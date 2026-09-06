from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import horse_database
from app_state import dataframe_to_horses
from data_fetcher import NetkeibaRaceResultProvider, decode_html, horse_url_from_id, is_valid_horse_id
from errors import RaceDataFetchError
from horse_analyzer import HorseAnalyzer
from main import _coerce_horses, _wrap_provider
from netkeiba_fetcher import NetkeibaRaceFetcher
from prediction_explainability import build_prediction_confidence
from race_config import RaceConfig


HISTORY_HTML = '''<html><head><title>Alpha | Horse</title></head><body><h1>Alpha</h1>
<table><tr><th>date</th><th>race_name</th><th>distance</th><th>finish</th>
<th>passing_order</th><th>last3f</th><th>track_condition</th><th>time</th><th>field_size</th></tr>
<tr><td>2026/05/01</td><td>Race A</td><td>1600</td><td>2</td>
<td>3-3-2-2</td><td>34.8</td><td>良</td><td>1:34.0</td><td>12</td></tr></table></body></html>'''


def response(text=HISTORY_HTML, status=200):
    return {"text": text, "status": status, "html_bytes": len(text.encode()),
            "title": "Alpha | Horse", "encoding": "utf-8",
            "final_url": horse_url_from_id("2020000001")}


def recent_row():
    return {"race_name": "Race A", "distance": 1600, "finish": 2, "surface": "芝",
            "track_condition": "良", "passing_order": "3-3-2-2", "last3f": 34.8,
            "time": "1:34.0", "field_size": 12, "date": "2026/05/01"}


class IdProvider:
    def __init__(self, failed=()):
        self.calls = []
        self.failed = set(failed)

    def get_recent_results_by_horse_id(self, horse_id, horse_name="", limit=10):
        self.calls.append(horse_id)
        if horse_id in self.failed:
            raise ValueError("HTTP 429")
        return [recent_row()]

    def get_recent_results(self, *args, **kwargs):
        raise AssertionError("Horse ID was lost; name-based fetching should not run")


def entries():
    return [{"horse_name": name, "horse_id": f"202000000{index}", "frame": index,
             "horse_number": index, "horse_url": horse_url_from_id(f"202000000{index}")}
            for index, name in enumerate(["Alpha", "Beta"], start=1)]


class HorseHistoryRegressionTest(unittest.TestCase):
    def test_horse_id_is_extracted_for_all_entries(self):
        html = '''<table><tr><th>枠</th><th>馬番</th><th>馬名</th><th>騎手</th><th>斤量</th></tr>
        <tr><td>1</td><td>1</td><td><a href="https://db.netkeiba.com/horse/2020000001/">Alpha</a></td><td>A</td><td>56</td></tr>
        <tr><td>2</td><td>2</td><td><a href="https://db.netkeiba.com/horse/2020000002/">Beta</a></td><td>B</td><td>56</td></tr></table>'''
        fetcher = NetkeibaRaceFetcher()
        with patch.object(fetcher, "_request_text", return_value=(html, "https://race.netkeiba.com/")):
            fetched = fetcher.fetch_race_entries("202605020811")
        converted = _coerce_horses(dataframe_to_horses(pd.DataFrame(fetched)))
        self.assertEqual([entry.horse_id for entry in converted], ["2020000001", "2020000002"])

    def test_horse_history_url_is_valid(self):
        self.assertEqual(horse_url_from_id("2020000001"), "https://db.netkeiba.com/horse/2020000001/")
        for invalid in [None, "", float("nan"), 0, "abc", "202605020811", pd.NA]:
            self.assertFalse(is_valid_horse_id(invalid))
            with self.assertRaises(ValueError):
                horse_url_from_id(invalid)

    def test_horse_history_fetch_returns_rows(self):
        provider = NetkeibaRaceResultProvider()
        with patch.object(provider, "request_text_response", return_value=response()) as request:
            rows = provider.get_recent_results_by_horse_id("2020000001", "Alpha")
        self.assertEqual(rows[0]["finish"], 2)
        self.assertEqual(len(rows), 1)
        self.assertEqual(provider.last_debug["parse_rows"], 1)
        self.assertEqual(request.call_count, 1)  # Static history needs no AJAX request.

    def test_missing_agari_and_passing_do_not_discard_valid_finish(self):
        provider = NetkeibaRaceResultProvider()
        html = HISTORY_HTML.replace("3-3-2-2", "-").replace("34.8", "-")
        with patch.object(provider, "request_text_response", return_value=response(html)):
            adapter = _wrap_provider(provider)
            adapter.set_race_entries(entries())
            rows = adapter.get_recent_results("Alpha", 10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].passing_order, [])
        self.assertEqual(rows[0].final_3f, 0.0)

    def test_empty_history_is_not_cached_as_success(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(horse_database, "HORSE_DB_DIR", Path(directory)):
            for payload in [None, {"recent_races": []}, {"recent_races": [{}]}]:
                self.assertIsNone(horse_database.get_or_fetch_horse_profile("Alpha", lambda: payload, horse_id="2020000001"))
                self.assertEqual(list(Path(directory).glob("*.json")), [])
            with self.assertRaises(ValueError):
                horse_database.get_or_fetch_horse_profile("Alpha", lambda: [], horse_id=None)

    def test_http_failure_is_not_cached_as_success(self):
        for status in [403, 404, 429, 500, 503]:
            provider = NetkeibaRaceResultProvider()
            with tempfile.TemporaryDirectory() as directory, patch.object(horse_database, "HORSE_DB_DIR", Path(directory)), patch.object(provider, "request_text_response", return_value=response(status=status)):
                adapter = _wrap_provider(provider, use_local_database=True)
                adapter.set_race_entries(entries())
                with self.assertRaises(RaceDataFetchError):
                    adapter.get_recent_results("Alpha", 10)
                self.assertEqual(adapter.get_fetch_debug()[0]["http_status"], status)
                self.assertEqual(list(Path(directory).glob("*.json")), [])

    def test_cache_hit_debug_belongs_to_the_correct_horse(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(horse_database, "HORSE_DB_DIR", Path(directory)):
            provider = IdProvider()
            adapter = _wrap_provider(provider, use_local_database=True)
            adapter.set_race_entries(entries())
            adapter.get_recent_results("Alpha", 10)
            adapter.get_recent_results("Beta", 10)
            adapter.get_recent_results("Alpha", 10)
            debug = adapter.get_fetch_debug()[-1]
            self.assertEqual(debug["horse_id"], "2020000001")
            self.assertEqual(debug["horse_database"], "hit")
            self.assertEqual(provider.calls, ["2020000001", "2020000002"])

    def test_prediction_calls_horse_history_fetch(self):
        provider = IdProvider()
        adapter = _wrap_provider(provider)
        adapter.set_race_entries(entries())
        abilities = HorseAnalyzer(adapter).analyze_many(_coerce_horses(entries()))
        self.assertEqual(provider.calls, ["2020000001", "2020000002"])
        self.assertTrue(all(ability.recent_results for ability in abilities))

    def test_prediction_uses_history_after_recent5_ui_removed(self):
        import app
        from public_prediction import build_public_prediction_result

        provider = _wrap_provider(IdProvider())
        config = {**RaceConfig.default().to_dict(), "distance": 1600}
        with patch.object(app.simulation_main, "load_provider", return_value=provider):
            result = app.run_public_prediction_analysis(config, entries(), provider_module=None, provider_factory=None, use_local_database=False, force_refresh_data=False)
        prediction = build_public_prediction_result(config, entries(), result["abilities"], result["pace"])
        self.assertEqual(len(prediction["prediction_table"]), 2)
        self.assertTrue(all(ability.recent_results for ability in result["abilities"]))
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertNotIn('st.subheader("近走5走データ確認")', source)

    def test_partial_history_failure_does_not_stop_prediction(self):
        adapter = _wrap_provider(IdProvider(failed=["2020000002"]))
        adapter.set_race_entries(entries())
        abilities = HorseAnalyzer(adapter).analyze_many(_coerce_horses(entries()))
        self.assertEqual(len(abilities), 2)
        self.assertTrue(abilities[0].recent_results)
        self.assertEqual(abilities[1].recent_results, [])
        self.assertEqual(abilities[1].history_confidence, 0)
        self.assertEqual(build_prediction_confidence(abilities[1])["prediction_confidence"], "Very Low")

    def test_all_history_failure_shows_diagnostic_error(self):
        adapter = _wrap_provider(IdProvider(failed=["2020000001", "2020000002"]))
        adapter.set_race_entries(entries())
        with self.assertRaisesRegex(RaceDataFetchError, "近走取得処理そのものが失敗") as context:
            HorseAnalyzer(adapter).analyze_many(_coerce_horses(entries()))
        self.assertEqual(len(context.exception.debug_records), 2)

    def test_encoding_variants(self):
        for encoding in ["utf-8", "euc-jp", "cp932"]:
            html = f'<meta charset="{encoding}"><h1>近走成績</h1>'
            self.assertIn("近走成績", decode_html(html.encode(encoding), None))


if __name__ == "__main__":
    unittest.main()
