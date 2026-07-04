from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluation import evaluate_prediction, save_prediction_log
from horse_analyzer import parse_race_time
from netkeiba_fetcher import NetkeibaRaceFetcher
from race_data_fetcher import load_prediction_race_data


class NetkeibaRaceFetcherTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fetcher = NetkeibaRaceFetcher(min_interval_sec=0.0)

    def test_search_race_id_returns_none_when_not_found(self) -> None:
        self.fetcher._request_text = lambda url: (
            '<html><a href="/race/shutuba.html?race_id=202509030811">別のレース</a></html>',
            url,
        )
        self.assertIsNone(self.fetcher.search_race_id_by_name_and_date("宝塚記念", "2025-06-15"))

    def test_fetch_entries_no_dummy_data(self) -> None:
        self.fetcher._request_text = lambda url: ("<html><body>出馬表なし</body></html>", url)
        self.assertEqual(self.fetcher.fetch_race_entries("202509030811"), [])

    def test_fetch_result_no_dummy_data(self) -> None:
        self.fetcher._request_text = lambda url: ("<html><body>結果未確定</body></html>", url)
        self.assertEqual(self.fetcher.fetch_race_result("202509030811"), [])

    def test_prediction_loader_does_not_fetch_result(self) -> None:
        calls: list[str] = []

        def search(name: str, race_date: str) -> dict[str, str]:
            calls.append("search")
            return {"race_id": "202509030811", "race_name": name, "race_date": race_date}

        def entries(race_id: str) -> list[dict[str, object]]:
            calls.append("entries")
            return [{"horse_name": "テストホース", "horse_number": 1, "frame": 1}]

        def metadata(race_id: str) -> dict[str, object]:
            calls.append("metadata")
            return {"distance": 2200}

        loaded = load_prediction_race_data(
            "宝塚記念",
            "2025-06-15",
            search_fn=search,
            entries_fn=entries,
            metadata_fn=metadata,
        )
        self.assertIsNotNone(loaded)
        self.assertEqual(calls, ["search", "entries", "metadata"])

    def test_parse_race_time(self) -> None:
        self.assertAlmostEqual(parse_race_time("2:10.2"), 130.2)
        self.assertAlmostEqual(parse_race_time("1:58.3"), 118.3)


class EvaluationTest(unittest.TestCase):
    def prediction_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"印": "◎", "馬番": 1, "馬名": "A", "prediction_score": 70, "win_rate": 0.3, "top3_rate": 0.7},
                {"印": "○", "馬番": 2, "馬名": "B", "prediction_score": 65, "win_rate": 0.2, "top3_rate": 0.6},
                {"印": "▲", "馬番": 3, "馬名": "C", "prediction_score": 60, "win_rate": 0.1, "top3_rate": 0.5},
                {"印": "△", "馬番": 4, "馬名": "D", "prediction_score": 55, "win_rate": 0.1, "top3_rate": 0.4},
                {"印": "☆", "馬番": 5, "馬名": "E", "prediction_score": 50, "win_rate": 0.05, "top3_rate": 0.3},
            ]
        )

    def test_prediction_log_contains_race_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = save_prediction_log(
                race_id="202509030811",
                source_url="https://race.netkeiba.com/race/shutuba.html?race_id=202509030811",
                fetched_entries=[],
                race_metadata={"race_name": "宝塚記念", "race_date": "2025-06-15"},
                prediction_table=self.prediction_table(),
                simulation_result={},
                output_dir=directory,
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["race_id"], "202509030811")
            self.assertIsNone(payload["actual_result"])

    def test_evaluation_compares_prediction_and_actual_result(self) -> None:
        prediction_log = {"prediction_table": self.prediction_table().to_dict("records")}
        actual = [
            {"horse_number": 2, "finish": 1},
            {"horse_number": 1, "finish": 2},
            {"horse_number": 6, "finish": 3},
            {"horse_number": 3, "finish": 4},
            {"horse_number": 4, "finish": 5},
            {"horse_number": 5, "finish": 6},
        ]
        result = evaluate_prediction(prediction_log, actual)
        self.assertEqual(result["mark_finishes"]["◎"], 2)
        self.assertTrue(result["top3_hit"])
        self.assertFalse(result["win_hit"])
        self.assertEqual(result["marked_top3_count"], 2)


if __name__ == "__main__":
    unittest.main()
