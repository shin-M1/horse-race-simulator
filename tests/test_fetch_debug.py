from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from errors import RaceDataFetchError
from main import _wrap_provider


class DataFrameProvider:
    def __init__(self) -> None:
        self.raw_df = pd.DataFrame(
            [
                {
                    "race_name": "テストS",
                    "distance": 1800,
                    "surface": "芝",
                    "track_condition": "良",
                    "finish": 1,
                    "passing_order": "1-1-1-1",
                    "last3f": 35.2,
                    "field_size": 16,
                }
            ]
        )
        self.last_debug = {}

    def get_recent_results(self, horse_name: str, limit: int = 5):
        self.last_debug = {
            horse_name: {
                "search_result_horse_name": horse_name,
                "horse_id": "2020123456",
                "url": "https://db.netkeiba.com/horse/2020123456/",
                "raw_race_df": self.raw_df,
            }
        }
        return self.raw_df


class FetchDebugTest(unittest.TestCase):
    def test_provider_adapter_keeps_raw_dataframe_and_lookup_metadata(self) -> None:
        adapter = _wrap_provider(DataFrameProvider())

        results = adapter.get_recent_results("入力馬", limit=10)
        debug = adapter.get_fetch_debug()[0]

        self.assertEqual(len(results), 1)
        self.assertEqual(debug["input_horse_name"], "入力馬")
        self.assertEqual(debug["search_result_horse_name"], "入力馬")
        self.assertEqual(debug["horse_id"], "2020123456")
        self.assertIs(debug["name_match"], True)
        self.assertIsInstance(debug["raw_race_df"], pd.DataFrame)
        self.assertEqual(debug["recent_races"][0]["race_name"], "テストS")

    def test_provider_adapter_stops_on_horse_name_mismatch(self) -> None:
        provider = DataFrameProvider()
        adapter = _wrap_provider(provider)

        def mismatched_get_recent_results(horse_name: str, limit: int = 5):
            rows = provider.raw_df
            provider.last_debug = {
                horse_name: {
                    "search_result_horse_name": "別の馬",
                    "horse_id": "2020123456",
                    "url": "https://db.netkeiba.com/horse/2020123456/",
                    "raw_race_df": rows,
                }
            }
            return rows

        provider.get_recent_results = mismatched_get_recent_results  # type: ignore[method-assign]

        with self.assertRaises(RaceDataFetchError):
            adapter.get_recent_results("入力馬", limit=10)

        debug = adapter.get_fetch_debug()[0]
        self.assertEqual(debug["search_result_horse_name"], "別の馬")
        self.assertIs(debug["name_match"], False)


if __name__ == "__main__":
    unittest.main()
