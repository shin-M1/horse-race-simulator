from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from race_trend_analyzer import (
    analyze_same_race_trends,
    apply_race_trend_scores,
    calculate_race_trend_score,
)
from race_trend_fetcher import fetch_same_race_history
from race_trend_database import analyze_same_race_trend_database, save_same_race_trend_database
from race_trend_scorer import compute_race_trend_match_score


def sample_history() -> list[dict]:
    return [
        {
            "year": 2025,
            "race_id": "202501",
            "finish": 1,
            "horse_name": "Trend Winner",
            "frame": 7,
            "horse_number": 13,
            "popularity": 4,
            "passing_order": "8-8-7-3",
            "last3f": 33.8,
            "jockey_switch": "継続",
            "field_size": 16,
            "track_condition": "良",
            "sire_line": "Sunday",
        },
        {
            "year": 2025,
            "race_id": "202501",
            "finish": 2,
            "horse_name": "Trend Place",
            "frame": 8,
            "horse_number": 15,
            "popularity": 2,
            "passing_order": "10-10-9-4",
            "last3f": 34.0,
            "jockey_switch": "継続",
            "field_size": 16,
            "track_condition": "良",
            "sire_line": "Kingman",
        },
        {
            "year": 2024,
            "race_id": "202401",
            "finish": 1,
            "horse_name": "Front Winner",
            "frame": 2,
            "horse_number": 3,
            "popularity": 1,
            "passing_order": "2-2-2-1",
            "last3f": 34.6,
            "jockey_switch": "乗り替わり",
            "field_size": 14,
            "track_condition": "稍重",
            "sire_line": "Roberto",
        },
    ]


class RaceTrendTest(unittest.TestCase):
    def test_fetch_same_race_history_uses_real_results_without_dummy_rows(self) -> None:
        def search_fn(race_name: str, race_date: str) -> dict | None:
            year = race_date[:4]
            if year in {"2025", "2024", "2023"}:
                return {"race_id": f"{year}01", "race_name": race_name, "race_date": race_date, "venue": "阪神"}
            return None

        def result_fn(race_id: str) -> list[dict]:
            if race_id == "202301":
                return []
            return [
                {"finish": 1, "horse_name": f"Horse {race_id}", "frame": 1, "horse_number": 1, "popularity": 1},
                {"finish": 2, "horse_name": f"Runner {race_id}", "frame": 2, "horse_number": 2, "popularity": 3},
            ]

        def metadata_fn(race_id: str) -> dict:
            return {"race_name": "宝塚記念", "venue": "阪神", "distance": 2200, "surface": "芝"}

        history = fetch_same_race_history(
            "宝塚記念",
            "2026-06-28",
            years=3,
            search_fn=search_fn,
            result_fn=result_fn,
            metadata_fn=metadata_fn,
        )

        self.assertEqual(len(history), 4)
        self.assertEqual({row["race_id"] for row in history}, {"202501", "202401"})
        self.assertTrue(all(row["horse_name"] for row in history))

    def test_analyze_same_race_trends_returns_required_sections(self) -> None:
        trends = analyze_same_race_trends(sample_history())

        for key in [
            "frame_bias",
            "horse_number_bias",
            "running_style_bias",
            "popularity_bias",
            "agari_bias",
            "jockey_switch_bias",
            "bloodline_bias",
            "summary_bullets",
            "trend_scores",
            "details",
        ]:
            self.assertIn(key, trends)
        self.assertGreater(trends["details"]["sample_size"], 0)
        self.assertIn("agari_importance", trends["trend_scores"])

    def test_race_trend_score_changes_prediction_table(self) -> None:
        trends = analyze_same_race_trends(sample_history())
        table = pd.DataFrame(
            [
                {"馬番": 13, "馬名": "A", "枠順": 7, "actual_running_style": "差し", "late_kick_score": 80},
                {"馬番": 4, "馬名": "B", "枠順": 4, "actual_running_style": "逃げ", "late_kick_score": 45},
            ]
        )

        scored = apply_race_trend_scores(table, trends)

        self.assertIn("race_trend_score", scored.columns)
        self.assertIn("trend_match_comment", scored.columns)
        self.assertNotEqual(float(scored.loc[0, "race_trend_score"]), 50.0)

    def test_calculate_race_trend_score_has_neutral_fallback(self) -> None:
        score, comment = calculate_race_trend_score({"馬番": 1, "枠順": 1}, None)

        self.assertEqual(score, 50.0)
        self.assertTrue(comment)

    def test_same_race_trend_database_saved(self) -> None:
        database = {
            "race_name": "Trend Stakes",
            "venue": "Tokyo",
            "distance": 2000,
            "rows": [
                {
                    "race_name": "Trend Stakes",
                    "year": 2025,
                    "race_id": "202501",
                    "venue": "Tokyo",
                    "surface": "芝",
                    "distance": 2000,
                    "track_condition": "良",
                    "finish": 1,
                    "horse_name": "A",
                    "frame": 1,
                    "horse_number": 1,
                    "age": 4,
                    "sex": "牡",
                    "carried_weight": 57.0,
                    "jockey": "J",
                    "previous_jockey": "J",
                    "jockey_change_type": "継続",
                    "running_style": "差し",
                    "passing_order": "8-8-7-3",
                    "fourth_corner_pos": 3,
                    "last3f": 33.8,
                    "last3f_rank": 1,
                    "previous_race_class": "G2",
                    "previous_distance": 2000,
                    "sire": "Sunday",
                    "broodmare_sire": "Kingman",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            paths = save_same_race_trend_database(database, directory)

            self.assertTrue(Path(paths["json"]).exists())
            self.assertTrue(Path(paths["csv"]).exists())

    def test_compute_race_trend_match_score(self) -> None:
        database = {
            "rows": [
                {
                    "race_name": "Trend Stakes",
                    "year": 2025,
                    "race_id": "202501",
                    "venue": "Tokyo",
                    "surface": "芝",
                    "distance": 2000,
                    "track_condition": "良",
                    "finish": 1,
                    "horse_name": "A",
                    "frame": 1,
                    "horse_number": 1,
                    "age": 4,
                    "sex": "牡",
                    "carried_weight": 57.0,
                    "jockey": "J",
                    "previous_jockey": "J",
                    "jockey_change_type": "継続",
                    "running_style": "差し",
                    "passing_order": "8-8-7-3",
                    "fourth_corner_pos": 3,
                    "last3f": 33.8,
                    "last3f_rank": 1,
                    "previous_race_class": "G2",
                    "previous_distance": 2000,
                    "sire": "Sunday",
                    "broodmare_sire": "Kingman",
                },
                {
                    "race_name": "Trend Stakes",
                    "year": 2024,
                    "race_id": "202401",
                    "venue": "Tokyo",
                    "surface": "芝",
                    "distance": 2000,
                    "track_condition": "良",
                    "finish": 2,
                    "horse_name": "B",
                    "frame": 1,
                    "horse_number": 2,
                    "age": 4,
                    "sex": "牡",
                    "carried_weight": 57.0,
                    "jockey": "K",
                    "previous_jockey": "K",
                    "jockey_change_type": "継続",
                    "running_style": "差し",
                    "passing_order": "7-7-6-2",
                    "fourth_corner_pos": 2,
                    "last3f": 34.0,
                    "last3f_rank": 2,
                    "previous_race_class": "G2",
                    "previous_distance": 2000,
                    "sire": "Sunday",
                    "broodmare_sire": "Roberto",
                },
            ],
            "year_status": [],
            "save_paths": {},
        }
        trends = analyze_same_race_trend_database(database)
        score = compute_race_trend_match_score(
            horse={"horse_number": 1, "frame": 1, "carried_weight": 57.0},
            horse_analysis={
                "primary_running_style": "差し",
                "late_kick_score": 85.0,
                "weighted_avg_last_corner_ratio": 0.20,
                "age": 4,
                "jockey_change_type": "継続",
                "previous_race_class": "G2",
                "previous_distance": 2000,
                "sire": "Sunday",
            },
            race_trends=trends,
            race_config={"course": "Tokyo", "distance": 2000},
        )

        self.assertIn("race_trend_score", score)
        self.assertIn("trend_match_comment", score)
        self.assertGreater(score["race_trend_score"], 50.0)


if __name__ == "__main__":
    unittest.main()
