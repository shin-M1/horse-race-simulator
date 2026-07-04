from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from horse_analyzer import HorseAnalyzer, RaceResult
from monte_carlo import (
    _assign_marks,
    run_monte_carlo_prediction,
    select_highest_expected_value_trial,
    select_representative_trial,
)
from race_config import HorseEntry, RaceConfig


class StaticProvider:
    def get_recent_results(self, horse_name: str, limit: int = 10) -> list[RaceResult]:
        patterns = {
            "Fast": ["1-1-1-1", "2-2-2-2", "3-3-3-2", "2-2-2-1", "4-4-4-3"],
            "Late": ["12-12-11-3", "10-10-9-4", "8-8-7-4", "11-11-10-5", "9-9-8-4"],
            "Stable": ["4-4-4-3", "5-5-5-4", "6-6-5-3", "4-4-3-2", "5-5-4-3"],
        }
        orders = patterns.get(horse_name, patterns["Stable"])
        return [
            RaceResult(
                race_name=f"Test G1 {index}",
                distance=2000,
                surface="芝",
                track_condition="良",
                finish_position=min(3 + index % 3, 8),
                margin=0.2 + index * 0.05,
                passing_order=order,
                final_3f=34.5 + index * 0.1,
                field_size=16,
                race_class="G1",
            )
            for index, order in enumerate(orders[:limit])
        ]


class MonteCarloPredictionTest(unittest.TestCase):
    def test_prediction_table_has_rates_and_saved_outputs(self) -> None:
        horses = [HorseEntry("Fast", 1, 1), HorseEntry("Late", 4, 2), HorseEntry("Stable", 7, 3)]
        provider = StaticProvider()
        abilities = HorseAnalyzer(provider).analyze_many(horses)

        result = run_monte_carlo_prediction(
            RaceConfig.default(),
            horses,
            n_simulations=30,
            seed=123,
            abilities=abilities,
            output_dir=str(ROOT / "work" / "test_outputs"),
        )

        table = result["prediction_table"]
        self.assertEqual(len(table), 3)
        for column in ["馬番", "馬名", "斤量", "primary_running_style", "actual_running_style", "win_rate", "top2_rate", "top3_rate", "prediction_score", "race_power", "score", "印", "予想根拠"]:
            self.assertIn(column, table.columns)
        for column in ["race_trend_score", "style_trend_score", "agari_trend_score"]:
            self.assertIn(column, table.columns)
        for column in ["odds", "implied_probability", "value_score", "popularity_score"]:
            self.assertNotIn(column, table.columns)
        self.assertTrue(((table["win_rate"] >= 0) & (table["win_rate"] <= 1)).all())
        self.assertTrue(((table["prediction_score"] >= 0) & (table["prediction_score"] <= 100)).all())
        self.assertTrue(((table["race_power"] >= 40) & (table["race_power"] <= 95)).all())
        self.assertTrue(((table["score"] >= 0) & (table["score"] <= 100)).all())
        self.assertTrue(Path(result["summary"]["saved_paths"]["prediction_table"]).exists())

    def test_race_trend_score_used_in_prediction_table(self) -> None:
        horses = [HorseEntry("Fast", 1, 1), HorseEntry("Late", 4, 2), HorseEntry("Stable", 7, 3)]
        abilities = HorseAnalyzer(StaticProvider()).analyze_many(horses)
        trend_analysis = {
            "trend_scores": {"front_advantage": 75, "closer_advantage": 45, "agari_importance": 60},
            "details": {
                "frame": {"1": {"top3_rate": 0.9}, "4": {"top3_rate": 0.4}, "7": {"top3_rate": 0.2}},
                "horse_number": {"1": {"top3_rate": 0.8}, "2": {"top3_rate": 0.5}, "3": {"top3_rate": 0.3}},
                "style": {"逃げ": {"top3_rate": 0.85}, "差し": {"top3_rate": 0.4}},
            },
        }

        result = run_monte_carlo_prediction(
            RaceConfig.default(),
            horses,
            n_simulations=5,
            seed=456,
            abilities=abilities,
            trend_analysis=trend_analysis,
            output_dir=str(ROOT / "work" / "test_outputs"),
        )
        table = result["prediction_table"]

        self.assertIn("race_trend_score", table.columns)
        self.assertIn("trend_match_comment", table.columns)
        self.assertTrue((table["race_trend_score"] != 50).any())

    def test_assign_marks_is_limited_to_top_five(self) -> None:
        table = pd.DataFrame({"score": [90, 80, 70, 60, 50, 40]})

        self.assertEqual(_assign_marks(table), ["◎", "○", "▲", "△", "☆", ""])

    def test_select_representative_trial_returns_nearest_ranking(self) -> None:
        prediction_table = pd.DataFrame(
            {
                "馬番": [1, 2, 3],
                "score": [80.0, 70.0, 60.0],
                "win_rate": [0.4, 0.3, 0.2],
                "top3_rate": [0.8, 0.7, 0.6],
            }
        )
        far_trial = {
            "trial_index": 0,
            "seed": 100,
            "race_timeline": [{"horses": []}],
            "result_df": pd.DataFrame({"着順": [1, 2, 3], "馬番": [3, 2, 1]}),
        }
        near_trial = {
            "trial_index": 1,
            "seed": 101,
            "race_timeline": [{"horses": [{"horse_number": 1}]}],
            "result_df": pd.DataFrame({"着順": [1, 2, 3], "馬番": [1, 2, 3]}),
        }

        selected = select_representative_trial(prediction_table, [far_trial, near_trial])

        self.assertEqual(selected["trial_index"], 1)
        self.assertEqual(selected["ranking_distance"], 0.0)
        self.assertEqual(selected["top5_overlap_count"], 3)

    def test_representative_trial_has_timeline(self) -> None:
        prediction_table = pd.DataFrame(
            {"馬番": [1, 2], "score": [70.0, 60.0], "win_rate": [0.5, 0.2], "top3_rate": [0.9, 0.7]}
        )
        timeline = [{"horses": [{"horse_number": 1, "position_m": 100.0}]}]
        selected = select_representative_trial(
            prediction_table,
            [{"trial_index": 0, "seed": 42, "race_timeline": timeline, "result_df": pd.DataFrame({"着順": [1, 2], "馬番": [1, 2]})}],
        )

        self.assertIs(selected["race_timeline"], timeline)
        self.assertTrue(selected["race_timeline"])

    def test_single_result_matches_representative_trial(self) -> None:
        prediction_table = pd.DataFrame(
            {"馬番": [2, 1], "score": [75.0, 70.0], "win_rate": [0.5, 0.4], "top3_rate": [0.8, 0.8]}
        )
        result_df = pd.DataFrame({"着順": [1, 2], "馬番": [2, 1]})
        selected = select_representative_trial(
            prediction_table,
            [{"trial_index": 0, "seed": 42, "race_timeline": [{"horses": []}], "result_df": result_df}],
        )

        pd.testing.assert_frame_equal(selected["result_df"], result_df)

    def test_select_highest_expected_value_trial(self) -> None:
        prediction_table = pd.DataFrame(
            {"馬番": [1, 2, 3], "prediction_score": [90.0, 60.0, 30.0]}
        )
        low_value = {
            "trial_index": 0,
            "seed": 100,
            "result_df": pd.DataFrame({"着順": [1, 2, 3], "馬番": [3, 2, 1]}),
            "race_timeline": [{"horses": []}],
        }
        high_value = {
            "trial_index": 1,
            "seed": 101,
            "result_df": pd.DataFrame({"着順": [1, 2, 3], "馬番": [1, 2, 3]}),
            "race_timeline": [{"horses": [{"horse_number": 1}]}],
        }

        selected = select_highest_expected_value_trial(prediction_table, [low_value, high_value])

        self.assertEqual(selected["trial_index"], 1)
        self.assertGreater(selected["representative_value_score"], 0.0)
        self.assertEqual(selected["top5_horses_in_selected_trial"], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
