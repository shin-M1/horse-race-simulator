from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from public_prediction import (
    calculate_final_public_score,
    calculate_race_suitability_score,
    build_public_prediction_result,
    estimate_probabilities_from_scores,
    should_use_public_prediction,
)


def ability(
    number: int,
    name: str,
    *,
    horse_ability_score: float = 70.0,
    race_strength_score: float = 65.0,
    normalized_elo_score: float = 68.0,
    late_kick_score: float = 62.0,
    mud_aptitude: float = 55.0,
    course_fit_score: float = 60.0,
    pace_fit_score: float = 58.0,
    track_bias_fit_score: float = 55.0,
    race_trend_score: float = 50.0,
    jockey_score: float = 50.0,
    weight_penalty: float = 3.0,
    style: str = "先行",
) -> SimpleNamespace:
    return SimpleNamespace(
        horse_number=number,
        horse_name=name,
        frame=number,
        carried_weight=56.0,
        primary_running_style=style,
        actual_running_style=style,
        horse_ability_score=horse_ability_score,
        race_strength_score=race_strength_score,
        normalized_elo_score=normalized_elo_score,
        late_kick_score=late_kick_score,
        mud_aptitude=mud_aptitude,
        course_fit_score=course_fit_score,
        pace_fit_score=pace_fit_score,
        track_bias_fit_score=track_bias_fit_score,
        race_trend_score=race_trend_score,
        jockey_score=jockey_score,
        weight_penalty=weight_penalty,
        jockey="Test",
    )


class PublicPredictionTest(unittest.TestCase):
    def test_public_prediction_result_without_monte_carlo(self) -> None:
        result = build_public_prediction_result(
            {"course": "東京", "distance": 1600},
            [{"horse_name": "A", "frame": 1, "horse_number": 1}],
            [ability(1, "A")],
            seed=1,
        )

        self.assertTrue(result["public_prediction_mode"])
        self.assertEqual(result["simulation_trials"], [])
        self.assertEqual(result["representative_trial"], {})
        self.assertEqual(result["summary"]["n_simulations"], 0)

    def test_estimate_probabilities_sum_to_one(self) -> None:
        probabilities = estimate_probabilities_from_scores([80, 70, 60], temperature=12)
        self.assertAlmostEqual(sum(item["win_rate"] for item in probabilities), 1.0, places=6)

    def test_top3_rate_range(self) -> None:
        probabilities = estimate_probabilities_from_scores([95, 70, 55, 40], temperature=12)
        for item in probabilities:
            self.assertGreaterEqual(item["top3_rate"], 0.05)
            self.assertLessEqual(item["top3_rate"], 0.85)

    def test_public_prediction_contains_required_columns(self) -> None:
        result = build_public_prediction_result(
            {"course": "阪神", "distance": 2200},
            [{"horse_name": "A", "frame": 1, "horse_number": 1}],
            [ability(1, "A")],
        )
        table = result["prediction_table"]

        required = {
            "印",
            "予想着順",
            "馬番",
            "馬名",
            "斤量",
            "脚質",
            "能力スコア",
            "今回条件適性スコア",
            "AIスコア",
            "推定勝率",
            "推定連対率",
            "推定複勝率",
            "prediction_score",
            "score",
            "win_rate",
            "top2_rate",
            "top3_rate",
            "horse_number",
            "frame",
            "carried_weight",
            "primary_running_style",
            "actual_running_style",
            "final_prediction_score",
        }
        self.assertTrue(required.issubset(set(table.columns)))

    def test_cloud_skips_monte_carlo(self) -> None:
        self.assertTrue(should_use_public_prediction(is_cloud=True, public_prediction_only=False))

    def test_local_can_still_use_monte_carlo(self) -> None:
        self.assertFalse(should_use_public_prediction(is_cloud=False, public_prediction_only=False))

    def test_public_prediction_log_compatible(self) -> None:
        result = build_public_prediction_result(
            {"race_name": "Test"},
            [{"horse_name": "A", "frame": 1, "horse_number": 1}],
            [ability(1, "A")],
        )
        record = result["prediction_table"].to_dict("records")[0]

        for key in ["馬番", "馬名", "prediction_score", "score", "win_rate", "top3_rate"]:
            self.assertIn(key, record)

    def test_two_layer_score_contains_ability_and_suitability(self) -> None:
        result = build_public_prediction_result(
            {"race_name": "Test"},
            [{"horse_name": "A", "frame": 1, "horse_number": 1}],
            [ability(1, "A")],
        )
        table = result["prediction_table"]

        self.assertIn("能力スコア", table.columns)
        self.assertIn("今回条件適性スコア", table.columns)

    def test_weight_penalty_is_reversed(self) -> None:
        low_penalty = calculate_race_suitability_score(ability(1, "A", weight_penalty=2.0))
        high_penalty = calculate_race_suitability_score(ability(1, "A", weight_penalty=30.0))

        self.assertGreater(low_penalty, high_penalty)

    def test_final_score_combines_two_layers(self) -> None:
        self.assertAlmostEqual(calculate_final_public_score(80.0, 60.0), 73.0, places=4)

    def test_public_prediction_short_comment_has_three_parts(self) -> None:
        result = build_public_prediction_result(
            {"race_name": "Test"},
            [{"horse_name": "A", "frame": 1, "horse_number": 1}],
            [ability(1, "A")],
        )
        comment = str(result["prediction_table"].iloc[0]["短評"])

        self.assertIn("能力面", comment)
        self.assertIn("今回条件", comment)
        self.assertIn("リスク", comment)


if __name__ == "__main__":
    unittest.main()
