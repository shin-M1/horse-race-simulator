from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from horse_analyzer import RaceResult
from prediction_explainability import (
    build_prediction_confidence,
    build_shap_like_contributions,
    classify_racing_affiliation,
    compute_track_speed_index,
    field_strength_score,
)
from public_prediction import build_public_prediction_result
from weight_optimizer import walk_forward_validation


def race(
    *,
    race_class: str = "G3",
    race_name: str = "テストS",
    course: str = "東京",
    date: str = "2025-10-01",
    distance: int = 1800,
    surface: str = "芝",
    time_sec: float = 108.0,
    avg_time: float | None = None,
    opponent: float = 60.0,
    field_size: int = 16,
) -> RaceResult:
    raw = {"average_opponent_score": opponent}
    if avg_time is not None:
        raw["same_condition_avg_time"] = avg_time
    return RaceResult(
        race_name=race_name,
        distance=distance,
        surface=surface,
        track_condition="良",
        finish_position=2,
        margin=0.1,
        passing_order="6-6-5-2",
        final_3f=34.4,
        race_time_seconds=time_sec,
        field_size=field_size,
        race_class=race_class,
        course=course,
        date=date,
        raw=raw,
    )


def ability(number: int = 1, recent_results: list[RaceResult] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        horse_number=number,
        horse_name=f"H{number}",
        frame=number,
        carried_weight=56.0,
        primary_running_style="差し",
        actual_running_style="差し",
        horse_ability_score=72.0,
        race_strength_score=68.0,
        race_strength_adjusted_score=70.0,
        normalized_elo_score=66.0,
        late_kick_score=74.0,
        course_fit_score=64.0,
        pace_fit_score=58.0,
        jockey_score=55.0,
        weight_penalty=2.0,
        mud_aptitude=50.0,
        recent_results=recent_results or [race()] * 5,
        jockey="TestJockey",
    )


class PredictionExplainabilityTest(unittest.TestCase):
    def test_jra_result_weighted_higher_than_local_general(self) -> None:
        jra = race(race_class="G2", race_name="毎日王冠", course="東京", opponent=75.0)
        local_general = race(race_class="C1", race_name="地方一般", course="大井", opponent=40.0)

        self.assertEqual(classify_racing_affiliation(jra), "JRA")
        self.assertEqual(classify_racing_affiliation(local_general), "地方一般")
        self.assertGreater(field_strength_score(jra), field_strength_score(local_general))

    def test_field_strength_score(self) -> None:
        jra_maiden = race(race_class="未勝利", race_name="未勝利", course="東京", opponent=45.0)
        exchange = race(race_class="Jpn2", race_name="交流重賞", course="大井", opponent=82.0)

        self.assertEqual(classify_racing_affiliation(exchange), "地方交流")
        self.assertGreater(field_strength_score(exchange), field_strength_score(jra_maiden))

    def test_track_speed_index(self) -> None:
        fast = race(time_sec=107.0, avg_time=108.0)
        slow = race(time_sec=109.0, avg_time=108.0)

        fast_index = compute_track_speed_index(fast)
        slow_index = compute_track_speed_index(slow)

        self.assertLess(fast_index["track_speed_index"], 0)
        self.assertGreater(fast_index["track_speed_score"], slow_index["track_speed_score"])

    def test_prediction_confidence(self) -> None:
        low = build_prediction_confidence(
            ability(recent_results=[race(course="大井", race_class="C1", surface="ダート")]),
            {"course": "東京", "surface": "芝", "distance": 2400, "race_date": "2026-05-01"},
        )

        self.assertIn(low["prediction_confidence"], {"Low", "Very Low"})
        self.assertTrue(low["prediction_confidence_reason"])

    def test_walk_forward_validation(self) -> None:
        frame = pd.DataFrame(
            [
                {"race_id": "2022a", "race_date": "2022-05-01", "horse_number": 1, "finish": 1},
                {"race_id": "2023a", "race_date": "2023-05-01", "horse_number": 1, "finish": 2},
                {"race_id": "2024a", "race_date": "2024-05-01", "horse_number": 1, "finish": 3},
            ]
        )
        splits = walk_forward_validation(frame)

        self.assertGreaterEqual(len(splits), 2)
        self.assertTrue(all(item["leakage_free"] for item in splits))
        self.assertEqual(splits[0]["validation_year"], 2023)

    def test_shap_like_contributions(self) -> None:
        contributions = build_shap_like_contributions(ability(), {"course": "東京", "surface": "芝", "distance": 1800})

        self.assertIn("能力評価", contributions)
        self.assertIn("相手レベル", contributions)
        self.assertIn("最終AIスコア", contributions)
        self.assertGreater(contributions["最終AIスコア"], 50.0)

    def test_jra_to_nar_transition(self) -> None:
        confidence = build_prediction_confidence(
            ability(recent_results=[race(course="東京") for _ in range(5)]),
            {"course": "大井", "surface": "芝", "distance": 1800, "race_date": "2026-01-01"},
        )

        self.assertIn("中央→地方", confidence["prediction_confidence_reason"])

    def test_nar_to_jra_transition(self) -> None:
        confidence = build_prediction_confidence(
            ability(recent_results=[race(course="大井", race_class="C1", surface="ダート") for _ in range(5)]),
            {"course": "東京", "surface": "芝", "distance": 1800, "race_date": "2026-01-01"},
        )

        self.assertIn("地方→中央", confidence["prediction_confidence_reason"])

    def test_unknown_race_level(self) -> None:
        unknown = race(race_class="", race_name="謎レース", course="")

        self.assertGreater(field_strength_score(unknown), 0.0)
        self.assertLessEqual(field_strength_score(unknown), 100.0)

    def test_public_prediction_uses_same_scores(self) -> None:
        result = build_public_prediction_result(
            {"course": "東京", "surface": "芝", "distance": 1800, "race_date": "2026-01-01"},
            [{"horse_name": "H1", "horse_number": 1, "frame": 1}],
            [ability()],
            seed=1,
        )
        table = result["prediction_table"]

        for column in [
            "field_strength_score",
            "track_speed_index",
            "prediction_confidence",
            "contribution_details",
            "AIスコア",
        ]:
            self.assertIn(column, table.columns)
        self.assertEqual(result["simulation_trials"], [])


if __name__ == "__main__":
    unittest.main()
