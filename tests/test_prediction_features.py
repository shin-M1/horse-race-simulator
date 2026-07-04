from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from course_db import estimate_course_fit_score
from horse_analyzer import (
    HorseAnalyzer,
    RaceResult,
    build_late_kick_score,
    update_elo_ratings,
)
from monte_carlo import _build_prediction_table
from pace_predictor import PacePredictor
from race_config import HorseEntry, RaceConfig


class StaticProvider:
    def __init__(self, results: list[RaceResult]) -> None:
        self.results = results

    def get_recent_results(self, horse_name: str, limit: int = 5) -> list[RaceResult]:
        return self.results[:limit]


def past_race(*, opponent_score: float = 50.0, order: str = "2-2-2-1") -> RaceResult:
    return RaceResult(
        race_name="test",
        distance=1800,
        surface="芝",
        track_condition="良",
        finish_position=1,
        margin=0.0,
        passing_order=order,
        final_3f=34.5,
        race_time_seconds=108.0,
        field_size=16,
        race_class="G2",
        popularity="2",
        course="東京",
        raw={
            "average_opponent_score": opponent_score,
            "direction": "左",
            "field_avg_last3f": 35.5,
        },
    )


class PredictionFeatureTest(unittest.TestCase):
    def test_race_strength_affects_ability(self) -> None:
        strong = HorseAnalyzer(StaticProvider([past_race(opponent_score=90.0)] * 5)).analyze(
            HorseEntry("StrongRace", 1, 1)
        )
        weak = HorseAnalyzer(StaticProvider([past_race(opponent_score=25.0)] * 5)).analyze(
            HorseEntry("WeakRace", 1, 1)
        )

        self.assertGreater(strong.race_strength_score, weak.race_strength_score)
        self.assertGreater(strong.horse_ability_score, weak.horse_ability_score)

    def test_elo_rating_updates_from_finish_order(self) -> None:
        ratings = update_elo_ratings(
            [
                {"race_id": "R1", "horse_name": "Winner", "finish": 1},
                {"race_id": "R1", "horse_name": "Second", "finish": 2},
                {"race_id": "R1", "horse_name": "Third", "finish": 3},
            ],
            {},
        )

        self.assertGreater(ratings["Winner"], ratings["Second"])
        self.assertGreater(ratings["Second"], ratings["Third"])

    def test_pace_uses_early_push_score(self) -> None:
        profile = {"逃げ": 0.30, "先行": 0.20, "差し": 0.30, "追込": 0.20}
        high = SimpleNamespace(
            horse_name="HighPush",
            primary_running_style="先行",
            base_style_profile=profile,
            early_push_score=100.0,
        )
        low = SimpleNamespace(
            horse_name="LowPush",
            primary_running_style="先行",
            base_style_profile=profile,
            early_push_score=0.0,
        )

        self.assertEqual(PacePredictor().predict([high]).pace, "high")
        self.assertEqual(PacePredictor().predict([low]).pace, "slow")

    def test_relative_agari_score_used(self) -> None:
        fast = build_late_kick_score(
            [{"passing_order": "8-8-7-3", "finish": 3, "field_size": 16, "last3f": 34.0, "race_avg_last3f": 36.0}]
        )
        average = build_late_kick_score(
            [{"passing_order": "8-8-7-3", "finish": 3, "field_size": 16, "last3f": 36.0, "race_avg_last3f": 36.0}]
        )

        self.assertGreater(float(fast["relative_agari_score"]), float(average["relative_agari_score"]))
        self.assertGreater(float(fast["late_kick_score"]), float(average["late_kick_score"]))

    def test_course_fit_score_used(self) -> None:
        config = RaceConfig(
            course="東京",
            surface="芝",
            distance=1800,
            direction="左",
            weather="晴",
            track_condition="良",
        )
        matching = [
            {
                "course": "東京",
                "surface": "芝",
                "distance": 1800,
                "direction": "左",
                "finish": 1,
                "field_size": 16,
                "margin": 0.0,
            }
        ]

        self.assertGreater(estimate_course_fit_score(matching, config), 50.0)

    def test_jockey_score_used(self) -> None:
        results = [past_race()] * 5
        neutral = HorseAnalyzer(StaticProvider(results)).analyze(
            HorseEntry("Neutral", 1, 1, jockey="A", jockey_score=50.0)
        )
        strong = HorseAnalyzer(StaticProvider(results)).analyze(
            HorseEntry("Strong", 1, 1, jockey="B", jockey_score=90.0)
        )

        self.assertAlmostEqual(strong.horse_ability_score, neutral.horse_ability_score)
        config = RaceConfig.default()
        neutral_table = _build_prediction_table(
            config,
            {
                1: {
                    "ability": neutral,
                    "finishes": [1, 1, 1],
                    "times": [108.0, 108.1, 107.9],
                    "actual_styles": [neutral.primary_running_style] * 3,
                }
            },
            1,
            PacePredictor().predict([neutral]),
        )
        strong_table = _build_prediction_table(
            config,
            {
                1: {
                    "ability": strong,
                    "finishes": [1, 1, 1],
                    "times": [108.0, 108.1, 107.9],
                    "actual_styles": [strong.primary_running_style] * 3,
                }
            },
            1,
            PacePredictor().predict([strong]),
        )
        self.assertGreater(
            float(strong_table.iloc[0]["final_prediction_score"]),
            float(neutral_table.iloc[0]["final_prediction_score"]),
        )

    def test_track_bias_changes_prediction(self) -> None:
        standard_config = RaceConfig(
            course="東京",
            surface="芝",
            distance=1800,
            direction="左",
            weather="晴",
            track_condition="良",
            track_bias="標準",
        )
        front_config = replace(standard_config, track_bias="前残り")
        ability = HorseAnalyzer(StaticProvider([past_race(order="1-1-1-1")] * 5), standard_config).analyze(
            HorseEntry("Runner", 1, 1)
        )
        aggregates = {
            1: {
                "ability": ability,
                "finishes": [1, 1, 1],
                "times": [108.0, 108.1, 107.9],
                "actual_styles": ["逃げ", "逃げ", "逃げ"],
            }
        }
        pace = PacePredictor().predict([ability])
        standard = _build_prediction_table(standard_config, aggregates, 1, pace)
        front = _build_prediction_table(front_config, aggregates, 1, pace)

        self.assertGreater(
            float(front.iloc[0]["final_prediction_score"]),
            float(standard.iloc[0]["final_prediction_score"]),
        )


if __name__ == "__main__":
    unittest.main()
