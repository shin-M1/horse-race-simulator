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

import horse_database
import race_database
import trend_database
from elo_rating import (
    expected_score,
    load_elo_ratings,
    normalize_elo_score,
    save_elo_ratings,
    update_elo_from_race_result,
)
from weight_optimizer import (
    DEFAULT_WEIGHTS,
    build_training_dataset,
    load_prediction_weights,
    optimize_prediction_weights,
    save_prediction_weights,
    score_with_weights,
)


class DatabaseEloOptimizerTest(unittest.TestCase):
    def test_horse_database_save_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            original = horse_database.HORSE_DB_DIR
            horse_database.HORSE_DB_DIR = Path(directory)
            try:
                horse_database.save_horse_profile("テストホース", {"recent_races": [{"race_name": "A"}]})
                loaded = horse_database.load_horse_profile("テストホース")
            finally:
                horse_database.HORSE_DB_DIR = original
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["recent_races"][0]["race_name"], "A")

    def test_race_database_save_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            original = race_database.RACE_DB_DIR
            race_database.RACE_DB_DIR = Path(directory)
            try:
                key = race_database.race_db_key("R1", "テストS", "2026-07-05")
                race_database.save_race_cache(key, {"race_id": "R1", "entries": [{"horse_name": "A"}]})
                loaded = race_database.load_race_cache(key)
            finally:
                race_database.RACE_DB_DIR = original
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["entries"][0]["horse_name"], "A")

    def test_trend_database_save_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            original = trend_database.TREND_DB_DIR
            trend_database.TREND_DB_DIR = Path(directory)
            try:
                trend_database.save_trend_cache("テストS", "東京", 1600, {"row_count": 10})
                loaded = trend_database.load_trend_cache("テストS", "東京", 1600)
            finally:
                trend_database.TREND_DB_DIR = original
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["row_count"], 10)

    def test_get_or_fetch_uses_cache(self) -> None:
        calls = {"count": 0}

        def fetcher() -> dict:
            calls["count"] += 1
            return {"recent_races": [{"race_name": "A"}]}

        with tempfile.TemporaryDirectory() as directory:
            original = horse_database.HORSE_DB_DIR
            horse_database.HORSE_DB_DIR = Path(directory)
            try:
                first = horse_database.get_or_fetch_horse_profile("CacheHorse", fetcher)
                second = horse_database.get_or_fetch_horse_profile("CacheHorse", fetcher)
            finally:
                horse_database.HORSE_DB_DIR = original
        self.assertEqual(calls["count"], 1)
        self.assertEqual(first["_database_status"], "miss")
        self.assertEqual(second["_database_status"], "hit")

    def test_elo_expected_score(self) -> None:
        self.assertAlmostEqual(expected_score(1500, 1500), 0.5)
        self.assertGreater(expected_score(1600, 1500), 0.5)

    def test_update_elo_from_race_result(self) -> None:
        ratings = update_elo_from_race_result(
            [
                {"horse_name": "A", "finish": 1},
                {"horse_name": "B", "finish": 2},
                {"horse_name": "C", "finish": 3},
            ],
            {},
        )
        self.assertGreater(ratings["A"], ratings["B"])
        self.assertGreater(ratings["B"], ratings["C"])

    def test_elo_save_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "elo.json"
            save_elo_ratings({"A": 1510.0}, path)
            loaded = load_elo_ratings(path)
        self.assertEqual(loaded["A"], 1510.0)

    def test_normalize_elo_score(self) -> None:
        self.assertEqual(normalize_elo_score(1300), 0.0)
        self.assertEqual(normalize_elo_score(1800), 100.0)
        self.assertAlmostEqual(normalize_elo_score(1550), 50.0)

    def test_build_training_dataset_without_popularity(self) -> None:
        logs = [_evaluation_log()]
        dataset = build_training_dataset(logs)
        self.assertIn("normalized_elo_score", dataset.columns)
        self.assertNotIn("popularity", dataset.columns)
        self.assertNotIn("popularity_score", dataset.columns)

    def test_optimize_prediction_weights(self) -> None:
        dataset = build_training_dataset([_evaluation_log(index) for index in range(3)])
        result = optimize_prediction_weights(dataset, n_trials=10)
        self.assertAlmostEqual(sum(result["weights"].values()), 1.0, places=5)
        self.assertIn("normalized_elo_score", result["weights"])

    def test_prediction_weights_save_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prediction_weights.json"
            save_prediction_weights({"weights": {"horse_ability_score": 1.0}}, path)
            loaded = load_prediction_weights(path)
        self.assertAlmostEqual(sum(loaded.values()), 1.0)
        self.assertGreater(loaded["horse_ability_score"], 0.5)

    def test_prediction_score_uses_optimized_weights(self) -> None:
        frame = pd.DataFrame(
            [
                {"horse_number": 1, "horse_ability_score": 95, "race_strength_score": 90, "normalized_elo_score": 90},
                {"horse_number": 2, "horse_ability_score": 45, "race_strength_score": 50, "normalized_elo_score": 50},
            ]
        )
        scores = score_with_weights(frame, DEFAULT_WEIGHTS)
        self.assertGreater(float(scores.iloc[0]), float(scores.iloc[1]))


def _evaluation_log(index: int = 0) -> dict:
    prediction_table = []
    actual_result = []
    for number in range(1, 5):
        strength = 95 - number * 10 + index
        prediction_table.append(
            {
                "horse_number": number,
                "horse_name": f"Horse {number}",
                "prediction_score": strength,
                "horse_ability_score": strength,
                "race_strength_score": strength - 2,
                "normalized_elo_score": strength - 1,
                "late_kick_score": strength - 3,
                "course_fit_score": strength - 4,
                "pace_fit_score": strength - 5,
                "track_bias_fit_score": 50,
                "race_trend_score": 50,
            }
        )
        actual_result.append({"horse_number": number, "finish": number})
    return {
        "race_id": f"race_{index}",
        "race_name": f"Race {index}",
        "race_date": "2026-07-05",
        "prediction_table": prediction_table,
        "actual_result": actual_result,
    }


if __name__ == "__main__":
    unittest.main()
