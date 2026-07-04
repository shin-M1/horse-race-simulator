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

from evaluation import evaluate_prediction, save_evaluation_log, save_prediction_log
from metrics import aggregate_evaluation_logs
from ml_model import ML_FEATURES, build_ml_dataset, resolve_prediction_engine, train_prediction_model
from weight_optimizer import (
    build_training_dataset,
    load_model_weights,
    optimize_prediction_weights,
    save_model_weights,
)


def prediction_rows(offset: float = 0.0) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    marks = ["◎", "○", "▲", "△", "☆", ""]
    for index in range(1, 7):
        strength = 95.0 - index * 8.0 + offset
        rows.append(
            {
                "印": marks[index - 1],
                "馬番": index,
                "馬名": f"Horse {index}",
                "枠順": min(8, index),
                "斤量": 56.0,
                "prediction_score": strength,
                "win_rate": max(0.01, 0.35 - index * 0.04),
                "top3_rate": max(0.05, 0.75 - index * 0.08),
                "horse_ability_score": strength,
                "race_strength_score": strength - 2,
                "normalized_elo_score": strength - 1,
                "late_kick_score": strength - 3,
                "course_fit_score": strength - 4,
                "pace_fit_score": strength - 5,
                "jockey_score": 60 + index,
                "track_bias_fit_score": 55 + index,
                "weight_penalty": 0.0,
                "mud_aptitude": 50 + index,
                "popularity_score": strength,
                "finish_score": strength - 2,
                "margin_score": strength - 3,
                "time_score": strength - 4,
                "last3f_score": strength - 5,
                "carried_weight": 56.0,
                "frame": min(8, index),
                "horse_number": index,
                "avg_finish": float(index),
            }
        )
    return rows


def evaluation_logs() -> list[dict[str, object]]:
    logs: list[dict[str, object]] = []
    for race_index in range(3):
        actual = [
            {"horse_number": number, "finish": number}
            for number in range(1, 7)
        ]
        prediction = prediction_rows(float(race_index))
        metrics = evaluate_prediction({"prediction_table": prediction}, actual)
        logs.append(
            {
                "race_id": f"race_{race_index}",
                "race_name": f"Race {race_index}",
                "race_date": f"2026-01-0{race_index + 1}",
                "prediction_table": prediction,
                "actual_result": actual,
                "evaluation": metrics,
                "evaluation_metrics": metrics,
            }
        )
    return logs


class LearningPipelineTest(unittest.TestCase):
    def test_prediction_log_saved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = save_prediction_log(
                race_id="race_1",
                source_url="https://example.test/race_1",
                fetched_entries=[],
                race_metadata={"race_name": "Race 1", "race_date": "2026-01-01"},
                prediction_table=prediction_rows(),
                simulation_result={"horse_inputs": [], "race_timeline": []},
                output_dir=directory,
            )
            self.assertTrue(path.is_file())
            self.assertTrue(path.with_suffix(".csv").is_file())
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("AI予想印", payload)
            self.assertIn("horse_analysis", payload)

    def test_evaluation_log_saved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prediction = {"race_id": "race_1", "race_name": "Race 1", "race_date": "2026-01-01", "prediction_table": prediction_rows()}
            actual = [{"horse_number": number, "finish": number} for number in range(1, 7)]
            path = save_evaluation_log(
                prediction_log=prediction,
                actual_results=actual,
                output_dir=directory,
            )
            self.assertTrue(path.is_file())
            self.assertTrue(path.with_suffix(".csv").is_file())
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("race_review", payload)
            self.assertIn("evaluation_metrics", payload)

    def test_aggregate_evaluation_logs(self) -> None:
        metrics = aggregate_evaluation_logs(evaluation_logs())
        self.assertEqual(metrics["race_count"], 3)
        self.assertEqual(metrics["honmei_win_rate"], 1.0)
        self.assertEqual(metrics["second_mark_top3_rate"], 1.0)

    def test_build_training_dataset(self) -> None:
        dataset = build_training_dataset(evaluation_logs())
        self.assertEqual(len(dataset), 18)
        self.assertIn("horse_ability_score", dataset.columns)
        self.assertIn("race_trend_score", dataset.columns)
        self.assertNotIn("popularity_score", dataset.columns)
        self.assertIn("is_top3", dataset.columns)

    def test_popularity_score_not_used_in_prediction_features(self) -> None:
        self.assertNotIn("popularity_score", ML_FEATURES)
        self.assertIn("race_trend_score", ML_FEATURES)

    def test_optimize_prediction_weights(self) -> None:
        dataset = build_training_dataset(evaluation_logs())
        result = optimize_prediction_weights(dataset, n_trials=20)
        self.assertAlmostEqual(sum(result["weights"].values()), 1.0, places=5)
        self.assertGreaterEqual(result["score"], result["baseline_score"])

    def test_model_weights_saved_and_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.json"
            saved = save_model_weights({"weights": {"horse_ability_score": 1.0}}, path)
            loaded = load_model_weights(saved)
            self.assertTrue(saved.is_file())
            self.assertAlmostEqual(sum(loaded.values()), 1.0)
            self.assertGreater(loaded["horse_ability_score"], 0.5)

    def test_build_ml_dataset(self) -> None:
        X, y = build_ml_dataset(evaluation_logs())
        self.assertEqual(len(X), len(y))
        self.assertEqual(len(X), 18)
        self.assertIn("carried_weight", X.columns)

    def test_train_prediction_model_fallback(self) -> None:
        X, y = build_ml_dataset(evaluation_logs())
        model = train_prediction_model(X, y, model_type="logistic")
        probabilities = model.predict_proba(X)
        self.assertEqual(probabilities.shape, (len(X), 2))

    def test_prediction_engine_fallback_when_no_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.pkl"
            self.assertEqual(resolve_prediction_engine("ml_model", missing), "rule_based")


if __name__ == "__main__":
    unittest.main()
