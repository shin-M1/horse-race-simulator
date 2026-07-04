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

from analysis_reporter import analyze_prediction_failure, generate_race_review
from evaluation import save_evaluation_log
from metrics import aggregate_failure_tags
from report_generator import generate_prediction_report, save_prediction_report


def prediction_rows() -> list[dict[str, object]]:
    styles = ["先行", "逃げ", "先行", "差し", "追込", "差し", "自在", "追込", "先行"]
    marks = ["◎", "○", "▲", "△", "☆", "", "", "", ""]
    rows = []
    for index in range(1, 10):
        rows.append(
            {
                "印": marks[index - 1],
                "馬番": index,
                "馬名": f"Horse {index}",
                "primary_running_style": styles[index - 1],
                "actual_running_style": styles[index - 1],
                "prediction_score": 100 - index * 5,
                "win_rate": 0.25 - index * 0.01,
                "top3_rate": 0.7 - index * 0.04,
                "horse_ability_score": 90 - index * 3,
                "late_kick_score": 55 + index,
                "course_fit_score": 60,
                "pace_fit_score": 60,
                "popularity": index,
                "popularity_score": max(30, 95 - index * 7),
                "race_strength_score": 60,
                "avg_opponent_strength_score": 60,
                "jockey_score": 50,
                "斤量": 56.0,
            }
        )
    return rows


def prediction_log() -> dict[str, object]:
    return {
        "race_id": "race-1",
        "race_name": "宝塚記念",
        "race_date": "2026-06-21",
        "race_config": {
            "course": "阪神", "surface": "芝", "distance": 2200,
            "weather": "晴", "track_condition": "良", "track_bias": "標準",
        },
        "prediction_table": prediction_rows(),
        "pace_prediction": {
            "pace": "high", "front_group": ["Horse 2"],
            "middle_group": ["Horse 1", "Horse 3"], "back_group": ["Horse 4", "Horse 5"],
        },
        "comments_table": [
            {"印": row["印"], "馬番": row["馬番"], "馬名": row["馬名"], "脚質": row["actual_running_style"], "短評": "テスト短評"}
            for row in prediction_rows()
        ],
        "single_result": [{"着順": 1, "馬番": 2}, {"着順": 2, "馬番": 1}],
        "video_path": "outputs/race.mp4",
    }


class PredictionReportTest(unittest.TestCase):
    def test_generate_prediction_report(self) -> None:
        report = generate_prediction_report(prediction_log())
        self.assertIn("宝塚記念", report["title"])
        self.assertIn("阪神", report["race_info"])
        self.assertIn("オープニング", report["youtube_script"])
        self.assertIn("#競馬予想", report["sns_text"])

    def test_generate_prediction_report_accepts_dataframes(self) -> None:
        log = prediction_log()
        log["prediction_table"] = pd.DataFrame(log["prediction_table"])
        log["comments_table"] = pd.DataFrame(log["comments_table"])
        log["single_result"] = pd.DataFrame(log["single_result"])
        report = generate_prediction_report(log)
        self.assertIn("AI期待値最大", report["simulation_summary"])

    def test_prediction_report_contains_marks(self) -> None:
        report = generate_prediction_report(prediction_log())
        marks = report["marks_table"]
        self.assertIsInstance(marks, pd.DataFrame)
        self.assertEqual(marks["印"].tolist(), ["◎", "○", "▲", "△", "☆"])
        self.assertIn("短評", marks.columns)

    def test_prediction_report_saves_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = save_prediction_report(
                generate_prediction_report(prediction_log()),
                "outputs/prediction_logs/test.json",
                directory,
            )
            self.assertTrue(paths["markdown"].is_file())
            self.assertTrue(paths["json"].is_file())
            self.assertIn("YouTube台本", paths["markdown"].read_text(encoding="utf-8"))
            payload = json.loads(paths["json"].read_text(encoding="utf-8"))
            self.assertEqual(payload["prediction_log_path"], "outputs/prediction_logs/test.json")


class FailureAnalysisTest(unittest.TestCase):
    def test_analyze_prediction_failure_adds_tags(self) -> None:
        actual = [{"horse_number": 1, "finish": 8}, {"horse_number": 2, "finish": 1}]
        analysis = analyze_prediction_failure(prediction_log(), actual, {})
        self.assertIn("能力評価過大", analysis["miss_reason_tags"])
        self.assertTrue(analysis["improvement_suggestions"])

    def test_pace_misread_tag(self) -> None:
        actual = [
            {"horse_number": 2, "finish": 1},
            {"horse_number": 1, "finish": 2},
            {"horse_number": 3, "finish": 3},
        ]
        analysis = analyze_prediction_failure(prediction_log(), actual, {})
        self.assertIn("ペース読み違い", analysis["miss_reason_tags"])
        self.assertIn("前残り過小評価", analysis["miss_reason_tags"])

    def test_overrated_favorite_tag(self) -> None:
        actual = [{"horse_number": 1, "finish": 9}]
        analysis = analyze_prediction_failure(prediction_log(), actual, {})
        self.assertIn("能力評価過大", analysis["miss_reason_tags"])

    def test_underrated_longshot_tag(self) -> None:
        log = prediction_log()
        rows = log["prediction_table"]
        rows[7]["popularity"] = 12
        rows[7]["popularity_score"] = 30
        actual = [{"horse_number": 8, "finish": 2}]
        analysis = analyze_prediction_failure(log, actual, {})
        self.assertIn("能力評価過小", analysis["miss_reason_tags"])
        self.assertIn("人気薄好走見逃し", analysis["miss_reason_tags"])

    def test_failure_tag_aggregation(self) -> None:
        logs = [
            {"failure_analysis": {"miss_reason_tags": ["能力評価過大", "ペース読み違い"]}},
            {"evaluation": {"failure_analysis": {"miss_reason_tags": ["能力評価過大"]}}},
        ]
        table = aggregate_failure_tags(logs)
        count = int(table.loc[table["タグ名"] == "能力評価過大", "回数"].iloc[0])
        self.assertEqual(count, 2)

    def test_race_review_contains_extended_fields(self) -> None:
        actual = [{"horse_number": 1, "finish": 8}, {"horse_number": 2, "finish": 1}]
        review = generate_race_review(prediction_log(), actual, {"mark_finishes": {"◎": 8}})
        self.assertIn("failure_analysis", review)
        self.assertIn("prediction_reason", review["horse_reviews"][0])
        self.assertIn("next_adjustment", review["horse_reviews"][0])

    def test_evaluation_log_saves_failure_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = prediction_log()
            log["_path"] = "outputs/prediction_logs/test.json"
            actual = [{"horse_number": 1, "finish": 9}, {"horse_number": 2, "finish": 1}]
            path = save_evaluation_log(prediction_log=log, actual_results=actual, output_dir=directory)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("failure_analysis", payload)
            self.assertIn("race_review", payload)
            self.assertIn("能力評価過大", payload["failure_analysis"]["miss_reason_tags"])


if __name__ == "__main__":
    unittest.main()
