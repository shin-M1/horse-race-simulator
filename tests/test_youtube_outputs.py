from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from thumbnail_generator import EXTERNAL_MEDIA_ASSETS_USED as THUMB_EXTERNAL_ASSETS
from thumbnail_generator import REVEALS_PREDICTION_MARKS_OR_HORSE_NAMES
from thumbnail_generator import generate_youtube_thumbnail
from youtube_video_builder import (
    EXTERNAL_MEDIA_ASSETS_USED as VIDEO_EXTERNAL_ASSETS,
    REQUIRED_SECTION_ORDER,
    build_youtube_prediction_video,
    build_youtube_video_structure,
    generate_race_trend_summary,
)


def prediction_log() -> dict[str, object]:
    rows = [
        {"印": "◎", "馬番": 7, "馬名": "サンプルホースA", "枠順": 4, "actual_running_style": "差し", "win_rate": 0.24, "top3_rate": 0.62, "horse_ability_score": 82, "late_kick_score": 76},
        {"印": "○", "馬番": 3, "馬名": "サンプルホースB", "枠順": 2, "actual_running_style": "先行", "win_rate": 0.18, "top3_rate": 0.58, "horse_ability_score": 78, "pace_fit_score": 74},
        {"印": "▲", "馬番": 11, "馬名": "サンプルホースC", "枠順": 6, "actual_running_style": "追込", "win_rate": 0.13, "top3_rate": 0.44, "late_kick_score": 81},
        {"印": "△", "馬番": 5, "馬名": "サンプルホースD", "枠順": 3, "actual_running_style": "逃げ", "win_rate": 0.10, "top3_rate": 0.35},
        {"印": "☆", "馬番": 12, "馬名": "サンプルホースE", "枠順": 7, "actual_running_style": "自在", "win_rate": 0.08, "top3_rate": 0.30},
    ]
    return {
        "race_name": "宝塚記念",
        "race_date": "2026-06-28",
        "race_config": {"course": "阪神", "surface": "芝", "distance": 2200, "direction": "右", "track_bias": "外差し有利", "track_condition": "良"},
        "prediction_table": rows,
        "comments_table": [
            {"印": row["印"], "馬番": row["馬番"], "馬名": row["馬名"], "脚質": row["actual_running_style"], "評価": "A", "短評": "テスト短評"}
            for row in rows
        ],
        "pace_prediction": {"pace": "high", "front_group": ["サンプルホースD"], "middle_group": ["サンプルホースB"], "back_group": ["サンプルホースA", "サンプルホースC"]},
        "single_result": [{"着順": 1, "馬番": 7}, {"着順": 2, "馬番": 3}],
        "same_race_trend_analysis": {
            "summary_bullets": ["過去10年では外差しの好走が目立つ", "上り上位馬の複勝率が高い"],
            "trend_scores": {"outer_advantage": 68, "agari_importance": 72},
        },
    }


class YoutubeOutputTest(unittest.TestCase):
    def test_generate_youtube_thumbnail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = generate_youtube_thumbnail(prediction_log(), str(Path(directory) / "thumb.png"))
            self.assertTrue(Path(path).is_file())

    def test_thumbnail_file_exists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(generate_youtube_thumbnail(prediction_log(), directory))
            self.assertEqual(path.suffix, ".png")
            self.assertGreater(path.stat().st_size, 0)

    def test_generate_race_trend_summary(self) -> None:
        summary = generate_race_trend_summary(prediction_log())
        self.assertEqual(summary["title"], "同レースの過去傾向")
        self.assertEqual(summary["data_source"], "same_race_history")
        self.assertTrue(summary["bullets"])

    def test_full_diagnosis_is_horse_number_order_without_mark(self) -> None:
        structure = build_youtube_video_structure(prediction_log(), race_video_path="")
        diagnosis_rows = structure[1]["rows"]

        self.assertEqual([row["馬番"] for row in diagnosis_rows], [3, 5, 7, 11, 12])
        self.assertTrue(all("印" not in row for row in diagnosis_rows))

    def test_featured_horses_use_requested_order_and_detailed_reasons(self) -> None:
        structure = build_youtube_video_structure(prediction_log(), race_video_path="")
        featured_rows = structure[3]["rows"]

        self.assertEqual([row["印"] for row in featured_rows], ["☆", "△", "▲", "○", "◎"])
        for row in featured_rows:
            short_comment = "テスト短評"
            self.assertIn("detailed_reason", row)
            self.assertIn("risk", row)
            self.assertNotEqual(row["detailed_reason"], short_comment)
            self.assertTrue(row["risk"])

    def test_thumbnail_policy_does_not_reveal_predictions(self) -> None:
        self.assertFalse(REVEALS_PREDICTION_MARKS_OR_HORSE_NAMES)

    def test_build_youtube_prediction_video(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = build_youtube_prediction_video(
                prediction_log(),
                race_video_path="",
                output_path=str(Path(directory) / "youtube.mp4"),
                fps=1,
                trend_section_sec=1,
                diagnosis_section_sec=1,
                featured_section_sec=1,
            )
            self.assertTrue(Path(path).is_file())

    def test_youtube_video_file_exists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(
                build_youtube_prediction_video(
                    prediction_log(),
                    race_video_path="",
                    output_path=directory,
                    fps=1,
                    trend_section_sec=1,
                    diagnosis_section_sec=1,
                    featured_section_sec=1,
                )
            )
            self.assertGreater(path.stat().st_size, 0)

    def test_youtube_video_uses_required_section_order(self) -> None:
        structure = build_youtube_video_structure(prediction_log(), race_video_path="")
        self.assertEqual([item["section"] for item in structure], REQUIRED_SECTION_ORDER)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(
                build_youtube_prediction_video(
                    prediction_log(),
                    race_video_path="",
                    output_path=str(Path(directory) / "youtube.mp4"),
                    fps=1,
                    trend_section_sec=1,
                    diagnosis_section_sec=1,
                    featured_section_sec=1,
                )
            )
            metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["section_order"], REQUIRED_SECTION_ORDER)

    def test_youtube_video_uses_narration_based_section_durations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(
                build_youtube_prediction_video(
                    prediction_log(),
                    race_video_path="",
                    output_path=str(Path(directory) / "youtube.mp4"),
                    fps=1,
                    use_narration=False,
                )
            )
            metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))

            self.assertGreaterEqual(metadata["section_durations"]["trend"], 6)
            self.assertGreaterEqual(metadata["section_durations"]["diagnosis"], 10)
            self.assertGreaterEqual(metadata["section_durations"]["featured"], 12)
            self.assertTrue(metadata["section_scripts"]["trend"])

    def test_youtube_video_records_uploaded_audio_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bgm = Path(directory) / "bgm.wav"
            race_bgm = Path(directory) / "race.wav"
            start = Path(directory) / "start.wav"
            for path in [bgm, race_bgm, start]:
                path.write_bytes(b"not-a-real-wav")

            path = Path(
                build_youtube_prediction_video(
                    prediction_log(),
                    race_video_path="",
                    output_path=str(Path(directory) / "youtube.mp4"),
                    fps=1,
                    trend_section_sec=1,
                    diagnosis_section_sec=1,
                    featured_section_sec=1,
                    bgm_path=str(bgm),
                    race_bgm_path=str(race_bgm),
                    start_se_path=str(start),
                    use_narration=False,
                )
            )
            metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))

            self.assertEqual(metadata["bgm_path"], str(bgm))
            self.assertEqual(metadata["race_bgm_path"], str(race_bgm))
            self.assertEqual(metadata["start_se_path"], str(start))

    def test_no_external_media_assets_used(self) -> None:
        self.assertFalse(THUMB_EXTERNAL_ASSETS)
        self.assertFalse(VIDEO_EXTERNAL_ASSETS)


if __name__ == "__main__":
    unittest.main()
