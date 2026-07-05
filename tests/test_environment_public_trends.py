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

import trend_database
from runtime_mode import resolve_environment_mode, should_reload_modules


class EnvironmentAndPublicTrendTest(unittest.TestCase):
    def test_environment_mode_auto(self) -> None:
        cloud = resolve_environment_mode("自動判定", cloud_detected=True)
        local = resolve_environment_mode("自動判定", cloud_detected=False)

        self.assertEqual(cloud["effective_mode"], "Cloud公開版")
        self.assertTrue(cloud["public_prediction_only"])
        self.assertEqual(local["effective_mode"], "ローカル高品質版")
        self.assertFalse(local["public_prediction_only"])

    def test_environment_mode_cloud_public(self) -> None:
        state = resolve_environment_mode("Cloud公開版", cloud_detected=False)

        self.assertEqual(state["effective_mode"], "Cloud公開版")
        self.assertTrue(state["public_prediction_only"])
        self.assertFalse(state["allow_heavy_features"])

    def test_environment_mode_local_high_quality(self) -> None:
        local = resolve_environment_mode("ローカル高品質版", cloud_detected=False)
        cloud = resolve_environment_mode("ローカル高品質版", cloud_detected=True)

        self.assertEqual(local["effective_mode"], "ローカル高品質版")
        self.assertTrue(local["allow_heavy_features"])
        self.assertEqual(cloud["effective_mode"], "Cloud公開版")
        self.assertTrue(cloud["public_prediction_only"])
        self.assertTrue(cloud["warning"])

    def test_cloud_skips_heavy_imports(self) -> None:
        source = (ROOT / "app.py").read_text(encoding="utf-8")

        top_import_section = source.split("APP_IMPORTS_COMPLETED_AT", 1)[0]
        self.assertNotIn("from video_renderer import", top_import_section)
        self.assertNotIn("from youtube_video_builder import", top_import_section)
        self.assertNotIn("from thumbnail_generator import", top_import_section)
        self.assertIn("from video_renderer import", source)
        self.assertIn("from youtube_video_builder import", source)

    def test_public_trend_database_loads_from_data_public(self) -> None:
        original_local = trend_database.TREND_DB_DIR
        original_public = trend_database.PUBLIC_TREND_DB_DIR
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                trend_database.TREND_DB_DIR = root / "data" / "trend_database"
                trend_database.PUBLIC_TREND_DB_DIR = root / "data_public" / "trend_database"
                public_path = Path(trend_database.public_trend_cache_path("テストS", "東京", 1600))
                public_path.parent.mkdir(parents=True, exist_ok=True)
                public_path.write_text(json.dumps({"race_name": "テストS", "row_count": 12}, ensure_ascii=False), encoding="utf-8")

                loaded = trend_database.load_trend_cache("テストS", "東京", 1600)
        finally:
            trend_database.TREND_DB_DIR = original_local
            trend_database.PUBLIC_TREND_DB_DIR = original_public
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["_database_source"], "data_public")
        self.assertEqual(loaded["_database_status"], "public_hit")

    def test_export_trend_database_to_public_dir(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "data" / "trend_database"
            target = root / "data_public" / "trend_database"
            source.mkdir(parents=True)
            (source / "race_a.json").write_text('{"row_count": 1}', encoding="utf-8")
            (source / "ignore.txt").write_text("x", encoding="utf-8")

            result = trend_database.export_trend_database_to_public_dir(source, target)

            self.assertEqual(result["copied_count"], 1)
            self.assertTrue((target / "race_a.json").is_file())

    def test_startup_reload_only_debug_mode(self) -> None:
        self.assertFalse(should_reload_modules(""))
        self.assertFalse(should_reload_modules("0"))
        self.assertTrue(should_reload_modules("1"))
        self.assertTrue(should_reload_modules("true"))


if __name__ == "__main__":
    unittest.main()
