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
from public_prediction import should_use_public_prediction
from runtime_mode import get_runtime_mode, is_streamlit_cloud, should_reload_modules


class EnvironmentAndPublicTrendTest(unittest.TestCase):
    def test_runtime_mode_cloud(self) -> None:
        self.assertEqual(get_runtime_mode(True), "CLOUD")

    def test_runtime_mode_local(self) -> None:
        self.assertEqual(get_runtime_mode(False), "LOCAL")
        self.assertIsInstance(is_streamlit_cloud(), bool)

    def test_cloud_hides_environment_selector(self) -> None:
        source = (ROOT / "app.py").read_text(encoding="utf-8")

        self.assertNotIn("実行環境モード", source)
        self.assertNotIn("ENVIRONMENT_MODES", source)
        self.assertNotIn("resolve_environment_mode", source)

    def test_local_shows_public_prediction_only_checkbox(self) -> None:
        source = (ROOT / "app.py").read_text(encoding="utf-8")

        self.assertIn("公開版AI予想のみ実行", source)
        self.assertIn('if runtime_mode == "CLOUD":', source)

    def test_cloud_skips_heavy_imports(self) -> None:
        source = (ROOT / "app.py").read_text(encoding="utf-8")

        top_import_section = source.split("APP_IMPORTS_COMPLETED_AT", 1)[0]
        self.assertNotIn("from video_renderer import", top_import_section)
        self.assertNotIn("from youtube_video_builder import", top_import_section)
        self.assertNotIn("from thumbnail_generator import", top_import_section)

    def test_cloud_skips_monte_carlo(self) -> None:
        self.assertTrue(should_use_public_prediction(is_cloud=True, public_prediction_only=False))

    def test_local_can_use_monte_carlo(self) -> None:
        self.assertFalse(should_use_public_prediction(is_cloud=False, public_prediction_only=False))

    def test_public_trend_database_loads_from_data_public(self) -> None:
        original_local = trend_database.TREND_DB_DIR
        original_public = trend_database.PUBLIC_TREND_DB_DIR
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                trend_database.TREND_DB_DIR = root / "data" / "trend_database"
                trend_database.PUBLIC_TREND_DB_DIR = root / "data_public" / "trend_database"
                public_path = Path(trend_database.public_trend_cache_path("test_race", "tokyo", 1600))
                public_path.parent.mkdir(parents=True, exist_ok=True)
                public_path.write_text(
                    json.dumps({"race_name": "test_race", "row_count": 12}),
                    encoding="utf-8",
                )

                loaded = trend_database.load_trend_cache("test_race", "tokyo", 1600)
        finally:
            trend_database.TREND_DB_DIR = original_local
            trend_database.PUBLIC_TREND_DB_DIR = original_public
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded["_database_source"], "data_public")
        self.assertEqual(loaded["_database_status"], "public_hit")

    def test_export_trend_database_to_public_dir(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "data" / "trend_database"
            public = root / "data_public" / "trend_database"
            source.mkdir(parents=True)
            (source / "race_a.json").write_text('{"row_count": 1}', encoding="utf-8")
            (source / "ignore.txt").write_text("x", encoding="utf-8")

            result = trend_database.export_trend_database_to_public_dir(
                source_dir=source,
                public_dir=public,
            )

            self.assertEqual(result["copied_count"], 1)
            self.assertEqual(result["public_dir"], str(public))
            self.assertTrue((public / "race_a.json").is_file())
            self.assertFalse((public / "ignore.txt").exists())

    def test_startup_reload_only_debug_mode(self) -> None:
        self.assertFalse(should_reload_modules(""))
        self.assertFalse(should_reload_modules("0"))
        self.assertTrue(should_reload_modules("1"))
        self.assertTrue(should_reload_modules("true"))


if __name__ == "__main__":
    unittest.main()
