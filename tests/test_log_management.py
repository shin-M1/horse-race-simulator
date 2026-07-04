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

from evaluation import save_evaluation_log, save_prediction_log
from log_manager import (
    DuplicateLogError,
    delete_log_files,
    find_duplicate_logs,
    log_exists,
)


class LogManagementTest(unittest.TestCase):
    def _save_prediction(self, directory: str, duplicate_action: str = "skip") -> Path:
        return save_prediction_log(
            race_id="race-1",
            source_url="https://example.test/race-1",
            fetched_entries=[],
            race_metadata={"race_name": "Test Race", "race_date": "2026-06-22"},
            prediction_table=[{"馬番": 1, "馬名": "Horse 1", "prediction_score": 70}],
            simulation_result={"race_timeline": []},
            output_dir=directory,
            duplicate_action=duplicate_action,
        )

    def test_duplicate_prediction_is_rejected_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self._save_prediction(directory)
            with self.assertRaises(DuplicateLogError):
                self._save_prediction(directory)
            self.assertEqual(len(list(Path(directory).glob("*.json"))), 1)

    def test_prediction_can_be_overwritten_or_renamed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = self._save_prediction(directory)
            overwritten = self._save_prediction(directory, "上書き保存")
            self.assertEqual(first, overwritten)
            renamed = self._save_prediction(directory, "別名で保存")
            self.assertNotEqual(first, renamed)
            self.assertEqual(len(list(Path(directory).glob("*.json"))), 2)

    def test_evaluation_duplicate_includes_prediction_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prediction = {
                "race_id": "race-1",
                "race_name": "Test Race",
                "race_date": "2026-06-22",
                "_path": "outputs/prediction_logs/a.json",
                "prediction_table": [{"印": "◎", "馬番": 1, "prediction_score": 70}],
            }
            actual = [{"horse_number": 1, "finish": 1}]
            save_evaluation_log(prediction_log=prediction, actual_results=actual, output_dir=directory)
            with self.assertRaises(DuplicateLogError):
                save_evaluation_log(prediction_log=prediction, actual_results=actual, output_dir=directory)

            other_prediction = dict(prediction, _path="outputs/prediction_logs/b.json")
            save_evaluation_log(prediction_log=other_prediction, actual_results=actual, output_dir=directory)
            self.assertEqual(len(list(Path(directory).glob("*.json"))), 2)

    def test_duplicate_detection_and_log_exists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self._save_prediction(directory)
            self._save_prediction(directory, "rename")
            self.assertTrue(log_exists("race-1", "Test Race", "2026-06-22", directory))
            duplicates = find_duplicate_logs(directory, "prediction")
            self.assertEqual(len(duplicates), 1)
            self.assertEqual(duplicates[0]["count"], 2)

    def test_delete_removes_json_csv_and_ignores_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._save_prediction(directory)
            result = delete_log_files([str(path), str(Path(directory) / "missing.json")])
            self.assertFalse(path.exists())
            self.assertFalse(path.with_suffix(".csv").exists())
            self.assertEqual(result["failed"], {})
            self.assertGreaterEqual(len(result["missing"]), 1)

    def test_delete_removes_timeline_companion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = save_prediction_log(
                race_id="race-2",
                source_url=None,
                fetched_entries=[],
                race_metadata={"race_name": "Timeline Race", "race_date": "2026-06-22"},
                prediction_table=[],
                simulation_result={"race_timeline": [{"time": 0, "horses": []}]},
                output_dir=directory,
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            timeline_path = Path(payload["race_timeline_path"])
            self.assertTrue(timeline_path.exists())
            delete_log_files([str(path)])
            self.assertFalse(timeline_path.exists())


if __name__ == "__main__":
    unittest.main()
