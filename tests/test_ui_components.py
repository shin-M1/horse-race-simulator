from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ui_components import default_horse_dataframe


class UIComponentsTest(unittest.TestCase):
    def test_carried_weight_input_exists(self) -> None:
        table = default_horse_dataframe(3)
        self.assertIn("carried_weight", table.columns)
        self.assertEqual(table["carried_weight"].tolist(), [56.0, 56.0, 56.0])


if __name__ == "__main__":
    unittest.main()
