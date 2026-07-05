from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ui_components import default_horse_dataframe, make_arrow_safe_dataframe


class UIComponentsTest(unittest.TestCase):
    def test_carried_weight_input_exists(self) -> None:
        table = default_horse_dataframe(3)
        self.assertIn("carried_weight", table.columns)
        self.assertEqual(table["carried_weight"].tolist(), [56.0, 56.0, 56.0])

    def test_make_arrow_safe_dataframe_stringifies_mixed_count_column(self) -> None:
        table = pd.DataFrame(
            {
                "件数": [3, "4", None],
                "score": [10.5, 20.0, 30.0],
            }
        )

        safe = make_arrow_safe_dataframe(table)

        self.assertEqual(safe["件数"].tolist(), ["3", "4", ""])
        self.assertEqual(safe["score"].tolist(), [10.5, 20.0, 30.0])
        try:
            import pyarrow as pa
        except ImportError:
            return
        pa.Table.from_pandas(safe)


if __name__ == "__main__":
    unittest.main()
