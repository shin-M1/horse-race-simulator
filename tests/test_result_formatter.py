from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from result_formatter import (
    build_single_race_result_from_timeline,
    build_horse_comments_table,
    build_recent_races_table,
    extract_value,
    format_analysis_table,
    style_group_table,
)


class ResultFormatterTest(unittest.TestCase):
    def test_single_result_matches_timeline_final_frame(self) -> None:
        timeline = [
            {
                "horses": [
                    {"horse_number": 1, "horse_name": "A", "frame": 1, "actual_running_style": "逃げ", "position_m": 100.0, "rank": 1},
                    {"horse_number": 2, "horse_name": "B", "frame": 2, "actual_running_style": "追込", "position_m": 98.0, "rank": 2},
                ]
            },
            {
                "horses": [
                    {"horse_number": 1, "horse_name": "A", "frame": 1, "actual_running_style": "逃げ", "position_m": 2198.0, "rank": 2},
                    {"horse_number": 2, "horse_name": "B", "frame": 2, "actual_running_style": "追込", "position_m": 2200.0, "rank": 1},
                ]
            },
        ]

        table = build_single_race_result_from_timeline(timeline, [])

        self.assertEqual(table["馬番"].tolist(), [2, 1])
        self.assertEqual(table["着順"].tolist(), [1, 2])
        self.assertEqual(float(table.iloc[1]["gap_from_winner"]), 2.0)

    def test_extract_value_uses_first_non_empty_candidate(self) -> None:
        row = {"race_name": "", "name": None, "race": "大阪杯"}
        self.assertEqual(extract_value(row, ["race_name", "name", "race"], "-"), "大阪杯")
        self.assertEqual(extract_value(row, ["missing"], "-"), "-")

    def test_build_recent_races_table_handles_key_variants_and_limits_to_five(self) -> None:
        horse_results = [
            {
                "horse_name": "テストホース",
                "recent_races": [
                    {
                        "name": f"レース{i}",
                        "class": "G1",
                        "favorite": i,
                        "rank": i,
                        "corner_order": [12, 12, 11, 3],
                        "agari": 34.5,
                        "time": "1:58.3",
                        "race_level_weight": 1.4,
                        "margin_score": 93.0,
                        "opponent_strength_score": 82.5,
                        "race_score": 78.1,
                        "date": "2026-01-01",
                        "place": "阪神",
                        "distance": 2200,
                        "condition": "良",
                    }
                    for i in range(1, 7)
                ],
            }
        ]

        table = build_recent_races_table(horse_results)

        self.assertEqual(len(table), 5)
        self.assertEqual(
            list(table.columns),
            [
                "馬名",
                "レース名",
                "クラス",
                "人気順",
                "着順",
                "着差",
                "通過順",
                "上り",
                "タイム",
                "time_sec",
                "avg_speed",
                "late_gain",
                "recent_time_score",
                "レースレベル重み",
                "着差補正",
                "相手関係補正",
                "レース評価スコア",
                "日付",
                "競馬場",
                "距離",
                "馬場状態",
            ],
        )
        self.assertEqual(table.iloc[0]["馬名"], "テストホース")
        self.assertEqual(table.iloc[0]["レース名"], "レース1")
        self.assertEqual(table.iloc[0]["通過順"], "12-12-11-3")
        self.assertEqual(table.iloc[0]["タイム"], "1:58.3")
        self.assertEqual(table.iloc[0]["レースレベル重み"], "1.40")
        self.assertEqual(table.iloc[0]["着差補正"], "93.0")
        self.assertEqual(table.iloc[0]["相手関係補正"], "82.5")
        self.assertEqual(table.iloc[0]["レース評価スコア"], "78.1")
        self.assertEqual(table.iloc[4]["レース名"], "レース5")

    def test_format_analysis_table_does_not_show_passing_order_list(self) -> None:
        analysis = pd.DataFrame(
            [
                {
                    "horse_name": "テストホース",
                    "primary_running_style": "差し",
                    "debug_passing_orders": ["8-8-7-4"],
                    "weighted_avg_first_ratio": 0.5,
                }
            ]
        )

        table = format_analysis_table(analysis)

        self.assertNotIn("debug_passing_orders", table.columns)
        self.assertNotIn("通過順一覧", table.columns)

    def test_style_group_table_uses_three_race_flow_groups(self) -> None:
        table = style_group_table(
            {
                "front_group": ["逃げ馬"],
                "middle_group": ["差し馬"],
                "back_group": ["追込馬"],
            }
        )

        self.assertEqual(table["区分"].tolist(), ["先頭候補", "中団候補", "後方候補"])
        self.assertNotIn("front_candidates", table.to_string())
        self.assertNotIn("closer_candidates", table.to_string())

    def test_build_horse_comments_table_outputs_all_horse_comments(self) -> None:
        prediction_table = pd.DataFrame(
            [
                {
                    "印": "◎",
                    "馬番": 1,
                    "馬名": "テストホース",
                    "枠順": 1,
                    "primary_running_style": "先行",
                    "win_rate": 0.25,
                    "top3_rate": 0.60,
                    "score": 82.0,
                }
            ]
        )
        horse_analysis = pd.DataFrame(
            [
                {
                    "horse_name": "テストホース",
                    "frame": 1,
                    "primary_running_style": "先行",
                    "adjusted_逃げ": 0.20,
                    "adjusted_先行": 0.55,
                    "adjusted_差し": 0.20,
                    "adjusted_追込": 0.05,
                }
            ]
        )

        table = build_horse_comments_table(
            prediction_table,
            horse_analysis,
            {"pace": "slow"},
            {"track_condition": "良", "distance": 2000, "race_course_day": "1日目", "course_layout": "A"},
        )

        self.assertEqual(list(table.columns), ["印", "馬番", "馬名", "斤量", "脚質", "勝率", "複勝率", "評価", "短評"])
        self.assertEqual(table.iloc[0]["評価"], "A")
        self.assertIn("勝ち切り候補", table.iloc[0]["短評"])


if __name__ == "__main__":
    unittest.main()
