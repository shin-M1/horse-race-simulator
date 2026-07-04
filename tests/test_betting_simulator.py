from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from betting_simulator import (
    aggregate_return_rates,
    calculate_return_rate,
    generate_bets_from_prediction,
)
from payout_fetcher import fetch_race_payouts, parse_payouts_from_html


def prediction_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"印": "◎", "馬番": 7, "馬名": "A"},
            {"印": "○", "馬番": 3, "馬名": "B"},
            {"印": "▲", "馬番": 11, "馬名": "C"},
            {"印": "△", "馬番": 5, "馬名": "D"},
            {"印": "☆", "馬番": 1, "馬名": "E"},
        ]
    )


def actual_result() -> list[dict]:
    return [
        {"finish": 1, "horse_number": 7},
        {"finish": 2, "horse_number": 3},
        {"finish": 3, "horse_number": 11},
        {"finish": 4, "horse_number": 5},
    ]


def payouts() -> dict:
    return {
        "単勝": [{"combination": "7", "payout": 420}],
        "複勝": [
            {"combination": "7", "payout": 180},
            {"combination": "3", "payout": 220},
            {"combination": "11", "payout": 310},
        ],
        "馬連": [{"combination": "3-7", "payout": 1280}],
        "ワイド": [
            {"combination": "3-7", "payout": 520},
            {"combination": "7-11", "payout": 840},
            {"combination": "3-11", "payout": 960},
        ],
        "三連複": [{"combination": "3-7-11", "payout": 4280}],
    }


class BettingSimulatorTest(unittest.TestCase):
    def test_fetch_race_payouts_no_dummy(self) -> None:
        self.assertEqual(fetch_race_payouts("invalid"), {})

    def test_parse_payouts_from_html(self) -> None:
        html = """
        <table>
          <tr><th>単勝</th><td>7</td><td>420円</td></tr>
          <tr><th>馬連</th><td>3-7</td><td>1,280円</td></tr>
        </table>
        """
        parsed = parse_payouts_from_html(html)

        self.assertEqual(parsed["単勝"][0]["combination"], "7")
        self.assertEqual(parsed["単勝"][0]["payout"], 420)
        self.assertEqual(parsed["馬連"][0]["combination"], "3-7")

    def test_generate_bets_from_prediction(self) -> None:
        bets = generate_bets_from_prediction(prediction_table(), box_size=5, stake_per_bet=100)

        self.assertEqual(len(bets["単勝"]), 1)
        self.assertEqual(len(bets["複勝"]), 1)
        self.assertEqual(len(bets["馬連"]), 10)
        self.assertEqual(len(bets["ワイド"]), 10)
        self.assertEqual(len(bets["三連複"]), 10)
        self.assertEqual(bets["単勝"][0]["combination"], "7")

    def test_tansho_hit(self) -> None:
        analysis = calculate_return_rate(generate_bets_from_prediction(prediction_table()), payouts(), actual_result())

        tansho = next(row for row in analysis["by_bet_type"] if row["bet_type"] == "単勝")
        self.assertTrue(tansho["hit"])
        self.assertEqual(tansho["return"], 420)

    def test_fukusho_hit(self) -> None:
        analysis = calculate_return_rate(generate_bets_from_prediction(prediction_table()), payouts(), actual_result())

        fukusho = next(row for row in analysis["by_bet_type"] if row["bet_type"] == "複勝")
        self.assertTrue(fukusho["hit"])
        self.assertEqual(fukusho["return"], 180)

    def test_umaren_box_hit(self) -> None:
        analysis = calculate_return_rate(generate_bets_from_prediction(prediction_table()), payouts(), actual_result())

        umaren = next(row for row in analysis["by_bet_type"] if row["bet_type"] == "馬連")
        self.assertTrue(umaren["hit"])
        self.assertEqual(umaren["return"], 1280)

    def test_wide_box_multiple_hits(self) -> None:
        analysis = calculate_return_rate(generate_bets_from_prediction(prediction_table()), payouts(), actual_result())

        wide = next(row for row in analysis["by_bet_type"] if row["bet_type"] == "ワイド")
        self.assertTrue(wide["hit"])
        self.assertEqual(wide["return"], 2320)

    def test_sanrenpuku_box_hit(self) -> None:
        analysis = calculate_return_rate(generate_bets_from_prediction(prediction_table()), payouts(), actual_result())

        sanrenpuku = next(row for row in analysis["by_bet_type"] if row["bet_type"] == "三連複")
        self.assertTrue(sanrenpuku["hit"])
        self.assertEqual(sanrenpuku["return"], 4280)

    def test_calculate_return_rate(self) -> None:
        analysis = calculate_return_rate(generate_bets_from_prediction(prediction_table()), payouts(), actual_result())

        self.assertEqual(analysis["summary"]["total_stake"], 3200)
        self.assertEqual(analysis["summary"]["total_return"], 8480)
        self.assertAlmostEqual(analysis["summary"]["return_rate"], 265.0)

    def test_aggregate_return_rates(self) -> None:
        analysis = calculate_return_rate(generate_bets_from_prediction(prediction_table()), payouts(), actual_result())
        table = aggregate_return_rates([{"return_analysis": analysis}])

        self.assertIn("全体", set(table["券種"]))
        overall = table[table["券種"] == "全体"].iloc[0]
        self.assertEqual(int(overall["購入額"]), 3200)
        self.assertEqual(int(overall["払戻額"]), 8480)


if __name__ == "__main__":
    unittest.main()
