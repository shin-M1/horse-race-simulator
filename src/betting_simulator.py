from __future__ import annotations

from itertools import combinations
from typing import Any

import pandas as pd


BET_TYPES = ("単勝", "複勝", "馬連", "ワイド", "三連複")
MARK_ORDER = ("◎", "○", "▲", "△", "☆")


def generate_bets_from_prediction(
    prediction_table: pd.DataFrame,
    box_size: int = 5,
    stake_per_bet: int = 100,
) -> dict[str, list[dict[str, Any]]]:
    """Generate standard bets from marked prediction rows."""
    table = prediction_table.copy() if isinstance(prediction_table, pd.DataFrame) else pd.DataFrame(prediction_table)
    if table.empty:
        return {bet_type: [] for bet_type in BET_TYPES}

    mark_column = _first_existing_column(table, ["印", "mark", "蜊ｰ"])
    horse_column = _first_existing_column(table, ["馬番", "horse_number", "鬥ｬ逡ｪ"])
    if horse_column is None:
        return {bet_type: [] for bet_type in BET_TYPES}

    marked = []
    if mark_column is not None:
        for mark in MARK_ORDER:
            match = table[table[mark_column].astype(str) == mark]
            if not match.empty:
                marked.append(match.iloc[0])
    if not marked:
        marked = [row for _, row in table.head(box_size).iterrows()]
    marked = marked[: max(1, int(box_size))]
    numbers = [_to_int(row.get(horse_column)) for row in marked]
    numbers = [number for number in numbers if number > 0]
    honmei = numbers[0] if numbers else 0

    bets: dict[str, list[dict[str, Any]]] = {bet_type: [] for bet_type in BET_TYPES}
    if honmei:
        bets["単勝"].append(_bet("単勝", [honmei], stake_per_bet))
        bets["複勝"].append(_bet("複勝", [honmei], stake_per_bet))
    for left, right in combinations(numbers, 2):
        bets["馬連"].append(_bet("馬連", [left, right], stake_per_bet))
        bets["ワイド"].append(_bet("ワイド", [left, right], stake_per_bet))
    for trio in combinations(numbers, 3):
        bets["三連複"].append(_bet("三連複", list(trio), stake_per_bet))
    return bets


def calculate_return_rate(
    bets: dict,
    payouts: dict,
    actual_result: list[dict],
    stake_per_bet: int = 100,
) -> dict[str, Any]:
    """Calculate total and per-bet-type return rates."""
    result_order = _actual_order(actual_result)
    payout_lookup = {
        bet_type: {_normalize_combination(row.get("combination", "")): _to_int(row.get("payout")) for row in rows}
        for bet_type, rows in (payouts or {}).items()
        if isinstance(rows, list)
    }
    winning_combinations = _winning_combinations(result_order)
    by_type: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    total_stake = 0
    total_return = 0
    for bet_type in BET_TYPES:
        bet_rows = [dict(row) for row in (bets or {}).get(bet_type, [])]
        stake = sum(_to_int(row.get("stake", stake_per_bet)) or stake_per_bet for row in bet_rows)
        payout_return = 0
        hit = False
        for row in bet_rows:
            combination = _normalize_combination(row.get("combination", ""))
            is_hit = combination in winning_combinations.get(bet_type, set())
            payout = payout_lookup.get(bet_type, {}).get(combination, 0) if is_hit else 0
            row_stake = _to_int(row.get("stake", stake_per_bet)) or stake_per_bet
            row_return = int(round(payout * row_stake / 100)) if payout > 0 else 0
            hit = hit or is_hit
            payout_return += row_return
            details.append(
                {
                    "bet_type": bet_type,
                    "combination": combination,
                    "stake": row_stake,
                    "hit": is_hit,
                    "payout": payout,
                    "return": row_return,
                }
            )
        total_stake += stake
        total_return += payout_return
        by_type.append(
            {
                "bet_type": bet_type,
                "num_bets": len(bet_rows),
                "stake": stake,
                "return": payout_return,
                "return_rate": round(payout_return / stake * 100, 2) if stake else 0.0,
                "hit": hit,
            }
        )
    return {
        "summary": {
            "total_stake": total_stake,
            "total_return": total_return,
            "return_rate": round(total_return / total_stake * 100, 2) if total_stake else 0.0,
        },
        "by_bet_type": by_type,
        "details": details,
    }


def aggregate_return_rates(evaluation_logs: list[dict]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    totals = {
        bet_type: {"stake": 0, "return": 0, "hit_count": 0, "race_count": 0}
        for bet_type in BET_TYPES
    }
    for log in evaluation_logs:
        analysis = log.get("return_analysis") or (log.get("evaluation") or {}).get("return_analysis") or {}
        for row in analysis.get("by_bet_type", []) if isinstance(analysis, dict) else []:
            bet_type = str(row.get("bet_type", ""))
            if bet_type not in totals:
                continue
            totals[bet_type]["stake"] += _to_int(row.get("stake"))
            totals[bet_type]["return"] += _to_int(row.get("return"))
            totals[bet_type]["hit_count"] += int(bool(row.get("hit")))
            totals[bet_type]["race_count"] += 1
    overall_stake = sum(item["stake"] for item in totals.values())
    overall_return = sum(item["return"] for item in totals.values())
    for bet_type in BET_TYPES:
        item = totals[bet_type]
        rows.append(
            {
                "券種": bet_type,
                "購入額": item["stake"],
                "払戻額": item["return"],
                "回収率": round(item["return"] / item["stake"] * 100, 2) if item["stake"] else 0.0,
                "的中率": round(item["hit_count"] / item["race_count"] * 100, 2) if item["race_count"] else 0.0,
                "対象レース数": item["race_count"],
            }
        )
    rows.append(
        {
            "券種": "全体",
            "購入額": overall_stake,
            "払戻額": overall_return,
            "回収率": round(overall_return / overall_stake * 100, 2) if overall_stake else 0.0,
            "的中率": "",
            "対象レース数": max((item["race_count"] for item in totals.values()), default=0),
        }
    )
    return pd.DataFrame(rows)


def bets_to_dataframe(bets: dict) -> pd.DataFrame:
    rows = []
    for bet_type in BET_TYPES:
        for row in (bets or {}).get(bet_type, []):
            rows.append({"券種": bet_type, "買い目": row.get("combination", ""), "購入額": row.get("stake", 0)})
    return pd.DataFrame(rows, columns=["券種", "買い目", "購入額"])


def payouts_to_dataframe(payouts: dict) -> pd.DataFrame:
    rows = []
    for bet_type in BET_TYPES:
        for row in (payouts or {}).get(bet_type, []):
            rows.append({"券種": bet_type, "組み合わせ": row.get("combination", ""), "払戻": row.get("payout", 0)})
    return pd.DataFrame(rows, columns=["券種", "組み合わせ", "払戻"])


def _bet(bet_type: str, numbers: list[int], stake: int) -> dict[str, Any]:
    normalized = sorted(_to_int(number) for number in numbers)
    return {"bet_type": bet_type, "combination": "-".join(str(number) for number in normalized), "stake": int(stake)}


def _winning_combinations(order: list[int]) -> dict[str, set[str]]:
    top1 = order[:1]
    top2 = order[:2]
    top3 = order[:3]
    return {
        "単勝": {_normalize_combination(top1)} if len(top1) == 1 else set(),
        "複勝": {_normalize_combination([number]) for number in top3},
        "馬連": {_normalize_combination(top2)} if len(top2) == 2 else set(),
        "ワイド": {_normalize_combination(pair) for pair in combinations(top3, 2)} if len(top3) >= 2 else set(),
        "三連複": {_normalize_combination(top3)} if len(top3) == 3 else set(),
    }


def _actual_order(actual_result: list[dict]) -> list[int]:
    rows = []
    for row in actual_result or []:
        finish = _to_int(row.get("finish", row.get("着順")))
        number = _to_int(row.get("horse_number", row.get("馬番")))
        if finish > 0 and number > 0:
            rows.append((finish, number))
    return [number for _, number in sorted(rows)]


def _normalize_combination(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        numbers = [_to_int(item) for item in value]
    else:
        numbers = [_to_int(item) for item in str(value).replace("－", "-").replace("ー", "-").split("-")]
        if not numbers or all(number == 0 for number in numbers):
            import re

            numbers = [_to_int(item) for item in re.findall(r"\d+", str(value))]
    numbers = sorted(number for number in numbers if number > 0)
    return "-".join(str(number) for number in numbers)


def _first_existing_column(table: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in table.columns:
            return column
    return None


def _to_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


__all__ = [
    "aggregate_return_rates",
    "bets_to_dataframe",
    "calculate_return_rate",
    "generate_bets_from_prediction",
    "payouts_to_dataframe",
]
