from __future__ import annotations

import re
import time
from io import StringIO
from typing import Any

import pandas as pd
import requests


RACE_BASE_URL = "https://race.netkeiba.com"
BET_TYPES = ("単勝", "複勝", "馬連", "ワイド", "三連複")


def fetch_race_payouts(race_id: str) -> dict[str, list[dict[str, int | str]]]:
    """Fetch payout rows for a completed race.

    Returns an empty dict when payouts are unavailable. No dummy payout rows are
    created.
    """
    if not re.fullmatch(r"\d{12}", str(race_id).strip()):
        return {}
    url = f"{RACE_BASE_URL}/race/result.html?race_id={race_id}"
    try:
        html_text = _request_text(url)
        return parse_payouts_from_html(html_text)
    except (requests.RequestException, ValueError, pd.errors.ParserError, ImportError):
        return {}


def parse_payouts_from_html(html_text: str) -> dict[str, list[dict[str, int | str]]]:
    payouts: dict[str, list[dict[str, int | str]]] = {}
    try:
        tables = pd.read_html(StringIO(html_text))
    except (ValueError, ImportError, pd.errors.ParserError):
        return {}
    for table in tables:
        frame = _flatten_columns(table)
        for _, row in frame.iterrows():
            parsed = _parse_payout_row(row)
            if parsed is None:
                continue
            bet_type, entries = parsed
            if bet_type in BET_TYPES:
                payouts.setdefault(bet_type, []).extend(entries)
    return {key: value for key, value in payouts.items() if value}


def _parse_payout_row(row: pd.Series) -> tuple[str, list[dict[str, int | str]]] | None:
    cells = [_clean_text(value) for value in row.tolist() if _clean_text(value)]
    if not cells:
        return None
    row_text = " ".join(cells)
    bet_type = next((name for name in BET_TYPES if name in row_text), "")
    if not bet_type:
        return None

    type_index = next((index for index, value in enumerate(cells) if bet_type in value), 0)
    remaining = cells[type_index + 1 :] or cells
    payout_cell_index = next(
        (index for index, value in enumerate(remaining) if "円" in value or re.search(r"\d{3,}(?:,\d{3})*", value)),
        -1,
    )
    if payout_cell_index < 0:
        return None

    combination_text = " ".join(remaining[:payout_cell_index])
    payout_text = remaining[payout_cell_index]
    combinations = _extract_combinations(combination_text)
    payout_values = _extract_payout_values(payout_text)
    if not combinations or not payout_values:
        return None
    if len(payout_values) == 1 and len(combinations) > 1:
        payout_values = payout_values * len(combinations)
    entries = [
        {"combination": combination, "payout": payout}
        for combination, payout in zip(combinations, payout_values)
        if combination and payout > 0
    ]
    return (bet_type, entries) if entries else None


def _extract_combinations(value: str) -> list[str]:
    text = _normalize_hyphen(value)
    parts = re.split(r"[\n\r/／、,，\s]+", text)
    combinations: list[str] = []
    for part in parts:
        numbers = re.findall(r"\d+", part)
        if not numbers:
            continue
        if len(numbers) == 1:
            combinations.append(numbers[0])
        elif len(numbers) <= 3:
            combinations.append("-".join(numbers))
    if combinations:
        return combinations
    numbers = re.findall(r"\d+", text)
    return ["-".join(numbers)] if 1 <= len(numbers) <= 3 else []


def _extract_payout_values(value: str) -> list[int]:
    return [int(match.replace(",", "")) for match in re.findall(r"\d{2,}(?:,\d{3})*", str(value))]


def _request_text(url: str) -> str:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "ja,en;q=0.8",
        }
    )
    time.sleep(0.2)
    response = session.get(url, timeout=15)
    response.raise_for_status()
    if not response.encoding or response.encoding.lower() == "iso-8859-1":
        response.encoding = response.apparent_encoding or "utf-8"
    return response.text


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    flattened = frame.copy()
    flattened.columns = [
        " ".join(str(part) for part in column if str(part) != "nan").strip()
        if isinstance(column, tuple)
        else str(column)
        for column in flattened.columns
    ]
    return flattened


def _clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\xa0", " ")).strip()


def _normalize_hyphen(value: str) -> str:
    return str(value).replace("－", "-").replace("ー", "-").replace("―", "-").replace("–", "-")


__all__ = ["fetch_race_payouts", "parse_payouts_from_html"]
