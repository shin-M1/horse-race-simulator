from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import json

from netkeiba_fetcher import (
    fetch_race_metadata,
    fetch_race_result,
    search_race_id_by_name_and_date,
)


RACE_TREND_DIR = Path("outputs/race_trends")


def fetch_same_race_history(
    race_name: str,
    race_date: str,
    years: int = 10,
    *,
    search_fn: Callable[[str, str], dict[str, Any] | None] = search_race_id_by_name_and_date,
    result_fn: Callable[[str], list[dict[str, Any]]] = fetch_race_result,
    metadata_fn: Callable[[str], dict[str, Any]] = fetch_race_metadata,
) -> list[dict]:
    """Fetch completed same-race histories for up to the previous `years`.

    This function never fabricates fallback race rows. If a year cannot be
    resolved or a result table is unavailable, that year is skipped.
    """
    base_date = _parse_date(race_date)
    normalized_name = str(race_name or "").strip()
    if not normalized_name or base_date is None:
        return []

    histories: list[dict[str, Any]] = []
    seen_race_ids: set[str] = set()
    current_year = datetime.now().year
    for offset in range(1, max(1, int(years)) + 1):
        target_year = base_date.year - offset
        if target_year > current_year:
            continue
        lookup = _find_same_race_near_date(
            normalized_name,
            _replace_year_safe(base_date, target_year),
            search_fn=search_fn,
        )
        if not lookup:
            continue
        race_id = str(lookup.get("race_id", ""))
        if not race_id or race_id in seen_race_ids:
            continue
        results = result_fn(race_id)
        if not results:
            continue
        try:
            metadata = metadata_fn(race_id) or {}
        except Exception:
            metadata = {}
        seen_race_ids.add(race_id)
        field_size = len(results)
        for row in results:
            item = _normalize_history_row(
                row=row,
                lookup=lookup,
                metadata=metadata,
                year=target_year,
                field_size=field_size,
            )
            if item.get("horse_name"):
                histories.append(item)
    return histories


def save_same_race_history(
    history: list[dict],
    race_name: str,
    race_date: str,
    output_dir: str | Path = RACE_TREND_DIR,
) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(ch if ch.isalnum() else "_" for ch in str(race_name or "race")).strip("_") or "race"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = directory / f"same_race_history_{safe_name}_{str(race_date).replace('-', '')}_{stamp}.json"
    path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _find_same_race_near_date(
    race_name: str,
    target_date: date,
    *,
    search_fn: Callable[[str, str], dict[str, Any] | None],
    search_window_days: int = 14,
) -> dict[str, Any] | None:
    offsets = [0]
    for day in range(1, search_window_days + 1):
        offsets.extend([-day, day])
    for offset in offsets:
        candidate_date = target_date + timedelta(days=offset)
        try:
            lookup = search_fn(race_name, candidate_date.isoformat())
        except Exception:
            lookup = None
        if lookup and lookup.get("race_id"):
            return lookup
    return None


def _normalize_history_row(
    row: dict[str, Any],
    lookup: dict[str, Any],
    metadata: dict[str, Any],
    year: int,
    field_size: int,
) -> dict[str, Any]:
    race_name = str(metadata.get("race_name") or lookup.get("race_name") or "")
    return {
        "year": year,
        "race_id": str(lookup.get("race_id", "")),
        "race_name": race_name,
        "race_date": str(metadata.get("race_date") or lookup.get("race_date") or ""),
        "course": str(metadata.get("venue") or metadata.get("course") or lookup.get("venue") or ""),
        "venue": str(metadata.get("venue") or lookup.get("venue") or ""),
        "distance": _to_int(metadata.get("distance")),
        "surface": str(metadata.get("surface", "")),
        "track_condition": str(metadata.get("track_condition", "")),
        "race_class": str(metadata.get("race_class", "")),
        "finish": _to_int(row.get("finish") or row.get("着順")),
        "horse_name": str(row.get("horse_name") or row.get("馬名") or ""),
        "frame": _to_int(row.get("frame") or row.get("枠順")),
        "horse_number": _to_int(row.get("horse_number") or row.get("馬番")),
        "age": _to_int(row.get("age") or row.get("馬齢") or _age_from_sex_age(row.get("sex_age") or row.get("性齢"))),
        "sex": str(row.get("sex") or row.get("性別") or _sex_from_sex_age(row.get("sex_age") or row.get("性齢")) or ""),
        "carried_weight": _to_float(row.get("carried_weight") or row.get("斤量")),
        "popularity": _to_int(row.get("popularity") or row.get("人気")),
        "running_style": str(row.get("running_style") or row.get("脚質") or ""),
        "passing_order": str(row.get("passing_order") or row.get("通過順") or ""),
        "fourth_corner_pos": _to_int(
            row.get("fourth_corner_pos")
            or row.get("4角")
            or row.get("４角")
            or _fourth_corner_from_passing(row.get("passing_order") or row.get("通過順") or "")
        ),
        "last3f": _to_float(row.get("last3f") or row.get("上り")),
        "last3f_rank": _to_int(row.get("last3f_rank") or row.get("上り順位") or row.get("上がり順位")),
        "jockey": str(row.get("jockey") or row.get("騎手") or ""),
        "previous_jockey": str(row.get("previous_jockey") or row.get("前走騎手") or ""),
        "jockey_change_type": str(row.get("jockey_change_type") or row.get("騎手変更") or _jockey_switch(row)),
        "previous_race_class": str(row.get("previous_race_class") or row.get("前走クラス") or ""),
        "previous_distance": _to_int(row.get("previous_distance") or row.get("前走距離")),
        "jockey_switch": _jockey_switch(row),
        "sire": str(row.get("sire") or row.get("父") or ""),
        "broodmare_sire": str(row.get("broodmare_sire") or row.get("母父") or ""),
        "sire_line": str(row.get("sire_line") or row.get("父系統") or ""),
        "broodmare_sire_line": str(row.get("broodmare_sire_line") or row.get("母父系統") or ""),
        "field_size": field_size,
        "time": row.get("time") or row.get("タイム") or "",
        "time_sec": _to_float(row.get("time_sec")),
        "margin": row.get("margin") or row.get("着差") or "",
        "margin_sec": _to_float(row.get("margin_sec")),
    }


def _jockey_switch(row: dict[str, Any]) -> str:
    current = str(row.get("jockey") or row.get("騎手") or "").strip()
    previous = str(row.get("previous_jockey") or row.get("前走騎手") or "").strip()
    if not current or not previous:
        return ""
    return "継続" if current == previous else "乗り替わり"


def _sex_from_sex_age(value: Any) -> str:
    text = str(value or "").strip()
    return text[:1] if text and text[:1] in {"牡", "牝", "セ", "騙"} else ""


def _age_from_sex_age(value: Any) -> int:
    text = str(value or "")
    for char in text:
        if char.isdigit():
            return int(char)
    return 0


def _fourth_corner_from_passing(value: Any) -> int:
    import re

    numbers = [int(item) for item in re.findall(r"\d+", str(value or ""))]
    return numbers[-1] if numbers else 0


def _parse_date(value: Any) -> date | None:
    text = str(value or "").strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _replace_year_safe(value: date, year: int) -> date:
    try:
        return value.replace(year=year)
    except ValueError:
        return value.replace(year=year, day=28)


def _to_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
