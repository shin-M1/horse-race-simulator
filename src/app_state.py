from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    errors: list[str]


def validate_inputs(race_config: dict[str, Any], horses: list[dict[str, Any]]) -> ValidationResult:
    errors: list[str] = []
    if int(race_config.get("distance", 0) or 0) <= 0:
        errors.append("距離は正の整数で入力してください。")
    if len(horses) < 2:
        errors.append("出走頭数は2頭以上にしてください。")
    if len(horses) > 18:
        errors.append("出走頭数は18頭以内にしてください。")

    horse_numbers: list[int] = []
    for index, horse in enumerate(horses, start=1):
        name = str(horse.get("horse_name", "")).strip()
        frame = _to_int(horse.get("frame", 0))
        horse_number = _to_int(horse.get("horse_number", 0))
        carried_weight = _to_float(horse.get("carried_weight", 56.0), 56.0)
        if not name:
            errors.append(f"{index}行目の馬名が空です。")
        if not 1 <= frame <= 8:
            errors.append(f"{name or index} の枠順は1〜8で入力してください。")
        if not 1 <= horse_number <= 18:
            errors.append(f"{name or index} の馬番は1〜18で入力してください。")
        if not 40.0 <= carried_weight <= 65.0:
            errors.append(f"{name or index} の斤量は40.0〜65.0kgで入力してください。")
        horse_numbers.append(horse_number)

    duplicated = sorted({number for number in horse_numbers if horse_numbers.count(number) > 1})
    if duplicated:
        errors.append(f"馬番が重複しています: {duplicated}")
    return ValidationResult(is_valid=not errors, errors=errors)


def dataframe_to_horses(df: pd.DataFrame, count: int | None = None) -> list[dict[str, Any]]:
    source = df.head(count) if count is not None else df
    rows = source.fillna("").to_dict("records")
    horses: list[dict[str, Any]] = []
    for row in rows:
        name = str(row.get("horse_name", "")).strip()
        frame = _to_int(row.get("frame", 0))
        horse_number = _to_int(row.get("horse_number", 0))
        carried_weight = _to_float(row.get("carried_weight", 56.0), 56.0)
        jockey = str(row.get("jockey", "")).strip()
        if not name and frame == 0 and horse_number == 0:
            continue
        horses.append(
            {
                "horse_name": name,
                "frame": frame,
                "horse_number": horse_number,
                "carried_weight": carried_weight,
                "jockey": jockey,
            }
        )
    return horses


def _to_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
