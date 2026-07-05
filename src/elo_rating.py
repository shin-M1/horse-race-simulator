from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


DEFAULT_ELO = 1500.0
K_FACTOR = 24.0
ELO_PATH = Path("data/elo/horse_elo_ratings.json")


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((float(rating_b) - float(rating_a)) / 400.0))


def update_pairwise_elo(
    rating_a: float,
    rating_b: float,
    result_a: float,
    k_factor: float = K_FACTOR,
) -> tuple[float, float]:
    expected_a = expected_score(rating_a, rating_b)
    change_a = float(k_factor) * (float(result_a) - expected_a)
    return float(rating_a) + change_a, float(rating_b) - change_a


def update_elo_from_race_result(
    race_result: list[dict[str, Any]],
    ratings: dict[str, float],
    k_factor: float = K_FACTOR,
) -> dict[str, float]:
    try:
        return _update_elo_from_race_result_impl(race_result, ratings, k_factor)
    except Exception:
        logger.exception("EloRating update failed")
        raise


def _update_elo_from_race_result_impl(
    race_result: list[dict[str, Any]],
    ratings: dict[str, float],
    k_factor: float = K_FACTOR,
) -> dict[str, float]:
    updated = {str(name): float(value) for name, value in (ratings or {}).items()}
    rows = []
    for row in race_result or []:
        if not isinstance(row, dict):
            continue
        name = str(row.get("horse_name") or row.get("name") or row.get("馬名") or "").strip()
        try:
            finish = int(float(row.get("finish") or row.get("rank") or row.get("着順")))
        except (TypeError, ValueError):
            continue
        if name and finish > 0:
            rows.append({"horse_name": name, "finish": finish})
            updated.setdefault(name, DEFAULT_ELO)
    if len(rows) < 2:
        return updated

    deltas = {row["horse_name"]: 0.0 for row in rows}
    divisor = max(1, len(rows) - 1)
    for index, left in enumerate(rows):
        for right in rows[index + 1 :]:
            left_name = str(left["horse_name"])
            right_name = str(right["horse_name"])
            if int(left["finish"]) == int(right["finish"]):
                actual_left = 0.5
            else:
                actual_left = 1.0 if int(left["finish"]) < int(right["finish"]) else 0.0
            expected_left = expected_score(updated[left_name], updated[right_name])
            change = float(k_factor) * (actual_left - expected_left) / divisor
            deltas[left_name] += change
            deltas[right_name] -= change
    for name, delta in deltas.items():
        updated[name] = updated[name] + delta
    return updated


def load_elo_ratings(path: str | Path = ELO_PATH) -> dict[str, float]:
    target = Path(path)
    if not target.is_file():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if isinstance(payload, dict) and isinstance(payload.get("ratings"), dict):
        payload = payload["ratings"]
    if not isinstance(payload, dict):
        return {}
    ratings: dict[str, float] = {}
    for name, value in payload.items():
        try:
            ratings[str(name)] = float(value)
        except (TypeError, ValueError):
            continue
    return ratings


def save_elo_ratings(ratings: dict[str, float], path: str | Path = ELO_PATH) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ratings": {str(name): float(value) for name, value in sorted((ratings or {}).items())}}
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(target)


def normalize_elo_score(rating: float, min_rating: float = 1300, max_rating: float = 1800) -> float:
    if max_rating <= min_rating:
        return 50.0
    score = (float(rating) - float(min_rating)) / (float(max_rating) - float(min_rating)) * 100.0
    return max(0.0, min(100.0, score))
