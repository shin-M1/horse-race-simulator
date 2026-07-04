from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from cache_utils import DATA_DIR, is_fresh, read_json, safe_key, utc_now_iso, write_json


TREND_DB_DIR = DATA_DIR / "trend_database"


def trend_cache_key(race_name: str, venue: str, distance: int) -> str:
    return safe_key(race_name, venue, int(distance or 0))


def trend_cache_path(race_name: str, venue: str, distance: int) -> str:
    return str(TREND_DB_DIR / f"{trend_cache_key(race_name, venue, distance)}.json")


def load_trend_cache(race_name: str, venue: str, distance: int) -> dict[str, Any] | None:
    payload = read_json(trend_cache_path(race_name, venue, distance))
    if payload is None:
        return None
    payload["_database_status"] = "hit"
    payload["_database_path"] = trend_cache_path(race_name, venue, distance)
    return payload


def save_trend_cache(race_name: str, venue: str, distance: int, trend_data: dict[str, Any]) -> str:
    payload = dict(trend_data)
    payload.setdefault("race_name", race_name)
    payload.setdefault("venue", venue)
    payload.setdefault("distance", int(distance or 0))
    payload.setdefault("updated_at", utc_now_iso())
    return write_json(trend_cache_path(race_name, venue, distance), payload)


def get_or_build_trend_data(
    race_name: str,
    venue: str,
    distance: int,
    builder_func: Callable[[], dict[str, Any] | None],
    force_refresh: bool = False,
    max_age_days: int = 180,
) -> dict[str, Any] | None:
    cached = load_trend_cache(race_name, venue, distance)
    if cached and not force_refresh and is_fresh(cached.get("updated_at"), max_age_days):
        cached["_database_status"] = "hit"
        return cached
    try:
        built = builder_func()
    except Exception:
        if cached and not force_refresh:
            cached["_database_status"] = "stale_hit"
            return cached
        return None
    if not isinstance(built, dict):
        return None
    path = save_trend_cache(race_name, venue, distance, built)
    built = dict(built)
    built["_database_status"] = "refresh" if force_refresh else "miss"
    built["_database_path"] = path
    return built


def ensure_trend_database_dir() -> Path:
    TREND_DB_DIR.mkdir(parents=True, exist_ok=True)
    return TREND_DB_DIR
