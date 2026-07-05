from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Callable

from cache_utils import DATA_DIR, is_fresh, read_json, safe_key, utc_now_iso, write_json


logger = logging.getLogger(__name__)


TREND_DB_DIR = DATA_DIR / "trend_database"
PUBLIC_TREND_DB_DIR = Path("data_public") / "trend_database"


def trend_cache_key(race_name: str, venue: str, distance: int) -> str:
    return safe_key(race_name, venue, int(distance or 0))


def trend_cache_path(race_name: str, venue: str, distance: int) -> str:
    return str(TREND_DB_DIR / f"{trend_cache_key(race_name, venue, distance)}.json")


def public_trend_cache_path(race_name: str, venue: str, distance: int) -> str:
    return str(PUBLIC_TREND_DB_DIR / f"{trend_cache_key(race_name, venue, distance)}.json")


def load_trend_cache(race_name: str, venue: str, distance: int) -> dict[str, Any] | None:
    local_path = trend_cache_path(race_name, venue, distance)
    payload = read_json(local_path)
    if payload is not None:
        payload["_database_status"] = "hit"
        payload["_database_path"] = local_path
        payload["_database_source"] = "data"
        return payload

    public_path = public_trend_cache_path(race_name, venue, distance)
    payload = read_json(public_path)
    if payload is not None:
        payload["_database_status"] = "public_hit"
        payload["_database_path"] = public_path
        payload["_database_source"] = "data_public"
        return payload
    return None


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
    if cached and not force_refresh and cached.get("_database_source") == "data_public":
        cached["_database_status"] = "public_hit"
        return cached
    if cached and not force_refresh and is_fresh(cached.get("updated_at"), max_age_days):
        cached["_database_status"] = "hit"
        return cached
    try:
        built = builder_func()
    except Exception:
        logger.exception("TrendDatabase builder failed: race_name=%s venue=%s distance=%s", race_name, venue, distance)
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


def ensure_public_trend_database_dir() -> Path:
    PUBLIC_TREND_DB_DIR.mkdir(parents=True, exist_ok=True)
    return PUBLIC_TREND_DB_DIR


def export_trend_database_to_public_dir(
    source_dir: str | Path = "data/trend_database",
    public_dir: str | Path = "data_public/trend_database",
) -> dict[str, Any]:
    source = Path(source_dir)
    public = Path(public_dir)
    public.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    if not source.is_dir():
        return {
            "copied_count": 0,
            "source_dir": str(source),
            "public_dir": str(public),
            "copied_files": copied,
            "message": f"source_dir does not exist: {source}",
        }
    for item in sorted(source.glob("*.json")):
        if not item.is_file():
            continue
        destination = public / item.name
        shutil.copy2(item, destination)
        copied.append(str(destination))
    return {
        "copied_count": len(copied),
        "source_dir": str(source),
        "public_dir": str(public),
        "copied_files": copied,
        "message": f"copied {len(copied)} json files",
    }
