from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from cache_utils import DATA_DIR, is_fresh, read_json, safe_key, utc_now_iso, write_json


HORSE_DB_DIR = DATA_DIR / "horse_database"


def normalize_horse_name(name: str) -> str:
    return "".join(str(name or "").split()).lower()


def horse_db_path(horse_name: str) -> str:
    return str(HORSE_DB_DIR / f"{safe_key(normalize_horse_name(horse_name))}.json")


def load_horse_profile(horse_name: str) -> dict[str, Any] | None:
    payload = read_json(horse_db_path(horse_name))
    if payload is None:
        return None
    payload = _profile_payload(horse_name, payload)
    payload["_database_status"] = "hit"
    payload["_database_path"] = horse_db_path(horse_name)
    return payload


def save_horse_profile(horse_name: str, profile: dict[str, Any]) -> str:
    payload = _profile_payload(horse_name, profile)
    return write_json(horse_db_path(horse_name), payload)


def get_or_fetch_horse_profile(
    horse_name: str,
    fetcher_func: Callable[[], dict[str, Any] | list[dict[str, Any]] | None],
    force_refresh: bool = False,
    max_age_days: int = 14,
) -> dict[str, Any] | None:
    cached = load_horse_profile(horse_name)
    if cached and cached.get("recent_races") and not force_refresh and is_fresh(cached.get("updated_at"), max_age_days):
        cached["_database_status"] = "hit"
        return cached

    try:
        fetched = fetcher_func()
    except Exception:
        if cached and not force_refresh:
            cached["_database_status"] = "stale_hit"
            return cached
        return None

    if fetched is None:
        return None
    profile = _profile_payload(horse_name, fetched)
    path = save_horse_profile(horse_name, profile)
    profile["_database_status"] = "refresh" if force_refresh else "miss"
    profile["_database_path"] = path
    return profile


def _profile_payload(horse_name: str, value: dict[str, Any] | list[dict[str, Any]]) -> dict[str, Any]:
    if isinstance(value, dict):
        payload = dict(value)
    else:
        payload = {"recent_races": list(value)}
    payload.setdefault("horse_name", horse_name)
    payload.setdefault("updated_at", utc_now_iso())
    payload.setdefault("recent_races", [])
    payload.setdefault("pedigree", {})
    payload.setdefault("running_style_profile", {})
    payload.setdefault("primary_running_style", "")
    payload.setdefault("mud_aptitude", {})
    payload.setdefault("distance_aptitude", {})
    payload.setdefault("course_aptitude", {})
    payload.setdefault("jockey_history", [])
    payload["horse_name"] = str(payload.get("horse_name") or horse_name)
    payload["updated_at"] = str(payload.get("updated_at") or utc_now_iso())
    return payload


def ensure_horse_database_dir() -> Path:
    HORSE_DB_DIR.mkdir(parents=True, exist_ok=True)
    return HORSE_DB_DIR
