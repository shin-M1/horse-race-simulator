from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from cache_utils import DATA_DIR, is_fresh, read_json, safe_key, utc_now_iso, write_json
from data_fetcher import is_valid_horse_id


HORSE_DB_DIR = DATA_DIR / "horse_database"
HORSE_HISTORY_SCHEMA_VERSION = "horse_history_v2"


def normalize_horse_name(name: str) -> str:
    return "".join(str(name or "").split()).lower()


def horse_db_path(horse_name: str, horse_id: str | None = None) -> str:
    key_source = normalize_horse_id(horse_id)
    if not key_source:
        raise ValueError("horse history cache requires a valid horse_id")
    return str(HORSE_DB_DIR / f"{safe_key(key_source)}.json")


def load_horse_profile(horse_name: str, horse_id: str | None = None) -> dict[str, Any] | None:
    payload = read_json(horse_db_path(horse_name, horse_id))
    if payload is None:
        return None
    if payload.get("horse_id") != horse_id or payload.get("schema_version") != HORSE_HISTORY_SCHEMA_VERSION:
        return None
    payload = _profile_payload(horse_name, payload)
    payload["_database_status"] = "hit"
    payload["_database_path"] = horse_db_path(horse_name, horse_id)
    return payload


def save_horse_profile(horse_name: str, profile: dict[str, Any], horse_id: str | None = None) -> str:
    payload = _profile_payload(horse_name, profile)
    if not _has_valid_history(payload):
        raise ValueError("empty horse history must not be cached")
    if normalize_horse_id(horse_id):
        payload["horse_id"] = normalize_horse_id(horse_id)
    return write_json(horse_db_path(horse_name, horse_id), payload)


def get_or_fetch_horse_profile(
    horse_name: str,
    fetcher_func: Callable[[], dict[str, Any] | list[dict[str, Any]] | None],
    force_refresh: bool = False,
    max_age_days: int = 14,
    horse_id: str | None = None,
) -> dict[str, Any] | None:
    cached = load_horse_profile(horse_name, horse_id)
    if cached and _has_valid_history(cached) and not force_refresh and is_fresh(cached.get("updated_at"), max_age_days):
        cached["_database_status"] = "hit"
        return cached

    try:
        fetched = fetcher_func()
    except Exception:
        if cached and _has_valid_history(cached) and not force_refresh:
            cached["_database_status"] = "stale_hit"
            return cached
        raise

    if fetched is None:
        return None
    profile = _profile_payload(horse_name, fetched)
    if not _has_valid_history(profile):
        return None
    path = save_horse_profile(horse_name, profile, horse_id)
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
    payload["schema_version"] = str(payload.get("schema_version") or HORSE_HISTORY_SCHEMA_VERSION)
    return payload


def normalize_horse_id(horse_id: str | None) -> str:
    return str(horse_id).strip() if is_valid_horse_id(horse_id) else ""


def _has_valid_history(profile: dict[str, Any]) -> bool:
    rows = profile.get("recent_races")
    if not isinstance(rows, list):
        return False
    for row in rows:
        if not isinstance(row, dict) or not row.get("race_name"):
            continue
        try:
            if float(row.get("distance", 0)) > 0 and float(row.get("finish", row.get("finish_position", 0))) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def clear_empty_horse_history_cache() -> list[str]:
    """Remove only well-formed cache files containing no recent races."""
    removed = []
    for path in HORSE_DB_DIR.glob("*.json"):
        payload = read_json(path)
        if payload is None or payload.get("recent_races"):
            continue
        try:
            path.unlink(missing_ok=True)
            removed.append(path.name)
        except OSError:
            continue
    return removed


def ensure_horse_database_dir() -> Path:
    HORSE_DB_DIR.mkdir(parents=True, exist_ok=True)
    return HORSE_DB_DIR
