from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from cache_utils import DATA_DIR, is_fresh, read_json, safe_key, utc_now_iso, write_json


RACE_DB_DIR = DATA_DIR / "race_database"


def race_db_key(race_id: str | None, race_name: str, race_date: str) -> str:
    return safe_key(race_id or "", race_name, race_date)


def race_cache_path(key: str) -> str:
    return str(RACE_DB_DIR / f"{safe_key(key)}.json")


def load_race_cache(key: str) -> dict[str, Any] | None:
    payload = read_json(race_cache_path(key))
    if payload is None:
        return None
    payload = _race_payload(payload)
    payload["_database_status"] = "hit"
    payload["_database_path"] = race_cache_path(key)
    return payload


def save_race_cache(key: str, data: dict[str, Any]) -> str:
    payload = _race_payload(data)
    return write_json(race_cache_path(key), payload)


def get_or_fetch_race_data(
    race_name: str,
    race_date: str,
    race_id: str | None,
    fetcher_func: Callable[[], dict[str, Any] | None],
    force_refresh: bool = False,
    max_age_days: int = 30,
) -> dict[str, Any] | None:
    key = race_db_key(race_id, race_name, race_date)
    cached = load_race_cache(key)
    if cached and not force_refresh and is_fresh(cached.get("updated_at"), max_age_days):
        cached["_database_status"] = "hit"
        return cached

    try:
        fetched = fetcher_func()
    except Exception:
        if cached and not force_refresh:
            cached["_database_status"] = "stale_hit"
            return cached
        return None
    if not isinstance(fetched, dict):
        return None
    payload = _race_payload({**fetched, "race_name": race_name, "race_date": race_date, "race_id": fetched.get("race_id") or race_id or ""})
    final_key = race_db_key(str(payload.get("race_id") or race_id or ""), race_name, race_date)
    path = save_race_cache(key, payload)
    if final_key != key:
        save_race_cache(final_key, payload)
    payload["_database_status"] = "refresh" if force_refresh else "miss"
    payload["_database_path"] = path
    return payload


def _race_payload(data: dict[str, Any]) -> dict[str, Any]:
    payload = dict(data)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = payload.get("race_metadata") if isinstance(payload.get("race_metadata"), dict) else {}
    entries = payload.get("entries")
    if not isinstance(entries, list):
        entries = payload.get("fetched_entries") if isinstance(payload.get("fetched_entries"), list) else []
    payload.setdefault("race_id", "")
    payload.setdefault("race_name", metadata.get("race_name", ""))
    payload.setdefault("race_date", metadata.get("race_date", ""))
    payload["metadata"] = metadata
    payload["entries"] = entries
    payload.setdefault("fetched_entries", entries)
    payload.setdefault("race_metadata", metadata)
    payload.setdefault("result", [])
    payload.setdefault("payouts", {})
    payload.setdefault("updated_at", utc_now_iso())
    return payload


def ensure_race_database_dir() -> Path:
    RACE_DB_DIR.mkdir(parents=True, exist_ok=True)
    return RACE_DB_DIR
