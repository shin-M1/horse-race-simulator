from __future__ import annotations

from typing import Any, Callable

from netkeiba_fetcher import (
    fetch_race_entries,
    fetch_race_metadata,
    fetch_race_schedule,
    get_fetch_debug,
    search_race_id_by_name_and_date,
)


def load_prediction_race_data_by_id(
    race_id: str,
    selected_race: dict[str, Any] | None = None,
    entries_fn: Callable[[str], list[dict[str, Any]]] = fetch_race_entries,
    metadata_fn: Callable[[str], dict[str, Any]] = fetch_race_metadata,
) -> dict[str, Any]:
    """Load a selected pre-race card directly by race_id."""
    selected = dict(selected_race or {})
    entries = entries_fn(race_id)
    entries_debug = get_fetch_debug()
    metadata = metadata_fn(race_id)
    metadata_debug = get_fetch_debug()
    merged_metadata = {key: value for key, value in selected.items() if value not in (None, "", 0)}
    merged_metadata.update({key: value for key, value in metadata.items() if value not in (None, "", 0)})
    return {
        **selected,
        "race_id": race_id,
        "race_name": merged_metadata.get("race_name", selected.get("race_name", "")),
        "race_date": merged_metadata.get("race_date", selected.get("race_date", "")),
        "venue": merged_metadata.get("venue", selected.get("venue", "")),
        "source_url": merged_metadata.get("source_url", selected.get("source_url", "")),
        "fetched_entries": entries,
        "race_metadata": merged_metadata,
        "debug": {"entries": entries_debug, "metadata": metadata_debug},
    }


def load_prediction_race_data(
    race_name: str,
    race_date: str,
    search_fn: Callable[[str, str], dict[str, Any] | None] = search_race_id_by_name_and_date,
    entries_fn: Callable[[str], list[dict[str, Any]]] = fetch_race_entries,
    metadata_fn: Callable[[str], dict[str, Any]] = fetch_race_metadata,
) -> dict[str, Any] | None:
    """Load only pre-race data. This function intentionally has no result fetcher."""
    lookup = search_fn(race_name, race_date)
    if not lookup:
        return None
    search_debug = get_fetch_debug()
    race_id = str(lookup.get("race_id", ""))
    entries = entries_fn(race_id)
    entries_debug = get_fetch_debug()
    metadata = metadata_fn(race_id)
    metadata_debug = get_fetch_debug()
    return {
        **lookup,
        "race_id": race_id,
        "fetched_entries": entries,
        "race_metadata": metadata,
        "debug": {
            "search": search_debug,
            "entries": entries_debug,
            "metadata": metadata_debug,
        },
    }


__all__ = [
    "search_race_id_by_name_and_date",
    "fetch_race_entries",
    "fetch_race_schedule",
    "fetch_race_metadata",
    "load_prediction_race_data",
    "load_prediction_race_data_by_id",
]
