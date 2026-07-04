from __future__ import annotations

from typing import Any, Callable

from netkeiba_fetcher import (
    fetch_race_metadata,
    fetch_race_result,
    get_fetch_debug,
    search_race_id_by_name_and_date,
)
from payout_fetcher import fetch_race_payouts


def load_completed_race_result(
    race_name: str,
    race_date: str,
    search_fn: Callable[[str, str], dict[str, Any] | None] = search_race_id_by_name_and_date,
    result_fn: Callable[[str], list[dict[str, Any]]] = fetch_race_result,
    metadata_fn: Callable[[str], dict[str, Any]] = fetch_race_metadata,
    payout_fn: Callable[[str], dict[str, list[dict[str, Any]]]] = fetch_race_payouts,
) -> dict[str, Any] | None:
    """Load post-race data only after an explicit result action."""
    lookup = search_fn(race_name, race_date)
    if not lookup:
        return None
    search_debug = get_fetch_debug()
    race_id = str(lookup.get("race_id", ""))
    results = result_fn(race_id)
    result_debug = get_fetch_debug()
    metadata = metadata_fn(race_id) if results else {}
    metadata_debug = get_fetch_debug() if results else {}
    payouts = payout_fn(race_id) if results else {}
    return {
        **lookup,
        "race_id": race_id,
        "actual_result": results or None,
        "payouts": payouts or {},
        "race_metadata": metadata,
        "debug": {
            "search": search_debug,
            "result": result_debug,
            "metadata": metadata_debug,
        },
    }


__all__ = ["fetch_race_result", "fetch_race_payouts", "load_completed_race_result"]
