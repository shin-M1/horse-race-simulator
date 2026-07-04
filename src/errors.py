from __future__ import annotations

from typing import Any


class RaceDataFetchError(ValueError):
    """Raised when a horse's real recent-race data cannot be fetched safely."""

    def __init__(
        self,
        message: str,
        horse_name: str = "",
        debug_records: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.horse_name = horse_name
        self.debug_records = debug_records or []
