from __future__ import annotations

import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent / "src"
if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.trend_database import (  # noqa: E402
    PUBLIC_TREND_DB_DIR,
    TREND_DB_DIR,
    ensure_public_trend_database_dir,
    ensure_trend_database_dir,
    export_trend_database_to_public_dir,
    get_or_build_trend_data,
    load_trend_cache,
    public_trend_cache_path,
    save_trend_cache,
    trend_cache_key,
    trend_cache_path,
)


__all__ = [
    "PUBLIC_TREND_DB_DIR",
    "TREND_DB_DIR",
    "ensure_public_trend_database_dir",
    "ensure_trend_database_dir",
    "export_trend_database_to_public_dir",
    "get_or_build_trend_data",
    "load_trend_cache",
    "public_trend_cache_path",
    "save_trend_cache",
    "trend_cache_key",
    "trend_cache_path",
]
