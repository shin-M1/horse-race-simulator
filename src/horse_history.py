"""Explicit diagnostics; no network access happens merely by importing this module."""
from __future__ import annotations

import argparse
import json
import logging

from data_fetcher import NetkeibaRaceResultProvider, fetch_horse_history


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose real horse history requests")
    parser.add_argument("horse_id")
    parser.add_argument("--horse-name", default="")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    provider = NetkeibaRaceResultProvider()
    try:
        races = provider.get_recent_results_by_horse_id(args.horse_id, args.horse_name)
        success = True
    except Exception as exc:
        races = []
        success = False
        logging.error("Horse history diagnosis failed: %s", exc)
    debug = provider.get_debug_info()
    print(json.dumps({
        **{key: debug.get(key) for key in (
            "horse_id", "url", "http_status", "final_url", "html_bytes", "title",
            "parse_rows", "parser_status", "failure_stage",
        )},
        "recent_count": len(races),
    }, ensure_ascii=False, indent=2))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
