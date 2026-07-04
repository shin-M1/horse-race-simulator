from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class DuplicateLogError(ValueError):
    """Raised when saving would duplicate an existing log."""

    def __init__(self, paths: list[str]) -> None:
        self.paths = paths
        super().__init__("同じレースのログが既に存在します。")


def log_exists(
    race_id: str,
    race_name: str,
    race_date: str,
    log_dir: str | Path,
) -> bool:
    """Return whether a prediction log with the same race identity exists."""
    return bool(
        find_matching_logs(
            race_id=race_id,
            race_name=race_name,
            race_date=race_date,
            log_dir=log_dir,
        )
    )


def find_matching_logs(
    *,
    race_id: str,
    race_name: str,
    race_date: str,
    log_dir: str | Path,
    prediction_log_path: str | None = None,
) -> list[dict[str, Any]]:
    """Find logs whose identity fields all match the supplied values."""
    target = _identity_key(race_id, race_name, race_date)
    target_prediction = _normalized_path(prediction_log_path) if prediction_log_path is not None else None
    matches: list[dict[str, Any]] = []
    for path, payload in _iter_json_logs(log_dir):
        if _identity_key(
            payload.get("race_id", ""),
            payload.get("race_name", ""),
            payload.get("race_date", ""),
        ) != target:
            continue
        if target_prediction is not None:
            if _normalized_path(payload.get("prediction_log_path", "")) != target_prediction:
                continue
        item = dict(payload)
        item["_path"] = str(path)
        matches.append(item)
    return matches


def find_duplicate_logs(log_dir: str | Path, log_type: str) -> list[dict[str, Any]]:
    """Group duplicate prediction or evaluation JSON logs by their identity key."""
    groups: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    is_evaluation = str(log_type).strip().lower() in {"evaluation", "評価", "evaluation_log"}
    for path, payload in _iter_json_logs(log_dir):
        if is_evaluation:
            key = "_".join(
                [
                    _identity_key(
                        payload.get("race_id", ""),
                        payload.get("race_name", ""),
                        payload.get("race_date", ""),
                    ),
                    _normalized_path(payload.get("prediction_log_path", "")),
                ]
            )
        else:
            key = _identity_key(
                payload.get("race_id", ""),
                payload.get("race_name", ""),
                payload.get("race_date", ""),
            )
        groups.setdefault(key, []).append((path, payload))

    duplicates: list[dict[str, Any]] = []
    for key, items in groups.items():
        if len(items) < 2:
            continue
        first = items[0][1]
        duplicates.append(
            {
                "duplicate_key": key,
                "log_type": "evaluation" if is_evaluation else "prediction",
                "race_id": str(first.get("race_id", "")),
                "race_name": str(first.get("race_name", "")),
                "race_date": str(first.get("race_date", "")),
                "prediction_log_path": str(first.get("prediction_log_path", "")),
                "count": len(items),
                "paths": [str(path) for path, _ in items],
                "files": [path.name for path, _ in items],
            }
        )
    return sorted(duplicates, key=lambda item: item["duplicate_key"])


def delete_log_files(paths: list[str]) -> dict[str, Any]:
    """Delete selected JSON logs and their CSV/timeline companions safely."""
    targets: list[Path] = []
    for value in paths:
        if not value:
            continue
        path = Path(value)
        targets.append(path)
        if path.suffix.lower() in {".json", ".csv"}:
            targets.append(path.with_suffix(".csv" if path.suffix.lower() == ".json" else ".json"))
        if path.suffix.lower() == ".json" and path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                timeline_path = payload.get("race_timeline_path")
                if timeline_path:
                    targets.append(Path(str(timeline_path)))
            except (OSError, json.JSONDecodeError, TypeError):
                pass

    deleted: list[str] = []
    missing: list[str] = []
    failed: dict[str, str] = {}
    seen: set[str] = set()
    for path in targets:
        normalized = _normalized_path(path)
        if normalized in seen:
            continue
        seen.add(normalized)
        try:
            if path.is_file():
                path.unlink()
                deleted.append(str(path))
            else:
                missing.append(str(path))
        except OSError as exc:
            failed[str(path)] = str(exc)
    return {"deleted": deleted, "missing": missing, "failed": failed}


def log_inventory(log_dir: str | Path) -> list[dict[str, Any]]:
    """Build compact metadata rows for Streamlit log management."""
    rows: list[dict[str, Any]] = []
    for path, payload in _iter_json_logs(log_dir):
        rows.append(
            {
                "ファイル名": path.name,
                "race_id": str(payload.get("race_id", "")),
                "race_name": str(payload.get("race_name", "")),
                "race_date": str(payload.get("race_date", "")),
                "timestamp": str(payload.get("timestamp", "")),
                "path": str(path),
                "prediction_log_path": str(payload.get("prediction_log_path", "")),
            }
        )
    return rows


def _iter_json_logs(log_dir: str | Path) -> list[tuple[Path, dict[str, Any]]]:
    directory = Path(log_dir)
    if not directory.exists():
        return []
    rows: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            rows.append((path, payload))
    return rows


def _identity_key(race_id: Any, race_name: Any, race_date: Any) -> str:
    return "_".join(
        [
            str(race_id or "").strip().casefold(),
            "".join(str(race_name or "").split()).casefold(),
            str(race_date or "").strip(),
        ]
    )


def _normalized_path(value: str | Path | None) -> str:
    if not value:
        return ""
    try:
        return os.path.normcase(os.path.abspath(os.fspath(value)))
    except (OSError, TypeError, ValueError):
        return str(value)
