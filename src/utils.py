from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, TypeVar

import numpy as np


T = TypeVar("T")


def clamp(value: float, minimum: float = 0.0, maximum: float = 100.0) -> float:
    """Clamp a numeric value into a closed interval."""
    return float(max(minimum, min(maximum, value)))


def safe_mean(values: Iterable[float], default: float = 0.0) -> float:
    """Return a mean value without failing on empty inputs."""
    items = [float(v) for v in values if v is not None]
    if not items:
        return float(default)
    return float(np.mean(items))


def safe_std(values: Iterable[float], default: float = 0.0) -> float:
    """Return a standard deviation without failing on short inputs."""
    items = [float(v) for v in values if v is not None]
    if len(items) <= 1:
        return float(default)
    return float(np.std(items))


def pick(items: Sequence[T], index: int, default: T) -> T:
    """Return an indexed item or a default value."""
    if 0 <= index < len(items):
        return items[index]
    return default


def normalize_track_condition(condition: str) -> str:
    """Normalize Japanese track condition labels used by JRA/netkeiba data."""
    text = condition.strip()
    aliases = {
        "良": "良",
        "稍": "稍重",
        "稍重": "稍重",
        "重": "重",
        "不": "不良",
        "不良": "不良",
    }
    return aliases.get(text, "良")


def normalize_running_positions(value: str | Sequence[int] | None) -> list[int]:
    """Convert pass-order text such as '3-3-2' into integer positions."""
    if value is None:
        return []
    if isinstance(value, str):
        parts = value.replace(" ", "").replace("/", "-").split("-")
        positions: list[int] = []
        for part in parts:
            if part.isdigit():
                positions.append(int(part))
        return positions
    return [int(v) for v in value]


@dataclass(frozen=True)
class OutputPaths:
    gif_path: str
    mp4_path: str
