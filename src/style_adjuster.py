from __future__ import annotations

from typing import Any

from course_db import get_course_bias
from horse_analyzer import STYLE_KEYS, normalize_style_profile


def adjust_style_profile(
    base_profile: dict[str, float],
    race_config: dict[str, Any] | Any,
    pace_prediction: dict[str, Any],
    frame: int,
    horse_number: int,
) -> dict[str, float]:
    """Adjust base running-style probabilities for one race condition."""
    profile = normalize_style_profile(base_profile)
    multipliers = {style: 1.0 for style in STYLE_KEYS}
    pace = str(pace_prediction.get("pace", "medium"))

    if pace == "slow":
        multipliers["逃げ"] *= 1.03
        multipliers["先行"] *= 1.02
        multipliers["差し"] *= 0.98
        multipliers["追込"] *= 0.94
    elif pace == "high":
        multipliers["逃げ"] *= 0.92
        multipliers["先行"] *= 0.92
        multipliers["差し"] *= 1.08
        multipliers["追込"] *= 1.12

    track_condition = _config_value(race_config, "track_condition", "良")
    if track_condition in {"重", "不良"}:
        multipliers["逃げ"] *= 1.03
        multipliers["先行"] *= 1.02
        multipliers["追込"] *= 0.93

    distance = int(_config_value(race_config, "distance", 1800) or 1800)
    if distance <= 1400:
        multipliers["逃げ"] *= 1.03
        multipliers["先行"] *= 1.02
        multipliers["追込"] *= 0.95
    elif distance >= 2400:
        multipliers["逃げ"] *= 0.95
        multipliers["差し"] *= 1.05
        multipliers["追込"] *= 1.05

    course_bias = get_course_bias(race_config)
    front_bias = float(course_bias.get("front_bias", 0.0))
    closer_bias = float(course_bias.get("closer_bias", 0.0))
    bias_scale = 0.35
    multipliers["逃げ"] *= 1.0 + front_bias * bias_scale
    multipliers["先行"] *= 1.0 + front_bias * bias_scale
    multipliers["差し"] *= 1.0 + closer_bias * bias_scale
    multipliers["追込"] *= 1.0 + closer_bias * bias_scale

    adjusted = {style: profile[style] * multipliers[style] for style in STYLE_KEYS}
    return normalize_style_profile(adjusted)


def _config_value(race_config: dict[str, Any] | Any, key: str, default: Any) -> Any:
    if isinstance(race_config, dict):
        return race_config.get(key, default)
    return getattr(race_config, key, default)
