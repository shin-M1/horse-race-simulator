from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from race_config import RaceConfig


__all__ = [
    "TRACK_FACTOR",
    "FRAME_BIAS",
    "TRACK_BIAS_EFFECTS",
    "COURSE_FEATURES",
    "CourseProfile",
    "CourseDB",
    "get_course_bias",
    "get_track_bias_fit_score",
    "estimate_course_fit_score",
]


FRAME_BIAS: dict[int, float] = {
    1: 0.010,
    2: 0.008,
    3: 0.006,
    4: 0.003,
    5: 0.000,
    6: -0.003,
    7: -0.006,
    8: -0.008,
}


TRACK_FACTOR: dict[str, dict[str, float]] = {
    "良": {
        "speed": 1.04,
        "stamina": 0.98,
        "acceleration": 1.04,
        "mud": 0.96,
    },
    "稍重": {
        "speed": 1.00,
        "stamina": 1.01,
        "acceleration": 0.99,
        "mud": 1.03,
    },
    "重": {
        "speed": 0.96,
        "stamina": 1.06,
        "acceleration": 0.95,
        "mud": 1.08,
    },
    "不良": {
        "speed": 0.92,
        "stamina": 1.10,
        "acceleration": 0.91,
        "mud": 1.13,
    },
}


TRACK_BIAS_EFFECTS: dict[str, dict[str, float]] = {
    "標準": {},
    "前残り": {"逃げ": 6.0, "先行": 5.0, "差し": -3.0, "追込": -5.0},
    "差し有利": {"逃げ": -5.0, "先行": -3.0, "差し": 5.0, "追込": 4.0},
    "内有利": {"inner": 5.0, "outer": -3.0},
    "外差し有利": {"inner": -4.0, "outer": 5.0, "差し": 5.0, "追込": 6.0},
    "内前有利": {"inner": 5.0, "逃げ": 5.0, "先行": 5.0},
    "外伸び": {"outer": 6.0, "差し": 4.0, "追込": 4.0},
}


COURSE_FEATURES: dict[str, dict[str, float | bool]] = {
    "東京": {"straight_length": 526.0, "slope": True},
    "中山": {"straight_length": 310.0, "slope": True},
    "京都": {"straight_length": 404.0, "slope": False},
    "阪神": {"straight_length": 474.0, "slope": True},
    "中京": {"straight_length": 413.0, "slope": True},
    "札幌": {"straight_length": 266.0, "slope": False},
    "函館": {"straight_length": 262.0, "slope": True},
    "福島": {"straight_length": 292.0, "slope": True},
    "新潟": {"straight_length": 659.0, "slope": False},
    "小倉": {"straight_length": 293.0, "slope": True},
}


@dataclass(frozen=True)
class CourseProfile:
    course: str
    surface: str
    distance: int
    direction: str
    inner_frame_advantage: bool
    stamina_weight: float
    acceleration_weight: float
    turn_penalty: float


class CourseDB:
    """Course-specific heuristics used by the simulator."""

    def get_profile(self, config: RaceConfig) -> CourseProfile:
        inner_advantage = self.has_inner_frame_advantage(config)
        return CourseProfile(
            course=config.course,
            surface=config.surface,
            distance=config.distance,
            direction=config.direction,
            inner_frame_advantage=inner_advantage,
            stamina_weight=1.12 if config.distance >= 2000 else 0.94,
            acceleration_weight=1.10 if config.distance <= 1800 else 0.96,
            turn_penalty=0.985 if config.direction in {"右", "左"} else 1.0,
        )

    def has_inner_frame_advantage(self, config: RaceConfig) -> bool:
        if config.surface == "芝" and config.track_condition in {"良", "稍重"}:
            return config.course in {"阪神", "中山", "京都", "小倉", "札幌", "函館"}
        if config.surface == "ダート":
            return config.course in {"阪神", "中山", "京都"}
        return False

    def frame_bonus(self, frame: int, config: RaceConfig) -> float:
        """Return a very small multiplicative frame correction."""
        bias = FRAME_BIAS.get(int(frame), 0.0)
        if not self.has_inner_frame_advantage(config):
            bias *= 0.35
        return 1.0 + bias

    def track_bonus(self, config: RaceConfig, mud_aptitude: float, stamina: float, acceleration: float) -> float:
        factor = TRACK_FACTOR.get(config.track_condition, TRACK_FACTOR["良"])
        speed_component = factor["speed"]
        stamina_component = 1.0 + ((stamina - 50.0) / 500.0) * factor["stamina"]
        acceleration_component = 1.0 + ((acceleration - 50.0) / 800.0) * factor["acceleration"]
        mud_component = 1.0 + ((mud_aptitude - 50.0) / 450.0) * factor["mud"]
        if config.track_condition == "良":
            return speed_component * acceleration_component
        return speed_component * stamina_component * mud_component


def get_course_bias(race_config: dict[str, Any] | RaceConfig) -> dict[str, float | str]:
    """Return simple course-bias corrections from meeting day and rail layout."""
    day_text = str(_config_value(race_config, "race_course_day", "1日目"))
    layout = str(_config_value(race_config, "course_layout", "A")).upper()
    match = re.search(r"\d+", day_text)
    day = int(match.group(0)) if match else 1

    inner_bias = 0.0
    outer_bias = 0.0
    front_bias = 0.0
    closer_bias = 0.0
    day_comment = "開催中盤"

    if 1 <= day <= 4:
        inner_bias += 0.05
        front_bias += 0.05
        day_comment = "開催前半"
    elif 9 <= day <= 12:
        outer_bias += 0.05
        closer_bias += 0.05
        day_comment = "開催後半"

    layout_comment = f"{layout}コース"
    if layout == "A":
        inner_bias += 0.03
        front_bias += 0.02
    elif layout == "C":
        outer_bias += 0.03
        closer_bias += 0.02
    elif layout == "D":
        outer_bias += 0.04
        closer_bias += 0.03

    selected_bias = str(_config_value(race_config, "track_bias", "標準"))
    effects = TRACK_BIAS_EFFECTS.get(selected_bias, {})
    inner_bias += float(effects.get("inner", 0.0)) / 200.0
    outer_bias += float(effects.get("outer", 0.0)) / 200.0
    front_bias += (float(effects.get("逃げ", 0.0)) + float(effects.get("先行", 0.0))) / 400.0
    closer_bias += (float(effects.get("差し", 0.0)) + float(effects.get("追込", 0.0))) / 400.0

    if inner_bias + front_bias > outer_bias + closer_bias:
        advantage = "内・先行がやや有利"
    elif outer_bias + closer_bias > inner_bias + front_bias:
        advantage = "外・差しがやや有利"
    else:
        advantage = "大きな偏りは少ない"

    return {
        "inner_bias": round(inner_bias, 4),
        "outer_bias": round(outer_bias, 4),
        "front_bias": round(front_bias, 4),
        "closer_bias": round(closer_bias, 4),
        "comment": f"{day_comment}の{layout_comment}、トラックバイアスは{selected_bias}想定で、{advantage}です。",
    }


def get_track_bias_fit_score(
    race_config: dict[str, Any] | RaceConfig,
    running_style: str,
    frame: int,
) -> float:
    """Return a restrained 0-100 fit score for the selected track bias."""
    selected = str(_config_value(race_config, "track_bias", "標準"))
    effects = TRACK_BIAS_EFFECTS.get(selected, {})
    score = 50.0 + float(effects.get(running_style, 0.0))
    if int(frame) <= 3:
        score += float(effects.get("inner", 0.0))
    elif int(frame) >= 7:
        score += float(effects.get("outer", 0.0))
    return max(0.0, min(100.0, score))


def estimate_course_fit_score(
    recent_races: list[Any],
    race_config: dict[str, Any] | RaceConfig | None,
) -> float:
    """Estimate target-course fit from actual recent-race conditions."""
    if race_config is None:
        return 50.0
    target_course = str(_config_value(race_config, "course", ""))
    target_distance = int(_config_value(race_config, "distance", 0) or 0)
    target_surface = str(_config_value(race_config, "surface", ""))
    target_direction = str(_config_value(race_config, "direction", ""))
    target_features = COURSE_FEATURES.get(target_course, {})
    buckets: dict[str, list[float]] = {
        "venue": [],
        "distance": [],
        "surface": [],
        "direction": [],
        "straight_slope": [],
    }
    for race in list(recent_races or [])[:5]:
        row = _race_row(race)
        performance = _race_performance_score(row)
        course = str(row.get("course") or row.get("venue") or row.get("競馬場") or "")
        distance = _to_int(row.get("distance") or row.get("距離"), 0)
        surface = str(row.get("surface") or row.get("芝/ダート") or "")
        direction = str(row.get("direction") or row.get("turn_direction") or row.get("回り") or "")
        if target_course and course == target_course:
            buckets["venue"].append(performance)
        if target_distance > 0 and distance > 0 and abs(distance - target_distance) <= max(200, target_distance * 0.12):
            buckets["distance"].append(performance)
        if target_surface and surface == target_surface:
            buckets["surface"].append(performance)
        if target_direction and direction == target_direction:
            buckets["direction"].append(performance)
        past_features = COURSE_FEATURES.get(course, {})
        if target_features and past_features:
            straight_diff = abs(float(target_features["straight_length"]) - float(past_features["straight_length"]))
            same_slope = bool(target_features["slope"]) == bool(past_features["slope"])
            if straight_diff <= 150 and same_slope:
                buckets["straight_slope"].append(performance)

    component = {key: (sum(values) / len(values) if values else 50.0) for key, values in buckets.items()}
    base_score = (
        component["venue"] * 0.25
        + component["distance"] * 0.25
        + component["surface"] * 0.20
        + component["direction"] * 0.15
        + component["straight_slope"] * 0.15
    )
    return max(0.0, min(100.0, base_score))


def _race_row(race: Any) -> dict[str, Any]:
    if isinstance(race, dict):
        return dict(race)
    row = dict(getattr(race, "raw", None) or {})
    for key in ("course", "distance", "surface", "track_condition", "finish_position", "margin", "field_size"):
        value = getattr(race, key, None)
        if value not in (None, ""):
            row.setdefault(key, value)
    return row


def _race_performance_score(row: dict[str, Any]) -> float:
    finish = _to_int(row.get("finish_position") or row.get("finish") or row.get("着順"), 0)
    field_size = max(2, _to_int(row.get("field_size") or row.get("頭数") or row.get("出走頭数"), 18))
    margin = _to_float(row.get("winner_time_diff") or row.get("margin_sec") or row.get("margin") or row.get("着差"), 0.8)
    finish_component = 50.0 if finish <= 0 else 100.0 * (1.0 - min(finish, field_size) / field_size)
    margin_component = max(0.0, min(100.0, 100.0 - max(0.0, margin) * 35.0))
    return max(0.0, min(100.0, finish_component * 0.65 + margin_component * 0.35))


def _to_int(value: Any, default: int) -> int:
    try:
        match = re.search(r"-?\d+", str(value))
        return int(match.group(0)) if match else default
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float) -> float:
    try:
        match = re.search(r"-?\d+(?:\.\d+)?", str(value))
        return float(match.group(0)) if match else default
    except (TypeError, ValueError):
        return default


def _config_value(race_config: dict[str, Any] | RaceConfig, key: str, default: Any) -> Any:
    if isinstance(race_config, dict):
        return race_config.get(key, default)
    return getattr(race_config, key, default)
