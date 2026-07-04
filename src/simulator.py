from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from course_db import CourseDB, get_course_bias, get_track_bias_fit_score
from horse_analyzer import STYLE_KEYS, HorseAbility
from pace_predictor import RacePace
from race_config import RaceConfig, calculate_weight_penalty
from style_adjuster import adjust_style_profile


STYLE_RUNNER = "\u9003\u3052"
STYLE_STALKER = "\u5148\u884c"
STYLE_VERSATILE = "\u81ea\u5728"
STYLE_CLOSER = "\u5dee\u3057"
STYLE_DEEP_CLOSER = "\u8ffd\u8fbc"
STYLE_RANK_BANDS = {
    STYLE_RUNNER: (0.00, 0.18),
    STYLE_STALKER: (0.18, 0.42),
    STYLE_VERSATILE: (0.30, 0.58),
    STYLE_CLOSER: (0.42, 0.72),
    STYLE_DEEP_CLOSER: (0.72, 1.00),
}
STYLE_ORDER = {
    STYLE_RUNNER: 0,
    STYLE_STALKER: 1,
    STYLE_VERSATILE: 2,
    STYLE_CLOSER: 3,
    STYLE_DEEP_CLOSER: 4,
}
STYLE_SEQUENCE = list(STYLE_ORDER.keys())
STYLE_GAP_BANDS = {
    STYLE_RUNNER: (0.0, 2.0),
    STYLE_STALKER: (2.5, 5.5),
    STYLE_VERSATILE: (4.0, 8.0),
    STYLE_CLOSER: (6.5, 12.0),
    STYLE_DEEP_CLOSER: (10.5, 18.0),
}
MAX_GAP_UNTIL_MID = 18.0
MAX_LEADER_GAP_UNTIL_MID = 4.0
MAX_GAP_DELTA_PER_STEP = 1.8


def _prepare_controlled_horse(row: dict[str, Any], race_config: dict, rng: random.Random) -> dict[str, object]:
    style = _controlled_style(row)
    race_power = _clamp_float(_to_float(row.get("race_power", row.get("avg_race_score", 70.0)), 70.0), 40.0, 95.0)
    stamina = _clamp_float(_to_float(row.get("stamina", 60.0), 60.0), 0.0, 100.0)
    acceleration = _clamp_float(_to_float(row.get("acceleration", 60.0), 60.0), 0.0, 100.0)
    pace_fit = _controlled_pace_fit(row, style)
    course_bias_fit = _controlled_course_bias_fit(row, race_config, style)
    performance_index = (
        race_power * 0.45
        + stamina * 0.20
        + acceleration * 0.20
        + pace_fit * 0.10
        + course_bias_fit * 0.05
    )
    style_rank = STYLE_ORDER.get(style, STYLE_ORDER[STYLE_VERSATILE])
    early_order_value = (len(STYLE_ORDER) - style_rank) * 20.0 + race_power * 0.07 + rng.uniform(0.0, 0.2)
    return {
        "horse_name": row.get("horse_name", ""),
        "horse_number": _to_int(row.get("horse_number", row.get("number", 0)), 0),
        "frame": _to_int(row.get("frame", 1), 1),
        "actual_running_style": style,
        "actual_running_style_fixed": style,
        "race_power": race_power,
        "stamina": stamina,
        "acceleration": acceleration,
        "pace_fit": pace_fit,
        "course_bias_fit": course_bias_fit,
        "performance_index": _clamp_float(performance_index, 0.0, 100.0),
        "early_order_value": early_order_value,
    }


def _controlled_order_value(horse: dict[str, object], late_phase: float) -> float:
    early_order_value = float(horse["early_order_value"])
    performance_index = float(horse["performance_index"])
    return early_order_value * (1.0 - late_phase) + performance_index * late_phase


def _controlled_style(row: dict[str, Any]) -> str:
    style = str(
        row.get("actual_running_style_fixed")
        or row.get("actual_running_style")
        or row.get("running_style")
        or row.get("primary_running_style")
        or STYLE_CLOSER
    )
    return style if style in STYLE_ORDER else STYLE_VERSATILE


def _controlled_pace_fit(row: dict[str, Any], style: str) -> float:
    style_probability = _to_float(row.get(f"adjusted_{style}", row.get(f"base_{style}", 0.0)), 0.0)
    if style_probability > 0:
        return _clamp_float(50.0 + style_probability * 50.0, 0.0, 100.0)
    if style in {STYLE_RUNNER, STYLE_STALKER}:
        return 62.0
    if style == STYLE_VERSATILE:
        return 60.0
    return 58.0


def _controlled_course_bias_fit(row: dict[str, Any], race_config: dict, style: str) -> float:
    bias = get_course_bias(race_config)
    frame = _to_int(row.get("frame", 1), 1)
    value = 50.0
    if frame <= 3:
        value += float(bias.get("inner_bias", 0.0)) * 100.0
    elif frame >= 7:
        value += float(bias.get("outer_bias", 0.0)) * 100.0
    if style in {STYLE_RUNNER, STYLE_STALKER}:
        value += float(bias.get("front_bias", 0.0)) * 100.0
    elif style in {STYLE_CLOSER, STYLE_DEEP_CLOSER}:
        value += float(bias.get("closer_bias", 0.0)) * 100.0
    return _clamp_float(value, 0.0, 100.0)


def _controlled_lane(horse: dict[str, object], progress: float) -> float:
    lane = max(0.0, min(7.0, float(horse.get("frame", 1)) - 1.0 + (int(horse.get("horse_number", 0)) % 2) * 0.22))
    style = _formation_fixed_style(horse)
    if progress >= 0.60 and style in {STYLE_CLOSER, STYLE_DEEP_CLOSER}:
        lane += 0.45 if style == STYLE_CLOSER else 0.65
    return max(0.0, min(7.0, lane))


def _controlled_distance(race_config: dict | object) -> float:
    if isinstance(race_config, dict):
        value = race_config.get("distance", 2200)
    else:
        value = getattr(race_config, "distance", 2200)
    return max(100.0, _to_float(value, 2200.0))


def _config_get_for_formation(race_config: dict | object, key: str, default: object) -> object:
    if isinstance(race_config, dict):
        return race_config.get(key, default)
    return getattr(race_config, key, default)


def _to_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: object, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _optional_float(value: object) -> float | None:
    try:
        if value in (None, ""):
            return None
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


def _clamp_float(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))


EARLY_STYLE_GAP_RANGES = dict(
    zip(
        STYLE_SEQUENCE,
        [
            (0.0, 8.0),
            (12.0, 34.0),
            (26.0, 55.0),
            (45.0, 82.0),
            (75.0, 120.0),
        ],
    )
)
STYLE_GAP_RANGES = EARLY_STYLE_GAP_RANGES
FORMATION_STYLE_GAP_RANGES = EARLY_STYLE_GAP_RANGES
MIN_VISIBLE_MIDFIELD_SPREAD = 40.0
MAX_VISIBLE_MIDFIELD_SPREAD = 110.0
MAX_LATE_GAIN_M = 70.0
MAX_STRETCH_GAIN_M = 65.0
MIN_NIGE_SENKO_GAP = 8.0
STYLE_ATTACK_START = {
    STYLE_RUNNER: 0.00,
    STYLE_STALKER: 0.15,
    STYLE_VERSATILE: 0.25,
    STYLE_CLOSER: 0.35,
    STYLE_DEEP_CLOSER: 0.50,
}
PACE_STRETCH_ADJUSTMENT = {
    "slow": {STYLE_RUNNER: 8.0, STYLE_STALKER: 6.0, STYLE_VERSATILE: 2.0, STYLE_CLOSER: -3.0, STYLE_DEEP_CLOSER: -8.0},
    "medium": {style: 0.0 for style in STYLE_SEQUENCE},
    "high": {STYLE_RUNNER: -10.0, STYLE_STALKER: -5.0, STYLE_VERSATILE: 2.0, STYLE_CLOSER: 7.0, STYLE_DEEP_CLOSER: 10.0},
}


def generate_controlled_race_timeline(
    horses: list[dict],
    race_config: dict,
    n_frames: int = 300,
    seed: int | None = 42,
) -> list[dict]:
    """Generate a side-scroll friendly timeline using formation gaps.

    This controlled timeline intentionally avoids speed accumulation. Until
    60% progress, style-based gaps drive the visible formation; after that,
    race ability gradually reshapes the order.
    """
    if not horses:
        return []

    rng = random.Random(seed)
    distance = _controlled_distance(race_config)
    prepared_horses = [_formation_prepare_horse(row, race_config, rng) for row in horses]
    _formation_assign_style_slots(prepared_horses)
    _formation_assign_mid_race_move(prepared_horses)
    _formation_assign_final_performance(prepared_horses)
    _formation_assign_late_power(prepared_horses, seed)
    _formation_assign_final_stretch_scores(prepared_horses)

    frame_count = max(2, int(n_frames))
    total_time = distance / 15.5
    final_stretch_start = _formation_final_stretch_start(race_config, distance)
    previous_gaps: dict[int, float] = {}
    timeline: list[dict[str, object]] = []

    for frame_index in range(frame_count):
        progress = frame_index / max(1, frame_count - 1)
        leader_position = min(distance, progress * distance)
        late_phase = max(0.0, (progress - 0.60) / 0.40)

        gaps: dict[int, float] = {}
        for horse in prepared_horses:
            horse_number = int(horse["horse_number"])
            gap = _formation_gap_for_progress(horse, progress, final_stretch_start)
            if 0.45 <= progress < final_stretch_start and horse_number in previous_gaps:
                previous_gap = previous_gaps[horse_number]
                gap = previous_gap + _clamp_float(
                    gap - previous_gap,
                    -1.2,
                    1.2,
                )
            gaps[horse_number] = gap

        if progress < final_stretch_start:
            gaps = _formation_clamp_gaps_to_style_bands(gaps, prepared_horses, progress, final_stretch_start)
        else:
            gaps = _formation_normalize_gaps(gaps)
            if previous_gaps:
                gaps = {
                    number: previous_gaps.get(number, gap)
                    + _clamp_float(gap - previous_gaps.get(number, gap), -2.0, 2.0)
                    for number, gap in gaps.items()
                }
                gaps = _formation_normalize_gaps(gaps)

        nige_gaps = [
            gaps.get(int(horse.get("horse_number", 0)), 0.0)
            for horse in prepared_horses
            if _formation_fixed_style(horse) == STYLE_RUNNER
        ]
        senko_gaps = [
            gaps.get(int(horse.get("horse_number", 0)), 0.0)
            for horse in prepared_horses
            if _formation_fixed_style(horse) == STYLE_STALKER
        ]
        nige_senko_gap = min(senko_gaps) - min(nige_gaps) if nige_gaps and senko_gaps else None

        frame_horses: list[dict[str, object]] = []
        for horse in prepared_horses:
            horse_number = int(horse["horse_number"])
            fixed_style = _formation_fixed_style(horse)
            gap_from_leader = gaps.get(horse_number, 0.0)
            position_m = max(0.0, min(distance, leader_position - gap_from_leader))
            if progress > 0.95:
                position_m = max(0.0, position_m + float(horse.get("tie_breaker", 0.0)))
            frame_horses.append(
                {
                    "horse_number": horse_number,
                    "horse_name": str(horse.get("horse_name", "")),
                    "frame": int(horse.get("frame", 1)),
                    "actual_running_style": fixed_style,
                    "actual_running_style_fixed": fixed_style,
                    "position_m": round(position_m, 3),
                    "finish_position_m": round(position_m, 3),
                    "rank": 0,
                    "lane": round(_controlled_lane(horse, progress), 3),
                    "gap_from_leader": round(gap_from_leader, 3),
                    "nige_senko_gap": round(float(nige_senko_gap), 3) if nige_senko_gap is not None else "",
                    "early_gap_range": _formation_gap_range_label(fixed_style, progress, final_stretch_start),
                    "final_stretch_start_progress": round(final_stretch_start, 3),
                    "horse_ability_score": round(float(horse.get("horse_ability_score", horse.get("race_power", 70.0))), 3),
                    "popularity_score": round(float(horse.get("popularity_score", 50.0)), 3),
                    "race_level_score": round(float(horse.get("race_level_score", 60.0)), 3),
                    "finish_score": round(float(horse.get("finish_score", 50.0)), 3),
                    "margin_score": round(float(horse.get("margin_score", 50.0)), 3),
                    "time_score": round(float(horse.get("time_score", 50.0)), 3),
                    "race_power": round(float(horse["race_power"]), 3),
                    "performance_index": round(float(horse["performance_index"]), 3),
                    "mid_race_move_score": round(float(horse["mid_race_move_score"]), 3),
                    "normalized_mid_race_move": round(float(horse["normalized_mid_race_move"]), 3),
                    "final_performance_score": round(float(horse["final_performance_score"]), 3),
                    "late_power": round(float(horse["late_power"]), 3),
                    "random_noise": round(float(horse.get("late_random_noise", 0.0)), 3),
                    "normalized_final_performance": round(float(horse["normalized_final_performance"]), 3),
                    "normalized_late_power": round(float(horse["normalized_late_power"]), 3),
                    "late_phase": round(late_phase, 3),
                    "late_ratio": round(max(0.0, (progress - 0.80) / 0.20), 3),
                    "gap_adjustment": round(float(horse.get("_last_gap_adjustment", 0.0)), 3),
                    "escape_fade": round(float(horse.get("_last_escape_fade", 0.0)), 4),
                    "stalker_fade": round(float(horse.get("_last_stalker_fade", 0.0)), 4),
                    "fade_penalty": round(float(horse.get("_last_fade_penalty", 0.0)), 4),
                    "late_gain_multiplier": round(float(horse.get("_last_late_gain_multiplier", 1.0)), 3),
                    "tie_breaker": round(float(horse.get("tie_breaker", 0.0)), 3),
                    "ability_factor": round(_formation_ability_factor(progress), 3),
                    "early_aggressiveness": round(float(horse.get("early_aggressiveness", 0.5)), 3),
                    "mid_positioning": round(float(horse.get("mid_positioning", 0.5)), 3),
                    "late_kick_timing": round(float(horse.get("late_kick_timing", 0.5)), 3),
                    "sustain_speed": round(float(horse.get("sustain_speed", 0.55)), 3),
                    "time_reliability": round(float(horse.get("time_reliability", 0.0)), 3),
                    "recent_time_score": round(float(horse.get("recent_time_score", 55.0)), 3),
                    "early_push_score": round(float(horse.get("early_push_score", 50.0)), 3),
                    "mid_cruise_score": round(float(horse.get("mid_cruise_score", 55.0)), 3),
                    "fade_resistance_score": round(float(horse.get("fade_resistance_score", 55.0)), 3),
                    "sustain_speed_score": round(float(horse.get("sustain_speed_score", 55.0)), 3),
                    "pace_resilience_score": round(float(horse.get("pace_resilience_score", 55.0)), 3),
                    "agari_reliability": round(float(horse.get("agari_reliability", 50.0)), 3),
                    "stamina": round(float(horse.get("stamina", 60.0)), 3),
                    "acceleration": round(float(horse.get("acceleration", 60.0)), 3),
                    "last3f_score": round(float(horse.get("last3f_score", 60.0)), 3),
                    "late_kick_score": round(float(horse.get("late_kick_score", 55.0)), 3),
                    "avg_last3f": horse.get("avg_last3f"),
                    "best_last3f": horse.get("best_last3f"),
                    "last3f_consistency": round(float(horse.get("last3f_consistency", 50.0)), 3),
                    "late_gain_score": round(float(horse.get("late_gain_score", 50.0)), 3),
                    "carried_weight": round(float(horse.get("carried_weight", 56.0)), 1),
                    "weight_penalty": round(float(horse.get("weight_penalty", 0.0)), 3),
                    "pace_fit_score": round(float(horse.get("pace_fit_score", horse.get("pace_fit", 60.0))), 3),
                    "final_stretch_score": round(float(horse.get("final_stretch_score", 50.0)), 3),
                    "style_attack_ratio": round(float(horse.get("_last_style_attack_ratio", 0.0)), 3),
                    "straight_attack_score": round(float(horse.get("_last_straight_attack_score", 0.0)), 3),
                    "straight_ratio": round(float(horse.get("_last_straight_ratio", 0.0)), 3),
                    "pace": str(horse.get("race_pace", "")),
                }
            )

        frame_horses = sorted(
            frame_horses,
            key=lambda row: (
                -float(row["position_m"]),
                STYLE_ORDER.get(str(row.get("actual_running_style")), 99),
                int(row.get("horse_number", 0)),
            ),
        )
        for rank, horse in enumerate(frame_horses, start=1):
            horse["rank"] = rank

        previous_gaps = {
            int(horse["horse_number"]): float(horse["gap_from_leader"])
            for horse in frame_horses
        }
        timeline.append(
            {
                "time": round(total_time * progress, 3),
                "progress": round(progress, 4),
                "phase": _formation_phase(progress, final_stretch_start),
                "timeline_mode": "controlled",
                "horses": frame_horses,
            }
        )
    return timeline


def _formation_prepare_horse(row: dict[str, Any], race_config: dict, rng: random.Random) -> dict[str, object]:
    style = _controlled_style(row)
    horse_ability_score = _clamp_float(
        _to_float(row.get("horse_ability_score", row.get("race_power", row.get("avg_race_score", 70.0))), 70.0),
        35.0,
        98.0,
    )
    race_power = _clamp_float(_to_float(row.get("race_power", horse_ability_score), horse_ability_score), 40.0, 95.0)
    stamina = _clamp_float(_to_float(row.get("stamina", 60.0), 60.0), 0.0, 100.0)
    acceleration = _clamp_float(_to_float(row.get("acceleration", 60.0), 60.0), 0.0, 100.0)
    pace_fit = _clamp_float(
        _to_float(row.get("pace_fit", row.get("pace_fit_score")), _controlled_pace_fit(row, style)),
        0.0,
        100.0,
    )
    race_pace = str(row.get("race_pace", _config_get_for_formation(race_config, "pace", "medium")))
    pace_fit_score = _clamp_float(
        pace_fit + PACE_STRETCH_ADJUSTMENT.get(race_pace, PACE_STRETCH_ADJUSTMENT["medium"]).get(style, 0.0),
        0.0,
        100.0,
    )
    carried_weight = _clamp_float(_to_float(row.get("carried_weight", 56.0), 56.0), 40.0, 65.0)
    weight_penalty = calculate_weight_penalty(carried_weight)
    course_bias_fit = _controlled_course_bias_fit(row, race_config, style)
    course_fit_score = _clamp_float(_to_float(row.get("course_fit_score", 50.0), 50.0), 0.0, 100.0)
    track_bias_fit_score = _clamp_float(
        _to_float(
            row.get("track_bias_fit_score"),
            get_track_bias_fit_score(race_config, style, _to_int(row.get("frame", 1), 1)),
        ),
        0.0,
        100.0,
    )
    last3f_score = _clamp_float(_to_float(row.get("last3f_score", row.get("last3f", acceleration)), acceleration), 0.0, 100.0)
    late_kick_score = _clamp_float(_to_float(row.get("late_kick_score", last3f_score), last3f_score), 0.0, 100.0)
    avg_last3f = _optional_float(row.get("avg_last3f"))
    best_last3f = _optional_float(row.get("best_last3f"))
    last3f_consistency = _clamp_float(_to_float(row.get("last3f_consistency", 50.0), 50.0), 0.0, 100.0)
    late_gain_score = _clamp_float(_to_float(row.get("late_gain_score", 50.0), 50.0), 0.0, 100.0)
    recent_time_score = _clamp_float(_to_float(row.get("recent_time_score", 55.0), 55.0), 0.0, 100.0)
    early_push_score = _clamp_float(_to_float(row.get("early_push_score", 50.0), 50.0), 0.0, 100.0)
    mid_cruise_score = _clamp_float(_to_float(row.get("mid_cruise_score", 55.0), 55.0), 0.0, 100.0)
    fade_resistance_score = _clamp_float(_to_float(row.get("fade_resistance_score", 55.0), 55.0), 0.0, 100.0)
    pace_resilience_score = _clamp_float(_to_float(row.get("pace_resilience_score", 55.0), 55.0), 0.0, 100.0)
    agari_reliability = _clamp_float(_to_float(row.get("agari_reliability", max(last3f_consistency, last3f_score)), max(last3f_consistency, last3f_score)), 0.0, 100.0)
    early_aggressiveness = _clamp_float(_to_float(row.get("early_aggressiveness", 0.50), 0.50), 0.0, 1.0)
    mid_positioning = _clamp_float(_to_float(row.get("mid_positioning", 0.50), 0.50), 0.0, 1.0)
    late_kick_timing = _clamp_float(_to_float(row.get("late_kick_timing", 0.50), 0.50), 0.0, 1.0)
    sustain_speed = _clamp_float(_to_float(row.get("sustain_speed", 0.55), 0.55), 0.0, 1.0)
    sustain_speed_score = _clamp_float(_to_float(row.get("sustain_speed_score", sustain_speed * 100.0), sustain_speed * 100.0), 0.0, 100.0)
    time_reliability = _clamp_float(_to_float(row.get("time_reliability", 0.0), 0.0), 0.0, 1.0)
    performance_index = (
        race_power * 0.45
        + stamina * 0.20
        + acceleration * 0.20
        + pace_fit * 0.10
        + course_bias_fit * 0.05
    )
    final_performance_score = (
        horse_ability_score * 0.30
        + race_power * 0.15
        + late_kick_score * 0.20
        + stamina * 0.10
        + pace_fit * 0.10
        + course_fit_score * 0.10
        + track_bias_fit_score * 0.05
    )
    mid_race_move_score = (
        race_power * 0.20
        + sustain_speed_score * 0.25
        + stamina * 0.15
        + mid_cruise_score * 0.25
        + recent_time_score * 0.15
    )
    style_rank = STYLE_ORDER.get(style, STYLE_ORDER[STYLE_SEQUENCE[2]])
    early_order_value = (len(STYLE_ORDER) - style_rank) * 20.0 + race_power * 0.07 + rng.uniform(0.0, 0.2)
    return {
        "horse_name": row.get("horse_name", ""),
        "horse_number": _to_int(row.get("horse_number", row.get("number", 0)), 0),
        "frame": _to_int(row.get("frame", 1), 1),
        "actual_running_style": style,
        "actual_running_style_fixed": style,
        "horse_ability_score": horse_ability_score,
        "popularity_score": _clamp_float(_to_float(row.get("popularity_score", 50.0), 50.0), 0.0, 100.0),
        "race_level_score": _clamp_float(_to_float(row.get("race_level_score", 60.0), 60.0), 0.0, 100.0),
        "finish_score": _clamp_float(_to_float(row.get("finish_score", 50.0), 50.0), 0.0, 100.0),
        "margin_score": _clamp_float(_to_float(row.get("margin_score", 50.0), 50.0), 0.0, 100.0),
        "time_score": _clamp_float(_to_float(row.get("time_score", 50.0), 50.0), 0.0, 100.0),
        "race_power": race_power,
        "stamina": stamina,
        "acceleration": acceleration,
        "pace_fit": pace_fit,
        "race_pace": race_pace,
        "pace_fit_score": pace_fit_score,
        "carried_weight": carried_weight,
        "weight_penalty": weight_penalty,
        "distance": _controlled_distance(race_config),
        "track_condition": str(_config_get_for_formation(race_config, "track_condition", "")),
        "course_bias_fit": course_bias_fit,
        "course_fit_score": course_fit_score,
        "track_bias_fit_score": track_bias_fit_score,
        "jockey_score": _clamp_float(_to_float(row.get("jockey_score", 50.0), 50.0), 0.0, 100.0),
        "jockey": str(row.get("jockey", "")),
        "race_strength_score": _clamp_float(_to_float(row.get("race_strength_score", 50.0), 50.0), 0.0, 100.0),
        "race_strength_adjusted_score": _clamp_float(_to_float(row.get("race_strength_adjusted_score", 50.0), 50.0), 0.0, 100.0),
        "elo_rating": _to_float(row.get("elo_rating", 1500.0), 1500.0),
        "normalized_elo_score": _clamp_float(_to_float(row.get("normalized_elo_score", 50.0), 50.0), 0.0, 100.0),
        "relative_agari_score": _clamp_float(_to_float(row.get("relative_agari_score", 50.0), 50.0), 0.0, 100.0),
        "last3f_score": last3f_score,
        "late_kick_score": late_kick_score,
        "avg_last3f": avg_last3f,
        "best_last3f": best_last3f,
        "last3f_consistency": last3f_consistency,
        "late_gain_score": late_gain_score,
        "recent_time_score": recent_time_score,
        "early_push_score": early_push_score,
        "mid_cruise_score": mid_cruise_score,
        "fade_resistance_score": fade_resistance_score,
        "sustain_speed_score": sustain_speed_score,
        "pace_resilience_score": pace_resilience_score,
        "agari_reliability": agari_reliability,
        "early_aggressiveness": early_aggressiveness,
        "mid_positioning": mid_positioning,
        "late_kick_timing": late_kick_timing,
        "sustain_speed": sustain_speed,
        "time_reliability": time_reliability,
        "performance_index": _clamp_float(performance_index, 0.0, 100.0),
        "mid_race_move_score": _clamp_float(mid_race_move_score, 0.0, 100.0),
        "normalized_mid_race_move": 0.0,
        "final_performance_score": _clamp_float(final_performance_score, 0.0, 100.0),
        "late_power": _clamp_float(final_performance_score, 0.0, 100.0),
        "normalized_final_performance": 0.0,
        "normalized_late_power": 0.0,
        "late_random_noise": 0.0,
        "late_gap_jitter": 0.0,
        "tie_breaker": 0.0,
        "_last_gap_adjustment": 0.0,
        "_last_fade_penalty": 0.0,
        "_last_escape_fade": 0.0,
        "_last_stalker_fade": 0.0,
        "_last_late_gain_multiplier": 1.0,
        "_last_straight_attack_score": 0.0,
        "_last_straight_ratio": 0.0,
        "_last_style_attack_ratio": 0.0,
        "early_order_value": early_order_value,
        "style_target_gap": _formation_style_target_gap(style, race_power, stamina, acceleration, rng, early_push_score),
    }


def _formation_assign_style_slots(prepared_horses: list[dict[str, object]]) -> None:
    by_style: dict[str, list[dict[str, object]]] = {style: [] for style in STYLE_SEQUENCE}
    for horse in prepared_horses:
        by_style.setdefault(_formation_fixed_style(horse), []).append(horse)

    for style, group in by_style.items():
        if not group:
            continue
        low, high = FORMATION_STYLE_GAP_RANGES.get(style, FORMATION_STYLE_GAP_RANGES[STYLE_SEQUENCE[2]])
        group.sort(key=lambda row: (-float(row.get("early_order_value", 0.0)), int(row.get("horse_number", 0))))
        if len(group) == 1:
            horse = group[0]
            horse["style_target_gap"] = _clamp_float(float(horse["style_target_gap"]), low, high)
            continue
        for slot, horse in enumerate(group):
            slot_ratio = slot / max(1, len(group) - 1)
            slotted_gap = low + (high - low) * slot_ratio
            horse["style_target_gap"] = _clamp_float(
                float(horse["style_target_gap"]) * 0.58 + slotted_gap * 0.42,
                low,
                high,
            )


def _formation_assign_mid_race_move(prepared_horses: list[dict[str, object]]) -> None:
    scores = [float(horse.get("mid_race_move_score", 60.0)) for horse in prepared_horses]
    if not scores:
        return
    mean_score = float(np.mean(scores))
    std_score = max(1.0, float(np.std(scores)))
    for horse in prepared_horses:
        raw = (float(horse.get("mid_race_move_score", 60.0)) - mean_score) / (std_score * 1.25)
        horse["normalized_mid_race_move"] = _clamp_float(raw, -1.0, 1.0)


def _formation_assign_final_performance(prepared_horses: list[dict[str, object]]) -> None:
    scores = [float(horse.get("final_performance_score", 60.0)) for horse in prepared_horses]
    if not scores:
        return
    mean_score = float(np.mean(scores))
    std_score = max(1.0, float(np.std(scores)))
    for horse in prepared_horses:
        raw = (float(horse.get("final_performance_score", 60.0)) - mean_score) / (std_score * 1.25)
        horse["normalized_final_performance"] = _clamp_float(raw, -1.0, 1.0)


def _formation_assign_late_power(prepared_horses: list[dict[str, object]], seed: int | None) -> None:
    base_seed = 0 if seed is None else int(seed)
    for horse in prepared_horses:
        horse_number = int(horse.get("horse_number", 0))
        horse_rng = random.Random(base_seed + horse_number)
        random_noise = horse_rng.uniform(-3.0, 3.0)
        horse["late_random_noise"] = random_noise
        horse["late_gap_jitter"] = horse_rng.uniform(-2.0, 2.0)
        horse["tie_breaker"] = horse_rng.uniform(-0.8, 0.8)
        horse["late_power"] = _clamp_float(
            float(horse.get("final_performance_score", 60.0)) * 0.40
            + float(horse.get("horse_ability_score", horse.get("race_power", 70.0))) * 0.22
            + float(horse.get("late_kick_score", horse.get("last3f_score", 60.0))) * 0.15
            + float(horse.get("race_power", 70.0)) * 0.10
            + float(horse.get("stamina", 60.0)) * 0.08
            + float(horse.get("pace_fit", 60.0)) * 0.05
            + random_noise * 0.60,
            0.0,
            100.0,
        )

    scores = [float(horse.get("late_power", 60.0)) for horse in prepared_horses]
    if not scores:
        return
    mean_score = float(np.mean(scores))
    std_score = max(1.0, float(np.std(scores)))
    for horse in prepared_horses:
        raw = (float(horse.get("late_power", 60.0)) - mean_score) / (std_score * 1.15)
        horse["normalized_late_power"] = _clamp_float(raw, -1.25, 1.25)


def _formation_final_stretch_score(horse: dict[str, object]) -> float:
    """Score the final stretch without granting a fixed bonus for style."""
    score = (
        float(horse.get("late_kick_score", 50.0)) * 0.30
        + float(horse.get("late_gain_score", 50.0)) * 0.17
        + float(horse.get("pace_fit_score", horse.get("pace_fit", 50.0))) * 0.15
        + float(horse.get("stamina", 50.0)) * 0.10
        + float(horse.get("recent_time_score", 50.0)) * 0.10
        + float(horse.get("course_fit_score", 50.0)) * 0.10
        + float(horse.get("track_bias_fit_score", 50.0)) * 0.08
        - float(horse.get("weight_penalty", 0.0)) * 0.05
    )
    return _clamp_float(score, 0.0, 100.0)


def _formation_assign_final_stretch_scores(prepared_horses: list[dict[str, object]]) -> None:
    scores = [_formation_final_stretch_score(horse) for horse in prepared_horses]
    if not scores:
        return
    low = min(scores)
    high = max(scores)
    spread = high - low
    for horse, score in zip(prepared_horses, scores):
        horse["final_stretch_score"] = score
        if spread < 1e-9:
            attack_power = 0.50
        else:
            attack_power = (score - low) / spread
        horse["normalized_final_stretch_score"] = _clamp_float(attack_power, 0.0, 1.0)


def _formation_style_target_gap(
    style: str,
    race_power: float,
    stamina: float,
    acceleration: float,
    rng: random.Random,
    early_push_score: float = 50.0,
) -> float:
    low, high = FORMATION_STYLE_GAP_RANGES.get(style, FORMATION_STYLE_GAP_RANGES[STYLE_SEQUENCE[2]])
    ability = _clamp_float((race_power * 0.35 + stamina * 0.15 + acceleration * 0.25 + early_push_score * 0.25) / 100.0, 0.0, 1.0)
    gap_ratio = 1.0 - ability
    return _clamp_float(low + (high - low) * gap_ratio + rng.uniform(-1.25, 1.25), low, high)


def _formation_phase(progress: float, final_stretch_start: float = 0.75) -> str:
    if progress < 0.20:
        return "start_position_battle"
    if progress < 0.55:
        return "style_formation"
    if progress < final_stretch_start:
        return "mid_race_transition"
    return "final_stretch_battle"


def _formation_gap_for_progress(
    horse: dict[str, object],
    progress: float,
    final_stretch_start: float,
) -> float:
    style = _formation_fixed_style(horse)
    target_gap = float(horse.get("style_target_gap", 30.0))
    early_aggressiveness = _clamp_float(float(horse.get("early_aggressiveness", 0.5)), 0.0, 1.0)
    mid_positioning = _clamp_float(float(horse.get("mid_positioning", 0.5)), 0.0, 1.0)
    early_push = _clamp_float(float(horse.get("early_push_score", early_aggressiveness * 100.0)) / 100.0, 0.0, 1.0)
    mid_cruise = _clamp_float(float(horse.get("mid_cruise_score", mid_positioning * 100.0)) / 100.0, 0.0, 1.0)
    if progress < 0.20:
        formation_ratio = progress / 0.20
        gap = target_gap * formation_ratio
        if style == STYLE_SEQUENCE[0] and progress < 0.15:
            gap *= 0.45
        gap += (1.0 - max(early_aggressiveness, early_push)) * 10.0 * formation_ratio
        return gap

    gap = target_gap
    if progress < 0.45:
        mid_ratio = _clamp_float((progress - 0.20) / 0.25, 0.0, 1.0)
        early_adjustment = (1.0 - max(early_aggressiveness, early_push)) * 10.0
        mid_adjustment = (1.0 - max(mid_positioning, mid_cruise)) * 8.0
        gap += early_adjustment * (1.0 - mid_ratio) + mid_adjustment * mid_ratio
        return gap

    if progress < final_stretch_start:
        free_ratio = _clamp_float((progress - 0.45) / max(0.01, final_stretch_start - 0.45), 0.0, 1.0)
        move = float(horse.get("normalized_mid_race_move", 0.0))
        gap -= free_ratio * max(0.0, move) * 11.0
        gap += free_ratio * max(0.0, -move) * 6.0
        gap -= free_ratio * _clamp_float((float(horse.get("sustain_speed", 0.55)) - 0.55) / 0.45, -0.5, 1.0) * 4.0
        if progress >= 0.60 and style in {STYLE_SEQUENCE[3], STYLE_SEQUENCE[4]}:
            advance_ratio = _clamp_float((progress - 0.60) / max(0.01, final_stretch_start - 0.60), 0.0, 1.0)
            mild_kick = _clamp_float((float(horse.get("late_kick_score", 55.0)) - 55.0) / 45.0, 0.0, 1.0)
            gap -= advance_ratio * mild_kick * (6.0 if style == STYLE_SEQUENCE[3] else 8.0)
        return gap

    straight_ratio = _clamp_float((progress - final_stretch_start) / max(0.01, 1.0 - final_stretch_start), 0.0, 1.0)
    # Carry the completed mid-race advance into the stretch instead of
    # snapping every horse back to its original style target.
    move = float(horse.get("normalized_mid_race_move", 0.0))
    gap -= max(0.0, move) * 11.0
    gap += max(0.0, -move) * 6.0
    gap -= _clamp_float((float(horse.get("sustain_speed", 0.55)) - 0.55) / 0.45, -0.5, 1.0) * 4.0
    if style in {STYLE_SEQUENCE[3], STYLE_SEQUENCE[4]}:
        mild_kick = _clamp_float((float(horse.get("late_kick_score", 55.0)) - 55.0) / 45.0, 0.0, 1.0)
        gap -= mild_kick * (6.0 if style == STYLE_SEQUENCE[3] else 8.0)
    attack_start = STYLE_ATTACK_START.get(style, 0.25)
    style_attack_ratio = _clamp_float(
        (straight_ratio - attack_start) / max(0.01, 1.0 - attack_start),
        0.0,
        1.0,
    )
    attack_power = _clamp_float(float(horse.get("normalized_final_stretch_score", 0.5)), 0.0, 1.0)
    gap_adjustment = attack_power * style_attack_ratio * MAX_STRETCH_GAIN_M
    gap += float(horse.get("late_gap_jitter", 0.0)) * straight_ratio * 0.55
    gap -= gap_adjustment
    escape_fade = _formation_escape_fade(horse) if style == STYLE_SEQUENCE[0] else 0.0
    stalker_fade = _formation_stalker_fade(horse) if style == STYLE_SEQUENCE[1] else 0.0
    if style == STYLE_SEQUENCE[0]:
        gap += straight_ratio * escape_fade * 12.0
    elif style == STYLE_SEQUENCE[1]:
        gap += straight_ratio * stalker_fade * 10.0
    horse["_last_escape_fade"] = escape_fade
    horse["_last_stalker_fade"] = stalker_fade
    horse["_last_fade_penalty"] = escape_fade if style == STYLE_SEQUENCE[0] else stalker_fade
    horse["_last_late_gain_multiplier"] = 1.0
    horse["_last_gap_adjustment"] = gap_adjustment
    horse["_last_straight_attack_score"] = float(horse.get("final_stretch_score", 50.0))
    horse["_last_straight_ratio"] = straight_ratio
    horse["_last_style_attack_ratio"] = style_attack_ratio
    return _clamp_float(gap, -120.0, 140.0)


def _formation_acceleration_effect(horse: dict[str, object]) -> float:
    acceleration = float(horse.get("acceleration", 60.0))
    stamina = float(horse.get("stamina", 60.0))
    pace_fit = float(horse.get("pace_fit", 60.0))
    course_bias_fit = float(horse.get("course_bias_fit", 50.0))
    raw = (acceleration * 0.55 + stamina * 0.18 + pace_fit * 0.17 + course_bias_fit * 0.10) / 70.0
    return _clamp_float(raw, 0.65, 1.45)


def _formation_escape_fade(horse: dict[str, object]) -> float:
    stamina_factor = _clamp_float(float(horse.get("stamina", 60.0)) / 100.0, 0.0, 1.0)
    fade_resistance_factor = _clamp_float(float(horse.get("fade_resistance_score", 55.0)) / 100.0, 0.0, 1.0)
    pace_resilience_factor = _clamp_float(float(horse.get("pace_resilience_score", 55.0)) / 100.0, 0.0, 1.0)
    race_pace = str(horse.get("race_pace", "medium"))
    distance = float(horse.get("distance", 2000.0))
    track_condition = str(horse.get("track_condition", ""))

    base_fade = 0.16
    pace_pressure_factor = 1.3 if race_pace == "high" else 0.7 if race_pace == "slow" else 1.0
    distance_factor = 1.15 if distance >= 2200 else 1.0
    track_factor = 1.0
    if any(word in track_condition for word in ["重", "不良", "heavy", "bad"]):
        track_factor = 1.08
    elif "良" in track_condition:
        track_factor = 0.96

    fade = (
        base_fade
        * pace_pressure_factor
        * distance_factor
        * track_factor
        * (1.0 - fade_resistance_factor)
        * (1.0 - stamina_factor)
        * (1.0 - 0.35 * pace_resilience_factor)
    )
    return _clamp_float(fade, 0.0, 0.16)


def _formation_stalker_fade(horse: dict[str, object]) -> float:
    race_pace = str(horse.get("race_pace", "medium"))
    if race_pace == "high":
        pace_pressure = 1.0
    elif race_pace == "slow":
        pace_pressure = 0.12
    else:
        pace_pressure = 0.25
    stamina_factor = _clamp_float(float(horse.get("stamina", 60.0)) / 100.0, 0.0, 1.0)
    race_power_factor = _clamp_float(float(horse.get("race_power", 70.0)) / 100.0, 0.0, 1.0)
    sustain_factor = _clamp_float(float(horse.get("sustain_speed_score", 55.0)) / 100.0, 0.0, 1.0)
    fade_resistance_factor = _clamp_float(float(horse.get("fade_resistance_score", 55.0)) / 100.0, 0.0, 1.0)
    fade = (
        pace_pressure
        * (1.0 - stamina_factor)
        * 0.12
        * (1.0 - 0.20 * race_power_factor)
        * (1.0 - 0.25 * sustain_factor)
        * (1.0 - 0.35 * fade_resistance_factor)
    )
    return _clamp_float(fade, 0.0, 0.12)


def _formation_late_gain_multiplier(horse: dict[str, object]) -> float:
    style = _formation_fixed_style(horse)
    race_pace = str(horse.get("race_pace", "medium"))
    acceleration = float(horse.get("acceleration", 60.0))
    last3f_score = float(horse.get("last3f_score", acceleration))
    elite_kick = _clamp_float(((acceleration + last3f_score) / 2.0 - 85.0) / 15.0, 0.0, 1.0)
    if style == STYLE_SEQUENCE[1]:
        return _clamp_float(0.72 + elite_kick * 0.18, 0.70, 0.90)
    if style == STYLE_SEQUENCE[3]:
        if race_pace == "high":
            return 1.18
        if race_pace == "slow":
            return 0.95
        return 1.12
    if style == STYLE_SEQUENCE[4]:
        return 1.22 if race_pace == "high" else 1.15 if race_pace == "medium" else 0.85
    if style == STYLE_SEQUENCE[0]:
        return 0.95 if race_pace == "slow" else 0.88
    return 1.00


def _formation_straight_attack_score(horse: dict[str, object]) -> float:
    style = _formation_fixed_style(horse)
    stamina = float(horse.get("stamina", 60.0))
    race_power = float(horse.get("race_power", 70.0))
    sustain_speed = float(horse.get("sustain_speed_score", float(horse.get("sustain_speed", 0.55)) * 100.0))
    pace_fit = float(horse.get("pace_fit", 60.0))
    acceleration = float(horse.get("acceleration", 60.0))
    late_kick_score = float(horse.get("late_kick_score", horse.get("last3f_score", 60.0)))
    late_gain_score = float(horse.get("late_gain_score", 50.0))
    fade_resistance = float(horse.get("fade_resistance_score", 55.0))
    mid_cruise = float(horse.get("mid_cruise_score", 55.0))
    pace_resilience = float(horse.get("pace_resilience_score", 55.0))
    agari_reliability = float(horse.get("agari_reliability", 50.0))
    race_pace = str(horse.get("race_pace", "medium"))
    horse_ability_score = float(horse.get("horse_ability_score", race_power))
    final_performance_score = float(horse.get("final_performance_score", horse_ability_score))
    late_power = float(horse.get("late_power", final_performance_score))

    if style == STYLE_SEQUENCE[0]:
        score = fade_resistance * 0.40 + stamina * 0.25 + sustain_speed * 0.20 + pace_resilience * 0.15
    elif style == STYLE_SEQUENCE[1]:
        score = fade_resistance * 0.25 + mid_cruise * 0.20 + late_kick_score * 0.25 + stamina * 0.20 + sustain_speed * 0.10
    elif style == STYLE_SEQUENCE[3]:
        score = late_kick_score * 0.55 + agari_reliability * 0.15 + acceleration * 0.15 + race_power * 0.10 + pace_fit * 0.05
    elif style == STYLE_SEQUENCE[4]:
        score = late_kick_score * 0.42 + late_gain_score * 0.22 + agari_reliability * 0.20 + pace_resilience * 0.08 + acceleration * 0.08
        if race_pace == "high":
            score += 8.0
        elif race_pace == "slow":
            score -= 5.0
    else:
        score = late_kick_score * 0.30 + acceleration * 0.20 + stamina * 0.20 + race_power * 0.20 + pace_fit * 0.10
    score = score * 0.68 + final_performance_score * 0.22 + late_power * 0.10
    return _clamp_float(score, 0.0, 100.0)


def _formation_straight_max_gain(style: str) -> float:
    return {
        STYLE_SEQUENCE[0]: 8.0,
        STYLE_SEQUENCE[1]: 12.0,
        STYLE_SEQUENCE[2]: 26.0,
        STYLE_SEQUENCE[3]: 82.0,
        STYLE_SEQUENCE[4]: 92.0,
    }.get(style, 22.0)


def _formation_final_stretch_start(race_config: dict | object, distance: float) -> float:
    if isinstance(race_config, dict):
        straight_length = (
            race_config.get("straight_length")
            or race_config.get("straight_length_m")
            or race_config.get("final_stretch_m")
        )
    else:
        straight_length = (
            getattr(race_config, "straight_length", None)
            or getattr(race_config, "straight_length_m", None)
            or getattr(race_config, "final_stretch_m", None)
        )
    straight_m = _to_float(straight_length, distance * 0.25)
    return _clamp_float(max(0.70, 1.0 - straight_m / max(1.0, distance)), 0.60, 0.90)


def _formation_normalize_gaps(gaps: dict[int, float]) -> dict[int, float]:
    if not gaps:
        return {}
    min_gap = min(gaps.values())
    return {number: max(0.0, gap - min_gap) for number, gap in gaps.items()}


def _formation_style_band_bounds(style: str, progress: float, final_stretch_start: float) -> tuple[float, float]:
    low, high = EARLY_STYLE_GAP_RANGES.get(style, EARLY_STYLE_GAP_RANGES[STYLE_SEQUENCE[2]])
    return low, high


def _formation_clamp_gaps_to_style_bands(
    gaps: dict[int, float],
    prepared_horses: list[dict[str, object]],
    progress: float,
    final_stretch_start: float,
) -> dict[int, float]:
    adjusted: dict[int, float] = {}
    for horse in prepared_horses:
        number = int(horse.get("horse_number", 0))
        style = _formation_fixed_style(horse)
        low, high = _formation_style_band_bounds(style, progress, final_stretch_start)
        adjusted[number] = _clamp_float(gaps.get(number, float(horse.get("style_target_gap", low))), low, high)
    runner_gaps = [
        adjusted[int(horse.get("horse_number", 0))]
        for horse in prepared_horses
        if _formation_fixed_style(horse) == STYLE_RUNNER and int(horse.get("horse_number", 0)) in adjusted
    ]
    if runner_gaps:
        front_runner_gap = min(runner_gaps)
        for horse in prepared_horses:
            if _formation_fixed_style(horse) != STYLE_STALKER:
                continue
            number = int(horse.get("horse_number", 0))
            if number not in adjusted:
                continue
            low, high = _formation_style_band_bounds(STYLE_STALKER, progress, final_stretch_start)
            adjusted[number] = _clamp_float(
                max(adjusted[number], front_runner_gap + MIN_NIGE_SENKO_GAP, low),
                low,
                high,
            )
    return adjusted


def _formation_gap_range_label(style: str, progress: float, final_stretch_start: float) -> str:
    if progress >= final_stretch_start:
        return "final_stretch_free"
    low, high = _formation_style_band_bounds(style, progress, final_stretch_start)
    return f"{low:.1f}-{high:.1f}"


def _formation_enforce_style_order(
    gaps: dict[int, float],
    prepared_horses: list[dict[str, object]],
) -> dict[int, float]:
    ordered = sorted(
        prepared_horses,
        key=lambda horse: (
            STYLE_ORDER.get(_formation_fixed_style(horse), 99),
            gaps.get(int(horse.get("horse_number", 0)), 0.0),
            int(horse.get("horse_number", 0)),
        ),
    )
    adjusted: dict[int, float] = {}
    previous_gap = -0.35
    for horse in ordered:
        number = int(horse.get("horse_number", 0))
        gap = max(gaps.get(number, 0.0), previous_gap + 0.35)
        adjusted[number] = gap
        previous_gap = gap
    return _formation_normalize_gaps(adjusted)


def _formation_fixed_style(horse: dict[str, object]) -> str:
    style = str(
        horse.get("actual_running_style_fixed")
        or horse.get("actual_running_style")
        or STYLE_SEQUENCE[2]
    )
    return style if style in STYLE_ORDER else STYLE_VERSATILE


def _formation_enforce_visible_spread(gaps: dict[int, float]) -> dict[int, float]:
    if len(gaps) <= 1:
        return gaps
    normalized = _formation_normalize_gaps(gaps)
    spread = max(normalized.values()) - min(normalized.values())
    if spread <= 0:
        ordered_numbers = sorted(normalized)
        step = MIN_VISIBLE_MIDFIELD_SPREAD / max(1, len(ordered_numbers) - 1)
        return {number: index * step for index, number in enumerate(ordered_numbers)}
    if spread < MIN_VISIBLE_MIDFIELD_SPREAD:
        scale = MIN_VISIBLE_MIDFIELD_SPREAD / spread
        normalized = {number: gap * scale for number, gap in normalized.items()}
    elif spread > MAX_VISIBLE_MIDFIELD_SPREAD:
        scale = MAX_VISIBLE_MIDFIELD_SPREAD / spread
        normalized = {number: gap * scale for number, gap in normalized.items()}
    return normalized


def _formation_ability_factor(progress: float) -> float:
    if progress < 0.40:
        return 0.0
    if progress < 0.60:
        return 0.3
    if progress < 0.80:
        return 0.7
    return 1.0


@dataclass(frozen=True)
class SectionState:
    horse_name: str
    horse_number: int
    section_index: int
    distance_m: int
    current_speed: float
    elapsed_time: float
    position_m: float
    actual_running_style: str


@dataclass(frozen=True)
class SimulationResult:
    race_config: RaceConfig
    pace: RacePace
    states: list[SectionState]
    ranking: pd.DataFrame
    race_timeline: list[dict[str, object]]

    def states_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([state.__dict__ for state in self.states])

    def timeline_dataframe(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for frame in self.race_timeline:
            frame_time = float(frame.get("time", 0.0))
            frame_progress = float(frame.get("progress", 0.0))
            for horse in frame.get("horses", []):
                if not isinstance(horse, dict):
                    continue
                row = dict(horse)
                row["time"] = frame_time
                row["elapsed_time"] = frame_time
                row["progress"] = frame_progress
                rows.append(row)
        return pd.DataFrame(rows)


class RaceSimulator:
    """Run a 100m-section race simulation."""

    def __init__(self, course_db: CourseDB | None = None) -> None:
        self.course_db = course_db or CourseDB()

    def simulate(
        self,
        config: RaceConfig,
        abilities: list[HorseAbility],
        pace: RacePace,
        section_m: int = 100,
        seed: int | None = None,
        timeline_mode: str = "controlled",
    ) -> SimulationResult:
        rng = random.Random(seed)
        sections = int(np.ceil(config.distance / section_m))
        states: list[SectionState] = []
        final_rows: list[dict[str, float | int | str]] = []
        pace_dict = pace.to_dict()

        for lane_index, ability in enumerate(abilities):
            adjusted_profile = adjust_style_profile(
                base_profile=ability.base_style_profile,
                race_config=config,
                pace_prediction=pace_dict,
                frame=ability.frame,
                horse_number=ability.horse_number,
            )
            actual_style = self._sample_actual_style(
                primary_style=ability.primary_running_style,
                adjusted_profile=adjusted_profile,
                rng=rng,
            )
            track_bias_fit_score = get_track_bias_fit_score(config, actual_style, ability.frame)
            if pace.pace == "slow":
                base_pace_fit = (ability.base_style_profile.get(STYLE_RUNNER, 0.0) + ability.base_style_profile.get(STYLE_STALKER, 0.0)) * 100.0
            elif pace.pace == "high":
                base_pace_fit = (ability.base_style_profile.get(STYLE_CLOSER, 0.0) + ability.base_style_profile.get(STYLE_DEEP_CLOSER, 0.0)) * 100.0
            else:
                front_probability = ability.base_style_profile.get(STYLE_RUNNER, 0.0) + ability.base_style_profile.get(STYLE_STALKER, 0.0)
                base_pace_fit = (1.0 - abs(front_probability - 0.5)) * 100.0
            pace_fit_score = _clamp_float(base_pace_fit * 0.85 + track_bias_fit_score * 0.15, 0.0, 100.0)
            horse_jitter = min(1.04, max(0.96, rng.normalvariate(1.0, 0.014)))
            acceleration_shift_m = rng.uniform(-100.0, 120.0)
            elapsed = 0.0
            position = 0.0
            fatigue = 1.0

            for section_index in range(sections):
                section_start = section_index * section_m
                remaining = max(0, config.distance - section_start)
                actual_section_m = min(section_m, remaining)
                progress = section_start / config.distance
                current_speed = self._current_speed(
                    config=config,
                    ability=ability,
                    pace=pace,
                    progress=progress,
                    fatigue=fatigue,
                    actual_style=actual_style,
                    acceleration_shift_m=acceleration_shift_m,
                )
                section_jitter = min(1.025, max(0.975, rng.normalvariate(1.0, 0.006)))
                current_speed *= horse_jitter * section_jitter
                section_time = actual_section_m / max(1.0, current_speed)
                elapsed += section_time
                position = min(config.distance, position + actual_section_m)
                states.append(
                    SectionState(
                        horse_name=ability.horse_name,
                        horse_number=ability.horse_number,
                        section_index=section_index,
                        distance_m=int(position),
                        current_speed=current_speed,
                        elapsed_time=elapsed,
                        position_m=position,
                        actual_running_style=actual_style,
                    )
                )
                fatigue = self._next_fatigue(ability, progress, pace, fatigue, actual_style)

            final_rows.append(
                {
                    "horse_name": ability.horse_name,
                    "horse_number": ability.horse_number,
                    "frame": ability.frame,
                    "primary_running_style": ability.primary_running_style,
                    "actual_running_style": actual_style,
                    "actual_running_style_fixed": actual_style,
                    "running_style": actual_style,
                    "adjusted_style_profile": dict(adjusted_profile),
                    "finish_time": elapsed,
                    "lane_index": lane_index,
                    f"adjusted_{STYLE_RUNNER}": adjusted_profile.get(STYLE_RUNNER, 0.0),
                    f"adjusted_{STYLE_STALKER}": adjusted_profile.get(STYLE_STALKER, 0.0),
                    f"adjusted_{STYLE_CLOSER}": adjusted_profile.get(STYLE_CLOSER, 0.0),
                    f"adjusted_{STYLE_DEEP_CLOSER}": adjusted_profile.get(STYLE_DEEP_CLOSER, 0.0),
                    "early_speed": ability.early_speed,
                    "stamina": ability.stamina,
                    "acceleration": ability.acceleration,
                    "consistency": ability.consistency,
                    "mud_aptitude": ability.mud_aptitude,
                    "mud_source": ability.mud_source,
                    "race_strength_score": ability.race_strength_score,
                    "race_strength_adjusted_score": ability.race_strength_adjusted_score,
                    "elo_rating": ability.elo_rating,
                    "normalized_elo_score": ability.normalized_elo_score,
                    "relative_agari_score": ability.relative_agari_score,
                    "course_fit_score": ability.course_fit_score,
                    "jockey": ability.jockey,
                    "jockey_score": ability.jockey_score,
                    "track_bias_fit_score": track_bias_fit_score,
                    "pace_fit_score": pace_fit_score,
                    "avg_race_score": ability.avg_race_score,
                    "horse_ability_score": ability.horse_ability_score,
                    "popularity_score": ability.popularity_score,
                    "race_level_score": ability.race_level_score,
                    "finish_score": ability.finish_score,
                    "margin_score": ability.margin_score,
                    "time_score": ability.time_score,
                    "race_power": ability.race_power,
                    "early_aggressiveness": ability.early_aggressiveness,
                    "mid_positioning": ability.mid_positioning,
                    "late_kick_timing": ability.late_kick_timing,
                    "sustain_speed": ability.sustain_speed,
                    "time_reliability": ability.time_reliability,
                    "recent_time_score": ability.recent_time_score,
                    "late_kick_score": ability.late_kick_score,
                    "avg_last3f": ability.avg_last3f,
                    "best_last3f": ability.best_last3f,
                    "last3f_consistency": ability.last3f_consistency,
                    "late_gain_score": ability.late_gain_score,
                    "early_push_score": ability.early_push_score,
                    "mid_cruise_score": ability.mid_cruise_score,
                    "fade_resistance_score": ability.fade_resistance_score,
                    "sustain_speed_score": ability.sustain_speed_score,
                    "pace_resilience_score": ability.pace_resilience_score,
                    "agari_reliability": ability.agari_reliability,
                    "carried_weight": ability.carried_weight,
                    "weight_penalty": ability.weight_penalty,
                }
            )

        ranking = pd.DataFrame(final_rows).sort_values("finish_time").reset_index(drop=True)
        ranking.insert(0, "rank", range(1, len(ranking) + 1))
        if timeline_mode == "legacy":
            race_timeline = self._build_race_timeline(config=config, states=states, ranking=ranking, pace=pace)
        else:
            race_timeline = generate_controlled_race_timeline(
                horses=ranking.to_dict("records"),
                race_config={**config.to_dict(), "pace": pace.pace},
                n_frames=300,
                seed=seed,
            )
        return SimulationResult(
            race_config=config,
            pace=pace,
            states=states,
            ranking=ranking,
            race_timeline=race_timeline,
        )

    def _sample_actual_style(
        self,
        primary_style: str,
        adjusted_profile: dict[str, float],
        rng: random.Random,
    ) -> str:
        if primary_style != STYLE_VERSATILE:
            return primary_style if primary_style in STYLE_KEYS else STYLE_VERSATILE
        return rng.choices(
            STYLE_KEYS,
            weights=[adjusted_profile[style] for style in STYLE_KEYS],
            k=1,
        )[0]

    def _current_speed(
        self,
        config: RaceConfig,
        ability: HorseAbility,
        pace: RacePace,
        progress: float,
        fatigue: float,
        actual_style: str,
        acceleration_shift_m: float = 0.0,
    ) -> float:
        ability_factor = self._ability_factor(progress)
        base_speed = 15.2
        base_speed += ((ability.early_speed - 50.0) / 35.0 + (ability.consistency - 50.0) / 110.0) * ability_factor
        diff_start = max(0.0, 1.0 - max(350.0, 600.0 + acceleration_shift_m) / config.distance)
        deep_start = max(0.0, 1.0 - max(250.0, 400.0 + acceleration_shift_m) / config.distance)
        common_start = max(0.0, 1.0 - max(350.0, 600.0 + acceleration_shift_m) / config.distance)
        if actual_style == STYLE_CLOSER and progress >= diff_start:
            base_speed += (ability.acceleration - 45.0) / 22.0 * ability_factor
        elif actual_style == "霑ｽ霎ｼ" and progress >= deep_start:
            base_speed += (ability.acceleration - 45.0) / 18.0 * ability_factor
        elif progress >= common_start:
            base_speed += (ability.acceleration - 45.0) / 32.0 * ability_factor
        if config.distance >= 2000:
            base_speed += (ability.stamina - 50.0) / 90.0 * ability_factor

        pace_bonus = self._pace_bonus(ability, pace, progress, actual_style)
        style_bonus = self._style_bonus(ability, pace, progress, actual_style)
        target_bonus = self._target_position_bonus(actual_style, progress)
        raw_track_bonus = self.course_db.track_bonus(
            config=config,
            mud_aptitude=ability.mud_aptitude,
            stamina=ability.stamina,
            acceleration=ability.acceleration,
        )
        track_bonus = 1.0 + (raw_track_bonus - 1.0) * max(0.25, ability_factor)
        stamina_bonus = 1.0 + ((ability.stamina - 50.0) / 600.0) * ability_factor
        frame_bonus = 1.0 + (self.course_db.frame_bonus(ability.frame, config) - 1.0) * 0.5
        course_bias_bonus = 1.0 + (self._course_bias_bonus(config, actual_style) - 1.0) * max(0.25, ability_factor)
        power_bonus = 1.0 + self._power_effect(ability.race_power) * 0.03 * ability_factor
        return (
            base_speed
            * pace_bonus
            * style_bonus
            * target_bonus
            * track_bonus
            * stamina_bonus
            * frame_bonus
            * course_bias_bonus
            * power_bonus
            * fatigue
        )

    def _course_bias_bonus(self, config: RaceConfig, actual_style: str) -> float:
        bias = get_course_bias(config)
        bias_scale = 0.20
        bonus = 1.0
        if actual_style in {STYLE_RUNNER, STYLE_STALKER}:
            bonus += float(bias.get("front_bias", 0.0)) * bias_scale
        elif actual_style in {STYLE_CLOSER, STYLE_DEEP_CLOSER}:
            bonus += float(bias.get("closer_bias", 0.0)) * bias_scale
        return max(0.975, min(1.025, bonus))

    def _pace_bonus(self, ability: HorseAbility, pace: RacePace, progress: float, actual_style: str) -> float:
        if pace.pace == "high":
            if progress < 0.55:
                return 1.04 if actual_style in {STYLE_RUNNER, STYLE_STALKER} else 0.985
            return pace.closer_advantage if actual_style in {STYLE_CLOSER, STYLE_DEEP_CLOSER} else 0.965
        if pace.pace == "slow":
            if progress < 0.60:
                return 1.04 if actual_style in {STYLE_RUNNER, STYLE_STALKER} else 0.975
            if actual_style == STYLE_DEEP_CLOSER:
                return 0.965
            return 1.035 if ability.acceleration >= 70.0 else 1.0
        return 1.0 if progress < 0.70 else 1.0 + (ability.acceleration - 50.0) / 900.0

    def _style_bonus(self, ability: HorseAbility, pace: RacePace, progress: float, actual_style: str) -> float:
        if actual_style == STYLE_RUNNER:
            if progress < 0.35:
                return 1.085
            if progress > 0.78 and pace.pace == "high":
                return 0.935
            if progress > 0.78 and pace.pace == "slow":
                return 1.04
            return 1.0
        if actual_style == STYLE_STALKER:
            if progress < 0.45:
                return 1.04
            if progress > 0.72:
                return 0.99 + (ability.stamina - 50.0) / 1800.0
            return 1.012
        if actual_style == STYLE_CLOSER:
            if progress < 0.45:
                return 0.975
            if progress > 0.70:
                return 1.07 if pace.pace == "high" else 1.025
            return 1.0
        if actual_style == STYLE_DEEP_CLOSER:
            if progress < 0.55:
                return 0.955
            if progress > 0.82:
                return 1.115 if pace.pace == "high" else 1.03
            if pace.pace == "slow" and progress > 0.65:
                return 0.965
            return 1.0
        return 1.0

    def _target_position_bonus(self, actual_style: str, progress: float) -> float:
        """Approximate target position behavior through phase speed shaping."""
        if actual_style == STYLE_RUNNER:
            return 1.035 if progress < 0.40 else 0.995
        if actual_style == STYLE_STALKER:
            return 1.018 if progress < 0.50 else 1.0
        if actual_style == STYLE_CLOSER:
            return 0.985 if progress < 0.55 else 1.035
        if actual_style == STYLE_DEEP_CLOSER:
            return 0.970 if progress < 0.65 else 1.060
        return 1.0

    def _next_fatigue(
        self,
        ability: HorseAbility,
        progress: float,
        pace: RacePace,
        fatigue: float,
        actual_style: str,
    ) -> float:
        if progress < 0.62:
            return fatigue
        burden = 0.004 + max(0.0, 55.0 - ability.stamina) / 4500.0
        burden += max(0.0, ability.weight_penalty) / 5000.0
        if pace.pace == "high" and actual_style == STYLE_RUNNER:
            burden += 0.010
        elif pace.pace == "high" and actual_style == STYLE_STALKER:
            burden += 0.005
        if actual_style in {STYLE_CLOSER, STYLE_DEEP_CLOSER} and pace.pace == "high":
            burden -= 0.002
        return max(0.86, fatigue - burden)

    def _build_race_timeline(
        self,
        config: RaceConfig,
        states: list[SectionState],
        ranking: pd.DataFrame,
        pace: RacePace,
    ) -> list[dict[str, object]]:
        """Build synchronized horse positions used by all renderers."""
        states_df = pd.DataFrame([state.__dict__ for state in states])
        if states_df.empty or ranking.empty:
            return []

        distance = float(config.distance)
        max_time = float(ranking["finish_time"].max())
        frame_count = max(80, min(260, int(distance / 22.0) + 1))
        time_points = np.linspace(0.0, max_time, frame_count)
        style_slots = self._style_slots(ranking)
        style_group_sizes = self._style_group_sizes(ranking)
        mean_power, std_power = self._race_power_stats(ranking)

        race_timeline: list[dict[str, object]] = []
        ranking_rows = ranking.to_dict("records")
        previous_gaps: dict[int, float] = {}
        for frame_index, time_value in enumerate(time_points):
            frame_horses: list[dict[str, object]] = []
            race_progress = frame_index / max(1, frame_count - 1)
            leader_position = min(distance, race_progress * distance)
            for row in ranking_rows:
                horse_number = int(row["horse_number"])
                actual_style = str(row.get("actual_running_style_fixed", row["actual_running_style"]))
                race_power = self._race_power(row)
                normalized_power = self._normalized_race_power(race_power, mean_power, std_power)
                power_effect = self._power_effect(race_power)
                slot = style_slots.get(horse_number, 0)
                group_size = style_group_sizes.get(actual_style, 1)
                ability_factor = self._ability_factor(race_progress)
                target_rank_ratio = self._target_rank_ratio_from_band(
                    actual_style=actual_style,
                    slot=slot,
                    group_size=group_size,
                    pace=pace,
                )
                gap_from_leader = self._gap_from_leader(
                    row=row,
                    actual_style=actual_style,
                    slot=slot,
                    group_size=group_size,
                    progress=race_progress,
                    distance=distance,
                    pace=pace,
                    target_rank_ratio=target_rank_ratio,
                    normalized_power=normalized_power,
                )
                if race_progress >= 0.60 and horse_number in previous_gaps:
                    previous_gap = previous_gaps[horse_number]
                    gap_delta = self._clamp(gap_from_leader - previous_gap, -MAX_GAP_DELTA_PER_STEP, MAX_GAP_DELTA_PER_STEP)
                    gap_from_leader = previous_gap + gap_delta
                target_position = max(0.0, min(distance, leader_position - self._early_gap_from_band(actual_style, target_rank_ratio)))
                position = max(0.0, min(distance, leader_position - gap_from_leader))
                lane = self._timeline_lane(
                    frame=int(row["frame"]),
                    horse_number=horse_number,
                    actual_style=actual_style,
                    progress=race_progress,
                    distance=distance,
                )
                frame_horses.append(
                    {
                        "horse_number": horse_number,
                        "horse_name": str(row["horse_name"]),
                        "frame": int(row["frame"]),
                        "actual_running_style": actual_style,
                        "actual_running_style_fixed": actual_style,
                        "position_m": round(position, 3),
                        "rank": int(row["rank"]),
                        "lane": round(lane, 3),
                        "gap_from_leader": round(gap_from_leader, 3),
                        "race_power": round(race_power, 3),
                        "normalized_race_power": round(normalized_power, 3),
                        "power_effect": round(power_effect, 4),
                        "ability_factor": round(ability_factor, 3),
                        "target_rank_ratio": round(target_rank_ratio, 3),
                        "target_position_m": round(target_position, 3),
                        "_finish_time": float(row["finish_time"]),
                        "_final_rank": int(row["rank"]),
                    }
                )

            if race_progress < 0.60:
                frame_horses = self._cap_midrace_gaps(frame_horses, leader_position, distance)
            previous_gaps = {
                int(horse["horse_number"]): float(horse["gap_from_leader"])
                for horse in frame_horses
            }
            ranked_horses = sorted(
                frame_horses,
                key=lambda horse: (-float(horse["position_m"]), int(horse["_final_rank"])),
            )
            for rank, horse in enumerate(ranked_horses, start=1):
                horse["rank"] = rank
                horse.pop("_finish_time", None)
                horse.pop("_final_rank", None)
            race_timeline.append(
                {
                    "time": round(float(time_value), 3),
                    "progress": round(race_progress, 4),
                    "horses": ranked_horses,
                }
            )
        return race_timeline

    def _enforce_early_style_bands(
        self,
        horses: list[dict[str, object]],
        distance: float,
        frame_progress: float,
    ) -> list[dict[str, object]]:
        if frame_progress >= 0.60 or len(horses) <= 1:
            return horses

        field_size = len(horses)
        leader_position = max(float(horse["position_m"]) for horse in horses)
        gap = min(24.0, max(8.0, distance * 0.0065))
        by_style = {style: [] for style in STYLE_ORDER}
        for horse in horses:
            style = str(horse.get("actual_running_style_fixed", horse.get("actual_running_style", STYLE_CLOSER)))
            if style not in by_style:
                style = STYLE_VERSATILE
            by_style[style].append(horse)

        adjusted: list[dict[str, object]] = []
        used_slots: set[int] = set()
        for style in STYLE_ORDER:
            group = sorted(
                by_style[style],
                key=lambda item: (-float(item["position_m"]), int(item["horse_number"])),
            )
            if not group:
                continue
            band_start, band_end = STYLE_RANK_BANDS[style]
            for index, horse in enumerate(group):
                ratio = band_start + (band_end - band_start) * (index + 1) / (len(group) + 1)
                slot = int(round(ratio * max(1, field_size - 1)))
                while slot in used_slots:
                    slot += 1
                used_slots.add(slot)
                controlled_position = max(0.0, leader_position - slot * gap)
                updated = dict(horse)
                updated["position_m"] = round(min(distance * 0.997, controlled_position), 3)
                adjusted.append(updated)
        return adjusted

    def _position_at_time(self, table: pd.DataFrame, time_value: float, distance: float) -> float:
        times = table["elapsed_time"].to_numpy(dtype=float)
        positions = table["position_m"].to_numpy(dtype=float)
        if len(times) == 0:
            return 0.0
        if time_value <= times[0]:
            return float(positions[0] * time_value / max(times[0], 0.1))
        if time_value >= times[-1]:
            return float(distance)
        return float(np.interp(time_value, times, positions))

    def _style_slots(self, ranking: pd.DataFrame) -> dict[int, int]:
        counts: dict[str, int] = {}
        slots: dict[int, int] = {}
        for _, row in ranking.sort_values("horse_number").iterrows():
            style = str(row.get("actual_running_style_fixed", row["actual_running_style"]))
            slots[int(row["horse_number"])] = counts.get(style, 0)
            counts[style] = counts.get(style, 0) + 1
        return slots

    def _style_group_sizes(self, ranking: pd.DataFrame) -> dict[str, int]:
        sizes: dict[str, int] = {}
        for _, row in ranking.iterrows():
            style = str(row.get("actual_running_style_fixed", row["actual_running_style"]))
            sizes[style] = sizes.get(style, 0) + 1
        return sizes

    def _target_rank_ratio_from_band(
        self,
        actual_style: str,
        slot: int,
        group_size: int,
        pace: RacePace,
    ) -> float:
        style = actual_style if actual_style in STYLE_RANK_BANDS else STYLE_VERSATILE
        band_start, band_end = STYLE_RANK_BANDS[style]
        if style == STYLE_VERSATILE:
            if pace.pace == "slow":
                band_start, band_end = 0.24, 0.50
            elif pace.pace == "high":
                band_start, band_end = 0.40, 0.64
        within_group = (slot + 1) / max(2, group_size + 1)
        return min(1.0, max(0.0, band_start + (band_end - band_start) * within_group))

    def _early_gap_from_band(self, actual_style: str, target_rank_ratio: float) -> float:
        style = actual_style if actual_style in STYLE_GAP_BANDS else STYLE_VERSATILE
        band_min, band_max = STYLE_GAP_BANDS[style]
        gap = target_rank_ratio * MAX_GAP_UNTIL_MID
        return min(band_max, max(band_min, gap))

    def _gap_from_leader(
        self,
        row: dict[str, object],
        actual_style: str,
        slot: int,
        group_size: int,
        progress: float,
        distance: float,
        pace: RacePace,
        target_rank_ratio: float,
        normalized_power: float,
    ) -> float:
        base_gap = self._early_gap_from_band(actual_style, target_rank_ratio)
        ability_factor = self._ability_factor(progress)
        ability_strength = self._late_ability_strength(row, normalized_power)
        style = actual_style if actual_style in STYLE_GAP_BANDS else STYLE_VERSATILE

        if progress < 0.60:
            band_min, band_max = STYLE_GAP_BANDS[style]
            controlled_gap = base_gap - ability_strength * ability_factor * 0.35
            return min(MAX_GAP_UNTIL_MID, max(band_min, min(band_max, controlled_gap)))

        late_stage = min(1.0, max(0.0, (progress - 0.60) / 0.40))
        gap = base_gap
        positive = max(0.0, ability_strength)
        negative = max(0.0, -ability_strength)

        diff_start = max(0.60, 1.0 - 600.0 / max(1.0, distance))
        deep_start = max(0.72, 1.0 - 400.0 / max(1.0, distance))
        if style == STYLE_RUNNER:
            gap -= positive * 1.4 * late_stage
            if pace.pace == "high":
                gap += (2.8 + negative * 2.0) * late_stage
            elif pace.pace == "slow":
                gap -= 1.0 * late_stage
        elif style == STYLE_STALKER:
            gap -= positive * 2.6 * late_stage
            stamina = float(row.get("stamina", 50.0) or 50.0)
            if stamina < 55.0 and progress > 0.74:
                gap += (55.0 - stamina) / 12.0 * late_stage
        elif style == STYLE_VERSATILE:
            if pace.pace == "slow":
                gap -= (1.8 + positive * 2.0) * late_stage
            elif pace.pace == "high":
                gap -= positive * 3.0 * late_stage
            else:
                gap -= positive * 2.4 * late_stage
        elif style == STYLE_CLOSER:
            if progress >= diff_start:
                stage = min(1.0, (progress - diff_start) / max(0.01, 1.0 - diff_start))
                pace_boost = 1.20 if pace.pace == "high" else 0.75 if pace.pace == "slow" else 1.0
                gap -= (3.8 + positive * 7.0) * stage * pace_boost
        elif style == STYLE_DEEP_CLOSER:
            if progress >= deep_start:
                stage = min(1.0, (progress - deep_start) / max(0.01, 1.0 - deep_start))
                pace_boost = 1.25 if pace.pace == "high" else 0.62 if pace.pace == "slow" else 0.95
                gap -= (4.6 + positive * 8.2) * stage * pace_boost

        gap -= ability_strength * 1.5 * late_stage
        if progress >= 0.92:
            finish_blend = min(1.0, (progress - 0.92) / 0.08)
            final_gap = max(0.0, (int(row.get("rank", 1) or 1) - 1) * 2.2)
            gap = gap * (1.0 - finish_blend) + final_gap * finish_blend
        return max(0.0, min(36.0, gap))

    def _ability_factor(self, progress: float) -> float:
        if progress < 0.40:
            return 0.0
        if progress < 0.60:
            return 0.3
        if progress < 0.80:
            return 0.7
        return 1.0

    def _late_ability_strength(self, row: dict[str, object], normalized_power: float) -> float:
        early_speed = float(row.get("early_speed", 50.0) or 50.0)
        stamina = float(row.get("stamina", 50.0) or 50.0)
        acceleration = float(row.get("acceleration", 50.0) or 50.0)
        consistency = float(row.get("consistency", 50.0) or 50.0)
        race_power = self._race_power(row)
        blended_power = (
            early_speed * 0.10
            + stamina * 0.25
            + acceleration * 0.35
            + consistency * 0.15
            + race_power * 0.15
        )
        raw_strength = (blended_power - 65.0) / 45.0
        compressed_power = normalized_power / 1.5 * 0.35
        return self._clamp(raw_strength * 0.65 + compressed_power, -1.0, 1.0)

    def _race_power(self, row: dict[str, object]) -> float:
        return self._clamp(float(row.get("race_power", row.get("avg_race_score", 70.0)) or 70.0), 40.0, 95.0)

    def _power_effect(self, race_power: float) -> float:
        return self._clamp((race_power - 70.0) / 100.0, -0.30, 0.25)

    def _race_power_stats(self, ranking: pd.DataFrame) -> tuple[float, float]:
        values = [self._race_power(row) for row in ranking.to_dict("records")]
        if not values:
            return 70.0, 1.0
        mean_power = float(np.mean(values))
        std_power = float(np.std(values))
        return mean_power, max(1.0, std_power)

    def _normalized_race_power(self, race_power: float, mean_power: float, std_power: float) -> float:
        return self._clamp((race_power - mean_power) / max(1.0, std_power), -1.5, 1.5)

    def _cap_midrace_gaps(
        self,
        horses: list[dict[str, object]],
        leader_position: float,
        distance: float,
    ) -> list[dict[str, object]]:
        if len(horses) <= 1:
            return horses

        sorted_horses = sorted(
            horses,
            key=lambda horse: (float(horse.get("gap_from_leader", 0.0)), int(horse.get("_final_rank", 99))),
        )
        leader_gap = max(0.0, float(sorted_horses[0].get("gap_from_leader", 0.0)))
        previous_gap = leader_gap
        for index, horse in enumerate(sorted_horses):
            gap = max(leader_gap, min(MAX_GAP_UNTIL_MID, float(horse.get("gap_from_leader", 0.0))))
            if index == 1:
                gap = min(gap, leader_gap + MAX_LEADER_GAP_UNTIL_MID)
            elif index > 1:
                gap = max(gap, previous_gap + 0.12)
            gap = min(MAX_GAP_UNTIL_MID, gap)
            horse["gap_from_leader"] = round(gap, 3)
            horse["position_m"] = round(max(0.0, min(distance, leader_position - gap)), 3)
            previous_gap = gap
        return horses

    def _clamp(self, value: float, minimum: float, maximum: float) -> float:
        return min(maximum, max(minimum, value))

    def _timeline_position(
        self,
        raw_position: float,
        progress: float,
        distance: float,
        actual_style: str,
        pace: RacePace,
        slot: int,
        is_finished: bool,
    ) -> float:
        if is_finished:
            return distance
        offset = self._style_position_offset(actual_style, progress, distance, pace, slot)
        return max(0.0, min(distance * 0.997, raw_position + offset))

    def _style_position_offset(
        self,
        actual_style: str,
        progress: float,
        distance: float,
        pace: RacePace,
        slot: int,
    ) -> float:
        spread = min(130.0, max(50.0, distance * 0.055))
        target_ratio = self._target_rank_ratio(actual_style, progress, pace, slot)
        early_offset = (0.50 - target_ratio) * spread
        start_ramp = min(1.0, max(0.0, progress / 0.08))
        finish_fade = max(0.0, min(1.0, (0.985 - progress) / 0.18))
        offset = early_offset * start_ramp * finish_fade

        diff_start = max(0.0, 1.0 - 600.0 / max(1.0, distance))
        deep_start = max(0.0, 1.0 - 400.0 / max(1.0, distance))
        if actual_style == STYLE_CLOSER and progress >= diff_start:
            late_stage = min(1.0, (progress - diff_start) / max(0.01, 1.0 - diff_start))
            pace_boost = 1.15 if pace.pace == "high" else 0.78 if pace.pace == "slow" else 1.0
            offset += spread * 0.30 * late_stage * pace_boost * finish_fade
        elif actual_style == STYLE_DEEP_CLOSER and progress >= deep_start:
            late_stage = min(1.0, (progress - deep_start) / max(0.01, 1.0 - deep_start))
            pace_boost = 1.22 if pace.pace == "high" else 0.60 if pace.pace == "slow" else 0.95
            offset += spread * 0.48 * late_stage * pace_boost * finish_fade
        elif actual_style == STYLE_RUNNER and progress >= 0.74:
            late_stage = min(1.0, (progress - 0.74) / 0.26)
            if pace.pace == "high":
                offset -= spread * 0.16 * late_stage
            elif pace.pace == "slow":
                offset += spread * 0.08 * late_stage * finish_fade
        return offset

    def _target_rank_ratio(self, actual_style: str, progress: float, pace: RacePace, slot: int) -> float:
        if actual_style == STYLE_RUNNER:
            return min(0.15, 0.055 + slot * 0.045)
        if actual_style == STYLE_STALKER:
            return min(0.35, 0.20 + slot * 0.045)
        if actual_style == STYLE_CLOSER:
            late_start = 0.72
            if progress <= late_start:
                return min(0.65, 0.52 + slot * 0.035)
            late_stage = min(1.0, (progress - late_start) / max(0.01, 1.0 - late_start))
            return 0.52 - 0.17 * late_stage
        if actual_style == STYLE_DEEP_CLOSER:
            late_start = 0.80
            if progress <= late_start:
                return min(0.90, 0.76 + slot * 0.035)
            late_stage = min(1.0, (progress - late_start) / max(0.01, 1.0 - late_start))
            return 0.76 - 0.27 * late_stage
        if pace.pace == "slow":
            return 0.30
        if pace.pace == "high":
            return 0.52
        return 0.42

    def _timeline_lane(
        self,
        frame: int,
        horse_number: int,
        actual_style: str,
        progress: float,
        distance: float,
    ) -> float:
        lane = frame - 1 + (horse_number % 2) * 0.24
        late_start = 1.0 - (600.0 if actual_style == STYLE_CLOSER else 400.0) / max(1.0, distance)
        if actual_style in {STYLE_CLOSER, STYLE_DEEP_CLOSER} and progress >= late_start:
            lane += 0.45 if actual_style == STYLE_CLOSER else 0.65
        elif actual_style in {STYLE_RUNNER, STYLE_STALKER} and progress < 0.65:
            lane -= 0.12
        return min(7.0, max(0.0, lane))
