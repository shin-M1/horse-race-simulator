from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

import pandas as pd

from course_db import estimate_course_fit_score, get_track_bias_fit_score
from errors import RaceDataFetchError
from race_config import HorseEntry, calculate_weight_penalty
from utils import clamp, normalize_track_condition, safe_mean, safe_std


STYLE_KEYS = ["逃げ", "先行", "差し", "追込"]
STYLE_WEIGHTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
ABILITY_RECENCY_WEIGHTS = [1.00, 0.85, 0.70, 0.55, 0.40]
RACE_LEVEL_WEIGHT = {
    "G1": 1.40,
    "G2": 1.25,
    "G3": 1.15,
    "L": 1.08,
    "OP": 1.00,
    "3勝": 0.88,
    "2勝": 0.78,
    "1勝": 0.68,
    "未勝利": 0.55,
    "新馬": 0.50,
}
RACE_LEVEL_SCORE = {
    "G1": 95.0,
    "G2": 88.0,
    "G3": 82.0,
    "L": 75.0,
    "OP": 70.0,
    "3勝": 62.0,
    "2勝": 55.0,
    "1勝": 48.0,
    "未勝利": 40.0,
    "新馬": 38.0,
}


def parse_passing_order(passing_order: str | list[int] | tuple[int, ...] | None) -> list[int]:
    """Extract passing-order positions from common netkeiba/JRA text formats."""
    if passing_order is None:
        return []
    if isinstance(passing_order, (list, tuple)):
        parsed: list[int] = []
        for value in passing_order:
            try:
                number = int(value)
            except (TypeError, ValueError):
                continue
            if number > 0:
                parsed.append(number)
        return parsed
    text = str(passing_order).strip()
    if not text:
        return []
    return [int(value) for value in re.findall(r"\d+", text)]


def parse_race_time(value: Any) -> float | None:
    """Convert race time values such as '1:58.3' or '2分12秒5' to seconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        seconds = float(value)
        return seconds if seconds > 0 else None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    text = text.translate(str.maketrans("０１２３４５６７８９：．", "0123456789:."))
    japanese_tenth_match = re.search(r"(\d+)\s*分\s*(\d+)\s*秒\s*(\d+)", text)
    if japanese_tenth_match:
        return int(japanese_tenth_match.group(1)) * 60.0 + float(
            f"{japanese_tenth_match.group(2)}.{japanese_tenth_match.group(3)}"
        )
    minute_match = re.search(r"(\d+)\s*分\s*(\d+(?:\.\d+)?)\s*秒?", text)
    if minute_match:
        return int(minute_match.group(1)) * 60.0 + float(minute_match.group(2))
    compact_match = re.search(r"(\d+)\s*:\s*(\d+(?:\.\d+)?)", text)
    if compact_match:
        return int(compact_match.group(1)) * 60.0 + float(compact_match.group(2))
    dotted_match = re.search(r"(\d+)\.(\d{2})\.(\d+)", text)
    if dotted_match:
        return int(dotted_match.group(1)) * 60.0 + float(f"{dotted_match.group(2)}.{dotted_match.group(3)}")
    seconds_match = re.search(r"\d+(?:\.\d+)?", text)
    if seconds_match:
        seconds = float(seconds_match.group(0))
        return seconds if seconds > 0 else None
    return None


def normalize_style_profile(profile: dict[str, float]) -> dict[str, float]:
    cleaned = {style: max(0.0, float(profile.get(style, 0.0))) for style in STYLE_KEYS}
    total = sum(cleaned.values())
    if total <= 0:
        return {style: 1.0 / len(STYLE_KEYS) for style in STYLE_KEYS}
    return {style: cleaned[style] / total for style in STYLE_KEYS}


def primary_style_from_profile(profile: dict[str, float]) -> str:
    normalized = normalize_style_profile(profile)
    ranked = sorted(normalized.items(), key=lambda item: item[1], reverse=True)
    if ranked[0][1] < 0.40 or ranked[0][1] - ranked[1][1] < 0.10:
        return "自在"
    return ranked[0][0]


def create_base_style_profile(races: list[dict[str, Any] | "RaceResult"]) -> dict[str, float]:
    """Create a weighted soft running-style profile from up to ten past races."""
    scores = {style: 0.0 for style in STYLE_KEYS}
    used_count = 0
    for index, race in enumerate(races[:10]):
        snapshot = race_style_snapshot(race)
        if snapshot is None:
            continue
        weight = STYLE_WEIGHTS[index]
        race_profile = race_style_contribution(snapshot)
        for style in STYLE_KEYS:
            scores[style] += race_profile[style] * weight
        used_count += 1
    if used_count == 0:
        return normalize_style_profile(scores)
    return normalize_style_profile(scores)


def estimate_mud_aptitude(
    recent_races: list[dict[str, Any] | "RaceResult"],
    pedigree_info: dict[str, Any] | None = None,
) -> float:
    """Estimate wet-track aptitude, preferring actual race history over pedigree."""
    score, _ = estimate_mud_aptitude_with_source(recent_races, pedigree_info)
    return score


def estimate_mud_aptitude_with_source(
    recent_races: list[dict[str, Any] | "RaceResult"],
    pedigree_info: dict[str, Any] | None = None,
) -> tuple[float, str]:
    """Return wet-track aptitude and the evidence source used for it."""
    wet_scores: list[float] = []
    wet_weights: list[float] = []
    for index, race in enumerate(list(recent_races or [])[:5]):
        condition = _race_track_condition(race)
        if normalize_track_condition(condition) not in {"稍重", "重", "不良"}:
            continue
        snapshot = race_style_snapshot(race)
        if isinstance(race, RaceResult):
            finish_value = finish_score(race)
            margin_value = margin_score(race)
            time_value = time_score(race)
            last3f_value = last3f_score(race)
        else:
            finish_value = _finish_score_from_snapshot(snapshot)
            margin_value = _race_margin_score(race)
            _, _, time_value = _race_time_features(race)
            _, last3f_value = _race_last3f_features(race)
        wet_scores.append(
            clamp(
                finish_value * 0.35
                + margin_value * 0.25
                + time_value * 0.20
                + last3f_value * 0.20
            )
        )
        wet_weights.append(ABILITY_RECENCY_WEIGHTS[index])

    if wet_scores:
        return clamp(_weighted_average(wet_scores, wet_weights, default=50.0)), "race_history"

    pedigree_score = _pedigree_mud_score(pedigree_info)
    if pedigree_score is not None:
        return clamp(pedigree_score), "pedigree"
    return 50.0, "neutral"


@dataclass(frozen=True)
class RaceResult:
    race_name: str
    distance: int
    surface: str
    track_condition: str
    finish_position: int
    margin: float
    passing_order: list[int] | str | None
    final_3f: float
    race_time_seconds: float | None = None
    winner_time_diff: float | None = None
    field_size: int | None = 18
    race_class: str = ""
    popularity: str = ""
    date: str = ""
    course: str = ""
    race_id: str = ""
    raw: dict[str, Any] | None = None


def race_result_to_recent_dict(result: RaceResult) -> dict[str, Any]:
    """Convert a normalized RaceResult back into a display-friendly recent-race row."""
    row = dict(result.raw or {})
    passing_order = result.passing_order
    if isinstance(passing_order, (list, tuple)):
        passing_text = "-".join(str(position) for position in passing_order)
    else:
        passing_text = "" if passing_order is None else str(passing_order)
    avg_speed = (
        result.distance / result.race_time_seconds
        if result.race_time_seconds is not None and result.race_time_seconds > 0 and result.distance > 0
        else None
    )
    recent_time_score = time_score(result) if result.race_time_seconds is not None else None
    late_kick = build_late_kick_score([result])
    dynamics = build_running_dynamics_profile([result])
    snapshot = race_style_snapshot(result)
    late_gain = snapshot.late_gain if snapshot is not None else ""

    normalized_values = {
        "race_name": result.race_name,
        "race_class": result.race_class,
        "popularity": result.popularity,
        "finish": result.finish_position,
        "passing_order": passing_text,
        "last3f": result.final_3f,
        "time": result.race_time_seconds if result.race_time_seconds is not None else "",
        "race_time_seconds": result.race_time_seconds if result.race_time_seconds is not None else "",
        "time_sec": result.race_time_seconds if result.race_time_seconds is not None else "",
        "avg_speed": round(avg_speed, 4) if avg_speed is not None else "",
        "recent_time_score": round(recent_time_score, 2) if recent_time_score is not None else "",
        "late_kick_score": round(float(late_kick["late_kick_score"]), 2),
        "late_gain_score": round(float(late_kick["late_gain_score"]), 2),
        "relative_agari_score": round(float(late_kick["relative_agari_score"]), 2),
        "winner_time_diff": result.winner_time_diff if result.winner_time_diff is not None else "",
        "margin": result.margin,
        "late_gain": late_gain,
        "early_push_score": round(float(dynamics["early_push_score"]), 2),
        "mid_cruise_score": round(float(dynamics["mid_cruise_score"]), 2),
        "fade_resistance_score": round(float(dynamics["fade_resistance_score"]), 2),
        "sustain_speed_score": round(float(dynamics["sustain_speed_score"]), 2),
        "pace_resilience_score": round(float(dynamics["pace_resilience_score"]), 2),
        "agari_reliability": round(float(dynamics["agari_reliability"]), 2),
        "race_level_weight": race_level_weight(result),
        "race_level_score": race_level_score(result),
        "popularity_score": popularity_to_score(
            result.popularity,
            resolve_field_size(result.field_size, parse_passing_order(result.passing_order), result.finish_position),
            result.finish_position,
        ),
        "finish_score": finish_score(result),
        "date": result.date,
        "course": result.course,
        "distance": result.distance,
        "surface": result.surface,
        "track_condition": result.track_condition,
        "field_size": resolve_field_size(
            result.field_size,
            parse_passing_order(result.passing_order),
            result.finish_position,
        ),
        "field_size_warning": (
            resolve_field_size_with_warning(
                result.field_size,
                parse_passing_order(result.passing_order),
                result.finish_position,
            )[2]
        ),
        "margin_score": margin_score(result),
        "opponent_strength_score": opponent_strength_score(result),
        "race_score": race_score(result),
        "race_strength_score": race_strength_score(result),
        "race_strength_adjusted_score": adjusted_race_score(result),
        "race_id": getattr(result, "race_id", ""),
    }
    for key, value in normalized_values.items():
        if row.get(key) in (None, ""):
            row[key] = value
    return row


def infer_race_level(race: RaceResult) -> str:
    """Infer race class labels from fetched class fields or netkeiba race names."""
    text = f"{race.race_class} {race.race_name}".upper()
    if re.search(r"G3|GIII|GⅢ|ＧⅢ", text):
        return "G3"
    if re.search(r"G2|GII|GⅡ|ＧⅡ", text):
        return "G2"
    if re.search(r"G1|GI|GⅠ|ＧⅠ", text):
        return "G1"
    if re.search(r"LISTED|\(L\)|\bL\b", text):
        return "L"
    if "OP" in text or "オープン" in text:
        return "OP"
    for level in ["3勝", "2勝", "1勝", "未勝利", "新馬"]:
        if level in text:
            return level
    return "OP"


def race_level_weight(race: RaceResult) -> float:
    return RACE_LEVEL_WEIGHT.get(infer_race_level(race), 1.0)


def is_graded_race(race_class: str) -> bool:
    return str(race_class).upper() in {"G1", "G2", "G3"}


def fetch_race_entries(race_id: str) -> list[dict[str, Any]]:
    """Future extension point for loading all entries in a past race.

    The current app does not synthesize opponent rows. If real race-entry
    fetching is not implemented by a provider, callers should use the
    lightweight opponent-strength estimate from the horse's fetched row.
    """
    return []


def race_score(race: RaceResult) -> float:
    """Score one past race, including a lightweight opponent-strength estimate."""
    return clamp(
        finish_score(race) * 0.20
        + margin_score(race) * 0.25
        + time_score(race) * 0.20
        + last3f_score(race) * 0.15
        + class_score(race) * 0.10
        + opponent_strength_score(race) * 0.10
    )


def race_evaluation_score(race: RaceResult) -> float:
    """Score one past race for simulation ability without popularity weighting."""
    return clamp(
        race_level_score(race) * 0.25
        + finish_score(race) * 0.22
        + margin_score(race) * 0.18
        + time_score(race) * 0.15
        + last3f_score(race) * 0.12
        + opponent_strength_score(race) * 0.08
    )


def race_strength_score(race: RaceResult) -> float:
    """Quantify past-race strength without using popularity as a prediction weight."""
    return clamp(
        race_level_score(race) * 0.45
        + average_opponent_score(race) * 0.35
        + field_size_score(race) * 0.20
    )


def adjusted_race_score(race: RaceResult) -> float:
    return clamp(race_evaluation_score(race) * (race_strength_score(race) / 70.0))


def average_opponent_score(race: RaceResult) -> float:
    raw = race.raw or {}
    for key in ("average_opponent_score", "avg_opponent_score", "opponent_score", "相手平均スコア"):
        value = raw.get(key)
        if value not in (None, ""):
            try:
                return clamp(float(value))
            except (TypeError, ValueError):
                pass
    entries = raw.get("race_entries") or raw.get("entries")
    if isinstance(entries, list):
        values: list[float] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            for key in ("horse_ability_score", "race_power", "elo_score", "rating"):
                value = entry.get(key)
                if value not in (None, ""):
                    try:
                        values.append(float(value))
                    except (TypeError, ValueError):
                        pass
                    break
        if values:
            return clamp(sum(values) / len(values))
    return 50.0


def update_elo_ratings(
    race_results: list[dict[str, Any]],
    ratings: dict[str, float] | None = None,
    k_factor: float = 24.0,
) -> dict[str, float]:
    """Update simple pairwise ELO ratings from real same-race finish orders."""
    updated = {str(name): float(value) for name, value in (ratings or {}).items()}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for index, race in enumerate(race_results or []):
        if not isinstance(race, dict):
            continue
        entries = race.get("race_entries") or race.get("entries")
        if isinstance(entries, list):
            race_id = str(race.get("race_id") or race.get("id") or f"race_{index}")
            for entry in entries:
                if isinstance(entry, dict):
                    grouped.setdefault(race_id, []).append(dict(entry))
            continue
        race_id = str(race.get("race_id") or race.get("id") or "").strip()
        if race_id:
            grouped.setdefault(race_id, []).append(race)

    for entries in grouped.values():
        finish_by_name: dict[str, int] = {}
        for entry in entries:
            name = str(entry.get("horse_name") or entry.get("name") or entry.get("馬名") or "").strip()
            finish = _safe_int(entry.get("finish") or entry.get("finish_position") or entry.get("rank") or entry.get("着順"), 0)
            if name and finish > 0:
                finish_by_name[name] = finish
                updated.setdefault(name, 1500.0)
        valid = list(finish_by_name.items())
        if len(valid) < 2:
            continue
        deltas = {name: 0.0 for name, _ in valid}
        divisor = max(1, len(valid) - 1)
        for left_index, (left_name, left_finish) in enumerate(valid):
            for right_name, right_finish in valid[left_index + 1 :]:
                left_rating = updated[left_name]
                right_rating = updated[right_name]
                expected_left = 1.0 / (1.0 + 10.0 ** ((right_rating - left_rating) / 400.0))
                actual_left = 1.0 if left_finish < right_finish else 0.0 if left_finish > right_finish else 0.5
                change = k_factor * (actual_left - expected_left) / divisor
                deltas[left_name] += change
                deltas[right_name] -= change
        for name, change in deltas.items():
            updated[name] = updated[name] + change
    return updated


def normalized_elo_score(rating: float) -> float:
    return clamp(50.0 + (float(rating) - 1500.0) / 8.0)


def opponent_strength_score(race: RaceResult) -> float:
    """Estimate opponent strength from data already present in the race row."""
    field_score = field_size_score(race)
    return clamp(
        class_score(race) * 0.55
        + average_opponent_score(race) * 0.20
        + field_score * 0.10
        + margin_score(race) * 0.15
    )


def popularity_context_score(race: RaceResult) -> float:
    popularity = parse_popularity(race.popularity)
    good_run = race.finish_position <= 3 or effective_margin(race) <= 0.30
    poor_run = race.finish_position >= 6 or effective_margin(race) >= 0.80
    if popularity is None:
        return 65.0 if good_run else 55.0 if poor_run else 60.0
    if good_run:
        if popularity <= 3:
            return 70.0
        if popularity <= 8:
            return 80.0
        return 90.0
    if poor_run:
        if popularity <= 3:
            return 45.0
        if popularity <= 8:
            return 55.0
        return 60.0
    if popularity <= 3:
        return 62.0
    if popularity <= 8:
        return 68.0
    return 74.0


def parse_popularity(value: Any) -> int | None:
    match = re.search(r"\d+", str(value))
    return int(match.group(0)) if match else None


def popularity_to_score(popularity: Any, field_size: int | None = 18, finish: int | None = None) -> float:
    popularity_value = parse_popularity(popularity)
    if popularity_value is None:
        return 50.0
    resolved_field_size = max(1, int(field_size or 18))
    ratio = popularity_value / resolved_field_size
    if ratio <= 0.15:
        score = 90.0
    elif ratio <= 0.30:
        score = 80.0
    elif ratio <= 0.50:
        score = 65.0
    elif ratio <= 0.70:
        score = 50.0
    else:
        score = 35.0
    if finish is not None:
        try:
            finish_value = int(finish)
        except (TypeError, ValueError):
            finish_value = 0
        if popularity_value >= 8 and 0 < finish_value <= 3:
            score += 10.0
        if popularity_value <= 3 and finish_value >= 8:
            score -= 10.0
    return clamp(score)


def field_size_score(race: RaceResult) -> float:
    field_size = resolve_field_size(race.field_size, parse_passing_order(race.passing_order), race.finish_position)
    return clamp(field_size / 18.0 * 100.0)


def finish_score(race: RaceResult) -> float:
    field_size = resolve_field_size(race.field_size, parse_passing_order(race.passing_order), race.finish_position)
    finish_ratio = max(1, race.finish_position) / max(1, field_size)
    score = 100.0 * (1.0 - finish_ratio)
    if race.finish_position == 1:
        score = max(score, 90.0)
    elif race.finish_position <= 3:
        score = max(score, 78.0)
    elif race.finish_position <= 5:
        score = max(score, 65.0)
    return clamp(score)


def margin_score(race: RaceResult) -> float:
    margin = effective_margin(race)
    return clamp(100.0 - margin * 35.0 - max(0, race.finish_position - 1) * 2.0)


def effective_margin(race: RaceResult) -> float:
    if race.winner_time_diff is not None:
        return max(0.0, race.winner_time_diff)
    return max(0.0, race.margin)


def class_score(race: RaceResult) -> float:
    return clamp(50.0 + (race_level_weight(race) - 1.0) * 70.0)


def race_level_score(race: RaceResult) -> float:
    return RACE_LEVEL_SCORE.get(infer_race_level(race), 60.0)


def last3f_score(race: RaceResult) -> float:
    if race.final_3f <= 0:
        return 55.0
    return clamp(50.0 + (36.4 - race.final_3f) * 12.0)


def time_score(race: RaceResult) -> float:
    if race.race_time_seconds is None or race.race_time_seconds <= 0 or race.distance <= 0:
        return 55.0
    speed = race.distance / race.race_time_seconds
    expected = expected_speed(race)
    diff_penalty = effective_margin(race) * 7.0
    return clamp(55.0 + (speed - expected) * 42.0 - diff_penalty)


def expected_speed(race: RaceResult) -> float:
    surface = str(race.surface)
    if surface == "ダート":
        base = 15.65
    else:
        base = 16.35
    if race.distance <= 1400:
        base += 0.30
    elif race.distance >= 2400:
        base -= 0.25
    condition = normalize_track_condition(race.track_condition)
    if condition == "稍重":
        base -= 0.08
    elif condition == "重":
        base -= 0.22
    elif condition == "不良":
        base -= 0.30
    return base


class RaceResultProvider(Protocol):
    """Adapter protocol for the already implemented recent-race fetcher."""

    def get_recent_results(self, horse_name: str, limit: int = 5) -> list[RaceResult]:
        ...


@dataclass(frozen=True)
class RaceStyleSnapshot:
    debug_passing_order: str
    first_pos: int
    mid_pos: int
    last_corner_pos: int
    finish_pos: int
    field_size: int
    first_ratio: float
    mid_ratio: float
    last_corner_ratio: float
    position_ratio: float
    late_gain: float
    all_lead: bool
    front_like: bool
    back_like: bool
    field_size_inferred: bool
    field_size_warning: str


@dataclass(frozen=True)
class StyleMetrics:
    running_style: str
    base_style_profile: dict[str, float]
    front_runner_score: float
    stalker_score: float
    closer_score: float
    deep_closer_score: float
    versatile_score: float
    weighted_avg_first_ratio: float
    weighted_avg_mid_ratio: float
    weighted_avg_last_corner_ratio: float
    weighted_avg_late_gain: float
    weighted_avg_position_ratio: float
    position_variance: float
    style_sample_size: int
    debug_passing_orders: list[str]
    debug_field_sizes: list[int]
    debug_first_ratios: list[float]
    debug_mid_ratios: list[float]
    debug_last_corner_ratios: list[float]
    field_size_warnings: list[str]


@dataclass(frozen=True)
class HorseAbility:
    horse_name: str
    frame: int
    horse_number: int
    early_speed: float
    stamina: float
    acceleration: float
    front_runner: float
    stalker: float
    closer: float
    mud_aptitude: float
    consistency: float
    running_style: str
    primary_running_style: str
    base_style_profile: dict[str, float]
    recent_results: list[RaceResult]
    front_runner_score: float
    stalker_score: float
    closer_score: float
    deep_closer_score: float
    versatile_score: float
    weighted_avg_first_ratio: float
    weighted_avg_mid_ratio: float
    weighted_avg_last_corner_ratio: float
    weighted_avg_late_gain: float
    weighted_avg_position_ratio: float
    position_variance: float
    style_sample_size: int
    debug_passing_orders: list[str]
    avg_opponent_strength_score: float
    avg_race_score: float
    popularity_score: float
    race_level_score: float
    finish_score: float
    margin_score: float
    time_score: float
    last3f_score: float
    horse_ability_score: float
    race_power: float
    early_aggressiveness: float
    mid_positioning: float
    late_kick_timing: float
    sustain_speed: float
    time_reliability: float
    recent_time_score: float
    late_kick_score: float
    avg_last3f: float | None
    best_last3f: float | None
    last3f_consistency: float
    late_gain_score: float
    early_push_score: float
    mid_cruise_score: float
    fade_resistance_score: float
    sustain_speed_score: float
    pace_resilience_score: float
    agari_reliability: float
    carried_weight: float = 56.0
    weight_penalty: float = 0.0
    mud_source: str = "neutral"
    base_ability_score: float = 50.0
    race_strength_score: float = 50.0
    race_strength_adjusted_score: float = 50.0
    elo_rating: float = 1500.0
    normalized_elo_score: float = 50.0
    relative_agari_score: float = 50.0
    course_fit_score: float = 50.0
    jockey: str = ""
    jockey_score: float = 50.0
    track_bias_fit_score: float = 50.0
    pace_fit_score: float = 50.0
    debug_field_sizes: list[int] = field(default_factory=list)
    debug_first_ratios: list[float] = field(default_factory=list)
    debug_mid_ratios: list[float] = field(default_factory=list)
    debug_last_corner_ratios: list[float] = field(default_factory=list)
    field_size_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "horse_name": self.horse_name,
            "frame": self.frame,
            "horse_number": self.horse_number,
            "carried_weight": round(self.carried_weight, 1),
            "weight_penalty": round(self.weight_penalty, 2),
            "early_speed": round(self.early_speed, 2),
            "stamina": round(self.stamina, 2),
            "acceleration": round(self.acceleration, 2),
            "front_runner": round(self.front_runner, 2),
            "stalker": round(self.stalker, 2),
            "closer": round(self.closer, 2),
            "mud_aptitude": round(self.mud_aptitude, 2),
            "mud_source": self.mud_source,
            "base_ability_score": round(self.base_ability_score, 2),
            "race_strength_score": round(self.race_strength_score, 2),
            "race_strength_adjusted_score": round(self.race_strength_adjusted_score, 2),
            "elo_rating": round(self.elo_rating, 2),
            "normalized_elo_score": round(self.normalized_elo_score, 2),
            "relative_agari_score": round(self.relative_agari_score, 2),
            "course_fit_score": round(self.course_fit_score, 2),
            "jockey": self.jockey,
            "jockey_score": round(self.jockey_score, 2),
            "track_bias_fit_score": round(self.track_bias_fit_score, 2),
            "pace_fit_score": round(self.pace_fit_score, 2),
            "consistency": round(self.consistency, 2),
            "running_style": self.running_style,
            "primary_running_style": self.primary_running_style,
            "base_style_profile": self.base_style_profile,
            "base_逃げ": round(self.base_style_profile["逃げ"], 3),
            "base_先行": round(self.base_style_profile["先行"], 3),
            "base_差し": round(self.base_style_profile["差し"], 3),
            "base_追込": round(self.base_style_profile["追込"], 3),
            "front_runner_score": round(self.front_runner_score, 2),
            "stalker_score": round(self.stalker_score, 2),
            "closer_score": round(self.closer_score, 2),
            "deep_closer_score": round(self.deep_closer_score, 2),
            "versatile_score": round(self.versatile_score, 2),
            "weighted_avg_first_ratio": round(self.weighted_avg_first_ratio, 3),
            "weighted_avg_mid_ratio": round(self.weighted_avg_mid_ratio, 3),
            "weighted_avg_last_corner_ratio": round(self.weighted_avg_last_corner_ratio, 3),
            "avg_first_ratio": round(self.weighted_avg_first_ratio, 3),
            "avg_mid_ratio": round(self.weighted_avg_mid_ratio, 3),
            "avg_last_corner_ratio": round(self.weighted_avg_last_corner_ratio, 3),
            "weighted_avg_late_gain": round(self.weighted_avg_late_gain, 2),
            "weighted_avg_position_ratio": round(self.weighted_avg_position_ratio, 3),
            "position_variance": round(self.position_variance, 4),
            "style_sample_size": self.style_sample_size,
            "debug_passing_orders": self.debug_passing_orders,
            "debug_field_sizes": self.debug_field_sizes,
            "debug_first_ratios": [round(value, 4) for value in self.debug_first_ratios],
            "debug_mid_ratios": [round(value, 4) for value in self.debug_mid_ratios],
            "debug_last_corner_ratios": [round(value, 4) for value in self.debug_last_corner_ratios],
            "field_size_warnings": self.field_size_warnings,
            "avg_opponent_strength_score": round(self.avg_opponent_strength_score, 2),
            "avg_race_score": round(self.avg_race_score, 2),
            "popularity_score": round(self.popularity_score, 2),
            "race_level_score": round(self.race_level_score, 2),
            "finish_score": round(self.finish_score, 2),
            "margin_score": round(self.margin_score, 2),
            "time_score": round(self.time_score, 2),
            "last3f_score": round(self.last3f_score, 2),
            "horse_ability_score": round(self.horse_ability_score, 2),
            "race_power": round(self.race_power, 2),
            "early_aggressiveness": round(self.early_aggressiveness, 3),
            "mid_positioning": round(self.mid_positioning, 3),
            "late_kick_timing": round(self.late_kick_timing, 3),
            "sustain_speed": round(self.sustain_speed, 3),
            "time_reliability": round(self.time_reliability, 3),
            "recent_time_score": round(self.recent_time_score, 2),
            "late_kick_score": round(self.late_kick_score, 2),
            "avg_last3f": round(self.avg_last3f, 3) if self.avg_last3f is not None else None,
            "best_last3f": round(self.best_last3f, 3) if self.best_last3f is not None else None,
            "last3f_consistency": round(self.last3f_consistency, 2),
            "late_gain_score": round(self.late_gain_score, 2),
            "early_push_score": round(self.early_push_score, 2),
            "mid_cruise_score": round(self.mid_cruise_score, 2),
            "fade_resistance_score": round(self.fade_resistance_score, 2),
            "sustain_speed_score": round(self.sustain_speed_score, 2),
            "pace_resilience_score": round(self.pace_resilience_score, 2),
            "agari_reliability": round(self.agari_reliability, 2),
        }

    @property
    def avg_first_ratio(self) -> float:
        return self.weighted_avg_first_ratio

    @property
    def avg_last_corner_ratio(self) -> float:
        return self.weighted_avg_last_corner_ratio

    @property
    def avg_late_gain(self) -> float:
        return self.weighted_avg_late_gain


def race_style_snapshot(race: dict[str, Any] | RaceResult) -> RaceStyleSnapshot | None:
    if isinstance(race, RaceResult):
        passing_order = race.passing_order
        finish_pos = race.finish_position
        field_size = race.field_size
    else:
        passing_order = race.get("passing_order", race.get("通過", ""))
        finish_pos = race.get("finish_position", race.get("finish", race.get("着順", None)))
        field_size = _first_non_empty_value(
            race,
            ["field_size", "runners", "number_of_runners", "頭数", "出走頭数"],
        )

    positions = parse_passing_order(passing_order)
    if not positions:
        return None
    try:
        finish = int(finish_pos or positions[-1])
    except (TypeError, ValueError):
        finish = positions[-1]

    resolved_field_size, field_size_inferred, field_size_warning = resolve_field_size_with_warning(
        field_size,
        positions,
        finish,
    )
    first_pos = positions[0]
    if len(positions) >= 4:
        mid_pos = positions[(len(positions) // 2) - 1]
        last_corner_pos = positions[-2]
    else:
        mid_pos = positions[len(positions) // 2]
        last_corner_pos = positions[-1]

    first_ratio = first_pos / resolved_field_size
    mid_ratio = mid_pos / resolved_field_size
    last_ratio = last_corner_pos / resolved_field_size
    late_gain = float(last_corner_pos - finish)
    debug = "-".join(str(position) for position in positions)
    return RaceStyleSnapshot(
        debug_passing_order=debug,
        first_pos=first_pos,
        mid_pos=mid_pos,
        last_corner_pos=last_corner_pos,
        finish_pos=max(1, finish),
        field_size=resolved_field_size,
        first_ratio=first_ratio,
        mid_ratio=mid_ratio,
        last_corner_ratio=last_ratio,
        position_ratio=(first_ratio + mid_ratio + last_ratio) / 3.0,
        late_gain=late_gain,
        all_lead=all(position == 1 for position in positions),
        front_like=first_ratio <= 0.30 and last_ratio <= 0.35,
        back_like=first_ratio >= 0.50 or mid_ratio >= 0.50,
        field_size_inferred=field_size_inferred,
        field_size_warning=field_size_warning,
    )


def resolve_field_size(field_size: Any, positions: list[int], finish_pos: int) -> int:
    return resolve_field_size_with_warning(field_size, positions, finish_pos)[0]


def resolve_field_size_with_warning(
    field_size: Any,
    positions: list[int],
    finish_pos: int,
) -> tuple[int, bool, str]:
    try:
        parsed = int(field_size or 0)
    except (TypeError, ValueError):
        parsed = 0
    if parsed > 0:
        return max(2, parsed), False, ""
    if positions:
        inferred = max(max(positions) + 1, finish_pos, 2)
        return inferred, True, f"field_size欠損: 通過順から{inferred}頭として推定"
    return 18, True, "field_size欠損: 通過順から推定できないため18頭として仮置き"


def _first_non_empty_value(row: dict[str, Any], candidates: list[str]) -> Any:
    for key in candidates:
        value = row.get(key)
        if value is not None and str(value).strip() != "":
            return value
    return None


def build_race_tactics_profile(recent_races: list[dict[str, Any] | RaceResult]) -> dict[str, float]:
    """Build tactics features from the latest five races.

    Values are normalized to 0.0-1.0 except recent_time_score, which is 0-100
    for direct use beside the other ability scores.
    """
    races = list(recent_races or [])[:5]
    if not races:
        return {
            "early_aggressiveness": 0.5,
            "mid_positioning": 0.5,
            "late_kick_timing": 0.5,
            "sustain_speed": 0.55,
            "time_reliability": 0.0,
            "recent_time_score": 55.0,
        }

    weights = STYLE_WEIGHTS[: len(races)]
    early_values: list[float] = []
    early_weights: list[float] = []
    mid_values: list[float] = []
    mid_weights: list[float] = []
    late_values: list[float] = []
    late_weights: list[float] = []
    sustain_values: list[float] = []
    sustain_weights: list[float] = []
    time_scores: list[float] = []
    time_score_weights: list[float] = []
    valid_time_count = 0

    for index, race in enumerate(races):
        weight = weights[index]
        snapshot = race_style_snapshot(race)
        if snapshot is not None:
            early_values.append(_clamp_unit(1.0 - snapshot.first_ratio))
            early_weights.append(weight)
            mid_values.append(_clamp_unit(1.0 - snapshot.mid_ratio))
            mid_weights.append(weight)
            late_values.append(_clamp_unit(max(0.0, snapshot.late_gain) / max(1.0, snapshot.field_size * 0.35)))
            late_weights.append(weight)

        time_sec, distance, time_score_value = _race_time_features(race)
        if time_sec is not None and time_sec > 0 and distance > 0:
            valid_time_count += 1
            time_scores.append(time_score_value)
            time_score_weights.append(weight)
            sustain_values.append(_clamp_unit(time_score_value / 100.0))
            sustain_weights.append(weight)

    sample_size = min(5, len(races))
    time_reliability = valid_time_count / max(1, sample_size)
    recent_time_score = _weighted_average(time_scores, time_score_weights, default=55.0)

    return {
        "early_aggressiveness": _weighted_average(early_values, early_weights, default=0.5),
        "mid_positioning": _weighted_average(mid_values, mid_weights, default=0.5),
        "late_kick_timing": _weighted_average(late_values, late_weights, default=0.5),
        "sustain_speed": _weighted_average(sustain_values, sustain_weights, default=recent_time_score / 100.0),
        "time_reliability": _clamp_unit(time_reliability),
        "recent_time_score": clamp(recent_time_score, 0.0, 100.0),
    }


def build_late_kick_score(recent_races: list[dict[str, Any] | RaceResult]) -> dict[str, float | None]:
    """Estimate final-stretch quality from recent last-3F and late gains."""
    races = list(recent_races or [])[:5]
    if not races:
        return {
            "late_kick_score": 55.0,
            "avg_last3f": None,
            "best_last3f": None,
            "last3f_consistency": 50.0,
            "late_gain_score": 50.0,
            "relative_agari_score": 50.0,
        }

    weights = STYLE_WEIGHTS[: len(races)]
    last3f_values: list[float] = []
    last3f_weights: list[float] = []
    last3f_scores: list[float] = []
    relative_agari_scores: list[float] = []
    late_gain_scores: list[float] = []
    late_gain_weights: list[float] = []
    class_scores: list[float] = []
    class_weights: list[float] = []

    fallback_last3f_values = [
        value
        for race in races
        for value, _ in [_race_last3f_features(race)]
        if value is not None and value > 0
    ]
    fallback_avg_last3f = (
        sum(fallback_last3f_values) / len(fallback_last3f_values)
        if fallback_last3f_values
        else None
    )

    for index, race in enumerate(races):
        weight = weights[index]
        last3f_value, last3f_score_value = _race_last3f_features(race)
        if last3f_value is not None and last3f_value > 0:
            last3f_values.append(last3f_value)
            last3f_weights.append(weight)
            last3f_scores.append(last3f_score_value)
            race_avg_last3f = _race_average_last3f(race) or fallback_avg_last3f
            if race_avg_last3f is not None:
                relative_agari_scores.append(clamp(50.0 + (race_avg_last3f - last3f_value) * 15.0))
            else:
                relative_agari_scores.append(50.0)

        snapshot = race_style_snapshot(race)
        if snapshot is not None:
            gain_score = clamp(max(0.0, snapshot.late_gain) / max(1.0, snapshot.field_size * 0.35) * 100.0)
            late_gain_scores.append(gain_score)
            late_gain_weights.append(weight)

        class_scores.append(_race_class_score(race))
        class_weights.append(weight)

    avg_last3f = _weighted_average(last3f_values, last3f_weights, default=0.0) if last3f_values else None
    best_last3f = min(last3f_values) if last3f_values else None
    last3f_speed_score = _weighted_average(last3f_scores, last3f_weights, default=55.0)
    late_gain_score_value = _weighted_average(late_gain_scores, late_gain_weights, default=50.0)
    relative_agari_score_value = _weighted_average(relative_agari_scores, last3f_weights, default=50.0)

    if len(last3f_values) >= 2:
        mean = sum(last3f_values) / len(last3f_values)
        variance = sum((value - mean) ** 2 for value in last3f_values) / len(last3f_values)
        last3f_consistency = clamp(100.0 - variance ** 0.5 * 24.0)
    elif last3f_values:
        last3f_consistency = 65.0
    else:
        last3f_consistency = 50.0

    late_kick_score = clamp(
        relative_agari_score_value * 0.40
        + last3f_speed_score * 0.25
        + late_gain_score_value * 0.25
        + last3f_consistency * 0.10
    )
    return {
        "late_kick_score": late_kick_score,
        "avg_last3f": avg_last3f,
        "best_last3f": best_last3f,
        "last3f_consistency": last3f_consistency,
        "late_gain_score": late_gain_score_value,
        "relative_agari_score": relative_agari_score_value,
    }


def build_running_dynamics_profile(recent_races: list[dict[str, Any] | RaceResult]) -> dict[str, float]:
    """Build race-movement dynamics from the latest five races."""
    races = list(recent_races or [])[:5]
    if not races:
        return {
            "early_push_score": 50.0,
            "mid_cruise_score": 50.0,
            "late_kick_score": 55.0,
            "fade_resistance_score": 55.0,
            "sustain_speed_score": 55.0,
            "pace_resilience_score": 55.0,
            "agari_reliability": 0.0,
            "time_reliability": 0.0,
        }

    weights = STYLE_WEIGHTS[: len(races)]
    early_scores: list[float] = []
    early_weights: list[float] = []
    mid_scores: list[float] = []
    mid_weights: list[float] = []
    fade_scores: list[float] = []
    fade_weights: list[float] = []
    sustain_scores: list[float] = []
    sustain_weights: list[float] = []
    pace_scores: list[float] = []
    pace_weights: list[float] = []
    last3f_values: list[float] = []
    valid_time_count = 0

    for index, race in enumerate(races):
        weight = weights[index]
        snapshot = race_style_snapshot(race)
        time_sec, distance, time_score_value = _race_time_features(race)
        last3f_value, last3f_score_value = _race_last3f_features(race)
        margin_score_value = _race_margin_score(race)
        class_score_value = _race_class_score(race)
        sustain_score = clamp(time_score_value * 0.55 + margin_score_value * 0.25 + class_score_value * 0.20)

        if time_sec is not None and time_sec > 0 and distance > 0:
            valid_time_count += 1
            sustain_scores.append(sustain_score)
            sustain_weights.append(weight)
        if last3f_value is not None:
            last3f_values.append(last3f_value)

        if snapshot is None:
            continue

        early_score = clamp((1.0 - snapshot.first_ratio) * 100.0)
        mid_position_score = clamp((1.0 - snapshot.mid_ratio) * 100.0)
        late_position_loss = max(0.0, float(snapshot.finish_pos - snapshot.last_corner_pos))
        position_hold = clamp(100.0 - late_position_loss * 12.0)
        front_pressure = clamp((1.0 - min(snapshot.first_ratio, snapshot.mid_ratio)) * 100.0)
        pace_score = clamp(
            front_pressure * 0.30
            + margin_score_value * 0.25
            + last3f_score_value * 0.25
            + class_score_value * 0.20
        )

        early_scores.append(early_score)
        early_weights.append(weight)
        mid_scores.append(clamp(mid_position_score * 0.55 + time_score_value * 0.45))
        mid_weights.append(weight)
        fade_scores.append(clamp(position_hold * 0.35 + sustain_score * 0.25 + time_score_value * 0.25 + margin_score_value * 0.15))
        fade_weights.append(weight)
        pace_scores.append(pace_score)
        pace_weights.append(weight)

    late_profile = build_late_kick_score(races)
    sample_size = max(1, min(5, len(races)))
    agari_reliability = clamp((len(last3f_values) / sample_size) * 65.0 + float(late_profile["last3f_consistency"]) * 0.35)
    time_reliability = clamp(valid_time_count / sample_size * 100.0)
    sustain_speed_score = _weighted_average(sustain_scores, sustain_weights, default=55.0)

    return {
        "early_push_score": clamp(_weighted_average(early_scores, early_weights, default=50.0)),
        "mid_cruise_score": clamp(_weighted_average(mid_scores, mid_weights, default=50.0)),
        "late_kick_score": clamp(float(late_profile["late_kick_score"])),
        "fade_resistance_score": clamp(_weighted_average(fade_scores, fade_weights, default=55.0)),
        "sustain_speed_score": clamp(sustain_speed_score),
        "pace_resilience_score": clamp(_weighted_average(pace_scores, pace_weights, default=55.0)),
        "agari_reliability": agari_reliability,
        "time_reliability": time_reliability,
    }


def _race_time_features(race: dict[str, Any] | RaceResult) -> tuple[float | None, int, float]:
    if isinstance(race, RaceResult):
        time_sec = race.race_time_seconds
        distance = int(race.distance or 0)
        return time_sec, distance, time_score(race)

    time_sec = parse_race_time(
        race.get("time")
        or race.get("race_time")
        or race.get("result_time")
        or race.get("race_time_seconds")
        or race.get("time_sec")
        or race.get("走破タイム")
        or race.get("タイム")
    )
    distance = int(float(race.get("distance", race.get("距離", 0)) or 0))
    score_value = _time_score_from_values(time_sec, distance, race)
    return time_sec, distance, score_value


def _race_last3f_features(race: dict[str, Any] | RaceResult) -> tuple[float | None, float]:
    if isinstance(race, RaceResult):
        value = race.final_3f if race.final_3f > 0 else None
        return value, last3f_score(race)

    value = _safe_float(
        race.get("last3f")
        or race.get("final_3f")
        or race.get("agari")
        or race.get("closing_3f")
        or race.get("上り")
        or race.get("上がり")
        or race.get("上がり3F"),
        default=0.0,
    )
    if value <= 0:
        return None, 55.0
    distance = int(float(race.get("distance", race.get("距離", 1800)) or 1800))
    track_condition = str(race.get("track_condition", race.get("馬場状態", "")))
    reference = 36.4
    if distance <= 1400:
        reference -= 0.15
    elif distance >= 2400:
        reference += 0.20
    if any(word in track_condition for word in ["稍重", "重", "不良", "heavy", "bad"]):
        reference += 0.20
    return value, clamp(50.0 + (reference - value) * 12.0)


def _race_average_last3f(race: dict[str, Any] | RaceResult) -> float | None:
    row = race.raw or {} if isinstance(race, RaceResult) else race
    for key in (
        "race_avg_last3f",
        "average_last3f",
        "field_avg_last3f",
        "avg_last3f",
        "レース平均上り",
        "平均上り",
    ):
        value = row.get(key)
        if value not in (None, ""):
            parsed = _safe_float(value, 0.0)
            if parsed > 0:
                return parsed
    return None


def _race_class_score(race: dict[str, Any] | RaceResult) -> float:
    if isinstance(race, RaceResult):
        return class_score(race)
    race_class = str(race.get("race_class", race.get("class", race.get("grade", race.get("クラス", ""))))).upper()
    race_name = str(race.get("race_name", race.get("name", race.get("race", race.get("レース名", ""))))).upper()
    text = f"{race_class} {race_name}"
    for level, weight in RACE_LEVEL_WEIGHT.items():
        if level.upper() in text:
            return clamp(50.0 + (weight - 1.0) * 70.0)
    return 55.0


def _race_margin_score(race: dict[str, Any] | RaceResult) -> float:
    if isinstance(race, RaceResult):
        return margin_score(race)
    margin = _safe_float(
        race.get("winner_time_diff")
        or race.get("margin_sec")
        or race.get("margin")
        or race.get("着差")
        or 0.0
    )
    finish = int(_safe_float(race.get("finish_position", race.get("finish", race.get("着順", 1))), 1.0))
    return clamp(100.0 - max(0.0, margin) * 35.0 - max(0, finish - 1) * 2.0)


def _race_track_condition(race: dict[str, Any] | RaceResult) -> str:
    if isinstance(race, RaceResult):
        return str(race.track_condition)
    return str(
        race.get("track_condition")
        or race.get("condition")
        or race.get("馬場状態")
        or race.get("馬場")
        or ""
    )


def _finish_score_from_snapshot(snapshot: RaceStyleSnapshot | None) -> float:
    if snapshot is None:
        return 50.0
    ratio = snapshot.finish_pos / max(1, snapshot.field_size)
    value = 100.0 * (1.0 - ratio)
    if snapshot.finish_pos == 1:
        value = max(value, 90.0)
    elif snapshot.finish_pos <= 3:
        value = max(value, 78.0)
    elif snapshot.finish_pos <= 5:
        value = max(value, 65.0)
    return clamp(value)


def _pedigree_mud_score(pedigree_info: dict[str, Any] | None) -> float | None:
    if not isinstance(pedigree_info, dict):
        return None
    for key in ("pedigree_mud_score", "mud_aptitude", "mud_score", "道悪適性"):
        value = pedigree_info.get(key)
        if value is None or str(value).strip() == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _time_score_from_values(time_sec: float | None, distance: int, race: dict[str, Any]) -> float:
    if time_sec is None or time_sec <= 0 or distance <= 0:
        return 55.0
    avg_speed = distance / time_sec
    surface = str(race.get("surface", race.get("芝/ダート", "芝")))
    expected = 15.65 if surface == "ダート" else 16.35
    if distance <= 1400:
        expected += 0.30
    elif distance >= 2400:
        expected -= 0.25
    margin = _safe_float(
        race.get("winner_time_diff")
        or race.get("margin_sec")
        or race.get("margin")
        or race.get("着差")
        or 0.0
    )
    return clamp(55.0 + (avg_speed - expected) * 42.0 - max(0.0, margin) * 7.0)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        match = re.search(r"-?\d+", str(value))
        return int(match.group(0)) if match else default
    except (TypeError, ValueError):
        return default


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _weighted_average(values: list[float], weights: list[float], default: float = 0.0) -> float:
    if not values:
        return default
    usable_weights = weights[: len(values)]
    total_weight = sum(usable_weights)
    if total_weight <= 0:
        return default
    return sum(value * weight for value, weight in zip(values, usable_weights)) / total_weight


def race_style_contribution(snapshot: RaceStyleSnapshot) -> dict[str, float]:
    """Return a soft style contribution for a single race."""
    if snapshot.all_lead:
        return {"逃げ": 0.90, "先行": 0.10, "差し": 0.0, "追込": 0.0}
    if snapshot.first_pos <= 2 and snapshot.first_ratio <= 0.15 and snapshot.last_corner_ratio <= 0.20:
        return normalize_style_profile({"逃げ": 0.78, "先行": 0.20, "差し": 0.02, "追込": 0.0})

    front = max(0.0, (0.22 - snapshot.first_ratio) / 0.22) * 0.48
    front += max(0.0, (0.24 - snapshot.last_corner_ratio) / 0.24) * 0.34
    if snapshot.first_pos <= 2:
        front += 0.16

    stalker = max(0.0, 1.0 - abs(snapshot.first_ratio - 0.26) / 0.24) * 0.55
    stalker += max(0.0, 1.0 - abs(snapshot.last_corner_ratio - 0.30) / 0.24) * 0.38
    if snapshot.first_ratio <= 0.40 and snapshot.last_corner_ratio <= 0.40:
        stalker += 0.22
    if snapshot.first_pos <= 2 and snapshot.last_corner_ratio <= 0.22:
        stalker *= 0.72

    closer = max(0.0, (snapshot.first_ratio - 0.36) / 0.34) * 0.34
    closer += max(0.0, (snapshot.last_corner_ratio - 0.32) / 0.34) * 0.24
    closer += max(0.0, min(snapshot.late_gain, 6.0)) / 6.0 * 0.52
    if snapshot.first_ratio <= 0.35 and snapshot.last_corner_ratio <= 0.35:
        closer *= 0.55

    deep = max(0.0, (snapshot.first_ratio - 0.56) / 0.34) * 0.38
    deep += max(0.0, (snapshot.mid_ratio - 0.56) / 0.34) * 0.28
    deep += max(0.0, min(snapshot.late_gain - 2.0, 8.0)) / 8.0 * 0.62
    if snapshot.first_ratio < 0.58:
        deep *= 0.60

    if snapshot.first_ratio > 0.40 and snapshot.late_gain >= 2.0:
        stalker *= 0.55
        closer += 0.18
    if snapshot.first_ratio >= 0.60 and snapshot.mid_ratio >= 0.60 and snapshot.late_gain >= 3.0:
        stalker *= 0.30
        closer *= 0.62
        deep += 0.50

    return normalize_style_profile({"逃げ": front, "先行": stalker, "差し": closer, "追込": deep})


class HorseAnalyzer:
    """Estimate ability from five starts and style probabilities from up to ten."""

    def __init__(self, provider: RaceResultProvider, race_config: Any | None = None) -> None:
        self.provider = provider
        self.race_config = race_config

    def analyze(self, horse: HorseEntry) -> HorseAbility:
        try:
            style_results = self.provider.get_recent_results(horse.horse_name, limit=10)
        except RaceDataFetchError:
            raise
        except Exception as exc:
            debug_records = self.provider.get_fetch_debug() if hasattr(self.provider, "get_fetch_debug") else []
            raise RaceDataFetchError(
                f"{horse.horse_name} の近走データ取得に失敗しました: {exc}",
                horse_name=horse.horse_name,
                debug_records=debug_records,
            ) from exc
        if not style_results:
            debug_records = self.provider.get_fetch_debug() if hasattr(self.provider, "get_fetch_debug") else []
            raise RaceDataFetchError(
                f"{horse.horse_name} の近走データが取得できませんでした。馬名または取得処理を確認してください。",
                horse_name=horse.horse_name,
                debug_records=debug_records,
            )
        style_results = style_results[:10]
        ability_results = style_results[:5]
        if not ability_results:
            debug_records = self.provider.get_fetch_debug() if hasattr(self.provider, "get_fetch_debug") else []
            raise RaceDataFetchError(
                f"{horse.horse_name} の近走5走データが取得できませんでした。",
                horse_name=horse.horse_name,
                debug_records=debug_records,
            )

        style = self._style_metrics(style_results)
        recency_weights = ABILITY_RECENCY_WEIGHTS[: len(ability_results)]
        level_weights = [race_level_weight(race) for race in ability_results]
        ability_weights = [recency * level for recency, level in zip(recency_weights, level_weights)]
        avg_level_weight = self._weighted_mean(level_weights, recency_weights, default=1.0)
        level_bonus = (avg_level_weight - 1.0) * 18.0
        field_sizes = [
            resolve_field_size(race.field_size, parse_passing_order(race.passing_order), race.finish_position)
            for race in ability_results
        ]
        popularity_scores = [
            popularity_to_score(race.popularity, field_size, race.finish_position)
            for race, field_size in zip(ability_results, field_sizes)
        ]
        race_level_scores = [race_level_score(race) for race in ability_results]
        finish_scores = [finish_score(race) for race in ability_results]
        margin_scores = [margin_score(race) for race in ability_results]
        all_time_scores = [time_score(race) for race in ability_results]
        last3f_scores = [last3f_score(race) for race in ability_results]
        race_evaluation_scores = [race_evaluation_score(race) for race in ability_results]
        race_strength_scores = [race_strength_score(race) for race in ability_results]
        strength_adjusted_scores = [adjusted_race_score(race) for race in ability_results]
        avg_popularity_score = self._weighted_mean(popularity_scores, recency_weights, default=50.0)
        avg_race_level_score = self._weighted_mean(race_level_scores, recency_weights, default=60.0)
        avg_finish_score = self._weighted_mean(finish_scores, recency_weights, default=50.0)
        avg_margin_score = self._weighted_mean(margin_scores, recency_weights, default=50.0)
        avg_component_time_score = self._weighted_mean(all_time_scores, recency_weights, default=50.0)
        avg_last3f_score = self._weighted_mean(last3f_scores, recency_weights, default=50.0)
        base_ability_score = self._weighted_mean(race_evaluation_scores, recency_weights, default=55.0)
        avg_race_strength = self._weighted_mean(race_strength_scores, recency_weights, default=50.0)
        avg_strength_adjusted = self._weighted_mean(strength_adjusted_scores, recency_weights, default=50.0)
        jockey_score_value = self._resolve_jockey_score(horse)
        neutral_elo_score = 50.0
        horse_ability_score = clamp(base_ability_score)

        avg_finish = self._weighted_mean([r.finish_position for r in ability_results], ability_weights, default=6.0)
        avg_margin = self._weighted_mean([r.margin for r in ability_results], ability_weights, default=0.8)
        final_3f_values = [r.final_3f for r in ability_results if r.final_3f > 0]
        final_3f_weights = [weight for r, weight in zip(ability_results, ability_weights) if r.final_3f > 0]
        avg_final_3f = self._weighted_mean(final_3f_values, final_3f_weights, default=36.0)
        distance_mean = self._weighted_mean([r.distance for r in ability_results], ability_weights, default=1800.0)
        finish_std = self._weighted_variance([r.finish_position for r in ability_results], ability_weights) ** 0.5
        performance_values = [self._performance_score(r) for r in ability_results]
        avg_performance = self._weighted_mean(performance_values, ability_weights, default=55.0)
        opponent_values = [opponent_strength_score(r) for r in ability_results]
        race_score_values = [race_score(r) for r in ability_results]
        avg_opponent_strength = self._weighted_mean(opponent_values, ability_weights, default=55.0)
        avg_race_score = self._weighted_mean(race_score_values, ability_weights, default=55.0)
        race_power = clamp(horse_ability_score * 0.75 + avg_race_score * 0.25, 40.0, 95.0)
        time_scores = [
            self._time_score(r)
            for r in ability_results
            if r.race_time_seconds is not None and r.race_time_seconds > 0
        ]
        time_score_weights = [
            weight
            for r, weight in zip(ability_results, ability_weights)
            if r.race_time_seconds is not None and r.race_time_seconds > 0
        ]
        avg_time_score = self._weighted_mean(time_scores, time_score_weights, default=55.0)
        tactics_profile = build_race_tactics_profile(ability_results)
        late_kick_profile = build_late_kick_score(ability_results)
        dynamics_profile = build_running_dynamics_profile(ability_results)
        course_fit_score = estimate_course_fit_score(ability_results, self.race_config)
        track_bias_fit_score = get_track_bias_fit_score(
            self.race_config or {},
            style.running_style,
            horse.frame,
        )
        course_fit_score = clamp(course_fit_score * 0.90 + track_bias_fit_score * 0.10)
        recent_time_score = float(tactics_profile.get("recent_time_score", avg_time_score))
        performance_bonus = (horse_ability_score - 55.0) * 0.42 + (avg_performance - 55.0) * 0.12
        time_bonus = (avg_time_score - 55.0) * 0.22 if time_scores else 0.0

        weight_penalty = calculate_weight_penalty(horse.carried_weight)
        early_speed = clamp(108.0 - style.weighted_avg_first_ratio * 62.0 - avg_margin * 5.0 + style.front_runner_score * 0.12 + level_bonus + performance_bonus * 0.45 + time_bonus)
        acceleration = clamp(66.0 + (36.4 - avg_final_3f) * 10.0 + style.weighted_avg_late_gain * 4.2 + max(style.closer_score, style.deep_closer_score) * 0.16 + level_bonus + performance_bonus * 0.30 + time_bonus * 0.45 - weight_penalty * 0.65)
        stamina = clamp(45.0 + (distance_mean - 1400.0) / 22.0 + max(0.0, 4.0 - avg_finish) * 5.0 + style.weighted_avg_late_gain * 1.6 + level_bonus + performance_bonus * 0.35 - weight_penalty * 0.45)
        consistency = clamp(100.0 - finish_std * 13.0 - avg_margin * 10.0 + level_bonus * 0.7 + performance_bonus * 0.22)
        pedigree_info = getattr(horse, "pedigree_info", None)
        if pedigree_info is None and hasattr(self.provider, "get_pedigree_info"):
            try:
                pedigree_info = self.provider.get_pedigree_info(horse.horse_name)
            except Exception:
                pedigree_info = None
        mud_aptitude, mud_source = estimate_mud_aptitude_with_source(ability_results, pedigree_info)

        return HorseAbility(
            horse_name=horse.horse_name,
            frame=horse.frame,
            horse_number=horse.horse_number,
            early_speed=early_speed,
            stamina=stamina,
            acceleration=acceleration,
            front_runner=style.front_runner_score,
            stalker=style.stalker_score,
            closer=max(style.closer_score, style.deep_closer_score),
            mud_aptitude=mud_aptitude,
            consistency=consistency,
            running_style=style.running_style,
            primary_running_style=style.running_style,
            base_style_profile=style.base_style_profile,
            recent_results=ability_results,
            front_runner_score=style.front_runner_score,
            stalker_score=style.stalker_score,
            closer_score=style.closer_score,
            deep_closer_score=style.deep_closer_score,
            versatile_score=style.versatile_score,
            weighted_avg_first_ratio=style.weighted_avg_first_ratio,
            weighted_avg_mid_ratio=style.weighted_avg_mid_ratio,
            weighted_avg_last_corner_ratio=style.weighted_avg_last_corner_ratio,
            weighted_avg_late_gain=style.weighted_avg_late_gain,
            weighted_avg_position_ratio=style.weighted_avg_position_ratio,
            position_variance=style.position_variance,
            style_sample_size=style.style_sample_size,
            debug_passing_orders=style.debug_passing_orders,
            avg_opponent_strength_score=avg_opponent_strength,
            avg_race_score=avg_race_score,
            popularity_score=avg_popularity_score,
            race_level_score=avg_race_level_score,
            finish_score=avg_finish_score,
            margin_score=avg_margin_score,
            time_score=avg_component_time_score,
            last3f_score=avg_last3f_score,
            horse_ability_score=horse_ability_score,
            race_power=race_power,
            early_aggressiveness=float(tactics_profile["early_aggressiveness"]),
            mid_positioning=float(tactics_profile["mid_positioning"]),
            late_kick_timing=float(tactics_profile["late_kick_timing"]),
            sustain_speed=float(tactics_profile["sustain_speed"]),
            time_reliability=float(dynamics_profile["time_reliability"]),
            recent_time_score=recent_time_score,
            late_kick_score=float(late_kick_profile["late_kick_score"]),
            avg_last3f=late_kick_profile["avg_last3f"],
            best_last3f=late_kick_profile["best_last3f"],
            last3f_consistency=float(late_kick_profile["last3f_consistency"]),
            late_gain_score=float(late_kick_profile["late_gain_score"]),
            early_push_score=float(dynamics_profile["early_push_score"]),
            mid_cruise_score=float(dynamics_profile["mid_cruise_score"]),
            fade_resistance_score=float(dynamics_profile["fade_resistance_score"]),
            sustain_speed_score=float(dynamics_profile["sustain_speed_score"]),
            pace_resilience_score=float(dynamics_profile["pace_resilience_score"]),
            agari_reliability=float(dynamics_profile["agari_reliability"]),
            carried_weight=float(horse.carried_weight),
            weight_penalty=weight_penalty,
            mud_source=mud_source,
            base_ability_score=base_ability_score,
            race_strength_score=avg_race_strength,
            race_strength_adjusted_score=avg_strength_adjusted,
            elo_rating=1500.0,
            normalized_elo_score=neutral_elo_score,
            relative_agari_score=float(late_kick_profile["relative_agari_score"]),
            course_fit_score=course_fit_score,
            jockey=horse.jockey,
            jockey_score=jockey_score_value,
            track_bias_fit_score=track_bias_fit_score,
            pace_fit_score=50.0,
            debug_field_sizes=style.debug_field_sizes,
            debug_first_ratios=style.debug_first_ratios,
            debug_mid_ratios=style.debug_mid_ratios,
            debug_last_corner_ratios=style.debug_last_corner_ratios,
            field_size_warnings=style.field_size_warnings,
        )

    def analyze_many(self, horses: list[HorseEntry]) -> list[HorseAbility]:
        abilities = [self.analyze(horse) for horse in horses]
        elo_rows: list[dict[str, Any]] = []
        for ability in abilities:
            for race in reversed(ability.recent_results):
                if not race.race_id:
                    continue
                elo_rows.append(
                    {
                        "race_id": race.race_id,
                        "horse_name": ability.horse_name,
                        "finish": race.finish_position,
                    }
                )
                raw_entries = (race.raw or {}).get("race_entries") or (race.raw or {}).get("entries")
                if isinstance(raw_entries, list):
                    elo_rows.append({"race_id": race.race_id, "race_entries": raw_entries})
        ratings = update_elo_ratings(elo_rows, {})
        enriched: list[HorseAbility] = []
        for ability in abilities:
            rating = float(ratings.get(ability.horse_name, 1500.0))
            elo_score_value = normalized_elo_score(rating)
            ability_score = clamp(ability.base_ability_score)
            race_power = clamp(ability_score * 0.75 + ability.avg_race_score * 0.25, 40.0, 95.0)
            enriched.append(
                replace(
                    ability,
                    elo_rating=rating,
                    normalized_elo_score=elo_score_value,
                    horse_ability_score=ability_score,
                    race_power=race_power,
                )
            )
        return enriched

    def _resolve_jockey_score(self, horse: HorseEntry) -> float:
        score = float(horse.jockey_score or 50.0)
        if horse.jockey and hasattr(self.provider, "get_jockey_score"):
            try:
                fetched = self.provider.get_jockey_score(horse.jockey)
                if fetched not in (None, ""):
                    score = float(fetched)
            except Exception:
                score = float(horse.jockey_score or 50.0)
        return clamp(score)

    def to_dataframe(self, abilities: list[HorseAbility]) -> pd.DataFrame:
        return pd.DataFrame([ability.to_dict() for ability in abilities])

    def _style_metrics(self, results: list[RaceResult]) -> StyleMetrics:
        snapshots = [snapshot for result in results[:10] if (snapshot := race_style_snapshot(result)) is not None]
        weights = STYLE_WEIGHTS[: len(snapshots)]
        first_ratios = [item.first_ratio for item in snapshots]
        mid_ratios = [item.mid_ratio for item in snapshots]
        last_ratios = [item.last_corner_ratio for item in snapshots]
        position_ratios = [item.position_ratio for item in snapshots]
        late_gains = [item.late_gain for item in snapshots]

        weighted_first = self._weighted_mean(first_ratios, weights, default=0.45)
        weighted_mid = self._weighted_mean(mid_ratios, weights, default=0.45)
        weighted_last = self._weighted_mean(last_ratios, weights, default=0.45)
        weighted_late_gain = self._weighted_mean(late_gains, weights, default=0.0)
        weighted_position = self._weighted_mean(position_ratios, weights, default=0.45)
        position_variance = self._weighted_variance(position_ratios, weights)

        base_profile = create_base_style_profile(results)
        primary_style = primary_style_from_profile(base_profile)
        all_lead_rate = self._weighted_mean([1.0 if item.all_lead else 0.0 for item in snapshots], weights)
        front_like_rate = self._weighted_mean([1.0 if item.front_like else 0.0 for item in snapshots], weights)
        back_like_rate = self._weighted_mean([1.0 if item.back_like else 0.0 for item in snapshots], weights)
        mixed_front_back = front_like_rate >= 0.25 and back_like_rate >= 0.25
        versatile_score = clamp(position_variance * 1600.0 + (28.0 if mixed_front_back else 0.0) + min(front_like_rate, back_like_rate) * 45.0)
        if primary_style == "自在":
            versatile_score = max(versatile_score, 55.0)

        return StyleMetrics(
            running_style=primary_style,
            base_style_profile=base_profile,
            front_runner_score=base_profile["逃げ"] * 100.0,
            stalker_score=base_profile["先行"] * 100.0,
            closer_score=base_profile["差し"] * 100.0,
            deep_closer_score=base_profile["追込"] * 100.0,
            versatile_score=versatile_score,
            weighted_avg_first_ratio=weighted_first,
            weighted_avg_mid_ratio=weighted_mid,
            weighted_avg_last_corner_ratio=weighted_last,
            weighted_avg_late_gain=weighted_late_gain,
            weighted_avg_position_ratio=weighted_position,
            position_variance=position_variance,
            style_sample_size=len(snapshots),
            debug_passing_orders=[item.debug_passing_order for item in snapshots],
            debug_field_sizes=[item.field_size for item in snapshots],
            debug_first_ratios=[item.first_ratio for item in snapshots],
            debug_mid_ratios=[item.mid_ratio for item in snapshots],
            debug_last_corner_ratios=[item.last_corner_ratio for item in snapshots],
            field_size_warnings=[item.field_size_warning for item in snapshots if item.field_size_warning],
        )

    def _performance_score(self, race: RaceResult) -> float:
        return race_score(race)

    def _time_score(self, race: RaceResult) -> float:
        return time_score(race)

    def _expected_speed(self, race: RaceResult) -> float:
        return expected_speed(race)

    def _weighted_mean(self, values: list[float], weights: list[float], default: float = 0.0) -> float:
        if not values:
            return default
        total_weight = sum(weights[: len(values)])
        if total_weight <= 0:
            return default
        return sum(value * weight for value, weight in zip(values, weights)) / total_weight

    def _weighted_variance(self, values: list[float], weights: list[float]) -> float:
        if len(values) <= 1:
            return 0.0
        mean = self._weighted_mean(values, weights)
        total_weight = sum(weights[: len(values)])
        if total_weight <= 0:
            return 0.0
        return sum(weight * (value - mean) ** 2 for value, weight in zip(values, weights)) / total_weight
