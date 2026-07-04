from __future__ import annotations

from typing import Any


TREND_SCORE_WEIGHTS = {
    "frame_trend_score": 0.12,
    "style_trend_score": 0.18,
    "agari_trend_score": 0.18,
    "fourth_corner_trend_score": 0.12,
    "age_trend_score": 0.08,
    "weight_trend_score": 0.08,
    "jockey_continuity_score": 0.08,
    "previous_race_trend_score": 0.08,
    "bloodline_trend_score": 0.08,
}


def compute_race_trend_match_score(
    horse: dict,
    horse_analysis: dict,
    race_trends: dict,
    race_config: dict,
) -> dict:
    """Score how well a horse matches same-race historical trends.

    The function treats missing trend buckets as neutral 50 rather than
    fabricating any evidence.
    """
    horse = dict(horse or {})
    analysis = dict(horse_analysis or {})
    trends = dict(race_trends or {})
    neutral_reasons: list[str] = []

    frame = _integer(_pick(horse, analysis, ["枠順", "frame"]))
    horse_number = _integer(_pick(horse, analysis, ["馬番", "horse_number"]))
    style = str(_pick(horse, analysis, ["actual_running_style", "primary_running_style", "脚質"], ""))
    age = _integer(_pick(horse, analysis, ["age", "馬齢"]))
    weight = _number(_pick(horse, analysis, ["斤量", "carried_weight"]))
    jockey_change = str(_pick(horse, analysis, ["jockey_change_type", "jockey_switch", "騎手継続"], ""))
    previous_class = str(_pick(horse, analysis, ["previous_race_class", "前走クラス"], ""))
    previous_distance = _integer(_pick(horse, analysis, ["previous_distance", "前走距離"]))
    sire = str(_pick(horse, analysis, ["sire", "sire_line", "父"], ""))
    broodmare_sire = str(_pick(horse, analysis, ["broodmare_sire", "broodmare_sire_line", "母父"], ""))
    last_corner_ratio = _number(_pick(horse, analysis, ["weighted_avg_last_corner_ratio", "avg_last_corner_ratio"]))
    late_kick_score = _number(_pick(horse, analysis, ["late_kick_score", "last3f_score"], 50.0), 50.0)

    frame_score = _trend_score(trends.get("frame_trend"), frame, neutral_reasons, "枠順")
    horse_number_score = _trend_score(trends.get("horse_number_trend"), horse_number, neutral_reasons, "馬番")
    style_score = _trend_score(trends.get("style_trend"), style, neutral_reasons, "脚質")
    agari_score = _agari_score(trends.get("agari_trend"), late_kick_score, neutral_reasons)
    fourth_corner_score = _trend_score(
        trends.get("fourth_corner_trend"),
        _fourth_corner_bucket(last_corner_ratio),
        neutral_reasons,
        "4角位置",
    )
    age_score = _trend_score(trends.get("age_trend"), age, neutral_reasons, "馬齢")
    weight_score = _trend_score(trends.get("weight_trend"), _weight_bucket(weight), neutral_reasons, "斤量")
    jockey_score = _trend_score(
        trends.get("jockey_continuity_trend"),
        jockey_change,
        neutral_reasons,
        "騎手継続",
    )
    previous_class_score = _trend_score(
        trends.get("previous_class_trend"),
        previous_class,
        neutral_reasons,
        "前走クラス",
    )
    previous_distance_score = _trend_score(
        trends.get("previous_distance_trend"),
        _distance_bucket(previous_distance),
        neutral_reasons,
        "前走距離",
    )
    previous_race_score = (previous_class_score + previous_distance_score) / 2.0
    bloodline_score = _best_of(
        [
            _trend_score(trends.get("bloodline_trend"), sire, [], "血統"),
            _trend_score(trends.get("bloodline_trend"), broodmare_sire, [], "血統"),
        ],
        neutral_reasons,
        "血統",
    )

    race_trend_score = (
        frame_score * TREND_SCORE_WEIGHTS["frame_trend_score"]
        + style_score * TREND_SCORE_WEIGHTS["style_trend_score"]
        + agari_score * TREND_SCORE_WEIGHTS["agari_trend_score"]
        + fourth_corner_score * TREND_SCORE_WEIGHTS["fourth_corner_trend_score"]
        + age_score * TREND_SCORE_WEIGHTS["age_trend_score"]
        + weight_score * TREND_SCORE_WEIGHTS["weight_trend_score"]
        + jockey_score * TREND_SCORE_WEIGHTS["jockey_continuity_score"]
        + previous_race_score * TREND_SCORE_WEIGHTS["previous_race_trend_score"]
        + bloodline_score * TREND_SCORE_WEIGHTS["bloodline_trend_score"]
    )
    result = {
        "race_trend_score": round(_clamp(race_trend_score), 2),
        "frame_trend_score": round(_clamp(frame_score), 2),
        "horse_number_trend_score": round(_clamp(horse_number_score), 2),
        "style_trend_score": round(_clamp(style_score), 2),
        "agari_trend_score": round(_clamp(agari_score), 2),
        "fourth_corner_trend_score": round(_clamp(fourth_corner_score), 2),
        "age_trend_score": round(_clamp(age_score), 2),
        "weight_trend_score": round(_clamp(weight_score), 2),
        "jockey_continuity_score": round(_clamp(jockey_score), 2),
        "previous_race_trend_score": round(_clamp(previous_race_score), 2),
        "bloodline_trend_score": round(_clamp(bloodline_score), 2),
        "trend_match_comment": _comment(
            frame_score=frame_score,
            style_score=style_score,
            agari_score=agari_score,
            fourth_corner_score=fourth_corner_score,
            bloodline_score=bloodline_score,
            neutral_reasons=neutral_reasons,
        ),
    }
    return result


def _trend_score(trend: Any, key: Any, neutral_reasons: list[str], label: str) -> float:
    if key in (None, "", 0):
        neutral_reasons.append(label)
        return 50.0
    if not isinstance(trend, dict) or not trend:
        neutral_reasons.append(label)
        return 50.0
    value = trend.get(str(key))
    if not isinstance(value, dict):
        neutral_reasons.append(label)
        return 50.0
    if "score" in value:
        return _number(value.get("score"), 50.0)
    if "top3_rate" in value:
        return _clamp(_number(value.get("top3_rate"), 0.5) * 100.0)
    neutral_reasons.append(label)
    return 50.0


def _agari_score(trend: Any, late_kick_score: float, neutral_reasons: list[str]) -> float:
    if not isinstance(trend, dict) or not trend:
        neutral_reasons.append("上り")
        return 50.0
    top_score = max(
        _trend_score(trend, "上り1位", [], "上り"),
        _trend_score(trend, "上り2-3位", [], "上り"),
    )
    importance = _clamp(top_score - 50.0, 0.0, 50.0) / 50.0
    return _clamp(50.0 + (late_kick_score - 50.0) * (0.35 + 0.65 * importance))


def _best_of(scores: list[float], neutral_reasons: list[str], label: str) -> float:
    valid = [score for score in scores if score != 50.0]
    if not valid:
        neutral_reasons.append(label)
        return 50.0
    return max(valid)


def _comment(**values: Any) -> str:
    strengths: list[str] = []
    if values["frame_score"] >= 62:
        strengths.append("枠順傾向に合う")
    if values["style_score"] >= 62:
        strengths.append("脚質傾向に合う")
    if values["agari_score"] >= 62:
        strengths.append("上り傾向に合う")
    if values["fourth_corner_score"] >= 62:
        strengths.append("4角位置傾向に合う")
    if values["bloodline_score"] >= 62:
        strengths.append("血統傾向に合う")
    if not strengths:
        strengths.append("過去傾向は中立評価")
    if values["neutral_reasons"]:
        strengths.append("過去傾向データ不足のため一部中立評価")
    return "、".join(dict.fromkeys(strengths))


def _fourth_corner_bucket(last_corner_ratio: float) -> str:
    if last_corner_ratio <= 0:
        return ""
    if last_corner_ratio <= 0.25:
        return "4角前方"
    if last_corner_ratio <= 0.50:
        return "4角中団前"
    if last_corner_ratio <= 0.75:
        return "4角中団後"
    return "4角後方"


def _weight_bucket(weight: float) -> str:
    if weight <= 0:
        return ""
    if weight <= 54:
        return "54kg以下"
    if weight <= 56:
        return "54.5-56kg"
    if weight <= 58:
        return "56.5-58kg"
    return "58.5kg以上"


def _distance_bucket(distance: int) -> str:
    if distance <= 0:
        return ""
    if distance <= 1400:
        return "短距離"
    if distance <= 1800:
        return "マイル前後"
    if distance <= 2200:
        return "中距離"
    return "長距離"


def _pick(horse: dict, analysis: dict, keys: list[str], default: Any = "") -> Any:
    for key in keys:
        value = horse.get(key)
        if value not in (None, ""):
            return value
        value = analysis.get(key)
        if value not in (None, ""):
            return value
    return default


def _integer(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, float(value)))
