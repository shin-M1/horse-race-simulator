from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

import pandas as pd

try:
    from horse_analyzer import RaceResult, parse_race_time
except Exception:  # pragma: no cover - imported defensively for app startup.
    RaceResult = Any  # type: ignore[misc,assignment]

    def parse_race_time(value: Any) -> float | None:  # type: ignore[no-redef]
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


JRA_COURSES = {"東京", "中山", "京都", "阪神", "中京", "札幌", "函館", "福島", "新潟", "小倉"}
LOCAL_COURSES = {
    "門別",
    "盛岡",
    "水沢",
    "浦和",
    "船橋",
    "大井",
    "川崎",
    "金沢",
    "笠松",
    "名古屋",
    "園田",
    "姫路",
    "高知",
    "佐賀",
    "帯広",
}
RACE_LEVEL_SCORE = {
    "G1": 95.0,
    "JPN1": 92.0,
    "G2": 88.0,
    "JPN2": 85.0,
    "G3": 82.0,
    "JPN3": 80.0,
    "L": 75.0,
    "OP": 70.0,
    "地方重賞": 68.0,
    "3勝": 62.0,
    "2勝": 55.0,
    "1勝": 48.0,
    "地方一般": 44.0,
    "未勝利": 40.0,
    "新馬": 38.0,
}
RACE_LEVEL_ORDER = ["G1", "Jpn1", "G2", "Jpn2", "G3", "Jpn3", "L", "OP", "地方重賞", "3勝", "2勝", "1勝", "地方一般", "未勝利", "新馬", "不明"]


def field_strength_score(race: Any) -> float:
    """Estimate race strength without using popularity or odds."""
    level = race_level_quality_score(race)
    opponent = average_opponent_score_from_race(race)
    field_size = _field_size_score(race)
    affiliation = racing_affiliation_score(classify_racing_affiliation(race))
    return clamp(level * 0.35 + opponent * 0.30 + affiliation * 0.20 + field_size * 0.15)


def classify_racing_affiliation(race: Any) -> str:
    row = race_row(race)
    course = str(_first_value(row, ["course", "venue", "競馬場"], "") or getattr(race, "course", ""))
    race_class = str(_first_value(row, ["race_class", "class", "grade", "クラス"], "") or getattr(race, "race_class", ""))
    race_name = str(_first_value(row, ["race_name", "name", "race", "レース名"], "") or getattr(race, "race_name", ""))
    organization = str(_first_value(row, ["organization", "affiliation", "host", "開催区分"], ""))
    text = f"{course} {race_class} {race_name} {organization}".upper()

    if any(token.upper() in text for token in ["交流", "JPN", "JBC", "ダートグレード", "NAR交流"]):
        return "地方交流"
    if any(token in text for token in ["中央", "JRA"]) or course in JRA_COURSES:
        return "JRA"
    if course in LOCAL_COURSES or any(token in text for token in ["地方", "NAR", "大井", "船橋", "川崎", "浦和", "園田", "高知"]):
        if any(token.upper() in text for token in ["重賞", "S1", "S2", "S3", "H1", "H2", "H3"]) or any(
            token in text for token in ["G1", "G2", "G3", "JPN1", "JPN2", "JPN3"]
        ):
            return "地方重賞"
        return "地方一般"
    if any(token in text for token in ["G1", "G2", "G3", "OP", "L"]):
        return "JRA"
    return "不明"


def racing_affiliation_score(affiliation: str) -> float:
    return {
        "JRA": 72.0,
        "地方交流": 78.0,
        "地方重賞": 66.0,
        "地方一般": 45.0,
    }.get(str(affiliation), 55.0)


def central_local_adjustment_score(affiliations: list[str], field_strength_values: list[float]) -> float:
    if not affiliations:
        return 0.0
    adjustments = {"JRA": 2.0, "地方交流": 4.0, "地方重賞": -1.0, "地方一般": -6.0, "不明": -1.5}
    raw = sum(adjustments.get(item, -1.5) for item in affiliations) / len(affiliations)
    strength = sum(field_strength_values) / len(field_strength_values) if field_strength_values else 55.0
    if strength >= 72.0:
        raw += 2.0
    elif strength <= 45.0:
        raw -= 2.0
    return round(clamp(raw, -12.0, 8.0), 2)


def race_level_quality_score(race: Any) -> float:
    level = infer_race_level_text(race)
    if level in RACE_LEVEL_SCORE:
        return RACE_LEVEL_SCORE[level]
    normalized = level.upper().replace("Ｊ", "J")
    return RACE_LEVEL_SCORE.get(normalized, 60.0)


def infer_race_level_text(race: Any) -> str:
    row = race_row(race)
    race_class = str(_first_value(row, ["race_class", "class", "grade", "クラス"], "") or getattr(race, "race_class", ""))
    race_name = str(_first_value(row, ["race_name", "name", "race", "レース名"], "") or getattr(race, "race_name", ""))
    text = f"{race_class} {race_name}".upper().replace("Ｊ", "J")
    checks = [
        ("JPN1", "Jpn1"),
        ("JPN2", "Jpn2"),
        ("JPN3", "Jpn3"),
        ("G1", "G1"),
        ("G2", "G2"),
        ("G3", "G3"),
        ("L", "L"),
        ("OP", "OP"),
        ("3勝", "3勝"),
        ("2勝", "2勝"),
        ("1勝", "1勝"),
        ("未勝利", "未勝利"),
        ("新馬", "新馬"),
    ]
    for token, label in checks:
        if token in text:
            return label
    if "重賞" in text:
        return "地方重賞"
    if classify_racing_affiliation(race) == "地方一般":
        return "地方一般"
    return "不明"


def average_opponent_score_from_race(race: Any) -> float:
    row = race_row(race)
    for key in ("average_opponent_score", "avg_opponent_score", "field_average_score", "opponent_strength_score"):
        value = _number(row.get(key), None)
        if value is not None:
            return clamp(value)
    entries = row.get("race_entries") or row.get("entries") or row.get("opponents")
    values: list[float] = []
    if isinstance(entries, list):
        for item in entries:
            if not isinstance(item, dict):
                continue
            for key in ("ability_score", "horse_ability_score", "rating", "elo_score", "race_power"):
                value = _number(item.get(key), None)
                if value is not None:
                    values.append(value if value <= 100 else normalize_elo_score(value))
                    break
    if values:
        return clamp(sum(values) / len(values))
    return 50.0


def compute_track_speed_index(race: Any, peer_races: list[Any] | None = None) -> dict[str, Any]:
    """Return same-day/same-condition time correction data.

    track_speed_index is race_time - same_condition_average_time.
    Negative values mean the horse was faster than the local comparison set.
    """
    time_sec = race_time_seconds(race)
    average_time = same_condition_average_time(race, peer_races)
    if time_sec is None or average_time is None:
        return {
            "track_speed_index": 0.0,
            "track_speed_score": 50.0,
            "track_speed_category": "不明",
            "same_condition_avg_time": None,
        }
    diff = float(time_sec) - float(average_time)
    score = clamp(50.0 - diff * 8.0)
    if diff <= -0.5:
        category = "高速馬場"
    elif diff >= 0.5:
        category = "低速馬場"
    else:
        category = "標準馬場"
    return {
        "track_speed_index": round(diff, 3),
        "track_speed_score": round(score, 2),
        "track_speed_category": category,
        "same_condition_avg_time": round(float(average_time), 3),
    }


def build_prediction_confidence(ability: Any, race_config: dict[str, Any] | None = None, horse_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    race_config = dict(race_config or {})
    horse_meta = dict(horse_meta or {})
    races = recent_races_from_ability(ability)
    reasons: list[str] = []
    score = 100.0

    recent_count = len(races)
    if recent_count < 5:
        penalty = (5 - recent_count) * 8.0
        score -= penalty
        reasons.append("近走データ不足")

    missing_rate = missing_data_rate(races)
    if missing_rate > 0:
        score -= min(25.0, missing_rate * 35.0)
        if missing_rate >= 0.25:
            reasons.append("取得データ欠損が多い")

    current_affiliation = current_race_affiliation(race_config)
    past_affiliations = [classify_racing_affiliation(race) for race in races]
    if current_affiliation == "JRA" and past_affiliations and all(item != "JRA" for item in past_affiliations):
        score -= 15.0
        reasons.append("地方→中央初挑戦/転戦")
    elif current_affiliation.startswith("地方") and any(item == "JRA" for item in past_affiliations):
        score -= 8.0
        reasons.append("中央→地方転戦")

    current_distance = _number(race_config.get("distance"), None)
    if current_distance is not None and races:
        distances = [_number(_race_attr(race, "distance"), None) for race in races]
        known_distances = [value for value in distances if value is not None]
        if known_distances and min(abs(float(current_distance) - value) for value in known_distances) > 250:
            score -= 8.0
            reasons.append("初距離または距離変化が大きい")

    current_course = str(race_config.get("course") or race_config.get("venue") or "")
    if current_course and races and current_course not in [str(_race_attr(race, "course") or _race_attr(race, "venue") or "") for race in races]:
        score -= 6.0
        reasons.append("初コース")

    current_surface = str(race_config.get("surface") or "")
    past_surfaces = [str(_race_attr(race, "surface") or "") for race in races]
    if current_surface and races and current_surface not in past_surfaces:
        score -= 10.0
        reasons.append(f"初{current_surface}")

    layoff_days = days_since_latest_race(races, race_config.get("race_date"))
    if layoff_days is not None and layoff_days >= 180:
        score -= 18.0
        reasons.append("長期休養明け")
    elif layoff_days is not None and layoff_days >= 90:
        score -= 9.0
        reasons.append("休養明け")

    if bool(horse_meta.get("first_blinker") or race_config.get("first_blinker")):
        score -= 6.0
        reasons.append("初ブリンカー")

    current_jockey = str(horse_meta.get("jockey") or _ability_value(ability, "jockey", "") or "")
    if current_jockey:
        past_jockeys = {str(race_row(race).get("jockey", "") or "") for race in races}
        if past_jockeys and current_jockey not in past_jockeys:
            score -= 5.0
            reasons.append("初騎手")

    similar_count = similar_condition_count(races, race_config)
    if similar_count == 0:
        score -= 10.0
        reasons.append("過去類似条件なし")
    elif similar_count < 2:
        score -= 4.0
        reasons.append("過去類似条件が少ない")

    score = clamp(score)
    label = confidence_label(score)
    if not reasons:
        reasons.append("近走と今回条件の整合性が高い")
    return {
        "prediction_confidence_score": round(score, 2),
        "prediction_confidence": label,
        "prediction_confidence_stars": confidence_stars(score),
        "prediction_confidence_reasons": reasons,
        "prediction_confidence_reason": " / ".join(reasons),
        "similar_condition_count": int(similar_count),
        "missing_data_rate": round(float(missing_rate), 3),
    }


def build_shap_like_contributions(
    ability: Any,
    race_config: dict[str, Any] | None = None,
    confidence: dict[str, Any] | None = None,
    peer_races: list[Any] | None = None,
) -> dict[str, float]:
    races = recent_races_from_ability(ability)
    strength_values = [field_strength_score(race) for race in races]
    avg_field_strength = sum(strength_values) / len(strength_values) if strength_values else _ability_value(ability, "race_strength_score", 50.0)
    track_values = [compute_track_speed_index(race, peer_races).get("track_speed_score", 50.0) for race in races]
    track_score = sum(float(value) for value in track_values) / len(track_values) if track_values else 50.0
    affiliations = [classify_racing_affiliation(race) for race in races]
    central_local_bonus = central_local_adjustment_score(affiliations, strength_values)
    confidence_score = float((confidence or {}).get("prediction_confidence_score", 70.0))
    distance_fit = distance_fit_score(races, race_config or {})

    contributions = {
        "能力評価": (_ability_value(ability, "horse_ability_score", 50.0) - 50.0) * 0.34,
        "近走指数": (_ability_value(ability, "race_strength_adjusted_score", _ability_value(ability, "avg_race_score", 50.0)) - 50.0) * 0.17,
        "相手レベル": (avg_field_strength - 50.0) * 0.16,
        "タイム指数": (track_score - 50.0) * 0.13,
        "コース適性": (_ability_value(ability, "course_fit_score", 50.0) - 50.0) * 0.10,
        "距離適性": (distance_fit - 50.0) * 0.07,
        "脚質": (_ability_value(ability, "pace_fit_score", 50.0) - 50.0) * 0.07,
        "騎手": (_ability_value(ability, "jockey_score", 50.0) - 50.0) * 0.035,
        "斤量": -max(0.0, _ability_value(ability, "weight_penalty", 0.0)) * 0.55,
        "中央地方補正": central_local_bonus,
        "信頼度補正": (confidence_score - 70.0) * 0.08,
    }
    final_score = clamp(50.0 + sum(contributions.values()))
    rounded = {key: round(float(value), 2) for key, value in contributions.items()}
    rounded["最終AIスコア"] = round(final_score, 2)
    return rounded


def enhance_prediction_table(
    prediction_table: pd.DataFrame,
    abilities: list[Any] | None,
    race_config: dict[str, Any] | None = None,
    horses: list[dict[str, Any]] | None = None,
    *,
    recalc_score: bool = True,
) -> pd.DataFrame:
    if prediction_table is None or prediction_table.empty:
        return prediction_table
    race_config = dict(race_config or {})
    ability_lookup = ability_lookup_by_number(abilities or [])
    horse_lookup = horse_lookup_by_number(horses or [])
    result = prediction_table.copy()
    rows: list[dict[str, Any]] = []
    for _, row in result.iterrows():
        horse_number = int(_first_row_number(row, ["horse_number", "鬥ｬ逡ｪ"], 0))
        ability = ability_lookup.get(horse_number, row.to_dict())
        horse_meta = horse_lookup.get(horse_number, {})
        races = recent_races_from_ability(ability)
        strength_values = [field_strength_score(race) for race in races]
        avg_strength = sum(strength_values) / len(strength_values) if strength_values else _row_number(row, "race_strength_score", 50.0)
        affiliations = [classify_racing_affiliation(race) for race in races]
        primary_affiliation = most_common(affiliations) if affiliations else "不明"
        central_local_bonus = central_local_adjustment_score(affiliations, strength_values)
        track_profiles = [compute_track_speed_index(race, races) for race in races]
        track_score = sum(float(item["track_speed_score"]) for item in track_profiles) / len(track_profiles) if track_profiles else 50.0
        track_index = sum(float(item["track_speed_index"]) for item in track_profiles) / len(track_profiles) if track_profiles else 0.0
        confidence = build_prediction_confidence(ability, race_config, horse_meta)
        contributions = build_shap_like_contributions(ability, race_config, confidence, races)
        explainable_score = float(contributions["最終AIスコア"])
        existing_score = _row_number(row, "prediction_score", _row_number(row, "score", explainable_score))
        combined_score = clamp(existing_score * 0.55 + explainable_score * 0.45) if recalc_score else existing_score
        highest_grade = highest_race_grade(races)
        avg_level = average_race_level_score(races)
        ai_reason = build_ai_short_comment(
            row=row,
            ability=ability,
            confidence=confidence,
            affiliation=primary_affiliation,
            highest_grade=highest_grade,
            avg_strength=avg_strength,
            contributions=contributions,
        )
        update = {
            "field_strength_score": round(avg_strength, 2),
            "track_speed_index": round(track_index, 3),
            "track_speed_score": round(track_score, 2),
            "track_speed_category": most_common([str(item["track_speed_category"]) for item in track_profiles]) if track_profiles else "不明",
            "所属区分": primary_affiliation,
            "racing_affiliation": primary_affiliation,
            "最高レース格": highest_grade,
            "平均レースレベル": round(avg_level, 2),
            "中央地方補正": round(central_local_bonus, 2),
            "Prediction Confidence": confidence["prediction_confidence"],
            "prediction_confidence": confidence["prediction_confidence"],
            "prediction_confidence_score": confidence["prediction_confidence_score"],
            "prediction_confidence_stars": confidence["prediction_confidence_stars"],
            "prediction_confidence_reason": confidence["prediction_confidence_reason"],
            "similar_condition_count": confidence["similar_condition_count"],
            "missing_data_rate": confidence["missing_data_rate"],
            "contribution_details": contributions,
            "ai_short_comment": ai_reason,
            "予想理由": ai_reason,
            "AI短評": ai_reason,
            "ability_layer_score": round(
                _ability_value(ability, "horse_ability_score", _row_number(row, "horse_ability_score", 50.0)) * 0.55
                + avg_strength * 0.25
                + _ability_value(ability, "late_kick_score", _row_number(row, "late_kick_score", 50.0)) * 0.20,
                2,
            ),
            "suitability_layer_score": round(
                _ability_value(ability, "course_fit_score", _row_number(row, "course_fit_score", 50.0)) * 0.30
                + _ability_value(ability, "pace_fit_score", _row_number(row, "pace_fit_score", 50.0)) * 0.25
                + _ability_value(ability, "mud_aptitude", _row_number(row, "mud_aptitude", 50.0)) * 0.20
                + track_score * 0.25,
                2,
            ),
            "AIスコア": round(combined_score, 2),
            "explainable_ai_score": round(combined_score, 2),
            "prediction_score": round(combined_score, 2),
            "score": round(combined_score, 2),
            "期待値": _expected_value_display(row),
            "勝率": round(_row_number(row, "win_rate", 0.0) * 100.0, 1),
            "連対率": round(_row_number(row, "top2_rate", 0.0) * 100.0, 1),
            "複勝率": round(_row_number(row, "top3_rate", 0.0) * 100.0, 1),
        }
        rows.append(update)
    for column in set().union(*(item.keys() for item in rows)):
        result[column] = [item.get(column) for item in rows]
    sort_columns = [column for column in ["prediction_score", "win_rate", "top3_rate", "avg_finish"] if column in result.columns]
    if sort_columns:
        ascending = [False if column != "avg_finish" else True for column in sort_columns]
        result = result.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)
    else:
        result = result.reset_index(drop=True)
    result["AI順位"] = [index + 1 for index in range(len(result))]
    if "莠域Φ逹鬆・" in result.columns:
        result["莠域Φ逹鬆・"] = result["AI順位"]
    result["蜊ｰ"] = assign_marks(len(result))
    return result


def build_ai_evaluation_table(prediction_table: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "AI順位",
        "印",
        "馬番",
        "馬名",
        "AIスコア",
        "勝率",
        "連対率",
        "複勝率",
        "期待値",
        "Prediction Confidence",
        "prediction_confidence_stars",
        "所属区分",
        "最高レース格",
        "平均レースレベル",
        "中央地方補正",
        "field_strength_score",
        "track_speed_index",
        "track_speed_category",
    ]
    return prediction_table[[column for column in columns if column in prediction_table.columns]].copy()


def build_prediction_reason_table(prediction_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in prediction_table.iterrows():
        rows.append(
            {
                "印": row.get("印", row.get("蜊ｰ", "")),
                "AI順位": row.get("AI順位", ""),
                "馬番": row.get("馬番", row.get("鬥ｬ逡ｪ", row.get("horse_number", ""))),
                "馬名": row.get("馬名", row.get("鬥ｬ蜷・", row.get("horse_name", ""))),
                "AIスコア": row.get("AIスコア", row.get("prediction_score", "")),
                "主な理由": row.get("ai_short_comment", row.get("予想理由", "")),
            }
        )
    return pd.DataFrame(rows)


def build_prediction_confidence_table(prediction_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in prediction_table.iterrows():
        rows.append(
            {
                "AI順位": row.get("AI順位", ""),
                "馬番": row.get("馬番", row.get("鬥ｬ逡ｪ", row.get("horse_number", ""))),
                "馬名": row.get("馬名", row.get("鬥ｬ蜷・", row.get("horse_name", ""))),
                "信頼度": row.get("Prediction Confidence", row.get("prediction_confidence", "")),
                "星": row.get("prediction_confidence_stars", ""),
                "理由": row.get("prediction_confidence_reason", ""),
                "類似条件数": row.get("similar_condition_count", ""),
                "欠損率": row.get("missing_data_rate", ""),
            }
        )
    return pd.DataFrame(rows)


def contribution_table_from_row(row: pd.Series | dict[str, Any]) -> pd.DataFrame:
    data = row.get("contribution_details", {}) if isinstance(row, dict) else row.get("contribution_details", {})
    if not isinstance(data, dict):
        return pd.DataFrame()
    rows = [{"項目": key, "寄与度": value} for key, value in data.items() if key != "最終AIスコア"]
    rows.append({"項目": "最終AIスコア", "寄与度": data.get("最終AIスコア", "")})
    return pd.DataFrame(rows)


def walk_forward_validation_splits(records: list[dict[str, Any]] | pd.DataFrame) -> list[dict[str, Any]]:
    frame = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    if frame.empty or "race_date" not in frame.columns:
        return []
    frame = frame.copy()
    frame["_race_date"] = pd.to_datetime(frame["race_date"], errors="coerce")
    frame = frame.dropna(subset=["_race_date"]).sort_values("_race_date")
    years = sorted(int(year) for year in frame["_race_date"].dt.year.unique())
    splits: list[dict[str, Any]] = []
    for validation_year in years[1:]:
        train = frame[frame["_race_date"].dt.year < validation_year]
        validation = frame[frame["_race_date"].dt.year == validation_year]
        if train.empty or validation.empty:
            continue
        splits.append(
            {
                "train_end_year": validation_year - 1,
                "validation_year": validation_year,
                "train_count": int(len(train)),
                "validation_count": int(len(validation)),
                "max_train_date": train["_race_date"].max().date().isoformat(),
                "min_validation_date": validation["_race_date"].min().date().isoformat(),
                "leakage_free": bool(train["_race_date"].max() < validation["_race_date"].min()),
            }
        )
    return splits


def race_row(race: Any) -> dict[str, Any]:
    if isinstance(race, dict):
        return race
    raw = getattr(race, "raw", None)
    data: dict[str, Any] = dict(raw or {}) if isinstance(raw, dict) else {}
    for key in ("race_name", "distance", "surface", "track_condition", "finish_position", "margin", "final_3f", "race_time_seconds", "field_size", "race_class", "date", "course", "race_id", "passing_order"):
        if hasattr(race, key):
            data.setdefault(key, getattr(race, key))
    return data


def recent_races_from_ability(ability: Any) -> list[Any]:
    if isinstance(ability, dict):
        races = ability.get("recent_results") or ability.get("recent_races") or []
    else:
        races = getattr(ability, "recent_results", None) or getattr(ability, "recent_races", None) or []
    return list(races or [])[:5]


def ability_lookup_by_number(abilities: list[Any]) -> dict[int, Any]:
    lookup: dict[int, Any] = {}
    for ability in abilities:
        number = int(_ability_value(ability, "horse_number", _ability_value(ability, "鬥ｬ逡ｪ", 0)))
        if number:
            lookup[number] = ability
    return lookup


def horse_lookup_by_number(horses: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    lookup: dict[int, dict[str, Any]] = {}
    for horse in horses:
        if not isinstance(horse, dict):
            continue
        number = int(_number(horse.get("horse_number", horse.get("鬥ｬ逡ｪ", 0)), 0) or 0)
        if number:
            lookup[number] = horse
    return lookup


def average_race_level_score(races: list[Any]) -> float:
    if not races:
        return 60.0
    return round(sum(race_level_quality_score(race) for race in races) / len(races), 2)


def highest_race_grade(races: list[Any]) -> str:
    if not races:
        return "不明"
    levels = [infer_race_level_text(race) for race in races]
    order = {level.upper(): index for index, level in enumerate(RACE_LEVEL_ORDER)}
    return min(levels, key=lambda level: order.get(level.upper(), 999))


def current_race_affiliation(race_config: dict[str, Any]) -> str:
    course = str(race_config.get("course") or race_config.get("venue") or "")
    if course in JRA_COURSES:
        return "JRA"
    if course in LOCAL_COURSES:
        return "地方一般"
    text = str(race_config.get("affiliation") or race_config.get("organization") or "")
    if "中央" in text or "JRA" in text.upper():
        return "JRA"
    if "地方" in text or "NAR" in text.upper():
        return "地方一般"
    return "不明"


def missing_data_rate(races: list[Any]) -> float:
    if not races:
        return 1.0
    keys = ["race_time_seconds", "time", "passing_order", "final_3f", "last3f", "field_size", "race_class"]
    total = len(races) * 5
    missing = 0
    for race in races:
        row = race_row(race)
        checks = [
            race_time_seconds(race),
            _first_value(row, ["passing_order", "passing", "corner_order"], None),
            _first_value(row, ["final_3f", "last3f", "agari", "closing_3f"], None),
            _first_value(row, ["field_size", "runners", "number_of_runners"], None),
            _first_value(row, ["race_class", "class", "grade"], None),
        ]
        missing += sum(1 for item in checks if item in (None, ""))
    return missing / max(1, total)


def similar_condition_count(races: list[Any], race_config: dict[str, Any]) -> int:
    target_surface = str(race_config.get("surface") or "")
    target_course = str(race_config.get("course") or race_config.get("venue") or "")
    target_distance = _number(race_config.get("distance"), None)
    count = 0
    for race in races:
        surface_ok = not target_surface or str(_race_attr(race, "surface") or "") == target_surface
        course_ok = not target_course or str(_race_attr(race, "course") or _race_attr(race, "venue") or "") == target_course
        distance = _number(_race_attr(race, "distance"), None)
        distance_ok = target_distance is None or distance is None or abs(distance - target_distance) <= 200
        if surface_ok and distance_ok and (course_ok or not target_course):
            count += 1
    return count


def days_since_latest_race(races: list[Any], race_date_value: Any) -> int | None:
    dates = [_parse_date(_race_attr(race, "date")) for race in races]
    dates = [item for item in dates if item is not None]
    if not dates:
        return None
    target = _parse_date(race_date_value) or date.today()
    return max(0, (target - max(dates)).days)


def distance_fit_score(races: list[Any], race_config: dict[str, Any]) -> float:
    target = _number(race_config.get("distance"), None)
    if target is None or not races:
        return 50.0
    distances = [_number(_race_attr(race, "distance"), None) for race in races]
    known = [value for value in distances if value is not None]
    if not known:
        return 50.0
    best_diff = min(abs(float(target) - value) for value in known)
    return clamp(85.0 - best_diff / 12.0, 35.0, 90.0)


def race_time_seconds(race: Any) -> float | None:
    row = race_row(race)
    value = _first_value(row, ["race_time_seconds", "time_sec", "time", "race_time", "result_time", "走破タイム", "タイム"], None)
    return parse_race_time(value)


def same_condition_average_time(race: Any, peer_races: list[Any] | None) -> float | None:
    row = race_row(race)
    for key in ("same_condition_avg_time", "same_day_condition_avg_time", "track_avg_time", "condition_avg_time", "average_time"):
        value = _number(row.get(key), None)
        if value is not None and value > 0:
            return value
    if not peer_races:
        return None
    target_date = str(_race_attr(race, "date") or "")
    target_course = str(_race_attr(race, "course") or _race_attr(race, "venue") or "")
    target_distance = _number(_race_attr(race, "distance"), None)
    target_surface = str(_race_attr(race, "surface") or "")
    times: list[float] = []
    for peer in peer_races:
        if peer is race:
            continue
        if target_date and str(_race_attr(peer, "date") or "") != target_date:
            continue
        if target_course and str(_race_attr(peer, "course") or _race_attr(peer, "venue") or "") != target_course:
            continue
        if target_surface and str(_race_attr(peer, "surface") or "") != target_surface:
            continue
        peer_distance = _number(_race_attr(peer, "distance"), None)
        if target_distance is not None and peer_distance is not None and abs(peer_distance - target_distance) > 50:
            continue
        peer_time = race_time_seconds(peer)
        if peer_time is not None and peer_time > 0:
            times.append(peer_time)
    if times:
        return sum(times) / len(times)
    return None


def build_ai_short_comment(
    row: pd.Series,
    ability: Any,
    confidence: dict[str, Any],
    affiliation: str,
    highest_grade: str,
    avg_strength: float,
    contributions: dict[str, float],
) -> str:
    name = str(row.get("鬥ｬ蜷・", row.get("horse_name", "")))
    score = float(contributions.get("最終AIスコア", row.get("prediction_score", 50.0)))
    confidence_label_value = str(confidence.get("prediction_confidence", "Medium"))
    lead = f"{name}は" if name else ""
    if affiliation == "地方交流":
        base = "地方交流レベルで相手関係の価値があり"
    elif affiliation == "地方一般":
        base = "地方一般戦中心で相手比較の確認が必要だが"
    elif affiliation == "地方重賞":
        base = "地方重賞での実績を評価でき"
    else:
        base = f"{highest_grade}級の実績を評価でき"
    if score >= 72:
        judgment = "今回も能力上位の評価です"
    elif score >= 60:
        judgment = "上位争いに加わる余地があります"
    else:
        judgment = "強調材料は限定的です"
    reason = max(
        [(key, value) for key, value in contributions.items() if key != "最終AIスコア"],
        key=lambda item: item[1],
        default=("能力評価", 0.0),
    )[0]
    risk = "" if confidence_label_value in {"High", "Medium"} else " ただし信頼度が低く、条件替わりの不確実性があります。"
    return f"{lead}{base}、{reason}がスコアを押し上げています。{judgment}。{risk}"


def confidence_label(score: float) -> str:
    if score >= 80:
        return "High"
    if score >= 65:
        return "Medium"
    if score >= 50:
        return "Low"
    return "Very Low"


def confidence_stars(score: float) -> str:
    if score >= 85:
        count = 5
    elif score >= 70:
        count = 4
    elif score >= 55:
        count = 3
    elif score >= 40:
        count = 2
    else:
        count = 1
    return "★" * count + "☆" * (5 - count)


def assign_marks(count: int) -> list[str]:
    marks = ["◎", "○", "▲", "△", "☆"]
    return [marks[index] if index < min(5, count) else "" for index in range(count)]


def normalize_elo_score(rating: float) -> float:
    return clamp(50.0 + (float(rating) - 1500.0) / 10.0)


def most_common(values: list[str]) -> str:
    if not values:
        return ""
    return max(sorted(set(values)), key=values.count)


def clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return lower
    return max(lower, min(upper, float(value)))


def _field_size_score(race: Any) -> float:
    field_size = _number(_race_attr(race, "field_size"), 18.0)
    return clamp(float(field_size or 18.0) / 18.0 * 100.0)


def _race_attr(race: Any, key: str) -> Any:
    if isinstance(race, dict):
        return race.get(key)
    if hasattr(race, key):
        return getattr(race, key)
    raw = getattr(race, "raw", None)
    if isinstance(raw, dict):
        return raw.get(key)
    return None


def _first_value(row: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return default


def _number(value: Any, default: float | None = 50.0) -> float | None:
    try:
        if value is None or value == "":
            return default
        if isinstance(value, str):
            value = value.replace(",", "").replace("%", "").strip()
            if value == "":
                return default
        numeric = float(value)
        if math.isnan(numeric):
            return default
        return numeric
    except (TypeError, ValueError):
        return default


def _ability_value(ability: Any, key: str, default: Any = 50.0) -> Any:
    if isinstance(ability, dict):
        return _number(ability.get(key), default) if isinstance(default, (int, float)) else ability.get(key, default)
    value = getattr(ability, key, default)
    if isinstance(default, (int, float)):
        return _number(value, default)
    return value


def _row_number(row: pd.Series, key: str, default: float = 50.0) -> float:
    return float(_number(row.get(key), default) or default)


def _first_row_number(row: pd.Series, keys: list[str], default: int = 0) -> int:
    for key in keys:
        value = _number(row.get(key), None)
        if value is not None:
            return int(value)
    return int(default)


def _expected_value_display(row: pd.Series) -> str:
    odds = _number(row.get("odds"), None)
    win_rate = _row_number(row, "win_rate", 0.0)
    if odds is None or odds <= 0:
        return ""
    return f"{win_rate * odds * 100.0:.1f}"


def _parse_date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip().replace("/", "-")
    try:
        return datetime.fromisoformat(text[:10]).date()
    except ValueError:
        return None
