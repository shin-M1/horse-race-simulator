from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from horse_analyzer import parse_passing_order
from race_trend_fetcher import fetch_same_race_history


TREND_DATABASE_DIR = Path("outputs/race_trends")
TREND_DATABASE_COLUMNS = [
    "race_name",
    "year",
    "race_id",
    "venue",
    "surface",
    "distance",
    "track_condition",
    "finish",
    "horse_name",
    "frame",
    "horse_number",
    "age",
    "sex",
    "carried_weight",
    "jockey",
    "previous_jockey",
    "jockey_change_type",
    "running_style",
    "passing_order",
    "fourth_corner_pos",
    "last3f",
    "last3f_rank",
    "previous_race_class",
    "previous_distance",
    "sire",
    "broodmare_sire",
]


def build_same_race_trend_database(
    race_name: str,
    venue: str,
    distance: int,
    years: int = 10,
    race_date: str | None = None,
) -> dict:
    """Fetch and persist same-race history as a reusable trend database.

    Missing years are never filled with synthetic rows. The returned
    `year_status` records which years were actually represented in the saved
    rows, which is useful for Streamlit debug display.
    """
    target_date = str(race_date or date.today().isoformat())
    rows = fetch_same_race_history(race_name, target_date, years=years)
    normalized_rows = [_normalize_database_row(row, race_name, venue, distance) for row in rows]
    represented_years = {int(row.get("year", 0)) for row in normalized_rows if _to_int(row.get("year")) > 0}
    base_year = _parse_year(target_date) or date.today().year
    year_status = [
        {
            "year": base_year - offset,
            "status": "取得済み" if base_year - offset in represented_years else "未取得または結果なし",
            "row_count": sum(1 for row in normalized_rows if _to_int(row.get("year")) == base_year - offset),
        }
        for offset in range(1, max(1, int(years)) + 1)
    ]
    save_paths = save_same_race_trend_database(
        {
            "race_name": race_name,
            "venue": venue,
            "distance": int(distance or 0),
            "race_date": target_date,
            "years": int(years),
            "rows": normalized_rows,
            "year_status": year_status,
        }
    )
    return {
        "race_name": race_name,
        "venue": venue,
        "distance": int(distance or 0),
        "race_date": target_date,
        "years": int(years),
        "rows": normalized_rows,
        "row_count": len(normalized_rows),
        "race_count": len({str(row.get("race_id", "")) for row in normalized_rows if row.get("race_id")}),
        "year_status": year_status,
        "save_paths": save_paths,
    }


def save_same_race_trend_database(trend_database: dict, output_dir: str | Path = TREND_DATABASE_DIR) -> dict[str, str]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    race_name = str(trend_database.get("race_name") or "race")
    venue = str(trend_database.get("venue") or "venue")
    distance = _to_int(trend_database.get("distance"))
    base = _safe_filename(f"{race_name}_{venue}_{distance}")
    json_path = directory / f"{base}.json"
    csv_path = directory / f"{base}.csv"
    json_path.write_text(json.dumps(trend_database, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(trend_database.get("rows", []), columns=TREND_DATABASE_COLUMNS).to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )
    return {"json": str(json_path), "csv": str(csv_path)}


def analyze_same_race_trend_database(trend_database: dict) -> dict:
    rows = [dict(row) for row in trend_database.get("rows", []) if isinstance(row, dict)]
    if not rows:
        return _empty_trend_database_analysis(trend_database)
    frame = pd.DataFrame(rows)
    _prepare_frame(frame)
    frame["in_top3"] = frame["finish"].between(1, 3)
    frame["is_winner"] = frame["finish"].eq(1)
    frame["style_for_trend"] = frame.apply(_style_for_row, axis=1)
    frame["fourth_corner_bucket"] = frame.apply(_fourth_corner_bucket, axis=1)
    frame["weight_bucket"] = frame["carried_weight"].apply(_weight_bucket)
    frame["previous_distance_bucket"] = frame["previous_distance"].apply(_distance_bucket)
    frame["jockey_continuity"] = frame.apply(_jockey_continuity, axis=1)
    frame["bloodline_key"] = frame.apply(_bloodline_key, axis=1)
    frame["agari_bucket"] = _agari_bucket(frame)

    analysis = {
        "frame_trend": _group_trend(frame, "frame"),
        "horse_number_trend": _group_trend(frame, "horse_number"),
        "style_trend": _group_trend(frame, "style_for_trend"),
        "agari_trend": _group_trend(frame, "agari_bucket"),
        "fourth_corner_trend": _group_trend(frame, "fourth_corner_bucket"),
        "age_trend": _group_trend(frame, "age"),
        "sex_trend": _group_trend(frame, "sex"),
        "weight_trend": _group_trend(frame, "weight_bucket"),
        "jockey_continuity_trend": _group_trend(frame, "jockey_continuity"),
        "previous_class_trend": _group_trend(frame, "previous_race_class"),
        "previous_distance_trend": _group_trend(frame, "previous_distance_bucket"),
        "bloodline_trend": _group_trend(frame, "bloodline_key"),
        "track_condition_trend": _group_trend(frame, "track_condition"),
        "summary_bullets": [],
        "details": {
            "sample_size": int(len(frame)),
            "race_count": int(frame["race_id"].nunique()) if "race_id" in frame.columns else 0,
            "year_status": trend_database.get("year_status", []),
            "save_paths": trend_database.get("save_paths", {}),
        },
    }
    analysis["summary_bullets"] = _summary_bullets(analysis)
    return analysis


def _normalize_database_row(row: dict[str, Any], race_name: str, venue: str, distance: int) -> dict[str, Any]:
    source = dict(row)
    passing_order = str(source.get("passing_order") or "")
    fourth_corner = _to_int(source.get("fourth_corner_pos")) or _last_passing_position(passing_order)
    return {
        "race_name": str(source.get("race_name") or race_name),
        "year": _to_int(source.get("year")),
        "race_id": str(source.get("race_id") or ""),
        "venue": str(source.get("venue") or source.get("course") or venue),
        "surface": str(source.get("surface") or ""),
        "distance": _to_int(source.get("distance")) or int(distance or 0),
        "track_condition": str(source.get("track_condition") or ""),
        "finish": _to_int(source.get("finish")),
        "horse_name": str(source.get("horse_name") or ""),
        "frame": _to_int(source.get("frame")),
        "horse_number": _to_int(source.get("horse_number")),
        "age": _to_int(source.get("age")),
        "sex": str(source.get("sex") or ""),
        "carried_weight": _to_float(source.get("carried_weight")),
        "jockey": str(source.get("jockey") or ""),
        "previous_jockey": str(source.get("previous_jockey") or ""),
        "jockey_change_type": str(source.get("jockey_change_type") or source.get("jockey_switch") or ""),
        "running_style": str(source.get("running_style") or ""),
        "passing_order": passing_order,
        "fourth_corner_pos": fourth_corner,
        "last3f": _to_float(source.get("last3f")),
        "last3f_rank": _to_int(source.get("last3f_rank")),
        "previous_race_class": str(source.get("previous_race_class") or ""),
        "previous_distance": _to_int(source.get("previous_distance")),
        "sire": str(source.get("sire") or source.get("sire_line") or ""),
        "broodmare_sire": str(source.get("broodmare_sire") or source.get("broodmare_sire_line") or ""),
    }


def _prepare_frame(frame: pd.DataFrame) -> None:
    for column in ["finish", "frame", "horse_number", "age", "carried_weight", "fourth_corner_pos", "last3f", "last3f_rank", "previous_distance", "distance"]:
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    for column in ["sex", "track_condition", "previous_race_class", "jockey_change_type", "jockey", "previous_jockey", "sire", "broodmare_sire", "running_style", "passing_order"]:
        frame[column] = frame.get(column, pd.Series(dtype=str)).fillna("").astype(str)


def _group_trend(frame: pd.DataFrame, column: str) -> dict[str, dict[str, float]]:
    if column not in frame.columns:
        return {}
    output: dict[str, dict[str, float]] = {}
    work = frame.dropna(subset=[column])
    work = work[work[column].astype(str).str.strip() != ""]
    for value, group in work.groupby(column):
        key = _trend_key(value)
        output[key] = {
            "count": int(len(group)),
            "win_rate": round(float(group["is_winner"].mean()), 4),
            "top3_rate": round(float(group["in_top3"].mean()), 4),
            "score": round(_rate_to_score(float(group["in_top3"].mean()), len(group)), 2),
        }
    return output


def _style_for_row(row: pd.Series) -> str:
    style = str(row.get("running_style", "")).strip()
    if style in {"逃げ", "先行", "差し", "追込", "自在"}:
        return style
    positions = parse_passing_order(str(row.get("passing_order", "")))
    field_size = max(1, _to_int(row.get("field_size")) or 18)
    if not positions:
        return ""
    first_ratio = positions[0] / field_size
    last_ratio = positions[-1] / field_size
    finish = _to_float(row.get("finish"))
    late_gain = positions[-1] - finish if finish > 0 else 0
    if first_ratio <= 0.16 and last_ratio <= 0.22:
        return "逃げ"
    if first_ratio <= 0.40 and last_ratio <= 0.42:
        return "先行"
    if first_ratio >= 0.60 and late_gain >= 3:
        return "追込"
    return "差し"


def _fourth_corner_bucket(row: pd.Series) -> str:
    position = _to_float(row.get("fourth_corner_pos")) or _last_passing_position(row.get("passing_order"))
    field_size = max(1, _to_int(row.get("field_size")) or 18)
    if position <= 0:
        return ""
    ratio = position / field_size
    if ratio <= 0.25:
        return "4角前方"
    if ratio <= 0.50:
        return "4角中団前"
    if ratio <= 0.75:
        return "4角中団後"
    return "4角後方"


def _agari_bucket(frame: pd.DataFrame) -> pd.Series:
    if "last3f_rank" not in frame.columns or frame["last3f_rank"].fillna(0).le(0).all():
        ranks = frame.groupby("race_id")["last3f"].rank(method="min", ascending=True)
    else:
        ranks = frame["last3f_rank"]
    return ranks.apply(lambda value: "上り1位" if _to_int(value) == 1 else "上り2-3位" if 2 <= _to_int(value) <= 3 else "上り4位以下")


def _weight_bucket(value: Any) -> str:
    weight = _to_float(value)
    if weight <= 0:
        return ""
    if weight <= 54:
        return "54kg以下"
    if weight <= 56:
        return "54.5-56kg"
    if weight <= 58:
        return "56.5-58kg"
    return "58.5kg以上"


def _distance_bucket(value: Any) -> str:
    distance = _to_int(value)
    if distance <= 0:
        return ""
    if distance <= 1400:
        return "短距離"
    if distance <= 1800:
        return "マイル前後"
    if distance <= 2200:
        return "中距離"
    return "長距離"


def _jockey_continuity(row: pd.Series) -> str:
    explicit = str(row.get("jockey_change_type", "")).strip()
    if explicit:
        return explicit
    current = str(row.get("jockey", "")).strip()
    previous = str(row.get("previous_jockey", "")).strip()
    if not current or not previous:
        return ""
    return "継続" if current == previous else "乗り替わり"


def _bloodline_key(row: pd.Series) -> str:
    sire = str(row.get("sire", "")).strip()
    broodmare = str(row.get("broodmare_sire", "")).strip()
    return sire or broodmare


def _summary_bullets(analysis: dict[str, Any]) -> list[str]:
    return [
        _best_summary("枠順", analysis.get("frame_trend", {})),
        _best_summary("脚質", analysis.get("style_trend", {})),
        _best_summary("上り", analysis.get("agari_trend", {})),
        _best_summary("騎手", analysis.get("jockey_continuity_trend", {})),
        _best_summary("血統", analysis.get("bloodline_trend", {})),
    ]


def _best_summary(label: str, trend: dict[str, Any]) -> str:
    if not trend:
        return f"{label}傾向はデータ不足"
    key, value = max(trend.items(), key=lambda item: float(item[1].get("top3_rate", 0.0)))
    return f"{label}では{key}の複勝率が{float(value.get('top3_rate', 0.0)):.1%}"


def _empty_trend_database_analysis(trend_database: dict) -> dict:
    keys = [
        "frame_trend",
        "horse_number_trend",
        "style_trend",
        "agari_trend",
        "fourth_corner_trend",
        "age_trend",
        "sex_trend",
        "weight_trend",
        "jockey_continuity_trend",
        "previous_class_trend",
        "previous_distance_trend",
        "bloodline_trend",
        "track_condition_trend",
    ]
    return {
        **{key: {} for key in keys},
        "summary_bullets": ["過去傾向データ不足のため一部中立評価"],
        "details": {
            "sample_size": 0,
            "race_count": 0,
            "year_status": trend_database.get("year_status", []),
            "save_paths": trend_database.get("save_paths", {}),
        },
    }


def _rate_to_score(top3_rate: float, count: int) -> float:
    reliability = min(1.0, max(0, count) / 8.0)
    return max(0.0, min(100.0, 50.0 + (top3_rate * 100.0 - 50.0) * reliability))


def _last_passing_position(value: Any) -> int:
    positions = parse_passing_order(str(value or ""))
    return positions[-1] if positions else 0


def _trend_key(value: Any) -> str:
    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return str(value)


def _safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_") or "race_trend"


def _parse_year(value: str) -> int:
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").year
    except ValueError:
        return 0


def _to_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
