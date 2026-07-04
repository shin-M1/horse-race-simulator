from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import pandas as pd

from horse_analyzer import parse_passing_order


STYLE_ORDER = ["逃げ", "先行", "差し", "追込"]


def analyze_same_race_trends(history: list[dict]) -> dict:
    """Analyze same-race history rows without inventing missing source data."""
    rows = [dict(row) for row in history if isinstance(row, dict)]
    if not rows:
        return _empty_trends()
    frame = pd.DataFrame(rows)
    frame["finish"] = pd.to_numeric(frame.get("finish"), errors="coerce")
    frame["frame"] = pd.to_numeric(frame.get("frame"), errors="coerce")
    frame["horse_number"] = pd.to_numeric(frame.get("horse_number"), errors="coerce")
    frame["popularity"] = pd.to_numeric(frame.get("popularity"), errors="coerce")
    frame["last3f"] = pd.to_numeric(frame.get("last3f"), errors="coerce")
    frame["field_size"] = pd.to_numeric(frame.get("field_size"), errors="coerce").fillna(18)
    frame["in_top3"] = frame["finish"].between(1, 3)
    frame["is_winner"] = frame["finish"].eq(1)
    frame["inferred_style"] = frame.apply(_style_for_row, axis=1)

    frame_bias, frame_details = _frame_bias(frame)
    number_bias, number_details = _number_bias(frame)
    style_bias, style_details = _style_bias(frame)
    popularity_bias, popularity_details = _popularity_bias(frame)
    agari_bias, agari_details = _agari_bias(frame)
    jockey_bias, jockey_details = _jockey_switch_bias(frame)
    bloodline_bias, bloodline_details = _bloodline_bias(frame)

    trend_scores = {
        "inner_advantage": _inner_advantage_score(frame),
        "outer_advantage": _outer_advantage_score(frame),
        "front_advantage": _style_advantage_score(frame, ["逃げ", "先行"]),
        "closer_advantage": _style_advantage_score(frame, ["差し", "追込"]),
        "favorite_reliability": _favorite_reliability_score(frame),
        "agari_importance": _agari_importance_score(frame),
        "jockey_continuity_importance": _jockey_continuity_score(frame),
        "bloodline_mud_importance": _bloodline_mud_score(frame),
    }
    summary = [
        frame_bias,
        style_bias,
        popularity_bias,
        agari_bias,
        jockey_bias,
        bloodline_bias,
    ]
    return {
        "frame_bias": frame_bias,
        "horse_number_bias": number_bias,
        "running_style_bias": style_bias,
        "popularity_bias": popularity_bias,
        "agari_bias": agari_bias,
        "jockey_switch_bias": jockey_bias,
        "bloodline_bias": bloodline_bias,
        "summary_bullets": [item for item in summary if item],
        "trend_scores": {key: round(float(value), 2) for key, value in trend_scores.items()},
        "details": {
            "frame": frame_details,
            "horse_number": number_details,
            "style": style_details,
            "popularity": popularity_details,
            "agari": agari_details,
            "jockey_switch": jockey_details,
            "bloodline": bloodline_details,
            "sample_size": int(len(frame)),
            "race_count": int(frame.get("race_id", pd.Series(dtype=str)).nunique()),
        },
    }


def calculate_race_trend_score(horse_row: dict[str, Any], trend_analysis: dict[str, Any] | None) -> tuple[float, str]:
    """Return a 0-100 score describing how well a horse matches trends."""
    trends = trend_analysis if isinstance(trend_analysis, dict) else {}
    scores = trends.get("trend_scores", {}) if isinstance(trends.get("trend_scores"), dict) else {}
    details = trends.get("details", {}) if isinstance(trends.get("details"), dict) else {}
    if not scores and not details:
        return 50.0, "過去傾向データ不足のため中立評価"

    frame = _to_int(horse_row.get("枠順", horse_row.get("frame")))
    number = _to_int(horse_row.get("馬番", horse_row.get("horse_number")))
    style = str(horse_row.get("actual_running_style") or horse_row.get("primary_running_style") or horse_row.get("脚質") or "")
    late_kick = _to_float(horse_row.get("late_kick_score", horse_row.get("上り性能")), 50.0)
    jockey_switch = str(horse_row.get("jockey_switch") or horse_row.get("騎手継続") or "")

    score_parts: list[float] = []
    comments: list[str] = []
    if frame:
        frame_rate = _top3_rate_from_detail(details.get("frame", {}), frame)
        frame_score = frame_rate * 100.0 if frame_rate is not None else 50.0
        score_parts.append(frame_score)
        if frame_score >= 60:
            comments.append("枠順傾向に合う")
        elif frame_score <= 40:
            comments.append("枠順傾向はやや不利")
    if number:
        number_rate = _top3_rate_from_detail(details.get("horse_number", {}), number)
        number_score = number_rate * 100.0 if number_rate is not None else 50.0
        score_parts.append(number_score)
    style_detail = details.get("style", {})
    if style:
        style_rate = _top3_rate_from_detail(style_detail, style)
        if style_rate is None:
            if style in {"逃げ", "先行"}:
                style_rate = _to_float(scores.get("front_advantage"), 50.0) / 100.0
            elif style in {"差し", "追込"}:
                style_rate = _to_float(scores.get("closer_advantage"), 50.0) / 100.0
        style_score = max(0.0, min(100.0, style_rate * 100.0))
        score_parts.append(style_score)
        if style_score >= 60:
            comments.append(f"{style}傾向が合う")
    agari_importance = _to_float(scores.get("agari_importance"), 50.0)
    agari_score = 50.0 + (late_kick - 50.0) * (agari_importance / 100.0)
    score_parts.append(max(0.0, min(100.0, agari_score)))
    if agari_importance >= 60 and late_kick >= 65:
        comments.append("上り重視傾向に合う")
    if jockey_switch:
        switch_rate = _top3_rate_from_detail(details.get("jockey_switch", {}), jockey_switch)
        if switch_rate is not None:
            score_parts.append(switch_rate * 100.0)
            comments.append("騎手傾向を加味")

    score = sum(score_parts) / max(1, len(score_parts))
    return round(max(0.0, min(100.0, score)), 2), "、".join(comments) or "過去傾向は中立評価"


def apply_race_trend_scores(prediction_table: pd.DataFrame, trend_analysis: dict[str, Any] | None) -> pd.DataFrame:
    table = prediction_table.copy()
    scores: list[float] = []
    comments: list[str] = []
    for _, row in table.iterrows():
        score, comment = calculate_race_trend_score(row.to_dict(), trend_analysis)
        scores.append(score)
        comments.append(comment)
    table["race_trend_score"] = scores
    table["trend_match_comment"] = comments
    return table


def _empty_trends() -> dict:
    return {
        "frame_bias": "過去傾向データが不足しています。",
        "horse_number_bias": "過去傾向データが不足しています。",
        "running_style_bias": "過去傾向データが不足しています。",
        "popularity_bias": "過去傾向データが不足しています。",
        "agari_bias": "過去傾向データが不足しています。",
        "jockey_switch_bias": "過去傾向データが不足しています。",
        "bloodline_bias": "過去傾向データが不足しています。",
        "summary_bullets": ["過去傾向データが不足しているため、コース条件と出走馬構成から簡易推定"],
        "trend_scores": {
            "inner_advantage": 50,
            "outer_advantage": 50,
            "front_advantage": 50,
            "closer_advantage": 50,
            "favorite_reliability": 50,
            "agari_importance": 50,
            "jockey_continuity_importance": 50,
            "bloodline_mud_importance": 50,
        },
        "details": {"sample_size": 0, "race_count": 0},
    }


def _frame_bias(frame: pd.DataFrame) -> tuple[str, dict[str, dict[str, float]]]:
    detail = _group_rates(frame, "frame")
    inner = frame[frame["frame"].between(1, 3)]["in_top3"].mean()
    outer = frame[frame["frame"].between(6, 8)]["in_top3"].mean()
    if pd.isna(inner):
        inner = 0.0
    if pd.isna(outer):
        outer = 0.0
    if inner > outer + 0.04:
        return "内枠の複勝率が高く、内寄り有利の傾向。", detail
    if outer > inner + 0.04:
        return "外枠の複勝率が高く、外寄りの伸びに注意。", detail
    return "枠順の極端な偏りは小さく、フラット寄り。", detail


def _number_bias(frame: pd.DataFrame) -> tuple[str, dict[str, dict[str, float]]]:
    detail = _group_rates(frame, "horse_number")
    top_numbers = sorted(detail.items(), key=lambda item: item[1].get("top3_rate", 0.0), reverse=True)[:3]
    label = "、".join(f"{number}番" for number, _ in top_numbers)
    return f"馬番別では{label or '特定馬番なし'}の好走率を参考にしたい。", detail


def _style_bias(frame: pd.DataFrame) -> tuple[str, dict[str, dict[str, float]]]:
    detail = _group_rates(frame, "inferred_style")
    if not detail:
        return "脚質傾向は判定材料が不足。", detail
    best = max(detail.items(), key=lambda item: item[1].get("top3_rate", 0.0))[0]
    return f"脚質は{best}タイプの複勝率が相対的に高い。", detail


def _popularity_bias(frame: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    favorite = frame[frame["popularity"].eq(1)]
    top3_pop = frame[frame["popularity"].between(1, 3)]
    longshot_top3 = frame[(frame["popularity"] >= 8) & frame["in_top3"]]
    fav_top3 = float(favorite["in_top3"].mean()) if not favorite.empty else 0.0
    top3_stability = float(top3_pop["in_top3"].mean()) if not top3_pop.empty else 0.0
    longshot_rate = len(longshot_top3) / max(1, frame.get("race_id", pd.Series(dtype=str)).nunique())
    if fav_top3 >= 0.60:
        text = "1人気の信頼度が高く、上位人気を素直に評価しやすい。"
    elif longshot_rate >= 0.50:
        text = "人気薄の激走が目立ち、穴馬の拾い上げが重要。"
    else:
        text = "人気傾向は中庸で、能力・展開との組み合わせを重視。"
    return text, {
        "favorite_top3_rate": fav_top3,
        "top3_popularity_stability": top3_stability,
        "longshot_top3_per_race": longshot_rate,
    }


def _agari_bias(frame: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    data = frame.dropna(subset=["last3f"]).copy()
    if data.empty:
        return "上り傾向はデータ不足。", {}
    data["agari_rank"] = data.groupby("race_id")["last3f"].rank(method="min", ascending=True)
    fastest = data[data["agari_rank"].eq(1)]
    fastest_top3 = float(fastest["in_top3"].mean()) if not fastest.empty else 0.0
    agari_top3 = float(data[data["agari_rank"].between(1, 3)]["in_top3"].mean())
    if fastest_top3 >= 0.60:
        text = "上り最速馬の複勝率が高く、末脚性能を重視したい。"
    else:
        text = "上りだけでは決まりにくく、位置取りと持続力も重要。"
    return text, {"fastest_top3_rate": fastest_top3, "agari_top3_rank_top3_rate": agari_top3}


def _jockey_switch_bias(frame: pd.DataFrame) -> tuple[str, dict[str, dict[str, float]]]:
    if "jockey_switch" not in frame.columns or frame["jockey_switch"].astype(str).str.strip().eq("").all():
        return "騎手継続・乗り替わりの傾向はデータ不足。", {}
    detail = _group_rates(frame[frame["jockey_switch"].astype(str).str.strip() != ""], "jockey_switch")
    continue_rate = detail.get("継続", {}).get("top3_rate", 0.0)
    switch_rate = detail.get("乗り替わり", {}).get("top3_rate", 0.0)
    if continue_rate > switch_rate + 0.05:
        return "騎手継続馬の好走率がやや高い。", detail
    if switch_rate > continue_rate + 0.05:
        return "乗り替わり馬の好走例も目立つ。", detail
    return "騎手継続・乗り替わりの差は小さい。", detail


def _bloodline_bias(frame: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    candidates = []
    for key in ["sire_line", "broodmare_sire_line", "sire", "broodmare_sire"]:
        if key in frame.columns:
            candidates.extend(str(value) for value in frame.loc[frame["in_top3"], key].dropna().tolist() if str(value).strip())
    counts = Counter(candidates)
    if not counts:
        return "血統傾向は取得データ不足。", {}
    top = counts.most_common(3)
    label = "、".join(name for name, _ in top)
    return f"好走馬の血統では{label}が目立つ。", {"top_bloodlines": dict(top)}


def _group_rates(frame: pd.DataFrame, key: str) -> dict[str, dict[str, float]]:
    if key not in frame.columns:
        return {}
    output: dict[str, dict[str, float]] = {}
    for value, group in frame.dropna(subset=[key]).groupby(key):
        output[str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)] = {
            "count": int(len(group)),
            "win_rate": float(group["is_winner"].mean()),
            "top3_rate": float(group["in_top3"].mean()),
        }
    return output


def _style_for_row(row: pd.Series) -> str:
    style = str(row.get("running_style", "")).strip()
    if style in STYLE_ORDER:
        return style
    positions = parse_passing_order(str(row.get("passing_order", "")))
    field_size = int(row.get("field_size", 18) or 18)
    if not positions:
        return ""
    first_ratio = positions[0] / max(1, field_size)
    last_ratio = positions[-1] / max(1, field_size)
    finish = _to_float(row.get("finish"), 0.0)
    late_gain = positions[-1] - finish if finish > 0 else 0
    if first_ratio <= 0.16 and last_ratio <= 0.22:
        return "逃げ"
    if first_ratio <= 0.40 and last_ratio <= 0.42:
        return "先行"
    if first_ratio >= 0.60 and late_gain >= 3:
        return "追込"
    return "差し"


def _inner_advantage_score(frame: pd.DataFrame) -> float:
    return _relative_group_score(frame, frame["frame"].between(1, 3), frame["frame"].between(6, 8))


def _outer_advantage_score(frame: pd.DataFrame) -> float:
    return _relative_group_score(frame, frame["frame"].between(6, 8), frame["frame"].between(1, 3))


def _style_advantage_score(frame: pd.DataFrame, styles: list[str]) -> float:
    target = frame[frame["inferred_style"].isin(styles)]
    other = frame[~frame["inferred_style"].isin(styles)]
    target_rate = float(target["in_top3"].mean()) if not target.empty else 0.5
    other_rate = float(other["in_top3"].mean()) if not other.empty else 0.5
    return max(0.0, min(100.0, 50.0 + (target_rate - other_rate) * 100.0))


def _favorite_reliability_score(frame: pd.DataFrame) -> float:
    favorites = frame[frame["popularity"].between(1, 3)]
    if favorites.empty:
        return 50.0
    return max(0.0, min(100.0, float(favorites["in_top3"].mean()) * 100.0))


def _agari_importance_score(frame: pd.DataFrame) -> float:
    data = frame.dropna(subset=["last3f"]).copy()
    if data.empty:
        return 50.0
    data["agari_rank"] = data.groupby("race_id")["last3f"].rank(method="min", ascending=True)
    top_agari = data[data["agari_rank"].between(1, 3)]
    other = data[data["agari_rank"] > 3]
    target_rate = float(top_agari["in_top3"].mean()) if not top_agari.empty else 0.5
    other_rate = float(other["in_top3"].mean()) if not other.empty else 0.5
    return max(0.0, min(100.0, 50.0 + (target_rate - other_rate) * 100.0))


def _jockey_continuity_score(frame: pd.DataFrame) -> float:
    if "jockey_switch" not in frame.columns:
        return 50.0
    cont = frame[frame["jockey_switch"].eq("継続")]
    if cont.empty:
        return 50.0
    return max(0.0, min(100.0, float(cont["in_top3"].mean()) * 100.0))


def _bloodline_mud_score(frame: pd.DataFrame) -> float:
    if "track_condition" not in frame.columns:
        return 50.0
    wet = frame[frame["track_condition"].isin(["稍重", "重", "不良"])]
    if wet.empty:
        return 50.0
    return max(0.0, min(100.0, float(wet["in_top3"].mean()) * 100.0))


def _relative_group_score(frame: pd.DataFrame, target_mask: pd.Series, other_mask: pd.Series) -> float:
    target = frame[target_mask]
    other = frame[other_mask]
    target_rate = float(target["in_top3"].mean()) if not target.empty else 0.5
    other_rate = float(other["in_top3"].mean()) if not other.empty else 0.5
    return max(0.0, min(100.0, 50.0 + (target_rate - other_rate) * 100.0))


def _top3_rate_from_detail(detail: Any, key: Any) -> float | None:
    if not isinstance(detail, dict):
        return None
    value = detail.get(str(key))
    if isinstance(value, dict) and "top3_rate" in value:
        return _to_float(value.get("top3_rate"), 0.0)
    return None


def _to_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
