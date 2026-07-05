from __future__ import annotations

import math
import random
from typing import Any

import pandas as pd

try:
    from race_trend_scorer import compute_race_trend_match_score
except Exception:  # pragma: no cover - trend scoring is optional in public mode.
    compute_race_trend_match_score = None  # type: ignore[assignment]


ABILITY_SCORE_WEIGHTS = {
    "horse_ability_score": 0.35,
    "race_strength_score": 0.20,
    "normalized_elo_score": 0.20,
    "late_kick_score": 0.15,
    "mud_aptitude": 0.10,
}

SUITABILITY_SCORE_WEIGHTS = {
    "course_fit_score": 0.25,
    "pace_fit_score": 0.20,
    "track_bias_fit_score": 0.20,
    "race_trend_score": 0.15,
    "jockey_score": 0.10,
    "weight_penalty_reversed": 0.10,
}

MARKS = ["◎", "○", "▲", "△", "☆"]
PUBLIC_ENGINE_NAME = "public_two_layer"

PUBLIC_DISPLAY_COLUMNS = [
    "印",
    "予想着順",
    "馬番",
    "馬名",
    "枠順",
    "斤量",
    "脚質",
    "能力スコア",
    "今回条件適性スコア",
    "AIスコア",
    "推定勝率",
    "推定連対率",
    "推定複勝率",
    "能力評価",
    "レースレベル評価",
    "Elo",
    "上り評価",
    "コース適性",
    "展開適性",
    "トラックバイアス適性",
    "過去傾向適性",
    "馬場適性",
    "斤量補正",
    "評価ランク",
    "短評",
]

COMPATIBILITY_COLUMNS = [
    "prediction_score",
    "score",
    "win_rate",
    "top2_rate",
    "top3_rate",
    "horse_number",
    "frame",
    "carried_weight",
    "primary_running_style",
    "actual_running_style",
    "final_prediction_score",
    "prediction_engine",
    "horse_ability_score",
    "race_strength_score",
    "normalized_elo_score",
    "elo_score",
    "late_kick_score",
    "course_fit_score",
    "pace_fit_score",
    "track_bias_fit_score",
    "race_trend_score",
    "jockey_score",
    "weight_penalty",
    "mud_aptitude",
]


def should_use_public_prediction(is_cloud: bool, public_prediction_only: bool = False) -> bool:
    """Return True when Monte Carlo/video/timeline work must be skipped."""
    return bool(is_cloud or public_prediction_only)


def calculate_ability_score(ability: Any, weights: dict[str, Any] | None = None) -> float:
    active = _layer_weights(weights, "ability", ABILITY_SCORE_WEIGHTS)
    values = {
        "horse_ability_score": _feature_value(ability, "horse_ability_score", 50.0),
        "race_strength_score": _feature_value(ability, "race_strength_score", 50.0),
        "normalized_elo_score": _feature_value(ability, "normalized_elo_score", _feature_value(ability, "elo_score", 50.0)),
        "late_kick_score": _feature_value(ability, "late_kick_score", 50.0),
        "mud_aptitude": _feature_value(ability, "mud_aptitude", 50.0),
    }
    return _weighted_score(values, active)


def calculate_race_suitability_score(ability: Any, weights: dict[str, Any] | None = None) -> float:
    active = _layer_weights(weights, "suitability", SUITABILITY_SCORE_WEIGHTS)
    weight_penalty = _feature_value(ability, "weight_penalty", 0.0)
    values = {
        "course_fit_score": _feature_value(ability, "course_fit_score", 50.0),
        "pace_fit_score": _feature_value(ability, "pace_fit_score", 50.0),
        "track_bias_fit_score": _feature_value(ability, "track_bias_fit_score", 50.0),
        "race_trend_score": _feature_value(ability, "race_trend_score", 50.0),
        "jockey_score": _feature_value(ability, "jockey_score", 50.0),
        "weight_penalty_reversed": 100.0 - _clamp(weight_penalty, 0.0, 100.0),
    }
    return _weighted_score(values, active)


def calculate_final_public_score(ability_score: float, race_suitability_score: float) -> float:
    return _clamp(float(ability_score) * 0.65 + float(race_suitability_score) * 0.35)


def estimate_probabilities_from_scores(scores: list[float], temperature: float = 12.0) -> list[dict[str, float]]:
    """Estimate win/top2/top3 probabilities without running simulations."""
    if not scores:
        return []
    temp = max(1e-6, float(temperature))
    values = [float(score) for score in scores]
    max_score = max(values)
    exps = [math.exp((score - max_score) / temp) for score in values]
    total = sum(exps)
    if total <= 0:
        win_rates = [1.0 / len(values) for _ in values]
    else:
        win_rates = [value / total for value in exps]

    order = sorted(range(len(values)), key=lambda index: values[index], reverse=True)
    rank_bonus_by_index = {
        index: max(0.0, (len(values) - rank) / max(1, len(values))) * 0.08
        for rank, index in enumerate(order, start=1)
    }
    probabilities: list[dict[str, float]] = []
    for index, win_rate in enumerate(win_rates):
        rank_bonus = rank_bonus_by_index.get(index, 0.0)
        top2 = _clamp(win_rate * 1.85 + rank_bonus, win_rate, 0.82)
        top3 = _clamp(win_rate * 2.65 + rank_bonus + 0.05, 0.05, 0.85)
        probabilities.append(
            {
                "win_rate": float(win_rate),
                "top2_rate": round(float(max(win_rate, top2)), 6),
                "top3_rate": round(float(_clamp(max(top2, top3), 0.05, 0.85)), 6),
            }
        )
    return probabilities


def assign_evaluation_rank(score: float) -> str:
    value = float(score)
    if value >= 90:
        return "S"
    if value >= 80:
        return "A"
    if value >= 70:
        return "B+"
    if value >= 60:
        return "B"
    if value >= 50:
        return "C"
    return "D"


def build_public_prediction_result(
    race_config: dict[str, Any],
    horses: list[dict[str, Any]],
    abilities: list[Any],
    pace: Any | None = None,
    prediction_weights: dict[str, Any] | None = None,
    trend_analysis: dict[str, Any] | None = None,
    seed: int | None = 42,
    temperature: float = 12.0,
) -> dict[str, Any]:
    """Build a Cloud-safe prediction result without Monte Carlo or timeline generation."""
    rows: list[dict[str, Any]] = []
    horse_lookup = _horse_lookup(horses)
    rng = random.Random(seed)
    for ability in abilities or []:
        ability_map = _to_dict(ability)
        horse_number = int(_feature_value(ability_map, "horse_number", 0))
        horse_meta = horse_lookup.get(horse_number, {})
        trend_features = _trend_features(horse_meta, ability_map, trend_analysis, race_config)
        ability_map.update({key: value for key, value in trend_features.items() if key != "trend_match_comment"})
        ability_map["race_trend_score"] = trend_features.get("race_trend_score", ability_map.get("race_trend_score", 50.0))

        ability_score = calculate_ability_score(ability_map, prediction_weights)
        suitability_score = calculate_race_suitability_score(ability_map, prediction_weights)
        final_score = calculate_final_public_score(ability_score, suitability_score)
        final_score = _clamp(final_score + rng.uniform(-0.35, 0.35))

        primary_style = str(ability_map.get("primary_running_style") or ability_map.get("running_style") or "")
        rows.append(
            {
                "馬番": horse_number,
                "馬名": str(ability_map.get("horse_name") or horse_meta.get("horse_name") or horse_meta.get("馬名") or ""),
                "枠順": int(_feature_value(ability_map, "frame", _feature_value(horse_meta, "frame", 0))),
                "斤量": round(_feature_value(ability_map, "carried_weight", _feature_value(horse_meta, "carried_weight", 56.0)), 1),
                "脚質": primary_style,
                "primary_running_style": primary_style,
                "actual_running_style": primary_style,
                "能力スコア": round(ability_score, 2),
                "今回条件適性スコア": round(suitability_score, 2),
                "AIスコア": round(final_score, 2),
                "能力評価": round(_feature_value(ability_map, "horse_ability_score", 50.0), 2),
                "レースレベル評価": round(_feature_value(ability_map, "race_strength_score", 50.0), 2),
                "Elo": round(_feature_value(ability_map, "normalized_elo_score", _feature_value(ability_map, "elo_score", 50.0)), 2),
                "上り評価": round(_feature_value(ability_map, "late_kick_score", 50.0), 2),
                "コース適性": round(_feature_value(ability_map, "course_fit_score", 50.0), 2),
                "展開適性": round(_feature_value(ability_map, "pace_fit_score", 50.0), 2),
                "トラックバイアス適性": round(_feature_value(ability_map, "track_bias_fit_score", 50.0), 2),
                "過去傾向適性": round(_feature_value(ability_map, "race_trend_score", 50.0), 2),
                "馬場適性": round(_feature_value(ability_map, "mud_aptitude", 50.0), 2),
                "斤量補正": round(_feature_value(ability_map, "weight_penalty", 0.0), 2),
                "評価ランク": assign_evaluation_rank(final_score),
                "prediction_score": round(final_score, 2),
                "score": round(final_score, 2),
                "final_prediction_score": round(final_score, 2),
                "prediction_engine": PUBLIC_ENGINE_NAME,
                "horse_ability_score": round(_feature_value(ability_map, "horse_ability_score", 50.0), 2),
                "race_strength_score": round(_feature_value(ability_map, "race_strength_score", 50.0), 2),
                "normalized_elo_score": round(
                    _feature_value(ability_map, "normalized_elo_score", _feature_value(ability_map, "elo_score", 50.0)), 2
                ),
                "elo_score": round(_feature_value(ability_map, "normalized_elo_score", _feature_value(ability_map, "elo_score", 50.0)), 2),
                "late_kick_score": round(_feature_value(ability_map, "late_kick_score", 50.0), 2),
                "course_fit_score": round(_feature_value(ability_map, "course_fit_score", 50.0), 2),
                "pace_fit_score": round(_feature_value(ability_map, "pace_fit_score", 50.0), 2),
                "track_bias_fit_score": round(_feature_value(ability_map, "track_bias_fit_score", 50.0), 2),
                "race_trend_score": round(_feature_value(ability_map, "race_trend_score", 50.0), 2),
                "jockey_score": round(_feature_value(ability_map, "jockey_score", 50.0), 2),
                "weight_penalty": round(_feature_value(ability_map, "weight_penalty", 0.0), 2),
                "mud_aptitude": round(_feature_value(ability_map, "mud_aptitude", 50.0), 2),
                "carried_weight": round(_feature_value(ability_map, "carried_weight", _feature_value(horse_meta, "carried_weight", 56.0)), 1),
                "frame": int(_feature_value(ability_map, "frame", _feature_value(horse_meta, "frame", 0))),
                "horse_number": horse_number,
                "trend_match_comment": str(trend_features.get("trend_match_comment", "")),
                "jockey": str(ability_map.get("jockey") or horse_meta.get("jockey") or ""),
                "莠域Φ譬ｹ諡": "",
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return _empty_public_result(seed)

    table = table.sort_values(
        ["AIスコア", "能力スコア", "今回条件適性スコア"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    table["予想着順"] = [index + 1 for index in range(len(table))]
    table["印"] = _assign_marks(len(table))
    probabilities = estimate_probabilities_from_scores(table["AIスコア"].astype(float).tolist(), temperature=temperature)
    table["win_rate"] = [item["win_rate"] for item in probabilities]
    table["top2_rate"] = [item["top2_rate"] for item in probabilities]
    table["top3_rate"] = [item["top3_rate"] for item in probabilities]
    table["推定勝率"] = table["win_rate"].map(lambda value: round(float(value) * 100.0, 1))
    table["推定連対率"] = table["top2_rate"].map(lambda value: round(float(value) * 100.0, 1))
    table["推定複勝率"] = table["top3_rate"].map(lambda value: round(float(value) * 100.0, 1))
    table["avg_finish"] = table["予想着順"].astype(float)
    table["median_finish"] = table["予想着順"].astype(int)
    table["worst_finish"] = [min(len(table), rank + 2) for rank in table["予想着順"]]
    table["best_finish"] = [max(1, rank - 2) for rank in table["予想着順"]]
    table["avg_time"] = 0.0
    table["短評"] = table.apply(_public_short_comment, axis=1)
    table["莠域Φ譬ｹ諡"] = table["短評"]

    ordered_columns = [column for column in PUBLIC_DISPLAY_COLUMNS + COMPATIBILITY_COLUMNS if column in table.columns]
    remaining_columns = [column for column in table.columns if column not in ordered_columns]
    table = table[ordered_columns + remaining_columns]

    single_result = _single_result_from_prediction_table(table)
    return {
        "prediction_table": table,
        "simulation_trials": [],
        "representative_trial": {},
        "simulation_logs": ["Public prediction mode: Monte Carlo, representative trial search, timeline and video generation were skipped."],
        "summary": {
            "n_simulations": 0,
            "seed": seed,
            "saved_paths": {},
            "prediction_engine": PUBLIC_ENGINE_NAME,
            "public_prediction_mode": True,
        },
        "prediction_engine": PUBLIC_ENGINE_NAME,
        "public_prediction": True,
        "public_prediction_mode": True,
        "single_result": single_result,
    }


def _empty_public_result(seed: int | None) -> dict[str, Any]:
    return {
        "prediction_table": pd.DataFrame(),
        "simulation_trials": [],
        "representative_trial": {},
        "simulation_logs": ["Public prediction mode could not build rows."],
        "summary": {"n_simulations": 0, "seed": seed, "saved_paths": {}, "prediction_engine": PUBLIC_ENGINE_NAME},
        "prediction_engine": PUBLIC_ENGINE_NAME,
        "public_prediction": True,
        "public_prediction_mode": True,
        "single_result": pd.DataFrame(),
    }


def _single_result_from_prediction_table(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in table.sort_values("予想着順").iterrows():
        rank = int(row["予想着順"])
        gap = round((rank - 1) * 1.4, 3)
        rows.append(
            {
                "着順": rank,
                "馬番": int(row["馬番"]),
                "馬名": str(row["馬名"]),
                "枠順": int(row["枠順"]),
                "actual_running_style": str(row["actual_running_style"]),
                "position_m": round(1000.0 - gap, 3),
                "gap_from_winner": gap,
            }
        )
    return pd.DataFrame(rows)


def _assign_marks(count: int) -> list[str]:
    return [MARKS[index] if index < min(5, count) else "" for index in range(count)]


def _public_short_comment(row: pd.Series) -> str:
    ability = float(row.get("能力スコア", 50.0))
    suitability = float(row.get("今回条件適性スコア", 50.0))
    win_rate = float(row.get("win_rate", 0.0))
    weight_penalty = float(row.get("斤量補正", 0.0))
    trend = float(row.get("過去傾向適性", 50.0))
    style = str(row.get("脚質", ""))
    ability_text = "能力面では上位候補" if ability >= 72 else "能力面では相手候補" if ability >= 60 else "能力面では強調材料がやや薄い"
    condition_bits: list[str] = []
    if suitability >= 70:
        condition_bits.append("今回条件との噛み合いが良い")
    elif suitability <= 48:
        condition_bits.append("今回条件で割引が必要")
    else:
        condition_bits.append("今回条件は標準的")
    if trend >= 65:
        condition_bits.append("過去傾向にも合う")
    risk_bits: list[str] = []
    if win_rate < 0.08:
        risk_bits.append("勝ち切る確率は控えめ")
    if weight_penalty >= 8:
        risk_bits.append("斤量負担")
    if style in {"差し", "追込"} and float(row.get("展開適性", 50.0)) < 50:
        risk_bits.append("展開待ち")
    if not risk_bits:
        risk_bits.append("大きな弱点は少ないが展開次第")
    return f"能力面: {ability_text}。今回条件: {'、'.join(condition_bits)}。リスク: {'、'.join(risk_bits)}。"


def _trend_features(
    horse: dict[str, Any],
    analysis: dict[str, Any],
    trend_analysis: dict[str, Any] | None,
    race_config: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(trend_analysis, dict) or not trend_analysis:
        return {"race_trend_score": 50.0, "trend_match_comment": "過去傾向データなし"}
    race_trends = trend_analysis.get("details") if isinstance(trend_analysis.get("details"), dict) else trend_analysis
    if compute_race_trend_match_score is None:
        return {"race_trend_score": 50.0, "trend_match_comment": "過去傾向スコア計算なし"}
    try:
        return compute_race_trend_match_score(horse, analysis, race_trends, race_config)
    except Exception:
        return {"race_trend_score": 50.0, "trend_match_comment": "過去傾向スコア計算失敗"}


def _layer_weights(weights: dict[str, Any] | None, layer_name: str, defaults: dict[str, float]) -> dict[str, float]:
    source = dict(weights or {})
    nested = source.get(layer_name)
    if isinstance(nested, dict):
        return _normalize_weights({key: _number(nested.get(key, defaults[key])) for key in defaults})

    mapped: dict[str, float] = {}
    for key in defaults:
        if key == "weight_penalty_reversed":
            mapped[key] = _number(source.get("weight_penalty", defaults[key]))
        elif key in source:
            mapped[key] = _number(source[key], defaults[key])
        else:
            mapped[key] = defaults[key]
    return _normalize_weights(mapped)


def _weighted_score(values: dict[str, float], weights: dict[str, float]) -> float:
    score = 0.0
    for key, weight in weights.items():
        score += _clamp(values.get(key, 50.0)) * weight
    return _clamp(score)


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    cleaned = {key: max(0.0, float(value)) for key, value in weights.items()}
    total = sum(cleaned.values())
    if total <= 0:
        return {key: 1.0 / max(1, len(cleaned)) for key in cleaned}
    return {key: value / total for key, value in cleaned.items()}


def _horse_lookup(horses: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    lookup: dict[int, dict[str, Any]] = {}
    for index, horse in enumerate(horses or [], start=1):
        if not isinstance(horse, dict):
            continue
        number = int(_feature_value(horse, "horse_number", _feature_value(horse, "馬番", index)))
        lookup[number] = horse
    return lookup


def _to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "to_dict"):
        try:
            return dict(value.to_dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {}


def _feature_value(source: Any, key: str, default: float = 50.0) -> float:
    if isinstance(source, dict):
        value = source.get(key, default)
    else:
        value = getattr(source, key, default)
    return _number(value, default)


def _number(value: Any, default: float = 50.0) -> float:
    try:
        if value is None:
            return float(default)
        if isinstance(value, str):
            value = value.replace("%", "").replace(",", "").strip()
            if value == "":
                return float(default)
        numeric = float(value)
        if math.isnan(numeric):
            return float(default)
        return numeric
    except (TypeError, ValueError):
        return float(default)


def _clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(upper, float(value)))
