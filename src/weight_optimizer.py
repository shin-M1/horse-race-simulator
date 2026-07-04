from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_WEIGHTS = {
    "horse_ability_score": 0.25,
    "race_strength_score": 0.15,
    "elo_score": 0.15,
    "late_kick_score": 0.15,
    "course_fit_score": 0.10,
    "pace_fit_score": 0.10,
    "jockey_score": 0.05,
    "track_bias_fit_score": 0.05,
    "race_trend_score": 0.10,
}
WEIGHTS_PATH = Path("outputs/model_weights.json")
TRAINING_COLUMNS = [
    "race_id",
    "race_name",
    "race_date",
    "horse_number",
    "horse_name",
    "finish",
    "is_win",
    "is_top3",
    "prediction_score",
    "horse_ability_score",
    "race_strength_score",
    "elo_score",
    "late_kick_score",
    "course_fit_score",
    "pace_fit_score",
    "jockey_score",
    "track_bias_fit_score",
    "race_trend_score",
    "weight_penalty",
    "mud_aptitude",
    "finish_score",
    "margin_score",
    "time_score",
    "last3f_score",
    "carried_weight",
    "frame",
]


def build_training_dataset(evaluation_logs: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for log_index, log in enumerate(evaluation_logs):
        actual_by_number = {
            _integer(item.get("horse_number", item.get("馬番"))): _integer(item.get("finish", item.get("着順")))
            for item in (log.get("actual_result", []) or [])
        }
        prediction_rows = log.get("prediction_table", []) or log.get("prediction_log", {}).get("prediction_table", []) or []
        for prediction in prediction_rows:
            horse_number = _integer(prediction.get("馬番", prediction.get("horse_number")))
            finish = actual_by_number.get(horse_number)
            if horse_number <= 0 or finish is None or finish <= 0:
                continue
            row = {
                "race_id": str(log.get("race_id") or f"manual_{log_index}"),
                "race_name": str(log.get("race_name", "")),
                "race_date": str(log.get("race_date", "")),
                "horse_number": horse_number,
                "horse_name": str(prediction.get("馬名", prediction.get("horse_name", ""))),
                "finish": finish,
                "is_win": int(finish == 1),
                "is_top3": int(finish <= 3),
                "prediction_score": _number(prediction.get("prediction_score", prediction.get("score", 50.0))),
                "horse_ability_score": _number(prediction.get("horse_ability_score", prediction.get("race_power", 50.0))),
                "race_strength_score": _number(prediction.get("race_strength_score", 50.0)),
                "elo_score": _number(prediction.get("elo_score", prediction.get("normalized_elo_score", 50.0))),
                "late_kick_score": _number(prediction.get("late_kick_score", 50.0)),
                "course_fit_score": _number(prediction.get("course_fit_score", 50.0)),
                "pace_fit_score": _number(prediction.get("pace_fit_score", 50.0)),
                "jockey_score": _number(prediction.get("jockey_score", 50.0)),
                "track_bias_fit_score": _number(prediction.get("track_bias_fit_score", 50.0)),
                "race_trend_score": _number(prediction.get("race_trend_score", 50.0)),
                "weight_penalty": _number(prediction.get("weight_penalty", 0.0)),
                "mud_aptitude": _number(prediction.get("mud_aptitude", 50.0)),
                "finish_score": _number(prediction.get("finish_score", 50.0)),
                "margin_score": _number(prediction.get("margin_score", 50.0)),
                "time_score": _number(prediction.get("time_score", 50.0)),
                "last3f_score": _number(prediction.get("last3f_score", 50.0)),
                "carried_weight": _number(prediction.get("carried_weight", prediction.get("斤量", 56.0))),
                "frame": _integer(prediction.get("frame", prediction.get("枠順", 0))),
            }
            rows.append(row)
    return pd.DataFrame(rows, columns=TRAINING_COLUMNS)


def optimize_prediction_weights(
    training_df: pd.DataFrame,
    metric: str = "top3_hit_rate",
    n_trials: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    if training_df.empty:
        raise ValueError("training_df is empty")
    if metric not in {"top3_hit_rate", "winner_hit_rate", "brier_score", "log_loss"}:
        raise ValueError(f"unsupported metric: {metric}")
    n_trials = max(1, int(n_trials))
    rng = random.Random(seed)
    baseline = normalize_weights(DEFAULT_WEIGHTS)
    baseline_score = _evaluate_weights(training_df, baseline, metric)
    best_weights = baseline
    best_score = baseline_score
    keys = list(DEFAULT_WEIGHTS)

    for trial_index in range(n_trials):
        if trial_index % 2 == 0:
            values = [max(0.001, DEFAULT_WEIGHTS[key] * rng.uniform(0.35, 1.85)) for key in keys]
        else:
            values = [rng.gammavariate(1.3, 1.0) for _ in keys]
        total = sum(values)
        candidate = {key: value / total for key, value in zip(keys, values)}
        score = _evaluate_weights(training_df, candidate, metric)
        if score > best_score:
            best_score = score
            best_weights = candidate

    return {
        "weights": {key: round(value, 6) for key, value in best_weights.items()},
        "metric": metric,
        "score": round(float(best_score), 6),
        "baseline_score": round(float(baseline_score), 6),
        "n_trials": n_trials,
        "training_rows": len(training_df),
        "race_count": int(training_df["race_id"].nunique()),
    }


def save_model_weights(
    weights_or_result: dict[str, Any],
    path: str | Path = WEIGHTS_PATH,
) -> Path:
    weights = weights_or_result.get("weights", weights_or_result)
    payload = {
        "weights": normalize_weights(weights),
        "metric": weights_or_result.get("metric", "manual"),
        "score": weights_or_result.get("score"),
        "baseline_score": weights_or_result.get("baseline_score"),
        "training_rows": weights_or_result.get("training_rows"),
        "race_count": weights_or_result.get("race_count"),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def load_model_weights(path: str | Path = WEIGHTS_PATH) -> dict[str, float]:
    target = Path(path)
    if not target.is_file():
        return dict(DEFAULT_WEIGHTS)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        weights = payload.get("weights", payload)
        return normalize_weights(weights)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return dict(DEFAULT_WEIGHTS)


def normalize_weights(weights: dict[str, Any]) -> dict[str, float]:
    values = {key: max(0.0, _number(weights.get(key, DEFAULT_WEIGHTS[key]))) for key in DEFAULT_WEIGHTS}
    total = sum(values.values())
    if total <= 0:
        return dict(DEFAULT_WEIGHTS)
    return {key: value / total for key, value in values.items()}


def apply_prediction_weights(table: pd.DataFrame, weights: dict[str, Any] | None = None) -> pd.DataFrame:
    if table.empty:
        return table.copy()
    active = normalize_weights(weights or load_model_weights())
    result = table.copy()
    raw = pd.Series(0.0, index=result.index, dtype=float)
    for feature, weight in active.items():
        source = "normalized_elo_score" if feature == "elo_score" and feature not in result.columns else feature
        values = result[source] if source in result.columns else pd.Series(50.0, index=result.index)
        raw += pd.to_numeric(values, errors="coerce").fillna(50.0) * weight
    result["optimized_raw_score"] = raw.round(4)
    result["prediction_score"] = _compress_scores(raw)
    result["score"] = result["prediction_score"]
    result["prediction_engine"] = "optimized_weights"
    result = result.sort_values(
        ["prediction_score", "win_rate", "top3_rate", "avg_finish"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    marks = ("◎", "○", "▲", "△", "☆")
    result["印"] = [marks[index] if index < min(5, len(result)) else "" for index in range(len(result))]
    return result


def _evaluate_weights(training_df: pd.DataFrame, weights: dict[str, float], metric: str) -> float:
    work = training_df.copy()
    work["candidate_score"] = 0.0
    for feature, weight in weights.items():
        work["candidate_score"] += pd.to_numeric(work[feature], errors="coerce").fillna(50.0) * weight
    race_scores: list[float] = []
    for _, race in work.groupby("race_id", sort=False):
        ordered = race.sort_values("candidate_score", ascending=False)
        if metric == "top3_hit_rate":
            denominator = max(1, min(3, int(race["is_top3"].sum())))
            race_scores.append(float(ordered.head(3)["is_top3"].sum()) / denominator)
        elif metric == "winner_hit_rate":
            race_scores.append(float(ordered.iloc[0]["is_win"] == 1))
        else:
            centered = (race["candidate_score"] - race["candidate_score"].mean()) / 10.0
            probabilities = 1.0 / (1.0 + np.exp(-centered.clip(-20, 20)))
            targets = race["is_top3"].astype(float)
            if metric == "brier_score":
                race_scores.append(-float(np.mean((probabilities - targets) ** 2)))
            else:
                clipped = probabilities.clip(1e-6, 1 - 1e-6)
                loss = -(targets * np.log(clipped) + (1 - targets) * np.log(1 - clipped)).mean()
                race_scores.append(-float(loss))
    return float(np.mean(race_scores)) if race_scores else float("-inf")


def _compress_scores(scores: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(scores, errors="coerce").fillna(50.0)
    std = float(numeric.std(ddof=0))
    if std < 1e-9:
        return pd.Series([50.0] * len(numeric), index=numeric.index)
    return (50.0 + 10.0 * ((numeric - numeric.mean()) / std).clip(-2.0, 2.0)).clip(0, 100).round(2)


def _integer(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _number(value: Any) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else 0.0
    except (TypeError, ValueError):
        return 0.0
