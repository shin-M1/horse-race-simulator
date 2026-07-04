from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from course_db import CourseDB, get_course_bias, get_track_bias_fit_score
from horse_analyzer import HorseAbility, HorseAnalyzer
from main import _coerce_horses, _coerce_race_config, load_provider
from pace_predictor import PacePredictor, RacePace
from race_config import HorseEntry, RaceConfig
from result_formatter import build_single_race_result_from_timeline
from race_trend_analyzer import calculate_race_trend_score
from race_trend_scorer import compute_race_trend_match_score
from simulator import RaceSimulator
from ml_model import TOP3_MODEL_PATH, WIN_MODEL_PATH, apply_ml_prediction, load_model
from weight_optimizer import apply_prediction_weights, load_model_weights


PREDICTION_COLUMNS = [
    "印",
    "馬番",
    "馬名",
    "枠順",
    "斤量",
    "primary_running_style",
    "actual_running_style",
    "win_rate",
    "top2_rate",
    "top3_rate",
    "avg_finish",
    "median_finish",
    "worst_finish",
    "best_finish",
    "avg_time",
    "prediction_score",
    "prediction_engine",
    "horse_ability_score",
    "race_power",
    "race_strength_score",
    "race_strength_adjusted_score",
    "elo_rating",
    "normalized_elo_score",
    "elo_score",
    "relative_agari_score",
    "late_kick_score",
    "course_fit_score",
    "pace_fit_score",
    "jockey_score",
    "track_bias_fit_score",
    "race_trend_score",
    "frame_trend_score",
    "horse_number_trend_score",
    "style_trend_score",
    "agari_trend_score",
    "fourth_corner_trend_score",
    "age_trend_score",
    "weight_trend_score",
    "jockey_continuity_score",
    "previous_race_trend_score",
    "bloodline_trend_score",
    "trend_match_comment",
    "weight_penalty",
    "mud_aptitude",
    "finish_score",
    "margin_score",
    "time_score",
    "last3f_score",
    "carried_weight",
    "frame",
    "horse_number",
    "final_prediction_score",
    "score",
    "予想根拠",
]


def run_monte_carlo_prediction(
    race_config: dict[str, Any] | RaceConfig,
    horses: list[dict[str, Any]] | list[HorseEntry],
    n_simulations: int = 500,
    seed: int | None = 42,
    abilities: list[HorseAbility] | None = None,
    pace: RacePace | None = None,
    output_dir: str = "outputs",
    prediction_engine: str = "rule_based",
    prediction_weights: dict[str, float] | None = None,
    top3_model_path: str | Path = TOP3_MODEL_PATH,
    win_model_path: str | Path = WIN_MODEL_PATH,
    trend_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run repeated race simulations and aggregate finish-position probabilities."""
    raw_race_config = race_config
    if trend_analysis is None and isinstance(raw_race_config, dict):
        maybe_trends = raw_race_config.get("same_race_trend_analysis") or raw_race_config.get("race_trend_analysis")
        if isinstance(maybe_trends, dict):
            trend_analysis = maybe_trends
    config = _coerce_race_config(race_config)
    entries = _coerce_horses(horses)
    n_simulations = max(1, int(n_simulations))

    if abilities is None:
        provider = load_provider(None, None)
        abilities = HorseAnalyzer(provider, config).analyze_many(entries)
    if pace is None:
        pace = PacePredictor().predict(abilities)

    simulator = RaceSimulator()
    rng = random.Random(seed)
    horse_count = len(abilities)
    aggregates: dict[int, dict[str, Any]] = {
        ability.horse_number: {
            "ability": ability,
            "finishes": [],
            "times": [],
            "actual_styles": [],
        }
        for ability in abilities
    }

    logs = [
        f"Monte Carlo prediction started: n_simulations={n_simulations}, seed={seed}",
        f"Base pace prediction: {pace.pace}, front_pressure={pace.front_pressure:.3f}",
    ]
    horse_inputs = [entry.to_dict() for entry in entries]
    simulation_trials: list[dict[str, Any]] = []

    for trial_index in range(n_simulations):
        trial_seed = rng.randrange(1, 2**31 - 1)
        trial_pace = _perturb_pace(pace, random.Random(trial_seed + 17))
        result = simulator.simulate(config=config, abilities=abilities, pace=trial_pace, seed=trial_seed)
        result_df = build_single_race_result_from_timeline(result.race_timeline, horse_inputs)
        simulation_trials.append(
            {
                "trial_index": trial_index,
                "seed": trial_seed,
                "pace": trial_pace.pace,
                "ranking": result.ranking.copy(),
                "race_timeline": result.race_timeline,
                "result_df": result_df,
            }
        )
        for _, row in result.ranking.iterrows():
            horse_number = int(row["horse_number"])
            aggregates[horse_number]["finishes"].append(int(row["rank"]))
            aggregates[horse_number]["times"].append(float(row["finish_time"]))
            aggregates[horse_number]["actual_styles"].append(str(row["actual_running_style"]))
        if (trial_index + 1) % max(1, n_simulations // 5) == 0:
            logs.append(f"{trial_index + 1}/{n_simulations} simulations completed")

    prediction_table = _build_prediction_table(config, aggregates, horse_count, pace, trend_analysis=trend_analysis)
    active_engine = str(prediction_engine or "rule_based")
    if active_engine == "optimized_weights":
        prediction_table = apply_prediction_weights(prediction_table, prediction_weights or load_model_weights())
    elif active_engine == "ml_model":
        top3_model = load_model(top3_model_path)
        win_model = load_model(win_model_path)
        if top3_model is None:
            active_engine = "rule_based"
            logs.append("ML model was not found; rule_based ranking was used.")
        else:
            prediction_table = apply_ml_prediction(prediction_table, top3_model, win_model)
    prediction_table["prediction_engine"] = active_engine
    representative_trial = select_highest_expected_value_trial(prediction_table, simulation_trials)
    saved_paths = _save_prediction_outputs(
        prediction_table=prediction_table,
        config=config,
        horses=entries,
        output_dir=output_dir,
    )
    summary = _build_summary(prediction_table)
    summary.update(
        {
            "n_simulations": n_simulations,
            "seed": seed,
            "saved_paths": saved_paths,
            "representative_trial_index": representative_trial.get("trial_index"),
            "representative_trial_seed": representative_trial.get("seed"),
            "representative_value_score": representative_trial.get("representative_value_score"),
            "top5_horses_in_selected_trial": representative_trial.get("top5_horses_in_selected_trial", []),
            "prediction_engine": active_engine,
        }
    )
    logs.append(f"Prediction CSV saved: {saved_paths['prediction_table']}")
    return {
        "prediction_table": prediction_table,
        "simulation_trials": simulation_trials,
        "representative_trial": representative_trial,
        "simulation_logs": logs,
        "summary": summary,
        "prediction_engine": active_engine,
    }


def select_representative_trial(
    prediction_table: pd.DataFrame,
    simulation_trials: list[dict[str, Any]],
) -> dict[str, Any]:
    """Select the trial whose finish order is closest to the prediction table."""
    if prediction_table.empty:
        raise ValueError("prediction_table is empty")
    if not simulation_trials:
        raise ValueError("simulation_trials is empty")

    predicted_order = _prediction_order(prediction_table)
    predicted_rank = {horse_number: rank for rank, horse_number in enumerate(predicted_order, start=1)}
    predicted_top5 = set(predicted_order[:5])
    best_trial: dict[str, Any] | None = None
    best_distance: float | None = None
    best_top5_overlap = -1

    for trial in simulation_trials:
        trial_order = _trial_order(trial)
        if not trial_order:
            continue
        trial_rank = {horse_number: rank for rank, horse_number in enumerate(trial_order, start=1)}
        all_numbers = set(predicted_rank) | set(trial_rank)
        missing_rank = len(all_numbers) + 1
        distance = sum(
            abs(predicted_rank.get(number, missing_rank) - trial_rank.get(number, missing_rank))
            for number in all_numbers
        )
        top5_overlap = len(predicted_top5 & set(trial_order[:5]))
        if (
            best_distance is None
            or distance < best_distance
            or (distance == best_distance and top5_overlap > best_top5_overlap)
        ):
            best_trial = trial
            best_distance = float(distance)
            best_top5_overlap = top5_overlap

    if best_trial is None:
        raise ValueError("simulation_trials contain no valid ranking")

    selected = dict(best_trial)
    selected["ranking_distance"] = best_distance
    selected["top5_overlap_count"] = best_top5_overlap
    return selected


def select_highest_expected_value_trial(
    prediction_table: pd.DataFrame,
    simulation_trials: list[dict[str, Any]],
) -> dict[str, Any]:
    """Select the trial with the highest value for AI-rated horses."""
    if prediction_table.empty:
        raise ValueError("prediction_table is empty")
    if not simulation_trials:
        raise ValueError("simulation_trials is empty")

    horse_column = _first_existing_column(prediction_table, ["horse_number", "馬番"])
    score_column = _first_existing_column(prediction_table, ["prediction_score", "score"])
    if horse_column is None or score_column is None:
        raise ValueError("prediction_table requires horse number and prediction score")
    prediction_scores = {
        int(row[horse_column]): float(row[score_column])
        for _, row in prediction_table.iterrows()
    }
    finish_values = {1: 1.00, 2: 0.75, 3: 0.60, 4: 0.40, 5: 0.25}
    selected: dict[str, Any] | None = None
    selected_value: float | None = None

    for trial in simulation_trials:
        order = _trial_order(trial)
        if not order:
            continue
        finish_by_number = {number: rank for rank, number in enumerate(order, start=1)}
        value_score = sum(
            score * finish_values.get(finish_by_number.get(number, 999), 0.05)
            for number, score in prediction_scores.items()
        )
        if selected_value is None or value_score > selected_value:
            selected = trial
            selected_value = float(value_score)

    if selected is None:
        raise ValueError("simulation_trials contain no valid ranking")
    result = dict(selected)
    order = _trial_order(selected)
    result["representative_value_score"] = round(float(selected_value), 4)
    result["top5_horses_in_selected_trial"] = order[:5]
    return result


def _prediction_order(prediction_table: pd.DataFrame) -> list[int]:
    horse_column = _first_existing_column(prediction_table, ["horse_number", "馬番"])
    if horse_column is None:
        raise ValueError("prediction_table has no horse number column")
    score_column = "score" if "score" in prediction_table.columns else "prediction_score"
    sort_columns = [column for column in [score_column, "win_rate", "top3_rate"] if column in prediction_table.columns]
    sorted_table = prediction_table.sort_values(
        sort_columns,
        ascending=[False for _ in sort_columns],
    )
    return [int(value) for value in sorted_table[horse_column].tolist()]


def _trial_order(trial: dict[str, Any]) -> list[int]:
    result_df = trial.get("result_df")
    if isinstance(result_df, pd.DataFrame) and not result_df.empty:
        horse_column = _first_existing_column(result_df, ["horse_number", "馬番"])
        rank_column = _first_existing_column(result_df, ["rank", "着順"])
        if horse_column is not None:
            table = result_df
            if rank_column is not None:
                table = table.sort_values(rank_column, ascending=True)
            return [int(value) for value in table[horse_column].tolist()]

    ranking = trial.get("ranking")
    if isinstance(ranking, pd.DataFrame) and not ranking.empty:
        horse_column = _first_existing_column(ranking, ["horse_number", "馬番"])
        rank_column = _first_existing_column(ranking, ["rank", "着順"])
        if horse_column is not None:
            table = ranking
            if rank_column is not None:
                table = table.sort_values(rank_column, ascending=True)
            return [int(value) for value in table[horse_column].tolist()]
    if isinstance(ranking, list):
        rows = [row for row in ranking if isinstance(row, dict)]
        rows.sort(key=lambda row: int(row.get("rank", row.get("着順", 999)) or 999))
        return [int(row.get("horse_number", row.get("馬番", 0)) or 0) for row in rows if row.get("horse_number", row.get("馬番"))]
    return []


def _first_existing_column(table: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in table.columns:
            return column
    return None


def _perturb_pace(base_pace: RacePace, rng: random.Random) -> RacePace:
    pace = base_pace.pace
    roll = rng.random()
    if pace == "medium":
        if roll < 0.08:
            pace = "high"
        elif roll < 0.16:
            pace = "slow"
    elif pace == "high" and roll < 0.10:
        pace = "medium"
    elif pace == "slow" and roll < 0.10:
        pace = "medium"

    closer_base = {"high": 1.08, "medium": 1.0, "slow": 0.92}.get(pace, 1.0)
    return RacePace(
        pace=pace,
        front_group_size=base_pace.front_group_size,
        closer_advantage=closer_base * rng.uniform(0.985, 1.015),
        style_groups=base_pace.style_groups,
        front_pressure=base_pace.front_pressure,
        front_group=base_pace.front_group,
        middle_group=base_pace.middle_group,
        back_group=base_pace.back_group,
    )


def _build_prediction_table(
    config: RaceConfig,
    aggregates: dict[int, dict[str, Any]],
    horse_count: int,
    pace: RacePace | None = None,
    trend_analysis: dict[str, Any] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    course_bias = get_course_bias(config)
    for horse_number, data in aggregates.items():
        ability: HorseAbility = data["ability"]
        finishes = pd.Series(data["finishes"], dtype=float)
        times = pd.Series(data["times"], dtype=float)
        actual_styles = pd.Series(data["actual_styles"], dtype=str)
        most_common_actual_style = str(actual_styles.mode().iloc[0]) if not actual_styles.empty else ability.primary_running_style
        win_rate = float((finishes == 1).mean())
        top2_rate = float((finishes <= 2).mean())
        top3_rate = float((finishes <= 3).mean())
        avg_finish = float(finishes.mean())
        normalized_avg_finish_score = 1.0 if horse_count <= 1 else max(0.0, min(1.0, (horse_count - avg_finish) / (horse_count - 1)))
        raw_score = win_rate * 0.45 + top2_rate * 0.25 + top3_rate * 0.20 + normalized_avg_finish_score * 0.10
        monte_carlo_score = min(
            100.0,
            max(0.0, (raw_score + _course_bias_score_bonus(course_bias, ability.frame, most_common_actual_style)) * 100.0),
        )
        track_bias_fit_score = get_track_bias_fit_score(config, most_common_actual_style, ability.frame)
        pace_fit_score = _prediction_pace_fit_score(ability, pace.pace if pace is not None else "medium")
        pace_fit_score = max(0.0, min(100.0, pace_fit_score * 0.85 + track_bias_fit_score * 0.15))
        course_fit_score = max(0.0, min(100.0, ability.course_fit_score * 0.90 + track_bias_fit_score * 0.10))
        trend_row = {
            "馬番": ability.horse_number,
            "馬名": ability.horse_name,
            "枠順": ability.frame,
            "frame": ability.frame,
            "horse_number": ability.horse_number,
            "primary_running_style": ability.primary_running_style,
            "actual_running_style": most_common_actual_style,
            "late_kick_score": ability.late_kick_score,
            "jockey_switch": getattr(ability, "jockey_switch", ""),
        }
        trend_scores = _calculate_trend_scores(trend_row, ability, trend_analysis, config)
        race_trend_score = float(trend_scores["race_trend_score"])
        trend_match_comment = str(trend_scores["trend_match_comment"])
        final_prediction_score = (
            ability.horse_ability_score * 0.25
            + ability.race_strength_score * 0.15
            + ability.normalized_elo_score * 0.15
            + ability.late_kick_score * 0.15
            + course_fit_score * 0.10
            + pace_fit_score * 0.10
            + ability.jockey_score * 0.05
            + track_bias_fit_score * 0.05
            + race_trend_score * 0.10
        )
        raw_prediction_score = max(0.0, min(100.0, final_prediction_score * 0.70 + monte_carlo_score * 0.30))
        stability = max(0.0, 1.0 - float(finishes.std(ddof=0)) / max(1.0, horse_count - 1))
        rows.append(
            {
                "馬番": ability.horse_number,
                "馬名": ability.horse_name,
                "枠順": ability.frame,
                "斤量": round(ability.carried_weight, 1),
                "primary_running_style": ability.primary_running_style,
                "actual_running_style": most_common_actual_style,
                "win_rate": round(win_rate, 4),
                "top2_rate": round(top2_rate, 4),
                "top3_rate": round(top3_rate, 4),
                "avg_finish": round(avg_finish, 3),
                "median_finish": round(float(finishes.median()), 3),
                "worst_finish": int(finishes.max()),
                "best_finish": int(finishes.min()),
                "avg_time": round(float(times.mean()), 3),
                "raw_prediction_score": raw_prediction_score,
                "prediction_score": 0.0,
                "prediction_engine": "rule_based",
                "horse_ability_score": round(ability.horse_ability_score, 2),
                "race_power": round(ability.race_power, 2),
                "race_strength_score": round(ability.race_strength_score, 2),
                "race_strength_adjusted_score": round(ability.race_strength_adjusted_score, 2),
                "elo_rating": round(ability.elo_rating, 2),
                "normalized_elo_score": round(ability.normalized_elo_score, 2),
                "elo_score": round(ability.normalized_elo_score, 2),
                "relative_agari_score": round(ability.relative_agari_score, 2),
                "late_kick_score": round(ability.late_kick_score, 2),
                "course_fit_score": round(course_fit_score, 2),
                "pace_fit_score": round(pace_fit_score, 2),
                "jockey_score": round(ability.jockey_score, 2),
                "track_bias_fit_score": round(track_bias_fit_score, 2),
                "race_trend_score": round(race_trend_score, 2),
                "frame_trend_score": round(float(trend_scores.get("frame_trend_score", 50.0)), 2),
                "horse_number_trend_score": round(float(trend_scores.get("horse_number_trend_score", 50.0)), 2),
                "style_trend_score": round(float(trend_scores.get("style_trend_score", 50.0)), 2),
                "agari_trend_score": round(float(trend_scores.get("agari_trend_score", 50.0)), 2),
                "fourth_corner_trend_score": round(float(trend_scores.get("fourth_corner_trend_score", 50.0)), 2),
                "age_trend_score": round(float(trend_scores.get("age_trend_score", 50.0)), 2),
                "weight_trend_score": round(float(trend_scores.get("weight_trend_score", 50.0)), 2),
                "jockey_continuity_score": round(float(trend_scores.get("jockey_continuity_score", 50.0)), 2),
                "previous_race_trend_score": round(float(trend_scores.get("previous_race_trend_score", 50.0)), 2),
                "bloodline_trend_score": round(float(trend_scores.get("bloodline_trend_score", 50.0)), 2),
                "trend_match_comment": trend_match_comment,
                "weight_penalty": round(ability.weight_penalty, 2),
                "mud_aptitude": round(ability.mud_aptitude, 2),
                "finish_score": round(ability.finish_score, 2),
                "margin_score": round(ability.margin_score, 2),
                "time_score": round(ability.time_score, 2),
                "last3f_score": round(ability.last3f_score, 2),
                "carried_weight": round(ability.carried_weight, 1),
                "frame": ability.frame,
                "horse_number": ability.horse_number,
                "final_prediction_score": round(final_prediction_score, 2),
                "score": 0.0,
                "stability": round(stability, 3),
                "予想根拠": _build_reason(config, ability, win_rate, top2_rate, top3_rate, avg_finish, stability),
            }
        )

    table = pd.DataFrame(rows)
    table["prediction_score"] = _compress_prediction_scores(table["raw_prediction_score"])
    table["score"] = table["prediction_score"]
    table = table.sort_values(
        ["prediction_score", "win_rate", "top3_rate", "avg_finish"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    table["印"] = _assign_marks(table)
    return table[PREDICTION_COLUMNS]


def _calculate_trend_scores(
    horse_row: dict[str, Any],
    ability: HorseAbility,
    trend_analysis: dict[str, Any] | None,
    config: RaceConfig,
) -> dict[str, Any]:
    neutral = {
        "race_trend_score": 50.0,
        "frame_trend_score": 50.0,
        "horse_number_trend_score": 50.0,
        "style_trend_score": 50.0,
        "agari_trend_score": 50.0,
        "fourth_corner_trend_score": 50.0,
        "age_trend_score": 50.0,
        "weight_trend_score": 50.0,
        "jockey_continuity_score": 50.0,
        "previous_race_trend_score": 50.0,
        "bloodline_trend_score": 50.0,
        "trend_match_comment": "historical trend data unavailable; neutral score applied",
    }
    if not isinstance(trend_analysis, dict) or not trend_analysis:
        return neutral

    if any(key in trend_analysis for key in ("frame_trend", "style_trend", "agari_trend", "fourth_corner_trend")):
        analysis_row = ability.to_dict()
        analysis_row.update(horse_row)
        horse_source = dict(horse_row)
        horse_source.update(
            {
                "horse_name": ability.horse_name,
                "horse_number": ability.horse_number,
                "frame": ability.frame,
                "carried_weight": ability.carried_weight,
                "jockey": ability.jockey,
            }
        )
        return compute_race_trend_match_score(
            horse=horse_source,
            horse_analysis=analysis_row,
            race_trends=trend_analysis,
            race_config=config.to_dict(),
        )

    legacy_score, legacy_comment = calculate_race_trend_score(horse_row, trend_analysis)
    result = dict(neutral)
    result["race_trend_score"] = legacy_score
    result["trend_match_comment"] = legacy_comment
    return result


def _assign_marks(table: pd.DataFrame) -> list[str]:
    mark_order = ["◎", "○", "▲", "△", "☆"]
    return [mark_order[index] if index < min(5, len(table)) else "" for index in range(len(table))]


def _compress_prediction_scores(raw_scores: pd.Series) -> pd.Series:
    scores = pd.to_numeric(raw_scores, errors="coerce").fillna(0.0)
    mean_score = float(scores.mean())
    std_score = float(scores.std(ddof=0))
    if std_score < 1e-9:
        return pd.Series([50.0 for _ in scores], index=scores.index)
    z_scores = ((scores - mean_score) / std_score).clip(-2.0, 2.0)
    return (50.0 + 10.0 * z_scores).clip(0.0, 100.0).round(2)


def _course_bias_score_bonus(course_bias: dict[str, float | str], frame: int, actual_style: str) -> float:
    bias_scale = 0.25
    bonus = 0.0
    if frame <= 3:
        bonus += float(course_bias.get("inner_bias", 0.0)) * bias_scale
    elif frame >= 7:
        bonus += float(course_bias.get("outer_bias", 0.0)) * bias_scale
    if actual_style in {"逃げ", "先行"}:
        bonus += float(course_bias.get("front_bias", 0.0)) * bias_scale
    elif actual_style in {"差し", "追込"}:
        bonus += float(course_bias.get("closer_bias", 0.0)) * bias_scale
    return bonus


def _prediction_pace_fit_score(ability: HorseAbility, pace: str) -> float:
    profile = ability.base_style_profile
    front_probability = profile.get("逃げ", 0.0) + profile.get("先行", 0.0)
    closer_probability = profile.get("差し", 0.0) + profile.get("追込", 0.0)
    if pace == "slow":
        return front_probability * 100.0
    if pace == "high":
        return closer_probability * 100.0
    return (1.0 - abs(front_probability - closer_probability)) * 100.0


def _build_reason(
    config: RaceConfig,
    ability: HorseAbility,
    win_rate: float,
    top2_rate: float,
    top3_rate: float,
    avg_finish: float,
    stability: float,
) -> str:
    profile = ability.base_style_profile
    style_prob = profile.get(ability.primary_running_style, 0.0)
    course_db = CourseDB()
    frame_bonus = course_db.frame_bonus(ability.frame, config)
    course_bias = get_course_bias(config)
    reasons = [
        f"近走安定度{ability.consistency:.1f}",
        f"{ability.primary_running_style}傾向{style_prob:.0%}",
        f"馬場適性{ability.mud_aptitude:.1f}",
        f"平均着順{avg_finish:.2f}",
        f"複勝率{top3_rate:.1%}",
        f"安定度{stability:.1%}",
        f"レース強度{ability.race_strength_score:.1f}",
        f"ELO {ability.elo_rating:.0f}",
        f"コース適性{ability.course_fit_score:.1f}",
    ]
    selected_track_bias = str(getattr(config, "track_bias", "標準"))
    track_fit = get_track_bias_fit_score(config, ability.primary_running_style, ability.frame)
    if selected_track_bias != "標準" and track_fit > 50.0:
        reasons.append(f"{selected_track_bias}適合")
    if ability.jockey:
        reasons.append(f"騎手 {ability.jockey}")
    if config.distance >= 2200 and ability.stamina >= 65:
        reasons.append("距離延長向き")
    elif config.distance <= 1600 and max(ability.early_speed, ability.acceleration) >= 70:
        reasons.append("短距離の速力評価")
    if frame_bonus > 1.0:
        reasons.append("枠順利")
    elif frame_bonus < 1.0:
        reasons.append("枠順不利")
    if ability.frame <= 3 and float(course_bias.get("inner_bias", 0.0)) > 0:
        reasons.append("内バイアス利")
    elif ability.frame >= 7 and float(course_bias.get("outer_bias", 0.0)) > 0:
        reasons.append("外バイアス利")
    if ability.primary_running_style in {"逃げ", "先行"} and float(course_bias.get("front_bias", 0.0)) > 0:
        reasons.append("先行バイアス利")
    elif ability.primary_running_style in {"差し", "追込"} and float(course_bias.get("closer_bias", 0.0)) > 0:
        reasons.append("差しバイアス利")
    if win_rate >= 0.30:
        reasons.append("勝ち切り候補")
    elif top2_rate >= 0.45:
        reasons.append("連対圏安定")
    if ability.carried_weight >= 58.0:
        reasons.append("斤量負荷あり")
    elif ability.carried_weight <= 54.0:
        reasons.append("軽斤量")
    return " / ".join(reasons)


def _build_summary(prediction_table: pd.DataFrame) -> dict[str, Any]:
    if prediction_table.empty:
        return {"marked_horses": []}
    sorted_table = prediction_table[prediction_table["印"].astype(str) != ""].copy()
    return {
        "marked_horses": sorted_table[["印", "馬番", "馬名", "prediction_score"]].to_dict("records"),
    }


def _save_prediction_outputs(
    prediction_table: pd.DataFrame,
    config: RaceConfig,
    horses: list[HorseEntry],
    output_dir: str,
) -> dict[str, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prediction_dir = Path(output_dir) / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    prediction_csv = prediction_dir / f"prediction_table_{timestamp}.csv"
    race_config_csv = prediction_dir / f"race_config_{timestamp}.csv"
    horse_inputs_csv = prediction_dir / f"horse_inputs_{timestamp}.csv"
    metadata_json = prediction_dir / f"prediction_metadata_{timestamp}.json"

    prediction_table.to_csv(prediction_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame([config.to_dict()]).to_csv(race_config_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame([horse.to_dict() for horse in horses]).to_csv(horse_inputs_csv, index=False, encoding="utf-8-sig")
    metadata_json.write_text(
        json.dumps(
            {
                "timestamp": timestamp,
                "race_config": config.to_dict(),
                "horse_inputs": [horse.to_dict() for horse in horses],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "prediction_table": str(prediction_csv),
        "race_config": str(race_config_csv),
        "horse_inputs": str(horse_inputs_csv),
        "metadata": str(metadata_json),
    }
