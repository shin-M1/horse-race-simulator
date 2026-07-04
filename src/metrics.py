from __future__ import annotations

from typing import Any

import pandas as pd


def aggregate_evaluation_logs(evaluation_logs: list[dict[str, Any]]) -> dict[str, Any]:
    if not evaluation_logs:
        return _empty_metrics()
    race_count = len(evaluation_logs)
    evaluations = [log.get("evaluation_metrics", log.get("evaluation", {})) or {} for log in evaluation_logs]

    def mark_top3_rate(mark: str) -> float:
        valid = [metrics.get("mark_finishes", metrics.get("mark_results", {})).get(mark) for metrics in evaluations]
        known = [_integer(value) for value in valid if _integer(value) > 0]
        return sum(value <= 3 for value in known) / len(known) if known else 0.0

    hit_races: list[dict[str, Any]] = []
    missed_races: list[dict[str, Any]] = []
    for log, evaluation in zip(evaluation_logs, evaluations):
        item = {
            "race_id": log.get("race_id", ""),
            "race_name": log.get("race_name", ""),
            "race_date": log.get("race_date", ""),
        }
        (hit_races if evaluation.get("win_hit") or evaluation.get("top3_hit") else missed_races).append(item)

    return {
        "race_count": race_count,
        "honmei_win_rate": sum(bool(item.get("win_hit")) for item in evaluations) / race_count,
        "honmei_top3_rate": mark_top3_rate("◎"),
        "second_mark_top3_rate": mark_top3_rate("○"),
        "third_mark_top3_rate": mark_top3_rate("▲"),
        "average_marked_top3_count": sum(_number(item.get("marked_top3_count")) for item in evaluations) / race_count,
        "average_predicted_top3_count": sum(_number(item.get("predicted_top3_hit_count")) for item in evaluations) / race_count,
        "winner_hit_rate": sum(bool(item.get("winner_hit", item.get("win_hit"))) for item in evaluations) / race_count,
        "top5_hit_rate": sum(_number(item.get("top5_hit_count")) > 0 for item in evaluations) / race_count,
        "average_hit_count": sum(_number(item.get("top5_hit_count")) for item in evaluations) / race_count,
        "hit_races": hit_races,
        "missed_races": missed_races,
    }


def aggregate_failure_tags(evaluation_logs: list[dict[str, Any]]) -> pd.DataFrame:
    """Count failure tags once per evaluation log."""
    counts: dict[str, int] = {}
    for log in evaluation_logs:
        failure = _failure_for_log(log)
        for tag in dict.fromkeys(failure.get("miss_reason_tags", []) if isinstance(failure, dict) else []):
            text = str(tag).strip()
            if text:
                counts[text] = counts.get(text, 0) + 1
    total = max(1, len(evaluation_logs))
    rows = [
        {"タグ名": tag, "回数": count, "割合": count / total}
        for tag, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    return pd.DataFrame(rows, columns=["タグ名", "回数", "割合"])


def aggregate_improvement_suggestions(evaluation_logs: list[dict[str, Any]]) -> pd.DataFrame:
    """Aggregate recurring improvement suggestions from saved analyses."""
    counts: dict[str, int] = {}
    for log in evaluation_logs:
        failure = _failure_for_log(log)
        for suggestion in dict.fromkeys(failure.get("improvement_suggestions", []) if isinstance(failure, dict) else []):
            text = str(suggestion).strip()
            if text:
                counts[text] = counts.get(text, 0) + 1
    rows = [
        {"改善提案": suggestion, "回数": count}
        for suggestion, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    return pd.DataFrame(rows, columns=["改善提案", "回数"])


def _failure_for_log(log: dict[str, Any]) -> dict[str, Any]:
    failure = log.get("failure_analysis")
    if not isinstance(failure, dict):
        metrics = log.get("evaluation_metrics") or log.get("evaluation") or {}
        failure = metrics.get("failure_analysis", {}) if isinstance(metrics, dict) else {}
    if failure:
        return failure
    if log.get("prediction_table") and log.get("actual_result"):
        from analysis_reporter import analyze_prediction_failure

        metrics = log.get("evaluation_metrics") or log.get("evaluation") or {}
        return analyze_prediction_failure(log, list(log.get("actual_result", [])), metrics)
    return {}


def _empty_metrics() -> dict[str, Any]:
    return {
        "race_count": 0,
        "honmei_win_rate": 0.0,
        "honmei_top3_rate": 0.0,
        "second_mark_top3_rate": 0.0,
        "third_mark_top3_rate": 0.0,
        "average_marked_top3_count": 0.0,
        "average_predicted_top3_count": 0.0,
        "winner_hit_rate": 0.0,
        "top5_hit_rate": 0.0,
        "average_hit_count": 0.0,
        "hit_races": [],
        "missed_races": [],
    }


def _integer(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
