from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from log_manager import DuplicateLogError, find_matching_logs


PREDICTION_LOG_DIR = Path("outputs/prediction_logs")
EVALUATION_LOG_DIR = Path("outputs/evaluation_logs")
MARKS = ("◎", "○", "▲", "△", "☆")


def save_prediction_log(
    *,
    race_id: str | None,
    source_url: str | None,
    fetched_entries: list[dict[str, Any]],
    race_metadata: dict[str, Any],
    prediction_table: pd.DataFrame | list[dict[str, Any]] | None,
    simulation_result: dict[str, Any],
    output_dir: str | Path = PREDICTION_LOG_DIR,
    duplicate_action: str = "skip",
) -> Path:
    """Save a prediction snapshot. Actual results are intentionally absent here."""
    now = datetime.now()
    timestamp = now.isoformat(timespec="seconds")
    file_stamp = now.strftime("%Y%m%d_%H%M%S")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    existing = find_matching_logs(
        race_id=str(race_id or ""),
        race_name=str(race_metadata.get("race_name", "")),
        race_date=str(race_metadata.get("race_date", "")),
        log_dir=directory,
    )
    path = _resolve_save_path(
        directory=directory,
        base_name=f"prediction_{file_stamp}.json",
        existing=existing,
        duplicate_action=duplicate_action,
    )
    csv_path = path.with_suffix(".csv")
    timeline = [] if simulation_result.get("skip_timeline_log") else simulation_result.get("race_timeline") or simulation_result.get("controlled_timeline") or []
    timeline_path: Path | None = None
    if isinstance(timeline, list) and timeline:
        timeline_dir = directory / "timelines"
        timeline_dir.mkdir(parents=True, exist_ok=True)
        timeline_path = timeline_dir / f"race_timeline_{path.stem.removeprefix('prediction_')}.json"
        _write_json(timeline_path, {"race_timeline": timeline})
    prediction_records = _records(prediction_table)
    single_result = _records(simulation_result.get("single_result"))
    horse_analysis = _records(simulation_result.get("horse_analysis"))
    prediction_payload = simulation_result.get("prediction")
    prediction_payload = prediction_payload if isinstance(prediction_payload, dict) else {}
    selected_trial = _compact_trial(
        simulation_result.get("representative_trial")
        or simulation_result.get("selected_trial")
        or prediction_payload.get("representative_trial", {})
    )
    payload = {
        "log_type": "prediction",
        "timestamp": timestamp,
        "race_id": str(race_id or ""),
        "source_url": str(source_url or ""),
        "race_name": race_metadata.get("race_name", ""),
        "race_date": race_metadata.get("race_date", ""),
        "fetched_entries": fetched_entries,
        "race_metadata": race_metadata,
        "race_config": simulation_result.get("race_config", race_metadata),
        "horse_inputs": simulation_result.get("horse_inputs", fetched_entries),
        "prediction_table": prediction_records,
        "AI予想印": [
            {"印": row.get("印", ""), "馬番": row.get("馬番", row.get("horse_number")), "馬名": row.get("馬名", row.get("horse_name", ""))}
            for row in prediction_records
            if str(row.get("印", ""))
        ],
        "single_result": single_result,
        "selected_trial": selected_trial,
        "race_timeline_path": str(timeline_path) if timeline_path else "",
        "video_path": str(simulation_result.get("mp4_path", simulation_result.get("animation_path", "")) or ""),
        "horse_analysis": horse_analysis,
        "comments_table": _records(simulation_result.get("comments_table")),
        "pace_prediction": simulation_result.get("pace_prediction", {}),
        "simulation_result": _compact_simulation_result(simulation_result),
        "actual_result": None,
        "csv_path": str(csv_path),
    }
    _write_json(path, payload)
    prediction_frame = pd.DataFrame(prediction_records)
    if prediction_frame.empty:
        prediction_frame = pd.DataFrame([{"race_id": payload["race_id"], "race_name": payload["race_name"]}])
    else:
        prediction_frame.insert(0, "race_date", payload["race_date"])
        prediction_frame.insert(0, "race_name", payload["race_name"])
        prediction_frame.insert(0, "race_id", payload["race_id"])
    prediction_frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return path


def load_prediction_logs(output_dir: str | Path = PREDICTION_LOG_DIR) -> list[dict[str, Any]]:
    return _load_logs(output_dir)


def find_prediction_log(
    *,
    race_id: str | None = None,
    race_name: str | None = None,
    race_date: str | None = None,
    output_dir: str | Path = PREDICTION_LOG_DIR,
) -> dict[str, Any] | None:
    logs = load_prediction_logs(output_dir)
    normalized_name = _normalize(race_name)
    normalized_date = str(race_date or "")
    for log in reversed(logs):
        if race_id and str(log.get("race_id", "")) == str(race_id):
            return log
        if normalized_name and normalized_date:
            if _normalize(log.get("race_name")) == normalized_name and str(log.get("race_date", "")) == normalized_date:
                return log
    return None


def evaluate_prediction(
    prediction_log: dict[str, Any],
    actual_results: list[dict[str, Any]],
) -> dict[str, Any]:
    if not actual_results:
        raise ValueError("actual_results is empty")

    actual_by_number = {
        _to_int(row.get("horse_number", row.get("馬番"))): _to_int(row.get("finish", row.get("着順")))
        for row in actual_results
        if _to_int(row.get("horse_number", row.get("馬番"))) > 0
    }
    prediction_rows = prediction_log.get("prediction_table", []) or []
    marked_rows = [row for row in prediction_rows if str(row.get("印", row.get("mark", ""))) in MARKS]
    marked_rows.sort(key=lambda row: MARKS.index(str(row.get("印", row.get("mark", "")))))
    ranked_rows = sorted(
        prediction_rows,
        key=lambda row: (
            -_to_float(row.get("prediction_score", row.get("score", 0.0))),
            -_to_float(row.get("win_rate", 0.0)),
            -_to_float(row.get("top3_rate", 0.0)),
        ),
    )

    mark_finishes: dict[str, int | None] = {}
    for mark in MARKS:
        row = next((item for item in marked_rows if str(item.get("印", item.get("mark", ""))) == mark), None)
        horse_number = _to_int(row.get("馬番", row.get("horse_number"))) if row else 0
        mark_finishes[mark] = actual_by_number.get(horse_number)

    marked_finishes = [finish for finish in mark_finishes.values() if finish is not None]
    predicted_top3_numbers = [
        _to_int(row.get("馬番", row.get("horse_number"))) for row in ranked_rows[:3]
    ]
    predicted_top5_numbers = [
        _to_int(row.get("馬番", row.get("horse_number"))) for row in ranked_rows[:5]
    ]
    actual_top3_numbers = {number for number, finish in actual_by_number.items() if 0 < finish <= 3}
    actual_top5_numbers = {number for number, finish in actual_by_number.items() if 0 < finish <= 5}
    marked_top3_count = sum(1 for finish in marked_finishes if finish <= 3)
    predicted_top3_hit_count = sum(1 for number in predicted_top3_numbers if 0 < actual_by_number.get(number, 999) <= 3)
    honmei_finish = mark_finishes.get("◎")
    top5_hit_count = len(set(predicted_top5_numbers) & actual_top5_numbers)
    winner_hit = bool(predicted_top5_numbers and actual_by_number.get(predicted_top5_numbers[0]) == 1)
    trifecta_box_hit = bool(actual_top3_numbers) and actual_top3_numbers.issubset(set(predicted_top5_numbers))
    metrics = {
        "mark_finishes": mark_finishes,
        "mark_results": mark_finishes,
        "marked_top3_count": marked_top3_count,
        "marked_top3_rate": marked_top3_count / len(marked_finishes) if marked_finishes else 0.0,
        "predicted_top3_hit_count": predicted_top3_hit_count,
        "predicted_top3_top3_rate": predicted_top3_hit_count / max(1, len(predicted_top3_numbers)),
        "win_hit": honmei_finish == 1,
        "winner_hit": winner_hit,
        "top3_hit": honmei_finish is not None and honmei_finish <= 3,
        "trifecta_candidate_hit": trifecta_box_hit,
        "top5_hit_count": top5_hit_count,
        "hit_summary": {
            "honmei_win": honmei_finish == 1,
            "honmei_top3": honmei_finish is not None and honmei_finish <= 3,
            "marked_top3_count": marked_top3_count,
            "predicted_top3_hit_count": predicted_top3_hit_count,
            "winner_hit": winner_hit,
            "trifecta_candidate_hit": trifecta_box_hit,
            "top5_hit_count": top5_hit_count,
        },
    }
    metrics["comment"] = _evaluation_comment(metrics)
    return metrics


def save_evaluation_log(
    *,
    prediction_log: dict[str, Any],
    actual_results: list[dict[str, Any]],
    evaluation: dict[str, Any] | None = None,
    payouts: dict[str, Any] | None = None,
    bets: dict[str, Any] | None = None,
    return_analysis: dict[str, Any] | None = None,
    race_metadata: dict[str, Any] | None = None,
    source_url: str | None = None,
    output_dir: str | Path = EVALUATION_LOG_DIR,
    duplicate_action: str = "skip",
) -> Path:
    evaluation = dict(evaluation or evaluate_prediction(prediction_log, actual_results))
    if return_analysis is not None:
        evaluation["return_analysis"] = return_analysis
    from analysis_reporter import analyze_prediction_failure, generate_race_review

    now = datetime.now()
    timestamp = now.isoformat(timespec="seconds")
    file_stamp = now.strftime("%Y%m%d_%H%M%S")
    failure_analysis = analyze_prediction_failure(prediction_log, actual_results, evaluation)
    evaluation["failure_analysis"] = failure_analysis
    review = generate_race_review(prediction_log, actual_results, evaluation)
    evaluation["race_review"] = review
    payload = {
        "log_type": "evaluation",
        "timestamp": timestamp,
        "race_id": prediction_log.get("race_id", ""),
        "race_name": prediction_log.get("race_name", ""),
        "race_date": prediction_log.get("race_date", ""),
        "prediction_timestamp": prediction_log.get("timestamp", ""),
        "prediction_log_path": prediction_log.get("_path", ""),
        "source_url": source_url or "",
        "race_metadata": race_metadata or {},
        "actual_result": actual_results,
        "payouts": payouts or {},
        "bets": bets or {},
        "return_analysis": return_analysis or evaluation.get("return_analysis", {}),
        "prediction_table": prediction_log.get("prediction_table", []),
        "mark_results": evaluation.get("mark_finishes", {}),
        "hit_summary": evaluation.get("hit_summary", {}),
        "evaluation": evaluation,
        "evaluation_metrics": evaluation,
        "failure_analysis": failure_analysis,
        "race_review": review,
    }
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    existing = find_matching_logs(
        race_id=str(payload["race_id"]),
        race_name=str(payload["race_name"]),
        race_date=str(payload["race_date"]),
        prediction_log_path=str(payload["prediction_log_path"]),
        log_dir=directory,
    )
    path = _resolve_save_path(
        directory=directory,
        base_name=f"evaluation_{file_stamp}.json",
        existing=existing,
        duplicate_action=duplicate_action,
    )
    csv_path = path.with_suffix(".csv")
    payload["csv_path"] = str(csv_path)
    _write_json(path, payload)
    actual_frame = pd.DataFrame(actual_results)
    if not actual_frame.empty:
        prediction_by_number = {
            _to_int(row.get("馬番", row.get("horse_number"))): row
            for row in prediction_log.get("prediction_table", [])
        }
        actual_frame["prediction_score"] = actual_frame.apply(
            lambda row: prediction_by_number.get(_to_int(row.get("horse_number", row.get("馬番"))), {}).get("prediction_score"),
            axis=1,
        )
        actual_frame["印"] = actual_frame.apply(
            lambda row: prediction_by_number.get(_to_int(row.get("horse_number", row.get("馬番"))), {}).get("印", ""),
            axis=1,
        )
    actual_frame.insert(0, "race_date", payload["race_date"])
    actual_frame.insert(0, "race_name", payload["race_name"])
    actual_frame.insert(0, "race_id", payload["race_id"])
    actual_frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return path


def load_evaluation_logs(output_dir: str | Path = EVALUATION_LOG_DIR) -> list[dict[str, Any]]:
    return _load_logs(output_dir)


def summarize_evaluations(logs: list[dict[str, Any]]) -> dict[str, Any]:
    if not logs:
        return {
            "race_count": 0,
            "honmei_win_rate": 0.0,
            "honmei_top3_rate": 0.0,
            "marked_top3_rate": 0.0,
            "predicted_top3_rate": 0.0,
            "average_hit_count": 0.0,
        }
    evaluations = [log.get("evaluation", {}) for log in logs]
    count = len(evaluations)
    return {
        "race_count": count,
        "honmei_win_rate": sum(bool(item.get("win_hit")) for item in evaluations) / count,
        "honmei_top3_rate": sum(bool(item.get("top3_hit")) for item in evaluations) / count,
        "marked_top3_rate": sum(_to_float(item.get("marked_top3_rate")) for item in evaluations) / count,
        "predicted_top3_rate": sum(_to_float(item.get("predicted_top3_top3_rate")) for item in evaluations) / count,
        "average_hit_count": sum(_to_float(item.get("top5_hit_count")) for item in evaluations) / count,
    }


def evaluation_logs_table(logs: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for log in reversed(logs):
        metrics = log.get("evaluation", {})
        mark_finishes = metrics.get("mark_finishes", {})
        rows.append(
            {
                "開催日": log.get("race_date", ""),
                "レース名": log.get("race_name", ""),
                "race_id": log.get("race_id", ""),
                "◎着順": mark_finishes.get("◎"),
                "○着順": mark_finishes.get("○"),
                "▲着順": mark_finishes.get("▲"),
                "△着順": mark_finishes.get("△"),
                "☆着順": mark_finishes.get("☆"),
                "印5頭の3着以内率": metrics.get("marked_top3_rate", 0.0),
                "予想上位3頭の3着以内率": metrics.get("predicted_top3_top3_rate", 0.0),
                "1着的中": bool(metrics.get("win_hit")),
                "複勝圏的中": bool(metrics.get("top3_hit")),
                "3連複候補的中": bool(metrics.get("trifecta_candidate_hit")),
                "コメント": metrics.get("comment", ""),
            }
        )
    return pd.DataFrame(rows)


def _compact_simulation_result(result: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "race_config",
        "single_result",
        "pace_prediction",
        "horse_analysis",
        "single_result_source",
        "timeline_mode",
        "renderer_name",
        "video_layout",
        "mp4_path",
    )
    compact = {key: result.get(key) for key in keys if key in result}
    timeline = result.get("race_timeline") or result.get("controlled_timeline") or []
    compact["timeline_frame_count"] = len(timeline) if isinstance(timeline, list) else 0
    compact["timeline_final_frame"] = timeline[-1] if isinstance(timeline, list) and timeline else None
    return compact


def _compact_trial(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    keys = (
        "trial_index",
        "seed",
        "pace",
        "ranking_distance",
        "top5_overlap_count",
        "representative_value_score",
        "top5_horses_in_selected_trial",
        "result_df",
        "ranking",
    )
    return {key: value.get(key) for key in keys if key in value}


def _records(value: object) -> list[dict[str, Any]]:
    if isinstance(value, pd.DataFrame):
        return value.to_dict("records")
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    return []


def _available_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 1000):
        candidate = path.with_name(f"{path.stem}_{index:03d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise OSError(f"could not allocate log path: {path}")


def _resolve_save_path(
    *,
    directory: Path,
    base_name: str,
    existing: list[dict[str, Any]],
    duplicate_action: str,
) -> Path:
    action = str(duplicate_action or "skip").strip().lower()
    aliases = {
        "保存しない": "skip",
        "上書き保存": "overwrite",
        "別名で保存": "rename",
    }
    action = aliases.get(action, action)
    if action not in {"skip", "overwrite", "rename"}:
        raise ValueError(f"unsupported duplicate_action: {duplicate_action}")
    if existing:
        existing_paths = [str(item.get("_path", "")) for item in existing if item.get("_path")]
        if action == "skip":
            raise DuplicateLogError(existing_paths)
        if action == "overwrite" and existing_paths:
            return Path(existing_paths[-1])
    return _available_path(directory / base_name)


def _load_logs(output_dir: str | Path) -> list[dict[str, Any]]:
    directory = Path(output_dir)
    if not directory.exists():
        return []
    logs: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["_path"] = str(path)
            logs.append(payload)
        except (OSError, json.JSONDecodeError):
            continue
    return logs


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return value.to_dict("records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def _evaluation_comment(metrics: dict[str, Any]) -> str:
    if metrics.get("win_hit"):
        return "本命が勝利し、中心馬の評価が結果につながりました。"
    if metrics.get("top3_hit"):
        return "本命は複勝圏を確保しましたが、勝ち切り評価には改善余地があります。"
    if _to_float(metrics.get("marked_top3_rate")) >= 0.4:
        return "印上位には実績馬を含められましたが、本命選定が課題でした。"
    return "上位評価と実着順の乖離が大きく、展開・適性補正の検証が必要です。"


def _normalize(value: Any) -> str:
    return "".join(str(value or "").split()).casefold()


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
