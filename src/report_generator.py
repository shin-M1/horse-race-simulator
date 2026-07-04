from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


MARKS = ("◎", "○", "▲", "△", "☆")
REPORT_DIR = Path("outputs/reports")


def generate_prediction_report(
    prediction_log: dict,
    include_simulation_summary: bool = True,
    include_horse_comments: bool = True,
) -> dict:
    """Create publication-ready copy from a saved or in-memory prediction log."""
    config = _dict(prediction_log.get("race_config"))
    metadata = _dict(prediction_log.get("race_metadata"))
    prediction = _frame(prediction_log.get("prediction_table"))
    analysis = _frame(prediction_log.get("horse_analysis"))
    comments = _frame(prediction_log.get("comments_table"))
    pace = _dict(
        prediction_log.get("pace_prediction")
        or _dict(prediction_log.get("simulation_result")).get("pace_prediction")
    )
    race_name = str(prediction_log.get("race_name") or metadata.get("race_name") or "対象レース")
    race_date = str(prediction_log.get("race_date") or metadata.get("race_date") or "")
    marks = _marks_table(prediction, comments)
    if include_horse_comments and comments.empty:
        comments = _fallback_comments(prediction, analysis)
    elif not include_horse_comments:
        comments = pd.DataFrame()

    honmei = marks.iloc[0].to_dict() if not marks.empty else {}
    pace_label = _pace_label(pace.get("pace"))
    title = f"{race_name} AI予想｜本命は{honmei.get('馬番', '-')}番、展開のカギは前半ペース"
    race_info = _race_info(race_name, race_date, config, metadata, len(prediction))
    pace_text = _pace_text(pace, config)
    top_picks = _top_picks(marks)
    simulation = _simulation_text(prediction_log) if include_simulation_summary else ""
    risks = _risk_text(prediction, analysis, config, pace)
    marks_text = _marks_text(marks)
    youtube = (
        f"【オープニング】\n今回は{race_name}のAI予想を行っていきます。\n\n"
        f"【レース概要】\n{race_info}\n\n【展開予測】\n{pace_text}\n\n"
        f"【AI予想印】\n{marks_text}\n\n【注目馬解説】\n{top_picks}\n\n"
        f"【シミュレーション結果】\n{simulation or 'シミュレーション概要は省略します。'}\n\n"
        f"【まとめ】\n{risks}\n当日の馬場と気配も確認して最終判断してください。"
    )
    blog = (
        f"# {title}\n\n## レース概要\n{race_info}\n\n## 展開予測\n{pace_text}\n\n"
        f"## AI印\n{marks_text}\n\n## 注目馬\n{top_picks}\n\n"
        f"## 全頭短評\n{_comments_text(comments)}\n\n## 注意点\n{risks}\n\n"
        "## まとめ\n本予想は取得データとシミュレーション条件に基づく参考情報です。"
    )
    sns_lines = [f"【{race_name}AI予想】"]
    sns_lines.extend(f"{row['印']} {row['馬番']}番 {row['馬名']}" for _, row in marks.head(5).iterrows())
    sns_lines.extend([f"展開は{pace_label}想定。馬場と直前気配に注意。", "#競馬予想 #AI予想"])
    return {
        "title": title,
        "summary": f"{race_info}\n{pace_text}\n{top_picks}",
        "race_info": race_info,
        "pace_prediction": pace_text,
        "marks_table": marks,
        "top_picks": top_picks,
        "horse_comments": comments,
        "simulation_summary": simulation,
        "risk_factors": risks,
        "youtube_script": youtube,
        "blog_text": blog,
        "sns_text": "\n".join(sns_lines),
    }


def save_prediction_report(
    report: dict[str, Any],
    prediction_log_path: str = "",
    output_dir: str | Path = REPORT_DIR,
) -> dict[str, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    markdown_path = _available_path(directory / f"report_{now:%Y%m%d_%H%M%S}.md")
    json_path = markdown_path.with_suffix(".json")
    payload = {
        "prediction_log_path": str(prediction_log_path or ""),
        "generated_report": _json_ready(report),
        "timestamp": now.isoformat(timespec="seconds"),
    }
    markdown_path.write_text(_markdown(report, prediction_log_path), encoding="utf-8")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"markdown": markdown_path, "json": json_path}


def _marks_table(prediction: pd.DataFrame, comments: pd.DataFrame) -> pd.DataFrame:
    columns = ["印", "馬番", "馬名", "脚質", "勝率", "複勝率", "score", "短評"]
    comment_map = {
        _int(row.get("馬番")): str(row.get("短評", "")) for _, row in comments.iterrows()
    } if not comments.empty else {}
    rows = []
    for _, row in prediction.iterrows():
        mark = str(row.get("印", ""))
        if mark not in MARKS:
            continue
        number = _int(row.get("馬番", row.get("horse_number")))
        rows.append({
            "印": mark,
            "馬番": number,
            "馬名": str(row.get("馬名", row.get("horse_name", ""))),
            "脚質": str(row.get("actual_running_style", row.get("primary_running_style", ""))),
            "勝率": _float(row.get("win_rate")),
            "複勝率": _float(row.get("top3_rate")),
            "score": _float(row.get("prediction_score", row.get("score"))),
            "短評": comment_map.get(number, str(row.get("予想根拠", ""))),
        })
    rows.sort(key=lambda row: MARKS.index(row["印"]))
    return pd.DataFrame(rows, columns=columns)


def _fallback_comments(prediction: pd.DataFrame, analysis: pd.DataFrame) -> pd.DataFrame:
    analysis_map = {
        str(row.get("horse_name", "")): row for _, row in analysis.iterrows()
    } if not analysis.empty else {}
    rows = []
    for _, row in prediction.iterrows():
        name = str(row.get("馬名", row.get("horse_name", "")))
        details = analysis_map.get(name, {})
        style = str(row.get("actual_running_style", row.get("primary_running_style", details.get("primary_running_style", ""))))
        rows.append({
            "印": str(row.get("印", "")), "馬番": _int(row.get("馬番", row.get("horse_number"))),
            "馬名": name, "脚質": style,
            "短評": str(row.get("予想根拠", "")) or f"{style or '脚質不明'}での展開適合と総合能力が評価の鍵です。",
        })
    return pd.DataFrame(rows)


def _race_info(name: str, race_date: str, config: dict, metadata: dict, count: int) -> str:
    value = lambda key, default="-": config.get(key, metadata.get(key, default))
    return (
        f"{name}（{race_date or '-'}） / {value('course')} / {value('surface')}{value('distance', 0)}m / "
        f"{value('weather')} / 馬場 {value('track_condition')} / "
        f"トラックバイアス {value('track_bias', '標準')} / {count or _int(value('field_size', 0))}頭"
    )


def _pace_text(pace: dict, config: dict) -> str:
    comment = str(pace.get("comment") or {
        "high": "前が競り合えば差し・追込の進出余地が広がります。",
        "slow": "前方勢が余力を残しやすく、早めの位置取りが重要です。",
    }.get(str(pace.get("pace")), "前後の有利不利が極端になりにくい想定です。"))
    return (
        f"ペース予測は{_pace_label(pace.get('pace'))}。先頭候補は{_join(pace.get('front_group'))}、"
        f"中団候補は{_join(pace.get('middle_group'))}、後方候補は{_join(pace.get('back_group'))}。"
        f"トラックバイアスは{config.get('track_bias', '標準')}。{comment}"
    )


def _top_picks(marks: pd.DataFrame) -> str:
    by_mark = {str(row["印"]): row for _, row in marks.iterrows()}
    parts = []
    for mark, label in (("◎", "本命"), ("○", "対抗"), ("☆", "穴")):
        row = by_mark.get(mark)
        if row is not None:
            parts.append(f"{label}は{row['馬番']}番{row['馬名']}です。{row.get('短評') or '総合評価を重視しました。'}")
    return "".join(parts) or "注目馬を選定できませんでした。"


def _simulation_text(log: dict) -> str:
    single_result = log.get("single_result")
    if single_result is None:
        single_result = _dict(log.get("simulation_result")).get("single_result")
    result = _frame(single_result)
    order = "、".join(
        f"{_int(row.get('着順', row.get('rank')))}着 {_int(row.get('馬番', row.get('horse_number')))}番"
        for _, row in result.head(5).iterrows()
    ) or "着順データなし"
    video = str(log.get("video_path") or _dict(log.get("simulation_result")).get("mp4_path") or "未生成")
    return f"AI期待値最大の代表試行です。上位は{order}。動画: {video}。終盤の上りと粘り合いが見どころです。"


def _risk_text(prediction: pd.DataFrame, analysis: pd.DataFrame, config: dict, pace: dict) -> str:
    risks = []
    if str(pace.get("pace")) in {"slow", "high"}:
        risks.append("想定ペースが外れた場合、前後の有利不利が逆転する可能性があります。")
    if not analysis.empty and "mud_source" in analysis and (analysis["mud_source"].astype(str) == "neutral").any():
        risks.append("道悪適性を中立評価とした馬がいます。")
    if not analysis.empty and "primary_running_style" in analysis and (analysis["primary_running_style"].astype(str) == "自在").any():
        risks.append("自在型の位置取りは流れに左右されます。")
    if not analysis.empty and "agari_reliability" in analysis:
        reliability = pd.to_numeric(analysis["agari_reliability"], errors="coerce").fillna(0)
        if (reliability < 40).any():
            risks.append("上りデータが少なく、末脚評価に不確実性のある馬がいます。")
    weight_key = "斤量" if "斤量" in prediction else "carried_weight"
    if weight_key in prediction and (pd.to_numeric(prediction[weight_key], errors="coerce").fillna(0) >= 58).any():
        risks.append("重い斤量が終盤の持続力へ影響する可能性があります。")
    if "popularity_score" in prediction and (pd.to_numeric(prediction["popularity_score"], errors="coerce").fillna(50) < 45).any():
        risks.append("人気薄の評価は展開次第で振れやすい点に注意が必要です。")
    if str(config.get("track_condition")) in {"重", "不良"}:
        risks.append("道悪では時計と脚質傾向が変化する可能性があります。")
    return " ".join(dict.fromkeys(risks)) or "ペースと位置取りの変化には注意が必要です。"


def _markdown(report: dict, path: str) -> str:
    return (
        f"# {report.get('title', '')}\n\n予想ログ: `{path or '-'}`\n\n"
        f"## レース概要\n{report.get('race_info', '')}\n\n## 展開予測\n{report.get('pace_prediction', '')}\n\n"
        f"## AI予想印\n{_table_text(_frame(report.get('marks_table')))}\n\n"
        f"## 全頭短評\n{_table_text(_frame(report.get('horse_comments')))}\n\n"
        f"## シミュレーション概要\n{report.get('simulation_summary', '')}\n\n"
        f"## リスク要素\n{report.get('risk_factors', '')}\n\n"
        f"## YouTube台本\n{report.get('youtube_script', '')}\n\n"
        f"## ブログ本文\n{report.get('blog_text', '')}\n\n## SNS投稿文\n{report.get('sns_text', '')}\n"
    )


def _marks_text(frame: pd.DataFrame) -> str:
    return "\n".join(f"{row['印']} {row['馬番']}番 {row['馬名']}（{row['脚質'] or '脚質不明'}）" for _, row in frame.iterrows()) or "予想印なし"


def _comments_text(frame: pd.DataFrame) -> str:
    return "\n".join(f"- {row.get('印', '')}{row.get('馬番', '')}番 {row.get('馬名', '')}: {row.get('短評', '')}" for _, row in frame.iterrows()) or "短評データなし"


def _table_text(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "データなし"
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return frame.to_csv(index=False)


def _pace_label(value: Any) -> str:
    return {"high": "ハイペース", "slow": "スローペース", "medium": "ミドルペース"}.get(str(value), "ミドルペース")


def _join(value: Any) -> str:
    return "、".join(str(item) for item in value) if isinstance(value, (list, tuple)) and value else str(value or "該当なし")


def _frame(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    return pd.DataFrame(value) if isinstance(value, list) else pd.DataFrame()


def _dict(value: Any) -> dict:
    return dict(value) if isinstance(value, dict) else {}


def _json_ready(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return value.to_dict("records")
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _available_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 1000):
        candidate = path.with_name(f"{path.stem}_{index:03d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise OSError(f"could not allocate report path: {path}")


def _int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
