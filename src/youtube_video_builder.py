from __future__ import annotations

import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from narration_generator import (
    create_start_beep,
    estimate_narration_duration,
    generate_section_script,
    synthesize_narration_audio,
)
from report_generator import generate_prediction_report


YOUTUBE_VIDEO_DIR = Path("outputs/youtube_videos")
TMP_DIR = YOUTUBE_VIDEO_DIR / "tmp"
EXTERNAL_MEDIA_ASSETS_USED = False
REQUIRED_SECTION_ORDER = [
    "同レースの過去傾向",
    "全頭診断",
    "シミュレーション動画の再生",
    "注目馬の紹介と根拠説明",
]


def generate_race_trend_summary(
    prediction_log: dict,
    historical_logs: list[dict] | None = None,
) -> dict:
    """Summarize same-race trends from logs, or fall back to race conditions."""
    historical_logs = historical_logs or []
    existing_analysis = prediction_log.get("same_race_trend_analysis") or prediction_log.get("race_trend_analysis")
    if isinstance(existing_analysis, dict) and existing_analysis.get("summary_bullets"):
        return {
            "title": "同レースの過去傾向",
            "bullets": list(existing_analysis.get("summary_bullets", [])),
            "data_source": "same_race_history",
            "analysis": existing_analysis,
        }
    config = _dict(prediction_log.get("race_config") or prediction_log.get("race_metadata"))
    course = str(config.get("course", "対象コース"))
    surface = str(config.get("surface", "芝"))
    distance = _int(config.get("distance"), 0)
    direction = str(config.get("direction", ""))
    track_bias = str(config.get("track_bias", "標準"))
    prediction_rows = _frame(prediction_log.get("prediction_table"))

    if historical_logs:
        styles: list[str] = []
        winners: list[int] = []
        for log in historical_logs:
            actual = log.get("actual_result") or log.get("actual_results") or []
            prediction = _frame(log.get("prediction_table"))
            winner = _winner_number(actual)
            if winner > 0:
                winners.append(winner)
                match = prediction[pd.to_numeric(prediction.get("馬番", prediction.get("horse_number")), errors="coerce") == winner] if not prediction.empty else pd.DataFrame()
                if not match.empty:
                    styles.append(str(match.iloc[0].get("actual_running_style", match.iloc[0].get("primary_running_style", ""))))
        common_style = _mode([style for style in styles if style])
        bullets = [
            f"保存済み評価ログ{len(historical_logs)}件から同条件傾向を集計",
            f"勝ち馬の脚質傾向は{common_style or '分散'}",
            f"{course}{surface}{distance or ''}mは展開と馬場バイアスの影響を確認したい条件",
        ]
        data_source = "historical_logs"
    else:
        style_counts = prediction_rows.get("actual_running_style", prediction_rows.get("primary_running_style", pd.Series(dtype=str))).astype(str).value_counts().to_dict() if not prediction_rows.empty else {}
        front_count = int(style_counts.get("逃げ", 0) + style_counts.get("先行", 0))
        closer_count = int(style_counts.get("差し", 0) + style_counts.get("追込", 0))
        distance_comment = "短距離寄りで位置取りと序盤の反応が重要" if 0 < distance <= 1400 else "長めの距離で持続力と仕掛けどころが重要" if distance >= 2200 else "バランス型でペース判断が重要"
        bias_comment = {
            "前残り": "前残り想定のため、逃げ・先行勢の粘り込みに注意",
            "差し有利": "差し有利想定のため、上り性能の高い馬を重視",
            "外差し有利": "外差し有利なら、外枠の差し・追込が浮上しやすい",
            "内前有利": "内前有利なら、内枠の先行力が評価材料",
            "外伸び": "外伸び想定では、直線で外を通せる馬に注目",
        }.get(track_bias, "標準馬場なら能力と展開適性をバランスよく評価")
        bullets = [
            f"過去傾向データが不足しているため、{course}{surface}{distance or ''}m{direction}回りの条件から簡易推定",
            distance_comment,
            f"出走構成は前方型{front_count}頭、差し・追込型{closer_count}頭",
            bias_comment,
        ]
        data_source = "race_config_estimate"

    return {
        "title": "同レースの過去傾向",
        "bullets": bullets[:6],
        "data_source": data_source,
    }


def build_youtube_video_structure(
    prediction_log: dict,
    race_video_path: str = "",
    historical_logs: list[dict] | None = None,
) -> list[dict[str, Any]]:
    """Return the fixed YouTube section structure without rendering video."""
    trend = generate_race_trend_summary(prediction_log, historical_logs)
    comments = _full_diagnosis_table(_frame(prediction_log.get("comments_table")), _frame(prediction_log.get("prediction_table")))
    prediction = _frame(prediction_log.get("prediction_table"))
    featured = _featured_rows(prediction, comments, prediction_log)
    return [
        {"section": REQUIRED_SECTION_ORDER[0], "title": trend["title"], "bullets": trend["bullets"], "data_source": trend.get("data_source", "")},
        {"section": REQUIRED_SECTION_ORDER[1], "title": "全頭診断", "rows": _records(comments)},
        {"section": REQUIRED_SECTION_ORDER[2], "title": "AIシミュレーション再生", "race_video_path": str(race_video_path or prediction_log.get("video_path", ""))},
        {"section": REQUIRED_SECTION_ORDER[3], "title": "注目馬の紹介と根拠説明", "rows": featured},
    ]


def build_youtube_prediction_video(
    prediction_log: dict,
    race_video_path: str,
    output_path: str,
    video_format: str = "youtube",
    fps: int = 30,
    *,
    trend_section_sec: int | None = None,
    diagnosis_section_sec: int | None = None,
    featured_section_sec: int | None = None,
    bgm_path: str = "",
    race_bgm_path: str = "",
    start_se_path: str = "",
    use_narration: bool = True,
) -> str:
    """Build a YouTube-ready prediction video from original slides and the race video."""
    try:
        import imageio.v2 as imageio
    except Exception as exc:
        raise RuntimeError("YouTube動画生成には imageio と imageio-ffmpeg が必要です。") from exc

    output = _resolve_output_path(output_path, YOUTUBE_VIDEO_DIR, "youtube_prediction", ".mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    width, height = _video_dimensions(video_format)
    fps = max(1, int(fps))
    structure = build_youtube_video_structure(prediction_log, race_video_path)
    report = generate_prediction_report(prediction_log)
    section_scripts = {
        "trend": generate_section_script("trend", structure[0]),
        "diagnosis": generate_section_script("diagnosis", structure[1]),
        "featured": generate_section_script("featured", structure[3]),
    }
    section_durations = {
        "trend": float(trend_section_sec) if trend_section_sec else estimate_narration_duration(section_scripts["trend"], min_sec=6),
        "diagnosis": float(diagnosis_section_sec) if diagnosis_section_sec else estimate_narration_duration(section_scripts["diagnosis"], min_sec=10),
        "featured": float(featured_section_sec) if featured_section_sec else estimate_narration_duration(section_scripts["featured"], min_sec=12),
    }
    narration_paths = _generate_narrations(section_scripts, use_narration)
    metadata = {
        "section_order": [item["section"] for item in structure],
        "external_media_assets_used": EXTERNAL_MEDIA_ASSETS_USED,
        "race_video_path": str(race_video_path or ""),
        "video_format": video_format,
        "fps": fps,
        "section_scripts": section_scripts,
        "section_durations": section_durations,
        "narration_paths": narration_paths,
        "bgm_path": str(bgm_path or ""),
        "race_bgm_path": str(race_bgm_path or ""),
        "start_se_path": str(start_se_path or ""),
    }

    writer = imageio.get_writer(str(output), fps=fps, codec="libx264", quality=8, macro_block_size=1)
    try:
        _append_still(
            writer,
            _draw_trend_slide(structure[0], prediction_log, width, height),
            max(1, int(section_durations["trend"] * fps)),
        )
        diagnosis_slides = _draw_diagnosis_slides(structure[1], prediction_log, width, height, horses_per_slide=6)
        _append_slides_evenly(writer, diagnosis_slides, max(1, int(section_durations["diagnosis"] * fps)))
        _append_still(writer, _draw_title_card("AIシミュレーション再生", "代表試行のレース動画を再生します", prediction_log, width, height), max(1, fps * 2))
        inserted = _append_race_video(writer, race_video_path, width, height, fps, imageio)
        if not inserted:
            _append_still(writer, _draw_title_card("シミュレーション動画", "レース動画が未生成のため、ここでは構成カードを表示します", prediction_log, width, height), max(1, fps * 6))
        featured_slides = _draw_featured_slides(structure[3], prediction_log, width, height, report)
        _append_slides_evenly(writer, featured_slides, max(1, int(section_durations["featured"] * fps)))
    finally:
        writer.close()

    audio_attached = _attach_audio_if_possible(str(output), bgm_path=bgm_path, race_bgm_path=race_bgm_path, start_se_path=start_se_path)
    metadata["audio_attached"] = bool(audio_attached)
    output.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(output)


def _draw_trend_slide(section: dict[str, Any], prediction_log: dict, width: int, height: int) -> Image.Image:
    image, draw, fonts = _base_slide(prediction_log, width, height, section["title"])
    x = int(width * 0.09)
    y = int(height * 0.25)
    for bullet in section.get("bullets", []):
        y = _draw_wrapped(draw, f"・{bullet}", (x, y), fonts["body"], fill=(255, 255, 255, 245), max_width=int(width * 0.82), line_gap=int(height * 0.018))
        y += int(height * 0.018)
    if section.get("data_source") == "race_config_estimate":
        _draw_text_with_shadow(draw, (x, height * 0.83), "過去傾向データ不足: コース条件と出走馬構成から簡易推定", fonts["small"], fill=(255, 226, 105, 255))
    return image


def _draw_diagnosis_slides(section: dict[str, Any], prediction_log: dict, width: int, height: int, horses_per_slide: int) -> list[Image.Image]:
    rows = [row for row in section.get("rows", []) if isinstance(row, dict)]
    if not rows:
        return [_draw_title_card("全頭診断", "全頭短評データが不足しています", prediction_log, width, height)]
    slides: list[Image.Image] = []
    for page_index in range(0, len(rows), horses_per_slide):
        chunk = rows[page_index: page_index + horses_per_slide]
        image, draw, fonts = _base_slide(prediction_log, width, height, f"全頭診断 {page_index // horses_per_slide + 1}")
        x = int(width * 0.055)
        y = int(height * 0.20)
        row_h = int(height * 0.105)
        headers = ["馬番", "馬名", "脚質", "評価", "短評"]
        positions = [x, x + width * 0.075, x + width * 0.36, x + width * 0.46, x + width * 0.55]
        for header, hx in zip(headers, positions):
            draw.text((hx, y), header, font=fonts["small"], fill=(255, 238, 120, 255))
        y += int(height * 0.045)
        for row in chunk:
            draw.rounded_rectangle((x - 10, y - 8, width - x + 10, y + row_h - 12), radius=14, fill=(0, 0, 0, 94))
            values = [
                str(row.get("馬番", row.get("horse_number", ""))),
                str(row.get("馬名", row.get("horse_name", ""))),
                str(row.get("脚質", row.get("actual_running_style", row.get("primary_running_style", "")))),
                str(row.get("評価", row.get("rating", ""))),
            ]
            for idx, (value, vx) in enumerate(zip(values, positions[:4])):
                if idx == 1:
                    _draw_wrapped(draw, value, (vx, y), fonts["body"], fill=(255, 255, 255, 245), max_width=int(width * 0.27), line_gap=2, max_lines=2)
                else:
                    draw.text((vx, y), value, font=fonts["body"], fill=(255, 255, 255, 245))
            comment = str(row.get("短評", row.get("予想根拠", "")))
            _draw_wrapped(draw, comment, (positions[4], y), fonts["small"], fill=(240, 246, 240, 235), max_width=int(width * 0.38), line_gap=4, max_lines=2)
            y += row_h
        slides.append(image)
    return slides


def _draw_featured_slides(section: dict[str, Any], prediction_log: dict, width: int, height: int, report: dict) -> list[Image.Image]:
    rows = [row for row in section.get("rows", []) if isinstance(row, dict)]
    if not rows:
        return [_draw_title_card("注目馬", "印付き上位馬データがありません", prediction_log, width, height)]
    slides: list[Image.Image] = []
    comments = _comment_map(prediction_log)
    for row in rows:
        image, draw, fonts = _base_slide(prediction_log, width, height, "注目馬の紹介と根拠説明")
        mark = str(row.get("印", ""))
        number = str(row.get("馬番", row.get("horse_number", "")))
        name = str(row.get("馬名", row.get("horse_name", "")))
        style = str(row.get("actual_running_style", row.get("primary_running_style", "")))
        win = _percent(row.get("win_rate"))
        top3 = _percent(row.get("top3_rate"))
        draw.text((width * 0.08, height * 0.22), f"{mark} {number}番 {name}", font=fonts["headline"], fill=(255, 236, 92, 255))
        draw.text((width * 0.08, height * 0.36), f"脚質: {style}   勝率: {win}   複勝率: {top3}", font=fonts["body"], fill=(255, 255, 255, 245))
        reason = str(row.get("detailed_reason") or _reason_text(row))
        risk = str(row.get("risk") or _risk_text(row, prediction_log))
        y = _draw_wrapped(draw, f"根拠: {reason}", (width * 0.08, height * 0.47), fonts["body"], fill=(245, 250, 245, 245), max_width=int(width * 0.82), line_gap=8)
        _draw_wrapped(draw, f"リスク: {risk}", (width * 0.08, y + height * 0.04), fonts["body"], fill=(255, 225, 120, 245), max_width=int(width * 0.82), line_gap=8)
        slides.append(image)
    return slides


def _draw_title_card(title: str, subtitle: str, prediction_log: dict, width: int, height: int) -> Image.Image:
    image, draw, fonts = _base_slide(prediction_log, width, height, title)
    _draw_wrapped(draw, subtitle, (width * 0.10, height * 0.42), fonts["body"], fill=(255, 255, 255, 240), max_width=int(width * 0.80), line_gap=10)
    return image


def _base_slide(prediction_log: dict, width: int, height: int, title: str) -> tuple[Image.Image, ImageDraw.ImageDraw, dict[str, ImageFont.ImageFont]]:
    config = _dict(prediction_log.get("race_config") or prediction_log.get("race_metadata"))
    surface = str(config.get("surface", "芝"))
    image = Image.new("RGB", (width, height), "#255f35" if surface == "芝" else "#92643f")
    draw = ImageDraw.Draw(image, "RGBA")
    _draw_slide_background(draw, width, height, surface)
    fonts = _fonts(width, height)
    race_name = str(prediction_log.get("race_name") or config.get("race_name") or "対象レース")
    race_info = f"{race_name}  {config.get('course', '')} {config.get('surface', '')}{config.get('distance', '')}m"
    draw.rounded_rectangle((width * 0.04, height * 0.045, width * 0.96, height * 0.16), radius=20, fill=(0, 0, 0, 135))
    _draw_text_with_shadow(draw, (width * 0.065, height * 0.062), title, fonts["title"], fill=(255, 238, 100, 255))
    _draw_text_with_shadow(draw, (width * 0.065, height * 0.135), race_info, fonts["small"], fill=(235, 245, 255, 240))
    return image, draw, fonts


def _draw_slide_background(draw: ImageDraw.ImageDraw, width: int, height: int, surface: str) -> None:
    sky_h = int(height * 0.24)
    for y in range(sky_h):
        ratio = y / max(1, sky_h)
        color = (
            int(80 * (1 - ratio) + 175 * ratio),
            int(162 * (1 - ratio) + 215 * ratio),
            int(226 * (1 - ratio) + 246 * ratio),
            255,
        )
        draw.line((0, y, width, y), fill=color)
    base = (46, 130, 58, 255) if surface == "芝" else (166, 116, 72, 255)
    alt = (81, 160, 73, 90) if surface == "芝" else (203, 151, 98, 90)
    draw.rectangle((0, sky_h, width, height), fill=base)
    for x in range(-width, width * 2, max(42, width // 30)):
        draw.polygon([(x, sky_h), (x + width // 5, sky_h), (x + width, height), (x + width // 2, height)], fill=alt)
    draw.rectangle((0, 0, width, height), fill=(0, 0, 0, 45))


def _append_still(writer: Any, image: Image.Image, frame_count: int) -> None:
    array = np.asarray(image.convert("RGB"))
    for _ in range(max(1, int(frame_count))):
        writer.append_data(array)


def _append_slides_evenly(writer: Any, slides: list[Image.Image], total_frames: int) -> None:
    if not slides:
        return
    per_slide = max(1, math.ceil(total_frames / len(slides)))
    for slide in slides:
        _append_still(writer, slide, per_slide)


def _append_race_video(writer: Any, race_video_path: str, width: int, height: int, fps: int, imageio: Any) -> bool:
    path = Path(str(race_video_path or ""))
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        reader = imageio.get_reader(str(path))
    except Exception:
        return False
    try:
        for frame in reader:
            image = Image.fromarray(frame).convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
            writer.append_data(np.asarray(image))
    except Exception:
        return False
    finally:
        try:
            reader.close()
        except Exception:
            pass
    return True


def _marked_rows(prediction: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    if prediction.empty:
        return []
    order = {"◎": 0, "○": 1, "▲": 2, "△": 3, "☆": 4}
    rows = []
    for _, row in prediction.iterrows():
        data = row.to_dict()
        mark = str(data.get("印", data.get("mark", "")))
        if mark in order:
            data["印"] = mark
            rows.append(data)
    rows.sort(key=lambda item: order.get(str(item.get("印")), 99))
    return rows[:limit]


def _full_diagnosis_table(comments: pd.DataFrame, prediction: pd.DataFrame) -> pd.DataFrame:
    source = comments.copy() if not comments.empty else prediction.copy()
    if source.empty:
        return pd.DataFrame(columns=["馬番", "馬名", "脚質", "評価", "短評"])
    rename = {
        "horse_number": "馬番",
        "horse_name": "馬名",
        "actual_running_style": "脚質",
        "primary_running_style": "脚質",
        "rating": "評価",
        "comment": "短評",
        "予想根拠": "短評",
    }
    source = source.rename(columns={key: value for key, value in rename.items() if key in source.columns and value not in source.columns})
    for column in ["馬番", "馬名", "脚質", "評価", "短評"]:
        if column not in source.columns:
            source[column] = ""
    source["馬番"] = pd.to_numeric(source["馬番"], errors="coerce").fillna(999).astype(int)
    output = source[["馬番", "馬名", "脚質", "評価", "短評"]].sort_values("馬番").reset_index(drop=True)
    return output


def _featured_rows(prediction: pd.DataFrame, comments: pd.DataFrame, prediction_log: dict) -> list[dict[str, Any]]:
    rows = _marked_rows(prediction, limit=5)
    order = {"☆": 0, "△": 1, "▲": 2, "○": 3, "◎": 4}
    comment_by_number = _comment_map({"comments_table": comments.to_dict("records")})
    trend_analysis = prediction_log.get("same_race_trend_analysis") or prediction_log.get("race_trend_analysis") or {}
    trend_bullets = trend_analysis.get("summary_bullets", []) if isinstance(trend_analysis, dict) else []
    sorted_rows: list[dict[str, Any]] = []
    for row in rows:
        number = _int(row.get("馬番", row.get("horse_number", 0)))
        item = dict(row)
        short_comment = comment_by_number.get(number, "")
        item["detailed_reason"] = _detailed_reason(item, short_comment, trend_bullets, prediction_log)
        item["risk"] = _featured_risk(item, prediction_log, trend_bullets)
        sorted_rows.append(item)
    sorted_rows.sort(key=lambda item: order.get(str(item.get("印", "")), 99))
    return sorted_rows


def _detailed_reason(row: dict[str, Any], short_comment: str, trend_bullets: list[Any], prediction_log: dict) -> str:
    race_class = row.get("race_class", row.get("クラス", "近走上位クラス"))
    popularity = row.get("popularity", row.get("人気", ""))
    finish = row.get("finish", row.get("着順", ""))
    margin = row.get("margin", row.get("着差", ""))
    passing = row.get("passing_order", row.get("通過順", ""))
    last3f = row.get("last3f", row.get("上り", ""))
    trend = str(trend_bullets[0]) if trend_bullets else "過去10年傾向は中立評価"
    track_bias = str(_dict(prediction_log.get("race_config")).get("track_bias", "標準"))
    detail = (
        f"近走では{race_class}で人気{popularity or '-'}、着順{finish or '-'}、着差{margin or '-'}の内容を評価。"
        f"通過順{passing or '-'}、上り{last3f or '-'}から今回の展開適性を確認し、"
        f"{trend}。トラックバイアスは{track_bias}想定で、{short_comment or _reason_text(row)}"
    )
    return detail


def _featured_risk(row: dict[str, Any], prediction_log: dict, trend_bullets: list[Any]) -> str:
    style = str(row.get("actual_running_style", row.get("primary_running_style", "")))
    frame = _int(row.get("枠順", row.get("frame", 0)))
    weight = _float(row.get("斤量", row.get("carried_weight", 56.0)), 56.0) or 56.0
    track_condition = str(_dict(prediction_log.get("race_config")).get("track_condition", "良"))
    pace = str(_dict(prediction_log.get("pace_prediction")).get("pace", "medium"))
    risks = []
    if weight >= 58:
        risks.append("斤量負荷")
    if frame <= 2 and style in {"差し", "追込"}:
        risks.append("内で包まれるリスク")
    if frame >= 7 and style in {"逃げ", "先行"}:
        risks.append("外枠から脚を使うリスク")
    if track_condition in {"重", "不良"}:
        risks.append("道悪適性")
    if style in {"差し", "追込"} and pace == "slow":
        risks.append("スローで届かない展開")
    if style in {"逃げ", "先行"} and pace == "high":
        risks.append("ハイペース消耗")
    if trend_bullets and "外" in str(trend_bullets[0]) and frame <= 3:
        risks.append("過去傾向との枠順ズレ")
    return "、".join(risks) + "に注意。" if risks else "大きな減点は少ないが、直前の馬場と気配で最終判断したい。"


def _generate_narrations(section_scripts: dict[str, str], use_narration: bool) -> dict[str, str]:
    if not use_narration:
        return {}
    paths: dict[str, str] = {}
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for key, script in section_scripts.items():
        path = synthesize_narration_audio(script, str(Path("outputs/narration") / f"{key}_{stamp}.wav"))
        if path:
            paths[key] = path
    return paths


def _attach_audio_if_possible(
    video_path: str,
    *,
    bgm_path: str = "",
    race_bgm_path: str = "",
    start_se_path: str = "",
) -> bool:
    audio_sources = [path for path in [bgm_path, race_bgm_path, start_se_path] if path and Path(path).is_file()]
    generated_beep = ""
    if not start_se_path:
        generated_beep = create_start_beep(str(Path("outputs/narration") / f"start_beep_{datetime.now():%Y%m%d_%H%M%S}.wav"))
        if Path(generated_beep).is_file():
            audio_sources.append(generated_beep)
    if not audio_sources:
        return False
    try:
        from moviepy import AudioFileClip, CompositeAudioClip, VideoFileClip
    except Exception:
        try:
            from moviepy.editor import AudioFileClip, CompositeAudioClip, VideoFileClip  # type: ignore
        except Exception:
            return False
    source = Path(video_path)
    temp_output = source.with_name(f"{source.stem}_audio_tmp{source.suffix}")
    video_clip = None
    output_clip = None
    composite_clip = None
    clips = []
    try:
        video_clip = VideoFileClip(str(source))
        for audio_path in audio_sources:
            try:
                clip = AudioFileClip(str(audio_path))
            except Exception:
                continue
            if clip.duration > video_clip.duration:
                clip = clip.subclipped(0, video_clip.duration) if hasattr(clip, "subclipped") else clip.subclip(0, video_clip.duration)
            clips.append(clip)
        if not clips:
            return False
        composite_clip = CompositeAudioClip(clips)
        output_clip = video_clip.with_audio(composite_clip) if hasattr(video_clip, "with_audio") else video_clip.set_audio(composite_clip)
        output_clip.write_videofile(str(temp_output), codec="libx264", audio_codec="aac", logger=None)
        shutil.move(str(temp_output), str(source))
        return True
    except Exception:
        if temp_output.exists():
            temp_output.unlink(missing_ok=True)
        return False
    finally:
        if output_clip is not None and output_clip is not video_clip:
            try:
                output_clip.close()
            except Exception:
                pass
        if composite_clip is not None:
            try:
                composite_clip.close()
            except Exception:
                pass
        for clip in clips:
            try:
                clip.close()
            except Exception:
                pass
        if video_clip is not None:
            try:
                video_clip.close()
            except Exception:
                pass


def _comment_map(prediction_log: dict) -> dict[int, str]:
    comments = _frame(prediction_log.get("comments_table"))
    output: dict[int, str] = {}
    for _, row in comments.iterrows():
        number = _int(row.get("馬番", row.get("horse_number")), 0)
        if number:
            output[number] = str(row.get("短評", row.get("comment", "")))
    return output


def _reason_text(row: dict[str, Any]) -> str:
    parts = []
    for label, key in [
        ("総合能力", "horse_ability_score"),
        ("上り性能", "late_kick_score"),
        ("コース適性", "course_fit_score"),
        ("展開適性", "pace_fit_score"),
        ("騎手補正", "jockey_score"),
    ]:
        value = _float(row.get(key), None)
        if value is not None and value >= 60:
            parts.append(f"{label}が{value:.0f}点")
    return "、".join(parts) + "で評価を上げています。" if parts else "AI予想印とシミュレーション結果を総合して評価しています。"


def _risk_text(row: dict[str, Any], prediction_log: dict) -> str:
    style = str(row.get("actual_running_style", row.get("primary_running_style", "")))
    pace = str(_dict(prediction_log.get("pace_prediction")).get("pace", "medium"))
    if style in {"差し", "追込"} and pace == "slow":
        return "スローペースでは届き切らないリスクがあります。"
    if style in {"逃げ", "先行"} and pace == "high":
        return "前半が厳しくなると終盤の粘りが課題です。"
    return "馬場傾向や直前気配によって評価が変動します。"


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[float, float],
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
    max_width: int,
    line_gap: int,
    max_lines: int | None = None,
) -> int:
    x, y = xy
    lines = _wrap_text(draw, text, font, max_width)
    if max_lines is not None:
        lines = lines[:max_lines]
    line_height = _line_height(draw, font) + line_gap
    for index, line in enumerate(lines):
        _draw_text_with_shadow(draw, (x, y + index * line_height), line, font, fill)
    return int(y + len(lines) * line_height)


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for char in str(text):
        candidate = current + char
        if current and draw.textlength(candidate, font=font) > max_width:
            lines.append(current)
            current = char
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]


def _draw_text_with_shadow(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, font: ImageFont.ImageFont, fill: tuple[int, int, int, int]) -> None:
    x, y = xy
    draw.text((x + 2, y + 2), text, font=font, fill=(0, 0, 0, 190))
    draw.text((x, y), text, font=font, fill=fill)


def _fonts(width: int, height: int) -> dict[str, ImageFont.ImageFont]:
    return {
        "title": _font(int(width * 0.038), bold=True),
        "headline": _font(int(width * 0.050), bold=True),
        "body": _font(int(width * 0.026), bold=False),
        "small": _font(int(width * 0.019), bold=False),
    }


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/meiryob.ttc" if bold else "C:/Windows/Fonts/meiryo.ttc",
        "C:/Windows/Fonts/YuGothB.ttc" if bold else "C:/Windows/Fonts/YuGothM.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            if Path(path).exists():
                return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _line_height(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont) -> int:
    bbox = draw.textbbox((0, 0), "あいうABC", font=font)
    return max(1, bbox[3] - bbox[1])


def _video_dimensions(video_format: str) -> tuple[int, int]:
    if "tiktok" in str(video_format).lower() or "9:16" in str(video_format):
        return 1080, 1920
    return 1920, 1080


def _resolve_output_path(output_path: str, default_dir: Path, prefix: str, suffix: str) -> Path:
    if not output_path:
        return default_dir / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}{suffix}"
    path = Path(output_path)
    if path.suffix.lower() != suffix:
        return path / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}{suffix}"
    return path


def _winner_number(actual: Any) -> int:
    for row in actual if isinstance(actual, list) else []:
        if _int(row.get("finish", row.get("着順")), 999) == 1:
            return _int(row.get("horse_number", row.get("馬番")), 0)
    return 0


def _mode(values: list[str]) -> str:
    if not values:
        return ""
    counts = pd.Series(values).value_counts()
    return str(counts.index[0]) if not counts.empty else ""


def _records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, pd.DataFrame):
        return value.to_dict("records")
    if isinstance(value, list):
        return [dict(row) for row in value if isinstance(row, dict)]
    return []


def _frame(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, list):
        return pd.DataFrame(value)
    return pd.DataFrame()


def _dict(value: Any) -> dict:
    return dict(value) if isinstance(value, dict) else {}


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _percent(value: Any) -> str:
    number = _float(value, 0.0) or 0.0
    return f"{number:.1%}" if number <= 1.0 else f"{number:.1f}%"
