from __future__ import annotations

import math
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

try:
    from animation import FRAME_COLORS, get_text_color
except ImportError:
    FRAME_COLORS = {
        1: "#FFFFFF",
        2: "#000000",
        3: "#FF0000",
        4: "#0000FF",
        5: "#FFFF00",
        6: "#008000",
        7: "#FFA500",
        8: "#FFC0CB",
    }

    def get_text_color(frame: int) -> str:
        if frame in [1, 5, 8]:
            return "#000000"
        return "#FFFFFF"


VIDEO_FORMATS = {
    "youtube": (1920, 1080),
    "YouTube横長 16:9": (1920, 1080),
    "tiktok": (1080, 1920),
    "TikTok縦長 9:16": (1080, 1920),
}
RESULT_DISPLAY_SEC = 3


def render_race_video(
    simulation_result: dict[str, Any] | Any,
    race_config: dict[str, Any] | Any,
    horses: list[dict[str, Any]],
    output_path: str = "outputs/race_movie.mp4",
    video_format: str = "youtube",
    fps: int = 30,
    duration_sec: int = 60,
    prediction_table: pd.DataFrame | None = None,
    renderer_info: dict[str, str] | None = None,
) -> str:
    """Render an original CG-style race video as MP4.

    Blender is attempted first when available. If it is not installed or fails,
    a pure-Python Pillow renderer is used and encoded with imageio.
    """
    if isinstance(simulation_result, dict) and simulation_result.get("race_timeline"):
        rendered_path = render_race_video_from_timeline(
            race_timeline=simulation_result["race_timeline"],
            race_config=race_config,
            horses=horses,
            output_path=output_path,
            video_format=video_format,
            fps=fps,
            duration_sec=duration_sec,
            prediction_table=prediction_table if prediction_table is not None else _extract_prediction_table(simulation_result),
            renderer_info=renderer_info,
        )
        return rendered_path

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    width, height = _video_dimensions(video_format)
    duration_sec = max(5, int(duration_sec))
    fps = max(8, int(fps))

    sections = _extract_sections(simulation_result)
    ranking = _extract_ranking(simulation_result)
    prediction_table = prediction_table if prediction_table is not None else _extract_prediction_table(simulation_result)
    if sections.empty:
        raise ValueError("投稿用MP4を生成するためのシミュレーション区間データがありません。")

    render_payload = _build_render_payload(
        sections=sections,
        ranking=ranking,
        race_config=race_config,
        horses=horses,
        prediction_table=prediction_table,
        width=width,
        height=height,
        fps=fps,
        duration_sec=duration_sec,
    )

    if shutil.which("blender"):
        try:
            from blender_renderer import render_with_blender

            rendered_path = render_with_blender(render_payload, str(output))
            if renderer_info is not None:
                renderer_info["renderer_name"] = "Blender marker renderer"
                renderer_info["horse_display_mode"] = "marker"
            return rendered_path
        except Exception:
            # The Pillow renderer below is intentionally robust enough to keep
            # the Streamlit app usable when Blender is missing or misconfigured.
            pass

    rendered_path = _render_with_pillow(render_payload, str(output))
    if renderer_info is not None:
        renderer_info["renderer_name"] = "Pillow/imageio marker renderer"
        renderer_info["horse_display_mode"] = "marker"
    return rendered_path


def render_race_video_from_timeline(
    race_timeline: list[dict],
    race_config: dict[str, Any] | Any,
    horses: list[dict[str, Any]],
    output_path: str,
    video_format: str = "youtube",
    fps: int = 30,
    duration_sec: int = 60,
    prediction_table: pd.DataFrame | None = None,
    renderer_info: dict[str, str] | None = None,
) -> str:
    """Legacy overview renderer from a precomputed race_timeline."""
    if not race_timeline:
        raise ValueError("race_timeline is empty")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    width, height = _video_dimensions(video_format)
    fps = max(8, int(fps))
    duration_sec = max(1, int(duration_sec))
    total_frames = max(2, fps * duration_sec)
    result_total_frames = _result_frame_count(total_frames, fps)
    race_total_frames = total_frames - result_total_frames
    interpolated_timeline = _timeline_with_result_display(race_timeline, total_frames, fps)
    payload = _build_render_payload_from_timeline(
        race_timeline=interpolated_timeline,
        race_config=race_config,
        horses=horses,
        prediction_table=prediction_table,
        width=width,
        height=height,
        fps=fps,
        duration_sec=duration_sec,
    )
    rendered_path = _render_with_pillow(payload, str(output))
    rendered = Path(rendered_path)
    if not rendered.is_file() or rendered.stat().st_size <= 0:
        raise RuntimeError(f"動画ファイルを作成できませんでした: {rendered_path}")
    if renderer_info is not None:
        renderer_info["renderer_name"] = "Pillow/imageio timeline marker renderer"
        renderer_info["horse_display_mode"] = "marker"
        renderer_info["video_layout"] = "legacy_overview"
        renderer_info["duration_sec"] = str(duration_sec)
        renderer_info["fps"] = str(fps)
        renderer_info["total_frames"] = str(total_frames)
        renderer_info["race_duration_sec"] = f"{race_total_frames / fps:.3f}"
        renderer_info["result_display_sec"] = f"{result_total_frames / fps:.3f}"
        renderer_info["race_total_frames"] = str(race_total_frames)
        renderer_info["result_total_frames"] = str(result_total_frames)
        renderer_info["final_result_first_frame_index"] = str(race_total_frames)
        renderer_info["direction"] = str(payload.get("direction", ""))
        renderer_info["render_direction"] = str(payload.get("render_direction", ""))
        renderer_info["start_gate_side"] = str(payload.get("start_gate_side", ""))
        renderer_info["goal_side"] = str(payload.get("goal_side", ""))
    return rendered_path


def render_side_scroll_race_video(
    race_timeline: list[dict],
    race_config: dict[str, Any] | Any,
    horses: list[dict[str, Any]],
    output_path: str,
    video_format: str = "youtube",
    fps: int = 30,
    duration_sec: int = 60,
    prediction_table: pd.DataFrame | None = None,
    renderer_info: dict[str, str] | None = None,
) -> str:
    """Render a side-scrolling race video from race_timeline."""
    if not race_timeline:
        raise ValueError("race_timeline is empty")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    width, height = _video_dimensions(video_format)
    fps = max(8, int(fps))
    duration_sec = max(1, int(duration_sec))
    total_frames = max(2, fps * duration_sec)
    result_total_frames = _result_frame_count(total_frames, fps)
    race_total_frames = total_frames - result_total_frames
    interpolated_timeline = _timeline_with_result_display(race_timeline, total_frames, fps)
    payload = _build_side_scroll_payload(
        race_timeline=interpolated_timeline,
        race_config=race_config,
        horses=horses,
        prediction_table=prediction_table,
        width=width,
        height=height,
        fps=fps,
        duration_sec=duration_sec,
    )
    rendered_path = _render_with_pillow(payload, str(output))
    rendered = Path(rendered_path)
    if not rendered.is_file() or rendered.stat().st_size <= 0:
        raise RuntimeError(f"動画ファイルを作成できませんでした: {rendered_path}")
    if renderer_info is not None:
        renderer_info["renderer_name"] = "Pillow/imageio side-scroll marker renderer"
        renderer_info["horse_display_mode"] = "marker"
        renderer_info["video_layout"] = "side_scroll"
        renderer_info["duration_sec"] = str(duration_sec)
        renderer_info["fps"] = str(fps)
        renderer_info["total_frames"] = str(total_frames)
        renderer_info["race_duration_sec"] = f"{race_total_frames / fps:.3f}"
        renderer_info["result_display_sec"] = f"{result_total_frames / fps:.3f}"
        renderer_info["race_total_frames"] = str(race_total_frames)
        renderer_info["result_total_frames"] = str(result_total_frames)
        renderer_info["final_result_first_frame_index"] = str(race_total_frames)
    return rendered_path


def interpolate_timeline(race_timeline: list[dict], total_frames: int) -> list[dict]:
    """Resample race_timeline to total_frames using linear position interpolation."""
    valid_frames = [frame for frame in race_timeline if isinstance(frame, dict) and isinstance(frame.get("horses"), list)]
    if not valid_frames:
        return []
    if total_frames <= 1:
        return [valid_frames[-1]]

    source_progress = np.array(
        [float(frame.get("progress", index / max(1, len(valid_frames) - 1))) for index, frame in enumerate(valid_frames)],
        dtype=float,
    )
    if np.any(np.diff(source_progress) < 0):
        order = np.argsort(source_progress)
        source_progress = source_progress[order]
        valid_frames = [valid_frames[int(index)] for index in order]

    target_progress = np.linspace(0.0, 1.0, total_frames)
    output: list[dict[str, Any]] = []
    for frame_index, progress in enumerate(target_progress):
        right_index = int(np.searchsorted(source_progress, progress, side="left"))
        if right_index <= 0:
            blended = _copy_timeline_frame(valid_frames[0], progress, frame_index)
        elif right_index >= len(valid_frames):
            blended = _copy_timeline_frame(valid_frames[-1], progress, frame_index)
        else:
            left = valid_frames[right_index - 1]
            right = valid_frames[right_index]
            left_progress = float(source_progress[right_index - 1])
            right_progress = float(source_progress[right_index])
            ratio = 0.0 if right_progress == left_progress else (progress - left_progress) / (right_progress - left_progress)
            blended = _blend_timeline_frames(left, right, float(ratio), progress, frame_index)
        output.append(blended)
    return output


def _timeline_with_result_display(race_timeline: list[dict], total_frames: int, fps: int, result_display_sec: int = RESULT_DISPLAY_SEC) -> list[dict]:
    """Split the video into race frames and post-goal result frames."""
    total_frames = max(2, int(total_frames))
    result_frames = _result_frame_count(total_frames, fps, result_display_sec)
    race_frames = max(2, total_frames - result_frames)
    frames = interpolate_timeline(race_timeline, race_frames)
    if not frames:
        return []

    for index, frame in enumerate(frames):
        frame["index"] = index
        frame["is_post_goal_frame"] = False
        frame["final_result_display_started"] = False

    final_frame = dict(frames[-1])
    final_frame["progress"] = 1.0
    final_frame["is_post_goal_frame"] = True
    final_frame["final_result_display_started"] = True
    final_time = float(final_frame.get("time", 0.0))
    while len(frames) < total_frames:
        held = {
            key: ([dict(horse) for horse in value] if key == "horses" and isinstance(value, list) else value)
            for key, value in final_frame.items()
        }
        held["index"] = len(frames)
        held["time"] = final_time + (len(frames) - race_frames + 1) / max(1, fps)
        held["progress"] = 1.0
        held["is_post_goal_frame"] = True
        held["final_result_display_started"] = True
        frames.append(held)
    return frames[:total_frames]


def _result_frame_count(total_frames: int, fps: int, result_display_sec: int = RESULT_DISPLAY_SEC) -> int:
    desired_result_frames = max(1, int(max(0, result_display_sec) * max(1, fps)))
    return min(desired_result_frames, max(0, int(total_frames) - 2))


def _should_show_final_result(frame: dict[str, Any], distance: float) -> bool:
    return bool(frame.get("is_post_goal_frame", False) and frame.get("final_result_display_started", False))


def _render_with_pillow(payload: dict[str, Any], output_path: str) -> str:
    try:
        import imageio.v2 as imageio
    except Exception as exc:
        raise RuntimeError(
            "投稿用MP4の生成には imageio と imageio-ffmpeg が必要です。"
            " `pip install -r requirements.txt` を実行してください。"
        ) from exc

    frames = payload["frames"]
    fps = int(payload["fps"])
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8, macro_block_size=1)
    try:
        for frame in frames:
            if payload.get("video_layout") == "side_scroll":
                image = _draw_side_scroll_frame(payload, frame)
            else:
                image = _draw_frame(payload, frame)
            writer.append_data(np.asarray(image.convert("RGB")))
    finally:
        writer.close()
    return output_path


def _copy_timeline_frame(frame: dict[str, Any], progress: float, frame_index: int) -> dict[str, Any]:
    copied_horses = [dict(horse) for horse in frame.get("horses", []) if isinstance(horse, dict)]
    copied_horses = _rerank_timeline_horses(copied_horses)
    return {
        "time": float(frame.get("time", 0.0)),
        "progress": round(float(progress), 5),
        "index": frame_index,
        "horses": copied_horses,
    }


def _blend_timeline_frames(
    left: dict[str, Any],
    right: dict[str, Any],
    ratio: float,
    progress: float,
    frame_index: int,
) -> dict[str, Any]:
    left_horses = {
        int(horse.get("horse_number", 0) or 0): horse
        for horse in left.get("horses", [])
        if isinstance(horse, dict)
    }
    right_horses = {
        int(horse.get("horse_number", 0) or 0): horse
        for horse in right.get("horses", [])
        if isinstance(horse, dict)
    }
    horse_numbers = sorted(set(left_horses) | set(right_horses))
    horses: list[dict[str, Any]] = []
    for number in horse_numbers:
        lhorse = left_horses.get(number, right_horses.get(number, {}))
        rhorse = right_horses.get(number, lhorse)
        row = dict(rhorse or lhorse)
        for key in [
            "position_m",
            "lane",
            "gap_from_leader",
            "performance_index",
            "final_performance_score",
            "late_power",
            "normalized_final_performance",
            "normalized_late_power",
            "late_ratio",
            "gap_adjustment",
            "tie_breaker",
            "race_power",
        ]:
            if key in lhorse or key in rhorse:
                row[key] = _lerp(_to_float_or(lhorse.get(key), _to_float_or(rhorse.get(key), 0.0)), _to_float_or(rhorse.get(key), _to_float_or(lhorse.get(key), 0.0)), ratio)
        row["horse_number"] = number
        fixed_style = row.get(
            "actual_running_style_fixed",
            row.get("actual_running_style", lhorse.get("actual_running_style_fixed", lhorse.get("actual_running_style", ""))),
        )
        row["actual_running_style"] = fixed_style
        row["actual_running_style_fixed"] = fixed_style
        row["horse_name"] = row.get("horse_name", lhorse.get("horse_name", ""))
        row["frame"] = int(row.get("frame", lhorse.get("frame", 1)) or 1)
        horses.append(row)
    horses = _rerank_timeline_horses(horses)
    left_time = _to_float_or(left.get("time"), float(frame_index))
    right_time = _to_float_or(right.get("time"), left_time)
    return {
        "time": _lerp(left_time, right_time, ratio),
        "progress": round(float(progress), 5),
        "index": frame_index,
        "horses": horses,
    }


def _rerank_timeline_horses(horses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        horses,
        key=lambda horse: (-_to_float_or(horse.get("position_m"), 0.0), int(horse.get("rank", 999) or 999)),
    )
    for rank, horse in enumerate(ranked, start=1):
        horse["rank"] = rank
    return ranked


def _lerp(left: float, right: float, ratio: float) -> float:
    return float(left + (right - left) * max(0.0, min(1.0, ratio)))


def _to_float_or(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _draw_frame(payload: dict[str, Any], frame: dict[str, Any]) -> Image.Image:
    width = int(payload["width"])
    height = int(payload["height"])
    race_config = payload["race_config"]
    distance = int(payload["distance"])
    surface = str(_config_get(race_config, "surface", "芝"))
    track_condition = str(_config_get(race_config, "track_condition", "良"))
    course = str(_config_get(race_config, "course", ""))

    image = Image.new("RGB", (width, height), "#18311f" if surface == "芝" else "#302116")
    draw = ImageDraw.Draw(image, "RGBA")
    fonts = _fonts(width, height)
    is_vertical = height > width

    leader = max(frame["horses"], key=lambda item: item["position_m"])
    camera_shift = _camera_shift(leader, payload)
    _draw_track(draw, payload, camera_shift)
    _draw_header(draw, fonts, width, course, surface, distance, track_condition, frame["remaining_m"], payload)

    horses = sorted(frame["horses"], key=lambda item: item["y"])
    for horse in horses:
        _draw_marker(draw, fonts, horse, payload, camera_shift)

    _draw_ranking_panel(draw, fonts, frame["horses"], width, height, is_vertical)
    if _should_show_final_result(frame, distance):
        _draw_post_goal_result_frame(draw, fonts, payload["final_ranking"], width, height, is_vertical)
    return image


def _draw_side_scroll_frame(payload: dict[str, Any], frame: dict[str, Any]) -> Image.Image:
    width = int(payload["width"])
    height = int(payload["height"])
    race_config = payload["race_config"]
    distance = int(payload["distance"])
    surface = str(_config_get(race_config, "surface", "芝"))
    track_condition = str(_config_get(race_config, "track_condition", "良"))
    weather = str(_config_get(race_config, "weather", "晴"))
    course = str(_config_get(race_config, "course", ""))
    is_vertical = height > width

    background = payload.get("background_image")
    image = background.copy() if isinstance(background, Image.Image) else Image.new("RGB", (width, height), "#85c8f2")
    draw = ImageDraw.Draw(image, "RGBA")
    fonts = _fonts(width, height)
    if not isinstance(background, Image.Image):
        _draw_side_scroll_background(draw, payload, surface, weather)

    horses = list(frame.get("horses", []))
    if not horses:
        return image
    leader_position = max(float(horse.get("position_m", 0.0)) for horse in horses)
    visible_distance_m = float(payload.get("visible_distance_m", 200.0))
    progress = float(frame.get("progress", 0.0) or 0.0)
    if progress < 0.02:
        camera_x = -visible_distance_m * 0.12
    else:
        camera_x = max(0.0, leader_position - visible_distance_m * 0.65)
        camera_x = min(camera_x, leader_position - visible_distance_m * 0.35)
        camera_x = max(camera_x, -visible_distance_m * 0.10)

    _draw_distance_markers(draw, payload, camera_x)
    _draw_start_gate(draw, payload, horses, camera_x)
    _draw_side_header(
        draw=draw,
        fonts=fonts,
        width=width,
        height=height,
        course=course,
        surface=surface,
        distance=distance,
        track_condition=track_condition,
        remaining_m=max(0, int(round(distance - leader_position, -1))),
        is_vertical=is_vertical,
    )

    frame_index = int(frame.get("index", 0) or 0)
    for horse in sorted(horses, key=lambda item: float(item.get("lane", 0.0))):
        horse_number = int(horse.get("horse_number", 0) or 0)
        frame_number = int(horse.get("frame", 1) or 1)
        x = _side_scroll_marker_screen_x(payload, horse, camera_x, progress)
        y = _side_scroll_lane_y(payload, horse, frame_index=frame_index, wobble=True)
        create_horse_marker(
            horse_number=horse_number,
            frame=frame_number,
            location=(x, y, 0.0),
            radius=float(payload["marker_radius"]),
            draw=draw,
            fonts=fonts,
            mark="",
            scale=1.25 if not is_vertical else 1.12,
        )

    _draw_side_ranking_panel(draw, fonts, horses, width, height, is_vertical)
    if _should_show_final_result(frame, distance):
        _draw_post_goal_result_frame(draw, fonts, payload["final_ranking"], width, height, is_vertical)
    return image


def _draw_side_scroll_background(
    draw: ImageDraw.ImageDraw,
    payload: dict[str, Any],
    surface: str,
    weather: str,
) -> None:
    width = int(payload["width"])
    height = int(payload["height"])
    track_top = float(payload["track_top"])
    track_bottom = float(payload["track_bottom"])
    sky_top = (90, 152, 205) if weather == "雨" else (93, 180, 238)
    sky_bottom = (170, 181, 190) if weather == "雨" else (205, 235, 255)
    for y in range(int(track_top)):
        ratio = y / max(1.0, track_top)
        color = tuple(int(sky_top[i] * (1.0 - ratio) + sky_bottom[i] * ratio) for i in range(3))
        draw.line((0, y, width, y), fill=(*color, 255))

    track_color = (73, 151, 77, 255) if surface == "芝" else (176, 130, 83, 255)
    darker = (54, 124, 61, 120) if surface == "芝" else (135, 95, 61, 130)
    lighter = (109, 181, 105, 95) if surface == "芝" else (205, 158, 105, 90)
    draw.rectangle((0, track_top, width, track_bottom), fill=track_color)
    line_step = max(16, int(height * 0.018))
    for index, y in enumerate(range(int(track_top), int(track_bottom), line_step)):
        color = lighter if index % 2 == 0 else darker
        draw.line((0, y, width, y + height * 0.035), fill=color, width=max(1, width // 640))

    rail_y_top = track_top + (track_bottom - track_top) * 0.08
    rail_y_bottom = track_bottom - (track_bottom - track_top) * 0.08
    draw.line((0, rail_y_top, width, rail_y_top), fill=(255, 255, 255, 210), width=max(3, width // 520))
    draw.line((0, rail_y_bottom, width, rail_y_bottom), fill=(255, 255, 255, 210), width=max(3, width // 520))


def _draw_distance_markers(
    draw: ImageDraw.ImageDraw,
    payload: dict[str, Any],
    camera_x: float,
) -> None:
    width = int(payload["width"])
    height = int(payload["height"])
    distance = int(payload["distance"])
    fonts = payload["fonts"]
    track_top = float(payload["track_top"])
    track_bottom = float(payload["track_bottom"])
    for remaining in _distance_marker_values(distance, int(payload["distance_marker_interval"])):
        world_x = float(distance - remaining)
        screen_x = _side_scroll_screen_x(payload, world_x, camera_x)
        if screen_x < -80 or screen_x > width + 80:
            continue
        is_goal = remaining == 0
        line_width = max(5, width // 260) if is_goal else max(2, width // 700)
        color = (255, 255, 255, 245) if is_goal else (255, 255, 255, 185)
        draw.line((screen_x, track_top, screen_x, track_bottom), fill=color, width=line_width)
        label = "GOAL" if is_goal else f"残り{remaining}m"
        label_y = track_top - height * 0.045 if not is_goal else track_top - height * 0.07
        _draw_text_with_shadow(draw, (screen_x + 8, label_y), label, fonts["small" if not is_goal else "title"], fill=(255, 255, 255, 255))


def _draw_start_gate(
    draw: ImageDraw.ImageDraw,
    payload: dict[str, Any],
    horses: list[dict[str, Any]],
    camera_x: float,
) -> None:
    width = int(payload["width"])
    right_to_left = _side_scroll_is_right_to_left(payload)
    if camera_x < 0:
        gate_x = width * (0.92 if right_to_left else 0.08)
    else:
        gate_x = _side_scroll_screen_x(payload, 0.0, camera_x)
    if right_to_left:
        if gate_x < width * 0.64 or gate_x > width * 1.16:
            return
    elif gate_x < -width * 0.16 or gate_x > width * 0.36:
        return
    fonts = payload["fonts"]
    marker_radius = float(payload["marker_radius"]) * 0.78
    gate_width = max(42, width * 0.032)
    sorted_horses = sorted(horses, key=lambda item: int(item.get("horse_number", 0) or 0))
    for horse in sorted_horses:
        y = _side_scroll_lane_y(payload, horse, frame_index=0, wobble=False)
        y0 = y - marker_radius * 1.35
        y1 = y + marker_radius * 1.35
        if right_to_left:
            box = (gate_x - gate_width * 0.35, y0, gate_x + gate_width, y1)
            bar_x = gate_x + gate_width * 0.25
        else:
            box = (gate_x - gate_width, y0, gate_x + gate_width * 0.35, y1)
            bar_x = gate_x - gate_width * 0.25
        draw.rectangle(box, outline=(225, 225, 225, 210), width=max(2, width // 800), fill=(30, 35, 38, 75))
        draw.line((bar_x, y0, bar_x, y1), fill=(230, 230, 230, 180), width=max(2, width // 900))
    label_x = gate_x - gate_width * 1.2 if not right_to_left else gate_x - gate_width * 0.8
    _draw_text_with_shadow(draw, (label_x, float(payload["track_top"]) - 38), "START", fonts["small"], fill=(255, 255, 255, 255))


def _draw_side_header(
    draw: ImageDraw.ImageDraw,
    fonts: dict[str, ImageFont.ImageFont],
    width: int,
    height: int,
    course: str,
    surface: str,
    distance: int,
    track_condition: str,
    remaining_m: int,
    is_vertical: bool,
) -> None:
    pad = max(18, width // 55)
    panel_h = height * (0.095 if is_vertical else 0.075)
    draw.rounded_rectangle((pad, pad, width - pad, pad + panel_h), radius=16, fill=(0, 0, 0, 145))
    title = f"{course} {surface}{distance}m {track_condition}"
    remain_text = f"残り {remaining_m}m" if remaining_m > 0 else "GOAL"
    _draw_text_with_shadow(draw, (pad * 1.45, pad * 1.22), title, fonts["title"], fill=(255, 255, 255, 255))
    _draw_text_with_shadow(draw, (pad * 1.45, pad * 1.22 + panel_h * 0.45), remain_text, fonts["small"], fill=(255, 238, 105, 255))


def _draw_side_ranking_panel(
    draw: ImageDraw.ImageDraw,
    fonts: dict[str, ImageFont.ImageFont],
    horses: list[dict[str, Any]],
    width: int,
    height: int,
    is_vertical: bool,
) -> None:
    ranked = sorted(horses, key=lambda item: int(item.get("rank", 999)))[:5]
    if is_vertical:
        panel_w = width * 0.86
        panel_h = height * 0.17
        x0 = width * 0.07
        y0 = height * 0.78
    else:
        panel_w = width * 0.24
        panel_h = height * 0.24
        x0 = width - panel_w - width * 0.026
        y0 = height * 0.13
    draw.rounded_rectangle((x0, y0, x0 + panel_w, y0 + panel_h), radius=18, fill=(0, 0, 0, 150))
    _draw_text_with_shadow(draw, (x0 + 18, y0 + 14), "現在上位5頭", fonts["small"], fill=(255, 238, 120, 255))
    line_y = y0 + 52
    for index, horse in enumerate(ranked, start=1):
        text = f"{index}位 {int(horse.get('horse_number', 0))}番"
        _draw_text_with_shadow(draw, (x0 + 18, line_y), text, fonts["small"], fill=(255, 255, 255, 245))
        line_y += max(30, int(height * (0.027 if not is_vertical else 0.022)))


def _draw_text_with_shadow(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
) -> None:
    x, y = xy
    draw.text((x + 2, y + 2), text, fill=(0, 0, 0, 185), font=font)
    draw.text((x, y), text, fill=fill, font=font)


def _draw_track(draw: ImageDraw.ImageDraw, payload: dict[str, Any], shift: tuple[float, float]) -> None:
    width = int(payload["width"])
    height = int(payload["height"])
    cx, cy, rx, ry = _track_box(payload)
    sx, sy = shift
    outer = (cx - rx + sx, cy - ry + sy, cx + rx + sx, cy + ry + sy)
    inner = (cx - rx * 0.63 + sx, cy - ry * 0.58 + sy, cx + rx * 0.63 + sx, cy + ry * 0.58 + sy)
    track_color = "#4a9a58" if str(_config_get(payload["race_config"], "surface", "芝")) == "芝" else "#b48755"
    edge_color = "#e9e7df"
    infield = "#265f34" if str(_config_get(payload["race_config"], "surface", "芝")) == "芝" else "#3c2a1d"

    draw.rectangle((0, 0, width, height), fill=(19, 41, 25, 255))
    draw.ellipse(outer, fill=track_color, outline=(230, 230, 220, 255), width=max(6, width // 240))
    draw.ellipse(inner, fill=infield, outline=(245, 245, 236, 255), width=max(5, width // 280))

    for scale, alpha in [(1.02, 120), (0.61, 160)]:
        box = (
            cx - rx * scale + sx,
            cy - ry * scale + sy,
            cx + rx * scale + sx,
            cy + ry * scale + sy,
        )
        draw.ellipse(box, outline=(255, 255, 255, alpha), width=max(2, width // 500))

    goal_x, goal_y, _ = _track_xy(payload["distance"], payload["distance"], 3.5, payload)
    goal_x += sx
    goal_y += sy
    draw.line((goal_x, goal_y - ry * 0.18, goal_x, goal_y + ry * 0.18), fill=(255, 235, 80, 255), width=max(5, width // 260))
    draw.text((goal_x + width * 0.012, goal_y - height * 0.025), "GOAL", fill=(255, 255, 210, 255), font=payload["fonts"]["small"])


def _draw_header(
    draw: ImageDraw.ImageDraw,
    fonts: dict[str, ImageFont.ImageFont],
    width: int,
    course: str,
    surface: str,
    distance: int,
    track_condition: str,
    remaining_m: int,
    payload: dict[str, Any],
) -> None:
    pad = max(18, width // 55)
    title = f"{course} {surface}{distance}m {track_condition}"
    draw.rounded_rectangle((pad, pad, width - pad, pad + width * 0.055), radius=14, fill=(0, 0, 0, 155))
    draw.text((pad * 1.45, pad * 1.25), title, fill=(255, 255, 255, 255), font=fonts["title"])
    remain_text = f"残り {remaining_m}m" if remaining_m > 0 else "GOAL"
    tw = draw.textlength(remain_text, font=fonts["title"])
    draw.text((width - pad * 1.45 - tw, pad * 1.25), remain_text, fill=(255, 230, 100, 255), font=fonts["title"])
    if payload.get("render_mode"):
        draw.text((pad * 1.45, pad * 3.6), "ORIGINAL CG SIMULATION", fill=(230, 240, 230, 210), font=fonts["tiny"])


def _draw_marker(
    draw: ImageDraw.ImageDraw,
    fonts: dict[str, ImageFont.ImageFont],
    horse: dict[str, Any],
    payload: dict[str, Any],
    shift: tuple[float, float],
) -> None:
    create_horse_marker(
        horse_number=int(horse["horse_number"]),
        frame=int(horse["frame"]),
        location=(horse["x"] + shift[0], horse["y"] + shift[1], 0.0),
        radius=max(17.0, 23.0 * float(horse["scale"])),
        draw=draw,
        fonts=fonts,
        mark=str(horse.get("mark", "")),
        scale=float(horse["scale"]),
    )


def create_horse_marker(
    horse_number: int,
    frame: int,
    location: tuple[float, float, float] | tuple[float, float],
    radius: float = 0.35,
    *,
    draw: ImageDraw.ImageDraw | None = None,
    fonts: dict[str, ImageFont.ImageFont] | None = None,
    mark: str = "",
    scale: float = 1.0,
) -> dict[str, Any] | None:
    """Draw or describe a horse-number marker. No horse body is created."""
    if draw is None or fonts is None:
        return {
            "horse_number": horse_number,
            "frame": frame,
            "location": location,
            "radius": radius,
            "display_mode": "marker",
        }
    x = float(location[0])
    y = float(location[1])
    number = str(horse_number)
    color = FRAME_COLORS.get(int(frame), "#9aa0a6")
    outline = "#FFFFFF" if int(frame) == 2 else "#111111"

    draw.ellipse((x - radius * 1.10, y - radius * 0.35, x + radius * 1.10, y + radius * 0.90), fill=(0, 0, 0, 70))
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=color,
        outline=outline,
        width=max(3, int(4 * scale)),
    )
    num_font = fonts["horse_number"]
    num_fill = get_text_color(int(frame))
    nw = draw.textlength(number, font=num_font)
    try:
        bbox = draw.textbbox((0, 0), number, font=num_font)
        nh = bbox[3] - bbox[1]
    except Exception:
        nh = radius
    draw.text((x - nw / 2, y - nh / 2 - radius * 0.04), number, fill=num_fill, font=num_font)

    if mark:
        mw = draw.textlength(mark, font=fonts["tiny"])
        draw.ellipse((x + radius * 0.42, y - radius * 1.18, x + radius * 1.18, y - radius * 0.42), fill=(0, 0, 0, 150))
        draw.text((x + radius * 0.80 - mw / 2, y - radius * 1.13), mark, fill=(255, 240, 100, 255), font=fonts["tiny"])
    return None


def _draw_ranking_panel(draw: ImageDraw.ImageDraw, fonts: dict[str, ImageFont.ImageFont], horses: list[dict[str, Any]], width: int, height: int, is_vertical: bool) -> None:
    ranked = sorted(horses, key=lambda item: int(item.get("rank", 999)))[:5]
    panel_w = width * (0.40 if is_vertical else 0.25)
    panel_h = height * (0.19 if is_vertical else 0.25)
    x0 = width - panel_w - width * 0.025
    y0 = height * (0.12 if is_vertical else 0.15)
    draw.rounded_rectangle((x0, y0, x0 + panel_w, y0 + panel_h), radius=18, fill=(0, 0, 0, 145))
    draw.text((x0 + 18, y0 + 14), "現在順位 TOP5", fill=(255, 238, 120, 255), font=fonts["small"])
    line_y = y0 + 52
    for index, horse in enumerate(ranked, start=1):
        text = f"{index}位 {horse['horse_number']}番 {horse['mark']}"
        draw.text((x0 + 18, line_y), text, fill=(255, 255, 255, 240), font=fonts["small"])
        line_y += max(28, int(height * 0.026))


def _draw_final_telop(draw: ImageDraw.ImageDraw, fonts: dict[str, ImageFont.ImageFont], final_ranking: list[dict[str, Any]], width: int, height: int, is_vertical: bool) -> None:
    panel_w = width * (0.62 if is_vertical else 0.42)
    panel_h = height * (0.28 if is_vertical else 0.34)
    x0 = (width - panel_w) / 2
    y0 = height * (0.62 if is_vertical else 0.56)
    draw.rounded_rectangle((x0, y0, x0 + panel_w, y0 + panel_h), radius=24, fill=(0, 0, 0, 185), outline=(255, 225, 90, 240), width=4)
    title = "FINAL RESULT"
    tw = draw.textlength(title, font=fonts["title"])
    draw.text((x0 + panel_w / 2 - tw / 2, y0 + 22), title, fill=(255, 230, 90, 255), font=fonts["title"])
    line_y = y0 + 78
    for row in final_ranking[:6]:
        text = f"{row['rank']}位 {row['horse_number']}番 {row.get('mark', '')}"
        draw.text((x0 + 42, line_y), text, fill=(255, 255, 255, 245), font=fonts["small"])
        line_y += max(30, int(height * 0.032))


def _draw_post_goal_result_frame(
    draw: ImageDraw.ImageDraw,
    fonts: dict[str, ImageFont.ImageFont],
    final_ranking: list[dict[str, Any]],
    width: int,
    height: int,
    is_vertical: bool,
) -> None:
    _draw_final_telop(draw, fonts, final_ranking, width, height, is_vertical)


def _build_render_payload_from_timeline(
    race_timeline: list[dict[str, Any]],
    race_config: dict[str, Any] | Any,
    horses: list[dict[str, Any]],
    prediction_table: pd.DataFrame | None,
    width: int,
    height: int,
    fps: int,
    duration_sec: int,
) -> dict[str, Any]:
    distance = int(_config_get(race_config, "distance", _timeline_distance(race_timeline)))
    marks_by_number = _marks_by_number(prediction_table)
    metadata = _horse_metadata_by_number(horses)
    frames: list[dict[str, Any]] = []

    for frame_index, timeline_frame in enumerate(race_timeline):
        if not isinstance(timeline_frame, dict):
            continue
        timeline_horses = timeline_frame.get("horses", [])
        if not isinstance(timeline_horses, list) or not timeline_horses:
            continue
        frame_horses: list[dict[str, Any]] = []
        for fallback_index, horse in enumerate(timeline_horses):
            if not isinstance(horse, dict):
                continue
            horse_number = int(horse.get("horse_number", fallback_index + 1) or fallback_index + 1)
            meta = metadata.get(horse_number, {})
            frame_value = int(horse.get("frame", meta.get("frame", min(8, fallback_index + 1))) or min(8, fallback_index + 1))
            lane = float(horse.get("lane", meta.get("lane", min(7.0, max(0.0, frame_value - 1.0)))))
            position = float(horse.get("position_m", 0.0) or 0.0)
            x, y, scale = _track_xy(position, distance, lane, {"width": width, "height": height})
            frame_horses.append(
                {
                    "horse_name": str(horse.get("horse_name", meta.get("horse_name", ""))),
                    "horse_number": horse_number,
                    "frame": frame_value,
                    "mark": marks_by_number.get(horse_number, ""),
                    "position_m": position,
                    "rank": int(horse.get("rank", fallback_index + 1) or fallback_index + 1),
                    "lane": lane,
                    "x": x,
                    "y": y,
                    "scale": scale,
                }
            )
        if not frame_horses:
            continue
        leader_position = max(horse["position_m"] for horse in frame_horses)
        frames.append(
            {
                "index": frame_index,
                "time": float(timeline_frame.get("time", frame_index / max(1, fps))),
                "progress": float(timeline_frame.get("progress", leader_position / max(1, distance))),
                "is_post_goal_frame": bool(timeline_frame.get("is_post_goal_frame", False)),
                "final_result_display_started": bool(timeline_frame.get("final_result_display_started", False)),
                "remaining_m": max(0, int(round(distance - leader_position, -1))),
                "horses": sorted(frame_horses, key=lambda item: int(item.get("rank", 999))),
            }
        )

    if not frames:
        raise ValueError("race_timeline contains no drawable horse frames")

    payload = {
        "width": width,
        "height": height,
        "fps": fps,
        "duration_sec": duration_sec,
        "distance": distance,
        "race_config": _config_to_dict(race_config),
        "frames": frames,
        "final_ranking": _final_ranking_from_timeline(race_timeline, marks_by_number, metadata),
        "fonts": _fonts(width, height),
        "render_mode": "timeline",
        "video_layout": "legacy_overview",
    }
    return payload


def _build_side_scroll_payload(
    race_timeline: list[dict[str, Any]],
    race_config: dict[str, Any] | Any,
    horses: list[dict[str, Any]],
    prediction_table: pd.DataFrame | None,
    width: int,
    height: int,
    fps: int,
    duration_sec: int,
) -> dict[str, Any]:
    distance = int(_config_get(race_config, "distance", _timeline_distance(race_timeline)))
    marks_by_number = _marks_by_number(prediction_table)
    metadata = _horse_metadata_by_number(horses)
    is_vertical = height > width
    field_size = max(1, max((len(frame.get("horses", [])) for frame in race_timeline if isinstance(frame, dict)), default=len(horses)))
    if is_vertical:
        track_top = height * 0.18
        track_bottom = height * 0.96
        marker_radius = max(20.0, width * 0.028)
    else:
        track_top = height * 0.22
        track_bottom = height * 0.96
        marker_radius = max(22.0, width * 0.016)
    visible_distance_m = 200.0
    px_per_m = width / visible_distance_m
    screen_origin_x = 0.0
    camera_lead_m = visible_distance_m * 0.65
    direction = _race_direction(race_config)
    render_direction = "right_to_left" if direction == "右" else "left_to_right"
    start_gate_side = "right" if render_direction == "right_to_left" else "left"
    goal_side = "left" if render_direction == "right_to_left" else "right"
    lane_order_by_number = _side_scroll_lane_order(race_timeline, horses)
    lane_jitter_by_number = _side_scroll_lane_jitter(lane_order_by_number)

    frames: list[dict[str, Any]] = []
    for frame_index, timeline_frame in enumerate(race_timeline):
        if not isinstance(timeline_frame, dict):
            continue
        timeline_horses = timeline_frame.get("horses", [])
        if not isinstance(timeline_horses, list) or not timeline_horses:
            continue
        frame_horses: list[dict[str, Any]] = []
        for fallback_index, horse in enumerate(timeline_horses):
            if not isinstance(horse, dict):
                continue
            horse_number = int(horse.get("horse_number", fallback_index + 1) or fallback_index + 1)
            meta = metadata.get(horse_number, {})
            frame_value = int(horse.get("frame", meta.get("frame", min(8, fallback_index + 1))) or min(8, fallback_index + 1))
            lane = float(horse.get("lane", meta.get("lane", fallback_index)))
            frame_horses.append(
                {
                    "horse_name": str(horse.get("horse_name", meta.get("horse_name", ""))),
                    "horse_number": horse_number,
                    "frame": frame_value,
                    "mark": marks_by_number.get(horse_number, ""),
                    "position_m": float(horse.get("position_m", 0.0) or 0.0),
                    "rank": int(horse.get("rank", fallback_index + 1) or fallback_index + 1),
                    "lane": lane,
                    "actual_running_style": str(horse.get("actual_running_style_fixed", horse.get("actual_running_style", ""))),
                    "actual_running_style_fixed": str(horse.get("actual_running_style_fixed", horse.get("actual_running_style", ""))),
                }
            )
        if not frame_horses:
            continue
        leader_position = max(horse["position_m"] for horse in frame_horses)
        frames.append(
            {
                "index": frame_index,
                "time": float(timeline_frame.get("time", frame_index / max(1, fps))),
                "progress": float(timeline_frame.get("progress", leader_position / max(1, distance))),
                "is_post_goal_frame": bool(timeline_frame.get("is_post_goal_frame", False)),
                "final_result_display_started": bool(timeline_frame.get("final_result_display_started", False)),
                "remaining_m": max(0, int(round(distance - leader_position, -1))),
                "horses": sorted(frame_horses, key=lambda item: int(item.get("rank", 999))),
            }
        )

    if not frames:
        raise ValueError("race_timeline contains no drawable horse frames")

    payload = {
        "width": width,
        "height": height,
        "fps": fps,
        "duration_sec": duration_sec,
        "distance": distance,
        "race_config": _config_to_dict(race_config),
        "frames": frames,
        "final_ranking": _final_ranking_from_timeline(race_timeline, marks_by_number, metadata),
        "fonts": _fonts(width, height),
        "render_mode": "timeline",
        "video_layout": "side_scroll",
        "track_top": track_top,
        "track_bottom": track_bottom,
        "field_size": field_size,
        "px_per_m": px_per_m,
        "screen_origin_x": screen_origin_x,
        "camera_lead_m": camera_lead_m,
        "visible_distance_m": visible_distance_m,
        "marker_radius": marker_radius,
        "distance_marker_interval": 100,
        "direction": direction,
        "render_direction": render_direction,
        "start_gate_side": start_gate_side,
        "goal_side": goal_side,
        "lane_order_by_number": lane_order_by_number,
        "lane_jitter_by_number": lane_jitter_by_number,
    }
    payload["background_image"] = _create_side_scroll_background_image(payload)
    return payload


def _create_side_scroll_background_image(payload: dict[str, Any]) -> Image.Image:
    image = Image.new("RGB", (int(payload["width"]), int(payload["height"])), "#85c8f2")
    draw = ImageDraw.Draw(image, "RGBA")
    race_config = payload["race_config"]
    _draw_side_scroll_background(
        draw,
        payload,
        str(_config_get(race_config, "surface", "芝")),
        str(_config_get(race_config, "weather", "晴")),
    )
    return image


def _build_render_payload(
    sections: pd.DataFrame,
    ranking: pd.DataFrame,
    race_config: dict[str, Any] | Any,
    horses: list[dict[str, Any]],
    prediction_table: pd.DataFrame | None,
    width: int,
    height: int,
    fps: int,
    duration_sec: int,
) -> dict[str, Any]:
    distance = int(_config_get(race_config, "distance", float(sections["position_m"].max())))
    metadata = _horse_metadata(horses, ranking, prediction_table)
    max_time = float(sections["elapsed_time"].max())
    frame_count = max(2, int(fps * duration_sec))
    timeline = np.linspace(0.0, max_time, frame_count)
    tables = {
        str(name): table.sort_values("elapsed_time")
        for name, table in sections.groupby("horse_name")
    }
    frames = []
    for frame_index, t in enumerate(timeline):
        frame_horses = []
        for lane_index, (horse_name, table) in enumerate(tables.items()):
            meta = metadata.get(str(horse_name), {})
            position = _position_at_time(table, float(t), distance)
            lane = _lane_at_time(table, float(t), float(meta.get("lane", lane_index)))
            rank = _rank_at_time(table, float(t), None)
            x, y, scale = _track_xy(position, distance, lane, {"width": width, "height": height})
            frame_horses.append(
                {
                    "horse_name": str(horse_name),
                    "horse_number": int(meta.get("horse_number", lane_index + 1)),
                    "frame": int(meta.get("frame", min(8, lane_index + 1))),
                    "mark": str(meta.get("mark", "")),
                    "position_m": position,
                    "rank": rank if rank is not None else lane_index + 1,
                    "lane": lane,
                    "x": x,
                    "y": y,
                    "scale": scale,
                }
            )
        frame_horses.sort(key=lambda item: int(item.get("rank", 999)))
        frames.append(
            {
                "index": frame_index,
                "time": float(t),
                "progress": frame_index / max(1, frame_count - 1),
                "remaining_m": max(0, int(round(distance - max(horse["position_m"] for horse in frame_horses), -1))),
                "horses": frame_horses,
            }
        )

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "duration_sec": duration_sec,
        "distance": distance,
        "race_config": _config_to_dict(race_config),
        "frames": frames,
        "final_ranking": _final_ranking(ranking, metadata),
        "fonts": _fonts(width, height),
        "render_mode": "pillow",
    }


def _extract_sections(simulation_result: dict[str, Any] | Any) -> pd.DataFrame:
    if isinstance(simulation_result, dict):
        timeline = simulation_result.get("race_timeline")
        if timeline:
            return _normalize_sections(_timeline_to_dataframe(timeline))
        timeline_table = simulation_result.get("timeline")
        if isinstance(timeline_table, pd.DataFrame):
            return _normalize_sections(timeline_table.copy())
        sections = simulation_result.get("sections")
        if isinstance(sections, pd.DataFrame):
            return _normalize_sections(sections.copy())
        csv_paths = simulation_result.get("csv_paths", {})
        if isinstance(csv_paths, dict) and csv_paths.get("timeline"):
            path = Path(str(csv_paths["timeline"]))
            if path.exists():
                return _normalize_sections(pd.read_csv(path))
        if isinstance(csv_paths, dict) and csv_paths.get("sections"):
            path = Path(str(csv_paths["sections"]))
            if path.exists():
                return _normalize_sections(pd.read_csv(path))
        if hasattr(simulation_result.get("simulation_result"), "timeline_dataframe"):
            timeline_df = simulation_result["simulation_result"].timeline_dataframe()
            if not timeline_df.empty:
                return _normalize_sections(timeline_df)
        if hasattr(simulation_result.get("simulation_result"), "states_dataframe"):
            return _normalize_sections(simulation_result["simulation_result"].states_dataframe())
    if hasattr(simulation_result, "timeline_dataframe"):
        timeline_df = simulation_result.timeline_dataframe()
        if not timeline_df.empty:
            return _normalize_sections(timeline_df)
    if hasattr(simulation_result, "states_dataframe"):
        return _normalize_sections(simulation_result.states_dataframe())
    return pd.DataFrame()


def _timeline_to_dataframe(race_timeline: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for frame in race_timeline:
        frame_time = float(frame.get("time", 0.0))
        for horse in frame.get("horses", []):
            if not isinstance(horse, dict):
                continue
            row = dict(horse)
            row["time"] = frame_time
            row["elapsed_time"] = frame_time
            rows.append(row)
    return pd.DataFrame(rows)


def _normalize_sections(sections: pd.DataFrame) -> pd.DataFrame:
    if sections.empty:
        return sections
    normalized = sections.copy()
    if "elapsed_time" not in normalized.columns and "time" in normalized.columns:
        normalized["elapsed_time"] = normalized["time"]
    if "position_m" not in normalized.columns and "distance_m" in normalized.columns:
        normalized["position_m"] = normalized["distance_m"]
    return normalized


def _extract_ranking(simulation_result: dict[str, Any] | Any) -> pd.DataFrame:
    if isinstance(simulation_result, dict) and isinstance(simulation_result.get("ranking"), pd.DataFrame):
        return simulation_result["ranking"].copy()
    if hasattr(simulation_result, "ranking"):
        return simulation_result.ranking.copy()
    return pd.DataFrame()


def _extract_prediction_table(simulation_result: dict[str, Any] | Any) -> pd.DataFrame | None:
    if not isinstance(simulation_result, dict):
        return None
    prediction = simulation_result.get("prediction")
    if isinstance(prediction, dict) and isinstance(prediction.get("prediction_table"), pd.DataFrame):
        return prediction["prediction_table"]
    return None


def _marks_by_number(prediction_table: pd.DataFrame | None) -> dict[int, str]:
    marks: dict[int, str] = {}
    if isinstance(prediction_table, pd.DataFrame) and not prediction_table.empty:
        for _, row in prediction_table.iterrows():
            number = int(row.get("馬番", 0) or 0)
            if number > 0:
                marks[number] = str(row.get("印", ""))
    return marks


def _horse_metadata_by_number(horses: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    metadata: dict[int, dict[str, Any]] = {}
    for index, horse in enumerate(horses):
        number = int(horse.get("horse_number", horse.get("馬番", index + 1)) or index + 1)
        frame = int(horse.get("frame", horse.get("枠順", min(8, index + 1))) or min(8, index + 1))
        metadata[number] = {
            "horse_name": str(horse.get("horse_name", horse.get("馬名", ""))),
            "frame": frame,
            "lane": min(7.0, max(0.0, frame - 1 + (number % 2) * 0.28)),
        }
    return metadata


def _timeline_distance(race_timeline: list[dict[str, Any]]) -> float:
    positions: list[float] = []
    for frame in race_timeline:
        if not isinstance(frame, dict):
            continue
        horses = frame.get("horses", [])
        if not isinstance(horses, list):
            continue
        for horse in horses:
            if isinstance(horse, dict):
                positions.append(float(horse.get("position_m", 0.0) or 0.0))
    return max(100.0, max(positions, default=2200.0))


def _final_ranking_from_timeline(
    race_timeline: list[dict[str, Any]],
    marks_by_number: dict[int, str],
    metadata: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    if not race_timeline:
        return []
    final_frame = race_timeline[-1]
    horses = final_frame.get("horses", []) if isinstance(final_frame, dict) else []
    if not isinstance(horses, list):
        return []
    rows = []
    sorted_horses = sorted(
        [horse for horse in horses if isinstance(horse, dict)],
        key=lambda horse: (-float(horse.get("position_m", 0.0) or 0.0), int(horse.get("rank", 999) or 999)),
    )
    for rank, horse in enumerate(sorted_horses, start=1):
        number = int(horse.get("horse_number", 0) or 0)
        meta = metadata.get(number, {})
        rows.append(
            {
                "rank": rank,
                "horse_number": number,
                "horse_name": str(horse.get("horse_name", meta.get("horse_name", ""))),
                "mark": marks_by_number.get(number, ""),
            }
        )
    return rows


def _side_scroll_lane_order(race_timeline: list[dict[str, Any]], horses: list[dict[str, Any]]) -> dict[int, int]:
    numbers: list[int] = []
    for horse in horses:
        try:
            number = int(horse.get("horse_number", horse.get("鬥ｬ逡ｪ", 0)) or 0)
        except (TypeError, ValueError):
            number = 0
        if number > 0 and number not in numbers:
            numbers.append(number)
    if not numbers:
        for frame in race_timeline:
            if not isinstance(frame, dict):
                continue
            for horse in frame.get("horses", []):
                if not isinstance(horse, dict):
                    continue
                try:
                    number = int(horse.get("horse_number", 0) or 0)
                except (TypeError, ValueError):
                    number = 0
                if number > 0 and number not in numbers:
                    numbers.append(number)
    return {number: index for index, number in enumerate(sorted(numbers))}


def _side_scroll_lane_jitter(lane_order_by_number: dict[int, int]) -> dict[int, float]:
    return {
        number: random.Random(number).uniform(-4.0, 4.0)
        for number in lane_order_by_number
    }


def _race_direction(race_config: dict[str, Any] | Any) -> str:
    raw = str(_config_get(race_config, "direction", _config_get(race_config, "turn_direction", "左"))).strip()
    lowered = raw.lower()
    if raw == "右" or "右" in raw or lowered.startswith("right"):
        return "右"
    return "左"


def _side_scroll_is_right_to_left(payload: dict[str, Any]) -> bool:
    return str(payload.get("render_direction", "left_to_right")) == "right_to_left"


def _side_scroll_screen_x(payload: dict[str, Any], world_x: float, camera_x: float) -> float:
    width = float(payload["width"])
    visible_distance_m = float(payload.get("visible_distance_m", 200.0))
    normalized = (float(world_x) - float(camera_x)) / max(1.0, visible_distance_m)
    screen_x = normalized * width
    if _side_scroll_is_right_to_left(payload):
        return width - screen_x
    return screen_x


def _side_scroll_start_screen_x(payload: dict[str, Any], scale: float = 1.25) -> float:
    width = float(payload["width"])
    marker_radius = float(payload.get("marker_radius", 20.0)) * scale
    if _side_scroll_is_right_to_left(payload):
        return min(width * 0.85, width - marker_radius - 20.0)
    return max(width * 0.15, marker_radius + 20.0)


def _side_scroll_marker_screen_x(
    payload: dict[str, Any],
    horse: dict[str, Any],
    camera_x: float,
    progress: float,
) -> float:
    scroll_x = _side_scroll_screen_x(payload, float(horse.get("position_m", 0.0)), camera_x)
    start_x = _side_scroll_start_screen_x(payload)
    if progress < 0.03:
        return start_x
    transition_ratio = max(0.0, min(1.0, (progress - 0.03) / 0.05))
    return start_x * (1.0 - transition_ratio) + scroll_x * transition_ratio


def _side_scroll_lane_y(
    payload: dict[str, Any],
    horse: dict[str, Any],
    frame_index: int = 0,
    wobble: bool = True,
) -> float:
    track_top = float(payload["track_top"])
    track_bottom = float(payload["track_bottom"])
    field_size = max(1, int(payload.get("field_size", 1)))
    lane_order = payload.get("lane_order_by_number", {})
    horse_number = int(horse.get("horse_number", 0) or 0)
    lane_index = lane_order.get(horse_number) if isinstance(lane_order, dict) else None
    if lane_index is None:
        lane_index = int(max(0.0, min(float(field_size - 1), _to_float_or(horse.get("lane"), 0.0))))

    marker_radius = float(payload.get("marker_radius", 20.0))
    vertical_margin = max(marker_radius * 1.45, (track_bottom - track_top) * 0.025)
    usable_top = track_top + vertical_margin
    usable_bottom = track_bottom - vertical_margin
    if field_size <= 1:
        base_y = (usable_top + usable_bottom) / 2.0
    else:
        base_y = usable_top + (usable_bottom - usable_top) * (float(lane_index) + 0.5) / field_size
    if wobble:
        horse_number = int(horse.get("horse_number", 0) or 0)
        jitter = payload.get("lane_jitter_by_number", {})
        if isinstance(jitter, dict):
            base_y += float(jitter.get(horse_number, 0.0))
        base_y += math.sin(frame_index * 0.1 + horse_number) * 2.0
    return float(max(usable_top, min(usable_bottom, base_y)))


def _distance_marker_values(distance: int, interval: int) -> list[int]:
    values = [int(distance)]
    values.extend(range((int(distance) // interval) * interval, 0, -interval))
    values.append(0)
    deduped: list[int] = []
    for value in values:
        if value < 0 or value > distance:
            continue
        if value not in deduped:
            deduped.append(value)
    return deduped


def _horse_metadata(horses: list[dict[str, Any]], ranking: pd.DataFrame, prediction_table: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    marks_by_number = _marks_by_number(prediction_table)
    metadata: dict[str, dict[str, Any]] = {}
    for index, horse in enumerate(horses):
        name = str(horse.get("horse_name", horse.get("馬名", "")))
        number = int(horse.get("horse_number", horse.get("馬番", index + 1)))
        frame = int(horse.get("frame", horse.get("枠順", min(8, index + 1))))
        metadata[name] = {
            "horse_number": number,
            "frame": frame,
            "mark": marks_by_number.get(number, ""),
            "lane": min(7.0, max(0.0, frame - 1 + (number % 2) * 0.28)),
        }
    if not ranking.empty:
        for _, row in ranking.iterrows():
            name = str(row.get("horse_name", ""))
            if not name:
                continue
            number = int(row.get("horse_number", 0) or 0)
            metadata.setdefault(
                name,
                {
                    "horse_number": number,
                    "frame": int(row.get("frame", 1) or 1),
                    "mark": marks_by_number.get(number, ""),
                    "lane": min(7.0, max(0.0, int(row.get("frame", 1) or 1) - 1 + (number % 2) * 0.28)),
                },
            )
    return metadata


def _final_ranking(ranking: pd.DataFrame, metadata: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if ranking.empty:
        return []
    rows = []
    for _, row in ranking.sort_values("rank").iterrows():
        name = str(row.get("horse_name", ""))
        meta = metadata.get(name, {})
        rows.append(
            {
                "rank": int(row.get("rank", 0) or 0),
                "horse_number": int(row.get("horse_number", meta.get("horse_number", 0)) or 0),
                "horse_name": name,
                "mark": str(meta.get("mark", "")),
            }
        )
    return rows


def _position_at_time(table: pd.DataFrame, t: float, distance: int) -> float:
    times = table["elapsed_time"].to_numpy(dtype=float)
    positions = table["position_m"].to_numpy(dtype=float)
    if len(times) == 0:
        return 0.0
    if t <= times[0]:
        return float(positions[0] * t / max(times[0], 0.1))
    if t >= times[-1]:
        return float(distance)
    return float(np.interp(t, times, positions))


def _lane_at_time(table: pd.DataFrame, t: float, default_lane: float) -> float:
    if "lane" not in table.columns or table.empty:
        return default_lane
    times = table["elapsed_time"].to_numpy(dtype=float)
    lanes = table["lane"].to_numpy(dtype=float)
    if len(times) == 0:
        return default_lane
    if t <= times[0]:
        return float(lanes[0])
    if t >= times[-1]:
        return float(lanes[-1])
    return float(np.interp(t, times, lanes))


def _rank_at_time(table: pd.DataFrame, t: float, default_rank: int | None) -> int | None:
    if "rank" not in table.columns or table.empty:
        return default_rank
    times = table["elapsed_time"].to_numpy(dtype=float)
    ranks = table["rank"].to_numpy(dtype=float)
    if len(times) == 0:
        return default_rank
    if t <= times[0]:
        return int(round(float(ranks[0])))
    if t >= times[-1]:
        return int(round(float(ranks[-1])))
    index = int(np.abs(times - t).argmin())
    return int(round(float(ranks[index])))


def _track_xy(position_m: float, distance: float, lane: float, payload: dict[str, Any]) -> tuple[float, float, float]:
    width = int(payload["width"])
    height = int(payload["height"])
    cx, cy, rx, ry = _track_box(payload)
    progress = min(1.0, max(0.0, position_m / max(1.0, distance)))
    theta = 2.0 * math.pi * progress
    lane_ratio = (lane - 3.5) / 7.0
    x = cx + (rx * (0.77 + lane_ratio * 0.07)) * math.cos(theta)
    y = cy + (ry * (0.77 + lane_ratio * 0.07)) * math.sin(theta)
    depth = 0.86 + 0.28 * ((y - (cy - ry)) / max(1.0, 2 * ry))
    return float(x), float(y), float(depth * (width / 1920.0 if width > height else height / 1920.0))


def _track_box(payload: dict[str, Any]) -> tuple[float, float, float, float]:
    width = int(payload["width"])
    height = int(payload["height"])
    if height > width:
        return width * 0.50, height * 0.52, width * 0.43, height * 0.25
    return width * 0.52, height * 0.56, width * 0.39, height * 0.34


def _camera_shift(leader: dict[str, Any], payload: dict[str, Any]) -> tuple[float, float]:
    width = int(payload["width"])
    height = int(payload["height"])
    target_x = width * 0.52
    target_y = height * 0.56
    progress = min(1.0, max(0.0, leader["position_m"] / max(1.0, payload["distance"])))
    strength = 0.10 + 0.14 * max(0.0, progress - 0.70) / 0.30
    sx = max(-width * 0.08, min(width * 0.08, (target_x - leader["x"]) * strength))
    sy = max(-height * 0.06, min(height * 0.06, (target_y - leader["y"]) * strength))
    return sx, sy


def _video_dimensions(video_format: str) -> tuple[int, int]:
    return VIDEO_FORMATS.get(video_format, VIDEO_FORMATS.get(video_format.lower(), (1920, 1080)))


def _fonts(width: int, height: int) -> dict[str, ImageFont.ImageFont]:
    base = max(18, int(min(width, height) / 42))
    return {
        "title": _font(base + 14),
        "small": _font(base),
        "tiny": _font(max(13, base - 8)),
        "label": _font(max(14, base - 5)),
        "horse_number": _font(max(14, base - 4)),
    }


def _font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/meiryo.ttc",
        "C:/Windows/Fonts/YuGothM.ttc",
        "C:/Windows/Fonts/msgothic.ttc",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _config_to_dict(value: dict[str, Any] | Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    return dict(getattr(value, "__dict__", {}))


def _config_get(value: dict[str, Any] | Any, key: str, default: Any) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)
