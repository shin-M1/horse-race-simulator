from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

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
        return "#000000" if frame in [1, 5, 8] else "#FFFFFF"


THUMBNAIL_DIR = Path("outputs/thumbnails")
EXTERNAL_MEDIA_ASSETS_USED = False
REVEALS_PREDICTION_MARKS_OR_HORSE_NAMES = False


def generate_youtube_thumbnail(
    prediction_log: dict,
    output_path: str,
    size: tuple[int, int] = (1280, 720),
) -> str:
    """Generate a YouTube thumbnail using only original shapes and text."""
    output = _resolve_output_path(output_path, THUMBNAIL_DIR, "thumbnail", ".png")
    output.parent.mkdir(parents=True, exist_ok=True)

    width, height = size
    config = _dict(prediction_log.get("race_config") or prediction_log.get("race_metadata"))
    race_name = str(prediction_log.get("race_name") or config.get("race_name") or "対象レース")
    race_date = str(prediction_log.get("race_date") or config.get("race_date") or "")
    surface = str(config.get("surface", "芝"))

    image = Image.new("RGB", (width, height), "#245c33")
    draw = ImageDraw.Draw(image, "RGBA")
    _draw_background(draw, width, height, surface)

    fonts = _fonts(width)
    pad = int(width * 0.055)
    draw.rounded_rectangle(
        (pad * 0.55, pad * 0.42, width - pad * 0.55, height - pad * 0.48),
        radius=34,
        fill=(0, 0, 0, 86),
        outline=(255, 230, 90, 190),
        width=4,
    )

    _draw_text_with_shadow(draw, (pad, pad * 0.75), race_name, fonts["race"], fill=(255, 255, 255, 255))
    if race_date:
        _draw_text_with_shadow(draw, (pad, pad * 0.75 + 88), race_date, fonts["small"], fill=(230, 242, 255, 255))

    badge_text = "AIシミュレーション予想"
    badge_w = draw.textlength(badge_text, font=fonts["badge"]) + 44
    badge_h = 58
    badge_x = width - pad - badge_w
    badge_y = pad * 0.88
    draw.rounded_rectangle(
        (badge_x, badge_y, badge_x + badge_w, badge_y + badge_h),
        radius=18,
        fill=(255, 218, 72, 245),
    )
    draw.text((badge_x + 22, badge_y + 10), badge_text, font=fonts["badge"], fill=(20, 24, 28, 255))

    sub = "過去傾向 × 全頭診断 × レース再現"
    _draw_text_with_shadow(draw, (pad, height * 0.335), sub, fonts["sub"], fill=(255, 235, 120, 255))

    _draw_race_silhouette(draw, width, height)
    _draw_decorative_markers(draw, fonts, width, height)

    footer = "外部映像素材不使用・オリジナルCG予想"
    _draw_text_with_shadow(draw, (pad, height - pad * 1.18), footer, fonts["tiny"], fill=(235, 245, 235, 230))
    image.save(output)
    return str(output)


def _draw_background(draw: ImageDraw.ImageDraw, width: int, height: int, surface: str) -> None:
    sky_h = int(height * 0.22)
    for y in range(sky_h):
        ratio = y / max(1, sky_h)
        color = (
            int(74 * (1 - ratio) + 136 * ratio),
            int(160 * (1 - ratio) + 207 * ratio),
            int(221 * (1 - ratio) + 245 * ratio),
            255,
        )
        draw.line((0, y, width, y), fill=color)
    base = (50, 132, 59, 255) if surface == "芝" else (162, 111, 68, 255)
    dark = (30, 91, 43, 120) if surface == "芝" else (115, 75, 47, 130)
    light = (96, 174, 85, 95) if surface == "芝" else (205, 153, 98, 90)
    draw.rectangle((0, sky_h, width, height), fill=base)
    for index, x in enumerate(range(-width, width * 2, max(36, width // 28))):
        color = light if index % 2 == 0 else dark
        draw.polygon([(x, sky_h), (x + width // 4, sky_h), (x + width, height), (x + width // 2, height)], fill=color)
    draw.rectangle((0, 0, width, height), fill=(0, 0, 0, 28))


def _draw_race_silhouette(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
    base_y = int(height * 0.62)
    for index, offset in enumerate([0, 82, 164, 246]):
        x = int(width * 0.54 + offset)
        scale = 1.0 - index * 0.08
        body = (x, base_y - 46 * scale, x + 136 * scale, base_y + 14 * scale)
        draw.ellipse(body, fill=(18, 22, 20, 220))
        draw.ellipse((x + 112 * scale, base_y - 72 * scale, x + 172 * scale, base_y - 20 * scale), fill=(18, 22, 20, 220))
        for leg in [18, 50, 86, 118]:
            lx = x + leg * scale
            draw.line((lx, base_y, lx - 24 * scale, base_y + 78 * scale), fill=(16, 18, 17, 210), width=max(8, int(13 * scale)))
        draw.line((x - 88 * scale, base_y + 52 * scale, x + 230 * scale, base_y + 52 * scale), fill=(255, 255, 255, 75), width=max(3, int(5 * scale)))


def _draw_decorative_markers(draw: ImageDraw.ImageDraw, fonts: dict[str, ImageFont.ImageFont], width: int, height: int) -> None:
    marker_data = [(1, 1, 0.68, 0.80), (2, 3, 0.77, 0.74), (3, 5, 0.86, 0.79), (4, 7, 0.92, 0.69)]
    radius = int(height * 0.040)
    for number, frame, rx, ry in marker_data:
        cx = int(width * rx)
        cy = int(height * ry)
        color = FRAME_COLORS.get(frame, "#888888")
        outline = "#FFFFFF" if frame == 2 else "#111111"
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color, outline=outline, width=4)
        text = str(number)
        nw = draw.textlength(text, font=fonts["number"])
        bbox = draw.textbbox((0, 0), text, font=fonts["number"])
        draw.text((cx - nw / 2, cy - (bbox[3] - bbox[1]) / 2 - 2), text, font=fonts["number"], fill=get_text_color(frame))


def _top_mark_rows(prediction_log: dict, limit: int = 3) -> list[dict[str, Any]]:
    prediction = _frame(prediction_log.get("prediction_table"))
    if prediction.empty:
        prediction = _frame(prediction_log.get("AI予想印"))
    if prediction.empty:
        return []
    mark_order = {"◎": 0, "○": 1, "▲": 2, "△": 3, "☆": 4}
    rows = []
    for _, row in prediction.iterrows():
        data = row.to_dict()
        mark = str(data.get("印", data.get("mark", "")))
        if mark in mark_order:
            data["印"] = mark
            rows.append(data)
    rows.sort(key=lambda item: mark_order.get(str(item.get("印")), 99))
    return rows[:limit]


def _fonts(width: int) -> dict[str, ImageFont.ImageFont]:
    return {
        "race": _font(int(width * 0.068), bold=True),
        "badge": _font(int(width * 0.026), bold=True),
        "sub": _font(int(width * 0.034), bold=True),
        "mark": _font(int(width * 0.070), bold=True),
        "number": _font(int(width * 0.040), bold=True),
        "horse": _font(int(width * 0.046), bold=True),
        "small": _font(int(width * 0.026), bold=False),
        "tiny": _font(int(width * 0.020), bold=False),
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
            if path and Path(path).exists():
                return ImageFont.truetype(path, size)
        except Exception:
            continue
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _draw_text_with_shadow(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
) -> None:
    x, y = xy
    draw.text((x + 3, y + 3), text, fill=(0, 0, 0, 190), font=font)
    draw.text((x, y), text, fill=fill, font=font)


def _resolve_output_path(output_path: str, default_dir: Path, prefix: str, suffix: str) -> Path:
    if not output_path:
        return default_dir / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}{suffix}"
    path = Path(output_path)
    if path.suffix.lower() != suffix:
        return path / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}{suffix}"
    return path


def _frame(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, list):
        return pd.DataFrame(value)
    return pd.DataFrame()


def _dict(value: Any) -> dict:
    return dict(value) if isinstance(value, dict) else {}


def _int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0
