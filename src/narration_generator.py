from __future__ import annotations

import math
import wave
from pathlib import Path
from typing import Any

import numpy as np


NARRATION_DIR = Path("outputs/narration")


def generate_section_script(section_type: str, data: dict) -> str:
    """Create narration copy for a YouTube section."""
    section = str(section_type)
    if section == "trend":
        bullets = data.get("bullets", []) if isinstance(data, dict) else []
        return "同レースの過去傾向を確認します。" + "。".join(str(item).rstrip("。") for item in bullets if item) + "。"
    if section == "diagnosis":
        rows = data.get("rows", []) if isinstance(data, dict) else []
        lines = ["全頭診断です。馬番順に各馬の強みと不安点を見ていきます。"]
        for row in rows:
            if not isinstance(row, dict):
                continue
            lines.append(
                f"{row.get('馬番', row.get('horse_number', ''))}番、"
                f"{row.get('馬名', row.get('horse_name', ''))}。"
                f"{row.get('短評', row.get('comment', ''))}"
            )
        return " ".join(lines)
    if section == "featured":
        rows = data.get("rows", []) if isinstance(data, dict) else []
        lines = ["最後に注目馬を紹介します。印の軽い順から見て、最後に本命を確認します。"]
        for row in rows:
            if not isinstance(row, dict):
                continue
            lines.append(
                f"{row.get('印', '')}{row.get('馬番', row.get('horse_number', ''))}番、"
                f"{row.get('馬名', row.get('horse_name', ''))}。"
                f"根拠は{row.get('detailed_reason', row.get('reason', '総合評価です'))}。"
                f"リスクは{row.get('risk', '展開次第です')}。"
            )
        return " ".join(lines)
    return str(data.get("script", "")) if isinstance(data, dict) else ""


def synthesize_narration_audio(script: str, output_path: str) -> str:
    """Try local TTS. If unavailable, return an empty path without raising."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = str(script or "").strip()
    if not text:
        return ""
    try:
        import pyttsx3  # type: ignore

        engine = pyttsx3.init()
        engine.save_to_file(text, str(output))
        engine.runAndWait()
        return str(output) if output.is_file() and output.stat().st_size > 0 else ""
    except Exception:
        return ""


def estimate_narration_duration(script: str, min_sec: float = 4.0, chars_per_sec: float = 7.0) -> float:
    """Estimate Japanese narration duration from text length."""
    length = len(str(script or "").strip())
    return max(float(min_sec), min(90.0, math.ceil(length / max(1.0, chars_per_sec))))


def create_silent_audio(output_path: str, duration_sec: float, sample_rate: int = 44100) -> str:
    return _write_wave(output_path, np.zeros(max(1, int(duration_sec * sample_rate)), dtype=np.float32), sample_rate)


def create_start_beep(output_path: str, duration_sec: float = 0.45, sample_rate: int = 44100) -> str:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = max(1, int(duration_sec * sample_rate))
    t = np.linspace(0, duration_sec, count, endpoint=False)
    envelope = np.linspace(1.0, 0.1, count)
    samples = 0.25 * np.sin(2 * np.pi * 880.0 * t) * envelope
    return _write_wave(str(output), samples.astype(np.float32), sample_rate)


def _write_wave(output_path: str, samples: np.ndarray, sample_rate: int) -> str:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767).astype("<i2")
    with wave.open(str(output), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return str(output)
