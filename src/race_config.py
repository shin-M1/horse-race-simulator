from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


Surface = Literal["芝", "ダート"]
Direction = Literal["右", "左", "直線"]
Weather = Literal["晴", "曇", "雨", "雪"]
TrackCondition = Literal["良", "稍重", "重", "不良"]
BASE_CARRIED_WEIGHT = 56.0


def calculate_weight_penalty(carried_weight: float | int | None) -> float:
    """Return the modest performance penalty relative to the 56kg baseline."""
    try:
        weight = float(carried_weight) if carried_weight is not None else BASE_CARRIED_WEIGHT
    except (TypeError, ValueError):
        weight = BASE_CARRIED_WEIGHT
    return max(-4.0, min(6.0, (weight - BASE_CARRIED_WEIGHT) * 2.0))


@dataclass(frozen=True)
class RaceConfig:
    course: str
    surface: Surface
    distance: int
    direction: Direction
    weather: Weather
    track_condition: TrackCondition
    race_course_day: str = "1日目"
    course_layout: str = "A"
    track_bias: str = "標準"

    @classmethod
    def default(cls) -> "RaceConfig":
        return cls(
            course="阪神",
            surface="芝",
            distance=2200,
            direction="右",
            weather="晴",
            track_condition="良",
            race_course_day="1日目",
            course_layout="A",
            track_bias="標準",
        )

    def to_dict(self) -> dict[str, str | int]:
        return {
            "course": self.course,
            "surface": self.surface,
            "distance": self.distance,
            "direction": self.direction,
            "weather": self.weather,
            "track_condition": self.track_condition,
            "race_course_day": self.race_course_day,
            "course_layout": self.course_layout,
            "track_bias": self.track_bias,
        }


@dataclass(frozen=True)
class HorseEntry:
    horse_name: str
    frame: int
    horse_number: int
    carried_weight: float = BASE_CARRIED_WEIGHT
    pedigree_info: dict[str, Any] | None = None
    jockey: str = ""
    jockey_score: float = 50.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "horse_name": self.horse_name,
            "frame": self.frame,
            "horse_number": self.horse_number,
            "carried_weight": float(self.carried_weight),
            "pedigree_info": self.pedigree_info,
            "jockey": self.jockey,
            "jockey_score": float(self.jockey_score),
        }


def prompt_race_config(default: RaceConfig | None = None) -> RaceConfig:
    """Read race settings from stdin, using defaults on empty input."""
    base = default or RaceConfig.default()
    print("Race settings. Press Enter to keep the default value.")
    course = input(f"course [{base.course}]: ").strip() or base.course
    surface = input(f"surface(芝/ダート) [{base.surface}]: ").strip() or base.surface
    distance_text = input(f"distance [{base.distance}]: ").strip()
    direction = input(f"direction(右/左/直線) [{base.direction}]: ").strip() or base.direction
    weather = input(f"weather(晴/曇/雨/雪) [{base.weather}]: ").strip() or base.weather
    track_condition = input(
        f"track_condition(良/稍重/重/不良) [{base.track_condition}]: "
    ).strip() or base.track_condition
    race_course_day = input(f"race_course_day [{base.race_course_day}]: ").strip() or base.race_course_day
    course_layout = input(f"course_layout(A/B/C/D) [{base.course_layout}]: ").strip() or base.course_layout
    track_bias = input(f"track_bias [{base.track_bias}]: ").strip() or base.track_bias
    return RaceConfig(
        course=course,
        surface=surface,  # type: ignore[arg-type]
        distance=int(distance_text or base.distance),
        direction=direction,  # type: ignore[arg-type]
        weather=weather,  # type: ignore[arg-type]
        track_condition=track_condition,  # type: ignore[arg-type]
        race_course_day=race_course_day,
        course_layout=course_layout,
        track_bias=track_bias,
    )


def prompt_horses(default: list[HorseEntry] | None = None) -> list[HorseEntry]:
    """Read starter entries from stdin, using demo starters when input is empty."""
    defaults = default or demo_horses()
    count_text = input(f"number of horses [{len(defaults)}]: ").strip()
    count = int(count_text or len(defaults))
    horses: list[HorseEntry] = []
    for index in range(count):
        fallback = defaults[index] if index < len(defaults) else HorseEntry(f"Horse{index + 1}", index + 1, index + 1)
        name = input(f"horse {index + 1} name [{fallback.horse_name}]: ").strip() or fallback.horse_name
        frame_text = input(f"frame [{fallback.frame}]: ").strip()
        number_text = input(f"horse_number [{fallback.horse_number}]: ").strip()
        weight_text = input(f"carried_weight kg [{fallback.carried_weight}]: ").strip()
        horses.append(
            HorseEntry(
                horse_name=name,
                frame=int(frame_text or fallback.frame),
                horse_number=int(number_text or fallback.horse_number),
                carried_weight=float(weight_text or fallback.carried_weight),
                jockey=fallback.jockey,
                jockey_score=fallback.jockey_score,
            )
        )
    return horses


def demo_horses() -> list[HorseEntry]:
    return [
        HorseEntry("ミドリノハヤテ", 1, 1),
        HorseEntry("アカツキスター", 2, 3),
        HorseEntry("シロガネロンド", 4, 6),
        HorseEntry("クロノスウィフト", 6, 10),
        HorseEntry("カゼノブリッツ", 8, 14),
    ]
