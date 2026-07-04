from __future__ import annotations

from dataclasses import dataclass

from horse_analyzer import HorseAbility


@dataclass(frozen=True)
class RacePace:
    pace: str
    front_group_size: int
    closer_advantage: float
    style_groups: dict[str, list[str]]
    front_pressure: float
    front_group: list[str]
    middle_group: list[str]
    back_group: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "pace": self.pace,
            "front_group_size": self.front_group_size,
            "closer_advantage": round(self.closer_advantage, 3),
            "style_groups": self.style_groups,
            "front_pressure": round(self.front_pressure, 3),
            "front_group": self.front_group,
            "middle_group": self.middle_group,
            "back_group": self.back_group,
        }


class PacePredictor:
    """Predict race pace from each horse's base style probabilities."""

    def predict(self, abilities: list[HorseAbility]) -> RacePace:
        if not abilities:
            return RacePace(
                pace="medium",
                front_group_size=0,
                closer_advantage=0.0,
                style_groups={},
                front_pressure=0.0,
                front_group=[],
                middle_group=[],
                back_group=[],
            )

        style_groups = {style: [] for style in ["逃げ", "先行", "差し", "追込", "自在"]}
        front_pressure = 0.0
        front_group: list[str] = []
        middle_group: list[str] = []
        back_group: list[str] = []

        for ability in abilities:
            profile = ability.base_style_profile
            style_groups.setdefault(ability.primary_running_style, []).append(ability.horse_name)
            front_style_probability = profile.get("逃げ", 0.0) + profile.get("先行", 0.0) * 0.70
            front_pressure += ability.early_push_score * 0.50 + front_style_probability * 50.0

            front_score = profile.get("逃げ", 0.0) + 0.7 * profile.get("先行", 0.0)
            middle_score = 0.5 * profile.get("先行", 0.0) + profile.get("差し", 0.0)
            back_score = 0.5 * profile.get("差し", 0.0) + profile.get("追込", 0.0)
            group_scores = {
                "front": front_score,
                "middle": middle_score,
                "back": back_score,
            }
            best_group = max(group_scores, key=group_scores.get)
            if best_group == "front":
                front_group.append(ability.horse_name)
            elif best_group == "middle":
                middle_group.append(ability.horse_name)
            else:
                back_group.append(ability.horse_name)

        horse_count = len(abilities)
        high_threshold = 62.0 * horse_count
        slow_threshold = 38.0 * horse_count

        if front_pressure >= high_threshold:
            pace = "high"
            closer_advantage = 1.08
        elif front_pressure <= slow_threshold:
            pace = "slow"
            closer_advantage = 0.92
        else:
            pace = "medium"
            closer_advantage = 1.0

        front_group_size = len(front_group)
        return RacePace(
            pace=pace,
            front_group_size=front_group_size,
            closer_advantage=closer_advantage,
            style_groups=style_groups,
            front_pressure=front_pressure,
            front_group=front_group,
            middle_group=middle_group,
            back_group=back_group,
        )
