from __future__ import annotations

import math
import random
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from horse_analyzer import HorseAbility
from pace_predictor import RacePace
from race_config import RaceConfig
from simulator import (
    EARLY_STYLE_GAP_RANGES,
    RaceSimulator,
    STYLE_CLOSER,
    STYLE_DEEP_CLOSER,
    STYLE_RUNNER,
    STYLE_STALKER,
    _formation_escape_fade,
    _formation_final_stretch_score,
    generate_controlled_race_timeline,
)


def ability(
    name: str,
    frame: int,
    number: int,
    style: str,
    early_speed: float = 65.0,
    stamina: float = 65.0,
    acceleration: float = 65.0,
    consistency: float = 70.0,
    avg_race_score: float = 60.0,
    race_power: float | None = None,
    early_aggressiveness: float = 0.5,
    mid_positioning: float = 0.5,
    late_kick_timing: float = 0.5,
    sustain_speed: float = 0.55,
    time_reliability: float = 0.0,
    recent_time_score: float = 55.0,
    late_kick_score: float = 55.0,
    avg_last3f: float | None = None,
    best_last3f: float | None = None,
    last3f_consistency: float = 50.0,
    late_gain_score: float = 50.0,
    early_push_score: float = 50.0,
    mid_cruise_score: float = 55.0,
    fade_resistance_score: float = 55.0,
    sustain_speed_score: float = 55.0,
    pace_resilience_score: float = 55.0,
    agari_reliability: float = 50.0,
) -> HorseAbility:
    profile = {"逃げ": 0.0, "先行": 0.0, "差し": 0.0, "追込": 0.0}
    profile[style] = 1.0
    return HorseAbility(
        horse_name=name,
        frame=frame,
        horse_number=number,
        early_speed=early_speed,
        stamina=stamina,
        acceleration=acceleration,
        front_runner=profile["逃げ"] * 100.0,
        stalker=profile["先行"] * 100.0,
        closer=max(profile["差し"], profile["追込"]) * 100.0,
        mud_aptitude=55.0,
        consistency=consistency,
        running_style=style,
        primary_running_style=style,
        base_style_profile=profile,
        recent_results=[],
        front_runner_score=profile["逃げ"] * 100.0,
        stalker_score=profile["先行"] * 100.0,
        closer_score=profile["差し"] * 100.0,
        deep_closer_score=profile["追込"] * 100.0,
        versatile_score=0.0,
        weighted_avg_first_ratio=0.5,
        weighted_avg_mid_ratio=0.5,
        weighted_avg_last_corner_ratio=0.5,
        weighted_avg_late_gain=0.0,
        weighted_avg_position_ratio=0.5,
        position_variance=0.0,
        style_sample_size=5,
        debug_passing_orders=[],
        avg_opponent_strength_score=60.0,
        avg_race_score=avg_race_score,
        popularity_score=60.0,
        race_level_score=60.0,
        finish_score=60.0,
        margin_score=60.0,
        time_score=recent_time_score,
        last3f_score=late_kick_score,
        horse_ability_score=avg_race_score if race_power is None else race_power,
        race_power=avg_race_score if race_power is None else race_power,
        early_aggressiveness=early_aggressiveness,
        mid_positioning=mid_positioning,
        late_kick_timing=late_kick_timing,
        sustain_speed=sustain_speed,
        time_reliability=time_reliability,
        recent_time_score=recent_time_score,
        late_kick_score=late_kick_score,
        avg_last3f=avg_last3f,
        best_last3f=best_last3f,
        last3f_consistency=last3f_consistency,
        late_gain_score=late_gain_score,
        early_push_score=early_push_score,
        mid_cruise_score=mid_cruise_score,
        fade_resistance_score=fade_resistance_score,
        sustain_speed_score=sustain_speed_score,
        pace_resilience_score=pace_resilience_score,
        agari_reliability=agari_reliability,
    )


class SimulatorTimelineTest(unittest.TestCase):
    def test_actual_style_equals_primary_style_except_jizai(self) -> None:
        simulator = RaceSimulator()
        adjusted = {"逃げ": 0.70, "先行": 0.10, "差し": 0.10, "追込": 0.10}
        rng = random.Random(42)

        for primary_style in [STYLE_RUNNER, STYLE_STALKER, STYLE_CLOSER, STYLE_DEEP_CLOSER]:
            self.assertEqual(
                simulator._sample_actual_style(primary_style, adjusted, rng),
                primary_style,
            )
        self.assertIn(
            simulator._sample_actual_style("自在", adjusted, rng),
            adjusted,
        )

    def test_controlled_timeline_20_percent_style_order(self) -> None:
        timeline = generate_controlled_race_timeline(self._controlled_horses(), RaceConfig.default().to_dict(), seed=3)
        snapshot = self._timeline_snapshot(timeline, 0.20)
        self.assertEqual(
            [horse["actual_running_style"] for horse in snapshot["horses"]],
            ["逃げ", "先行", "自在", "差し", "追込"],
        )

    def test_controlled_timeline_40_percent_style_order(self) -> None:
        timeline = generate_controlled_race_timeline(self._controlled_horses(), RaceConfig.default().to_dict(), seed=5)
        snapshot = self._timeline_snapshot(timeline, 0.40)
        self.assertEqual(
            [horse["actual_running_style"] for horse in snapshot["horses"]],
            ["逃げ", "先行", "自在", "差し", "追込"],
        )

    def test_prediction_score_not_used_in_timeline(self) -> None:
        horses = self._controlled_horses()
        horses[-1]["prediction_score"] = 100.0
        horses[-1]["score"] = 100.0

        timeline = generate_controlled_race_timeline(horses, RaceConfig.default().to_dict(), seed=7)
        snapshot = self._timeline_snapshot(timeline, 0.20)
        deep = self._horse_by_number(snapshot, 5)

        self.assertEqual(deep["actual_running_style"], "追込")
        self.assertGreaterEqual(int(deep["rank"]), 4)
        self.assertNotIn("prediction_score", deep)
        self.assertNotIn("score", deep)

    def test_midfield_spread_is_visible_during_formation(self) -> None:
        timeline = generate_controlled_race_timeline(self._controlled_horses(), RaceConfig.default().to_dict(), seed=11)
        for progress in [0.20, 0.40]:
            snapshot = self._timeline_snapshot(timeline, progress)
            positions = [float(horse["position_m"]) for horse in snapshot["horses"]]
            spread = max(positions) - min(positions)

            self.assertGreaterEqual(spread, 40.0)
            self.assertLessEqual(spread, 115.0)

    def test_late_phase_allows_overtake(self) -> None:
        horses = [
            {"horse_name": "Front", "horse_number": 1, "frame": 1, "actual_running_style": "逃げ", "race_power": 40, "stamina": 30, "acceleration": 30},
            {"horse_name": "Stalker", "horse_number": 2, "frame": 3, "actual_running_style": "先行", "race_power": 45, "stamina": 35, "acceleration": 35},
            {"horse_name": "Closer", "horse_number": 3, "frame": 5, "actual_running_style": "差し", "race_power": 95, "stamina": 100, "acceleration": 100},
            {"horse_name": "Deep", "horse_number": 4, "frame": 7, "actual_running_style": "追込", "race_power": 90, "stamina": 95, "acceleration": 95},
        ]
        timeline = generate_controlled_race_timeline(horses, RaceConfig.default().to_dict(), seed=13)
        middle = self._timeline_snapshot(timeline, 0.50)
        late = self._timeline_snapshot(timeline, 0.95)
        middle_rank = self._horse_by_number(middle, 3)["rank"]
        late_rank = self._horse_by_number(late, 3)["rank"]

        self.assertLess(int(late_rank), int(middle_rank))

    def test_mid_race_free_allows_ability_move_after_45_percent(self) -> None:
        horses = [
            {"horse_name": "Front", "horse_number": 1, "frame": 1, "actual_running_style": STYLE_RUNNER, "race_power": 60, "stamina": 58, "acceleration": 55, "sustain_speed": 0.50, "mid_positioning": 0.50, "recent_time_score": 55},
            {"horse_name": "Stalker", "horse_number": 2, "frame": 3, "actual_running_style": STYLE_STALKER, "race_power": 61, "stamina": 58, "acceleration": 55, "sustain_speed": 0.50, "mid_positioning": 0.50, "recent_time_score": 55},
            {"horse_name": "Closer", "horse_number": 3, "frame": 5, "actual_running_style": STYLE_CLOSER, "race_power": 92, "stamina": 92, "acceleration": 70, "sustain_speed": 0.95, "mid_positioning": 0.90, "recent_time_score": 92},
        ]
        timeline = generate_controlled_race_timeline(horses, {"distance": 2000, "pace": "medium"}, seed=77)
        formation = self._timeline_snapshot(timeline, 0.40)
        mid_free = self._timeline_snapshot(timeline, 0.65)
        closer_formation_gap = float(self._horse_by_number(formation, 3)["gap_from_leader"])
        closer_mid_gap = float(self._horse_by_number(mid_free, 3)["gap_from_leader"])

        self.assertEqual(mid_free.get("phase"), "mid_race_transition")
        self.assertLess(closer_mid_gap, closer_formation_gap)
        self.assertGreaterEqual(closer_mid_gap, 28.0)

    def test_high_final_performance_closer_can_pass_weak_runner(self) -> None:
        horses = [
            {"horse_name": "Front", "horse_number": 1, "frame": 1, "actual_running_style": STYLE_RUNNER, "race_power": 40, "stamina": 35, "acceleration": 35, "last3f_score": 35},
            {"horse_name": "Stalker", "horse_number": 2, "frame": 2, "actual_running_style": STYLE_STALKER, "race_power": 48, "stamina": 45, "acceleration": 42, "last3f_score": 42},
            {"horse_name": "Closer", "horse_number": 3, "frame": 5, "actual_running_style": STYLE_CLOSER, "race_power": 95, "stamina": 96, "acceleration": 100, "last3f_score": 100},
        ]
        timeline = generate_controlled_race_timeline(horses, {"distance": 2000, "pace": "medium"}, seed=101)
        final = timeline[-1]

        self.assertEqual(int(final["horses"][0]["horse_number"]), 3)
        self.assertGreater(float(self._horse_by_number(final, 3)["final_performance_score"]), float(self._horse_by_number(final, 1)["final_performance_score"]))

    def test_high_pace_helps_deep_closer_finish_near_top(self) -> None:
        horses = [
            {"horse_name": "Front", "horse_number": 1, "frame": 1, "actual_running_style": STYLE_RUNNER, "race_power": 60, "stamina": 45, "acceleration": 45, "last3f_score": 45},
            {"horse_name": "Stalker", "horse_number": 2, "frame": 2, "actual_running_style": STYLE_STALKER, "race_power": 62, "stamina": 55, "acceleration": 50, "last3f_score": 50},
            {"horse_name": "Deep", "horse_number": 4, "frame": 7, "actual_running_style": STYLE_DEEP_CLOSER, "race_power": 92, "stamina": 94, "acceleration": 98, "last3f_score": 98, "pace_fit": 90},
        ]
        timeline = generate_controlled_race_timeline(horses, {"distance": 2000, "pace": "high"}, seed=102)
        final = timeline[-1]

        self.assertLessEqual(int(self._horse_by_number(final, 4)["rank"]), 2)

    def test_slow_pace_can_still_allow_much_stronger_closer_to_win(self) -> None:
        horses = [
            {"horse_name": "Front", "horse_number": 1, "frame": 1, "actual_running_style": STYLE_RUNNER, "race_power": 55, "stamina": 50, "acceleration": 45, "last3f_score": 45},
            {"horse_name": "Stalker", "horse_number": 2, "frame": 2, "actual_running_style": STYLE_STALKER, "race_power": 58, "stamina": 55, "acceleration": 48, "last3f_score": 48},
            {"horse_name": "Closer", "horse_number": 3, "frame": 5, "actual_running_style": STYLE_CLOSER, "race_power": 98, "stamina": 98, "acceleration": 100, "last3f_score": 100},
        ]
        timeline = generate_controlled_race_timeline(horses, {"distance": 2000, "pace": "slow"}, seed=103)
        final = timeline[-1]

        self.assertEqual(int(final["horses"][0]["horse_number"]), 3)

    def test_final_order_is_not_locked_to_running_style_order(self) -> None:
        horses = [
            {"horse_name": "Front", "horse_number": 1, "frame": 1, "actual_running_style": STYLE_RUNNER, "race_power": 42, "stamina": 35, "acceleration": 35, "last3f_score": 35},
            {"horse_name": "Stalker", "horse_number": 2, "frame": 2, "actual_running_style": STYLE_STALKER, "race_power": 45, "stamina": 42, "acceleration": 40, "last3f_score": 40},
            {"horse_name": "Closer", "horse_number": 3, "frame": 5, "actual_running_style": STYLE_CLOSER, "race_power": 96, "stamina": 95, "acceleration": 100, "last3f_score": 100},
            {"horse_name": "Deep", "horse_number": 4, "frame": 7, "actual_running_style": STYLE_DEEP_CLOSER, "race_power": 90, "stamina": 92, "acceleration": 95, "last3f_score": 95},
        ]
        timeline = generate_controlled_race_timeline(horses, {"distance": 2000, "pace": "medium"}, seed=104)
        final_styles = [horse["actual_running_style"] for horse in timeline[-1]["horses"]]

        self.assertNotEqual(final_styles, [STYLE_RUNNER, STYLE_STALKER, STYLE_CLOSER, STYLE_DEEP_CLOSER])

    def test_no_mass_tie_at_finish(self) -> None:
        timeline = generate_controlled_race_timeline(self._controlled_horses(), {"distance": 2000, "pace": "medium"}, seed=201)
        final = timeline[-1]
        rounded_positions = [round(float(horse["position_m"]), 2) for horse in final["horses"]]

        self.assertGreaterEqual(len(set(rounded_positions)), len(rounded_positions) - 1)

    def test_finish_positions_are_spread(self) -> None:
        timeline = generate_controlled_race_timeline(self._controlled_horses(), {"distance": 2000, "pace": "medium"}, seed=202)
        final = timeline[-1]
        positions = [float(horse["position_m"]) for horse in final["horses"]]

        self.assertGreater(max(positions) - min(positions), 3.0)

    def test_style_not_always_dominant(self) -> None:
        horses = [
            {"horse_name": "Front", "horse_number": 1, "frame": 1, "actual_running_style": STYLE_RUNNER, "race_power": 92, "stamina": 96, "acceleration": 82, "last3f_score": 78},
            {"horse_name": "Stalker", "horse_number": 2, "frame": 2, "actual_running_style": STYLE_STALKER, "race_power": 55, "stamina": 54, "acceleration": 50, "last3f_score": 50},
            {"horse_name": "Closer", "horse_number": 3, "frame": 5, "actual_running_style": STYLE_CLOSER, "race_power": 55, "stamina": 54, "acceleration": 50, "last3f_score": 50},
            {"horse_name": "Deep", "horse_number": 4, "frame": 7, "actual_running_style": STYLE_DEEP_CLOSER, "race_power": 90, "stamina": 94, "acceleration": 98, "last3f_score": 98},
        ]
        timeline = generate_controlled_race_timeline(horses, {"distance": 2000, "pace": "high"}, seed=203)
        top_two_styles = {horse["actual_running_style"] for horse in timeline[-1]["horses"][:2]}

        self.assertFalse(top_two_styles.issubset({STYLE_STALKER, STYLE_CLOSER}))

    def test_escape_horse_not_always_fades(self) -> None:
        horses = [
            {
                "horse_name": "StrongFront",
                "horse_number": 1,
                "frame": 1,
                "actual_running_style": STYLE_RUNNER,
                "race_power": 95,
                "stamina": 98,
                "acceleration": 76,
                "last3f_score": 78,
                "recent_time_score": 96,
                "pace_fit": 92,
            },
            {"horse_name": "Stalker", "horse_number": 2, "frame": 2, "actual_running_style": STYLE_STALKER, "race_power": 68, "stamina": 66, "acceleration": 62, "last3f_score": 62},
            {"horse_name": "Closer", "horse_number": 3, "frame": 5, "actual_running_style": STYLE_CLOSER, "race_power": 72, "stamina": 70, "acceleration": 72, "last3f_score": 72},
        ]

        timeline = generate_controlled_race_timeline(
            horses,
            {"distance": 2400, "pace": "high", "track_condition": "良"},
            seed=205,
        )
        final_front = self._horse_by_number(timeline[-1], 1)

        self.assertLessEqual(int(final_front["rank"]), 2)
        self.assertLessEqual(float(final_front["gap_from_leader"]), 8.0)
        self.assertLess(float(final_front["escape_fade"]), 0.05)

    def test_escape_fade_depends_on_fade_resistance(self) -> None:
        base = {
            "actual_running_style": STYLE_RUNNER,
            "stamina": 58.0,
            "race_pace": "high",
            "distance": 2200.0,
            "track_condition": "良",
            "pace_resilience_score": 55.0,
        }
        weak_hold = dict(base, fade_resistance_score=30.0)
        strong_hold = dict(base, fade_resistance_score=90.0)

        self.assertGreater(_formation_escape_fade(weak_hold), _formation_escape_fade(strong_hold))

    def test_senko_not_always_top(self) -> None:
        horses = [
            {"horse_name": "Front", "horse_number": 1, "frame": 1, "actual_running_style": STYLE_RUNNER, "race_power": 72, "stamina": 74, "acceleration": 70, "last3f_score": 70},
            {"horse_name": "SenkoA", "horse_number": 2, "frame": 2, "actual_running_style": STYLE_STALKER, "race_power": 72, "stamina": 72, "acceleration": 70, "last3f_score": 70},
            {"horse_name": "SenkoB", "horse_number": 3, "frame": 3, "actual_running_style": STYLE_STALKER, "race_power": 71, "stamina": 71, "acceleration": 69, "last3f_score": 69},
            {"horse_name": "Sashi", "horse_number": 4, "frame": 5, "actual_running_style": STYLE_CLOSER, "race_power": 73, "stamina": 72, "acceleration": 74, "last3f_score": 74},
            {"horse_name": "Deep", "horse_number": 5, "frame": 7, "actual_running_style": STYLE_DEEP_CLOSER, "race_power": 72, "stamina": 72, "acceleration": 75, "last3f_score": 75},
        ]
        timeline = generate_controlled_race_timeline(horses, {"distance": 2000, "pace": "medium"}, seed=301)
        top_three_styles = [horse["actual_running_style"] for horse in timeline[-1]["horses"][:3]]

        self.assertNotEqual(top_three_styles.count(STYLE_STALKER), 3)

    def test_high_pace_penalizes_low_stamina_senko(self) -> None:
        horses = [
            {"horse_name": "Front", "horse_number": 1, "frame": 1, "actual_running_style": STYLE_RUNNER, "race_power": 70, "stamina": 70, "acceleration": 64, "last3f_score": 64},
            {"horse_name": "LowStaminaSenko", "horse_number": 2, "frame": 2, "actual_running_style": STYLE_STALKER, "race_power": 74, "stamina": 35, "acceleration": 72, "last3f_score": 72},
            {"horse_name": "Sashi", "horse_number": 3, "frame": 5, "actual_running_style": STYLE_CLOSER, "race_power": 72, "stamina": 72, "acceleration": 78, "last3f_score": 80},
        ]
        timeline = generate_controlled_race_timeline(horses, {"distance": 2000, "pace": "high"}, seed=302)
        final_senko = self._horse_by_number(timeline[-1], 2)
        final_sashi = self._horse_by_number(timeline[-1], 3)

        self.assertGreater(float(final_senko["fade_penalty"]), 0.04)
        self.assertGreater(int(final_senko["rank"]), int(final_sashi["rank"]))

    def test_medium_pace_is_not_senko_favored(self) -> None:
        horses = [
            {"horse_name": "Senko", "horse_number": 2, "frame": 2, "actual_running_style": STYLE_STALKER, "race_power": 75, "stamina": 75, "acceleration": 75, "last3f_score": 75, "pace_fit": 60},
            {"horse_name": "Sashi", "horse_number": 3, "frame": 5, "actual_running_style": STYLE_CLOSER, "race_power": 75, "stamina": 75, "acceleration": 75, "last3f_score": 75, "pace_fit": 60},
        ]
        timeline = generate_controlled_race_timeline(horses, {"distance": 2000, "pace": "medium"}, seed=303)
        final_senko = self._horse_by_number(timeline[-1], 2)
        final_sashi = self._horse_by_number(timeline[-1], 3)

        self.assertEqual(float(final_senko["pace_fit_score"]), 60.0)
        self.assertEqual(float(final_sashi["pace_fit_score"]), 60.0)
        self.assertAlmostEqual(float(final_senko["final_stretch_score"]), float(final_sashi["final_stretch_score"]))

    def test_final_stretch_not_style_order_only(self) -> None:
        horses = [
            {"horse_name": "Runner", "horse_number": 1, "frame": 1, "actual_running_style": STYLE_RUNNER, "stamina": 45, "late_kick_score": 35, "late_gain_score": 35, "recent_time_score": 45},
            {"horse_name": "Stalker", "horse_number": 2, "frame": 2, "actual_running_style": STYLE_STALKER, "stamina": 55, "late_kick_score": 50, "late_gain_score": 50, "recent_time_score": 55},
            {"horse_name": "Closer", "horse_number": 3, "frame": 5, "actual_running_style": STYLE_CLOSER, "stamina": 92, "late_kick_score": 98, "late_gain_score": 95, "recent_time_score": 90},
        ]
        final = generate_controlled_race_timeline(horses, {"distance": 2000, "pace": "high"}, seed=812)[-1]
        self.assertEqual(int(final["horses"][0]["horse_number"]), 3)

    def test_high_late_kick_can_overturn_style_order(self) -> None:
        horses = [
            {"horse_name": "Runner", "horse_number": 1, "frame": 1, "actual_running_style": STYLE_RUNNER, "stamina": 50, "late_kick_score": 40, "late_gain_score": 40},
            {"horse_name": "Deep", "horse_number": 4, "frame": 7, "actual_running_style": STYLE_DEEP_CLOSER, "stamina": 90, "late_kick_score": 100, "late_gain_score": 100, "recent_time_score": 95},
        ]
        timeline = generate_controlled_race_timeline(horses, {"distance": 1800, "pace": "high"}, seed=813)
        self.assertGreater(int(self._horse_by_number(self._timeline_snapshot(timeline, 0.50), 4)["rank"]), 1)
        self.assertEqual(int(timeline[-1]["horses"][0]["horse_number"]), 4)

    def test_weight_penalty_affects_final_stretch_score(self) -> None:
        base = {
            "late_kick_score": 80.0,
            "late_gain_score": 75.0,
            "pace_fit_score": 70.0,
            "stamina": 75.0,
            "recent_time_score": 70.0,
        }
        light = _formation_final_stretch_score({**base, "weight_penalty": -4.0})
        heavy = _formation_final_stretch_score({**base, "weight_penalty": 6.0})
        self.assertGreater(light, heavy)

    def test_sashi_can_overtake_senko_late(self) -> None:
        horses = [
            {"horse_name": "Senko", "horse_number": 2, "frame": 2, "actual_running_style": STYLE_STALKER, "race_power": 72, "stamina": 58, "acceleration": 58, "last3f_score": 58},
            {"horse_name": "Sashi", "horse_number": 3, "frame": 5, "actual_running_style": STYLE_CLOSER, "race_power": 82, "stamina": 82, "acceleration": 96, "last3f_score": 96, "late_kick_timing": 0.85},
        ]
        timeline = generate_controlled_race_timeline(horses, {"distance": 2000, "pace": "medium"}, seed=304)
        middle_senko = self._horse_by_number(self._timeline_snapshot(timeline, 0.50), 2)
        final_senko = self._horse_by_number(timeline[-1], 2)
        final_sashi = self._horse_by_number(timeline[-1], 3)

        self.assertLess(int(middle_senko["rank"]), 2)
        self.assertLess(int(final_sashi["rank"]), int(final_senko["rank"]))

    def test_actual_running_style_not_overwritten(self) -> None:
        horses = self._controlled_horses()
        initial_styles = {int(horse["horse_number"]): horse["actual_running_style"] for horse in horses}
        timeline = generate_controlled_race_timeline(horses, {"distance": 2000, "pace": "medium"}, seed=401)

        for frame in timeline:
            for horse in frame["horses"]:
                self.assertEqual(horse["actual_running_style"], initial_styles[int(horse["horse_number"])])

    def test_actual_running_style_fixed_from_start_to_finish(self) -> None:
        horses = self._controlled_horses()
        initial_styles = {int(horse["horse_number"]): horse["actual_running_style"] for horse in horses}
        timeline = generate_controlled_race_timeline(horses, {"distance": 2000, "pace": "medium"}, seed=406)

        for frame in [timeline[0], self._timeline_snapshot(timeline, 0.50), timeline[-1]]:
            for horse in frame["horses"]:
                number = int(horse["horse_number"])
                self.assertEqual(horse["actual_running_style_fixed"], initial_styles[number])
                self.assertEqual(horse["actual_running_style"], initial_styles[number])

    def test_actual_style_same_first_and_final_frame(self) -> None:
        timeline = generate_controlled_race_timeline(self._controlled_horses(), {"distance": 2000, "pace": "medium"}, seed=410)
        first_styles = {int(horse["horse_number"]): horse["actual_running_style"] for horse in timeline[0]["horses"]}
        final_styles = {int(horse["horse_number"]): horse["actual_running_style"] for horse in timeline[-1]["horses"]}

        self.assertEqual(first_styles, final_styles)

    def test_actual_running_style_not_recomputed_from_rank(self) -> None:
        horses = [
            {"horse_name": "Senko", "horse_number": 2, "frame": 2, "actual_running_style": STYLE_STALKER, "race_power": 98, "stamina": 98, "acceleration": 95, "late_kick_score": 95},
            {"horse_name": "Closer", "horse_number": 3, "frame": 5, "actual_running_style": STYLE_CLOSER, "race_power": 45, "stamina": 45, "acceleration": 45, "late_kick_score": 45},
        ]
        timeline = generate_controlled_race_timeline(horses, {"distance": 2000, "pace": "medium"}, seed=407)
        winner = timeline[-1]["horses"][0]

        self.assertEqual(int(winner["horse_number"]), 2)
        self.assertEqual(winner["actual_running_style"], STYLE_STALKER)
        self.assertEqual(winner["actual_running_style_fixed"], STYLE_STALKER)

    def test_actual_running_style_not_changed_by_position(self) -> None:
        horses = [
            {"horse_name": "Sashi", "horse_number": 3, "frame": 5, "actual_running_style": STYLE_CLOSER, "horse_ability_score": 98, "race_power": 98, "stamina": 98, "acceleration": 100, "late_kick_score": 100},
            {"horse_name": "Deep", "horse_number": 4, "frame": 7, "actual_running_style": STYLE_DEEP_CLOSER, "horse_ability_score": 40, "race_power": 40, "stamina": 40, "acceleration": 40, "late_kick_score": 40},
        ]
        timeline = generate_controlled_race_timeline(horses, {"distance": 2000, "pace": "medium"}, seed=412)
        winner = timeline[-1]["horses"][0]

        self.assertEqual(int(winner["horse_number"]), 3)
        self.assertEqual(winner["actual_running_style"], STYLE_CLOSER)

    def test_actual_running_style_not_inferred_from_rank_or_gap(self) -> None:
        horses = [
            {"horse_name": "Senko", "horse_number": 1, "frame": 1, "actual_running_style": STYLE_STALKER, "late_kick_score": 92, "late_gain_score": 90, "stamina": 88},
            {"horse_name": "Runner", "horse_number": 2, "frame": 2, "actual_running_style": STYLE_RUNNER, "late_kick_score": 35, "late_gain_score": 35, "stamina": 42},
        ]
        timeline = generate_controlled_race_timeline(horses, {"distance": 1600, "pace": "high"}, seed=704)
        final_senko = self._horse_by_number(timeline[-1], 1)

        self.assertEqual(int(final_senko["rank"]), 1)
        self.assertEqual(final_senko["actual_running_style_fixed"], STYLE_STALKER)
        self.assertEqual(final_senko["actual_running_style"], STYLE_STALKER)

    def test_no_style_reclassification_from_position(self) -> None:
        horses = [
            {"horse_name": "Sashi", "horse_number": 3, "frame": 5, "actual_running_style": STYLE_CLOSER, "race_power": 99, "stamina": 99, "acceleration": 100, "late_kick_score": 100},
            {"horse_name": "Runner", "horse_number": 1, "frame": 1, "actual_running_style": STYLE_RUNNER, "race_power": 40, "stamina": 35, "acceleration": 35, "late_kick_score": 35},
        ]
        timeline = generate_controlled_race_timeline(horses, {"distance": 1800, "pace": "high"}, seed=705)
        final_sashi = self._horse_by_number(timeline[-1], 3)

        self.assertEqual(int(final_sashi["rank"]), 1)
        self.assertEqual(final_sashi["actual_running_style"], STYLE_CLOSER)
        self.assertEqual(final_sashi["actual_running_style_fixed"], STYLE_CLOSER)

    def test_senko_winner_still_senko(self) -> None:
        horses = [
            {"horse_name": "Senko", "horse_number": 2, "frame": 2, "actual_running_style": STYLE_STALKER, "race_power": 98, "stamina": 98, "acceleration": 96, "late_kick_score": 96, "fade_resistance_score": 98},
            {"horse_name": "Closer", "horse_number": 3, "frame": 5, "actual_running_style": STYLE_CLOSER, "race_power": 55, "stamina": 55, "acceleration": 55, "late_kick_score": 55},
        ]
        timeline = generate_controlled_race_timeline(horses, {"distance": 2000, "pace": "medium"}, seed=408)
        winner = timeline[-1]["horses"][0]

        self.assertEqual(int(winner["horse_number"]), 2)
        self.assertEqual(winner["actual_running_style"], STYLE_STALKER)

    def test_sashi_leader_late_still_sashi(self) -> None:
        horses = [
            {"horse_name": "Front", "horse_number": 1, "frame": 1, "actual_running_style": STYLE_RUNNER, "race_power": 42, "stamina": 35, "acceleration": 35, "late_kick_score": 35, "fade_resistance_score": 20},
            {"horse_name": "Senko", "horse_number": 2, "frame": 2, "actual_running_style": STYLE_STALKER, "race_power": 45, "stamina": 42, "acceleration": 40, "late_kick_score": 40},
            {"horse_name": "Sashi", "horse_number": 3, "frame": 5, "actual_running_style": STYLE_CLOSER, "race_power": 98, "stamina": 98, "acceleration": 100, "late_kick_score": 100, "late_gain_score": 95},
        ]
        timeline = generate_controlled_race_timeline(horses, {"distance": 2000, "pace": "medium"}, seed=409)
        leader_late = timeline[-1]["horses"][0]

        self.assertEqual(int(leader_late["horse_number"]), 3)
        self.assertEqual(leader_late["actual_running_style"], STYLE_CLOSER)
        self.assertEqual(leader_late["actual_running_style_fixed"], STYLE_CLOSER)

    def test_nige_ahead_of_senko_early(self) -> None:
        horses = [
            {"horse_name": "Front", "horse_number": 1, "frame": 1, "actual_running_style": STYLE_RUNNER, "race_power": 70, "stamina": 70, "acceleration": 65},
            {"horse_name": "Stalker", "horse_number": 2, "frame": 2, "actual_running_style": STYLE_STALKER, "race_power": 95, "stamina": 95, "acceleration": 95},
            {"horse_name": "Closer", "horse_number": 3, "frame": 5, "actual_running_style": STYLE_CLOSER, "race_power": 70, "stamina": 70, "acceleration": 70},
        ]
        timeline = generate_controlled_race_timeline(horses, {"distance": 2000, "pace": "medium"}, seed=405)
        snapshot = self._timeline_snapshot(timeline, 0.20)
        nige = self._horse_by_number(snapshot, 1)
        senko = self._horse_by_number(snapshot, 2)

        self.assertLess(float(nige["gap_from_leader"]), float(senko["gap_from_leader"]))
        self.assertGreaterEqual(float(senko["gap_from_leader"]) - float(nige["gap_from_leader"]), 8.0)

    def test_senko_does_not_enter_nige_gap_before_stretch(self) -> None:
        horses = [
            {"horse_name": "Front", "horse_number": 1, "frame": 1, "actual_running_style": STYLE_RUNNER, "race_power": 62, "stamina": 60, "acceleration": 60},
            {"horse_name": "StrongSenko", "horse_number": 2, "frame": 2, "actual_running_style": STYLE_STALKER, "race_power": 95, "stamina": 95, "acceleration": 95, "recent_time_score": 95},
            {"horse_name": "Closer", "horse_number": 3, "frame": 5, "actual_running_style": STYLE_CLOSER, "race_power": 70, "stamina": 70, "acceleration": 70},
        ]
        timeline = generate_controlled_race_timeline(horses, {"distance": 2000, "pace": "medium"}, seed=402)

        for frame in timeline:
            progress = float(frame["progress"])
            final_stretch_start = float(self._horse_by_number(frame, 2).get("final_stretch_start_progress", 0.70))
            if 0.20 <= progress < final_stretch_start:
                senko = self._horse_by_number(frame, 2)
                self.assertGreaterEqual(float(senko["gap_from_leader"]), EARLY_STYLE_GAP_RANGES[STYLE_STALKER][0])

    def test_senko_stays_senko_band_before_stretch(self) -> None:
        horses = [
            {"horse_name": "Front", "horse_number": 1, "frame": 1, "actual_running_style": STYLE_RUNNER, "horse_ability_score": 80, "race_power": 80, "stamina": 75, "acceleration": 70},
            {"horse_name": "Senko", "horse_number": 2, "frame": 2, "actual_running_style": STYLE_STALKER, "horse_ability_score": 42, "race_power": 42, "stamina": 40, "acceleration": 40},
            {"horse_name": "Sashi", "horse_number": 3, "frame": 5, "actual_running_style": STYLE_CLOSER, "horse_ability_score": 75, "race_power": 75, "stamina": 75, "acceleration": 75},
        ]
        timeline = generate_controlled_race_timeline(horses, {"distance": 2000, "pace": "medium"}, seed=411)

        for frame in timeline:
            senko = self._horse_by_number(frame, 2)
            progress = float(frame["progress"])
            final_stretch_start = float(senko.get("final_stretch_start_progress", 0.70))
            if progress < final_stretch_start:
                low, high = EARLY_STYLE_GAP_RANGES[STYLE_STALKER]
                self.assertGreaterEqual(float(senko["gap_from_leader"]), low)
                self.assertLessEqual(float(senko["gap_from_leader"]), high)

    def test_low_ability_horse_not_randomly_top_too_often(self) -> None:
        horses = [
            {"horse_name": "HighA", "horse_number": 1, "frame": 1, "actual_running_style": STYLE_RUNNER, "horse_ability_score": 88, "race_power": 88, "stamina": 86, "acceleration": 82, "late_kick_score": 78},
            {"horse_name": "HighB", "horse_number": 2, "frame": 2, "actual_running_style": STYLE_STALKER, "horse_ability_score": 86, "race_power": 86, "stamina": 84, "acceleration": 80, "late_kick_score": 80},
            {"horse_name": "Low", "horse_number": 3, "frame": 6, "actual_running_style": STYLE_CLOSER, "horse_ability_score": 38, "race_power": 40, "stamina": 45, "acceleration": 45, "late_kick_score": 45},
        ]
        top_count = 0
        for seed in range(430, 450):
            timeline = generate_controlled_race_timeline(horses, {"distance": 2000, "pace": "medium"}, seed=seed)
            if int(timeline[-1]["horses"][0]["horse_number"]) == 3:
                top_count += 1

        self.assertLessEqual(top_count, 1)

    def test_style_bands_preserved_until_final_stretch(self) -> None:
        timeline = generate_controlled_race_timeline(self._controlled_horses(), {"distance": 2000, "pace": "medium"}, seed=403)

        for progress in [0.20, 0.50]:
            snapshot = self._timeline_snapshot(timeline, progress)
            for horse in snapshot["horses"]:
                style = str(horse["actual_running_style"])
                low, high = EARLY_STYLE_GAP_RANGES[style]
                gap = float(horse["gap_from_leader"])
                self.assertGreaterEqual(gap, low)
                self.assertLessEqual(gap, high)

    def test_sashi_with_high_late_kick_can_attack_in_stretch(self) -> None:
        horses = [
            {"horse_name": "Front", "horse_number": 1, "frame": 1, "actual_running_style": STYLE_RUNNER, "race_power": 78, "stamina": 76, "acceleration": 68, "late_kick_score": 58, "late_gain_score": 40},
            {"horse_name": "Sashi", "horse_number": 3, "frame": 5, "actual_running_style": STYLE_CLOSER, "race_power": 80, "stamina": 82, "acceleration": 92, "late_kick_score": 96, "late_gain_score": 88, "late_kick_timing": 0.88},
        ]
        timeline = generate_controlled_race_timeline(horses, {"distance": 2000, "pace": "medium", "straight_length": 500}, seed=404)
        before_stretch = self._timeline_snapshot(timeline, 0.74)
        late = self._timeline_snapshot(timeline, 0.95)
        closer_before = self._horse_by_number(before_stretch, 3)
        closer_late = self._horse_by_number(late, 3)

        self.assertGreater(float(closer_late["late_kick_score"]), 90.0)
        self.assertGreater(float(closer_late["straight_attack_score"]), float(self._horse_by_number(late, 1)["straight_attack_score"]))
        self.assertLess(float(closer_late["gap_from_leader"]), float(closer_before["gap_from_leader"]))

    def test_early_position_matches_actual_running_style(self) -> None:
        config = RaceConfig.default()
        abilities = [
            ability("Front", 1, 1, "逃げ"),
            ability("Stalker", 3, 2, "先行"),
            ability("Closer", 5, 3, "差し"),
            ability("Deep", 7, 4, "追込"),
        ]
        pace = RacePace(
            pace="medium",
            front_group_size=1,
            closer_advantage=1.0,
            style_groups={},
            front_pressure=1.5,
            front_group=["Front"],
            middle_group=["Stalker", "Closer"],
            back_group=["Deep"],
        )

        result = RaceSimulator().simulate(config=config, abilities=abilities, pace=pace, seed=7)

        self.assertTrue(result.race_timeline)
        snapshot = min(
            result.race_timeline,
            key=lambda frame: abs(
                max(float(horse["position_m"]) for horse in frame["horses"]) / config.distance - 0.20
            ),
        )
        by_style = {str(horse["actual_running_style"]): horse for horse in snapshot["horses"]}
        horse_count = len(abilities)

        self.assertLessEqual(int(by_style["逃げ"]["rank"]), math.ceil(horse_count * 0.25))
        self.assertLessEqual(int(by_style["先行"]["rank"]), math.ceil(horse_count * 0.40))
        self.assertGreater(int(by_style["差し"]["rank"]), math.ceil(horse_count * 0.25))
        self.assertGreaterEqual(int(by_style["追込"]["rank"]), math.ceil(horse_count * 0.60))

        timeline_df = result.timeline_dataframe()
        for column in ["time", "horse_number", "actual_running_style", "position_m", "rank", "lane", "race_power"]:
            self.assertIn(column, timeline_df.columns)
        self.assertNotIn("prediction_score", timeline_df.columns)

    def test_style_order_is_preserved_until_middle_phase(self) -> None:
        config = RaceConfig.default()
        abilities = [
            ability("Front", 1, 1, "逃げ"),
            ability("Stalker", 3, 2, "先行"),
            ability("Closer", 5, 3, "差し"),
            ability("Deep", 7, 4, "追込"),
        ]
        pace = RacePace(
            pace="medium",
            front_group_size=1,
            closer_advantage=1.0,
            style_groups={},
            front_pressure=1.5,
            front_group=["Front"],
            middle_group=["Stalker", "Closer"],
            back_group=["Deep"],
        )

        result = RaceSimulator().simulate(config=config, abilities=abilities, pace=pace, seed=11)

        for progress in [0.20, 0.40]:
            snapshot = min(
                result.race_timeline,
                key=lambda frame: abs(float(frame.get("progress", 0.0)) - progress),
            )
            by_style = {str(horse["actual_running_style"]): horse for horse in snapshot["horses"]}
            self.assertLess(int(by_style["逃げ"]["rank"]), int(by_style["先行"]["rank"]))
            self.assertLess(int(by_style["先行"]["rank"]), int(by_style["差し"]["rank"]))
            self.assertLess(int(by_style["差し"]["rank"]), int(by_style["追込"]["rank"]))

        late_snapshot = min(
            result.race_timeline,
            key=lambda frame: abs(float(frame.get("progress", 0.0)) - 0.55),
        )
        self.assertGreaterEqual(float(late_snapshot.get("progress", 0.0)), 0.45)

    def test_middle_field_spread_stays_in_visible_range(self) -> None:
        result = self._style_order_result(seed=17)

        for progress in [0.20, 0.40]:
            snapshot = self._snapshot(result, progress)
            positions = [float(horse["position_m"]) for horse in snapshot["horses"]]
            spread = max(positions) - min(positions)

            self.assertGreaterEqual(spread, 40.0)
            self.assertLessEqual(spread, 115.0)

    def test_high_ability_horse_can_improve_after_eighty_percent(self) -> None:
        horses = [
            {"horse_name": "Front", "horse_number": 1, "frame": 1, "actual_running_style": STYLE_RUNNER, "race_power": 52, "stamina": 52, "acceleration": 50, "late_kick_score": 50},
            {"horse_name": "Stalker", "horse_number": 2, "frame": 3, "actual_running_style": STYLE_STALKER, "race_power": 54, "stamina": 55, "acceleration": 52, "late_kick_score": 52},
            {"horse_name": "Closer", "horse_number": 3, "frame": 5, "actual_running_style": STYLE_CLOSER, "race_power": 92, "stamina": 92, "acceleration": 96, "late_kick_score": 96, "late_gain_score": 78},
            {"horse_name": "Deep", "horse_number": 4, "frame": 7, "actual_running_style": STYLE_DEEP_CLOSER, "race_power": 55, "stamina": 60, "acceleration": 58, "late_kick_score": 58},
        ]

        timeline = generate_controlled_race_timeline(horses, {"distance": 2000, "pace": "medium"}, seed=23)
        middle = self._timeline_snapshot(timeline, 0.50)
        late = self._timeline_snapshot(timeline, 0.95)
        middle_rank = self._horse_by_number(middle, 3)["rank"]
        late_rank = self._horse_by_number(late, 3)["rank"]

        self.assertLess(int(late_rank), int(middle_rank))

    def test_deep_closer_does_not_lead_before_sixty_percent(self) -> None:
        result = self._style_order_result(seed=29)

        for progress in [0.10, 0.20, 0.40, 0.59]:
            snapshot = self._snapshot(result, progress)
            deep = {str(horse["actual_running_style"]): horse for horse in snapshot["horses"]}["追込"]
            self.assertNotEqual(int(deep["rank"]), 1)

    def _style_order_result(self, seed: int = 11):
        config = RaceConfig.default()
        abilities = [
            ability("Front", 1, 1, "逃げ"),
            ability("Stalker", 3, 2, "先行"),
            ability("Closer", 5, 3, "差し"),
            ability("Deep", 7, 4, "追込"),
        ]
        return RaceSimulator().simulate(config=config, abilities=abilities, pace=self._pace(), seed=seed)

    def _pace(self) -> RacePace:
        return RacePace(
            pace="medium",
            front_group_size=1,
            closer_advantage=1.0,
            style_groups={},
            front_pressure=1.5,
            front_group=["Front"],
            middle_group=["Stalker", "Closer"],
            back_group=["Deep"],
        )

    def _snapshot(self, result, progress: float):
        return min(
            result.race_timeline,
            key=lambda frame: abs(float(frame.get("progress", 0.0)) - progress),
        )

    def _horse_by_number(self, snapshot, horse_number: int):
        for horse in snapshot["horses"]:
            if int(horse["horse_number"]) == horse_number:
                return horse
        raise AssertionError(f"horse not found: {horse_number}")

    def _timeline_snapshot(self, timeline, progress: float):
        return min(
            timeline,
            key=lambda frame: abs(float(frame.get("progress", 0.0)) - progress),
        )

    def _controlled_horses(self):
        return [
            {"horse_name": "Front", "horse_number": 1, "frame": 1, "actual_running_style": "逃げ", "race_power": 65, "stamina": 60, "acceleration": 55},
            {"horse_name": "Stalker", "horse_number": 2, "frame": 3, "actual_running_style": "先行", "race_power": 70, "stamina": 65, "acceleration": 60},
            {"horse_name": "Flexible", "horse_number": 3, "frame": 4, "actual_running_style": "自在", "race_power": 72, "stamina": 68, "acceleration": 68},
            {"horse_name": "Closer", "horse_number": 4, "frame": 5, "actual_running_style": "差し", "race_power": 80, "stamina": 75, "acceleration": 80},
            {"horse_name": "Deep", "horse_number": 5, "frame": 7, "actual_running_style": "追込", "race_power": 85, "stamina": 80, "acceleration": 88},
        ]


if __name__ == "__main__":
    unittest.main()
