from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from errors import RaceDataFetchError
from horse_analyzer import (
    HorseAnalyzer,
    RaceResult,
    RACE_LEVEL_WEIGHT,
    build_late_kick_score,
    build_race_tactics_profile,
    build_running_dynamics_profile,
    create_base_style_profile,
    estimate_mud_aptitude,
    estimate_mud_aptitude_with_source,
    is_graded_race,
    opponent_strength_score,
    parse_passing_order,
    parse_race_time,
    popularity_to_score,
    race_score,
    race_style_snapshot,
    primary_style_from_profile,
)
from race_config import HorseEntry


def race(order: str, finish: int, field_size: int | None = 18, index: int = 0) -> RaceResult:
    return RaceResult(
        race_name=f"case{index}",
        distance=1800,
        surface="芝",
        track_condition="良",
        finish_position=finish,
        margin=0.3,
        passing_order=order,
        final_3f=35.0,
        field_size=field_size,
    )


class StaticProvider:
    def __init__(self, results: list[RaceResult]) -> None:
        self.results = results
        self.requested_limits: list[int] = []

    def get_recent_results(self, horse_name: str, limit: int = 5) -> list[RaceResult]:
        self.requested_limits.append(limit)
        return self.results[:limit]


class EmptyProvider:
    def get_recent_results(self, horse_name: str, limit: int = 5) -> list[RaceResult]:
        return []


class HorseAnalyzerStyleTest(unittest.TestCase):
    def analyze_ability(self, orders: list[str], finishes: list[int]):
        provider = StaticProvider([race(order, finish, index=i) for i, (order, finish) in enumerate(zip(orders, finishes))])
        return HorseAnalyzer(provider).analyze(HorseEntry("Test", 1, 1))

    def test_parse_passing_order_variants(self) -> None:
        self.assertEqual(parse_passing_order("12-12-11-3"), [12, 12, 11, 3])
        self.assertEqual(parse_passing_order("(12,12,11,3)"), [12, 12, 11, 3])
        self.assertEqual(parse_passing_order("12 12 11 3"), [12, 12, 11, 3])
        self.assertEqual(parse_passing_order(""), [])
        self.assertEqual(parse_passing_order(None), [])

    def test_style_detection_uses_field_size_ratio(self) -> None:
        large_field = {"passing_order": "3-3-3-3", "finish": 3, "field_size": 18}
        small_field = {"passing_order": "3-3-3-3", "finish": 3, "field_size": 8}

        large_snapshot = race_style_snapshot(large_field)
        small_snapshot = race_style_snapshot(small_field)
        self.assertIsNotNone(large_snapshot)
        self.assertIsNotNone(small_snapshot)
        assert large_snapshot is not None and small_snapshot is not None
        self.assertAlmostEqual(large_snapshot.first_ratio, 3 / 18)
        self.assertAlmostEqual(small_snapshot.first_ratio, 3 / 8)
        self.assertGreater(
            create_base_style_profile([large_field])["逃げ"],
            create_base_style_profile([small_field])["逃げ"],
        )

    def test_primary_style_uses_field_size_ratio(self) -> None:
        large_field = [{"passing_order": "4-4-4-3", "finish": 3, "field_size": 18}] * 3
        small_field = [{"passing_order": "4-4-4-3", "finish": 3, "field_size": 8}] * 3

        self.assertEqual(primary_style_from_profile(create_base_style_profile(large_field)), "先行")
        self.assertEqual(primary_style_from_profile(create_base_style_profile(small_field)), "差し")

    def test_mud_aptitude_uses_race_history_first(self) -> None:
        wet_race = RaceResult(
            race_name="wet",
            distance=1800,
            surface="芝",
            track_condition="重",
            finish_position=1,
            margin=0.0,
            passing_order="2-2-1-1",
            final_3f=35.0,
            race_time_seconds=110.0,
            field_size=16,
        )
        score, source = estimate_mud_aptitude_with_source(
            [wet_race],
            {"pedigree_mud_score": 10.0},
        )

        self.assertEqual(source, "race_history")
        self.assertGreater(score, 50.0)
        self.assertEqual(score, estimate_mud_aptitude([wet_race], {"pedigree_mud_score": 10.0}))

    def test_mud_aptitude_uses_pedigree_when_no_mud_history(self) -> None:
        dry_race = race("3-3-3-2", 2)
        score, source = estimate_mud_aptitude_with_source(
            [dry_race],
            {"pedigree_mud_score": 74.0},
        )

        self.assertEqual(source, "pedigree")
        self.assertEqual(score, 74.0)
        self.assertEqual(estimate_mud_aptitude([dry_race]), 50.0)

    def test_field_size_aliases_and_missing_warning(self) -> None:
        for key in ["runners", "number_of_runners", "頭数", "出走頭数"]:
            snapshot = race_style_snapshot(
                {"passing_order": "3-3-3-3", "finish": 3, key: 8}
            )
            self.assertIsNotNone(snapshot)
            assert snapshot is not None
            self.assertEqual(snapshot.field_size, 8)
            self.assertFalse(snapshot.field_size_inferred)

        inferred = race_style_snapshot(
            {"passing_order": "12-12-11-3", "finish": 3}
        )
        self.assertIsNotNone(inferred)
        assert inferred is not None
        self.assertEqual(inferred.field_size, 13)
        self.assertTrue(inferred.field_size_inferred)
        self.assertIn("field_size欠損", inferred.field_size_warning)

    def test_parse_race_time_variants(self) -> None:
        self.assertEqual(parse_race_time("1:58.3"), 118.3)
        self.assertEqual(parse_race_time("2:12.5"), 132.5)
        self.assertEqual(parse_race_time("1.58.3"), 118.3)
        self.assertEqual(parse_race_time("2分12秒5"), 132.5)
        self.assertIsNone(parse_race_time(""))

    def test_race_level_weights_and_graded_detection(self) -> None:
        self.assertEqual(RACE_LEVEL_WEIGHT["G1"], 1.40)
        self.assertEqual(RACE_LEVEL_WEIGHT["未勝利"], 0.55)
        self.assertTrue(is_graded_race("G1"))
        self.assertTrue(is_graded_race("G2"))
        self.assertTrue(is_graded_race("G3"))
        self.assertFalse(is_graded_race("OP"))

    def test_opponent_strength_rewards_tough_popularity_context(self) -> None:
        favorite_win = RaceResult(
            race_name="OP",
            distance=1800,
            surface="芝",
            track_condition="良",
            finish_position=1,
            margin=0.1,
            passing_order="3-3-3-1",
            final_3f=34.8,
            field_size=12,
            race_class="OP",
            popularity="1",
        )
        longshot_g1_near_miss = RaceResult(
            race_name="G1",
            distance=2000,
            surface="芝",
            track_condition="良",
            finish_position=2,
            margin=0.2,
            passing_order="8-8-7-2",
            final_3f=34.2,
            field_size=18,
            race_class="G1",
            popularity="10",
        )

        self.assertGreater(opponent_strength_score(longshot_g1_near_miss), opponent_strength_score(favorite_win))
        self.assertGreater(race_score(longshot_g1_near_miss), 70.0)

    def test_popularity_weight_affects_ability(self) -> None:
        popular_good = [
            RaceResult(
                race_name="G1",
                distance=2000,
                surface="芝",
                track_condition="良",
                finish_position=1,
                margin=0.1,
                passing_order="3-3-3-1",
                final_3f=34.0,
                race_time_seconds=119.0,
                field_size=18,
                race_class="G1",
                popularity="1",
            )
            for _ in range(5)
        ]
        unpopular_poor = [
            RaceResult(
                race_name="1勝",
                distance=2000,
                surface="芝",
                track_condition="良",
                finish_position=12,
                margin=1.8,
                passing_order="10-10-10-12",
                final_3f=37.5,
                race_time_seconds=125.0,
                field_size=18,
                race_class="1勝",
                popularity="14",
            )
            for _ in range(5)
        ]
        popular_ability = HorseAnalyzer(StaticProvider(popular_good)).analyze(HorseEntry("Popular", 1, 1))
        poor_ability = HorseAnalyzer(StaticProvider(unpopular_poor)).analyze(HorseEntry("Poor", 2, 2))

        self.assertGreater(popularity_to_score("1", 18, 1), 85.0)
        self.assertGreater(popular_ability.popularity_score, poor_ability.popularity_score)
        self.assertGreater(popular_ability.horse_ability_score, poor_ability.horse_ability_score + 25.0)

    def test_horse_ability_score_without_popularity(self) -> None:
        favorite_rows = [
            RaceResult(
                race_name="G2",
                distance=2000,
                surface="芝",
                track_condition="良",
                finish_position=2,
                margin=0.2,
                passing_order="4-4-4-2",
                final_3f=34.8,
                race_time_seconds=120.0,
                field_size=18,
                race_class="G2",
                popularity="1",
            )
            for _ in range(5)
        ]
        outsider_rows = [
            RaceResult(
                race_name="G2",
                distance=2000,
                surface="芝",
                track_condition="良",
                finish_position=2,
                margin=0.2,
                passing_order="4-4-4-2",
                final_3f=34.8,
                race_time_seconds=120.0,
                field_size=18,
                race_class="G2",
                popularity="14",
            )
            for _ in range(5)
        ]

        favorite = HorseAnalyzer(StaticProvider(favorite_rows)).analyze(HorseEntry("Favorite", 1, 1))
        outsider = HorseAnalyzer(StaticProvider(outsider_rows)).analyze(HorseEntry("Outsider", 1, 1))

        self.assertGreater(favorite.popularity_score, outsider.popularity_score)
        self.assertAlmostEqual(favorite.horse_ability_score, outsider.horse_ability_score, places=6)
        self.assertAlmostEqual(favorite.race_power, outsider.race_power, places=6)

    def test_deep_closer_case(self) -> None:
        ability = self.analyze_ability(
            ["12-12-11-3", "10-10-9-4", "14-14-13-5"],
            [3, 4, 5],
        )
        profile = ability.base_style_profile
        self.assertGreater(profile["差し"] + profile["追込"], 0.80)
        self.assertGreater(profile["追込"], 0.35)

    def test_closer_case(self) -> None:
        ability = self.analyze_ability(
            ["8-8-7-4", "7-7-6-3", "9-8-8-5"],
            [4, 3, 5],
        )
        self.assertEqual(ability.primary_running_style, "差し")
        self.assertGreater(ability.base_style_profile["差し"], ability.base_style_profile["先行"])

    def test_front_runner_case(self) -> None:
        ability = self.analyze_ability(
            ["1-1-1-1", "1-1-2-2", "2-2-2-3"],
            [1, 2, 3],
        )
        profile = ability.base_style_profile
        self.assertEqual(ability.primary_running_style, "逃げ")
        self.assertGreater(profile["逃げ"] + profile["先行"], 0.75)

    def test_stalker_case(self) -> None:
        ability = self.analyze_ability(
            ["3-3-3-2", "4-4-4-4", "5-5-5-3"],
            [2, 4, 3],
        )
        self.assertEqual(ability.primary_running_style, "先行")
        self.assertGreater(ability.base_style_profile["先行"], 0.50)

    def test_versatile_case(self) -> None:
        ability = self.analyze_ability(
            ["2-2-2-2", "10-10-8-5", "5-5-4-3"],
            [2, 5, 3],
        )
        self.assertEqual(ability.primary_running_style, "自在")
        self.assertLess(max(ability.base_style_profile.values()), 0.40)

    def test_create_base_style_profile_is_normalized(self) -> None:
        profile = create_base_style_profile(
            [
                {"passing_order": "1-1-1-1", "finish": 1, "field_size": 16},
                {"passing_order": "3-3-3-2", "finish": 2, "field_size": 16},
            ]
        )
        self.assertAlmostEqual(sum(profile.values()), 1.0)
        self.assertEqual(set(profile), {"逃げ", "先行", "差し", "追込"})

    def test_style_uses_ten_starts_and_ability_uses_five(self) -> None:
        results = [race("12-12-11-3", 3, index=i) for i in range(10)]
        provider = StaticProvider(results)
        ability = HorseAnalyzer(provider).analyze(HorseEntry("Test", 1, 1))
        self.assertEqual(provider.requested_limits, [10])
        self.assertEqual(ability.style_sample_size, 10)
        self.assertEqual(len(ability.recent_results), 5)

    def test_tactics_profile_uses_recent_5_races(self) -> None:
        recent_five = [
            RaceResult(
                race_name=f"recent{i}",
                distance=1800,
                surface="芝",
                track_condition="良",
                finish_position=1,
                margin=0.1,
                passing_order="2-2-2-1",
                final_3f=34.5,
                race_time_seconds=108.5 + i * 0.1,
                field_size=16,
            )
            for i in range(5)
        ]
        old_extreme = RaceResult(
            race_name="old",
            distance=1800,
            surface="芝",
            track_condition="良",
            finish_position=16,
            margin=3.0,
            passing_order="16-16-16-16",
            final_3f=40.0,
            race_time_seconds=130.0,
            field_size=16,
        )

        profile = build_race_tactics_profile(recent_five + [old_extreme])
        control = build_race_tactics_profile(recent_five)

        self.assertAlmostEqual(profile["early_aggressiveness"], control["early_aggressiveness"], places=5)
        self.assertAlmostEqual(profile["recent_time_score"], control["recent_time_score"], places=5)
        self.assertGreater(profile["early_aggressiveness"], 0.80)
        self.assertGreater(profile["time_reliability"], 0.99)
        self.assertGreater(profile["recent_time_score"], 55.0)

    def test_late_kick_score_uses_last3f(self) -> None:
        fast_finish = build_late_kick_score(
            [
                RaceResult(
                    race_name="G1",
                    distance=2000,
                    surface="芝",
                    track_condition="良",
                    finish_position=2,
                    margin=0.1,
                    passing_order="10-10-8-2",
                    final_3f=33.2,
                    field_size=16,
                    race_class="G1",
                ),
                RaceResult(
                    race_name="G2",
                    distance=1800,
                    surface="芝",
                    track_condition="良",
                    finish_position=3,
                    margin=0.2,
                    passing_order="9-9-7-3",
                    final_3f=33.5,
                    field_size=16,
                    race_class="G2",
                ),
            ]
        )
        slow_finish = build_late_kick_score(
            [
                RaceResult(
                    race_name="OP",
                    distance=2000,
                    surface="芝",
                    track_condition="良",
                    finish_position=5,
                    margin=0.8,
                    passing_order="5-5-5-5",
                    final_3f=37.2,
                    field_size=16,
                    race_class="OP",
                )
            ]
        )

        self.assertGreater(fast_finish["late_kick_score"], slow_finish["late_kick_score"])
        self.assertLess(fast_finish["best_last3f"], 34.0)
        self.assertGreater(fast_finish["late_gain_score"], slow_finish["late_gain_score"])

    def test_running_dynamics_uses_time_passing_agari(self) -> None:
        strong = build_running_dynamics_profile(
            [
                RaceResult(
                    race_name="G1",
                    distance=2000,
                    surface="芝",
                    track_condition="良",
                    finish_position=1,
                    margin=0.1,
                    passing_order="1-1-1-1",
                    final_3f=33.8,
                    race_time_seconds=118.0,
                    field_size=16,
                    race_class="G1",
                ),
                RaceResult(
                    race_name="G2",
                    distance=1800,
                    surface="芝",
                    track_condition="良",
                    finish_position=2,
                    margin=0.2,
                    passing_order="2-2-2-2",
                    final_3f=34.0,
                    race_time_seconds=107.8,
                    field_size=16,
                    race_class="G2",
                ),
            ]
        )
        weak = build_running_dynamics_profile(
            [
                RaceResult(
                    race_name="1勝",
                    distance=2000,
                    surface="芝",
                    track_condition="良",
                    finish_position=10,
                    margin=2.0,
                    passing_order="12-12-12-10",
                    final_3f=38.0,
                    race_time_seconds=126.0,
                    field_size=16,
                    race_class="1勝",
                )
            ]
        )

        self.assertGreater(strong["early_push_score"], weak["early_push_score"])
        self.assertGreater(strong["mid_cruise_score"], weak["mid_cruise_score"])
        self.assertGreater(strong["late_kick_score"], weak["late_kick_score"])
        self.assertGreater(strong["sustain_speed_score"], weak["sustain_speed_score"])
        self.assertGreater(strong["agari_reliability"], 0.0)

    def test_missing_field_size_is_inferred(self) -> None:
        provider = StaticProvider([race("12-12-11-3", 3, field_size=None)])
        ability = HorseAnalyzer(provider).analyze(HorseEntry("Test", 1, 1))
        self.assertEqual(ability.style_sample_size, 1)
        self.assertGreater(ability.weighted_avg_first_ratio, 0)

    def test_empty_fetch_result_raises_without_synthetic_races(self) -> None:
        with self.assertRaises(RaceDataFetchError):
            HorseAnalyzer(EmptyProvider()).analyze(HorseEntry("あああああああ", 1, 1))


if __name__ == "__main__":
    unittest.main()
