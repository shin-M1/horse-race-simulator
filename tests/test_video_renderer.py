from __future__ import annotations

import sys
import unittest
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from video_renderer import (
    _build_side_scroll_payload,
    _should_show_final_result,
    _side_scroll_lane_y,
    _side_scroll_marker_screen_x,
    _side_scroll_screen_x,
    _timeline_with_result_display,
    _video_dimensions,
    interpolate_timeline,
    render_race_video_from_timeline,
    render_side_scroll_race_video,
)


class VideoRendererTest(unittest.TestCase):
    def test_video_dimensions_support_youtube_and_tiktok_labels(self) -> None:
        self.assertEqual(_video_dimensions("YouTube横長 16:9"), (1920, 1080))
        self.assertEqual(_video_dimensions("TikTok縦長 9:16"), (1080, 1920))
        self.assertEqual(_video_dimensions("youtube"), (1920, 1080))
        self.assertEqual(_video_dimensions("tiktok"), (1080, 1920))

    def test_video_renderer_returns_existing_file(self) -> None:
        timeline = self._sample_timeline()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "race.mp4"
            rendered = render_race_video_from_timeline(
                race_timeline=timeline,
                race_config={"course": "東京", "surface": "芝", "distance": 1000, "track_condition": "良"},
                horses=[
                    {"horse_name": "A", "horse_number": 1, "frame": 1},
                    {"horse_name": "B", "horse_number": 2, "frame": 2},
                ],
                output_path=str(path),
                video_format="youtube",
                fps=8,
                duration_sec=1,
            )

            self.assertTrue(Path(rendered).is_file())
            self.assertGreater(Path(rendered).stat().st_size, 0)

    def test_side_scroll_renderer_returns_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "side_scroll.mp4"
            rendered = render_side_scroll_race_video(
                race_timeline=self._sample_timeline(),
                race_config={"course": "東京", "surface": "芝", "distance": 1000, "track_condition": "良", "weather": "晴"},
                horses=[
                    {"horse_name": "A", "horse_number": 1, "frame": 1},
                    {"horse_name": "B", "horse_number": 2, "frame": 2},
                ],
                output_path=str(path),
                video_format="youtube",
                fps=8,
                duration_sec=1,
            )

            self.assertTrue(Path(rendered).is_file())
            self.assertGreater(Path(rendered).stat().st_size, 0)

    def test_interpolate_timeline_matches_requested_frame_count(self) -> None:
        self.assertEqual(len(interpolate_timeline(self._sample_timeline(), 31)), 31)

    def test_side_scroll_payload_uses_large_track_area(self) -> None:
        timeline = interpolate_timeline(self._sample_timeline(), 8)
        youtube = _build_side_scroll_payload(
            race_timeline=timeline,
            race_config={"course": "Tokyo", "surface": "turf", "distance": 1000},
            horses=[{"horse_number": 1, "frame": 1}, {"horse_number": 2, "frame": 2}],
            prediction_table=None,
            width=1920,
            height=1080,
            fps=8,
            duration_sec=1,
        )
        tiktok = _build_side_scroll_payload(
            race_timeline=timeline,
            race_config={"course": "Tokyo", "surface": "turf", "distance": 1000},
            horses=[{"horse_number": 1, "frame": 1}, {"horse_number": 2, "frame": 2}],
            prediction_table=None,
            width=1080,
            height=1920,
            fps=8,
            duration_sec=1,
        )

        self.assertAlmostEqual(youtube["track_top"] / 1080, 0.22, places=2)
        self.assertAlmostEqual(tiktok["track_top"] / 1920, 0.18, places=2)
        self.assertGreater(youtube["track_bottom"] / 1080, 0.90)
        self.assertGreater(tiktok["track_bottom"] / 1920, 0.90)
        self.assertEqual(youtube["visible_distance_m"], 200.0)
        self.assertEqual(youtube["distance_marker_interval"], 100)
        self.assertAlmostEqual(youtube["px_per_m"], 1920 / 200.0)

    def test_direction_left_scrolls_left_to_right(self) -> None:
        payload = _build_side_scroll_payload(
            race_timeline=interpolate_timeline(self._sample_timeline(), 8),
            race_config={"course": "Tokyo", "surface": "turf", "distance": 1000, "direction": "左"},
            horses=[{"horse_number": 1, "frame": 1}, {"horse_number": 2, "frame": 2}],
            prediction_table=None,
            width=1920,
            height=1080,
            fps=8,
            duration_sec=1,
        )

        self.assertEqual(payload["render_direction"], "left_to_right")
        self.assertLess(_side_scroll_screen_x(payload, 0.0, 0.0), _side_scroll_screen_x(payload, 100.0, 0.0))

    def test_direction_right_scrolls_right_to_left(self) -> None:
        payload = _build_side_scroll_payload(
            race_timeline=interpolate_timeline(self._sample_timeline(), 8),
            race_config={"course": "Tokyo", "surface": "turf", "distance": 1000, "direction": "右"},
            horses=[{"horse_number": 1, "frame": 1}, {"horse_number": 2, "frame": 2}],
            prediction_table=None,
            width=1920,
            height=1080,
            fps=8,
            duration_sec=1,
        )

        self.assertEqual(payload["render_direction"], "right_to_left")
        self.assertGreater(_side_scroll_screen_x(payload, 0.0, 0.0), _side_scroll_screen_x(payload, 100.0, 0.0))

    def test_start_goal_side_changes_by_direction(self) -> None:
        left_payload = _build_side_scroll_payload(
            race_timeline=interpolate_timeline(self._sample_timeline(), 8),
            race_config={"course": "Tokyo", "surface": "turf", "distance": 1000, "direction": "左"},
            horses=[{"horse_number": 1, "frame": 1}, {"horse_number": 2, "frame": 2}],
            prediction_table=None,
            width=1920,
            height=1080,
            fps=8,
            duration_sec=1,
        )
        right_payload = _build_side_scroll_payload(
            race_timeline=interpolate_timeline(self._sample_timeline(), 8),
            race_config={"course": "Tokyo", "surface": "turf", "distance": 1000, "direction": "右"},
            horses=[{"horse_number": 1, "frame": 1}, {"horse_number": 2, "frame": 2}],
            prediction_table=None,
            width=1920,
            height=1080,
            fps=8,
            duration_sec=1,
        )

        self.assertEqual(left_payload["start_gate_side"], "left")
        self.assertEqual(left_payload["goal_side"], "right")
        self.assertEqual(right_payload["start_gate_side"], "right")
        self.assertEqual(right_payload["goal_side"], "left")

    def test_no_vertical_rank_layout(self) -> None:
        payload = _build_side_scroll_payload(
            race_timeline=interpolate_timeline(self._sample_timeline(), 8),
            race_config={"course": "Tokyo", "surface": "turf", "distance": 1000},
            horses=[{"horse_number": 1, "frame": 1}, {"horse_number": 2, "frame": 2}],
            prediction_table=None,
            width=1920,
            height=1080,
            fps=8,
            duration_sec=1,
        )
        front_rank_y = _side_scroll_lane_y(payload, {"horse_number": 1, "rank": 1}, frame_index=3, wobble=True)
        back_rank_y = _side_scroll_lane_y(payload, {"horse_number": 1, "rank": 9}, frame_index=3, wobble=True)

        self.assertEqual(front_rank_y, back_rank_y)

    def test_start_icons_inside_frame(self) -> None:
        payload = _build_side_scroll_payload(
            race_timeline=interpolate_timeline(self._sample_timeline(), 8),
            race_config={"course": "Tokyo", "surface": "turf", "distance": 1000},
            horses=[{"horse_number": 1, "frame": 1}, {"horse_number": 2, "frame": 2}],
            prediction_table=None,
            width=1920,
            height=1080,
            fps=8,
            duration_sec=1,
        )
        frame = payload["frames"][0]
        camera_x = -float(payload["visible_distance_m"]) * 0.12
        marker_radius = float(payload["marker_radius"]) * 1.25

        for horse in frame["horses"]:
            x = _side_scroll_marker_screen_x(payload, horse, camera_x, progress=0.0)
            y = _side_scroll_lane_y(payload, horse, frame_index=0, wobble=True)
            self.assertGreaterEqual(x, marker_radius + 10.0)
            self.assertLessEqual(x, float(payload["width"]) - marker_radius - 10.0)
            self.assertGreaterEqual(y, float(payload["track_top"]) + marker_radius)
            self.assertLessEqual(y, float(payload["track_bottom"]) - marker_radius)

    def test_final_result_not_drawn_in_race_frames(self) -> None:
        timeline = self._sample_timeline()
        interpolated = _timeline_with_result_display(timeline, total_frames=30, fps=10, result_display_sec=3)
        race_frames = [frame for frame in interpolated if not frame.get("is_post_goal_frame")]

        self.assertTrue(race_frames)
        self.assertFalse(any(_should_show_final_result(frame, 1000.0) for frame in race_frames))

    def test_final_result_drawn_only_in_post_goal_frames(self) -> None:
        timeline = self._sample_timeline()
        interpolated = _timeline_with_result_display(timeline, total_frames=30, fps=10, result_display_sec=3)
        post_goal_frames = [frame for frame in interpolated if frame.get("is_post_goal_frame")]

        self.assertTrue(post_goal_frames)
        self.assertTrue(all(_should_show_final_result(frame, 1000.0) for frame in post_goal_frames))

    def test_final_result_only_post_goal(self) -> None:
        timeline = self._sample_timeline()
        interpolated = _timeline_with_result_display(timeline, total_frames=30, fps=10, result_display_sec=3)

        for frame in interpolated:
            self.assertEqual(_should_show_final_result(frame, 1000.0), bool(frame.get("is_post_goal_frame")))

    def _sample_timeline(self):
        return [
            {
                "time": 0.0,
                "progress": 0.0,
                "horses": [
                    {"horse_number": 1, "horse_name": "A", "frame": 1, "actual_running_style": "逃げ", "position_m": 0.0, "rank": 1, "lane": 0.0},
                    {"horse_number": 2, "horse_name": "B", "frame": 2, "actual_running_style": "追込", "position_m": 0.0, "rank": 2, "lane": 1.0},
                ],
            },
            {
                "time": 1.0,
                "progress": 0.5,
                "horses": [
                    {"horse_number": 1, "horse_name": "A", "frame": 1, "actual_running_style": "逃げ", "position_m": 500.0, "rank": 1, "lane": 0.0},
                    {"horse_number": 2, "horse_name": "B", "frame": 2, "actual_running_style": "追込", "position_m": 497.5, "rank": 2, "lane": 1.0},
                ],
            },
            {
                "time": 2.0,
                "progress": 1.0,
                "horses": [
                    {"horse_number": 2, "horse_name": "B", "frame": 2, "actual_running_style": "追込", "position_m": 1000.0, "rank": 1, "lane": 1.0},
                    {"horse_number": 1, "horse_name": "A", "frame": 1, "actual_running_style": "逃げ", "position_m": 998.0, "rank": 2, "lane": 0.0},
                ],
            },
        ]

    def test_video_renderer_rejects_empty_timeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                render_race_video_from_timeline(
                    race_timeline=[],
                    race_config={"distance": 1000},
                    horses=[],
                    output_path=str(Path(tmpdir) / "race.mp4"),
                )


if __name__ == "__main__":
    unittest.main()
