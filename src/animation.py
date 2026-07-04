from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("work") / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter, writers

from simulator import SimulationResult
from utils import OutputPaths


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


def create_horse_marker(
    horse_number: int,
    frame: int,
    location: tuple[float, float, float] | tuple[float, float],
    radius: float = 0.35,
) -> dict[str, object]:
    """Return marker-only styling shared by Matplotlib and Plotly renderers."""
    return {
        "horse_number": int(horse_number),
        "frame": int(frame),
        "location": location,
        "radius": radius,
        "color": FRAME_COLORS.get(int(frame), "#9aa0a6"),
        "text_color": get_text_color(int(frame)),
        "edge_color": "#FFFFFF" if int(frame) == 2 else "#111111",
        "display_mode": "marker",
    }


def position_to_track_xyz(position_m: float, total_distance: float, lane: float) -> tuple[float, float, float]:
    """Convert race distance into an oval 3D track coordinate."""
    progress = min(1.0, max(0.0, position_m / max(1.0, total_distance)))
    theta = 2.0 * np.pi * progress
    radius = 0.82 + lane * 0.052
    x = radius * np.cos(theta)
    y = 0.58 * radius * np.sin(theta)
    z = 0.08 + lane * 0.006

    # Stretch the final straight visually so late moves are easier to see.
    if progress > 0.78:
        x += (progress - 0.78) * 0.72
        y *= 0.92
    return float(x), float(y), float(z)


class RaceAnimation:
    """Render race progress as an oval-course animation."""

    def __init__(self, result: SimulationResult) -> None:
        self.result = result
        timeline = result.timeline_dataframe() if hasattr(result, "timeline_dataframe") else pd.DataFrame()
        self.states = timeline if not timeline.empty else result.states_dataframe()
        self.distance = result.race_config.distance

    def save(
        self,
        output_dir: str = "outputs",
        output_name: str = "race",
        fps: int = 12,
        duration_seconds: float = 18.0,
        make_gif: bool = True,
        make_mp4: bool = True,
    ) -> OutputPaths:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        gif_path = str(Path(output_dir) / f"{output_name}.gif")
        mp4_path = str(Path(output_dir) / f"{output_name}.mp4")
        mp4_available = writers.is_available("ffmpeg")
        if make_mp4 and not mp4_available:
            print("MP4 output skipped: ffmpeg is not available.")
        if not make_gif and not (make_mp4 and mp4_available):
            return OutputPaths(gif_path=gif_path, mp4_path=mp4_path)

        animation = self._build_animation(fps=fps, duration_seconds=duration_seconds)

        if make_gif:
            animation.save(gif_path, writer=PillowWriter(fps=fps))
        if make_mp4 and mp4_available:
            animation.save(mp4_path, writer="ffmpeg", fps=fps, dpi=120)
        plt.close(animation._fig)
        return OutputPaths(gif_path=gif_path, mp4_path=mp4_path)

    def _build_animation(self, fps: int, duration_seconds: float) -> FuncAnimation:
        self._configure_fonts()
        horse_names = list(self.states["horse_name"].unique())
        max_time = float(self.states["elapsed_time"].max())
        frame_count = max(24, int(duration_seconds * fps))
        timeline = np.linspace(0.0, max_time, frame_count)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(
            f"{self.result.race_config.course} {self.result.race_config.surface}"
            f"{self.result.race_config.distance}m / pace={self.result.pace.pace}"
        )
        self._draw_course(ax)

        scatters = {
            horse_name: ax.plot(
                [],
                [],
                "o",
                markersize=18,
                markerfacecolor=str(self._marker_style(horse_name)["color"]),
                markeredgecolor=str(self._marker_style(horse_name)["edge_color"]),
                markeredgewidth=2,
            )[0]
            for index, horse_name in enumerate(horse_names)
        }
        labels = {
            horse_name: ax.text(
                0,
                0,
                str(self._horse_number(horse_name)),
                fontsize=9,
                weight="bold",
                ha="center",
                va="center",
                color=str(self._marker_style(horse_name)["text_color"]),
            )
            for horse_name in horse_names
        }
        time_text = ax.text(-1.25, 0.78, "", fontsize=11, weight="bold")

        position_tables = {
            horse_name: self.states[self.states["horse_name"] == horse_name].sort_values("elapsed_time")
            for horse_name in horse_names
        }

        def update(frame_index: int) -> list[object]:
            t = timeline[frame_index]
            artists: list[object] = []
            for lane_index, horse_name in enumerate(horse_names):
                position = self._position_at_time(position_tables[horse_name], t)
                lane = self._lane_at_time(position_tables[horse_name], t, float(lane_index))
                x, y = self._distance_to_xy(position, lane)
                scatters[horse_name].set_data([x], [y])
                labels[horse_name].set_position((x, y))
                artists.extend([scatters[horse_name], labels[horse_name]])
            time_text.set_text(f"{t:05.2f}s")
            artists.append(time_text)
            return artists

        return FuncAnimation(fig, update, frames=frame_count, interval=1000 / fps, blit=False)

    def _configure_fonts(self) -> None:
        candidates = ["Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP"]
        installed = {font.name for font in font_manager.fontManager.ttflist}
        for candidate in candidates:
            if candidate in installed:
                plt.rcParams["font.family"] = candidate
                break
        plt.rcParams["axes.unicode_minus"] = False

    def _draw_course(self, ax: plt.Axes) -> None:
        theta = np.linspace(0, 2 * np.pi, 500)
        ax.plot(np.cos(theta), 0.55 * np.sin(theta), color="#7a5136", linewidth=32, alpha=0.35)
        ax.plot(np.cos(theta), 0.55 * np.sin(theta), color="#bf8b5f", linewidth=24)
        ax.plot(np.cos(theta), 0.55 * np.sin(theta), color="white", linewidth=2)
        ax.text(1.06, 0.0, "GOAL", fontsize=10, weight="bold")
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-0.9, 0.9)

    def _position_at_time(self, table: pd.DataFrame, t: float) -> float:
        times = table["elapsed_time"].to_numpy(dtype=float)
        positions = table["position_m"].to_numpy(dtype=float)
        if len(times) == 0:
            return 0.0
        if t <= times[0]:
            return float(positions[0] * t / max(times[0], 0.1))
        if t >= times[-1]:
            return float(self.distance)
        return float(np.interp(t, times, positions))

    def _lane_at_time(self, table: pd.DataFrame, t: float, default_lane: float) -> float:
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

    def _distance_to_xy(self, distance_m: float, lane: float) -> tuple[float, float]:
        lane_offset = (lane - 3.5) * 0.010
        progress = min(1.0, max(0.0, distance_m / self.distance))
        theta = 2 * np.pi * progress
        radius_x = 1.0 + lane_offset
        radius_y = 0.55 + lane_offset
        x = radius_x * np.cos(theta)
        y = radius_y * np.sin(theta)
        return float(x), float(y)

    def _horse_number(self, horse_name: str) -> int:
        row = self.result.ranking[self.result.ranking["horse_name"] == horse_name].iloc[0]
        return int(row["horse_number"])

    def _horse_frame(self, horse_name: str) -> int:
        row = self.result.ranking[self.result.ranking["horse_name"] == horse_name].iloc[0]
        return int(row["frame"])

    def _marker_style(self, horse_name: str) -> dict[str, object]:
        return create_horse_marker(
            horse_number=self._horse_number(horse_name),
            frame=self._horse_frame(horse_name),
            location=(0.0, 0.0),
            radius=0.35,
        )


class Race3DAnimation:
    """Render a Plotly 3D-style race animation for Streamlit embedding."""

    def __init__(self, result: SimulationResult) -> None:
        self.result = result
        timeline = result.timeline_dataframe() if hasattr(result, "timeline_dataframe") else pd.DataFrame()
        self.states = timeline if not timeline.empty else result.states_dataframe()
        self.distance = result.race_config.distance
        self.ranking = result.ranking

    def save_html(
        self,
        output_dir: str = "outputs",
        output_name: str = "race_3d",
        duration_seconds: float = 18.0,
        fps: int = 8,
    ) -> str:
        import plotly.graph_objects as go

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        html_path = str(Path(output_dir) / f"{output_name}.html")
        fig = self._build_figure(go=go, duration_seconds=duration_seconds, fps=fps)
        fig.write_html(html_path, include_plotlyjs=True, full_html=True)
        return html_path

    def _build_figure(self, go: object, duration_seconds: float, fps: int):
        horse_names = list(self.states["horse_name"].unique())
        max_time = float(self.states["elapsed_time"].max())
        frame_count = max(32, int(duration_seconds * fps))
        timeline = np.linspace(0.0, max_time, frame_count)
        base_data = self._horse_frame(go, horse_names, 0.0)
        surface, inner_rail, outer_rail = self._course_traces(go)

        fig = go.Figure(data=[surface, inner_rail, outer_rail, base_data["shadow"], base_data["horses"]])
        frames = []
        for index, t in enumerate(timeline):
            frame_data = self._horse_frame(go, horse_names, float(t))
            frames.append(go.Frame(data=[frame_data["shadow"], frame_data["horses"]], traces=[3, 4], name=str(index)))
        fig.frames = frames

        frame_ms = int(1000 / max(1, fps))
        fig.update_layout(
            scene={
                "xaxis": {"visible": False},
                "yaxis": {"visible": False},
                "zaxis": {"visible": False, "range": [0, 0.55]},
                "aspectratio": {"x": 1.8, "y": 1.15, "z": 0.35},
                "camera": {"eye": {"x": 1.8, "y": -2.2, "z": 1.2}},
            },
            margin={"l": 0, "r": 0, "t": 40, "b": 0},
            title=f"{self.result.race_config.course} {self.result.race_config.surface}{self.distance}m 3D Race",
            showlegend=False,
            updatemenus=[
                {
                    "type": "buttons",
                    "showactive": False,
                    "x": 0.02,
                    "y": 0.98,
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [None, {"frame": {"duration": frame_ms, "redraw": True}, "fromcurrent": True}],
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                        },
                    ],
                }
            ],
            sliders=[
                {
                    "steps": [
                        {
                            "method": "animate",
                            "label": str(index),
                            "args": [[str(index)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
                        }
                        for index in range(frame_count)
                    ],
                    "x": 0.08,
                    "len": 0.84,
                    "y": 0.02,
                }
            ],
        )
        return fig

    def _course_traces(self, go: object):
        theta = np.linspace(0, 2 * np.pi, 160)
        radii = np.linspace(0.74, 1.18, 16)
        theta_grid, radius_grid = np.meshgrid(theta, radii)
        x = radius_grid * np.cos(theta_grid)
        y = 0.58 * radius_grid * np.sin(theta_grid)
        z = 0.02 * np.sin(theta_grid) ** 2
        is_dirt = self.result.race_config.surface == "ダート"
        color = "#b88758" if is_dirt else "#4f9a58"
        rail_color = "#f7f7f2"
        surface = go.Surface(
            x=x,
            y=y,
            z=z,
            colorscale=[[0, color], [1, color]],
            showscale=False,
            opacity=0.9,
            hoverinfo="skip",
        )
        inner_rail = go.Scatter3d(
            x=0.70 * np.cos(theta),
            y=0.58 * 0.70 * np.sin(theta),
            z=np.full_like(theta, 0.075),
            mode="lines",
            line={"color": rail_color, "width": 7},
            hoverinfo="skip",
        )
        outer_rail = go.Scatter3d(
            x=1.22 * np.cos(theta),
            y=0.58 * 1.22 * np.sin(theta),
            z=np.full_like(theta, 0.075),
            mode="lines",
            line={"color": rail_color, "width": 7},
            hoverinfo="skip",
        )
        return surface, inner_rail, outer_rail

    def _horse_frame(self, go: object, horse_names: list[str], t: float) -> dict[str, object]:
        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        shadow_zs: list[float] = []
        texts: list[str] = []
        colors: list[str] = []
        text_colors: list[str] = []
        hover_texts: list[str] = []
        sizes: list[int] = []

        for name in horse_names:
            meta = self._horse_meta(name)
            table = self.states[self.states["horse_name"] == name].sort_values("elapsed_time")
            position = self._position_at_time(table, t)
            lane = self._lane_at_time(table, t, self._lane_for(meta))
            x, y, z = position_to_track_xyz(position, self.distance, lane)
            xs.append(x)
            ys.append(y)
            zs.append(z)
            shadow_zs.append(0.035)
            texts.append(str(meta["horse_number"]))
            frame = int(meta["frame"])
            marker = create_horse_marker(int(meta["horse_number"]), frame, (x, y, z), radius=0.35)
            colors.append(str(marker["color"]))
            text_colors.append(str(marker["text_color"]))
            sizes.append(16)
            hover_texts.append(
                f"馬番: {meta['horse_number']}<br>枠: {meta['frame']}"
                f"<br>脚質: {meta['running_style']}<br>距離: {position:.0f}m"
            )

        shadow = go.Scatter3d(
            x=xs,
            y=ys,
            z=shadow_zs,
            mode="markers",
            marker={"size": [size + 5 for size in sizes], "color": "rgba(0,0,0,0.22)", "symbol": "circle"},
            hoverinfo="skip",
        )
        horses = go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="markers+text",
            marker={
                "size": sizes,
                "color": colors,
                "line": {"color": "#161616", "width": 2},
                "symbol": "circle",
            },
            text=texts,
            textposition="middle center",
            textfont={"color": text_colors, "size": 12},
            hovertext=hover_texts,
            hoverinfo="text",
        )
        return {"shadow": shadow, "horses": horses}

    def _horse_meta(self, horse_name: str) -> dict[str, object]:
        row = self.ranking[self.ranking["horse_name"] == horse_name].iloc[0]
        return {
            "horse_number": int(row["horse_number"]),
            "frame": int(row["frame"]),
            "running_style": str(row["running_style"]),
        }

    def _lane_for(self, meta: dict[str, object]) -> float:
        frame = int(meta["frame"])
        horse_number = int(meta["horse_number"])
        return min(7.0, max(0.0, (frame - 1) + (horse_number % 2) * 0.28))

    def _position_at_time(self, table: pd.DataFrame, t: float) -> float:
        times = table["elapsed_time"].to_numpy(dtype=float)
        positions = table["position_m"].to_numpy(dtype=float)
        if len(times) == 0:
            return 0.0
        if t <= times[0]:
            return float(positions[0] * t / max(times[0], 0.1))
        if t >= times[-1]:
            return float(self.distance)
        return float(np.interp(t, times, positions))

    def _lane_at_time(self, table: pd.DataFrame, t: float, default_lane: float) -> float:
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
