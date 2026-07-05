from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


COURSES = ["東京", "中山", "京都", "阪神", "中京", "札幌", "函館", "福島", "新潟", "小倉"]
SURFACES = ["芝", "ダート"]
DIRECTIONS = ["右", "左"]
WEATHERS = ["晴", "曇", "雨", "雪"]
TRACK_CONDITIONS = ["良", "稍重", "重", "不良"]
RACE_COURSE_DAYS = [f"{day}日目" for day in range(1, 13)]
COURSE_LAYOUTS = ["A", "B", "C", "D"]
TRACK_BIASES = ["標準", "前残り", "差し有利", "内有利", "外差し有利", "内前有利", "外伸び"]


def render_race_inputs(
    defaults: dict[str, object] | None = None,
    *,
    key_prefix: str = "prediction",
) -> dict[str, str | int]:
    defaults = defaults or {}
    st.sidebar.header("レース条件")
    course = st.sidebar.selectbox(
        "競馬場", COURSES, index=_option_index(COURSES, defaults.get("venue", defaults.get("course")), 3), key=f"{key_prefix}_course"
    )
    surface = st.sidebar.radio(
        "芝/ダート", SURFACES, index=_option_index(SURFACES, defaults.get("surface"), 0), horizontal=True, key=f"{key_prefix}_surface"
    )
    distance = st.sidebar.number_input(
        "距離", min_value=100, max_value=4000, value=_bounded_int(defaults.get("distance"), 2200, 100, 4000), step=100, key=f"{key_prefix}_distance"
    )
    direction = st.sidebar.selectbox(
        "回り", DIRECTIONS, index=_option_index(DIRECTIONS, defaults.get("direction"), 0), key=f"{key_prefix}_direction"
    )
    weather = st.sidebar.selectbox(
        "天候", WEATHERS, index=_option_index(WEATHERS, defaults.get("weather"), 0), key=f"{key_prefix}_weather"
    )
    track_condition = st.sidebar.selectbox(
        "馬場状態", TRACK_CONDITIONS, index=_option_index(TRACK_CONDITIONS, defaults.get("track_condition"), 0), key=f"{key_prefix}_track_condition"
    )
    race_course_day = st.sidebar.selectbox(
        "開催日数", RACE_COURSE_DAYS, index=_option_index(RACE_COURSE_DAYS, defaults.get("race_course_day"), 0), key=f"{key_prefix}_race_course_day"
    )
    course_layout = st.sidebar.selectbox(
        "使用コース", COURSE_LAYOUTS, index=_option_index(COURSE_LAYOUTS, defaults.get("course_layout"), 0), key=f"{key_prefix}_course_layout"
    )
    track_bias = st.sidebar.selectbox(
        "トラックバイアス", TRACK_BIASES, index=_option_index(TRACK_BIASES, defaults.get("track_bias"), 0), key=f"{key_prefix}_track_bias"
    )
    return {
        "course": course,
        "surface": surface,
        "distance": int(distance),
        "direction": direction,
        "weather": weather,
        "track_condition": track_condition,
        "race_course_day": race_course_day,
        "course_layout": course_layout,
        "track_bias": track_bias,
    }


def default_horse_dataframe(count: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "horse_name": [""] * count,
            "frame": [min(8, (index // 2) + 1) for index in range(count)],
            "horse_number": list(range(1, count + 1)),
            "carried_weight": [56.0] * count,
            "jockey": [""] * count,
        }
    )


def make_arrow_safe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a display-only DataFrame that Streamlit/Arrow can serialize."""
    safe_df = df.copy()
    for column in safe_df.columns:
        if safe_df[column].dtype == "object":
            safe_df[column] = safe_df[column].apply(lambda value: "" if value is None else str(value))
    return safe_df


def show_dataframe_safe(df: Any, **kwargs: Any) -> None:
    """Display a DataFrame after removing mixed object dtypes."""
    if df is None:
        return
    frame = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
    if frame.empty:
        st.info("表示するデータがありません。")
        return
    st.dataframe(make_arrow_safe_dataframe(frame), **kwargs)


def render_horse_editor(
    initial_count: int = 5,
    initial_dataframe: pd.DataFrame | list[dict[str, object]] | None = None,
    *,
    key: str = "horse_editor_dynamic",
) -> pd.DataFrame:
    st.subheader("出走馬入力")
    st.caption("表を直接編集できます。行の追加・削除で出走頭数を変更できます。斤量の未入力時は56.0kg、騎手未入力時の補正は中立です。")
    base = _normalize_horse_dataframe(initial_dataframe, initial_count)
    return st.data_editor(
        base,
        num_rows="dynamic",
        width="stretch",
        hide_index=True,
        column_config={
            "horse_name": st.column_config.TextColumn("馬名", required=True),
            "frame": st.column_config.NumberColumn("枠順", min_value=1, max_value=8, step=1, required=True),
            "horse_number": st.column_config.NumberColumn("馬番", min_value=1, max_value=18, step=1, required=True),
            "carried_weight": st.column_config.NumberColumn(
                "斤量 (kg)", min_value=40.0, max_value=65.0, step=0.5, format="%.1f", required=True
            ),
            "jockey": st.column_config.TextColumn("騎手"),
        },
        key=key,
    )


def _normalize_horse_dataframe(
    value: pd.DataFrame | list[dict[str, object]] | None,
    initial_count: int,
) -> pd.DataFrame:
    if value is None:
        return default_horse_dataframe(initial_count)
    frame = value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
    if frame.empty:
        return default_horse_dataframe(initial_count)
    defaults: dict[str, object] = {
        "horse_name": "",
        "frame": 1,
        "horse_number": 1,
        "carried_weight": 56.0,
        "jockey": "",
    }
    for column, default in defaults.items():
        if column not in frame.columns:
            frame[column] = default
    frame["carried_weight"] = pd.to_numeric(frame["carried_weight"], errors="coerce").fillna(56.0)
    return frame[list(defaults)]


def _option_index(options: list[str], value: object, fallback: int) -> int:
    text = str(value or "").strip()
    return options.index(text) if text in options else fallback


def _bounded_int(value: object, fallback: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = fallback
    return max(minimum, min(maximum, parsed))
