from __future__ import annotations

import importlib
import math
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
OUTPUT_DIR = Path("outputs")
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Streamlit keeps imported modules alive between reruns. Reload the project
# modules whose dataclasses/function signatures change often during development.
for module_name in [
    "race_config",
    "course_db",
    "horse_analyzer",
    "data_fetcher",
    "simulator",
    "animation",
    "app_state",
    "ui_components",
    "main",
    "weight_optimizer",
    "ml_model",
    "monte_carlo",
    "result_formatter",
    "video_renderer",
    "netkeiba_fetcher",
    "race_data_fetcher",
    "race_trend_fetcher",
    "race_trend_analyzer",
    "race_trend_database",
    "race_trend_scorer",
    "result_fetcher",
    "analysis_reporter",
    "report_generator",
    "thumbnail_generator",
    "youtube_video_builder",
    "narration_generator",
    "metrics",
    "log_manager",
    "evaluation",
]:
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])

from app_state import dataframe_to_horses, validate_inputs
from analysis_reporter import analyze_prediction_failure, generate_race_review
from betting_simulator import (
    aggregate_return_rates,
    bets_to_dataframe,
    calculate_return_rate,
    generate_bets_from_prediction,
    payouts_to_dataframe,
)
from course_db import get_course_bias
from errors import RaceDataFetchError
from evaluation import (
    EVALUATION_LOG_DIR,
    PREDICTION_LOG_DIR,
    evaluation_logs_table,
    evaluate_prediction,
    find_prediction_log,
    load_evaluation_logs,
    load_prediction_logs,
    save_evaluation_log,
    save_prediction_log,
    summarize_evaluations,
)
from log_manager import (
    DuplicateLogError,
    delete_log_files,
    find_duplicate_logs,
    find_matching_logs,
    log_inventory,
)
import main as simulation_main
from metrics import (
    aggregate_evaluation_logs,
    aggregate_failure_tags,
    aggregate_improvement_suggestions,
)
from ml_model import (
    ML_FEATURES,
    TOP3_MODEL_PATH,
    WIN_MODEL_PATH,
    build_ml_dataset,
    load_model,
    model_feature_importance,
    resolve_prediction_engine,
    save_model,
    train_prediction_model,
)
from monte_carlo import run_monte_carlo_prediction
from race_data_fetcher import load_prediction_race_data
from race_trend_analyzer import analyze_same_race_trends
from race_trend_database import analyze_same_race_trend_database, build_same_race_trend_database
from race_trend_fetcher import fetch_same_race_history, save_same_race_history
from report_generator import generate_prediction_report, save_prediction_report
from result_fetcher import load_completed_race_result
from result_formatter import (
    build_single_race_result_from_timeline,
    build_horse_comments_table,
    build_recent_races_table,
    format_analysis_table,
    pace_comment,
    style_group_table,
    style_probability_long_table,
)
from ui_components import render_horse_editor, render_race_inputs
from thumbnail_generator import generate_youtube_thumbnail
from video_renderer import render_race_video_from_timeline, render_side_scroll_race_video
from weight_optimizer import (
    build_training_dataset,
    load_model_weights,
    optimize_prediction_weights,
    save_model_weights,
)
from youtube_video_builder import (
    REQUIRED_SECTION_ORDER,
    build_youtube_prediction_video,
    build_youtube_video_structure,
    generate_race_trend_summary,
)


simulation_main = importlib.reload(simulation_main)
run_race_simulation = simulation_main.run_race_simulation

st.set_page_config(page_title="競馬レースシミュレーター", layout="wide")


@st.cache_data(ttl=3600, show_spinner=False)
def cached_prediction_race_data(race_name: str, race_date: str) -> dict[str, Any] | None:
    # Pre-race loading deliberately has no dependency on fetch_race_result.
    return load_prediction_race_data(race_name, race_date)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_completed_race_result(race_name: str, race_date: str) -> dict[str, Any] | None:
    return load_completed_race_result(race_name, race_date)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    st.title("競馬レースシミュレーター")
    tab_prediction, tab_result, tab_analysis, tab_youtube = st.tabs(["予想", "実結果入力", "成績分析", "YouTube出力"])
    with tab_prediction:
        render_prediction_tab()
    with tab_result:
        render_actual_result_tab()
    with tab_analysis:
        render_performance_tab()
    with tab_youtube:
        render_youtube_output_tab()


def render_prediction_tab() -> None:
    st.write("レース条件と出走馬を入力すると、近走分析、展開予測、シミュレーション、動画生成まで実行します。")

    lookup_col, date_col, fetch_col = st.columns([2, 1, 1])
    with lookup_col:
        race_name_input = st.text_input("レース名", key="prediction_race_name")
    with date_col:
        race_date_input = st.date_input("開催日", value=date.today(), key="prediction_race_date")
    with fetch_col:
        st.write("")
        fetch_entries = st.button("出馬表を取得", key="fetch_prediction_entries", width="stretch")

    if fetch_entries:
        try:
            fetched = cached_prediction_race_data(race_name_input.strip(), race_date_input.isoformat())
            if fetched is None:
                st.error("レースを特定できませんでした。レース名と開催日を確認してください。")
            elif not fetched.get("fetched_entries"):
                st.error("出馬表を取得できませんでした。手入力またはCSVアップロードを利用してください。")
                st.session_state["prediction_race_data"] = fetched
            else:
                st.session_state["prediction_race_data"] = fetched
                st.session_state["prediction_editor_version"] = int(st.session_state.get("prediction_editor_version", 0)) + 1
                st.success("netkeibaの出馬表を取得しました。内容を確認してから予想を実行してください。")
                st.rerun()
        except Exception as exc:
            st.error(f"出馬表の取得に失敗しました: {exc}")

    uploaded_entries = st.file_uploader(
        "出馬表CSV（取得できない場合）",
        type=["csv"],
        key="prediction_entries_csv",
        help="horse_name, frame, horse_number, carried_weight, jockey などの列に対応します。",
    )
    if uploaded_entries is not None and st.button("CSVを出走馬表へ反映", key="apply_prediction_csv"):
        try:
            uploaded_frame = pd.read_csv(uploaded_entries)
            st.session_state["prediction_csv_entries"] = uploaded_frame.to_dict("records")
            st.session_state["prediction_editor_version"] = int(st.session_state.get("prediction_editor_version", 0)) + 1
            st.rerun()
        except Exception as exc:
            st.error(f"CSVを読み込めませんでした: {exc}")

    prediction_race_data = st.session_state.get("prediction_race_data", {})
    fetched_entries_data = prediction_race_data.get("fetched_entries", []) if isinstance(prediction_race_data, dict) else []
    initial_entries = st.session_state.get("prediction_csv_entries") or fetched_entries_data
    metadata_defaults = prediction_race_data.get("race_metadata", {}) if isinstance(prediction_race_data, dict) else {}
    editor_version = int(st.session_state.get("prediction_editor_version", 0))

    if isinstance(prediction_race_data, dict) and prediction_race_data:
        st.caption(
            f"race_id: {prediction_race_data.get('race_id', '-')} / "
            f"取得元: {prediction_race_data.get('source_url', '-')}"
        )
        if fetched_entries_data:
            st.dataframe(pd.DataFrame(fetched_entries_data), width="stretch", hide_index=True)

    race_config = render_race_inputs(metadata_defaults, key_prefix=f"prediction_{editor_version}")
    if race_name_input.strip():
        race_config["race_name"] = race_name_input.strip()
        race_config["race_date"] = race_date_input.isoformat()

    st.sidebar.header("出走設定")
    cloud_environment = bool(
        os.getenv("STREAMLIT_SHARING_MODE")
        or os.getenv("STREAMLIT_CLOUD")
        or os.getenv("RENDER")
    )
    lightweight_mode = st.sidebar.checkbox(
        "軽量モード",
        value=cloud_environment,
        help="公開環境向けにMP4生成を避け、Plotly表示と短い計算設定を使います。",
    )
    animation_method = st.sidebar.radio("アニメーション形式", ["3D風", "簡易2D", "投稿用MP4"], index=0)
    timeline_mode = st.sidebar.selectbox("タイムライン生成方式", ["controlled", "legacy"], index=0)
    video_layout = st.sidebar.selectbox("動画レイアウト", ["side_scroll", "legacy_overview"], index=0)
    video_format_label = st.sidebar.selectbox(
        "動画フォーマット",
        ["YouTube横長 16:9", "TikTok縦長 9:16"],
    )
    animation_seconds = st.sidebar.selectbox(
        "動画の長さ",
        [30, 60, 90],
        index=1,
    )
    debug_mode = st.sidebar.checkbox("デバッグ情報を表示", value=False)
    prediction_mode = st.sidebar.checkbox("予想モード", value=False)
    prediction_engine_requested = st.sidebar.selectbox(
        "予想エンジン",
        ["rule_based", "optimized_weights", "ml_model"],
        index=0,
        disabled=not prediction_mode,
    )
    active_prediction_engine = resolve_prediction_engine(prediction_engine_requested, TOP3_MODEL_PATH)
    if prediction_mode and prediction_engine_requested == "ml_model" and active_prediction_engine != "ml_model":
        st.warning("学習済みMLモデルがないため、rule_basedへフォールバックします。")
    current_weights = load_model_weights()
    with st.sidebar.expander("使用中の予想重み"):
        st.json(current_weights)
    n_simulations = st.sidebar.selectbox("シミュレーション回数", [100, 300, 500, 1000], index=2, disabled=not prediction_mode)
    prediction_seed = st.sidebar.number_input("乱数seed", min_value=0, max_value=2_147_483_647, value=42, step=1, disabled=not prediction_mode)
    prediction_sort = st.sidebar.selectbox("結果の並び替え", ["prediction_score", "win_rate", "top3_rate", "avg_finish"], index=0, disabled=not prediction_mode)

    with st.sidebar.expander("既存の直近5走取得モジュール"):
        provider_module = st.text_input("provider module", value="", placeholder="例: my_fetcher")
        provider_factory = st.text_input("factory name", value="", placeholder="例: get_provider")
        st.caption("未入力の場合はnetkeibaから実データを取得します。取得できない馬名では停止します。")

    horse_df = render_horse_editor(
        initial_count=max(2, len(initial_entries) if initial_entries else 5),
        initial_dataframe=initial_entries or None,
        key=f"horse_editor_dynamic_{editor_version}",
    )
    horses = dataframe_to_horses(horse_df)
    st.sidebar.metric("現在の出走頭数", len(horses))
    if debug_mode and isinstance(prediction_race_data, dict) and prediction_race_data:
        render_netkeiba_debug(prediction_race_data.get("debug", {}), "予想用データ取得デバッグ")
    render_duration_sec = int(animation_seconds)
    effective_duration_sec = min(render_duration_sec, 30) if lightweight_mode else render_duration_sec
    effective_animation_method = "3D風" if lightweight_mode else animation_method

    prediction_duplicate_action = st.selectbox(
        "同じレースの予想ログがある場合",
        ["保存しない", "上書き保存", "別名で保存"],
        index=0,
        key="prediction_duplicate_action",
    )
    submitted = st.button("シミュレーション実行", type="primary", width="stretch")
    if submitted:
        validation = validate_inputs(race_config, horses)
        if not validation.is_valid:
            for error in validation.errors:
                st.error(error)
            st.stop()

        st.session_state.pop("simulation_result", None)
        try:
            with st.spinner("直近5走取得、分析、シミュレーション、動画生成を実行中です..."):
                result = run_race_simulation(
                    race_config=race_config,
                    horses=horses,
                    output_dir=str(OUTPUT_DIR),
                    provider_module=provider_module.strip() or None,
                    provider_factory=provider_factory.strip() or None,
                    make_gif=effective_animation_method == "簡易2D",
                    make_mp4=False,
                    animation_mode=animation_mode_key(effective_animation_method),
                    animation_seconds=float(effective_duration_sec),
                    video_format=video_format_label,
                    timeline_mode=timeline_mode,
                    seed=42,
                )
            ensure_recent_races_exist(result)
            if prediction_mode:
                trend_analysis = None
                trend_name = str(race_config.get("race_name") or race_name_input.strip())
                trend_venue = str(race_config.get("course") or "")
                try:
                    trend_distance = int(race_config.get("distance") or 0)
                except (TypeError, ValueError):
                    trend_distance = 0
                if trend_name and trend_venue and trend_distance > 0:
                    with st.spinner("同レース過去10年傾向を取得・集計中です..."):
                        try:
                            trend_database = build_same_race_trend_database(
                                race_name=trend_name,
                                venue=trend_venue,
                                distance=trend_distance,
                                years=10,
                                race_date=str(race_config.get("race_date") or race_date_input.isoformat()),
                            )
                            trend_analysis = analyze_same_race_trend_database(trend_database)
                            result["same_race_trend_database"] = trend_database
                            result["same_race_trend_analysis"] = trend_analysis
                            result.setdefault("log", []).append(
                                f"same race trend rows={trend_database.get('row_count', 0)}"
                            )
                        except Exception as exc:
                            result.setdefault("log", []).append(f"same race trend fetch skipped: {exc}")
                            if debug_mode:
                                st.warning(f"同レース過去10年傾向は取得できませんでした。中立スコアで続行します: {exc}")
                race_config_for_prediction = dict(race_config)
                if isinstance(trend_analysis, dict):
                    race_config_for_prediction["same_race_trend_analysis"] = trend_analysis
                with st.spinner("Monte Carlo予想を実行中です..."):
                    result["prediction"] = run_monte_carlo_prediction(
                        race_config=race_config_for_prediction,
                        horses=horses,
                        n_simulations=min(int(n_simulations), 100) if lightweight_mode else int(n_simulations),
                        seed=int(prediction_seed),
                        abilities=result.get("abilities"),
                        pace=result.get("pace"),
                        output_dir=str(OUTPUT_DIR),
                        prediction_engine=active_prediction_engine,
                        prediction_weights=current_weights,
                        trend_analysis=trend_analysis,
                    )
                _merge_prediction_trend_columns(result)
                result["prediction_engine_requested"] = prediction_engine_requested
                result["prediction_engine"] = result["prediction"].get("prediction_engine", active_prediction_engine)
                representative_trial = result["prediction"].get("representative_trial", {})
                if representative_trial:
                    result["representative_trial"] = representative_trial
                    result["race_timeline"] = representative_trial.get("race_timeline", result.get("race_timeline", []))
                    result["single_result"] = representative_trial.get("result_df")
                    result["single_result_source"] = "Monte Carlo trial with highest AI expected value"
            if not isinstance(result.get("single_result"), pd.DataFrame):
                result["single_result"] = build_single_race_result_from_timeline(
                    result.get("race_timeline", []),
                    result.get("horse_inputs", horses),
                )
                result["single_result_source"] = "controlled_timeline final_frame"
            result["prediction_result"] = result.get("prediction")
            result["controlled_timeline"] = result.get("race_timeline", [])
            result["lightweight_mode"] = lightweight_mode
            if lightweight_mode:
                result.setdefault("log", []).append("軽量モード: MP4生成を省略し、Plotly HTMLを使用しました。")
            else:
                with st.spinner("レース動画を生成中です..."):
                    render_video_for_result(
                        result=result,
                        race_config=race_config,
                        horses=result.get("horse_inputs", horses),
                        video_format_label=video_format_label,
                        video_layout=video_layout,
                        duration_sec=render_duration_sec,
                    )
            log_metadata = dict(race_config)
            if isinstance(metadata_defaults, dict):
                log_metadata.update({key: value for key, value in metadata_defaults.items() if value not in (None, "", 0)})
            if not log_metadata.get("race_name"):
                log_metadata["race_name"] = race_name_input.strip()
            if not log_metadata.get("race_date"):
                log_metadata["race_date"] = race_date_input.isoformat()
            result["race_metadata"] = log_metadata
            result["race_name"] = str(log_metadata.get("race_name", ""))
            result["race_date"] = str(log_metadata.get("race_date", ""))
            prediction_frame_for_comments = _prediction_table(result)
            analysis_frame_for_comments = result.get("horse_analysis")
            if (
                isinstance(prediction_frame_for_comments, pd.DataFrame)
                and not prediction_frame_for_comments.empty
                and isinstance(analysis_frame_for_comments, pd.DataFrame)
            ):
                result["comments_table"] = build_horse_comments_table(
                    prediction_frame_for_comments,
                    analysis_frame_for_comments,
                    result.get("pace_prediction", {}),
                    result.get("race_config", {}),
                )
            race_id_for_log = str(prediction_race_data.get("race_id", "")) if isinstance(prediction_race_data, dict) else ""
            existing_prediction_logs = find_matching_logs(
                race_id=race_id_for_log,
                race_name=str(log_metadata.get("race_name", "")),
                race_date=str(log_metadata.get("race_date", "")),
                log_dir=PREDICTION_LOG_DIR,
            )
            if existing_prediction_logs:
                st.warning("同じレースのログが既に存在します。")
            if existing_prediction_logs and prediction_duplicate_action == "保存しない":
                result["prediction_log_path"] = ""
                st.info("予想ログは保存しませんでした。")
            else:
                prediction_log_path = save_prediction_log(
                    race_id=race_id_for_log,
                    source_url=str(prediction_race_data.get("source_url", "")) if isinstance(prediction_race_data, dict) else "",
                    fetched_entries=list(initial_entries or []),
                    race_metadata=log_metadata,
                    prediction_table=_prediction_table(result),
                    simulation_result=result,
                    duplicate_action=prediction_duplicate_action,
                )
                result["prediction_log_path"] = str(prediction_log_path)
                st.success("予想ログを保存しました")
                st.write(str(prediction_log_path))
            st.session_state["simulation_result"] = result
        except DuplicateLogError as exc:
            st.warning(str(exc))
        except RaceDataFetchError as exc:
            st.error(str(exc))
            if debug_mode:
                render_fetch_debug({"fetch_debug": exc.debug_records})
            st.stop()
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
        except Exception as exc:
            st.error("シミュレーション中にエラーが発生しました。")
            st.exception(exc)
            st.stop()

    result = st.session_state.get("simulation_result")
    if result:
        render_results(result, debug_mode=debug_mode, prediction_sort=prediction_sort)
        render_prediction_report_section(result)
        if result.get("prediction_log_path"):
            st.caption(f"予想ログ: {result['prediction_log_path']}")
    else:
        st.info("入力後に「シミュレーション実行」を押してください。")


def render_actual_result_tab() -> None:
    st.write("レース後に実結果を取得し、保存済みの予想ログと比較します。")
    debug_mode = st.checkbox("実結果取得のデバッグ情報を表示", value=False, key="result_debug_mode")
    name_col, date_col, button_col = st.columns([2, 1, 1])
    with name_col:
        race_name = st.text_input("レース名", key="result_race_name")
    with date_col:
        race_date_value = st.date_input("開催日", value=date.today(), key="result_race_date")
    with button_col:
        st.write("")
        fetch_result_clicked = st.button(
            "netkeibaから実結果を取得",
            key="fetch_actual_result",
            type="primary",
            width="stretch",
        )

    if fetch_result_clicked:
        try:
            # This is the only UI action that calls the completed-result loader.
            fetched = cached_completed_race_result(race_name.strip(), race_date_value.isoformat())
            if fetched is None:
                st.error("レースを特定できませんでした。レース名と開催日を確認してください。")
            else:
                st.session_state["actual_result_fetch"] = fetched
                if not fetched.get("actual_result"):
                    cached_completed_race_result.clear()
                    st.warning("このレースはまだ結果が公開されていない可能性があります。レース後に再度取得してください。")
                else:
                    st.session_state["actual_result_rows"] = fetched["actual_result"]
                    st.session_state["actual_result_editor_version"] = int(
                        st.session_state.get("actual_result_editor_version", 0)
                    ) + 1
                    st.success("実結果を取得しました。内容を確認して保存してください。")
                    st.rerun()
        except Exception as exc:
            st.error(f"実結果の取得に失敗しました: {exc}")

    manual_finish_order = st.text_input(
        "全着順を馬番のカンマ区切りで入力",
        placeholder="7,3,11,5,1,8,12",
        key="manual_finish_order",
    )
    if st.button("手入力着順を実結果表へ反映", key="apply_manual_finish_order"):
        try:
            horse_numbers = [int(value.strip()) for value in manual_finish_order.split(",") if value.strip()]
            if not horse_numbers or any(number <= 0 for number in horse_numbers) or len(set(horse_numbers)) != len(horse_numbers):
                raise ValueError("馬番は重複のない正の整数で入力してください。")
            st.session_state["actual_result_rows"] = [
                {"finish": finish, "horse_number": horse_number, "horse_name": ""}
                for finish, horse_number in enumerate(horse_numbers, start=1)
            ]
            st.session_state["actual_result_editor_version"] = int(
                st.session_state.get("actual_result_editor_version", 0)
            ) + 1
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    uploaded_result = st.file_uploader(
        "実結果CSV（取得できない場合）",
        type=["csv"],
        key="actual_result_csv",
        help="horse_name, frame, horse_number, finish, time, margin, passing_order, last3f などの列に対応します。",
    )
    if uploaded_result is not None and st.button("CSVを実結果表へ反映", key="apply_actual_result_csv"):
        try:
            uploaded_frame = pd.read_csv(uploaded_result)
            st.session_state["actual_result_rows"] = uploaded_frame.to_dict("records")
            st.session_state["actual_result_editor_version"] = int(
                st.session_state.get("actual_result_editor_version", 0)
            ) + 1
            st.rerun()
        except Exception as exc:
            st.error(f"CSVを読み込めませんでした: {exc}")

    fetched_result = st.session_state.get("actual_result_fetch", {})
    if isinstance(fetched_result, dict) and fetched_result:
        st.caption(f"race_id: {fetched_result.get('race_id', '-')} / 取得元: {fetched_result.get('source_url', '-')}")
        metadata = fetched_result.get("race_metadata", {})
        if metadata:
            st.dataframe(pd.DataFrame([metadata]), width="stretch", hide_index=True)
        if debug_mode:
            render_netkeiba_debug(fetched_result.get("debug", {}), "実結果取得デバッグ")
        payouts = fetched_result.get("payouts", {})
        st.subheader("払い戻し情報")
        if payouts:
            st.dataframe(payouts_to_dataframe(payouts), width="stretch", hide_index=True)
        else:
            st.warning("払い戻し情報を取得できませんでした。結果未確定、または払い戻し表が未公開の可能性があります。")

    rows = st.session_state.get("actual_result_rows") or _empty_actual_result_rows()
    result_version = int(st.session_state.get("actual_result_editor_version", 0))
    st.subheader("実結果確認・手入力")
    result_frame = st.data_editor(
        pd.DataFrame(rows),
        num_rows="dynamic",
        width="stretch",
        hide_index=True,
        key=f"actual_result_editor_{result_version}",
    )

    evaluation_duplicate_action = st.selectbox(
        "同じレースの評価ログがある場合",
        ["保存しない", "上書き保存", "別名で保存"],
        index=0,
        key="evaluation_duplicate_action",
    )
    bet_col1, bet_col2 = st.columns(2)
    with bet_col1:
        stake_per_bet = st.number_input(
            "1点あたり購入金額",
            min_value=100,
            max_value=10000,
            value=100,
            step=100,
            key="evaluation_stake_per_bet",
        )
    with bet_col2:
        box_size = st.selectbox("BOX買い頭数", [3, 4, 5], index=2, key="evaluation_box_size")
    if st.button("実結果を保存・評価", key="save_and_evaluate_result", type="primary"):
        actual_results = _valid_actual_result_rows(result_frame)
        if not actual_results:
            st.warning("着順と馬番を含む実結果を入力してください。")
            return
        race_id = str(fetched_result.get("race_id", "")) if isinstance(fetched_result, dict) else ""
        prediction_log = find_prediction_log(
            race_id=race_id or None,
            race_name=race_name.strip(),
            race_date=race_date_value.isoformat(),
        )
        if prediction_log is None:
            st.warning("対応する保存済み予想ログが見つかりません。race_id、レース名、開催日を確認してください。")
            return
        evaluation = evaluate_prediction(prediction_log, actual_results)
        prediction_table_for_bets = pd.DataFrame(prediction_log.get("prediction_table", []))
        bets = generate_bets_from_prediction(
            prediction_table_for_bets,
            box_size=int(box_size),
            stake_per_bet=int(stake_per_bet),
        )
        payouts = fetched_result.get("payouts", {}) if isinstance(fetched_result, dict) else {}
        return_analysis = calculate_return_rate(
            bets=bets,
            payouts=payouts,
            actual_result=actual_results,
            stake_per_bet=int(stake_per_bet),
        )
        evaluation["return_analysis"] = return_analysis
        existing_evaluations = find_matching_logs(
            race_id=str(prediction_log.get("race_id", "")),
            race_name=str(prediction_log.get("race_name", "")),
            race_date=str(prediction_log.get("race_date", "")),
            prediction_log_path=str(prediction_log.get("_path", "")),
            log_dir=EVALUATION_LOG_DIR,
        )
        if existing_evaluations:
            st.warning("同じレースのログが既に存在します。")
        if existing_evaluations and evaluation_duplicate_action == "保存しない":
            st.info("評価ログは保存しませんでした。")
            st.session_state["latest_evaluation"] = evaluation
            return
        try:
            evaluation_path = save_evaluation_log(
                prediction_log=prediction_log,
                actual_results=actual_results,
                evaluation=evaluation,
                payouts=payouts,
                bets=bets,
                return_analysis=return_analysis,
                race_metadata=fetched_result.get("race_metadata", {}) if isinstance(fetched_result, dict) else {},
                source_url=str(fetched_result.get("source_url", "")) if isinstance(fetched_result, dict) else "",
                duplicate_action=evaluation_duplicate_action,
            )
        except DuplicateLogError as exc:
            st.warning(str(exc))
            return
        st.session_state["latest_evaluation"] = evaluation
        st.session_state["latest_payouts"] = payouts
        st.session_state["latest_bets"] = bets
        st.session_state["latest_return_analysis"] = return_analysis
        st.success(f"評価ログを保存しました: {evaluation_path}")

    latest_evaluation = st.session_state.get("latest_evaluation")
    if isinstance(latest_evaluation, dict):
        st.subheader("予想と実結果の比較")
        mark_finishes = latest_evaluation.get("mark_finishes", {})
        st.dataframe(
            pd.DataFrame([{"印": mark, "着順": mark_finishes.get(mark)} for mark in ("◎", "○", "▲", "△", "☆")]),
            width="stretch",
            hide_index=True,
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("印5頭の3着以内率", f"{float(latest_evaluation.get('marked_top3_rate', 0.0)):.1%}")
        c2.metric("予想上位3頭の3着以内率", f"{float(latest_evaluation.get('predicted_top3_top3_rate', 0.0)):.1%}")
        c3.metric("上位5頭内的中数", int(latest_evaluation.get("top5_hit_count", 0)))
        c4.metric("3連複候補", "的中" if latest_evaluation.get("trifecta_candidate_hit") else "不的中")
        st.info(str(latest_evaluation.get("comment", "")))
        latest_payouts = st.session_state.get("latest_payouts", {})
        latest_bets = st.session_state.get("latest_bets", {})
        latest_return = st.session_state.get("latest_return_analysis") or latest_evaluation.get("return_analysis", {})
        if isinstance(latest_payouts, dict):
            st.subheader("払い戻し情報")
            payout_frame = payouts_to_dataframe(latest_payouts)
            if payout_frame.empty:
                st.warning("払い戻し情報は未取得です。")
            else:
                st.dataframe(payout_frame, width="stretch", hide_index=True)
        if isinstance(latest_bets, dict):
            st.subheader("AI予想による買い目")
            st.dataframe(bets_to_dataframe(latest_bets), width="stretch", hide_index=True)
        if isinstance(latest_return, dict) and latest_return:
            st.subheader("回収率")
            summary = latest_return.get("summary", {})
            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("総購入額", f"{int(summary.get('total_stake', 0)):,}円")
            rc2.metric("総払戻額", f"{int(summary.get('total_return', 0)):,}円")
            rc3.metric("回収率", f"{float(summary.get('return_rate', 0.0)):.2f}%")
            st.dataframe(pd.DataFrame(latest_return.get("by_bet_type", [])), width="stretch", hide_index=True)


def render_performance_tab() -> None:
    logs = load_evaluation_logs()
    summary = aggregate_evaluation_logs(logs)
    st.subheader("予想成績")
    if not logs:
        st.info("評価ログはまだありません。レース後に実結果を保存すると、ここに集計されます。")
    else:
        columns = st.columns(5)
        columns[0].metric("評価レース数", int(summary["race_count"]))
        columns[1].metric("◎勝率", f"{summary['honmei_win_rate']:.1%}")
        columns[2].metric("◎複勝率", f"{summary['honmei_top3_rate']:.1%}")
        columns[3].metric("○複勝率", f"{summary['second_mark_top3_rate']:.1%}")
        columns[4].metric("▲複勝率", f"{summary['third_mark_top3_rate']:.1%}")
        hit_columns = st.columns(4)
        hit_columns[0].metric("印5頭の平均複勝圏内数", f"{summary['average_marked_top3_count']:.2f}")
        hit_columns[1].metric("予想上位3頭の平均複勝圏内数", f"{summary['average_predicted_top3_count']:.2f}")
        hit_columns[2].metric("1着的中率", f"{summary['winner_hit_rate']:.1%}")
        hit_columns[3].metric("平均的中数", f"{summary['average_hit_count']:.2f}")
        st.dataframe(evaluation_logs_table(logs), width="stretch", hide_index=True)
        st.subheader("券種別回収率")
        return_rate_table = aggregate_return_rates(logs)
        st.dataframe(return_rate_table, width="stretch", hide_index=True)

        st.subheader("勝因・敗因分析")
        for log in reversed(logs[-10:]):
            review = log.get("race_review", {})
            if not review:
                continue
            with st.expander(f"{log.get('race_date', '')} {log.get('race_name', '')}"):
                st.write(review.get("hit_summary", ""))
                for reason in review.get("miss_reason_candidates", []):
                    st.write(f"・{reason}")
                if review.get("horse_reviews"):
                    st.dataframe(pd.DataFrame(review["horse_reviews"]), width="stretch", hide_index=True)
                for feedback in review.get("model_feedback", []):
                    st.info(feedback)

    render_failure_analysis(logs)
    render_log_management()

    st.divider()
    st.subheader("予想重み最適化")
    current_weights = load_model_weights()
    st.caption("使用中の予想重み")
    st.json(current_weights)
    training_df = build_training_dataset(logs)
    st.write(f"学習データセット: {len(training_df)}行 / {training_df['race_id'].nunique() if not training_df.empty else 0}レース")
    if len(logs) < 20:
        st.warning("評価ログが少ないため、重み最適化は参考程度です。最低20レース以上の蓄積を推奨します。")
    optimizer_col1, optimizer_col2 = st.columns(2)
    with optimizer_col1:
        optimization_metric = st.selectbox(
            "最適化指標",
            ["top3_hit_rate", "winner_hit_rate", "brier_score", "log_loss"],
            key="optimization_metric",
        )
    with optimizer_col2:
        optimization_trials = st.selectbox("探索回数", [100, 300, 500, 1000], index=2, key="optimization_trials")
    if st.button("予想重みを最適化して保存", key="optimize_prediction_weights_button"):
        if training_df.empty:
            st.error("重み最適化に使える評価データがありません。")
        else:
            with st.spinner("予想重みを探索中です..."):
                optimized = optimize_prediction_weights(
                    training_df,
                    metric=optimization_metric,
                    n_trials=int(optimization_trials),
                )
                weights_path = save_model_weights(optimized)
            st.session_state["latest_weight_optimization"] = optimized
            st.success(f"最適化後の重みを保存しました: {weights_path}")
    optimized = st.session_state.get("latest_weight_optimization")
    if isinstance(optimized, dict):
        comparison = pd.DataFrame(
            {
                "feature": list(current_weights),
                "before": [current_weights[key] for key in current_weights],
                "after": [optimized.get("weights", {}).get(key, current_weights[key]) for key in current_weights],
            }
        )
        st.dataframe(comparison, width="stretch", hide_index=True)
        st.write(
            f"最適化前: {optimized.get('baseline_score', 0):.4f} / "
            f"最適化後: {optimized.get('score', 0):.4f}"
        )

    st.divider()
    st.subheader("機械学習モデル")
    ml_target = st.selectbox("目的変数", ["is_top3", "is_win"], key="ml_target")
    model_type = st.selectbox(
        "使用モデル",
        ["logistic", "random_forest", "lightgbm", "xgboost"],
        key="ml_model_type",
    )
    X, y = build_ml_dataset(logs, target=ml_target)
    st.write(f"MLデータセット件数: {len(X)}")
    if len(logs) < 50:
        st.warning("MLモデル学習にはデータが少なすぎます。最低50レース以上を推奨します。")
    if st.button("MLモデルを学習して保存", key="train_ml_model_button"):
        if X.empty:
            st.error("ML学習に使える評価データがありません。")
        else:
            with st.spinner("モデルを学習中です..."):
                model = train_prediction_model(X, y, model_type=model_type)
                model_path = TOP3_MODEL_PATH if ml_target == "is_top3" else WIN_MODEL_PATH
                save_model(model, model_path)
                predictions = pd.Series(model.predict(X), index=y.index)
                accuracy = float((predictions.astype(int) == y.astype(int)).mean())
                st.session_state["latest_ml_training"] = {
                    "path": str(model_path),
                    "accuracy": accuracy,
                    "target": ml_target,
                    "importance": model_feature_importance(model, ML_FEATURES),
                }
            st.success(f"学習済みモデルを保存しました: {model_path}")
    ml_training = st.session_state.get("latest_ml_training")
    if isinstance(ml_training, dict):
        st.metric("学習データ内の簡易正解率", f"{float(ml_training.get('accuracy', 0.0)):.1%}")
        importance = ml_training.get("importance")
        if isinstance(importance, pd.DataFrame) and not importance.empty:
            st.caption("特徴量重要度")
            st.dataframe(importance, width="stretch", hide_index=True)
            st.bar_chart(importance, x="feature", y="importance")
    elif TOP3_MODEL_PATH.is_file():
        existing_model = load_model(TOP3_MODEL_PATH)
        if existing_model is not None:
            importance = model_feature_importance(existing_model, ML_FEATURES)
            if not importance.empty:
                st.caption("保存済みtop3モデルの特徴量重要度")
                st.dataframe(importance, width="stretch", hide_index=True)


def render_prediction_report_section(result: dict[str, Any]) -> None:
    prediction_table = _prediction_table(result)
    if not isinstance(prediction_table, pd.DataFrame) or prediction_table.empty:
        return
    report_input = {
        "race_name": result.get("race_name", ""),
        "race_date": result.get("race_date", ""),
        "race_config": result.get("race_config", {}),
        "race_metadata": result.get("race_metadata", {}),
        "prediction_table": prediction_table,
        "horse_analysis": result.get("horse_analysis"),
        "comments_table": result.get("comments_table"),
        "pace_prediction": result.get("pace_prediction", {}),
        "same_race_trend_database": result.get("same_race_trend_database"),
        "same_race_trend_analysis": result.get("same_race_trend_analysis"),
        "single_result": result.get("single_result"),
        "video_path": result.get("mp4_path", result.get("animation_path", "")),
        "simulation_result": result,
    }
    report = generate_prediction_report(report_input)
    st.subheader("予想レポート")
    st.text_area("タイトル", value=str(report["title"]), height=70, key="prediction_report_title")
    st.text_area("レース概要", value=str(report["race_info"]), height=100, key="prediction_report_race_info")
    st.text_area("展開予測", value=str(report["pace_prediction"]), height=120, key="prediction_report_pace")
    marks = report.get("marks_table")
    if isinstance(marks, pd.DataFrame) and not marks.empty:
        st.caption("AI予想印")
        st.dataframe(marks, width="stretch", hide_index=True)
        st.text_area("AI予想印（コピー用）", value=marks.to_csv(index=False), height=150, key="prediction_report_marks")
    st.text_area("本命・対抗・穴の解説", value=str(report["top_picks"]), height=130, key="prediction_report_picks")
    comments = report.get("horse_comments")
    if isinstance(comments, pd.DataFrame) and not comments.empty:
        st.caption("全頭短評")
        st.dataframe(comments, width="stretch", hide_index=True)
        st.text_area("全頭短評（コピー用）", value=comments.to_csv(index=False), height=200, key="prediction_report_comments")
    st.text_area("シミュレーション概要", value=str(report["simulation_summary"]), height=120, key="prediction_report_simulation")
    st.text_area("リスク要素", value=str(report["risk_factors"]), height=130, key="prediction_report_risks")
    st.text_area("YouTube台本", value=str(report["youtube_script"]), height=340, key="prediction_report_youtube")
    st.text_area("ブログ本文", value=str(report["blog_text"]), height=420, key="prediction_report_blog")
    st.text_area("SNS投稿文", value=str(report["sns_text"]), height=180, key="prediction_report_sns")
    if st.button("予想レポートを保存", key="save_prediction_report_button"):
        paths = save_prediction_report(report, str(result.get("prediction_log_path", "")))
        st.session_state["latest_report_paths"] = {key: str(value) for key, value in paths.items()}
        st.success(f"レポートを保存しました: {paths['markdown']} / {paths['json']}")
    latest_paths = st.session_state.get("latest_report_paths")
    if isinstance(latest_paths, dict):
        st.caption(f"Markdown: {latest_paths.get('markdown', '')} / JSON: {latest_paths.get('json', '')}")


def render_youtube_output_tab() -> None:
    st.write("AI予想結果から、YouTube投稿に使いやすいサムネイルと構成動画を生成します。")
    prediction_log = _select_youtube_prediction_log()
    if not prediction_log:
        st.info("まず予想タブでシミュレーションを実行するか、予想ログを保存してください。")
        return

    race_name = str(prediction_log.get("race_name") or "対象レース")
    race_date = str(prediction_log.get("race_date") or "")
    st.caption(f"使用中の予想データ: {race_date} {race_name}")

    trend_key = f"{race_name}_{race_date}"
    trend_state = st.session_state.get("youtube_same_race_trends", {})
    if not isinstance(trend_state, dict):
        trend_state = {}
    stored_trend = trend_state.get(trend_key, {})
    if isinstance(stored_trend, dict) and stored_trend.get("analysis"):
        prediction_log = dict(prediction_log)
        prediction_log["same_race_history"] = stored_trend.get("history", [])
        prediction_log["same_race_trend_analysis"] = stored_trend.get("analysis", {})

    st.subheader("過去10年傾向")
    trend_col1, trend_col2 = st.columns([1, 3])
    with trend_col1:
        years = st.selectbox("取得年数", [5, 10], index=1, key="youtube_trend_years")
        fetch_trends_clicked = st.button("過去10年傾向を取得", key="fetch_same_race_trends_button")
    if fetch_trends_clicked:
        if not race_name or not race_date:
            st.warning("レース名と開催日が必要です。")
        else:
            with st.spinner("netkeibaから過去同一レースの結果を取得中です..."):
                race_config_for_trend = prediction_log.get("race_config", {})
                metadata_for_trend = prediction_log.get("race_metadata", {})
                if not isinstance(race_config_for_trend, dict):
                    race_config_for_trend = {}
                if not isinstance(metadata_for_trend, dict):
                    metadata_for_trend = {}
                venue = str(
                    race_config_for_trend.get("course")
                    or metadata_for_trend.get("venue")
                    or metadata_for_trend.get("course")
                    or ""
                )
                try:
                    distance = int(race_config_for_trend.get("distance") or metadata_for_trend.get("distance") or 0)
                except (TypeError, ValueError):
                    distance = 0
                if not venue or distance <= 0:
                    st.warning("過去10年傾向DBの作成には、競馬場と距離が必要です。")
                else:
                    trend_database = build_same_race_trend_database(
                        race_name=race_name,
                        venue=venue,
                        distance=distance,
                        years=int(years),
                        race_date=race_date,
                    )
                    analysis = analyze_same_race_trend_database(trend_database)
                    history = trend_database.get("rows", [])
                    trend_state[trend_key] = {
                        "history": history,
                        "database": trend_database,
                        "analysis": analysis,
                        "history_path": trend_database.get("save_paths", {}).get("json", ""),
                    }
                    st.session_state["youtube_same_race_trends"] = trend_state
                    prediction_log = dict(prediction_log)
                    prediction_log["same_race_history"] = history
                    prediction_log["same_race_trend_database"] = trend_database
                    prediction_log["same_race_trend_analysis"] = analysis
                    if history:
                        paths = trend_database.get("save_paths", {})
                        st.success(
                            f"過去傾向DBを作成しました: {len(history)}行 / "
                            f"{paths.get('json', '-')} / {paths.get('csv', '-')}"
                        )
                    else:
                        st.warning("過去傾向データを取得できませんでした。動画ではコース条件から簡易推定します。")
    stored_trend = st.session_state.get("youtube_same_race_trends", {}).get(trend_key, {})
    if isinstance(stored_trend, dict) and stored_trend.get("analysis"):
        with trend_col2:
            analysis = stored_trend.get("analysis", {})
            st.write(" / ".join(analysis.get("summary_bullets", [])[:3]))
            st.json(analysis.get("trend_scores", {}))
        history = stored_trend.get("history", [])
        if history:
            with st.expander("過去傾向取得データ"):
                st.dataframe(pd.DataFrame(history), width="stretch", hide_index=True)

    report = generate_prediction_report(prediction_log)
    structure = build_youtube_video_structure(prediction_log, str(prediction_log.get("video_path", "")))
    trend = generate_race_trend_summary(prediction_log)

    st.subheader("動画構成")
    st.write(" → ".join(REQUIRED_SECTION_ORDER))
    with st.expander("同レースの過去傾向", expanded=True):
        for bullet in trend.get("bullets", []):
            st.write(f"・{bullet}")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "順番": index + 1,
                    "セクション": item.get("section", ""),
                    "内容": item.get("title", ""),
                    "件数": len(item.get("rows", [])) if isinstance(item.get("rows"), list) else "",
                }
                for index, item in enumerate(structure)
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    diagnosis_rows = next((item.get("rows", []) for item in structure if item.get("section") == "全頭診断"), [])
    featured_rows = next((item.get("rows", []) for item in structure if item.get("section") == "注目馬の紹介と根拠説明"), [])
    st.subheader("全頭診断プレビュー")
    st.dataframe(pd.DataFrame(diagnosis_rows), width="stretch", hide_index=True)
    st.subheader("注目馬紹介プレビュー")
    st.dataframe(pd.DataFrame(featured_rows), width="stretch", hide_index=True)

    st.subheader("YouTube用サムネイル")
    if st.button("サムネイル生成", key="generate_youtube_thumbnail_button"):
        thumbnail_path = generate_youtube_thumbnail(prediction_log, "outputs/thumbnails")
        st.session_state["latest_youtube_thumbnail"] = thumbnail_path
        st.success(f"サムネイルを生成しました: {thumbnail_path}")
    thumbnail_path = st.session_state.get("latest_youtube_thumbnail")
    if isinstance(thumbnail_path, str) and Path(thumbnail_path).is_file():
        st.image(thumbnail_path)
        with open(thumbnail_path, "rb") as file:
            st.download_button(
                "サムネイルをダウンロード",
                data=file.read(),
                file_name=Path(thumbnail_path).name,
                mime="image/png",
                key="download_youtube_thumbnail",
            )

    st.subheader("YouTube予想動画")
    video_col1, video_col2 = st.columns(2)
    with video_col1:
        video_format = st.selectbox("動画フォーマット", ["YouTube横長 16:9", "TikTok縦長 9:16"], key="youtube_video_format")
    with video_col2:
        youtube_fps = st.selectbox("FPS", [12, 24, 30], index=1, key="youtube_video_fps")
    bgm_file = st.file_uploader("BGMファイルをアップロード", type=["mp3", "wav"], key="youtube_bgm_file")
    race_bgm_file = st.file_uploader("シミュレーション用BGM", type=["mp3", "wav"], key="youtube_race_bgm_file")
    start_se_file = st.file_uploader("スタート効果音", type=["mp3", "wav"], key="youtube_start_se_file")
    use_narration = st.checkbox("AI読み上げを使う", value=False, key="youtube_use_narration")

    race_video_path = st.text_input(
        "挿入するシミュレーション動画",
        value=str(prediction_log.get("video_path", "")),
        key="youtube_race_video_path",
        help="未生成または見つからない場合は、構成カードで代替します。",
    )
    if st.button("YouTube予想動画を生成", key="generate_youtube_prediction_video_button", type="primary"):
        output_path = Path("outputs/youtube_videos") / f"youtube_prediction_{datetime.now():%Y%m%d_%H%M%S}.mp4"
        bgm_path = _save_uploaded_media(bgm_file, "bgm")
        race_bgm_path = _save_uploaded_media(race_bgm_file, "race_bgm")
        start_se_path = _save_uploaded_media(start_se_file, "start_se")
        with st.spinner("YouTube向け動画を生成中です..."):
            try:
                youtube_video_path = build_youtube_prediction_video(
                    prediction_log=prediction_log,
                    race_video_path=race_video_path,
                    output_path=str(output_path),
                    video_format=video_format,
                    fps=int(youtube_fps),
                    bgm_path=bgm_path,
                    race_bgm_path=race_bgm_path,
                    start_se_path=start_se_path,
                    use_narration=use_narration,
                )
                st.session_state["latest_youtube_video"] = youtube_video_path
                st.success(f"YouTube予想動画を生成しました: {youtube_video_path}")
            except Exception as exc:
                st.error(f"YouTube予想動画の生成に失敗しました: {exc}")

    youtube_video_path = st.session_state.get("latest_youtube_video")
    if isinstance(youtube_video_path, str):
        path = Path(youtube_video_path)
        if path.is_file() and path.stat().st_size > 0:
            st.video(str(path))
            with open(path, "rb") as file:
                st.download_button(
                    "YouTube予想動画をダウンロード",
                    data=file.read(),
                    file_name=path.name,
                    mime="video/mp4",
                    key="download_youtube_prediction_video",
                )
            metadata_path = path.with_suffix(".json")
            if metadata_path.is_file():
                with st.expander("動画生成メタデータ"):
                    st.json(metadata_path.read_text(encoding="utf-8"))
        else:
            st.warning(f"動画ファイルが見つかりません: {youtube_video_path}")

    st.subheader("コピー用テキスト")
    st.text_area("YouTubeタイトル", value=str(report.get("title", "")), height=70, key="youtube_copy_title")
    st.text_area("概要欄", value=str(report.get("summary", "")), height=190, key="youtube_copy_summary")
    st.text_area("ハッシュタグ", value=_hashtags_from_report(report), height=80, key="youtube_copy_hashtags")
    st.text_area("台本", value=str(report.get("youtube_script", "")), height=360, key="youtube_copy_script")


def render_failure_analysis(logs: list[dict[str, Any]]) -> None:
    st.divider()
    st.subheader("的中・不的中の原因分析")
    tag_table = aggregate_failure_tags(logs)
    if tag_table.empty:
        st.info("原因分析を含む評価ログはまだありません。新しく実結果を保存すると集計されます。")
    else:
        display_tags = tag_table.copy()
        display_tags["割合"] = display_tags["割合"].map(lambda value: f"{float(value):.1%}")
        st.caption("原因タグ集計")
        st.dataframe(display_tags, width="stretch", hide_index=True)

    suggestions = aggregate_improvement_suggestions(logs)
    st.caption("改善提案一覧")
    if suggestions.empty:
        st.write("改善提案はまだありません。")
    else:
        st.dataframe(suggestions, width="stretch", hide_index=True)

    missed_logs = []
    for log in logs:
        metrics = log.get("evaluation_metrics", log.get("evaluation", {})) or {}
        if not bool(metrics.get("top3_hit")):
            missed_logs.append(log)
    if not missed_logs:
        return
    selected_index = st.selectbox(
        "外れレース詳細",
        list(range(len(missed_logs))),
        format_func=lambda index: f"{missed_logs[index].get('race_date', '')} {missed_logs[index].get('race_name', '')}",
        key="missed_race_detail_selector",
    )
    selected = missed_logs[int(selected_index)]
    prediction_rows = pd.DataFrame(selected.get("prediction_table", []))
    if not prediction_rows.empty:
        mark_column = prediction_rows.get("印", pd.Series(dtype=str)).astype(str)
        marked = prediction_rows[mark_column.isin(["◎", "○", "▲", "△", "☆"])]
        st.caption("予想印")
        st.dataframe(marked, width="stretch", hide_index=True)
    actual = pd.DataFrame(selected.get("actual_result", []))
    if not actual.empty:
        st.caption("実結果")
        st.dataframe(actual, width="stretch", hide_index=True)
    failure = selected.get("failure_analysis")
    if not isinstance(failure, dict):
        failure = (selected.get("evaluation_metrics") or selected.get("evaluation") or {}).get("failure_analysis", {})
    if not failure and selected.get("prediction_table") and selected.get("actual_result"):
        failure = analyze_prediction_failure(
            selected,
            list(selected.get("actual_result", [])),
            selected.get("evaluation_metrics", selected.get("evaluation", {})) or {},
        )
    st.write("原因タグ:", "、".join(failure.get("miss_reason_tags", [])) or "なし")
    review = selected.get("race_review", {})
    if (not isinstance(review, dict) or not review) and selected.get("actual_result"):
        review = generate_race_review(
            selected,
            list(selected.get("actual_result", [])),
            selected.get("evaluation_metrics", selected.get("evaluation", {})) or {},
        )
    horse_reviews = pd.DataFrame(review.get("horse_reviews", [])) if isinstance(review, dict) else pd.DataFrame()
    if not horse_reviews.empty:
        st.caption("各馬レビュー")
        st.dataframe(horse_reviews, width="stretch", hide_index=True)
    for suggestion in failure.get("improvement_suggestions", []):
        st.info(str(suggestion))


def render_log_management() -> None:
    st.divider()
    st.subheader("ログ管理")
    prediction_rows = log_inventory(PREDICTION_LOG_DIR)
    evaluation_rows = log_inventory(EVALUATION_LOG_DIR)

    st.caption("予想ログ一覧")
    prediction_frame = pd.DataFrame(prediction_rows)
    if prediction_frame.empty:
        st.info("予想ログはありません。")
    else:
        st.dataframe(
            prediction_frame[["ファイル名", "race_id", "race_name", "race_date", "timestamp"]],
            width="stretch",
            hide_index=True,
        )

    st.caption("評価ログ一覧")
    evaluation_frame = pd.DataFrame(evaluation_rows)
    if evaluation_frame.empty:
        st.info("評価ログはありません。")
    else:
        st.dataframe(
            evaluation_frame[["ファイル名", "race_id", "race_name", "race_date", "timestamp"]],
            width="stretch",
            hide_index=True,
        )

    prediction_paths = [row["path"] for row in prediction_rows]
    evaluation_paths = [row["path"] for row in evaluation_rows]
    selected_predictions = st.multiselect(
        "削除する予想ログを選択",
        prediction_paths,
        format_func=lambda value: Path(value).name,
        key="delete_prediction_logs",
    )
    delete_related = st.checkbox(
        "関連する評価ログも削除する",
        value=False,
        key="delete_related_evaluations",
    )
    selected_evaluations = st.multiselect(
        "削除する評価ログを選択",
        evaluation_paths,
        format_func=lambda value: Path(value).name,
        key="delete_evaluation_logs",
    )
    confirm_delete = st.checkbox("本当に削除します", value=False, key="confirm_log_delete")
    if st.button("選択したログを削除", key="delete_selected_logs"):
        if not confirm_delete:
            st.warning("削除前に「本当に削除します」を確認してください。")
        elif not selected_predictions and not selected_evaluations:
            st.warning("削除するログを選択してください。")
        else:
            targets = list(selected_predictions) + list(selected_evaluations)
            if delete_related and selected_predictions:
                for row in evaluation_rows:
                    if any(_same_log_path(row.get("prediction_log_path", ""), path) for path in selected_predictions):
                        targets.append(row["path"])
            outcome = delete_log_files(targets)
            if outcome["failed"]:
                st.error(f"削除できなかったファイルがあります: {outcome['failed']}")
            else:
                st.success(f"{len(outcome['deleted'])}ファイルを削除しました。")
                st.rerun()

    st.divider()
    if st.button("重複ログ検出", key="detect_duplicate_logs"):
        st.session_state["duplicate_log_scan"] = {
            "prediction": find_duplicate_logs(PREDICTION_LOG_DIR, "prediction"),
            "evaluation": find_duplicate_logs(EVALUATION_LOG_DIR, "evaluation"),
        }
    duplicate_scan = st.session_state.get("duplicate_log_scan")
    if isinstance(duplicate_scan, dict):
        duplicate_rows: list[dict[str, Any]] = []
        duplicate_paths: list[str] = []
        for log_type in ("prediction", "evaluation"):
            for group in duplicate_scan.get(log_type, []):
                duplicate_rows.append(
                    {
                        "種類": "予想" if log_type == "prediction" else "評価",
                        "race_id": group.get("race_id", ""),
                        "race_name": group.get("race_name", ""),
                        "race_date": group.get("race_date", ""),
                        "件数": group.get("count", 0),
                        "ファイル": ", ".join(group.get("files", [])),
                    }
                )
                duplicate_paths.extend(group.get("paths", []))
        if duplicate_rows:
            st.warning(f"重複ログを{len(duplicate_rows)}組検出しました。")
            st.dataframe(pd.DataFrame(duplicate_rows), width="stretch", hide_index=True)
            selected_duplicates = st.multiselect(
                "削除する重複ログを選択",
                list(dict.fromkeys(duplicate_paths)),
                format_func=lambda value: Path(value).name,
                key="delete_duplicate_logs",
            )
            confirm_duplicate_delete = st.checkbox(
                "重複ログの削除を確認する",
                value=False,
                key="confirm_duplicate_log_delete",
            )
            if st.button("選択した重複ログを削除", key="delete_selected_duplicate_logs"):
                if not confirm_duplicate_delete:
                    st.warning("削除前に確認チェックを入れてください。")
                elif not selected_duplicates:
                    st.warning("削除する重複ログを選択してください。")
                else:
                    outcome = delete_log_files(selected_duplicates)
                    if outcome["failed"]:
                        st.error(f"削除できなかったファイルがあります: {outcome['failed']}")
                    else:
                        st.session_state.pop("duplicate_log_scan", None)
                        st.success(f"{len(outcome['deleted'])}ファイルを削除しました。")
                        st.rerun()
        else:
            st.success("重複ログは見つかりませんでした。")


def _same_log_path(left: object, right: object) -> bool:
    if not left or not right:
        return False
    try:
        return os.path.normcase(os.path.abspath(os.fspath(left))) == os.path.normcase(os.path.abspath(os.fspath(right)))
    except (OSError, TypeError, ValueError):
        return str(left) == str(right)


def render_netkeiba_debug(debug: object, heading: str) -> None:
    st.subheader(heading)
    if not isinstance(debug, dict) or not debug:
        st.info("デバッグ情報はありません。")
        return
    for section_name, section in debug.items():
        with st.expander(str(section_name), expanded=False):
            if isinstance(section, pd.DataFrame):
                st.dataframe(section, width="stretch", hide_index=True)
                continue
            if not isinstance(section, dict):
                st.write(section)
                continue
            scalar_values = {
                key: value
                for key, value in section.items()
                if not isinstance(value, (pd.DataFrame, list, dict))
            }
            if scalar_values:
                st.json(scalar_values)
            for key, value in section.items():
                if isinstance(value, pd.DataFrame):
                    st.markdown(f"**{key}**")
                    st.dataframe(value, width="stretch", hide_index=True)
                elif isinstance(value, list):
                    st.markdown(f"**{key}**")
                    st.dataframe(pd.DataFrame(value), width="stretch", hide_index=True)
                elif isinstance(value, dict):
                    st.markdown(f"**{key}**")
                    st.json(value)


def _empty_actual_result_rows(count: int = 5) -> list[dict[str, Any]]:
    return [
        {
            "finish": None,
            "horse_name": "",
            "frame": None,
            "horse_number": None,
            "carried_weight": None,
            "jockey": "",
            "popularity": None,
            "time": "",
            "margin": "",
            "passing_order": "",
            "last3f": None,
        }
        for _ in range(count)
    ]


def _valid_actual_result_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        try:
            finish = int(float(row.get("finish", row.get("着順"))))
            horse_number = int(float(row.get("horse_number", row.get("馬番"))))
        except (TypeError, ValueError):
            continue
        if finish <= 0 or horse_number <= 0:
            continue
        cleaned = {key: (None if pd.isna(value) else value) for key, value in row.items()}
        cleaned["finish"] = finish
        cleaned["horse_number"] = horse_number
        rows.append(cleaned)
    return rows


def ensure_recent_races_exist(result: dict[str, object]) -> None:
    for item in result.get("recent_races", []):
        if not isinstance(item, dict):
            continue
        horse_name = str(item.get("horse_name", ""))
        recent_races = item.get("recent_races", [])
        if not recent_races:
            raise RaceDataFetchError(f"{horse_name} の近走データが取得できませんでした。")


def animation_mode_key(label: str) -> str:
    if label == "投稿用MP4":
        return "post_mp4"
    if label == "簡易2D":
        return "2d_gif"
    return "3d_plotly"


def render_results(result: dict[str, object], debug_mode: bool = False, prediction_sort: str = "prediction_score") -> None:
    horse_analysis = result["horse_analysis"]
    pace_prediction = result["pace_prediction"]
    if result.get("lightweight_mode"):
        st.info("軽量モードで実行しました。投稿用MP4を省略し、ブラウザ向け表示を優先しています。")

    st.subheader("今回のシミュレーション着順（AI期待値最大の代表例）")
    single_result = result.get("single_result")
    if isinstance(single_result, pd.DataFrame) and not single_result.empty:
        st.caption(str(result.get("single_result_source", "controlled_timeline final_frame")))
        st.dataframe(single_result, width="stretch", hide_index=True)
    else:
        st.warning("今回のシミュレーション着順を作成できませんでした。")

    st.subheader("レース展開予測")
    c1, c2, c3 = st.columns(3)
    c1.metric("予測ペース", str(pace_prediction.get("pace", "")))
    c2.metric("front_pressure", f"{float(pace_prediction.get('front_pressure', 0.0)):.2f}")
    c3.metric("差し有利度", f"{float(pace_prediction.get('closer_advantage', 0.0)):.2f}")
    st.write(pace_comment(pace_prediction))
    st.info(f"コースバイアス：{get_course_bias(result.get('race_config', {})).get('comment', '')}")
    st.dataframe(style_group_table(pace_prediction), width="stretch", hide_index=True)

    prediction = result.get("prediction")
    if isinstance(prediction, dict):
        render_prediction_results(
            prediction,
            sort_by=prediction_sort,
            horse_analysis=horse_analysis,
            pace_prediction=pace_prediction,
            race_config=result.get("race_config", {}),
        )

    st.subheader("各馬の分析結果")
    st.dataframe(format_analysis_table(horse_analysis), width="stretch", hide_index=True)
    with st.expander("脚質確率プロファイル"):
        probability_table = style_probability_long_table(horse_analysis)
        if not probability_table.empty:
            st.bar_chart(probability_table, x="馬名", y="確率", color="脚質")

    st.subheader("近走5走データ確認")
    recent_races_df = build_recent_races_table(result.get("recent_races", []))
    if recent_races_df.empty:
        st.info("近走データが取得できませんでした。")
    else:
        st.dataframe(recent_races_df, width="stretch", hide_index=True)

    if debug_mode:
        render_fetch_debug(result)
        render_style_detection_debug(result)
        render_timeline_debug(result)

    st.subheader("レース動画")
    renderer_name = str(result.get("renderer_name", "marker renderer"))
    horse_display_mode = str(result.get("horse_display_mode", "marker"))
    st.info(f"使用レンダラー: {renderer_name}")
    st.info(f"馬表示モード: {horse_display_mode}")
    html_path = existing_file_path(result.get("plotly_html_path"))
    gif_path = existing_file_path(result.get("gif_path"))
    mp4_path = existing_file_path(result.get("mp4_path"))
    if html_path is not None:
        st.iframe(str(html_path), height=720, width="stretch")
        st.download_button(
            "3D HTMLをダウンロード",
            data=html_path.read_bytes(),
            file_name=html_path.name,
            mime="text/html",
        )
    if gif_path is not None:
        st.image(str(gif_path), caption="race.gif")
        st.download_button(
            "GIFをダウンロード",
            data=gif_path.read_bytes(),
            file_name=gif_path.name,
            mime="image/gif",
        )
    video_error = str(result.get("video_error", "") or "")
    if video_error:
        st.error(video_error)
    if mp4_path is None:
        raw_path = result.get("mp4_path")
        if raw_path:
            st.error(f"動画ファイルが存在しません: {raw_path}")
    elif mp4_path.stat().st_size <= 0:
        st.error(f"動画ファイルが空です: {mp4_path}")
    else:
        st.video(str(mp4_path))
        st.download_button(
            "動画をダウンロード",
            data=mp4_path.read_bytes(),
            file_name="race_movie.mp4" if mp4_path.name == "race_movie.mp4" else mp4_path.name,
            mime="video/mp4",
        )

    if debug_mode:
        render_video_debug(result)

    with st.expander("実行ログ"):
        for line in result.get("log", []):
            st.write(line)


def render_video_for_result(
    result: dict[str, object],
    race_config: dict[str, object],
    horses: list[dict[str, object]],
    video_format_label: str,
    video_layout: str,
    duration_sec: int,
) -> None:
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"race_movie_{timestamp}.mp4"
    renderer_info: dict[str, str] = {}
    try:
        render_kwargs = {
            "race_timeline": result.get("race_timeline", []),
            "race_config": race_config,
            "horses": horses,
            "output_path": str(output_path),
            "video_format": video_format_label,
            "fps": 30,
            "duration_sec": duration_sec,
            "prediction_table": _prediction_table(result),
            "renderer_info": renderer_info,
        }
        if video_layout == "legacy_overview":
            video_path = render_race_video_from_timeline(**render_kwargs)
        else:
            video_path = render_side_scroll_race_video(**render_kwargs)
        result["mp4_path"] = video_path
        result["animation_path"] = video_path
        result["renderer_name"] = renderer_info.get("renderer_name", "Pillow/imageio timeline marker renderer")
        result["horse_display_mode"] = renderer_info.get("horse_display_mode", "marker")
        result["video_layout"] = renderer_info.get("video_layout", video_layout)
        result["video_format"] = video_format_label
        result["video_duration_sec"] = int(renderer_info.get("duration_sec", duration_sec))
        result["video_fps"] = int(renderer_info.get("fps", 30))
        result["video_total_frames"] = int(renderer_info.get("total_frames", 0) or 0)
        result["race_duration_sec"] = renderer_info.get("race_duration_sec", "")
        result["result_display_sec"] = renderer_info.get("result_display_sec", "")
        result["race_total_frames"] = int(renderer_info.get("race_total_frames", 0) or 0)
        result["result_total_frames"] = int(renderer_info.get("result_total_frames", 0) or 0)
        result["final_result_first_frame_index"] = int(renderer_info.get("final_result_first_frame_index", 0) or 0)
        if video_path is None:
            result["video_error"] = "動画生成に失敗しました: video_path が None です"
        elif not Path(str(video_path)).exists():
            result["video_error"] = f"動画ファイルが存在しません: {video_path}"
        else:
            result["video_error"] = ""
    except Exception as exc:
        result["mp4_path"] = str(output_path)
        result["video_layout"] = video_layout
        result["video_format"] = video_format_label
        result["video_duration_sec"] = duration_sec
        result["video_fps"] = 30
        result["video_total_frames"] = 30 * duration_sec
        result["video_error"] = f"動画生成に失敗しました: {exc}"
        result.setdefault("log", []).append(result["video_error"])


def _merge_prediction_trend_columns(result: dict[str, Any]) -> None:
    prediction = _prediction_table(result)
    analysis = result.get("horse_analysis")
    if not isinstance(prediction, pd.DataFrame) or prediction.empty:
        return
    if not isinstance(analysis, pd.DataFrame) or analysis.empty:
        return
    if "horse_number" not in analysis.columns:
        return
    trend_columns = [
        "horse_number",
        "race_trend_score",
        "frame_trend_score",
        "horse_number_trend_score",
        "style_trend_score",
        "agari_trend_score",
        "fourth_corner_trend_score",
        "age_trend_score",
        "weight_trend_score",
        "jockey_continuity_score",
        "previous_race_trend_score",
        "bloodline_trend_score",
        "trend_match_comment",
    ]
    existing = [column for column in trend_columns if column in prediction.columns]
    if "horse_number" not in existing:
        return
    trend_frame = prediction[existing].copy()
    base = analysis.drop(columns=[column for column in existing if column != "horse_number" and column in analysis.columns])
    result["horse_analysis"] = base.merge(trend_frame, on="horse_number", how="left")


def _prediction_table(result: dict[str, object]) -> pd.DataFrame | None:
    prediction = result.get("prediction")
    if isinstance(prediction, dict) and isinstance(prediction.get("prediction_table"), pd.DataFrame):
        return prediction["prediction_table"]
    return None


def _select_youtube_prediction_log() -> dict[str, Any] | None:
    current_result = st.session_state.get("simulation_result")
    options: list[tuple[str, dict[str, Any]]] = []
    if isinstance(current_result, dict) and _prediction_table(current_result) is not None:
        options.append(("現在のシミュレーション結果", _prediction_log_from_result(current_result)))

    saved_logs = load_prediction_logs()
    for log in reversed(saved_logs[-50:]):
        label = f"{log.get('race_date', '')} {log.get('race_name', '')} ({Path(str(log.get('_path', ''))).name or 'saved'})"
        options.append((label, log))

    if not options:
        return None
    selected_label = st.selectbox(
        "YouTube出力に使う予想データ",
        [label for label, _ in options],
        index=0,
        key="youtube_prediction_log_selector",
    )
    for label, log in options:
        if label == selected_label:
            return log
    return options[0][1]


def _prediction_log_from_result(result: dict[str, Any]) -> dict[str, Any]:
    prediction_table = _prediction_table(result)
    race_config = result.get("race_config", {})
    race_metadata = result.get("race_metadata", {})
    race_name = result.get("race_name") or (race_config.get("race_name") if isinstance(race_config, dict) else "") or (
        race_metadata.get("race_name") if isinstance(race_metadata, dict) else ""
    )
    race_date = result.get("race_date") or (race_config.get("race_date") if isinstance(race_config, dict) else "") or (
        race_metadata.get("race_date") if isinstance(race_metadata, dict) else ""
    )
    return {
        "race_id": result.get("race_id", ""),
        "race_name": race_name,
        "race_date": race_date,
        "race_config": race_config,
        "race_metadata": race_metadata,
        "prediction_table": prediction_table if prediction_table is not None else [],
        "horse_analysis": result.get("horse_analysis"),
        "comments_table": result.get("comments_table"),
        "pace_prediction": result.get("pace_prediction", {}),
        "single_result": result.get("single_result"),
        "video_path": str(result.get("mp4_path", result.get("animation_path", "")) or ""),
        "simulation_result": result,
        "prediction_log_path": str(result.get("prediction_log_path", "")),
    }


def _hashtags_from_report(report: dict[str, Any]) -> str:
    sns = str(report.get("sns_text", ""))
    tags = [token for token in sns.replace("\n", " ").split(" ") if token.startswith("#")]
    if not tags:
        tags = ["#競馬予想", "#AI予想", "#競馬シミュレーション"]
    return " ".join(dict.fromkeys(tags))


def _save_uploaded_media(uploaded_file: Any, prefix: str) -> str:
    if uploaded_file is None:
        return ""
    suffix = Path(getattr(uploaded_file, "name", "")).suffix.lower() or ".wav"
    output_dir = Path("outputs/youtube_videos/tmp")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S_%f}{suffix}"
    path.write_bytes(uploaded_file.getvalue())
    return str(path)


def render_video_debug(result: dict[str, object]) -> None:
    video_path = str(result.get("mp4_path", "") or "")
    path = Path(video_path) if video_path else None
    exists = bool(path and path.exists())
    st.subheader("動画生成デバッグ")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "video_path": video_path,
                    "exists": exists,
                    "file_size": path.stat().st_size if exists and path is not None else 0,
                    "renderer_name": result.get("renderer_name", ""),
                    "horse_display_mode": result.get("horse_display_mode", ""),
                    "video_layout": result.get("video_layout", ""),
                    "duration_sec": result.get("video_duration_sec", ""),
                    "race_duration_sec": result.get("race_duration_sec", ""),
                    "result_display_sec": result.get("result_display_sec", ""),
                    "fps": result.get("video_fps", ""),
                    "total_frames": result.get("video_total_frames", ""),
                    "race_total_frames": result.get("race_total_frames", ""),
                    "result_total_frames": result.get("result_total_frames", ""),
                    "final_result_first_frame_index": result.get("final_result_first_frame_index", ""),
                    "output_path": video_path,
                    "timeline_mode": result.get("timeline_mode", ""),
                    "timeline_frames": len(result.get("race_timeline", []) or []),
                }
            ]
        ),
        width="stretch",
        hide_index=True,
    )


def existing_file_path(value: object) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text)
    return path if path.is_file() else None


def render_prediction_results(
    prediction: dict[str, object],
    sort_by: str = "prediction_score",
    horse_analysis: pd.DataFrame | None = None,
    pace_prediction: dict[str, object] | None = None,
    race_config: dict[str, object] | None = None,
) -> None:
    prediction_table = prediction.get("prediction_table")
    if not isinstance(prediction_table, pd.DataFrame) or prediction_table.empty:
        return

    st.subheader("AI予想ランキング")
    sorted_table = sort_prediction_table(prediction_table, sort_by)
    st.dataframe(sorted_table, width="stretch", hide_index=True)

    st.subheader("勝率・連対率・複勝率グラフ")
    rate_table = prediction_table[["馬名", "win_rate", "top2_rate", "top3_rate"]].melt(
        id_vars="馬名",
        var_name="指標",
        value_name="確率",
    )
    st.bar_chart(rate_table, x="馬名", y="確率", color="指標")

    if horse_analysis is not None and pace_prediction is not None and race_config is not None:
        comments_df = build_horse_comments_table(sorted_table, horse_analysis, pace_prediction, race_config)
        st.subheader("全頭短評")
        st.dataframe(comments_df, width="stretch", hide_index=True)

    with st.expander("予想根拠"):
        for _, row in sorted_table.iterrows():
            mark = str(row.get("印", ""))
            st.markdown(f"**{mark} {row['馬番']} {row['馬名']}**")
            st.write(row.get("予想根拠", ""))

    logs = prediction.get("simulation_logs", [])
    if logs:
        with st.expander("Monte Carloログ"):
            for line in logs:
                st.write(line)


def sort_prediction_table(prediction_table: pd.DataFrame, sort_by: str) -> pd.DataFrame:
    score_column = "prediction_score" if "prediction_score" in prediction_table.columns else "score"
    if sort_by not in prediction_table.columns:
        sort_by = score_column
    if sort_by in {"score", "prediction_score"}:
        return prediction_table.sort_values(
            [score_column, "win_rate", "top3_rate", "avg_finish"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
    if sort_by == "avg_finish":
        return prediction_table.sort_values(
            ["avg_finish", score_column, "win_rate", "top3_rate"],
            ascending=[True, False, False, False],
        ).reset_index(drop=True)
    return prediction_table.sort_values(
        [sort_by, score_column, "win_rate", "top3_rate", "avg_finish"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)


def render_fetch_debug(result: dict[str, object]) -> None:
    st.subheader("データ取得デバッグ")
    records = result.get("fetch_debug", [])
    if not records:
        st.info("データ取得デバッグ情報がありません。既存の取得プロバイダで raw_race_df などを保持している場合は自動表示されます。")
        return

    for record in records:
        if not isinstance(record, dict):
            continue
        input_name = str(record.get("input_horse_name", ""))
        fetched_name = str(record.get("search_result_horse_name", "") or "")
        horse_id = str(record.get("horse_id", "") or "")
        url = str(record.get("url", "") or "")
        title = f"{input_name} / horse_id: {horse_id or '-'}"
        with st.expander(title, expanded=True):
            c1, c2, c3 = st.columns(3)
            c1.write("入力馬名")
            c1.code(input_name or "-", language="text")
            c2.write("検索結果で取得した馬名")
            c2.code(fetched_name or "-", language="text")
            c3.write("horse_id")
            c3.code(horse_id or "-", language="text")

            st.write("取得URL")
            if url:
                st.markdown(f"[{url}]({url})")
            else:
                st.code("-", language="text")

            name_match = record.get("name_match")
            if name_match is True:
                st.success("入力馬名と取得ページ上の馬名は一致しています。")
            elif name_match is False:
                st.warning("入力馬名と取得ページ上の馬名が一致していません。馬名検索または horse_id 取得の誤りを確認してください。")
            else:
                st.info("取得ページ上の馬名を確認できないため、入力馬名との一致判定ができません。")

            st.markdown("**raw_race_df（分析に使う前の生データ）**")
            render_debug_table_like(record.get("raw_race_df"))

            st.markdown("**整形後のrecent_races**")
            render_debug_table_like(record.get("recent_races"))


def render_debug_table_like(value: Any) -> None:
    if isinstance(value, list) and value and all(isinstance(item, pd.DataFrame) for item in value):
        for index, frame in enumerate(value, start=1):
            st.caption(f"table {index}")
            st.dataframe(frame, width="stretch", hide_index=True)
        return

    frame = to_debug_dataframe(value)
    st.dataframe(frame, width="stretch", hide_index=True)
    if frame.empty:
        st.caption("表示できるデータがありません。")


def to_debug_dataframe(value: Any) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, dict):
        return pd.DataFrame([simplify_debug_row(value)])
    if isinstance(value, (list, tuple)):
        return pd.DataFrame([simplify_debug_row(item) for item in value])
    if hasattr(value, "__dict__"):
        return pd.DataFrame([simplify_debug_row(value)])
    return pd.DataFrame({"value": [str(value)]})


def simplify_debug_row(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        row = dict(value)
    elif hasattr(value, "__dict__"):
        row = dict(value.__dict__)
    else:
        return {"value": str(value)}

    simplified: dict[str, Any] = {}
    for key, item in row.items():
        if isinstance(item, pd.DataFrame):
            simplified[key] = f"<DataFrame shape={item.shape}>"
        elif isinstance(item, (list, tuple)) and item and all(isinstance(v, pd.DataFrame) for v in item):
            simplified[key] = f"<{len(item)} DataFrames>"
        elif isinstance(item, (list, tuple)):
            simplified[key] = "-".join(str(v) for v in item)
        elif isinstance(item, dict):
            simplified[key] = str(item)
        else:
            simplified[key] = item
    return simplified


def render_style_detection_debug(result: dict[str, object]) -> None:
    st.subheader("脚質判定デバッグ")
    analysis = result.get("horse_analysis")
    if not isinstance(analysis, pd.DataFrame) or analysis.empty:
        st.info("脚質判定のデバッグ情報がありません。")
        return

    rows: list[dict[str, object]] = []
    warning_messages: list[str] = []
    for _, item in analysis.iterrows():
        horse_name = str(item.get("horse_name", ""))
        warnings = _debug_list(item.get("field_size_warnings", []))
        warning_messages.extend(f"{horse_name}: {warning}" for warning in warnings if warning)
        rows.append(
            {
                "馬名": horse_name,
                "field_size一覧": _debug_join(item.get("debug_field_sizes", [])),
                "passing_order一覧": _debug_join(item.get("debug_passing_orders", [])),
                "first_ratio一覧": _debug_join(item.get("debug_first_ratios", [])),
                "mid_ratio一覧": _debug_join(item.get("debug_mid_ratios", [])),
                "last_corner_ratio一覧": _debug_join(item.get("debug_last_corner_ratios", [])),
                "base_style_profile": item.get("base_style_profile", {}),
                "adjusted_style_profile": item.get("adjusted_style_profile", {}),
                "primary_running_style": item.get("primary_running_style", ""),
                "actual_running_style_fixed": item.get(
                    "actual_running_style_fixed",
                    item.get("actual_running_style", ""),
                ),
                "style_sample_size": item.get("style_sample_size", 0),
                "first_ratio平均": item.get("weighted_avg_first_ratio", ""),
                "mid_ratio平均": item.get("weighted_avg_mid_ratio", ""),
                "last_corner_ratio平均": item.get("weighted_avg_last_corner_ratio", ""),
                "late_gain平均": item.get("weighted_avg_late_gain", ""),
                "mud_aptitude": item.get("mud_aptitude", ""),
                "mud_source": item.get("mud_source", "neutral"),
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    for message in warning_messages:
        st.warning(message)


def _debug_list(value: object) -> list[object]:
    if isinstance(value, (list, tuple)):
        return list(value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    return [value]


def _debug_join(value: object) -> str:
    return " | ".join(str(item) for item in _debug_list(value))


def render_timeline_debug(result: dict[str, object]) -> None:
    st.subheader("シミュレーション隊列デバッグ")
    st.info(f"timeline_mode: {result.get('timeline_mode', 'controlled')}")
    representative_trial = result.get("representative_trial")
    if isinstance(representative_trial, dict) and representative_trial:
        st.write(
            {
                "selected_trial_index": representative_trial.get("trial_index"),
                "selected_trial_seed": representative_trial.get("seed"),
                "representative_value_score": representative_trial.get("representative_value_score"),
                "top5_horses_in_selected_trial": representative_trial.get("top5_horses_in_selected_trial", []),
            }
        )
    prediction = result.get("prediction", {})
    prediction_table = prediction.get("prediction_table") if isinstance(prediction, dict) else None
    if isinstance(prediction_table, pd.DataFrame) and not prediction_table.empty:
        debug_columns = [
            column
            for column in ["馬番", "馬名", "prediction_score", "race_power", "win_rate", "top3_rate"]
            if column in prediction_table.columns
        ]
        st.markdown("**prediction_table debug**")
        st.dataframe(prediction_table[debug_columns], width="stretch", hide_index=True)

    style_mismatches: list[dict[str, object]] = []
    for frame in result.get("race_timeline", []):
        if not isinstance(frame, dict):
            continue
        for horse in frame.get("horses", []):
            if not isinstance(horse, dict):
                continue
            fixed = str(horse.get("actual_running_style_fixed", ""))
            in_frame = str(horse.get("actual_running_style", ""))
            if fixed and in_frame and fixed != in_frame:
                style_mismatches.append(
                    {
                        "progress": frame.get("progress", ""),
                        "馬番": horse.get("horse_number", ""),
                        "actual_running_style_fixed": fixed,
                        "actual_running_style_in_frame": in_frame,
                    }
                )
    if style_mismatches:
        st.warning("固定したactual_running_styleとフレーム内の脚質が一致しません。")
        st.dataframe(pd.DataFrame(style_mismatches), width="stretch", hide_index=True)

    table = build_timeline_debug_table(
        result.get("race_timeline", []),
        prediction_table=prediction_table,
        horse_analysis=result.get("horse_analysis"),
        race_config=result.get("race_config"),
        video_format=result.get("video_format", "YouTube讓ｪ髟ｷ 16:9"),
    )
    if table.empty:
        st.info("race_timeline のデバッグ情報がありません。")
        return
    for target in ["20%", "50%", "80%", "95%", "100%"]:
        target_table = table[table["target"] == target]
        if target_table.empty:
            continue
        st.markdown(f"**{target}地点順位表**")
        st.dataframe(target_table.drop(columns=["target"]), width="stretch", hide_index=True)
    render_finish_distribution_debug(result.get("race_timeline", []))


def build_timeline_debug_table(
    race_timeline: object,
    prediction_table: object | None = None,
    horse_analysis: object | None = None,
    race_config: object | None = None,
    video_format: object = "YouTube讓ｪ髟ｷ 16:9",
) -> pd.DataFrame:
    if not isinstance(race_timeline, list) or not race_timeline:
        return pd.DataFrame()
    prediction_by_number: dict[int, dict[str, object]] = {}
    if isinstance(prediction_table, pd.DataFrame) and not prediction_table.empty and "馬番" in prediction_table.columns:
        prediction_by_number = {
            int(row["馬番"]): row.to_dict()
            for _, row in prediction_table.iterrows()
            if str(row.get("馬番", "")).strip() != ""
        }
    analysis_by_number: dict[int, dict[str, object]] = {}
    if isinstance(horse_analysis, pd.DataFrame) and not horse_analysis.empty and "horse_number" in horse_analysis.columns:
        analysis_by_number = {
            int(row["horse_number"]): row.to_dict()
            for _, row in horse_analysis.iterrows()
            if str(row.get("horse_number", "")).strip() != ""
        }
    lane_order = debug_lane_order(race_timeline)
    frames = [frame for frame in race_timeline if isinstance(frame, dict)]
    direction = debug_race_direction(race_config)
    render_direction = "right_to_left" if direction == "右" else "left_to_right"
    start_gate_side = "right" if render_direction == "right_to_left" else "left"
    goal_side = "left" if render_direction == "right_to_left" else "right"
    first_styles = debug_style_by_number(frames[0]) if frames else {}
    final_styles = debug_style_by_number(frames[-1]) if frames else {}
    rows: list[dict[str, object]] = []
    for target in [0.20, 0.50, 0.80, 0.95, 1.00]:
        if not frames:
            continue
        frame = min(frames, key=lambda item: abs(float(item.get("progress", 0.0)) - target))
        progress = float(frame.get("progress", 0.0))
        horses = frame.get("horses", [])
        if not isinstance(horses, list):
            continue
        positions = [debug_float(horse.get("position_m"), 0.0) for horse in horses if isinstance(horse, dict)]
        gaps = [debug_float(horse.get("gap_from_leader"), 0.0) for horse in horses if isinstance(horse, dict)]
        leader_position = max(positions, default=0.0)
        race_distance = debug_race_distance(race_config, default=leader_position)
        final_result_display_started = bool(frame.get("is_post_goal_frame", False) and frame.get("final_result_display_started", False))
        nige_gaps = [
            debug_float(horse.get("gap_from_leader"), 0.0)
            for horse in horses
            if isinstance(horse, dict) and debug_fixed_style(horse) == "\u9003\u3052"
        ]
        senko_gaps = [
            debug_float(horse.get("gap_from_leader"), 0.0)
            for horse in horses
            if isinstance(horse, dict) and debug_fixed_style(horse) == "\u5148\u884c"
        ]
        nige_gap = min(nige_gaps) if nige_gaps else ""
        senko_gap = min(senko_gaps) if senko_gaps else ""
        nige_senko_gap = round(float(senko_gap) - float(nige_gap), 3) if nige_gaps and senko_gaps else ""
        field_spread = max(positions, default=0.0) - min(positions, default=0.0)
        leader_gap_range = f"{min(gaps, default=0.0):.1f}-{max(gaps, default=0.0):.1f}"
        screen_params = debug_side_scroll_params(video_format)
        visible_distance_m = screen_params["visible_distance_m"]
        if progress < 0.02:
            camera_x = -visible_distance_m * 0.12
        else:
            camera_x = max(0.0, leader_position - visible_distance_m * 0.65)
            camera_x = min(camera_x, leader_position - visible_distance_m * 0.35)
            camera_x = max(camera_x, -visible_distance_m * 0.10)
        marker_radius_px = screen_params["marker_radius"]
        start_screen_x = debug_start_screen_x(screen_params, render_direction)
        initial_x_values: list[float] = []
        initial_y_values: list[float] = []
        for horse in horses:
            if not isinstance(horse, dict):
                continue
            horse_number = int(horse.get("horse_number", 0) or 0)
            prediction_row = prediction_by_number.get(horse_number, {})
            analysis_row = analysis_by_number.get(horse_number, {})
            position_m = debug_float(horse.get("position_m"), 0.0)
            screen_x = debug_side_scroll_screen_x(screen_params, position_m, camera_x, render_direction)
            screen_y = debug_lane_screen_y(
                lane_order=lane_order,
                horse_number=horse_number,
                field_size=max(1, len(lane_order)),
                track_top=screen_params["track_top"],
                track_bottom=screen_params["track_bottom"],
                marker_radius=screen_params["marker_radius"],
                frame_index=int(frame.get("index", 0) or 0),
            )
            if progress < 0.03:
                initial_x_values.append(start_screen_x)
                initial_y_values.append(screen_y)
            fixed_style = debug_fixed_style(horse)
            rows.append(
                {
                    "target": f"{int(target * 100)}%",
                    "direction": direction,
                    "render_direction": render_direction,
                    "start_gate_side": start_gate_side,
                    "goal_side": goal_side,
                    "current_frame_index": frame.get("index", ""),
                    "is_post_goal_frame": bool(frame.get("is_post_goal_frame", False)),
                    "progress": round(progress, 3),
                    "rank": horse.get("rank", ""),
                    "horse_number": horse_number or "",
                    "actual_running_style_fixed": fixed_style,
                    "actual_running_style": horse.get("actual_running_style", ""),
                    "actual_running_style_in_frame": horse.get("actual_running_style", ""),
                    "actual_running_style_first_frame": first_styles.get(horse_number, ""),
                    "actual_running_style_final_frame": final_styles.get(horse_number, ""),
                    "horse_ability_score": horse.get("horse_ability_score", analysis_row.get("horse_ability_score", "")),
                    "race_level_score": horse.get("race_level_score", analysis_row.get("race_level_score", "")),
                    "finish_score": horse.get("finish_score", analysis_row.get("finish_score", "")),
                    "margin_score": horse.get("margin_score", analysis_row.get("margin_score", "")),
                    "time_score": horse.get("time_score", analysis_row.get("time_score", "")),
                    "race_trend_score": prediction_row.get("race_trend_score", analysis_row.get("race_trend_score", "")),
                    "style_trend_score": prediction_row.get("style_trend_score", analysis_row.get("style_trend_score", "")),
                    "agari_trend_score": prediction_row.get("agari_trend_score", analysis_row.get("agari_trend_score", "")),
                    "race_power": horse.get("race_power", analysis_row.get("race_power", "")),
                    "prediction_score": prediction_row.get("prediction_score", prediction_row.get("score", "")),
                    "performance_index": horse.get("performance_index", ""),
                    "final_performance_score": horse.get("final_performance_score", ""),
                    "late_power": horse.get("late_power", ""),
                    "random_noise": horse.get("random_noise", ""),
                    "stamina": horse.get("stamina", ""),
                    "acceleration": horse.get("acceleration", ""),
                    "last3f_score": horse.get("last3f_score", ""),
                    "late_kick_score": horse.get("late_kick_score", analysis_row.get("late_kick_score", "")),
                    "early_push_score": horse.get("early_push_score", analysis_row.get("early_push_score", "")),
                    "mid_cruise_score": horse.get("mid_cruise_score", analysis_row.get("mid_cruise_score", "")),
                    "fade_resistance_score": horse.get("fade_resistance_score", analysis_row.get("fade_resistance_score", "")),
                    "sustain_speed_score": horse.get("sustain_speed_score", analysis_row.get("sustain_speed_score", "")),
                    "pace_resilience_score": horse.get("pace_resilience_score", analysis_row.get("pace_resilience_score", "")),
                    "agari_reliability": horse.get("agari_reliability", analysis_row.get("agari_reliability", "")),
                    "avg_last3f": horse.get("avg_last3f", analysis_row.get("avg_last3f", "")),
                    "best_last3f": horse.get("best_last3f", analysis_row.get("best_last3f", "")),
                    "late_gain_score": horse.get("late_gain_score", analysis_row.get("late_gain_score", "")),
                    "pace_fit_score": horse.get("pace_fit_score", ""),
                    "carried_weight": horse.get("carried_weight", analysis_row.get("carried_weight", "")),
                    "weight_penalty": horse.get("weight_penalty", analysis_row.get("weight_penalty", "")),
                    "final_stretch_score": horse.get("final_stretch_score", ""),
                    "style_attack_ratio": horse.get("style_attack_ratio", ""),
                    "straight_attack_score": horse.get("straight_attack_score", ""),
                    "early_gap_range": horse.get("early_gap_range", ""),
                    "final_stretch_start_progress": horse.get("final_stretch_start_progress", ""),
                    "pace": horse.get("pace", ""),
                    "late_ratio": horse.get("late_ratio", ""),
                    "gap_adjustment": horse.get("gap_adjustment", ""),
                    "fade_penalty": horse.get("fade_penalty", ""),
                    "late_gain_multiplier": horse.get("late_gain_multiplier", ""),
                    "tie_breaker": horse.get("tie_breaker", ""),
                    "position_m": horse.get("position_m", ""),
                    "finish_position_m": horse.get("finish_position_m", horse.get("position_m", "")),
                    "gap_from_leader": horse.get("gap_from_leader", ""),
                    "nige_gap": nige_gap,
                    "senko_gap": senko_gap,
                    "nige_senko_gap": horse.get("nige_senko_gap", nige_senko_gap),
                    "final_result_display_started": final_result_display_started,
                    "start_screen_x": round(start_screen_x, 1),
                    "marker_radius_px": round(marker_radius_px, 1),
                    "min_initial_screen_x": round(min(initial_x_values), 1) if initial_x_values else "",
                    "max_initial_screen_x": round(max(initial_x_values), 1) if initial_x_values else "",
                    "min_initial_screen_y": round(min(initial_y_values), 1) if initial_y_values else "",
                    "max_initial_screen_y": round(max(initial_y_values), 1) if initial_y_values else "",
                    "gap_from_winner": round(max(0.0, leader_position - position_m), 3),
                    "screen_x": round(screen_x, 1),
                    "screen_y": round(screen_y, 1),
                    "leader_gap_range": leader_gap_range,
                    "field_spread": round(field_spread, 1),
                    "ability_factor": horse.get("ability_factor", ""),
                }
            )
    return pd.DataFrame(rows)


def render_finish_distribution_debug(race_timeline: object) -> None:
    if not isinstance(race_timeline, list) or not race_timeline:
        return
    final_frame = race_timeline[-1]
    if not isinstance(final_frame, dict):
        return
    horses = [horse for horse in final_frame.get("horses", []) if isinstance(horse, dict)]
    if not horses:
        return

    rounded_positions = [round(debug_float(horse.get("position_m"), 0.0), 3) for horse in horses]
    duplicated_position_count = len(rounded_positions) - len(set(rounded_positions))
    if duplicated_position_count > 0:
        st.warning("Finish positions contain near-duplicates. Check late spread / tie-breaker.")

    top5 = sorted(horses, key=lambda horse: int(horse.get("rank", 999) or 999))[:5]
    top5_table = pd.DataFrame(
        [
            {
                "着順": int(horse.get("rank", 999) or 999),
                "馬番": int(horse.get("horse_number", 0) or 0),
                "脚質": horse.get("actual_running_style", ""),
                "stamina": horse.get("stamina", ""),
                "acceleration": horse.get("acceleration", ""),
                "last3f_score": horse.get("last3f_score", ""),
                "late_kick_score": horse.get("late_kick_score", ""),
                "late_gain_score": horse.get("late_gain_score", ""),
                "pace_fit_score": horse.get("pace_fit_score", ""),
                "weight_penalty": horse.get("weight_penalty", ""),
                "final_stretch_score": horse.get("final_stretch_score", ""),
                "pace": horse.get("pace", ""),
                "late_power": horse.get("late_power", ""),
                "fade_penalty": horse.get("fade_penalty", ""),
                "final_position_m": horse.get("position_m", ""),
            }
            for horse in top5
        ]
    )
    if not top5_table.empty:
        st.markdown("**最終着順上位5頭の脚質分布**")
        st.dataframe(top5_table, width="stretch", hide_index=True)

    style_order = {"逃げ": 0, "先行": 1, "自在": 2, "差し": 3, "追込": 4}
    finish_styles = [style_order.get(str(horse.get("actual_running_style", "")), 99) for horse in sorted(horses, key=lambda item: int(item.get("rank", 999) or 999))]
    if len(finish_styles) > 1 and finish_styles == sorted(finish_styles):
        st.warning("最終着順が脚質順に寄りすぎています。終盤補正を確認してください。")

    distribution = pd.DataFrame(
        [
            {
                "actual_running_style": horse.get("actual_running_style", ""),
                "rank": int(horse.get("rank", 999) or 999),
                "horse_count": 1,
            }
            for horse in horses
        ]
    )
    if distribution.empty:
        return
    style_distribution = (
        distribution.groupby("actual_running_style", dropna=False)
        .agg(
            horse_count=("horse_count", "sum"),
            best_rank=("rank", "min"),
            avg_rank=("rank", "mean"),
        )
        .reset_index()
    )
    st.markdown("**finish style distribution**")
    st.dataframe(style_distribution, width="stretch", hide_index=True)


def debug_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def debug_race_distance(race_config: object, default: float = 0.0) -> float:
    if isinstance(race_config, dict):
        return debug_float(race_config.get("distance"), default)
    return debug_float(getattr(race_config, "distance", default), default)


def debug_config_get(race_config: object, key: str, default: object = "") -> object:
    if isinstance(race_config, dict):
        return race_config.get(key, default)
    return getattr(race_config, key, default)


def debug_race_direction(race_config: object) -> str:
    value = str(debug_config_get(race_config, "direction", debug_config_get(race_config, "turn_direction", "左"))).strip()
    if value == "右" or "右" in value or value.lower().startswith("right"):
        return "右"
    return "左"


def debug_fixed_style(horse: dict[str, object]) -> str:
    return str(horse.get("actual_running_style_fixed", horse.get("actual_running_style", "")))


def debug_style_by_number(frame: dict[str, object]) -> dict[int, str]:
    horses = frame.get("horses", []) if isinstance(frame, dict) else []
    if not isinstance(horses, list):
        return {}
    return {
        int(horse.get("horse_number", 0) or 0): debug_fixed_style(horse)
        for horse in horses
        if isinstance(horse, dict) and int(horse.get("horse_number", 0) or 0) > 0
    }


def debug_lane_order(race_timeline: object) -> dict[int, int]:
    numbers: list[int] = []
    if isinstance(race_timeline, list):
        for frame in race_timeline:
            if not isinstance(frame, dict):
                continue
            horses = frame.get("horses", [])
            if not isinstance(horses, list):
                continue
            for horse in horses:
                if not isinstance(horse, dict):
                    continue
                number = int(horse.get("horse_number", 0) or 0)
                if number > 0 and number not in numbers:
                    numbers.append(number)
            if numbers:
                break
    return {number: index for index, number in enumerate(sorted(numbers))}


def debug_side_scroll_params(video_format: object) -> dict[str, float]:
    label = str(video_format)
    is_vertical = "TikTok" in label or label.lower() == "tiktok"
    width = 1080.0 if is_vertical else 1920.0
    height = 1920.0 if is_vertical else 1080.0
    track_top = height * (0.18 if is_vertical else 0.22)
    track_bottom = height * 0.96
    visible_distance_m = 200.0
    return {
        "width": width,
        "height": height,
        "track_top": track_top,
        "track_bottom": track_bottom,
        "px_per_m": width / visible_distance_m,
        "screen_origin_x": 0.0,
        "camera_lead_m": visible_distance_m * 0.65,
        "visible_distance_m": visible_distance_m,
        "marker_radius": max(20.0, width * (0.028 if is_vertical else 0.016)),
    }


def debug_side_scroll_screen_x(
    screen_params: dict[str, float],
    world_x: float,
    camera_x: float,
    render_direction: str,
) -> float:
    width = screen_params["width"]
    visible_distance_m = screen_params["visible_distance_m"]
    screen_x = (world_x - camera_x) / max(1.0, visible_distance_m) * width
    if render_direction == "right_to_left":
        return width - screen_x
    return screen_x


def debug_start_screen_x(screen_params: dict[str, float], render_direction: str) -> float:
    width = screen_params["width"]
    marker_radius = screen_params["marker_radius"] * 1.25
    if render_direction == "right_to_left":
        return min(width * 0.85, width - marker_radius - 20.0)
    return max(width * 0.15, marker_radius + 20.0)


def debug_lane_screen_y(
    lane_order: dict[int, int],
    horse_number: int,
    field_size: int,
    track_top: float,
    track_bottom: float,
    marker_radius: float,
    frame_index: int,
) -> float:
    lane_index = lane_order.get(horse_number, max(0, horse_number - 1))
    vertical_margin = max(marker_radius * 1.45, (track_bottom - track_top) * 0.025)
    usable_top = track_top + vertical_margin
    usable_bottom = track_bottom - vertical_margin
    if field_size <= 1:
        base_y = (usable_top + usable_bottom) / 2.0
    else:
        base_y = usable_top + (usable_bottom - usable_top) * (lane_index + 0.5) / field_size
    base_y += math.sin(frame_index * 0.1 + horse_number) * 2.0
    return max(usable_top, min(usable_bottom, base_y))


if __name__ == "__main__":
    main()
