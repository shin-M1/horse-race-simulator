from __future__ import annotations

import pandas as pd

from course_db import get_course_bias
from horse_analyzer import RACE_LEVEL_WEIGHT


RECENT_RACE_COLUMNS = [
    "馬名",
    "レース名",
    "クラス",
    "人気順",
    "着順",
    "着差",
    "通過順",
    "上り",
    "タイム",
    "time_sec",
    "avg_speed",
    "late_gain",
    "recent_time_score",
    "レースレベル重み",
    "着差補正",
    "相手関係補正",
    "レース評価スコア",
    "日付",
    "競馬場",
    "距離",
    "馬場状態",
]


def extract_value(row: dict, candidates: list[str], default: str = "") -> str:
    """
    複数の候補キーから値を取得する。
    最初に見つかった非空値を返す。
    """
    row_dict = _as_row_dict(row)
    for key in candidates:
        value = row_dict.get(key)
        if not _is_empty(value):
            return _format_value(value)
    return default


def build_recent_races_table(horse_results: list[dict]) -> pd.DataFrame:
    """
    各馬の近走5走分のレース名、クラス、人気順、着順、通過順、上りをまとめたDataFrameを作成する。
    """
    rows: list[dict[str, str]] = []
    for horse_result in horse_results:
        horse_row = _as_row_dict(horse_result)
        horse_name = extract_value(horse_row, ["horse_name", "馬名"], "-")
        races = _extract_races(horse_row)
        for race in races[:5]:
            race_row = _as_row_dict(race)
            race_name = extract_value(race_row, ["race_name", "name", "race", "レース名"], "-")
            race_class = extract_value(race_row, ["race_class", "class", "grade", "クラス"], "-")
            rows.append(
                {
                    "馬名": horse_name,
                    "レース名": race_name,
                    "クラス": race_class,
                    "人気順": extract_value(race_row, ["popularity", "odds_rank", "favorite", "人気", "人気順"], "-"),
                    "着順": extract_value(race_row, ["finish", "finish_position", "rank", "result", "着順"], "-"),
                    "着差": extract_value(race_row, ["winner_time_diff", "margin_sec", "margin", "着差"], "-"),
                    "通過順": extract_value(race_row, ["passing_order", "passing", "corner_order", "通過", "通過順"], "-"),
                    "上り": extract_value(race_row, ["last3f", "final_3f", "agari", "closing_3f", "上り", "上がり", "上がり3F"], "-"),
                    "タイム": extract_value(race_row, ["time", "race_time", "result_time", "race_time_seconds", "走破タイム", "タイム"], "-"),
                    "time_sec": extract_value(race_row, ["time_sec", "race_time_seconds"], "-"),
                    "avg_speed": _score_text(race_row, ["avg_speed"]),
                    "late_gain": _score_text(race_row, ["late_gain"]),
                    "recent_time_score": _score_text(race_row, ["recent_time_score"]),
                    "レースレベル重み": _race_level_weight_text(race_row, race_name, race_class),
                    "着差補正": _score_text(race_row, ["margin_score", "着差補正"]),
                    "相手関係補正": _score_text(race_row, ["opponent_strength_score", "opponent_score", "相手関係補正"]),
                    "レース評価スコア": _score_text(race_row, ["race_score", "レース評価スコア"]),
                    "日付": extract_value(race_row, ["date", "日付"], "-"),
                    "競馬場": extract_value(race_row, ["course", "venue", "place", "競馬場"], "-"),
                    "距離": extract_value(race_row, ["distance", "距離"], "-"),
                    "馬場状態": extract_value(race_row, ["track_condition", "condition", "馬場状態", "馬場"], "-"),
                }
            )
    return pd.DataFrame(rows, columns=RECENT_RACE_COLUMNS)


def build_single_race_result_from_timeline(
    race_timeline: list[dict],
    horses: list[dict],
) -> pd.DataFrame:
    """Build the one-run video result table from the final timeline frame."""
    columns = ["着順", "馬番", "馬名", "枠順", "斤量", "actual_running_style", "position_m", "gap_from_winner"]
    if not race_timeline:
        return pd.DataFrame(columns=columns)

    final_frame = race_timeline[-1]
    final_horses = final_frame.get("horses", []) if isinstance(final_frame, dict) else []
    if not isinstance(final_horses, list) or not final_horses:
        return pd.DataFrame(columns=columns)

    metadata = _horse_metadata_by_number(horses)
    rows: list[dict[str, object]] = []
    sorted_horses = sorted(
        [horse for horse in final_horses if isinstance(horse, dict)],
        key=lambda horse: (-_to_float(horse.get("position_m", 0.0)), int(horse.get("rank", 999) or 999)),
    )
    winner_position = _to_float(sorted_horses[0].get("position_m", 0.0)) if sorted_horses else 0.0
    for rank, horse in enumerate(sorted_horses, start=1):
        horse_number = _to_int(horse.get("horse_number", 0))
        meta = metadata.get(horse_number, {})
        position_m = _to_float(horse.get("position_m", 0.0))
        rows.append(
            {
                "着順": rank,
                "馬番": horse_number,
                "馬名": str(horse.get("horse_name") or meta.get("horse_name", "")),
                "枠順": _to_int(horse.get("frame", meta.get("frame", 0))),
                "斤量": round(_to_float(horse.get("carried_weight", meta.get("carried_weight", 56.0))), 1),
                "actual_running_style": str(horse.get("actual_running_style_fixed", horse.get("actual_running_style", ""))),
                "position_m": round(position_m, 3),
                "gap_from_winner": round(max(0.0, winner_position - position_m), 3),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _horse_metadata_by_number(horses: list[dict]) -> dict[int, dict[str, object]]:
    metadata: dict[int, dict[str, object]] = {}
    for horse in horses:
        horse_row = _as_row_dict(horse)
        number = _to_int(horse_row.get("horse_number", horse_row.get("馬番", 0)))
        if number <= 0:
            continue
        metadata[number] = {
            "horse_name": horse_row.get("horse_name", horse_row.get("馬名", "")),
            "frame": horse_row.get("frame", horse_row.get("枠順", "")),
            "carried_weight": horse_row.get("carried_weight", horse_row.get("斤量", 56.0)),
        }
    return metadata


def _race_level_weight_text(race_row: dict[str, object], race_name: str, race_class: str) -> str:
    value = extract_value(race_row, ["race_level_weight", "level_weight", "レースレベル重み"], "")
    if value:
        try:
            return f"{float(value):.2f}"
        except ValueError:
            return value
    text = f"{race_class} {race_name}".upper()
    for level in ["G1", "G2", "G3", "L", "OP", "3勝", "2勝", "1勝", "未勝利", "新馬"]:
        if level in text:
            return f"{RACE_LEVEL_WEIGHT[level]:.2f}"
    return "-"


def _score_text(race_row: dict[str, object], candidates: list[str]) -> str:
    value = extract_value(race_row, candidates, "")
    if value:
        try:
            return f"{float(value):.1f}"
        except ValueError:
            return value
    return "-"


def _extract_races(horse_row: dict[str, object]) -> list[object]:
    for key in ["recent_races", "recent_results", "races", "results"]:
        races = horse_row.get(key)
        if _is_empty(races):
            continue
        if hasattr(races, "to_dict"):
            return list(races.to_dict("records"))
        return list(races)
    return []


def _as_row_dict(row: object) -> dict[str, object]:
    if isinstance(row, dict):
        return dict(row)
    if hasattr(row, "to_dict"):
        try:
            return dict(row.to_dict())
        except TypeError:
            pass
    if hasattr(row, "raw") and getattr(row, "raw"):
        raw = dict(getattr(row, "raw"))
    else:
        raw = {}
    if hasattr(row, "__dict__"):
        raw.update(dict(getattr(row, "__dict__")))
        return raw
    return {}


def _is_empty(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _format_value(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return "-".join(str(item) for item in value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def add_ranking_columns(ranking: pd.DataFrame, horse_analysis: pd.DataFrame) -> pd.DataFrame:
    analysis = horse_analysis.copy()
    if "race_power" in analysis.columns:
        analysis["overall_power"] = analysis["race_power"]
    else:
        analysis["overall_power"] = analysis[
            ["early_speed", "stamina", "acceleration", "mud_aptitude", "consistency"]
        ].mean(axis=1)
    merged = ranking.merge(
        analysis[["horse_name", "overall_power"]],
        on="horse_name",
        how="left",
    )
    winner_time = float(merged["finish_time"].min())
    merged["margin"] = merged["finish_time"] - winner_time
    return merged.rename(
        columns={
            "rank": "着順",
            "horse_number": "馬番",
            "horse_name": "馬名",
            "frame": "枠順",
            "primary_running_style": "代表脚質",
            "actual_running_style": "採用脚質",
            "overall_power": "総合能力",
            "finish_time": "ゴールタイム",
            "margin": "着差",
        }
    )[
        ["着順", "馬番", "馬名", "枠順", "代表脚質", "採用脚質", "総合能力", "ゴールタイム", "着差"]
    ].round({"総合能力": 2, "ゴールタイム": 3, "着差": 3})


def format_analysis_table(horse_analysis: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "horse_name",
        "base_style_profile",
        "adjusted_style_profile",
        "primary_running_style",
        "actual_running_style",
        "carried_weight",
        "weight_penalty",
        "base_逃げ",
        "base_先行",
        "base_差し",
        "base_追込",
        "adjusted_逃げ",
        "adjusted_先行",
        "adjusted_差し",
        "adjusted_追込",
        "style_sample_size",
        "debug_field_sizes",
        "weighted_avg_first_ratio",
        "weighted_avg_mid_ratio",
        "weighted_avg_last_corner_ratio",
        "weighted_avg_late_gain",
        "position_variance",
        "early_speed",
        "stamina",
        "acceleration",
        "mud_aptitude",
        "mud_source",
        "consistency",
        "avg_opponent_strength_score",
        "avg_race_score",
        "race_level_score",
        "finish_score",
        "margin_score",
        "time_score",
        "last3f_score",
        "horse_ability_score",
        "race_power",
        "race_strength_score",
        "race_strength_adjusted_score",
        "elo_rating",
        "normalized_elo_score",
        "relative_agari_score",
        "course_fit_score",
        "jockey",
        "jockey_score",
        "track_bias_fit_score",
        "pace_fit_score",
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
        "early_aggressiveness",
        "mid_positioning",
        "late_kick_timing",
        "sustain_speed",
        "time_reliability",
        "recent_time_score",
        "late_kick_score",
        "avg_last3f",
        "best_last3f",
        "last3f_consistency",
        "late_gain_score",
        "early_push_score",
        "mid_cruise_score",
        "fade_resistance_score",
        "sustain_speed_score",
        "pace_resilience_score",
        "agari_reliability",
    ]
    existing_columns = [column for column in columns if column in horse_analysis.columns]
    return horse_analysis[existing_columns].rename(
        columns={
            "horse_name": "馬名",
            "primary_running_style": "代表脚質",
            "actual_running_style": "採用脚質",
            "base_style_profile": "base_style_profile",
            "adjusted_style_profile": "adjusted_style_profile",
            "carried_weight": "斤量",
            "weight_penalty": "weight_penalty",
            "style_sample_size": "脚質判定走数",
            "debug_field_sizes": "field_size一覧",
            "weighted_avg_first_ratio": "first_ratio平均",
            "weighted_avg_mid_ratio": "mid_ratio平均",
            "weighted_avg_last_corner_ratio": "last_corner_ratio平均",
            "weighted_avg_late_gain": "late_gain平均",
            "mud_source": "mud_source",
            "avg_opponent_strength_score": "平均相手関係補正",
            "avg_race_score": "平均レース評価",
            "race_level_score": "レースレベルスコア",
            "finish_score": "着順スコア",
            "margin_score": "着差スコア",
            "time_score": "タイムスコア",
            "last3f_score": "上りスコア",
            "horse_ability_score": "horse_ability_score",
            "race_power": "race_power",
            "race_strength_score": "race_strength_score",
            "race_strength_adjusted_score": "race_strength_adjusted_score",
            "elo_rating": "elo_rating",
            "normalized_elo_score": "normalized_elo_score",
            "relative_agari_score": "relative_agari_score",
            "course_fit_score": "course_fit_score",
            "jockey": "騎手",
            "jockey_score": "jockey_score",
            "track_bias_fit_score": "track_bias_fit_score",
            "pace_fit_score": "pace_fit_score",
            "race_trend_score": "race_trend_score",
            "frame_trend_score": "frame_trend_score",
            "horse_number_trend_score": "horse_number_trend_score",
            "style_trend_score": "style_trend_score",
            "agari_trend_score": "agari_trend_score",
            "fourth_corner_trend_score": "fourth_corner_trend_score",
            "age_trend_score": "age_trend_score",
            "weight_trend_score": "weight_trend_score",
            "jockey_continuity_score": "jockey_continuity_score",
            "previous_race_trend_score": "previous_race_trend_score",
            "bloodline_trend_score": "bloodline_trend_score",
            "trend_match_comment": "trend_match_comment",
        }
    ).round(3)


def style_probability_long_table(horse_analysis: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in horse_analysis.iterrows():
        horse_name = row["horse_name"]
        for style in ["逃げ", "先行", "差し", "追込"]:
            rows.append({"馬名": horse_name, "脚質": f"base_{style}", "確率": float(row.get(f"base_{style}", 0.0))})
            rows.append({"馬名": horse_name, "脚質": f"adjusted_{style}", "確率": float(row.get(f"adjusted_{style}", 0.0))})
    return pd.DataFrame(rows)


def pace_comment(pace_prediction: dict[str, object]) -> str:
    pace = str(pace_prediction.get("pace", "medium"))
    front_pressure = float(pace_prediction.get("front_pressure", 0.0))
    if pace == "high":
        return f"前に行く確率の高い馬が多く、ハイペース想定です。front_pressure={front_pressure:.2f}"
    if pace == "slow":
        return f"前に行く確率の高い馬が少なく、スローペース想定です。front_pressure={front_pressure:.2f}"
    return f"前後の圧力が拮抗した平均ペース想定です。front_pressure={front_pressure:.2f}"


def style_group_table(pace_prediction: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "区分": "先頭候補",
                "該当馬": "、".join(_as_str_list(pace_prediction.get("front_group", []))) or "-",
            },
            {
                "区分": "中団候補",
                "該当馬": "、".join(_as_str_list(pace_prediction.get("middle_group", []))) or "-",
            },
            {
                "区分": "後方候補",
                "該当馬": "、".join(_as_str_list(pace_prediction.get("back_group", []))) or "-",
            },
        ]
    )


def build_horse_comments_table(
    prediction_table: pd.DataFrame,
    horse_analysis: pd.DataFrame,
    pace_prediction: dict[str, object],
    race_config: dict[str, object],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if prediction_table.empty:
        return pd.DataFrame(columns=["印", "馬番", "馬名", "斤量", "脚質", "勝率", "複勝率", "評価", "短評"])

    analysis_by_name = {}
    if isinstance(horse_analysis, pd.DataFrame) and not horse_analysis.empty and "horse_name" in horse_analysis.columns:
        analysis_by_name = {str(row["horse_name"]): row for _, row in horse_analysis.iterrows()}

    for _, row in prediction_table.iterrows():
        horse_name = str(row.get("馬名", ""))
        analysis = analysis_by_name.get(horse_name)
        win_rate = _to_float(row.get("win_rate", 0.0))
        top3_rate = _to_float(row.get("top3_rate", 0.0))
        style = str(row.get("primary_running_style", ""))
        if analysis is not None:
            style = str(analysis.get("primary_running_style", style))
        comment_eval_score = _comment_eval_score(row, analysis, pace_prediction)
        carried_weight = _to_float(row.get("斤量", analysis.get("carried_weight", 56.0) if analysis is not None else 56.0), 56.0)
        comment = _append_weight_comment(
            _build_horse_comment(row, analysis, pace_prediction, race_config),
            carried_weight,
        )
        rows.append(
            {
                "印": str(row.get("印", "")),
                "馬番": row.get("馬番", ""),
                "馬名": horse_name,
                "斤量": round(carried_weight, 1),
                "脚質": style,
                "勝率": round(win_rate, 4),
                "複勝率": round(top3_rate, 4),
                "評価": "",
                "短評": _add_trend_comment(comment, row),
                "_comment_eval_score": comment_eval_score,
            }
        )

    _assign_relative_comment_grades(rows)
    return pd.DataFrame(rows, columns=["印", "馬番", "馬名", "斤量", "脚質", "勝率", "複勝率", "評価", "短評"])


def _add_trend_comment(comment: str, prediction_row: pd.Series) -> str:
    trend_comment = str(prediction_row.get("trend_match_comment", "")).strip()
    trend_score = _to_float(prediction_row.get("race_trend_score", 50.0), 50.0)
    if trend_comment and trend_comment != "過去傾向は中立評価":
        return f"{comment} 過去傾向では{trend_comment}。"
    if trend_score >= 62:
        return f"{comment} 過去10年傾向との相性も評価できる。"
    if trend_score <= 42:
        return f"{comment} 過去10年傾向とはやや噛み合わない点に注意。"
    return comment


def _build_horse_comment(
    prediction_row: pd.Series,
    analysis_row: pd.Series | None,
    pace_prediction: dict[str, object],
    race_config: dict[str, object],
) -> str:
    horse_name = str(prediction_row.get("馬名", ""))
    style = str(prediction_row.get("primary_running_style", ""))
    frame = _to_int(prediction_row.get("枠順", 0))
    if analysis_row is not None:
        style = str(analysis_row.get("primary_running_style", style))
        frame = _to_int(analysis_row.get("frame", frame))

    win_rate = _to_float(prediction_row.get("win_rate", 0.0))
    top3_rate = _to_float(prediction_row.get("top3_rate", 0.0))
    prediction_score = _prediction_score(prediction_row)
    pace = str(pace_prediction.get("pace", "medium"))
    track_condition = str(_config_get(race_config, "track_condition", "良"))
    distance = _to_int(_config_get(race_config, "distance", 0))
    day = str(_config_get(race_config, "race_course_day", ""))
    layout = str(_config_get(race_config, "course_layout", ""))
    course_bias = get_course_bias(race_config)
    track_bias = str(_config_get(race_config, "track_bias", "標準"))

    adjusted_profile = _adjusted_profile(analysis_row)
    front_prob = adjusted_profile.get("逃げ", 0.0) + adjusted_profile.get("先行", 0.0)
    closer_prob = adjusted_profile.get("差し", 0.0) + adjusted_profile.get("追込", 0.0)
    opponent_score = _to_float(analysis_row.get("avg_opponent_strength_score", 0.0)) if analysis_row is not None else 0.0
    race_power = _race_power_from_rows(prediction_row, analysis_row)
    if win_rate >= 0.20 and race_power >= 70:
        if opponent_score >= 75:
            return "強い相手に僅差の内容があり、勝率とレース内能力からも勝ち切り候補として評価できる。"
        return f"勝率とレース内能力が高く、{day}{layout}コース想定でも勝ち切り候補として評価できる。"
    if opponent_score >= 75 and race_power >= 65:
        return "強い相手に僅差の内容があり、クラス上位でも通用する下地はある。"
    if opponent_score <= 55 and race_power >= 60 and top3_rate < 0.45:
        return "相手関係に恵まれた好走が多く、今回のメンバーレベルでは過信禁物。"
    if track_bias in {"外差し有利", "外伸び"} and style in {"差し", "追込"}:
        return f"{track_bias}の馬場なら、相対的な上り性能と外を伸びる形を活かして浮上可能。"
    if track_bias in {"前残り", "内前有利"} and style in {"逃げ", "先行"}:
        return f"{track_bias}想定では先行力を活かしやすく、直線まで余力を残せれば粘り込み可能。"
    if prediction_score >= 65 and top3_rate >= 0.45 and win_rate < 0.15:
        return "予想順位は上位だが勝率は抜けておらず、相手候補として堅実に評価したい。"
    if top3_rate >= 0.45 and win_rate < 0.12:
        return "複勝率は高く安定感はあるが、勝ち切るには展開の助けが必要。"
    if style in {"逃げ", "先行"} or front_prob >= closer_prob:
        if pace == "slow" or float(course_bias.get("front_bias", 0.0)) > 0:
            frame_note = "内枠と" if frame <= 3 else ""
            return f"先行力が高く、{frame_note}{pace}ペース想定を活かせれば粘り込み可能。"
        return f"前で運べる強みはあるが、{pace}ペースでは終盤の踏ん張りが鍵になる。"
    if style in {"差し", "追込"} or closer_prob > front_prob:
        if pace == "slow":
            return "差し脚は安定しているが、スロー想定では届き切らないリスクがある。"
        if float(course_bias.get("closer_bias", 0.0)) > 0 or frame >= 7:
            return f"外差しバイアスと{track_condition}馬場への対応次第で、終盤に浮上できる。"
        return f"{distance}mで末脚を活かせれば上位圏だが、位置取り次第で評価が揺れる。"
    return f"{horse_name}は展開待ちの面があり、馬場とペースの噛み合いが上位進出の条件。"


def _append_weight_comment(comment: str, carried_weight: float) -> str:
    if carried_weight >= 58.0:
        return f"{comment} 斤量負荷をこなせるかが鍵。"
    if carried_weight <= 54.0:
        return f"{comment} 軽斤量を活かせる。"
    return comment


def _comment_eval_score(
    prediction_row: pd.Series,
    analysis_row: pd.Series | None,
    pace_prediction: dict[str, object],
) -> float:
    win_rate = _to_float(prediction_row.get("win_rate", 0.0))
    top3_rate = _to_float(prediction_row.get("top3_rate", 0.0))
    consistency = _to_float(analysis_row.get("consistency", 50.0)) / 100.0 if analysis_row is not None else 0.50
    race_power_score = _race_power_from_rows(prediction_row, analysis_row) / 100.0
    pace_fit_score = _pace_fit_score(analysis_row, pace_prediction)
    return (
        win_rate * 0.30
        + top3_rate * 0.30
        + consistency * 0.15
        + race_power_score * 0.15
        + pace_fit_score * 0.10
    )


def _assign_relative_comment_grades(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    order = sorted(
        range(len(rows)),
        key=lambda index: float(rows[index].get("_comment_eval_score", 0.0)),
        reverse=True,
    )
    denominator = max(1, len(rows) - 1)
    for rank_index, row_index in enumerate(order):
        percentile = rank_index / denominator
        if percentile <= 0.15:
            grade = "A"
        elif percentile <= 0.35:
            grade = "B"
        elif percentile <= 0.60:
            grade = "C"
        elif percentile <= 0.80:
            grade = "D"
        else:
            grade = "E"
        rows[row_index]["評価"] = grade


def _pace_fit_score(analysis_row: pd.Series | None, pace_prediction: dict[str, object]) -> float:
    if analysis_row is None:
        return 0.50
    adjusted_profile = _adjusted_profile(analysis_row)
    pace = str(pace_prediction.get("pace", "medium"))
    front_prob = adjusted_profile.get("逃げ", 0.0) + adjusted_profile.get("先行", 0.0)
    closer_prob = adjusted_profile.get("差し", 0.0) + adjusted_profile.get("追込", 0.0)
    if pace == "slow":
        return max(0.0, min(1.0, front_prob))
    if pace == "high":
        return max(0.0, min(1.0, closer_prob))
    return max(0.0, min(1.0, 1.0 - abs(front_prob - closer_prob)))


def _race_power_from_rows(prediction_row: pd.Series, analysis_row: pd.Series | None) -> float:
    if analysis_row is not None:
        race_power = _to_float(analysis_row.get("race_power", 0.0))
        if race_power > 0:
            return race_power
        race_score = _to_float(analysis_row.get("avg_race_score", 0.0))
        if race_score > 0:
            return race_score
    return _to_float(prediction_row.get("race_power", 70.0)) or 70.0


def _prediction_score(prediction_row: pd.Series) -> float:
    score = _to_float(prediction_row.get("prediction_score", 0.0))
    if score > 0:
        return score
    return _to_float(prediction_row.get("score", 0.0))


def _adjusted_profile(analysis_row: pd.Series | None) -> dict[str, float]:
    if analysis_row is None:
        return {"逃げ": 0.0, "先行": 0.0, "差し": 0.0, "追込": 0.0}
    return {
        "逃げ": _to_float(analysis_row.get("adjusted_逃げ", 0.0)),
        "先行": _to_float(analysis_row.get("adjusted_先行", 0.0)),
        "差し": _to_float(analysis_row.get("adjusted_差し", 0.0)),
        "追込": _to_float(analysis_row.get("adjusted_追込", 0.0)),
    }


def _as_str_list(value: object) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    if value in (None, ""):
        return []
    return [str(value)]


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: object) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _config_get(race_config: dict[str, object] | object, key: str, default: object) -> object:
    if isinstance(race_config, dict):
        return race_config.get(key, default)
    return getattr(race_config, key, default)
