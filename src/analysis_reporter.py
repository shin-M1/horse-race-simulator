from __future__ import annotations

from typing import Any


MARKS = ("◎", "○", "▲", "△", "☆")


def analyze_prediction_failure(
    prediction_log: dict[str, Any],
    actual_result: list[dict[str, Any]],
    evaluation_result: dict[str, Any],
) -> dict[str, Any]:
    """Explain prediction gaps with stable, aggregatable reason tags."""
    prediction_rows = list(prediction_log.get("prediction_table", []) or [])
    ranked_rows = sorted(
        prediction_rows,
        key=lambda row: (
            -_number(row.get("prediction_score", row.get("score", 0.0))),
            -_number(row.get("win_rate", 0.0)),
            -_number(row.get("top3_rate", 0.0)),
        ),
    )
    predicted_rank = {
        _integer(row.get("馬番", row.get("horse_number"))): index
        for index, row in enumerate(ranked_rows, start=1)
    }
    rows_by_number = {
        _integer(row.get("馬番", row.get("horse_number"))): row for row in prediction_rows
    }
    actual_by_number = {
        _integer(row.get("horse_number", row.get("馬番"))): _integer(row.get("finish", row.get("着順")))
        for row in actual_result
    }
    top3_rows = [rows_by_number.get(number, {}) for number, finish in actual_by_number.items() if 0 < finish <= 3]
    top3_styles = [_style(row) for row in top3_rows]
    front_count = sum(style in {"逃げ", "先行"} for style in top3_styles)
    closer_count = sum(style in {"差し", "追込"} for style in top3_styles)
    pace = prediction_log.get("pace_prediction") or prediction_log.get("simulation_result", {}).get("pace_prediction", {})
    pace_name = str(pace.get("pace", "medium")) if isinstance(pace, dict) else "medium"
    config = prediction_log.get("race_config", {}) if isinstance(prediction_log.get("race_config"), dict) else {}
    track_bias = str(config.get("track_bias", "標準"))
    condition = str(config.get("track_condition", "良"))

    tags: list[str] = []
    if pace_name == "high" and front_count >= 2:
        tags.extend(["ペース読み違い", "前残り過小評価"])
    if pace_name == "slow" and closer_count >= 2:
        tags.extend(["ペース読み違い", "差し有利過小評価"])
    if track_bias in {"差し有利", "外差し有利", "外伸び"} and front_count >= 2:
        tags.extend(["トラックバイアス読み違い", "前残り過小評価"])
    if track_bias in {"前残り", "内前有利"} and closer_count >= 2:
        tags.extend(["トラックバイアス読み違い", "差し有利過小評価"])

    for number, actual_finish in actual_by_number.items():
        row = rows_by_number.get(number, {})
        rank = predicted_rank.get(number, len(ranked_rows) + 1)
        late_kick = _number(row.get("late_kick_score"))
        ability = _number(row.get("horse_ability_score"))
        popularity = _integer(row.get("popularity", row.get("人気")))
        popularity_score = _number(row.get("popularity_score", 50))
        if rank <= 3 and actual_finish >= 8:
            tags.append("能力評価過大")
            if late_kick >= 65:
                tags.append("上り性能過大評価")
        if rank >= 8 and actual_finish <= 3:
            tags.append("能力評価過小")
            if _style(row) in {"差し", "追込"} or late_kick >= 65:
                tags.extend(["上り性能過小評価", "差し有利過小評価"])
        if actual_finish <= 3 and rank > 5 and (popularity >= 8 or popularity_score < 50):
            tags.append("人気薄好走見逃し")
        if condition in {"重", "不良"}:
            mud = _number(row.get("mud_aptitude", 50))
            if (mud >= 70 and actual_finish >= 8) or (mud <= 40 and actual_finish <= 3):
                tags.append("道悪適性評価ミス")
        weight = _number(row.get("斤量", row.get("carried_weight", 56)))
        if (weight >= 58 and rank <= 3 and actual_finish >= 8) or (weight <= 54 and rank > 5 and actual_finish <= 3):
            tags.append("斤量影響過小評価")
        jockey = _number(row.get("jockey_score", 50))
        if (jockey >= 70 and rank <= 3 and actual_finish >= 8) or (jockey <= 40 and rank > 5 and actual_finish <= 3):
            tags.append("騎手補正不足")
        primary = str(row.get("primary_running_style", ""))
        actual_style = str(row.get("actual_running_style", primary))
        if primary and actual_style and primary != "自在" and primary != actual_style:
            tags.append("脚質判定ミス")
        if actual_finish <= 3 and rank > 5 and _number(row.get("race_strength_score", 50)) < 45:
            tags.append("レースレベル補正ミス")
        if actual_finish <= 3 and rank > 5 and _number(row.get("avg_opponent_strength_score", 50)) < 45:
            tags.append("相手関係評価不足")
        if ability >= 75 and actual_finish >= 8 and rank <= 3:
            tags.append("能力評価過大")

    tags = list(dict.fromkeys(tags))
    suggestions = list(dict.fromkeys(_suggestion(tag) for tag in tags))
    return {
        "race_level_analysis": _analysis_sentence(tags, "レースレベル補正ミス", "レース強度と相手関係の補正を再確認します。", "レースレベル補正に顕著な乖離は見られません。"),
        "pace_analysis": _analysis_sentence(tags, "ペース読み違い", f"{pace_name}想定と上位馬の脚質構成にずれがありました。", "ペース想定と上位馬の脚質は概ね整合しています。"),
        "bias_analysis": _analysis_sentence(tags, "トラックバイアス読み違い", f"{track_bias}想定と実着順の傾向が一致しませんでした。", f"{track_bias}想定に大きな矛盾は見られません。"),
        "style_analysis": _analysis_sentence(tags, "脚質判定ミス", "代表脚質と採用脚質の整合を再確認します。", f"上位3頭は前方型{front_count}頭、後方型{closer_count}頭でした。"),
        "late_kick_analysis": _tag_group_sentence(tags, {"上り性能過小評価", "上り性能過大評価"}, "上り性能の評価差が結果に影響しました。", "上り性能評価に明確な過不足は見られません。"),
        "ability_analysis": _tag_group_sentence(tags, {"能力評価過大", "能力評価過小"}, "総合能力の順位付けに見直し余地があります。", "能力上位・下位の逆転は許容範囲でした。"),
        "weight_analysis": _analysis_sentence(tags, "斤量影響過小評価", "斤量差を終盤性能へ強めに反映する余地があります。", "斤量差が主因となった兆候は限定的です。"),
        "jockey_analysis": _analysis_sentence(tags, "騎手補正不足", "騎手補正と実着順の関係を再検証します。", "騎手補正に顕著な乖離は見られません。"),
        "miss_reason_tags": tags,
        "improvement_suggestions": suggestions,
    }


def generate_race_review(
    prediction_log: dict[str, Any],
    actual_result: list[dict[str, Any]],
    evaluation_result: dict[str, Any],
) -> dict[str, Any]:
    prediction_rows = list(prediction_log.get("prediction_table", []) or [])
    ranked_rows = sorted(
        prediction_rows,
        key=lambda row: (
            -_number(row.get("prediction_score", row.get("score", 0.0))),
            -_number(row.get("win_rate", 0.0)),
            -_number(row.get("top3_rate", 0.0)),
        ),
    )
    actual_by_number = {
        _integer(row.get("horse_number", row.get("馬番"))): _integer(row.get("finish", row.get("着順")))
        for row in actual_result
    }
    predicted_rank = {
        _integer(row.get("馬番", row.get("horse_number"))): index
        for index, row in enumerate(ranked_rows, start=1)
    }
    marks = {
        _integer(row.get("馬番", row.get("horse_number"))): str(row.get("印", row.get("mark", "")))
        for row in prediction_rows
    }
    failure_analysis = evaluation_result.get("failure_analysis")
    if not isinstance(failure_analysis, dict):
        failure_analysis = analyze_prediction_failure(prediction_log, actual_result, evaluation_result)

    horse_reviews: list[dict[str, Any]] = []
    for row in ranked_rows:
        horse_number = _integer(row.get("馬番", row.get("horse_number")))
        actual_finish = actual_by_number.get(horse_number)
        if not horse_number or actual_finish is None:
            continue
        horse_reviews.append(
            {
                "horse_number": horse_number,
                "horse_name": str(row.get("馬名", row.get("horse_name", ""))),
                "mark": marks.get(horse_number, ""),
                "predicted_rank": predicted_rank.get(horse_number),
                "actual_finish": actual_finish,
                "prediction_reason": _prediction_reason(row),
                "cause_candidate": _horse_cause(row, predicted_rank.get(horse_number, 0), actual_finish),
                "next_adjustment": _horse_adjustment(row, predicted_rank.get(horse_number, 0), actual_finish),
                "review": _horse_review(row, predicted_rank.get(horse_number, 0), actual_finish),
            }
        )

    miss_reasons: list[str] = []
    model_feedback: list[str] = []
    honmei_finish = evaluation_result.get("mark_finishes", {}).get("◎")
    actual_winner = next((number for number, finish in actual_by_number.items() if finish == 1), 0)
    winner_row = next(
        (row for row in prediction_rows if _integer(row.get("馬番", row.get("horse_number"))) == actual_winner),
        {},
    )
    if honmei_finish is not None and honmei_finish > 3:
        miss_reasons.append("◎が複勝圏外となり、能力・展開・適性のいずれかを過大評価した可能性があります。")
        model_feedback.append("◎の敗因に対応する特徴量を評価ログ上で比較し、上り・展開適性の重みを再検証してください。")
    if actual_winner and str(winner_row.get("印", "")) not in MARKS:
        miss_reasons.append("勝ち馬を無印としており、人気薄実績や相手関係を拾い切れていない可能性があります。")
        model_feedback.append("無印好走馬のレース強度・人気・コース適性を学習データで再評価してください。")
    if _number(evaluation_result.get("predicted_top3_top3_rate")) < 2 / 3:
        miss_reasons.append("予想上位3頭と実際の複勝圏にずれがあり、順位付けの改善余地があります。")
    else:
        model_feedback.append("上位評価馬は概ね複勝圏に入り、軸候補の抽出は妥当でした。")

    pace_prediction = prediction_log.get("pace_prediction") or prediction_log.get("simulation_result", {}).get("pace_prediction", {})
    track_bias = prediction_log.get("race_config", {}).get("track_bias") or prediction_log.get("race_metadata", {}).get("track_bias")
    if pace_prediction.get("pace") == "slow" and honmei_finish and honmei_finish > 3:
        miss_reasons.append("スローペース想定に対する前残り・末脚評価の配分が合わなかった可能性があります。")
    if track_bias and track_bias != "標準":
        model_feedback.append(f"選択したトラックバイアス「{track_bias}」と実着順の整合性を確認してください。")
    miss_reasons.extend(str(tag) for tag in failure_analysis.get("miss_reason_tags", []))
    model_feedback.extend(str(item) for item in failure_analysis.get("improvement_suggestions", []))

    hit_summary = (
        "◎が勝利し、1着評価が的中しました。"
        if evaluation_result.get("win_hit")
        else "◎は複勝圏を確保し、軸評価は概ね妥当でした。"
        if evaluation_result.get("top3_hit")
        else "◎は複勝圏外で、中心馬評価の見直しが必要です。"
    )
    race_name = prediction_log.get("race_name", "")
    return {
        "race_summary": f"{race_name or '対象レース'}の予想と実結果を比較しました。",
        "hit_summary": hit_summary,
        "miss_reason_candidates": miss_reasons,
        "horse_reviews": horse_reviews,
        "model_feedback": model_feedback,
        "failure_analysis": failure_analysis,
    }


def _prediction_reason(row: dict[str, Any]) -> str:
    candidates = [
        ("総合能力", _number(row.get("horse_ability_score"))),
        ("上り性能", _number(row.get("late_kick_score"))),
        ("コース適性", _number(row.get("course_fit_score"))),
        ("展開適性", _number(row.get("pace_fit_score"))),
    ]
    strongest = sorted(candidates, key=lambda item: item[1], reverse=True)[:2]
    return "・".join(name for name, score in strongest if score > 0) or "総合予想スコア"


def _horse_cause(row: dict[str, Any], predicted_rank: int, actual_finish: int) -> str:
    if predicted_rank <= 3 and actual_finish >= 8:
        return "上位評価に対して大敗しており、能力・展開適性を過大評価した可能性があります。"
    if predicted_rank >= 8 and actual_finish <= 3:
        return "低評価から好走しており、末脚・人気薄実績を拾い切れなかった可能性があります。"
    if actual_finish <= 3:
        return "評価した能力と適性を実戦で発揮しました。"
    return "展開、位置取り、馬場適性のいずれかが想定を下回った可能性があります。"


def _horse_adjustment(row: dict[str, Any], predicted_rank: int, actual_finish: int) -> str:
    if predicted_rank <= 3 and actual_finish >= 8:
        return "同型馬の数、ペース耐性、斤量を次回は強めに確認します。"
    if predicted_rank >= 8 and actual_finish <= 3:
        return "上り性能、相手関係、人気薄好走歴の重みを再検討します。"
    return "同条件での再現性を次走データで確認します。"


def _suggestion(tag: str) -> str:
    mapping = {
        "ペース読み違い": "early_push_scoreと逃げ・先行確率からペース閾値を再調整してください。",
        "前残り過小評価": "前方脚質のfade_resistanceとトラックバイアス補正を再確認してください。",
        "差し有利過小評価": "late_kick_scoreと相対上り評価の重みを上げる余地があります。",
        "トラックバイアス読み違い": "開催日数・使用コース・当日バイアスの整合を確認してください。",
        "上り性能過小評価": "late_kick_scoreの重みを上げる余地があります。",
        "上り性能過大評価": "上りタイムを馬場・位置取り・レースレベルで補正してください。",
        "能力評価過大": "高評価馬の凡走条件をhorse_ability_scoreへ負例として反映してください。",
        "能力評価過小": "低評価好走馬のレース強度と相手関係を再評価してください。",
        "人気薄好走見逃し": "人気薄の好走歴とクラス上位実績を補助特徴量として確認してください。",
        "道悪適性評価ミス": "道悪実績を血統補完より優先して再集計してください。",
        "斤量影響過小評価": "距離と脚質に応じた斤量補正を再検証してください。",
        "騎手補正不足": "騎手補正は小幅のまま、コース別実績を追加検証してください。",
        "脚質判定ミス": "通過順比率と出走頭数の取得精度を確認してください。",
        "レースレベル補正ミス": "race_strength_scoreとELOの寄与を再検証してください。",
        "相手関係評価不足": "opponent_strength_scoreの欠損率と補完値を確認してください。",
    }
    return mapping.get(tag, f"{tag}に関連する特徴量を再確認してください。")


def _analysis_sentence(tags: list[str], target: str, hit: str, miss: str) -> str:
    return hit if target in tags else miss


def _tag_group_sentence(tags: list[str], targets: set[str], hit: str, miss: str) -> str:
    return hit if targets.intersection(tags) else miss


def _style(row: dict[str, Any]) -> str:
    return str(row.get("actual_running_style", row.get("primary_running_style", row.get("脚質", ""))))


def _horse_review(row: dict[str, Any], predicted_rank: int, actual_finish: int) -> str:
    name = str(row.get("馬名", row.get("horse_name", "対象馬")))
    if actual_finish <= 3 and predicted_rank <= 3:
        return f"{name}は予想{predicted_rank}位から{actual_finish}着。上位評価は概ね妥当でした。"
    if actual_finish <= 3 < predicted_rank:
        if _number(row.get("late_kick_score")) >= 65:
            return f"{name}は{actual_finish}着。高い上り性能を順位へ十分反映できませんでした。"
        return f"{name}は{actual_finish}着。能力または適性を過小評価した可能性があります。"
    if predicted_rank <= 3 and actual_finish > 3:
        strongest = max(
            (
                ("能力評価", _number(row.get("horse_ability_score"))),
                ("上り性能", _number(row.get("late_kick_score"))),
                ("展開適性", _number(row.get("pace_fit_score"))),
                ("コース適性", _number(row.get("course_fit_score"))),
            ),
            key=lambda item: item[1],
        )[0]
        return f"{name}は予想{predicted_rank}位から{actual_finish}着。{strongest}を過大評価した可能性があります。"
    return f"{name}は予想{predicted_rank}位、実着順{actual_finish}着でした。"


def _integer(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
