from __future__ import annotations

import argparse
import importlib
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from errors import RaceDataFetchError
from horse_analyzer import (
    HorseAnalyzer,
    RaceResult,
    RaceResultProvider,
    parse_passing_order,
    parse_race_time,
    race_result_to_recent_dict,
)
from horse_database import get_or_fetch_horse_profile
from pace_predictor import PacePredictor
from race_config import RaceConfig, HorseEntry, demo_horses, prompt_horses, prompt_race_config
from simulator import RaceSimulator
from data_fetcher import NetkeibaRaceResultProvider


def load_provider(
    module_name: str | None,
    factory_name: str | None,
    use_local_database: bool = True,
    force_refresh_data: bool = False,
) -> RaceResultProvider:
    """Load an existing recent-result fetcher, or use strict netkeiba fetching."""
    if not module_name:
        return _wrap_provider(NetkeibaRaceResultProvider(), use_local_database, force_refresh_data)
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name or "get_provider", None)
    if factory is None:
        provider = getattr(module, "provider", None)
        if provider is None:
            raise AttributeError(f"{module_name} must expose get_provider() or provider")
        return _wrap_provider(provider, use_local_database, force_refresh_data)
    return _wrap_provider(factory(), use_local_database, force_refresh_data)


def _wrap_provider(provider: Any, use_local_database: bool = False, force_refresh_data: bool = False) -> RaceResultProvider:
    """Accept providers returning RaceResult objects, dicts, or DataFrames."""

    class ProviderAdapter:
        def __init__(self, inner_provider: Any) -> None:
            self.inner_provider = inner_provider
            self.fetch_debug: list[dict[str, Any]] = []
            self._codex_provider_adapter = True
            self.use_local_database = bool(use_local_database)
            self.force_refresh_data = bool(force_refresh_data)
            self.fetch_count = 0

        def get_recent_results(self, horse_name: str, limit: int = 5) -> list[RaceResult]:
            raw: Any = []
            profile: dict[str, Any] | None = None
            try:
                if self.use_local_database:
                    profile = get_or_fetch_horse_profile(
                        horse_name,
                        lambda: self._fetch_profile(horse_name, limit),
                        force_refresh=self.force_refresh_data,
                    )
                    raw = (profile or {}).get("recent_races", [])
                else:
                    raw = self.inner_provider.get_recent_results(horse_name, limit=limit)
                    self.fetch_count += 1
                records = _raw_to_records(raw)
                normalized = [_to_race_result(item) for item in records[:limit]]
            except RaceDataFetchError as exc:
                debug_record = _build_fetch_debug_record(self.inner_provider, horse_name, raw, [])
                self.fetch_debug.append(debug_record)
                exc.debug_records = self.get_fetch_debug()
                raise
            except Exception as exc:
                debug_record = _build_fetch_debug_record(self.inner_provider, horse_name, raw, [])
                self.fetch_debug.append(debug_record)
                raise RaceDataFetchError(
                    f"{horse_name} の近走データ取得に失敗しました: {exc}",
                    horse_name=horse_name,
                    debug_records=self.get_fetch_debug(),
                ) from exc

            debug_record = _build_fetch_debug_record(self.inner_provider, horse_name, raw, normalized)
            if profile is not None:
                debug_record["horse_database"] = profile.get("_database_status", "")
                debug_record["horse_database_path"] = profile.get("_database_path", "")
                debug_record["fetch_count"] = self.fetch_count
            self.fetch_debug.append(debug_record)
            if not normalized:
                raise RaceDataFetchError(
                    f"{horse_name} の近走データが取得できませんでした。馬名または取得処理を確認してください。",
                    horse_name=horse_name,
                    debug_records=self.get_fetch_debug(),
                )
            fetched_name = str(debug_record.get("search_result_horse_name", "") or "")
            if fetched_name and not _horse_name_match(horse_name, fetched_name):
                raise RaceDataFetchError(
                    f"{horse_name} と取得ページ上の馬名（{fetched_name}）が一致しないため、シミュレーションを停止しました。",
                    horse_name=horse_name,
                    debug_records=self.get_fetch_debug(),
                )
            return normalized

        def get_fetch_debug(self) -> list[dict[str, Any]]:
            return list(self.fetch_debug)

        def _fetch_profile(self, horse_name: str, limit: int) -> dict[str, Any] | None:
            raw = self.inner_provider.get_recent_results(horse_name, limit=limit)
            self.fetch_count += 1
            records: list[dict[str, Any]] = []
            for item in _raw_to_records(raw):
                if isinstance(item, RaceResult):
                    records.append(race_result_to_recent_dict(item))
                elif isinstance(item, dict):
                    records.append(dict(item))
            if not records:
                return None
            return {
                "horse_name": horse_name,
                "recent_races": records,
                "fetch_debug": _provider_debug_info(self.inner_provider, horse_name),
            }

        def get_pedigree_info(self, horse_name: str) -> dict[str, Any] | None:
            if not hasattr(self.inner_provider, "get_pedigree_info"):
                return None
            value = self.inner_provider.get_pedigree_info(horse_name)
            return value if isinstance(value, dict) else None

        def get_jockey_score(self, jockey_name: str) -> float | None:
            if not hasattr(self.inner_provider, "get_jockey_score"):
                return None
            value = self.inner_provider.get_jockey_score(jockey_name)
            return float(value) if value not in (None, "") else None

    return ProviderAdapter(provider)


def _raw_to_records(raw: Any) -> list[Any]:
    if hasattr(raw, "to_dict"):
        try:
            return list(raw.to_dict("records"))
        except TypeError:
            return [raw.to_dict()]
    if raw is None:
        return []
    if isinstance(raw, dict):
        return [raw]
    return list(raw)


def _build_fetch_debug_record(
    provider: Any,
    input_horse_name: str,
    raw: Any,
    normalized_results: list[RaceResult],
) -> dict[str, Any]:
    provider_debug = _provider_debug_info(provider, input_horse_name)
    search_result_horse_name = _pick(
        provider_debug,
        ["search_result_horse_name", "fetched_horse_name", "page_horse_name", "matched_horse_name", "horse_name_on_page", "取得馬名", "ページ馬名"],
        "",
    )
    horse_id = _pick(provider_debug, ["horse_id", "id", "netkeiba_horse_id"], "")
    url = _pick(provider_debug, ["url", "horse_url", "page_url", "取得URL"], "")
    raw_race_df = _pick(
        provider_debug,
        ["raw_race_df", "race_df", "raw_df", "read_html_df", "read_html_result", "tables", "scraped_df"],
        raw,
    )
    if not url and horse_id:
        url = f"https://db.netkeiba.com/horse/{horse_id}/"

    recent_races = [race_result_to_recent_dict(result) for result in normalized_results[:5]]
    name_match = _horse_name_match(input_horse_name, str(search_result_horse_name)) if search_result_horse_name else None
    return {
        "input_horse_name": input_horse_name,
        "search_result_horse_name": search_result_horse_name,
        "horse_id": horse_id,
        "url": url,
        "raw_race_df": raw_race_df,
        "recent_races": recent_races,
        "name_match": name_match,
    }


def _provider_debug_info(provider: Any, horse_name: str) -> dict[str, Any]:
    for method_name in ["get_fetch_debug", "get_debug_info", "get_last_debug"]:
        method = getattr(provider, method_name, None)
        if not callable(method):
            continue
        for args in [(horse_name,), ()]:
            try:
                return _select_debug_record(method(*args), horse_name)
            except TypeError:
                continue
    for attr_name in ["last_debug", "debug_info", "fetch_debug", "debug_records", "last_fetch_debug", "last_lookup"]:
        if hasattr(provider, attr_name):
            return _select_debug_record(getattr(provider, attr_name), horse_name)
    return {}


def _select_debug_record(value: Any, horse_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        if horse_name in value and isinstance(value[horse_name], dict):
            return dict(value[horse_name])
        normalized_horse_name = _normalize_horse_name(horse_name)
        for key, item in value.items():
            if _normalize_horse_name(str(key)) == normalized_horse_name and isinstance(item, dict):
                return dict(item)
        return dict(value)
    if isinstance(value, list):
        normalized_horse_name = _normalize_horse_name(horse_name)
        dict_items = [item for item in value if isinstance(item, dict)]
        for item in reversed(dict_items):
            candidate = _pick(item, ["input_horse_name", "horse_name", "馬名"], "")
            if candidate and _normalize_horse_name(str(candidate)) == normalized_horse_name:
                return dict(item)
        return dict(dict_items[-1]) if dict_items else {}
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {}


def _horse_name_match(input_horse_name: str, fetched_horse_name: str) -> bool:
    normalized_input = _normalize_horse_name(input_horse_name)
    normalized_fetched = _normalize_horse_name(fetched_horse_name)
    if not normalized_input or not normalized_fetched:
        return False
    return normalized_input in normalized_fetched or normalized_fetched in normalized_input


def _normalize_horse_name(value: str) -> str:
    return re.sub(r"[\s\u3000]+", "", str(value)).casefold()


def _to_race_result(item: Any) -> RaceResult:
    if isinstance(item, RaceResult):
        return item
    row = _as_dict(item)
    race_name = _required_pick(row, ["race_name", "name", "race", "レース名"], "race_name")
    distance_value = _required_pick(row, ["distance", "距離"], "distance")
    track_condition = _required_pick(row, ["track_condition", "馬場状態", "馬場"], "track_condition")
    finish_position = _required_pick(row, ["finish_position", "finish", "rank", "result", "着順"], "finish")
    passing_order = _required_pick(row, ["passing_order", "passing", "corner_order", "通過", "通過順"], "passing_order")
    final_3f = _required_pick(row, ["final_3f", "last3f", "agari", "closing_3f", "上り", "上がり", "上がり3F"], "last3f")
    race_time = _pick(row, ["time", "race_time", "result_time", "走破タイム", "タイム"], None)
    winner_time_diff = _pick(row, ["winner_time_diff", "time_diff", "勝ち時計との差", "タイム差"], None)
    race_id = _pick(row, ["race_id", "id", "レースID"], "")
    surface = _pick(row, ["surface", "芝ダート", "馬場種別"], "")
    if not surface:
        surface = _surface_from_distance(distance_value)
    distance = _to_int(distance_value, 0)
    finish = _to_int(finish_position, 0)
    parsed_passing_order = parse_passing_order(passing_order)
    parsed_final_3f = _to_float(final_3f, 0.0)
    if distance <= 0:
        raise ValueError(f"invalid distance: {distance_value}")
    if finish <= 0:
        raise ValueError(f"invalid finish: {finish_position}")
    if not parsed_passing_order:
        raise ValueError(f"invalid passing_order: {passing_order}")
    if parsed_final_3f <= 0:
        raise ValueError(f"invalid last3f: {final_3f}")
    return _make_race_result(
        race_name=str(race_name),
        distance=distance,
        surface=str(surface),
        track_condition=str(track_condition),
        finish_position=finish,
        margin=_to_float(_pick(row, ["margin", "margin_sec", "着差", "着差秒"], 0.0), 0.0),
        passing_order=parsed_passing_order,
        final_3f=parsed_final_3f,
        race_time_seconds=parse_race_time(race_time),
        winner_time_diff=parse_race_time(winner_time_diff),
        field_size=_to_optional_int(
            _pick(
                row,
                ["field_size", "runners", "number_of_runners", "field", "頭数", "出走頭数"],
                None,
            )
        ),
        race_class=str(_pick(row, ["race_class", "class", "grade", "クラス"], "")),
        popularity=str(_pick(row, ["popularity", "odds_rank", "favorite", "人気", "人気順"], "")),
        date=str(_pick(row, ["date", "日付"], "")),
        course=str(_pick(row, ["course", "競馬場"], "")),
        race_id=str(race_id),
        raw=row,
    )


def _make_race_result(**kwargs: Any) -> RaceResult:
    fields = getattr(RaceResult, "__dataclass_fields__", {})
    if fields:
        kwargs = {key: value for key, value in kwargs.items() if key in fields}
    return RaceResult(**kwargs)


def _as_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return dict(item)
    if hasattr(item, "to_dict"):
        return dict(item.to_dict())
    if hasattr(item, "__dict__"):
        return dict(item.__dict__)
    raise TypeError(f"Unsupported race result row: {type(item)!r}")


def _pick(row: dict[str, Any], keys: list[str], default: Any = "") -> Any:
    for key in keys:
        value = row.get(key)
        if _has_value(value):
            return value
    return default


def _required_pick(row: dict[str, Any], keys: list[str], field_name: str) -> Any:
    value = _pick(row, keys, None)
    if not _has_value(value):
        raise ValueError(f"required field missing: {field_name}")
    return value


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return str(value).strip() != ""


def _to_int(value: Any, default: int) -> int:
    try:
        if isinstance(value, str):
            value = value.replace(",", "").strip()
            match = re.search(r"-?\d+", value)
            if match:
                value = match.group(0)
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_optional_int(value: Any) -> int | None:
    if not _has_value(value):
        return None
    return _to_int(value, 0) or None


def _to_float(value: Any, default: float) -> float:
    try:
        if isinstance(value, str):
            value = value.replace(",", "").strip()
            match = re.search(r"-?\d+(?:\.\d+)?", value)
            if match:
                value = match.group(0)
        return float(value)
    except (TypeError, ValueError):
        return default


def _surface_from_distance(value: Any) -> str:
    text = str(value)
    if "ダ" in text:
        return "ダート"
    if "障" in text:
        return "障害"
    return "芝"


def run_race_simulation(
    race_config: dict[str, Any] | RaceConfig,
    horses: list[dict[str, Any]] | list[HorseEntry],
    output_dir: str = "outputs",
    provider: RaceResultProvider | None = None,
    provider_module: str | None = None,
    provider_factory: str | None = None,
    make_gif: bool = True,
    make_mp4: bool = False,
    animation_mode: str = "2d_gif",
    animation_seconds: float = 18.0,
    video_format: str = "youtube",
    timeline_mode: str = "controlled",
    seed: int | None = None,
    make_animation: bool = True,
    save_timeline_csv: bool = True,
    use_local_database: bool = True,
    force_refresh_data: bool = False,
) -> dict[str, Any]:
    """Run the full pipeline for CLI, Streamlit, or tests.

    Existing recent-race fetchers can be passed as `provider`, or loaded via
    `provider_module` / `provider_factory`.
    """
    if seed is not None:
        import random

        import numpy as np

        random.seed(seed)
        np.random.seed(seed)

    config = _coerce_race_config(race_config)
    entries = _coerce_horses(horses)
    provider = provider or load_provider(provider_module, provider_factory, use_local_database, force_refresh_data)
    if not getattr(provider, "_codex_provider_adapter", False):
        provider = _wrap_provider(provider, use_local_database, force_refresh_data)

    logs: list[str] = []
    logs.append("直近5走データを取得します。")
    analyzer = HorseAnalyzer(provider, config)
    abilities = analyzer.analyze_many(entries)
    ability_table = analyzer.to_dataframe(abilities)

    logs.append("近走傾向から脚質と能力値を推定しました。")
    pace = PacePredictor().predict(abilities)
    logs.append(f"レース展開を {pace.pace} pace と予測しました。")

    result = RaceSimulator().simulate(
        config=config,
        abilities=abilities,
        pace=pace,
        seed=seed,
        timeline_mode=timeline_mode,
    )
    if timeline_mode == "legacy":
        logs.append("100m区間ごとのレースシミュレーションを実行しました。")
    else:
        logs.append("隊列制御型のrace_timelineを生成しました。")

    style_result_columns = [
        "horse_name",
        "actual_running_style",
        "actual_running_style_fixed",
        "adjusted_style_profile",
        "adjusted_逃げ",
        "adjusted_先行",
        "adjusted_差し",
        "adjusted_追込",
    ]
    ability_table = ability_table.merge(
        result.ranking[style_result_columns],
        on="horse_name",
        how="left",
    )
    recent_races = [
        {
            "horse_name": ability.horse_name,
            "recent_races": [race_result_to_recent_dict(race) for race in ability.recent_results[:5]],
        }
        for ability in abilities
    ]
    fetch_debug = provider.get_fetch_debug() if hasattr(provider, "get_fetch_debug") else []

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"race_{timestamp}"

    ability_csv = output_path / f"horse_analysis_{timestamp}.csv"
    ranking_csv = output_path / f"race_result_{timestamp}.csv"
    sections_csv = output_path / f"race_sections_{timestamp}.csv"
    timeline_csv = output_path / f"race_timeline_{timestamp}.csv"
    sections_table = result.states_dataframe()
    timeline_table = result.timeline_dataframe() if save_timeline_csv else pd.DataFrame()
    ability_table.to_csv(ability_csv, index=False, encoding="utf-8-sig")
    result.ranking.to_csv(ranking_csv, index=False, encoding="utf-8-sig")
    sections_table.to_csv(sections_csv, index=False, encoding="utf-8-sig")
    if save_timeline_csv:
        timeline_table.to_csv(timeline_csv, index=False, encoding="utf-8-sig")

    html_path = ""
    renderer_name = "Matplotlib marker renderer"
    horse_display_mode = "marker"
    from animation import Race3DAnimation, RaceAnimation

    if not make_animation or animation_mode == "none":
        paths = type("AnimationPaths", (), {"gif_path": "", "mp4_path": ""})()
        animation_path = ""
        logs.append("動画生成をスキップしました。")
    elif animation_mode == "3d_plotly":
        renderer_name = "Plotly marker renderer"
        html_path = Race3DAnimation(result).save_html(
            output_dir=str(output_path),
            output_name="race_3d",
            duration_seconds=animation_seconds,
        )
        paths = RaceAnimation(result).save(
            output_dir=str(output_path),
            output_name=output_name,
            duration_seconds=animation_seconds,
            make_gif=False,
            make_mp4=False,
        )
        animation_path = html_path
    elif animation_mode == "post_mp4":
        paths = RaceAnimation(result).save(
            output_dir=str(output_path),
            output_name=output_name,
            duration_seconds=animation_seconds,
            make_gif=False,
            make_mp4=False,
        )
        from video_renderer import render_race_video

        try:
            renderer_info: dict[str, str] = {}
            movie_path = render_race_video(
                simulation_result={
                    "ranking": result.ranking,
                    "sections": sections_table,
                    "timeline": timeline_table,
                    "race_timeline": result.race_timeline,
                },
                race_config=config.to_dict(),
                horses=[entry.to_dict() for entry in entries],
                output_path=str(output_path / "race_movie.mp4"),
                video_format=video_format,
                duration_sec=int(animation_seconds),
                renderer_info=renderer_info,
            )
            renderer_name = renderer_info.get("renderer_name", "Pillow/imageio marker renderer")
            horse_display_mode = renderer_info.get("horse_display_mode", "marker")
        except Exception as exc:
            logs.append(f"投稿用MP4の生成をスキップしました: {exc}")
            movie_path = ""
        paths = type(paths)(gif_path=paths.gif_path, mp4_path=movie_path)
        animation_path = movie_path
    else:
        paths = RaceAnimation(result).save(
            output_dir=str(output_path),
            output_name=output_name,
            duration_seconds=animation_seconds,
            make_gif=make_gif,
            make_mp4=make_mp4,
        )
        animation_path = paths.gif_path if make_gif else paths.mp4_path
    if make_animation and animation_mode != "none":
        logs.append("レース動画を生成しました。")

    return {
        "race_config": config.to_dict(),
        "ranking": result.ranking,
        "sections": sections_table,
        "timeline": timeline_table,
        "race_timeline": result.race_timeline,
        "horse_analysis": ability_table,
        "pace_prediction": pace.to_dict(),
        "abilities": abilities,
        "pace": pace,
        "recent_races": recent_races,
        "horse_inputs": [entry.to_dict() for entry in entries],
        "fetch_debug": fetch_debug,
        "animation_path": animation_path,
        "gif_path": paths.gif_path,
        "mp4_path": paths.mp4_path,
        "plotly_html_path": html_path,
        "renderer_name": renderer_name,
        "horse_display_mode": horse_display_mode,
        "timeline_mode": timeline_mode,
        "csv_paths": {
            "horse_analysis": str(ability_csv),
            "ranking": str(ranking_csv),
            "sections": str(sections_csv),
            "timeline": str(timeline_csv) if save_timeline_csv else "",
        },
        "log": logs,
    }


def _coerce_race_config(value: dict[str, Any] | RaceConfig) -> RaceConfig:
    if isinstance(value, RaceConfig):
        return value
    return RaceConfig(
        course=str(value["course"]),
        surface=str(value["surface"]),  # type: ignore[arg-type]
        distance=int(value["distance"]),
        direction=str(value["direction"]),  # type: ignore[arg-type]
        weather=str(value["weather"]),  # type: ignore[arg-type]
        track_condition=str(value["track_condition"]),  # type: ignore[arg-type]
        race_course_day=str(value.get("race_course_day", "1日目")),
        course_layout=str(value.get("course_layout", "A")),
        track_bias=str(value.get("track_bias", "標準")),
    )


def _coerce_horses(values: list[dict[str, Any]] | list[HorseEntry]) -> list[HorseEntry]:
    entries: list[HorseEntry] = []
    for value in values:
        if isinstance(value, HorseEntry):
            entries.append(value)
        else:
            entries.append(
                HorseEntry(
                    horse_name=str(value["horse_name"]).strip(),
                    frame=int(value["frame"]),
                    horse_number=int(value["horse_number"]),
                    carried_weight=float(value.get("carried_weight", 56.0) or 56.0),
                    pedigree_info=value.get("pedigree_info") if isinstance(value.get("pedigree_info"), dict) else None,
                    jockey=str(value.get("jockey", "")).strip(),
                    jockey_score=float(value.get("jockey_score", 50.0) or 50.0),
                )
            )
    return entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Horse race simulation app")
    parser.add_argument("--demo", action="store_true", help="run without interactive prompts")
    parser.add_argument("--provider-module", default=None, help="module containing your existing result fetcher")
    parser.add_argument("--provider-factory", default=None, help="factory name. default: get_provider")
    parser.add_argument("--output-dir", default="outputs", help="directory for race.gif and race.mp4")
    parser.add_argument("--animation-seconds", type=float, default=18.0, help="rendered video length")
    parser.add_argument("--no-mp4", action="store_true", help="skip MP4 generation")
    parser.add_argument("--no-gif", action="store_true", help="skip GIF generation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    interactive = sys.stdin.isatty() and not args.demo
    config = prompt_race_config() if interactive else RaceConfig.default()
    horses = prompt_horses() if interactive else demo_horses()

    provider = load_provider(args.provider_module, args.provider_factory)
    analyzer = HorseAnalyzer(provider)
    abilities = analyzer.analyze_many(horses)
    ability_table = analyzer.to_dataframe(abilities)

    pace = PacePredictor().predict(abilities)
    result = RaceSimulator().simulate(config=config, abilities=abilities, pace=pace)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ability_table.to_csv(output_dir / "horse_analysis.csv", index=False, encoding="utf-8-sig")
    result.ranking.to_csv(output_dir / "race_result.csv", index=False, encoding="utf-8-sig")
    result.states_dataframe().to_csv(output_dir / "race_sections.csv", index=False, encoding="utf-8-sig")
    result.timeline_dataframe().to_csv(output_dir / "race_timeline.csv", index=False, encoding="utf-8-sig")

    from animation import RaceAnimation

    paths = RaceAnimation(result).save(
        output_dir=str(output_dir),
        duration_seconds=args.animation_seconds,
        make_gif=not args.no_gif,
        make_mp4=not args.no_mp4,
    )

    print("Race config")
    print(config.to_dict())
    print("\nHorse analysis")
    print(ability_table.to_string(index=False))
    print("\nPace prediction")
    print(pace.to_dict())
    print("\nResult")
    print(result.ranking.to_string(index=False))
    print(f"\nGIF: {paths.gif_path}")
    print(f"MP4: {paths.mp4_path}")


if __name__ == "__main__":
    main()
