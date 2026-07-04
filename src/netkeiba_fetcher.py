from __future__ import annotations

import re
import time
from io import StringIO
from typing import Any
from urllib.parse import parse_qs, urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

from horse_analyzer import parse_race_time


RACE_BASE_URL = "https://race.netkeiba.com"
DB_BASE_URL = "https://db.netkeiba.com"
VENUE_CODES = {
    "01": "札幌",
    "02": "函館",
    "03": "福島",
    "04": "新潟",
    "05": "東京",
    "06": "中山",
    "07": "中京",
    "08": "京都",
    "09": "阪神",
    "10": "小倉",
}


class NetkeibaRaceFetcher:
    """Fetch race cards and completed results without creating fallback rows."""

    def __init__(self, timeout: float = 15.0, min_interval_sec: float = 0.8) -> None:
        self.timeout = timeout
        self.min_interval_sec = min_interval_sec
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept-Language": "ja,en;q=0.8",
            }
        )
        self._last_request_at = 0.0
        self.last_debug: dict[str, Any] = {}

    def search_race_id_by_name_and_date(self, race_name: str, race_date: str) -> dict[str, Any] | None:
        normalized_name = _normalize_text(race_name)
        date_compact = re.sub(r"\D", "", str(race_date))
        if not normalized_name or len(date_compact) != 8:
            return None
        date_list_url = f"{RACE_BASE_URL}/top/race_list_get_date_list.html?kaisai_date={date_compact}"
        date_list_html, date_list_final_url = self._request_text(date_list_url)
        date_soup = BeautifulSoup(date_list_html, "lxml")
        date_anchor = date_soup.select_one(f"li[date='{date_compact}'] a[href*='race_list_sub']")
        race_list_url = (
            urljoin(date_list_final_url, str(date_anchor.get("href", "")))
            if date_anchor is not None
            else date_list_url
        )
        if race_list_url != date_list_url:
            html_text, final_url = self._request_text(race_list_url)
        else:
            html_text, final_url = date_list_html, date_list_final_url
        soup = BeautifulSoup(html_text, "lxml")
        candidates: list[dict[str, str]] = []
        seen_race_ids: set[str] = set()
        for anchor in soup.select("a[href*='race_id='], a[href*='/race/']"):
            href = str(anchor.get("href", ""))
            race_id = _extract_race_id(href)
            label = _clean_text(anchor.get_text(" ", strip=True))
            if race_id and label and race_id not in seen_race_ids:
                candidates.append({"race_id": race_id, "race_name": label, "href": href})
                seen_race_ids.add(race_id)
        match = next(
            (
                candidate
                for candidate in candidates
                if normalized_name == _normalize_text(candidate["race_name"])
                or normalized_name in _normalize_text(candidate["race_name"])
                or _normalize_text(candidate["race_name"]) in normalized_name
            ),
            None,
        )
        self.last_debug = {
            "operation": "search_race_id",
            "race_name_input": race_name,
            "race_date_input": str(race_date),
            "fetch_url": race_list_url,
            "source_url": final_url,
            "race_id": match["race_id"] if match else "",
            "raw_dataframe": pd.DataFrame(candidates),
            "parsed_entries": [],
            "parsed_results": [],
            "race_metadata": {},
        }
        if match is None:
            return None
        race_id = match["race_id"]
        return {
            "race_id": race_id,
            "race_name": match["race_name"],
            "race_date": str(race_date),
            "venue": venue_from_race_id(race_id),
            "source_url": urljoin(final_url, match["href"]),
        }

    def fetch_race_entries(self, race_id: str) -> list[dict[str, Any]]:
        if not _valid_race_id(race_id):
            return []
        url = f"{RACE_BASE_URL}/race/shutuba.html?race_id={race_id}"
        html_text, final_url = self._request_text(url)
        raw_frame = _find_table(html_text, required_groups=[("馬名",), ("馬番",), ("騎手",), ("斤量",)])
        entries = [] if raw_frame is None else [_parse_entry_row(row) for _, row in raw_frame.iterrows()]
        entries = [entry for entry in entries if entry.get("horse_name") and entry.get("horse_number")]
        self.last_debug = {
            "operation": "fetch_race_entries",
            "race_id": race_id,
            "fetch_url": url,
            "source_url": final_url,
            "raw_dataframe": raw_frame if raw_frame is not None else pd.DataFrame(),
            "parsed_entries": entries,
            "parsed_results": [],
            "race_metadata": {},
        }
        return entries

    def fetch_race_result(self, race_id: str) -> list[dict[str, Any]]:
        if not _valid_race_id(race_id):
            return []
        url = f"{RACE_BASE_URL}/race/result.html?race_id={race_id}"
        try:
            html_text, final_url = self._request_text(url)
            raw_frame = _find_table(html_text, required_groups=[("着順",), ("馬名",), ("馬番",), ("タイム",)])
        except (requests.RequestException, ValueError, pd.errors.ParserError):
            self.last_debug = {
                "operation": "fetch_race_result",
                "race_id": race_id,
                "fetch_url": url,
                "source_url": url,
                "raw_dataframe": pd.DataFrame(),
                "parsed_results": [],
            }
            return []
        results = [] if raw_frame is None else [_parse_result_row(row) for _, row in raw_frame.iterrows()]
        results = [row for row in results if row.get("horse_name") and row.get("finish")]
        self.last_debug = {
            "operation": "fetch_race_result",
            "race_id": race_id,
            "fetch_url": url,
            "source_url": final_url,
            "raw_dataframe": raw_frame if raw_frame is not None else pd.DataFrame(),
            "parsed_entries": [],
            "parsed_results": results,
            "race_metadata": {},
        }
        return results

    def fetch_race_metadata(self, race_id: str) -> dict[str, Any]:
        if not _valid_race_id(race_id):
            return {}
        url = f"{RACE_BASE_URL}/race/shutuba.html?race_id={race_id}"
        html_text, final_url = self._request_text(url)
        soup = BeautifulSoup(html_text, "lxml")
        race_name = next(
            (
                _clean_text(node.get_text(" ", strip=True))
                for node in soup.select("h1.RaceName, .RaceName, h1")
                if _clean_text(node.get_text(" ", strip=True))
            ),
            "",
        )
        page_title = _clean_text(soup.title.get_text(" ", strip=True)) if soup.title else ""
        data_text = _clean_text(" ".join(node.get_text(" ", strip=True) for node in soup.select(".RaceData01, .RaceData02, .RaceData")))
        distance_match = re.search(r"(芝|ダート|ダ|障害)\s*(\d{3,4})m", data_text)
        surface_text = distance_match.group(1) if distance_match else ""
        surface = "ダート" if surface_text in {"ダ", "ダート"} else "障害" if surface_text == "障害" else "芝" if surface_text else ""
        distance = int(distance_match.group(2)) if distance_match else 0
        direction_match = re.search(r"(?:右|左|直線)", data_text)
        weather_match = re.search(r"天候[:：]?\s*(晴|曇|雨|雪)", data_text)
        condition_match = re.search(r"(?:馬場|芝|ダート)[:：]?\s*(良|稍重|稍|重|不良)", data_text)
        date_match = re.search(r"(20\d{2})年\s*(\d{1,2})月\s*(\d{1,2})日", f"{data_text} {page_title}")
        race_date = (
            f"{int(date_match.group(1)):04d}-{int(date_match.group(2)):02d}-{int(date_match.group(3)):02d}"
            if date_match
            else ""
        )
        class_text = _clean_text(
            f"{page_title} "
            + " ".join(node.get_text(" ", strip=True) for node in soup.select(".Icon_GradeType, .RaceName span, .RaceData02"))
        )
        class_match = re.search(r"G[123]|GⅠ|GⅡ|GⅢ|リステッド|Listed|オープン|\d勝|未勝利|新馬", class_text, re.I)
        race_class = _normalize_race_class(class_match.group(0)) if class_match else ""
        metadata = {
            "race_name": race_name,
            "race_date": race_date,
            "venue": venue_from_race_id(race_id),
            "surface": surface,
            "distance": distance,
            "direction": direction_match.group(0) if direction_match else "",
            "weather": weather_match.group(1) if weather_match else "",
            "track_condition": "稍重" if condition_match and condition_match.group(1) == "稍" else condition_match.group(1) if condition_match else "",
            "race_class": race_class,
            "source_url": final_url,
        }
        self.last_debug = {
            "operation": "fetch_race_metadata",
            "race_id": race_id,
            "fetch_url": url,
            "source_url": final_url,
            "raw_dataframe": pd.DataFrame([{"race_text": data_text, "class_text": class_text}]),
            "parsed_entries": [],
            "parsed_results": [],
            "race_metadata": metadata,
        }
        return metadata

    def get_debug_info(self) -> dict[str, Any]:
        return dict(self.last_debug)

    def _request_text(self, url: str) -> tuple[str, str]:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.min_interval_sec:
            time.sleep(self.min_interval_sec - elapsed)
        response = self.session.get(url, timeout=self.timeout)
        self._last_request_at = time.monotonic()
        response.raise_for_status()
        if not response.encoding or response.encoding.lower() == "iso-8859-1":
            response.encoding = response.apparent_encoding or "utf-8"
        return response.text, response.url


def venue_from_race_id(race_id: str) -> str:
    return VENUE_CODES.get(str(race_id)[4:6], "") if len(str(race_id)) >= 6 else ""


def _find_table(html_text: str, required_groups: list[tuple[str, ...]]) -> pd.DataFrame | None:
    try:
        tables = pd.read_html(StringIO(html_text))
    except (ValueError, ImportError, pd.errors.ParserError):
        return None
    for table in tables:
        frame = _flatten_columns(table)
        columns = [_normalize_text(column) for column in frame.columns]
        if all(any(any(_normalize_text(name) in column for name in group) for column in columns) for group in required_groups):
            return frame
    return None


def _parse_entry_row(row: pd.Series) -> dict[str, Any]:
    return {
        "horse_name": _pick(row, ["馬名", "horse_name"]),
        "frame": _parse_int(_pick(row, ["枠", "枠番", "frame"])),
        "horse_number": _parse_int(_pick(row, ["馬番", "horse_number"])),
        "carried_weight": _parse_float_or_none(_pick(row, ["斤量", "carried_weight"])),
        "jockey": _pick(row, ["騎手", "jockey"]),
        "sex_age": _pick(row, ["性齢", "sex_age"]),
        "trainer": _pick(row, ["厩舎", "調教師", "trainer"]),
        "horse_weight": _pick(row, ["馬体重", "horse_weight"]) or None,
        "popularity": _parse_int_or_none(_pick(row, ["人気", "人気順", "popularity"])),
        "odds": _parse_float_or_none(_pick(row, ["オッズ", "単勝", "odds"])),
    }


def _parse_result_row(row: pd.Series) -> dict[str, Any]:
    time_text = _pick(row, ["タイム", "time"])
    margin_text = _pick(row, ["着差", "margin"])
    return {
        "finish": _parse_int_or_none(_pick(row, ["着順", "finish", "rank"])),
        "horse_name": _pick(row, ["馬名", "horse_name"]),
        "frame": _parse_int_or_none(_pick(row, ["枠", "枠番", "frame"])),
        "horse_number": _parse_int_or_none(_pick(row, ["馬番", "horse_number"])),
        "carried_weight": _parse_float_or_none(_pick(row, ["斤量", "carried_weight"])),
        "jockey": _pick(row, ["騎手", "jockey"]),
        "popularity": _parse_int_or_none(_pick(row, ["人気", "人気順", "popularity"])),
        "time": time_text,
        "time_sec": parse_race_time(time_text),
        "margin": margin_text,
        "margin_sec": _margin_to_seconds(margin_text),
        "passing_order": _pick(row, ["通過", "通過順", "passing_order"]),
        "last3f": _parse_float_or_none(_pick(row, ["上り", "上がり", "後3F", "last3f"])),
        "odds": _parse_float_or_none(_pick(row, ["オッズ", "単勝", "odds"])),
    }


def _pick(row: pd.Series, candidates: list[str]) -> str:
    for candidate in candidates:
        normalized_candidate = _normalize_text(candidate)
        for column in row.index:
            if normalized_candidate in _normalize_text(column):
                value = row[column]
                if value is not None and not pd.isna(value):
                    text = _clean_text(value)
                    if text and text.lower() != "nan":
                        return text
    return ""


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    flattened = frame.copy()
    flattened.columns = [
        " ".join(str(part) for part in column if str(part) != "nan").strip()
        if isinstance(column, tuple)
        else str(column)
        for column in flattened.columns
    ]
    return flattened


def _extract_race_id(value: str) -> str:
    query_id = parse_qs(urlparse(value).query).get("race_id", [""])[0]
    if re.fullmatch(r"\d{12}", query_id):
        return query_id
    match = re.search(r"(?:race_id=|/race/)(\d{12})", value)
    return match.group(1) if match else ""


def _valid_race_id(value: str) -> bool:
    return bool(re.fullmatch(r"\d{12}", str(value).strip()))


def _normalize_text(value: Any) -> str:
    return re.sub(r"[\s\u3000・･（）()\[\]]+", "", str(value)).casefold()


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def _parse_int(value: Any) -> int:
    match = re.search(r"\d+", str(value))
    return int(match.group(0)) if match else 0


def _parse_int_or_none(value: Any) -> int | None:
    parsed = _parse_int(value)
    return parsed if parsed > 0 else None


def _parse_float_or_none(value: Any) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else None


def _margin_to_seconds(value: Any) -> float | None:
    text = _clean_text(value)
    if text in {"", "-", "同着"}:
        return 0.0 if text in {"", "同着"} else None
    named = {"ハナ": 0.05, "アタマ": 0.10, "クビ": 0.15, "大差": 3.0}
    if text in named:
        return named[text]
    fractions = {"1/2": 0.10, "3/4": 0.15, "1.1/4": 0.25, "1.1/2": 0.30, "1.3/4": 0.35}
    if text in fractions:
        return fractions[text]
    number = _parse_float_or_none(text)
    return number * 0.2 if number is not None else None


def _normalize_race_class(value: str) -> str:
    upper = value.upper()
    return upper.replace("GⅠ", "G1").replace("GⅡ", "G2").replace("GⅢ", "G3").replace("リステッド", "L")


_DEFAULT_FETCHER = NetkeibaRaceFetcher()


def search_race_id_by_name_and_date(race_name: str, race_date: str) -> dict[str, Any] | None:
    return _DEFAULT_FETCHER.search_race_id_by_name_and_date(race_name, race_date)


def fetch_race_entries(race_id: str) -> list[dict[str, Any]]:
    return _DEFAULT_FETCHER.fetch_race_entries(race_id)


def fetch_race_result(race_id: str) -> list[dict[str, Any]]:
    return _DEFAULT_FETCHER.fetch_race_result(race_id)


def fetch_race_metadata(race_id: str) -> dict[str, Any]:
    return _DEFAULT_FETCHER.fetch_race_metadata(race_id)


def get_fetch_debug() -> dict[str, Any]:
    return _DEFAULT_FETCHER.get_debug_info()
