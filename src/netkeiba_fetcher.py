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
NAR_RACE_BASE_URL = "https://nar.netkeiba.com"
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
    "30": "門別",
    "35": "盛岡",
    "36": "水沢",
    "42": "浦和",
    "43": "船橋",
    "44": "大井",
    "45": "川崎",
    "46": "金沢",
    "47": "笠松",
    "48": "名古屋",
    "50": "園田",
    "51": "姫路",
    "54": "高知",
    "55": "佐賀",
    "65": "帯広",
}

JRA_VENUE_CODES = {code for code in VENUE_CODES if int(code) <= 10}


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

    def fetch_race_schedule(self, race_date: str) -> list[dict[str, Any]]:
        """Return the JRA and NAR race cards published for one date."""
        date_compact = re.sub(r"\D", "", str(race_date))
        if len(date_compact) != 8:
            return []
        races: list[dict[str, Any]] = []
        errors: list[str] = []
        for organization, base_url in (("JRA", RACE_BASE_URL), ("NAR", NAR_RACE_BASE_URL)):
            try:
                races.extend(self._fetch_schedule_for_site(date_compact, organization, base_url))
            except (requests.RequestException, ValueError, pd.errors.ParserError) as exc:
                errors.append(f"{organization}: {exc}")
        races.sort(key=lambda row: (str(row.get("organization", "")), str(row.get("venue", "")), int(row.get("race_number", 0))))
        self.last_debug = {
            "operation": "fetch_race_schedule",
            "race_date_input": date_compact,
            "raw_dataframe": pd.DataFrame(races),
            "parsed_entries": races,
            "errors": errors,
        }
        return races

    def _fetch_schedule_for_site(
        self,
        date_compact: str,
        organization: str,
        base_url: str,
    ) -> list[dict[str, Any]]:
        date_url = f"{base_url}/top/race_list_get_date_list.html?kaisai_date={date_compact}"
        date_html, date_final_url = self._request_text(date_url)
        date_soup = BeautifulSoup(date_html, "lxml")
        anchors = date_soup.select(f"li[date='{date_compact}'] a[href*='race_list_sub']")
        if not anchors:
            anchors = []
            for anchor in date_soup.select("a[href*='race_list_sub']"):
                parent = anchor.find_parent("li")
                if parent is not None and str(parent.get("date", "")) == date_compact:
                    anchors.append(anchor)
        sub_urls = [urljoin(date_final_url, str(anchor.get("href", ""))) for anchor in anchors]
        if not sub_urls:
            return []

        races: list[dict[str, Any]] = []
        seen: set[str] = set()
        for sub_url in dict.fromkeys(sub_urls):
            html_text, final_url = self._request_text(sub_url)
            soup = BeautifulSoup(html_text, "lxml")
            blocks = soup.select("dl.RaceList_DataList") or soup.select("dl.RaceList_Box") or [soup]
            for block in blocks:
                meeting_node = block.select_one(".RaceList_DataTitle, .RaceList_DataHeader")
                meeting = _clean_text(meeting_node.get_text(" ", strip=True)) if meeting_node else ""
                venue, meeting_number, meeting_day = _parse_meeting(meeting)
                if venue:
                    meeting = f"{venue}{meeting_number}回{meeting_day}日"
                for item in block.select("li.RaceList_DataItem"):
                    anchor = item.select_one("a[href*='race_id=']")
                    if anchor is None:
                        continue
                    href = str(anchor.get("href", ""))
                    race_id = _extract_race_id(href)
                    if not race_id or race_id in seen:
                        continue
                    race_number = _parse_int(_node_text(item, ".Race_Num"))
                    race_name = _node_text(item, ".ItemTitle") or _clean_text(anchor.get_text(" ", strip=True))
                    race_data = _node_text(item, ".RaceData")
                    start_time = _node_text(item, ".RaceList_Itemtime")
                    course_text = _node_text(item, ".RaceList_ItemLong")
                    if not start_time:
                        time_match = re.search(r"\b(\d{1,2}:\d{2})\b", race_data)
                        start_time = time_match.group(1) if time_match else ""
                    if not course_text:
                        course_match = re.search(r"(芝|ダート|ダ|障害)\s*(\d{3,4})m", race_data)
                        course_text = "" if course_match is None else f"{course_match.group(1)}{course_match.group(2)}m"
                    course_match = re.search(r"(芝|ダート|ダ|障害)\s*(\d{3,4})m", course_text)
                    surface_text = course_match.group(1) if course_match else ""
                    surface = "ダート" if surface_text in {"ダ", "ダート"} else surface_text
                    distance = int(course_match.group(2)) if course_match else 0
                    resolved_venue = venue or venue_from_race_id(race_id)
                    races.append(
                        {
                            "race_id": race_id,
                            "race_date": f"{date_compact[:4]}-{date_compact[4:6]}-{date_compact[6:]}",
                            "organization": organization,
                            "venue": resolved_venue,
                            "meeting": meeting,
                            "meeting_number": meeting_number,
                            "meeting_day": meeting_day,
                            "race_number": race_number,
                            "race_name": race_name,
                            "start_time": start_time,
                            "surface": surface,
                            "distance": distance,
                            "source_url": urljoin(final_url, href),
                        }
                    )
                    seen.add(race_id)
        return races

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
        base_url = _race_site_base(race_id)
        url = f"{base_url}/race/shutuba.html?race_id={race_id}"
        html_text, final_url = self._request_text(url)
        raw_frame = _find_table(html_text, required_groups=[("馬名",), ("馬番",), ("騎手",), ("斤量",)])
        entries = [] if raw_frame is None else [_parse_entry_row(row) for _, row in raw_frame.iterrows()]
        entries = [entry for entry in entries if entry.get("horse_name") and entry.get("horse_number")]
        horse_links = _extract_entry_horse_links(html_text, final_url)
        _merge_horse_ids_into_entries(entries, horse_links)
        self.last_debug = {
            "operation": "fetch_race_entries",
            "race_id": race_id,
            "fetch_url": url,
            "source_url": final_url,
            "raw_dataframe": raw_frame if raw_frame is not None else pd.DataFrame(),
            "parsed_entries": entries,
            "horse_id_count": sum(1 for entry in entries if entry.get("horse_id")),
            "horse_links": horse_links,
            "parsed_results": [],
            "race_metadata": {},
        }
        return entries

    def fetch_race_result(self, race_id: str) -> list[dict[str, Any]]:
        if not _valid_race_id(race_id):
            return []
        url = f"{_race_site_base(race_id)}/race/result.html?race_id={race_id}"
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
        base_url = _race_site_base(race_id)
        url = f"{base_url}/race/shutuba.html?race_id={race_id}"
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
        meeting_match = re.search(r"(\d+)回\s*([^\s]+)\s*(\d+)日目", data_text)
        meeting = (
            f"{meeting_match.group(2)}{meeting_match.group(1)}回{meeting_match.group(3)}日"
            if meeting_match
            else ""
        )
        date_match = re.search(r"(20\d{2})年\s*(\d{1,2})月\s*(\d{1,2})日", f"{data_text} {page_title}")
        race_date = (
            f"{int(date_match.group(1)):04d}-{int(date_match.group(2)):02d}-{int(date_match.group(3)):02d}"
            if date_match
            else ""
        )
        course_layout_match = re.search(r"(?:芝)?\s*([ABCD])コース", data_text, re.I)
        course_layout = course_layout_match.group(1).upper() if course_layout_match else ""
        if not course_layout and base_url == RACE_BASE_URL and surface == "芝":
            course_layout = self._fetch_jra_course_layout(race_id, race_date)
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
            "meeting": meeting,
            "meeting_number": int(meeting_match.group(1)) if meeting_match else 0,
            "meeting_day": int(meeting_match.group(3)) if meeting_match else 0,
            "race_course_day": f"{int(meeting_match.group(3))}日目" if meeting_match else "",
            "course_layout": course_layout,
            "organization": "JRA" if base_url == RACE_BASE_URL else "NAR",
            "corner_count": _infer_corner_count(venue_from_race_id(race_id), surface, distance, direction_match.group(0) if direction_match else ""),
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

    def _fetch_jra_course_layout(self, race_id: str, race_date: str) -> str:
        date_compact = re.sub(r"\D", "", race_date)
        if len(date_compact) != 8:
            date_compact = str(race_id)[:4] + "0101"
        try:
            html_text, _ = self._request_text(f"{RACE_BASE_URL}/race/track.html?kaisai_date={date_compact}")
        except (requests.RequestException, ValueError):
            return ""
        soup = BeautifulSoup(html_text, "lxml")
        venue_code = str(race_id)[4:6]
        venue_node = soup.select_one(f".Jyo_{venue_code}")
        search_root = venue_node.parent if venue_node is not None and venue_node.parent is not None else soup
        text = _clean_text(search_root.get_text(" ", strip=True))
        match = re.search(r"(?:芝)?\s*([ABCD])コース", text, re.I)
        return match.group(1).upper() if match else ""

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
    trainer = _pick(row, ["厩舎", "調教師", "trainer"])
    affiliation = _trainer_affiliation(trainer)
    return {
        "horse_name": _pick(row, ["馬名", "horse_name"]),
        "frame": _parse_int(_pick(row, ["枠", "枠番", "frame"])),
        "horse_number": _parse_int(_pick(row, ["馬番", "horse_number"])),
        "carried_weight": _parse_float_or_none(_pick(row, ["斤量", "carried_weight"])),
        "jockey": _pick(row, ["騎手", "jockey"]),
        "sex_age": _pick(row, ["性齢", "sex_age"]),
        "trainer": trainer,
        "affiliation": affiliation,
        "organization": "JRA" if affiliation in {"美浦", "栗東"} else "NAR" if affiliation else "",
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


def _race_site_base(race_id: str) -> str:
    return RACE_BASE_URL if str(race_id)[4:6] in JRA_VENUE_CODES else NAR_RACE_BASE_URL


def _node_text(node: Any, selector: str) -> str:
    child = node.select_one(selector)
    return _clean_text(child.get_text(" ", strip=True)) if child is not None else ""


def _parse_meeting(value: str) -> tuple[str, int, int]:
    match = re.search(r"(\d+)回\s*([^\s]+)\s*(\d+)日目", value)
    if not match:
        return "", 0, 0
    return match.group(2), int(match.group(1)), int(match.group(3))


def _trainer_affiliation(trainer: str) -> str:
    text = _clean_text(trainer)
    for affiliation in ("美浦", "栗東", "門別", "盛岡", "水沢", "浦和", "船橋", "大井", "川崎", "金沢", "笠松", "名古屋", "園田", "姫路", "高知", "佐賀", "帯広"):
        if affiliation in text:
            return affiliation
    return ""


def _infer_corner_count(venue: str, surface: str, distance: int, direction: str) -> int:
    if direction == "直線":
        return 0
    if distance <= 1600:
        return 2
    if venue == "東京" and distance <= 2000:
        return 2
    if venue in {"京都", "阪神"} and surface == "芝" and distance <= 1800:
        return 2
    return 4


def _extract_entry_horse_links(html_text: str, base_url: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(html_text, "lxml")
    links: list[dict[str, str]] = []
    seen: set[str] = set()
    for anchor in soup.select("a[href*='/horse/']"):
        href = str(anchor.get("href", ""))
        match = re.search(r"/horse/(\d+)/?", href)
        if not match:
            continue
        horse_id = match.group(1)
        if horse_id in seen:
            continue
        horse_name = _clean_text(anchor.get_text(" ", strip=True))
        if not horse_name:
            continue
        links.append(
            {
                "horse_name": horse_name,
                "horse_id": horse_id,
                "horse_url": urljoin(base_url, href),
            }
        )
        seen.add(horse_id)
    return links


def _merge_horse_ids_into_entries(entries: list[dict[str, Any]], links: list[dict[str, str]]) -> None:
    unused_links = list(links)
    for entry in entries:
        entry.setdefault("horse_id", "")
        entry.setdefault("horse_url", "")
        entry_name = _normalize_text(entry.get("horse_name", ""))
        matched_index = next(
            (
                link_index
                for link_index, link in enumerate(unused_links)
                if entry_name and entry_name == _normalize_text(link.get("horse_name", ""))
            ),
            None,
        )
        if matched_index is None:
            continue
        link = unused_links.pop(matched_index)
        entry["horse_id"] = link.get("horse_id", "")
        entry["horse_url"] = link.get("horse_url", "")


_DEFAULT_FETCHER = NetkeibaRaceFetcher()


def search_race_id_by_name_and_date(race_name: str, race_date: str) -> dict[str, Any] | None:
    return _DEFAULT_FETCHER.search_race_id_by_name_and_date(race_name, race_date)


def fetch_race_entries(race_id: str) -> list[dict[str, Any]]:
    return _DEFAULT_FETCHER.fetch_race_entries(race_id)


def fetch_race_schedule(race_date: str) -> list[dict[str, Any]]:
    return _DEFAULT_FETCHER.fetch_race_schedule(race_date)


def fetch_race_result(race_id: str) -> list[dict[str, Any]]:
    return _DEFAULT_FETCHER.fetch_race_result(race_id)


def fetch_race_metadata(race_id: str) -> dict[str, Any]:
    return _DEFAULT_FETCHER.fetch_race_metadata(race_id)


def get_fetch_debug() -> dict[str, Any]:
    return _DEFAULT_FETCHER.get_debug_info()
