from __future__ import annotations

import html
import json
import re
import urllib.parse
import urllib.request
from io import StringIO
from typing import Any

import pandas as pd


NETKEIBA_HOME = "https://www.netkeiba.com/"
NETKEIBA_DB = "https://db.netkeiba.com"


class NetkeibaRaceResultProvider:
    """Fetch real recent-race records from netkeiba and never synthesize races."""

    def __init__(self, timeout: float = 15.0) -> None:
        self.timeout = timeout
        self.last_debug: dict[str, Any] = {}

    def get_recent_results(self, horse_name: str, limit: int = 10) -> list[dict[str, Any]]:
        input_name = horse_name.strip()
        if not input_name:
            raise ValueError("horse_name is empty")

        self.last_debug = {
            "input_horse_name": input_name,
            "search_result_horse_name": "",
            "horse_id": "",
            "url": "",
            "raw_race_df": pd.DataFrame(),
            "recent_races": [],
        }

        horse_id, fetched_name, horse_url = self.search_horse(input_name)
        self.last_debug.update(
            {
                "search_result_horse_name": fetched_name,
                "horse_id": horse_id,
                "url": horse_url,
            }
        )
        if not horse_id:
            raise ValueError(f"horse_id not found: {input_name}")
        if fetched_name and not names_compatible(input_name, fetched_name):
            raise ValueError(f"horse_name mismatch: input={input_name}, fetched={fetched_name}")

        page_html, _ = self.request_text(horse_url)
        page_name = extract_page_horse_name(page_html) or fetched_name
        if page_name:
            self.last_debug["search_result_horse_name"] = page_name
        if page_name and not names_compatible(input_name, page_name):
            raise ValueError(f"horse_name mismatch: input={input_name}, page={page_name}")

        raw_race_df = self.read_race_table(page_html, horse_id)
        self.last_debug["raw_race_df"] = raw_race_df
        recent_races = normalize_race_table(raw_race_df)
        if not recent_races:
            raise ValueError(f"race history not found: {input_name}")

        limited = recent_races[:limit]
        self.last_debug["recent_races"] = limited
        return limited

    def search_horse(self, horse_name: str) -> tuple[str, str, str]:
        search_url = f"{NETKEIBA_DB}/?pid=horse_list&word={urllib.parse.quote(horse_name)}"
        search_html, final_url = self.request_text(search_url)

        direct_id = extract_horse_id(final_url)
        if direct_id:
            page_name = extract_page_horse_name(search_html)
            return direct_id, page_name, horse_url_from_id(direct_id)

        candidates = extract_search_candidates(search_html)
        normalized_input = normalize_name(horse_name)
        for horse_id, candidate_name, candidate_url in candidates:
            if normalize_name(candidate_name) == normalized_input:
                return horse_id, candidate_name, candidate_url

        raise ValueError(f"horse_id not found: {horse_name}")

    def request_text(self, url: str) -> tuple[str, str]:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept-Language": "ja,en;q=0.8",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            raw = response.read()
            final_url = response.geturl()
            charset = response.headers.get_content_charset()
        return decode_html(raw, charset), final_url

    def read_race_table(self, page_html: str, horse_id: str = "") -> pd.DataFrame:
        if horse_id:
            ajax_url = f"{NETKEIBA_DB}/horse/ajax_horse_results.html?{urllib.parse.urlencode({'input': 'UTF-8', 'output': 'json', 'id': horse_id})}"
            ajax_text, _ = self.request_text(ajax_url)
            payload = json.loads(ajax_text)
            if payload.get("status") == "OK" and payload.get("data"):
                try:
                    ajax_frame = self.find_race_table(str(payload["data"]))
                except ValueError:
                    ajax_frame = None
                if ajax_frame is not None:
                    return ajax_frame

        try:
            page_frame = self.find_race_table(page_html)
        except ValueError:
            page_frame = None
        if page_frame is not None:
            return page_frame
        raise ValueError("race history table not found")

    def find_race_table(self, html_text: str) -> pd.DataFrame | None:
        try:
            tables = pd.read_html(StringIO(html_text))
        except ValueError as exc:
            raise ValueError("read_html found no tables") from exc

        for table in tables:
            frame = flatten_columns(table)
            if is_race_history_table(frame):
                return frame
        return None

    def get_debug_info(self, horse_name: str | None = None) -> dict[str, Any]:
        return dict(self.last_debug)

    def fetch_race_entries(self, race_id: str) -> list[dict[str, Any]]:
        """Future hook for fetching all runners in a past race.

        It intentionally returns no synthetic rows. Until full race-entry
        scraping is implemented, caller-side analysis should fall back to the
        lightweight opponent-strength score from the fetched horse row.
        """
        if not str(race_id).strip():
            return []
        return []


def get_provider() -> NetkeibaRaceResultProvider:
    return NetkeibaRaceResultProvider()


def fetch_race_entries(race_id: str) -> list[dict[str, Any]]:
    return NetkeibaRaceResultProvider().fetch_race_entries(race_id)


def decode_html(raw: bytes, charset: str | None) -> str:
    encodings = [encoding for encoding in [charset, "euc-jp", "cp932", "utf-8"] if encoding]
    for encoding in encodings:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def extract_horse_id(url: str) -> str:
    match = re.search(r"/horse/(\d+)/?", url)
    return match.group(1) if match else ""


def horse_url_from_id(horse_id: str) -> str:
    return f"{NETKEIBA_DB}/horse/{horse_id}/"


def extract_search_candidates(search_html: str) -> list[tuple[str, str, str]]:
    candidates: list[tuple[str, str, str]] = []
    pattern = re.compile(r"<a[^>]+href=[\"']([^\"']*/horse/(\d+)/?[^\"']*)[\"'][^>]*>(.*?)</a>", re.I | re.S)
    for href, horse_id, label in pattern.findall(search_html):
        name = strip_tags(label)
        if not horse_id or not name:
            continue
        candidates.append((horse_id, name, urllib.parse.urljoin(NETKEIBA_DB, href)))
    return candidates


def extract_page_horse_name(page_html: str) -> str:
    for pattern in [
        r"<h1[^>]*>\s*(.*?)\s*</h1>",
        r"<title[^>]*>\s*(.*?)\s*[|｜]",
    ]:
        match = re.search(pattern, page_html, re.I | re.S)
        if match:
            name = strip_tags(match.group(1))
            if name:
                return name
    return ""


def strip_tags(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", value))).strip()


def names_compatible(input_name: str, fetched_name: str) -> bool:
    normalized_input = normalize_name(input_name)
    normalized_fetched = normalize_name(fetched_name)
    if not normalized_input or not normalized_fetched:
        return False
    return normalized_input in normalized_fetched or normalized_fetched in normalized_input


def normalize_name(value: str) -> str:
    return re.sub(r"[\s\u3000]+", "", str(value)).casefold()


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    flattened = frame.copy()
    flattened.columns = [
        " ".join(str(part) for part in column if str(part) != "nan").strip()
        if isinstance(column, tuple)
        else str(column)
        for column in flattened.columns
    ]
    return flattened


def is_race_history_table(frame: pd.DataFrame) -> bool:
    normalized_columns = {normalize_column(column) for column in frame.columns}
    score = 0
    for keyword in ["日付", "レース名", "着順", "通過", "上り", "距離"]:
        if any(keyword in column for column in normalized_columns):
            score += 1
    return score >= 4


def normalize_race_table(raw_race_df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for _, row in raw_race_df.iterrows():
        record = normalize_race_row(row)
        if record is not None:
            records.append(record)
    return records


def normalize_race_row(row: pd.Series) -> dict[str, Any] | None:
    race_name = pick_cell(row, ["レース名", "race_name", "name"])
    finish = pick_cell(row, ["着順", "finish", "rank"])
    passing_order = pick_cell(row, ["通過", "通過順", "passing_order", "passing"])
    last3f = pick_cell(row, ["上り", "上がり", "last3f", "final_3f"])
    distance_text = pick_cell(row, ["距離", "distance"])
    track_condition = pick_cell(row, ["馬場", "馬場状態", "track_condition"])
    race_time = pick_cell(row, ["タイム", "走破タイム", "time", "race_time", "result_time"])
    winner_time_diff = pick_cell(row, ["勝ち時計との差", "タイム差", "time_diff", "winner_time_diff"])

    if not passing_order or not looks_passing_order(passing_order):
        passing_order = find_passing_order(row)
    if not last3f or parse_float(last3f) <= 0:
        last3f = find_last3f_after_passing(row, passing_order)

    if not all([race_name, finish, passing_order, last3f, distance_text, track_condition]):
        return None

    distance = parse_int(distance_text)
    finish_number = parse_int(finish)
    final_3f = parse_float(last3f)
    if distance <= 0 or finish_number <= 0 or final_3f <= 0:
        return None

    return {
        "date": pick_cell(row, ["日付", "date"]),
        "course": pick_cell(row, ["開催", "競馬場", "course"]),
        "race_name": race_name,
        "race_class": pick_cell(row, ["クラス", "class", "grade"]),
        "popularity": pick_cell(row, ["人気", "人気順", "popularity"]),
        "finish": finish_number,
        "passing_order": passing_order,
        "last3f": final_3f,
        "time": race_time,
        "race_time": race_time,
        "winner_time_diff": winner_time_diff,
        "race_id": pick_cell(row, ["race_id", "レースID"]),
        "distance": distance,
        "surface": parse_surface(distance_text),
        "track_condition": track_condition,
        "field_size": parse_int(
            pick_cell(row, ["頭数", "出走頭数", "field_size", "runners", "number_of_runners"])
        ),
        "margin": parse_float(pick_cell(row, ["着差", "margin"])),
    }


def pick_cell(row: pd.Series, candidates: list[str]) -> str:
    by_normalized_column = {normalize_column(column): column for column in row.index}
    for candidate in candidates:
        normalized_candidate = normalize_column(candidate)
        for normalized_column, original_column in by_normalized_column.items():
            if column_matches(normalized_candidate, normalized_column):
                value = clean_cell(row[original_column])
                if value:
                    return value
    return ""


def column_matches(normalized_candidate: str, normalized_column: str) -> bool:
    if re.fullmatch(r"[A-Za-z0-9_]+", normalized_candidate):
        return normalized_candidate.casefold() == normalized_column.casefold()
    return normalized_candidate in normalized_column


def looks_passing_order(value: Any) -> bool:
    return bool(re.fullmatch(r"\d+(?:-\d+)+", str(value).strip()))


def find_passing_order(row: pd.Series) -> str:
    for value in row.tolist():
        text = clean_cell(value)
        if looks_passing_order(text):
            return text
    return ""


def find_last3f_after_passing(row: pd.Series, passing_order: str) -> str:
    values = [clean_cell(value) for value in row.tolist()]
    start_index = 0
    if passing_order in values:
        start_index = values.index(passing_order) + 1
    for value in values[start_index:]:
        if "-" in value:
            continue
        number = parse_float(value)
        if 30.0 <= number <= 45.0:
            return value
    return ""


def clean_cell(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def normalize_column(value: Any) -> str:
    return re.sub(r"\s+", "", str(value))


def parse_int(value: Any) -> int:
    match = re.search(r"\d+", str(value))
    return int(match.group(0)) if match else 0


def parse_float(value: Any) -> float:
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else 0.0


def parse_surface(distance_text: str) -> str:
    text = str(distance_text)
    if "ダ" in text:
        return "ダート"
    if "障" in text:
        return "障害"
    return "芝"
