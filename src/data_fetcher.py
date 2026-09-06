from __future__ import annotations

import html
import json
import logging
import re
import urllib.parse
import urllib.error
import urllib.request
from io import StringIO
from typing import Any

import pandas as pd


NETKEIBA_HOME = "https://www.netkeiba.com/"
NETKEIBA_DB = "https://db.netkeiba.com"
HORSE_HISTORY_SCHEMA_VERSION = "horse_history_v2"
logger = logging.getLogger(__name__)


class NetkeibaRaceResultProvider:
    """Fetch real recent-race records from netkeiba and never synthesize races."""

    def __init__(self, timeout: float = 15.0) -> None:
        self.timeout = timeout
        self.last_debug: dict[str, Any] = {}
        self.race_entries_by_name: dict[str, dict[str, Any]] = {}

    def set_race_entries(self, entries: list[dict[str, Any]] | list[Any]) -> None:
        """Allow Prediction to pass horse_id values already parsed from the race card."""
        self.race_entries_by_name = {}
        for entry in entries or []:
            row = dict(entry) if isinstance(entry, dict) else dict(getattr(entry, "__dict__", {}))
            horse_name = str(row.get("horse_name") or row.get("name") or row.get("馬名") or "").strip()
            if not horse_name:
                continue
            self.race_entries_by_name[normalize_name(horse_name)] = row

    def get_recent_results(self, horse_name: str, limit: int = 10) -> list[dict[str, Any]]:
        input_name = horse_name.strip()
        if not input_name:
            raise ValueError("horse_name is empty")

        self._start_debug(input_name)
        entry = self.race_entries_by_name.get(normalize_name(input_name), {})
        entry_horse_id = str(entry.get("horse_id") or entry.get("netkeiba_horse_id") or "").strip()
        if is_valid_horse_id(entry_horse_id):
            fetched_name = str(entry.get("horse_name") or input_name)
            horse_url = str(entry.get("horse_url") or horse_url_from_id(entry_horse_id))
            self._debug_stage(
                "HorseEntry",
                search_result_horse_name=fetched_name,
                horse_id=entry_horse_id,
                url=horse_url,
                horse_id_source="race_entry",
            )
            return self.get_recent_results_by_horse_id(
                entry_horse_id,
                horse_name=input_name,
                fetched_name=fetched_name,
                limit=limit,
                horse_url=horse_url,
            )

        horse_id, fetched_name, horse_url = self.search_horse(input_name)
        self._debug_stage(
            "HorseSearch",
            search_result_horse_name=fetched_name,
            horse_id=horse_id,
            url=horse_url,
            horse_id_source="horse_name_search",
        )
        if not is_valid_horse_id(horse_id):
            raise ValueError(f"horse_id not found: {input_name}")
        if fetched_name and not names_compatible(input_name, fetched_name):
            raise ValueError(f"horse_name mismatch: input={input_name}, fetched={fetched_name}")
        return self.get_recent_results_by_horse_id(
            horse_id,
            horse_name=input_name,
            fetched_name=fetched_name,
            limit=limit,
            horse_url=horse_url,
        )

    def get_recent_results_by_horse_id(
        self,
        horse_id: str,
        horse_name: str = "",
        fetched_name: str = "",
        limit: int = 10,
        horse_url: str = "",
    ) -> list[dict[str, Any]]:
        expected_name = str(horse_name or fetched_name or "").strip()
        input_name = expected_name or str(horse_id).strip()
        if not self.last_debug or self.last_debug.get("input_horse_name") != input_name:
            self._start_debug(input_name)
        if not is_valid_horse_id(horse_id):
            self._debug_stage("HorseIdInvalid", horse_id=horse_id, failure_stage="horse_id")
            raise ValueError(f"invalid horse_id: {horse_id}")

        # The ID, never a caller-supplied race URL, defines the history endpoint.
        horse_url = horse_url_from_id(horse_id)
        self._debug_stage(
            "HorseHistory Request",
            search_result_horse_name=fetched_name or input_name,
            horse_id=str(horse_id),
            url=horse_url,
            horse_url=horse_url,
        )

        page_response = self.request_text_response(horse_url)
        self._debug_stage(
            "HorseHistory Response",
            request_url=horse_url,
            http_status=page_response["status"],
            final_url=page_response["final_url"],
            html_bytes=page_response["html_bytes"],
            title=page_response["title"],
            encoding=page_response["encoding"],
        )
        if page_response["status"] != 200:
            self._debug_stage("HorseHistory Failed", failure_stage="http", error=f"HTTP {page_response['status']}")
            raise ValueError(f"horse history HTTP {page_response['status']}: {horse_url}")
        page_html = str(page_response["text"])
        if extract_horse_id(str(page_response["final_url"])) != str(horse_id):
            self._debug_stage("HorseHistory Failed", failure_stage="redirect", error="unexpected page redirect")
            raise ValueError("horse history redirected to a different page")
        page_name = extract_page_horse_name(page_html) or fetched_name
        if page_name:
            self._debug_stage("HorseHistory Page", search_result_horse_name=page_name, page_title=page_response["title"])
        if expected_name and page_name and not names_compatible(expected_name, page_name):
            self._debug_stage("HorseHistory Failed", failure_stage="identity", error="horse_name mismatch")
            raise ValueError(f"horse_name mismatch: input={expected_name}, page={page_name}")

        raw_race_df = self.read_race_table(page_html, horse_id)
        self._debug_stage("HorseHistory Parse", raw_race_df=raw_race_df, parse_rows=len(raw_race_df))
        recent_races = normalize_race_table(raw_race_df)
        if not recent_races:
            self._debug_stage("HorseHistory Normalize", recent_count=0, failure_stage="normalize")
            raise ValueError(f"race history not found: {input_name}")

        limited = recent_races[:limit]
        self._debug_stage("HorseHistory Normalize", recent_races=limited, recent_count=len(limited))
        return limited

    def search_horse(self, horse_name: str) -> tuple[str, str, str]:
        search_url = f"{NETKEIBA_DB}/?pid=horse_list&word={urllib.parse.quote(horse_name)}"
        search_response = self.request_text_response(search_url)
        self._debug_stage(
            "HorseSearch Request",
            search_url=search_url,
            http_status=search_response["status"],
            final_url=search_response["final_url"],
            html_bytes=search_response["html_bytes"],
            title=search_response["title"],
        )
        if search_response["status"] != 200:
            raise ValueError(f"horse search HTTP {search_response['status']}: {search_url}")
        search_html = str(search_response["text"])
        final_url = str(search_response["final_url"])

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
        response = self.request_text_response(url)
        if response["status"] != 200:
            raise ValueError(f"HTTP {response['status']}: {url}")
        return str(response["text"]), str(response["final_url"])

    def request_text_response(self, url: str) -> dict[str, Any]:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept-Language": "ja,en;q=0.8",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
                final_url = response.geturl()
                charset = response.headers.get_content_charset()
                status = int(getattr(response, "status", 200) or 200)
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            final_url = exc.geturl()
            charset = exc.headers.get_content_charset() if exc.headers else None
            status = int(exc.code)
        except (urllib.error.URLError, OSError) as exc:
            self._debug_stage("HorseHistory Failed", request_url=url, failure_stage="request", error=str(exc))
            raise
        text = decode_html(raw, charset)
        return {
            "text": text,
            "status": status,
            "final_url": final_url,
            "html_bytes": len(raw),
            "title": extract_html_title(text),
            "encoding": charset or "",
        }

    def read_race_table(self, page_html: str, horse_id: str = "") -> pd.DataFrame:
        page_frame = self.find_race_table(page_html)
        if page_frame is not None and not page_frame.empty:
            return page_frame
        if horse_id:
            ajax_url = f"{NETKEIBA_DB}/horse/ajax_horse_results.html?{urllib.parse.urlencode({'input': 'UTF-8', 'output': 'json', 'id': horse_id})}"
            ajax_response = self.request_text_response(ajax_url)
            self._debug_stage(
                "HorseHistory Ajax",
                ajax_url=ajax_url,
                ajax_status=ajax_response["status"],
                ajax_final_url=ajax_response["final_url"],
                ajax_html_bytes=ajax_response["html_bytes"],
            )
            if ajax_response["status"] == 200:
                try:
                    payload = json.loads(str(ajax_response["text"]))
                except json.JSONDecodeError:
                    payload = {}
                    self._debug_stage("HorseHistory AjaxParse", ajax_json_ok=False)
                if payload.get("status") == "OK" and payload.get("data"):
                    try:
                        ajax_frame = self.find_race_table(str(payload["data"]))
                    except ValueError:
                        ajax_frame = None
                    if ajax_frame is not None:
                        return ajax_frame

        self._debug_stage("HorseHistory Failed", failure_stage="parse", error="race history table missing or empty; page structure/access may have changed")
        raise ValueError("race history table not found")

    def find_race_table(self, html_text: str) -> pd.DataFrame | None:
        try:
            tables = pd.read_html(StringIO(html_text))
        except ValueError:
            self._debug_stage("HorseHistory Table", parser_status="no_tables", parse_rows=0, table_count=0)
            return None

        for table in tables:
            frame = flatten_columns(table)
            if is_race_history_table(frame):
                self._debug_stage("HorseHistory Table", parser_status="ok", parse_rows=len(frame), table_count=len(tables))
                return frame
        self._debug_stage("HorseHistory Table", parser_status="table_not_found", parse_rows=0, table_count=len(tables))
        return None

    def get_debug_info(self, horse_name: str | None = None) -> dict[str, Any]:
        return dict(self.last_debug)

    def _start_debug(self, input_name: str) -> None:
        self.last_debug = {
            "schema_version": HORSE_HISTORY_SCHEMA_VERSION,
            "input_horse_name": input_name,
            "search_result_horse_name": "",
            "horse_id": "",
            "horse_id_valid": False,
            "url": "",
            "raw_race_df": pd.DataFrame(),
            "recent_races": [],
            "debug_logs": [],
        }

    def _debug_stage(self, stage: str, **fields: Any) -> None:
        self.last_debug.update(fields)
        if "horse_id" in fields:
            self.last_debug["horse_id_valid"] = is_valid_horse_id(fields.get("horse_id"))
        log_line = _format_debug_log(stage, {**self.last_debug, **fields})
        self.last_debug.setdefault("debug_logs", []).append(log_line)
        logger.info(log_line)

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


def fetch_horse_history(horse_id: str, horse_name: str = "", limit: int = 10) -> list[dict[str, Any]]:
    """Diagnostic entry point for one-horse recent-history fetching."""
    return NetkeibaRaceResultProvider().get_recent_results_by_horse_id(
        horse_id=horse_id,
        horse_name=horse_name,
        limit=limit,
    )


def fetch_race_entries(race_id: str) -> list[dict[str, Any]]:
    return NetkeibaRaceResultProvider().fetch_race_entries(race_id)


def decode_html(raw: bytes, charset: str | None) -> str:
    declared = re.search(br'charset\s*=\s*["\x27]?([\w-]+)', raw[:8192], re.I)
    meta_charset = declared.group(1).decode("ascii") if declared else None
    encodings = [encoding for encoding in [meta_charset, charset, "utf-8", "euc-jp", "cp932"] if encoding]
    for encoding in encodings:
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


def extract_horse_id(url: str) -> str:
    match = re.search(r"/horse/(\d+)/?", url)
    return match.group(1) if match else ""


def is_valid_horse_id(value: Any) -> bool:
    text = str(value).strip()
    return bool(re.fullmatch(r"[0-9]{10}", text)) and int(text) > 0


def horse_url_from_id(horse_id: str) -> str:
    if not is_valid_horse_id(horse_id):
        raise ValueError(f"invalid horse_id: {horse_id}")
    return f"{NETKEIBA_DB}/horse/{horse_id}/"


def extract_html_title(html_text: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", html_text or "", re.I | re.S)
    return strip_tags(match.group(1)) if match else ""


def _format_debug_log(stage: str, fields: dict[str, Any]) -> str:
    if stage == "HorseHistory Request":
        return f"[HorseHistory Request] url={fields.get('url', '')}"
    if stage == "HorseHistory Response":
        return (
            f"[HorseHistory Response] status={fields.get('http_status', '')} "
            f"html_bytes={fields.get('html_bytes', '')} final_url={fields.get('final_url', '')}"
        )
    if stage == "HorseHistory Parse":
        return f"[HorseHistory Parse] rows={fields.get('parse_rows', 0)}"
    if stage in {"HorseSearch", "HorseEntry"}:
        return (
            f"[Horse] name={fields.get('input_horse_name', '')} "
            f"horse_id={fields.get('horse_id', '')} url={fields.get('url', '')}"
        )
    return f"[{stage}] horse_id={fields.get('horse_id', '')} rows={fields.get('parse_rows', '')}"


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
    english_keywords = [
        "date",
        "race_name",
        "finish",
        "rank",
        "passing_order",
        "passing",
        "last3f",
        "final_3f",
        "distance",
        "track_condition",
    ]
    english_score = sum(1 for keyword in english_keywords if any(keyword in column.casefold() for column in normalized_columns))
    if english_score >= 4:
        return True
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
    if not any(normalize_column(column) in {"上り", "上がり", "last3f", "final_3f"} for column in row.index):
        last3f = find_last3f_after_passing(row, passing_order)

    if not all([race_name, finish, distance_text]):
        return None

    distance = parse_int(distance_text)
    finish_number = parse_int(finish)
    final_3f = parse_float(last3f)
    if distance <= 0 or finish_number <= 0:
        return None

    return {
        "date": pick_cell(row, ["日付", "date"]),
        "course": pick_cell(row, ["開催", "競馬場", "course"]),
        "race_name": race_name,
        "race_class": pick_cell(row, ["race_class", "クラス", "class", "grade"]),
        "popularity": pick_cell(row, ["人気", "人気順", "popularity"]),
        "finish": finish_number,
        "passing_order": passing_order,
        "last3f": final_3f if final_3f > 0 else None,
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
