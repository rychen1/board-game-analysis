"""Corpus integrity and missingness summaries for ingested `Game` records.

These helpers describe the processed corpus. They do not score games,
infer quality, or rewrite source data.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

from pydantic import ValidationError

from board_game_analysis.domain.game import Game
from board_game_analysis.ingestion.dictionary import game_field_names

SCALAR_FIELDS: tuple[str, ...] = (
    "id",
    "title",
    "release_year",
    "min_players",
    "max_players",
    "min_play_time_minutes",
    "max_play_time_minutes",
    "popularity",
    "rating",
    "rating_count",
    "complexity",
)

LIST_FIELDS: tuple[str, ...] = (
    "designers",
    "publishers",
    "categories",
    "mechanics",
    "sources",
)

NUMERIC_FIELDS: tuple[str, ...] = (
    "release_year",
    "min_players",
    "max_players",
    "min_play_time_minutes",
    "max_play_time_minutes",
    "rating",
    "rating_count",
    "complexity",
)

BGG_RATING_MIN = 1.0
BGG_RATING_MAX = 10.0
BGG_COMPLEXITY_MIN = 1.0
BGG_COMPLEXITY_MAX = 5.0
YEAR_MIN_PLAUSIBLE = 1800
PUBLISHER_LIST_FLAG = 30
MECHANIC_LIST_FLAG = 15
CATEGORY_LIST_FLAG = 8
PLAYER_COUNT_FLAG = 20
PLAY_TIME_FLAG_MINUTES = 480


@dataclass
class LoadError:
    line_number: int
    reason: str
    raw_id: str | None = None


@dataclass
class CorpusLoad:
    games: list[Game]
    errors: list[LoadError]
    raw_objects: list[dict[str, Any]]


@dataclass
class IntegrityReport:
    n_jsonl_records: int
    n_unique_ids: int
    n_unique_titles: int
    duplicate_ids: list[str]
    duplicate_titles: list[str]
    n_manifest_requested: int | None
    n_manifest_ok: int | None
    n_manifest_skipped: int | None
    n_manifest_errors: int | None
    jsonl_ids_missing_from_manifest: list[str]
    manifest_ok_ids_missing_from_jsonl: list[str]
    load_errors: list[LoadError]
    provenance_issues: list[str]
    expansion_like_titles: list[str]
    manifest_item_types: dict[str, int]


@dataclass
class MissingnessRow:
    field: str
    kind: str
    present: int
    missing_null: int
    empty_list: int | None
    nonempty_list: int | None
    pct_missing_null: float
    pct_empty_list: float | None


@dataclass
class NumericSummary:
    field: str
    count: int
    minimum: float | None
    median: float | None
    mean: float | None
    maximum: float | None
    stdev: float | None


@dataclass
class ListLengthSummary:
    field: str
    count: int
    minimum: int
    median: float
    mean: float
    maximum: int


@dataclass
class Anomaly:
    game_id: str
    title: str
    code: str
    detail: str


def load_games_jsonl(path: Path) -> CorpusLoad:
    """Load JSONL through `Game`. Keep invalid lines instead of dropping them."""
    games: list[Game] = []
    errors: list[LoadError] = []
    raw_objects: list[dict[str, Any]] = []
    text = path.read_text(encoding="utf-8")
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(LoadError(line_number, f"invalid JSON: {exc}"))
            continue
        if not isinstance(raw, dict):
            errors.append(LoadError(line_number, "JSONL record is not an object"))
            continue
        raw_objects.append(raw)
        raw_id = raw.get("id") if isinstance(raw.get("id"), str) else None
        try:
            games.append(Game.model_validate(raw))
        except ValidationError as exc:
            errors.append(
                LoadError(line_number, f"Game validation failed: {exc}", raw_id)
            )
    return CorpusLoad(games=games, errors=errors, raw_objects=raw_objects)


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = "manifest is not a JSON object"
        raise ValueError(msg)
    return payload


def integrity_report(
    loaded: CorpusLoad,
    manifest: dict[str, Any] | None = None,
) -> IntegrityReport:
    ids = [game.id for game in loaded.games]
    titles = [game.title for game in loaded.games]
    id_counts = Counter(ids)
    title_counts = Counter(titles)
    jsonl_ids = set(ids)

    n_requested = n_ok = n_skipped = n_errors = None
    missing_from_manifest: list[str] = []
    missing_from_jsonl: list[str] = []
    item_types: dict[str, int] = {}
    if manifest is not None:
        requested = [str(item) for item in manifest.get("requested_ids", [])]
        ok_rows = list(manifest.get("ok", []))
        n_requested = len(requested)
        n_ok = len(ok_rows)
        n_skipped = len(manifest.get("skipped", []))
        n_errors = len(manifest.get("errors", []))
        manifest_game_ids = {
            str(row["game_id"]) for row in ok_rows if row.get("game_id")
        }
        missing_from_manifest = sorted(jsonl_ids - manifest_game_ids)
        missing_from_jsonl = sorted(manifest_game_ids - jsonl_ids)
        item_types = dict(Counter(str(row.get("item_type")) for row in ok_rows))

    return IntegrityReport(
        n_jsonl_records=len(loaded.raw_objects),
        n_unique_ids=len(jsonl_ids),
        n_unique_titles=len(set(titles)),
        duplicate_ids=sorted(key for key, count in id_counts.items() if count > 1),
        duplicate_titles=sorted(
            key for key, count in title_counts.items() if count > 1
        ),
        n_manifest_requested=n_requested,
        n_manifest_ok=n_ok,
        n_manifest_skipped=n_skipped,
        n_manifest_errors=n_errors,
        jsonl_ids_missing_from_manifest=missing_from_manifest,
        manifest_ok_ids_missing_from_jsonl=missing_from_jsonl,
        load_errors=list(loaded.errors),
        provenance_issues=_provenance_issues(loaded.games),
        expansion_like_titles=_expansion_like_titles(loaded.games),
        manifest_item_types=item_types,
    )


def missingness_rows(games: list[Game]) -> list[MissingnessRow]:
    n = len(games)
    rows: list[MissingnessRow] = []
    for name in SCALAR_FIELDS:
        values = [getattr(game, name) for game in games]
        missing = sum(value is None for value in values)
        present = n - missing
        rows.append(
            MissingnessRow(
                field=name,
                kind="scalar",
                present=present,
                missing_null=missing,
                empty_list=None,
                nonempty_list=None,
                pct_missing_null=_pct(missing, n),
                pct_empty_list=None,
            )
        )
    for name in LIST_FIELDS:
        values = [getattr(game, name) for game in games]
        nulls = sum(value is None for value in values)
        empty = sum(isinstance(value, list) and len(value) == 0 for value in values)
        nonempty = sum(isinstance(value, list) and len(value) > 0 for value in values)
        rows.append(
            MissingnessRow(
                field=name,
                kind="list",
                present=nonempty,
                missing_null=nulls,
                empty_list=empty,
                nonempty_list=nonempty,
                pct_missing_null=_pct(nulls, n),
                pct_empty_list=_pct(empty, n),
            )
        )
    catalog = game_field_names()
    reported = {row.field for row in rows}
    if catalog != reported:
        msg = f"missingness fields drifted from Game: {sorted(catalog ^ reported)}"
        raise AssertionError(msg)
    return rows


def numeric_summaries(games: list[Game]) -> list[NumericSummary]:
    summaries: list[NumericSummary] = []
    for name in NUMERIC_FIELDS:
        values = [
            float(value) for game in games if (value := getattr(game, name)) is not None
        ]
        if not values:
            summaries.append(
                NumericSummary(
                    field=name,
                    count=0,
                    minimum=None,
                    median=None,
                    mean=None,
                    maximum=None,
                    stdev=None,
                )
            )
            continue
        summaries.append(
            NumericSummary(
                field=name,
                count=len(values),
                minimum=min(values),
                median=median(values),
                mean=mean(values),
                maximum=max(values),
                stdev=stdev(values) if len(values) > 1 else 0.0,
            )
        )
    return summaries


def list_length_summaries(games: list[Game]) -> list[ListLengthSummary]:
    summaries: list[ListLengthSummary] = []
    for name in LIST_FIELDS:
        lengths = [len(getattr(game, name)) for game in games]
        summaries.append(
            ListLengthSummary(
                field=name,
                count=len(lengths),
                minimum=min(lengths) if lengths else 0,
                median=float(median(lengths)) if lengths else 0.0,
                mean=float(mean(lengths)) if lengths else 0.0,
                maximum=max(lengths) if lengths else 0,
            )
        )
    return summaries


def category_counts(games: list[Game]) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    for game in games:
        counts.update(game.categories)
    return counts.most_common()


def mechanic_counts(games: list[Game]) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    for game in games:
        counts.update(mechanic.name for mechanic in game.mechanics)
    return counts.most_common()


def flag_anomalies(
    games: list[Game],
    *,
    current_year: int,
) -> list[Anomaly]:
    """Heuristic flags only. Not correction rules and not Game validation."""
    flags: list[Anomaly] = []
    for game in games:
        flags.extend(_flags_for_game(game, current_year=current_year))
    return flags


def _flags_for_game(game: Game, *, current_year: int) -> list[Anomaly]:
    flags: list[Anomaly] = []

    def add(code: str, detail: str) -> None:
        flags.append(
            Anomaly(game_id=game.id, title=game.title, code=code, detail=detail)
        )

    year = game.release_year
    if year is not None and (year < YEAR_MIN_PLAUSIBLE or year > current_year):
        add("implausible_year", f"release_year={year}")
    if (
        game.min_players is not None
        and game.max_players is not None
        and game.min_players > game.max_players
    ):
        add("player_range_inverted", f"{game.min_players}>{game.max_players}")
    if (
        game.min_play_time_minutes is not None
        and game.max_play_time_minutes is not None
        and game.min_play_time_minutes > game.max_play_time_minutes
    ):
        add(
            "play_time_range_inverted",
            f"{game.min_play_time_minutes}>{game.max_play_time_minutes}",
        )
    for name in (
        "min_players",
        "max_players",
        "min_play_time_minutes",
        "max_play_time_minutes",
        "rating_count",
    ):
        value = getattr(game, name)
        if value is not None and value < 0:
            add("negative_value", f"{name}={value}")
        if value == 0 and name != "rating_count":
            add("unexpected_zero", f"{name}={value}")
    if game.max_players is not None and game.max_players > PLAYER_COUNT_FLAG:
        add("large_player_count", f"max_players={game.max_players}")
    if (
        game.max_play_time_minutes is not None
        and game.max_play_time_minutes > PLAY_TIME_FLAG_MINUTES
    ):
        add("long_play_time", f"max_play_time_minutes={game.max_play_time_minutes}")
    if game.rating is not None and not (
        BGG_RATING_MIN <= game.rating <= BGG_RATING_MAX
    ):
        add("rating_off_scale", f"rating={game.rating}")
    if game.complexity is not None and not (
        BGG_COMPLEXITY_MIN <= game.complexity <= BGG_COMPLEXITY_MAX
    ):
        add("complexity_off_scale", f"complexity={game.complexity}")
    if game.rating is None and game.rating_count not in (None, 0):
        add(
            "rating_count_without_rating",
            f"rating is null but rating_count={game.rating_count}",
        )
    if game.rating is not None and game.rating_count in (None, 0):
        add(
            "rating_without_count",
            f"rating={game.rating} rating_count={game.rating_count}",
        )
    if len(game.publishers) >= PUBLISHER_LIST_FLAG:
        add("large_publisher_list", f"publishers={len(game.publishers)}")
    if len(game.mechanics) >= MECHANIC_LIST_FLAG:
        add("large_mechanic_list", f"mechanics={len(game.mechanics)}")
    if len(game.categories) >= CATEGORY_LIST_FLAG:
        add("large_category_list", f"categories={len(game.categories)}")
    return flags


def _provenance_issues(games: list[Game]) -> list[str]:
    issues: list[str] = []
    for game in games:
        if len(game.sources) != 1:
            issues.append(f"{game.id}: expected 1 source, found {len(game.sources)}")
            continue
        source = game.sources[0]
        if source.source != "boardgamegeek":
            issues.append(f"{game.id}: source={source.source!r}")
        if source.source_type != "xmlapi2":
            issues.append(f"{game.id}: source_type={source.source_type!r}")
        if not source.url or "thing?id=" not in source.url:
            issues.append(f"{game.id}: unexpected url={source.url!r}")
        if not source.source_identifier:
            issues.append(f"{game.id}: missing source_identifier")
        elif game.id != f"bgg-{source.source_identifier}":
            issues.append(
                f"{game.id}: id does not match source_identifier="
                f"{source.source_identifier!r}"
            )
        if source.retrieved_at is None:
            issues.append(f"{game.id}: missing retrieved_at")
    return issues


def _expansion_like_titles(games: list[Game]) -> list[str]:
    """Title heuristic only. Not a substitute for BGG item@type."""
    notes: list[str] = []
    markers = ("expansion", "accessory", "promo", "fan expansion")
    for game in games:
        lowered = game.title.lower()
        if any(marker in lowered for marker in markers):
            notes.append(f"{game.id}: {game.title}")
    return notes


def _pct(part: int, whole: int) -> float:
    if whole == 0:
        return 0.0
    return 100.0 * part / whole
