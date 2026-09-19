"""Deterministic descriptive transforms for a Game corpus.

These helpers summarize observed BGG metadata. They do not infer
population parameters, causality, or mechanic families.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import mean, median, quantiles, stdev
from typing import Literal

from board_game_analysis.domain.game import Game
from board_game_analysis.ingestion.quality import missingness_rows

ANALYSIS_ID = "corpus_descriptive_v1"
ANALYSIS_VERSION = "0.1.0"
ANALYSIS_LOGICAL_KEY = "bga:analysis/corpus-descriptive:v1"

PlayerProfile = Literal["two_only", "upto_four", "five_plus", "unknown"]
PlayTimeKind = Literal["point", "range", "unknown"]
EraBin = Literal[
    "pre_1970",
    "1970_1994",
    "1995_2009",
    "2010_2016",
    "2017_plus",
    "unknown",
]

PLAYER_PROFILE_ORDER: tuple[PlayerProfile, ...] = (
    "two_only",
    "upto_four",
    "five_plus",
    "unknown",
)
ERA_ORDER: tuple[EraBin, ...] = (
    "pre_1970",
    "1970_1994",
    "1995_2009",
    "2010_2016",
    "2017_plus",
    "unknown",
)

MIN_SPEARMAN_N = 8
MIN_MECHANIC_FOR_PROFILE = 5


@dataclass(frozen=True)
class NumericBox:
    n: int
    n_missing: int
    minimum: float | None
    q1: float | None
    median: float | None
    q3: float | None
    maximum: float | None
    mean: float | None
    stdev: float | None


@dataclass(frozen=True)
class PlayerRow:
    game_id: str
    title: str
    min_players: int | None
    max_players: int | None
    player_span: int | None
    profile: PlayerProfile
    min_play_time_minutes: int | None
    max_play_time_minutes: int | None
    play_time_span: int | None
    play_time_kind: PlayTimeKind
    play_time_midpoint: float | None
    complexity: float | None
    release_year: int | None
    era: EraBin
    mechanic_names: tuple[str, ...]


def player_profile(min_players: int | None, max_players: int | None) -> PlayerProfile:
    """Exclusive interval profile. Not typical table size."""
    if min_players is None or max_players is None:
        return "unknown"
    if min_players == 2 and max_players == 2:
        return "two_only"
    if max_players >= 5:
        return "five_plus"
    return "upto_four"


def play_time_kind(min_minutes: int | None, max_minutes: int | None) -> PlayTimeKind:
    if min_minutes is None or max_minutes is None:
        return "unknown"
    if min_minutes == max_minutes:
        return "point"
    return "range"


def play_time_midpoint(
    min_minutes: int | None, max_minutes: int | None
) -> float | None:
    """Midpoint of the reported range. Not a measured session length."""
    if min_minutes is None or max_minutes is None:
        return None
    return (min_minutes + max_minutes) / 2.0


def era_bin(year: int | None) -> EraBin:
    if year is None:
        return "unknown"
    if year < 1970:
        return "pre_1970"
    if year < 1995:
        return "1970_1994"
    if year < 2010:
        return "1995_2009"
    if year < 2017:
        return "2010_2016"
    return "2017_plus"


def game_row(game: Game) -> PlayerRow:
    min_p, max_p = game.min_players, game.max_players
    min_t, max_t = game.min_play_time_minutes, game.max_play_time_minutes
    span_p = None if min_p is None or max_p is None else max_p - min_p
    span_t = None if min_t is None or max_t is None else max_t - min_t
    names = tuple(mechanic.name for mechanic in game.mechanics)
    return PlayerRow(
        game_id=game.id,
        title=game.title,
        min_players=min_p,
        max_players=max_p,
        player_span=span_p,
        profile=player_profile(min_p, max_p),
        min_play_time_minutes=min_t,
        max_play_time_minutes=max_t,
        play_time_span=span_t,
        play_time_kind=play_time_kind(min_t, max_t),
        play_time_midpoint=play_time_midpoint(min_t, max_t),
        complexity=game.complexity,
        release_year=game.release_year,
        era=era_bin(game.release_year),
        mechanic_names=names,
    )


def build_rows(games: list[Game]) -> list[PlayerRow]:
    return [game_row(game) for game in games]


def numeric_box(values: list[float], *, n_missing: int = 0) -> NumericBox:
    if not values:
        return NumericBox(
            n=0,
            n_missing=n_missing,
            minimum=None,
            q1=None,
            median=None,
            q3=None,
            maximum=None,
            mean=None,
            stdev=None,
        )
    q1, _, q3 = _quartile_ends(values)
    return NumericBox(
        n=len(values),
        n_missing=n_missing,
        minimum=min(values),
        q1=q1,
        median=float(median(values)),
        q3=q3,
        maximum=max(values),
        mean=float(mean(values)),
        stdev=float(stdev(values)) if len(values) > 1 else 0.0,
    )


def _quartile_ends(values: list[float]) -> tuple[float, float, float]:
    if len(values) == 1:
        only = float(values[0])
        return only, only, only
    parts = quantiles(values, n=4, method="inclusive")
    return float(parts[0]), float(parts[1]), float(parts[2])


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index
        while end + 1 < len(order) and values[order[end + 1]] == values[order[index]]:
            end += 1
        average = (index + end) / 2.0 + 1.0
        for pos in range(index, end + 1):
            ranks[order[pos]] = average
        index = end + 1
    return ranks


def pearson_r(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = mean(xs)
    mean_y = mean(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    den_x = sum((x - mean_x) ** 2 for x in xs)
    den_y = sum((y - mean_y) ** 2 for y in ys)
    if den_x == 0.0 or den_y == 0.0:
        return None
    return num / (den_x * den_y) ** 0.5


def spearman_rho(xs: list[float], ys: list[float]) -> float | None:
    """Exploratory rank association. No p-value. None if n is too small."""
    if len(xs) != len(ys) or len(xs) < MIN_SPEARMAN_N:
        return None
    return pearson_r(_rank(xs), _rank(ys))


def paired_spearman(
    rows: list[PlayerRow],
    x_attr: str,
    y_attr: str,
) -> dict[str, object]:
    pairs = [
        (getattr(row, x_attr), getattr(row, y_attr))
        for row in rows
        if getattr(row, x_attr) is not None and getattr(row, y_attr) is not None
    ]
    xs = [float(left) for left, _ in pairs]
    ys = [float(right) for _, right in pairs]
    return {
        "n": len(pairs),
        "rho": spearman_rho(xs, ys),
        "x": x_attr,
        "y": y_attr,
    }


def coverage_table(games: list[Game]) -> list[dict[str, object]]:
    rows = []
    for item in missingness_rows(games):
        rows.append(
            {
                "field": item.field,
                "kind": item.kind,
                "present": item.present,
                "missing_null": item.missing_null,
                "empty_list": item.empty_list,
                "pct_missing_null": item.pct_missing_null,
                "pct_empty_list": item.pct_empty_list,
            }
        )
    return rows


def count_map(values: list[str], order: tuple[str, ...]) -> dict[str, int]:
    counts = Counter(values)
    return {key: counts.get(key, 0) for key in order}


def profile_table(rows: list[PlayerRow]) -> dict[str, object]:
    n = len(rows)
    counts = count_map([row.profile for row in rows], PLAYER_PROFILE_ORDER)
    by_profile: dict[str, dict[str, object]] = {}
    for profile in PLAYER_PROFILE_ORDER:
        group = [row for row in rows if row.profile == profile]
        by_profile[profile] = {
            "n": len(group),
            "share": (len(group) / n) if n else 0.0,
            "player_span": _box_from_optional([row.player_span for row in group]),
            "min_players": _box_from_optional([row.min_players for row in group]),
            "max_players": _box_from_optional([row.max_players for row in group]),
        }
    return {
        "n": n,
        "counts": counts,
        "player_span_overall": _box_from_optional([row.player_span for row in rows]),
        "by_profile": by_profile,
    }


def play_time_table(rows: list[PlayerRow]) -> dict[str, object]:
    n = len(rows)
    kinds = count_map(
        [row.play_time_kind for row in rows], ("point", "range", "unknown")
    )
    by_profile: dict[str, dict[str, object]] = {}
    for profile in PLAYER_PROFILE_ORDER:
        group = [row for row in rows if row.profile == profile]
        by_profile[profile] = {
            "n": len(group),
            "kinds": count_map(
                [row.play_time_kind for row in group],
                ("point", "range", "unknown"),
            ),
            "min_play_time_minutes": _box_from_optional(
                [row.min_play_time_minutes for row in group]
            ),
            "max_play_time_minutes": _box_from_optional(
                [row.max_play_time_minutes for row in group]
            ),
            "play_time_span": _box_from_optional([row.play_time_span for row in group]),
            "play_time_midpoint": _box_from_optional(
                [row.play_time_midpoint for row in group]
            ),
        }
    return {
        "n": n,
        "kinds": kinds,
        "min_play_time_minutes": _box_from_optional(
            [row.min_play_time_minutes for row in rows]
        ),
        "max_play_time_minutes": _box_from_optional(
            [row.max_play_time_minutes for row in rows]
        ),
        "play_time_span": _box_from_optional([row.play_time_span for row in rows]),
        "by_profile": by_profile,
    }


def complexity_table(rows: list[PlayerRow]) -> dict[str, object]:
    by_profile: dict[str, object] = {}
    for profile in PLAYER_PROFILE_ORDER:
        group = [row for row in rows if row.profile == profile]
        by_profile[profile] = _box_from_optional([row.complexity for row in group])
    return {
        "overall": _box_from_optional([row.complexity for row in rows]),
        "by_profile": by_profile,
        "associations": [
            paired_spearman(rows, "play_time_midpoint", "complexity"),
            paired_spearman(rows, "min_play_time_minutes", "complexity"),
            paired_spearman(rows, "max_play_time_minutes", "complexity"),
            paired_spearman(rows, "max_players", "complexity"),
            paired_spearman(rows, "player_span", "complexity"),
        ],
    }


def era_table(rows: list[PlayerRow]) -> dict[str, object]:
    n = len(rows)
    counts = count_map([row.era for row in rows], ERA_ORDER)
    return {
        "n": n,
        "counts": counts,
        "shares": {key: (counts[key] / n) if n else 0.0 for key in ERA_ORDER},
        "release_year": _box_from_optional([row.release_year for row in rows]),
    }


def mechanic_table(rows: list[PlayerRow]) -> dict[str, object]:
    n = len(rows)
    counts: Counter[str] = Counter()
    per_game = [len(row.mechanic_names) for row in rows]
    empty = sum(length == 0 for length in per_game)
    for row in rows:
        counts.update(row.mechanic_names)
    prevalence = [
        {
            "name": name,
            "games": count,
            "share_of_games": count / n if n else 0.0,
        }
        for name, count in counts.most_common()
    ]
    profile_n = Counter(row.profile for row in rows)
    known = [row for row in rows if row.profile != "unknown"]
    known_n = len(known)
    known_profile_n = Counter(row.profile for row in known)
    lifts: list[dict[str, object]] = []
    for name, count in counts.most_common():
        if count < MIN_MECHANIC_FOR_PROFILE:
            continue
        holders = [row for row in known if name in row.mechanic_names]
        if len(holders) < MIN_MECHANIC_FOR_PROFILE:
            continue
        by_profile: dict[str, dict[str, float | int | None]] = {}
        for profile in ("two_only", "upto_four", "five_plus"):
            baseline = known_profile_n[profile] / known_n if known_n else 0.0
            in_profile = sum(1 for row in holders if row.profile == profile)
            share = in_profile / len(holders) if holders else 0.0
            lift = (share / baseline) if baseline else None
            by_profile[profile] = {
                "games_with_mechanic": in_profile,
                "share_among_mechanic": share,
                "baseline_share": baseline,
                "lift": lift,
            }
        lifts.append(
            {
                "name": name,
                "games": len(holders),
                "by_profile": by_profile,
            }
        )
    return {
        "n_games": n,
        "n_empty_mechanic_list": empty,
        "mechanics_per_game": _box_from_optional(list(per_game)),
        "n_distinct_mechanics": len(counts),
        "prevalence": prevalence,
        "profile_n": dict(profile_n),
        "profile_lifts": lifts,
        "min_games_for_profile_compare": MIN_MECHANIC_FOR_PROFILE,
        "note": (
            "Mechanics are multi-label source strings. Shares can sum "
            "above 1. Mechanic.category is unused in BGG ingest."
        ),
    }


def _box_from_optional(values: list[int | float | None]) -> dict[str, object]:
    present = [float(value) for value in values if value is not None]
    box = numeric_box(present, n_missing=len(values) - len(present))
    return {
        "n": box.n,
        "n_missing": box.n_missing,
        "minimum": box.minimum,
        "q1": box.q1,
        "median": box.median,
        "q3": box.q3,
        "maximum": box.maximum,
        "mean": box.mean,
        "stdev": box.stdev,
    }


def questions() -> list[dict[str, str]]:
    return [
        {
            "id": "q1_player_profiles",
            "layer": "descriptive_observation",
            "question": (
                "What player-count profiles appear in this corpus, and how "
                "wide are the reported player ranges?"
            ),
        },
        {
            "id": "q2_play_time_by_profile",
            "layer": "exploratory_association",
            "question": (
                "How does reported play time (min, max, span; point vs range) "
                "vary across player-count profiles?"
            ),
        },
        {
            "id": "q3_complexity_associations",
            "layer": "exploratory_association",
            "question": (
                "How does BGG weight (complexity) vary with player-count "
                "profile and reported play time?"
            ),
        },
        {
            "id": "q4_mechanic_prevalence_and_profile",
            "layer": "exploratory_association",
            "question": (
                "Which BGG mechanic labels are most common, and do common "
                "labels appear at different rates across player-count profiles?"
            ),
        },
    ]


def build_tables(games: list[Game]) -> dict[str, object]:
    rows = build_rows(games)
    return {
        "coverage": coverage_table(games),
        "era": era_table(rows),
        "player_profiles": profile_table(rows),
        "play_time": play_time_table(rows),
        "complexity": complexity_table(rows),
        "mechanics": mechanic_table(rows),
    }


def top_mechanic_names(
    tables: dict[str, object], *, limit: int = 15
) -> list[tuple[str, int]]:
    mechanics = tables["mechanics"]
    if not isinstance(mechanics, dict):
        return []
    prevalence = mechanics.get("prevalence", [])
    if not isinstance(prevalence, list):
        return []
    out: list[tuple[str, int]] = []
    for item in prevalence[:limit]:
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            games = item.get("games")
            if isinstance(games, int):
                out.append((item["name"], games))
    return out


def profile_groups(rows: list[PlayerRow]) -> dict[PlayerProfile, list[PlayerRow]]:
    groups: dict[PlayerProfile, list[PlayerRow]] = defaultdict(list)
    for row in rows:
        groups[row.profile].append(row)
    return groups
