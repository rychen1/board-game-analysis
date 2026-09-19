"""Second-pass robustness checks for the corpus-descriptive analysis."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from board_game_analysis.analysis.descriptive import (
    PlayerRow,
    _box_from_optional,
    build_rows,
    paired_spearman,
    spearman_rho,
)
from board_game_analysis.domain.game import Game
from board_game_analysis.ingestion.quality import missingness_rows

ROBUSTNESS_ID = "corpus_descriptive_v1_robustness"
ROBUSTNESS_VERSION = "0.1.0"
ROBUSTNESS_LOGICAL_KEY = "bga:analysis/corpus-descriptive-robustness:v1"

MECHANIC_THRESHOLDS = (1, 2, 3, 5, 10, 20)
EXTREME_MAX_PLAYERS = 12
EXTREME_PLAYER_SPAN = 20
EXTREME_PLAY_TIME_MAX = 300


def player_midpoint(min_players: int | None, max_players: int | None) -> float | None:
    """Midpoint of the reported player interval. A proxy, not table size."""
    if min_players is None or max_players is None:
        return None
    return (min_players + max_players) / 2.0


def log_positive(value: float | int | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    if number <= 0:
        return None
    return math.log(number)


def correlation_row(
    rows: list[PlayerRow],
    x_attr: str,
    y_attr: str = "complexity",
    *,
    exclude_game_ids: set[str] | None = None,
) -> dict[str, object]:
    filtered = rows
    if exclude_game_ids:
        filtered = [row for row in rows if row.game_id not in exclude_game_ids]
    result = paired_spearman(filtered, x_attr, y_attr)
    result["missing_x"] = sum(getattr(row, x_attr) is None for row in filtered)
    result["missing_y"] = sum(getattr(row, y_attr) is None for row in filtered)
    if exclude_game_ids:
        result["excluded_game_ids"] = sorted(exclude_game_ids)
    return result


def time_weight_sensitivity(rows: list[PlayerRow]) -> dict[str, object]:
    measures = [
        ("min_play_time_minutes", "Reported minimum minutes"),
        ("max_play_time_minutes", "Reported maximum minutes"),
        ("play_time_midpoint", "Midpoint proxy (min+max)/2"),
        ("play_time_span", "Duration span (max−min)"),
    ]
    correlations: list[dict[str, object]] = []
    for attr, label in measures:
        row = correlation_row(rows, attr, "complexity")
        row["label"] = label
        row["transform"] = "raw"
        correlations.append(row)
        log_pairs: list[tuple[float, float]] = []
        for row in rows:
            logged = log_positive(getattr(row, attr))
            if logged is None or row.complexity is None:
                continue
            log_pairs.append((logged, row.complexity))
        xs = [left for left, _ in log_pairs]
        ys = [right for _, right in log_pairs]
        correlations.append(
            {
                "x": attr,
                "y": "complexity",
                "label": label,
                "transform": "log",
                "n": len(log_pairs),
                "missing_x": sum(log_positive(getattr(r, attr)) is None for r in rows),
                "missing_y": sum(r.complexity is None for r in rows),
                "rho": spearman_rho(xs, ys),
            }
        )
    outliers = extreme_player_games(rows)
    leave_one_out: list[dict[str, object]] = []
    for item in outliers:
        for attr in (
            "min_play_time_minutes",
            "max_play_time_minutes",
            "play_time_midpoint",
            "play_time_span",
        ):
            full = correlation_row(rows, attr, "complexity")
            game_id = str(item["game_id"])
            dropped = correlation_row(
                rows, attr, "complexity", exclude_game_ids={game_id}
            )
            leave_one_out.append(
                {
                    "game_id": item["game_id"],
                    "title": item["title"],
                    "reason": item["reason"],
                    "x": attr,
                    "rho_full": full["rho"],
                    "rho_without": dropped["rho"],
                    "n_full": full["n"],
                    "n_without": dropped["n"],
                }
            )
    return {
        "correlations": correlations,
        "leave_one_out": leave_one_out,
        "note": (
            "Positive Spearman ρ means heavier games tend toward longer "
            "published times in this file. Log transforms require values > 0."
        ),
    }


def player_interval_analysis(rows: list[PlayerRow]) -> dict[str, object]:
    midpoints = [player_midpoint(row.min_players, row.max_players) for row in rows]
    distributions = {
        "min_players": _box_from_optional([row.min_players for row in rows]),
        "max_players": _box_from_optional([row.max_players for row in rows]),
        "player_span": _box_from_optional([row.player_span for row in rows]),
        "player_midpoint_proxy": _box_from_optional(midpoints),
    }
    time_attrs = (
        "min_play_time_minutes",
        "max_play_time_minutes",
        "play_time_midpoint",
    )
    player_attrs = ("min_players", "max_players", "player_span")
    mid_pairs: list[tuple[float, float]] = []
    for row in rows:
        midpoint = player_midpoint(row.min_players, row.max_players)
        if midpoint is None or row.complexity is None:
            continue
        mid_pairs.append((midpoint, row.complexity))
    complexity_associations = [
        correlation_row(rows, attr, "complexity") for attr in player_attrs
    ]
    complexity_associations.append(
        {
            "x": "player_midpoint_proxy",
            "y": "complexity",
            "n": len(mid_pairs),
            "missing_x": sum(
                player_midpoint(r.min_players, r.max_players) is None for r in rows
            ),
            "missing_y": sum(r.complexity is None for r in rows),
            "rho": spearman_rho(
                [left for left, _ in mid_pairs],
                [right for _, right in mid_pairs],
            ),
        }
    )
    play_time_associations: list[dict[str, object]] = []
    for p_attr in player_attrs + ("player_midpoint_proxy",):
        for t_attr in time_attrs:
            if p_attr == "player_midpoint_proxy":
                pairs: list[tuple[float, float]] = []
                for row in rows:
                    midpoint = player_midpoint(row.min_players, row.max_players)
                    time_val = getattr(row, t_attr)
                    if midpoint is None or time_val is None:
                        continue
                    pairs.append((midpoint, float(time_val)))
                play_time_associations.append(
                    {
                        "x": p_attr,
                        "y": t_attr,
                        "n": len(pairs),
                        "rho": spearman_rho(
                            [left for left, _ in pairs],
                            [right for _, right in pairs],
                        ),
                    }
                )
            else:
                play_time_associations.append(correlation_row(rows, p_attr, t_attr))
    return {
        "distributions": distributions,
        "extreme_games": extreme_player_games(rows),
        "complexity_associations": complexity_associations,
        "play_time_associations": play_time_associations,
        "without_extreme_max_players": _subset_correlations(
            rows,
            lambda r: (
                r.max_players is not None and r.max_players <= EXTREME_MAX_PLAYERS
            ),
        ),
        "without_extreme_player_span": _subset_correlations(
            rows,
            lambda r: (
                r.player_span is not None and r.player_span <= EXTREME_PLAYER_SPAN
            ),
        ),
    }


def _subset_correlations(rows: list[PlayerRow], keep: Any) -> dict[str, object]:
    subset = [row for row in rows if keep(row)]
    return {
        "n": len(subset),
        "complexity_vs_max_players": correlation_row(
            subset, "max_players", "complexity"
        ),
        "complexity_vs_play_time_midpoint": correlation_row(
            subset, "play_time_midpoint", "complexity"
        ),
    }


def extreme_player_games(rows: list[PlayerRow]) -> list[dict[str, object]]:
    flagged: list[dict[str, object]] = []
    for row in rows:
        reasons: list[str] = []
        if row.max_players is not None and row.max_players >= EXTREME_MAX_PLAYERS:
            reasons.append(f"max_players={row.max_players}")
        if row.player_span is not None and row.player_span >= EXTREME_PLAYER_SPAN:
            reasons.append(f"player_span={row.player_span}")
        if (
            row.max_play_time_minutes is not None
            and row.max_play_time_minutes >= EXTREME_PLAY_TIME_MAX
        ):
            reasons.append(f"max_play_time_minutes={row.max_play_time_minutes}")
        if reasons:
            flagged.append(
                {
                    "game_id": row.game_id,
                    "title": row.title,
                    "min_players": row.min_players,
                    "max_players": row.max_players,
                    "player_span": row.player_span,
                    "min_play_time_minutes": row.min_play_time_minutes,
                    "max_play_time_minutes": row.max_play_time_minutes,
                    "complexity": row.complexity,
                    "reason": "; ".join(reasons),
                }
            )

    def _extreme_sort_key(item: dict[str, object]) -> tuple[int, int]:
        max_players = item["max_players"]
        player_span = item["player_span"]
        return (
            -(int(max_players) if isinstance(max_players, int) else 0),
            -(int(player_span) if isinstance(player_span, int) else 0),
        )

    flagged.sort(key=_extreme_sort_key)
    return flagged


def mechanic_robustness(rows: list[PlayerRow]) -> dict[str, object]:
    n = len(rows)
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(row.mechanic_names)
    freq_of_freq = Counter(counts.values())
    frequency_histogram = [
        {"games_listing_label": freq, "n_labels": freq_of_freq[freq]}
        for freq in sorted(freq_of_freq)
    ]
    threshold_survival = []
    for threshold in MECHANIC_THRESHOLDS:
        surviving = sum(1 for count in counts.values() if count >= threshold)
        threshold_survival.append(
            {
                "min_games": threshold,
                "n_labels": surviving,
                "share_of_labels": surviving / len(counts) if counts else 0.0,
            }
        )
    lift_by_threshold: list[dict[str, object]] = []
    for threshold in (3, 5, 10, 20):
        lifts = _profile_lifts(rows, min_games=threshold)
        notable: list[dict[str, object]] = []
        for item in lifts:
            by_profile = item.get("by_profile")
            if not isinstance(by_profile, dict):
                continue
            if any(
                isinstance(block, dict)
                and block.get("lift") is not None
                and float(block["lift"]) >= 1.5
                for block in by_profile.values()
            ):
                notable.append(item)
        lift_by_threshold.append(
            {
                "min_games": threshold,
                "n_labels_compared": len(lifts),
                "n_notable_lift_ge_1_5": len(notable),
                "notable_examples": notable[:5],
            }
        )
    return {
        "n_games": n,
        "n_distinct_labels": len(counts),
        "frequency_histogram": frequency_histogram,
        "n_labels_occurrence_1": freq_of_freq.get(1, 0),
        "n_labels_occurrence_2": freq_of_freq.get(2, 0),
        "threshold_survival": threshold_survival,
        "lift_by_threshold": lift_by_threshold,
        "prevalence_top_15": [
            {"name": name, "games": count, "share": count / n if n else 0.0}
            for name, count in counts.most_common(15)
        ],
        "note": (
            "Prevalence counts games listing a label. Lift compares a "
            "label's profile mix to the corpus profile mix. Small cells "
            "are unstable."
        ),
    }


def _profile_lifts(rows: list[PlayerRow], *, min_games: int) -> list[dict[str, object]]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(row.mechanic_names)
    known = [row for row in rows if row.profile != "unknown"]
    known_n = len(known)
    known_profile_n = Counter(row.profile for row in known)
    lifts: list[dict[str, object]] = []
    for name, count in counts.most_common():
        if count < min_games:
            continue
        holders = [row for row in known if name in row.mechanic_names]
        if len(holders) < min_games:
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
        lifts.append({"name": name, "games": len(holders), "by_profile": by_profile})
    return lifts


def corpus_composition_audit(
    games: list[Game], rows: list[PlayerRow]
) -> dict[str, object]:
    coverage = [
        {
            "field": item.field,
            "present": item.present,
            "missing_null": item.missing_null,
            "pct_missing_null": item.pct_missing_null,
        }
        for item in missingness_rows(games)
        if item.field
        in (
            "release_year",
            "min_players",
            "max_players",
            "min_play_time_minutes",
            "max_play_time_minutes",
            "complexity",
            "popularity",
            "mechanics",
            "rating",
        )
    ]
    per_game_mech = [len(row.mechanic_names) for row in rows]
    return {
        "n_games": len(games),
        "coverage": coverage,
        "release_year": _box_from_optional([row.release_year for row in rows]),
        "complexity": _box_from_optional([row.complexity for row in rows]),
        "min_players": _box_from_optional([row.min_players for row in rows]),
        "max_players": _box_from_optional([row.max_players for row in rows]),
        "player_span": _box_from_optional([row.player_span for row in rows]),
        "min_play_time_minutes": _box_from_optional(
            [row.min_play_time_minutes for row in rows]
        ),
        "max_play_time_minutes": _box_from_optional(
            [row.max_play_time_minutes for row in rows]
        ),
        "play_time_span": _box_from_optional([row.play_time_span for row in rows]),
        "mechanics_per_game": _box_from_optional(list(per_game_mech)),
        "extreme_observations": extreme_player_games(rows),
        "missing_mechanics": [
            {"game_id": row.game_id, "title": row.title}
            for row in rows
            if not row.mechanic_names
        ],
        "missing_play_time": [
            {"game_id": row.game_id, "title": row.title}
            for row in rows
            if row.min_play_time_minutes is None or row.max_play_time_minutes is None
        ],
        "missing_complexity": [
            {"game_id": row.game_id, "title": row.title}
            for row in rows
            if row.complexity is None
        ],
    }


def build_robustness_tables(games: list[Game]) -> dict[str, object]:
    rows = build_rows(games)
    return {
        "time_weight_sensitivity": time_weight_sensitivity(rows),
        "player_intervals": player_interval_analysis(rows),
        "mechanics": mechanic_robustness(rows),
        "composition_audit": corpus_composition_audit(games, rows),
    }
