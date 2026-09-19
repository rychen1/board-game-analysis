"""Assemble the corpus-descriptive report and findings text."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from board_game_analysis.analysis.descriptive import (
    ANALYSIS_ID,
    ANALYSIS_VERSION,
    questions,
)
from board_game_analysis.domain.game import Game

CAVEAT = (
    "This is a convenience / coverage corpus, not a probability sample, "
    "not representative of all board games, and not a rank or best-of list. "
    "Numbers describe this file only."
)


def build_report(
    games: list[Game],
    tables: dict[str, Any],
    *,
    corpus_id: str,
    jsonl_name: str,
    corpus_payload_id: str,
    started_at: datetime,
    load_errors: int,
    ingest_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    corpus: dict[str, Any] = {
        "corpus_id": corpus_id,
        "jsonl_name": jsonl_name,
        "n_games": len(games),
        "n_load_errors": load_errors,
        "payload_id": corpus_payload_id,
    }
    if ingest_counts is not None:
        corpus["ingest_counts"] = ingest_counts
    return {
        "analysis_id": ANALYSIS_ID,
        "analysis_version": ANALYSIS_VERSION,
        "layer": "descriptive_observation",
        "caveat": CAVEAT,
        "unsupported": [
            "causal interpretation",
            "population inference",
            "mechanic families (Mechanic.category is unused)",
            "popularity (unmapped)",
            "rating as quality or fun",
        ],
        "questions": questions(),
        "corpus": corpus,
        "generated_at": started_at.isoformat(),
        "tables": tables,
    }


def render_findings(report: dict[str, Any]) -> str:
    tables = report["tables"]
    corpus = report["corpus"]
    profiles = tables["player_profiles"]
    play = tables["play_time"]
    complexity = tables["complexity"]
    mechanics = tables["mechanics"]
    era = tables["era"]
    coverage = {row["field"]: row for row in tables["coverage"]}

    lines = [
        "# Corpus descriptive findings",
        "",
        "Generated from `report.json`. Layer labels are those in",
        "[analysis-corpus-descriptive-v1.md](analysis-corpus-descriptive-v1.md).",
        "",
        CAVEAT,
        "",
        f"- Corpus: `{corpus['corpus_id']}` (`{corpus['jsonl_name']}`)",
        (
            f"- n games in JSONL: **{corpus['n_games']}** "
            f"(load errors: {corpus['n_load_errors']})"
        ),
        *_ingest_lines(corpus),
        f"- Corpus payload_id: `{corpus['payload_id']}`",
        f"- Generated: {report['generated_at']}",
        "",
        "## Coverage",
        "",
        _coverage_line(coverage, "min_players"),
        _coverage_line(coverage, "max_players"),
        _coverage_line(coverage, "min_play_time_minutes"),
        _coverage_line(coverage, "max_play_time_minutes"),
        _coverage_line(coverage, "complexity"),
        _coverage_line(coverage, "popularity"),
        _coverage_line(coverage, "mechanics"),
        "",
        "`Mechanic.category` is not a corpus field; ingest leaves it null.",
        "",
        "## Q1. Player-count profiles (descriptive observation)",
        "",
        "Profiles are exclusive partitions of the **reported** min–max interval.",
        "They are not typical table size.",
        "",
    ]
    counts = profiles["counts"]
    n = profiles["n"]
    for name in ("two_only", "upto_four", "five_plus", "unknown"):
        count = counts[name]
        share = (count / n * 100) if n else 0.0
        lines.append(f"- `{name}`: {count} ({share:.1f}%)")
    span = profiles["player_span_overall"]
    lines.extend(
        [
            "",
            _box_sentence("Player-range width (max − min)", span),
            "",
            "## Q2. Play time by profile (exploratory association)",
            "",
            "Play time is the published BGG range. A midpoint is a proxy only.",
            "",
            (
                f"- Point estimates (min = max): {play['kinds']['point']}; "
                f"ranges: {play['kinds']['range']}; unknown: {play['kinds']['unknown']}"
            ),
            _box_sentence("Reported minimum minutes", play["min_play_time_minutes"]),
            _box_sentence("Reported maximum minutes", play["max_play_time_minutes"]),
            _box_sentence("Range span (minutes)", play["play_time_span"]),
            "",
        ]
    )
    for name in ("two_only", "upto_four", "five_plus"):
        block = play["by_profile"][name]
        if block["n"] == 0:
            continue
        mid = block["play_time_midpoint"]
        lines.append(
            f"- `{name}` (n={block['n']}): midpoint-proxy median "
            f"{_fmt(mid['median'])} min "
            f"(IQR {_fmt(mid['q1'])}–{_fmt(mid['q3'])}); "
            f"span median {_fmt(block['play_time_span']['median'])}"
        )
    lines.extend(
        [
            "",
            "Do not read this as “larger groups cause longer games.” Publisher",
            "time bands, rulebooks, and this convenience sample all confound it.",
            "",
            "## Q3. Complexity / weight (exploratory association)",
            "",
            "BGG `averageweight` is a community poll on a 1–5 scale, present only",
            "when voters exist. It is not a measured cognitive load.",
            "",
            _box_sentence("Complexity", complexity["overall"]),
            "",
        ]
    )
    for name in ("two_only", "upto_four", "five_plus"):
        box = complexity["by_profile"][name]
        if box["n"] == 0:
            continue
        lines.append(
            f"- `{name}`: n={box['n']} median {_fmt(box['median'])} "
            f"(IQR {_fmt(box['q1'])}–{_fmt(box['q3'])})"
        )
    lines.extend(["", "Spearman rank correlations (no p-values):", ""])
    for item in complexity["associations"]:
        rho = item["rho"]
        rho_text = "n/a (n too small or no variation)" if rho is None else f"{rho:.2f}"
        lines.append(f"- {item['x']} vs {item['y']}: n={item['n']}, ρ={rho_text}")
    lines.extend(
        [
            "",
            "A positive play-time / weight rank association in this file would",
            "still not imply that longer games are heavier *because* of time,",
            "nor that the same pattern holds outside this list.",
            "",
            "## Q4. Mechanic labels (exploratory association)",
            "",
            "Labels are multi-label BGG strings. Shares sum to more than 100%.",
            "There is no mechanic-family taxonomy in this ingest.",
            "",
            (
                f"- Distinct labels: {mechanics['n_distinct_mechanics']}; "
                f"empty mechanic lists: {mechanics['n_empty_mechanic_list']}"
            ),
            _box_sentence("Mechanics listed per game", mechanics["mechanics_per_game"]),
            "",
            "Most common labels (share of games listing the name):",
            "",
        ]
    )
    for item in mechanics["prevalence"][:15]:
        lines.append(
            f"- {item['name']}: {item['games']} ({item['share_of_games'] * 100:.1f}%)"
        )
    lines.extend(
        [
            "",
            "Profile lift for labels on at least "
            f"{mechanics['min_games_for_profile_compare']} games is in",
            "`report.json` (`tables.mechanics.profile_lifts`). Lift > 1 means the",
            "label is more common in that profile than the corpus mix. Small",
            "cells are noisy; do not treat lift as a discovery of a mechanic",
            "type that “belongs” to a player count.",
            "",
            "## Era (corpus composition only)",
            "",
            "These bins describe how the convenience list was filled, not the",
            "history of board games.",
            "",
        ]
    )
    for name in (
        "pre_1970",
        "1970_1994",
        "1995_2009",
        "2010_2016",
        "2017_plus",
        "unknown",
    ):
        count = era["counts"][name]
        share = era["shares"][name] * 100
        lines.append(f"- `{name}`: {count} ({share:.1f}%)")
    lines.extend(
        [
            "",
            "## What this does not show",
            "",
            "- No causal effects of player count, time, or mechanics.",
            "- No ranking of games or claims about “the hobby.”",
            "- No extracted interpretations (cooperative, hidden information, …).",
            "- Ratings were not modeled as quality.",
            "",
        ]
    )
    return "\n".join(lines)


def _ingest_lines(corpus: dict[str, Any]) -> list[str]:
    counts = corpus.get("ingest_counts")
    if not isinstance(counts, dict):
        return []
    requested = counts.get("requested")
    ok = counts.get("ok")
    skipped = counts.get("skipped")
    errors = counts.get("errors")
    return [
        (
            f"- Ingest list: requested={requested}, ok={ok}, "
            f"skipped={skipped}, errors={errors}"
        )
    ]


def _coverage_line(coverage: dict[str, Any], field: str) -> str:
    row = coverage[field]
    if row["kind"] == "list":
        empty = row["empty_list"]
        return (
            f"- `{field}`: nonempty={row['present']}, empty_list={empty}, "
            f"null={row['missing_null']}"
        )
    return (
        f"- `{field}`: present={row['present']}, "
        f"null={row['missing_null']} ({row['pct_missing_null']:.1f}%)"
    )


def _box_sentence(label: str, box: dict[str, Any]) -> str:
    if box["n"] == 0:
        return f"- {label}: n=0 (all missing; missing={box['n_missing']})"
    return (
        f"- {label}: n={box['n']} (missing={box['n_missing']}); "
        f"median {_fmt(box['median'])} "
        f"(IQR {_fmt(box['q1'])}–{_fmt(box['q3'])}); "
        f"mean {_fmt(box['mean'])} (SD {_fmt(box['stdev'])}); "
        f"range {_fmt(box['minimum'])}–{_fmt(box['maximum'])}"
    )


def _fmt(value: float | None) -> str:
    if value is None:
        return "—"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}"
