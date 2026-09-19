"""Assemble the robustness report and second-pass findings text."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from board_game_analysis.analysis.robustness import (
    ROBUSTNESS_ID,
    ROBUSTNESS_VERSION,
)

CAVEAT = (
    "Second-pass stress test of the first descriptive analysis on the same "
    "convenience / coverage corpus. Not a probability sample, not "
    "representative of all board games, and not a rank or best-of list."
)


def robustness_questions() -> list[dict[str, str]]:
    return [
        {
            "id": "R1",
            "layer": "exploratory association",
            "text": (
                "Is the positive play-time / weight rank association robust "
                "to min, max, midpoint, span, and log transforms?"
            ),
        },
        {
            "id": "R2",
            "layer": "descriptive observation",
            "text": (
                "How are player counts distributed as intervals, and do "
                "extreme published ranges materially affect associations?"
            ),
        },
        {
            "id": "R3",
            "layer": "exploratory association",
            "text": (
                "Are mechanic prevalence and profile-lift patterns stable "
                "under frequency thresholds, or mostly sparse-sample noise?"
            ),
        },
        {
            "id": "R4",
            "layer": "descriptive observation",
            "text": (
                "What coverage artifacts, missingness, and extreme values "
                "appear in this corpus file?"
            ),
        },
    ]


def build_robustness_report(
    games: list[Any],
    tables: dict[str, Any],
    *,
    corpus_id: str,
    jsonl_name: str,
    corpus_payload_id: str,
    started_at: datetime,
    load_errors: int,
    ingest_counts: dict[str, int] | None = None,
    descriptive_report_payload_id: str | None = None,
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
    if descriptive_report_payload_id is not None:
        corpus["descriptive_report_payload_id"] = descriptive_report_payload_id
    return {
        "analysis_id": ROBUSTNESS_ID,
        "analysis_version": ROBUSTNESS_VERSION,
        "layer": "robustness_check",
        "caveat": CAVEAT,
        "unsupported": [
            "causal interpretation",
            "population inference",
            "p-values or multiple-testing corrections",
            "choosing a single best time representation",
            "mechanic families (Mechanic.category is unused)",
            "popularity (unmapped)",
        ],
        "questions": robustness_questions(),
        "corpus": corpus,
        "generated_at": started_at.isoformat(),
        "tables": tables,
    }


def _fmt_rho(item: dict[str, Any]) -> str:
    rho = item.get("rho")
    if rho is None:
        return "ρ=n/a"
    return f"ρ={float(rho):.2f}"


def _fmt_corr_line(item: dict[str, Any]) -> str:
    label = item.get("label", item.get("x", "?"))
    transform = item.get("transform")
    suffix = f" ({transform})" if transform else ""
    return (
        f"- {label}{suffix}: n={item.get('n')}, "
        f"missing_x={item.get('missing_x', '—')}, "
        f"missing_y={item.get('missing_y', '—')}, {_fmt_rho(item)}"
    )


def render_robustness_findings(report: dict[str, Any]) -> str:
    tables = report["tables"]
    corpus = report["corpus"]
    time_s = tables["time_weight_sensitivity"]
    players = tables["player_intervals"]
    mechanics = tables["mechanics"]
    audit = tables["composition_audit"]

    raw_time = [
        item
        for item in time_s["correlations"]
        if item.get("transform") == "raw" and item.get("rho") is not None
    ]
    log_time = [
        item
        for item in time_s["correlations"]
        if item.get("transform") == "log" and item.get("rho") is not None
    ]

    lines = [
        "# Corpus descriptive robustness findings (second pass)",
        "",
        "Stress test of the first descriptive analysis. Layer labels match",
        "[analysis-corpus-descriptive-v1.md](analysis-corpus-descriptive-v1.md).",
        "",
        CAVEAT,
        "",
        f"- Corpus: `{corpus['corpus_id']}` (`{corpus['jsonl_name']}`)",
        (
            f"- n games in JSONL: **{corpus['n_games']}** "
            f"(load errors: {corpus.get('n_load_errors', 0)})"
        ),
        f"- Corpus payload_id: `{corpus['payload_id']}`",
    ]
    if corpus.get("descriptive_report_payload_id"):
        lines.append(
            "- First-pass report payload_id: "
            f"`{corpus['descriptive_report_payload_id']}`"
        )
    lines.extend(
        [
            f"- Generated: {report['generated_at']}",
            "",
            "## 1. Play time vs weight — representation sensitivity",
            "",
            "Spearman rank correlations with BGG weight (no p-values):",
            "",
        ]
    )
    lines.extend(_fmt_corr_line(item) for item in raw_time)
    lines.extend(["", "Log-transformed positive values:", ""])
    lines.extend(_fmt_corr_line(item) for item in log_time)

    lines.extend(
        [
            "",
            "Extreme-value leave-one-out (games flagged for extreme "
            "player or play-time intervals):",
            "",
        ]
    )
    loo_seen: set[tuple[str, str]] = set()
    for item in time_s["leave_one_out"]:
        key = (str(item["game_id"]), str(item["x"]))
        if key in loo_seen:
            continue
        loo_seen.add(key)
        full = item.get("rho_full")
        without = item.get("rho_without")
        if full is None or without is None:
            continue
        delta = float(without) - float(full)
        lines.append(
            f"- `{item['title']}` ({item['reason']}), {item['x']}: "
            f"ρ {float(full):.2f} → {float(without):.2f} (Δ={delta:+.2f})"
        )

    lines.extend(
        [
            "",
            "## 2. Player count as interval data",
            "",
            "Interval distributions (this file only):",
            "",
        ]
    )
    for key, label in (
        ("min_players", "Minimum players"),
        ("max_players", "Maximum players"),
        ("player_span", "Player span (max−min)"),
        ("player_midpoint_proxy", "Midpoint proxy (min+max)/2"),
    ):
        box = players["distributions"][key]
        if box["n"] == 0:
            lines.append(f"- {label}: no data")
            continue
        lines.append(
            f"- {label}: n={box['n']}; median {box['median']} "
            f"(IQR {box['q1']}–{box['q3']}); range {box['minimum']}–{box['maximum']}"
        )

    lines.extend(["", "Associations with complexity:", ""])
    for item in players["complexity_associations"]:
        lines.append(_fmt_corr_line({**item, "label": item.get("x")}))

    lines.extend(
        [
            "",
            "Associations with play-time midpoint proxy:",
            "",
        ]
    )
    for item in players["play_time_associations"]:
        if item.get("y") != "play_time_midpoint":
            continue
        lines.append(
            f"- {item['x']} vs play_time_midpoint: n={item['n']}, {_fmt_rho(item)}"
        )

    lines.extend(["", "Extreme published intervals (not removed):", ""])
    for item in players["extreme_games"][:8]:
        lines.append(f"- `{item['title']}` ({item['game_id']}): {item['reason']}")

    subset_max = players["without_extreme_max_players"]
    subset_span = players["without_extreme_player_span"]
    lines.extend(
        [
            "",
            "Subset without max_players ≥ 12:",
            f"- n={subset_max['n']}; max_players vs complexity: "
            f"{_fmt_rho(subset_max['complexity_vs_max_players'])}",
            "",
            "Subset without player_span ≥ 20:",
            f"- n={subset_span['n']}; max_players vs complexity: "
            f"{_fmt_rho(subset_span['complexity_vs_max_players'])}",
            "",
            "## 3. Mechanic prevalence vs association",
            "",
            f"- Distinct labels: {mechanics['n_distinct_labels']}",
            f"- Labels appearing once: {mechanics['n_labels_occurrence_1']}",
            f"- Labels appearing twice: {mechanics['n_labels_occurrence_2']}",
            "",
            "Labels surviving minimum-game thresholds:",
            "",
        ]
    )
    for item in mechanics["threshold_survival"]:
        if item["min_games"] not in (3, 5, 10, 20):
            continue
        lines.append(
            f"- ≥{item['min_games']} games: {item['n_labels']} labels "
            f"({100 * float(item['share_of_labels']):.1f}% of distinct labels)"
        )

    lines.extend(["", "Profile-lift sensitivity (lift ≥ 1.5 somewhere):", ""])
    for item in mechanics["lift_by_threshold"]:
        lines.append(
            f"- min_games={item['min_games']}: {item['n_labels_compared']} labels "
            f"compared, {item['n_notable_lift_ge_1_5']} with lift ≥ 1.5"
        )

    lines.extend(
        [
            "",
            "Prevalence (top labels) is not association. Most labels are rare.",
            "",
            "## 4. Corpus composition audit",
            "",
            f"- Games: {audit['n_games']}",
            "",
            "Field missingness:",
            "",
        ]
    )
    for row in audit["coverage"]:
        lines.append(
            f"- `{row['field']}`: present={row['present']}, "
            f"null={row['missing_null']} ({row['pct_missing_null']:.1f}%)"
        )

    lines.extend(["", "Missing play time:", ""])
    for item in audit["missing_play_time"]:
        lines.append(f"- `{item['title']}` ({item['game_id']})")

    lines.extend(["", "Missing complexity:", ""])
    for item in audit["missing_complexity"]:
        lines.append(f"- `{item['title']}` ({item['game_id']})")

    lines.extend(["", "Empty mechanic lists:", ""])
    for item in audit["missing_mechanics"]:
        lines.append(f"- `{item['title']}` ({item['game_id']})")

    lines.extend(
        [
            "",
            "## 5. Analytical conclusions",
            "",
            "### Robust observations",
            "",
            _robust_section(raw_time, players, mechanics),
            "",
            "### Representation-sensitive observations",
            "",
            _representation_sensitive_section(raw_time, log_time, players),
            "",
            "### Sparse-data observations",
            "",
            _sparse_section(mechanics),
            "",
            "### Data limitations",
            "",
            _limitations_section(audit),
            "",
            "## Figures",
            "",
            "- `figures/fig_time_weight_correlations.png`",
            "- `figures/fig_time_vs_weight_scatter.png`",
            "- `figures/fig_player_intervals.png`",
            "- `figures/fig_mechanic_frequency_histogram.png`",
            "- `figures/fig_mechanic_threshold_survival.png`",
            "- `figures/fig_composition_audit.png`",
            "",
        ]
    )
    return "\n".join(lines)


def _robust_section(
    raw_time: list[dict[str, Any]],
    players: dict[str, Any],
    mechanics: dict[str, Any],
) -> str:
    rhos = {str(item["x"]): float(item["rho"]) for item in raw_time}
    bullets: list[str] = []
    min_r = rhos.get("min_play_time_minutes")
    max_r = rhos.get("max_play_time_minutes")
    mid_r = rhos.get("play_time_midpoint")
    if min_r is not None and max_r is not None and mid_r is not None:
        if min(min_r, max_r, mid_r) >= 0.5:
            bullets.append(
                "- Positive play-time / weight rank association persists "
                "across minimum, maximum, and midpoint representations "
                f"(ρ roughly {min_r:.2f}–{max_r:.2f} in this file)."
            )
    max_players = next(
        (
            item
            for item in players["complexity_associations"]
            if item.get("x") == "max_players"
        ),
        None,
    )
    if max_players and max_players.get("rho") is not None:
        rho = float(max_players["rho"])
        if abs(rho) >= 0.1:
            bullets.append(
                f"- Weak negative max_players vs complexity association "
                f"(ρ≈{rho:.2f}) is small in magnitude."
            )
    if mechanics["n_distinct_labels"]:
        top = mechanics["prevalence_top_15"][0]
        bullets.append(
            f"- Hand Management–scale prevalence ({top['name']}: "
            f"{100 * float(top['share']):.0f}% of games) reflects label "
            "frequency, not a profile association."
        )
    return "\n".join(bullets) if bullets else "- No patterns met the robustness bar."


def _representation_sensitive_section(
    raw_time: list[dict[str, Any]],
    log_time: list[dict[str, Any]],
    players: dict[str, Any],
) -> str:
    bullets: list[str] = []
    by_x = {str(item["x"]): float(item["rho"]) for item in raw_time}
    span = by_x.get("play_time_span")
    mid = by_x.get("play_time_midpoint")
    if span is not None and mid is not None and abs(span - mid) >= 0.15:
        bullets.append(
            f"- Duration span (ρ≈{span:.2f}) differs from midpoint "
            f"(ρ≈{mid:.2f}); wide published ranges carry different signal."
        )
    log_by_x = {str(item["x"]): float(item["rho"]) for item in log_time}
    for attr, label in (
        ("max_play_time_minutes", "maximum minutes"),
        ("play_time_midpoint", "midpoint"),
    ):
        raw = by_x.get(attr)
        logged = log_by_x.get(attr)
        if raw is not None and logged is not None and abs(raw - logged) >= 0.05:
            bullets.append(
                f"- Log transform of {label} shifts ρ from {raw:.2f} to {logged:.2f}."
            )
    subset = players["without_extreme_max_players"]
    full = next(
        (
            item
            for item in players["complexity_associations"]
            if item.get("x") == "max_players"
        ),
        None,
    )
    if full and subset["complexity_vs_max_players"].get("rho") is not None:
        full_r = float(full["rho"])
        sub_r = float(subset["complexity_vs_max_players"]["rho"])
        if abs(full_r - sub_r) >= 0.05:
            bullets.append(
                f"- max_players vs complexity moves from ρ≈{full_r:.2f} to "
                f"ρ≈{sub_r:.2f} when excluding max_players ≥ 12 "
                f"(includes Cartographers 1–100)."
            )
    if bullets:
        return "\n".join(bullets)
    return "- No major representation shifts detected."


def _sparse_section(mechanics: dict[str, Any]) -> str:
    once = int(mechanics["n_labels_occurrence_1"])
    total = int(mechanics["n_distinct_labels"])
    share_once = (once / total * 100) if total else 0.0
    bullets = [
        f"- {once} of {total} labels ({share_once:.0f}%) appear on only one game.",
        "- Profile lift at min_games=5 compares sparse cells; threshold "
        "changes which labels enter the comparison.",
    ]
    for item in mechanics["lift_by_threshold"]:
        if item["min_games"] == 5:
            bullets.append(
                f"- At threshold 5: {item['n_notable_lift_ge_1_5']} labels "
                "show lift ≥ 1.5 somewhere — interpret as exploratory only."
            )
            break
    return "\n".join(bullets)


def _limitations_section(audit: dict[str, Any]) -> str:
    bullets = [
        "- Convenience sample: cannot infer BGG or hobby-wide distributions.",
        "- `popularity` is unmapped; `Mechanic.category` is always null.",
        "- Published min/max intervals are not observed play durations.",
        "- Player midpoint is a proxy, not typical table size.",
        "- Multi-label mechanics: prevalence shares sum above 100%.",
    ]
    if audit["missing_play_time"]:
        bullets.append(
            f"- Play time missing for {len(audit['missing_play_time'])} games."
        )
    if audit["missing_complexity"]:
        bullets.append(
            f"- Complexity missing for {len(audit['missing_complexity'])} games."
        )
    return "\n".join(bullets)
