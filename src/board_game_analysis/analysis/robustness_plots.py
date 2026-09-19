"""Figures for the corpus-descriptive robustness pass."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from board_game_analysis.analysis.descriptive import PlayerRow
from board_game_analysis.analysis.robustness import player_midpoint


def write_robustness_figures(
    rows: list[PlayerRow],
    tables: dict[str, Any],
    directory: Path,
) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    return [
        _time_weight_correlations(
            tables, directory / "fig_time_weight_correlations.png"
        ),
        _time_vs_weight_scatter(rows, directory / "fig_time_vs_weight_scatter.png"),
        _player_intervals(rows, directory / "fig_player_intervals.png"),
        _mechanic_frequency(tables, directory / "fig_mechanic_frequency_histogram.png"),
        _mechanic_thresholds(tables, directory / "fig_mechanic_threshold_survival.png"),
        _composition_audit(rows, directory / "fig_composition_audit.png"),
    ]


def _time_weight_correlations(tables: dict[str, Any], path: Path) -> Path:
    items = [
        item
        for item in tables["time_weight_sensitivity"]["correlations"]
        if item.get("transform") == "raw" and item.get("rho") is not None
    ]
    labels = [str(item["label"]).replace("Reported ", "") for item in items]
    rhos = [float(item["rho"]) for item in items]
    figure, axes = plt.subplots(figsize=(7.0, 4.0))
    colors = ["#4C6F7A" if rho >= 0 else "#C47B2B" for rho in rhos]
    axes.barh(labels, rhos, color=colors)
    axes.axvline(0, color="#666666", linewidth=0.8)
    axes.set_xlabel("Spearman ρ with BGG weight")
    axes.set_title("Play-time representation vs weight (raw)")
    axes.set_xlim(-1, 1)
    _caption(axes, "Exploratory rank association in this corpus only. No p-values.")
    return _save(figure, path)


def _time_vs_weight_scatter(rows: list[PlayerRow], path: Path) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(8.5, 7.0), sharey=True)
    panels = (
        ("min_play_time_minutes", "Minimum minutes"),
        ("max_play_time_minutes", "Maximum minutes"),
        ("play_time_midpoint", "Midpoint proxy"),
        ("play_time_span", "Span (max−min)"),
    )
    for axis, (attr, title) in zip(axes.flat, panels, strict=True):
        xs: list[float] = []
        ys: list[float] = []
        for row in rows:
            x_val = getattr(row, attr)
            if x_val is None or row.complexity is None:
                continue
            xs.append(float(x_val))
            ys.append(float(row.complexity))
        axis.scatter(xs, ys, alpha=0.45, s=18, color="#4C6F7A")
        axis.set_title(title)
        axis.set_xlabel("Minutes")
        if axis is axes[0, 0]:
            axis.set_ylabel("averageweight (1–5)")
    figure.suptitle("BGG weight vs play-time representations")
    figure.tight_layout()
    return _save(figure, path)


def _player_intervals(rows: list[PlayerRow], path: Path) -> Path:
    mins = [float(row.min_players) for row in rows if row.min_players is not None]
    maxs = [float(row.max_players) for row in rows if row.max_players is not None]
    spans = [float(row.player_span) for row in rows if row.player_span is not None]
    mids: list[float] = []
    for row in rows:
        midpoint = player_midpoint(row.min_players, row.max_players)
        if midpoint is not None:
            mids.append(midpoint)
    figure, axes = plt.subplots(figsize=(7.0, 4.0))
    axes.boxplot(
        [mins, maxs, spans, mids],
        tick_labels=["min", "max", "span", "midpoint"],
    )
    axes.set_title("Reported player-count intervals")
    axes.set_ylabel("Players")
    _caption(axes, "Midpoint is a proxy. Cartographers (1–100) is included.")
    return _save(figure, path)


def _mechanic_frequency(tables: dict[str, Any], path: Path) -> Path:
    hist = tables["mechanics"]["frequency_histogram"]
    xs = [item["games_listing_label"] for item in hist]
    ys = [item["n_labels"] for item in hist]
    figure, axes = plt.subplots(figsize=(7.0, 4.0))
    axes.bar(xs, ys, color="#4C6F7A")
    axes.set_xlabel("Games listing the label (count)")
    axes.set_ylabel("Number of distinct labels")
    axes.set_title("Mechanic label frequency distribution")
    _caption(axes, "How many labels appear once, twice, etc.")
    return _save(figure, path)


def _mechanic_thresholds(tables: dict[str, Any], path: Path) -> Path:
    surv = tables["mechanics"]["threshold_survival"]
    xs = [item["min_games"] for item in surv]
    ys = [item["n_labels"] for item in surv]
    figure, axes = plt.subplots(figsize=(6.5, 4.0))
    axes.plot(xs, ys, marker="o", color="#4C6F7A")
    axes.set_xlabel("Minimum games threshold")
    axes.set_ylabel("Labels surviving threshold")
    axes.set_title("Mechanic labels above frequency cutoffs")
    _caption(axes, "Most labels are rare; profile lift is threshold-sensitive.")
    return _save(figure, path)


def _composition_audit(rows: list[PlayerRow], path: Path) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(8.5, 6.5))
    panels: list[tuple[str, list[float]]] = [
        (
            "Release year",
            [float(row.release_year) for row in rows if row.release_year is not None],
        ),
        (
            "BGG weight",
            [float(row.complexity) for row in rows if row.complexity is not None],
        ),
        (
            "Max players",
            [float(row.max_players) for row in rows if row.max_players is not None],
        ),
        (
            "Mechanics/game",
            [float(len(row.mechanic_names)) for row in rows],
        ),
    ]
    for axis, (title, values) in zip(axes.flat, panels, strict=True):
        if not values:
            axis.set_title(f"{title} (no data)")
            axis.axis("off")
            continue
        axis.boxplot([values], tick_labels=[title])
        axis.set_title(title)
    figure.suptitle("Corpus composition audit (this file only)")
    figure.tight_layout()
    return _save(figure, path)


def _caption(axes: Axes, text: str) -> None:
    axes.text(
        0.0,
        -0.22,
        text,
        transform=axes.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        wrap=True,
    )


def _save(figure: Figure, path: Path) -> Path:
    figure.tight_layout()
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return path
