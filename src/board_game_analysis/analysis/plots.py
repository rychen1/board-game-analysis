"""Matplotlib figures for the corpus-descriptive pass."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from board_game_analysis.analysis.descriptive import (
    PLAYER_PROFILE_ORDER,
    PlayerProfile,
    PlayerRow,
    profile_groups,
)

PROFILE_LABELS = {
    "two_only": "2-only",
    "upto_four": "up to 4",
    "five_plus": "5+",
    "unknown": "unknown",
}
PROFILE_COLORS = {
    "two_only": "#4C6F7A",
    "upto_four": "#C47B2B",
    "five_plus": "#6B4C7A",
    "unknown": "#7A7A7A",
}


def write_figures(
    rows: list[PlayerRow],
    tables: dict[str, Any],
    directory: Path,
) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = [
        _player_profiles(tables, directory / "fig_player_profiles.png"),
        _play_time_by_profile(rows, directory / "fig_play_time_by_profile.png"),
        _complexity_by_profile(rows, directory / "fig_complexity_by_profile.png"),
        _complexity_vs_time(rows, directory / "fig_complexity_vs_play_time.png"),
        _top_mechanics(tables, directory / "fig_top_mechanics.png"),
    ]
    return paths


def _player_profiles(tables: dict[str, Any], path: Path) -> Path:
    counts = tables["player_profiles"]["counts"]
    labels = [PROFILE_LABELS[name] for name in PLAYER_PROFILE_ORDER]
    values = [counts[name] for name in PLAYER_PROFILE_ORDER]
    colors = [PROFILE_COLORS[name] for name in PLAYER_PROFILE_ORDER]
    figure, axes = plt.subplots(figsize=(6.5, 3.8))
    axes.bar(labels, values, color=colors)
    axes.set_title("Player-count profiles in this corpus")
    axes.set_ylabel("Games")
    axes.set_xlabel("Reported interval profile")
    _annotate_bars(axes, values)
    _caption(
        axes,
        "Exclusive partition of min–max players. Not typical table size.",
    )
    return _save(figure, path)


def _play_time_by_profile(rows: list[PlayerRow], path: Path) -> Path:
    names: list[PlayerProfile] = [
        name
        for name in ("two_only", "upto_four", "five_plus")
        if any(row.profile == name for row in rows)
    ]
    groups = profile_groups(rows)
    mins = [
        [
            float(row.min_play_time_minutes)
            for row in groups[name]
            if row.min_play_time_minutes is not None
        ]
        for name in names
    ]
    maxs = [
        [
            float(row.max_play_time_minutes)
            for row in groups[name]
            if row.max_play_time_minutes is not None
        ]
        for name in names
    ]
    kept = [
        (label, lo, hi)
        for label, lo, hi in zip(
            [PROFILE_LABELS[name] for name in names], mins, maxs, strict=True
        )
        if lo and hi
    ]
    labels = [item[0] for item in kept]
    mins = [item[1] for item in kept]
    maxs = [item[2] for item in kept]
    if not labels:
        figure, axes = plt.subplots(figsize=(6.5, 3.2))
        axes.set_title("Play-time ranges by player-count profile")
        axes.text(0.5, 0.5, "No play-time values in known profiles", ha="center")
        axes.axis("off")
        return _save(figure, path)
    figure, axes = plt.subplots(1, 2, figsize=(8.5, 4.0), sharey=True)
    axes[0].boxplot(mins, tick_labels=labels)
    axes[0].set_title("Reported minimum minutes")
    axes[0].set_ylabel("Minutes")
    axes[1].boxplot(maxs, tick_labels=labels)
    axes[1].set_title("Reported maximum minutes")
    figure.suptitle("Play-time ranges by player-count profile")
    _caption(
        axes[0],
        "Boxes are the published range endpoints, not timed sessions.",
    )
    figure.tight_layout()
    return _save(figure, path)


def _complexity_by_profile(rows: list[PlayerRow], path: Path) -> Path:
    names: list[PlayerProfile] = [
        name
        for name in ("two_only", "upto_four", "five_plus")
        if any(row.complexity is not None and row.profile == name for row in rows)
    ]
    groups = profile_groups(rows)
    data = [
        [float(row.complexity) for row in groups[name] if row.complexity is not None]
        for name in names
    ]
    figure, axes = plt.subplots(figsize=(6.5, 3.8))
    axes.boxplot(data, tick_labels=[PROFILE_LABELS[name] for name in names])
    axes.set_title("BGG weight by player-count profile")
    axes.set_ylabel("averageweight (1–5)")
    axes.set_xlabel("Reported interval profile")
    _caption(axes, "Community poll. Missing weights are omitted.")
    return _save(figure, path)


def _complexity_vs_time(rows: list[PlayerRow], path: Path) -> Path:
    figure, axes = plt.subplots(figsize=(6.8, 4.2))
    for name in ("two_only", "upto_four", "five_plus"):
        xs: list[float] = []
        ys: list[float] = []
        xerr_lo: list[float] = []
        xerr_hi: list[float] = []
        for row in rows:
            if row.profile != name:
                continue
            if row.complexity is None or row.play_time_midpoint is None:
                continue
            if row.min_play_time_minutes is None or row.max_play_time_minutes is None:
                continue
            mid = row.play_time_midpoint
            xs.append(mid)
            ys.append(row.complexity)
            xerr_lo.append(mid - row.min_play_time_minutes)
            xerr_hi.append(row.max_play_time_minutes - mid)
        if not xs:
            continue
        axes.errorbar(
            xs,
            ys,
            xerr=[xerr_lo, xerr_hi],
            fmt="o",
            color=PROFILE_COLORS[name],
            ecolor=PROFILE_COLORS[name],
            alpha=0.65,
            label=PROFILE_LABELS[name],
            markersize=4,
            elinewidth=0.8,
            capsize=2,
        )
    axes.set_title("BGG weight vs reported play-time range")
    axes.set_xlabel("Play-time midpoint (minutes); whiskers are min–max")
    axes.set_ylabel("averageweight (1–5)")
    axes.legend(title="Profile", frameon=False)
    _caption(
        axes,
        "Whiskers preserve the published time range. Midpoint is a proxy.",
    )
    return _save(figure, path)


def _top_mechanics(tables: dict[str, Any], path: Path) -> Path:
    prevalence = tables["mechanics"]["prevalence"][:15]
    names = [item["name"] for item in reversed(prevalence)]
    values = [item["games"] for item in reversed(prevalence)]
    figure, axes = plt.subplots(figsize=(7.2, 5.4))
    axes.barh(names, values, color="#4C6F7A")
    axes.set_title("Most common BGG mechanic labels")
    axes.set_xlabel("Games listing the label (multi-label)")
    _caption(
        axes,
        "A game can list many labels. Not a taxonomy and not exclusive classes.",
    )
    figure.tight_layout()
    return _save(figure, path)


def _annotate_bars(axes: Axes, values: list[int]) -> None:
    for index, value in enumerate(values):
        axes.text(index, value, str(value), ha="center", va="bottom", fontsize=8)


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
