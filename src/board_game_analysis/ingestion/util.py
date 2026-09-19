"""Helpers shared by the ingestion pipeline."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def try_git_code_ref(cwd: Path | None = None) -> dict[str, Any] | None:
    """Return git sha and dirty flag, or None if Git is unavailable."""
    working = Path.cwd() if cwd is None else cwd
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=working,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty_out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=working,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    if not sha:
        return None
    return {"git_sha": sha, "dirty": bool(dirty_out.strip())}


def chunked[T](items: Sequence[T], size: int) -> list[list[T]]:
    """Split `items` into consecutive chunks of at most `size`."""
    if size < 1:
        msg = "chunk size must be >= 1"
        raise ValueError(msg)
    return [list(items[i : i + size]) for i in range(0, len(items), size)]
