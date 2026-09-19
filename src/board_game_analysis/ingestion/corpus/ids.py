"""Load frozen BoardGameGeek thing-id lists."""

from __future__ import annotations

from pathlib import Path

DEFAULT_CORPUS_ID = "bgg_boardgames_v0"
DEFAULT_CORPUS_VERSION = "0.1.0"
DEFAULT_CORPUS_FILENAME = "bgg_boardgames_v0.txt"


def default_corpus_path() -> Path:
    return Path(__file__).resolve().parent / DEFAULT_CORPUS_FILENAME


def load_source_ids(
    path: Path | None = None,
    *,
    offset: int = 0,
    limit: int | None = None,
) -> list[str]:
    """Load unique thing ids, skipping comments and blank lines.

    Order is part of the corpus definition. Duplicate ids keep the first
    occurrence. `offset` / `limit` slice the list after uniquing.
    """
    if offset < 0:
        msg = "offset must be >= 0"
        raise ValueError(msg)
    if limit is not None and limit < 0:
        msg = "limit must be >= 0"
        raise ValueError(msg)
    source = path or default_corpus_path()
    ids: list[str] = []
    seen: set[str] = set()
    for raw_line in source.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.split("#", 1)[0].strip()
        if not stripped:
            continue
        token = stripped.split()[0]
        if token in seen:
            continue
        seen.add(token)
        ids.append(token)
    sliced = ids[offset:]
    if limit is not None:
        sliced = sliced[:limit]
    return sliced
