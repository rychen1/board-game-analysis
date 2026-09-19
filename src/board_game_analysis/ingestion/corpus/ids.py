"""Load frozen BoardGameGeek thing-id lists."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_CORPUS_ID = "bgg_boardgames_v0"
DEFAULT_CORPUS_VERSION = "0.1.0"
DEFAULT_CORPUS_FILENAME = "bgg_boardgames_v0.txt"


@dataclass(frozen=True)
class CorpusSpec:
    """Packaged corpus definition. Selection methodology lives in the id file."""

    corpus_id: str
    version: str
    filename: str
    jsonl_name: str
    dataset_logical_key: str
    quality_logical_key: str


CORPORA: dict[str, CorpusSpec] = {
    "bgg_boardgames_v0": CorpusSpec(
        corpus_id="bgg_boardgames_v0",
        version="0.1.0",
        filename="bgg_boardgames_v0.txt",
        jsonl_name="corpus_v0.jsonl",
        dataset_logical_key="bga:corpus:v0",
        quality_logical_key="bga:quality/corpus:v0",
    ),
    "bgg_boardgames_v1": CorpusSpec(
        corpus_id="bgg_boardgames_v1",
        version="0.1.0",
        filename="bgg_boardgames_v1.txt",
        jsonl_name="corpus_v1.jsonl",
        dataset_logical_key="bga:corpus:v1",
        quality_logical_key="bga:quality/corpus:v1",
    ),
}


def corpus_spec(corpus_id: str) -> CorpusSpec:
    try:
        return CORPORA[corpus_id]
    except KeyError as exc:
        known = ", ".join(sorted(CORPORA))
        msg = f"unknown corpus_id {corpus_id!r}; expected one of: {known}"
        raise ValueError(msg) from exc


def corpus_path(corpus_id: str = DEFAULT_CORPUS_ID) -> Path:
    return Path(__file__).resolve().parent / corpus_spec(corpus_id).filename


def default_corpus_path() -> Path:
    return corpus_path(DEFAULT_CORPUS_ID)


def load_source_ids(
    path: Path | None = None,
    *,
    corpus_id: str = DEFAULT_CORPUS_ID,
    offset: int = 0,
    limit: int | None = None,
) -> list[str]:
    """Load unique thing ids, skipping comments and blank lines.

    Order is part of the corpus definition. Duplicate ids keep the first
    occurrence. `offset` / `limit` slice the list after uniquing.
    ``path`` overrides the packaged file for ``corpus_id``.
    """
    if offset < 0:
        msg = "offset must be >= 0"
        raise ValueError(msg)
    if limit is not None and limit < 0:
        msg = "limit must be >= 0"
        raise ValueError(msg)
    source = path if path is not None else corpus_path(corpus_id)
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
