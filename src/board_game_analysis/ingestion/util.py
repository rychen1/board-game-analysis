"""Helpers shared by the ingestion pipeline."""

from collections.abc import Sequence


def chunked[T](items: Sequence[T], size: int) -> list[list[T]]:
    """Split `items` into consecutive chunks of at most `size`."""
    if size < 1:
        msg = "chunk size must be >= 1"
        raise ValueError(msg)
    return [list(items[i : i + size]) for i in range(0, len(items), size)]
