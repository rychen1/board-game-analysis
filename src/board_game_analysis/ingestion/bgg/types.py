"""Source-specific parsed BGG types. Not domain models."""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class RawArtifact:
    """An archived HTTP response before normalization."""

    source: str
    source_identifier: str
    retrieved_at: datetime
    request_url: str
    content_type: str
    http_status: int
    body: str


@dataclass(frozen=True)
class BggLink:
    kind: str
    source_id: str
    value: str


@dataclass(frozen=True)
class BggThing:
    """Parsed BGG `<item>` payload. Missing numeric fields are None, not 0."""

    bgg_id: str
    item_type: str
    primary_name: str
    year_published: int | None
    min_players: int | None
    max_players: int | None
    min_play_time_minutes: int | None
    max_play_time_minutes: int | None
    rating: float | None
    rating_count: int | None
    complexity: float | None
    designers: list[str] = field(default_factory=list)
    publishers: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    mechanics: list[BggLink] = field(default_factory=list)
