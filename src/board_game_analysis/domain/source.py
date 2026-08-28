"""Provenance for observed facts and extracted claims."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class SourceReference(BaseModel):
    """Where a piece of information came from.

    This is intentionally a citation, not a full source catalog. Later layers
    should be able to trace a claim back to an API record, database row, or
    rulebook page.
    """

    model_config = ConfigDict(extra="forbid")

    source: str
    source_type: str | None = None
    url: str | None = None
    source_identifier: str | None = None
    retrieved_at: datetime | None = None
    page_number: int | None = Field(default=None, ge=1)
