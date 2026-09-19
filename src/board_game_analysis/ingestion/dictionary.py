"""Canonical field catalog for ingested `Game` records.

This is the machine-readable data dictionary. The prose version is
`docs/data-dictionary.md`. Values here are observed facts only.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from board_game_analysis.domain.game import Game
from board_game_analysis.domain.mechanic import Mechanic
from board_game_analysis.domain.source import SourceReference

LAYER_OBSERVED_FACT = "observed_fact"
GAME_SCHEMA_ID = (
    "https://github.com/rychen1/board-game-analysis/schemas/game.schema.json"
)


class FieldSpec(BaseModel):
    """One canonical field and how ingestion treats it."""

    model_config = ConfigDict(extra="forbid")

    model: str
    name: str
    path: str
    python_type: str
    nullable: bool
    layer: str = LAYER_OBSERVED_FACT
    description: str
    bgg_source: str | None = None
    missing_policy: str
    notes: str | None = None


def game_field_specs() -> tuple[FieldSpec, ...]:
    return _GAME_FIELDS


def mechanic_field_specs() -> tuple[FieldSpec, ...]:
    return _MECHANIC_FIELDS


def source_reference_field_specs() -> tuple[FieldSpec, ...]:
    return _SOURCE_FIELDS


def all_field_specs() -> tuple[FieldSpec, ...]:
    return _GAME_FIELDS + _MECHANIC_FIELDS + _SOURCE_FIELDS


def game_field_names() -> set[str]:
    return {field.name for field in _GAME_FIELDS}


def game_json_schema() -> dict[str, Any]:
    """JSON Schema for interchange, aligned with `Game.model_json_schema()`."""
    schema = Game.model_json_schema()
    return {
        "$id": GAME_SCHEMA_ID,
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        **schema,
    }


def assert_dictionary_matches_models() -> None:
    """Raise if the dictionary drifts from the domain models."""
    _assert_fields("Game", Game, game_field_names())
    _assert_fields(
        "Mechanic",
        Mechanic,
        {field.name for field in _MECHANIC_FIELDS},
    )
    _assert_fields(
        "SourceReference",
        SourceReference,
        {field.name for field in _SOURCE_FIELDS},
    )


def _assert_fields(
    label: str,
    model: type[BaseModel],
    catalog: set[str],
) -> None:
    model_fields = set(model.model_fields)
    missing = model_fields - catalog
    extra = catalog - model_fields
    if missing or extra:
        msg = (
            f"data dictionary mismatch for {label}: "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )
        raise AssertionError(msg)


_GAME_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec(
        model="Game",
        name="id",
        path="Game.id",
        python_type="str",
        nullable=False,
        description="Canonical game identifier.",
        bgg_source="item@id, prefixed with 'bgg-'",
        missing_policy="required; ingest fails without a BGG item id",
    ),
    FieldSpec(
        model="Game",
        name="title",
        path="Game.title",
        python_type="str",
        nullable=False,
        description="Primary display title as reported by the source.",
        bgg_source="name[@type=primary]/@value",
        missing_policy="required; ingest fails if no name node is present",
    ),
    FieldSpec(
        model="Game",
        name="release_year",
        path="Game.release_year",
        python_type="int | None",
        nullable=True,
        description="Calendar year of first publication as reported by the source.",
        bgg_source="yearpublished/@value",
        missing_policy=(
            "omitted node or BGG sentinel 0 → null; do not infer from other dates"
        ),
    ),
    FieldSpec(
        model="Game",
        name="min_players",
        path="Game.min_players",
        python_type="int | None",
        nullable=True,
        description="Minimum player count on the box / listing.",
        bgg_source="minplayers/@value",
        missing_policy="omitted or 0 → null; never default to 1",
    ),
    FieldSpec(
        model="Game",
        name="max_players",
        path="Game.max_players",
        python_type="int | None",
        nullable=True,
        description="Maximum player count on the box / listing.",
        bgg_source="maxplayers/@value",
        missing_policy="omitted or 0 → null",
        notes="Rejected if both bounds are present and min > max.",
    ),
    FieldSpec(
        model="Game",
        name="min_play_time_minutes",
        path="Game.min_play_time_minutes",
        python_type="int | None",
        nullable=True,
        description="Lower bound of listed play time in minutes.",
        bgg_source=(
            "minplaytime/@value, else playingtime/@value if both min and max missing"
        ),
        missing_policy="omitted or 0 → null; never default to 0 minutes",
    ),
    FieldSpec(
        model="Game",
        name="max_play_time_minutes",
        path="Game.max_play_time_minutes",
        python_type="int | None",
        nullable=True,
        description="Upper bound of listed play time in minutes.",
        bgg_source=(
            "maxplaytime/@value, else playingtime/@value if both min and max missing"
        ),
        missing_policy="omitted or 0 → null",
    ),
    FieldSpec(
        model="Game",
        name="popularity",
        path="Game.popularity",
        python_type="float | None",
        nullable=True,
        description="Optional popularity signal. Not filled from BGG in v0.",
        bgg_source=None,
        missing_policy=(
            "always null for BGG v0 "
            "(rank is inverted; owned count is a different concept)"
        ),
    ),
    FieldSpec(
        model="Game",
        name="rating",
        path="Game.rating",
        python_type="float | None",
        nullable=True,
        description="Source user-average rating. Not Geek Rating / Bayes average.",
        bgg_source="statistics/ratings/average/@value",
        missing_policy=(
            "null when usersrated is omitted or 0; "
            "do not store BGG placeholder 0.0 as a rating"
        ),
    ),
    FieldSpec(
        model="Game",
        name="rating_count",
        path="Game.rating_count",
        python_type="int | None",
        nullable=True,
        description=(
            "Number of ratings behind `rating`. Distinct from the rating value."
        ),
        bgg_source="statistics/ratings/usersrated/@value",
        missing_policy="omitted → null; 0 is a known zero, not missing",
    ),
    FieldSpec(
        model="Game",
        name="complexity",
        path="Game.complexity",
        python_type="float | None",
        nullable=True,
        description="Source weight / complexity score (BGG averageweight, 1–5).",
        bgg_source="statistics/ratings/averageweight/@value",
        missing_policy="null when numweights is omitted or 0",
        notes="Scale is source-specific; do not assume it matches other sites.",
    ),
    FieldSpec(
        model="Game",
        name="designers",
        path="Game.designers",
        python_type="list[str]",
        nullable=False,
        description="Designer names in source order.",
        bgg_source="link[@type=boardgamedesigner]/@value",
        missing_policy="no links → empty list, not null",
    ),
    FieldSpec(
        model="Game",
        name="publishers",
        path="Game.publishers",
        python_type="list[str]",
        nullable=False,
        description="Publisher names in source order (often many).",
        bgg_source="link[@type=boardgamepublisher]/@value",
        missing_policy="no links → empty list",
    ),
    FieldSpec(
        model="Game",
        name="categories",
        path="Game.categories",
        python_type="list[str]",
        nullable=False,
        description="Source-reported category labels. Not interpretations.",
        bgg_source="link[@type=boardgamecategory]/@value",
        missing_policy="no links → empty list",
        notes="Must not be copied onto ExtractedInterpretation.",
    ),
    FieldSpec(
        model="Game",
        name="mechanics",
        path="Game.mechanics",
        python_type="list[Mechanic]",
        nullable=False,
        description="Source-reported mechanic labels with stable source ids.",
        bgg_source="link[@type=boardgamemechanic] → Mechanic(id=bgg-{id}, name=@value)",
        missing_policy="no links → empty list",
        notes="Not a controlled taxonomy. Nested fields are listed under Mechanic.",
    ),
    FieldSpec(
        model="Game",
        name="sources",
        path="Game.sources",
        python_type="list[SourceReference]",
        nullable=False,
        description="Provenance for the record as a whole.",
        bgg_source="request URL, thing id, retrieval timestamp",
        missing_policy="BGG ingest always writes one SourceReference",
        notes=(
            "retrieved_at is provenance, not a game attribute. Nested fields "
            "are listed under SourceReference."
        ),
    ),
)

_MECHANIC_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec(
        model="Mechanic",
        name="id",
        path="Mechanic.id",
        python_type="str",
        nullable=False,
        description="Stable mechanic identifier within this corpus.",
        bgg_source="link[@type=boardgamemechanic]/@id, prefixed with 'bgg-'",
        missing_policy="required for each mechanic link that has a name",
    ),
    FieldSpec(
        model="Mechanic",
        name="name",
        path="Mechanic.name",
        python_type="str",
        nullable=False,
        description="Source-reported mechanic label.",
        bgg_source="link[@type=boardgamemechanic]/@value",
        missing_policy="links with empty names are dropped",
        notes="Not a taxonomy. The same name may appear on many games.",
    ),
    FieldSpec(
        model="Mechanic",
        name="category",
        path="Mechanic.category",
        python_type="str | None",
        nullable=True,
        description="Optional free-form grouping. Not filled from BGG in v0.",
        bgg_source=None,
        missing_policy="always null for BGG v0",
    ),
    FieldSpec(
        model="Mechanic",
        name="description",
        path="Mechanic.description",
        python_type="str | None",
        nullable=True,
        description="Optional prose description. Not filled from BGG in v0.",
        bgg_source=None,
        missing_policy="always null for BGG v0",
    ),
)

_SOURCE_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec(
        model="SourceReference",
        name="source",
        path="SourceReference.source",
        python_type="str",
        nullable=False,
        description="Short name of the originating system.",
        bgg_source="literal 'boardgamegeek'",
        missing_policy="required",
    ),
    FieldSpec(
        model="SourceReference",
        name="source_type",
        path="SourceReference.source_type",
        python_type="str | None",
        nullable=True,
        description="Kind of source document or API.",
        bgg_source="literal 'xmlapi2'",
        missing_policy="BGG ingest always sets xmlapi2",
    ),
    FieldSpec(
        model="SourceReference",
        name="url",
        path="SourceReference.url",
        python_type="str | None",
        nullable=True,
        description="Request URL used to retrieve the record.",
        bgg_source="GET /thing?id=...&stats=1 (may be a batched URL)",
        missing_policy="BGG ingest always sets the request URL",
    ),
    FieldSpec(
        model="SourceReference",
        name="source_identifier",
        path="SourceReference.source_identifier",
        python_type="str | None",
        nullable=True,
        description="Identifier in the source system (BGG thing id).",
        bgg_source="item@id",
        missing_policy="BGG ingest always sets the thing id",
    ),
    FieldSpec(
        model="SourceReference",
        name="retrieved_at",
        path="SourceReference.retrieved_at",
        python_type="datetime | None",
        nullable=True,
        description="When the raw artifact was fetched. Provenance only.",
        bgg_source="local fetch timestamp",
        missing_policy="BGG ingest always sets a timezone-aware timestamp",
        notes="Not a game attribute. Same XML + same timestamp → identical Game.",
    ),
    FieldSpec(
        model="SourceReference",
        name="page_number",
        path="SourceReference.page_number",
        python_type="int | None",
        nullable=True,
        description="Rulebook page citation. Unused for BGG API ingest.",
        bgg_source=None,
        missing_policy="always null for BGG v0",
    ),
)
