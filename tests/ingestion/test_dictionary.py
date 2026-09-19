"""Data dictionary stays aligned with domain models and the committed schema."""

from __future__ import annotations

import json
from pathlib import Path

from board_game_analysis.domain.game import Game
from board_game_analysis.domain.mechanic import Mechanic
from board_game_analysis.domain.source import SourceReference
from board_game_analysis.ingestion.dictionary import (
    GAME_SCHEMA_ID,
    all_field_specs,
    assert_dictionary_matches_models,
    game_field_names,
    game_json_schema,
    mechanic_field_specs,
    source_reference_field_specs,
)

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "game.schema.json"


def test_dictionary_covers_game_mechanic_and_source_fields() -> None:
    assert_dictionary_matches_models()
    assert game_field_names() == set(Game.model_fields)
    assert {field.name for field in mechanic_field_specs()} == set(
        Mechanic.model_fields
    )
    assert {field.name for field in source_reference_field_specs()} == set(
        SourceReference.model_fields
    )


def test_every_field_has_a_missing_data_policy() -> None:
    for spec in all_field_specs():
        assert spec.layer == "observed_fact"
        assert spec.missing_policy.strip()
        assert spec.description.strip()


def test_committed_json_schema_matches_game_model() -> None:
    committed = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    live = game_json_schema()
    assert committed == live
    assert committed["$id"] == GAME_SCHEMA_ID
    assert set(committed["properties"]) == set(Game.model_fields)
    defs = committed.get("$defs", {})
    assert "Mechanic" in defs
    assert "SourceReference" in defs
    assert set(defs["Mechanic"]["properties"]) == set(Mechanic.model_fields)
    assert set(defs["SourceReference"]["properties"]) == set(
        SourceReference.model_fields
    )
