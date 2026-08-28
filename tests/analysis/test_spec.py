"""Conceptual structure of analysis specification v0. No metrics are computed."""

import pytest
from pydantic import ValidationError

from board_game_analysis.analysis import (
    EvidenceLevel,
    MeasurementRole,
    MeasurementSpec,
    MeasurementStatus,
    measurement_catalog,
    spec_by_id,
)
from board_game_analysis.domain import DerivedMeasurement, Game


def test_catalog_ids_and_names_are_unique() -> None:
    catalog = measurement_catalog()
    ids = [spec.id for spec in catalog]
    names = [spec.name for spec in catalog]
    assert len(ids) == len(set(ids))
    assert len(names) == len(set(names))
    assert catalog


def test_catalog_roles_and_statuses_are_known() -> None:
    roles = {
        MeasurementRole.DESCRIPTIVE,
        MeasurementRole.PREDICTIVE,
        MeasurementRole.OUTCOME,
    }
    statuses = {
        MeasurementStatus.PROPOSED,
        MeasurementStatus.HYPOTHESIS,
        MeasurementStatus.FUTURE,
    }
    for spec in measurement_catalog():
        assert spec.role in roles
        assert spec.status in statuses
        assert spec.description


def test_layers_are_not_collapsed_in_the_catalog() -> None:
    hidden = spec_by_id("hidden_information")
    uncertainty = spec_by_id("information_uncertainty")
    chance = spec_by_id("randomness")
    assert hidden.id in uncertainty.distinct_from
    assert chance.id in uncertainty.distinct_from
    assert hidden.id != uncertainty.id != chance.id


def test_decision_counts_are_not_meaningfulness() -> None:
    available = spec_by_id("available_decision_count")
    legal = spec_by_id("legal_decision_count")
    meaningful = spec_by_id("meaningful_action_count")
    diversity = spec_by_id("decision_diversity")
    dominance = spec_by_id("strategic_dominance")
    assert meaningful.computable_from_ontology_v0 is False
    assert available.id in meaningful.distinct_from
    assert legal.id in meaningful.distinct_from
    assert diversity.id in meaningful.distinct_from
    assert dominance.id in meaningful.distinct_from
    assert meaningful.status == MeasurementStatus.FUTURE


def test_tension_is_a_hypothesis_not_a_formula() -> None:
    tension = spec_by_id("tension_trajectory")
    assert tension.role == MeasurementRole.PREDICTIVE
    assert tension.status == MeasurementStatus.HYPOTHESIS
    assert tension.computable_from_ontology_v0 is False


def test_fun_is_not_a_game_field_or_structural_metric() -> None:
    with pytest.raises(ValidationError):
        Game(id="g", title="G", fun=8.0)  # type: ignore[call-arg]
    names = {spec.name for spec in measurement_catalog()}
    assert "fun" not in names
    rating = spec_by_id("outcome_rating")
    assert rating.role == MeasurementRole.OUTCOME
    assert rating.computable_from_ontology_v0 is False


def test_structural_and_behavioral_interaction_are_split() -> None:
    structural = spec_by_id("structural_interaction")
    behavioral = spec_by_id("behavioral_interaction")
    assert structural.computable_from_ontology_v0 is True
    assert behavioral.computable_from_ontology_v0 is False
    assert behavioral.id in structural.distinct_from


def test_derived_measurement_can_record_scope_without_a_value() -> None:
    measurement = DerivedMeasurement(
        id="m-hanabi-branching",
        game_id="hanabi",
        name="available_decision_count",
        value=None,
        unit="actions",
        method="not_computed",
        scope="player",
        player_id="hanabi-alice",
        state_id="hanabi-s0",
        version="analysis-spec-v0",
        model_ref="tests.fixtures.games.hanabi",
        confidence=None,
    )
    assert measurement.value is None
    assert measurement.scope == "player"
    assert measurement.version == "analysis-spec-v0"


def test_evidence_levels_are_named_and_distinct() -> None:
    levels = {
        EvidenceLevel.OBSERVED_FACT,
        EvidenceLevel.EXTRACTED_INTERPRETATION,
        EvidenceLevel.STRUCTURAL_MEASUREMENT,
        EvidenceLevel.SIMULATION_RESULT,
        EvidenceLevel.HUMAN_OUTCOME,
    }
    assert len(levels) == 5


def test_measurement_spec_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        MeasurementSpec(
            id="x",
            name="x",
            role=MeasurementRole.DESCRIPTIVE,
            status=MeasurementStatus.PROPOSED,
            scope="game",
            description="d",
            formula="a*b",  # type: ignore[call-arg]
        )
