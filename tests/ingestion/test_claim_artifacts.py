"""ds-platform claim_set boundary. Fixture-only; no extraction engine."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from ds_platform import (
    ArtifactKind,
    Citation,
    Claim,
    ClaimLayer,
    Evidence,
    LocalStore,
    RelationType,
    payload_id,
)
from pydantic import ValidationError

from board_game_analysis.domain.interpretation import ExtractedInterpretation
from board_game_analysis.ingestion.bgg.normalizer import normalize_bgg_artifact
from board_game_analysis.ingestion.bgg.types import RawArtifact
from board_game_analysis.ingestion.claim_artifacts import (
    claim_from_interpretation,
    claim_set_json_bytes,
    claim_set_logical_key,
    fixture_catan_claims,
    store_claim_set,
)
from board_game_analysis.ingestion.document_artifacts import store_game_document
from board_game_analysis.ingestion.pipeline import processed_game_json_bytes
from board_game_analysis.ingestion.raw_artifacts import (
    bga_run_context,
    store_raw_bgg_artifact,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "ingestion"
RETRIEVED = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)


def _artifact() -> RawArtifact:
    body = (FIXTURES / "bgg_thing_13.xml").read_text(encoding="utf-8")
    return RawArtifact(
        source="boardgamegeek",
        source_identifier="13",
        retrieved_at=RETRIEVED,
        request_url="https://boardgamegeek.com/xmlapi2/thing?id=13&stats=1",
        content_type="text/xml",
        http_status=200,
        body=body,
    )


def test_identical_claim_set_bytes_produce_identical_payload_id() -> None:
    claims = fixture_catan_claims(payload_id(b"game-doc"))
    first = claim_set_json_bytes(claims)
    second = claim_set_json_bytes(claims)
    assert first == second
    assert payload_id(first) == payload_id(second)


def test_claim_requires_layer_and_evidence_requires_excerpt() -> None:
    citation = Citation(payload_id=payload_id(b"game"))
    with pytest.raises(ValidationError):
        Claim(statement="CATAN has negotiation")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        Evidence(citation=citation, excerpt="")


def test_citation_is_not_source_reference() -> None:
    citation = Citation(payload_id=payload_id(b"game"), pointer="/categories")
    dumped = citation.model_dump()
    assert dumped["payload_id"] == payload_id(b"game")
    assert "url" not in dumped
    assert "source" not in dumped
    assert "source_identifier" not in dumped
    assert "retrieved_at" not in dumped


def test_conflicting_claims_are_allowed_in_one_set(tmp_path: Path) -> None:
    run = bga_run_context(started_at=RETRIEVED)
    store = LocalStore(tmp_path)
    artifact = _artifact()
    raw_pid, _, _ = store_raw_bgg_artifact(store, artifact, run)
    game = normalize_bgg_artifact(artifact)
    game_json = processed_game_json_bytes(game)
    document_pid, _, _ = store_game_document(
        store,
        game_json,
        raw_payload_id=raw_pid,
        run=run,
        game_id=game.id,
    )
    claims = fixture_catan_claims(document_pid)
    assert {claim.layer for claim in claims} == {
        ClaimLayer.OBSERVED,
        ClaimLayer.INFERRED,
    }
    payload = claim_set_json_bytes(claims)
    claim_pid, _rid, record = store_claim_set(
        store,
        payload,
        subject_payload_id=document_pid,
        run=run,
        game_id=game.id,
    )
    assert record.kind is ArtifactKind.CLAIM_SET
    assert record.logical_key == claim_set_logical_key("bgg-13")
    assert record.related[0].rel is RelationType.CLAIMS_ABOUT
    assert record.related[0].payload_id == document_pid
    assert store.get(document_pid) == game_json
    assert store.get(claim_pid) == payload
    assert claim_pid != document_pid
    dumped = record.model_dump()
    assert "location" not in dumped
    assert "statement" not in dumped
    assert "url" not in dumped


def test_game_payload_unchanged_when_claim_set_is_added(tmp_path: Path) -> None:
    run = bga_run_context(started_at=RETRIEVED)
    store = LocalStore(tmp_path)
    artifact = _artifact()
    raw_pid, _, _ = store_raw_bgg_artifact(store, artifact, run)
    game = normalize_bgg_artifact(artifact)
    game_json = processed_game_json_bytes(game)
    document_pid, _, document = store_game_document(
        store,
        game_json,
        raw_payload_id=raw_pid,
        run=run,
        game_id=game.id,
    )
    store_claim_set(
        store,
        claim_set_json_bytes(fixture_catan_claims(document_pid)),
        subject_payload_id=document_pid,
        run=run,
        game_id=game.id,
    )
    assert store.get(document_pid) == game_json
    assert "claims" not in document.model_dump()
    assert not hasattr(game, "claims")


def test_interpretation_maps_to_claim_without_source_reference() -> None:
    subject = payload_id(b"game-doc")
    interpretation = ExtractedInterpretation(
        id="hanabi-interp-asymmetric",
        game_id="hanabi",
        label="asymmetric_information",
        description="Players see others' cards, not their own.",
        confidence=0.9,
        extractor="human",
    )
    claim = claim_from_interpretation(
        interpretation,
        subject_payload_id=subject,
        excerpt="Players see others' cards, not their own.",
        pointer="/title",
    )
    assert claim.layer is ClaimLayer.INFERRED
    assert claim.confidence_policy == "extractor:human"
    assert claim.subject_payload_id == subject
    dumped = claim.model_dump()
    assert "confidence" not in dumped
    assert "url" not in dumped
    assert "source_identifier" not in dumped
    assert claim.evidence is not None
    assert claim.evidence[0].citation.payload_id == subject
