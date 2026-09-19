"""Thin ds-platform adapter for fixture claim sets.

ExtractedInterpretation and SourceReference stay in BGA. This module maps
those domain objects to Claim/Evidence/Citation at the store boundary.
"""

from __future__ import annotations

import json
from datetime import datetime

from ds_platform import (
    ArtifactKind,
    ArtifactRecord,
    Citation,
    Claim,
    ClaimLayer,
    Evidence,
    RelatedRef,
    RelationType,
    RunContext,
    Store,
    payload_id,
    put_record,
)

from board_game_analysis.domain.interpretation import ExtractedInterpretation

CLAIM_SET_MEDIA_TYPE = "application/json"


def claim_set_logical_key(game_id: str) -> str:
    return f"bga:claims/game/{game_id}"


def claim_set_json_bytes(claims: list[Claim]) -> bytes:
    """Exact claim-set payload bytes. Not platform record canonicalization."""
    payload = [claim.model_dump(mode="json", exclude_none=True) for claim in claims]
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def claim_from_interpretation(
    interpretation: ExtractedInterpretation,
    *,
    subject_payload_id: str,
    excerpt: str,
    pointer: str | None = None,
    layer: ClaimLayer = ClaimLayer.INFERRED,
) -> Claim:
    """Map a BGA interpretation to a Claim. SourceReference is not copied in."""
    citation = Citation(payload_id=subject_payload_id, pointer=pointer)
    policy = None
    if interpretation.extractor is not None:
        policy = f"extractor:{interpretation.extractor}"
    statement = interpretation.label
    if interpretation.description:
        statement = f"{interpretation.label}: {interpretation.description}"
    return Claim(
        statement=statement,
        layer=layer,
        confidence_policy=policy,
        subject_payload_id=subject_payload_id,
        evidence=[
            Evidence(
                citation=citation,
                excerpt=excerpt,
                method=interpretation.extractor,
            )
        ],
    )


def fixture_catan_claims(game_payload_id: str) -> list[Claim]:
    """One observed catalog claim and a conflicting inferred claim."""
    citation = Citation(payload_id=game_payload_id, pointer="/categories")
    observed = Claim(
        statement="CATAN is listed with the source-reported category Negotiation",
        layer=ClaimLayer.OBSERVED,
        confidence_policy="source-reported",
        subject_payload_id=game_payload_id,
        evidence=[
            Evidence(
                citation=citation,
                excerpt="Negotiation",
                method="bgg-category",
            )
        ],
    )
    conflicting = Claim(
        statement="CATAN is not a negotiation game",
        layer=ClaimLayer.INFERRED,
        confidence_policy="unresolved-conflict",
        subject_payload_id=game_payload_id,
        citations=[citation],
    )
    return [observed, conflicting]


def store_claim_set(
    store: Store,
    claim_set_bytes: bytes,
    *,
    subject_payload_id: str,
    run: RunContext,
    game_id: str,
    created_at: datetime | None = None,
) -> tuple[str, str, ArtifactRecord]:
    """Store exact claim-set JSON; cite the subject via related claims_about."""
    pid = payload_id(claim_set_bytes)
    store.put(pid, claim_set_bytes, media_type=CLAIM_SET_MEDIA_TYPE)
    record = ArtifactRecord(
        payload_id=pid,
        media_type=CLAIM_SET_MEDIA_TYPE,
        kind=ArtifactKind.CLAIM_SET,
        produced_by=run,
        created_at=created_at or run.started_at,
        logical_key=claim_set_logical_key(game_id),
        related=[
            RelatedRef(rel=RelationType.CLAIMS_ABOUT, payload_id=subject_payload_id)
        ],
    )
    rid = put_record(store, record)
    return pid, rid, record
