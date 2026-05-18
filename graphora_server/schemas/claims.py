"""B1-prob slice 2a: Pydantic wire models for the contradictions surface.

A :class:`Contradiction` is the API-level projection of the
service-layer :class:`graphora_server.services.claims_service.Contradiction`
dataclass. The split between schemas (Pydantic) and services
(plain dataclasses) mirrors the convention the rest of the
Gate-4 surfaces use — keeps the service unit-testable without
Pydantic ceremony, and keeps the API layer free of dataclass
serialization concerns.

The shape mirrors the existing diff endpoint's contradiction-
adjacent surfaces: a list of (target_id, target_kind,
property_key) groups, each carrying its competing claims sorted
by confidence DESC and a ``severity`` count (distinct-value
count above the confidence floor). The "winning" claim — the
one the pipeline picked — is the first entry in
``competing_claims``.
"""

from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class Claim(BaseModel):
    """One extractor's assertion about a target's property.

    Matches the service-layer ``Claim`` dataclass field-for-field
    so JSON serialization roundtrips cleanly. The
    ``target_kind`` field is the string value (``"node"`` /
    ``"edge"``) rather than the enum, mirroring how the wire
    surfaces the same enum elsewhere (Decision Log endpoints).
    """

    id: str
    transform_id: str
    target_id: str
    target_kind: str = Field(
        ...,
        description="``node`` or ``edge`` — the kind of target this claim is about.",
    )
    property_key: str
    value: Any = Field(
        ...,
        description=(
            "The claimed property value. JSON-serializable; can be a "
            "string, number, list, or nested object."
        ),
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Extractor-reported confidence in [0.0, 1.0].",
    )
    source_chunk_id: Optional[str] = None
    source_extractor_model: Optional[str] = None
    source_prompt_version: Optional[str] = None
    user_id: Optional[str] = None
    created_at: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class Contradiction(BaseModel):
    """A (target, property) pair carrying multiple distinct claimed values.

    A group with 2+ distinct values above the
    ``min_confidence`` floor surfaces as a contradiction; the
    ``competing_claims`` list is sorted by confidence DESC so the
    "winning" value is the first entry.

    ``severity`` is the distinct-value count — a heuristic on
    "how many alternative values disagree on this property." A
    slice 3 refinement may swap to a confidence-weighted entropy
    measure once enough real data lands; the wire shape stays the
    same (severity is just an int).
    """

    target_id: str
    target_kind: str
    property_key: str
    competing_claims: List[Claim] = Field(
        ...,
        description=(
            "All claims about this (target, property) above the "
            "confidence floor, sorted by confidence DESC. The "
            "first entry is the 'winning' claim the pipeline "
            "picked; the rest are alternatives the contradiction "
            "detector wants you to know about."
        ),
    )
    severity: int = Field(
        ...,
        ge=2,
        description=(
            "Distinct-value count (always >= 2 — single-value "
            "groups don't surface as contradictions)."
        ),
    )

    model_config = ConfigDict(from_attributes=True)


class ContradictionsResponse(BaseModel):
    """Wire envelope for GET /api/v1/graph/{tx}/contradictions.

    The envelope (rather than a bare list) gives room for future
    pagination + summary stats without breaking the wire
    contract. The shape mirrors the diff endpoint's response
    pattern — a summary at the top, the per-item list below.
    """

    transform_id: str
    min_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence floor applied to the underlying claim set.",
    )
    contradictions: List[Contradiction]
    # Reserved for slice 2b once pipeline hooks emit claims;
    # zero until then.
    total_claims_scanned: int = Field(
        default=0,
        description=(
            "Number of claims considered for contradiction detection "
            "(post-confidence-filter). Returns 0 until B1-prob slice "
            "2b's pipeline hooks emit claims at extraction time."
        ),
    )

    model_config = ConfigDict(from_attributes=True)
