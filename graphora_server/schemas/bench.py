"""B4-bench Pydantic wire models for the public benchmark endpoint.

Mirrors the service-layer dataclasses in
``graphora_server.services.bench.models`` field-for-field so JSON
serialization roundtrips cleanly AND the OpenAPI schema exposes
the actual shape — corpus_size, micro/macro F1, per-entry detail,
error rows — to generated clients.

Reviewer-flagged Medium on commit 06fc210: pre-fix the route
returned ``Dict[str, Any]`` with no ``response_model``, so the
committed OpenAPI snapshot for ``/api/v1/bench/run`` was a
permissive ``{"additionalProperties": true, "type": "object"}``.
Without a typed response_model, the snapshot can't catch wire-
shape regressions (e.g., dropping ``micro_node_f1`` or renaming
``errored_count``) — exactly the load-bearing claim of a public
reproducibility-promising API.

The split mirrors the convention the contradictions surface uses
(see ``schemas/claims.py`` alongside ``services/claims_service.py``):
plain dataclasses for service-layer unit testing without Pydantic
ceremony, Pydantic models for the wire to satisfy OpenAPI typing.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class BenchEntryScore(BaseModel):
    """One extractor's score on one corpus entry.

    Matches ``services.bench.models.BenchEntryScore`` field-for-
    field. ``error`` is set when the runner couldn't score this
    entry (missing actual.json, malformed shape, etc.); when
    non-None the numeric fields are zeroed and the entry is
    excluded from the extractor's aggregate. The frontend
    distinguishes "extractor failed on this doc" from "extractor
    scored 0 on this doc" via this field — both look like zero
    F1 otherwise.
    """

    corpus_slug: str
    node_precision: float = Field(..., ge=0.0, le=1.0)
    node_recall: float = Field(..., ge=0.0, le=1.0)
    node_f1: float = Field(..., ge=0.0, le=1.0)
    edge_precision: float = Field(..., ge=0.0, le=1.0)
    edge_recall: float = Field(..., ge=0.0, le=1.0)
    edge_f1: float = Field(..., ge=0.0, le=1.0)
    node_true_positives: int = Field(..., ge=0)
    node_false_positives: int = Field(..., ge=0)
    node_false_negatives: int = Field(..., ge=0)
    edge_true_positives: int = Field(..., ge=0)
    edge_false_positives: int = Field(..., ge=0)
    edge_false_negatives: int = Field(..., ge=0)
    error: Optional[str] = Field(
        default=None,
        description=(
            "Failure reason when this (extractor, slug) pair couldn't "
            "be scored. ``None`` means scoring succeeded. Coverage "
            "gaps (missing extractor output for a corpus entry) "
            "surface here rather than as silent zero-F1 rows so the "
            "denominator stays honest."
        ),
    )

    model_config = ConfigDict(from_attributes=True)


class BenchExtractorReport(BaseModel):
    """Aggregated scores for one extractor across the full corpus.

    Both micro (entity-count weighted) and macro (per-doc mean)
    F1 surface so consumers can pick the right headline. Micro
    answers "across all the entities, what fraction did the
    extractor get right?" — a 50-node doc dominates a 3-node
    doc. Macro asks "across all the docs, what's the average per-
    doc F1?" — every doc gets equal weight. An extractor that
    wins on micro but loses on macro is strong on dense docs and
    weak on sparse ones; the headline F1 alone would hide that.

    ``errored_count`` is required (not optional with a default)
    so OpenAPI consumers can't treat it as ignorable. It's the
    load-bearing signal for distinguishing "writer healthy" from
    "writer missing data for some entries" — the same posture as
    ``total_claims_scanned`` on the contradictions surface
    (reviewer-flagged Medium on commit 86c1dbd / 66987b2).
    """

    extractor_name: str
    total_entries: int = Field(..., ge=0)
    scored_count: int = Field(..., ge=0)
    errored_count: int = Field(
        ...,
        ge=0,
        description=(
            "Number of corpus entries the extractor failed to score "
            "(missing output, malformed JSON, etc.). Required field "
            "— callers must NOT treat it as optional, since collapsing "
            "errored + scored into one bucket hides coverage gaps."
        ),
    )
    micro_node_precision: float = Field(..., ge=0.0, le=1.0)
    micro_node_recall: float = Field(..., ge=0.0, le=1.0)
    micro_node_f1: float = Field(..., ge=0.0, le=1.0)
    micro_edge_precision: float = Field(..., ge=0.0, le=1.0)
    micro_edge_recall: float = Field(..., ge=0.0, le=1.0)
    micro_edge_f1: float = Field(..., ge=0.0, le=1.0)
    macro_node_f1: float = Field(..., ge=0.0, le=1.0)
    macro_edge_f1: float = Field(..., ge=0.0, le=1.0)
    entries: List[BenchEntryScore]

    model_config = ConfigDict(from_attributes=True)


class BenchRunReport(BaseModel):
    """Top-level public benchmark report. One ``extractors`` row
    per ``bench/results/<extractor>/`` directory.

    ``corpus_size`` is the discovered count of ``golden/<slug>/``
    entries — exposed so frontend consumers can render "N of M
    entries scored" without computing it from ``entries``.
    """

    corpus_size: int = Field(..., ge=0)
    extractor_count: int = Field(..., ge=0)
    extractors: List[BenchExtractorReport]

    model_config = ConfigDict(from_attributes=True)
