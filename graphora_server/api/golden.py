"""B4-test (Gate 4) — scoring endpoint for golden-corpus regression.

The companion to the corpus + scorer landed in commit 48dbe0a
(b4-corpus seed): given a hand-curated ground-truth graph and a
live transform_id, compute precision/recall/F1 by routing through
``CorpusScorer`` (which wraps DiffService for identity matching).

The CLI side (``graphora test --against golden/``) lives in the
graphora-client repo and orchestrates: register ontology, upload
document, await extraction, fetch transform, POST here for
scoring. This module is intentionally tight — one endpoint, one
shape — so the CLI can call it without ceremony.

Tenant-scoped: the actual graph is fetched via the same user-
isolation helper the /diff endpoint uses, so callers can only
score against transforms they own. The ``expected`` payload is
treated as caller-supplied ground truth (no DB write); a request
that lies about expected only hurts its own scoring report.
"""

from __future__ import annotations

import traceback
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from graphora_server.auth import AuthContext, get_current_auth
from graphora_server.schemas.graph import GraphResponse
from graphora_server.services.golden_corpus import CorpusScorer
from graphora_server.utils.logger import logger

# Lazily-imported to avoid pulling the graph_for_diff loader's
# transitive deps (storage backends, user-DB service) at module
# load. The endpoint pays one attribute-lookup per call; cleaner
# than copy-pasting the loader. ``_check_truncated`` rides along
# because the truncation guard contract must stay aligned with
# the diff endpoint — see the High reviewer fix below.
from graphora_server.api.graph import _check_truncated, _load_graph_for_diff


router = APIRouter(prefix="/api/v1/golden", tags=["Golden Corpus"])


class ScoreRequest(BaseModel):
    """Request body for POST /api/v1/golden/score.

    ``expected`` is the ground-truth graph (typically loaded from
    a ``golden/<slug>/expected.json`` corpus file by the CLI runner).
    ``transform_id`` is a live transform owned by the authenticated
    user — the actual graph is fetched server-side via the same
    backend-selection logic ``/diff`` uses (in-memory vs staging
    DB), so the caller doesn't have to ship the actual payload.
    ``corpus_slug`` is echoed onto the response for batch-aggregation
    convenience (the CLI tags each report with its slug).
    """

    expected: GraphResponse = Field(
        ...,
        description="Ground-truth graph for scoring (caller-supplied).",
    )
    transform_id: str = Field(
        ...,
        description=(
            "Live transform whose extraction will be compared "
            "against ``expected``. Must belong to the authenticated "
            "user — cross-tenant attempts return 404 without "
            "leaking whether the transform exists."
        ),
    )
    corpus_slug: str = Field(
        default="",
        description="Optional corpus directory name; echoed back on the response.",
    )


@router.post(
    "/score",
    description=(
        "Score a live transform's extraction against a caller-supplied "
        "ground-truth graph. Returns per-type and aggregate "
        "precision/recall/F1 plus raw TP/FP/FN counts. The scoring "
        "logic wraps the same identity-matching service the diff "
        "endpoint uses, so a node's canonical_id / canonical_key "
        "drives match identity — see graphora_server/services/"
        "golden_corpus/scorer.py for the full contract."
    ),
)
async def score_against_golden(
    body: ScoreRequest,
    auth: AuthContext = Depends(get_current_auth),
) -> Dict[str, Any]:
    """B4-test endpoint. The CLI runner POSTs here per corpus
    document; the server fetches the actual graph for the
    user-owned transform, scores it against the request's
    ``expected`` payload, and returns the structured
    ``ScoringReport`` dict.

    Tenant scoping mirrors the /diff endpoint: ``_load_graph_for_diff``
    uses the per-user storage factory, so a request that names
    another user's transform_id receives the same "no graph"
    treatment as a request for a missing transform — no leakage
    of cross-tenant existence.
    """
    try:
        actual = await _load_graph_for_diff(body.transform_id, auth.user_id)
    except HTTPException:
        # _load_graph_for_diff doesn't raise HTTPException itself,
        # but other layers below it can (e.g. user-DB lookup
        # failures). Pass through without wrapping so the status
        # code stays accurate.
        raise
    except Exception as exc:
        logger.error(
            "Error loading graph for golden scoring (transform=%s, user=%s): %s",
            body.transform_id,
            auth.user_id,
            exc,
        )
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error loading graph for scoring: {exc}",
        )

    # Reviewer-flagged High on commit 9fb4909: mirror the /diff
    # endpoint's truncation guard. The same 10k-node loader cap
    # applies — without this guard, a transform that exceeds the
    # cap would be scored against a partial actual graph, silently
    # producing a confident-but-wrong report (typically a string
    # of false negatives that aren't really missing). Surface as
    # 413 so the CLI runner can either short-circuit the corpus
    # entry or surface a clear error rather than write a bogus
    # score to its batch report.
    _check_truncated(actual, "actual", body.transform_id)

    # Reviewer-flagged Medium on commit 9fb4909: pre-fix this
    # branch rejected an empty ``actual.nodes`` as 404, conflating
    # three distinct cases — missing transform, cross-tenant
    # attempt, and a legitimate "the extractor produced zero
    # entities" outcome. For B4-test, the third case is exactly
    # what scoring should surface (every expected node becomes
    # FN, recall drops to 0, the CLI runner flags the doc as a
    # regression). Mirroring /diff's posture: proceed with
    # scoring regardless of emptiness; missing/cross-tenant
    # transforms produce the same all-FN report a real
    # zero-extraction would. The trade-off is a malformed
    # transform_id no longer 404s — but the CLI runner controls
    # transform_id (it just uploaded the doc), and direct API
    # users get the same posture /diff already has.
    report = CorpusScorer().score(
        expected=body.expected,
        actual=actual,
        corpus_slug=body.corpus_slug or None,
    )
    return report.to_dict()
