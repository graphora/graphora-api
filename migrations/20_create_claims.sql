-- B1-prob slice 1: claims are property-level assertions emitted
-- by the extraction pipeline. Each row represents "extractor X
-- read chunk Y and concluded that node N has property K = value V
-- with confidence C". The pipeline currently picks a single
-- winner per (node, property) and drops the rest; claims keep
-- the full distribution so contradictions stay visible.
--
-- Slice 1 lands the foundation. Slice 2 hooks the extraction
-- pipeline to emit claims at write time + adds a /contradictions
-- API surface. Slice 3 wires the Evidence Explorer's
-- Contradictions tab into the new data.
--
-- The shape mirrors extraction_decisions (B0-log migration 14):
-- same JSONB-evidence pattern, same per-tenant user_id column,
-- same target_kind constraint. The two surfaces are siblings —
-- decisions log the pipeline's choice; claims log the data the
-- choice was made over.
CREATE TABLE IF NOT EXISTS claims (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    transform_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    target_kind TEXT NOT NULL CHECK (target_kind IN ('node', 'edge')),
    property_key TEXT NOT NULL,
    -- JSONB so claims about non-string properties (numbers,
    -- lists, structured values) round-trip cleanly. The
    -- contradiction detector compares values JSON-wise — two
    -- claims with the same JSON value are NOT contradictions
    -- even if they came from different chunks.
    value JSONB NOT NULL,
    -- 0..1 inclusive. Pipeline writes the extractor's reported
    -- confidence; downstream UIs interpret. Storing as
    -- DOUBLE PRECISION (not NUMERIC) keeps arithmetic cheap;
    -- the precision tradeoff is acceptable — confidence is a
    -- gradient, not an audit-grade value.
    confidence DOUBLE PRECISION NOT NULL,
    -- Provenance trio. Mirrors the source-span columns the
    -- pipeline already stores on node/edge properties — slicing
    -- by these lets the API show "all claims from chunk-3" or
    -- "all claims this prompt version emitted" without a
    -- secondary join.
    source_chunk_id TEXT,
    source_extractor_model TEXT,
    source_prompt_version TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Confidence must be a valid probability. The pipeline can
    -- emit weird values when LLM responses fail to parse; pin
    -- the invariant at the DB layer so a buggy writer can't
    -- corrupt the contradiction-detection math downstream.
    CONSTRAINT claims_confidence_in_range CHECK (
        confidence >= 0.0 AND confidence <= 1.0
    )
);

-- Most-frequent query: "all claims about target T in transform X".
-- The contradiction detector groups by (target_id, property_key)
-- so the index covers both lookups.
CREATE INDEX IF NOT EXISTS idx_claims_transform_target
    ON claims (transform_id, target_id, property_key);
-- Per-user list view + tenant scoping at read time.
CREATE INDEX IF NOT EXISTS idx_claims_user_transform
    ON claims (user_id, transform_id);
