CREATE TABLE IF NOT EXISTS disputed_pairs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    transform_id TEXT NOT NULL,
    node_a_id TEXT NOT NULL,
    node_b_id TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    node_a_canonical_key TEXT,
    node_b_canonical_key TEXT,
    similarity_score NUMERIC(6, 4),
    source_stage TEXT NOT NULL CHECK (source_stage IN (
        'property_blocker',
        'embedding_blocker',
        'splink_blocker',
        'llm_review'
    )),
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN (
        'pending',
        'labeled_match',
        'labeled_not_match',
        'skipped'
    )),
    labeled_at TIMESTAMPTZ,
    labeled_by_user_id TEXT,
    label_reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- Pre-fix invariant: a labeled row carries both labeled_at
    -- and labeled_by_user_id. A pending row carries neither.
    -- Mixed state would mean either "labeled but we don't know
    -- when" or "we know when but not who" — both ambiguous.
    CONSTRAINT disputed_pairs_label_consistency CHECK (
        (status = 'pending' AND labeled_at IS NULL AND labeled_by_user_id IS NULL)
        OR (status != 'pending' AND labeled_at IS NOT NULL AND labeled_by_user_id IS NOT NULL)
    )
);

-- Common query: "show me this user's pending queue, newest
-- first". The DESC index supports both the queue surface
-- (newest-pending = most recent ER guess to review) and the
-- LIMIT pagination.
CREATE INDEX IF NOT EXISTS idx_disputed_pairs_user_status_created
    ON disputed_pairs (user_id, status, created_at DESC);

-- Per-transform lookup for "show me what ER produced for this
-- run". Same tenant scoping.
CREATE INDEX IF NOT EXISTS idx_disputed_pairs_user_transform
    ON disputed_pairs (user_id, transform_id);
