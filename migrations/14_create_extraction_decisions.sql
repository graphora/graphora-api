CREATE TABLE IF NOT EXISTS extraction_decisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    transform_id TEXT NOT NULL,
    target_id TEXT,
    target_kind TEXT NOT NULL,
    decision_type TEXT NOT NULL,
    reason TEXT,
    evidence JSONB NOT NULL DEFAULT '{}'::jsonb,
    alternatives JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_extraction_decisions_transform_target
    ON extraction_decisions (transform_id, target_id);
CREATE INDEX IF NOT EXISTS idx_extraction_decisions_transform_type
    ON extraction_decisions (transform_id, decision_type);
