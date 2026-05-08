CREATE TABLE IF NOT EXISTS extraction_decisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    transform_id TEXT NOT NULL,
    target_id TEXT,
    target_kind TEXT NOT NULL CHECK (target_kind IN ('node', 'edge', 'schema')),
    decision_type TEXT NOT NULL CHECK (decision_type IN (
        'schema_inferred',
        'entity_merged',
        'relationship_accepted',
        'relationship_rejected',
        'confidence_marked',
        'llm_disambiguated'
    )),
    reason TEXT,
    evidence JSONB NOT NULL DEFAULT '{}'::jsonb,
    alternatives JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- Schema-level decisions don't pin to a specific node/edge id
    -- (e.g. "ontology inferred from chunk-1"); node/edge decisions
    -- always do. Without this constraint a stray INSERT can land
    -- target_kind='node' with target_id=NULL and the for_target
    -- query would silently exclude it. Pinning the invariant at the
    -- DB layer prevents that drift regardless of which writer
    -- (service, manual SQL, future microservice) inserts the row.
    CONSTRAINT extraction_decisions_target_kind_target_id_consistent CHECK (
        (target_kind = 'schema' AND target_id IS NULL)
        OR (target_kind IN ('node', 'edge') AND target_id IS NOT NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_extraction_decisions_transform_target
    ON extraction_decisions (transform_id, target_id);
CREATE INDEX IF NOT EXISTS idx_extraction_decisions_transform_type
    ON extraction_decisions (transform_id, decision_type);
