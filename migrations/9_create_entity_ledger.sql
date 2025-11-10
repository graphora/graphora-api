CREATE TABLE IF NOT EXISTS entity_ledger (
    user_id TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    canonical_key TEXT NOT NULL,
    canonical_id TEXT NOT NULL,
    features JSONB NOT NULL DEFAULT '{}'::jsonb,
    confidence DOUBLE PRECISION,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, entity_type, canonical_key)
);

CREATE INDEX IF NOT EXISTS idx_entity_ledger_user_type ON entity_ledger (user_id, entity_type);
CREATE INDEX IF NOT EXISTS idx_entity_ledger_canonical_id ON entity_ledger (canonical_id);
