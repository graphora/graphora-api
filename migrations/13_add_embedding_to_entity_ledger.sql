-- Slice 1 of cross-document entity linking: persist precomputed
-- embeddings on the entity_ledger so the similarity-search path
-- doesn't re-embed every stored entry on every transform.
--
-- The EntityLedgerEntry dataclass already has an `embedding` field
-- (added preemptively in earlier work) but no storage path existed.
-- This migration closes the gap.
--
-- Why JSONB and not pgvector: pgvector requires the extension to
-- be installed on the Postgres server, which we can't assume in
-- every deployment (in-memory dev, Supabase free-tier, etc.). JSONB
-- is universally available; the cost is ~50% larger payload and no
-- ANN indexing. For ledger sizes <= ~10k entries per (user, type)
-- the linear scan is still well under 100ms — the bigger win is
-- avoiding the embed-every-entry recomputation that was happening.
-- A future migration can introduce a pgvector column when scale
-- demands it; this migration is forward-compatible.
--
-- embedding_model is recorded so we can detect (and ignore) entries
-- embedded under a different model when the operator upgrades
-- ENTITY_RESOLUTION_EMBEDDING_MODEL — without it, mixing
-- 384-dim MiniLM vectors with 768-dim MPNet vectors silently
-- returns garbage similarity scores.
ALTER TABLE entity_ledger
    ADD COLUMN IF NOT EXISTS embedding JSONB,
    ADD COLUMN IF NOT EXISTS embedding_model TEXT;
