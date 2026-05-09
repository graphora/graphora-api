-- Reviewer-flagged on commit eb22a79 (P1): the /decisions endpoint
-- went straight to DecisionLogService and the table was keyed only
-- by transform_id/target_id. Any authenticated user who knew
-- another transform's id could fetch its decision log.
--
-- Add user_id to enforce tenant ownership at the data layer. The
-- column is nullable to keep additive backward-compat with rows
-- that may already exist in dev environments (production tables
-- are empty as of 2026-05-09 since B0-log only just shipped).
-- The endpoint enforces non-NULL match so legacy NULLs land at
-- "deny" rather than "allow anyone".

ALTER TABLE extraction_decisions
    ADD COLUMN IF NOT EXISTS user_id TEXT;

-- Backfill index supports the common read shape: "give me this
-- user's decisions for this transform". Composite (user_id,
-- transform_id) means lookups don't have to scan all decisions
-- for the transform and then filter by user.
CREATE INDEX IF NOT EXISTS idx_extraction_decisions_user_transform
    ON extraction_decisions (user_id, transform_id);
