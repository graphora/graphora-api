-- Add optional metadata columns used by the new merge repository
ALTER TABLE change_logs
    ADD COLUMN IF NOT EXISTS match_confidence DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS match_strategy TEXT;
