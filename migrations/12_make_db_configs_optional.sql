-- Make staging and production database configurations optional
-- Staging DB is optional (system uses in-memory storage if not configured)
-- Production DB is required for merge operations but not for initial config

-- Drop the different_databases constraint first (it will fail with NULLs)
ALTER TABLE configs DROP CONSTRAINT IF EXISTS different_databases;

-- Make staging_db_id nullable
ALTER TABLE configs ALTER COLUMN staging_db_id DROP NOT NULL;

-- Make prod_db_id nullable
ALTER TABLE configs ALTER COLUMN prod_db_id DROP NOT NULL;

-- Add a new constraint that only checks when both are non-null
ALTER TABLE configs ADD CONSTRAINT different_databases
    CHECK (staging_db_id IS NULL OR prod_db_id IS NULL OR staging_db_id != prod_db_id);

-- Add constraint to ensure at least one database is configured
ALTER TABLE configs ADD CONSTRAINT at_least_one_db
    CHECK (staging_db_id IS NOT NULL OR prod_db_id IS NOT NULL);
