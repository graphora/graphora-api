CREATE TABLE IF NOT EXISTS user_budgets (
    user_id TEXT PRIMARY KEY,
    monthly_cap_usd NUMERIC(10, 4) NOT NULL CHECK (monthly_cap_usd >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- A user has at most one budget (PK enforces that). Reads always
-- key by user_id from the auth context, so no additional index
-- is needed beyond the PK.
