-- B6-scenario slice 1: scenarios are named, point-in-time snapshots
-- of a transform's graph. Slice 1 materializes the full graph as
-- JSONB per scenario; the CoW (diff-from-parent storage) split
-- lands in slice 2 once the read/write API shape stabilizes.
--
-- parent_scenario_id stays NULL for slice 1 (scenarios are always
-- branched from a transform, not from another scenario). Keeping
-- the column reserved up front avoids a follow-up migration when
-- slice 2 introduces branching-from-scenario.
CREATE TABLE IF NOT EXISTS scenarios (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    transform_id TEXT NOT NULL,
    parent_scenario_id UUID,
    name TEXT NOT NULL,
    description TEXT,
    -- Materialized graph snapshot at creation time. JSONB shape
    -- mirrors GraphResponse: {nodes: [...], edges: [...]}.
    -- Reading a scenario means reading this column verbatim, no
    -- chain walk required. Slice 2 may replace this with a
    -- diff-from-parent layout to avoid duplicating large graphs.
    graph_snapshot JSONB NOT NULL DEFAULT '{"nodes": [], "edges": []}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- A scenario name must be unique per (user, transform) so the
    -- CLI's `graphora scenario create --name foo` is idempotent
    -- by name and the list view stays scannable.
    CONSTRAINT scenarios_name_unique_per_transform
        UNIQUE (user_id, transform_id, name),

    -- Slice 1 always sets parent_scenario_id NULL. Keeping a
    -- FK-style integrity check here once slice 2 enables
    -- branching from scenarios; for now this just documents the
    -- intent and stops a stray non-NULL insert from a future
    -- writer that doesn't know about the slice gating.
    CONSTRAINT scenarios_parent_null_in_slice_1 CHECK (
        parent_scenario_id IS NULL
    )
);

CREATE INDEX IF NOT EXISTS idx_scenarios_user_transform
    ON scenarios (user_id, transform_id);
CREATE INDEX IF NOT EXISTS idx_scenarios_user_created
    ON scenarios (user_id, created_at DESC);
