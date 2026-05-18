-- B6-scenario slice 2c: CoW (copy-on-write) storage. Mutations
-- on a scenario no longer rewrite ``graph_snapshot`` — that
-- column becomes the IMMUTABLE base captured at create time.
-- A new ``graph_diff`` column accumulates the canonical-state
-- delta (added / removed / updated nodes + edges). Reading the
-- scenario means resolving (base + diff) at access time.
--
-- The CoW win: a mutation writes only the diff (typically a
-- few KB), not the full snapshot (could be megabytes for
-- 10k-node graphs). Existing scenarios automatically get an
-- empty diff via the column default, so their resolved view
-- equals their graph_snapshot — zero behaviour change on the
-- materialized path.
--
-- The diff shape mirrors graphora_server.schemas.graph_changes
-- but uses the resolved node/edge payloads rather than the
-- request DTOs: callers want to see the full post-mutation
-- node, not just an id + property delta.
ALTER TABLE scenarios
    ADD COLUMN IF NOT EXISTS graph_diff JSONB NOT NULL
    DEFAULT '{
        "added_nodes": [],
        "removed_node_ids": [],
        "updated_nodes": [],
        "added_edges": [],
        "removed_edge_ids": [],
        "updated_edges": []
    }'::jsonb;
