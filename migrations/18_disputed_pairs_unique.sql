-- B2-active backend slice B (reviewer-flagged P2 on commit 150677a):
-- the slice-B write hook in graph_transformer.py fires every time
-- a 2-node candidate comes back as all-singletons. With deterministic
-- node IDs, task retries / re-extractions surface the same candidate
-- multiple times — and slice A's schema has no uniqueness constraint
-- on the pair key. Without this index the queue accumulates duplicate
-- PENDING rows; labeling one leaves the duplicates still pending.
--
-- The unique key is (user_id, transform_id, source_stage, unordered
-- pair). Order-independent because (a, b) and (b, a) represent the
-- same candidate. ``source_stage`` IS part of the key — the same node
-- pair surfaced by ``property_blocker`` vs ``embedding_blocker`` is
-- conceptually two different signals worth tracking separately on
-- the dashboard (reviewers filter by stage).
--
-- Expression-based unique index (vs unique constraint) because the
-- LEAST/GREATEST pair-canonicalization is an expression, not a
-- single column. The service then references this exact expression
-- list in its INSERT ... ON CONFLICT (...) DO NOTHING clause so
-- Postgres's unique-index inference picks up the right index.
--
-- Safe to run against a populated table only when no duplicates
-- exist today. Slice A landed in the previous commit; the table is
-- empty in every environment (slice B's hook hadn't fired yet at the
-- time slice A's migration was applied) so the index creates cleanly.
CREATE UNIQUE INDEX IF NOT EXISTS idx_disputed_pairs_unique_pair
    ON disputed_pairs (
        user_id,
        transform_id,
        source_stage,
        LEAST(node_a_id, node_b_id),
        GREATEST(node_a_id, node_b_id)
    );
