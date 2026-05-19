# two_people_same_org

Two people working at the same organization. Tests entity
deduplication on the shared `Organization`: even though "Acme
Corp" appears twice in the document, the expected graph has a
single `Organization` node referenced by both employees'
`WORKS_AT` edges.

## What this exercises

- **Cross-entity dedup** via `canonical_id`. The two `Person`
  nodes have distinct canonical_ids; the `Organization` is
  shared.
- **Property-level extraction** of `title`/`role` per person.
- **Multi-edge graph** where both edges share a target.

## Failure signals

- An extractor that creates two separate `Organization` nodes
  for the two mentions of "Acme Corp" drops **precision** on
  the Organization type. With one expected node and two
  actual nodes that share the helper-derived `canonical_id`
  ("Organization:name=acme corp"), the scorer records
  `TP=1, FP=1, FN=0` → `Organization.precision ≈ 0.5`,
  `Organization.recall = 1.0`. The expected node still has a
  match; the duplicate just contaminates the actual set with
  an unmatched extra.
  Edges are **not** affected in this case: both `WORKS_AT`
  edges in the actual graph point at the same canonical Org
  identity, so the canonical-id grouping in the edge-matching
  layer aligns them with the two expected edges and
  `WORKS_AT` stays at `TP=2, FP=0, FN=0`. Edges only take a
  hit when the duplicate has a *different* canonical_id from
  the matched one (the next bullet).
- Worse: an extractor that fails to compute `canonical_id` at
  all (or computes it differently per chunk — e.g., from a
  name with stray whitespace) splits the two Org references
  into two distinct canonical identities. Now only one Org
  matches, the other is unmatched, and one of the `WORKS_AT`
  edges points at the unmatched node — that edge fails
  source/target identity match. Both Organization and
  WORKS_AT precision drop below 1.0; recall on WORKS_AT
  drops too.
