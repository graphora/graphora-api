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
  (one per mention of "Acme Corp") drops **precision** on the
  Organization type — one expected, one of the two actuals
  matches it, the other is an extra FP. The scorer reports
  `Organization.precision ≈ 0.5` (1 TP / (1 TP + 1 FP)) while
  `Organization.recall` stays 1.0 (the expected node was
  still matched). Edges then take a hit too: one of the two
  WORKS_AT edges points at the unmatched duplicate Org, so
  that edge fails source/target identity match and the
  scorer records `WORKS_AT` precision **and** recall around
  0.5.
- Worse: an extractor that uses the per-chunk node id (instead
  of the helper-derived `canonical_id`) splits the two
  references entirely and both `WORKS_AT` edges go dangling.
