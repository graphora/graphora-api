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
  (one per mention of "Acme Corp") fails recall on the shared
  Organization — `Organization.recall = 0.5` (one expected,
  one extracted matches; the second extracted is a duplicate
  that doesn't appear in expected).
- Worse: an extractor that uses the per-chunk node id (instead
  of the helper-derived `canonical_id`) splits the two
  references entirely and both `WORKS_AT` edges go dangling.
