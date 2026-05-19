# employment_history

A short career biography spanning three employers. Tests
one-to-many edge fan-out from a shared source node.

## What this exercises

- **One Person, three WORKED_AT edges**: All three edges
  share the same source node (Maria Lopez). Pin against an
  extractor that creates duplicate Person nodes per
  employer mention.
- **Multi-mention dedup on the Person**: Maria Lopez is
  named four times; she must collapse to one node.
- **Sequential-in-narrative, parallel-in-graph**: The
  document presents the employments as a temporal sequence,
  but the ontology models them as a set of independent
  edges. Pin that the extractor doesn't smuggle in
  artificial PRECEDED_BY edges between employers.

## Failure signals

- An extractor that emits a chain (StartupCo PRECEDED_BY
  BigCorp PRECEDED_BY NewVenture) inflates relationship FP
  — those edges aren't modeled.
- An extractor that fuses BigCorp and NewVenture into one
  node (e.g., "current employer") drops Organization recall.

## Intentional non-extractions

- "software engineer", "senior engineer", "founding
  engineer", "infrastructure", "distributed systems", "2015",
  "2019", "2024" are content the document carries but the
  ontology does not model. Property-level extraction of
  start_date / end_date / title is a future slice; this
  entry pins the edge-level shape only.
