# org_subsidiaries

A short parent-company description. Tests hierarchical
same-type edges converging on a shared parent node.

## What this exercises

- **Three same-type edges sharing a target**: All three
  subsidiaries point at the same Voltaic Holdings node via
  SUBSIDIARY_OF. The expected graph has one parent node
  referenced by three edges.
- **Pure hierarchical relationship**: The ontology has only
  one relationship type. Pin against an extractor that
  invents siblings or peer relationships not in the source.
- **Same-name prefix dedup**: All four companies share the
  "Voltaic" prefix but have distinct canonical_keys (the
  full lowercased name). A name-parser that strips common
  prefixes would collapse them incorrectly.

## Failure signals

- An extractor that creates one Voltaic Holdings node per
  paragraph (three mentions) drops Holdings precision: 1
  expected vs. 3 actuals.
- An extractor that emits sibling edges between subsidiaries
  (e.g., Voltaic Energy SIBLING_OF Voltaic Storage) inflates
  relationship FP — those edges aren't in the ground truth
  even though peers can be inferred from the parent.

## Intentional non-extractions

- "executive committee", "quarterly basis", "solar-generation
  arm", "battery storage portfolio", "transmission-and-
  distribution operations" are descriptive content the
  ontology does not model.
