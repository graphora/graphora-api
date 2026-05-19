# financial_transaction

Wire transfer between two accounts. Tests direction-sensitive
relationships (which side is debited vs credited) — the
source/target ordering on the edges is load-bearing here.

## What this exercises

- **Non-name identity**: `Account.account_number` is the
  unique property (vs `name` on Person/Org/Patient). The
  canonical_id helper handles any unique-flagged property
  type; this entry pins that path.
- **Symmetric entity type with asymmetric edges**: both
  accounts are `Account` nodes, but they're distinguished by
  edge direction — one `DEBITED_FROM`, the other
  `CREDITED_TO`. A scorer regression that ignores
  source/target direction would silently match swapped edges
  as correct.
- **Property with currency symbol**: `amount = "$5,000 USD"`
  is intentionally a string with currency intact, not a float.
  Parsing is out of scope for B4-test; downstream tools that
  want numbers should call a separate normalizer.

## Failure signals

- An extractor that flips the direction of `DEBITED_FROM` /
  `CREDITED_TO` → both edges fail identity match (source +
  target don't both resolve), → `edges.recall = 0`.
- An extractor that confuses the two accounts (uses #4521 for
  both ends) → one edge dangles, the other duplicates → both
  precision and recall drop.

## Intentional non-extractions

- "Wire transfer", "clearing house", "business day" — narrative
  context not modeled by the ontology. Surfacing them would
  be FPs.
