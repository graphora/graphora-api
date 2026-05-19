# legal_simple_agreement

Two-party service agreement. Tests the "multiple entity types
all referenced multiple times" dedup case, plus optional
property extraction (`effective_date` on the Agreement).

## What this exercises

- **Three-way dedup**: "Globex Industries", "Initech LLC", and
  "Service Agreement" each appear twice in the document — all
  three must collapse to single canonical nodes.
- **Optional property capture**: `effective_date = "April 1,
  2026"` is in the source as a string. The extractor doesn't
  need to parse it into a datetime; the corpus pins the
  string representation as it appears.
- **Symmetric edges**: both parties have the same edge type
  (`PARTY_TO`) pointing at the same target. The two edges
  are distinct (different sources) so neither collapses.

## Failure signals

- An extractor that produces separate Agreement nodes for the
  two "Service Agreement" mentions → `Agreement.recall = 0.5`,
  one of the two `PARTY_TO` edges dangles.
- An extractor that drops `effective_date` because it sits
  outside the parties' immediate context → `Agreement` itself
  still matches via canonical_id, but the changed-properties
  delta surfaces. (Property-level scoring isn't part of the
  B4-test scorer yet, so this only flags via diff if the
  caller uses /diff.)

## Intentional non-extractions

- "cloud hosting services" and "two years" are mentioned but
  the ontology doesn't model Service or Duration. False
  positives if surfaced.
