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

- An extractor that produces two separate Agreement nodes for
  the two "Service Agreement" mentions drops **precision** on
  the Agreement type. With one expected and two actuals that
  share the helper-derived canonical_id
  ("Agreement:name=service agreement"), the scorer records
  `TP=1, FP=1, FN=0` → `Agreement.precision ≈ 0.5`,
  `Agreement.recall = 1.0`.
  `PARTY_TO` edges are **not** affected in this case: both
  edges in the actual graph point at the same canonical
  Agreement identity, so the edge-matching layer aligns them
  with the two expected edges and `PARTY_TO` stays at
  `TP=2, FP=0, FN=0`. Edges only take a hit when the
  duplicate Agreement has a *different* canonical_id from
  the matched one (e.g., from name-parsing drift like
  "Service Agreement" vs "Service agreement.") — then one of
  the `PARTY_TO` edges points at the unmatched duplicate and
  both PARTY_TO precision and recall drop.
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
