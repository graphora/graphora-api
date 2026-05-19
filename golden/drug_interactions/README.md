# drug_interactions

A short pharmacology summary. Tests same-type edges in a
"hub-and-spoke" pattern where one drug interacts with two
others.

## What this exercises

- **Hub-and-spoke same-type pattern**: warfarin is the hub
  connected to two spokes (aspirin, amoxicillin). The
  expected graph has both INTERACTS_WITH edges sharing the
  source. Pins that the edge-matching layer doesn't
  conflate same-source-different-target edges.
- **Directional reading of an inherently symmetric
  relation**: drug interactions are logically symmetric, but
  the ontology models them as a directional edge per pair.
  An extractor that emits both directions (warfarin→aspirin
  AND aspirin→warfarin) drops precision.
- **Explicit non-relationship**: the document states "no
  established interaction" between aspirin and amoxicillin.
  Pin against an extractor that hallucinates the third edge
  to "complete" the triangle.

## Failure signals

- An extractor that emits an aspirin→amoxicillin
  INTERACTS_WITH edge despite the document's explicit
  negation drops relationship FP.
- An extractor that emits both directions for each
  documented interaction doubles the INTERACTS_WITH count
  and halves precision.

## Intentional non-extractions

- "international normalized ratio", "INR", "bleeding risk",
  "anticoagulant", "antiplatelet", "dose adjustment" are
  content the ontology does not model.
