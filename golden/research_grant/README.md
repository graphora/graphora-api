# research_grant

A short grant-award announcement. Tests a four-entity
star pattern centered on the Grant.

## What this exercises

- **Grant as the center of three different edge types**:
  AWARDED_BY (Grant → Funder), PI_OF (Researcher → Grant),
  FUNDS (Grant → Project). Pin that the edge-matching
  layer handles directional asymmetry — one edge points AT
  the Grant; two point AWAY.
- **Honorific in canonical_key**: ``Dr. Rohan Mehta``
  carries the ``Dr.`` prefix in the canonical_key, mirroring
  medical_specialty's Dr. Aisha Patel.
- **Multi-mention dedup on the Grant**: ``NSF-2401234``
  appears five times.

## Failure signals

- An extractor that emits the Grant as a property on the
  Researcher (rather than as its own entity) loses the
  Grant node entirely.
- An extractor that reverses the FUNDS direction (Project
  → Grant) reads as "the project funds the grant" —
  semantic-mirror error.

## Intentional non-extractions

- "$1.2M over three years", "two cohorts of graduate
  students", "formal methods", "machine learning" are
  content the ontology does not model.
