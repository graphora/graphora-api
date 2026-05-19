# medical_specialty

A short physician profile. Tests two parallel edges of the
same type from one source (BOARD_CERTIFIED_IN), plus an edge
of a different type from the same source.

## What this exercises

- **Two same-type edges sharing a source**: Both
  BOARD_CERTIFIED_IN edges originate at Dr. Aisha Patel and
  fan out to different Specialty nodes. Pin against an
  extractor that conflates "double board-certified" into a
  single edge with a composite target.
- **Honorific in canonical_key**: ``Dr. Aisha Patel`` keeps
  the ``Dr.`` prefix in the canonical_key. The expected
  encodes this as ``Doctor:name=dr. aisha patel``.
- **Specialty-as-shared-vocabulary**: ``cardiology`` and
  ``internal medicine`` are referenced twice each. Pin that
  the helper handles multi-word specialty names without
  normalization.

## Failure signals

- An extractor that strips the ``Dr.`` honorific for the
  canonical_key produces a different canonical_id from the
  helper recomputation — the invariant test catches this
  class.
- An extractor that emits a single BOARD_CERTIFIED_IN edge
  with target "cardiology and internal medicine" (parsing
  the conjunction as a compound noun) drops Specialty
  recall.

## Intentional non-extractions

- "preventive interventions", "complex multi-organ cases",
  "fellowship in 2014" are descriptive content the ontology
  does not model.
