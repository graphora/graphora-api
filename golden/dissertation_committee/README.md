# dissertation_committee

A short dissertation-committee description. Tests
ASCII-foldable non-ASCII names and overlapping
relationship types (advisor is also a committee member).

## What this exercises

- **Non-ASCII in canonical_key**: ``Ana Rodríguez`` carries
  the ``í`` character. The canonical_key
  (``Student:name=ana rodríguez``) preserves the
  non-ASCII; pin against a normalizer that ASCII-folds.
- **Two edges of different types sharing the same
  source-target pair**: Prof. Karim Hassan has both ADVISES
  and COMMITTEE_FOR pointing at Ana Rodríguez. The two
  edges coexist — the advisor IS on the committee. Pin
  against an extractor that picks one and drops the other.
- **Three faculty, one student**: Three COMMITTEE_FOR
  edges converge on Ana Rodríguez. Pin that the
  edge-matching layer handles the converge pattern.

## Failure signals

- An extractor that ASCII-folds ``Rodríguez`` to ``Rodriguez``
  fails the helper-recomputation invariant.
- An extractor that emits only ADVISES for Prof. Karim
  Hassan (treating the advisor role as overriding committee
  membership) drops COMMITTEE_FOR recall.

## Intentional non-extractions

- "primary advisor", "second reader", "external perspective",
  "examining committee", "fall 2026" are content the ontology
  models only partially — the advisor/reader distinction
  isn't captured at the edge-property level.
