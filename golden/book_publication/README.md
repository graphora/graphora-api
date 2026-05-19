# book_publication

A short book-release summary. Tests ISBN-as-identity and
non-identity title field.

## What this exercises

- **ISBN-hyphenated canonical_key**: ``978-1-2345-6789-0``
  stays hyphenated in the canonical_key. Pin against a
  normalizer that strips hyphens.
- **Title as non-identity property**: The book's title
  rides as a property; identity is the ISBN. Three of the
  four mentions in the document reference the book by
  title; the ontology's choice of ISBN-as-identity means
  the title-only mentions still need to resolve back to
  the same Book node.
- **Author + Publisher both single-edge fan-out**: The
  Book has one AUTHORED_BY and one PUBLISHED_BY edge.

## Failure signals

- An extractor that creates separate Book nodes for the
  ISBN-bearing mentions and the title-only mentions
  (because canonical_id derivation would diverge) drops
  Book precision badly. The pin lives in the
  helper-recomputation invariant.

## Intentional non-extractions

- "debut novel", "five years writing", "at auction in
  2024", "advance galleys", "national tour" are content the
  ontology does not model.
