# mortgage_origination

A short residential-mortgage summary. Tests street-address
canonical identity with punctuation.

## What this exercises

- **Full-address canonical_key**: ``412 Magnolia Lane,
  Charleston, SC 29401`` includes comma-separated components.
  The canonical_key lowercases (``Property:address=412
  magnolia lane, charleston, sc 29401``) but keeps commas
  and spaces intact. Pin against an extractor that
  normalizes addresses (e.g., USPS standard form) — that
  would diverge from the helper output.
- **Three-mention dedup on Property**: The full address
  appears three times.
- **Two edges sharing a target**: Both ORIGINATED and
  BORROWER_OF point at the Property. Pin that
  different-edge-type edges to the same target stay
  distinct.

## Failure signals

- An extractor that emits two Property nodes (one for
  "412 Magnolia Lane" shortform, one for the full address)
  drops Property precision.
- An extractor that swaps the direction of ORIGINATED
  (Property → Bank instead of Bank → Property) reads as
  "the property originated a mortgage at the bank" —
  semantically nonsense — and drops edge precision.

## Intentional non-extractions

- "30-year fixed", "80 percent", "6.85 percent", "primary
  residence" are content the ontology does not model.
