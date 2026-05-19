# case_law_citation

A short legal opinion summary. Tests Bluebook-citation
identity and judge attribution.

## What this exercises

- **Long-string canonical_key**: The Case's citation is a
  35+ character Bluebook string. Pin that the helper handles
  long, punctuated, mixed-case strings as canonical_keys
  without losing or normalizing structure.
- **Multi-mention dedup on the Case**: The full citation
  string appears three times in the document and must
  collapse to one node.
- **Court-vs-Judge as separate entity types**: Both have
  ``name`` as their unique property, but the type
  discriminator keeps them distinct in the canonical_key
  (``Court:name=...`` vs ``Judge:name=...``). Pin against
  an extractor that merges them on shared property names.

## Failure signals

- An extractor that creates two Case nodes (one for the
  full Bluebook citation, one for "Brennan v. Adler"
  shortform) drops Case precision.
- An extractor that surfaces "Third Circuit" as a separate
  Court node from "United States Court of Appeals for the
  Third Circuit" (treating them as different courts) drops
  Court precision.

## Intentional non-extractions

- "qualified immunity", "majority opinion", "prior
  precedent" are content the ontology does not model.
