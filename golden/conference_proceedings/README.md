# conference_proceedings

A short conference-acceptance summary. Tests one Author
linked to two distinct Papers (Hana Suzuki).

## What this exercises

- **One Author, multiple Papers**: Hana Suzuki has two
  AUTHORED_BY edges incoming. The expected graph keeps her
  as one node referenced by two separate edges. Pin
  against an extractor that creates a "Hana Suzuki (paper
  1)" / "Hana Suzuki (paper 2)" disambiguation when no
  ambiguity exists.
- **Long-title canonical_key**: paper titles are 50+
  character strings. Pin alongside case_law_citation for
  the general long-string identity coverage.
- **Three ACCEPTED_AT edges sharing a target**: All three
  Papers point at NeurIPS 2025 via ACCEPTED_AT.

## Failure signals

- An extractor that title-cases or trims the paper titles
  (changing punctuation or case) produces canonical_keys
  that don't match the helper output.
- An extractor that misses Felipe Costa (who is named only
  once, in a co-author context) drops Author recall.

## Intentional non-extractions

- "oral presentation", "spotlight", "poster", "our group"
  are content the ontology does not model.
