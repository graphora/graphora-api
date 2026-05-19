# degree_conferred

A short degree-conferral summary. Tests degree-as-record
entity separate from the institution.

## What this exercises

- **Long-form Institution name**: ``Massachusetts Institute
  of Technology`` is the canonical form; ``MIT`` would
  have to collapse to the same canonical_id. Pin against a
  pre-processor that uses the abbreviated form as identity.
- **Degree-as-record**: The Degree node carries the
  identifier; the Person earns it; the Institution
  confers it. Three-way star pattern with the Degree
  intermediating.
- **Multi-mention dedup on the Degree**:
  ``MIT-PhD-EECS-2025-OPark`` appears twice; the institution
  three times; the person three times.

## Failure signals

- An extractor that creates an Institution node for "MIT"
  separately from "Massachusetts Institute of Technology"
  (treating the abbreviation as a distinct identity)
  inflates Institution FP. The text doesn't actually use
  "MIT" — that's a future-test signal for an extended
  version of this entry.
- An extractor that merges Person and Degree (Olivia Park
  is both the person AND the degree-bearer) loses the
  Degree node entirely.

## Intentional non-extractions

- "neural circuit design", "hardware constraints",
  "industrial research lab", "spring 2025 commencement
  ceremony", "registrar's records" are content the
  ontology does not model.
