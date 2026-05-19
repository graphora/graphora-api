# experimental_dataset

A short experimental-dataset summary. Tests two
COLLECTED_BY edges sharing a source plus a Method edge.

## What this exercises

- **Two same-type edges sharing a source**: COLLECTED_BY
  fan-out to both researchers. Pin against an extractor
  that parses "Dr. Yumi Tanaka and Dr. Marcus Webb" as a
  compound author.
- **Method-as-shared-vocabulary**: ``X-ray diffraction``
  could just as easily appear in another entry; pin that
  the canonical_key (``Method:name=x-ray diffraction``)
  lowercases without further normalization.
- **Multi-mention dedup on the Dataset**: ``DS-2026-CRYSTAL-001``
  appears four times.

## Failure signals

- An extractor that creates separate Method nodes for "X-ray
  diffraction" and "synchrotron beamline" (treating both
  technique descriptions as distinct methods) inflates
  Method FP.
- An extractor that combines the two researchers into a
  single Researcher node ("Tanaka and Webb") drops
  Researcher recall.

## Intentional non-extractions

- "4,800 measurements" (the count rides as a property),
  "synchrotron beamline", "consortium", "community use",
  "collection and curation" are content the ontology does
  not model.
