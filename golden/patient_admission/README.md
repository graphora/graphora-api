# patient_admission

A short hospital-admission summary. Tests PHI-safe identity
patterns and date-as-property.

## What this exercises

- **PHI-safe identity by design**: Patient identity is the
  MRN, not a name. The canonical_key is
  ``Patient:mrn=mrn-88241``. Pin so an extractor that
  promotes "patient name" into the canonical_key in some
  future variant doesn't slip through.
- **Three-mention dedup across two entity types**:
  ``MRN-88241`` and ``Riverside General Hospital`` are each
  referenced three times.
- **Date-as-property**: ``admission_date`` rides along on
  the Patient as metadata; not part of the canonical_key.

## Failure signals

- An extractor that derives a Patient identity from
  contextual cues (e.g., creating a "patient who had
  cholecystectomy" identity rather than using MRN) drops
  Patient recall against the MRN-keyed expected.
- An extractor that emits a separate Procedure node for the
  follow-up clinic visit (when the ontology doesn't
  distinguish initial-vs-followup) inflates Procedure FP.

## Intentional non-extractions

- "May 3, 2026", "May 5, 2026", "two weeks post-discharge",
  "surgery clinic", "without complications" are content the
  document carries but the ontology does not model.
