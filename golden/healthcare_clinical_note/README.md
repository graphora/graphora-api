# healthcare_clinical_note

Single clinical encounter: Patient + Doctor + Diagnosis with
two relationship types. First non-business-domain entry — adds
healthcare to the corpus mix.

## What this exercises

- **Cross-domain entity types** (Patient / Doctor / Diagnosis)
  to validate the extractor doesn't anchor only on
  business-tech entity names.
- **Multiple edge types from one source** (Patient → Doctor
  via SEEN_BY, Patient → Diagnosis via DIAGNOSED_WITH).
- **Honorific normalization**: the document says "Dr. Robert
  Smith" but the expected Doctor canonical_key is
  `Doctor:name=robert smith` (lowercased, honorific stripped).
  An extractor that keeps "Dr." in the name property would
  fail the canonical_id match.

## Failure signals

- An extractor that emits the wrong Doctor canonical_id
  (e.g., from leaving "Dr." in the name) → `Doctor.recall = 0`,
  and the `SEEN_BY` edge dangles.
- An extractor that creates a `Doctor` + `Person` for the same
  individual → schema-level drift (the ontology has no
  Person type here).

## Intentional non-extractions

- "Mercy Hospital" appears in the document but the ontology
  doesn't declare a `Hospital` entity. Per the ontology
  contract, extractions for types not in the schema should be
  dropped — surfacing a Hospital node would be a false positive
  the scorer would surface as `nodes.precision < 1.0`.
- "Metformin therapy" is a treatment the ontology doesn't
  model. Same intentional drop.
