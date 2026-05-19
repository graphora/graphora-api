# political_campaign

A short campaign-launch summary. Tests a four-entity star
pattern centered on the Candidate.

## What this exercises

- **Title-as-prefix in canonical_key**: ``Senator Rebecca
  Cho`` carries the ``Senator`` honorific in the
  canonical_key (``Candidate:name=senator rebecca cho``).
  Pin alongside medical_specialty's ``Dr. Aisha Patel`` for
  the honorific-preservation coverage.
- **Three different-typed targets from one source**: The
  Candidate has three edges (RUNNING_FOR, IN_DISTRICT,
  ENDORSED_BY) each pointing at a different entity type.
  Pin that the edge-matching layer doesn't fold them on
  source identity.
- **Identifier-keyed District**: ``Oregon`` is keyed on
  ``identifier`` not ``name`` — pin coverage for
  jurisdiction-style identities.

## Failure signals

- An extractor that strips the ``Senator`` prefix from
  the canonical_key creates a name-only identity
  (``Rebecca Cho``) that doesn't match the helper
  recomputation.
- An extractor that combines ``Office`` and ``District``
  into a single "Governor of Oregon" entity loses the
  modeled separation.

## Intentional non-extractions

- "infrastructure investment", "third candidate", "campaign
  launch speech", "this cycle" are content the ontology
  does not model.
