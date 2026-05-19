# incident_response

A short incident-response timeline. Tests one Incident with
fan-out to two Systems and two Engineers.

## What this exercises

- **Two parallel fan-outs from one source**: The Incident
  has two IMPACTS edges and two RESPONDED_BY edges. Pin
  that the edge-matching layer handles four edges sharing
  a source plus two pairs sharing target type.
- **System-vs-Engineer separation**: Both entity types use
  ``name`` as identity. Pin that the type discriminator
  keeps ``ledger-db`` and "Priya Ramanathan" from
  colliding even though the canonical_key shape is
  identical.

## Failure signals

- An extractor that elevates the SEV classification (SEV-2)
  to identity would diverge from the helper recomputation
  on identifier-keyed Incident.
- An extractor that creates a single composite Engineer
  node for "Priya Ramanathan and Daniel Schmidt" parses
  the conjunction as a compound noun and drops Engineer
  recall.

## Intentional non-extractions

- "14:23 UTC", "47 minutes", "transaction log", "5xx
  rates", "triage", "rotated" are content the ontology
  does not model.
