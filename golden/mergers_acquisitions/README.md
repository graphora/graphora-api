# mergers_acquisitions

A short M&A announcement. Tests a transaction entity with
non-identity properties (closing_date) and same-type
self-referential edges (Company → Company via ACQUIRED).

## What this exercises

- **Date-as-property, not identity**: ``closing_date`` is on
  the Acquisition's properties but NOT part of the
  canonical_key. The helper-derived canonical_key is
  ``Acquisition:reference=hex-oct-2026-03`` — purely
  reference-based. Pins that ``unique: true`` selects which
  properties drive identity; the rest ride along as
  metadata.
- **Same-type self-referential edge**: ``ACQUIRED`` has
  Company for both source and target. The expected graph has
  one such edge (Hexagon → Octane).
- **Three-way relationship around a transaction**: The
  Acquisition node carries the transaction identity; the two
  Company nodes carry the parties. The three edges (ACQUIRED,
  ACQUISITION_OF, ACQUISITION_BY) form a triangle that lets
  downstream queries reach the deal record from either party
  AND get the parties from the deal record.

## Failure signals

- An extractor that merges the deal record into the Company
  nodes (e.g., adds ``acquired_on`` as a property on Hexagon
  Industries) loses the Acquisition entity entirely — recall
  drops on Acquisition.
- An extractor that creates separate Acquisition nodes for
  "HEX-OCT-2026-03" and "the deal" (anaphoric reference)
  inflates Acquisition precision noise.

## Intentional non-extractions

- "three-state manufacturing footprint", "autonomous-systems
  group", "robotics division", and the "eighteen months
  post-closing" timeline are content the ontology does not
  model. A faithful extractor should not surface them.
