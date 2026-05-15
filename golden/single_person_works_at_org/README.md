# single_person_works_at_org

The canonical Gate-2 example: one Person, one Organization, one
WORKS_AT edge connecting them. Smallest doc that exercises the
end-to-end extraction → identity-matching → relationship surface
without combinatorial entity-resolution concerns.

## What this tests

- **Entity extraction**: Person + Organization from two short
  sentences. Both have clear surface forms (`Alice Martinez`,
  `Acme Corp`) so canonical_key derivation is unambiguous.
- **Relationship extraction**: A WORKS_AT edge with an optional
  `start_date` property. The date appears inline in the source.
- **Optional property extraction**: `title="senior engineer"` on
  the Person node. Tests whether the extractor surfaces
  schema-declared-but-not-required properties.

## What this does NOT test

- Multi-entity disambiguation (no second "Alice" / "Acme")
- Cross-sentence coreference (`She works on...` is third-party
  context, not a new fact we expect surfaced)
- Schema inference (the ontology is pre-supplied)
- ER weighting / disputed-pair queueing (B2-active surfaces)

## Source

Synthetic. Written for this corpus — no third-party content.
