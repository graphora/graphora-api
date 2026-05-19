# manufacturing_supply

A short manufacturing description: one Product consumes two
Components, each Component sourced from a different
Supplier. Tests a two-step chain through an intermediate
entity type.

## What this exercises

- **Two-step transitive structure**: Product → Component →
  Supplier. The supplier relationship is anchored on the
  Component, not directly on the Product. A flatter ontology
  ("Product SUPPLIED_BY Supplier") would collapse the chain
  but lose which component each supplier delivers; the
  Component-as-bridge pattern is the test signal.
- **Multi-paragraph dedup across three entity types**:
  "Model X" appears three times, "Acme Materials" twice,
  "Beta Metals" twice, "lithium battery" twice, "aluminum
  frame" twice. Each must collapse to a single canonical node.
- **Asymmetric edges**: the two USES edges share the Product
  source but differ in target; the two SUPPLIED_BY edges
  share no node. Verifies the edge-grouping logic doesn't
  conflate "different edges from same source" with
  duplicates.

## Failure signals

- An extractor that flattens the chain into a direct
  `Product → Supplier` edge (skipping Component) would emit
  edges with mismatched ontology types and drop relationship
  precision against the published vocabulary.
- An extractor that emits `lithium battery` and `aluminum
  frame` as siblings under a generic "Material" type ignores
  the Component type and drops both Component recall (no
  matched Component nodes) and SUPPLIED_BY recall (no
  Component-source for the edge).

## Intentional non-extractions

- "Ohio-based cell manufacturer", "Pittsburgh", "quality
  assurance review", "specification", and "master schedule"
  are content the document carries but the ontology does
  not model. A faithful extractor should not surface them
  as unmodeled entity types or speculative location/QA
  edges.
