# gene_protein

A short molecular-biology annotation. Tests a three-entity
chain (Gene → Protein → Function) and non-name canonical
identity on Gene.

## What this exercises

- **Three-entity transitive chain**: Gene → Protein →
  Function. The protein appears as the target of one edge
  and the source of the next; pin that the edge-matching
  layer doesn't conflate the protein's two roles.
- **Non-name canonical identity**: Gene uses ``symbol`` not
  ``name`` as the unique property. The canonical_key is
  ``Gene:symbol=brca1``. Pin alongside government_regulation
  for the broader "identity is whatever the ontology says,
  not name" coverage.
- **Long entity names**: the protein name spans seven words
  ("breast cancer type 1 susceptibility protein"). Pin
  against an extractor that truncates or normalizes long
  names.

## Failure signals

- An extractor that creates two Protein nodes (one for
  "breast cancer type 1 susceptibility protein" and one
  for an abbreviation it hallucinates) drops precision.
- An extractor that emits a Gene → Function edge directly
  (skipping Protein) drops the modeled topology.

## Intentional non-extractions

- "loss-of-function variants", "hereditary breast and
  ovarian cancer", "downstream effectors" are content the
  ontology does not model.
