# family_relationships

A short family description: two parents, two children, one
marriage. Tests a single-entity-type ontology with multiple
same-type relationships.

## What this exercises

- **Single-entity-type ontology**: Person is the only modeled
  entity type. All relationships have `Person` for both
  source and target. Pins that the ontology system handles
  the degenerate case where every node and every edge
  endpoint is the same type.
- **Multiple same-type relationships**: `MARRIED_TO` and
  `PARENT_OF` both connect Person to Person — distinguished
  only by the relationship name. The expected graph has 1
  MARRIED_TO edge and 4 PARENT_OF edges between the same
  pool of four nodes; pin that edge-type discrimination
  isn't conflated with endpoint-type discrimination.
- **Directional reading of inherently symmetric relations**:
  marriage is logically symmetric, but the ontology models
  it as a single directional edge (Alice → Bob, not both
  directions). An extractor that emits two edges (one each
  way) would inflate the MARRIED_TO count and drop precision.
- **Multi-source convergence on the same target**: Charlie
  has two PARENT_OF edges pointing at him (one from Alice,
  one from Bob); same for Diana. Pin that the edge-matching
  layer treats these as distinct edges (different sources)
  not duplicates.

## Failure signals

- An extractor that emits a second MARRIED_TO edge
  (`Bob → Alice`) to capture the symmetric reading drops
  MARRIED_TO precision: 1 expected vs. 2 actuals would
  record `TP=1, FP=1, FN=0` → `MARRIED_TO.precision = 0.5`,
  while recall stays at 1.0.
- An extractor that misses one parent for one child (e.g.,
  only emits `Alice → Charlie` and `Bob → Diana`, missing
  the cross pairs) drops PARENT_OF recall: 4 expected vs. 2
  actuals → `TP=2, FP=0, FN=2`.
- A name-parser that strips the shared "Carter" surname for
  the canonical_key (treating it as a redundant suffix)
  would collapse Alice/Bob/Charlie/Diana incorrectly only if
  the parser ALSO collapses the first names — but the more
  common failure mode is over-trimming whitespace or
  punctuation, which this corpus doesn't exercise.

## Intentional non-extractions

- "Spring of 2018", "born in 2020", "born in 2023", "software
  engineer", "high school physics teacher", and "Lincoln
  Park neighborhood" are content the document carries but
  the ontology does not model. A faithful extractor should
  not surface them as unmodeled entity types or
  speculative occupation/location edges.
