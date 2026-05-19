# academic_paper_citation

Multi-author paper citing another paper. Tests **same-type
self-referential edges** (`Paper → Paper` via `CITES`) plus
**multi-edge from same source** (one Paper, two `AUTHORED_BY`
edges).

## What this exercises

- **Same-type endpoints on an edge**: `CITES` source AND
  target are both `Paper`. The scorer must distinguish them
  by id, not type — a regression that uses (source_type,
  target_type) as edge identity would conflate
  `pgm-cites-bn` with any other `Paper→Paper` edge.
- **Multi-edge from one source**: the "Probabilistic Graph
  Models" Paper has two `AUTHORED_BY` edges (to Patel and to
  Chen). Both must surface as distinct.
- **Bare-name reference resolution**: the document says "Pearl"
  in body text after introducing him as "Judea Pearl" in the
  first sentence. The expected canonical_id for Pearl is
  keyed on the full name; the extractor should resolve the
  bare reference back to the full name (one canonical
  `Author`, not a separate "Pearl"-only node).

## Failure signals

- An extractor that creates a "Pearl" Author distinct from
  "Judea Pearl" → `Author.precision < 1.0` (extra node) and
  one of the citation-related edges may dangle.
- An extractor that puts the cited paper's `year` on the wrong
  Paper node (assigning 2025 to "Bayesian Networks" instead
  of 1988) → still matches identity but property delta
  surfaces via /diff.

## Intentional non-extractions

- "Probabilistic graphical models" (the field) and the
  introductory text are descriptive context, not entities.
  No `Field` type in the ontology — surfacing such a node
  would be an FP.
