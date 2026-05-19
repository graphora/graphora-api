# patent_inventors

A short patent-grant announcement. Tests three INVENTED_BY
edges sharing a source plus one ASSIGNED_TO edge.

## What this exercises

- **Three same-type edges sharing a source**: All three
  INVENTED_BY edges originate at the Patent and fan out to
  distinct Inventor nodes. Pin against an extractor that
  emits one composite edge with target "Tanaka, Ahmed, and
  Rossi" (parsing the list as a compound noun).
- **Star pattern with mixed edge types**: Patent is the
  center; three INVENTED_BY edges plus one ASSIGNED_TO edge
  share the Patent source. Pin that the edge-matching
  layer doesn't conflate edges of different types from the
  same source.
- **Order-of-listing irrelevant to identity**: Kenji Tanaka
  is the "first-named inventor", but the ontology doesn't
  model inventor order. Pin that an extractor doesn't smuggle
  in an ORDERED_FIRST property to capture this.

## Failure signals

- An extractor that misses one inventor (e.g., only
  captures the first-named one) drops INVENTED_BY recall by
  2/3.
- An extractor that emits an ASSIGNED_TO edge from each
  Inventor (parsing "All three inventors assigned their
  rights") inflates ASSIGNED_TO from 1 to 4.

## Intentional non-extractions

- "March 17, 2026", "first-named inventor", "co-inventors",
  "patent portfolio" are content the ontology does not
  model.
