# cloud_deployment

A short cloud-architecture summary. Tests duplicate parallel
fan-outs (two sources, both pointing at the same set of three
regions).

## What this exercises

- **Two parallel three-way fan-outs**: Service and Cluster
  both connect to the same three Region nodes. Six edges
  total, with three Region nodes shared between two
  different-type sources. Pin that the edge-matching layer
  handles deeply-shared target reuse.
- **Region-code canonical identity**: ``us-east-1`` etc.
  use ``code`` as the unique property. Pin against a
  normalizer that splits the dashed code into multiple
  fields.

## Failure signals

- An extractor that creates separate Region nodes per
  fan-out source (six Region nodes total instead of three)
  drops Region precision badly.
- An extractor that misses the third region (``ap-southeast-2``)
  is asymmetric — it might pass DEPLOYED_TO but fail SPANS
  if the second paragraph's mentions don't link back.

## Intentional non-extractions

- "multi-AZ", "early 2025", "March 2026", "control plane",
  "quorum-based consensus" are content the ontology does
  not model.
