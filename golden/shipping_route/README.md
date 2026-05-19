# shipping_route

A short multi-carrier route summary. Tests two carriers
sharing identical origin/destination ports.

## What this exercises

- **Shared origin and destination across carriers**: Both
  Carriers point at the same Shanghai (origin) and Long
  Beach (destination) Port nodes. Pin that the dedup logic
  doesn't merge the carriers based on shared port
  endpoints.
- **Two-word Port identity**: ``Long Beach`` is keyed on
  its space-separated name. Pin against a tokenizer that
  splits on whitespace.

## Failure signals

- An extractor that creates Long Beach-A (Pacific Star
  side) and Long Beach-B (Trident side) as separate Port
  nodes inflates Port FP.
- An extractor that swaps ORIGIN_OF and DESTINATION_OF
  for one carrier (semantic-mirror error) drops edge
  precision.

## Intentional non-extractions

- "Wednesday-departure", "Saturday-departure", "deep-water
  terminal", "alternating sailings", "twice-weekly
  capacity", "Shanghai-Long Beach corridor" are content
  the ontology does not model.
