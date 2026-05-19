# investment_round

A short startup-funding announcement. Tests fan-in to a Round
node from multiple Investors plus one Startup.

## What this exercises

- **Two PARTICIPATED_IN edges sharing a target**: Both
  investors point at the same Round. Pin against an
  extractor that conflates "co-led by Sequoia Capital and
  Andreessen Horowitz" into one composite investor node.
- **Multiple non-identity properties on the Round**:
  ``amount`` (``$45M``) and ``round_type`` (``Series B``)
  ride along; only ``identifier`` is unique-flagged.
- **Multi-mention dedup on the Round entity**:
  Polyhedron-Series-B-2026 is referenced four times.

## Failure signals

- An extractor that creates separate Round nodes per
  mention (``Series B funding round`` vs
  ``Polyhedron-Series-B-2026`` vs ``the round``) drops
  Round precision.
- An extractor that emits a phantom Investor for the
  generic "co-leads" phrase (without naming specific firms)
  inflates Investor FP.

## Intentional non-extractions

- "April 15, 2026", "seed round in 2023", "training
  infrastructure" are content the ontology does not model.
