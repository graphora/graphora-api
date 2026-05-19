# stock_purchase

A short equity-block-trade summary. Tests direction-sensitive
edges (PURCHASED vs SOLD) on the same Stock.

## What this exercises

- **Two opposite-direction edges sharing a target**: Both
  edges point at the same Stock; the direction of the
  trade lives in the edge type (PURCHASED vs SOLD). Pin
  against an extractor that conflates them or swaps the
  edge type.
- **Ticker-as-identity (uppercase source)**: ``GLBX`` is
  uppercase in the document but the canonical_key
  lowercases. Pin that the helper-recomputation invariant
  doesn't trip on the case discrepancy.
- **Multiple non-identity numerical properties**:
  ``quantity`` and ``price`` ride along on the Stock node
  but neither is part of the canonical_key.

## Failure signals

- An extractor that emits ``PURCHASED`` for both edges
  (taking the buyer-perspective for both) loses the
  bilateral structure of the trade.
- An extractor that creates a separate Stock node per
  trade (treating ``250,000 shares of GLBX`` as a distinct
  identity from the bare ``GLBX``) inflates Stock count.

## Intentional non-extractions

- "T+1 basis", "4.2 percent stake", "since 2021",
  "realized a gain" are content the ontology does not
  model.
