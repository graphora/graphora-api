# vendor_invoice

A short commercial-invoice description. Tests numeric/currency
properties as metadata on a transaction entity.

## What this exercises

- **Multiple non-identity properties**: Invoice carries
  ``total_amount`` and ``line_item_count`` alongside
  ``number``. Only ``number`` is unique-flagged, so the
  canonical_key is ``Invoice:number=inv-2026-0421``. Pins
  that multiple non-unique properties ride along as
  metadata without affecting identity.
- **Five-mention dedup on the transaction entity**:
  INV-2026-0421 appears five times across the document. Pin
  on an extractor that creates a fresh Invoice node per
  mention.
- **Currency formatting preservation**: ``$3,481.50 USD``
  keeps its symbol, comma, and currency label intact in the
  expected ``total_amount``. Pin against an extractor that
  parses-and-reformats the amount.

## Failure signals

- An extractor that creates two Invoice nodes (one for
  "invoice INV-2026-0421" and one for "INV-2026-0421"
  bare) drops Invoice precision.
- An extractor that parses the amount into a float
  (3481.50) loses the currency-label context.

## Intentional non-extractions

- "Net 30 terms", "accounts payable team", "April 23", and
  "April 21, 2026" are content the document carries but the
  ontology does not model.
