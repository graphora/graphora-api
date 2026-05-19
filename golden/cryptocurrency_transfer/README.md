# cryptocurrency_transfer

A short on-chain transfer summary. Tests case-insensitive
hex-address dedup and same-type Wallet → Wallet edges.

## What this exercises

- **Same-type edge with hex-address identity**: The SENT_TO
  edge connects two Wallets. Each Wallet is identified by
  its hex address (40 chars + ``0x`` prefix). Pin against
  an extractor that strips the ``0x`` prefix from one
  occurrence but not the other.
- **Case-insensitive canonical_key on hex addresses**: The
  document uses mixed-case checksummed addresses
  (``0x742d35Cc...``). The canonical_key lowercases. Pin
  that the helper-recomputation invariant doesn't trip on
  this normalization.
- **Token-amount-as-property**: The 1,250,000 USDC amount
  is metadata on the Token node, not identity. canonical_key
  is ``Token:symbol=usdc`` — pin that "amount" isn't
  promoted to identity even though it's the most prominent
  detail in the document.

## Failure signals

- An extractor that creates separate Wallet nodes for the
  mixed-case and lowercase forms of the same address drops
  Wallet precision.
- An extractor that creates separate Token nodes per
  transfer mention (three mentions of "USDC" in this doc)
  drops Token precision.

## Intentional non-extractions

- "May 8, 2026", "prior transfer", "one block earlier",
  "centralized exchange deposit address", "single block"
  are content the ontology does not model.
