# loan_collateral

A short secured-loan summary. Tests two distinct entities of
the same type (Asset) both connecting to the same target.

## What this exercises

- **Two same-type-source edges**: Both COLLATERAL_FOR edges
  share the Borrower target; their sources are different
  Asset nodes. Pin that the dedup logic doesn't conflate
  Assets on shared target.
- **Equipment identifier as canonical identity**: Asset
  uses ``identifier`` (e.g., ``EQUIP-CRANE-7831``) as the
  unique property. Mixed letters / numbers / hyphens stay
  in the lowercased canonical_key.
- **Modeling choice: Asset → Borrower, not Asset → Lender**:
  Pin against an extractor that wires assets to the lender
  ("Lender holds collateral") — the ontology vocabulary
  models the obligation side.

## Failure signals

- An extractor that creates one composite Asset node for
  "two pieces of equipment" (parsing the conjunction)
  drops Asset recall.
- An extractor that emits COLLATERAL_FOR from Asset to
  Lender (semantic-mirror error) drops relationship
  precision against the published vocabulary.

## Intentional non-extractions

- "$4.2M term loan", "security agreement", "perfected
  security interest", "loan term", "operational use" are
  content the ontology does not model.
