# legislation_sponsorship

A short legislative summary. Tests a Bill referenced four
times by its prefixed number plus two committee referrals.

## What this exercises

- **Bill number with punctuation**: ``H.R. 4521`` carries
  the period after ``H.R`` and a space before ``4521``.
  Pin that the canonical_key preserves both (lowercased).
- **Two REFERRED_TO edges sharing a source**: Both committee
  edges originate at the Bill. Pin against an extractor
  that emits one composite "joint committee" Committee
  node.
- **Multi-mention dedup on the Bill**: ``H.R. 4521``
  appears five times.

## Failure signals

- An extractor that strips the period from ``H.R. 4521``
  (yielding ``HR 4521``) produces a different canonical_key.
- An extractor that misses the second committee referral
  (``House Ways and Means Committee``) drops REFERRED_TO
  recall.

## Intentional non-extractions

- "initial markup", "tax provisions", "floor action", "fall
  session", "next month" are content the ontology does not
  model.
