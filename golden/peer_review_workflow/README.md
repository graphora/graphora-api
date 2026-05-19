# peer_review_workflow

A short peer-review workflow snapshot. Tests pseudonymous
identity (``Reviewer 1``) and reviewer/editor role
separation.

## What this exercises

- **Pseudonymous canonical_key**: ``Reviewer 1`` and
  ``Reviewer 2`` are functional pseudonyms — their
  canonical_keys (``Reviewer:name=reviewer 1`` and
  ``Reviewer:name=reviewer 2``) collide on the integer
  suffix only. Pin against a normalizer that strips
  numeric suffixes.
- **Reviewer-vs-Editor type separation**: Both use ``name``
  as identity; the type discriminator keeps them apart.

## Failure signals

- An extractor that merges ``Reviewer 1`` and ``Reviewer 2``
  on the ``Reviewer`` prefix (treating the suffix as
  positional rather than as part of the identity) drops
  Reviewer count.
- An extractor that creates a single composite
  Reviewer + Editor node for "the review committee" drops
  type-level recall.

## Intentional non-extractions

- "March 1, 2026", "minor revisions", "major revisions",
  "mediate the differences" are content the ontology does
  not model.
