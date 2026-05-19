# interview_quotes

A short interview-summary description. Tests one Speaker
discussing multiple Topics.

## What this exercises

- **Two DISCUSSED edges from one source**: The Speaker has
  two outgoing DISCUSSED edges. Pin against an extractor
  that conflates "two main topics" into one composite
  Topic.
- **Multi-word Topic identity**: ``AI safety governance``
  and ``open-source model release`` are 3-4 word
  canonical_keys. Pin that the helper preserves spaces and
  hyphens in the lowercased form.
- **Publication-as-entity**: ``The Atlantic`` keeps its
  definite article in the canonical_key
  (``Publication:name=the atlantic``).

## Failure signals

- An extractor that strips ``The`` from ``The Atlantic``
  (yielding just ``Atlantic``) produces a different
  canonical_key than the helper.
- An extractor that promotes "mandatory licensing regime"
  to a Topic (treating Dr. Marshall's position as the
  topic itself) inflates Topic FP.

## Intentional non-extractions

- "long-form interview", "mandatory licensing regime",
  "frontier models", "permissive stance", "AI policy",
  "follow-up conversation" are content the ontology does
  not model.
