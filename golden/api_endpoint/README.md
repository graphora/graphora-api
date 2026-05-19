# api_endpoint

A short API-documentation summary. Tests path-with-
placeholder identity and HTTP-method enumerations.

## What this exercises

- **Path-with-placeholder canonical_key**: ``{id}`` stays
  literal in the canonical_key — pin against a normalizer
  that resolves placeholders.
- **Short-string canonical identity**: ``GET`` and ``PATCH``
  are three- and five-character canonical_keys. Pin that
  the helper handles very short identities without
  collision (e.g., not stripping common HTTP-method-shaped
  words from text).
- **Two ACCEPTS edges sharing a source**: Both Method
  edges originate at the Endpoint. Pin against an extractor
  that conflates the two methods into one composite Method
  node.

## Failure signals

- An extractor that normalizes the path to ``/api/v1/users``
  (stripping the placeholder) produces a canonical_key that
  doesn't match the helper output.
- An extractor that omits one of the two Methods inflates
  the asymmetry between GET and PATCH coverage.

## Intentional non-extractions

- "partial-update body", "field-level diffs", "authenticated
  callers" are content the ontology does not model.
