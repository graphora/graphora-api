# commit_authorship

A short commit-history snapshot. Tests email-as-identity and
high-cardinality fan-in (six edges sharing two targets).

## What this exercises

- **Email-as-canonical-identity**: Author uses ``email``
  not ``name``. Common convention in source control; pin
  it as the corpus example for that pattern.
- **Six edges from three commits**: Each Commit has two
  outgoing edges (AUTHORED_BY + IN_REPOSITORY). All six
  share two targets (the Author and the Repository
  respectively). Pin that the edge-matching layer doesn't
  collapse them based on shared targets.
- **Commit message as non-identity property**: The
  ``message`` field carries human-readable text but doesn't
  affect canonical_key.

## Failure signals

- An extractor that emits one composite Commit node for
  "three commits this week" (parsing the count phrase as a
  single entity) collapses three to one.
- An extractor that uses the Author's name (which doesn't
  appear in the document) instead of email fails
  helper-recomputation against the email-based identity.

## Intentional non-extractions

- "Monday", "Wednesday", "Friday", "pushed" are content the
  ontology does not model — temporal sequencing is
  intentionally left to a future slice.
