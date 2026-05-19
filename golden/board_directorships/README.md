# board_directorships

A short governance profile. Tests one-to-many edges from a
single Person to multiple Companies. Mirror of
employment_history but with a different relationship type.

## What this exercises

- **Repeat coverage on the one-to-many pattern**: Pairs with
  employment_history to verify the pattern holds for
  multiple relationship types. A regression that breaks
  many-edge fan-out on one type would surface in both
  entries.
- **Multi-mention dedup on three Company nodes**: Each of
  the three companies is referenced twice. All three must
  collapse.

## Failure signals

- An extractor that misses one of the three board seats
  (e.g., fails to extract Crestview Bank from the second
  paragraph) drops DIRECTOR_OF recall.
- An extractor that emits a fourth DIRECTOR_OF edge
  fabricated from the "three public companies" preamble
  (without naming the fourth) inflates relationship FP.

## Intentional non-extractions

- "audit committee", "governance", "financial oversight",
  "staggered", "renewal in 2027" are descriptive content the
  ontology does not model.
