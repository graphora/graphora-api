# software_dependency

A short ecosystem-overview paragraph describing react, its
dependencies, and the organization that maintains them.
Tests same-type relationships and a shared maintainer node.

## What this exercises

- **Same-type edges (Package → Package)**: `DEPENDS_ON` has
  identical source and target types. The expected graph has
  two such edges (`react → react-dom`, `react → scheduler`).
  An extractor that requires distinct source/target types
  in the relationship spec would fail to emit either.
- **Shared-target convergence**: all three Package nodes
  point at the same Organization (Meta) via MAINTAINED_BY.
  The expected graph has one Organization node referenced
  by three edges — pin against an extractor that creates a
  separate Organization per package mention.
- **Hyphenated names through canonical_key**: `react-dom`
  carries a hyphen in the source name. The helper lower-cases
  but preserves the hyphen (`Package:name=react-dom`), so a
  refactor that strips non-alphanumerics from the key would
  fail the recompute invariant.

## Failure signals

- An extractor that creates a separate `Meta` Organization for
  each package mention drops Organization precision: one
  expected vs. three actuals sharing the canonical_key would
  record `TP=1, FP=2, FN=0` → `Organization.precision ≈ 0.33`.
- An extractor that mis-types `DEPENDS_ON` as e.g.
  `REQUIRES` or `USES` drops relationship precision against
  the ontology vocabulary.
- An extractor that emits a phantom `scheduler → react` edge
  (reading the dependency direction backwards) drops
  `DEPENDS_ON` precision while leaving recall intact.

## Intentional non-extractions

- "JavaScript library", "user interfaces", "DOM-rendering
  primitives", "cooperative task scheduling", and
  "unstable_scheduleCallback" are descriptive content the
  ontology does not model. A faithful extractor should not
  surface them as unmodeled entity types.
