# Golden Corpus

B4-corpus (Gate 4, Track B). A small set of curated documents with
ground-truth extractions for benchmarking and regression testing.

## Contract

Each document lives in its own subdirectory under `golden/`. The
contract per directory:

```
golden/<slug>/
  document.txt        # The source text the extractor reads.
                      # Plain-text only at this stage — PDF / DOCX
                      # support will land alongside B4-test's runner
                      # once the text-only path is locked in.
  ontology.yaml       # The ontology the extraction runs against.
                      # Matches the format /api/v1/ontology accepts.
  expected.json       # Ground-truth nodes + edges. Same shape as
                      # the /api/v1/graph/{transform_id} response,
                      # minus pagination metadata. Identity matching
                      # happens by canonical_id (preferred) or
                      # `<type>:<canonical_key>` fallback — see the
                      # graphora_server/services/diff_service.py
                      # contract.
  README.md           # Brief description of what this doc tests:
                      # which entity types, what edge patterns, any
                      # known edge cases.
```

## What "ground truth" means

Ground truth is the EXTRACTION we'd accept as correct against the
given ontology. It's not "every fact in the source" — it's "every
fact the ontology asks the extractor to surface, with the
canonical_key derivation rules applied." A doc that mentions
"Alice" five times has ONE `Person:alice` node in expected.json.

Property differences on matched nodes are scored as "changed" by
the scorer (see `graphora_server/services/golden_corpus/scorer.py`)
— TP for identity, partial-credit for properties. The exact
weighting is a tunable on the scorer, not the corpus.

## Licensing

Seed docs are **synthetic** — written for this corpus, no
third-party content. That keeps the corpus distributable without
attribution complications. Real-world documents (with explicit
license tags) come in subsequent corpus additions; the per-doc
README's "source" field is where attribution lands when applicable.

## Adding a new doc

1. Pick a slug that describes the pattern under test
   (`single_person_works_at_org`, `multi_org_acquisition`, etc.).
2. Drop the four files above. Keep document.txt under 2KB for
   the seed tier — the runner reads them all into memory.
3. Verify locally: `pytest tests/unit/services/test_golden_corpus_scorer.py`
   exercises the scorer; once B4-test's runner ships, it'll do
   the full extract → score pass.
4. Update this README's roster table below.

## Current roster

| Slug | Pattern | Entity types | Edge types |
|---|---|---|---|
| `single_person_works_at_org` | One Person, one Organization, one WORKS_AT edge | Person, Organization | WORKS_AT |
