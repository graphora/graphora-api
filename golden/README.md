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
                      # minus pagination metadata.
                      #
                      # Identity matching uses `canonical_id` /
                      # `canonical_key` on each node — read by the
                      # diff service via the property bag. The Node
                      # schema (graphora_server/schemas/graph.py)
                      # only declares id/label/type/properties; any
                      # top-level canonical_* would be silently
                      # dropped by Pydantic. Put canonical_id and
                      # canonical_key INSIDE properties:
                      #
                      #   {"id":"alice", "type":"Person",
                      #    "properties":{"canonical_id":"alice",
                      #                  "canonical_key":"alice",
                      #                  "name":"Alice"}}
                      #
                      # Failing to do so makes the scorer fall back
                      # to per-side local IDs, which never match
                      # across an expected/actual pair — every fact
                      # ends up as added+removed.
                      #
                      # The VALUES must match what the extraction
                      # helpers actually compute. Hand-writing
                      # ``"canonical_id": "alice-martinez"`` is a
                      # foot-gun: the live extractor calls
                      # ``_generate_node_key`` +
                      # ``_make_canonical_node_id`` (in
                      # graphora_server/services/transform/helpers.py)
                      # which produce a UUID-shaped canonical_id
                      # derived from the ontology's ``unique: true``
                      # properties. If your expected canonical_id
                      # differs from the helper output, the
                      # DiffService's "conflicting canonical IDs
                      # stay unmatched" rule (asymmetric ER
                      # constraint, commit a261321) refuses the
                      # canonical_key fallback and every node
                      # surfaces as FP+FN. The corpus contract test
                      # verifies the values match the helpers.
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

| Slug | Domain | Pattern | Entity types | Edge types |
|---|---|---|---|---|
| `single_person_works_at_org` | Business | One Person, one Organization, one WORKS_AT edge | Person, Organization | WORKS_AT |
| `two_people_same_org` | Business | Two people at one shared Organization — dedup pin on the shared node | Person, Organization | WORKS_AT |
| `healthcare_clinical_note` | Healthcare | Patient + Doctor + Diagnosis; honorific normalization + cross-type edges | Patient, Doctor, Diagnosis | SEEN_BY, DIAGNOSED_WITH |
| `legal_simple_agreement` | Legal | Two Parties + one Agreement; multi-reference dedup across three nodes | Party, Agreement | PARTY_TO |
| `financial_transaction` | Finance | Two Accounts + one Transaction; direction-sensitive debit/credit edges | Account, Transaction | DEBITED_FROM, CREDITED_TO |
| `academic_paper_citation` | Academic | Multi-author paper citing another paper; same-type self-referential CITES edge | Paper, Author | AUTHORED_BY, CITES |

### Growth target

Plan calls for 50+ documents at Gate-4 exit. As of 2026-05-19
we're at 6. Each new entry should add either a new domain or
a new pattern not yet covered above — duplicating an existing
pattern hurts coverage diversity more than it helps test
volume.
