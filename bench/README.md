# Bench

B4-bench (Gate 4, Track B). Public benchmark comparing Graphora's
extraction quality against competing knowledge-graph extractors on
the [golden corpus](../golden/).

## Reproducibility contract

The benchmark numbers are derived from files committed to this
repository. Anyone with a checkout can re-run the calculation
locally:

```
curl http://localhost:8000/api/v1/bench/run
```

…or via the `BenchRunner` directly. The endpoint reads
`bench/results/` and `golden/`, scores each pair via
`CorpusScorer`, and returns the aggregated report. There is no
hidden state; the score on the public page == the score you get
from a fresh checkout.

## Directory layout

```
bench/
  README.md                              # this file
  results/                               # populated by extractor runs
    <extractor>/                         # e.g., "graphora", "langextract"
      <slug>.json                        # extractor's output for golden/<slug>/
      <slug>.json
      ...
```

Each `<slug>.json` matches the wire shape of
`GET /api/v1/graph/{transform_id}` — same `GraphResponse` schema
(`nodes` + `edges`). The bench scorer uses Pydantic
`model_validate`, so a missing field surfaces as an errored entry
rather than crashing the full run.

## Populating an extractor

Pick a stable name for the extractor (e.g., `graphora`,
`langextract`, `graphrag`, `graphiti`, `langchain_gt`). Create the
subdirectory:

```
mkdir -p bench/results/<extractor>
```

For each corpus entry under `golden/<slug>/`, run the extractor
on `golden/<slug>/document.txt` with `golden/<slug>/ontology.yaml`
as the schema input. Write the resulting graph to
`bench/results/<extractor>/<slug>.json` in the `GraphResponse`
shape.

Slugs without an output file surface as **errored entries** in
the bench report — visible coverage gaps rather than silent
zero-deflation. Empty extractor directories are valid (slot
reserved with no scores).

## Running the bench

Once `bench/results/` is populated:

```
# Via the API
curl http://localhost:8000/api/v1/bench/run | jq .

# Via the Python service directly
from pathlib import Path
from graphora_server.services.bench import BenchRunner
report = BenchRunner(repo_root=Path(".")).run()
print(report.to_dict())
```

The report carries:

* `corpus_size` — how many golden entries were discovered.
* `extractors[]` — one per `bench/results/<extractor>/` directory.
  Each carries:
  * `extractor_name`
  * `total_entries` / `scored_count` / `errored_count`
  * Micro-averaged P/R/F1 for nodes and edges (weighted by entity
    count — favors high-density docs).
  * Macro-averaged F1 for nodes and edges (unweighted mean across
    entries — treats each doc equally).
  * `entries[]` — per-entry detail with raw TP/FP/FN counts.

Both micro and macro are reported because they answer different
questions:

* **Micro** asks: "across all the entities in the corpus, what
  fraction did the extractor get right?" A 50-node document
  dominates a 3-node document.
* **Macro** asks: "across all the documents in the corpus, what's
  the average per-document F1?" Every document gets equal weight.

If an extractor wins on micro but loses on macro, it's strong on
information-dense documents and weak on sparse ones — a
diagnostic the headline F1 alone would hide.

## What this slice does NOT include

* **Cost / latency / token-budget columns**: the score endpoint
  measures only graph-shape correctness. Cost + latency live on
  the B5-obs surface (`/api/v1/transforms/<id>/cost`) and can be
  joined in a future slice; for now the bench dashboard shows
  quality only.
* **Property-level scoring**: matched-identity nodes with
  different property values count as TP for identity, with the
  property delta surfaced through `/diff` but not weighted into
  F1 yet. See `services/golden_corpus/scorer.py` for the
  "future slice" note.
* **Per-domain breakdowns**: aggregate scores treat all 50
  entries as one corpus. A "score per domain" pivot (Business /
  Healthcare / Legal / etc.) follows from the roster table in
  `golden/README.md` but isn't part of this surface yet.

## Auth

The `/api/v1/bench/run` endpoint is **unauthenticated**.
Benchmark numbers are public by design — the reproducibility
claim depends on anyone being able to fetch them. No tenant
scoping; the data is repo artifacts, not user content.
