# government_regulation

A short regulatory press release. Tests cross-paragraph
deduplication of an agency, a specific CFR citation, and an
industry segment — each referenced multiple times across the
document.

## What this exercises

- **Multi-mention dedup on three entity types**: "EPA" appears
  four times, "40 CFR Part 60" three times (twice as the full
  string, once with the "Rule" prefix), and "coal-fired power
  generation" twice. Each must collapse to a single canonical
  node.
- **Non-name canonical identity**: the Regulation's identity
  comes from `citation`, not `name`. Verifies that the
  helper's `unique: true` flag drives identity off whatever
  property the ontology says — `Regulation:citation=40 cfr part 60`
  is the canonical_key, not anything derived from a "name"
  field.
- **Citation normalization**: the source document writes both
  "Rule 40 CFR Part 60" and "40 CFR Part 60". The pre-dedup
  pass lower-cases for the canonical_key but keeps the source
  string intact in `properties.citation`; the ontology test
  pins the citation string as it should land.

## Failure signals

- An extractor that creates two separate Regulation nodes for
  "Rule 40 CFR Part 60" vs "40 CFR Part 60" (i.e., treats the
  "Rule" prefix as part of the citation) drops Regulation
  precision. The two actuals would carry different
  canonical_keys and not collapse.
- An extractor that misses the explicit "ISSUED_BY" reading
  and emits e.g. "AUTHORED_BY" or "CREATED_BY" drops relation
  precision against the ontology's published vocabulary —
  ISSUED_BY is the only valid type for Regulation → Agency in
  this ontology.

## Intentional non-extractions

- "Federal Register", "Q3 2026", "three years", "90 percent",
  and the implementation-extension request from trade groups
  are content the document carries but the ontology does not
  model. A faithful extractor should not surface them as
  unmodeled entity types or speculative edges.
