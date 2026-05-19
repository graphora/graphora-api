# news_attribution

A short news-byline summary. Tests descriptive (non-personal)
source names as canonical identities.

## What this exercises

- **Descriptive Source identity**: ``Emerald spokesperson``
  and ``industry analyst Yuri Lev`` are not personal names
  but role/role+name composites. The canonical_key uses the
  full descriptive string lowercased. Pin against a
  normalizer that strips role qualifiers ("analyst") to
  derive a personal-name identity.
- **Two CITES_SOURCE edges sharing a source**: Both
  citations originate at the Article. Pin against an
  extractor that conflates "both sources contacted" into
  one composite Source.

## Failure signals

- An extractor that elevates ``Yuri Lev`` out of "industry
  analyst Yuri Lev" creates a different canonical_key
  (``Source:name=yuri lev``) than the helper recomputes
  from the full string.
- An extractor that misses the second source (Yuri Lev,
  appearing only in the second paragraph) drops
  CITES_SOURCE recall.

## Intentional non-extractions

- "strategic rationale", "integration risk", "went to
  press", "additional comment" are content the ontology
  does not model.
