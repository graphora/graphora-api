# clinical_trial

A short clinical-trial summary. Tests a four-entity
star-pattern with one Trial node at the center.

## What this exercises

- **Star pattern around a transaction-like entity**: Trial
  is the center; Sponsor, Drug, and Condition are the
  spokes. Three edges, all rooted in Trial. Pin against an
  extractor that drops the Trial node and tries to wire
  Sponsor → Drug → Condition directly.
- **Non-name canonical identity (variant)**: Trial uses
  ``nct_id``. The phase rides along as a non-identity
  property.
- **Acronym preservation in identity**: The NCT ID
  ``NCT04501239`` mixes letters and digits. canonical_key
  lower-cases (``trial:nct_id=nct04501239``) but preserves
  the alphanumeric structure.

## Failure signals

- An extractor that combines Drug and Condition into a
  single "indication" node loses the modeled separation
  between intervention and target.
- An extractor that misses TARGETS_CONDITION while keeping
  SPONSORED_BY and EVALUATES drops relationship recall in a
  way that's hard to see from precision alone.

## Intentional non-extractions

- "240 patients", "fifteen sites", "intravenously every
  three weeks", "primary endpoint", "late 2027" are content
  the ontology does not model.
