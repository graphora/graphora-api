# trademark_registration

A short trademark-registration summary. Tests numeric-code
canonical identity on the Class entity.

## What this exercises

- **Numeric-code identity on Class**: Class uses ``code``
  (``25``) as the unique property. canonical_key is
  ``Class:code=25`` — a numeric string. Pin that the helper
  handles numeric-string identities without coercing to
  int.
- **Two-property Trademark node**: registration_number is
  identity; ``text`` (the mark itself, ``STARFOLD``) rides
  along. Pin that the canonical_key uses the registration
  number, not the mark text.
- **Four-mention dedup on the Trademark**: USPTO-6543210
  appears four times.

## Failure signals

- An extractor that elevates the mark text (``STARFOLD``)
  into the canonical_key — common foot-gun since the mark
  text is more memorable than the registration number —
  produces a key that doesn't match the helper recomputation.
- An extractor that omits the Class node (treating "Class
  25" as a property string on the Trademark) drops Class
  recall.

## Intentional non-extractions

- "clothing and apparel", "Section 8 declaration", "2022",
  "2024", "first use dated 2021", "five years" are content
  the ontology does not model.
