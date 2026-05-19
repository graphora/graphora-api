# flight_itinerary

A short two-leg flight itinerary. Tests sequential edges
through shared intermediate airports.

## What this exercises

- **ORD as shared intermediate**: ORD is both the
  arrival airport of UA-2841 AND the departure airport of
  UA-417. The expected graph has one ORD node with two
  edges pointing in (DEPARTS_FROM from UA-417, ARRIVES_AT
  from UA-2841). Pin that the dedup logic doesn't
  duplicate ORD per role.
- **Six edges across three entity types**: Two BOOKED_ON,
  two DEPARTS_FROM, two ARRIVES_AT. Pin that the
  edge-matching layer handles this density.
- **Short-code Airport identity**: Three-letter IATA codes
  (SFO, ORD, FRA) are short canonical_keys. Pin against
  collision with similar three-letter substrings in other
  contexts.

## Failure signals

- An extractor that creates ORD-departure and ORD-arrival
  as separate Airport nodes (specializing on role)
  inflates Airport count.
- An extractor that conflates UA-2841 and UA-417 into a
  single "itinerary" entity loses the modeled per-flight
  structure.

## Intentional non-extractions

- "8:15 AM", "two-hour layover", "transatlantic", "the
  following morning local time" are content the ontology
  does not model.
