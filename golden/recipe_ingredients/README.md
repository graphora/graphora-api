# recipe_ingredients

A short recipe description. Tests three USES_INGREDIENT edges
sharing a source + one IN_CUISINE edge.

## What this exercises

- **Three same-type edges from one source**: All three
  USES_INGREDIENT edges originate at the Recipe. Pin
  against an extractor that emits one composite ingredient
  node for "chicken thighs, lemon, and garlic" (parsing the
  comma-and-conjunction list as a compound noun).
- **High-density mention pattern**: Each of the three
  ingredients is mentioned three times across the document
  in different paragraphs. Pin that the dedup logic handles
  the high recurrence without spurious entity splits.

## Failure signals

- An extractor that creates a single composite Ingredient
  ("chicken thighs, lemon, and garlic") drops Ingredient
  recall by 2/3.
- An extractor that promotes "Mediterranean tradition" and
  "Mediterranean classic" to separate Cuisine nodes
  inflates Cuisine FP.

## Intentional non-extractions

- "at least two hours", "before cooking", "simple
  combinations", "roasted until golden" are content the
  ontology does not model.
