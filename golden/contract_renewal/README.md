# contract_renewal

A short contract-renewal description. Sibling to
legal_simple_agreement but with a renewal-date property and
deeper multi-paragraph dedup.

## What this exercises

- **Same pattern, different stress**: legal_simple_agreement
  pins Party + Agreement with two PARTY_TO edges. This entry
  uses the same edge structure but with a non-trivial
  contract reference (``MSA-2023-NORTHWIND``) that's named
  four times in the document. Pin that the dedup pressure
  scales with mention count.
- **Renewal-date as non-identity property**: ``renewal_date``
  rides along on the Contract. Pin that the ontology's
  ``unique: true`` flag on ``reference`` selects identity
  independently of the date field.

## Failure signals

- An extractor that creates separate Contract nodes for
  "Master Services Agreement" vs "MSA-2023-NORTHWIND" (one
  capturing the long form, the other the reference) drops
  Contract precision.
- An extractor that elevates ``renewal_date`` to identity
  (so two contracts renewing on the same date collapse)
  would fail the helper-recomputation invariant.

## Intentional non-extractions

- "three-year term", "written confirmation", "sole service
  provider", "initial term", "since 2023" are content the
  ontology does not model.
