# compliance_audit

A short compliance-audit summary. Tests fan-out from an
audit firm to two clients plus one standard.

## What this exercises

- **Two AUDITED edges from one source**: Both AUDITED edges
  originate at Quintar Compliance and fan out to Northwind
  Logistics and Southshore Holdings. Pin against an
  extractor that conflates "two clients" into one composite
  Company node.
- **Standard reference with mixed punctuation**: ``ISO
  27001:2022`` carries a colon and a hyphen pattern. Pin
  that the canonical_key preserves the structure (lowercased
  but not stripped).
- **Modeling choice: firm-to-standard, not company-to-
  standard**: The ontology models AUDIT_AGAINST between the
  AuditFirm and the Standard — a firm specializes in a
  standard. Pin against an extractor that emits four
  ternary edges (firm→company→standard).

## Failure signals

- An extractor that emits AUDIT_AGAINST from each Company
  (rather than from the AuditFirm) doubles edge count and
  misaligns with the ontology vocabulary.
- An extractor that normalizes ``ISO 27001:2022`` to
  ``iso-27001-2022`` (stripping the colon) produces a
  canonical_key the helper recomputation can't match.

## Intentional non-extractions

- "three observations", "access management", "remediation
  plans", "Q3", "re-audit" are content the ontology does
  not model.
