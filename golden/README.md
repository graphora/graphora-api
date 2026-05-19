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
| `government_regulation` | Government | Agency + CFR citation + Industry; non-name canonical identity (citation, not name) | Agency, Regulation, Industry | ISSUED_BY, TARGETS |
| `software_dependency` | Software | Same-type DEPENDS_ON between packages + shared maintainer convergence | Package, Organization | DEPENDS_ON, MAINTAINED_BY |
| `manufacturing_supply` | Manufacturing | Product → Component → Supplier two-step chain; multiple components per product | Product, Component, Supplier | USES, SUPPLIED_BY |
| `family_relationships` | Family / personal | Single-entity-type ontology; same-source-and-target Person-Person edges (MARRIED_TO + PARENT_OF) | Person | MARRIED_TO, PARENT_OF |
| `mergers_acquisitions` | Business | Acquisition record with date-as-property; same-type Company → Company ACQUIRED edge | Company, Acquisition | ACQUIRED, ACQUISITION_OF, ACQUISITION_BY |
| `org_subsidiaries` | Business | Parent + three subsidiaries; three same-type edges converging on a shared parent | Company | SUBSIDIARY_OF |
| `vendor_invoice` | Business | Currency-amount as non-identity property; five-mention dedup on the Invoice | Vendor, Customer, Invoice | ISSUED_BY, BILLED_TO |
| `employment_history` | Business | One Person, three sequential Organizations; sequence-in-narrative parallel-in-graph | Person, Organization | WORKED_AT |
| `board_directorships` | Business | One Director on three Boards; fan-out from a shared source | Person, Company | DIRECTOR_OF |
| `drug_interactions` | Healthcare | Hub-and-spoke same-type INTERACTS_WITH; explicit non-extraction of negated triangle | Drug | INTERACTS_WITH |
| `gene_protein` | Biology | Gene → Protein → Function chain; symbol-as-identity on Gene | Gene, Protein, Function | ENCODES, PARTICIPATES_IN |
| `clinical_trial` | Healthcare | Trial-centered star pattern; NCT identifier with letter/digit mix | Trial, Sponsor, Drug, Condition | SPONSORED_BY, EVALUATES, TARGETS_CONDITION |
| `patient_admission` | Healthcare | MRN-as-PHI-safe identity; date-as-non-identity property | Patient, Hospital, Procedure | ADMITTED_TO, UNDERWENT |
| `medical_specialty` | Healthcare | One Doctor, two Specialties (same-source same-type fan-out); honorific in canonical_key | Doctor, Specialty, Hospital | BOARD_CERTIFIED_IN, PRACTICES_AT |
| `case_law_citation` | Legal | Bluebook-citation canonical identity; long-string canonical_key | Case, Court, Judge | DECIDED_BY, AUTHORED_BY |
| `patent_inventors` | Legal | Three same-type INVENTED_BY edges from one Patent; star + assignee | Patent, Inventor, Company | INVENTED_BY, ASSIGNED_TO |
| `compliance_audit` | Legal | Firm-to-standard relationship modeling choice; punctuated standard reference | AuditFirm, Company, Standard | AUDITED, AUDIT_AGAINST |
| `contract_renewal` | Legal | Pattern repeat of legal_simple_agreement under deeper dedup pressure | Contract, Party | PARTY_TO |
| `trademark_registration` | Legal | Numeric-string Class identity; mark text as non-identity | Trademark, Company, Class | OWNED_BY, REGISTERED_IN |
| `stock_purchase` | Finance | Direction-sensitive PURCHASED vs SOLD; uppercase ticker source → lowercase canonical | Investor, Stock | PURCHASED, SOLD |
| `investment_round` | Finance | Two Investors converging on one Round; amount + round_type as non-identity | Startup, Investor, Round | RAISED, PARTICIPATED_IN |
| `cryptocurrency_transfer` | Finance | Hex-address identity with mixed-case source; same-type SENT_TO | Wallet, Token | SENT_TO, TRANSFERRED |
| `loan_collateral` | Finance | Two Asset → Borrower edges; Asset-anchored collateral modeling | Lender, Borrower, Asset | LOANED_TO, COLLATERAL_FOR |
| `mortgage_origination` | Finance | Full-address canonical_key with commas and ZIP | Bank, Property, Person | ORIGINATED, BORROWER_OF |
| `cve_vulnerability` | Software | CVE-as-identifier; one CVE affecting multiple Software products | CVE, Software | AFFECTS |
| `api_endpoint` | Software | Path-with-placeholder identity; short HTTP-method canonical_keys | Endpoint, Service, Method | EXPOSED_BY, ACCEPTS |
| `commit_authorship` | Software | Email-as-author-identity; six edges, two shared targets | Author, Repository, Commit | AUTHORED_BY, IN_REPOSITORY |
| `incident_response` | Software | Two parallel fan-outs (Systems + Engineers) from one Incident | Incident, System, Engineer | IMPACTS, RESPONDED_BY |
| `cloud_deployment` | Software | Two parallel three-way fan-outs sharing three Region targets | Service, Region, Cluster | DEPLOYED_TO, SPANS |
| `conference_proceedings` | Academic | One Author with two Paper edges; long-title canonical_keys | Conference, Paper, Author | ACCEPTED_AT, AUTHORED_BY |
| `research_grant` | Academic | Grant-centered star with three different-typed targets | Grant, Funder, Researcher, Project | AWARDED_BY, PI_OF, FUNDS |
| `dissertation_committee` | Academic | Non-ASCII name in canonical_key; overlapping ADVISES + COMMITTEE_FOR | Student, Faculty, Department | ADVISES, COMMITTEE_FOR, ENROLLED_IN |
| `experimental_dataset` | Academic | Two COLLECTED_BY edges sharing a source; method-as-shared-vocabulary | Dataset, Researcher, Method | COLLECTED_BY, USES_METHOD |
| `peer_review_workflow` | Academic | Pseudonymous Reviewer canonical_keys (Reviewer 1 / Reviewer 2) | Paper, Reviewer, Editor | REVIEWED_BY, HANDLED_BY |
| `news_attribution` | Media | Descriptive role-based Source identities; two CITES_SOURCE edges | Article, Reporter, Source | REPORTED_BY, CITES_SOURCE |
| `political_campaign` | Politics | Title-prefix in canonical_key; four-entity star pattern | Candidate, Office, District, Endorser | RUNNING_FOR, IN_DISTRICT, ENDORSED_BY |
| `legislation_sponsorship` | Politics | Bill number with periods; two REFERRED_TO edges sharing source | Bill, Legislator, Committee | SPONSORED_BY, REFERRED_TO |
| `interview_quotes` | Media | Two Topics from one Speaker; "The" article preserved in Publication name | Speaker, Topic, Publication | DISCUSSED, INTERVIEWED_BY |
| `book_publication` | Media | ISBN-hyphenated canonical_key; title as non-identity property | Book, Author, Publisher | AUTHORED_BY, PUBLISHED_BY |
| `flight_itinerary` | Travel | Shared intermediate airport (ORD = arrival AND departure for different flights) | Passenger, Flight, Airport | BOOKED_ON, DEPARTS_FROM, ARRIVES_AT |
| `course_enrollment` | Education | 2×2 bipartite (one student in two courses + one instructor of both) | Student, Course, Instructor | ENROLLED_IN, TAUGHT_BY |
| `shipping_route` | Logistics | Two carriers sharing identical origin + destination Ports | Port, Carrier | ORIGIN_OF, DESTINATION_OF |
| `degree_conferred` | Education | Long-form Institution name in canonical_key; degree-as-record entity | Person, Degree, Institution | EARNED, CONFERRED_BY |
| `recipe_ingredients` | Food | Three USES_INGREDIENT edges from one Recipe; high-density ingredient mentions | Recipe, Ingredient, Cuisine | USES_INGREDIENT, IN_CUISINE |

### Growth target

Plan calls for 50+ documents at Gate-4 exit. As of 2026-05-19
we're at **50** — target met. Further growth is welcome but
not gating on B4-bench; new entries should still add a new
domain or pattern not yet covered above rather than padding
the count.
