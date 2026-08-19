# SKO-034 Germany-first source laboratory - implementation evidence

**Task:** SKO-034
**Status:** Implemented and tested; operationally unvalidated, not promoted, not accepted and not complete
**Branch:** `codex/sko-034-german-source-lab`
**Reconciled base:** 2334e88f794c27d1e5921c51b464372e32183644 (accepted SKO-031)

## Boundary compliance

- Production `sources.yaml` unchanged.
- `scrape_ei_registers.py` unchanged.
- Canonical stores/materialisation unchanged.
- Verified overlays and client matching unchanged.
- No client data used.
- No live scrape or Registerportal crawler implemented.
- Every candidate source is disabled.
- Parsing source records does not create canonical entities.

## Implemented decisions

- Candidate semantic layers: Core, Associated, Review-only, Reject.
- Explicit source identity, jurisdiction, access, format, native key, provenance, terms, rejection and acceptance-state contract.
- German typed identifier vocabulary: `DE_HRB`, `DE_HRA`, `DE_GNR`, `DE_VR`, `DE_PR`, `DE_GSR`, `DE_EUID`, `DE_VAT`, `DE_ZER_INTERNAL`.
- Court-scoped register keys require a verified court.
- ZER internal identifiers have `internal_source` scope and are never reusable tax identifiers.
- German legal-form normalization produces evidence only and never policy classification.
- BAG IF and ZER offline parsers preserve accepted and rejected records plus metrics and provenance.

## Inspectable behavioural evidence

The focused suite verifies:

- config validity and disabled candidates;
- rejection of an enabled candidate;
- every approved German identifier type;
- cross-court collision avoidance;
- court-required rejection;
- ZER non-reusability;
- `e.V.`, `eG`, `gGmbH`, `GmbH`, `Stiftung` variants;
- `H.E.G.` and embedded-letter false-positive protection;
- BAG IF multiline names, missing websites, incomplete addresses, invalid names, provenance, rejection and duplicate metrics;
- ZER provenance, source-local keys, empty tax IDs, duplicate/missing-ID/null-name metrics and tax-designation semantics;
- deterministic BAG IF reruns;
- Switzerland UID formatting.

## Research evidence

- 16-Länder authority/source matrix.
- Welfare and specialist-source catalogue.
- National/regional combination assessment.
- Manual individual-query Registerportal protocol.
- Brief Switzerland comparator.
- Global catalogue only.

## Deferred shared-core reconciliation

The production BAG IF parser does not expose rejected rows or full provenance and was not changed. The ZER production builders currently place source-local values in the normalized `tax_id` column and were not changed. SKO-031 is accepted. The pre-reconciliation audit found no overlapping paths or shared-core conflicts. A future integration task must compare the accepted laboratory contract with shared-core source/identifier work, resolve field vocabulary conflicts, and propose a separately reviewed production migration. No historical data migration belongs to SKO-034.

## Acceptance gate

Implementation evidence is not acceptance. Owner review is required before integration, production config changes, source enabling, historical migration or status completion.
