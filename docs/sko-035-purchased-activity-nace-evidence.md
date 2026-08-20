# SKO-035 — Governed purchased-activity / NACE evidence

**Status:** Implemented and tested for owner review; not accepted
**Date:** 20 August 2026
**Baseline:** `7caf6da6a5ab97923a6e8f36ad467502e6bdd261`
**Target classification:** NACE Rev. 2 for the SKO-035 v1 pilot

## Scope

SKO-035 materialises governed human/research decisions at supplier-spend-observation × economic-activity allocation grain. It preserves purchased-activity evidence separately, validates confidence and NACE coherence, supports evidence-backed splits and explicit unresolved decisions, escalates material weak/conflicting cases, and produces deterministic records and QA.

NACE Rev. 2 is the versioned v1 pilot target. It is not asserted as permanently authoritative for future skopia work. Rev. 2.1 may be added later through an explicitly versioned migration or extension.

## Files changed

- `.gitignore` — narrow addition allowing only explicitly synthetic SKO-035 CSV fixtures.
- `purchased_activity_nace.py` — additive local validation, governance, materialisation, QA and CLI.
- `config/purchased_activity_nace.yaml` — production Rev. 2 target/catalogue, evidence/confidence governance, and no national-code crosswalks.
- `schemas/purchased_activity_evidence.schema.json` — inspectable evidence contract.
- `schemas/purchased_activity_assertion.schema.json` — inspectable assertion contract.
- `tests/test_purchased_activity_nace.py` — T01–T35 behavioural contract.
- `tests/fixtures/purchased_activity_nace/` — synthetic observations, evidence, decisions and test-only configuration.

## Crosswalk governance

Production configuration contains `crosswalks: {}`. No CNAE, CCAE, UK SIC or other production correspondence is asserted.

The synthetic test configuration contains one deliberately test-only correspondence:

- ID: `synthetic-cnae-8121-to-nace-8121-test-only`
- source: CNAE / `synthetic-cnae-2009` / `8121`
- target: NACE / `Rev. 2` / `81.21`
- provenance: `synthetic-test-only-correspondence-v1`
- authority: synthetic behavioural fixture only; not an official correspondence

T11 proves the national scheme remains CNAE evidence. T12 proves the explicit test correspondence enables the governed mapping and that production configuration, with the correspondence absent, rejects it.

## Behavioural evidence

The 35 focused behaviours cover selected-only processing; entity pass-through and observation-level validity; absence of canonical mutation; evidence hierarchy; confidence constraints; coherent/versioned NACE; preserved national schemes and explicit-only crosswalking; conflicts and escalation; single, split, broader and unresolved treatments; allocation and spend reconciliation; provenance; deterministic reruns/order/CLI; duplicate governance; schema contracts; synthetic-only fixtures; and prohibited coefficient, impact, network, policy and publication paths.

## Validation results

```text
python -m py_compile purchased_activity_nace.py
PASS

python -m unittest -v tests.test_purchased_activity_nace
Ran 35 tests — OK

python -m unittest -v tests.test_activity_evidence
Ran 28 tests — OK

python -m unittest discover -s tests -v
Ran 327 tests — OK

JSON Schema validation
7 evidence rows and 6 assertion rows validated
```

## Synthetic QA

| Measure | Count |
|---|---:|
| selected observations | 5 |
| observations with assignment | 5 |
| high / medium / low confidence rows | 2 / 3 / 1 |
| unresolved | 1 |
| broader-level assignments | 1 |
| contract-specific observations | 1 |
| multi-activity observations | 2 |
| spend-split observations | 1 |
| materiality-escalated | 1 |
| conflicting-evidence observations | 1 |
| procurement-category-only evidence | 0 |
| authoritative/strong evidence observations | 4 |
| allocation reconciliation failures | 0 |
| invalid NACE/version failures | 0 |

Counts are synthetic behavioural evidence, not a live pilot result.

## Explicit exclusions

No coefficients, GVA, labour income, employment, production tax, GHG, FIGARO, automatic free-text classification, canonical creation/mutation, social-economy reclassification, directory/publication mutation, Airtable/network integration, or live client data are introduced.

## Unresolved limitations

- Production national-code correspondences remain empty pending separately evidenced authoritative sources and versions.
- The production NACE catalogue is intentionally minimal and must be governed before a real pilot expands it.
- Materiality labels are supplied by the governed input context; SKO-035 does not impose one universal monetary threshold.
- The real approximately 35-observation pilot remains local and uncommitted.
- SKO-035 remains unaccepted until owner review of this evidence.
