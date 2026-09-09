# SKO-036A — Live-source contract refinement evidence

**Parent task:** SKO-036  
**Refinement scope:** live-source semantic contract only  
**Decision date:** 9 September 2026  
**Owner decision:** Approved by Charlie  
**Implementation status:** Implemented and focused-tested; SKO-036 overall remains in progress pending the real coverage build

## 1. Purpose

Apply the owner-approved refinements arising from the SKO-036 read-only live-source validation before building the full country × sector × year × outcome coefficient coverage layer.

This refinement does not alter the supplier-agnostic architecture or introduce client, canonical-entity, social-economy, indirect/Leontief or environmental logic.

## 2. Owner-approved decisions implemented

1. Keep the hybrid denominator architecture unchanged: national-accounts `P1` output for ordinary sectors and turnover for `G45`, `G46`, `G47`, with no silent P1 fallback for trade.
2. Normalize Eurostat `nama_10_a64_e` employment using the compound `na_item + unit` interpretation rather than source concept alone.
3. Keep business-statistics labour costs distinct from ESA `D1` compensation of employees. Eurostat `EXPN_SAL_BEN_MEUR` maps to governed `LABOUR_COSTS`, not `D1`.
4. Expand the UK governed source catalogue to include BRES, dedicated industry GFCF and capital-stock/CFC sources; keep ASHE paid-hours use deferred because it is not automatically comparable with national-accounts total hours worked.
5. Keep automatic year fallback disabled until the real country × sector × year × outcome coverage matrix provides evidence for a bounded rule.

## 3. Files changed

- `config/direct_economic_coefficients.yaml` — replacement/update of governed source semantics and source catalogue.
- `tests/fixtures/direct_economic_coefficients/source_rows.csv` — replacement/update of the synthetic trade fixture so employee benefits expense is represented as `LABOUR_COSTS`, not `D1`.
- `tests/test_sko036_live_source_contract.py` — added focused behavioural tests for the approved refinements.
- `docs/sko-036a-live-source-contract-refinement-evidence.md` — this evidence record.

No client data or live source rows were added to committed fixtures.

## 4. Behaviour implemented

### Eurostat employment

The config now records `nama_10_a64_e` raw dimensions and an explicit compound mapping:

- `EMP_DC | THS_PER` → `EMP_PERSONS`, scale 1,000, normalized unit `persons`;
- `EMP_DC | THS_HW` → `EMP_HOURS`, scale 1,000, normalized unit `hours`.

The mapping rule is explicitly recorded as `na_item_and_unit_required`.

### Trade labour costs

A separate `LABOUR_COSTS` outcome is now present. For Eurostat SBS:

- `EXPN_SAL_BEN_MEUR` → `LABOUR_COSTS`;
- it is not mapped to `D1`.

The synthetic G46 trade fixture was corrected accordingly. Therefore the coefficient engine can calculate a turnover-basis labour-cost coefficient where the compatible business-statistics numerator exists, while ESA employee compensation remains held out rather than fabricated.

### UK source catalogue

The governed catalogue now includes:

- ONS Supply and Use Tables;
- Annual Business Survey;
- Business Register and Employment Survey;
- annual GFCF by industry and asset;
- capital stocks and fixed capital consumption;
- ASHE industry hours as an explicitly deferred/not-comparable v1 source.

UK source classifications remain UK SIC 2007 with transparent mapping to NACE-compatible model sectors where applicable.

### Year fallback

The config now states:

- `automatic_fallback_enabled: false`;
- status `deferred_pending_real_coverage_matrix`;
- exact-year use only until an owner-approved bounded fallback rule exists.

## 5. Focused validation

A focused six-test suite validates:

- unchanged hybrid denominator architecture;
- compound Eurostat employment mapping;
- labour-cost versus D1 semantic separation;
- corrected trade fixture semantics;
- expanded UK source catalogue and deferred ASHE hours;
- disabled year fallback pending coverage evidence.

Result: **6 tests passed**.

The new Python test module was also checked with `python -m py_compile` successfully.

The tests use only configuration and synthetic fixture data.

## 6. Acceptance boundary

These refinements are owner-approved and implemented. They do **not** complete SKO-036.

SKO-036 remains in progress because the following acceptance evidence is still missing:

- the real country × A*64 × year × outcome coverage matrix;
- live Eurostat/ONS extraction and normalization at scale;
- measured missing/confidential/incompatible cells;
- evidence supporting a bounded year-fallback rule;
- application of the governed coefficient layer to the 35-observation methodology cohort;
- owner acceptance of the resulting real coefficient methodology.

## 7. Recommended next task

Build the first governed real source coverage matrix across Eurostat and ONS, classifying each country × model sector × reference year × outcome cell as available, missing, confidential, semantically incompatible, denominator-missing or mapping-unavailable. Use that evidence to set the year-fallback rule and then apply the resulting coefficient layer to the 35-observation pilot.
