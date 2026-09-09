# SKO-036D — Currency normalisation and cross-family employment evidence

Status: **implemented; validation pending owner-local test and pilot rerun**

## Decision

Owner agreed on 2026-09-09 that non-euro absolute-unit outcomes require an explicit currency-normalisation rule before SKO-036 closeout.

The v1 rule is:

- procurement spend is represented in EUR;
- monetary coefficients such as GVA/P1 and P2/P1 are dimensionless ratios, so no FX conversion is applied;
- absolute-unit coefficients such as persons per million denominator currency and hours per million denominator currency require procurement spend to be converted into the denominator currency before the coefficient is applied;
- the FX year is the same fixed reference year as the coefficient release: 2023;
- annual-average rates are used;
- a missing governed FX rate causes the absolute-unit attribution to be held out rather than silently substituted.

## Governed 2023 GBP rate

For UK coefficients the configured rate is **0.86979 GBP per EUR** for 2023.

Source:

- European Central Bank
- series: `EXR.A.GBP.EUR.SP00.A`
- annual average pound sterling / euro reference exchange rate
- quote convention: denominator currency per EUR

## Employment numerator compatibility

The UK M72 pilot has:

- ONS Supply and Use Tables 2023 current-price output P1 = GBP 45,979m;
- ONS BRES 2023 total employment for SIC 72 = 175.4 thousand persons.

The coefficient engine now permits a governed route-specific source-family order for `EMPLOYMENT_PERSONS`:

- default output route: prefer `national_accounts`, then `business_statistics`;
- trade-turnover route: use `business_statistics` only.

This means Eurostat national-accounts employment remains preferred where available, while the UK M72 pilot can use BRES employment against the ONS SUT output denominator without relabelling BRES as national-accounts employment.

Provenance retains both source families when a coefficient combines them.

## Implementation

Changed:

- `config/direct_economic_coefficients.yaml`
  - fixed 2023 FX policy;
  - ECB GBP/EUR rate and provenance;
  - route-specific employment numerator source-family order;
  - UK v1 source-scope clarification.
- `direct_economic_coefficients.py`
  - governed cross-family numerator selection;
  - combined source-family provenance.
- `direct_economic_coverage.py`
  - denominator currency exposed in coverage matrix.
- `apply_direct_economic_coefficients.py`
  - FX-aware absolute-unit attribution;
  - no-FX treatment for dimensionless monetary ratios;
  - explicit FX provenance in outcome output;
  - missing-rate holdout.
- `tests/test_apply_direct_economic_coefficients.py`
  - GBP absolute-unit attribution test.
- `tests/test_sko036_fx_and_cross_family.py`
  - cross-family employment numerator;
  - denominator-currency propagation;
  - missing-FX holdout.

## Acceptance boundary

This work is not yet accepted solely by implementation.

Still required:

1. owner-local `py_compile` pass;
2. focused test pass including the new FX/cross-family tests;
3. rebuild the combined Eurostat + ONS coverage matrix;
4. rerun the 35-observation pilot;
5. confirm UK employment is modelled using the governed 2023 GBP/EUR rate and that the eight unresolved-NACE observations remain the only observation-level holdouts;
6. broader relevant regression and SKO-036 closeout evidence before acceptance.
