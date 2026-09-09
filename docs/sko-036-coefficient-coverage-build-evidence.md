# SKO-036 — Coefficient coverage build evidence

**Task:** SKO-036  
**Scope:** governed country × model-sector × year × outcome coefficient coverage diagnostics  
**Date:** 9 September 2026  
**Status:** Coverage-builder implemented; live authoritative coverage execution pending

## Purpose

Extend the accepted SKO-036 coefficient architecture with a deterministic coverage matrix that makes real coefficient availability inspectable before procurement attribution.

The intended sequence remains:

`authoritative normalized source observations → coefficient materialisation → coverage matrix → year diagnostics → owner-approved fallback policy → pilot attribution`

This task does not perform supplier attribution, social-economy classification, environmental modelling, FIGARO/Leontief modelling, canonical mutation, directory integration or publication.

## Files added

- `direct_economic_coverage.py`
- `tests/test_direct_economic_coverage.py`
- `docs/sko-036-coefficient-coverage-build-evidence.md`

The previously owner-approved SKO-036 source-contract refinements remain in `config/direct_economic_coefficients.yaml` and the synthetic trade fixture.

## Implemented coverage contract

For every materialised coefficient record the coverage layer exposes:

- country;
- model sector code/label;
- reference year;
- denominator route and denominator concept;
- outcome and numerator concept;
- availability state (`available` or `held_out`);
- holdout reason;
- coefficient value/unit/ID where calculated;
- source family, dataset IDs, release versions and release dates.

A separate year-diagnostic table records for each country × sector × outcome:

- all available years;
- earliest and latest usable year;
- count of usable years;
- internal missing years;
- current fallback-policy state.

No automatic year fallback is introduced. The approved configuration remains `deferred_pending_real_coverage_matrix`.

## Behavioural tests

The committed focused tests cover:

1. available and held-out coverage states;
2. explicit turnover route for G46;
3. business-statistics `LABOUR_COSTS` availability;
4. trade `EMPLOYEE_COMPENSATION` holdout when genuine D1 is unavailable;
5. multi-year diagnostics;
6. deferred fallback state;
7. deterministic output under input reordering.

The module was also syntax-checked in the current execution environment. A full repository test run has not been asserted from this environment.

## Live execution status

The coverage builder is ready to consume real normalized Eurostat/ONS observations. However, this execution environment cannot directly access the local ignored SKO-036 source extracts under `/home/charliewigglesworth/ei-scraper/data/...`, and direct runtime retrieval of Eurostat API cells is unavailable here.

Therefore this evidence does **not** claim that the required real country × A*64 × year × outcome coverage matrix has yet been produced.

The real build remains required before SKO-036 can be accepted or marked complete.

## Required next execution

Run the coverage builder locally against the governed real normalized source extract, producing ignored outputs such as:

- `data/pilots/sko-036/coefficient_coverage_matrix.csv`
- `data/pilots/sko-036/coefficient_year_diagnostics.csv`
- `data/pilots/sko-036/coefficient_coverage_summary.json`

Then review at minimum:

- pilot-country and pilot-sector coverage;
- output-route versus turnover-route coverage;
- outcome-specific missingness;
- latest usable year by country/sector/outcome;
- internal year gaps;
- whether a bounded preceding-year fallback is empirically necessary;
- prospective coverage for the accepted 35-observation cohort.

No live client rows should be committed. Aggregate QA and non-client source coverage evidence may be committed after review.

## Acceptance status

SKO-036 remains **In progress**.

Owner acceptance requires the real authoritative coverage build and review in addition to the already completed architecture, source-contract refinement and synthetic behavioural evidence.
