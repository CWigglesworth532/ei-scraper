# SKO-036C — Fixed coefficient reference-year policy evidence

**Status:** Implemented for validation; SKO-036 remains in progress / not accepted  
**Date:** 9 September 2026

## Owner decision

Following the live Eurostat coverage build and year diagnostics, the owner approved a fixed reference-year policy for the v1 direct-economic coefficient release rather than an automatic year-fallback hierarchy.

The v1 policy is:

- primary coefficient reference year: **2023**;
- procurement spend year does not need to equal coefficient reference year;
- use exact 2023 country × model-sector × outcome coefficients only;
- no automatic fallback to 2022, 2024 or another year;
- if the 2023 coefficient is unavailable, hold the outcome out for review;
- always record spend year and coefficient reference year separately in downstream attribution outputs;
- future coefficient releases may move to a newer common reference year after coverage review.

## Evidence supporting the decision

The live Eurostat year diagnostics showed:

- 2023 exact-year availability: **6544 / 6558 = 99.8%** among country × sector × outcome diagnostics with at least one available year;
- 2024 exact-year availability: **4649 / 6558 = 70.9%**;
- 2024 plus immediately preceding year: **6550 / 6558 = 99.9%**;
- only six diagnostics had internal year gaps, each with two gap years.

The owner concluded that a stable common reference year is clearer and more reproducible than mixing coefficient years within one release, particularly because annual movements in the national-accounts/business-statistics ratios are not the intended analytical focus of the procurement-impact methodology.

## Implemented changes

- `config/direct_economic_coefficients.yaml`
  - replaced the deferred fallback policy with `reference_year_policy`;
  - set `mode: fixed_release_year`;
  - set `primary_reference_year: 2023`;
  - disabled automatic fallback;
  - explicitly separated spend-year and coefficient-year provenance;
  - specified holdout when 2023 is unavailable.
- `direct_economic_coverage.py`
  - reports the approved reference-year policy and primary reference year;
  - adds exact-2023 coverage metrics to the summary;
  - year diagnostics record reference-year policy status and primary year.
- tests updated to assert the fixed-2023 policy and exact-year-only summary behaviour.

## Acceptance boundary

This decision does **not** complete SKO-036.

Remaining acceptance work includes:

1. rerun focused regression after this policy change;
2. regenerate live coverage and confirm exact-2023 coverage under the governed policy;
3. apply the fixed-year coefficient release to the real 35-observation procurement cohort;
4. measure observation- and spend-weighted coverage, outcome availability, unresolved NACE rows and denominator-route behaviour;
5. build/validate the UK ONS coefficient route for the UK pilot observations;
6. record final owner acceptance evidence before marking SKO-036 complete.

## Methodological implication

The v1 calculation chain is now:

`procurement observation (spend year retained) → supplier country → purchased activity / NACE → fixed 2023 coefficient release → modelled direct economic outcomes`

This deliberately avoids silently treating coefficient vintage as identical to procurement spend year.
