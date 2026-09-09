# SKO-036B — Route applicability and operating-surplus refinement evidence

**Task:** SKO-036  
**Date:** 9 September 2026  
**Status:** Implemented; owner-local regression and live coverage rerun pending

## Owner-approved refinements

Following the first governed live Eurostat coverage build, the owner approved three refinements:

1. use a 0.1 million EUR absolute tolerance for the `P1 = P2 + B1G` accounting identity, reflecting observed published-data rounding;
2. ingest `B2A3N` and derive gross operating surplus/mixed income as `B2A3G = B2A3N + P51C`, retaining both component source records in coefficient provenance;
3. distinguish route-incompatible outcomes as `not_applicable` rather than penalising them as missing coverage.

## Implemented changes

- `config/direct_economic_coefficients.yaml`
  - accounting identity tolerance changed from `0.000001` to `0.1` million EUR;
  - explicit `applicable_routes` added to all outcomes;
  - operating-surplus derivation contract added using `B2A3N + P51C`;
  - Eurostat national-accounts catalogue updated to describe the derived concept and provenance rule.
- `extract_eurostat_direct_economic.py`
  - national-accounts extraction now requests `B2A3N` rather than the unavailable `B2A3G` item;
  - existing A*64 constraint and total-fixed-assets P51G selection remain in place.
- `direct_economic_coefficients.py`
  - pre-indexed group materialisation retained for live-scale performance;
  - route applicability now yields `not_applicable` status where appropriate;
  - derived numerators are supported with component source-record provenance;
  - `B2A3G` can be calculated from `B2A3N + P51C` when direct B2A3G is absent;
  - QA now counts not-applicable and derived numerator records separately.
- `direct_economic_coverage.py`
  - exposes `available`, `held_out`, and `not_applicable` states;
  - reports an applicability-adjusted availability rate.
- focused tests updated/added for route applicability, derivation provenance, accounting tolerance, and extractor semantics.

## Acceptance boundary

This evidence does not claim SKO-036 acceptance. Required next evidence is:

- owner-local `py_compile` and focused/full regression pass;
- regenerated live Eurostat extract containing `B2A3N`;
- regenerated governed coverage matrix using the refined route semantics;
- review of applicability-adjusted coverage and year diagnostics;
- owner decision on bounded year fallback;
- UK/ONS layer and methodology-cohort attribution remain subsequent work.

The tracker must remain **In progress** until those acceptance conditions are evidenced.
