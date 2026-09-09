# SKO-036E — Combined Eurostat + ONS pilot and FX validation evidence

## Status

Tested on the governed 35-observation methodology cohort. Not yet accepted/complete: broader regression and owner acceptance remain outstanding.

## Scope

This evidence records the first combined Eurostat + ONS procurement attribution run after implementing:

- the fixed 2023 coefficient reference-year policy;
- Eurostat national-accounts and SBS hybrid denominator routes;
- ONS 2023 SIC 72/M72 current-price P1/P2/GVA extraction;
- BRES 2023 SIC 72 employment evidence;
- governed cross-family numerator use for UK employment;
- governed currency normalisation for absolute-unit outcomes only.

The calculation remains supplier-agnostic. Social-economy status is not required by the core model.

## Governing currency rule

For dimensionless monetary ratios such as GVA/P1 and P2/P1, no FX conversion is required when applying a country coefficient to client spend because numerator and denominator are in the same source currency and the ratio is currency-invariant.

For absolute-unit coefficients such as persons per million denominator currency or hours per million denominator currency, procurement spend must first be converted from the client spend currency into the coefficient denominator currency using the governed reference-year FX rate.

For the 2023 UK pilot, EUR-denominated client spend is converted to GBP before applying the UK employment coefficient. The attribution output records the currency-normalisation policy and FX adjustment status.

## Combined coefficient-layer result

The combined Eurostat + ONS coverage build produced:

- countries: 10;
- country × model-sector pairs: 856;
- outcomes: 10;
- coefficient matrix records: 50,670;
- applicable records: 44,631;
- available records: 37,320;
- held-out records: 7,311;
- not-applicable records: 6,039;
- all-year applicable availability rate: 83.6190%;
- primary reference year: 2023;
- 2023 matrix records: 8,560;
- 2023 applicable records: 7,542;
- 2023 available records: 6,556;
- 2023 applicable availability rate: 86.9265%;
- accounting identity checks: 4,755;
- accounting identity failures: 0;
- conflicting source rows: 0;
- duplicate source rows: 0.

## 35-observation procurement pilot

Input cohort:

- observations: 35;
- total spend: EUR 25,077,933.78;
- spend-year values present: 0;
- fixed coefficient reference year: 2023;
- automatic year fallback: disabled.

Model mapping outcomes:

- 23 observations mapped by division-to-A*64 range;
- 4 observations mapped by the governed most-specific A*64-sector rule;
- 8 observations held out because no NACE code was assigned.

Attribution outcomes:

- observations with at least one available outcome: 27/35 (77.14%);
- spend with at least one available outcome: EUR 22,102,562.52 / EUR 25,077,933.78 (88.14%);
- observation statuses: 13 modelled, 14 modelled partial, 8 held out for unresolved NACE;
- country-source gaps: 0;
- mapped observations with no available outcomes: 0;
- FX-adjusted available outcomes: 2.

Headline modelled outcomes on supported spend:

- GVA: EUR 10,340,270.83 across 27 observations / EUR 22,102,562.52 spend;
- employment supported: 120.8037 persons across 27 observations / EUR 22,102,562.52 spend;
- intermediate consumption: EUR 9,845,228.89 across 23 observations / EUR 19,873,893.55 applicable spend;
- employee compensation: EUR 6,379,931.10 across 21 observations / EUR 16,288,335.87 applicable spend;
- operating surplus and mixed income: EUR 1,984,630.13 across 20 observations / EUR 16,143,604.02 applicable spend;
- consumption of fixed capital: EUR 1,429,215.38 across 20 observations / EUR 16,143,604.02 applicable spend;
- gross fixed capital formation: EUR 1,585,182.66 across 14 observations / EUR 14,070,531.45 applicable spend;
- production taxes less subsidies: EUR -472,037.22 across 20 observations / EUR 16,143,604.02 applicable spend;
- employment hours: 76,710.99 hours across 14 observations / EUR 7,296,116.31 applicable spend;
- trade labour costs: EUR 183,133.04 across 4 observations / EUR 2,228,668.97 applicable spend.

These outcome totals must not be summed together as a single impact total because several are accounting components of GVA or otherwise overlapping measures.

## Methodological conclusion

The pilot demonstrates that the governed method can take procurement observations with country and purchased-activity/NACE mapping and convert them into direct economic outcomes using a fixed 2023 country × sector coefficient release.

For the present cohort, all non-modelled observations are now explained by unresolved purchased-activity/NACE assignment rather than failure of the economic coefficient layer. The 8 unresolved observations remain deliberately held out; no NACE values or spend splits are invented.

The combined Eurostat + ONS run therefore demonstrates the core methodology at pilot level for supplier-agnostic procurement spend. It does not yet constitute task acceptance.

## Remaining acceptance evidence

Before SKO-036 may be marked accepted/complete:

1. run the broader relevant regression suite after the final FX/cross-family changes;
2. confirm py_compile for the touched Python modules;
3. retain a reproducible final coverage/pilot build with coherent source retrieval and generation timestamps;
4. record owner acceptance;
5. update the authoritative skopia development tracker, separating completed work, evidence, decisions, status changes, unresolved items and recommended next task.

## Unresolved items

- 8 pilot observations have no accepted purchased-activity/NACE assignment and remain held out;
- outcome availability varies by source and sector; lower-coverage outcomes remain supplementary rather than headline measures;
- the current ONS implementation is intentionally narrow (2023 SIC 72/M72 pilot) and should be expanded only through a separately governed source-coverage task if broader UK sector coverage is required.
