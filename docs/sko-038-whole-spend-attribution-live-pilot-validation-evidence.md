# SKO-038 whole-spend attribution live-pilot validation evidence

**Task:** SKO-038 — governed whole-spend direct attribution engine  
**Epic / milestone:** E4 / E4.11  
**Status:** Implemented, tested, live-pilot validated; **owner acceptance pending**.  
**Branch:** `sko-038-whole-spend-attribution`  
**Reference year:** 2023 fixed coefficient release; no automatic year fallback.

## Purpose

SKO-038 composes the accepted SKO-036 direct-economic and SKO-037 direct-GHG application layers into one supplier-agnostic procurement attribution workflow:

`procurement spend → supplier country → purchased activity / NACE → accepted country × sector coefficient → modelled direct outcome`

The implementation does not reconstruct or override coefficient methodology. It consumes the accepted coefficient/coverage outputs and produces one governed observation/outcome/portfolio/QA contract.

Social-economy status is not required by the calculation layer and remains a separate optional analytical segmentation.

## Implementation delivered

Production module:

- `whole_spend_attribution.py`

Focused behavioural tests:

- `tests/test_whole_spend_attribution.py`

The implementation composes:

- `apply_direct_economic_coefficients.py`
- `apply_direct_environmental_coefficients.py`

and emits:

- observation-level attribution;
- canonical long-form outcome ledger;
- portfolio aggregation by outcome;
- deterministic QA breakdowns by country, model sector, denominator route, coefficient specificity and hold-out reason;
- machine-readable reconciliation summary.

The composer preserves exact coefficient IDs, source lineage, denominator route, denominator currency, FX provenance, coefficient source sector, fallback/specificity provenance and attribution status/reason.

## Synthetic behavioural validation

`whole_spend_attribution.py` passed `py_compile`.

Focused SKO-038 suite:

- **16 tests run**;
- **16 passed**;
- **0 failures / 0 errors**.

Covered behaviours include:

- one workflow composing economic and environmental outcomes;
- unresolved-NACE hold-outs surviving explicitly;
- trade rows restricted to `trade_turnover`;
- approved environmental parent fallback exposed;
- unapproved parent fallback rejected;
- exact coefficient/source lineage required for available outcomes;
- economic rows normalised to exact specificity;
- reference-year mismatch rejected;
- duplicate procurement observation IDs rejected;
- portfolio totals reconcile to the outcome ledger;
- coverage denominator uses full portfolio spend;
- missing FX remains an explicit hold-out;
- `not_applicable` remains distinct from `held_out`;
- social-economy fields do not affect attribution;
- identical runs are deterministic;
- governed QA breakdown dimensions are produced.

## Accepted-layer regression validation

SKO-036 governed regression:

- **59 tests run**;
- **59 passed**.

SKO-037 governed regression:

- **14 tests run**;
- **14 passed**.

No regression in the accepted direct-economic or direct-GHG layers was identified.

## Governed live pilot

Input cohort:

- observations: **35**;
- total spend: **€25,077,933.78**;
- spend currency: EUR;
- coefficient reference year: 2023.

Whole-spend results:

- observations with at least one modelled outcome: **27 / 35**;
- held-out observations: **8 / 35**;
- modelled spend: **€22,102,562.52**;
- spend coverage: **88.1355007709092%**;
- all eight held-out observations are `nace_code_missing` upstream hold-outs;
- outcome ledger rows: **385**;
- portfolio outcomes: **11**.

Core reconciliation checks all passed:

- coefficient lineage complete: **true**;
- duplicate observation/outcome rows: **0**;
- trade route invariant: **passed**;
- fallback invariant: **passed**;
- observation-to-outcome reconciliation: **passed**;
- portfolio-total reconciliation: **passed**.

## Portfolio outcome validation

### Direct economic

The unified workflow reproduces accepted SKO-036 pilot totals and outcome-specific availability:

- Gross value added: **€10,340,270.8294613796202361243** across 27 observations / 88.14% spend;
- Employment persons: **120.8036961641447104460677169 persons-equivalent** across 27 observations / 88.14% spend;
- Intermediate consumption: **€9,845,228.886619714582720422736**;
- Compensation of employees: **€6,379,931.10388207875753238911**;
- Gross operating surplus / mixed income: **€1,984,630.126822531593682143003**;
- Consumption of fixed capital: **€1,429,215.378778661338809326351**;
- Production taxes less subsidies: **-€472,037.223226011500534981895**;
- Gross fixed capital formation: **€1,585,182.656990501668678169371**;
- Employment hours: **76,710.99213534919107171110245 hours**;
- Trade-route labour costs: **€183,133.0423580427412535893066**.

Outcome-specific `not_applicable` and held-out states remain distinct.

### Direct environmental

The unified workflow reproduces the accepted SKO-037 headline result:

- direct GHG: **387.8432903074818 tCO2e**;
- modelled observations: **27 / 35**;
- covered spend: **€22,102,562.52**;
- spend coverage: **88.14%**.

Boundary remains direct residence-based production effects only.

## UK M72 FX and lineage spot-check

The two accepted UK M72 observations retain the governed GBP denominator route and exact environmental coefficient lineage:

### SKO036-002 — University Of Birmingham

- model sector: `M72`;
- coefficient source sector: `M72`;
- denominator route: `default` / P1-output route;
- denominator currency: GBP;
- EUR→GBP rate: **0.86979**;
- coefficient: **4.676047760934339589812740599 tCO2e/£m**;
- modelled GHG: **13.15996619801270145066225886 tCO2e**;
- coefficient ID: `envc_d78f7adf85ba4c28258459c2`.

### SKO036-029 — NIAB

- model sector: `M72`;
- coefficient source sector: `M72`;
- denominator route: `default` / P1-output route;
- denominator currency: GBP;
- EUR→GBP rate: **0.86979**;
- coefficient: **4.676047760934339589812740599 tCO2e/£m**;
- modelled GHG: **1.423140788105917919050001087 tCO2e**;
- coefficient ID: `envc_d78f7adf85ba4c28258459c2`.

## FR P85 approved environmental fallback

Observation `SKO036-007` retains the purchased-activity model sector while exposing the lower-specificity environmental coefficient source:

- model sector: `P85`;
- coefficient source sector: `P`;
- specificity: `approved_parent_fallback`;
- reason: Eurostat `env_ac_ainah_r2` does not publish P85 at the required granularity; owner-approved P-sector environmental fallback dated 2026-09-09;
- modelled direct GHG: **23.87058971725718435121576686 tCO2e**;
- coefficient ID: `envc_b0cf3187585da19c4f2ead8a`.

The purchased activity is not silently relabelled.

## Trade-route validation

All available outcomes for the four resolved trade observations use `trade_turnover` in both domains:

- `SKO036-004` AT G46;
- `SKO036-022` ES G47;
- `SKO036-023` DE G46;
- `SKO036-030` ES G46.

For each, available economic outcomes (GVA, employment persons, labour costs) and direct GHG use `trade_turnover`. No trade observation silently falls back to P1.

## Deterministic rerun

The complete live-pilot workflow was rerun with identical inputs and all five governed outputs compared byte-for-byte using `cmp`:

- `whole_spend_observations.csv`;
- `whole_spend_outcomes.csv`;
- `whole_spend_portfolio_summary.csv`;
- `whole_spend_qa_breakdowns.csv`;
- `whole_spend_attribution_summary.json`.

Result:

`DETERMINISTIC CMP EXIT: 0`

The live outputs are therefore deterministic for identical inputs.

## Known limitation / non-blocking note

The governed 35-observation cohort has no populated `spend_year` values, so the live run records `spend_year_present_observations = 0`. The output contract preserves spend year separately from coefficient reference year, but this particular pilot cannot demonstrate populated spend-year provenance. Synthetic SKO-038 test inputs exercise the spend-year field. This is not a methodology blocker.

## Validation conclusion

SKO-038 has demonstrated the required whole-spend direct attribution behaviour on the governed 35-observation pilot:

- accepted economic and GHG layers are composed through one workflow;
- 27 resolved observations model successfully;
- eight unresolved-NACE observations remain explicit upstream hold-outs;
- economic and environmental portfolio totals reconcile to observation-level outputs;
- exact coefficient/source lineage survives;
- trade turnover routing is preserved;
- UK FX normalisation is preserved;
- the approved FR P85→P environmental fallback is exposed rather than silently relabelled;
- social-economy status is not part of calculation eligibility;
- deterministic reruns are byte-identical;
- no coefficient methodology is recomputed or overridden by the composer.

**Current programme state:** SKO-038 is **implemented, tested and live-pilot validated; owner acceptance pending**. Do not mark SKO-038 or E4.11 Complete / Accepted until Charlie explicitly accepts this evidence.
