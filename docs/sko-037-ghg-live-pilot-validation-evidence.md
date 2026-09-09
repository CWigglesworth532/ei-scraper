# SKO-037 GHG live pilot validation evidence

**Task:** SKO-037 — direct environmental impact coefficient layer, headline direct GHG

**Status:** Implemented, targeted-tested, live-pilot validated; **owner acceptance pending**.

**Branch:** `sko-037-direct-environmental-coefficients`

**Reference year:** 2023 fixed release year; no automatic year fallback.

## Scope and boundary

The implemented v1 layer models direct, residence-based production greenhouse-gas emissions attributable to procurement spend:

supplier spend → supplier country → purchased economic activity / NACE → governed country × sector coefficient → modelled direct GHG

The boundary is direct resident-unit production effects only. The implementation excludes supplier-reported emissions, product carbon footprints, client Scope 3 inventories, territorial inventories, upstream supply-chain emissions and consumption footprints.

Social-economy status is not required or used by the calculation layer.

## Governed denominator architecture

The environmental layer reuses the accepted SKO-036 denominator architecture:

- ordinary sectors: environmental numerator / P1 national-accounts output;
- G45/G46/G47: environmental numerator / turnover;
- trade sectors must not silently fall back to P1;
- absolute physical outcomes use procurement spend converted into denominator currency before applying the coefficient;
- the 2023 EUR→GBP annual-average rate used for UK attribution is 0.86979 GBP per EUR.

## Live source validation

### Eurostat

Governing environmental source: `env_ac_ainah_r2`, 2023, `GHG`.

The live extractor materialised every required non-UK environmental cell for the resolved pilot cohort and normalized Eurostat `THS_T` values to `tonne_co2e` before coefficient construction.

Published Eurostat quality flags `p` (provisional) and `e` (estimated) are retained in provenance and are treated as usable numeric observations. Suppressed/confidential/unavailable observations remain holdouts.

### ONS UK M72

Governing UK source: ONS Environmental Accounts, `GHG total` worksheet, current release used during validation.

Workbook metadata explicitly states:

- mass of air emissions per annum in **thousand tonnes of carbon dioxide equivalent**;
- **UK residence basis**.

The exact 2023 SIC72 value is 215, normalized to **215,000 tCO2e**.

Accepted SKO-036 P1 denominator for UK M72/SIC72: **£45,979 million**.

Derived coefficient:

`215,000 / 45,979 = 4.676047760934339589812740599 tCO2e per £m`

The ONS workbook SHA-256 recorded in the live run is:

`bf3a7e22a695be13ef380180d4cce4858607d5a15e21d81fc6530fe00cf1d8ec`

## Approved FR granularity fallback

Owner-approved rule:

- purchased activity remains `P85`;
- exact P85 environmental cell is structurally unpublished;
- coefficient source sector moves to `P`;
- both numerator and denominator move to `P`;
- route remains the ordinary P1-output route;
- the lower-specificity coefficient source is explicitly disclosed.

Pilot evidence:

- observation `SKO036-007` Intergroupe Francophone Du Myelome;
- `model_sector=P85`;
- `coefficient_source_sector=P`;
- modelled direct GHG = **23.87058971725718435121576686 tCO2e**.

## Behavioural test evidence

Targeted SKO-037 test suite result after live-source integration and status-flag refinement:

- **14 tests run**;
- **14 passed**;
- compilation succeeded for the SKO-037 Python modules.

Tests cover, among other behaviours:

- ordinary P1 denominator route;
- trade turnover route and no P1 fallback;
- confidential/suppressed source holdout;
- provisional/estimated statistical flags remain usable;
- wrong environmental unit holdout;
- UK EUR→GBP attribution;
- approved FR P85→P fallback;
- no unapproved parent fallback;
- no automatic year fallback;
- social-economy status does not affect the result;
- Eurostat thousand-tonnes normalization;
- ONS unit and residence-boundary assertions.

## Live coefficient QA

Live source materialisation produced:

- target country × sector cells: **21**;
- environmental rows: **21**;
- denominator rows: **21**;
- missing environmental cells: **0**;
- missing denominators: **0**.

Coefficient QA after the status-flag governance fix:

- coefficient records: **21**;
- calculated coefficients: **21**;
- held-out coefficients: **0**;
- conflicting source rows: **0**;
- duplicate source rows: **0**;
- missing denominator records: **0**;
- missing numerator records: **0**;
- source-status holdouts: **0**;
- unit-mismatch holdouts: **0**;
- output-route groups: **17**;
- trade-route groups: **4**.

## 35-observation pilot validation

Pilot cohort:

- observations: **35**;
- total spend: **€25,077,933.78**.

Results:

- modelled observations: **27**;
- held out unresolved activity: **8**;
- observation coverage: **27/35 = 77.14%**;
- covered spend: **€22,102,562.52**;
- spend coverage: **88.14%**;
- approved parent-fallback observations: **1**;
- modelled direct GHG across covered spend: **387.8432903074818 tCO2e**.

All eight holdouts are due solely to missing purchased-activity NACE codes:

- SKO036-010 — istituto nazionale per lo
- SKO036-011 — ILUNION
- SKO036-016 — Mapi Research Trust
- SKO036-017 — Katholieke Universiteit Leuven
- SKO036-018 — Institut Gustave Roussy
- SKO036-025 — EIT Food ivzw
- SKO036-032 — Fondazione IRCCS Ist. Naz. Tumori
- SKO036-035 — ILUNION LIMPIEZA Y MEDIOAMBIENTE

Each is recorded as `nace_code_missing` / `held_out_unresolved_activity`.

## UK attribution checks

Two UK M72 observations validate the currency-normalised absolute-outcome route:

- `SKO036-002` University Of Birmingham — €3,235,649.16 → **13.15996619801270145066225886 tCO2e**;
- `SKO036-029` NIAB — €349,908.52 → **1.423140788105917919050001087 tCO2e**.

Combined UK M72 modelled direct GHG: **14.583106986118619369712259947 tCO2e**.

Both use:

- `model_sector=M72`;
- `coefficient_source_sector=M72`;
- denominator route `default`;
- denominator currency GBP;
- EUR→GBP rate 0.86979;
- coefficient `4.676047760934339589812740599 tCO2e per £m`.

## Trade-route checks

Every resolved trade observation uses `trade_turnover`, not P1:

- AT `SKO036-004` G46 — available;
- ES `SKO036-022` G47 — available;
- DE `SKO036-023` G46 — available;
- ES `SKO036-030` G46 — available.

All four use EUR-denominated turnover denominators.

## Validation conclusion

The SKO-037 headline direct-GHG capability has reached its intended methodological ceiling for the governed 35-observation pilot:

- all 21 required live coefficient cells materialised and calculated;
- all 27 observations with resolved purchased activity are modelled;
- all 8 unresolved-NACE observations remain explicitly held out;
- spend coverage is 88.14%, matching the accepted SKO-036 resolved-spend ceiling;
- UK physical-unit and FX handling is validated;
- the owner-approved FR P85→P fallback is exercised and disclosed;
- trade turnover routing is preserved;
- supplier social-economy status is irrelevant to the calculation.

**Acceptance state:** evidence is complete for owner review. Do not mark SKO-037 accepted until the owner explicitly accepts this evidence.
