# SKO-036 — Versioned direct economic coefficient layer implementation evidence

**Task:** SKO-036  
**Scope:** supplier-agnostic direct economic coefficient source/materialisation layer  
**Implementation date:** 9 September 2026  
**Status:** Implemented and synthetically tested; owner acceptance pending  

## 1. Purpose

Implement the approved architecture for converting generic procurement spend into modelled **direct** economic outcomes through governed country × sector × year coefficients, without requiring canonical identity or social-economy status.

This task does **not** implement environmental coefficients, FIGARO/Leontief indirect effects, supplier-specific social outcomes, social-economy classification, directory functions or canonical entity creation.

## 2. Owner-approved architecture carried into implementation

The implementation reflects the 9 September 2026 owner decisions:

- use a hybrid denominator architecture;
- use national-accounts output at basic prices where procurement spend is a defensible output proxy;
- use a turnover/revenue route for wholesale/retail sectors rather than applying full invoice spend to trade-margin output;
- preserve transparent holdouts where the compatible denominator or numerator is unavailable;
- build the source layer around a broader production-account scope, not only the original five headline outcomes;
- preserve NACE Rev. 2 as the model target and keep source classification/version explicit;
- keep coefficient materialisation supplier-agnostic.

## 3. Files implemented

- `direct_economic_coefficients.py`
- `config/direct_economic_coefficients.yaml`
- `schemas/direct_economic_source.schema.json`
- `schemas/direct_economic_coefficient.schema.json`
- `tests/test_direct_economic_coefficients.py`
- `tests/fixtures/direct_economic_coefficients/source_rows.csv`
- `docs/sko-036-direct-economic-coefficient-evidence.md`

## 4. Source families represented in configuration

### EU / Eurostat

- `nama_10_a64` — national accounts aggregates by industry, up to NACE Rev. 2 A*64;
- `nama_10_a64_e` — employment by industry;
- `nama_10_a64_p5` — capital formation by industry;
- `sbs_ovw_act` — structural business statistics route for turnover-compatible coefficients.

### UK / ONS

- Input-output supply and use tables — industry inputs, output and GVA under UK SIC 2007;
- Employment multipliers and effects — intended only for direct FTE/output information when normalized into the source contract; no type-1/indirect multiplier logic is implemented;
- Annual Business Survey — intended turnover-compatible route where required.

The source catalogue is metadata/config only at this stage. Real source downloads remain a separately inspectable input step; committed tests use synthetic data only.

## 5. Economic concepts in v1 source contract

The configuration supports:

- `P1` output;
- `P2` intermediate consumption;
- `B1G` gross value added;
- `D1` compensation of employees;
- `D29X39` other taxes less subsidies on production;
- `B2A3G` gross operating surplus and mixed income;
- `P51C` consumption of fixed capital;
- `EMP_PERSONS` persons employed;
- `EMP_HOURS` hours worked;
- `P51G` gross fixed capital formation;
- `TURNOVER` for the business-statistics denominator route.

The code does not silently derive missing economic concepts. Missing numerators are retained as `held_out` coefficient records.

## 6. Denominator logic

### Default route

For ordinary sectors:

`outcome coefficient = numerator / P1 output`

with `P1` required in normalized `million_currency` units and denominator method recorded as `output_basic_prices`.

### Trade route

For NACE A*64 trade sectors `G45`, `G46`, `G47`:

`outcome coefficient = compatible business-statistics numerator / TURNOVER`

The calculation will not substitute national-accounts `P1` if turnover is missing. The coefficient is held out instead.

This prevents full wholesaler/retailer invoice values being treated as though they were national-accounts trade-margin output.

## 7. Versioning and provenance

Every source record carries:

- source family/organisation/dataset;
- release version/date;
- retrieval date and source URL;
- licence;
- source and model classifications and versions;
- source/model sector codes;
- year, concept, value, normalized unit, currency and price basis;
- source fingerprint and schema version.

Every coefficient record additionally carries:

- coefficient schema/release version;
- outcome and numerator/denominator concepts;
- denominator method and compatibility class;
- coefficient value/unit;
- source-record IDs/fingerprints;
- transformation method;
- QA status/reason;
- deterministic record fingerprint.

## 8. QA behaviours

Implemented QA includes:

- required-column validation;
- explicit model classification/version gate;
- deterministic source-record and coefficient IDs;
- identical duplicate suppression and conflicting duplicate rejection;
- no calculation with missing or non-positive denominator;
- explicit numerator-unit and denominator-unit compatibility checks;
- `P1 = P2 + B1G` accounting-identity check where all components exist;
- negative `D29X39` values preserved;
- coverage records by country × model sector × year;
- transparent `complete`, `partial` and `unmodelled` states;
- deterministic rerun and shuffled-input logical equivalence;
- no environmental, indirect/Leontief, canonical-entity, social-economy or Airtable logic.

## 9. Synthetic validation result

Focused test suite: **25 tests passed**.

Validation covered ordinary output-basis coefficients, trade turnover-basis coefficients, missing numerator/denominator holdouts, accounting identities, negative production taxes, source duplication/conflict behaviour, classification version controls, deterministic output and CLI reproducibility.

The focused test data are entirely synthetic and use `synthetic://` source references.

## 10. What is not yet evidenced

This implementation does **not** yet establish:

- successful normalization of live Eurostat downloads;
- successful normalization of live ONS workbooks;
- actual country × A*64 × year completeness;
- approved year-fallback rules;
- real pilot coverage for the 35 researched observations;
- final harmonisation of EU persons/hours with UK FTE;
- whether all trade-sector outcomes can be sourced consistently from SBS/ABS;
- production/client acceptance.

Accordingly SKO-036 must not yet be marked accepted or complete.

## 11. Recommended next implementation step

Perform a **read-only live-source extraction/normalisation run** for a small governed cohort of countries/sectors/years, then report:

- source-cell availability by concept;
- accounting-identity results;
- Eurostat/ONS classification and unit differences;
- trade-route availability;
- likely coefficient-year fallback need;
- prospective coverage of the 35-observation methodology pilot.

Only after that evidence should the coefficient layer be applied to pilot procurement spend.
