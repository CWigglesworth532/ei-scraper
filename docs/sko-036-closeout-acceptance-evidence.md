# SKO-036 closeout and acceptance evidence

Status: **Accepted**  
Acceptance date: **2026-09-09**  
Owner acceptance: user confirmed final regression was all good after the governed test suite passed.

## Scope accepted

SKO-036 delivers the versioned direct economic impact coefficient layer for supplier-agnostic procurement spend using the governed chain:

`procurement spend → supplier country → purchased activity / NACE → fixed 2023 country × sector coefficient release → modelled direct economic outcomes`

Social-economy status is not required for the core calculation and remains an optional analytical segmentation.

## Work actually completed

- Implemented governed direct-economic source and coefficient contracts with deterministic fingerprints and provenance.
- Implemented hybrid denominator architecture:
  - ordinary sectors use national-accounts output (`P1`) where spend-as-output proxy is methodologically defensible;
  - trade sectors G45/G46/G47 use business-statistics turnover and never silently fall back to output.
- Implemented direct outcomes for GVA, intermediate consumption, compensation of employees, production taxes net of subsidies, gross operating surplus/mixed income, consumption of fixed capital, employment persons, employment hours, GFCF, and trade-route labour costs where source-compatible.
- Implemented `B2A3G` derivation as `B2A3N + P51C` with component provenance retained.
- Implemented 0.1m accounting-identity tolerance for published current-price source data.
- Implemented live Eurostat extraction for AT, BE, CH, DE, ES, FI, FR, IE and IT, including A*64 controls and explicit SBS trade-sector retention.
- Implemented fixed 2023 coefficient release policy with no automatic year fallback.
- Implemented NACE-to-model-sector mapping using the unique narrowest compatible A*64 sector and explicit hold-out for unresolved/equally-specific ambiguity.
- Implemented procurement attribution against the coefficient coverage matrix.
- Implemented targeted UK 2023 ONS SUT + BRES pilot path for SIC 72 / model M72.
- Implemented cross-family numerator support where governed, allowing ONS SUT output denominator with BRES employment numerator while retaining both source families in provenance.
- Implemented governed FX normalization for absolute-unit outcomes: spend is converted from EUR into the coefficient denominator currency for persons/hours; dimensionless monetary ratios are not FX-adjusted.
- Implemented combined Eurostat + ONS source build and coverage diagnostics.

## Acceptance evidence

### Final combined pilot

35 procurement observations; total spend **€25,077,933.78**.

- observations with at least one modelled outcome: **27 / 35 (77.14%)**
- spend with at least one modelled outcome: **€22,102,562.52 / €25,077,933.78 (88.14%)**
- remaining held-outs: **8**, all due to unresolved purchased-activity/NACE assignment
- country-source gaps: **0** in the final combined pilot
- GVA available: **27 / 35**, covering **€22,102,562.52 (88.14%)** of cohort spend
- employment persons available: **27 / 35**, covering **€22,102,562.52 (88.14%)** of cohort spend
- modelled direct GVA on covered spend: **€10,340,270.83**
- modelled employment supported: **120.80 persons-equivalent**

The employment result is an attributed estimate of employment supported, not a claim of jobs created.

### Combined coefficient coverage

- countries: **10**
- country × sector pairs: **856**
- coefficient matrix records: **50,670**
- primary 2023 applicable records: **7,542**
- primary 2023 available records: **6,556**
- primary-year applicability-adjusted availability rate: **86.93%**
- accounting identity checks: **4,755**
- accounting identity failures: **0**
- conflicting source rows: **0**

### Regression gate

Final governed regression command:

`python3 -m unittest discover -s tests -p 'test_*direct_economic*.py' -v`

Result supplied by owner:

- **59 tests run**
- **59 passed**
- **0 failures / 0 errors**
- runtime: **0.160s**

Relevant Python modules also passed `py_compile` during the working sequence.

## Decisions accepted

- 2023 is the fixed v1 coefficient reference year.
- Procurement spend year does not have to equal the coefficient reference year; both are retained separately in provenance when available.
- No automatic coefficient-year fallback is allowed in v1.
- Ordinary sectors use the output-proxy route; G45/G46/G47 use turnover-compatible trade routing only.
- Employment source semantics require concept + unit; Eurostat `EMP_DC + THS_PER` maps to persons and `EMP_DC + THS_HW` maps to hours.
- Business-statistics labour costs are distinct from ESA D1 compensation of employees.
- SBS detailed sectors are not silently promoted to A*64; explicit governed trade-sector retention is allowed for G45/G46/G47.
- `B2A3G` may be derived from `B2A3N + P51C` with both components retained in provenance.
- NACE mapping selects the unique narrowest compatible model-sector span; equal-specificity ambiguity remains unresolved.
- UK SIC 2007 is not relabelled as NACE; mapping to the NACE-compatible model layer is explicit.
- Cross-family numerators are allowed only where explicitly governed and source semantics remain preserved.
- Absolute-unit metrics for non-EUR coefficient denominators require governed FX normalization; dimensionless monetary ratios do not.
- Eight unresolved pilot rows remain intentionally held out rather than assigned or split without evidence.

## Unresolved items / boundaries

These are not blockers to SKO-036 acceptance:

- the eight pilot observations with unresolved purchased activity/NACE remain unmodelled pending future activity research;
- UK coverage beyond the targeted SIC 72 pilot remains a future source-expansion task;
- hours and capital-related outcome coverage remains lower than GVA/employment and should be disclosed outcome-by-outcome;
- supplier-specific social outcomes, indirect/induced economic effects, environmental/FIGARO extensions and wider impact modelling are outside SKO-036 scope.

## Acceptance conclusion

The methodology has been demonstrated end-to-end on real procurement spend across Eurostat and UK ONS sources. For observations with a resolved purchased activity and supported country/sector cell, the governed 2023 coefficient layer produces reproducible direct economic outcomes with explicit provenance, route compatibility, hold-outs and FX treatment. SKO-036 therefore meets its acceptance objective and is accepted.

## Recommended next task

Update the authoritative skopia development tracker to mark SKO-036 **Accepted**, preserving the distinction between completed implementation/evidence/decisions and the non-blocking unresolved items above. Only then move to the next roadmap task.