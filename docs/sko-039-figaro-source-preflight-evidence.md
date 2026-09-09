# SKO-039 FIGARO live-source preflight evidence

**Task:** SKO-039 — Prototype indirect supply-chain attribution using FIGARO  
**Stage:** source preflight / source-contract validation  
**Status:** Implemented and synthetically tested; governing FIGARO 2026 / reference-year 2023 source candidate identified; live source not yet normalized or pilot-run.  
**Branch:** `sko-039-figaro-indirect-attribution`

## Purpose

Establish the governed source boundary required before running the SKO-039 indirect attribution prototype against the governed 35-observation pilot.

SKO-038 remains authoritative for direct attribution. This preflight does not alter SKO-036, SKO-037 or SKO-038 methodology or outputs.

## Governing source decision

Owner approved changing the governing FIGARO release from the 2025 edition to the current 2026 edition while retaining reference year 2023.

The reason is source governance and reproducibility rather than a methodological preference for one vintage. The 2026 release is the current official FIGARO release available from Eurostat/CIRCABC and still provides the required 2023 reference year. Annual FIGARO releases may revise historical values, so edition and reference year are both frozen explicitly.

The primary source is therefore:

- organisation: Eurostat;
- product: FIGARO inter-country supply-use and input-output tables (`naio_10_fcp`);
- edition: 2026;
- reference year: 2023;
- table: industry-by-industry inter-country input-output;
- classification: NACE Rev. 2 A*64;
- valuation: basic prices;
- currency/unit: nominal EUR million.

If the 2025/2023 matrix is subsequently recovered, it is reserved as a release-revision sensitivity rather than the governing primary source.

## Live matrix candidate identified

The owner supplied the official CIRCABC file:

`matrix_eu-ic-io_ind-by-ind_26ed_2023.csv`

Read-only inspection of that supplied file established:

- file size: 49,350,189 bytes;
- SHA-256: `030ff10a923a5d8949e4d05c16c7f2260d59c91f23d74f9529a029ba9249d7a5`;
- 3,206 data rows plus header;
- 3,451 columns;
- 3,200 country×industry rows, consistent with 50 country/region nodes × 64 industries;
- row/column node labels such as `AL_A01`, `DE_C20` and `FR_M72`;
- bottom accounting rows including `W2_D1`, `W2_D29X39` and `W2_B2A3G`.

This confirms that the supplied file is the correct native FIGARO industry-by-industry ICIO matrix type required for the Leontief model. It does not yet constitute completed live-source validation because the native matrix has not yet been normalized into the governed transaction/output/satellite contract and passed through the validator.

## External source findings confirmed during preflight

Eurostat publishes FIGARO inter-country supply, use and input-output tables annually and exposes:

- industry-by-industry input-output tables;
- product-by-product input-output tables;
- 64 NACE Rev. 2 industries;
- nominal million-euro transactions at basic prices;
- country/region coverage including the EU, United Kingdom, major trading partners and Rest of World;
- a 2026 edition covering 2010-2024.

Eurostat also publishes environmental footprints based on FIGARO. Existing environmental-footprint metadata available during this work documents a FIGARO-based 2023 GHG result, but compatibility of the environmental satellite with the governing 2026 FIGARO matrix must be demonstrated before GHG is treated as live-source validated. No 2025-aligned environmental satellite is silently treated as 2026-aligned.

These findings support the approved SKO-039 v1 decision to use an industry-by-industry architecture and a 2023 reference year. Publication-level evidence does not replace file-level validation.

## Source governance implemented

`figaro_source_validation.py` adds a hard gate between downloaded/normalized FIGARO source material and `figaro_indirect_attribution.py`.

A locally frozen normalized package must contain:

- `figaro_source_manifest.json`;
- one normalized transaction file;
- one normalized output-vector file;
- one normalized satellite file.

The manifest must bind exact:

- organisation;
- product ID;
- FIGARO edition;
- reference year;
- table type;
- classification;
- valuation basis;
- currency and unit;
- source filenames;
- source-file roles;
- SHA-256 checksums.

The validator rejects any mismatch between the frozen manifest and the governed SKO-039 config.

## Normalized source contract

### Transactions

Required fields:

`origin_country | origin_sector | destination_country | destination_sector | value_million_eur`

Rules:

- both origin and destination nodes must exist in the frozen output vector;
- normalized transaction values must be non-negative for v1;
- duplicate normalized origin×destination transaction pairs are rejected.

### Output vector

Required fields:

`country | sector | output_million_eur`

Rules:

- country×sector nodes must be unique;
- output must be strictly positive;
- blank country/sector nodes are rejected.

### Satellites

Required fields:

`country | sector | outcome | value | unit`

Governed outcomes are:

- `GVA`;
- `GHG`;
- `EMPLOYMENT_PERSONS`.

Rules:

- satellite nodes must exist in the frozen output vector;
- duplicate outcome×node cells are rejected;
- unit is mandatory;
- missing node coverage is reported explicitly;
- partial employment coverage is permitted at source-validation stage so it can be held out downstream rather than silently imputed as zero;
- a package with no rows at all for one governed outcome is rejected.

## Deterministic source freeze

A valid source package receives a deterministic package fingerprint derived from:

- the exact manifest;
- verified file SHA-256 values;
- normalized contract diagnostics.

Pilot execution is permitted only after all manifest, checksum and normalized-contract gates pass.

## Test execution evidence

The repository environment executed:

```text
python -m py_compile figaro_indirect_attribution.py figaro_source_validation.py
python -m unittest tests.test_figaro_indirect_attribution tests.test_figaro_source_validation -v
```

Result on 2026-09-09:

- indirect-attribution behavioural tests: 14/14 passed;
- source-validation behavioural tests: 11/11 passed;
- combined: 25 tests in 0.038 seconds;
- result: `OK`;
- no failures or errors.

This is synthetic/repository-level test evidence. It is not live FIGARO source validation.

## Live-source validation still required

Before the governed pilot may run:

1. freeze the supplied official `matrix_eu-ic-io_ind-by-ind_26ed_2023.csv` locally under the SKO-039 source area and verify its SHA-256;
2. normalize the 3,200×3,200 inter-industry block into the governed transaction contract;
3. derive the country×industry output vector from the native FIGARO accounting structure and reconcile it against matrix accounting identities;
4. derive the GVA satellite from the native FIGARO value-added/accounting rows where supported and document the exact accounting construction;
5. obtain/freeze a 2023 GHG satellite and demonstrate compatibility with the governing 2026 matrix before using it as a primary indirect outcome;
6. obtain/freeze the employment source and identify where global origin-node coverage is incomplete;
7. create the normalized satellite file without zero-filling missing origin nodes;
8. create `figaro_source_manifest.json` binding governing source files and checksums;
9. run `figaro_source_validation.py` and retain the JSON source-validation summary;
10. rerun focused SKO-039 tests plus relevant SKO-038 regression;
11. only then run the 35-observation live pilot.

## Current status assessment

### Work actually completed

- FIGARO indirect-attribution core implemented on the SKO-039 branch.
- Governed FIGARO source-freeze/source-contract validator implemented.
- Source-validation behavioural tests implemented.
- Focused SKO-039 synthetic test suites executed successfully, 25/25.
- Governing source decision updated to FIGARO 2026 edition / reference year 2023.
- Official 2026/2023 native industry-by-industry matrix candidate identified and structurally inspected.

### Evidence created

- `figaro_indirect_attribution.py`;
- `figaro_source_validation.py`;
- `tests/test_figaro_indirect_attribution.py`;
- `tests/test_figaro_source_validation.py`;
- local executed test output, 25/25 passed;
- supplied native matrix filename, dimensions and SHA-256 recorded in this evidence note.

### Decisions made/preserved

- industry-by-industry FIGARO architecture;
- 2023 primary reference year;
- 2026 FIGARO edition as governing primary release;
- 2025/2023 reserved as release-revision sensitivity if recovered;
- exact country×A*64 mapping only;
- no inheritance of the SKO-037 FR P85→P environmental-direct fallback;
- G45/G46/G47 primary-case valuation hold-out;
- GVA and GHG primary indirect outcomes, subject to live source validation;
- employment as coverage-qualified secondary outcome;
- `L-I` upstream operator;
- no induced household effects;
- gross portfolio upstream sums labelled non-network-deduplicated.

### Status changes

SKO-039 is now legitimately **Implemented / Tested** at synthetic repository level.

SKO-039 remains **not live-source validated, not live-pilot validated, not complete and not accepted**.

No accepted SKO-036, SKO-037 or SKO-038 status has changed.

### Unresolved items

- native FIGARO 2026/2023 matrix normalizer;
- output-vector and GVA accounting extraction/reconciliation from the native matrix;
- aligned/compatible 2023 GHG satellite validation for the 2026 matrix;
- defensible global employment-node coverage;
- complete normalized package manifest and source-validator run;
- relevant SKO-038 regression after live-source adapter work;
- live 35-observation pilot.

### Recommended next task

Implement and test a native FIGARO 2026 matrix normalizer against the supplied `matrix_eu-ic-io_ind-by-ind_26ed_2023.csv`. The normalizer should emit governed transactions and output/GVA structures without altering the source file, reconcile dimensions and accounting totals, and preserve the source SHA-256. Once that live matrix gate passes, proceed to GHG/employment satellite construction and the governed source-package validator.
