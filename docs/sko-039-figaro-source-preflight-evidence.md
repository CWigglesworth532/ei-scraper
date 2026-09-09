# SKO-039 FIGARO live-source preflight evidence

**Task:** SKO-039 — Prototype indirect supply-chain attribution using FIGARO  
**Stage:** source preflight / source-contract validation  
**Status:** Implementation added; synthetic source-contract validation available; live FIGARO package not yet frozen or pilot-run.  
**Branch:** `sko-039-figaro-indirect-attribution`

## Purpose

Establish the governed source boundary required before running the SKO-039 indirect attribution prototype against the governed 35-observation pilot.

SKO-038 remains authoritative for direct attribution. This preflight does not alter SKO-036, SKO-037 or SKO-038 methodology or outputs.

## External source findings confirmed during preflight

Eurostat currently publishes FIGARO inter-country supply, use and input-output tables annually and exposes:

- industry-by-industry input-output tables;
- product-by-product input-output tables;
- 64 NACE Rev. 2 industries;
- nominal million-euro transactions at basic prices;
- country/region coverage including the EU, United Kingdom, major trading partners and Rest of World;
- a 2026 edition covering 2010-2024.

Eurostat also publishes environmental footprints based on FIGARO. The environmental database explicitly states that the 2025-edition GHG footprint results use Leontief-type modelling with air-emissions accounts and FIGARO industry-by-industry inter-country input-output tables. The 2023 greenhouse-gas footprint result remains available.

These findings support the already approved SKO-039 v1 decision to use an industry-by-industry architecture and a 2023 reference year. They do not by themselves constitute file-level live-source validation.

## Source governance implemented

`figaro_source_validation.py` adds a hard gate between downloaded/normalized FIGARO source material and `figaro_indirect_attribution.py`.

A locally frozen package must contain:

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

## Tests added

`tests/test_figaro_source_validation.py` covers:

1. valid frozen-package acceptance and deterministic fingerprint;
2. reference-year mismatch rejection;
3. FIGARO-edition mismatch rejection;
4. checksum mismatch rejection;
5. unknown transaction-node rejection;
6. duplicate output-node rejection;
7. negative normalized transaction rejection;
8. unsupported satellite-outcome rejection;
9. missing governed satellite-outcome rejection;
10. partial employment coverage reported rather than imputed;
11. identical package fingerprint determinism.

The connected GitHub environment does not provide an execution runner and no repository CI status was attached to the commit at the time of this preflight. Therefore these tests are **implemented but not claimed as executed evidence in this document**. They must be run in the repository environment before this source-validation patch is treated as tested.

## Live-source validation still required

Before the governed pilot may run:

1. download the exact FIGARO industry-by-industry 2025-edition / 2023 source table from the official Eurostat/CIRCABC release;
2. record original official filename, retrieval route and SHA-256;
3. normalize the inter-industry transactions and output vector into the contract above;
4. obtain/freeze the aligned 2023 GHG satellite source and record exact source lineage;
5. obtain/freeze the employment source and identify where global origin-node coverage is incomplete;
6. build the normalized satellite file without zero-filling missing origin nodes;
7. create `figaro_source_manifest.json`;
8. run `figaro_source_validation.py` and retain the JSON source-validation summary;
9. run `py_compile`, the SKO-039 focused suites and relevant SKO-038 regression;
10. only then run the 35-observation live pilot.

## Current status assessment

### Work actually completed

- FIGARO indirect-attribution core implemented on the SKO-039 branch.
- Governed FIGARO source-freeze/source-contract validator implemented.
- Source-validation behavioural tests added.
- Eurostat publication architecture and current source availability reviewed.

### Evidence created

- `figaro_source_validation.py`;
- `tests/test_figaro_source_validation.py`;
- this source-preflight evidence record.

### Decisions preserved

- industry-by-industry FIGARO architecture;
- 2023 primary reference year;
- 2025 FIGARO edition as primary release for the initial prototype, with 2026/2023 reserved as release-revision sensitivity;
- exact country×A*64 mapping only;
- no inheritance of the SKO-037 FR P85→P environmental-direct fallback;
- G45/G46/G47 primary-case valuation hold-out;
- GVA and GHG primary indirect outcomes;
- employment as coverage-qualified secondary outcome;
- `L-I` upstream operator;
- no induced household effects;
- gross portfolio upstream sums labelled non-network-deduplicated.

### Status changes

None to accepted programme milestones.

SKO-039 remains **not accepted** and must not be described as live-source validated or live-pilot validated yet.

### Unresolved items

- exact official FIGARO 2025/2023 file freeze and checksum;
- normalization of the actual FIGARO flat/matrix source structure;
- aligned GHG source freeze;
- defensible global employment-node coverage;
- source-validator test execution in repository environment;
- live 35-observation pilot.

### Recommended next task

Run the new source-validation tests locally, then freeze and normalize the actual official FIGARO 2025/2023 files and execute the source-validation gate. If that passes, proceed directly to the governed 35-observation SKO-039 pilot without changing SKO-038 direct outputs.
