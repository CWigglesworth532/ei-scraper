# SKO-028 generic external-indicator ingestion evidence

## Status and scope

SKO-028 / E4.3 is **implemented and tested, but not accepted or complete**.
This prototype ingests tiny, local, versioned reference extracts into one
source-agnostic observation contract. Indicators describe socioeconomic or
environmental context associated with a geography; they do not establish that
supplier presence or client spend caused any regional outcome.

The authoritative tracker and master-context filenames named in the approved
request were not available under the repository, WSL home, or accessible Windows
user tree during implementation. The supplied approved design and completed
preflight were followed; neither programme record was modified or fabricated.

## Work actually completed

- Added a standard-library CSV ingestion module and CLI with explicit config
  validation, caller-supplied `retrieved_at`, raw-file SHA-256 provenance,
  deterministic IDs/fingerprints, stable output ordering, rejection output, and
  hard conflict detection.
- Added config-driven source, dataset, indicator, field-map, unit, and geography
  governance without source-specific normalized columns.
- Added one generic JSON Schema with one row per source/dataset/indicator,
  geography scheme/version/level/code, period, unit, and version context.
- Added tiny synthetic/source-shaped Eurostat, JRC/EDGAR, and ONS fixtures.
- Added T01-T22 behavioural tests and this evidence document.
- Added only the narrow `.gitignore` exception needed for synthetic CSVs.

Files changed are `external_indicator_ingestion.py`,
`config/external_indicators.yaml`,
`schemas/external_indicator_observation.schema.json`,
`tests/test_external_indicator_ingestion.py`, the three CSVs under
`tests/fixtures/external_indicators/`, this document, and the narrow
`.gitignore` exception. Accepted SKO-027 files and behaviour are unchanged.

## Proof set and normalized contract

The synthetic proof set exercises:

- Eurostat-shaped regional unemployment rate and GDP observations on NUTS 2024;
- JRC/EDGAR-shaped regional GHG emissions on the same generic NUTS contract;
- an ONS-shaped regional GVA observation on ITL 2021 using the same lightweight
  local CSV path.

NUTS and ITL remain distinct and are never converted. The exact later linkage
boundary is `geography_scheme + geography_version + geography_level +
geography_code`.

The schema retains observation/source/dataset/indicator metadata, indicator
description/theme, complete geography boundary, period/type, decimal value,
unit, source value status, source/dataset versions and reference, retrieval
time, transformation method/version, input SHA, ingestion version, and
deterministic fingerprint. It contains no entity, supplier, classification, or
client-spend field.

## Governance and deterministic behaviour

- Only configured indicators and units can normalize.
- Unsupported geography metadata and unconfigured indicators enter deterministic
  rejection output rather than silently resolving.
- Exact duplicates deduplicate; conflicting duplicates for the same fully
  versioned key raise `ConflictingObservationError` rather than overwrite.
- Period, unit, dataset version, geography version, and source version preserve
  materially distinct observations.
- Shuffled row order gives the same stable logical observations. Raw file SHA and
  its dependent fingerprint legitimately change when file bytes change.
- Source references are explicit synthetic labels. Operational files remain
  outside Git under ignored data paths.

## Validation evidence

Observed on 2026-08-18:

- `.venv/bin/python -m py_compile external_indicator_ingestion.py` passed.
- Focused suite passed 22/22 tests.
- Full discovery passed 228/228 tests, against the accepted baseline of 206.
- JSON schema parsing and production config-loader validation passed.
- `git diff --check` passed.
- Two CLI runs produced five observations and two deterministic rejections with
  byte-identical output files.
- Observation output SHA-256:
  `281a067f6194bebbbb32f9af09f2ee3d55d880357c51b67aa110ae9060ad8093`.
- Rejection output SHA-256:
  `721a730cf09ad601ea1f1c748fd789ff943169a3b1d79e7b6cad84a675f6cd46`.
- A path-scoped diff confirmed all accepted SKO-027 files are unchanged.

These results constitute implementation evidence only and do not imply owner
acceptance.

## Deliberately not implemented

- No live API/network retrieval, bulk mirrors, or large source datasets.
- No NUTS-to-ITL crosswalk, grid aggregation, geocoding, or geography inference.
- No change to accepted SKO-027 classification behaviour.
- No supplier/canonical linkage, entity creation/mutation, client data, NACE,
  reporting UI, or whole-chain operational analysis.
- No causal attribution from supplier presence or spend to jobs, GDP,
  unemployment, GVA, or emissions.

## Unresolved operational follow-ons

- Supply and configure approved operational Eurostat/GISCO, JRC/EDGAR, and ONS
  reference extracts outside Git.
- Confirm operational source flags, units, dataset releases, and source licenses
  when those extracts are selected.
- Coverage and update cadence depend on authoritative runtime extracts.
- Owner review and explicit acceptance remain required before SKO-028 can be
  called accepted or complete.
