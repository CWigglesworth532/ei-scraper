# SKO-027 governed on-demand geographic classification evidence

## Status and scope

SKO-027 / E4.2 is **implemented and tested, but not accepted or complete**. This
change adds a standalone, selected-only bridge from supplier location
observations to versioned statistical geography. The owner-review follow-up
generalises the assertion from NUTS-only fields to scheme-aware
`geography_scheme/version/level/code/name` fields supporting governed EU NUTS
and UK ONS ITL. It does not alter canonical identity, matching, directory
integration, canonical stores, or client data.

### Owner acceptance

Accepted by the owner on 2026-08-18:

> SKO-027 is accepted. The governed on-demand statistical-geography bridge is
> implemented and tested. It supports explicitly selected canonical entities
> and supplier observations; preserves canonical identity and registered/HQ
> versus contracted-activity location roles; maps only through governed
> country-and-postcode correspondence; represents EU NUTS and UK ONS ITL as
> explicit versioned statistical-geography schemes; rejects unsupported
> scheme/country/version combinations rather than inferring precision; retains
> mapping method, source, version, SHA and confidence provenance; preserves
> ambiguous and insufficient-evidence outcomes; and produces deterministic
> outputs. Validation passed with 24 focused tests, 206 full repository tests,
> schema validation, git diff --check, and byte-identical reruns. No live client
> data, indicator ingestion, geocoding, fuzzy geography, canonical-ID creation
> or universal enrichment capability was introduced. Operational
> Eurostat/GISCO and ONS correspondence coverage remains a runtime dependency
> and E4.3 remains separately governed.

### Files changed

- `.gitignore` — narrow synthetic-fixture exception.
- `geographic_classification.py` — standalone classifier and CLI.
- `config/geographic_classification.yaml` — explicit geography governance.
- `schemas/geographic_classification.schema.json` — generic output contract.
- `tests/test_geographic_classification.py` — T01-T24 behavioural suite.
- `tests/fixtures/geographic_classification/selected_suppliers.csv` — synthetic subjects.
- `tests/fixtures/geographic_classification/postcode_correspondence.csv` —
  synthetic NUTS, ITL, duplicate, ambiguous, and invalid GB/NUTS assertions.
- `docs/sko-027-geographic-classification-evidence.md` — this evidence record.

## Governed behaviour

- Only rows whose `selected` value is explicitly configured as `true` execute.
- Supported subjects are `canonical_entity` and `supplier_observation`.
- Existing `entity_id` values are passed through unchanged. Empty observation
  entity IDs remain empty; the classifier has no entity-creation path.
- `location_role` is validated and preserved as `registered_or_hq`,
  `contracted_activity`, or `unknown`. One role is never inferred from another.
- Version 1 uses only exact country plus conservatively normalized postcode.
- Country accepts ISO2 or an explicit config mapping. Unsupported values are
  retained with `unsupported_country` status.
- Postcodes are trimmed, uppercased, and whitespace-collapsed only.
- Every assertion carries explicit scheme, version, level, code, and name.
- Configuration binds each scheme to allowed countries, versions, and levels:
  EU countries may emit governed NUTS 2024; GB may emit governed ITL 2021.
- A row is not assumed to be NUTS because it contains a region code. The
  synthetic GB/NUTS 2024 row is rejected by governance.
- There is no historic-UK-NUTS to current-ITL translation or assumption.
- One distinct supported result resolves, duplicates of that result deduplicate,
  multiple distinct results are ambiguous, and no result is insufficient.
- City and address remain provenance fields only. No fuzzy matching, geocoding,
  coordinate, city, province, or address inference exists.
- Caller-supplied `classified_at`, mapping-source name/version, source-file SHA,
  evidence, confidence, versions, deterministic fingerprints, and classifier
  version are retained on every output.
- Output is stable-sorted. IDs and fingerprints derive from governed content.

## Synthetic fixture evidence

Committed fixtures are synthetic only. They exercise EU/NUTS and GB/ITL
resolution, config-driven country normalization, duplicates, a rejected
GB/NUTS 2024 row, an ambiguous Belgian postcode, missing/no-match geography,
an unsupported country, separate location roles, and an unselected row.

Observed synthetic outcomes:

| subject_id | role | status | scheme | version | level | code |
|---|---|---|---|---|---|---|
| `ent_eu` | `registered_or_hq` | `resolved` | NUTS | 2024 | 3 | `BE211` |
| `ent_alpha` | `registered_or_hq` | `resolved` | ITL | 2021 | 3 | `TLI32` |
| `obs_activity` | `contracted_activity` | `resolved` | ITL | 2021 | 3 | `TLM50` |
| `obs_ambiguous` | `unknown` | `ambiguous` | empty | empty | empty | empty |
| `obs_no_match` | `unknown` | `insufficient_evidence` | empty | empty | empty | empty |
| `obs_unsupported` | `unknown` | `unsupported_country` | empty | empty | empty | empty |

`ent_not_selected` produces no output. Canonical IDs `ent_alpha` and `ent_eu`
remain unchanged; every supplier observation retains an empty `entity_id`.
The invalid GB/NUTS 2024 row alone yields `insufficient_evidence`, proving GB
cannot be falsely labelled as current NUTS.

## Behavioural and regression validation

The focused suite defines T01-T24 in
`tests/test_geographic_classification.py`. It covers resolution, country and
postcode normalization, ambiguity and deduplication, unsupported/missing
geography, selected-only execution, identity non-mutation/non-creation,
subject types, role separation, provenance/version/SHA retention, explicit
time, deterministic reruns and shuffled inputs, prohibited city/address
inference, validation failures, stable serialization, and absence of indicator
fields.

Observed on 2026-08-18:

- `.venv/bin/python -m py_compile geographic_classification.py` — passed.
- `.venv/bin/python -m unittest tests.test_geographic_classification -v` —
  24 tests passed.
- `.venv/bin/python -m unittest discover -s tests -v` — 206 tests passed.
- The JSON schema parsed successfully.
- Two independent CLI runs produced byte-identical CSV output with SHA-256
  `52aabdc4a4cb9fccc5c5d89a6682214a8add2f7986f02b5a1fb2f3ad25057e8c`.
- Eight fixture input rows produced seven output rows plus a header: the sole
  `selected=false` row produced no classification.
- Canonical outputs retained `ent_alpha` and `ent_eu` unchanged. All five
  `supplier_observation` outputs retained empty `entity_id` values.
- HQ `ent_alpha` remained `registered_or_hq`; work-site
  `obs_activity` remained `contracted_activity`.

These are inspectable implementation results, not owner acceptance. SKO-027 is
not marked accepted or complete.

## Decisions implemented

- EU regional geography uses explicitly governed NUTS correspondence.
- UK regional geography uses explicitly governed ONS ITL correspondence.
- Scheme/version are explicit content validated against configuration; code
  shapes never determine the scheme.
- Historic UK NUTS is not silently translated into current ITL.
- Eurostat, GISCO, and ONS indicator ingestion remains E4.3.

## Limitations and follow-up

- Runtime operators must supply an authoritative correspondence file and its
  meaningful source/version labels; only its SHA is computed automatically.
- Coverage and precision cannot exceed that correspondence source.
- Ambiguous postcodes remain unresolved; there is intentionally no fallback.
- No socioeconomic or environmental indicator ingestion is included (E4.3).
- There is no population-wide or universal canonical enrichment path.
- Acceptance and programme-status updates remain owner actions.
- Authoritative operational Eurostat/GISCO postcode correspondence remains to
  be supplied and configured outside Git.
- Authoritative operational ONS ITL correspondence remains to be supplied and
  configured outside Git.
