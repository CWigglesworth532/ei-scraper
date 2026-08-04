# E1.6 Canonical Directory-Candidate Export — Acceptance Evidence

**Task:** SKO-011  
**Epic:** E1 — Canonical skopia entity layer  
**Milestone:** E1.6  
**Status:** Implemented and tested; awaiting owner acceptance  
**Branch:** `e1-6-directory-candidate-export`

## Scope

E1.6 creates a CSV-only directory-candidate export using `entity_id` as the stable directory key.

The export is a review and workflow product. It does not:

- publish records to Airtable or Softr;
- treat canonical identity as proof of social economy classification;
- treat a technical match as proof of directory eligibility;
- assign publication approval;
- expose live client spend records;
- commit live client or directory data to Git.

Airtable synchronization remains deferred.

## Agreed design decisions

### Stable key

Each exported row is keyed by the persistent canonical `entity_id`.

### Candidate population

The export includes one row per canonical entity that:

- has a non-empty `entity_id`;
- has `record_status = active`;
- has at least one accepted, resolved supplier link;
- does not require identity review.

Merged, split, deprecated, quarantined, unlinked and review-required entities are excluded.

### Classification

Existing source-recognition fields are retained as source evidence only.

They do not populate governed classification fields.

Where no governed classification table exists, classification fields remain blank.

### Directory workflow defaults

Unassessed entities enter the export as:

- `directory_candidate_status = research_needed`
- `directory_inclusion_decision = review`
- `Data Status = Review`
- `Verified = No`

### Directory field ownership

skopia is authoritative for:

- stable identity;
- identifiers;
- source linkage;
- match provenance;
- classification evidence;
- candidate assessment;
- internal commercial-materiality indicators.

Airtable remains authoritative for:

- directory display and editorial content;
- website and summary;
- Sector and Social Mission;
- Countries Served;
- public client references;
- final publication status.

### Spend and materiality

Spend remains in a separate confidential entity-linked activity layer.

The directory-candidate export includes internal materiality placeholders, but no live spend data is populated because no governed spend table currently exists.

High spend may later affect review priority and procurement relevance. It does not prove classification or directory eligibility.

## Files added

### `build_directory_candidate_export.py`

New standalone exporter that:

- reads canonical Parquet tables;
- selects eligible active entities;
- aggregates accepted supplier links;
- selects a valid primary identifier;
- extracts source recognition and register provenance;
- preserves classification fields separately;
- adds conservative directory-review defaults;
- includes Airtable-aligned profile columns;
- exposes missing-data QA flags;
- includes internal materiality placeholders;
- writes a deterministic UTF-8 CSV;
- rejects duplicate `entity_id` values.

### `tests/test_build_directory_candidate_export.py`

Synthetic behavioural tests covering:

- one row per eligible active entity;
- exclusion of unlinked and merged entities;
- aggregation of multiple supplier links;
- primary identifier selection;
- rejection of invalid identifier rows;
- preservation of source evidence without automatic classification;
- conservative directory defaults;
- blank spend fields where no governed spend source exists;
- deterministic schema and output order.

No live client or directory records are used in committed tests.

## Validation commands

```bash
python -m py_compile build_directory_candidate_export.py
python -m py_compile tests/test_build_directory_candidate_export.py
python -m unittest tests.test_build_directory_candidate_export -v
python -m unittest discover -s tests -v
```

## Test result

```text
Ran 67 tests in 6.695s

OK
```

This includes all previous regression tests and six new E1.6 behavioural tests.

## Operational export command

```bash
python build_directory_candidate_export.py \
  --canonical-dir data/canonical/e1-5-operational \
  --output data/outputs/directory/directory_candidates.csv
```

## Operational export result

```text
canonical_entity_rows=39
accepted_supplier_link_rows=42
accepted_linked_entity_count=39
eligible_export_rows=39
duplicate_entity_id_count=0
research_needed_count=39
missing_website_count=39
missing_country_hq_count=0
```

## Output QA

```text
rows=39
columns=65
unique_entity_ids=39
duplicate_entity_ids=0
research_needed=39
missing_website=39
missing_country_hq=0
missing_business_summary=39
missing_sector=39
missing_social_mission=39
```

## Behaviour confirmed

- Every exported row has one stable `entity_id`.
- No duplicate canonical IDs appear in the CSV.
- Multiple supplier links are aggregated.
- Invalid identifiers are not selected.
- Source evidence does not create automatic classification.
- All unassessed entities enter the research/review workflow.
- Airtable fields are present without fabricated enrichment.
- Spend fields remain blank without a governed spend source.
- Output schema and order are deterministic.
- No Airtable or Softr synchronization occurs.

## Current limitations and follow-ons

1. No governed `entity_classifications.parquet` currently exists.
2. No governed `directory_readiness.parquet` currently exists.
3. No Airtable export is available for duplicate comparison.
4. No governed client activity or spend table currently exists.
5. All 39 candidates remain `research_needed`.
6. Directory enrichment and publication remain downstream work.
7. Airtable/Softr synchronization remains deferred.
8. A later activity layer should link client, reporting period, supplier record and spend to `entity_id`.

## Acceptance assessment

The required sample export keyed by `entity_id` now exists locally.

It has passed:

- schema validation;
- duplicate-ID validation;
- synthetic behavioural tests;
- full regression testing;
- operational QA against the accepted E1.5 canonical layer.

E1.6 remains **implemented and tested, awaiting owner acceptance**.
