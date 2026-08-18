# SKO-030 context integration implementation evidence

## Status and scope

Implemented for owner review on branch `sko-022-directory-integration-prototype`. This is an additive orchestration-only layer. It consumes accepted SKO-027 geography classifications, SKO-028 external indicator observations, and SKO-029 activity evidence without changing those modules, their configs, schemas, fixtures, or tests.

The implementation does not establish task acceptance. Owner review remains required.

## Files added

- `context_integration.py` — deterministic local-CSV orchestration and QA CLI.
- `config/context_integration.yaml` — explicit subject/geography join and evidence-summary governance.
- `schemas/context_integration_record.schema.json` — Draft 2020-12 output record contract.
- `tests/test_context_integration.py` — behavioural, schema, deterministic, shuffle, dependency, and QA tests.
- `tests/fixtures/context_integration/*` — wholly synthetic ten-subject production-shaped cohort.
- `docs/sko-030-context-integration-evidence.md` — this evidence record.

No file was replaced or deleted.

## Behaviour and boundaries

- Subject join key is exactly `subject_type + subject_id`.
- `entity_id` is retained and checked as identity context. A blank `entity_id` is supported for supplier observations. It is never used as the sole join key.
- Geography/indicator join key is exactly `geography_scheme + geography_version + geography_level + geography_code`.
- No NUTS/ITL crosswalk, geography version conversion, fuzzy matching, or partial geography match exists.
- Activity evidence is grouped by subject into deterministic counts and JSON ID/status/authority summaries before contextual rows are emitted. Evidence rows do not multiply indicator rows and no industry is inferred.
- Enriched records make regional context available beside a subject. They do not assert that the subject caused, contributed to, or is responsible for an indicator value.
- Resolved classifications with no exact-compatible indicator emit `no_compatible_indicator` staging records. Non-resolved classifications emit `unresolved_geography` staging records, retaining activity summaries.

## Synthetic cohort coverage and deterministic QA

The fixtures contain 10 selected subjects: 6 canonical entities and 4 supplier observations. They cover a fully enriched NUTS subject, GB/ITL, a blank supplier `entity_id`, resolved/no-indicator, unresolved-with-activity, contextual/weak-only activity, ambiguous activity, multiple activity rows, a geography-version mismatch, and multiple indicators/periods.

| QA measure | Expected |
|---|---:|
| selected_subjects | 10 |
| canonical_selected_subjects | 6 |
| non_canonical_selected_subjects | 4 |
| geography_classification_rows | 10 |
| resolved_geography_assertions | 8 |
| unresolved_geography_assertions | 2 |
| subjects_with_resolved_geography | 8 |
| subjects_with_unresolved_geography | 2 |
| subjects_with_compatible_indicators | 6 |
| subjects_with_resolved_geography_no_compatible_indicator | 2 |
| subjects_with_any_external_indicator | 6 |
| subjects_with_no_compatible_external_indicator | 4 |
| subjects_with_all_three_layers | 5 |
| subjects_with_geography_and_indicators_only | 1 |
| subjects_with_geography_and_activity_only | 0 |
| subjects_with_activity_only | 2 |
| distinct_indicator_ids | 4 |
| distinct_geography_schemes | 2 |
| distinct_geography_versions | 2 |
| blocked_geography_version_mismatch_count | 1 |
| compatible_indicator_observations / enriched_rows | 11 |
| staging_rows | 4 |
| output_rows | 15 |
| subjects_with_any_activity_evidence | 7 |
| subjects_with_authoritative_or_strong_activity_evidence | 4 |
| subjects_with_contextual_or_weak_activity_evidence_only | 2 |
| subjects_with_multiple_activity_evidence_items | 3 |
| subjects_with_no_activity_evidence | 3 |
| activity_evidence_rows | 11 |
| activity_review_required_rows | 3 |
| activity_ambiguous_rows | 2 |
| exact_geography_join_matches | 11 |
| entity_id_only_join_matches | 0 |
| case_j_multiple_resolved_levels | 0 |

The 15 output records comprise 11 exact-join enriched records plus 4 explicit staging records: 2 unresolved-geography and 2 resolved/no-compatible-indicator.

The subject-level layer combinations are defined as follows:

- `all_three_layers`: resolved geography, at least one exact-compatible external indicator, and activity evidence.
- `geography_and_indicators_only`: resolved geography and at least one exact-compatible indicator, with no activity evidence.
- `geography_and_activity_only`: resolved geography and activity evidence, with no exact-compatible indicator.
- `activity_only`: unresolved geography and activity evidence.
- `no_compatible_external_indicator`: any selected subject without an exact-compatible indicator, including unresolved geography and resolved geography with no compatible observation.

The mismatch count is subject-level: one resolved subject was blocked because only an otherwise matching scheme/level/code observation with a different geography version existed.

## Accepted SKO-027 Case J constraint

Case J (multiple resolved geography levels for one selected subject/location) is not applicable under accepted SKO-027 v1. That contract emits one classification row per selected subject/location and represents multiple distinct supported geography results as one `ambiguous` row, not multiple resolved levels. The fixtures and QA therefore record Case J as zero/not applicable; the orchestration layer does not manufacture it. Revisit only if the owner changes the SKO-027 contract.

## Validation evidence

The focused suite validates the config contract, every output record against the JSON Schema with date-time format checking, deterministic CLI byte equivalence across two runs, shuffled-input equivalence, exact QA counts, activity non-multiplication, scheme/version mismatch staging, blank supplier identity context, subject-key separation, Case J non-manufacture, and an AST source check excluding network/API client dependencies.

Commands required for owner handoff:

```text
python -m py_compile context_integration.py
python -m unittest tests.test_context_integration -v
python -m unittest discover -s tests -v
git diff --check
```

The environment exposes `python3` and the repository virtual-environment interpreter rather than a `python` executable. Final command results and upstream hash verification are recorded in the implementation report delivered to the owner.

## Limitations and owner decisions

- No authoritative tracker workbook or standalone SKO-030 task brief was present in this checkout. The numeric QA contract above is explicit, exhaustive for the requested ten-case cohort, and enforced in tests; the owner should compare it with any external task record during review.
- This layer accepts already-produced local CSVs. It deliberately does not invoke or change upstream producers.
- Output activity summaries preserve evidence metadata only. They are not activity classification, NACE crosswalking, eligibility, publication approval, or impact evidence.
