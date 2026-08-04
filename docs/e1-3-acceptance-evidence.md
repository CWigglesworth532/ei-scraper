# E1.3 acceptance evidence

**Task:** E1.3 — Implement persisted canonical entity IDs and source-record linkage
**Epic:** Epic 1 — Canonical skopia entity layer
**Evidence date:** 4 August 2026
**Status:** Implemented and tested; awaiting review and acceptance

## 1. Implementation boundary

E1.3 implements an incremental, selection-driven canonical entity layer.

The full normalized register remains the matching universe. Canonical entities and source-record mappings are materialised only when records become relevant through supplier matching, review, overlay processing or an explicit selection.

This task does not alter supplier matching, heuristic detection, policy classification or publication behaviour.

## 2. Files implemented

- `canonical_entity_layer.py`
- `tests/test_canonical_entity_layer.py`
- `tests/fixtures/canonical/source_records.csv`
- `tests/fixtures/canonical/selection.csv`
- `requirements.txt`
- `.gitignore`

## 3. Tables implemented

- `canonical_entities`
- `source_records`
- `entity_identifiers`
- `entity_events`

Only entity-creation events are implemented in E1.3.

## 4. Persisted outputs

- `canonical_entities.parquet`
- `source_records.parquet`
- `entity_identifiers.parquet`
- `entity_events.parquet`
- `schema_manifest.json`
- `canonical_qa.duckdb`

Parquet tables are authoritative. DuckDB provides local relational QA views.

## 5. Selection-driven build counts

- Input fixture rows before selection: 8
- Selected input rows: 3
- Persisted source records: 3
- Canonical entities: 2
- Entity identifiers: 2
- Entity events: 2
- Linked records: 3
- Quarantined records: 0

## 6. Rerun persistence

- Rows equal: `True`
- Source-record IDs stable: `True`
- Canonical entity IDs stable: `True`

## 7. QA results

- Duplicate source-record IDs: 0
- Duplicate canonical entity IDs: 0
- Identifier conflicts: 0
- Entities without source records: 0

## 8. Synthetic sampled evidence

- `sko_src_a5869776-4253-42f0-836b-f7df9eebe0b1`
  - entity: `sko_ent_3be80e76-af68-43bb-8590-8db99b2583a0`
  - name: Example Community Cooperative Ltd
  - country: IE
  - status: linked
  - method: persisted_source_mapping
- `sko_src_9441eaaf-3b59-42f7-8340-b425306c3fbf`
  - entity: `sko_ent_3be80e76-af68-43bb-8590-8db99b2583a0`
  - name: Example Community Co-operative
  - country: IE
  - status: linked
  - method: persisted_source_mapping
- `sko_src_2edd8154-9989-41da-b523-195f1c239197`
  - entity: `sko_ent_b9360db4-ba98-4d49-af82-f1e5c63fcec3`
  - name: Example Foundation
  - country: NL
  - status: linked
  - method: persisted_source_mapping

The two Irish source observations resolve to one canonical entity through a shared country-scoped identifier. The selected Dutch source record resolves to a separate entity.

Committed tests also demonstrate that:

- equal normalized names with different identifiers do not merge;
- French SIRETs sharing a SIREN resolve to one legal entity;
- missing-country records are retained and quarantined;
- shuffled input preserves IDs;
- duplicate continuity keys keep distinct source-record IDs;
- exact duplicate observations retain stable IDs;
- incremental selections preserve previously materialised records;
- identifier rows survive unchanged reruns.

## 9. Unit-test output

    test_build_creates_required_outputs (tests.test_canonical_entity_layer.CanonicalEntityLayerTests.test_build_creates_required_outputs) ... ok
    test_duckdb_qa_views_are_clean (tests.test_canonical_entity_layer.CanonicalEntityLayerTests.test_duckdb_qa_views_are_clean) ... ok
    test_duplicate_continuity_keys_keep_distinct_source_ids (tests.test_canonical_entity_layer.CanonicalEntityLayerTests.test_duplicate_continuity_keys_keep_distinct_source_ids) ... ok
    test_entity_ids_are_opaque_uuid_values (tests.test_canonical_entity_layer.CanonicalEntityLayerTests.test_entity_ids_are_opaque_uuid_values) ... ok
    test_exact_duplicate_observations_keep_stable_ids (tests.test_canonical_entity_layer.CanonicalEntityLayerTests.test_exact_duplicate_observations_keep_stable_ids) ... ok
    test_french_sirets_share_siren_entity (tests.test_canonical_entity_layer.CanonicalEntityLayerTests.test_french_sirets_share_siren_entity) ... ok
    test_identifiers_survive_unchanged_rerun (tests.test_canonical_entity_layer.CanonicalEntityLayerTests.test_identifiers_survive_unchanged_rerun) ... ok
    test_incremental_selections_preserve_prior_records (tests.test_canonical_entity_layer.CanonicalEntityLayerTests.test_incremental_selections_preserve_prior_records) ... ok
    test_missing_country_is_quarantined_and_retained (tests.test_canonical_entity_layer.CanonicalEntityLayerTests.test_missing_country_is_quarantined_and_retained) ... ok
    test_rerun_reuses_source_and_entity_ids (tests.test_canonical_entity_layer.CanonicalEntityLayerTests.test_rerun_reuses_source_and_entity_ids) ... ok
    test_same_identifier_resolves_to_one_entity (tests.test_canonical_entity_layer.CanonicalEntityLayerTests.test_same_identifier_resolves_to_one_entity) ... ok
    test_same_name_different_identifiers_do_not_merge (tests.test_canonical_entity_layer.CanonicalEntityLayerTests.test_same_name_different_identifiers_do_not_merge) ... ok
    test_selection_limits_materialised_records (tests.test_canonical_entity_layer.CanonicalEntityLayerTests.test_selection_limits_materialised_records) ... ok
    test_shuffled_input_preserves_ids (tests.test_canonical_entity_layer.CanonicalEntityLayerTests.test_shuffled_input_preserves_ids) ... ok

    ----------------------------------------------------------------------
    Ran 14 tests in 1.440s

    OK

## 10. Data-safety checks

- All committed fixtures are synthetic.
- No live client supplier rows are included.
- Generated Parquet and DuckDB outputs are ignored by Git.
- `match_suppliers_v2.py` was not modified or staged in this worktree.

## 11. Decisions made

### Incremental materialisation

E1.3 does not perform a full-register canonical backfill. IDs and mappings are created when records become operationally relevant. A future full backfill remains possible without changing the stable-ID convention.

### Identifier-first linkage

A valid, non-conflicted, country-scoped legal-entity identifier is the strongest automatic resolution signal. Name equality alone does not create a canonical merge.

### Singleton fallback

A relevant record without an accepted identifier may become a separate singleton entity rather than being aggressively merged. Important exceptions can be manually reviewed later.

## 12. Deferred work

- verified-overlay migration;
- production selection generation from supplier-match results;
- aliases and evidence tables;
- merge, split and redirect execution;
- supplier-match propagation of `matched_entity_id`;
- classification and directory-readiness tables;
- full-register backfill.

## 13. Acceptance status

Implementation and behaviour-based evidence exist. E1.3 remains awaiting review and Charlie’s acceptance in the programme tracker.
