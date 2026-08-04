# E1.4 Canonical Linkage and Governed Safe List — Acceptance Evidence

## Scope

E1.4 introduces a demand-driven identity-memory layer for accepted supplier
matches.

It does not create canonical IDs for the complete normalized register.

It provides:

- reuse-before-create canonical identity decisions;
- persistent supplier-to-entity links;
- materialisation of accepted entities, identifiers and aliases;
- separate group and establishment relationships;
- a governed safe-list export;
- controlled manual additions;
- explicit legacy trusted-list migration;
- matcher integration returning persistent entity IDs.

## Implemented components

### `canonical_entity_linkage.py`

Pure decision engine that:

- reuses persisted source-record mappings;
- reuses accepted identifiers;
- reuses reviewed aliases;
- never resolves identity from normalized-name equality alone;
- sends ambiguous or conflicting identity signals to review;
- allocates a new opaque entity ID only where no unresolved candidate remains;
- keeps organisational relationships separate from legal identity.

### `run_canonical_linkage.py`

Persists supplier-to-entity decisions in:

- `supplier_entity_links.parquet`;
- `identity_review_queue.parquet`;
- `identity_review_queue.csv`;
- `canonical_linkage_qa.duckdb`.

Repeated supplier records reuse their persisted entity ID.

### `materialise_canonical_links.py`

Materialises accepted decisions into:

- `canonical_entities.parquet`;
- `entity_identifiers.parquet`;
- `entity_aliases.parquet`;
- `entity_relationships.parquet`;
- `entity_events.parquet`.

Parent, subsidiary, group and establishment relationships do not collapse
separate legal entities.

### `build_canonical_safe_list.py`

Creates:

- authoritative `trusted_match_terms.parquet`;
- derived `trusted_match_terms.csv`;
- `trusted_match_terms_qa.duckdb`.

Only accepted entity-linked legal names, reviewed aliases, client variants,
brands and identifiers are exported.

Unreviewed, rejected, conflicted and superseded terms are excluded.

### Manual additions

`config/manual_trusted_terms.csv` provides a controlled editable route for
manual additions.

`ingest_manual_trusted_terms.py` requires:

- an existing active `entity_id`;
- an allowed term type;
- explicit matching approval;
- reviewer and approval date;
- a reason;
- country;
- non-conflicting identifiers.

Invalid rows enter `manual_trusted_terms_review`.

### Legacy trusted-list migration

`prepare_legacy_trusted_migration.py` preserves every historical trusted-list
row in either:

- `legacy_trusted_link_candidates.csv`; or
- `legacy_trusted_migration_review.csv`.

No legacy name is silently discarded.

The legacy operational roster has not yet been automatically approved or
materialised. Migration remains a reviewed operational step.

### Matcher integration

`match_suppliers_v2.py` now supports:

`--canonical-safe-list trusted_match_terms.csv`

The matcher can resolve:

- exact accepted identifiers;
- exact country-aware legal names and aliases;
- bounded approved brand phrases.

Successful safe-list matches populate:

- `matched_entity_id`;
- `matched_source_record_id`;
- `canonical_safe_term_type`.

Legacy `--trusted-entities` behaviour remains available.

Canonical safe-list integration does not alter:

- fuzzy thresholds;
- name heuristics;
- commercial-form safeguards;
- coop/association vetoes;
- SIREN/SIRET handling;
- candidate consolidation;
- classification;
- publication decisions.

## Behavioural evidence

At the final E1.4 regression run:

`Ran 59 tests in 4.158s — OK`

Evidence includes tests for:

- source-record reuse;
- identifier reuse;
- reviewed-alias reuse;
- name-only review;
- conflicting identity signals;
- stable new IDs across reruns;
- canonical table materialisation;
- cross-client identifier rediscovery;
- cross-client alias rediscovery;
- relationship separation;
- duplicate prevention;
- safe-list approval controls;
- exclusion of conflicted and unreviewed terms;
- manual alias and identifier ingestion;
- rejection of unknown entity IDs;
- complete legacy-row reconciliation;
- country-aware matcher resolution;
- ambiguity rejection;
- bounded brand matching;
- cross-border alias protection;
- persistent IDs in matcher outputs;
- full E1.3 regression compatibility.

## Acceptance boundaries

E1.4 implementation and synthetic behavioural testing are complete.

The following remain operational follow-on work:

- reviewing and migrating the live legacy trusted roster;
- running the workflow against a real accepted client-review file;
- deciding which historical directory entries should be approved as aliases,
  brands or separate canonical entities;
- updating operational run documentation;
- merging and pushing the E1.4 branch after review.

## Acceptance decision

Status: Accepted with operational follow-ons.

Accepted by: Charlie  
Accepted on: 2026-08-04

The implementation and synthetic behavioural evidence are accepted.

The following remain open as operational follow-on work:

1. review and migrate the live legacy trusted roster;
2. run the workflow against a real accepted client-review file;
3. document the operational run sequence;
4. review historical directory entries as aliases, brands or separate entities.
