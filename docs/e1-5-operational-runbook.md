# E1.5 Canonical Linkage and Legacy Migration Runbook

## Purpose

Process accepted supplier matches through the demand-driven canonical entity layer. Create canonical IDs only for accepted, reviewed, verified, directory-relevant or otherwise operationally relevant entities. The normalized register remains the wider matching universe.

## Environment and data safety

Run from the repository root with the Parquet-capable environment:

```bash
cd ~/ei-scraper
source ~/ei-scraper-e1-3/.venv/bin/activate
git status --short --branch
which python
python --version
```

Keep live client files, canonical tables and migration outputs under ignored `data/` paths. Commit only synthetic fixtures and documentation.

## Operational sequence

### 1. Prepare accepted matches

The accepted-match CSV must include supplier key, original and normalized name, country, identifier type/value, acceptance status, client provenance, reviewer details, matched source-record ID where available, and any relationship fields.

Use governed statuses such as `accepted`, `verified`, `reviewed_confirmed`, `directory_approved` or `reporting_approved`. Do not allocate IDs to probable or possible rows.

### 2. Resolve source evidence

Prefer persisted source-record mappings, then accepted identifiers, then reviewed aliases, then explicit relationship evidence. Never resolve identity from normalized-name equality alone.

### 3. Run linkage

```bash
python run_canonical_linkage.py \
  --accepted-matches <accepted_matches.csv> \
  --canonical-dir <canonical_directory>
```

Review `linkage_manifest.json`, `supplier_entity_links.parquet`, `identity_review_queue.parquet` and `canonical_linkage_qa.duckdb`.

Stop for investigation if rows do not reconcile, resolved rows lack IDs, duplicate supplier keys appear, unexpected new allocations occur, or ambiguity is auto-resolved.

### 4. Review exceptions

Rows with `review_required = true` remain unresolved pending evidence. Typical reasons are `name_candidate_only`, `ambiguous_identity_candidate` and `conflicting_identity_signals`. Do not create a new entity merely to bypass ambiguity.

### 5. Materialise accepted links

```bash
python materialise_canonical_links.py \
  --canonical-dir <canonical_directory>
```

Check canonical entities, identifiers, aliases, relationships and materialisation review outputs. Confirm no duplicate IDs, no missing linked IDs and no unsupported materialisation.

### 6. Prove rerun stability

Rerun the same accepted input. Existing links should resolve through `persisted_supplier_link`; entity and source-record IDs must remain unchanged; no duplicate allocation should occur.

### 7. Build the governed safe list

```bash
python build_canonical_safe_list.py \
  --canonical-dir <canonical_directory>
```

Use only terms with appropriate verification, review and `approved_for_matching` states. Every governed term must remain linked to a canonical entity.

### 8. Run matching

Use the canonical route with:

```text
--canonical-safe-list <trusted_match_terms.csv>
```

Canonical matches should retain `matched_entity_id`, `matched_source_record_id` and `canonical_safe_term_type`.

Legacy compatibility remains available through:

```text
--trusted-entities <legacy_trusted_file.csv>
```

Legacy trusted names use `match_type = known_social_brand` and must not receive canonical IDs until governed migration is complete.

### 9. Prepare legacy migration

```bash
python prepare_legacy_trusted_migration.py \
  --legacy trusted_entities_prioritized_clean.csv \
  --canonical-dir <canonical_directory> \
  --output-dir <migration_output_directory>
```

Reconcile unique link candidates plus review rows to the total legacy rows. A unique name candidate is not automatic approval. Convert terms only after entity identity, country, term type, evidence, status and matching approval are reviewed.

## Required evidence

Retain input and accepted counts; reused, new, review and ineligible counts; reconciliation totals; canonical table counts before and after; duplicate and missing-ID QA; safe-list counts; migration outcomes; test output; branch and commit; and confirmation that live files remained ignored.

## Current E1.5 result

The AZ validation produced 42 accepted rows, 42 reused links, 39 unique entities, stable IDs on rerun and 128 governed safe-list terms.

The cleaned legacy roster produced 308 rows: 278 unlinked, 30 invalid-country, 0 unique links, 0 ambiguous and 0 invalid-status. No legacy terms were approved or materialised.

## Stop conditions

Stop when canonical initialisation would be rerun, live files appear in Git, a name-only candidate is treated as confirmed identity, identifiers conflict, IDs change on rerun, legacy terms receive IDs without governed migration, totals fail to reconcile, or acceptance evidence is missing.
