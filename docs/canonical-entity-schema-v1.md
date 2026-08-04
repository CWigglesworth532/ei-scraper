# Canonical entity schema v1

**Task:** SKO-003 — Define canonical entity schema v1  
**Epic:** E1 — Canonical skopia entity layer  
**Milestone:** E1.2  
**Schema version:** 1.0.0  
**Document status:** Accepted  
**Decision date:** 4 August 2026
**Accepted by:** Charlie
**Acceptance date:** 4 August 2026  
**Implementation status:** Not started

## 1. Purpose

This document defines canonical entity schema v1 for skopia.

The canonical entity layer will establish a stable, auditable identity model above the existing record-level normalized and combined master datasets. It will separate:

- canonical entities;
- source records;
- typed identifiers;
- aliases;
- evidence;
- verified overlay records;
- canonical attribute assertions;
- policy classification;
- directory readiness;
- entity merge and split governance.

This document defines the schema and governance model only. It does not implement the schema, migrate data, alter matching logic, or complete milestone E1.2 by itself.

## 2. Authoritative inputs

This schema is based on:

- `Eu Social Economy Matching Master Context_v4_July_2026.pdf`;
- `skopia_development_tracker.xlsx`;
- `docs/AGENTS.md`;
- `docs/epic-1-repository-inspection.md`.

The accepted repository inspection found that:

1. the normalized and combined master datasets are record-level datasets without a persistent canonical `entity_id`;
2. the existing `make_entity_id()` in `resolve_trusted_entities_v2.py` is provisional and downstream;
3. overlay identity is not propagated into the current combined master;
4. current deduplication is local to individual scripts rather than canonical;
5. supplier matches are not linked to a stable entity identifier;
6. the existing unstaged `match_suppliers_v2.py` change is separate work and must not be included in Epic 1 schema commits.

## 3. Core design principles

### 3.1 Canonical entity versus source record

A canonical entity represents the real-world organisation skopia currently believes exists.

A source record represents what one source said about an organisation at a particular time.

Multiple source records may resolve to one canonical entity. A source record must never be overwritten to simulate a canonical correction.

### 3.2 Matching is not classification

The schema must keep separate:

- technical identity matching;
- source evidence;
- policy classification;
- confidence;
- publication status.

A correct identity match does not automatically prove social economy status, client eligibility, or directory readiness.

### 3.3 Stable identity must not depend on mutable attributes

Canonical `entity_id` values must not be generated from:

- names;
- normalized names;
- countries;
- tax identifiers;
- register identifiers;
- source names;
- classifications;
- directory status.

Those attributes may support resolution, but they must not determine the permanent canonical ID.

### 3.4 Source and overlay separation

Official or scraped source records must remain separate from human-reviewed overlay records.

An accepted overlay may create or strengthen:

- an entity link;
- an identifier;
- an alias;
- evidence;
- a classification decision.

It must not overwrite raw source data.

### 3.5 Provenance and reversibility

Every material assertion, merge, split, reassignment and classification decision must be traceable to:

- its source;
- its evidence;
- its processing batch or decision event;
- its reviewer or rule;
- its date;
- its schema and policy version.

### 3.6 Preserve unresolved and rejected records

Unresolved, ambiguous, rejected, excluded and quarantined records must remain internally traceable.

They must not disappear silently during canonicalisation.

## 4. DEC-001 — storage format

### 4.1 Accepted decision

Canonical entity layer v1 will use:

- versioned Parquet tables as the authoritative persisted format;
- DuckDB as the local relational query and QA engine;
- derived CSV files for compatibility, manual review and downstream interchange.

A hosted or server database is deferred beyond entity layer v1.

### 4.2 Rationale

This approach:

- supports multi-million-row typed datasets;
- enables efficient relational joins;
- fits the current local, file-based repository workflow;
- is more robust than CSV as the authoritative store;
- avoids premature operational dependency on a database server;
- permits versioned table-level replacement and reproducible QA.

### 4.3 Proposed storage layout

```text
data/canonical/
    v1/
        canonical_entities.parquet
        source_records.parquet
        entity_identifiers.parquet
        entity_aliases.parquet
        entity_evidence.parquet
        canonical_attribute_assertions.parquet
        overlay_records.parquet
        entity_classifications.parquet
        directory_readiness.parquet
        entity_events.parquet
        entity_redirects.parquet
        classification_evidence.parquet
        directory_readiness_evidence.parquet
        entity_event_evidence.parquet
        schema_manifest.json
```

CSV files generated from these tables are derived products and are not authoritative.

## 5. Stable ID convention

### 5.1 Accepted decision

Canonical entity IDs will use persisted opaque identifiers in the form:

```text
sko_ent_<UUID>
```

Example:

```text
sko_ent_0194d7a8-7a2e-7c21-a923-c43ae829d310
```

### 5.2 Rules

A canonical `entity_id`:

- is allocated once;
- is persisted;
- is never recomputed from mutable entity attributes;
- contains no country, source, name, identifier or classification meaning;
- is never reused;
- remains resolvable after merge, split, deprecation or correction.

### 5.3 Existing provisional implementation

`resolve_trusted_entities_v2.py::make_entity_id()` may be used during migration as a provisional clustering helper or comparison key.

It must not become the canonical ID implementation because it derives IDs from tax identifiers, register identifiers or normalized names.

## 6. Logical data model

The principal tables are:

1. `canonical_entities`
2. `source_records`
3. `entity_identifiers`
4. `entity_aliases`
5. `entity_evidence`
6. `canonical_attribute_assertions`
7. `overlay_records`
8. `entity_classifications`
9. `directory_readiness`
10. `entity_events`
11. `entity_redirects`

Supporting relationship tables are:

- `classification_evidence`
- `directory_readiness_evidence`
- `entity_event_evidence`

## 7. Table definitions

## 7.1 `canonical_entities`

One row per organisation currently represented as a distinct canonical entity.

| Field | Required | Description |
|---|---:|---|
| `entity_id` | Yes | Permanent opaque skopia entity identifier |
| `canonical_name` | Yes | Preferred current display or legal name |
| `canonical_name_norm` | Yes | Current normalized name for search |
| `country` | Yes | Primary registration jurisdiction, ISO2 |
| `entity_status` | Yes | `active`, `inactive`, `dissolved`, `unknown`, `superseded` |
| `record_status` | Yes | `active`, `merged`, `split`, `deprecated`, `quarantined` |
| `legal_form_local` | No | Preferred current local legal form |
| `base_legal_form_family` | No | Normalized legal-form family |
| `primary_identifier_id` | No | Preferred identifier-row reference |
| `primary_source_record_id` | No | Best current source-record reference |
| `identity_confidence` | Yes | Confidence in the canonical grouping |
| `identity_review_status` | Yes | `unreviewed`, `machine_resolved`, `reviewed`, `contested` |
| `created_at` | Yes | Entity creation timestamp |
| `created_by` | Yes | Process or reviewer that created the entity |
| `updated_at` | Yes | Last update timestamp |
| `schema_version` | Yes | Schema version |

### Ownership rule

This table contains selected current canonical values only.

It must not own:

- raw source names;
- raw source addresses;
- source classifications;
- client publication status;
- directory readiness;
- technical supplier match results.

## 7.2 `source_records`

One row per ingested source observation.

| Field | Required | Description |
|---|---:|---|
| `source_record_id` | Yes | Stable or versioned source-observation ID |
| `entity_id` | No | Linked canonical entity, where resolved |
| `source_id` | Yes | Stable source definition ID |
| `source_record_key` | No | Native source record key |
| `ingest_batch_id` | Yes | Retrieval or build batch |
| `source_row_number` | No | Row traceability where no native key exists |
| `entity_name_raw` | Yes | Source name exactly as provided |
| `entity_name_norm` | Yes | Normalized source name |
| `country` | Yes | Source-record country |
| `address_raw` | No | Source address |
| `city` | No | Source city |
| `postcode` | No | Source postcode |
| `legal_form_local` | No | Source legal form |
| `source_status` | No | Source-specific status |
| `valid_from` | No | Source validity start |
| `valid_to` | No | Source validity end |
| `retrieved_at` | Yes | Retrieval timestamp |
| `source_url` | No | Source or record URL |
| `source_page` | No | Page or sub-resource reference |
| `record_fingerprint` | Yes | Change-detection fingerprint |
| `resolution_status` | Yes | See allowed values below |
| `resolution_method` | No | Resolution method |
| `resolution_confidence` | No | Confidence in source-record-to-entity link |
| `resolution_reason` | No | Rationale for resolution state |
| `schema_version` | Yes | Schema version |

Allowed `resolution_status` values:

```text
unresolved
candidate_link
linked
ambiguous
excluded
quarantined
```

Exact normalized-name equality alone must not create an approved canonical merge.

## 7.3 `entity_identifiers`

One row per identifier asserted for an entity.

| Field | Required | Description |
|---|---:|---|
| `identifier_id` | Yes | Internal identifier-row ID |
| `entity_id` | Yes | Owning entity |
| `identifier_type` | Yes | SIREN, SIRET, KBO, VAT, HRB, GnR, VR, KvK, etc. |
| `identifier_value_raw` | Yes | Original representation |
| `identifier_value_normalized` | Yes | Country/type-normalized value |
| `country` | Yes | Identifier jurisdiction |
| `issuing_authority` | No | Issuing registry or authority |
| `identifier_scope` | Yes | Identifier scope |
| `is_primary` | Yes | Preferred identifier flag |
| `verification_status` | Yes | Assertion status |
| `valid_from` | No | Identifier start date |
| `valid_to` | No | Identifier end date |
| `source_record_id` | No | Supporting source record |
| `evidence_id` | No | Supporting evidence |
| `created_at` | Yes | Audit timestamp |
| `schema_version` | Yes | Schema version |

Allowed `identifier_scope` values:

```text
legal_entity
establishment
tax_registration
register_entry
group
internal_source
```

Allowed `verification_status` values:

```text
asserted
source_verified
human_verified
rejected
conflicted
superseded
```

Uniqueness must be assessed by:

```text
country + identifier_type + identifier_value_normalized
```

Identifier values must not be treated as globally unique without type and jurisdiction.

## 7.4 `entity_aliases`

One row per alternative entity name.

| Field | Required | Description |
|---|---:|---|
| `alias_id` | Yes | Alias-row ID |
| `entity_id` | Yes | Owning entity |
| `alias_name` | Yes | Alternative name |
| `alias_name_norm` | Yes | Normalized alias |
| `alias_type` | Yes | Alias category |
| `language` | No | Language code |
| `country` | No | Relevant country |
| `is_preferred` | Yes | Preferred within context |
| `valid_from` | No | Alias validity start |
| `valid_to` | No | Alias validity end |
| `source_record_id` | No | Supporting source record |
| `evidence_id` | No | Supporting evidence |
| `verification_status` | Yes | Assertion status |
| `created_at` | Yes | Audit timestamp |
| `schema_version` | Yes | Schema version |

Allowed `alias_type` values:

```text
legal
former_legal
trading
brand
acronym
source_variant
client_variant
```

Client supplier names must not automatically become aliases. They require an approved resolution or review decision.

## 7.5 `entity_evidence`

Append-oriented evidence ledger.

| Field | Required | Description |
|---|---:|---|
| `evidence_id` | Yes | Evidence-row ID |
| `entity_id` | Yes | Entity concerned |
| `source_record_id` | No | Related source record |
| `evidence_type` | Yes | Evidence category |
| `claim_type` | Yes | Claim supported |
| `claim_value` | No | Structured or textual claim |
| `evidence_strength` | Yes | Strength of evidence |
| `evidence_status` | Yes | Current evidence state |
| `source_name` | Yes | Source name |
| `source_url` | No | Evidence URL |
| `source_date` | No | Date of underlying evidence |
| `retrieved_at` | No | Retrieval date |
| `reviewed_at` | No | Human review date |
| `reviewed_by` | No | Reviewer |
| `notes` | No | Concise rationale |
| `content_hash` | No | Evidence-change fingerprint |
| `schema_version` | Yes | Schema version |

Allowed `evidence_strength` values:

```text
authoritative
strong
supporting
contextual
weak
```

Evidence strength must remain separate from classification confidence.

## 7.6 `canonical_attribute_assertions`

One row per candidate or selected canonical attribute value.

| Field | Required | Description |
|---|---:|---|
| `assertion_id` | Yes | Assertion-row ID |
| `entity_id` | Yes | Entity concerned |
| `attribute_name` | Yes | Canonical field name |
| `attribute_value` | Yes | Asserted value |
| `source_record_id` | No | Supporting source record |
| `evidence_id` | No | Supporting evidence |
| `selection_status` | Yes | Assertion state |
| `valid_from` | No | Validity start |
| `valid_to` | No | Validity end |
| `selected_by` | Yes | Rule or reviewer |
| `selected_at` | Yes | Selection timestamp |
| `schema_version` | Yes | Schema version |

Allowed `selection_status` values:

```text
candidate
selected
rejected
superseded
conflicted
```

This table provides provenance for canonical names, legal forms, country and status values.

## 7.7 `overlay_records`

One row per submitted or accepted verified-overlay record.

| Field | Required | Description |
|---|---:|---|
| `overlay_record_id` | Yes | Permanent overlay-row ID |
| `entity_id` | No | Linked entity after resolution |
| `overlay_key` | Yes | Existing normalized overlay key |
| `overlay_status` | Yes | Overlay workflow state |
| `country` | Yes | Jurisdiction |
| `submitted_name` | Yes | Raw candidate name |
| `verified_name` | No | Reviewed legal name |
| `verified_id_type` | No | Reviewed identifier type |
| `verified_id_normalized` | No | Reviewed identifier value |
| `verified_source` | No | Verification source |
| `verified_url` | No | Verification URL |
| `reviewer` | No | Reviewer |
| `reviewed_at` | No | Review timestamp |
| `review_notes` | No | Review rationale |
| `ingest_source_file` | Yes | Input-file provenance |
| `ingest_batch_id` | Yes | Overlay import batch |
| `accepted_evidence_id` | No | Evidence created on acceptance |
| `supersedes_overlay_record_id` | No | Overlay history link |
| `schema_version` | Yes | Schema version |

Allowed `overlay_status` values:

```text
submitted
accepted
rejected
superseded
conflicted
```

An accepted overlay must not bypass identifiers, evidence or classification governance.

## 7.8 `entity_classifications`

One row per entity classification decision, policy and period.

| Field | Required | Description |
|---|---:|---|
| `classification_id` | Yes | Decision ID |
| `entity_id` | Yes | Classified entity |
| `classification_scheme` | Yes | Policy or scheme name |
| `classification_status` | Yes | Decision status |
| `social_economy_category` | No | Cooperative, charity, work integration, etc. |
| `recognition_type` | No | Legal form, tax designation, official register, overlay |
| `recognition_name` | No | ANBI, ESUS, RUNTS, ZER, etc. |
| `confidence_level` | Yes | Decision confidence |
| `classification_reason` | Yes | Auditable rationale |
| `policy_rule_id` | No | Applied rule reference |
| `client_scope` | No | Null for general skopia policy |
| `is_current` | Yes | Current-decision flag |
| `valid_from` | Yes | Validity start |
| `valid_to` | No | Validity end |
| `decided_by` | Yes | Rule engine or reviewer |
| `decided_at` | Yes | Decision timestamp |
| `schema_version` | Yes | Schema version |

Allowed `classification_status` values:

```text
core
associated
probable
review
excluded
```

Classification must not create, merge or split entities.

## 7.9 `classification_evidence`

Relationship table between classification decisions and evidence.

| Field | Required | Description |
|---|---:|---|
| `classification_id` | Yes | Classification decision |
| `evidence_id` | Yes | Supporting evidence |
| `relationship_type` | Yes | `supports`, `contradicts`, `contextual` |

## 7.10 `directory_readiness`

One row per entity directory assessment and version.

| Field | Required | Description |
|---|---:|---|
| `directory_assessment_id` | Yes | Assessment ID |
| `entity_id` | Yes | Assessed entity |
| `directory_status` | Yes | Readiness state |
| `procurement_relevant` | No | Clear B2B product or service |
| `policy_eligible` | No | Meets directory policy |
| `website_verified` | No | Working official website |
| `profile_sufficient` | No | Sufficient profile information |
| `sector_assigned` | No | Sector taxonomy complete |
| `social_mission_assigned` | No | Mission taxonomy complete |
| `duplicate_check_complete` | No | Publication duplicate check |
| `directory_display_name` | No | User-facing directory name |
| `readiness_reason` | Yes | Inclusion, exclusion or outstanding work |
| `airtable_record_id` | No | Downstream directory reference |
| `assessed_by` | Yes | Reviewer or process |
| `assessed_at` | Yes | Assessment timestamp |
| `next_review_at` | No | Currency-control date |
| `is_current` | Yes | Current assessment flag |
| `schema_version` | Yes | Schema version |

Allowed `directory_status` values:

```text
not_assessed
research_needed
review
ready
published
excluded
needs_update
```

Directory readiness must remain separate from identity and classification.

## 7.11 `directory_readiness_evidence`

Relationship table between directory assessments and evidence.

| Field | Required | Description |
|---|---:|---|
| `directory_assessment_id` | Yes | Directory assessment |
| `evidence_id` | Yes | Supporting evidence |
| `relationship_type` | Yes | `supports`, `contradicts`, `contextual` |

## 7.12 `entity_events`

Governance ledger for identity changes.

| Field | Required | Description |
|---|---:|---|
| `event_id` | Yes | Event ID |
| `event_type` | Yes | Governance event type |
| `subject_entity_id` | Yes | Entity acted upon |
| `target_entity_id` | No | Survivor or new related entity |
| `reason` | Yes | Decision rationale |
| `decision_status` | Yes | Event state |
| `decided_by` | Yes | Reviewer |
| `decided_at` | Yes | Decision timestamp |
| `effective_at` | Yes | Effective date |
| `migration_batch_id` | No | Migration provenance |
| `schema_version` | Yes | Schema version |

Allowed `event_type` values:

```text
create
merge
split
reassign_record
identifier_conflict
deprecate
restore
```

Allowed `decision_status` values:

```text
proposed
approved
rejected
reversed
```

## 7.13 `entity_event_evidence`

Relationship table between governance events and evidence.

| Field | Required | Description |
|---|---:|---|
| `event_id` | Yes | Governance event |
| `evidence_id` | Yes | Supporting evidence |
| `relationship_type` | Yes | `supports`, `contradicts`, `contextual` |

## 7.14 `entity_redirects`

Permanent mapping for retired canonical IDs.

| Field | Required | Description |
|---|---:|---|
| `from_entity_id` | Yes | Retired entity ID |
| `to_entity_id` | Yes | Surviving or successor entity ID |
| `redirect_type` | Yes | `merge`, `split_successor`, `correction` |
| `event_id` | Yes | Governing event |
| `effective_at` | Yes | Effective date |
| `is_current` | Yes | Current redirect flag |

A retired ID must remain resolvable and must never be reused.

## 8. Relationships

```text
canonical_entities
    ├──< source_records
    ├──< entity_identifiers
    ├──< entity_aliases
    ├──< entity_evidence
    ├──< canonical_attribute_assertions
    ├──< overlay_records
    ├──< entity_classifications
    ├──< directory_readiness
    ├──< entity_events
    └──< entity_redirects
```

Key cardinalities:

- one entity may have many source records;
- one source record may link to zero or one current entity;
- one entity may have many identifiers;
- one active identifier should belong to one entity unless explicitly conflicted;
- one entity may have many classifications under different policies;
- one entity may have many historical directory assessments;
- one overlay record may resolve to no more than one entity;
- one retired entity may redirect to one survivor or to governed split successors.

## 9. Canonical field ownership

### 9.1 Source-owned values

The following remain owned by `source_records`:

- original source name;
- original address;
- source legal form;
- source status;
- source registration details;
- source URL;
- retrieval date;
- source-specific evidence.

### 9.2 Canonical values

The following may appear in `canonical_entities` as selected current values:

- canonical name;
- canonical country;
- canonical legal form;
- canonical entity status;
- primary identifier;
- primary source record.

Their provenance must be represented through `canonical_attribute_assertions`.

### 9.3 Decision-owned values

The following must not be treated as source or identity attributes:

- policy classification;
- confidence in classification;
- client eligibility;
- publication status;
- directory readiness.

## 10. Source-record identity and versioning

### 10.1 Native source keys

Where a source provides a reliable native record key:

- retain the native key;
- retain source and batch context;
- use it for source-record continuity.

### 10.2 Sources without native keys

Where no reliable native key exists:

- allocate a `source_record_id`;
- retain source row number and batch;
- use `record_fingerprint` for change detection;
- do not treat the fingerprint as a canonical entity ID.

### 10.3 Repeated observations

Repeated observations may represent:

- unchanged source records;
- updates to a source record;
- separate establishments;
- separate legal entities;
- duplicates within the source.

The canonicalisation process must not assume which case applies without evidence.

## 11. Merge governance

A merge means that two canonical records were found to represent the same legal entity.

Rules:

1. one entity is selected as the survivor;
2. the other entity is marked `record_status = merged`;
3. the retired ID is preserved in `entity_redirects`;
4. source records, identifiers and aliases are reassigned through an approved event;
5. historical evidence and match outputs retain the ID used at the time;
6. conflicting identifiers are not silently combined;
7. every merge requires evidence, rationale, reviewer and timestamp.

Default survivor preference:

1. entity with authoritative verified identifier;
2. entity with accepted overlay or reviewed history;
3. entity referenced by existing reviewed matches;
4. oldest persistent `entity_id`;
5. manual decision where the criteria conflict.

The survivor must not be selected solely because it has the longest or newest name.

## 12. Split governance

A split means that one canonical record incorrectly combined two or more real entities.

Rules:

1. create new entity IDs for the separated entities;
2. mark the original entity as `split` or `deprecated`;
3. explicitly reassign source records and identifiers;
4. do not duplicate identifiers across children without a conflict record;
5. preserve the split event and rationale;
6. retain traceability from historical matches to the original entity;
7. unresolved historical matches return to review.

A split requires stronger review evidence than an ordinary record link because it may affect historical matching and reporting.

## 13. No destructive deletion

Canonical entities should normally be:

- merged;
- split;
- deprecated;
- quarantined;
- restored.

They should not normally be physically deleted.

## 14. Overlay migration rules

The current verified overlay architecture remains the accepted path for reviewed additions and corrections.

Overlay migration must:

- preserve the existing `overlay_key`;
- create a permanent `overlay_record_id`;
- retain source-file and batch provenance;
- resolve to an existing entity or create a reviewed new entity;
- create or confirm identifier records;
- create evidence records;
- report duplicate and identifier conflicts;
- preserve rejected rows;
- produce before/after counts;
- avoid overwriting or orphaning existing verified records.

Required migration evidence:

- audit file;
- rejected-row output;
- duplicate/conflict report;
- before/after row counts;
- sampled accepted records;
- sampled rejected records;
- explicit unresolved items.

## 15. Initial migration approach

## 15.1 Phase 1 — freeze and inventory

Create immutable snapshots and checksums for:

- `ei_registers_normalized.csv`;
- `ei_registers_normalized_headered.csv`, where used;
- `ei_registers_verified_overlay.csv`;
- `ei_registers_master_plus_overlay.csv`;
- current trusted-entity resolutions;
- known local regression outputs approved for comparison.

The existing unstaged `match_suppliers_v2.py` change must remain untouched and outside the schema commit.

## 15.2 Phase 2 — create source records

For every normalized source row:

- allocate `source_record_id`;
- assign stable `source_id`;
- preserve all source fields;
- extract typed identifier candidates;
- retain unresolved records;
- produce reconciliation counts.

Required reconciliation:

```text
input normalized rows
= linked source records
+ unresolved source records
+ explicitly excluded source records
+ quarantined source records
```

No row may disappear silently.

## 15.3 Phase 3 — migrate overlay records

For every overlay row:

- allocate `overlay_record_id`;
- preserve `overlay_key`;
- retain ingest provenance;
- resolve to an existing entity or reviewed new entity;
- create evidence;
- create identifiers;
- report conflicts.

## 15.4 Phase 4 — build candidate entity clusters

Resolution order:

1. authoritative country-scoped legal-entity identifiers;
2. accepted overlay identity;
3. consistent register identifier plus country;
4. exact-name plus corroborating evidence;
5. manual review for ambiguous clusters.

Fuzzy names and heuristics may create candidate links but must not independently force canonical merges.

## 15.5 Phase 5 — allocate permanent IDs

For every accepted cluster:

- allocate one `sko_ent_<UUID>`;
- persist source-record mappings;
- retain unresolved records;
- produce duplicate and identifier-conflict queues;
- reuse prior mappings on rerun.

Migration reruns must consume the existing mapping ledger rather than allocate new IDs from scratch.

## 15.6 Phase 6 — generate compatibility exports

A flattened compatibility export should include at least:

```text
entity_id
source_record_id
entity_name
entity_name_norm
country
tax_id
tax_id_root
ei_registration_number
ei_register_name
source_url
base_legal_form_family
se_recognition_type
se_recognition_name
se_recognition_evidence
record_origin
```

Allowed `record_origin` values must include:

```text
register
verified_overlay
```

Compatibility exports remain derived products.

## 15.7 Phase 7 — downstream migration

Subsequent implementation tasks should:

- propagate `matched_entity_id` into supplier matches;
- propagate `source_record_id` where relevant;
- update trusted-entity resolution;
- migrate overlay ingestion fully;
- create directory-candidate exports;
- test known matching regressions.

These actions are outside SKO-003.

## 16. Required invariants

1. Every active canonical entity has one unique `entity_id`.
2. Every source record has one unique `source_record_id`.
3. A source record links to at most one current canonical entity.
4. Every active identifier belongs to one entity unless marked conflicted.
5. Exact normalized-name equality alone cannot merge entities.
6. Classification cannot create, merge or split entities.
7. Directory readiness cannot alter classification or identity.
8. Overlay acceptance cannot overwrite raw source records.
9. A merged ID remains resolvable and is never reused.
10. A split requires an event and explicit source-record reassignment.
11. Every material assertion has traceable provenance.
12. Supplier matches must eventually retain `matched_entity_id`.
13. Compatibility exports are derived, not authoritative.
14. Unresolved and rejected records remain internally traceable.
15. Canonical IDs must not be regenerated when names, identifiers or normalization rules change.

## 17. Regression and QA requirements

Implementation of this schema must produce behaviour-based evidence.

Depending on the affected component, validation must include:

- before/after row counts;
- source-record reconciliation;
- duplicate and identifier-conflict checks;
- overlay acceptance and rejection counts;
- sampled entity clusters;
- sampled unresolved records;
- merge and split examples;
- known-regression comparisons;
- explicit limitations.

Known failure modes to protect include:

- French placeholder VAT/SIREN values;
- non-French IDs misread as French identifiers;
- cross-border exact-name matching;
- embedded `eG` false positives;
- cooperative-to-association and association-to-cooperative matches;
- generic or single-token fuzzy matches;
- known social brands missed by legal-entity matching;
- heuristic-only candidates disappearing;
- partial multi-sheet Excel loading;
- technically correct charity or ecosystem matches being overclassified.

## 18. Worked examples required before acceptance

The repository document must include or be accompanied by four inspectable examples:

1. two source records resolving to one canonical entity;
2. an accepted overlay resolving to an existing canonical entity;
3. a canonical entity merge with redirect;
4. a canonical entity split or identifier-conflict case.

These examples must use synthetic data only.


### 18.1 Two source records resolving to one canonical entity

Synthetic source records:

| `source_record_id` | Source | Source name | Country | Identifier |
|---|---|---|---|---|
| `sko_src_001` | Synthetic cooperative register | Example Community Cooperative Ltd | IE | `IE_COOP_12345` |
| `sko_src_002` | Synthetic tax-designation register | Example Community Co-operative | IE | `IE_COOP_12345` |

Resolution:

```text
entity_id: sko_ent_11111111-1111-4111-8111-111111111111
resolution_method: authoritative_identifier
identity_review_status: machine_resolved
```

Both source records resolve to the same canonical entity because they share the same country-scoped, legal-entity identifier.

The alternative spelling from `sko_src_002` is retained as a `source_variant` alias. Neither source record is deleted or overwritten.

### 18.2 Accepted overlay resolving to an existing entity

Existing canonical entity:

```text
entity_id: sko_ent_22222222-2222-4222-8222-222222222222
canonical_name: Synthetic Inclusion Enterprise GmbH
country: DE
```

Accepted overlay:

```text
overlay_record_id: sko_ovr_001
overlay_key: DE|DE_HRB|HRB12345
verified_name: Synthetic Inclusion Enterprise gGmbH
verified_id_type: DE_HRB
verified_id_normalized: HRB12345
overlay_status: accepted
```

Resolution outcome:

- the overlay links to the existing entity;
- an `entity_identifiers` record is created or confirmed;
- an `entity_evidence` record is created;
- the verified name may be added as an alias or selected canonical attribute;
- the raw overlay record remains unchanged and auditable.

The overlay does not create a duplicate entity and does not overwrite any scraped source record.

### 18.3 Canonical entity merge and redirect

Before review:

```text
sko_ent_33333333-3333-4333-8333-333333333333
sko_ent_44444444-4444-4444-8444-444444444444
```

Evidence shows that both IDs represent the same Belgian legal entity with enterprise number `0123456789`.

Approved merge:

```text
surviving_entity_id:
sko_ent_33333333-3333-4333-8333-333333333333

retired_entity_id:
sko_ent_44444444-4444-4444-8444-444444444444
```

Required governance records:

- an approved `merge` event;
- supporting evidence links;
- reassignment of current source records, identifiers and aliases;
- `record_status = merged` for the retired entity;
- an `entity_redirects` row from the retired ID to the survivor.

Historical outputs that used the retired ID remain unchanged but can resolve through the redirect.

### 18.4 Canonical entity split and identifier conflict

A provisional canonical entity contains two source records:

| Source record | Name | Identifier |
|---|---|---|
| `sko_src_010` | Synthetic Foundation North | `NL_KVK_10000001` |
| `sko_src_011` | Synthetic Foundation South | `NL_KVK_10000002` |

Further evidence shows that these are separate legal entities rather than branches of one organisation.

Approved split:

```text
original_entity_id:
sko_ent_55555555-5555-4555-8555-555555555555

new_entity_id_1:
sko_ent_66666666-6666-4666-8666-666666666666

new_entity_id_2:
sko_ent_77777777-7777-4777-8777-777777777777
```

Required actions:

- mark the original entity as `split`;
- create an approved `split` event;
- reassign each source record and identifier to the correct new entity;
- retain the original ID for historical traceability;
- return any supplier matches that cannot be assigned confidently to review.

No identifier may be duplicated across the two new entities unless it is explicitly marked conflicted.


## 19. Deferred decisions

The following are intentionally deferred to implementation tasks:

- UUID version and allocation library;
- exact Parquet partitioning strategy;
- DuckDB schema and view definitions;
- physical build orchestration;
- canonical attribute-selection scoring;
- automated merge thresholds;
- supplier-match propagation implementation;
- directory export implementation;
- hosted database or API architecture.

## 20. Acceptance evidence

SKO-003 and milestone E1.2 may be marked complete only when:

- this document is saved as `docs/canonical-entity-schema-v1.md`;
- the document is versioned in the repository;
- DEC-001 is recorded as accepted;
- the stable-ID convention is recorded as accepted;
- keys, relationships and invariants are defined;
- merge and split governance is defined;
- overlay treatment is defined;
- classification and directory readiness are separate from identity;
- migration and compatibility requirements are defined;
- four synthetic worked examples exist;
- the document is committed and pushed;
- Charlie records acceptance in `skopia_development_tracker.xlsx`.

## 21. Change history

| Version | Date | Status | Notes |
|---|---|---|---|
| `1.0.0` | 2026-08-04 | Accepted | Canonical entity schema v1 accepted by Charlie; DEC-001 and stable-ID convention approved |
