# Epic 1 repository inspection

**Task:** SKO-002 — Run read-only repository inspection  
**Epic:** E1 — Canonical skopia entity layer  
**Inspection date:** 4 August 2026  
**Repository commit inspected:** `c0aa25a`  
**Inspection mode:** Read-only; no repository files changed

## Purpose

Assess the current repository implementation relevant to a canonical skopia entity layer, including:

- normalized register records;
- verified overlay ingestion;
- combined master construction;
- existing entity-ID logic;
- supplier-match linkage;
- deduplication;
- tracked configuration and data-safety controls;
- current uncommitted work that must remain separate.

## Evidence reviewed

The inspection used read-only repository commands to capture:

- current commit and working-tree status;
- tracked-file and ignore-rule summaries;
- Python function inventories;
- entity-ID, overlay-key and deduplication references;
- headers for the normalized master, verified overlay and combined master;
- relevant code sections from:
  - `scrape_ei_registers.py`
  - `ingest_verified_queue.py`
  - `resolve_trusted_entities_v2.py`
  - `match_suppliers_v2.py`
- the existing uncommitted diff in `match_suppliers_v2.py`.

## Findings

### 1. No canonical entity layer is currently implemented

The normalized and combined master datasets are record-level datasets. Their current schemas do not contain a persistent `entity_id`.

Current files inspected:

- `ei_registers_normalized.csv`
- `ei_registers_verified_overlay.csv`
- `ei_registers_master_plus_overlay.csv`

### 2. Existing entity-ID logic is downstream and provisional

`resolve_trusted_entities_v2.py` contains `make_entity_id(...)`, which builds a deterministic hash from:

1. country plus tax identifier;
2. otherwise country plus register name and registration number;
3. otherwise country plus normalized name.

This is useful as a prototype, but it is not written back into the normalized or combined master and therefore is not yet canonical.

The normalized-name fallback is not sufficiently stable for a permanent canonical ID because the ID could change when a name is corrected, normalization changes, or a stronger identifier becomes available.

### 3. Overlay identity is not propagated into the combined master

`ingest_verified_queue.py` creates an auditable `overlay_key` using:

`country | canonical identifier type | normalized identifier`

This key supports overlay validation and upsert behaviour. However, the current 21-column master schema does not retain `overlay_key`, so the explicit overlay link is lost when overlay rows are converted into the combined master.

### 4. Current deduplication is local rather than canonical

Different scripts use different deduplication rules, including:

- entity name plus postcode and city;
- country plus tax ID;
- overlay key;
- temporary combined-master keys;
- matcher indexes for tax IDs and SIRENs.

These rules support individual processing steps but do not establish that multiple source records belong to one canonical entity.

### 5. Combined-master reconciliation may not operate as intended

`build_master_plus_overlay()` only applies its strongest identifier-based deduplication when country, identifier type and identifier value are all available in the master schema.

The inspected master schema includes country and tax ID but no explicit identifier-type field. This indicates that equivalent core and overlay records may be appended rather than reconciled. This should be confirmed during schema design.

### 6. Supplier matches are not linked to a stable entity identifier

`match_suppliers_v2.py` currently returns matched name, register, method, score, country and region. It does not retain:

- canonical entity ID;
- source-record ID;
- matched master tax ID;
- matched registration number.

Supplier results therefore cannot reliably reconnect to the same entity after source ordering, names or master files change.

### 7. The uncommitted matcher change is separate work

The working tree contains an uncommitted change to `match_suppliers_v2.py` of 252 additions and 7 deletions.

The change consolidates technical matches and heuristic-only candidates and creates an explicit candidate output. It aligns with the July 2026 staged-review approach, but it does not implement canonical entity identity and should remain separate from Epic 1 schema work unless deliberately reviewed and accepted.

### 8. Current preferred starting components

Subject to later acceptance decisions:

| Function | Preferred current component |
|---|---|
| Register ingestion | `scrape_ei_registers.py` |
| Verified overlay ingestion | `ingest_verified_queue.py` |
| Supplier matching | `match_suppliers_v2.py` |
| Entity-ID prototype | `resolve_trusted_entities_v2.py::make_entity_id()` |
| Normalized source records | `ei_registers_normalized.csv` |
| Verified overlay audit records | `ei_registers_verified_overlay.csv` |
| Current matching master | `ei_registers_master_plus_overlay.csv` |

Earlier overlay and country-specific ingestion scripts should be treated as legacy or specialist until remaining use is established.

## Risks for Epic 1

- **Identity churn:** IDs derived from mutable names or identifier formatting could change between builds.
- **False merging:** tax roots or identifiers may represent branches, legal units or group relationships requiring explicit policy.
- **Duplicate canonical entities:** core and overlay records may represent the same organisation without reconciliation.
- **Lost provenance:** canonicalisation could discard source-specific evidence unless source records remain separate from entities.
- **Matching regression:** changing master construction may alter duplicate selection and established outputs.
- **Uncommitted-work contamination:** the existing matcher diff could be accidentally included in unrelated Epic 1 commits.

## Conclusion

The repository has normalized source records, verified overlay ingestion, a combined matching master, supplier matching and a provisional entity-ID helper. It does not yet have a canonical entity identifier consistently generated, stored and propagated through those layers.

## Recommendation

Proceed to SKO-003 only after this inspection report is accepted.

The next design task should define canonical entity schema v1, including:

- immutable `entity_id`;
- separate source-record identifiers;
- aliases;
- typed identifiers;
- provenance and evidence links;
- canonical-versus-source field ownership;
- merge and split controls;
- overlay linkage;
- directory-readiness fields;
- migration and regression requirements.

## Acceptance record

- Inspection completed: Yes
- Repository changed during inspection: No
- Report versioned in repository: Pending
- Charlie acceptance: Pending
