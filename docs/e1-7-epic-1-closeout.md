# E1.7 — Epic 1 formal acceptance and closeout

**Epic:** E1 — Canonical skopia entity layer  
**Task:** E1.7 — Formal Epic 1 acceptance and closeout  
**Closeout date:** 4 August 2026  
**Closeout status:** Accepted  
**Repository branch reviewed:** `main`  
**Repository commit reviewed:** `c4e5b8d`  
**Owner acceptance:** Accepted by Charlie on 4 August 2026

## 1. Accepted Epic 1 scope

Epic 1 establishes a stable, reusable identity-memory layer for suppliers that are operationally relevant to skopia.

Canonical entities and persistent IDs are created only for matched, verified, reviewed, directory-relevant, reporting-relevant or otherwise explicitly selected entities. The full normalized social economy register remains the matching universe and does not receive a bulk canonical-ID backfill.

The accepted outcome is demand-driven canonical identity that supports:

- stable opaque `entity_id` values;
- reuse-before-create identity decisions;
- cross-client supplier identity reuse;
- persistent supplier-to-entity linkage;
- canonical entities, source records, identifiers, aliases and provenance;
- governed trusted/safe-list generation;
- matcher outputs retaining `matched_entity_id` and `matched_source_record_id`;
- explicit separation of legal entities, establishments and group relationships;
- operational validation against accepted client records;
- controlled reconciliation of the legacy trusted roster;
- a directory-candidate CSV keyed by `entity_id`;
- separation of matching, identity, classification, directory readiness and publication.

Epic 1 does not establish that every technically matched entity is policy-approved, directory-ready or publishable.

## 2. Repository and governance state

The closeout review confirmed:

- `main` is the current branch;
- `HEAD`, `origin/main`, `origin/HEAD` and the E1.6 branch point to `c4e5b8d`;
- the local branch is neither ahead of nor behind `origin/main`;
- the working tree is clean;
- E1.6 is merged into `main`;
- all required Epic 1 evidence and runbook documents are present;
- live client data and generated operational outputs remain outside Git under ignored paths;
- committed tests use synthetic fixtures.

Epic 1 added 31 tracked files or modifications, with approximately 9,840 insertions and 13 deletions across canonical identity, linkage, materialisation, trusted-term governance, matcher integration, directory export, tests and evidence.

## 3. Delivered capability closeout matrix

| Intended capability | Planned | Implemented | Tested | Operationally validated | Accepted before E1.7 | Remaining follow-on |
|---|---|---|---|---|---|---|
| Demand-driven canonical population only | Yes | Yes | Yes | Yes — operational store contained 39 relevant entities rather than the full register | Yes | Full-register backfill remains explicitly out of scope unless separately approved |
| Stable persistent opaque `entity_id` | Yes | Yes | Yes | Yes — stable across accepted-file reruns and isolated new-entity reruns | Yes | Merge/split lifecycle execution has not been operationally exercised |
| Stable `source_record_id` and source linkage | Yes | Yes | Yes | Yes — 42 accepted records retained persistent source-record IDs | Yes | Broader source-family operational coverage |
| Reuse-before-create identity decisions | Yes | Yes | Yes | Yes — 42 accepted rows reused existing links; isolated new entity reused on second run | Yes | Continue governed review of ambiguous candidates |
| Cross-client supplier identity reuse | Yes | Yes | Yes | Partly — reuse behaviour and matcher rediscovery validated; multi-client production history was not separately reported | Yes | Expand evidence as further client datasets are processed |
| Persistent supplier-to-entity linkage | Yes | Yes | Yes | Yes — 42 persisted supplier links, 39 unique entities, no duplicate supplier keys | Yes | Link future activity and spend records to the same entities |
| Canonical entities, identifiers, aliases and provenance | Yes | Yes | Yes | Yes — 39 entities, 47 identifiers, 42 aliases and retained source provenance | Yes | Broader evidence/assertion tables may be added as needed |
| Governed safe-list generation | Yes | Yes | Yes | Yes — 128 entity-linked trusted terms generated with QA | Yes | Review and convert eligible legacy terms through governance |
| Controlled manual trusted-term additions | Yes | Yes | Yes | Synthetic validation only | Yes | Use operationally when approved additions arise |
| Matcher returns `matched_entity_id` and `matched_source_record_id` | Yes | Yes | Yes | Yes — canonical safe-list route retained both IDs | Yes | Broader matcher regression catalogue and CI |
| Legacy trusted-name compatibility | Yes | Yes | Yes | Yes — regression found and fixed; legacy route remains positive without assigning canonical IDs | Yes | Governed migration of eligible legacy terms |
| Legal-entity, establishment and group separation | Yes | Yes | Yes | Not exercised in the 42-row operational run; operational relationship count was zero | Yes as an implemented E1.4 capability | Validate against real relationship cases when encountered |
| Ambiguous/conflicting identity review | Yes | Yes | Yes | Yes — controlled ambiguous alias produced review and no new ID | Yes | Ongoing operational review process |
| Operational validation against accepted client records | Yes | Yes | Yes | Yes — 42 accepted AstraZeneca rows reconciled and rerun stably | Yes | Repeat through normal operations for additional clients |
| Legacy trusted-roster reconciliation | Yes | Yes | Yes | Yes — all 308 rows reconciled into governed outputs | Yes | 278 unlinked rows; 30 unsupported country labels; no terms yet approved/materialised |
| Directory-candidate CSV keyed by `entity_id` | Yes | Yes | Yes | Yes — 39 rows, 39 unique IDs, zero duplicates | Yes | Classification, readiness, enrichment, duplicate comparison and synchronization |
| Separation of identity, matching, classification, readiness and publication | Yes | Yes | Yes | Yes — export retained evidence but left governed classification blank and defaulted readiness to review | Yes | Implement downstream governed classification and directory-readiness persistence |
| No live client data committed | Yes | Yes | Yes | Yes — operational files remained ignored | Yes | Continue as a standing governance control |
| Broader historical regression protection and CI | Recognised | Partly | Current Epic 1 suite passes | No CI evidence | Not an Epic 1 acceptance requirement | Epic 2 |

## 4. Evidence by milestone

### E1.1 — Repository inspection

The read-only inspection established that the repository had record-level normalized data, overlay ingestion, matching and provisional downstream identity logic, but no persistent canonical entity layer propagated into supplier outputs.

**Evidence:** `docs/epic-1-repository-inspection.md`  
**Acceptance:** Accepted and versioned before schema implementation.

### E1.2 — Canonical entity schema

The accepted schema defined:

- opaque stable IDs;
- separate canonical entities and source records;
- typed identifiers and aliases;
- provenance and evidence requirements;
- separation of identity, classification and directory readiness;
- merge/split governance;
- Parquet as authoritative persistence, DuckDB for QA and CSV as derived interchange.

The later accepted scope clarification superseded the schema’s universal full-register migration wording for the current programme. The data model and governance rules remained valid, while population became selection-driven.

**Evidence:** `docs/canonical-entity-schema-v1.md`  
**Acceptance:** Accepted by Charlie on 4 August 2026.

### E1.3 — Persisted canonical IDs

E1.3 implemented selection-driven persistence for canonical entities, source records, identifiers and creation events.

Evidence included:

- 14 passing tests;
- opaque UUID IDs;
- identifier-first resolution;
- stable reruns;
- shuffled-input stability;
- non-merging of equal names with different identifiers;
- French SIRET-to-SIREN legal-entity handling;
- retention and quarantine of incomplete records;
- no full-register backfill.

**Evidence:** `docs/e1-3-acceptance-evidence.md`  
**Acceptance:** Recorded as accepted in the authoritative tracker.

### E1.4 — Cross-client linkage and governed safe list

E1.4 implemented:

- reuse-before-create decisions;
- persisted supplier links;
- canonical materialisation;
- aliases and explicit relationships;
- governed safe-list generation;
- controlled manual additions;
- controlled legacy migration preparation;
- matcher integration with persistent IDs.

The final E1.4 regression run reported 59 passing tests.

**Evidence:** `docs/e1-4-acceptance-evidence.md`  
**Acceptance:** Accepted with operational follow-ons.

### E1.5 — Operational validation and legacy reconciliation

The accepted AstraZeneca review supplied 42 confirmed rows. Results included:

- 42 reconciled accepted rows;
- 42 reused links;
- 39 unique canonical entities;
- stable entity and source-record IDs on rerun;
- 47 identifiers and 42 aliases after materialisation;
- 128 governed trusted terms;
- a successful isolated new-ID allocation and reuse test;
- a successful ambiguity-to-review test;
- canonical matcher output retaining persistent IDs;
- restoration of legacy trusted-name matching;
- all 308 legacy rows reconciled.

No legacy term was automatically approved or materialised.

**Evidence:**  
- `docs/e1-5-operational-validation-evidence.md`  
- `docs/e1-5-operational-runbook.md`

**Acceptance:** Accepted with operational follow-ons. The authoritative tracker records 61 passing tests for the completed milestone.

### E1.6 — Directory-candidate export

E1.6 implemented a deterministic CSV review product keyed by `entity_id`.

Operational QA produced:

- 39 eligible rows;
- 39 unique entity IDs;
- zero duplicate IDs;
- preserved source and match provenance;
- conservative `research_needed` and `review` defaults;
- no fabricated classification, enrichment or spend;
- 67 passing tests across the full suite.

**Evidence:** `docs/e1-6-directory-candidate-export-evidence.md`  
**Acceptance:** Accepted with operational follow-ons.

## 5. Test and operational validation summary

The evidence shows progressive regression coverage:

- E1.3: 14 tests;
- E1.4: 59 tests;
- E1.5 tracker acceptance record: 61 tests;
- E1.6: 67 tests.

The final accepted E1.6 suite includes the earlier canonical identity, linkage, materialisation, safe-list, matcher and legacy migration tests as regressions.

Operational evidence is strongest for:

- stable ID reuse;
- accepted supplier linkage;
- canonical materialisation;
- safe-list generation;
- ambiguous-candidate review;
- canonical matcher ID propagation;
- legacy matcher compatibility;
- complete legacy-roster reconciliation;
- directory-candidate export uniqueness and conservative defaults.

Operational evidence is not yet available for a real parent/subsidiary/group/establishment relationship because the accepted 42-row validation produced zero relationship rows. This is a limitation in exercised evidence, not a missing implementation required to close Epic 1.

## 6. Accepted architectural decisions

1. **Demand-driven population.** Canonical IDs are allocated only when entities become operationally relevant.
2. **No bulk backfill.** The normalized register remains the matching universe.
3. **Opaque persisted IDs.** IDs are not derived from names, countries or business identifiers.
4. **Identifier-first, conservative resolution.** Name equality alone cannot force identity.
5. **Reuse before creation.** Persisted links, accepted identifiers and reviewed aliases are checked before allocating a new ID.
6. **Legal identity and organisational relationships are distinct.** Group, parent, subsidiary and establishment relationships do not collapse separate legal entities.
7. **Source and overlay data remain distinct.** Reviewed assertions do not overwrite raw source observations.
8. **Matching is not classification.** Identity, source evidence, policy classification, readiness and publication remain separate.
9. **Governed safe-list only.** Only approved, entity-linked names, aliases, brands and identifiers enter the canonical matching safe list.
10. **Legacy compatibility without false canonicalisation.** Unlinked legacy trusted names may continue matching but do not receive canonical IDs.
11. **Parquet is authoritative.** DuckDB supports QA; CSVs are derived interchange products.
12. **Airtable remains downstream.** The E1.6 export is a review product, not publication or synchronization.

## 7. Known limitations and classification of remaining work

| Remaining item | Epic 1 acceptance requirement? | Classification | Destination |
|---|---|---|---|
| 278 unlinked legacy trusted-roster rows | No | Operational follow-on | Controlled trusted-term migration backlog; process as entities become relevant |
| 30 unsupported country labels | No | Operational follow-on | Trusted-roster migration/configuration task |
| Governed conversion of eligible legacy terms | No | Operational follow-on | Future canonical safe-list operations |
| Governed entity-classification persistence | No | Downstream capability | Epic 3 directory integration, with policy governance kept separate from identity |
| Directory-readiness persistence | No | Downstream capability | Epic 3 |
| Airtable duplicate comparison using `entity_id` | No | Downstream integration | Epic 3 |
| Airtable/Softr synchronization | No | Downstream integration | Epic 3 |
| Entity-linked client activity and spend | No | Reporting/data capability | Epic 4 |
| Broader historical regression catalogue | No | Regression hardening | Epic 2 |
| Automated CI | No | Regression hardening | Epic 2 |
| Real operational relationship examples | No | Operational follow-on | Validate when parent/subsidiary/group/establishment cases arise |
| Merge, split and redirect execution beyond current creation/linkage workflow | No under accepted Epic 1 outcome | Unplanned future enhancement unless a real correction requires it | Future governed identity-maintenance task |
| Full-register canonical-ID backfill | No; explicitly excluded | Out of scope | Only by a future explicit programme decision |

## 8. Documentation inconsistencies identified during closeout

The authoritative tracker and owner decisions establish the accepted status, but three historical evidence documents contain metadata or wording that should be clarified in this closeout record:

1. `docs/canonical-entity-schema-v1.md` still says **Implementation status: Not started**. This correctly described the schema task when accepted, but it is no longer the current programme implementation status.
2. `docs/e1-3-acceptance-evidence.md` ends with E1.3 awaiting acceptance. The tracker records E1.3 as accepted by Charlie on 4 August 2026.
3. `docs/e1-5-operational-validation-evidence.md` contains pre-acceptance stop conditions and says E1.5 should not yet be accepted. The tracker and subsequent owner decision record that E1.5 was accepted with operational follow-ons.

These documents should not be rewritten as though their original review state never existed. This E1.7 closeout records the later owner decisions and resolves the programme-status interpretation.

## 9. Final acceptance assessment

### Required for Epic 1 acceptance

The following required elements exist:

- accepted demand-driven scope;
- accepted stable-ID convention and canonical schema;
- persisted opaque entity and source-record IDs;
- reuse-before-create linkage;
- persistent supplier links;
- canonical materialisation with identifiers, aliases and provenance;
- governed safe-list generation;
- persistent IDs in matcher outputs;
- relationship separation;
- real accepted-file operational validation;
- controlled legacy-roster reconciliation;
- entity-keyed directory-candidate export;
- regression and behavioural evidence;
- repository and data-safety governance;
- milestone acceptance records through E1.6.

No identified missing item is necessary to implement before Epic 1 can be accepted under its agreed scope.

### Recommendation

**Recommend accepting E1.7 and closing Epic 1 with operational follow-ons.**

Recommended acceptance wording:

> Epic 1 is accepted and closed with operational follow-ons. skopia now has a demand-driven canonical identity layer for operationally relevant suppliers, providing stable persistent entity IDs, reuse-before-create cross-client identity decisions, persistent supplier linkage, canonical identifiers and aliases with provenance, governed safe-list generation, matcher ID propagation, controlled legacy-roster reconciliation and an entity-keyed directory-candidate export. The full normalized register remains the matching universe and has not received a bulk canonical-ID backfill. Remaining work on legacy-term conversion, classification, directory readiness, Airtable integration, spend/activity linkage, broader regression coverage and CI is assigned to operational follow-ons or later epics and is not required for Epic 1 acceptance.

## 10. Owner review and acceptance record

**Owner decision:** Accepted and closed with operational follow-ons  
**Accepted by:** Charlie  
**Acceptance date:** 4 August 2026  

**Accepted wording:**

> Epic 1 is accepted and closed with operational follow-ons. skopia now has a demand-driven canonical identity layer for operationally relevant suppliers, providing stable persistent entity IDs, reuse-before-create cross-client identity decisions, persistent supplier linkage, canonical identifiers and aliases with provenance, governed safe-list generation, matcher ID propagation, controlled legacy-roster reconciliation and an entity-keyed directory-candidate export. The full normalized register remains the matching universe and has not received a bulk canonical-ID backfill. Remaining work on legacy-term conversion, classification, directory readiness, Airtable integration, spend/activity linkage, broader regression coverage and CI is assigned to operational follow-ons or later epics and is not required for Epic 1 acceptance.

E1.7 and Epic 1 are formally accepted and closed.

## 11. Tracker update prepared for post-acceptance use

### Work actually completed

- Confirmed clean and synchronized `main` at commit `c4e5b8d`.
- Confirmed E1.6 is merged into `main`.
- Inspected the Epic 1 implementation change set, governance rules, schema, milestone evidence and operational runbook.
- Compared delivered capability against the accepted demand-driven scope and tracker outcome.
- Produced the Epic 1 capability closeout matrix.
- Classified all identified follow-ons by acceptance relevance and destination.

### Evidence created

- `docs/e1-7-epic-1-closeout.md` draft.
- Consolidated capability, testing, operational-validation and limitation matrix.
- Final acceptance recommendation and proposed owner wording.

### Decisions made

- No remaining item has been identified as necessary to implement before Epic 1 acceptance.
- Relationship handling is accepted as implemented and tested, while real operational relationship validation remains a follow-on.
- Merge/split/redirect execution beyond the delivered workflow is a future governed enhancement, not a blocker under the accepted Epic 1 outcome.
- Follow-ons are allocated to operations, Epic 2, Epic 3 or Epic 4 rather than being implemented during E1.7.

### Status changes

- E1.7: **Complete and accepted**.
- Epic 1: **Complete and closed with operational follow-ons**.
- E1.1–E1.6: no change; remain accepted/complete as recorded.

### Unresolved items

- Optional documentation metadata clarifications for historical evidence files.
- All listed operational and downstream follow-ons.

### Recommended next task

1. Commit and push the accepted E1.7 closeout document.
2. Use the updated tracker as the authoritative programme record.
3. Start Epic 2 with `SKO-004 — Catalogue known matching regressions`.
