# E1.5 Operational Validation and Legacy Trusted-Roster Migration Evidence

## Scope

E1.5 operationally validates the E1.4 canonical linkage and governed
safe-list workflow against a real accepted client-review file and prepares
the legacy trusted roster for controlled migration.

This phase does not automatically approve or materialise legacy trusted
terms.

It validates:

- reuse of persistent canonical entity and source-record IDs;
- materialisation of accepted entities, aliases and identifiers;
- governed safe-list generation;
- stable allocation and reuse of a genuinely new accepted entity ID;
- review handling for ambiguous identity candidates;
- persistent-ID output through canonical safe-list matching;
- preservation of legacy trusted-name matcher behaviour;
- complete reconciliation of the legacy trusted roster.

## Repository state

Operational validation was completed on branch:

`e1-5-operational-validation`

Legacy matcher regression fix commit:

`f7171dc Restore legacy trusted-entity matching`

The branch was pushed to:

`origin/e1-5-operational-validation`

## Real accepted client-review validation

### Input

The operational validation used the accepted rows from the AstraZeneca
client-review workbook:

`AZ_client_review_telos-final.xlsx`

Worksheet:

`All_Matches`

Acceptance status mapped into the canonical workflow:

`Confirmed` → `reviewed_confirmed`

Live client data and generated operational outputs remained under ignored
`data/` paths and were not committed to Git.

### Extraction and reconciliation

Confirmed input rows:

- 42 accepted rows
- 40 tax-based rows
- 2 name-only rows
- 0 unresolved rows
- 42 reconciled rows

Relevant canonical source records:

- 42 source-record observations
- 39 unique canonical entities
- 39 unique mapped source-record IDs selected for supplier linkage
- 3 accepted rows associated with identifiers represented by multiple source
  observations but one canonical entity

### Initial canonical store

The operational canonical store was created once at:

`data/canonical/e1-5-operational`

Initial results:

- 42 source records
- 39 canonical entities
- 37 source-derived identifiers
- 39 entity events
- 0 quarantined records
- 0 identifier conflicts
- 0 duplicate entity IDs
- 0 duplicate source-record IDs
- 0 duplicate identifier IDs

Canonical initialisation was not rerun after creation.

## Accepted linkage validation

The 42 accepted client rows were linked against the populated operational
canonical store.

Results:

- 42 input rows
- 42 reused links
- 42 persisted supplier links
- 0 new allocations
- 0 review rows
- 0 ineligible rows
- 39 unique matched entity IDs
- 42 persistent entity IDs present
- 42 persistent source-record IDs present
- 0 duplicate supplier keys

All 42 rows resolved using:

`persisted_source_record`

Linkage QA passed.

## Materialisation validation

Accepted links were materialised into the operational canonical layer.

Results after materialisation:

- 39 canonical entities
- 42 source records
- 47 entity identifiers
- 42 entity aliases
- 39 entity events
- 42 supplier links
- 0 entity relationships
- 0 materialisation review rows

New materialised evidence:

- 42 client aliases
- 10 identifiers
- 0 new canonical entities
- 0 new relationships

QA confirmed:

- no duplicate entity IDs;
- no duplicate source-record IDs;
- no duplicate identifier IDs;
- no duplicate alias IDs;
- no linked entity IDs missing from the entity table;
- no linked source-record IDs missing from the source-record table.

## Rerun stability

The same 42 accepted rows were processed again after materialisation.

Results:

- 42 reused links
- 0 new allocations
- 0 review rows
- entity IDs unchanged
- source-record IDs unchanged
- all 42 rows resolved through `persisted_supplier_link`

Rerun stability QA passed.

## Governed safe-list generation

The governed canonical safe list was generated from the operational store.

Results:

- 39 legal-name terms
- 42 client-variant terms
- 47 identifier terms
- 0 brand terms
- 0 reviewed-alias terms
- 128 total trusted terms

Verification status:

- 76 source-verified terms
- 52 human-verified terms

Review status:

- 89 accepted terms
- 37 machine-resolved terms
- 2 unreviewed terms

QA confirmed:

- 0 blank entity IDs;
- 0 unknown entity IDs;
- 0 duplicate trusted-term rows.

The two unreviewed terms remained governed by their
`approved_for_matching` status and were not treated as automatically
approved solely because they appeared in the generated table.

## Legacy trusted-roster migration preparation

### Input

Cleaned legacy roster:

`trusted_entities_prioritized_clean.csv`

Input structure:

- 308 rows
- 301 `DIRECTORY_V22`
- 7 `SOCIAL_BRANDS`
- all rows marked `published`
- no blank required fields
- no duplicate country and casefolded-name combinations

### Migration result

Preparation-only migration results:

- 308 legacy rows
- 0 unique link candidates
- 308 review rows
- 308 reconciled rows
- 0 ambiguous candidates
- 278 unlinked candidates
- 30 invalid-country rows
- 0 invalid-status rows
- 0 missing-name rows
- 0 other review rows

Invalid-country distribution:

- United Kingdom: 11
- Finland: 5
- Latvia: 4
- Romania: 3
- Estonia: 3
- Poland: 2
- Greece: 1
- Lithuania: 1

No legacy term uniquely linked to the current operational canonical store.

No legacy term was approved, converted into a governed manual term, or
materialised.

The migration result reflects the deliberately demand-driven canonical
store: the store contains only operationally relevant accepted entities,
not the complete normalized register.

## Isolated synthetic operational validation

A complete copy of the populated operational canonical store was created
under an ignored validation directory.

The live operational store was not modified.

### New accepted entity

A synthetic accepted supplier with a previously unseen identifier was
processed.

First run:

- resolution status: `new`
- resolution method: `new_accepted_identifier_entity`
- new entity allocated: yes
- review required: no
- persistent entity ID created

Materialisation:

- 1 new canonical entity
- 1 new identifier
- 1 new client alias
- 0 materialisation review rows

Second run:

- same entity ID reused
- resolution status: `reused`
- resolution method: `persisted_supplier_link`
- no second allocation

Stable new-ID validation passed.

### Ambiguous identity candidate

A controlled reviewed alias was attached to two canonical entities in the
same country within the isolated validation store.

The accepted supplier row using that alias produced:

- resolution status: `review`
- resolution method: `ambiguous_identity_candidate`
- review required: yes
- allocation of new entity: no
- blank matched entity ID
- 2 candidate entity IDs

Ambiguity review validation passed.

## Matcher output validation

### Canonical safe-list route

A canonical safe-list client-variant term was matched through
`match_suppliers_v2.py`.

The output retained:

- `social_enterprise_supplier = YES`
- `match_type = canonical_safe_name_exact`
- persistent `matched_entity_id`
- persistent `matched_source_record_id`
- `canonical_safe_term_type = client_variant`

Canonical persistent-ID matcher QA passed.

### Legacy trusted-name route

Operational validation identified a regression: the legacy
`--trusted-entities` file was loaded and its regex matched, but the matcher
used it only to bypass commercial-form suppression and did not produce a
positive trusted-brand match.

The regression was fixed on the E1.5 branch.

After the fix, a synthetic legacy trusted name produced:

- `social_enterprise_supplier = YES`
- `match_type = known_social_brand`
- `match_score = 100`
- matched legacy brand name retained
- blank `matched_entity_id`
- blank `matched_source_record_id`

This separation is intentional:

- governed canonical safe-list terms provide persistent canonical IDs;
- unlinked legacy trusted names preserve historical matching behaviour but
  do not receive canonical IDs.

A regression test was added:

`tests/test_matcher_legacy_trusted_entities.py`

The test confirms:

- a published legacy trusted brand remains a positive match;
- an unrelated supplier is not whitelisted;
- legacy matching does not assign canonical IDs.

## Test evidence

The E1.5 defect-fix validation suite completed with:

- 43 tests run
- 43 tests passed
- 0 failures
- 0 errors

This included linkage, materialisation, safe-list, canonical matcher,
legacy migration and restored legacy trusted-name behaviour tests.

## Git and data safety

All live client inputs, canonical operational tables, migration outputs and
synthetic validation outputs remained under ignored `data/` paths.

Git safety checks confirmed that these files were excluded by `.gitignore`.

The committed change contains only:

- matcher code restoring the legacy trusted-name route;
- synthetic regression tests.

## Acceptance assessment

Evidence now exists for:

- real accepted-file linkage;
- persistent ID reuse;
- accepted-link materialisation;
- governed safe-list generation;
- migration reconciliation;
- stable new entity-ID allocation;
- ambiguity review handling;
- canonical matcher persistent-ID output;
- preservation of legacy trusted-name behaviour;
- regression testing;
- Git and live-data isolation.

Operational follow-ons remain for the legacy roster because none of its
308 rows currently link to an operational canonical entity.

E1.5 should not be marked accepted until:

- this evidence document has been reviewed;
- the operational runbook has been reviewed;
- the programme tracker has been updated;
- the treatment of the 30 unsupported country labels has been decided;
- the next controlled migration approach for the 278 unlinked rows has been
  agreed.
