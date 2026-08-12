# SKO-022 Directory Integration Prototype — Acceptance Evidence

**Task:** SKO-022
**Branch:** `sko-022-directory-integration-prototype`
**Status:** Accepted
**Accepted by:** Charlie
**Acceptance date:** 12 August 2026
**Mode:** Proposal-only; no live Airtable writes authorised or implemented

## Owner acceptance

> “I accept SKO-022 as implemented and tested, with no live Airtable write capability.”

T01–T20 all passed. The focused suite passed 23 tests and the full regression
suite passed 113 tests before acceptance.

## Scope and fixture provenance

This prototype evaluates canonical directory candidates against four separate
states: identity, classification, directory readiness, and Airtable-owned
publication. An approved crosswalk establishes identity linkage only. It does
not approve readiness or alter publication.

The repository did not contain the earlier external SKO-020 fixture ZIP. The
ten committed CSV fixtures are therefore a repository-native transcription of
the approved synthetic fixture design and T01-T20 contract. They contain no
real client or Airtable rows.

Fixture accounting (data rows, excluding headers):

| Fixture | Rows |
|---|---:|
| `activity_metrics.csv` | 2 |
| `airtable_profiles.csv` | 17 |
| `canonical_entities.csv` | 17 |
| `classifications.csv` | 16 |
| `crosswalks.csv` | 17 |
| `directory_candidates.csv` | 19 |
| `expected_behaviours.csv` | 20 |
| `integration_history.csv` | 2 |
| `lifecycle_events.csv` | 3 |
| `readiness.csv` | 16 |

All active synthetic canonical entities have a non-empty `counting_entity_id`.

## Behavioural results

| ID | Result | Evidence |
|---|---|---|
| T01 | PASS | New candidate is `new_profile_candidate`; no proposal/write generated. |
| T02 | PASS | Approved crosswalk resolves to existing `rec_reuse`. |
| T03 | PASS | Name-only similarity is `possible_duplicate` and review-only. |
| T04 | PASS | Group, division, and service-line profiles share one counting entity. |
| T05 | PASS | Two entities claiming `rec_conflict` create hard conflicts and no proposals. |
| T06 | PASS | Incomplete enrichment evaluates `not_ready`. |
| T07 | PASS | Ineligible classification is neither ready nor publishable. |
| T08 | PASS | Ready profile remains independently unpublished. |
| T09 | PASS | Crosswalk evaluation preserves publication state. |
| T10 | PASS | Publication feedback preserves identity and classification. |
| T11 | PASS | Merge redirects to survivor and blocks pending review. |
| T12 | PASS | Split suspends synchronisation pending reassignment. |
| T13 | PASS | Changed record fingerprint blocks stale proposal. |
| T14 | PASS | Rollback contains integration-owned fields only and preserves editorial name. |
| T15 | PASS | Supplier, spend, and impact aggregate once by `counting_entity_id`. |
| T16 | PASS | All 19 transitions have batch, actor, timestamp, reason, and evidence. |
| T17 | PASS | Proposal schema is exactly the eight accepted fields. |
| T18 | PASS | Mutation attempt raises `ProposalOnlyViolation`. |
| T19 | PASS | Crosswalk approval leaves publication values byte-for-byte unchanged. |
| T20 | PASS | Retired profile history remains; no delete operation is generated. |

## Persistence and idempotency

The local run produced ten proposal rows. Authoritative outputs are Parquet;
DuckDB provides QA tables/view; CSV and JSON are derived review artefacts. All
are under ignored `data/outputs/directory/sko-022-synthetic/`.

Two independent temporary-directory runs produced identical Parquet hashes for
all seven output tables. The persisted manifest SHA-256 after the explicit
rerun was:

```text
e7d81241e444871446a3c9fb9067a8ba8af1f0fc7b4b84355165c9442194cab1
```

## Input preservation

The operational canonical store was read only for checksum comparison and was
not supplied to the prototype. Its aggregate before/after SHA-256 was unchanged:

```text
before  2ddb9a1b65f9c8ce46f32c57559a29d6426be62830358a711e3be759b9e6ab3
after   2ddb9a1b65f9c8ce46f32c57559a29d6426be62830358a711e3be759b9e6ab3
```

No separate live/frozen Airtable snapshot was present in the repository. The
committed synthetic frozen Airtable snapshot was unchanged across the rerun:

```text
before  300a6eed7ef2a922fd63e5fcdf2cd94a61e492e4234afb493f16b06abcb8bc8a
after   300a6eed7ef2a922fd63e5fcdf2cd94a61e492e4234afb493f16b06abcb8bc8a
```

## Protected-field and safety QA

Proposal columns are exactly:

```text
entity_id
profile_relationship_type
counting_entity_id
crosswalk_status
skopia_readiness_status
skopia_readiness_assessed_at
skopia_readiness_reference
integration_batch_id
```

Protected proposal fields: **0**. The DuckDB `proposal_field_qa` view reports
`protected_field_count = 0`. Rollback payloads use the same allowlist.

The implementation imports no HTTP/network client, reads no credentials or
environment secrets, defines no API call or webhook, and has no mutation path.
`request_mutation()` is only a fail-closed guard and always raises a hard error.
Configuration separately disables network, credentials, and mutation.

There is no Airtable, network, credential, or executable mutation capability.
Live Airtable integration, concurrency control, and operational rollback remain
explicitly out of scope and require separate future approval.

## Validation

```text
python -m py_compile directory_integration.py                         PASS
python -m py_compile tests/test_directory_integration.py              PASS
python -m unittest tests.test_directory_integration -v                PASS (23 tests)
python -m unittest discover -s tests -v                               PASS (113 tests)
git diff --check                                                      PASS
git status -sb                                                        reviewed; expected uncommitted SKO-022 files only
```

## Unresolved limitations

1. The earlier external SKO-020 fixture ZIP was unavailable; committed fixtures
   are the repository-native synthetic transcription described above.
2. No real Airtable snapshot, API, schema discovery, authentication, network
   exchange, or live write has been exercised or implemented.
3. Name similarity is deliberately a conservative local review signal, not a
   production entity-resolution algorithm.
4. The prototype models proposals and local rollback fields; it cannot validate
   Airtable concurrency or operational rollback without a separately authorised
   future integration phase.
5. Generated Parquet/DuckDB/review artefacts remain local and ignored.

## Acceptance status

**Accepted by Charlie on 12 August 2026.** Acceptance is limited to the tested
proposal-only prototype and does not authorise live Airtable integration.
