# SKO-023 Production-Shaped Synthetic Directory Validation Evidence

**Task:** SKO-023
**Baseline:** `ef4a8ad` — accepted SKO-022 proposal-only prototype
**Status:** Accepted
**Scope:** Entirely synthetic validation; not real Airtable validation
**Acceptance date:** 12 August 2026

## Owner acceptance

> “SKO-023 is accepted. The production-shaped synthetic directory integration validation demonstrates that the accepted SKO-022 proposal-only architecture behaves correctly on a larger mixed synthetic population, including one-to-many directory profiles, counting-entity aggregation, duplicate/conflict handling, stale fingerprint blocking, lifecycle cases, separate identity/classification/readiness/publication gates, zero protected-field changes and zero publication mutations. The implementation remains synthetic and no-write; it does not constitute validation against live or historic Airtable data.”

## Scope and safeguards

SKO-023 validates the accepted SKO-022 engine against a larger deterministic,
production-shaped mixed population. All organisation names, identifiers,
profiles, events, evidence references, and activity metrics are synthetic.

The accepted `directory_integration.py` behaviour and eight-field proposal
allowlist were unchanged. No Airtable or other network client, credentials,
webhook, API call, or executable mutation path was added. Protected editorial
fields and Airtable-owned publication values are inputs for gate validation
only and are never proposed or modified.

## Fixture accounting

| Table | Rows |
|---|---:|
| Canonical entities | 330 |
| Directory candidates | 430 |
| Existing synthetic Airtable-shaped profiles | 400 |
| Approved crosswalk rows | 385 |
| Classifications | 330 |
| Readiness assessments | 330 |
| Activity observations | 430 |
| Lifecycle events | 30 |
| Integration-history rows | 10 |
| Expected-behaviour rows | 430 |

Every active mapped profile/entity has a non-empty `counting_entity_id`.
Twenty name-only possible-duplicate profiles are intentionally unmapped review
records and cannot enter counting or proposal totals.

## Scenario accounting

| Scenario | Candidates |
|---|---:|
| Clean approved existing crosswalk | 280 |
| New profile candidate | 25 |
| Possible duplicate | 20 |
| Identity conflict | 10 |
| Stale fingerprint | 10 |
| Canonical merge | 10 |
| Split pending | 10 |
| Retired profile | 10 |
| Eligible but incomplete readiness | 15 |
| Ineligible classification | 15 |
| Ready but unpublished | 15 |
| Published but not ready | 10 |
| **Total** | **430** |

## Candidate reconciliation

| Engine outcome | Count | Proposal generated |
|---|---:|---:|
| `approved_crosswalk` | 335 | 335 |
| `new_profile_candidate` | 25 | 0 |
| `possible_duplicate_review` | 20 | 0 |
| `identity_conflict` | 10 | 0 |
| `stale_proposal_blocked` | 10 | 0 |
| `merge_review_required` | 10 | 0 |
| `split_suspended` | 10 | 0 |
| `retired_history_retained` | 10 | 0 |
| **Total** | **430** | **335** |

Every input candidate has exactly one decision. All duplicate, conflict, stale,
merge, split, and retirement cases are blocked from proposal generation.
Retired profiles retain ten history rows and generate no deletion operation.

## Relationships and counting QA

| Relationship type | Candidate rows |
|---|---:|
| Group | 330 |
| Division | 80 |
| Service line | 20 |

The fixture contains 100 additional profiles beyond the one-group-profile
baseline: 80 division profiles and 20 service-line profiles. The proposal set
has 235 distinct `counting_entity_id` values. Activity QA has 320 distinct
counting entities after ten approved canonical merge redirects.

Anti-double-counting results:

- aggregated `supplier_count` total: 320, one per counting entity;
- spend is the per-counting-entity maximum of repeated profile observations;
- impact is the per-counting-entity maximum of repeated profile observations;
- grouped engine metrics exactly equal independently computed expected metrics;
- extra division and service-line profiles add no supplier, spend, or impact.

## Four-gate validation

Identity, classification, readiness, and publication remained separate:

- 15 eligible and ready profiles remained unpublished;
- 10 published profiles remained not ready;
- 15 approved-crosswalk profiles were ineligible and therefore not ready;
- 15 approved-crosswalk profiles had incomplete enrichment and were not ready;
- total records blocked at the readiness gate: 40;
- crosswalk approval did not alter readiness or publication;
- readiness did not alter publication;
- protected editorial-field mutations: 0;
- publication-state mutations: 0.

## Proposal and security QA

Proposal columns remained exactly:

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

- proposal column count: 8;
- protected fields in proposals: 0;
- transitions with complete batch, actor, timestamp, reason, and evidence: 430;
- mutation attempts still raise `ProposalOnlyViolation`;
- network capability: absent;
- credential handling: absent;
- Airtable API/webhook capability: absent.

## Determinism

- the committed fixture pack is byte-identical to an independently generated
  pack from `build_directory_integration_production_fixture.py`;
- two independent engine reruns are dataframe-identical;
- proposal serialisation hashes match across independent reruns;
- independently shuffled rows in every input table produce logically identical
  proposals, decisions, references, transitions, metrics, rollback fields, and
  history.

## Validation results

```text
python -m py_compile build_directory_integration_production_fixture.py       PASS
python -m py_compile tests/test_directory_integration_production_shaped.py   PASS
python -m unittest tests.test_directory_integration_production_shaped -v     PASS (15 tests)
python -m unittest tests.test_directory_integration -v                       PASS (23 tests)
python -m unittest discover -s tests -v                                      PASS (128 tests)
git diff --check                                                            PASS
```

## Limitations

1. This is production-shaped synthetic validation, not real Airtable validation.
2. It does not validate Airtable authentication, network behaviour, schema
   discovery, rate limits, concurrency, or remote-write rollback.
3. Synthetic publication and editorial fields model gate separation but do not
   claim parity with a current Airtable base.
4. Name-only duplicate detection remains a conservative review signal, not a
   production identity decision.
5. The validation exercises the accepted proposal-only engine and does not
   authorise a live integration phase.

## Acceptance status

**Accepted by the owner on 12 August 2026.** Acceptance is limited to the
production-shaped synthetic, proposal-only validation described here.
