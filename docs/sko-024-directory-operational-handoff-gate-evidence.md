# SKO-024 — Controlled Directory Operational Handoff Gate — Acceptance Evidence

**Task:** SKO-024 — Controlled directory operational handoff gate
**Baseline:** `a042ffe` (accepted SKO-023)
**Status:** Accepted with non-blocking follow-ons
**Owner acceptance date:** 12 August 2026

## Owner acceptance

> **I accept SKO-024 with the identified non-blocking follow-ons.**

Acceptance is limited to the deterministic, synthetic, no-write operational
handoff gate described here. It does not constitute live Airtable validation,
authorise an Airtable write adapter, or authorise Airtable mutation.

## Accepted architecture and behavior

SKO-024 implements the separate hard boundary:

`PROPOSAL → CANDIDATE_BATCH → REVIEWABLE_BATCH → APPROVED_BATCH → HANDOFF_READY → hard boundary → future separately authorised adapter`

There is no `APPLIED` state, Airtable client, credential handling, network
call, mutation function, or apply/write CLI.

The accepted handoff gate:

- uses a format-validated batch ID as the authoritative
  `integration_batch_id`, retaining differing proposal values only as source
  metadata and normalizing reviewed operations to the authoritative ID;
- binds proposals only through unique approved entity/relationship crosswalks
  and unique Airtable-shaped Record IDs, blocking duplicate keys;
- compares approved crosswalk fingerprints with the frozen Airtable-shaped
  current fingerprint;
- validates relationship types against the supplied no-write configuration;
- validates proposed `counting_entity_id` values against the accepted canonical
  entity mapping without creating or reinterpreting canonical identity;
- preserves the accepted eight-field integration allowlist;
- hashes canonical, sorted reviewed items and derives future operations only
  from those hash-covered items;
- requires the approval batch ID and content hash to match the recalculated
  reviewed batch;
- blocks the entire batch when any holdout exists;
- models no-op, partial-failure, retry and idempotency behavior synthetically.

`directory_integration.py` remains unchanged.

## Approval, audit and rollback contract

Approval records contain the batch ID, batch content hash, decision, approver,
ISO-8601 timestamp and optional note. Same-person preparation and approval are
permitted at this no-write stage.

The content hash covers the canonical items from which future operations are
regenerated. Convenience views such as `actionable` are non-authoritative;
mutation or injection into them cannot enter the handoff package. Changes to
canonical items invalidate approval.

SKO-024 retains `reviewed_before_values`, meaning the proposal snapshot reviewed
during batch preparation. It does not claim those values are actual
application-time state. A future separately authorised adapter must recheck the
current fingerprint immediately before application and capture actual immediate
pre-application integration-owned values before mutation.

SKO-024 produces only non-executable `rollback_planning_candidate` packages.
Application-time before-state is present only when explicitly supplied in a
synthetic application outcome. Rollback planning remains integration-field-only
and requires fresh validation and approval.

## Accepted validation results

```text
SKO-024 focused suite                                               PASS (13 tests)
SKO-022 regression suite                                           PASS (23 tests)
SKO-023 production-shaped synthetic regression suite               PASS (15 tests)
Full repository suite                                              PASS (141 tests)
Python compilation                                                 PASS
JSON/schema checks                                                 PASS
git diff --check                                                   PASS
Executable safety scan                                             PASS
```

The tests cover deterministic and shuffled input, order-insensitive proposal
hashes, Record ID binding, stale fingerprints, duplicate keys, approval batch
identity, post-approval injection, canonical counting consistency, strict
holdout blocking, allowlist/protected/publication-field injection, all modeled
application outcomes, and rollback planning with and without explicitly
supplied application-time state.

## Data and capability statement

- No live Airtable data was used.
- No live Airtable validation occurred.
- No live client data, generated private data, credentials, API keys or tokens
  were introduced.
- No Airtable client, network call, webhook or Airtable API URL was introduced.
- No Airtable mutation or publication capability exists.
- All committed test inputs are synthetic.
- The programme tracker was not changed during this closeout.

## Accepted non-blocking follow-ons

1. Bind or recompute remaining manifest/audit metadata at packaging.
2. Require validated configuration provenance rather than permitting
   unvalidated or default configuration.
3. Add direct materially conflicting duplicate-row tests.
4. Add full JSON Schema instance validation.
5. Clean remaining evidence encoding and formatting artifacts.

These follow-ons were explicitly accepted as non-blocking and were not
implemented during this acceptance closeout.

## Remaining scope boundary

SKO-024 remains synthetic and no-write. Actual application-time fingerprint
checking, pre-write state capture, authentication, concurrency, rate limiting,
remote behavior and any Airtable adapter remain outside SKO-024 and require
separate design and authorisation.
