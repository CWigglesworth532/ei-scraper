# SKO-026 — Governed directory review-feedback import evidence

**Task:** SKO-026 — Governed directory review-feedback import and audit trail
**Baseline:** `3a22826b98ee2f40a0f27d733d053befa786e51f` (accepted SKO-025)
**Status:** Implemented and tested; not accepted
**Operationally validated against real directory feedback:** No

## Status boundary

- **Designed:** Yes; owner-approved design supplied for SKO-026.
- **Implemented:** Yes; offline governed import, append-only ledger and derived current state.
- **Tested:** Yes; committed synthetic fixtures and regressions.
- **Operationally validated:** Synthetic only. No approved real feedback file exists in scope.
- **Accepted:** No. Owner acceptance remains required.

SKO-026 and E3.4 are not marked accepted by this evidence.

## Reused architecture and separation

The implementation consumes exact accepted artifacts through a caller-supplied,
read-only context. It reuses SKO-025A exact Airtable Record-ID binding,
`airtable-profile-fingerprint-v1`, the six-value relationship vocabulary, the
accepted canonical `counting_entity_id` authority, forward-flow candidate IDs,
and the SKO-024 blank-Record-ID handoff boundary.

No accepted earlier module was changed. Canonical `entity_events`, SKO-022
lifecycle events and integration rollback history are not used as the directory
feedback ledger. The SKO-026 ledger is directory-decision-specific and
append-only; current state is derived deterministically from it.

## Implemented behavior

The v1 JSON Schema defines immutable feedback IDs and the complete governed
payload. The importer validates exact target IDs, exact crosswalk IDs and
Record-ID bindings, fingerprints, relationship vocabulary, canonical/counting
IDs, reviewer and timezone-aware ISO-8601 timestamp, source hash, prior-state
hash and explicit supersession.

Identical feedback-ID/payload replay is an idempotent duplicate. A changed
payload under an existing ID is rejected. Contradictory current decisions
require `supersedes_feedback_id`; valid supersession retains both ledger entries
and exposes one current decision. Same-batch conflicting decisions are rejected
independently of input order.

Readiness decisions affect only readiness state. New-profile
`approve_future_creation` remains a decision to enter a future separately
governed creation process: it has no Airtable Record ID, is not crosswalk-bound,
is not publishable, and remains SKO-024-ineligible.

## Synthetic behavioral evidence

The committed clean fixture produces:

| Aggregate | Result |
|---|---:|
| Feedback rows received / accepted / rejected | 3 / 3 / 0 |
| Decisions by type | crosswalk 1; readiness 1; new profile 1; relationship 0; counting 0 |
| Approved/rejected crosswalk decisions | 1 |
| Current governed decisions | 3 |
| SKO-024-ineligible new-profile candidates | 1 |
| Protected/editorial mutation attempts | 0 |
| Publication mutation attempts | 0 |
| Network calls | 0 |
| Write capability / apply capability | false / false |
| Deterministic rerun | PASS |
| Shuffled-input logical equivalence | PASS |
| Governed-state SHA-256 | `fdca3902fbf4b0a650cfca14d9fde80511f44d751f983fedcbef69e7526fb173` |

Focused negative and history scenarios additionally demonstrate:

| Required evidence | Result |
|---|---|
| Rejected crosswalk | governed current rejection retained |
| Relationship decisions | approval accepted; unsupported type blocked |
| Counting decisions | default self and approved non-default accepted |
| Invalid counting mappings | unsupported and unapproved non-default blocked |
| Readiness outcomes | hold and research-needed history/supersession tested |
| New-profile decisions | future-creation approval and rejection tested |
| Stale feedback blocks | stale fingerprint, wrong version and prior-hash mismatch blocked |
| Duplicate/conflict blocks | identical replay idempotent; changed ID payload and same-batch conflict blocked |
| Malformed target blocks | unknown target, malformed Record ID and exact-binding mismatch blocked |
| Superseded decisions | two historical entries, exactly one derived current decision |
| Fabricated Record IDs | blocked |
| Idempotent repeated batch | no duplicate historical/current decision |

Rejected rows are returned deterministically with stable reason codes and never
enter the governed ledger. Shuffling input produces logically identical ledger,
current state, rejection output, aggregate evidence and governed-state hash.

## Validation

```text
python -m py_compile directory_integration_feedback.py tests/test_directory_integration_feedback.py  PASS
python -m unittest tests.test_directory_integration_feedback -v                         PASS (15)
python -m unittest tests.test_directory_integration -v                                  PASS (23)
python -m unittest tests.test_directory_integration_production_shaped -v                PASS (15)
python -m unittest tests.test_directory_integration_handoff -v                          PASS (13)
SKO-025A focused regression suites                                                       PASS (17)
python -m unittest tests.test_sko025_operational_validation -v                          PASS (9)
```

Full-suite and final diff/safety validation results are recorded at task
closeout after this document exists.

Effective full discovery (`python -m unittest discover -s tests -v`): **PASS (182 tests)**. The literal command without `-s tests` discovers zero tests under the repository's existing layout and is not presented as validation.

## Safety boundary

The implementation imports only Python standard-library hashing, JSON,
timestamp, regular-expression and typing facilities. It has no HTTP/network
client, Airtable client/API, credentials, webhook, CLI, `--write`, `--apply`,
record-creation, publication-mutation or accepted-artifact mutation path.

## Limitations

1. Validation uses committed synthetic feedback only; no real owner-approved
   directory feedback file was available or imported.
2. The library accepts already-loaded rows and accepted context artifacts; a
   future operational wrapper may govern local file discovery and persistence,
   but must preserve the same no-network/no-apply boundary.
3. Acceptance and any future Airtable creation/write process remain separate
   owner-governed steps.

## Recommendation boundary

Subject to the recorded full-suite and final safety checks passing, SKO-026 is
ready to recommend for owner review and acceptance as a synthetic, offline,
no-write capability. This document does not itself accept SKO-026 or E3.4.
