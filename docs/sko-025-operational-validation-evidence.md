# SKO-025 — Governed read-only operational validation evidence

**Task:** SKO-025
**Accepted baseline:** `fbd98251337904ed5ba13682d0811b217a1d78d5` (SKO-025A)
**Status:** Accepted
**Owner acceptance:** Accepted
**Operational date:** 17 August 2026
**Owner acceptance date:** 17 August 2026

## Owner acceptance

> I accept SKO-025 as governed read-only operational validation against the
> checksum-bound frozen 321-row Airtable directory in its current
> pre-integration state. The run confirmed exact input and Record-ID integrity,
> 0/8 integration-owned fields present, 321 conservatively unresolved profiles,
> zero approved bindings, zero initial proposals, zero handoff-ready batches,
> deterministic output, and no Airtable, network, write, apply,
> publication-mutation or fabricated-ID capability. Approved-binding, counting,
> forward-flow and SKO-024 handoff behavior remains supported by synthetic
> accepted regression evidence rather than exercised by this all-unresolved
> real run.

## Outcome

The accepted SKO-022 → SKO-024 → SKO-025A architecture behaved conservatively
against the frozen real 321-row Airtable directory in its actual pre-integration
state. Validation was local and read-only. No Airtable connection, credential,
network, write, apply, creation, merge, split or publication capability was
introduced.

Safe partial linkage is the governing interpretation. Zero approved bindings is
an operational finding, not evidence that the approved path is broken. No
unresolved Airtable profile was auto-canonicalised and unresolved coverage was
not treated as a validation failure.

## Operationally validated against the real frozen export

| Check | Aggregate result |
|---|---:|
| SHA-256 | `eb76de3c04973502b2008ca98f028c0abbe59cc70e54036c4f73d64366df4cad` |
| Checksum matched accepted baseline | yes |
| Rows | 321 |
| Columns | 10 |
| Exact schema | yes |
| Blank Airtable Record IDs | 0 |
| Duplicate Airtable Record IDs | 0 |
| Structurally invalid Airtable Record IDs | 0 |
| Git-ignored input and output | yes |

The exact columns were `Organisation`, `Country`, `Website`, `Business
Summary`, `Identified Clients`, `Sector`, `Clients Publicly Referenced
(Yes/No)`, `Social Mission`, `Countries served`, and `Airtable Record ID`.
No real row content is included in this evidence.

The eight integration-owned Airtable columns were present **0 of 8**. They have
not yet been deployed. Their absence is the real before-state; blank
before-values were not invented, and absence was not treated as a mismatch.
Any future values at this boundary are initial proposed values, not updates to
existing integration values.

## Real crosswalk and proposal findings

| Outcome | Count |
|---|---:|
| Exact approved existing-profile bindings | 0 |
| Approved crosswalks missing current Record ID | 0 |
| Stale fingerprints | 0 |
| Unresolved/unlinked legacy profiles | 321 |
| Invalid/conflicting bindings | 0 |
| Invalid counting mappings | 0 |
| Unsupported non-default counting mappings | 0 |
| Initial valid integration proposals | 0 |
| Gate/readiness holdouts | 0 |
| Approved-binding no-ops | 0 |
| Total profiles receiving no proposal | 321 |
| Handoff-ready existing-profile batches | 0 |

The governed crosswalk input contained 321 rows, all with status `unresolved`.
Only explicit approved decisions can bind by exact Airtable Record ID and
accepted fingerprint v1. Consequently this real run had no approved binding
from which to generate an initial existing-profile proposal and generated no
handoff structure. The zero handoff-ready count is derived from the empty set
of generated handoff structures, not from fabricated proposal data.

Real approved-profile counting results were therefore zero: profile count,
unique `entity_id`, unique `counting_entity_id`, one-to-many entity count and
all relationship-type counts were 0.

## Synthetic accepted regression boundary

The following were not exercised by the all-unresolved real run and remain
covered by focused synthetic tests and accepted SKO-024/SKO-025A regressions:

- approved exact existing-profile binding and fingerprint-v1 comparison;
- stale fingerprint and invalid/conflicting binding handling;
- default self-counting and explicitly approved non-default counting through
  the accepted canonical `counting_entity_id` mapping;
- blocking an unsupported non-default counting mapping;
- one-to-many profiles without supplier-count inflation;
- review-only new-profile candidates and readiness holdouts;
- direct SKO-024 rejection/holdout of a new-profile candidate with no Record ID;
- SKO-024 proposal, approval and handoff behavior.

A new-profile candidate cannot fabricate an Airtable Record ID. The focused
SKO-025 regression submits a synthetic candidate directly to the accepted
SKO-024 gate and confirms it is held out, non-actionable and not handoff-ready.
No real new-profile candidate was generated in this run.

Counting defaults to `counting_entity_id = entity_id`. A non-default value is
valid only when it equals the accepted canonical entity artifact's governed
`counting_entity_id`; unsupported aggregation is blocked. No aggregation is
inferred from names, brands or relationship type.

## Publication protection limitation

The frozen export does not contain wider Airtable publication controls such as
`Data Status` or `Verified`. SKO-025 does **not** claim a literal before/after
comparison for absent publication fields. Protection is structural: the exact
eight-field allowlist, proposal schema, proposal-only/no-write architecture and
derived QA over generated proposal artifacts.

## Derived hard-invariant QA

| Invariant | Result |
|---|---:|
| Allowlist violations | 0 |
| Protected/editorial mutation attempts | 0 |
| Publication mutation attempts | 0 |
| Fabricated Airtable Record IDs | 0 |
| Handoff-ready existing-profile batches | 0 |
| Operational network calls | 0 |
| Write/apply capability | false |

The first five counts are derived from the generated proposal field set,
candidate artifacts and generated handoff structures. Network/write absence is
an architectural and source-scan result; it is not presented as remote runtime
telemetry because no remote integration exists.

## Determinism and private outputs

Two real-data executions with identical input, canonical counting artifact and
batch metadata produced byte-identical console aggregate JSON and aggregate QA
JSON. Deterministic rerun result: **PASS**.

Private artifacts remain under ignored
`data/outputs/directory/sko-025-operational/`. The approved-binding and initial-
proposal extracts contain headers only because the real run found no approved
binding.

## Leakage interpretation

No real SKO-025 operational row content, Airtable Record IDs, URLs or supplier-
specific data were introduced into tracked files by the SKO-025 change set.
The change-set scan found zero real Record IDs, zero real URLs, zero copied real
rows and zero new supplier-name leakage.

A separate repository-wide exact-name scan found six pre-existing organisation-
name overlaps in tracked public/test material. These files predate SKO-025 and
were neither modified nor removed. The repository-wide scan found zero frozen
Airtable Record IDs and zero frozen URLs in tracked text files.

## Validation results

| Validation | Result |
|---|---:|
| Changed-Python compilation | PASS |
| SKO-025 focused suite | PASS — 9 tests |
| SKO-025A regression | PASS — 17 tests |
| SKO-024 regression | PASS — 13 tests |
| SKO-023 regression | PASS — 15 tests |
| SKO-022 regression | PASS — 23 tests |
| Full repository suite | PASS — 167 tests |
| Deterministic real-data rerun | PASS |
| Input/output ignored-path checks | PASS |
| No-network/no-write capability scan | PASS |
| SKO-025 change-set leakage scan | PASS with the qualified interpretation above |
| Repository-wide exact-name scan | 6 pre-existing overlaps; 0 new SKO-025 leakage |
| Tracked `git diff --check` | PASS |
| Equivalent untracked-file whitespace/error check | PASS |

## Status and owner boundary

SKO-025 is **accepted** within the exact real-versus-synthetic evidence
boundary recorded above. Accepted SKO-022, SKO-023, SKO-024 and SKO-025A
behavior was not changed. Acceptance does not authorize Airtable connection,
record creation, mutation, publication changes, write or apply capability.
