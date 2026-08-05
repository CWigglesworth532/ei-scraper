# Pfizer Canonical Promotion — Acceptance Evidence

## Status

**Accepted with operational follow-ons.**

Owner acceptance was recorded on 2026-08-05 after controlled promotion, post-promotion structural validation and full regression testing.

## Scope

Promote the validated Pfizer canonical working copy into the operational canonical store.

The promotion includes only the owner-reviewed and accepted Pfizer population.

It excludes:

- three held ambiguous identities;
- one rejected prior identity;
- 54 unsupported aggregate-only rows;
- four missing-country rows.

These 62 records remain outside the canonical store.

## Accepted Pfizer population

The accepted population comprises:

- 45 supplier links;
- 6 verified reuses of existing canonical entities;
- 39 new canonical entities;
- 42 newly created aliases;
- 3 pre-existing aliases reused;
- complete alias representation for all 45 accepted supplier names;
- zero unresolved identity-review rows;
- zero materialisation-review rows.

## Verified cross-client entity reuses

Six existing canonical entities were reused:

1. Mapi Research Trust
2. Institut Gustave Roussy
3. Fundacion Casa Del Corazon
4. Amyloidosis Alliance — The Voice of Patients
5. Vrije Universiteit Brussel
6. Jessa Ziekenhuis

Five reuses were supported by accepted identifiers. Fundacion Casa Del Corazon was supported by an accepted reviewed alias.

## Working-copy validation

The isolated Pfizer working copy was created from:

`data/canonical/e1-5-operational`

The validated working store was:

`data/outputs/canonical-intake/pfizer-canonical-working-copy/store`

Validation established:

- 45 Pfizer supplier links;
- 6 existing-entity reuses;
- 39 new entity allocations;
- no duplicate supplier keys;
- no duplicate new entity IDs;
- no resolved links without entity IDs;
- no review-required rows;
- complete alias coverage;
- rerun stability;
- 67 regression tests passed;
- operational canonical store remained unchanged before promotion.

## Promotion controls

Staging timestamp:

`20260805T112121Z`

Swap timestamp:

`20260805T112438Z`

Operational store:

`data/canonical/e1-5-operational`

Validated staged store:

`data/canonical/e1-5-operational-pfizer-staged-20260805T112121Z`

Rollback store:

`data/canonical/e1-5-operational-pre-pfizer-swap-20260805T112438Z`

Independent backup:

`data/canonical/backups/e1-5-operational-before-pfizer-20260805T112121Z`

Promotion evidence:

`data/outputs/canonical-intake/pfizer-promotion-evidence/20260805T112121Z`

Controls completed:

- pre-promotion regression suite: 67 tests passed;
- operational backup checksum comparison: passed;
- staged-copy checksum comparison: passed;
- staged population validation: passed;
- prohibited-record exclusion check: passed;
- controlled directory swap completed;
- promoted-store checksum comparison: passed;
- post-promotion structural QA: passed;
- post-promotion regression suite: 67 tests passed.

## Final operational counts

| Table | Final count |
|---|---:|
| Canonical entities | 125 |
| Supplier links | 139 |
| Entity aliases | 132 |
| Entity identifiers | 47 |
| Source records | 42 |
| Entity events | 125 |
| Entity relationships | 0 |
| Identity review rows | 0 |
| Materialisation review rows | 0 |

## Preserved outside the canonical store

| Decision category | Rows |
|---|---:|
| Held ambiguous identities | 3 |
| Rejected prior identity | 1 |
| Unsupported aggregate-only rows | 54 |
| Blocked missing-country rows | 4 |
| **Total** | **62** |

No excluded row was assigned a canonical entity ID through this promotion.

## Owner acceptance

The owner accepted the Pfizer canonical-store promotion with operational follow-ons:

> I accept the Pfizer canonical-store promotion with operational follow-ons. The validated Pfizer working copy has been promoted into the operational canonical store through a controlled and reversible directory swap. The promoted store matches the accepted working copy, contains 45 Pfizer supplier links, 39 new canonical entities and 42 new aliases, and preserves the six verified cross-client entity reuses. Structural QA and the full 67-test post-promotion regression suite passed. The rollback directory and independent pre-promotion backup may now be retained according to normal operational policy. Follow-ons remain for the three held ambiguous identities, one rejected prior identity, 54 unsupported aggregate-only rows and four missing-country rows.

## Final status

The Pfizer canonical-store promotion is:

**Accepted with operational follow-ons.**

The rollback directory and independent backup may be retained under normal operational retention policy.

## Operational follow-ons

- resolve or reject the three held ambiguous identities;
- retain the rejected prior identity outside the canonical store unless new evidence changes the decision;
- obtain accepted identity evidence for the 54 unsupported aggregate-only rows before any canonical materialisation;
- resolve country evidence for the four missing-country rows;
- update the authoritative skopia development tracker;
- select the next governed supplier population.
