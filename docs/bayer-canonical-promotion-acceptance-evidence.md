# Bayer Canonical Promotion — Acceptance Evidence

## Status

**Implemented and tested; awaiting explicit owner acceptance.**

This document does not record owner acceptance and does not mark the task
complete.

## Scope

Promote the validated Bayer canonical working copy into the operational
canonical store using a controlled, reversible process.

The promotion excludes the unresolved generic Fundación record and the two
unsupported aggregate-only records.

## Repository state

- Branch: `ops-bayer-canonical-ingestion`
- Commit at evidence creation: `d9df2e7`
- Evidence created: `2026-08-05T10:40:44.911324+00:00`
- Operational store: `data/canonical/e1-5-operational`
- Validated working store: `data/canonical/bayer-linkage-working`

## Pre-promotion controls

- Full regression suite before promotion: **67 tests passed**
- Operational-store backup: `data/canonical/backups/e1-5-operational-before-bayer-20260805T103638Z`
- Backup checksum comparison: **passed**
- Operational source store before promotion: **39 canonical entities and 42 supplier links**

## Preserved unresolved records

Preservation directory:

`data/outputs/canonical-intake/bayer-promotion-evidence/20260805T103638Z`

Exactly three records were excluded from canonical promotion:

- `SB00511` — generic `FUNDACION PARA LA INVESTIGACION`; held for source-record reconstruction;
- `SB00510` — `FUNDACION PARA INVESTIGACION`; held pending accepted evidence;
- `SB00516` — `FUNDACION SANT JOAN DE DEU`; held pending accepted evidence.

Control results:

- blocked rows: `3`
- canonical entity IDs present on blocked rows: `0`
- promotion decision: `excluded_from_canonical_promotion`

## Validated Bayer manifest

- Bayer supplier links: `52`
- Unique Bayer supplier-record keys: `52`
- Unique Bayer entity IDs: `50`
- New canonical entities materialised: `47`
- New supplier links materialised: `52`
- New aliases materialised: `48`
- Review-required linkage rows: `0`
- Working-copy validation passed: `True`
- Acceptance failures: `{}`

## Promotion process

The validated working copy was copied to a staged directory, checksum-verified,
and promoted by a controlled directory swap.

Rollback directory retained pending owner acceptance:

`data/canonical/e1-5-operational-pre-swap-20260805T103858Z`

The promoted operational store matches the validated Bayer working copy by
SHA-256 checksum.

## Before and after counts

| Table | Before | After | Change |
|---|---:|---:|---:|
| Canonical entities | 39 | 86 | +47 |
| Supplier links | 42 | 94 | +52 |
| Entity aliases | 42 | 90 | +48 |
| Entity identifiers | 47 | 47 | +0 |
| Source records | 42 | 42 | +0 |
| Entity relationships | 0 | 0 | +0 |
| Materialisation review | 0 | 0 | +0 |
| Identity review queue | 0 | 0 | +0 |

## Post-promotion QA

- Promoted-store counts: **passed**
- Working-to-promoted checksum comparison: **passed**
- Prohibited blocked-record absence check: **passed**
- Full post-promotion regression: **67 tests passed**
- Post-promotion regression exit status: **0**

## Acceptance position

The controlled promotion has been implemented and tested.

The task remains **awaiting explicit owner acceptance**. The rollback directory
and timestamped backup must be retained until acceptance is recorded.

## Operational follow-ons

- Resolve or reject the generic Fundación record after reconstructing the three
  underlying Bayer source records.
- Obtain separate accepted evidence before considering the two aggregate-only
  suppliers for canonical materialisation.
- Update the authoritative skopia development tracker after owner acceptance.
