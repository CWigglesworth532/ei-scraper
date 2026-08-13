# SKO-025A — Governed directory crosswalk and forward-flow acceptance evidence

**Task:** SKO-025A
**Acceptance date:** 13 August 2026
**Status:** Accepted
**Scope:** Offline, proposal/review-only governance; no Airtable creation or write authority

## Owner acceptance

> **SKO-025A is accepted.**
> The governed directory crosswalk and Airtable profile fingerprint layer has been implemented and tested. It supports exact Airtable Record-ID binding, a versioned fingerprint based on Organisation, Country and Website hostname, a six-value relationship vocabulary, separate governed counting mappings, conservative owner-review handling, and compatibility with multiple same-type profiles for one canonical entity.
>
> Real validation against the frozen 321-row Airtable directory correctly produced no approved bindings where accepted canonical evidence was insufficient; known examples including ILUNION, Gureak and AfB were diagnostically checked and did not expose an implementation bug. Existing directory records are not required to be fully canonicalised.
>
> The more important forward-flow capability has been validated synthetically: a verified canonical supplier can enrich an existing directory profile, generate a governed new-directory-profile candidate where no profile exists, or be held out where directory readiness is insufficient. One-to-many profiles preserve a common `counting_entity_id` and do not inflate supplier counts.
>
> No Airtable API, credentials, network capability, write/apply path, publication mutation, canonical merge/split or fabricated Record ID has been introduced. Full regression suite: **158 passed**.
>
> SKO-025A acceptance does not authorise Airtable creation or writes. A future creation/write boundary remains a separate governed task.

## Delivery status and evidence boundary

- **Implemented:** offline crosswalk governance, exact Record-ID binding, fingerprint v1, owner-review candidates, counting governance, SKO-024 compatibility and forward-flow proposals.
- **Tested:** synthetic behavioral, compatibility, safety, determinism and regression suites passed.
- **Real-data reviewed:** the ignored frozen 321-row directory snapshot was evaluated conservatively; private row-level material remained under ignored `data/` paths.
- **Forward-flow validated:** existing-profile enrichment, new-profile proposal and readiness holdout paths were demonstrated synthetically.
- **Accepted:** owner acceptance recorded on 13 August 2026. Acceptance remains no-write.

## Accepted architecture

The active crosswalk binding is the exact Airtable Record ID. Each governed row has an immutable `crosswalk_id`; one Record ID cannot be actively approved for competing canonical identities. Multiple Record IDs, including multiple profiles of the same relationship type, may bind to one canonical entity.

The accepted relationship vocabulary is:

- `group`
- `division`
- `service_line`
- `brand`
- `establishment`
- `operating_unit`

Counting defaults to `entity_id` for an ordinary standalone profile. Any different `counting_entity_id` requires a separate owner-approved aggregation mapping; the Epic 1 canonical schema is unchanged.

Fingerprint `airtable-profile-fingerprint-v1` uses normalized Organisation, ISO2 Country and lowercase Website hostname without leading `www.`. Record ID, integration-owned fields, publication fields and non-identity editorial/client-reference fields are excluded. Approval and current-state fingerprints are compared exactly.

SKO-024 now resolves active crosswalks by exact Record ID while validating canonical entity and relationship consistency. Its approval, hash, allowlist, holdout and no-write handoff semantics remain intact.

## Real-data review

The owner-local frozen directory input contained 321 profiles with unique, non-blank, structurally valid Airtable Record IDs. It remained Git-ignored. Aggregate review outcomes were:

| Outcome | Profiles |
|---|---:|
| Approved bindings | 0 |
| Tier A candidates | 0 |
| Tier B candidates | 0 |
| Ambiguous candidates | 0 |
| Unresolved | 321 |

This is an accepted conservative outcome, not a validation failure. High canonical coverage of the existing directory is not required, and existing unlinked directory suppliers need not be forcibly canonicalised.

The named diagnostic found, structurally:

- **ILUNION:** relevant directory variants and accepted canonical evidence existed with aligned country, but the exact profile binding was not established by the governed evidence rules; no bug was found.
- **Gureak:** directory variants existed, but no corresponding current canonical entity, accepted alias or persisted supplier link was available.
- **AfB:** a directory profile existed, but no corresponding current canonical entity, accepted alias or persisted supplier link was available.

No thresholds or governance rules were weakened to manufacture candidates.

## Synthetic forward-flow validation

The strategic flow is supported:

`client matching → verified supplier → canonical entity/materialisation → crosswalk → enrich existing directory profile OR propose new directory profile`

The bounded proposal layer demonstrated:

| Synthetic outcome | Count |
|---|---:|
| Existing-profile enrichment proposals | 2 |
| New-directory-profile candidates | 2 |
| Readiness holdouts | 1 |
| Additional one-to-many profile | 1 |

Four represented profiles resolved to three counting entities. The additional division shared the group counting entity and did not inflate unique supplier count. New-profile candidates retained canonical identity, counting identity, evidence and batch provenance, had no fabricated Airtable Record ID, remained unpublished and stayed outside the SKO-024 handoff boundary.

## Final validation

| Validation | Result |
|---|---:|
| SKO-025A focused tests | PASS — 17 |
| SKO-022 regression | PASS — 23 |
| SKO-023 regression | PASS — 15 |
| SKO-024 regression | PASS — 13 |
| Full repository suite | PASS — 158 |
| Changed-Python compilation | PASS |
| Deterministic reruns | PASS |
| Safety/no-network/no-write scan | PASS |
| `git diff --check` | PASS |

Hard safety outcomes were zero protected/editorial mutation attempts, zero publication mutation attempts and zero fabricated Record IDs. No Airtable client, API, credentials, webhook, network path, `--write`, `--apply`, canonical merge/split or executable creation path exists.

## Remaining governed boundary

Actual Airtable record creation or mutation is not implemented or authorised. A future separately designed, tested and accepted boundary must govern owner approval of new-profile candidates, Airtable record creation, exact returned Record-ID capture, approval-time fingerprinting, crosswalk activation, concurrency and application outcomes. SKO-025A acceptance does not authorize that future work.