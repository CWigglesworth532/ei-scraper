# SKO-029 governed activity/industry evidence capture

## Status and authoritative context

SKO-029 / E4.4 is **implemented and tested, but not accepted or complete**.
Owner acceptance remains a separate step.

Implementation followed the read-only programme records supplied on 18 August
2026: `skopia_development_tracker_2026-08-18_E4_pilot_led_scope.xlsx` and the
owner-designated July 2026 matching context
`Eu_Social_Economy_Matching_Master_Context_v5_July_2026.pdf`. The tracker records
DEC-022: retain selected-only typed activity evidence, make NACE optional
downstream, and judge the work by later pilot usefulness rather than taxonomy
coverage. The matching context preserves staged/materiality-led client review
and separation of matching, evidence, classification, and publication.

The passed preflight was branch `sko-022-directory-integration-prototype`, HEAD
`ffc997cff093966209a68eb43909188405c0eb97`, a clean worktree, and upstream
divergence `0 ahead / 0 behind`.

## Work implemented

- Added a local-file-only CLI and library that joins small configured evidence
  files to explicitly selected `canonical_entity` and `supplier_observation`
  subjects.
- Made `subject_type + subject_id` the fundamental reference. An existing
  `entity_id` passes through from selection only; canonical subjects require one,
  while supplier observations may leave it blank.
- Preserved source code and description text without padding, truncation,
  translation, version inference, level inference, or principal-activity
  selection.
- Added config-driven subject types, evidence types, schemes, principal-value
  semantics, authority/status rules, source metadata, and source field maps.
- Added deterministic evidence IDs, fingerprints, ordering, exact-duplicate
  handling, source-file SHA-256 provenance, and explicit timestamps.
- Preserved materially conflicting evidence as separate rows and marked each
  conflicting row `ambiguous` rather than selecting a winner.
- Added deterministic pilot QA output covering evidence coverage, strength,
  multiplicity, gaps, review requirements, and ambiguity.

Files added are `activity_evidence.py`, `config/activity_evidence.yaml`,
`schemas/activity_evidence.schema.json`, `tests/test_activity_evidence.py`, four
synthetic CSV fixtures under `tests/fixtures/activity_evidence/`, and this
evidence note. `.gitignore` has only the narrow exception needed to retain those
synthetic CSVs. Accepted SKO-027 and SKO-028 paths are unchanged.

## Evidence contract

The exact ordered row contract is:

1. `activity_evidence_id`
2. `subject_type`
3. `subject_id`
4. `entity_id`
5. `evidence_type`
6. `source_name`
7. `source_record_id`
8. `source_reference`
9. `source_code_field`
10. `source_description_field`
11. `activity_scheme`
12. `activity_version`
13. `activity_code_raw`
14. `activity_description_raw`
15. `is_principal_activity`
16. `evidence_authority`
17. `evidence_status`
18. `source_version`
19. `source_file_sha256`
20. `retrieved_at`
21. `extracted_at`
22. `evidence_fingerprint`
23. `schema_version`

Supported evidence types are `official_activity_code`,
`official_activity_description`, `official_company_description`,
`procurement_category`, `directory_sector`, `directory_business_summary`, and
`other_activity_text`. The synthetic proof exercises all except
`other_activity_text`; it remains governed for later explicitly configured local
sources.

Authority is separate from status. Allowed authority values are
`authoritative`, `strong`, `supporting`, `contextual`, and `weak`. Allowed status
values are `usable`, `review_required`, `ambiguous`, `insufficient_evidence`, and
`unsupported`. The proof config treats official register evidence as
authoritative/strong, directory evidence as contextual/weak, and procurement
category as contextual. A structured CCAE code with no explicit version remains
authoritative source evidence but is `review_required`; no version is inferred.

Principal activity uses `true`, `false`, or `unknown`. Only configured explicit
source values produce true or false. A single code never implies principal
status.

## Synthetic pilot QA

The fixture contains seven selected subjects plus one deliberately unselected
subject. It covers known-version CNAE code/description evidence, version-unresolved
CCAE evidence, official description-only evidence, official company description,
directory Sector and Business Summary, procurement category, no evidence, and a
material official-code conflict.

Observed deterministic QA:

```text
selected_subjects                                      7
subjects_with_any_activity_evidence                    6
subjects_with_authoritative_or_strong_evidence         4
subjects_with_contextual_or_weak_evidence_only         2
subjects_with_multiple_evidence_items                  3
subjects_with_no_activity_evidence                     1
review_required                                        3 evidence rows
ambiguous                                              2 evidence rows
```

The evidence CSV contains nine evidence rows plus its header.

## Validation evidence

Observed on 18 August 2026:

- `.venv/bin/python -m py_compile activity_evidence.py` passed.
- Focused suite passed 28/28 tests.
- Full discovery passed 256/256 tests, against the accepted 228-test baseline.
- The production config loader passed with schema version
  `sko-029-activity-evidence-v1` and three configured synthetic sources.
- JSON Schema Draft 2020-12 validation passed for all nine CLI output rows using
  the locally available system validator.
- `git diff --check` passed.
- Path-scoped diffs confirmed accepted SKO-027 and SKO-028 files are unchanged.
- Two CLI runs produced byte-identical evidence and QA files.
- Evidence CSV SHA-256:
  `d47c422c3cd1f018bdd28eafdc1b5888811490425cde4c654297504dd7906447`.
- QA JSON SHA-256:
  `2e83f89bc75b751a8493a6d80e6c30fd14841d8faddf5ed82ff7ca0ad34b15c3`.

## Deliberately excluded

- No deterministic NACE classification, CNAE/CCAE crosswalk, NACE version
  conversion, text classifier, fuzzy classifier, or principal-activity inference.
- No full-register scan, normalized-register rebuild, parser expansion, live
  scrape, API, or other network path.
- No canonical entity creation/mutation, identity change, social-economy
  classification, readiness/publication change, or directory write.
- No geography field, geography inference, or change to accepted SKO-027/028.
- No client spend, reporting/UI, impact value, supplier-impact assertion, or
  causal claim.
- No real register row, client row, personal data, or confidential data in Git.

## Operational follow-ons and owner decision

Later pilot operation still requires owner-approved local evidence extracts and
source-specific config outside Git, with explicit confirmation of each source's
field semantics, scheme, version, principal marker, release/version, reference,
and retrieval time. Sources whose semantics are not explicit should not be
configured. Any downstream NACE assertion remains a separate future capability.

No stop condition was approached: implementation did not require canonical
changes, population-wide enrichment, real data, guessed source semantics, a
crosswalk, network access, or reporting/impact logic. The behavioural evidence
supports owner review of SKO-029 as ready for acceptance, but this document does
not itself record acceptance or completion.
