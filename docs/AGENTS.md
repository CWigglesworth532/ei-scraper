# AGENTS.md

## Purpose

This repository contains the skopia social economy data, matching and reporting pipeline. Work in this repository must preserve auditability, protect client data, and keep entity resolution separate from policy classification.

These instructions apply to all automated coding agents and contributors working in this repository. More specific `AGENTS.md` files in subdirectories may add local rules, but must not weaken the requirements below.

## Authoritative programme records

- `skopia_development_tracker.xlsx` is the authoritative programme-status record.
- `Eu Social Economy Matching Master Context_v4_July_2026.pdf` is the current operating and methodology context for EU social economy supplier matching.
- Do not infer completion from discussion, plans, comments, branches, filenames or partially implemented code.
- A task or milestone is complete only when its stated acceptance evidence exists and has been reviewed or accepted by the named owner.
- Distinguish clearly between planned, discussed, implemented, tested and accepted work.

## Current delivery approach

Use a staged client-review process:

1. **Initial discovery and prioritisation** — broad but clearly tiered recall. Retain tax-ID, exact-name, fuzzy-name, known-brand and heuristic-only candidates. Separate Confirmed, Probable and Possible / Review Required candidates.
2. **Materiality-led verification** — prioritise suppliers above the agreed spend threshold, normally EUR 100,000 annual spend, strategically important suppliers, and entities intended for external reporting.
3. **Confirmed reporting and publication** — high precision. Include only verified and policy-approved suppliers in confirmed supplier or spend totals.

Uncertain candidates may appear in an initial client review file when explicitly labelled. They must not inflate confirmed totals.

## Core architectural principles

### Matching is not classification

Keep these concepts separate in code, data models and outputs:

- **technical match** — whether a supplier appears to correspond to a registered entity;
- **source evidence** — what the source or register proves;
- **policy classification** — whether the entity counts for the relevant client use case;
- **confidence** — how reliable the match or classification decision is;
- **publication status** — whether the record is approved for a client-facing total or external claim.

A correct register match does not automatically prove social economy status or client eligibility.

### Preserve source and client traceability

- Preserve original supplier names and source fields unchanged wherever possible.
- Use separate normalized fields for matching.
- Preserve source register, source URL, match method, score, heuristic trigger, confidence tier, review status and rationale.
- Preserve rejected and unresolved records internally, even when they are not shared with clients.
- Do not overwrite raw register or client-source data with reviewed or derived values.

### Country alignment

- Country alignment is required for deterministic and fuzzy name matching unless a specific, documented cross-border rule is being tested.
- Cross-border matching must not be enabled as a broad default.
- Country-specific identifiers must be validated and normalized using country-specific rules.

### Canonical entities and stable IDs

Epic 1 is creating the canonical skopia entity layer. Until its schema is accepted:

- do not introduce a new canonical entity-ID convention without an explicit design note and approval;
- locate and document all existing `entity_id`, overlay-key, trusted-entity-key and deduplication logic before changing any of it;
- prefer stable official identifiers where available;
- do not rely on normalized name alone as a permanent identity key when a stable identifier exists;
- treat name-derived keys as fallback identifiers whose stability risks must be documented;
- do not merge entities solely because their normalized names are equal;
- preserve aliases, source records and provenance separately from the canonical entity record;
- ensure identifier changes can be migrated and audited rather than silently regenerated.

### Source data and verified overlays

- Keep official or scraped source records separate from human-verified overlays.
- Do not edit source data to simulate a verified correction.
- Use the verified overlay architecture for reviewed additions or corrections.
- Prefer `ingest_verified_queue.py` as the canonical overlay-ingestion path unless repository inspection establishes that a newer accepted implementation supersedes it.
- Any overlay migration must produce an audit file, rejected-row output, duplicate/conflict checks and before/after counts.

### Matcher branch

- Prefer `match_suppliers_v2.py` for precision-controlled matching unless repository inspection establishes that a newer accepted matcher supersedes it.
- Do not remove safeguards such as country blocking, meaningful-token requirements, cooperative/association vetoes, placeholder-ID rejection or fuzzy-score caps without tests and an explicit decision record.
- Heuristics are candidate signals, not proof of classification.
- Heuristic-only candidates must remain visible in run outputs or an automatically generated candidate file.

## Data safety and confidentiality

- Never commit live client supplier data, client outputs, credentials, access tokens, personal data or confidential commercial data to a public repository.
- Use synthetic fixtures for committed tests.
- Known client runs may be used only in approved local environments and must remain outside version control.
- Do not copy real client rows into examples, documentation, issue text or test fixtures.
- Check `.gitignore` before creating local outputs.
- When uncertain whether a file contains client or personal data, stop and ask before committing or uploading it.

## Initial repository inspection rule

The first repository inspection under task `SKO-002` is read-only.

During that inspection:

- do not modify, create, delete, rename or format repository files;
- do not install packages or update lock files;
- do not run destructive, migration or scrape commands;
- do not rebuild large datasets;
- do not alter Git state;
- inspect the repository structure, active scripts, configuration, tests, documentation and existing ID logic;
- report contradictions between documentation and code;
- identify which scripts appear current, duplicated, obsolete or unverified;
- identify all data paths that could contain live client information;
- recommend changes, but do not implement them.

## Change requirements after inspection

Every implementation task must state:

- task or issue ID;
- files changed;
- change type: addition, replacement, deletion or full-file rewrite;
- rationale and affected behaviour;
- tests or validation commands run;
- evidence created;
- unresolved risks or follow-up work.

For code patches, provide exact placement instructions when the whole file is not being replaced.

## Testing and acceptance evidence

### Minimum validation

For Python changes, run at least:

```bash
python -m py_compile <changed_file.py>
```

Also run the most relevant unit, fixture or behavioural tests available for the affected logic.

### Behaviour-based evidence

Because code review capacity is limited, acceptance should rely on inspectable behavioural evidence. Depending on the task, evidence should include:

- unit-test results;
- synthetic fixture inputs and expected outputs;
- row-count and match-type comparisons;
- country mismatch checks;
- duplicate and identifier-conflict checks;
- before/after schema or migration reports;
- sampled output rows;
- known-regression results;
- explicit limitations and unresolved failures.

A successful command alone is not acceptance evidence when the output has not been checked.

### Regression protection

Do not accept changes that affect matching, identifiers, normalization, overlays or classification without testing relevant known failure modes, including where applicable:

- French placeholder VAT/SIREN values;
- non-French IDs misread as French identifiers;
- cross-border exact-name matching;
- `eG` false positives such as embedded letter sequences;
- cooperative-to-association and association-to-cooperative fuzzy matches;
- generic or single-token fuzzy matches;
- known social brands missed by legal-entity matching;
- heuristic-only candidates disappearing from outputs;
- multi-sheet Excel files being only partially loaded;
- technically correct charity or ecosystem matches being overclassified.

## Configuration and governance

- Prefer config-driven rules over mutable policy logic hardcoded in Python.
- Keep these governance concepts distinct:
  - commercial fuzzy-match suppression or blacklist;
  - policy denylist;
  - trusted social-brand whitelist;
  - source-semantic classification;
  - country identifier rules;
  - legal-form policy.
- Every governance-list entry must retain a reason and, where relevant, source or review evidence.

## Output and QA requirements

Every material matching run should retain or produce:

- input row and unique-supplier counts;
- candidate and match counts by method;
- country counts;
- heuristic-only count;
- country mismatch count;
- empty matched-entity count;
- blacklist and denylist exclusion counts;
- unresolved or review-required counts;
- run metadata sufficient to reproduce the result.

Client-facing outputs must clearly separate initial-review candidates from confirmed reporting totals.

## Working-session closeout

At the end of each substantive development session, prepare a tracker update separating:

- work actually completed;
- evidence created;
- decisions made;
- status changes;
- unresolved items;
- recommended next task.

Do not update a task or milestone to complete unless the stated acceptance evidence exists. Update `skopia_development_tracker.xlsx` before beginning a new substantive task where relevant.

## Stop conditions

Stop and request a decision before proceeding when:

- the change would establish or alter the canonical entity schema or stable-ID convention;
- the change could expose or commit live client data;
- two scripts appear to be competing canonical implementations;
- a migration could overwrite or orphan verified records;
- expected acceptance evidence cannot be produced;
- the requested action conflicts with the master context or programme tracker;
- source semantics are unclear enough that inclusion could affect confirmed client totals.
