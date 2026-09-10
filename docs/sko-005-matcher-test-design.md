# SKO-005 — Matcher test framework and fixture structure

## Purpose

Establish a small, repeatable regression-test structure for `match_suppliers_v2.py` without changing accepted matcher behaviour.

SKO-005 is a testing/maintenance task. It does not alter matching thresholds, canonical identity logic, legal-form classification policy or accepted SKO-004 heuristic behaviour.

## Framework decision

Use Python's existing `unittest` framework. The repository already uses `unittest` extensively, so introducing another test dependency would add complexity without improving the acceptance outcome.

## Matcher regression convention

All matcher regression test modules should use the filename prefix:

`test_matcher*.py`

This makes the canonical fast matcher gate:

```bash
python3 -m unittest discover -s tests -p 'test_matcher*.py' -v
```

The final repository gate remains:

```bash
python3 -m unittest discover -s tests -v
```

The naming convention is part of the contract: matcher-focused regression modules that must run in the fast gate need the `test_matcher` prefix. `tests/test_matcher_canonical_resolution.py` was renamed from the earlier canonical-safe-list filename so country-aware alias resolution, identifier precedence, ambiguity, brand and approval-state safeguards are included automatically.

## Fixture convention

Committed matcher fixtures live under:

`tests/fixtures/matcher/`

Fixtures must be synthetic. Live client rows must not be committed. When a live diagnostic exposes a regression, preserve the behavioural feature but translate the example into a synthetic fixture before commit.

The initial governed catalogue is:

`tests/fixtures/matcher/expected_cases.csv`

Each row has a stable `case_id`, country, synthetic supplier name and explicit expected candidate outcome. The table is intended to be readable by a reviewer without reading test code.

## Behaviour boundaries

Legal-form heuristic positives are review candidates only. They are not proof that an organisation belongs to the social economy.

Country-scoped aliases must remain country-scoped. Short aliases should be constrained to safe positions where appropriate. Negative regression cases are required where abbreviation broadening creates plausible false-positive risk.

Accepted SKO-004 behaviour is the starting baseline. Future matcher changes should add or update a regression fixture before or alongside implementation.

## Test responsibilities

`tests/test_matcher_legal_forms.py`
: Executes the synthetic legal-form fixture catalogue against `classify_name_candidates()`. Each fixture row is run under `subTest(case_id=...)` so failures identify the exact governed case.

`tests/test_matcher_regression_fixture.py`
: Validates the fixture schema, stable IDs, governed YES/NO values and basic reviewability.

`tests/test_matcher_canonical_resolution.py`
: Covers canonical safe-list resolution behaviour, including country boundaries, exact aliases, identifier precedence, ambiguity, brand matching and exclusion of unapproved/conflicted rows.

Existing matcher-prefixed modules continue to cover legacy trusted-entity behaviour and end-to-end matcher output.

## Acceptance evidence for SKO-005

SKO-005 can be put to owner acceptance when:

1. the fixture convention is committed and documented;
2. a synthetic shared matcher fixture is present;
3. accepted legal-form regression behaviour is represented without committed live-client rows;
4. the canonical `test_matcher*.py` command catches the matcher regression suite including legal-form fixtures and canonical-resolution safeguards;
5. the matcher gate passes;
6. the full repository suite passes; and
7. no matcher behaviour code has changed as part of SKO-005.
