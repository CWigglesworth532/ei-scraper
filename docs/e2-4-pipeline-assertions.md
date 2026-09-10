# E2.4 — Matcher pipeline assertions

## Purpose

Prove that the accepted matcher behaves correctly as an assembled pipeline on a controlled synthetic supplier file.

This milestone is testing only. It does not change matching thresholds, matcher precedence, legal-form heuristics, canonical-resolution policy or accepted SKO-004/SKO-005 behaviour.

## Governed synthetic mini-pipeline

The fixture under `tests/fixtures/matcher/` contains eight supplier rows covering:

1. exact canonical alias resolution;
2. approved brand resolution;
3. exact identifier resolution;
4. a canonical technical match that also carries legal-form heuristic support;
5. a heuristic-only cooperative candidate;
6. a heuristic-only nonprofit/association candidate;
7. an ordinary unflagged supplier;
8. a cross-border canonical-alias negative.

The master-register fixture is deliberately unrelated so fuzzy matching cannot obscure the behaviours under test.

## Expected pipeline summary

The accepted expected shape is:

- input rows: 8
- technical matches: 4
- heuristic-only candidates: 2
- technical matches with heuristic support: 1
- total distinct candidates: 6
- unflagged rows: 2

`tests/test_matcher_pipeline_assertions.py` also compares every output row against `pipeline_expected.csv`, including match type, persistent entity ID, cooperative flag, marker flag and outcome bucket.

## Why both counts and row-level checks matter

Count assertions catch pipeline-wide regressions such as unexpected candidate expansion, dropped matches or precedence changes. Row-level assertions prevent a false pass where the aggregate counts remain the same but the wrong supplier occupies a category.

## Acceptance boundary

E2.4 can be put to owner acceptance when:

1. the controlled synthetic fixture runs through the real `match_suppliers()` entry point;
2. all six governed summary counts match exactly;
3. all eight row-level expected outcomes match exactly;
4. the canonical fast matcher gate passes;
5. the full repository suite passes;
6. no matcher behaviour code changes are required; and
7. the branch is clean and the implementation is committed/pushed.

If the pipeline assertions expose a genuine defect, that defect must be treated as separately governed matcher work rather than silently fixed inside E2.4.
