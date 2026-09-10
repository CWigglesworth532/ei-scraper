# Matcher regression fixtures

This directory contains committed **synthetic-only** fixtures for matcher regression testing.

## Rules

- Never copy live client rows into committed fixtures.
- Preserve the behavioural feature under test (for example punctuation, legal-form abbreviation, country alignment or suffix position) while using synthetic names.
- Each fixture row must have a stable identifier or an explicit row-level expectation.
- Positive heuristic cases are candidate-generation signals only; they do not prove social-economy classification.
- Country-scoped legal-form aliases must remain country-scoped.
- Short legal-form aliases should be constrained to safe positions where appropriate.
- A new real-world regression should be translated into a synthetic fixture before commit.

## Legal-form fixture

`expected_cases.csv` is the readable catalogue for legal-form candidate behaviour. It is intentionally small and reviewable.

Columns:

- `case_id`: stable test identifier
- `country`: ISO country code supplied to the matcher
- `supplier_name`: synthetic supplier name
- `expected_coop`: expected `name_coop_candidate` value
- `expected_marker`: expected `name_marker_candidate` value
- `expected_reason`: expected standardized reason string
- `notes`: human-readable rationale

## Pipeline assertion fixture

E2.4 adds a controlled synthetic mini-pipeline made up of:

- `pipeline_suppliers.csv`: eight headerless supplier rows supplied to the real matcher entry point;
- `pipeline_master.csv`: deliberately unrelated master-register row so canonical and heuristic behaviour can be tested without accidental fuzzy positives;
- `pipeline_safe_list.csv`: synthetic accepted canonical aliases, brand and identifier terms;
- `pipeline_expected.csv`: row-level expected match type, entity ID, heuristic flags and final outcome bucket.

The governed summary for this mini-pipeline is:

- input rows: 8
- technical matches: 4
- heuristic-only candidates: 2
- technical matches with heuristic support: 1
- total distinct candidates: 6
- unflagged rows: 2

The row set deliberately includes exact canonical resolution, brand resolution, identifier resolution, a technical match with a legal-form heuristic, heuristic-only cooperative and nonprofit candidates, an ordinary non-match, and a cross-border canonical-alias negative.

## Standard commands

Fast matcher regression gate:

```bash
python3 -m unittest discover -s tests -p 'test_matcher*.py' -v
```

Full repository gate:

```bash
python3 -m unittest discover -s tests -v
```

The fast matcher gate is only authoritative if all matcher regression files use the `test_matcher*.py` naming convention.
