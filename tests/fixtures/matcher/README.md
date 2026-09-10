# Matcher regression fixtures

This directory contains committed **synthetic-only** fixtures for matcher regression testing.

## Rules

- Never copy live client rows into committed fixtures.
- Preserve the behavioural feature under test (for example punctuation, legal-form abbreviation, country alignment or suffix position) while using synthetic names.
- Each fixture row must have a stable `case_id` and an explicit expected outcome.
- Positive heuristic cases are candidate-generation signals only; they do not prove social-economy classification.
- Country-scoped legal-form aliases must remain country-scoped.
- Short legal-form aliases should be constrained to safe positions where appropriate.
- A new real-world regression should be translated into a synthetic fixture before commit.

## Primary fixture

`expected_cases.csv` is the readable catalogue for legal-form candidate behaviour. It is intentionally small and reviewable.

Columns:

- `case_id`: stable test identifier
- `country`: ISO country code supplied to the matcher
- `supplier_name`: synthetic supplier name
- `expected_coop`: expected `name_coop_candidate` value
- `expected_marker`: expected `name_marker_candidate` value
- `expected_reason`: expected standardized reason string
- `notes`: human-readable rationale

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
