# SKO-036D — A*64 specificity mapping refinement evidence

## Task
SKO-036 — direct economic impact coefficient attribution pilot.

## Problem observed
The 35-observation procurement pilot produced four `ambiguous_a64_sector_for_division` holds for valid NACE assignments:

- 18.12 — printing
- 70.22 — management consultancy
- 71.20 — technical testing and analysis
- 71.12 — engineering activities

Inspection of the 2023 coefficient vocabulary showed that Eurostat rows included both a broader published aggregate and a more specific compatible sector, for example:

- division 18: `C16-C18` and `C18`
- division 70: `M69-M71` and `M69_M70`
- division 71: `M69-M71` and `M71`

The attribution mapper previously treated every overlapping span as equally eligible, causing valid detailed NACE assignments to be held out.

## Change implemented
`apply_direct_economic_coefficients.py` now chooses the narrowest compatible division span when multiple published model-sector rows contain the same NACE division.

Governance rule:

1. exact model-sector code remains preferred;
2. otherwise identify all compatible sector spans containing the NACE division;
3. choose the unique narrowest span;
4. if two or more equally specific candidates remain, hold out as `ambiguous_equally_specific_a64_sectors` rather than guessing.

The mapping reason `division_to_most_specific_a64_sector` is recorded when a broader aggregate was explicitly rejected in favour of the more specific sector.

## Tests added
`tests/test_apply_direct_economic_coefficients.py` now tests:

- C18 selected over C16-C18 for NACE 18.12;
- M69_M70 selected over M69-M71 for NACE 70.22;
- equal-specificity collisions remain unresolved.

## Separate unresolved issue
The Spanish G46/G47 2023 trade cells are held out because the turnover denominator is missing. This is not treated as a NACE mapping defect. No aggregation or substitution has been introduced for those trade cells. The source-layer cause must be diagnosed separately before any change.

## Acceptance boundary
This refinement is implemented but not yet accepted. Required evidence remains:

- focused tests passing after the change;
- rerun of the 35-observation pilot showing the recovered mappings;
- separate diagnosis of the Spanish trade denominator gap;
- UK/ONS layer for the two UK observations;
- final SKO-036 acceptance evidence and tracker update.
