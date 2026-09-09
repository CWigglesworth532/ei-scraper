# SKO-036C — fixed-year procurement attribution pilot evidence

Status: **implemented; local cohort execution and owner acceptance pending**

## Purpose

Demonstrate that the SKO-036 supplier-agnostic direct economic methodology can take procurement observations and apply the governed fixed-year coefficient release:

`procurement spend → supplier country → purchased activity / NACE → A*64 model sector → 2023 country × sector × outcome coefficient → modelled direct outcome`

Social-economy status is not required or used by this calculation.

## Owner-approved year policy

The v1 coefficient layer uses a single fixed reference year of 2023. The procurement spend year does not need to match the coefficient reference year. Automatic year fallback is disabled. Missing 2023 coefficient cells are held out rather than silently substituted. Attribution outputs record the coefficient reference year separately; the current 35-observation pilot input does not provide a spend-year column, so spend year remains blank rather than inferred.

## Implementation

Added `apply_direct_economic_coefficients.py`.

The adapter:

- reads the governed reference-year policy from `config/direct_economic_coefficients.yaml`;
- consumes the existing coefficient coverage matrix rather than recomputing source statistics;
- maps class/group/division NACE codes to a unique governed A*64 model sector using the division represented by the A*64 sector code/range;
- retains unresolved or ambiguous NACE mappings as held out;
- retains countries absent from the coefficient layer (including UK before the ONS layer is built) as held out;
- preserves route-level `not_applicable` outcomes separately from true held-out outcomes;
- applies monetary coefficients directly to spend and person/hour coefficients per EUR million denominator;
- emits observation-level results, long-form outcome results, and a JSON summary;
- reports both observation coverage and spend-weighted coverage by outcome;
- retains coefficient ID, reference year, denominator route, and source release provenance.

## Pilot source contract

The owner-local ignored cohort `data/pilots/sko-036/research_review_pass1.csv` has 35 observations and includes:

- `selection_id`
- `supplier`
- `client`
- `country`
- `spend_eur`
- `purchased_activity_description`
- `proposed_nace_rev2_code`
- `nace_level`
- `nace_description`
- `confidence`
- evidence/rationale fields
- `treatment`
- allocation/review fields

No invented spend splits are introduced by the attribution adapter.

## Tests added

Added `tests/test_apply_direct_economic_coefficients.py` covering:

- NACE class → A*64 mapping;
- NACE group → A*64 mapping;
- grouped A*64 division ranges;
- monetary and employment attribution arithmetic;
- explicit `not_applicable` handling;
- explicit country-source holdout for UK before the ONS layer;
- explicit unresolved-NACE holdout;
- fixed 2023 reference-year policy and no automatic fallback.

## Acceptance boundary

This evidence does **not** mark SKO-036 complete or accepted. Acceptance still requires:

1. local py_compile and focused tests passing against the repository;
2. execution against the real 35-observation cohort and live 2023 Eurostat matrix;
3. inspection of observation- and spend-weighted coverage and modelled outcomes;
4. UK/ONS coefficient-layer work for UK observations where required;
5. owner review of the pilot evidence and final SKO-036 acceptance decision.
