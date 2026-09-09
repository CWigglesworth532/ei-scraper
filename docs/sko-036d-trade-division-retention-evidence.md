# SKO-036D — Trade-division SBS retention evidence

## Task

SKO-036 — fixed-year direct economic coefficient layer for supplier-agnostic procurement spend.

## Trigger

The 35-observation pilot showed two Spanish trade observations mapped to G46/G47 but held out with `denominator_missing`. A direct live Eurostat SBS diagnostic for Spain, 2023 showed authoritative `sbs_ovw_act` rows for G46 and G47 for:

- `NETTUR_MEUR` — net turnover;
- `AV_MEUR` — value added;
- `EXPN_SAL_BEN_MEUR` — employee benefits expense;
- `EMP_NR` — persons employed.

The rows were therefore available upstream but absent from the governed normalized source layer.

## Root cause

`constrain_to_a64_model()` retained SBS rows only where the SBS NACE code was also present in the country-specific `nama_10_a64` vocabulary. This was correct for preventing detailed SBS classes/groups from becoming pseudo-model sectors, but it conflicted with the already accepted hybrid denominator architecture for trade:

- G45 / G46 / G47 must use turnover-compatible SBS denominators;
- trade sectors must not silently fall back to national-accounts P1/output.

Where `nama_10_a64` exposes a broader trade aggregate, valid G45/G46/G47 SBS division rows were being discarded.

## Implemented refinement

`extract_eurostat_direct_economic.py` now defines governed trade-division overrides `{G45, G46, G47}`.

For SBS rows:

- ordinary rows are still retained only when present in the country-specific A*64 vocabulary;
- G45/G46/G47 are retained explicitly even when absent from that A*64 vocabulary;
- more detailed trade rows such as G466 remain excluded from the model layer.

The extraction summary now records `sbs_trade_override_rows_retained`.

## Tests

`tests/test_extract_eurostat_direct_economic.py` now checks:

1. the governed trade-division override set is exactly G45/G46/G47;
2. a G46 SBS row is retained when national accounts expose only a broader trade aggregate;
3. a more detailed SBS row such as G466 is still excluded.

## Acceptance boundary

This refinement is implemented but not yet acceptance evidence for SKO-036 as a whole. The live Eurostat source must be regenerated, coverage rebuilt, and the 35-observation pilot rerun to confirm that the Spanish G46/G47 observations move from `denominator_missing` to modelled outcomes where 2023 SBS data are available.
