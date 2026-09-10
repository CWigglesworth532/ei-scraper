# SKO-039 Closeout — FIGARO indirect supply-chain attribution

**Status:** Complete / Accepted  
**Owner acceptance date:** 2026-09-10  
**Milestone:** E4.12  
**Branch:** `sko-039-figaro-indirect-attribution`

## Purpose

SKO-039 proves a governed, supplier-agnostic indirect supply-chain attribution method for procurement spend using FIGARO. The accepted calculation chain is:

supplier spend → supplier country → purchased activity / NACE → FIGARO country × industry shock → Leontief upstream requirements → indirect GVA / GHG.

Social-economy status is not required for the core calculation and remains an optional analytical segmentation only.

## Accepted methodology boundary

- FIGARO 2026 edition, reference year 2023.
- Industry-by-industry A*64 table at basic prices.
- Procurement spend is treated as an explicit basic-price output proxy, not as an equivalence claim.
- Technical coefficients use `A = Z / output` by column.
- Leontief inverse uses `L = (I-A)^-1`.
- Upstream-only operator is `U = L-I`.
- No induced household-consumption effects.
- Accepted SKO-038 direct effects remain authoritative and are not recalculated.
- Combined result = accepted direct + FIGARO indirect on the same eligible observation boundary.
- Exact country × A*64 mapping only; no silent parent-sector fallback.
- SKO-037 FR `P85 → P` environmental-direct fallback is not inherited by FIGARO; FIGARO remains on the governed model sector.
- Trade procurement in G45/G46/G47 is mapped but held out of the primary FIGARO attribution because purchase value is not comparable with FIGARO trade-industry output, which is margin-based.
- Portfolio aggregation is gross / observation-based and is not network-deduplicated.
- Primary indirect outcomes are GVA and GHG.
- Employment remains secondary/deferred; when no governed employment satellite is supplied it is omitted rather than represented as zero.

## Governed source package

### FIGARO economic source

- Raw source: `matrix_eu-ic-io_ind-by-ind_26ed_2023.csv`
- Raw SHA256: `030ff10a923a5d8949e4d05c16c7f2260d59c91f23d74f9529a029ba9249d7a5`
- Compact model: `figaro_model_26ed_2023.npz`
- Compact model SHA256: `f177056152d48a9e3775adb2eea9bd16231b58dec05d1ca14021e03592e008ba`
- Usable FIGARO nodes: 3,135
- Non-zero intermediate transactions: 6,357,196
- Accounting reconstruction max absolute difference: `6.984919309616089e-09` million EUR.

### GHG satellite

- Raw source: `env_ac_ghgfp_2023_totals.csv`
- Raw SHA256: `c1875958df6e7033bac0ac9c3ae011ccc7f97714bdd70ee4a51c8cd9ece9851b`
- Normalized GHG satellite SHA256: `9771c2708f32b8366344ed413c6bf32d7768d95efa764bd6a9c540596cc10f10`
- Coverage: 3,135 / 3,135 usable FIGARO nodes.
- Source emissions preserved: 9,708,184,545 tCO2e.
- Total allocation difference: 0.0 tCO2e.

### GVA satellite

- Normalized GVA satellite SHA256: `1db7d5a222d2ab562c06eb552dd20fc39b69dd083ba2cfa5ce9c8e8689f01ff3`
- Coverage: 3,135 / 3,135 usable FIGARO nodes.

### Source-package validation

- Source-package fingerprint: `47b6256e3ae7115e3465a1ce9bb3117a6d005bdc10113927c6fde2e3778c0be6`
- Validation status: `source_package_validated`
- Pilot execution permitted: true.

## Governed exact-equivalence mappings

The live pilot exposed five representation mismatches rather than substantive coverage gaps. They were resolved through explicit governed equivalence mappings, while preserving original cohort values in output lineage:

- country: `UK → GB`
- sector notation: `C10-C12 → C10T12`
- sector notation: `M69_M70 → M69_70`
- sector notation: `N80-N82 → N80T82`

These are exact representation equivalences, not parent-sector fallbacks.

## Live-pilot evidence

Governed cohort: 35 observations, total spend €25,077,933.78.

Final live-pilot structure:

- mapped eligible observations: 23
- held-out observations: 12
- unresolved-NACE holdouts: 8
- trade valuation holdouts: 4
- unexplained mapping losses: 0
- mapped eligible spend: €19,873,893.55
- mapped eligible spend coverage: 79.2485287039465%

### Primary outcome results

| Outcome | Accepted direct | FIGARO indirect | Combined | Unit |
| --- | ---: | ---: | ---: | --- |
| GVA | 10,028,664.6633803 | 9,013,276.88421664 | 19,041,941.5475969 | EUR |
| GHG | 376.629940299729 | 1,880.13257429721 | 2,256.76251459694 | tCO2e |

The direct figures above are the accepted SKO-038 direct effects for the same 23 observations eligible for FIGARO, not the entire SKO-038 direct portfolio.

### Reconciliation and determinism

- Direct input recalculated: false.
- Leontief inverse max reconciliation error in live run: `0`.
- Country × sector contribution reconciliation failures: 0.
- Deterministic rerun: confirmed by identical live-pilot summaries before the final reporting-wording-only change.
- Final post-wording live run retained identical mapping structure and GVA/GHG values.
- Final live-pilot fingerprint after governed reporting wording update: `132cd4e3192d48169e2cb438a7c595ba2f49689b87269b60c88779a55ae571a3`.

## Reporting boundary

Required portfolio wording:

> Gross modelled upstream activity associated with the analysed purchases; supplier-level upstream requirements may overlap and are not deduplicated across the procurement portfolio.

This wording is emitted by the final governed configuration.

## Implementation evidence

Relevant SKO-039 commits include:

- `1919cc5e2399aaa0c830176f108b6cc925c66bb6` — live source-package validator refinement.
- `2361bc61951bc64e0abad9b679fd3044badfa0f8` — source validation tests.
- `5c8f6df76cb14103ecd0a3f95f7c7fe07c75d738` — live pilot runner.
- `94bf40534e09da227e02f85c0528a0f9558f646f` — live pilot runner tests.
- `5dbd901c1c5047df20a755e26214341799a462fd` — governed exact-equivalence mapping config.
- `8a0f9d8b4218d8ff1a5c1769a04caa54c0fffa15` — explicit country/sector equivalence support.
- `93126211c2851d4e855d3c3b6ba76bc0a237e1b3` — equivalence mapping tests.
- `326c6f088eca99ad33fe068eba3dcafe6b8a8277` — omit entirely unsupplied secondary outcomes from attribution outputs.
- `d2345ade81dfe54f24c9cfc97f6b23b06f29e3f8` — secondary-outcome tests.
- `8dedcaac887a59738cfbda1c84659ae151fe87a6` — final governed portfolio wording.

A separate SKO-036 stale regression assertion was corrected during the programme-wide regression gate in commit `5d5892f746b97201a8c5cf06c0f5c75230ff6128`; this did not change SKO-036 methodology.

## Test evidence

- Focused source-validator gate: 17/17 passed.
- Earlier focused SKO-039 regression gate: 63/63 passed.
- Full repository regression gate after the SKO-036 stale-test correction: 492/492 passed.
- Final SKO-039 closeout regression after all SKO-039 patches: 72/72 passed.
- Working tree reported clean before closeout.

## Acceptance

Owner explicitly accepted SKO-039 on 2026-09-10 after review of the final live-pilot outputs and reporting wording.

**Final status: SKO-039 Complete / Accepted.**  
**Milestone E4.12: Complete.**

## Remaining / deferred items

- Indirect employment remains secondary and deferred until a governed employment satellite with defensible coverage is implemented and validated.
- Trade-sector indirect attribution remains a governed holdout in the primary case pending a defensible valuation treatment.
- These deferred items do not block SKO-039 acceptance.
