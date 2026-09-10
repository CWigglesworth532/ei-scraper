# SKO-040 — Owner acceptance

**Task:** SKO-040 — Produce methodology proof and scalable whole-spend impact pilot  
**Epic / milestone:** E4 / E4.13  
**Branch:** `sko-040-whole-spend-methodology-proof`  
**Owner acceptance date:** 2026-09-10  
**Owner:** Charlie Wigglesworth  
**Status:** Complete / Accepted

## Acceptance decision

The owner explicitly accepted SKO-040 on 10 September 2026 after review of the implemented methodology proof and the governed local reconciliation evidence.

## Accepted evidence

Repository methodology proof:

- `docs/sko-040-whole-spend-methodology-proof.md`

Governed local evidence:

- `data/pilots/sko-040/whole_spend_methodology_proof.csv`
- `data/pilots/sko-040/whole_spend_methodology_proof_summary.json`

Deterministic evidence fingerprints:

- proof CSV SHA256: `9802db909515c77c9bbe19354b8cf79e483a9328aaa5ee4ee7b6f0d49016277b`
- summary JSON SHA256: `fe71efe3c1f13a41d808688278b45816d9a6cf2ad4e29c34ef62f18fffa65621`

Both files were regenerated and reproduced byte-identically.

## Accepted methodology proof boundary

The accepted proof demonstrates the supplier-agnostic chain:

`procurement spend → supplier country → purchased activity / NACE → governed direct country × sector coefficient → modelled direct economic/environmental outcomes → optional governed FIGARO upstream indirect extension`

Social-economy status is not part of the calculation eligibility gate. It remains a separate optional analytical segmentation. Supplier-specific verified social outcomes remain a separate evidence overlay.

The proof reuses the accepted SKO-036, SKO-037, SKO-038 and SKO-039 layers without recalculating or altering those methodologies.

## Accepted reconciliation evidence

The 35-observation methodology test set reconciled as follows:

- total observations: **35**
- total spend: **€25,077,933.78**
- country present: **35/35**
- NACE resolved: **27/35**
- direct modelled: **27/35**
- direct modelled spend: **€22,102,562.52**
- direct spend coverage: **88.14%**
- FIGARO eligible: **23/35**
- FIGARO eligible spend: **€19,873,893.55**
- FIGARO spend coverage: **79.25%**
- unresolved-NACE hold-outs: **8**
- trade valuation hold-outs: **4**

Accepted headline direct outcomes:

- GVA: **€10,340,270.82946137962023612430**
- employment: **120.8036961641447104460677169 persons-equivalent**
- direct GHG: **387.8432903074818001402078293 tCO2e**

On the common 23-observation FIGARO-eligible boundary:

- direct GVA: **€10,028,664.6633803**
- indirect GVA: **€9,013,276.88421664**
- combined GVA: **€19,041,941.5475969**
- direct GHG: **376.629940299729 tCO2e**
- indirect GHG: **1,880.13257429721 tCO2e**
- combined GHG: **2,256.76251459694 tCO2e**

## Accepted controls and claim boundaries

- unresolved NACE remains explicitly held out;
- direct trade rows use the governed `trade_turnover` denominator route;
- the approved FR P85 environmental fallback is explicitly labelled and remains direct-layer-only;
- UK→GB and other governed FIGARO representation equivalences are explicit rather than silent fallbacks;
- G45/G46/G47 trade procurement remains held out of primary FIGARO attribution because purchase value is not comparable with margin-based trade-industry output;
- `partially_modelled` means one or more direct outcomes are available while other secondary outcomes may be unavailable; it is not a whole-observation calculation failure;
- direct and indirect results are only combined on the same eligible observation boundary;
- procurement spend is an explicit output proxy for FIGARO, not an equivalence claim;
- employment is modelled persons-equivalent, not verified supplier headcount or jobs created;
- GHG is a modelled production-emissions estimate, not an audited supplier footprint;
- upstream outputs are gross, observation-based and not network-deduplicated.

Required FIGARO portfolio wording remains:

> Gross modelled upstream activity associated with the analysed purchases; supplier-level upstream requirements may overlap and are not deduplicated across the procurement portfolio.

## Remaining limitations after acceptance

Acceptance of SKO-040 does not imply universal production-scale coverage. Known limitations remain:

- `spend_year` is present in the governed output contract but blank across the current 35-observation pilot;
- eight observations remain unresolved for purchased activity / NACE;
- four trade observations remain held out from primary FIGARO indirect attribution;
- indirect employment remains deferred;
- the current proof uses a social-economy-origin pilot cohort even though social-economy status is outside the calculation gate;
- a larger ordinary-procurement run remains the next scaling validation step, not a prerequisite for this acceptance.

## Programme status consequence

With explicit owner acceptance and the required governed evidence present:

- **SKO-040: Complete / Accepted**
- **E4.13: Complete**

The next programme task is **SKO-041 — Complete Epic 4 formal acceptance and closeout / E4.14**.
