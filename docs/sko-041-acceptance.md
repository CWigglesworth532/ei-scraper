# SKO-041 / Epic 4 acceptance record

**Task:** SKO-041 — Complete Epic 4 formal acceptance and closeout  
**Milestone:** E4.14 — Epic 4 accepted  
**Status:** Complete / Accepted  
**Owner acceptance date:** 2026-09-10  
**Branch:** `sko-041-epic-4-closeout`

## Owner decision

Charlie explicitly accepted SKO-041 / Epic 4 on 10 September 2026 after review of `docs/sko-041-epic-4-closeout.md`.

## Acceptance basis

Epic 4 is accepted on the basis that skopia has demonstrated a reproducible, supplier-agnostic spend-attribution methodology with the governed chain:

`procurement spend → supplier country → purchased activity / NACE → governed country × sector coefficient → modelled direct economic/environmental outcome → optional governed FIGARO upstream indirect outcome`

The accepted programme state includes:

- governed purchased-activity/NACE evidence and assertions;
- versioned direct economic coefficient layers;
- versioned direct GHG coefficient layers;
- deterministic whole-spend direct attribution;
- explicit mapped/unmapped spend and hold-out reasons;
- governed denominator routing and FX handling;
- explicit bounded fallback treatment;
- optional reconciled FIGARO Type I upstream attribution;
- same-boundary direct + indirect combination only;
- deterministic SKO-040 supplier-agnostic methodology proof;
- social-economy status as an optional segmentation only;
- supplier-specific verified social outcomes as a separate evidence overlay;
- documented client-facing claim boundaries;
- a documented pathway to larger ordinary procurement files;
- explicit limitations and deferred items.

## Accepted pilot evidence

Governed cohort: **35 observations / €25,077,933.78 total spend**.

Direct accepted boundary:

- 27 / 35 observations modelled;
- €22,102,562.52 modelled spend;
- 88.14% spend coverage;
- GVA €10,340,270.829461...;
- employment 120.803696... persons-equivalent;
- direct GHG 387.8432903074818 tCO2e;
- eight unresolved-NACE hold-outs.

FIGARO accepted boundary:

- 23 / 35 observations eligible;
- €19,873,893.55 eligible spend;
- 79.25% spend coverage;
- eight unresolved-NACE hold-outs plus four trade valuation hold-outs;
- direct GVA €10,028,664.6633803;
- indirect GVA €9,013,276.88421664;
- combined GVA €19,041,941.5475969;
- direct GHG 376.629940299729 tCO2e;
- indirect GHG 1,880.13257429721 tCO2e;
- combined GHG 2,256.76251459694 tCO2e.

SKO-040 deterministic proof fingerprints:

- CSV SHA256: `9802db909515c77c9bbe19354b8cf79e483a9328aaa5ee4ee7b6f0d49016277b`
- summary JSON SHA256: `fe71efe3c1f13a41d808688278b45816d9a6cf2ad4e29c34ef62f18fffa65621`

## Accepted governance boundaries

- v1 direct coefficients use fixed 2023 reference-year releases; no automatic year fallback.
- Ordinary sectors use P1/output where governed; G45/G46/G47 use turnover-compatible routing and never silently fall back to P1.
- Absolute non-EUR outcomes use governed FX normalization.
- Purchased activity is the modelling target; unresolved or ambiguous activity remains held out.
- The FR P85→P environmental fallback is explicit, owner-approved, direct-only and visible in lineage.
- FIGARO uses exact country × A*64 mapping plus governed representation equivalences only; no silent parent fallback.
- Trade procurement remains held out of the primary FIGARO case pending a defensible valuation treatment.
- FIGARO uses `L-I` Type I upstream effects only; no induced household-consumption effects.
- Procurement spend is an explicit basic-price output proxy for FIGARO, not an equivalence claim.
- Upstream portfolio results are gross / observation-based and not network-deduplicated.
- Direct and indirect outcomes are combined only on the same eligible observation boundary.

Required FIGARO portfolio wording remains:

> Gross modelled upstream activity associated with the analysed purchases; supplier-level upstream requirements may overlap and are not deduplicated across the procurement portfolio.

## Remaining limitations / follow-ons

The following do not block Epic 4 acceptance:

- eight of 35 pilot observations remain unresolved for purchased activity/NACE;
- four resolved trade observations remain held out from primary FIGARO attribution;
- indirect employment remains deferred pending a governed satellite;
- the current pilot has no populated `spend_year` values;
- wider UK sector coverage remains source-dependent;
- secondary direct outcomes have outcome-specific coverage;
- the current 35-observation cohort originated from a social-economy-oriented population;
- larger ordinary-procurement scaling validation remains desirable;
- current methodology proof does not establish universal country × sector coverage for every arbitrary client file;
- supplier-specific verified social outcomes remain separate from the generic attribution calculation.

For production scaling, material spend should be prioritised before exhaustive tail-spend classification. A larger staged ordinary-procurement validation remains a productisation follow-on rather than an Epic 4 acceptance prerequisite.

## Final programme status

**SKO-041: Complete / Accepted.**  
**E4.14: Complete.**  
**Epic 4: Complete / Accepted.**

The accepted closeout evidence is `docs/sko-041-epic-4-closeout.md` together with the accepted SKO-035–SKO-040 evidence chain and the authoritative programme tracker.
