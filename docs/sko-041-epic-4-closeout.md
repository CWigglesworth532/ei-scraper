# SKO-041 — Epic 4 formal closeout

**Task:** SKO-041  
**Epic / milestone:** E4 / E4.14  
**Status:** Closeout document implemented for owner review; Epic 4 not yet accepted.  
**Branch:** `sko-041-epic-4-closeout`  
**Accepted technical baseline:** SKO-035, SKO-036, SKO-037, SKO-038, SKO-039 and SKO-040  

## 1. Purpose and acceptance question

Epic 4 was established to build a governed spend-attribution capability that can translate procurement spend into economic and environmental outcomes using supplier country, purchased economic activity and versioned official statistical coefficients, while retaining explicit provenance, mapped/unmapped coverage, fallback controls and bounded reporting claims.

The final acceptance question is:

> Has skopia demonstrated a reproducible, supplier-agnostic methodology that can take real procurement spend through country and purchased-activity/NACE resolution to governed direct economic/environmental attribution, with an optional governed upstream FIGARO extension, while preserving explicit limitations, hold-outs, lineage and a credible scaling pathway to ordinary client procurement files?

The evidence assembled below supports owner review of that question. This document does not itself mark Epic 4 accepted; explicit owner acceptance remains required.

## 2. Final accepted architecture

The accepted Epic 4 architecture is:

`procurement spend → supplier country → purchased activity / NACE → governed country × sector coefficient → modelled direct outcome → optional FIGARO upstream indirect outcome`

The architecture is supplier-agnostic. Social-economy status is not required for calculation eligibility and is retained only as an optional analytical segmentation.

Supplier-specific verified social or environmental outcomes are a separate evidence overlay. They are not inferred from generic coefficient-based modelling and are not mathematically combined with generic GVA, employment or GHG estimates unless a separately governed methodology explicitly supports that operation.

## 3. Accepted Epic 4 delivery chain

### 3.1 SKO-035 — purchased activity / NACE governance

SKO-035 established governed purchased-activity evidence and NACE determination at supplier-spend-observation × activity-allocation grain.

Accepted capabilities include:

- preserving purchased-activity evidence separately from assertions;
- NACE Rev. 2 as the governed v1 target;
- single-sector, evidence-backed split, broader-level and unresolved outcomes;
- explicit confidence and materiality escalation;
- national classification evidence retained without silent relabelling;
- production crosswalks empty unless an explicit governed/versioned correspondence exists;
- no automatic free-text classifier in the accepted v1 layer;
- deterministic assertions and QA;
- unresolved activity remains unresolved rather than being forced into a sector.

This provides the activity-resolution gate used by later coefficient and attribution stages.

### 3.2 SKO-036 — direct economic coefficient layer

SKO-036 established the supplier-agnostic direct economic coefficient layer for procurement spend.

Accepted methodology:

- fixed 2023 coefficient release;
- ordinary sectors use national-accounts output (`P1`) as the denominator route where spend-as-output proxy is methodologically defensible;
- trade sectors G45/G46/G47 use turnover-compatible business-statistics routing and do not silently fall back to `P1`;
- absolute non-EUR outcomes use governed FX normalization;
- dimensionless monetary ratios are not FX-adjusted;
- NACE maps to the unique narrowest compatible model-sector span;
- ambiguity remains unresolved;
- source lineage, coefficient IDs, denominator route, source sector, reference year and provenance are retained.

Accepted direct economic outcomes include:

- gross value added;
- intermediate consumption;
- compensation of employees;
- production taxes less subsidies;
- gross operating surplus / mixed income;
- consumption of fixed capital;
- employment persons;
- employment hours;
- gross fixed capital formation;
- trade-route labour costs where source-compatible.

Final accepted live-pilot evidence:

- 35 procurement observations;
- total spend **€25,077,933.78**;
- 27 observations with at least one modelled economic outcome;
- **€22,102,562.52** modelled spend;
- **88.14%** spend coverage;
- eight held-out rows, all unresolved purchased-activity/NACE;
- direct GVA **€10,340,270.83**;
- employment **120.80 persons-equivalent**;
- 4,755 / 4,755 accounting identity checks passed;
- final direct-economic regression **59 / 59** tests passed.

### 3.3 SKO-037 — direct environmental / GHG coefficient layer

SKO-037 established the supplier-agnostic direct residence-based production GHG layer.

Accepted methodology:

- fixed 2023 reference year;
- ordinary sectors use the accepted `P1` denominator route;
- G45/G46/G47 use turnover;
- Eurostat `env_ac_ainah_r2` as the primary environmental source for the resolved non-UK cohort;
- ONS Environmental Accounts for the targeted UK M72 case;
- provisional/estimated published numeric observations remain usable with flags preserved;
- suppressed/confidential/unavailable observations remain hold-outs;
- no automatic year fallback;
- the FR P85 environmental granularity gap is handled through one explicit owner-approved P85 → P coefficient-source fallback while retaining purchased activity as P85;
- the fallback is direct-environmental only and is not inherited by FIGARO.

Final live-pilot evidence:

- 21 / 21 required country × sector coefficient cells calculated;
- 27 / 35 observations modelled;
- eight unresolved-NACE hold-outs;
- **€22,102,562.52 / €25,077,933.78 = 88.14%** spend coverage;
- direct GHG **387.8432903074818 tCO2e**;
- UK M72 physical-unit and FX route validated;
- all four resolved trade observations use turnover;
- targeted SKO-037 tests **14 / 14** passed.

### 3.4 SKO-038 — governed whole-spend direct attribution

SKO-038 composed the accepted SKO-036 and SKO-037 layers into a single direct whole-spend workflow without recalculating or overriding coefficient methodology.

The composer emits:

- observation-level attribution;
- a canonical long-form outcome ledger;
- portfolio aggregation by outcome;
- QA breakdowns by country, model sector, denominator route, coefficient specificity and hold-out reason;
- machine-readable reconciliation summaries.

It preserves:

- coefficient IDs;
- source lineage;
- coefficient reference year;
- denominator route and currency;
- FX provenance;
- coefficient source sector;
- fallback/specificity provenance;
- attribution status and reason.

Accepted live-pilot evidence:

- 35 observations / **€25,077,933.78**;
- 27 observations modelled;
- eight unresolved-NACE hold-outs;
- **€22,102,562.52 = 88.14%** spend coverage;
- 385 observation/outcome ledger rows;
- 11 portfolio outcomes;
- GVA **€10,340,270.829461...**;
- employment **120.803696... persons-equivalent**;
- direct GHG **387.8432903074818 tCO2e**;
- UK FX, FR fallback, trade routing and coefficient lineage invariants passed;
- byte-identical deterministic rerun of all five governed outputs;
- focused SKO-038 tests **16 / 16**, SKO-036 regression **59 / 59**, SKO-037 regression **14 / 14**.

### 3.5 SKO-039 — optional FIGARO upstream indirect attribution

SKO-039 established the optional supplier-agnostic Type I upstream extension.

Accepted methodology boundary:

- FIGARO 2026 edition, reference year 2023;
- industry-by-industry A*64 table at basic prices;
- procurement spend treated as an explicit basic-price output proxy, not an equivalence claim;
- technical coefficients `A = Z / output` by column;
- Leontief inverse `L = (I-A)^-1`;
- upstream-only operator `U = L-I`;
- no induced household-consumption effects;
- accepted SKO-038 direct effects remain authoritative and are not recalculated;
- combined values use direct + indirect only on the same eligible observation boundary;
- exact country × A*64 mapping only, with governed representation equivalences where economic meaning is unchanged;
- no silent parent-sector fallback;
- trade G45/G46/G47 remains held out of the primary FIGARO case because procurement purchase value is not comparable with margin-based trade-industry output;
- portfolio aggregation is gross / observation-based and not network-deduplicated;
- primary indirect outcomes are GVA and GHG;
- indirect employment remains secondary/deferred.

Governed representation equivalences:

- `UK → GB`;
- `C10-C12 → C10T12`;
- `M69_M70 → M69_70`;
- `N80-N82 → N80T82`.

Accepted live-pilot evidence:

- 35 observations;
- 23 mapped eligible;
- 12 governed hold-outs: eight unresolved NACE + four trade valuation hold-outs;
- no unexplained mapping losses;
- mapped eligible spend **€19,873,893.55**;
- spend coverage **79.25%**;
- on the same 23-observation boundary:
  - GVA: **€10,028,664.6633803 direct + €9,013,276.88421664 indirect = €19,041,941.5475969 combined**;
  - GHG: **376.629940299729 direct + 1,880.13257429721 indirect = 2,256.76251459694 tCO2e combined**;
- contribution reconciliation failures: **0**;
- live-pilot output deterministic;
- source-package fingerprint `47b6256e3ae7115e3465a1ce9bb3117a6d005bdc10113927c6fde2e3778c0be6`;
- final live-pilot fingerprint `132cd4e3192d48169e2cb438a7c595ba2f49689b87269b60c88779a55ae571a3`;
- final SKO-039 regression **72 / 72** passed.

Required portfolio wording:

> Gross modelled upstream activity associated with the analysed purchases; supplier-level upstream requirements may overlap and are not deduplicated across the procurement portfolio.

### 3.6 SKO-040 — supplier-agnostic methodology proof

SKO-040 consolidated the accepted architecture into one explicit whole-spend methodology proof and scaling pathway.

The governed local proof joined accepted SKO-038 and SKO-039 outputs deterministically by `selection_id` without rebuilding either accepted method.

Accepted proof boundary:

- total: **35 observations / €25,077,933.78**;
- supplier country present: **35 / 35**;
- NACE resolved: **27 / 35 / €22,102,562.52**;
- direct modelled: **27 / 35 / €22,102,562.52 = 88.14% spend**;
- FIGARO eligible: **23 / 35 / €19,873,893.55 = 79.25% spend**;
- unresolved-NACE hold-outs: **8**;
- trade FIGARO hold-outs: **4**.

The proof preserves direct and FIGARO lineage, coefficient specificity, denominator routing, fallback status, valuation eligibility and hold-out reasons.

The proof outputs reproduced byte-identically across reruns:

- `whole_spend_methodology_proof.csv` SHA256: `9802db909515c77c9bbe19354b8cf79e483a9328aaa5ee4ee7b6f0d49016277b`;
- `whole_spend_methodology_proof_summary.json` SHA256: `fe71efe3c1f13a41d808688278b45816d9a6cf2ad4e29c34ef62f18fffa65621`.

SKO-040 also formalised the minimum client input contract, staged enrichment/hold-out path, client reporting structure, claim boundaries and the path to a larger ordinary-procurement validation cohort.

## 4. Supplier-agnostic capability and limitation matrix

| Capability | Accepted state | Evidence / boundary | Current limitation |
| --- | --- | --- | --- |
| Purchased-activity/NACE determination | Accepted | Governed evidence/assertion architecture with single/split/broader/unresolved treatments | Requires evidence; unresolved activity remains held out |
| Direct GVA | Accepted | 27/35 pilot observations; 88.14% spend coverage | Coverage depends on country/activity/coefficient availability |
| Direct employment persons-equivalent | Accepted | 27/35 pilot observations; 88.14% spend coverage | Modelled persons-equivalent, not verified jobs created |
| Additional direct economic outcomes | Accepted where source-compatible | 11 portfolio outcomes in SKO-038 | Outcome-specific coverage varies |
| Direct GHG | Accepted | 27/35 pilot observations; 387.8433 tCO2e | Modelled production emissions, not supplier-audited carbon footprint |
| UK non-EUR absolute outcomes | Accepted for governed route | EUR→GBP 2023 FX retained in provenance | Wider UK sector coverage remains source-dependent |
| Direct trade attribution | Accepted | G45/G46/G47 use turnover | Must not use P1 fallback |
| Environmental granularity fallback | Accepted where explicitly governed | One FR P85→P direct-GHG case | Fallback must remain visible and is not generalised automatically |
| FIGARO upstream GVA | Accepted optional extension | 23/35 eligible; 79.25% spend coverage | Procurement spend is a basic-price output proxy |
| FIGARO upstream GHG | Accepted optional extension | 23/35 eligible; 79.25% spend coverage | Gross upstream estimate; not network-deduplicated |
| FIGARO trade attribution | Held out in primary case | Four resolved trade observations held out | Requires a future defensible valuation treatment |
| Indirect employment | Deferred | No governed employment satellite supplied | Omitted rather than represented as zero |
| Social-economy segmentation | Accepted as optional overlay | Does not affect calculation eligibility | Classification quality remains a separate data problem |
| Supplier-specific social outcomes | Separate evidence overlay | Must be separately verified | Not inferred from generic coefficients |
| Arbitrary whole-client procurement file | Methodology proven; operational scaling not yet validated | 35-observation proof plus scaling pathway | Larger ordinary-procurement validation still needed |

## 5. Source and coefficient governance accepted at Epic level

The accepted Epic 4 governance model is based on explicit versioning, provenance and controlled failure rather than silent approximation.

### 5.1 Reference-year governance

- v1 direct coefficients use 2023 as the fixed reference year.
- No automatic coefficient-year fallback is permitted.
- Procurement spend year and coefficient reference year are conceptually separate and should both be preserved.
- The current 35-observation pilot has blank `spend_year`; this is a known input limitation, not a coefficient-methodology defect.

### 5.2 Denominator governance

- Ordinary sectors use `P1` / output where accepted.
- Trade sectors G45/G46/G47 use turnover-compatible routing.
- Trade must not silently fall back to `P1`.
- Absolute outcomes in a non-EUR denominator currency require governed FX normalization.

### 5.3 Activity and sector governance

- Purchased activity is the calculation target, not merely the supplier's broad corporate activity.
- NACE Rev. 2 is the current governed target for the accepted v1 layer.
- National classification codes remain source evidence unless an explicit governed/versioned correspondence exists.
- Sector mapping uses the unique narrowest compatible model-sector span.
- Equal-specificity ambiguity remains unresolved.
- FIGARO mapping uses exact country × A*64 nodes plus explicit governed representation equivalences.
- No silent FIGARO parent-sector fallback is accepted.

### 5.4 Environmental governance

- Direct GHG uses residence-based production emissions.
- Published provisional/estimated numeric statistical observations may be used with flags retained.
- Suppressed/confidential/unavailable values remain hold-outs.
- The FR P85→P direct-environmental fallback is explicit, owner-approved and visible in lineage.
- That fallback does not propagate into FIGARO.

### 5.5 FIGARO governance

- FIGARO 2026 / 2023 industry-by-industry A*64 at basic prices is the accepted v1 upstream basis.
- `L-I` defines upstream-only Type I effects.
- No induced household-consumption effects are included.
- Direct SKO-038 effects are authoritative and are not recalculated.
- Combined reporting requires the same eligible observation boundary.
- Trade procurement remains held out until a defensible valuation treatment exists.

## 6. Baseline-versus-improved reconciliation

### 6.1 Reconciliation principle

Epic 4 should not claim a numerical improvement percentage unless the pre-Epic-4 and post-Epic-4 calculations share the same cohort, outcome definition, sector assignment, denominator route, source year and reporting boundary.

The currently available governed repository evidence does not provide one single pre-Epic-4 numerical baseline artifact that can be reconciled line-for-line against the accepted 35-observation output across all of those dimensions.

Accordingly, the defensible reconciliation is primarily methodological, with numerical comparison only where a genuinely like-for-like baseline exists.

### 6.2 Methodological baseline

The pre-Epic-4 analytical baseline can be characterised as broad spend-based impact estimation without the full accepted governance now present around:

- purchased activity at supplier-spend-observation grain;
- versioned NACE assertions;
- country × sector coefficient coverage;
- denominator route compatibility;
- trade-specific denominator treatment;
- reference-year governance;
- FX provenance;
- coefficient identifiers and source lineage;
- mapped/unmapped spend disclosure;
- fallback specificity;
- deterministic observation/outcome ledgers;
- direct environmental coefficients;
- governed upstream FIGARO attribution;
- same-boundary direct/indirect reconciliation;
- explicit client-reporting claim boundaries.

### 6.3 Improved accepted state

The accepted Epic 4 state adds those controls and therefore changes the nature of the analytical product from a broad estimate to a governed, inspectable attribution model.

The key measurable improvements demonstrated on the 35-observation pilot are:

- 100% of pilot spend retains an explicit country signal;
- 88.14% of spend has resolved purchased activity and reaches the direct model;
- unresolved activity is held out rather than guessed;
- direct GVA, employment and GHG reconcile to observation-level outputs;
- coefficient lineage and denominator routing are preserved row by row;
- trade valuation is handled explicitly rather than using a silent ordinary-sector assumption;
- one environmental parent fallback is visible and governed rather than hidden;
- 79.25% of spend reaches the optional FIGARO upstream case;
- upstream trade exclusions are explicit;
- direct and indirect values reconcile only on the same 23-observation boundary;
- SKO-038 and SKO-040 proof outputs are deterministic across identical reruns.

### 6.4 Numerical comparison boundary

No generic claim such as “Epic 4 increased GVA by X%” or “reduced/increased employment by Y%” is supported by the current governed evidence because the accepted methodology has materially changed the classification, denominator, coverage and reporting boundaries.

Where a future client project has a frozen earlier output on the same supplier-spend rows, skopia can produce a proper baseline-versus-improved reconciliation using:

- row-level mapping changes;
- coefficient/source changes;
- denominator-route changes;
- fallback changes;
- mapped/unmapped spend changes;
- outcome deltas attributable to those changes.

That should be treated as a future application of the accepted methodology, not retroactively manufactured for this closeout.

## 7. Final pilot findings

The 35-observation cohort is sufficient to prove the methodology because it exercises:

- multiple countries;
- UK and Eurostat source routes;
- ordinary sectors and trade sectors;
- unresolved activity;
- exact coefficients and one explicit approved environmental fallback;
- non-EUR absolute-outcome FX handling;
- direct economic and direct environmental attribution;
- FIGARO exact and governed-equivalence mapping;
- trade valuation hold-outs;
- deterministic evidence composition.

The key portfolio results are:

### Direct full 27-observation boundary

- spend: **€22,102,562.52**;
- spend coverage: **88.14%**;
- GVA: **€10,340,270.829461...**;
- employment: **120.803696... persons-equivalent**;
- direct GHG: **387.8432903074818 tCO2e**.

### FIGARO-eligible 23-observation boundary

- spend: **€19,873,893.55**;
- spend coverage: **79.25%**;
- direct GVA: **€10,028,664.6633803**;
- indirect GVA: **€9,013,276.88421664**;
- combined GVA: **€19,041,941.5475969**;
- direct GHG: **376.629940299729 tCO2e**;
- indirect GHG: **1,880.13257429721 tCO2e**;
- combined GHG: **2,256.76251459694 tCO2e**.

The full 27-observation direct totals must not be added to the 23-observation indirect totals. Combined figures are valid only on the same FIGARO-eligible boundary.

## 8. Client-ready claim boundaries

### 8.1 Supported wording

With coverage and method disclosure, the accepted methodology supports claims such as:

- “modelled direct gross value added associated with analysed procurement spend”;
- “modelled direct employment persons-equivalent associated with analysed procurement spend”;
- “modelled direct production GHG emissions associated with analysed procurement spend”;
- “modelled upstream GVA / GHG associated with analysed purchases”;
- “combined direct and upstream modelled effects on the same eligible observation boundary”.

### 8.2 Claims that remain out of scope

The methodology must not imply that:

- procurement spend is literally equivalent to producer output;
- employment persons-equivalent is verified supplier headcount or jobs created;
- direct GHG is a supplier-specific audited carbon footprint;
- FIGARO identifies actual named Tier-2 or Tier-3 suppliers;
- direct or upstream modelled values are causal impacts created by the client;
- generic economic coefficients establish supplier-specific social outcomes;
- social-economy status changes calculation eligibility;
- upstream networks have been deduplicated;
- trade indirect effects are covered where the accepted valuation hold-out applies.

## 9. Whole-spend scaling pathway

The methodology is designed to extend beyond the current social-economy-derived test cohort to ordinary client procurement files.

A production-scale workflow should proceed through the following governed stages:

1. **Ingest and normalize procurement observations** — preserve client row/vendor key, supplier name, spend amount, currency and reporting period.
2. **Resolve supplier country** — use client fields first, then controlled enrichment from addresses, identifiers or canonical supplier evidence.
3. **Resolve purchased activity / NACE** — use procurement category, PO/invoice/service descriptions, supplier evidence and governed research; do not force unresolved cases.
4. **Materiality-led review** — prioritize ambiguous/unresolved rows by spend and reporting significance rather than seeking row-count perfection.
5. **Apply direct coefficients** — use accepted country × sector coefficients, denominator routes, FX and fallback controls.
6. **Produce direct portfolio outputs and QA** — show mapped/unmapped spend, outcome-specific coverage, fallbacks and hold-outs.
7. **Optionally apply FIGARO** — only to eligible observations under the accepted valuation/mapping rules.
8. **Report with bounded claims** — separate direct, indirect and combined same-boundary results.
9. **Apply optional overlays** — social-economy segmentation and separately verified supplier-specific social outcomes remain distinct from generic modelling.

The recommended next operational validation cohort is approximately **250–500 ordinary supplier-spend observations**, selected independently of social-economy status and deliberately retaining incomplete or difficult records.

That larger run is a scaling validation and productisation step. It is not required to prove the methodology already demonstrated by SKO-040.

## 10. Remaining limitations and deferred items

The following are explicit accepted limitations or follow-ons rather than hidden failures:

- eight of the 35 pilot observations remain unresolved for purchased activity/NACE;
- four resolved trade observations remain held out from primary FIGARO attribution;
- indirect employment remains deferred pending a governed employment satellite;
- the 35-observation pilot has no populated `spend_year` values, although the output contract preserves the field;
- wider UK sector coverage remains dependent on future source expansion;
- outcome-specific direct coverage varies beyond the headline GVA/employment/GHG set;
- FIGARO upstream results use procurement spend as an explicit basic-price output proxy;
- FIGARO portfolio results are gross and not network-deduplicated;
- the current methodology proof does not establish universal country × sector coverage for every arbitrary client file;
- the current 35-observation cohort was selected from a social-economy-oriented pilot population, so a larger ordinary-procurement run remains desirable operational evidence of scale;
- supplier-specific verified social outcomes remain a separate evidence stream and are not yet integrated into the generic calculation layer.

None of these items invalidates the accepted supplier-agnostic calculation architecture. They define the current product boundary and future development priorities.

## 11. Historical evidence-status note

Several earlier evidence documents were written before later owner acceptance and therefore contain historical status lines such as “owner acceptance pending”. Those status lines are accurate snapshots of the state at the time those documents were produced and should not be retroactively rewritten.

For current programme status, the authoritative tracker and later acceptance records supersede those historical status labels. SKO-035, SKO-036, SKO-037, SKO-038, SKO-039 and SKO-040 are all accepted in the current programme record.

## 12. Epic 4 acceptance criteria

Epic 4 can be accepted when the owner is satisfied that the evidence demonstrates all of the following:

- [x] purchased activity / NACE is governed and evidence-backed;
- [x] direct economic coefficients are versioned and source-governed;
- [x] direct GHG coefficients are versioned and source-governed;
- [x] direct whole-spend attribution is implemented and deterministic;
- [x] mapped/unmapped spend and hold-out reasons are explicit;
- [x] denominator routing and FX handling are governed;
- [x] fallback use is explicit and bounded;
- [x] optional FIGARO upstream attribution is implemented and reconciled;
- [x] direct and indirect outcomes are combined only on the same eligible boundary;
- [x] supplier-agnostic methodology proof is complete and deterministic;
- [x] social-economy status is separate from calculation eligibility;
- [x] supplier-specific social outcomes remain a separate evidence overlay;
- [x] client-facing claim boundaries are documented;
- [x] a scaling pathway to larger ordinary procurement files is documented;
- [x] limitations and deferred items are explicit;
- [x] the absence of a genuinely like-for-like historic numerical baseline is disclosed rather than filled with a fabricated comparison;
- [ ] explicit owner acceptance of Epic 4 / E4.14.

## 13. Acceptance recommendation

On the evidence assembled from accepted SKO-035 through SKO-040, the technical and methodological criteria for Epic 4 closeout are met.

Epic 4 now has a reproducible supplier-agnostic spend-attribution architecture with governed activity resolution, direct economic and environmental coefficients, deterministic whole-spend attribution, optional FIGARO upstream extension, transparent coverage and failure behaviour, source/coefficient lineage, bounded client claims and a defined scaling pathway.

The recommended programme action is therefore to put **E4.14 / SKO-041 to the owner for explicit acceptance**.

Until that acceptance is given, SKO-041 remains **implemented for review, not accepted**, and Epic 4 remains **In progress**.
