# SKO-040 — Whole-spend methodology proof and scalable supplier-agnostic impact pilot

**Task:** SKO-040  
**Epic / milestone:** E4 / E4.13  
**Status:** Implemented and locally reconciled; deterministic rerun and owner review pending.  
**Branch:** `sko-040-whole-spend-methodology-proof`  
**Accepted technical baseline:** SKO-036, SKO-037, SKO-038 and SKO-039  

## 1. Purpose

SKO-040 demonstrates how the accepted skopia attribution architecture can support a client-facing whole-spend impact offer across ordinary procurement spend.

The core calculation chain is:

`procurement spend → supplier country → purchased activity / NACE → governed country × sector coefficient → modelled direct outcome → optional FIGARO upstream indirect outcome`

The calculation is supplier-agnostic. Social-economy status is not required to calculate direct or indirect economic/environmental effects.

Two additional analytical layers remain separate from the generic whole-spend calculation:

1. social-economy segmentation, for example total procurement impact versus the share associated with verified social-economy suppliers; and
2. supplier-specific verified social outcomes, which are evidence overlays and must not be blended into generic coefficient-based estimates.

SKO-040 does not rebuild or alter the accepted SKO-036–039 methodologies. Its role is to make the accepted end-to-end capability explicit, prove the calculation chain on the governed 35-observation pilot, define the minimum client input/enrichment contract, expose hold-outs and claim boundaries, and document the path to a larger ordinary procurement pilot.

## 2. Accepted architecture

### 2.1 Direct layer

SKO-038 is the accepted whole-spend direct attribution baseline.

The direct architecture is intentionally thin. It composes the accepted SKO-036 direct-economic and SKO-037 direct-GHG application layers rather than reconstructing coefficients.

For each procurement observation, the direct composer retains:

- supplier/client traceability;
- supplier country;
- spend and spend currency;
- spend year where available;
- purchased activity / NACE assertion;
- governed model-sector mapping;
- coefficient reference year;
- coefficient identifier and source lineage;
- denominator route and denominator currency;
- FX provenance where required;
- coefficient specificity and approved fallback status;
- attribution status and hold-out reason;
- modelled direct outcomes.

Accepted direct controls include:

- unresolved NACE remains held out;
- trade sectors use the governed trade-turnover denominator route;
- the accepted FR P85 environmental parent fallback remains explicit and direct-layer-only;
- social-economy fields do not determine calculation eligibility;
- coefficient methodology is not recomputed by the whole-spend composer.

### 2.2 Optional indirect layer

SKO-039 is the accepted optional FIGARO extension.

The accepted indirect chain is:

`procurement spend → supplier country → purchased activity / NACE → FIGARO country × A*64 industry shock → L-I upstream requirements → indirect GVA / GHG`

Accepted FIGARO controls are:

- FIGARO 2026 edition, reference year 2023;
- industry-by-industry A*64 table at basic prices;
- procurement spend treated as an explicit basic-price output proxy, not an equivalence claim;
- `L-I` only, with no induced household-consumption effects;
- accepted SKO-038 direct values are not recalculated;
- exact country × A*64 mapping only;
- governed exact equivalences may be used where representation differs but economic meaning is unchanged;
- no silent parent-sector fallback;
- G45/G46/G47 trade procurement remains held out of primary FIGARO attribution because purchase value is not comparable with margin-based trade-industry output;
- portfolio aggregation is gross / observation-based and not network-deduplicated;
- primary indirect outcomes are GVA and GHG;
- indirect employment remains secondary/deferred.

Required portfolio wording:

> Gross modelled upstream activity associated with the analysed purchases; supplier-level upstream requirements may overlap and are not deduplicated across the procurement portfolio.

## 3. Minimum client input contract

The minimum viable client input should contain enough information to establish a unique procurement observation, spend value and a credible supplier-country/activity resolution path.

### 3.1 Required client fields

| Field | Requirement | Purpose |
| --- | --- | --- |
| `supplier_record_id` or equivalent stable row/vendor key | Required | Preserves traceability and enables deterministic reconciliation |
| `supplier_name` | Required | Supplier resolution and enrichment |
| `spend_amount` | Required | Calculation base |
| `spend_currency` | Required | Currency normalization and denominator compatibility |
| `reporting_period` or spend year | Required for production use | Separates procurement period from coefficient reference year |
| supplier country or evidence from which country can be derived | Required at calculation gate | Country is part of every governed coefficient / FIGARO node |

A client does not need to supply a verified NACE code, canonical skopia entity ID or social-economy classification for the whole-spend method to operate. Those may be supplied by the client where available, but they are enrichment fields rather than calculation prerequisites at ingestion.

### 3.2 Strongly recommended client fields

The following materially improve purchased-activity resolution and should be requested where available:

- procurement category;
- category hierarchy;
- commodity or UNSPSC-like code;
- purchase-order description;
- invoice line or service description;
- cost centre;
- contract title / service family;
- supplier address / postcode;
- tax or registration identifier.

These fields improve enrichment quality but should not be silently substituted for evidence where their meaning is ambiguous.

## 4. Enrichment and derivation stages

A production-scale client file should pass through explicit staged gates.

### Stage A — observation and spend normalization

Preserve the original client row key and supplier name. Normalize spend amount and currency without overwriting source fields. Separate spend period from coefficient reference year.

Possible hold-outs include:

- missing or invalid spend;
- unsupported or unresolved currency;
- duplicate observation key where uniqueness is required.

### Stage B — supplier-country resolution

Use client country fields first where reliable, then controlled enrichment from addresses, registration identifiers or canonical supplier evidence.

Country must not be silently inferred from a supplier name alone.

Possible states should include at minimum:

- supplied / accepted;
- derived / accepted;
- ambiguous;
- missing / held out.

### Stage C — purchased activity / NACE resolution

Resolve what the client purchased, not simply the supplier's broad corporate activity.

Use the accepted SKO-035 purchased-activity architecture and preserve:

- proposed NACE code;
- level / specificity;
- description;
- evidence source;
- treatment / review status.

Where purchased activity cannot be resolved defensibly, the row remains a governed hold-out rather than receiving an arbitrary sector.

### Stage D — direct coefficient mapping

Map supplier country × governed model sector to the accepted direct economic and environmental coefficient layers.

Expose:

- model sector;
- coefficient reference year;
- coefficient ID;
- denominator route;
- coefficient source sector;
- specificity / fallback status;
- FX provenance;
- attribution status and reason.

### Stage E — direct outcome calculation

The currently accepted direct outcome set includes:

- gross value added;
- employment persons-equivalent;
- intermediate consumption;
- compensation of employees;
- gross operating surplus / mixed income;
- consumption of fixed capital;
- production taxes less subsidies;
- gross fixed capital formation;
- employment hours;
- trade-route labour costs where applicable;
- direct GHG emissions.

Outcome availability is governed by source and denominator coverage; `not_applicable` must remain distinct from `held_out`.

A direct observation may be `partially_modelled` where headline outcomes are available but one or more secondary outcomes is unavailable (for example because a numerator is absent). `partially_modelled` therefore does not mean that the whole observation has failed attribution.

### Stage F — optional FIGARO mapping and upstream attribution

For eligible observations only, map the supplier country and model sector to the governed FIGARO node.

Expose separately:

- exact or governed-exact mapping status;
- FIGARO country;
- FIGARO sector;
- valuation case;
- eligibility;
- hold-out reason;
- indirect GVA;
- indirect GHG;
- combined direct + indirect value on the same eligible observation boundary;
- country × sector contribution lineage.

## 5. Coverage framework

Client reporting should not use one undifferentiated coverage percentage. Coverage should be shown as a sequence of gates, with both observation-count and spend-weighted coverage where useful.

Recommended stages are:

1. supplier / country resolved;
2. purchased activity / NACE resolved;
3. direct coefficient-covered;
4. direct outcome-covered by outcome;
5. FIGARO mapped and valuation-eligible;
6. indirect outcome-covered by outcome.

Every unresolved row should retain a machine-readable reason.

Typical reasons include:

- `country_missing`;
- `nace_code_missing`;
- country × sector coefficient unavailable;
- missing FX;
- approved direct fallback unavailable;
- `figaro_country_sector_node_missing`;
- `trade_sector_primary_case_valuation_holdout`;
- satellite coverage missing for upstream nodes.

A lower coverage rate is preferable to silent approximation where the missing step would materially change the claim.

## 6. Accepted pilot evidence

### 6.1 Direct whole-spend pilot — SKO-038

Governed cohort:

- total observations: **35**;
- total spend: **€25,077,933.78**;
- observations with at least one modelled direct outcome: **27**;
- direct held-out observations: **8**;
- direct modelled spend: **€22,102,562.52**;
- direct spend coverage: **88.14%**;
- all eight direct hold-outs are unresolved-NACE (`nace_code_missing`) cases.

Headline direct outcomes:

| Outcome | Accepted result |
| --- | ---: |
| Gross value added | €10,340,270.8295 |
| Employment | 120.803696 persons-equivalent |
| Direct GHG | 387.8432903074818 tCO2e |

The direct composer also produced 385 observation/outcome ledger rows and 11 portfolio outcomes, with coefficient lineage complete, zero duplicate observation/outcome rows, trade-route invariants passed, fallback invariants passed and portfolio reconciliation passed.

### 6.2 FIGARO upstream pilot — SKO-039

On the same 35-observation cohort:

- mapped eligible observations: **23**;
- held-out observations: **12**;
- unresolved-NACE hold-outs: **8**;
- trade valuation hold-outs: **4**;
- unexplained mapping losses: **0**;
- mapped eligible spend: **€19,873,893.55**;
- mapped eligible spend coverage: **79.25%**.

On the same 23-observation FIGARO-eligible boundary:

| Outcome | Accepted direct | FIGARO indirect | Combined |
| --- | ---: | ---: | ---: |
| GVA | €10,028,664.6633803 | €9,013,276.88421664 | €19,041,941.5475969 |
| GHG | 376.629940299729 tCO2e | 1,880.13257429721 tCO2e | 2,256.76251459694 tCO2e |

These direct values are the accepted SKO-038 direct results for the same FIGARO-eligible observation boundary; they are not a recalculation of the direct model.

### 6.3 SKO-040 composed methodology proof — reconciled 2026-09-10

The local proof outputs were generated from the accepted SKO-038 and SKO-039 outputs without recalculating the accepted direct results or rebuilding the indirect methodology:

- `data/pilots/sko-040/whole_spend_methodology_proof.csv`
- `data/pilots/sko-040/whole_spend_methodology_proof_summary.json`

The composed proof reconciled all governed coverage boundaries:

| Gate | Observations | Spend | Spend coverage |
| --- | ---: | ---: | ---: |
| Input / country present | 35 / 35 | €25,077,933.78 | 100% |
| NACE resolved | 27 / 35 | €22,102,562.52 | 88.14% |
| Direct modelled | 27 / 35 | €22,102,562.52 | 88.14% |
| FIGARO eligible | 23 / 35 | €19,873,893.55 | 79.25% |

All extended reconciliation gates passed, including:

- 35 input observations and total spend;
- 27 NACE-resolved / direct-modelled observations;
- 23 FIGARO-eligible observations;
- eight unresolved-NACE hold-outs;
- four trade valuation hold-outs;
- full direct headline GVA, employment and GHG totals;
- FIGARO-eligible-boundary direct GVA/GHG;
- indirect GVA/GHG;
- combined GVA/GHG on the same 23-observation boundary.

The proof preserves exact direct coefficient IDs, denominator routes, direct coefficient specificity, FIGARO mapping version, valuation status and mapping reason. It visibly retains the accepted FR P85 environmental `approved_parent_fallback`, the governed UK→GB FIGARO exact-equivalence treatment, and the `trade_turnover` denominator for resolved trade observations.

Proof CSV SHA256:

`9802db909515c77c9bbe19354b8cf79e483a9328aaa5ee4ee7b6f0d49016277b`

The generated summary status is `methodology_proof_generated_and_reconciled_not_accepted`.

## 7. Direct and indirect capability matrix

| Outcome / capability | Direct | Indirect FIGARO | Current reporting position |
| --- | --- | --- | --- |
| Gross value added | Accepted | Accepted | Client-ready as modelled estimate with method/coverage disclosure |
| Employment persons-equivalent | Accepted | Deferred | Direct client-ready as modelled estimate; indirect not primary |
| Employment hours | Accepted where coefficient available | Deferred | Direct analytical output |
| Compensation of employees | Accepted where coefficient available | Not primary | Direct analytical output |
| Operating surplus / mixed income | Accepted where coefficient available | Not primary | Direct analytical output |
| Intermediate consumption | Accepted where coefficient available | Not primary | Direct analytical output |
| Fixed capital formation / consumption | Accepted where coefficient available | Not primary | Direct analytical output |
| Production taxes less subsidies | Accepted where coefficient available | Not primary | Direct analytical output |
| GHG emissions | Accepted | Accepted | Client-ready as modelled estimate with direct/upstream separation |
| Supplier-specific social outcomes | Separate evidence overlay | N/A | Report only where separately evidenced |
| Social-economy segmentation | Optional segmentation | Optional segmentation | Classification overlay, not calculation requirement |

## 8. Claim boundaries

### 8.1 Permitted claims

Where coverage and lineage are disclosed, the methodology supports wording such as:

- “modelled direct gross value added associated with analysed procurement spend”;
- “modelled direct employment persons-equivalent associated with analysed procurement spend”;
- “modelled direct production GHG emissions associated with analysed procurement spend”;
- “modelled upstream GVA / GHG associated with analysed purchases”;
- “combined direct and upstream modelled effects on the same eligible observation boundary”.

### 8.2 Claims that must remain bounded

The methodology must not imply that:

- procurement spend is literally equivalent to producer output;
- modelled employment persons-equivalent represents verified new jobs created;
- modelled GHG is a supplier-specific audited carbon footprint;
- upstream FIGARO results identify actual named Tier-2/Tier-3 suppliers;
- direct and upstream outcomes are causal impacts created by the client;
- social-economy status changes the generic coefficient calculation;
- supplier-specific social outcomes can be inferred from generic economic coefficients;
- portfolio-level upstream supplier networks have been deduplicated;
- trade-sector indirect effects are covered where the accepted valuation hold-out applies.

### 8.3 Direct / indirect distinction

Direct outcomes represent modelled effects associated with the supplier country × purchased-activity production layer under the accepted coefficient methodology.

Indirect outcomes represent modelled upstream supply-chain requirements generated through FIGARO `L-I` and must be reported separately before any combined presentation.

### 8.4 Social-economy distinction

Social-economy status is a classification and analytical segmentation. It may be used to compare:

- total procurement spend / impact;
- spend / impact associated with verified social-economy suppliers.

It must not be presented as an eligibility requirement for the whole-spend calculation.

### 8.5 Supplier-specific impact distinction

Supplier-provided or independently verified social/environmental outputs belong in a separate evidence layer. They may complement the generic whole-spend estimates but should not be mathematically added to GVA, employment persons-equivalent or generic GHG estimates unless a separately governed methodology explicitly supports that operation.

## 9. Client-facing output structure

A client-facing whole-spend report or dashboard should separate four analytical panels.

### Panel 1 — portfolio coverage and data quality

Show:

- spend analysed;
- observations analysed;
- country resolution rate;
- NACE/activity resolution rate;
- direct model coverage by spend and observation;
- FIGARO eligibility by spend and observation;
- hold-out reasons;
- coefficient/fallback mix.

### Panel 2 — direct economic and environmental effects

Show direct outcomes first, with clear modelled-estimate language and coefficient reference year.

Recommended headline outputs are:

- direct GVA;
- direct employment persons-equivalent;
- direct GHG.

Additional accepted direct economic outcomes may sit below the headline layer or in technical appendices depending on client need.

### Panel 3 — optional upstream indirect effects

Show FIGARO indirect GVA and GHG separately from direct effects, followed by combined values only where the same observation boundary is used.

The required gross / non-network-deduplicated wording must accompany the portfolio view.

### Panel 4 — optional overlays

Keep separate:

- social-economy segmentation;
- supplier-specific verified outputs / outcomes;
- geographic or policy context;
- client-specific opportunity analysis.

These enrich interpretation but do not alter the generic direct / indirect calculation.

## 10. SKO-040 governed proof-output contract

The final SKO-040 evidence is generated locally from accepted SKO-038 and SKO-039 outputs without copying live client rows into committed repository documentation.

Local output directory:

`data/pilots/sko-040/`

Files:

- `whole_spend_methodology_proof.csv`
- `whole_spend_methodology_proof_summary.json`

The CSV contains one row per governed pilot observation, joined deterministically by `selection_id`, with observation identity and spend, country, spend year, proposed NACE and governed model sector, direct mapping/hold-out status, direct headline outcomes, denominator route and coefficient specificity, FIGARO country/sector, FIGARO mapping status/reason, valuation case and eligibility, indirect/combined headline outcomes and lineage references.

The JSON summary records input and gate coverage, hold-outs, direct and indirect headline totals, fallback/specificity and denominator-route mixes, claim-boundary text and input/proof fingerprints.

No accepted SKO-036–039 result is recalculated merely to populate this proof. The proof reconciles and exposes the accepted outputs.

## 11. Scaling pathway to a larger ordinary procurement dataset

The current 35-observation pilot proves the architecture but is not intended to estimate production-scale client coverage.

The next validation pilot should use an ordinary client procurement extract selected independently of social-economy status.

Recommended initial scale:

- approximately **250–500 supplier-spend observations**;
- several countries;
- multiple procurement categories;
- multiple spend bands;
- deliberate retention of incomplete / low-quality rows rather than pre-cleaning all failures away.

The principal validation questions should be:

1. What proportion of observations and spend has a defensible supplier country?
2. What proportion has a defensible purchased-activity / NACE assignment?
3. What proportion reaches direct coefficient coverage?
4. What proportion reaches FIGARO eligibility?
5. Which failure reasons dominate by count and spend?
6. Which client fields materially improve NACE/activity resolution?
7. Which repeated manual enrichment steps should become a later governed adapter or workflow capability?
8. Can the output structure remain deterministic and auditable at this larger scale?

Success should be defined by transparent coverage and governed failure behaviour, not by maximising the absolute amount of modelled spend.

## 12. Known limitations and open items

- The governed 35-observation cohort has no populated `spend_year` values. The accepted SKO-038 output contract preserves the field, but the current live pilot does not demonstrate populated spend-year provenance.
- Eight current pilot observations remain held out because NACE is unresolved.
- Four resolved trade observations remain held out from primary FIGARO attribution under the accepted valuation policy.
- Some direct observations are `partially_modelled` because one or more secondary economic outcomes have unavailable numerators; this does not prevent available headline outcomes from being modelled.
- Indirect employment is deferred until a governed employment satellite with defensible coverage exists.
- FIGARO upstream portfolio results are gross and not network-deduplicated.
- The current pilot was selected from a social-economy cohort, but social-economy status is outside the calculation gate. A larger ordinary-procurement validation is still needed to evidence operational scaling beyond the test cohort.
- The current pilot proves methodology compatibility and controlled hold-outs; it does not establish universal country × sector coverage for arbitrary client portfolios.

## 13. Acceptance boundary

This document and the reconciled local proof do not by themselves make SKO-040 complete or accepted.

SKO-040 should remain in progress until:

1. the governed SKO-040 proof CSV and summary JSON are generated from the accepted SKO-038 and SKO-039 outputs — **done**;
2. totals, coverage and hold-outs reconcile to accepted evidence — **done**;
3. the proof exposes the required lineage / fallback / valuation information — **done**;
4. deterministic behaviour is checked where practical — **pending**;
5. owner review confirms that the methodology proof and scaling pathway are sufficient for E4.13 — **pending**.

No change to SKO-036–039 methodology is required unless a genuine incompatibility is discovered during proof generation.

## 14. Current implementation state

### Work actually completed

- SKO-040 methodology/capability proof documented.
- Minimum client input contract documented.
- Enrichment and hold-out stages documented.
- Direct / indirect capability matrix documented.
- Accepted 35-observation direct and FIGARO pilot evidence consolidated.
- Claim boundaries and client-facing output structure documented.
- Local one-row-per-observation methodology proof generated.
- Extended summary generated with coverage gates, headline outcome reconciliation, hold-outs, lineage/specificity mixes and fingerprints.
- All extended reconciliation gates passed.
- Larger ordinary-procurement validation pathway documented.

### Evidence created

- `docs/sko-040-whole-spend-methodology-proof.md`
- `data/pilots/sko-040/whole_spend_methodology_proof.csv`
- `data/pilots/sko-040/whole_spend_methodology_proof_summary.json`
- proof CSV SHA256 `9802db909515c77c9bbe19354b8cf79e483a9328aaa5ee4ee7b6f0d49016277b`
- behaviour evidence showing 35/35 country-present, 27/35 NACE/direct-modelled, 23/35 FIGARO-eligible, eight unresolved-NACE hold-outs, four trade hold-outs and complete headline-outcome reconciliation.

### Decisions made

- No new modelling engine is required for SKO-040; accepted SKO-038 and SKO-039 outputs can be composed directly.
- A larger 250–500-row ordinary procurement pilot is the next scaling validation, not a prerequisite for the current 35-row methodology proof.
- `partially_modelled` is interpreted at outcome coverage level, not as whole-observation failure.
- Social-economy status remains an optional segmentation and supplier-specific verified social outcomes remain a separate evidence overlay.

### Status changes

- SKO-040 moved from documentation-only implementation to **implemented and locally reconciled**.
- SKO-040 / E4.13 are **not accepted** pending deterministic rerun and explicit owner review.

### Unresolved items

- Deterministic rerun / byte-level or hash-level comparison of SKO-040 local proof outputs.
- Explicit owner acceptance of the methodology proof and scaling pathway.
- Later ordinary-procurement scaling validation.

### Recommended next task

Run the SKO-040 proof composition a second time with identical accepted inputs and confirm the proof CSV and summary JSON are unchanged. If deterministic, present the final E4.13 evidence for owner acceptance before moving to SKO-041 / E4.14 closeout.

**Current SKO-040 status:** In progress — implemented and locally reconciled; deterministic rerun and owner acceptance pending.
