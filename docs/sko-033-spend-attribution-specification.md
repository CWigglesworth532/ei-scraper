# SKO-033 — Spend-attribution baseline reconciliation and direct impact-model specification

**Epic:** E4 — Socioeconomic and environmental supply-chain intelligence
**Task:** SKO-033 — Spend-attribution baseline reconciliation and impact-model specification
**Document status:** Implemented as specification evidence for owner review; not yet accepted
**Date:** 19 August 2026
**Repository baseline:** `2334e88f794c27d1e5921c51b464372e32183644` before the local unaccepted SKO-030 QA correction
**Change type:** Additive documentation/specification only

## 1. Purpose

SKO-033 reconstructs the earlier exploratory “hidden impact” calculation as far as surviving evidence permits and freezes the governed specification for future direct spend-attribution work.

The objective is not to build the coefficient layer or attribution engine in this task.

The agreed target architecture is:

```text
accepted supplier observation
    → actual client spend
    → supplier country
    → evidence-backed purchased economic activity / NACE sector
    → versioned country × sector coefficient
    → modelled direct attributable outcome
```

This specification keeps modelled economic and environmental attribution separate from supplier-specific verified social-impact evidence such as social mission, beneficiary groups, work-integration characteristics or actual supplier-reported outputs and outcomes.

Direct effects come first. FIGARO direct+indirect supply-chain modelling remains a later task.

## 2. Governing principles

SKO-033 applies the following accepted programme principles:

1. Matching is not classification.
2. Source evidence is not automatically policy eligibility.
3. Canonical materialisation remains demand-driven.
4. Impact attribution must not create or alter canonical identity.
5. Regional context is enabling infrastructure, not the target output.
6. Procurement category is not equivalent to economic activity or NACE.
7. Sector determination is an evidence problem before it is a coefficient problem.
8. The more a supplier can move the headline result, the stronger the sector evidence required.
9. Supplier-specific social-impact evidence must remain separate from modelled economic/environmental attribution.
10. Estimates must not be presented as audited supplier outcomes or causal impacts without separate evidence.

## 3. Historical hidden-impact MVP reconstruction

### 3.1 Recovered headline baseline

The earlier GPT-produced exploratory calculation used approximately **EUR 54.34 million** of supplier expenditure and produced:

| Outcome | Historical result |
|---|---:|
| Direct GVA supported | ~EUR 31.13m |
| Labour income supported | ~EUR 22.16m |
| Employment supported | ~809 economic employment-supported roles |
| Production taxes supported | ~EUR 1.53m |

The historical model was exploratory. It was not the final published Hidden Impact reporting methodology.

### 3.2 Arithmetically implied weighted intensities

The four headline results imply the following portfolio-weighted intensities:

| Outcome | Implied weighted intensity |
|---|---:|
| GVA / spend | 57.29% |
| Labour income / spend | 40.78% |
| Employment / spend | 14.89 roles per EUR 1m |
| Production taxes / spend | 2.82% |

Additional implied relationships are:

- labour income / modelled GVA ≈ 71.2%;
- production taxes / modelled GVA ≈ 4.9%;
- expenditure per employment-supported role ≈ EUR 67,170.

These values are arithmetic reconciliation targets only. They are **not evidence that these exact aggregate ratios were hard-coded into the original model**.

### 3.3 Reconstructed historical architecture

The surviving evidence supports the following interpretation of the MVP:

```text
supplier spend
    ↓
client/procurement category
    ↓
heuristic category → broad economic sector / NACE bridge
    ↓
provisional sector economic coefficient
    ↓
spend × coefficient
    ↓
GVA / labour income / employment / production tax
    ↓
portfolio aggregation
```

The intended statistical family was Eurostat national-accounts activity data, including work discussed around sector economic aggregates and employment. However, the exact coefficient extraction, table versions, coefficient year and complete mapping used to generate the original four figures have not been recovered.

The historical coefficient provenance must therefore be recorded conservatively as:

> GPT-generated provisional economic coefficients intended to approximate sector-level national-accounts relationships; exact source extraction and coefficient set not recovered.

It must not be restated as a fully evidenced Eurostat coefficient model unless later evidence establishes that.

### 3.4 Known historical weakness

The central weakness was the bridge between broad procurement categories and economic sectors.

Broad buyer-oriented categories may contain materially different underlying activities. For example:

- facilities may include cleaning, catering, security, property management or maintenance;
- digital/IT may represent software, consulting, hardware wholesale, device repair or refurbishment;
- professional services may include legal, accounting, recruitment, consultancy or technical services;
- food may represent manufacturing, wholesale, catering or hospitality;
- circular-economy suppliers may operate in wholesale, repair, waste management or manufacturing.

Therefore an economically precise coefficient can still produce a poor estimate if the purchased activity has been assigned to the wrong sector.

### 3.5 Later proportional scaling

Later work scaled the original four headline figures proportionately when the spend denominator increased, rather than rerunning a fresh supplier-by-sector model.

That approach implicitly assumes unchanged sector composition and is acceptable only as rough illustrative extrapolation.

It is not part of the governed SKO-033 specification.

## 4. Direct attribution boundary

SKO-033 governs **direct spend-attributable economic and environmental estimates** only.

The modelled direct outcomes may include, where defensible:

- GVA supported;
- labour income / compensation supported;
- employment or FTE-equivalent supported;
- production taxes supported;
- GHG emissions associated with expenditure.

The following remain separate evidence layers:

- supplier social mission;
- beneficiary groups;
- work-integration characteristics;
- social-economy legal or policy status;
- supplier-reported outputs;
- supplier-reported outcomes;
- case-study evidence;
- actual contract-specific environmental performance.

The attribution model must not convert supplier social classification into economic or environmental coefficients.

## 5. Sector determination is the primary evidence gate

A supplier must not be assigned to an economic sector from procurement category alone where stronger evidence is available.

The assigned sector should represent the **economic activity associated with the client spend**, not merely the supplier’s broad corporate identity.

### 5.1 Evidence hierarchy

Preferred evidence order:

1. **Contract-specific description, purchase description, statement of work or service-line evidence**
2. **Supplier website evidence identifying the relevant purchased activity**
3. **Official register or authoritative company activity classification**
4. **Other strong business-activity evidence**
5. **Client procurement category as supporting evidence**
6. **Broad inferred fallback where no better evidence exists**

Evidence sources may support or conflict with one another. No single evidence source automatically overrides all others.

### 5.2 Activity versus supplier identity

A diversified supplier may perform many activities.

The correct question is:

> What economic activity is the client expenditure paying for?

not:

> What is the supplier’s single overall industry?

Where contract-specific evidence identifies the purchased activity, that activity should control the attribution even if the supplier has several other material business lines.

## 6. Required sector-assignment record

Each attributed supplier-spend observation should retain at least:

- `subject_type`;
- `subject_id`;
- `entity_id` where available;
- client identifier;
- original spend amount;
- original currency;
- reporting spend amount;
- spend period/year;
- supplier country;
- assigned NACE scheme/version;
- assigned NACE code;
- assigned NACE level;
- sector description;
- purchased-activity description;
- evidence source(s);
- evidence date(s);
- assignment method;
- assignment confidence;
- review status;
- reviewer where applicable;
- `contract_specific` flag;
- `multi_activity` flag;
- `spend_split_applied` flag;
- allocation share where applicable;
- rationale/notes;
- provenance sufficient to reproduce the decision.

The assignment must remain separate from canonical entity identity and social-economy classification.

## 7. Sector-assignment confidence

Use three operational confidence levels initially.

### High

Use where the purchased activity is clearly evidenced by:

- contract or service-line evidence; or
- strong supplier/official activity evidence that aligns unambiguously to the client expenditure.

### Medium

Use where:

- supplier activity is clear but contract-specific evidence is absent; or
- several closely related sector assignments remain plausible.

### Low

Use where assignment relies mainly on:

- broad procurement category;
- generic supplier description;
- weak contextual evidence; or
- a broad fallback.

Low-confidence assignments may remain analytically useful, but they must not be treated as equivalent to high-confidence assignments in headline reporting.

## 8. Proportional assurance and materiality

Evidence requirements increase with the supplier’s ability to change the headline result.

Governance principle:

> **The more a supplier can move the portfolio estimate, the stronger the sector evidence required.**

The model must expose concentration rather than hide it.

At minimum, later implementation should support:

- supplier contribution to total analysed spend;
- supplier contribution to each modelled outcome;
- sector-assignment confidence;
- coefficient fallback status;
- identification of outcome-dominant suppliers.

A high-spend observation with weak sector evidence must be escalated for stronger review, subjected to sensitivity analysis, or excluded from precise headline estimates.

SKO-033 does not freeze one universal monetary threshold. Materiality may be set for the pilot or client context.

## 9. Multi-activity suppliers

Do not automatically force a diversified supplier into a single sector.

Use one of three treatments.

### A. Single-sector attribution

Use where evidence clearly identifies the purchased activity.

### B. Evidence-backed spend split

Use where evidence supports a defensible allocation across activities.

Example:

```text
60% cleaning
40% catering
```

Each component receives the coefficient for its evidenced activity.

No percentage split may be invented merely to increase modelling sophistication.

### C. Unresolved or broader attribution

Use where the activity mix cannot be defensibly separated.

The observation should then:

- use a justified broader sector if one exists; or
- remain unresolved / low-confidence; or
- be excluded from the most precise headline estimate.

False precision is not an acceptable substitute for missing evidence.

## 10. Coefficient contract

Future economic and environmental coefficient records should be versioned at the grain:

```text
country × economic sector × reference year × outcome
```

where source data supports that grain.

Each coefficient record should retain:

- coefficient ID;
- dataset/source;
- dataset version;
- source table;
- source reference/URL where appropriate;
- retrieval date;
- source file or extract hash where practical;
- reference year;
- country;
- NACE scheme/version;
- NACE code/aggregation;
- outcome type;
- numerator measure;
- denominator measure;
- coefficient value;
- coefficient unit;
- source currency where relevant;
- price basis where relevant;
- transformation method;
- fallback level;
- quality/confidence status;
- notes.

Initial direct coefficient families are expected to include:

- GVA/output;
- compensation of employees/output;
- employment/output;
- production taxes/output;
- GHG emissions/output or expenditure.

SKO-036 will implement the direct economic coefficient layer.

SKO-037 will implement the environmental intensity layer.

## 11. Year alignment

The model must preserve separately:

- spend year;
- coefficient reference year;
- calculation/reporting year.

Preferred coefficient alignment:

1. same-year coefficient where available;
2. nearest defensible prior year;
3. documented fallback.

The system must not silently represent an older coefficient as contemporary.

Any inflation or price-basis adjustment must be explicit, reproducible and justified by the denominator used in the coefficient.

No automatic inflation adjustment is assumed by SKO-033.

## 12. Currency treatment

The model must retain:

- original spend;
- original currency;
- converted reporting spend;
- FX rate where conversion occurs;
- FX source;
- FX date or averaging convention;
- reporting currency.

European reporting is expected normally to aggregate in EUR, but the implementation must not destroy original-currency traceability.

Where a coefficient is a same-currency ratio, currency may mathematically cancel within the coefficient. This does not remove the need for consistent reporting-currency treatment of client expenditure.

## 13. Missing coefficient fallback

Fallbacks must be explicit and governed.

Preferred hierarchy:

1. exact country × sector;
2. country × broader sector aggregation;
3. EU or governed peer aggregate × sector;
4. country-wide economy coefficient;
5. unresolved / no estimate.

Every fallback must:

- be recorded;
- reduce confidence where appropriate;
- remain visible in output;
- be reproducible.

No hidden substitution is permitted.

## 14. Direct outcome formulas

Conceptually:

### GVA supported

```text
eligible spend × GVA/output coefficient
```

### Labour income supported

```text
eligible spend × compensation-of-employees/output coefficient
```

### Employment supported

```text
eligible spend × employment/output coefficient
```

The employment unit must be explicit, for example FTE-equivalent or persons per EUR million of output.

### Production taxes supported

```text
eligible spend × production-taxes/output coefficient
```

### GHG emissions associated with expenditure

```text
eligible spend × emissions/output or emissions/expenditure coefficient
```

The implementation must record coefficient units rather than rely on implied scale.

## 15. Confidence propagation

Impact confidence depends on more than coefficient quality.

At minimum, later attribution should consider:

- sector-assignment confidence;
- coefficient quality;
- coefficient fallback;
- year alignment;
- spend-data quality;
- allocation/split quality where relevant.

A high-quality coefficient applied to a weak activity assignment remains a weak estimate.

The pilot does not require a complex probabilistic scoring system. A transparent governed confidence classification is preferred to false mathematical precision.

## 16. Sensitivity analysis

Sensitivity analysis is required where a materially significant observation has more than one plausible sector assignment.

For example:

```text
Supplier A → sector X
versus
Supplier A → sector Y
```

The model should calculate the effect on the relevant headline estimate where practical.

Sensitivity should be used to quantify uncertainty caused by sector ambiguity instead of applying arbitrary blanket confidence bands.

Portfolio reporting should identify whether headline conclusions are robust to plausible alternative assignments for material suppliers.

## 17. Permissible reporting language

### Estimated

Appropriate for model-derived outputs.

Preferred example:

> Estimated direct GVA associated with the analysed supplier expenditure.

### Supported

Permissible for economic activity where the modelled nature is clear.

Preferred example:

> The analysed spend is estimated to support EUR X of direct GVA and approximately Y FTE-equivalent roles.

Do not state that jobs were “created” unless separate evidence supports actual new roles.

### Associated with

Preferred default for environmental estimates and generally conservative model wording.

Preferred example:

> An estimated X tCO2e is associated with the expenditure analysed.

### Attributable

Use only with an explicit model qualifier, for example:

> modelled spend-attributable direct outcome

Do not imply causal additionality from proportional attribution.

## 18. Prohibited or unsupported claims

Without separate evidence, do not describe modelled estimates as:

- audited supplier impact;
- actual supplier GVA generated by the client contract;
- jobs created;
- causal carbon emissions caused by the client;
- supplier-reported outcomes;
- social impact experienced by beneficiaries;
- Telos-created historic impact.

The model must distinguish identification, attribution and causation.

## 19. Pilot output structure

The eventual deep pilot for 30–40 suppliers should be capable of presenting, for each supplier:

```text
supplier
→ client spend
→ purchased activity
→ evidence
→ NACE assignment
→ sector confidence
→ direct GVA
→ labour income
→ employment
→ production taxes
→ GHG
→ separate verified supplier-specific social-impact evidence
```

Portfolio reporting should include:

- total analysed spend;
- spend coverage by sector-assignment confidence;
- spend excluded or unresolved;
- direct outcome estimates;
- share of estimates derived from high/medium/low-confidence assignments;
- top outcome contributors;
- coefficient fallback coverage;
- sensitivity to material uncertain assignments;
- separate supplier-specific social evidence.

## 20. SKO-033 acceptance-question assessment

### Can the earlier four headline figures be reproduced/explained?

**Partially reconstructed and substantially explained.**

The spend denominator, headline results, direct spend-attribution architecture and implied aggregate intensities are recovered.

Exact original sector assignments and coefficient table have not been recovered, so exact byte/row-level reproduction has not been demonstrated.

This limitation must remain explicit.

### Can the datasets, coefficient years, denominators and mappings that drove them be identified?

**Only partially.**

The intended national-accounts style denominator and sector-ratio logic are reconstructed. Exact historical coefficient values, exact extraction, coefficient year and complete mapping are not recovered.

The governed replacement now requires explicit provenance, year, numerator, denominator, units and fallback status.

### Can broad or incorrect spend-category assumptions be identified as a source of error?

**Yes.**

The historical category→sector bridge is the primary identified methodological weakness.

Future work will quantify its effect through evidence-led sector determination and materiality-led sensitivity analysis.

### Is there an evidence standard for assigning NACE/economic sector?

**Yes.**

The standard is defined in Sections 5–9.

### Are coefficient provenance, units, year alignment, currency treatment, confidence and missingness specified?

**Yes.**

The required contract is defined in Sections 10–16.

### Is permissible reporting language established?

**Yes.**

The permitted and prohibited wording is defined in Sections 17–18.

## 21. Explicit exclusions

SKO-033 does not:

- bulk-classify suppliers to NACE;
- create canonical entities;
- alter canonical identity;
- change SKO-027 geography logic;
- change SKO-028 external-indicator ingestion;
- change SKO-029 activity evidence;
- change SKO-030 context integration;
- implement sector determination for the pilot;
- implement coefficient acquisition;
- implement the direct attribution engine;
- implement FIGARO direct+indirect effects;
- add large new external datasets;
- infer supplier-specific social outcomes from spend;
- connect to Airtable or Softr;
- modify live client data;
- commit live client data to Git.

## 22. Dependencies and next tasks

The intended sequence after SKO-033 is:

- **SKO-035:** governed evidence-led NACE/sector determination for 30–40 material suppliers;
- **SKO-036:** direct economic coefficient layer;
- **SKO-037:** environmental intensity coefficients;
- **SKO-038:** direct spend-attribution engine;
- **SKO-039:** FIGARO indirect-impact prototype;
- **SKO-040:** deep 30–40 supplier impact pilot;
- **SKO-041:** Epic 4 formal closeout.

SKO-035 must remain demand-driven and materiality-led. It must not become a bulk NACE-classification exercise.

## 23. Acceptance status

This document codifies the owner-agreed SKO-033 methodology as of 19 August 2026.

At creation:

- historical baseline reconstruction: completed to the limit of surviving evidence;
- direct attribution specification: documented;
- sector evidence hierarchy: documented;
- proportional assurance: documented;
- multi-activity treatment: documented;
- coefficient contract: documented;
- confidence and sensitivity treatment: documented;
- reporting language: documented;
- code implementation: not part of SKO-033;
- owner acceptance of SKO-033: pending review of this artifact.

No task should be marked accepted solely because this file exists. Acceptance remains owner-controlled and requires review of this artifact against the tracker criteria.
