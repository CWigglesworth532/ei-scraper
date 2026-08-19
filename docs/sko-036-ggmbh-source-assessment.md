# SKO-036 — German gGmbH / tax-privileged company source feasibility

**Status:** Research complete for owner review; not accepted, not promoted
**Branch:** `codex/sko-036-ggmbh-source-feasibility`
**Base:** accepted SKO-034 commit `165f3e3c1a1dbc8e47e3476ecb4eae168d67bc8a`
**Production changes:** None
**Live client data:** None

## 1. Purpose

Determine whether skopia can lawfully and reproducibly build a reusable German company-source population that captures tax-privileged limited companies, especially organisations trading as `gGmbH`, without scraping or mirroring the official Handelsregister/Unternehmensregister in breach of access terms.

This task is deliberately separate from SKO-035. SKO-035 concerns Vereinsregister evidence for associations (`DE_VR`). SKO-036 concerns company-register routes, primarily HRB-registered limited companies.

The practical question is:

> Can skopia obtain a scalable population of German `gGmbH` and related tax-privileged companies, with stable court-scoped register identity, from a lawful reusable source?

## 2. Governing skopia boundaries

This assessment preserves the accepted programme rules:

- matching is not classification;
- source evidence is not automatically policy eligibility;
- canonical materialisation is demand-driven;
- a source population does not become a canonical population;
- source ingestion must not itself create canonical entities;
- German register identifiers remain court-scoped;
- `DE_HRB` is the relevant Handelsregister identifier type for most gGmbH entities;
- production `sources.yaml` must not change during feasibility work;
- no CAPTCHA, authentication, payment, account creation or access-control bypass is permitted without owner approval.

## 3. Legal semantics of `gGmbH`

### 3.1 What the name itself means

German GmbH law provides an unusually strong legal-form signal. Section 4 GmbHG states that a company pursuing exclusively and directly tax-privileged purposes under sections 51–68 AO may use the abbreviation `gGmbH` in its registered company name.

Primary source:

- GmbHG § 4: https://www.gesetze-im-internet.de/gmbhg/__4.html
- AO §§ 51–68: https://www.gesetze-im-internet.de/ao_1977/BJNR006130976.html

This means an official registered name containing the legally used `gGmbH` form is stronger than an ordinary heuristic such as the word `gemeinnützig` appearing in a website description.

### 3.2 What it does not safely prove forever

`gGmbH` should not be treated as a timeless substitute for current tax-status evidence. A register name is legal identity evidence and a strong tax-privileged-purpose signal, but skopia should still distinguish:

- registered company identity;
- registered company name / legal-form signal;
- current tax-benefit / donation-recipient status;
- social-economy policy classification;
- procurement relevance;
- directory readiness.

The BZSt Zuwendungsempfängerregister (ZER) remains the stronger source for current donation-recipient/public-benefit evidence where the organisation is present.

### 3.3 `gUG` / tax-privileged UG

Section 5a GmbHG requires a low-capital company to use `Unternehmergesellschaft (haftungsbeschränkt)` or `UG (haftungsbeschränkt)`. The statute does not mirror § 4 by defining `gUG` as a separate statutory abbreviation. SKO-036 therefore does **not** treat free-text `gUG` detection as equivalent to the explicit statutory `gGmbH` signal.

Any tax-privileged UG population should be supported by stronger evidence such as ZER/public-benefit status, not name heuristics alone.

Primary source: https://www.gesetze-im-internet.de/gmbhg/__5a.html

## 4. Official national register routes

### 4.1 Registerportal der Länder / Handelsregister

**Authority:** Landesjustizverwaltungen; portal operated on their behalf by North Rhine-Westphalia.

**Coverage:** Handelsregister A/B, Genossenschaftsregister, Partnerschaftsregister, Gesellschaftsregister and Vereinsregister.

**Useful HRB fields:**

- registered name;
- register court;
- register type;
- register number;
- seat;
- status;
- current / chronological / historical extracts;
- structured register content (`SI`) as XML where available.

**Technical suitability for individual verification:** High.

**Suitability for building a reusable skopia gGmbH index:** No.

The Registerportal terms prohibit systematic retrieval used to construct or update a parallel full or partial register. The same core restriction therefore applies whether the intended subset is all companies, all gGmbH entities or another systematic Handelsregister slice.

Primary source: https://www.handelsregister.de/rp_web/information/welcome.xhtml

**Conclusion:** retain as authoritative targeted verification, not systematic source-population ingestion.

### 4.2 Unternehmensregister

**Authority / operator:** the statutory Unternehmensregister, operated by Bundesanzeiger Verlag as the register-running body.

**Coverage relevant to this task:** direct access to Handelsregister, Genossenschaftsregister, Gesellschaftsregister and Partnerschaftsregister information supplied from the judicial registers.

**Structured content:** `SI` is available as XML for automated downstream processing on an individual retrieved entity where available.

Primary source: https://www.unternehmensregister.de/de/so-gehts/registerinformationen

#### Terms assessment

The current Allgemeine Nutzungsbedingungen are decisive. They distinguish data held directly in the Unternehmensregister from original judicial-register data accessed through searches. For original register data, § 7(3) states that users must not use retrieved data to build or maintain their own register parallel to the Handelsregister, nor provide it to others for that purpose.

Primary source, current terms PDF:
https://www.unternehmensregister.de/i18n-doc/D061_UReg_nutz_0118_de.pdf

The same terms note that access to original register data is mediated through search results and the Länder justice administrations (§ 3(6)).

**Conclusion:** the Unternehmensregister is not a lawful workaround for bulk harvesting a gGmbH slice from the Handelsregister.

### 4.3 Unternehmensregister submission Webservice

The Unternehmensregister offers a software Webservice, but the official documentation describes it as a channel **for submitting** disclosure/publication material, not a general bulk-read API for company-register records.

Primary source: https://unternehmensregister.de/de/so-gehts/uebermitteln

**Conclusion:** not an ingestion route.

## 5. Official tax-status route — BZSt Zuwendungsempfängerregister

**Authority:** Bundeszentralamt für Steuern (BZSt).

**Coverage:** organisations entitled to issue donation receipts / tax-privileged recipients, including domestic tax-privileged corporations.

**Public fields described by BZSt:**

- organisation name;
- address;
- tax-privileged purposes under the AO;
- date of latest exemption/determination decision;
- later voluntary website information.

Primary source: https://www.bzst.de/SharedDocs/Pressemitteilungen/DE/20231129_start_zer.html

**Semantic strength:** stronger public-benefit evidence than `gGmbH` name alone.

**Important limitation for identity:** public BZSt documentation located in this task does not establish that ZER exposes Handelsregister court + HRB number as a reusable native identifier. It therefore should not replace HRB identity.

**Systematic-access finding:** the BZSt does operate mass-data interfaces for specific tax procedures, but no official public documentation was found establishing a ZER bulk-download/read API for building an external reusable ZER index. Existing skopia ZER ingestion remains a separate accepted laboratory issue requiring its own terms/reuse confirmation before promotion.

**Practical implication:** ZER is best treated as a classification/evidence complement to HRB identity, not yet as the solved source for the complete gGmbH identity population.

## 6. Licensed intermediary routes

Where official portals forbid creation of a parallel register, a separately licensed company-data supplier is the realistic scalable route, provided the commercial licence explicitly permits local storage, matching, derivative filtering and recurring refresh.

### 6.1 OpenRegister

**Provider:** OpenRegister.

**Claimed coverage:** 4M+ German companies assembled from official registers.

**Interfaces:**

- REST API;
- advanced company search;
- legal-form filtering;
- company records with register type, register number, register court, status and legal form;
- optional real-time retrieval from the Handelsregister;
- batch / bulk data licensing of the full company index.

Primary documentation:

- https://openregister.de/api
- https://docs.openregister.de/introduction
- https://docs.openregister.de/endpoint/filter-company
- https://docs.openregister.de/endpoint/company

The provider explicitly offers bulk-data licensing for large-scale downstream processing:
https://openregister.de/en

**Technical fit for skopia:** High.

A licensed export or legal-form-filtered population could preserve:

- `entity_name_raw`;
- `country = DE`;
- `identifier_type = DE_HRB`;
- register number;
- register court;
- active/status signal;
- legal form;
- address/location;
- provider-native company key for source continuity;
- source/provider provenance;
- refresh date.

The API documentation shows identifiers shaped like `DE-HRB-...` and returns `register_number`, `register_type`, `register_court`, `active` and `legal_form` in company search results.

**Cost:** normal API plans are published, with enterprise/bulk exports quoted separately. Current public pricing includes API tiers from free testing through paid Pro/Business plans; bulk-index licensing is a commercial discussion.

**Unresolved before use:**

1. Does the bulk licence permit skopia to persist a filtered gGmbH population indefinitely?
2. Is redistribution prohibited while internal matching/derived outputs are permitted?
3. Can the export be restricted to selected legal forms/name patterns to reduce cost?
4. What refresh cadence and deletion/correction semantics are supplied?
5. Are court names normalized or supplied in official form?
6. Does the source contract permit storing official-source lineage/document URLs?
7. What fields are included in the bulk product versus API-only enrichment?

No account, API key, paid service or test request was used during SKO-036.

### 6.2 North Data

**Provider:** North Data GmbH.

**Interfaces documented by provider:**

- Data API in JSON/XML;
- company lookup by German register ID **plus register court/city**;
- power/universal search;
- quarterly exports described as complete company datasets for supported countries;
- export and API products intended for synchronization.

Primary provider documentation:

- https://github.com/northdata/api
- https://github.com/northdata/api/blob/master/doc/data-api-userguide/data-api-userguide.md
- https://help.northdata.com/de/center/data-service

The documentation explicitly preserves the German court-scoped nature of register identity: a German register ID query requires both the register ID and register city/court context.

**Technical fit for skopia:** High.

**Unresolved before use:** commercial licence, fields in the German export, legal-form filtering capability, current pricing, reuse/storage rights and provenance granularity.

No account, API key or paid request was used.

## 7. Comparative assessment

| Route | Authoritative identity | Systematic lawful skopia population | Court + HRB | Tax/public-benefit semantics | Automation | Assessment |
|---|---|---:|---:|---:|---:|---|
| Registerportal | Yes | No | Yes | No | Targeted only | Verification source |
| Unternehmensregister register access | Yes | No | Yes | No | Targeted only | Verification source |
| Unternehmensregister submission Webservice | N/A | No | N/A | N/A | Submission only | Reject |
| BZSt ZER | Yes for tax status | Unresolved as bulk route | Not established from public docs | Yes, strong | Current skopia route needs separate governance | Complementary evidence source |
| OpenRegister licensed bulk/API | Derived from official sources | **Potentially yes under licence** | Yes | No by itself | High | Strongest immediate candidate |
| North Data licensed exports/API | Derived from official/public sources | **Potentially yes under licence** | Yes | No by itself | High | Strong comparator |

## 8. Recommended operating model

### Recommendation: licensed identity population + official/tax evidence layering

The best governed route identified is **not** to scrape the Handelsregister. It is to license a reusable German company index from a provider whose contract explicitly permits skopia's intended internal use.

Recommended architecture:

1. **Population layer — licensed company index**
   - obtain a filtered or filterable German company dataset;
   - retain registered name, court, register type/number, status, address and legal form;
   - identify explicit statutory `gGmbH` name/legal-form signals;
   - preserve source/provider provenance and refresh timestamp;
   - do not materialise all rows as canonical entities.

2. **Matching universe**
   - add eligible source records to the normalized matching universe only after licence and parser acceptance;
   - use the population to find supplier candidates by identifier/name/country;
   - preserve `DE_HRB` as court-scoped identity.

3. **Evidence strengthening**
   - where a supplier candidate is material, query/verify current official Handelsregister information through the governed individual-verification route;
   - join to ZER where available for current public-benefit/donation-recipient evidence.

4. **Canonical layer**
   - materialise only matched, verified, reviewed, directory-relevant or reporting-relevant entities;
   - reuse before create;
   - source ingestion itself creates no canonical entity.

5. **Classification**
   - registered `gGmbH` is strong statutory tax-privileged-purpose evidence;
   - current ZER evidence is stronger for current tax-benefit/donation-recipient status;
   - neither alone establishes procurement relevance or directory readiness.

## 9. Why this is materially better than manual-only Germany

This model gives skopia a scalable German discovery universe while preserving official verification for the small number of suppliers that matter operationally.

It fits the accepted skopia staged model:

- broad but tiered discovery from a reusable population;
- materiality-led verification against official evidence;
- high precision for final reporting.

It also avoids bulk canonicalisation: a licensed dataset may contain millions of German companies, but only matching/relevant suppliers receive canonical skopia IDs.

## 10. Candidate normalized source semantics

For a licensed company-index source, the minimum normalized record should include:

```text
entity_name_raw
entity_name_norm
country = DE
legal_form_local
base_legal_form_family
identifier_type = DE_HRB
identifier_value_raw
identifier_value_normalized
issuing_authority / register_court
register_number
source_status
city
postcode
address_raw
source_id
source_record_key
source_url or provider reference where contract permits
retrieved_at
provenance
resolution_status
classification_semantics
```

For `DE_HRB`, identity remains scoped as:

```text
normalized_register_court | normalized_register_number
```

The provider's own internal company ID may be retained as a `source_record_key`, but must not replace the authoritative court-scoped register identifier or become a skopia canonical ID.

## 11. Candidate source entries — proposal only

Do **not** add these to production `sources.yaml` yet.

```yaml
- source_id: de_openregister_company_index_candidate
  enabled: false
  acceptance_state: research
  country: DE
  jurisdiction_level: national
  jurisdiction: DE
  source_name: OpenRegister licensed German company index
  publisher: OpenRegister
  source_family: licensed_identity_register_intermediary
  access_method: licensed_bulk_or_api
  source_format: JSON
  native_record_key: provider_company_id
  semantic_evidence_layer: Review-only
  source_evidence: Licensed structured German company identity data derived from official registers; gGmbH registered-name signal may provide strong statutory tax-privileged-purpose evidence, but policy classification remains separate.
  policy_classification: none
  rejected_unresolved_policy: preserve_with_reason
```

```yaml
- source_id: de_northdata_company_index_candidate
  enabled: false
  acceptance_state: research
  country: DE
  jurisdiction_level: national
  jurisdiction: DE
  source_name: North Data licensed German company export
  publisher: North Data GmbH
  source_family: licensed_identity_register_intermediary
  access_method: licensed_export_or_api
  source_format: JSON
  native_record_key: provider_company_id
  semantic_evidence_layer: Review-only
  source_evidence: Licensed structured German company identity data with court-scoped Handelsregister references; policy classification remains separate.
  policy_classification: none
  rejected_unresolved_policy: preserve_with_reason
```

## 12. Recommended next controlled experiment

Do **not** purchase or authenticate yet.

The next experiment should be a provider-level evaluation after owner approval:

1. obtain written confirmation of internal-storage and matching rights from OpenRegister and/or North Data;
2. request a sample/sandbox extract containing at least:
   - exact registered name;
   - register court;
   - register type;
   - register number;
   - legal form;
   - active/status;
   - address/seat;
   - provider source/provenance;
   - update timestamp;
3. test filtering for explicit `gGmbH` names;
4. test five synthetic or public known examples against official Handelsregister identity;
5. measure duplicate court/number handling and name-change behaviour;
6. separately test ZER joins for tax-status strengthening;
7. price the smallest adequate refresh model rather than licensing unnecessary enrichment fields.

This requires owner approval because it may involve account creation, provider contact, credentials, a commercial quote or paid access.

## 13. Recommendation

**Recommended disposition: 4 — Licensed intermediary required for systematic company-source expansion.**

More specifically:

- do **not** scrape Registerportal or Unternehmensregister to build a gGmbH population;
- do build toward a reusable gGmbH discovery population if a commercial licence permits internal storage and matching;
- evaluate OpenRegister first because it explicitly offers bulk licensing of its German company index and exposes the exact fields skopia needs;
- retain North Data as a serious comparator;
- preserve official Registerportal/Unternehmensregister as the materiality-led verification layer;
- use ZER as a separate tax/public-benefit evidence layer where available.

This means Germany need not be manual-only. The likely scalable architecture is **licensed broad identity discovery + targeted official verification + ZER classification evidence**.

## 14. Work actually completed

- Created a dedicated SKO-036 branch from the accepted SKO-034 commit.
- Reviewed official GmbHG/AO semantics for `gGmbH`.
- Reviewed official Handelsregister/Registerportal and Unternehmensregister access and terms.
- Confirmed that the Unternehmensregister read route has the same parallel-register restriction for original Handelsregister data.
- Confirmed the Unternehmensregister Webservice documented publicly is a submission route, not a bulk-read company API.
- Reviewed BZSt ZER public scope and semantics.
- Reviewed OpenRegister API, search, bulk-data and pricing documentation.
- Reviewed North Data API/export documentation.
- Designed a governed layered German company-source operating model.

## 15. Evidence created

- `docs/sko-036-ggmbh-source-assessment.md`
- Official-source URLs and provider-primary documentation captured in this document.

No technical-evidence document was created because no authenticated, paid or provider-sandbox experiment was authorised.

## 16. Decisions proposed, not accepted

- Treat statutory `gGmbH` registered-name evidence as a strong tax-privileged-purpose signal, but keep current tax status and policy classification separate.
- Do not treat `gUG` free-text as equivalent to statutory `gGmbH` evidence.
- Do not use Registerportal/Unternehmensregister systematic retrieval to populate a gGmbH list.
- Prefer a licensed intermediary for systematic HRB identity population.
- Evaluate OpenRegister first and North Data second.
- Join ZER where available for stronger current tax/public-benefit evidence.

## 17. Unresolved items

- Commercial licence terms for local storage, derived matching and refresh.
- Bulk-export price and minimum contract size.
- Exact fields and provenance in provider batch products.
- Whether a filtered gGmbH-only export is commercially available.
- Current ZER systematic-reuse terms and supported stable identifiers.
- Operational treatment of tax-privileged UG entities that do not have the statutory `gGmbH` signal.

## 18. Status

SKO-036 is **researched and documented for owner review**.

It is **not complete or accepted**, no provider has been selected, no purchase or credentials have been used, and no source has been promoted.