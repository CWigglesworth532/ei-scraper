# SKO-035 — German Vereinsregister access and ingestion feasibility

**Task:** SKO-035 — German Vereinsregister access and ingestion feasibility  
**Status:** Research/feasibility evidence for owner review; **not accepted, not complete, not promoted**  
**Base:** accepted SKO-034 commit `165f3e3c1a1dbc8e47e3476ecb4eae168d67bc8a`  
**Branch:** `codex/sko-035-vereinsregister-feasibility`  
**Assessment date:** 19 August 2026

## 1. Practical conclusion

The best governed use of German Vereinsregister evidence for skopia is **controlled targeted verification**, not national Registerportal scraping.

The official Registerportal is a strong authoritative source for verifying a specific known supplier candidate. It exposes the German justice registers, including Vereinsregister, and offers individual search/view access plus structured register content (`SI`) as XML. However, its current usage rules expressly limit public use to individual informational retrieval and prohibit systematic retrieval intended to build, expand or update a parallel full or partial register. The portal also limits ordinary access to 60 searches/entity calls per hour and can block abusive or excessive use. A higher-frequency IP registration can be requested where justified, but it does **not** remove the prohibition on parallel-register construction.

Therefore:

- **Use case 1 — targeted verification:** viable and valuable now, under a governed individual-query workflow.
- **Use case 2 — source expansion:** no robust official national bulk-ingestion route has been identified in this assessment. Larger-scale ingestion should not be built against Registerportal. If broader VR coverage becomes operationally material, the next route to evaluate is a **licensed intermediary** with explicit contractual reuse rights, while preserving official-source provenance and court-scoped identity semantics.

**Recommended closeout category: 2 — Controlled targeted verification.**

This recommendation is intentionally compatible with skopia's materiality-led model: hundreds or low thousands of operationally relevant supplier candidates do not require a full German association register to be mirrored locally.

## 2. Governing boundaries

This assessment follows the accepted SKO-034 source-lab controls:

- matching is not classification;
- source evidence is not automatically policy eligibility;
- source ingestion must not create canonical entities;
- canonical materialisation remains demand-driven;
- German register identity is court-scoped;
- `DE_VR` is the Vereinsregister identifier type;
- `DE_VR` identity preserves issuing court plus register number;
- production `sources.yaml` is unchanged;
- no CAPTCHA, authentication, payment or access restriction may be bypassed;
- no live client data is used;
- no broad crawl is permitted in this task.

The accepted SKO-034 implementation already enforces court scope for reusable German register identifiers and treats register evidence as identity/legal-form evidence rather than policy classification.

## 3. Official legal and technical basis

### 3.1 Registerportal authority

**Provider/authority:** German Länder justice administrations; the portal is operated on their behalf by the State of North Rhine-Westphalia.

**Service:** Gemeinsames Registerportal der Länder / Registerportal.

Official information and usage rules:  
https://www.handelsregister.de/rp_web/information/welcome.xhtml

The portal states that it provides data from the justice entity registers, including:

- Handelsregister A and B;
- Genossenschaftsregister;
- Partnerschaftsregister;
- Gesellschaftsregister;
- Vereinsregister.

For an entity it may expose:

- index data;
- documents;
- entity-holder data;
- announcements;
- current extract (`AD`);
- chronological extract (`CD`);
- historic extract (`HD`);
- structured register content (`SI`) as XML.

This is authoritative identity evidence because the portal states that searches access the authentic register data held by the register courts.

### 3.2 Legal basis for public inspection

The public right to inspect the Vereinsregister is established by BGB § 79. The provision allows anyone to inspect the register and submitted documents and permits automated procedures where the scope of inspection remains lawful and use can be audited.

Official text:  
https://www.gesetze-im-internet.de/bgb/__79.html

The Vereinsregisterverordnung (VRV) governs the structure and automated access to the register. In particular:

- VRV § 1 establishes that Vereinsregister are court registers and may be centralised by Land;
- VRV § 2 makes the register sheet number local to the register context and explicitly allows local court numbering arrangements;
- VRV § 3 / the statutory register form records name and seat, representation, statutes/other legal relationships and registration events;
- VRV §§ 31–33 govern inspection and automated data access;
- VRV § 36 requires logging of automated retrievals.

Official VRV:  
https://www.gesetze-im-internet.de/vrv/

These provisions reinforce the SKO-034 rule that a VR number must not be treated as nationally unique without its register court/jurisdiction.

### 3.3 Current portal-use restrictions

The Registerportal usage rules, checked on 19 August 2026, state that:

1. individual retrieval for informational purposes is permitted;
2. systematic retrieval to build, expand or update a parallel full or partial register is prohibited;
3. ordinary use is limited to 60 searches/entity calls per hour;
4. the portal may block IP addresses or sessions where misuse or overload is suspected;
5. a registered/whitelisted IP can be requested for a demonstrated legitimate need for higher retrieval frequency, but use must still comply with the usage rules.

This is the principal reason national Registerportal crawling is not a governed production-ingestion route for skopia.

## 4. Access-route assessment

| Route | Authority/provider | Coverage | Access / format | Auth / CAPTCHA / rate | Reuse / bulk position | Useful fields | Automation feasibility | Confidence | Assessment |
|---|---|---|---|---|---|---|---|---|---|
| Registerportal individual query/entity view | Länder justice administrations | National federated justice registers; Vereinsregister availability depends on electronic register coverage | Web search; entity views; AD/CD/HD; SI XML where available | No routine account stated for public search; anti-abuse controls; 60 searches/entity calls per hour | Individual informational retrieval permitted; systematic parallel-register construction prohibited | Name; register type; court; VR number; seat; representation; statutes/legal relationships; dates; structured content where available | **High for manual/governed targeted verification; unsuitable for bulk harvesting** | High | Preferred official verification route |
| Registerportal higher-frequency whitelisted IP | Registerportal service office / AG Hagen | Same portal | Registered IP after application | Application and purpose disclosure required | Higher frequency may be allowed; parallel-register prohibition remains | Same | Potentially useful only for a bounded verification workflow whose purpose is accepted by the service office | High on existence; medium on suitability for skopia | Possible future governance option, not needed for current materiality-led scale |
| Official national machine-readable/bulk download | None identified | No national VR bulk dataset identified | No official public bulk download/API identified in this research | N/A | N/A | N/A | Not established | Medium-high | **No viable route identified** |
| Länder open-data registry exports | Länder / municipalities | Fragmented, local datasets exist | JSON/CSV/open-data APIs vary | Usually open | Dataset-specific licences | Often association names/categories/contact details; may not be legal register extracts | Low for authoritative national identity | Medium | Useful only case-by-case; do not treat as statutory VR replacement |
| Court-level/regional sources | Individual register courts / Land justice services | Local/regional | Court/service pages; register access ultimately through official system | Varies | No general bulk reuse right identified | May provide court competence and access routing | Low for systematic ingestion | Medium | Not a scalable national source |
| Register announcements (`Registerbekanntmachungen`) | Länder justice administrations | Current announcements, including VR announcements | Web publication; time-bounded announcement view | Public | Not a substitute for full register | Court, register type/number, name, seat and event context can appear | Potentially useful as event evidence, not full identity master | High | Secondary evidence only; not a completeness source |
| Municipal/Land association directories via GovData | Municipal/Land open-data publishers | Local and heterogeneous | JSON/CSV etc.; example: Stadt Herrenberg open-data Vereinsregister | Dataset-specific | May have open licences (example CC0) | Usually directory names/contact/category, not statutory VR court identity | Technically easy but semantically weak for legal identity | High for existence, low for statutory equivalence | Do not label as Vereinsregister legal evidence unless exact source proves it |
| North Data licensed API/exports | North Data | Broad German organisation coverage; provider states e.V. coverage | API JSON/XML; quarterly exports; API key | Authentication/API key required | Commercial licence/contract controls reuse | Register ID + court/city, names, address, status, EUID, subject and other derived fields depending product | High if licensed | Medium-high | Strong commercial candidate for source expansion; licence/source lineage must be reviewed |
| OpenRegister licensed API | OpenRegister | German register types including VR according to current documentation | REST API; search/company endpoints; JSON | API credential; commercial credit model | Commercial terms apply | Register number, register type, register court, active status, legal form; broader company details | High if licensed | Medium-high | Strong technical candidate; source provenance/licensing and VR field completeness need due diligence |
| OpenCorporates | OpenCorporates | Germany sourced from Common Register Portal; broad entity dataset | API/bulk products depending licence | Commercial/API conditions vary | Contract/licence review required | Core identity attributes; Germany source listed as Common Register Portal | Medium-high | Medium | Possible comparator/intermediary; not preferred without VR-specific coverage and licence confirmation |
| Manual verification | Registerportal + human reviewer | Any material supplier candidate that can be found | Individual official query | No bypass; reviewer operates portal normally | Fits individual informational-use model | Court-scoped VR identity and source evidence | Low throughput but high governance fit | High | **Recommended fallback/default under current scale** |

## 5. Official alternative-source findings

### 5.1 No national open-data VR mirror identified

Searches of GovData and official justice sources did not identify an official national machine-readable bulk export of statutory Vereinsregister records. GovData does expose datasets named “Vereinsregister”, but the identified example is a municipal participation/association directory from Stadt Herrenberg, not the statutory court register. It is published as JSON and has an open licence, but its semantics are local civic-directory semantics rather than court-register identity.

Example:  
https://www.govdata.de/suche/daten/vereinsregister859c0

This distinction is critical: **an open-data dataset with “Vereinsregister” in its title must not automatically be normalized as `DE_VR`.** `DE_VR` should be asserted only where the issuing court and statutory VR registration are actually evidenced.

### 5.2 Register announcements are useful but incomplete

The official Registerportal announcement service publishes time-limited register announcements and visibly includes some `VR` records alongside other register types. This demonstrates that court + register type + number + name + seat are exposed in official public event notices.

However, announcements are event-driven and time-bounded, so they are not a complete population of currently registered associations. They may be retained as authoritative event/context evidence but should not be treated as a replacement for the Vereinsregister itself.

Official service:  
https://www.handelsregister.de/rp_web/bekanntmachungen/welcome.xhtml

## 6. Licensed intermediary assessment

A licensed intermediary is the most credible route if skopia later needs larger-scale German association identity coverage.

### 6.1 North Data

North Data documents a Data API returning JSON/XML and requires an API key. For German register identity, its API explicitly requires both register ID and register city, matching the court-scoped identity requirement. Its documentation also states that e.V. organisations are included in Germany and that quarterly exports are available for supported countries.

Relevant documentation:

- https://github.com/northdata/api
- https://help.northdata.com/en/center/how-to-make-an-api-call-using-the-registerid
- https://help.northdata.com/de/center/welche-organisationstypen-in-deutschland-sind-in-north-data-enthalten

Potential strengths:

- API/export path rather than portal scraping;
- court/register-aware German identity;
- machine-readable output;
- broad entity data and update processes;
- potentially suitable for targeted API checks or larger licensed extracts.

Open questions before any use:

- exact VR coverage/completeness versus official Registerportal;
- legal basis/source lineage for VR data;
- permitted storage and downstream reuse in skopia;
- retention/redistribution constraints;
- price for expected volume;
- whether source timestamps and original official provenance can be preserved at field/record level.

### 6.2 OpenRegister

OpenRegister documentation currently lists `VR` as a supported German register type and exposes search filters for register number/type/court plus active status. It uses a credentialled credit-based API and documents a commercial pricing model.

Relevant documentation:

- https://docs.openregister.de/endpoint/search-company
- https://docs.openregister.de/sources/handelsregister
- https://docs.openregister.de/pricing

Potential strengths:

- simple API contract;
- explicit `register_type=VR` support;
- court and register number are first-class fields;
- status is exposed;
- technically aligned with skopia's normalized identity model.

Open questions:

- whether its stated “all German legal entities” VR coverage is complete enough for skopia;
- exact provenance and refresh behaviour for Vereinsregister rather than Handelsregister-only records;
- licence rights to persist and use derived identity records;
- whether street address, seat, purpose and historic status are consistently available for VR entries;
- commercial cost at the relevant volume.

### 6.3 Intermediary governance rule

A commercial source must not be promoted merely because it is easy to query. Before source promotion, skopia should require:

1. contractual authority to retrieve, store and use the data for the intended purpose;
2. a documented source lineage to the official register;
3. stable court + register-number identity;
4. field-level provenance and retrieval timestamps;
5. measured VR coverage/completeness on a small permitted comparison sample;
6. deterministic parser/schema tests;
7. explicit separation of intermediary identity evidence from social-economy classification;
8. owner review before enabling the candidate.

## 7. Field-level assessment for Vereinsregister

### 7.1 What the statutory register establishes

The VRV register form and Registerportal together support the following as authoritative or potentially authoritative register facts:

- register court / issuing authority;
- register type `VR`;
- register sheet / VR number;
- registered association name;
- registered seat (`Sitz`);
- representation rules and registered representatives;
- statutes and changes / other legal relationships;
- registration/event dates;
- current/chronological/historic register content depending available view;
- structured register content in XML where `SI` is available.

### 7.2 Address caveat

`Sitz` is a registered seat and is not automatically a full postal street address. The statutory register form centres on name and seat. Any street address exposed elsewhere in documents, index metadata, an intermediary or a related source should be recorded with its own provenance and must not be silently treated as the statutory seat field.

Therefore the normalized model should distinguish:

- `city` / registered seat where supported;
- `address_raw` only where an address is actually exposed and evidenced;
- `postcode` only where directly available or separately derived through governed enrichment.

### 7.3 Purpose/activity caveat

The Vereinsregister does not provide a standardized procurement/activity classification. Statutes or filed documents may describe organisational purpose, but that is not the same as a normalized business activity or procurement category. Purpose text, if accessed, should be preserved as source evidence and must not auto-create social-economy or directory classification.

## 8. Court-scoped `DE_VR` identity design

The accepted SKO-034 rule should remain unchanged:

```text
country = DE
identifier_type = DE_VR
issuing_authority = <verified register court>
register_number = <VR number without unsafe national uniqueness assumptions>
identifier_value_raw = <source representation>
identifier_value_normalized = <NORMALIZED_COURT>|<NORMALIZED_REGISTER_NUMBER>
identifier_scope = register_entry
```

Example shape only:

```text
identifier_type: DE_VR
identifier_value_raw: VR 12345
issuing_authority: Amtsgericht Berlin (Charlottenburg)
identifier_value_normalized: AMTSGERICHTBERLINCHARLOTTENBURG|12345
```

The normalized token is an implementation key, not a replacement for preserving the raw court and raw register number.

### Same-number rule

`VR 12345` at Court A and `VR 12345` at Court B are distinct register identities. No canonical resolution may collapse them solely because the numeric portion matches.

## 9. Proposed normalized source observation

A future permitted adapter should preserve at least:

```text
entity_name_raw
entity_name_norm
country = DE
identifier_type = DE_VR
identifier_value_raw
identifier_value_normalized
issuing_authority
register_number
source_status
city
postcode
address_raw
source_id
source_record_key
source_url
retrieved_at
provenance
resolution_status
resolution_method
resolution_confidence
resolution_reason
```

Recommended additional fields where the route exposes them:

```text
register_type_raw
registered_seat_raw
last_register_entry_date
euid
structured_content_type
source_payload_hash
source_format
source_provider
source_authority
source_access_route
source_terms_assessed_at
```

`source_record_key` should be based on the stable court-scoped register identity, not normalized name.

## 10. Classification boundary

### A. Vereinsregister **proves / authoritatively supports**

Where obtained from the official register or a contractually acceptable authoritative intermediary with preserved source lineage:

- registered association identity;
- registered name;
- VR registration;
- issuing/register court;
- registered seat where present;
- status or legal-event facts where present in the retrieved content.

### B. Vereinsregister **may support investigation of**

- association legal form (`e.V.` / registered association);
- public-benefit/nonprofit status investigation;
- links to tax-designation or ZER evidence;
- corroboration of a supplier candidate's legal identity.

### C. Vereinsregister **does not by itself prove**

- tax-exempt/public-benefit status;
- charitable tax recognition;
- social enterprise status;
- Core social-economy policy eligibility;
- procurement relevance;
- directory eligibility;
- corporate supplier readiness;
- a procurable product/service;
- client publication status.

**No rule should auto-classify all `e.V.` records as Core social economy solely from Vereinsregister presence.**

## 11. Materiality-led operating model

### Use case 1 — targeted supplier verification

**Recommended operational path now.**

For a known German supplier candidate:

1. start from the candidate's supplied name/city and any asserted register details;
2. make an individual Registerportal query using the narrowest available search parameters;
3. verify exact court + `VR` number + name + seat;
4. retain the official evidence, timestamp and retrieval context;
5. normalize the identifier as `DE_VR` using court scope;
6. use the result to strengthen identity resolution/matching;
7. separately decide whether tax, social-economy, procurement and directory evidence is required;
8. materialise/reuse a canonical entity only under the accepted demand-driven canonical workflow.

This is proportionate to skopia's current operating scale and consistent with the Registerportal information-purpose model.

### Use case 2 — source expansion

**Do not implement against Registerportal.**

No official national bulk/public API route suitable for building a reusable skopia association source was identified in this research. If a future requirement justifies association-scale ingestion, run a separately governed commercial-source evaluation comparing at least North Data and OpenRegister on:

- VR coverage;
- court/number accuracy;
- active/inactive handling;
- seat/address availability;
- update lag;
- source lineage;
- licence/storage rights;
- price per targeted lookup and per large extract;
- deterministic technical integration.

The acceptance criterion should be practical value for material supplier verification/discovery, not maximal national association coverage.

## 12. Candidate SKO-034-compatible source entry — proposed only, disabled

This candidate describes the recommended **targeted verification** source. It is a proposal and must remain disabled unless a separate production-promotion task is accepted.

```yaml
- source_id: de_vereinsregister_targeted_verification
  enabled: false
  acceptance_state: research
  country: DE
  jurisdiction_level: national_federated
  jurisdiction: DE
  source_name: Gemeinsames Registerportal der Länder — Vereinsregister targeted verification
  publisher: Landesjustizverwaltungen
  source_family: identity_register
  access_method: manual_individual_query
  source_format: XML
  landing_page: https://www.handelsregister.de/
  native_record_key: court_register_type_register_number
  provenance:
    preserve_raw_record: true
    preserve_source_url: true
    preserve_retrieved_at: true
  access_terms:
    terms_url: https://www.handelsregister.de/rp_web/information/welcome.xhtml
    assessed_on: "2026-08-19"
    systematic_reuse: prohibited_for_parallel_register
    authentication_required: false
    payment_required: false
    captcha_observed: not_required_for_assessment
    notes: >-
      Individual informational queries are permitted. Systematic retrieval to build,
      expand or update a parallel full or partial register is prohibited. Ordinary
      access is limited to 60 searches/entity calls per hour. Any higher-frequency
      registered-IP use requires separate service-office approval and does not remove
      the parallel-register restriction.
  semantic_evidence_layer: Review-only
  source_evidence: >-
    Authoritative German justice-register identity evidence for registered associations;
    preserve VR number and issuing court. Register presence is not policy eligibility.
  policy_classification: none
  rejected_unresolved_policy: preserve_with_reason
```

A separate commercial candidate should be created only after contractual review; it should not reuse `de_vereinsregister_targeted_verification` because the authority, retrieval method and licence semantics differ.

## 13. Technical-experiment decision

No broad or systematic Registerportal experiment was performed.

During this phase, only public official information/terms pages, legal texts and publicly indexed interface evidence were reviewed. That is sufficient to determine that a crawler or bulk Registerportal prototype would conflict with the stated use restrictions.

A `docs/sko-035-vereinsregister-technical-evidence.md` file is therefore **not yet justified**. The next permitted technical evidence should be created only after one of the following occurs:

- an owner-supervised individual Registerportal verification is performed using a known non-client/synthetic/public example without bypassing controls; or
- a licensed intermediary sandbox/test-data route is approved for a small technical comparison.

This task must not manufacture technical evidence by automating a portal flow whose terms make the intended systematic use unsuitable.

## 14. Confidence and unresolved questions

### High-confidence findings

- Registerportal is the authoritative official route for individual German Vereinsregister inspection.
- `SI` structured register content exists as XML where offered.
- court scope is required for safe `DE_VR` identity.
- systematic retrieval to build/update a parallel full or partial register is prohibited by current portal rules.
- ordinary portal use is limited to 60 searches/entity calls per hour.
- Vereinsregister identity does not prove charitable/social-economy/procurement eligibility.
- municipal open-data association directories must not be treated as statutory `DE_VR` evidence without explicit court-register data.

### Medium-confidence / unresolved

- the exact field completeness of `SI` XML for Vereinsregister across all Länder/courts;
- whether all targeted supplier cases expose status in a consistent machine-readable form;
- whether full street address/postcode is available in a consistent official entity view rather than only seat;
- practical success rate for name-only versus court+VR searches;
- any Länder-specific officially licensed bulk route not discoverable through the national/official sources reviewed;
- exact commercial licence and VR coverage terms for North Data/OpenRegister/OpenCorporates;
- whether a whitelisted-IP verification workflow would ever be necessary at skopia's materiality-led scale.

## 15. Recommendation

### Recommended operating model: **2 — Controlled targeted verification**

Adopt Vereinsregister as an **authoritative identity-verification source** for German supplier candidates, using individual Registerportal queries and court-scoped `DE_VR` identifiers. Do not attempt national source expansion through Registerportal.

If operational demand later shows that manual verification is a bottleneck, launch a separate **licensed German register intermediary comparison** rather than a scraping project. That task should test a small known public sample, measure VR completeness and court/number fidelity, and review contractual storage/reuse rights before any source promotion.

## 16. Explicit non-changes / safety evidence

This feasibility phase does **not**:

- modify production `sources.yaml`;
- enable any SKO-034 candidate source;
- integrate a production parser;
- perform a broad crawl;
- bypass CAPTCHA or anti-automation controls;
- use credentials or accounts;
- purchase documents or paid data;
- use live client data;
- create or materialise canonical entities;
- migrate historical German identifiers;
- alter policy classification;
- promote any source to production.

## 17. Sources reviewed

### Official primary sources

- Registerportal information, terms and FAQ: https://www.handelsregister.de/rp_web/information/welcome.xhtml
- Registerportal announcements: https://www.handelsregister.de/rp_web/bekanntmachungen/welcome.xhtml
- BGB § 79: https://www.gesetze-im-internet.de/bgb/__79.html
- Vereinsregisterverordnung: https://www.gesetze-im-internet.de/vrv/
- GovData example municipal “Vereinsregister”: https://www.govdata.de/suche/daten/vereinsregister859c0

### Commercial/secondary technical sources (used only to evaluate alternative licensed access)

- North Data API documentation: https://github.com/northdata/api
- North Data German register-ID guidance: https://help.northdata.com/en/center/how-to-make-an-api-call-using-the-registerid
- North Data German organisation coverage: https://help.northdata.com/de/center/welche-organisationstypen-in-deutschland-sind-in-north-data-enthalten
- OpenRegister company search: https://docs.openregister.de/endpoint/search-company
- OpenRegister source coverage: https://docs.openregister.de/sources/handelsregister
- OpenRegister pricing: https://docs.openregister.de/pricing
- OpenCorporates Germany coverage: https://knowledge.opencorporates.com/knowledge-base/de/

## 18. Owner-review checkpoint

SKO-035 is **not accepted or complete**.

Before further implementation, owner review should decide whether to:

1. accept the operating recommendation of Registerportal targeted verification only; and/or
2. authorise a separate small licensed-intermediary comparison using sandbox/test data; and/or
3. perform one owner-supervised individual Registerportal verification to create technical evidence.

No source promotion should occur at this checkpoint.
