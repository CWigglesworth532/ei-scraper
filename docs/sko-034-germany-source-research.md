# SKO-034 Germany-first source research

**Status:** Implemented research evidence; operationally unvalidated, not promoted and not accepted
**Assessment date:** 19 August 2026
**Production effect:** None

## Research method and semantic rule

This matrix records candidate discovery and feasibility only. A source listing is source evidence, not automatic policy classification. The controlled semantic layers are:

- **Core** - strong official recognition, tax-designation or deterministic legal-form evidence;
- **Associated** - governed specialist, federation, programme or ecosystem association;
- **Review-only** - useful identity/discovery evidence requiring verification or policy review;
- **Reject** - unsuitable for ingestion under current access, quality or semantic conditions.

The BIH confirms that all 16 Länder have responsible integration/inclusion authorities and publishes comparable counts by Land. REHADAT provides an organisation-level nationwide directory containing names, postcodes, legal-form descriptions and Länder. These establish national discovery coverage but do not remove the need to assess each Land's authoritative organisation-level route.

Primary governed references:

- BIH authority directory: https://www.bih.de/bih/die-bih/mitglieder/
- BIH 2024 inclusion-enterprise statistics: https://jahresberichte.bih.de/jahresbericht-2024/leistungen-der-integrationsaemter/leistungen-an-inklusionsbetriebe
- REHADAT inclusion-enterprise directory: https://www.rehadat-adressen.de/adressen/arbeit-beschaeftigung/inklusionsbetriebe/
- BAG IF national directory: https://bag-if.de/unternehmensuebersicht-fuer-menschen-mit-blindheit-und-sehbehinderung/

## 16-Länder matrix

| Land | Responsible authority / route | Organisation-level source/list | Access | Typical fields/identifiers | Update frequency | Semantics | Systematic reuse feasibility | National overlap / next verification |
|---|---|---|---|---|---|---|---|---|
| Baden-Württemberg | KVJS Integrationsamt | REHADAT/BAG IF; authority-held funded population to confirm | Public query + authority enquiry | Name, address, postcode, activity; official register ID generally absent | REHADAT ongoing; authority unknown | Associated; official funding status would be Core if released | Directory reuse terms review; authority extract unknown | Compare KVJS population with REHADAT/BAG IF; request permitted structured list |
| Bayern | ZBFS Inklusionsamt | REHADAT/BAG IF; LAG IF Bayern specialist coverage | Public directory / specialist site | Name, locality, website, sector; legal identifiers inconsistent | Unknown | Associated | Review required | Establish whether ZBFS publishes a current funded-enterprise list |
| Berlin | LAGeSo Integrationsamt | REHADAT/BAG IF; Berlin programme pages | Public query/manual | Name, postcode, address, legal-form text | Unknown | Associated / Review-only | Review required | Confirm organisation-level LAGeSo export and status dates |
| Brandenburg | LASV Integrationsamt | REHADAT/BAG IF; LASV programme information | Public query/manual | Name/location via directories; official status via authority | Unknown | Associated / Review-only | Review required | Seek current LASV recognised/funded list |
| Bremen | Amt für Versorgung und Integration Bremen | REHADAT/BAG IF | Public query/manual | Name, locality, website; few stable IDs | Unknown | Associated | Review required | Small population suited to authority-verified manual reconciliation |
| Hamburg | Sozialbehörde / Integrationsamt Hamburg | REHADAT/BAG IF | Public query/manual | Name, address, postcode, sector | Unknown | Associated | Review required | Confirm whether Hamburg publishes its funded population as open data |
| Hessen | LWV Hessen Integrationsamt | REHADAT/BAG IF; LWV information route | Public query/manual | Name, address, postcode, legal-form description | Unknown | Associated / Review-only | Review required | Request current organisation list and update policy from LWV |
| Mecklenburg-Vorpommern | LAGuS Integrationsamt | REHADAT/BAG IF | Public query/manual | Name, location, website | Unknown | Associated | Review required | Verify LAGuS organisation-level availability |
| Niedersachsen | Niedersächsisches Integrationsamt | REHADAT/BAG IF | Public query/manual | Name, locality, legal-form text, services | Unknown | Associated | Review required | Check state programme/funding list and identifiers |
| Nordrhein-Westfalen | LVR-Inklusionsamt + LWL-Inklusionsamt Arbeit | REHADAT/BAG IF; two regional authority populations | Public query/manual | Name, address, postcode, activities; regional status | Unknown | Associated; authority recognition could be Core | Review required | Treat Rheinland and Westfalen-Lippe separately; reconcile to national directories |
| Rheinland-Pfalz | LSJV Integrationsamt | REHADAT/BAG IF | Public query/manual | Name, locality, website, sector | Unknown | Associated | Review required | Seek LSJV funded-enterprise list and effective dates |
| Saarland | Landesamt für Soziales Integrationsamt | REHADAT/BAG IF | Public query/manual | Name, locality, contact | Unknown | Associated | Review required | Small population suitable for manual authority confirmation |
| Sachsen | Kommunaler Sozialverband Sachsen Integrationsamt | REHADAT/BAG IF | Public query/manual | Name, address, postcode, activity | Unknown | Associated | Review required | Determine whether KSV Sachsen releases a current list |
| Sachsen-Anhalt | Landesverwaltungsamt Integrationsamt | REHADAT/BAG IF | Public query/manual | Name, locality, website | Unknown | Associated | Review required | Determine organisation-level authority route and status timestamps |
| Schleswig-Holstein | Integrationsamt Schleswig-Holstein | REHADAT/BAG IF | Public query/manual | Name, postcode, address, services | Unknown | Associated | Review required | Check state open-data or permitted list route |
| Thüringen | Thüringer Landesverwaltungsamt Integrationsamt | REHADAT/BAG IF | Public query/manual | Name, location, activity | Unknown | Associated | Review required | Seek current authority population and reuse terms |

### Matrix conclusions

1. REHADAT is the strongest immediately visible organisation-level nationwide discovery route, but its directory semantics and reuse terms require assessment before ingestion.
2. BAG IF is a governed specialist directory and useful corroboration source, but membership/listing is not identical to statutory recognition.
3. BIH statistics prove that authority-level recognised/funded populations exist across all Länder, but the statistical tables are aggregate and cannot identify organisations.
4. NRW requires two authority tracks. Other Länder still require direct verification of whether the authority exposes a current, permitted organisation-level list.
5. No Land should be marked unavailable merely because a bulk dataset was not found; permitted authority enquiry and manual verification remain valid routes.

## Welfare and specialist source catalogue

| Source family | Candidate | Access and fields | Semantic treatment | Feasibility / limitation |
|---|---|---|---|---|
| Welfare umbrella | BAGFW | Aggregate statistics and links to six member federations | Associated | BAGFW statistics are not an organisation register; use federation routes for entities |
| Welfare federation | Arbeiterwohlfahrt (AWO) | National/regional facility and provider directories | Associated | May contain facilities, brands and local units rather than legal entities |
| Welfare federation | Deutscher Caritasverband | Facility/provider directories | Associated | Requires legal-entity resolution; Catholic structures can be nested |
| Welfare federation | Der Paritätische | Member directories by Land | Associated | Strong membership evidence; fragmented regional presentation |
| Welfare federation | Deutsches Rotes Kreuz | Association/facility networks | Associated | Chapters, facilities and legal entities must remain distinct |
| Welfare federation | Diakonie Deutschland | Provider/facility directories | Associated | Regional churches, operators and facilities require careful identity handling |
| Welfare federation | Zentralwohlfahrtsstelle der Juden in Deutschland | Member/facility information | Associated | Coverage and structured access require direct assessment |
| Inclusion specialist | REHADAT-Adressen | Nationwide queryable organisation records with location/legal-form text | Associated / Review-only | Strong discovery coverage; reuse terms and record-key stability need confirmation |
| Inclusion specialist | BAG IF | Names, addresses, websites, activities | Associated | Existing parser requires production reconciliation after laboratory acceptance |
| Cooperative identity | Registerportal GnR | Court, GnR number, legal name, status, structured content when available | Review-only identity/legal-form | Individual query only; systematic parallel-register construction prohibited |
| Association identity | Registerportal VR | Court, VR number, legal name, status | Review-only identity/legal-form | Same individual-query restriction; registration does not prove policy eligibility |
| Tax designation | BZSt ZER | Organisation, seat/purpose context and source-local key | Core tax-designation | Existing local-build mechanism and reuse conditions require revalidation |

BAGFW reference: https://www.bagfw.de/veroeffentlichungen/statistik/gesamtstatistiken-vorjahre

## National and regional cooperative/social-economy conclusion

Germany has no single permitted bulk social-economy identity register identified in this phase. The best achievable combination is:

1. ZER for official tax-designation evidence;
2. BIH/authority populations, REHADAT and BAG IF for inclusion-enterprise discovery and corroboration;
3. GnR/VR/HRB individual verification for identity and legal form;
4. federation/member directories for governed welfare association evidence;
5. Länder authority enquiry where national directory evidence is incomplete.

## Switzerland comparator

Zefix is an official identity/legal-form comparator with structured public REST and linked-data routes. Normalize UID as `CHE-123.456.789`. German, French and Italian legal-form labels should map to a language-independent code/family while retaining the raw label. Zefix evidence establishes registered identity, status and legal form, not social-economy eligibility. Known limitations include associations not required to enter the commercial register, missing purpose/address fields, and distinctions between head office and establishments. CH-ID and FCRO-ID remain outside matching scope unless separately approved.

Reference: https://opendata.swiss/en/dataset/zefix-zentraler-firmenindex

## Global opportunity catalogue (no ingestion)

| Jurisdiction | Governed opportunity | Organisation-level potential | Status |
|---|---|---|---|
| United Kingdom | Charity Commission registers; FCA Mutuals Public Register; Companies House | Charity, cooperative/mutual and company identities | Catalogue only |
| Canada | CRA List of charities; federal/provincial corporate registries | Registered-charity evidence and identity | Catalogue only |
| United States | IRS Tax Exempt Organization Search bulk data | Tax-exempt designation and EIN | Catalogue only |
| Australia | ACNC Charity Register | Charity status, ABN and organisation details | Catalogue only |
| New Zealand | Charities Services register; Companies Office societies | Charity/association identity and status | Catalogue only |
| India | NGO Darpan and statutory corporate/cooperative routes | Governed but access/coverage complexity | Catalogue only |

No global source was probed, downloaded or implemented under SKO-034.
