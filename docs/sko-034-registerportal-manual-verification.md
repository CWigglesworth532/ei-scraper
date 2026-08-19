# SKO-034 Registerportal manual individual-query protocol

**Status:** Implemented and reviewed as a laboratory protocol; operationally unvalidated, not promoted and not accepted
**Prohibited:** systematic retrieval, crawling, pagination harvesting, or building/updating a parallel full or partial register

The Registerportal terms explicitly prohibit systematic retrieval for constructing or updating parallel full or partial registers: https://www.handelsregister.de/rp_web/information/welcome.xhtml

## Permitted laboratory workflow

1. Start with a specific organisation candidate and its asserted country `DE`.
2. Record the supplied raw name, city/postcode and any asserted register type/number without altering them.
3. Search manually using the narrowest available parameters: exact name, seat, Land, court, register type or number.
4. Inspect an individual result. Prefer the UT/entity view and structured register content where legitimately available.
5. Record separately:
   - official legal name;
   - verified register court exactly as displayed;
   - register type (`HRB`, `HRA`, `GnR`, `VR`, `PR`, `GsR`);
   - register number;
   - EUID, if displayed;
   - seat;
   - status;
   - retrieval timestamp and result URL/context;
   - reviewer and decision rationale.
6. Normalize only after verifying court and register type. The laboratory key is `DE_<TYPE>` plus `<NORMALIZED_COURT>|<REGISTER_NUMBER>`.
7. If name, seat or court evidence is ambiguous, retain every candidate as unresolved. Do not choose by normalized name alone.
8. Store evidence as an identity assertion. Do not classify the organisation as social economy solely from register presence or legal form.

## Register-specific interpretation

| Register | Establishes | Does not establish |
|---|---|---|
| HRB | Registered company identity, court/number, legal form/status and available filed facts | Charitable status, inclusion-enterprise recognition or policy eligibility; `gGmbH` naming alone is not universal eligibility |
| HRA | Registered partnership/merchant identity and available status | Social-economy or non-profit status |
| GnR | Registered cooperative identity and legal form | That every cooperative meets the client's social-economy policy |
| VR | Registered association identity | Tax-exempt/charitable status or procurement relevance |
| PR | Registered professional partnership identity | Social-economy status |
| GsR | Registered civil-law society identity | Social-economy status |
| EUID | Cross-system official identity reference where present | Classification |

## Stop and rejection rules

- Stop if CAPTCHA, authentication, payment or access restrictions would need bypassing.
- Stop if the task expands from individual verification into systematic collection.
- Reject a proposed reusable register identifier if court or register type is unverified.
- Preserve no-hit, multiple-hit, expired-session and unavailable-state outcomes with timestamps.
- Do not paste personal data from filings into committed fixtures or evidence.
- Do not use source URLs containing sensitive or short-lived session tokens as durable identifiers.

## Minimum review evidence

One reviewed verification record must contain the raw query, result count, selected or unresolved result, court/type/number, name/seat comparison, source timestamp, reviewer, and explicit statement that identity evidence did not create policy classification.
