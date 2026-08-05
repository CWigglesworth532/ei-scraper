# CBRE ILUNION Operational Promotion — Acceptance Evidence

## Status

**Implemented, tested and owner-accepted with operational follow-ons.**

Owner acceptance recorded on **5 August 2026**:

> I accept the CBRE ILUNION operational promotion with operational follow-ons.

## Governing decision

Decision **DEC-007** requires all ILUNION legal entities, operating-company names and trading variants to be represented by one governed canonical and directory entry.

Surviving entity:

`sko_ent_184dcde9-369f-4ac4-a1f6-27469964bb28`

Merged and superseded entity:

`sko_ent_652899e9-5728-4c88-bfcc-f8b2a587e48d`

## Promotion run

Promotion evidence directory:

`data/outputs/canonical-intake/cbre-ilunion-promotion/20260805T121926Z`

The promotion was executed through a reversible staged swap with:

- a timestamped pre-promotion operational backup;
- byte-for-byte validation of the staged store against the accepted working copy;
- verification that the backup matched the pre-promotion operational store;
- automatic rollback protection;
- pre- and post-promotion regression testing;
- post-promotion canonical-store QA.

## Test evidence

- Pre-promotion tests: **90 passed**
- Post-promotion tests: **90 passed**
- Post-promotion QA: **passed**
- Promoted store matched the accepted isolated working copy.
- Operational backup was verified.
- Rollback remained available after promotion.

## Accepted operational result

The promoted operational store contains:

| Table | Rows |
|---|---:|
| canonical_entities | 125 |
| supplier_entity_links | 143 |
| entity_aliases | 136 |
| entity_identifiers | 47 |
| source_records | 42 |
| trusted_match_terms | 132 |
| entity_events | 126 |
| entity_relationships | 1 |
| identity_review_queue | 0 |
| canonical_materialisation_review | 0 |

## ILUNION result

The surviving entity is active and named **ILUNION**.

The former Bayer-created ILUNION entity is retained for auditability but marked:

- `entity_status = superseded`
- `record_status = merged`

The promotion:

- added four accepted CBRE supplier links;
- preserved the two accepted AstraZeneca links;
- redirected the accepted Bayer supplier link;
- added the accepted CBRE aliases and trusted terms;
- created one approved merge event;
- created one accepted `merged_into` relationship.

The four promoted CBRE rows are:

| Supplier baseline ID | Supplier name |
|---|---|
| SB00173 | ILUNION |
| SB00174 | ILUNION LAVANDERIAS Y SERVICIOS |
| SB00175 | ILUNION LIMPIEZA Y MEDIOAMBIENTE |
| SB00176 | ILUNION SEGURIDAD |

All four use:

`sko_ent_184dcde9-369f-4ac4-a1f6-27469964bb28`

## Duplicate-reference controls

After promotion, the merged entity has:

- **0** live supplier links;
- **0** live aliases;
- **0** live trusted terms.

The operational store contains exactly:

- one merge event;
- one `merged_into` relationship.

## Acceptance conclusion

The CBRE ILUNION operational promotion is accepted.

The acceptance covers:

- the one-entry ILUNION canonical and directory treatment;
- the four CBRE link additions;
- the Bayer link redirection;
- preservation of the superseded entity and its audit trail;
- the merge event and relationship;
- the promoted operational counts and QA result.

## Operational follow-ons

The following remain outside this acceptance:

- materiality-led review of the remaining 118 CBRE suppliers;
- identity verification and owner decisions for the remaining new-entity candidates;
- resolution or exclusion of the ten unsupported aggregate-only rows;
- final CBRE population promotion;
- broader Epic 2 regression and CI development.

These follow-ons do not invalidate the accepted ILUNION promotion.
