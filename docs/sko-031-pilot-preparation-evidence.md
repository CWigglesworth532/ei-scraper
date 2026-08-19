# SKO-031 pilot preparation evidence

## Status

SKO-031 is implemented and tested for owner review. It is not accepted, and acceptance remains owner-controlled.

## Implemented scope

The additive `pilot_preparation.py` layer validates a selected pilot subject population, preserves local source traceability, produces deterministic manifest and pre-pilot QA records, presents already-produced SKO-030 context as a human-inspectable CSV, and emits one of the three governed readiness states. Subject identity remains keyed by `subject_type + subject_id`; `entity_id` is retained as context for supplier observations and is required for canonical-entity subjects.

The manifest records hashes, counts, caller-supplied repository commit and timestamp, declared dependencies, and live-data safety metadata. It does not reproduce input rows. Pre-pilot contextual metrics supplied by SKO-030 are reflected without changing their integration semantics.

## Population and data safety

The accepted 143-row real supplier-link population is the governed future real-run spine, but it is not reproduced in Git. All committed fixtures are wholly synthetic. Live inputs and generated outputs remain below the ignored `data/` root, with the intended output root `data/pilots/sko-031/`.

The implementation does not create, mutate, materialise, or import canonical entities. It does not acquire sources, geocode, infer locations, perform NACE classification, build a dashboard or UI, attribute impact, or integrate Airtable or Softr. It makes no changes to accepted SKO-027, SKO-028, SKO-029, or SKO-030 files.

Explicit exclusions include preserved or unaccepted client rows, aggregate-only evidence, ambiguous rows, and candidate/review populations. The preparation layer does not manufacture supplier observations or collapse distinct supplier observations solely by `entity_id`.

## Behavioural evidence

Synthetic tests cover both accepted subject types, blank and retained supplier-observation entity context, canonical entity-ID enforcement, duplicate/conflict handling, permitted subject and location-role values, missing-postcode QA, source traceability, manifest hashes without row content, direct reuse of SKO-030 QA semantics, explicit unresolved/no-compatible-indicator states, activity summarisation without row multiplication, all readiness states, byte-deterministic reruns, input-order independence, JSON Schema validation, absence of network/API imports, absence of canonical creation functions, and absence of impact/causal proxy fields.

## Validation

Focused and full-suite results are recorded in the implementation handoff after commands are run. A successful implementation and test run is evidence for owner review, not acceptance.

No real-data pilot was run during implementation. Real-run evidence may be added only after the local ignored run occurs and only if the test suites pass. Known runtime gaps for postcode correspondence, external contextual extracts, and activity-evidence inputs are expected to produce `ready_with_known_gaps`, not a structural blocker.

## Remaining owner-controlled step

After review, the owner may authorise governed mapping of the accepted local 143-row population to source records and execution under `data/pilots/sko-031/`. The resulting local manifest, QA, contextual review, and readiness artifacts must remain ignored and must not copy live supplier rows into Git, tests, documentation, examples, or asserted error text.
