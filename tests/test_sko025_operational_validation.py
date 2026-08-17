import copy
import inspect
import sys
import unittest
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sko025_operational_validation as operational
import directory_integration_handoff as handoff
from scripts.sko025a_crosswalk import fingerprint

FP_CONFIG = yaml.safe_load((ROOT / "config/airtable_profile_fingerprint_v1.yaml").read_text())


def snapshot():
    rows = []
    for index in range(321):
        rows.append({column: "" for column in operational.EXPECTED_COLUMNS})
        rows[-1].update({
            "Organisation": f"Synthetic Supplier {index}", "Country": "United Kingdom",
            "Website": f"supplier{index}.example", "Airtable Record ID": f"recSynthetic{index:03d}",
        })
    return pd.DataFrame(rows, columns=operational.EXPECTED_COLUMNS)


class Tests(unittest.TestCase):
    def test_pre_integration_snapshot_contract(self):
        frame = snapshot()
        qa = operational.validate_snapshot(frame, actual_sha256="a" * 64, expected_sha256="a" * 64)
        self.assertEqual(qa["integration_fields_present"], 0)
        self.assertIn("no blank values invented", qa["before_state_interpretation"])

    def test_snapshot_integrity_fail_closed(self):
        frame = snapshot()
        for broken in (
            frame.iloc[:-1], frame.assign(**{"Airtable Record ID": ["bad"] * 321}),
            frame.assign(entity_id=""),
        ):
            with self.assertRaises(operational.OperationalValidationError):
                operational.validate_snapshot(broken, actual_sha256="a", expected_sha256="a")

    def test_crosswalk_classifications_are_exact_and_conservative(self):
        frame = snapshot()
        exact = frame.iloc[0].to_dict()
        stale = frame.iloc[1].to_dict()
        crosswalks = pd.DataFrame([
            {"entity_id": "ent_exact", "airtable_record_id": exact["Airtable Record ID"], "profile_relationship_type": "group", "counting_entity_id": "ent_exact", "crosswalk_status": "approved", "fingerprint_version": operational.FINGERPRINT_VERSION, "approved_fingerprint": fingerprint(exact, FP_CONFIG)},
            {"entity_id": "ent_stale", "airtable_record_id": stale["Airtable Record ID"], "profile_relationship_type": "brand", "counting_entity_id": "ent_stale", "crosswalk_status": "approved", "fingerprint_version": operational.FINGERPRINT_VERSION, "approved_fingerprint": "apf1:stale"},
            {"entity_id": "ent_missing", "airtable_record_id": "recMissing", "profile_relationship_type": "group", "counting_entity_id": "ent_missing", "crosswalk_status": "approved", "fingerprint_version": operational.FINGERPRINT_VERSION, "approved_fingerprint": "apf1:missing"},
            {"entity_id": "ent_conflict_a", "airtable_record_id": frame.iloc[2]["Airtable Record ID"], "profile_relationship_type": "group", "counting_entity_id": "ent_conflict_a", "crosswalk_status": "approved", "fingerprint_version": operational.FINGERPRINT_VERSION, "approved_fingerprint": fingerprint(frame.iloc[2], FP_CONFIG)},
            {"entity_id": "ent_conflict_b", "airtable_record_id": frame.iloc[2]["Airtable Record ID"], "profile_relationship_type": "group", "counting_entity_id": "ent_conflict_b", "crosswalk_status": "approved", "fingerprint_version": operational.FINGERPRINT_VERSION, "approved_fingerprint": fingerprint(frame.iloc[2], FP_CONFIG)},
        ])
        canonical = pd.DataFrame([
            {"entity_id": "ent_exact", "counting_entity_id": "ent_exact"},
            {"entity_id": "ent_stale", "counting_entity_id": "ent_stale"},
        ])
        valid, qa = operational.classify_crosswalks(
            frame, crosswalks, fingerprint_config=FP_CONFIG,
            canonical_entities=canonical,
        )
        self.assertEqual(valid.entity_id.tolist(), ["ent_exact"])
        self.assertEqual(qa["approved_existing_profile_bindings"], 1)
        self.assertEqual(qa["approved_crosswalk_missing_current_record_id"], 1)
        self.assertEqual(qa["stale_fingerprints"], 1)
        self.assertEqual(qa["invalid_conflicting_bindings"], 1)
        self.assertEqual(qa["unresolved_legacy_profiles"], 318)

    def test_governed_counting_default_approved_nondefault_and_unsupported(self):
        frame = snapshot().iloc[:3].copy()
        entities = ["ent_self", "ent_child", "ent_unsupported"]
        counting = ["", "ent_group", "ent_group"]
        crosswalks = pd.DataFrame([
            {"entity_id": entity, "airtable_record_id": frame.iloc[index]["Airtable Record ID"],
             "profile_relationship_type": "division", "counting_entity_id": counting[index],
             "crosswalk_status": "approved", "fingerprint_version": operational.FINGERPRINT_VERSION, "approved_fingerprint": fingerprint(frame.iloc[index], FP_CONFIG)}
            for index, entity in enumerate(entities)
        ])
        canonical = pd.DataFrame([
            {"entity_id": "ent_self", "counting_entity_id": "ent_self"},
            {"entity_id": "ent_child", "counting_entity_id": "ent_group"},
            {"entity_id": "ent_unsupported", "counting_entity_id": "ent_unsupported"},
        ])
        valid, qa = operational.classify_crosswalks(
            frame, crosswalks, fingerprint_config=FP_CONFIG,
            canonical_entities=canonical,
        )
        self.assertEqual(valid[["entity_id", "counting_entity_id"]].values.tolist(), [
            ["ent_self", "ent_self"], ["ent_child", "ent_group"],
        ])
        self.assertEqual(qa["invalid_counting_mappings"], 1)
        self.assertEqual(qa["unsupported_non_default_counting_mappings"], 1)
        self.assertEqual(qa["invalid_conflicting_bindings"], 1)

    def test_initial_proposals_and_counting(self):
        valid = pd.DataFrame([
            {"entity_id": "ent_group", "profile_relationship_type": "group", "counting_entity_id": "ent_group"},
            {"entity_id": "ent_group", "profile_relationship_type": "division", "counting_entity_id": "ent_group"},
            {"entity_id": "ent_hold", "profile_relationship_type": "brand", "counting_entity_id": "ent_hold"},
        ])
        readiness = pd.DataFrame([
            {"entity_id": "ent_group", "readiness_status": "ready", "assessed_at": "2026-08-17T00:00:00Z", "evidence_reference": "synthetic-ready"},
            {"entity_id": "ent_hold", "readiness_status": "not_ready", "assessed_at": "2026-08-17T00:00:00Z", "evidence_reference": "synthetic-hold"},
        ])
        proposals, qa = operational.build_initial_proposals(valid, readiness, batch_id="sko-025-test")
        self.assertEqual(list(proposals.columns), operational.PROPOSAL_FIELDS)
        self.assertEqual(qa, {"initial_valid_proposals": 2, "holdouts": 1, "unresolved_no_proposal": 0})
        counting = operational.counting_qa(proposals)
        self.assertEqual(counting["profile_count"], 2)
        self.assertEqual(counting["unique_entity_ids"], 1)
        self.assertEqual(counting["unique_counting_entity_ids"], 1)
        self.assertEqual(counting["one_to_many_entity_count"], 1)

    def test_new_profile_candidates_are_review_only_and_rejected_by_handoff(self):
        candidate = {"entity_id": "ent_new", "airtable_record_id": "",
                     "candidate_status": "new_directory_profile_candidate"}
        qa = operational.validate_forward_flow([candidate], [{"entity_id": "ent_hold"}])
        self.assertEqual(qa["new_profile_candidates"], 1)
        config = handoff.load_config(ROOT / "config/directory_integration_handoff_no_write.yaml")
        batch = handoff.build_candidate_batch(
            [candidate], [], [], [], [], [], batch_id="sko-025-new-profile-test",
            created_at="2026-08-17T00:00:00Z", created_by="test", config=config,
        )
        self.assertEqual(batch["manifest"]["holdout_count"], 1)
        self.assertEqual(batch["manifest"]["actionable_count"], 0)
        self.assertFalse(batch["manifest"]["handoff_ready"])
        with self.assertRaises(operational.OperationalValidationError):
            operational.validate_forward_flow([{"airtable_record_id": "recFabricated"}])

    def test_hard_invariants_and_mixed_no_proposal_counts_are_derived(self):
        proposals = pd.DataFrame([{**{field: "" for field in operational.PROPOSAL_FIELDS},
                                   "Organisation": "Synthetic", "publication_status": "published",
                                   "unexpected": "value"}])
        qa = operational.derive_hard_invariants(
            proposals, handoff_batches=[{"manifest": {"handoff_ready": True}}],
        )
        self.assertEqual(qa["allowlist_violations"], 1)
        self.assertEqual(qa["protected_editorial_mutation_attempts"], 1)
        self.assertEqual(qa["publication_mutation_attempts"], 1)
        self.assertEqual(qa["handoff_ready_existing_profile_batch_count"], 1)
        outcomes = operational.proposal_outcome_qa(
            {"unresolved_legacy_profiles": 4, "stale_fingerprints": 2, "invalid_conflicting_bindings": 1},
            {"holdouts": 3},
        )
        self.assertEqual(outcomes["unresolved_no_proposal"], 4)
        self.assertEqual(outcomes["total_no_proposal_profiles"], 10)

    def test_configuration_and_source_have_no_remote_or_apply_capability(self):
        config = yaml.safe_load((ROOT / "config/directory_integration_no_write.yaml").read_text())
        operational.validate_config(config)
        operational.validate_fingerprint_config(FP_CONFIG)
        changed = dict(FP_CONFIG, fingerprint_version="airtable-profile-fingerprint-v2")
        with self.assertRaises(operational.OperationalValidationError):
            operational.validate_fingerprint_config(changed)
        source = inspect.getsource(operational).casefold()
        forbidden = ["requests", "httpx", "pyairtable", "airtable.com", "--write", "--apply"]
        self.assertFalse([token for token in forbidden if token in source])

    def test_deterministic_aggregate_logic(self):
        frame = snapshot()
        first = operational.classify_crosswalks(frame, [], fingerprint_config=FP_CONFIG)
        second = operational.classify_crosswalks(copy.deepcopy(frame), [], fingerprint_config=FP_CONFIG)
        pd.testing.assert_frame_equal(first[0], second[0])
        self.assertEqual(first[1], second[1])


if __name__ == "__main__":
    unittest.main()
