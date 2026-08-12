"""Synthetic behavioural tests for SKO-022 (T01-T20)."""

from __future__ import annotations

import hashlib
import inspect
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import directory_integration as integration


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "directory"
CONFIG = ROOT / "config" / "directory_integration_no_write.yaml"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DirectoryIntegrationBehaviourTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tables = integration.load_fixture_pack(FIXTURES)
        cls.results = integration.build_proposals(cls.tables)
        cls.decisions = cls.results["decisions"].set_index("candidate_id")

    def decision(self, candidate_id: str, field: str) -> str:
        return str(self.decisions.loc[candidate_id, field])

    def test_t01_new_candidate_no_write(self) -> None:
        self.assertEqual(self.decision("c_new", "outcome"), "new_profile_candidate")
        self.assertEqual(self.decision("c_new", "proposal_generated"), "false")

    def test_t02_approved_crosswalk_reuses_record(self) -> None:
        self.assertEqual(self.decision("c_reuse", "airtable_record_id"), "rec_reuse")
        self.assertEqual(self.decision("c_reuse", "identity_status"), "approved_crosswalk")

    def test_t03_name_similarity_is_review_only(self) -> None:
        self.assertEqual(self.decision("c_similar", "identity_status"), "possible_duplicate")
        self.assertEqual(self.decision("c_similar", "proposal_generated"), "false")

    def test_t04_multiple_profiles_one_counting_entity(self) -> None:
        candidates = self.tables["directory_candidates"]
        rows = candidates.loc[candidates["entity_id"].eq("ent_multi")]
        self.assertEqual(set(rows["profile_relationship_type"]), {"group", "division", "service_line"})
        self.assertEqual(self.tables["canonical_entities"].loc[
            self.tables["canonical_entities"]["entity_id"].eq("ent_multi"), "counting_entity_id"
        ].nunique(), 1)

    def test_t05_identity_conflict_blocks_proposals(self) -> None:
        for candidate in ("c_conflict_a", "c_conflict_b"):
            self.assertEqual(self.decision(candidate, "outcome"), "identity_conflict")
            self.assertEqual(self.decision(candidate, "proposal_generated"), "false")

    def test_t06_incomplete_enrichment_not_ready(self) -> None:
        self.assertEqual(self.decision("c_incomplete", "readiness_status"), "not_ready")

    def test_t07_ineligible_not_ready_or_publishable(self) -> None:
        self.assertEqual(self.decision("c_ineligible", "readiness_status"), "not_ready")
        self.assertEqual(self.decision("c_ineligible", "publishable"), "false")

    def test_t08_ready_does_not_imply_published(self) -> None:
        self.assertEqual(self.decision("c_ready", "readiness_status"), "ready")
        self.assertEqual(self.decision("c_ready", "publication_status"), "unpublished")

    def test_t09_crosswalk_does_not_alter_publication(self) -> None:
        self.assertEqual(self.decision("c_reuse", "identity_status"), "approved_crosswalk")
        self.assertEqual(self.decision("c_reuse", "publication_status"), "published")

    def test_t10_publication_feedback_preserves_identity_classification(self) -> None:
        self.assertEqual(self.decision("c_feedback", "publication_status"), "published")
        self.assertEqual(self.decision("c_feedback", "identity_status"), "approved_crosswalk")
        self.assertEqual(self.decision("c_feedback", "classification_status"), "eligible")

    def test_t11_merge_redirect_requires_review(self) -> None:
        self.assertEqual(self.decision("c_merge", "entity_id"), "ent_merge_survivor")
        self.assertEqual(self.decision("c_merge", "outcome"), "merge_review_required")
        self.assertEqual(self.decision("c_merge", "proposal_generated"), "false")

    def test_t12_split_suspends_synchronization(self) -> None:
        self.assertEqual(self.decision("c_split", "outcome"), "split_suspended")
        self.assertEqual(self.decision("c_split", "proposal_generated"), "false")

    def test_t13_changed_fingerprint_blocks_stale_proposal(self) -> None:
        self.assertEqual(self.decision("c_stale", "outcome"), "stale_proposal_blocked")
        self.assertEqual(self.decision("c_stale", "proposal_generated"), "false")

    def test_t14_rollback_only_integration_owned_fields(self) -> None:
        fields = set(self.results["rollback_fields"].columns)
        self.assertTrue(fields)
        self.assertTrue(fields <= set(integration.PROPOSAL_FIELDS))
        self.assertFalse(fields & integration.PROTECTED_FIELDS)
        profile = self.tables["airtable_profiles"].set_index("airtable_record_id")
        self.assertEqual(profile.loc["rec_rollback", "Organisation"], "Synthetic Editorial Name Later")

    def test_t15_profiles_do_not_inflate_metrics(self) -> None:
        row = self.results["counting_metrics"].set_index("counting_entity_id").loc["ent_count"]
        self.assertEqual(str(row["supplier_count"]), "1")
        self.assertEqual(str(row["spend_eur"]), "1000")
        self.assertEqual(str(row["impact_count"]), "2")

    def test_t16_transition_metadata_is_complete(self) -> None:
        required = {"batch_id", "actor", "timestamp", "reason", "evidence"}
        self.assertTrue(required <= set(self.results["transitions"].columns))
        self.assertFalse(self.results["transitions"][list(required)].eq("").any().any())
        self.assertEqual(len(self.results["transitions"]), len(self.tables["directory_candidates"]))

    def test_t17_proposals_have_only_allowed_fields(self) -> None:
        self.assertEqual(list(self.results["proposals"].columns), integration.PROPOSAL_FIELDS)
        self.assertFalse(set(self.results["proposals"].columns) & integration.PROTECTED_FIELDS)

    def test_t18_mutation_attempt_is_hard_error(self) -> None:
        with self.assertRaises(integration.ProposalOnlyViolation):
            integration.request_mutation("rec_reuse", {"entity_id": "ent_reuse"})

    def test_t19_crosswalk_approval_leaves_publication_unchanged(self) -> None:
        tables = {name: frame.copy() for name, frame in self.tables.items()}
        before = tables["airtable_profiles"].set_index("airtable_record_id")["publication_status"].to_dict()
        integration.build_proposals(tables)
        after = tables["airtable_profiles"].set_index("airtable_record_id")["publication_status"].to_dict()
        self.assertEqual(before, after)

    def test_t20_retired_profile_keeps_history_without_delete(self) -> None:
        self.assertEqual(self.decision("c_retired", "outcome"), "retired_history_retained")
        self.assertIn("ent_retired", set(self.results["history"]["entity_id"]))
        operations = set(self.results["integration_references"].get("proposal_operation", pd.Series(dtype=str)))
        self.assertNotIn("delete", operations)

    def test_fixture_accounting_and_active_counting_ids(self) -> None:
        self.assertEqual(set(self.tables["expected_behaviours"]["test_id"]), {f"T{i:02d}" for i in range(1, 21)})
        active = self.tables["canonical_entities"].loc[
            self.tables["canonical_entities"]["entity_status"].eq("active")
        ]
        self.assertFalse(active["counting_entity_id"].eq("").any())

    def test_idempotent_rerun_and_input_checksums(self) -> None:
        before = {path.name: sha256(path) for path in sorted(FIXTURES.glob("*.csv"))}
        second = integration.build_proposals(integration.load_fixture_pack(FIXTURES))
        for name in self.results:
            pd.testing.assert_frame_equal(self.results[name], second[name])
        after = {path.name: sha256(path) for path in sorted(FIXTURES.glob("*.csv"))}
        self.assertEqual(before, after)
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first_hashes = integration.write_outputs(self.results, Path(first_dir))
            second_hashes = integration.write_outputs(second, Path(second_dir))
            self.assertEqual(first_hashes, second_hashes)

    def test_config_and_source_have_no_network_or_credentials(self) -> None:
        config = integration.load_config(CONFIG)
        self.assertFalse(config["network_enabled"])
        self.assertFalse(config["credentials_enabled"])
        self.assertFalse(config["mutation_enabled"])
        source = inspect.getsource(integration)
        forbidden = ["import requests", "import urllib", "httpx", "airtable_api", "os.environ", "webhook"]
        self.assertFalse([token for token in forbidden if token in source.casefold()])


if __name__ == "__main__":
    unittest.main()
