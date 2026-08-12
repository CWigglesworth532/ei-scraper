"""SKO-023 production-shaped, entirely synthetic validation tests."""

from __future__ import annotations

import hashlib
import inspect
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import build_directory_integration_production_fixture as fixture_builder
import directory_integration as integration


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "directory_production_shaped"


def logical_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a row/column-normalized frame for logical comparisons."""
    normalized = frame.copy()
    normalized.columns = normalized.columns.astype(str)
    normalized = normalized.reindex(sorted(normalized.columns), axis=1)
    if not normalized.empty:
        normalized = normalized.astype(str).sort_values(
            list(normalized.columns), kind="stable"
        )
    return normalized.reset_index(drop=True)


class ProductionShapedDirectoryIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tables = integration.load_fixture_pack(FIXTURES)
        cls.results = integration.build_proposals(
            cls.tables,
            actor="sko-023-synthetic-validation",
            timestamp="2026-08-12T10:00:00Z",
        )
        cls.decisions = cls.results["decisions"]
        cls.expected = cls.tables["expected_behaviours"]

    def test_fixture_scale_and_complete_candidate_reconciliation(self) -> None:
        self.assertEqual(len(self.tables["canonical_entities"]), 330)
        self.assertEqual(len(self.tables["directory_candidates"]), 430)
        self.assertEqual(len(self.tables["airtable_profiles"]), 400)
        self.assertEqual(len(self.tables["crosswalks"]), 385)
        self.assertEqual(len(self.decisions), 430)
        self.assertEqual(self.decisions["candidate_id"].nunique(), 430)
        self.assertEqual(
            set(self.decisions["candidate_id"]),
            set(self.tables["directory_candidates"]["candidate_id"]),
        )

    def test_reconciliation_outcomes_and_proposal_counts(self) -> None:
        outcomes = self.decisions["outcome"].value_counts().to_dict()
        self.assertEqual(outcomes, {
            "approved_crosswalk": 335,
            "new_profile_candidate": 25,
            "possible_duplicate_review": 20,
            "identity_conflict": 10,
            "stale_proposal_blocked": 10,
            "merge_review_required": 10,
            "split_suspended": 10,
            "retired_history_retained": 10,
        })
        self.assertEqual(len(self.results["proposals"]), 335)
        self.assertEqual(
            int(self.decisions["proposal_generated"].eq("true").sum()), 335
        )

    def test_scenario_accounting(self) -> None:
        scenarios = self.expected["expected_outcome"].value_counts().to_dict()
        self.assertEqual(scenarios, {
            "clean_existing": 280,
            "new_profile": 25,
            "possible_duplicate": 20,
            "ineligible": 15,
            "incomplete_readiness": 15,
            "ready_unpublished": 15,
            "identity_conflict": 10,
            "stale_fingerprint": 10,
            "canonical_merge": 10,
            "split_pending": 10,
            "retired_profile": 10,
            "published_not_ready": 10,
        })

    def test_duplicate_conflict_stale_merge_split_retirement_blocks(self) -> None:
        expected_outcomes = {
            "possible_duplicate": "possible_duplicate_review",
            "identity_conflict": "identity_conflict",
            "stale_fingerprint": "stale_proposal_blocked",
            "canonical_merge": "merge_review_required",
            "split_pending": "split_suspended",
            "retired_profile": "retired_history_retained",
        }
        joined = self.expected.merge(self.decisions, left_on="test_id", right_on="candidate_id")
        for scenario, outcome in expected_outcomes.items():
            rows = joined.loc[joined["expected_outcome"].eq(scenario)]
            self.assertTrue(rows["outcome"].eq(outcome).all())
            self.assertTrue(rows["proposal_generated"].eq("false").all())
        self.assertEqual(len(self.results["history"]), 10)
        operations = self.results["integration_references"]["proposal_operation"]
        self.assertNotIn("delete", set(operations))

    def test_relationship_types_and_one_to_many_counting(self) -> None:
        relationship_counts = (
            self.tables["directory_candidates"]["profile_relationship_type"]
            .value_counts().to_dict()
        )
        self.assertEqual(relationship_counts, {
            "group": 330, "division": 80, "service_line": 20,
        })
        multi = self.results["proposals"].loc[
            self.results["proposals"]["entity_id"].eq("prod_ent_000")
        ]
        self.assertEqual(
            set(multi["profile_relationship_type"]),
            {"group", "division", "service_line"},
        )
        self.assertEqual(multi["counting_entity_id"].nunique(), 1)
        self.assertEqual(
            self.results["proposals"]["counting_entity_id"].nunique(), 235
        )

    def test_every_active_entity_and_profile_has_counting_entity(self) -> None:
        active_entities = self.tables["canonical_entities"].loc[
            self.tables["canonical_entities"]["entity_status"].eq("active")
        ]
        self.assertFalse(active_entities["counting_entity_id"].eq("").any())
        active_records = set(
            self.tables["airtable_profiles"].loc[
                self.tables["airtable_profiles"]["profile_status"].eq("active"),
                "airtable_record_id",
            ]
        )
        mapped = self.tables["crosswalks"].loc[
            self.tables["crosswalks"]["airtable_record_id"].isin(active_records)
        ].merge(
            self.tables["canonical_entities"][["entity_id", "counting_entity_id"]],
            on="entity_id", how="left",
        )
        # Name-only possible-duplicate profiles are intentionally unmapped review records.
        mapped_records = set(mapped["airtable_record_id"])
        expected_unmapped = {
            value for value in active_records if value.startswith("prod_rec_similar_")
        }
        self.assertEqual(active_records - mapped_records, expected_unmapped)
        self.assertFalse(mapped["counting_entity_id"].eq("").any())

    def test_counting_metrics_do_not_inflate(self) -> None:
        metrics = self.results["counting_metrics"]
        self.assertEqual(metrics["counting_entity_id"].nunique(), 320)
        self.assertEqual(int(metrics["supplier_count"].astype(int).sum()), 320)
        entities = self.tables["canonical_entities"][["entity_id", "counting_entity_id"]]
        expected = self.tables["activity_metrics"].merge(entities, on="entity_id")
        expected = expected.groupby("counting_entity_id", as_index=False).agg(
            supplier_count=("supplier_count", "max"),
            spend_eur=("spend_eur", "max"),
            impact_count=("impact_count", "max"),
        )
        pd.testing.assert_frame_equal(logical_frame(metrics), logical_frame(expected))

    def test_four_gates_remain_independent(self) -> None:
        joined = self.expected.merge(self.decisions, left_on="test_id", right_on="candidate_id")
        ready_unpublished = joined.loc[joined["expected_outcome"].eq("ready_unpublished")]
        self.assertTrue(ready_unpublished["identity_status"].eq("approved_crosswalk").all())
        self.assertTrue(ready_unpublished["classification_status"].eq("eligible").all())
        self.assertTrue(ready_unpublished["readiness_status"].eq("ready").all())
        self.assertTrue(ready_unpublished["publication_status"].eq("unpublished").all())
        published_not_ready = joined.loc[joined["expected_outcome"].eq("published_not_ready")]
        self.assertTrue(published_not_ready["publication_status"].eq("published").all())
        self.assertTrue(published_not_ready["readiness_status"].eq("not_ready").all())
        ineligible = joined.loc[joined["expected_outcome"].eq("ineligible")]
        self.assertTrue(ineligible["identity_status"].eq("approved_crosswalk").all())
        self.assertTrue(ineligible["readiness_status"].eq("not_ready").all())
        self.assertEqual(int(self.decisions["readiness_status"].eq("not_ready").sum()), 40)

    def test_proposal_allowlist_and_zero_protected_fields(self) -> None:
        self.assertEqual(
            list(self.results["proposals"].columns), integration.PROPOSAL_FIELDS
        )
        self.assertEqual(len(integration.PROPOSAL_FIELDS), 8)
        self.assertFalse(
            set(self.results["proposals"].columns) & integration.PROTECTED_FIELDS
        )

    def test_zero_editorial_and_publication_mutations(self) -> None:
        tables = {name: frame.copy(deep=True) for name, frame in self.tables.items()}
        profiles_before = tables["airtable_profiles"].copy(deep=True)
        integration.build_proposals(tables)
        pd.testing.assert_frame_equal(profiles_before, tables["airtable_profiles"])
        self.assertEqual(
            profiles_before["publication_status"].tolist(),
            tables["airtable_profiles"]["publication_status"].tolist(),
        )

    def test_no_mutation_network_or_credentials(self) -> None:
        with self.assertRaises(integration.ProposalOnlyViolation):
            integration.request_mutation("prod_rec_000_group", {})
        sources = (
            inspect.getsource(integration) + inspect.getsource(fixture_builder)
        ).casefold()
        forbidden = [
            "import requests", "import urllib", "httpx", "aiohttp",
            "pyairtable", "airtable_api", "os.environ", "webhook",
        ]
        self.assertFalse([token for token in forbidden if token in sources])

    def test_transition_audit_is_complete(self) -> None:
        transitions = self.results["transitions"]
        self.assertEqual(len(transitions), 430)
        required = ["batch_id", "actor", "timestamp", "reason", "evidence"]
        self.assertFalse(transitions[required].eq("").any().any())

    def test_independent_reruns_are_identical(self) -> None:
        second = integration.build_proposals(
            integration.load_fixture_pack(FIXTURES),
            actor="sko-023-synthetic-validation",
            timestamp="2026-08-12T10:00:00Z",
        )
        for name in self.results:
            pd.testing.assert_frame_equal(self.results[name], second[name])
        first_hash = hashlib.sha256(
            self.results["proposals"].to_csv(index=False).encode()
        ).hexdigest()
        second_hash = hashlib.sha256(
            second["proposals"].to_csv(index=False).encode()
        ).hexdigest()
        self.assertEqual(first_hash, second_hash)

    def test_shuffled_input_is_logically_identical(self) -> None:
        shuffled = {
            name: frame.sample(frac=1, random_state=23000 + index).reset_index(drop=True)
            for index, (name, frame) in enumerate(self.tables.items())
        }
        shuffled_results = integration.build_proposals(
            shuffled,
            actor="sko-023-synthetic-validation",
            timestamp="2026-08-12T10:00:00Z",
        )
        for name in self.results:
            pd.testing.assert_frame_equal(
                logical_frame(self.results[name]),
                logical_frame(shuffled_results[name]),
                obj=name,
            )

    def test_committed_fixture_pack_matches_generator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            generated = Path(temporary)
            fixture_builder.write_fixture_pack(generated)
            expected_files = sorted(path.name for path in FIXTURES.glob("*.csv"))
            self.assertEqual(
                expected_files,
                sorted(path.name for path in generated.glob("*.csv")),
            )
            for filename in expected_files:
                self.assertEqual(
                    (FIXTURES / filename).read_bytes(),
                    (generated / filename).read_bytes(),
                    filename,
                )


if __name__ == "__main__":
    unittest.main()
