"""Synthetic tests for the E1.6 directory-candidate export."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from build_directory_candidate_export import (
    OUTPUT_COLUMNS,
    build_directory_candidate_export,
    write_export,
)


class DirectoryCandidateExportTests(unittest.TestCase):
    """Test directory-candidate behaviour with synthetic data only."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base = Path(self.temp_dir.name) / "canonical"
        self.base.mkdir(parents=True)

        entities = pd.DataFrame(
            [
                {
                    "entity_id": "sko_ent_active_1",
                    "canonical_name": "Synthetic Circular Works",
                    "canonical_name_norm": "syntheticcircularworks",
                    "country": "IE",
                    "entity_status": "active",
                    "record_status": "active",
                    "primary_source_record_id": "sko_src_1",
                    "identity_confidence": "high",
                    "identity_review_status": "reviewed",
                },
                {
                    "entity_id": "sko_ent_active_2",
                    "canonical_name": "Synthetic Inclusion Services",
                    "canonical_name_norm": "syntheticinclusionservices",
                    "country": "DE",
                    "entity_status": "unknown",
                    "record_status": "active",
                    "primary_source_record_id": "sko_src_2",
                    "identity_confidence": "high",
                    "identity_review_status": "machine_resolved",
                },
                {
                    "entity_id": "sko_ent_unlinked",
                    "canonical_name": "Synthetic Unlinked Entity",
                    "canonical_name_norm": "syntheticunlinkedentity",
                    "country": "FR",
                    "entity_status": "active",
                    "record_status": "active",
                    "primary_source_record_id": "sko_src_3",
                    "identity_confidence": "high",
                    "identity_review_status": "reviewed",
                },
                {
                    "entity_id": "sko_ent_merged",
                    "canonical_name": "Synthetic Merged Entity",
                    "canonical_name_norm": "syntheticmergedentity",
                    "country": "BE",
                    "entity_status": "superseded",
                    "record_status": "merged",
                    "primary_source_record_id": "sko_src_4",
                    "identity_confidence": "high",
                    "identity_review_status": "reviewed",
                },
            ]
        )

        identifiers = pd.DataFrame(
            [
                {
                    "identifier_id": "sko_id_1_secondary",
                    "entity_id": "sko_ent_active_1",
                    "identifier_type": "IE_OTHER",
                    "identifier_value_normalized": "secondary",
                    "is_primary": False,
                    "verification_status": "source_verified",
                },
                {
                    "identifier_id": "sko_id_1_primary",
                    "entity_id": "sko_ent_active_1",
                    "identifier_type": "IE_CHARITY",
                    "identifier_value_normalized": "20000001",
                    "is_primary": True,
                    "verification_status": "human_verified",
                },
                {
                    "identifier_id": "sko_id_2",
                    "entity_id": "sko_ent_active_2",
                    "identifier_type": "DE_HRB",
                    "identifier_value_normalized": "hrb12345",
                    "is_primary": True,
                    "verification_status": "source_verified",
                },
                {
                    "identifier_id": "sko_id_rejected",
                    "entity_id": "sko_ent_active_2",
                    "identifier_type": "DE_VAT",
                    "identifier_value_normalized": "rejectedvalue",
                    "is_primary": True,
                    "verification_status": "rejected",
                },
            ]
        )

        source_records = pd.DataFrame(
            [
                {
                    "source_record_id": "sko_src_1",
                    "entity_id": "sko_ent_active_1",
                    "source_url": "https://example.test/ie/1",
                    "resolution_method": (
                        "authoritative_identifier"
                    ),
                    "raw_payload_json": json.dumps(
                        {
                            "ei_register_name": (
                                "Synthetic Irish Register"
                            ),
                            "se_recognition_type": (
                                "official_social_enterprise_register"
                            ),
                            "se_recognition_name": "SYNTHETIC_IE",
                            "se_recognition_evidence": (
                                "Synthetic recognition evidence"
                            ),
                            "base_legal_form_family": "CHARITY",
                        }
                    ),
                },
                {
                    "source_record_id": "sko_src_2",
                    "entity_id": "sko_ent_active_2",
                    "source_url": "https://example.test/de/2",
                    "resolution_method": (
                        "new_authoritative_identifier_cluster"
                    ),
                    "raw_payload_json": json.dumps(
                        {
                            "ei_register_name": (
                                "Synthetic German Register"
                            ),
                            "se_recognition_type": "legal_form",
                            "se_recognition_name": "GGMBH",
                            "se_recognition_evidence": (
                                "Synthetic legal-form evidence"
                            ),
                            "base_legal_form_family": (
                                "WORK_INTEGRATION"
                            ),
                        }
                    ),
                },
                {
                    "source_record_id": "sko_src_3",
                    "entity_id": "sko_ent_unlinked",
                    "source_url": "https://example.test/fr/3",
                    "resolution_method": (
                        "authoritative_identifier"
                    ),
                    "raw_payload_json": "{}",
                },
                {
                    "source_record_id": "sko_src_4",
                    "entity_id": "sko_ent_merged",
                    "source_url": "https://example.test/be/4",
                    "resolution_method": (
                        "authoritative_identifier"
                    ),
                    "raw_payload_json": "{}",
                },
            ]
        )

        links = pd.DataFrame(
            [
                {
                    "supplier_record_key": "client_a_1",
                    "matched_entity_id": "sko_ent_active_1",
                    "acceptance_status": "reviewed_confirmed",
                    "review_required": False,
                    "source_client": "Synthetic Client A",
                    "reviewed_at": "2026-08-01T10:00:00+00:00",
                },
                {
                    "supplier_record_key": "client_b_1",
                    "matched_entity_id": "sko_ent_active_1",
                    "acceptance_status": "reviewed_confirmed",
                    "review_required": False,
                    "source_client": "Synthetic Client B",
                    "reviewed_at": "2026-08-02T10:00:00+00:00",
                },
                {
                    "supplier_record_key": "client_a_2",
                    "matched_entity_id": "sko_ent_active_2",
                    "acceptance_status": "reviewed_confirmed",
                    "review_required": False,
                    "source_client": "Synthetic Client A",
                    "reviewed_at": "2026-08-03T10:00:00+00:00",
                },
                {
                    "supplier_record_key": "merged_link",
                    "matched_entity_id": "sko_ent_merged",
                    "acceptance_status": "reviewed_confirmed",
                    "review_required": False,
                    "source_client": "Synthetic Client A",
                    "reviewed_at": "2026-08-03T10:00:00+00:00",
                },
                {
                    "supplier_record_key": "review_link",
                    "matched_entity_id": "sko_ent_unlinked",
                    "acceptance_status": "reviewed_confirmed",
                    "review_required": True,
                    "source_client": "Synthetic Client C",
                    "reviewed_at": "2026-08-03T10:00:00+00:00",
                },
            ]
        )

        entities.to_parquet(
            self.base / "canonical_entities.parquet",
            index=False,
        )
        identifiers.to_parquet(
            self.base / "entity_identifiers.parquet",
            index=False,
        )
        source_records.to_parquet(
            self.base / "source_records.parquet",
            index=False,
        )
        links.to_parquet(
            self.base / "supplier_entity_links.parquet",
            index=False,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_one_row_per_eligible_active_entity(self) -> None:
        output, qa = build_directory_candidate_export(
            self.base
        )

        self.assertEqual(len(output), 2)
        self.assertEqual(
            set(output["entity_id"]),
            {
                "sko_ent_active_1",
                "sko_ent_active_2",
            },
        )
        self.assertEqual(
            output["entity_id"].duplicated().sum(),
            0,
        )
        self.assertEqual(qa["eligible_export_rows"], 2)
        self.assertEqual(qa["duplicate_entity_id_count"], 0)

    def test_supplier_links_are_aggregated(self) -> None:
        output, _ = build_directory_candidate_export(
            self.base
        )

        row = output.loc[
            output["entity_id"].eq("sko_ent_active_1")
        ].iloc[0]

        self.assertEqual(
            row["accepted_supplier_link_count"],
            2,
        )
        self.assertEqual(row["source_client_count"], 2)
        self.assertEqual(
            row["source_clients"],
            "Synthetic Client A; Synthetic Client B",
        )
        self.assertEqual(
            row["known_client_relationship_count"],
            2,
        )
        self.assertEqual(
            row["last_link_reviewed_at"],
            "2026-08-02T10:00:00+00:00",
        )

    def test_primary_identifier_is_selected(self) -> None:
        output, _ = build_directory_candidate_export(
            self.base
        )

        row = output.loc[
            output["entity_id"].eq("sko_ent_active_1")
        ].iloc[0]

        self.assertEqual(
            row["primary_identifier_type"],
            "IE_CHARITY",
        )
        self.assertEqual(
            row["primary_identifier_value"],
            "20000001",
        )

        row_2 = output.loc[
            output["entity_id"].eq("sko_ent_active_2")
        ].iloc[0]

        self.assertEqual(
            row_2["primary_identifier_type"],
            "DE_HRB",
        )
        self.assertNotEqual(
            row_2["primary_identifier_value"],
            "rejectedvalue",
        )

    def test_source_evidence_is_not_classification(self) -> None:
        output, _ = build_directory_candidate_export(
            self.base
        )

        row = output.loc[
            output["entity_id"].eq("sko_ent_active_1")
        ].iloc[0]

        self.assertEqual(
            row["source_register_name"],
            "Synthetic Irish Register",
        )
        self.assertEqual(
            row["source_recognition_type"],
            "official_social_enterprise_register",
        )
        self.assertEqual(row["classification_status"], "")
        self.assertEqual(
            row["directory_candidate_status"],
            "research_needed",
        )
        self.assertEqual(
            row["directory_inclusion_decision"],
            "review",
        )
        self.assertEqual(row["Verified"], "No")

    def test_airtable_and_materiality_defaults(self) -> None:
        output, _ = build_directory_candidate_export(
            self.base
        )

        row = output.iloc[0]

        self.assertEqual(row["Website"], "")
        self.assertTrue(row["missing_website"])
        self.assertEqual(row["Data Status"], "Review")
        self.assertEqual(row["known_spend_eur"], "")
        self.assertEqual(
            row["spend_data_status"],
            "not_available",
        )
        self.assertEqual(
            row["has_100k_plus_relationship"],
            "",
        )

    def test_output_schema_and_csv_are_deterministic(self) -> None:
        first, _ = build_directory_candidate_export(
            self.base
        )
        second, _ = build_directory_candidate_export(
            self.base
        )

        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(list(first.columns), OUTPUT_COLUMNS)

        output_path = Path(self.temp_dir.name) / "output.csv"
        write_export(self.base, output_path)

        written = pd.read_csv(
            output_path,
            dtype=str,
            keep_default_na=False,
        )

        self.assertEqual(
            written["entity_id"].tolist(),
            sorted(written["entity_id"].tolist()),
        )


if __name__ == "__main__":
    unittest.main()
