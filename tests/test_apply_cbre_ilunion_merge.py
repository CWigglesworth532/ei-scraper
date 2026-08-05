"""Synthetic tests for the DEC-007 ILUNION merge."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from apply_cbre_ilunion_merge import (
    DUPLICATE_ENTITY_ID,
    ILUNION_SUPPLIERS,
    SURVIVOR_ENTITY_ID,
    apply_merge,
    checksum_directory,
)


class ApplyCbreIlunionMergeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)

        self.operational = root / "operational"
        self.working = root / "working"
        self.evidence = root / "evidence"

        self.operational.mkdir()
        self.working.mkdir()

        self._write_store(self.operational)
        self._write_store(self.working)

        owner_rows = []
        for supplier_id, supplier_name in (
            ILUNION_SUPPLIERS.items()
        ):
            owner_rows.append(
                {
                    "supplier_baseline_id": supplier_id,
                    "supplier_name": supplier_name,
                    "country_code": "ES",
                    "owner_decision": (
                        "reuse_existing_entity"
                    ),
                    "owner_approved_entity_id": (
                        SURVIVOR_ENTITY_ID
                    ),
                    "owner_review_notes": "DEC-007",
                }
            )

        self.owner_review = root / "owner_review.csv"
        pd.DataFrame(owner_rows).to_csv(
            self.owner_review,
            index=False,
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_store(self, store: Path) -> None:
        pd.DataFrame(
            [
                {
                    "entity_id": SURVIVOR_ENTITY_ID,
                    "canonical_name": "ILUNION RETAIL",
                    "canonical_name_norm": "ilunionretail",
                    "country": "ES",
                    "entity_status": "unknown",
                    "record_status": "active",
                    "legal_form_local": "",
                    "base_legal_form_family": "",
                    "primary_identifier_id": "",
                    "primary_source_record_id": "src1",
                    "identity_confidence": "high",
                    "identity_review_status": (
                        "machine_resolved"
                    ),
                    "created_at": "2026-01-01",
                    "created_by": "test",
                    "updated_at": "2026-01-01",
                    "schema_version": "1.0.0",
                },
                {
                    "entity_id": DUPLICATE_ENTITY_ID,
                    "canonical_name": "ILUNION LAVANDERIAS",
                    "canonical_name_norm": (
                        "ilunionlavanderias"
                    ),
                    "country": "ES",
                    "entity_status": "unknown",
                    "record_status": "active",
                    "legal_form_local": "",
                    "base_legal_form_family": "",
                    "primary_identifier_id": "",
                    "primary_source_record_id": "",
                    "identity_confidence": "reviewed",
                    "identity_review_status": "reviewed",
                    "created_at": "2026-01-02",
                    "created_by": "test",
                    "updated_at": "2026-01-02",
                    "schema_version": "1.0.0",
                },
            ]
        ).to_parquet(
            store / "canonical_entities.parquet",
            index=False,
        )

        pd.DataFrame(
            [
                {
                    "supplier_record_key": "bayer_test",
                    "supplier_name_original": (
                        "ILUNION LAVANDERIAS"
                    ),
                    "supplier_name_norm": (
                        "ilunionlavanderias"
                    ),
                    "supplier_country": "ES",
                    "supplier_identifier_type": "",
                    "supplier_identifier_value": "",
                    "acceptance_status": (
                        "reviewed_confirmed"
                    ),
                    "source_client": "Bayer",
                    "source_client_file": "bayer.csv",
                    "reviewed_by": "owner",
                    "reviewed_at": "2026-01-02",
                    "matched_entity_id": DUPLICATE_ENTITY_ID,
                    "matched_source_record_id": "",
                    "resolution_status": "new",
                    "resolution_method": (
                        "new_accepted_singleton_entity"
                    ),
                    "allocate_new_entity": True,
                    "review_required": False,
                    "candidate_entity_ids": "",
                    "resolution_reason": "",
                    "relationship_type": "",
                    "related_entity_id": "",
                }
            ]
        ).to_parquet(
            store / "supplier_entity_links.parquet",
            index=False,
        )

        pd.DataFrame(
            [
                {
                    "alias_id": "alias_duplicate",
                    "entity_id": DUPLICATE_ENTITY_ID,
                    "alias_name": "ILUNION LAVANDERIAS",
                    "alias_name_norm": (
                        "ilunionlavanderias"
                    ),
                    "alias_type": "client_variant",
                    "language": "",
                    "country": "ES",
                    "is_preferred": False,
                    "valid_from": None,
                    "valid_to": None,
                    "source_record_id": "",
                    "evidence_id": "",
                    "verification_status": (
                        "human_verified"
                    ),
                    "review_status": "accepted",
                    "source_client": "Bayer",
                    "supplier_record_key": "bayer_test",
                    "created_at": "2026-01-02",
                    "schema_version": "1.0.0",
                }
            ]
        ).to_parquet(
            store / "entity_aliases.parquet",
            index=False,
        )

        pd.DataFrame(
            columns=[
                "identifier_id",
                "entity_id",
                "identifier_type",
                "identifier_value_raw",
                "identifier_value_normalized",
                "country",
                "identifier_scope",
                "is_primary",
                "verification_status",
                "source_record_id",
                "created_at",
                "schema_version",
            ]
        ).to_parquet(
            store / "entity_identifiers.parquet",
            index=False,
        )

        pd.DataFrame(
            [
                {
                    "source_record_id": "src1",
                    "entity_id": SURVIVOR_ENTITY_ID,
                    "source_id": "source",
                    "source_record_key": "source|1",
                    "continuity_key": "source|1",
                    "continuity_method": "tax",
                    "record_occurrence": 0,
                    "observation_key": "obs1",
                    "ingest_batch_id": "batch1",
                    "source_row_number": 1,
                    "entity_name_raw": "ILUNION RETAIL",
                    "entity_name_norm": "ilunionretail",
                    "country": "ES",
                    "address_raw": "",
                    "city": "",
                    "postcode": "",
                    "legal_form_local": "",
                    "source_status": "",
                    "retrieved_at": "2026-01-01",
                    "source_url": "",
                    "source_page": "",
                    "record_fingerprint": "fingerprint",
                    "resolution_status": "linked",
                    "resolution_method": "identifier",
                    "resolution_confidence": "high",
                    "resolution_reason": "",
                    "raw_payload_json": "{}",
                    "schema_version": "1.0.0",
                }
            ]
        ).to_parquet(
            store / "source_records.parquet",
            index=False,
        )

        pd.DataFrame(
            columns=[
                "trusted_term_id",
                "entity_id",
                "source_record_id",
                "term_type",
                "term_raw",
                "term_normalized",
                "country",
                "identifier_type",
                "identifier_value_normalized",
                "identifier_scope",
                "relationship_type",
                "verification_status",
                "review_status",
                "approved_for_matching",
                "approved_by",
                "approved_at",
                "evidence_reference",
                "source_client",
                "supplier_record_key",
                "created_at",
                "schema_version",
            ]
        ).to_parquet(
            store / "trusted_match_terms.parquet",
            index=False,
        )

        pd.DataFrame(
            columns=[
                "event_id",
                "event_type",
                "subject_entity_id",
                "target_entity_id",
                "reason",
                "decision_status",
                "decided_by",
                "decided_at",
                "effective_at",
                "migration_batch_id",
                "schema_version",
            ]
        ).to_parquet(
            store / "entity_events.parquet",
            index=False,
        )

        pd.DataFrame(
            columns=[
                "relationship_id",
                "subject_entity_id",
                "object_entity_id",
                "relationship_type",
                "relationship_status",
                "evidence_reference",
                "source_client",
                "supplier_record_key",
                "reviewed_by",
                "reviewed_at",
                "created_at",
                "schema_version",
            ]
        ).to_parquet(
            store / "entity_relationships.parquet",
            index=False,
        )

    def run_merge(self):
        return apply_merge(
            working_store=self.working,
            operational_store=self.operational,
            owner_review_path=self.owner_review,
            evidence_dir=self.evidence,
            decision_at="2026-08-05T12:00:00+00:00",
            decided_by="Charlie Wigglesworth",
            source_file="aggregate.xlsx",
        )

    def test_merge_redirects_duplicate_and_adds_links(
        self,
    ) -> None:
        summary = self.run_merge()

        self.assertTrue(
            summary["operational_store_unchanged"]
        )
        self.assertEqual(
            summary["qa"]["cbre_supplier_links_added"],
            4,
        )

        links = pd.read_parquet(
            self.working
            / "supplier_entity_links.parquet"
        )

        self.assertEqual(len(links), 5)
        self.assertTrue(
            links["matched_entity_id"]
            .eq(SURVIVOR_ENTITY_ID)
            .all()
        )

        bayer = links.loc[
            links["supplier_record_key"].eq("bayer_test")
        ]

        self.assertEqual(len(bayer), 1)
        self.assertEqual(
            bayer.iloc[0]["matched_entity_id"],
            SURVIVOR_ENTITY_ID,
        )
        self.assertEqual(
            bayer.iloc[0]["resolution_method"],
            "owner_approved_canonical_merge",
        )
        self.assertFalse(
            bool(bayer.iloc[0]["allocate_new_entity"])
        )

    def test_duplicate_entity_is_preserved_as_merged(
        self,
    ) -> None:
        self.run_merge()

        entities = pd.read_parquet(
            self.working / "canonical_entities.parquet"
        )

        survivor = entities.loc[
            entities["entity_id"].eq(
                SURVIVOR_ENTITY_ID
            )
        ].iloc[0]
        duplicate = entities.loc[
            entities["entity_id"].eq(
                DUPLICATE_ENTITY_ID
            )
        ].iloc[0]

        self.assertEqual(
            survivor["canonical_name"],
            "ILUNION",
        )
        self.assertEqual(
            duplicate["record_status"],
            "merged",
        )
        self.assertEqual(
            duplicate["entity_status"],
            "superseded",
        )

    def test_merge_creates_relationship_and_event(
        self,
    ) -> None:
        self.run_merge()

        relationships = pd.read_parquet(
            self.working
            / "entity_relationships.parquet"
        )
        events = pd.read_parquet(
            self.working / "entity_events.parquet"
        )

        self.assertEqual(len(relationships), 1)
        self.assertEqual(
            relationships.iloc[0][
                "relationship_type"
            ],
            "merged_into",
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(
            events.iloc[0]["event_type"],
            "merge",
        )

    def test_operational_store_is_unchanged(self) -> None:
        before = checksum_directory(self.operational)
        self.run_merge()
        after = checksum_directory(self.operational)

        self.assertEqual(before, after)

    def test_rerun_is_idempotent(self) -> None:
        first_summary = self.run_merge()
        first = checksum_directory(self.working)

        second_summary = self.run_merge()
        second = checksum_directory(self.working)

        self.assertEqual(first, second)
        self.assertGreater(
            first_summary[
                "working_store_file_write_count"
            ],
            0,
        )
        self.assertEqual(
            second_summary[
                "working_store_file_write_count"
            ],
            0,
        )
        self.assertEqual(
            second_summary[
                "working_store_files_written"
            ],
            [],
        )


if __name__ == "__main__":
    unittest.main()
