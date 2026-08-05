"""Synthetic tests for aggregate supplier reconciliation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from build_aggregate_supplier_reconciliation import (
    build_manifest_summary,
    checksum_directory,
    checksum_manifest_digest,
    decide_reconciliation_category,
    expand_source_rows,
    normalize_identifier,
    normalize_name,
    select_baseline_rows,
    validate_reconciliation,
)


class AggregateSupplierReconciliationTests(unittest.TestCase):
    """Test governed reconciliation behaviour using synthetic data."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base = Path(self.temp_dir.name)
        self.store = self.base / "canonical"
        self.store.mkdir()

        (self.store / "canonical_entities.parquet").write_bytes(
            b"synthetic entities"
        )
        (self.store / "supplier_entity_links.parquet").write_bytes(
            b"synthetic links"
        )

        self.aggregate = pd.DataFrame(
            [
                {
                    "supplier_baseline_id": "SB00001",
                    "baseline_id": "Input 2",
                    "supplier_name": "Synthetic Cooperative",
                    "supplier_name_norm": "synthetic cooperative",
                    "country_code": "IE",
                    "country_name": "Ireland",
                    "spend_total_eur": 100.25,
                    "source_record_ids_merged": (
                        "ROW00001; ROW00002"
                    ),
                    "supplier_name_variants": (
                        "Synthetic Cooperative"
                    ),
                },
                {
                    "supplier_baseline_id": "SB00002",
                    "baseline_id": "Input 2",
                    "supplier_name": "Synthetic Foundation",
                    "supplier_name_norm": "synthetic foundation",
                    "country_code": "FR",
                    "country_name": "France",
                    "spend_total_eur": 200.75,
                    "source_record_ids_merged": "ROW00003",
                    "supplier_name_variants": (
                        "Synthetic Foundation"
                    ),
                },
                {
                    "supplier_baseline_id": "SB99999",
                    "baseline_id": "Input 1",
                    "supplier_name": "Other Client Supplier",
                    "supplier_name_norm": "other client supplier",
                    "country_code": "DE",
                    "country_name": "Germany",
                    "spend_total_eur": 999.00,
                    "source_record_ids_merged": "ROW99999",
                    "supplier_name_variants": (
                        "Other Client Supplier"
                    ),
                },
            ]
        )

        self.row_level = pd.DataFrame(
            [
                {
                    "record_id": "ROW00001",
                    "source_input": "Input 2",
                    "source_row_number": 2,
                    "supplier_name": "Synthetic Cooperative",
                    "supplier_name_norm": "synthetic cooperative",
                    "original_supplier_name": (
                        "Synthetic Cooperative"
                    ),
                    "erp_supplier_name": "",
                    "erp_supplier_number": "",
                    "country_code": "IE",
                    "spend_amount_eur": 40.00,
                },
                {
                    "record_id": "ROW00002",
                    "source_input": "Input 2",
                    "source_row_number": 3,
                    "supplier_name": "Synthetic Cooperative",
                    "supplier_name_norm": "synthetic cooperative",
                    "original_supplier_name": (
                        "Synthetic Cooperative"
                    ),
                    "erp_supplier_name": "",
                    "erp_supplier_number": "",
                    "country_code": "IE",
                    "spend_amount_eur": 60.25,
                },
                {
                    "record_id": "ROW00003",
                    "source_input": "Input 2",
                    "source_row_number": 4,
                    "supplier_name": "Synthetic Foundation",
                    "supplier_name_norm": "synthetic foundation",
                    "original_supplier_name": (
                        "Synthetic Foundation"
                    ),
                    "erp_supplier_name": "",
                    "erp_supplier_number": "",
                    "country_code": "FR",
                    "spend_amount_eur": 200.75,
                },
            ]
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def selected(self) -> pd.DataFrame:
        return select_baseline_rows(
            self.aggregate,
            "Input 2",
        )

    def reconciliation_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "supplier_baseline_id": "SB00001",
                    "spend_total_eur": 100.25,
                    "reconciliation_category": (
                        "reuse_existing_identifier_or_source_supported"
                    ),
                    "evidence_basis": (
                        "accepted_identifier_or_persisted_source_evidence"
                    ),
                },
                {
                    "supplier_baseline_id": "SB00002",
                    "spend_total_eur": 200.75,
                    "reconciliation_category": (
                        "create_new_entity_candidate"
                    ),
                    "evidence_basis": (
                        "historical_positive_evidence_without_existing_entity"
                    ),
                },
            ]
        )

    def test_name_and_identifier_normalisation(self) -> None:
        self.assertEqual(
            normalize_name("  Société  Coopérative, S.L. "),
            "societe cooperative s l",
        )
        self.assertEqual(
            normalize_identifier(" IE-123 456 "),
            "IE123456",
        )

    def test_source_row_expansion_and_exact_resolution(self) -> None:
        resolved = expand_source_rows(
            self.selected(),
            self.row_level,
        )

        self.assertEqual(len(resolved), 3)
        self.assertEqual(
            set(resolved["source_record_id"]),
            {"ROW00001", "ROW00002", "ROW00003"},
        )
        self.assertEqual(
            resolved["supplier_baseline_id"].nunique(),
            2,
        )

    def test_unresolved_source_id_fails(self) -> None:
        aggregate = self.selected()
        aggregate.loc[
            aggregate["supplier_baseline_id"].eq("SB00001"),
            "source_record_ids_merged",
        ] = "ROW00001; ROW_MISSING"

        with self.assertRaisesRegex(
            ValueError,
            "do not resolve",
        ):
            expand_source_rows(
                aggregate,
                self.row_level,
            )

    def test_duplicate_source_id_across_suppliers_fails(
        self,
    ) -> None:
        aggregate = self.selected()
        aggregate.loc[
            aggregate["supplier_baseline_id"].eq("SB00002"),
            "source_record_ids_merged",
        ] = "ROW00002"

        with self.assertRaisesRegex(
            ValueError,
            "multiple aggregate suppliers",
        ):
            expand_source_rows(
                aggregate,
                self.row_level,
            )

    def test_duplicate_row_level_record_id_fails(self) -> None:
        duplicate = pd.concat(
            [
                self.row_level,
                self.row_level.loc[
                    self.row_level["record_id"].eq("ROW00001")
                ],
            ],
            ignore_index=True,
        )

        with self.assertRaisesRegex(
            ValueError,
            "resolve multiple times",
        ):
            expand_source_rows(
                self.selected(),
                duplicate,
            )

    def test_identifier_supported_reuse(self) -> None:
        category, candidates, basis = (
            decide_reconciliation_category(
                identifier_or_source_candidates={"sko_ent_1"},
                reviewed_evidence_candidates=set(),
                name_only_candidates=set(),
                has_historical_positive_evidence=True,
            )
        )

        self.assertEqual(
            category,
            "reuse_existing_identifier_or_source_supported",
        )
        self.assertEqual(candidates, {"sko_ent_1"})
        self.assertEqual(
            basis,
            "accepted_identifier_or_persisted_source_evidence",
        )

    def test_reviewed_alias_supported_reuse(self) -> None:
        category, candidates, _ = (
            decide_reconciliation_category(
                identifier_or_source_candidates=set(),
                reviewed_evidence_candidates={"sko_ent_2"},
                name_only_candidates=set(),
                has_historical_positive_evidence=False,
            )
        )

        self.assertEqual(
            category,
            "reuse_existing_reviewed_evidence",
        )
        self.assertEqual(candidates, {"sko_ent_2"})

    def test_name_only_candidate_remains_review_only(self) -> None:
        category, candidates, basis = (
            decide_reconciliation_category(
                identifier_or_source_candidates=set(),
                reviewed_evidence_candidates=set(),
                name_only_candidates={"sko_ent_3"},
                has_historical_positive_evidence=True,
            )
        )

        self.assertEqual(
            category,
            "ambiguous_identity_owner_review",
        )
        self.assertEqual(candidates, {"sko_ent_3"})
        self.assertEqual(
            basis,
            "name_country_candidate_only",
        )

    def test_conflicting_strong_candidates_are_ambiguous(
        self,
    ) -> None:
        category, candidates, _ = (
            decide_reconciliation_category(
                identifier_or_source_candidates={
                    "sko_ent_1",
                    "sko_ent_2",
                },
                reviewed_evidence_candidates=set(),
                name_only_candidates=set(),
                has_historical_positive_evidence=True,
            )
        )

        self.assertEqual(
            category,
            "ambiguous_identity_owner_review",
        )
        self.assertEqual(
            candidates,
            {"sko_ent_1", "sko_ent_2"},
        )

    def test_historical_positive_becomes_new_candidate(
        self,
    ) -> None:
        category, candidates, _ = (
            decide_reconciliation_category(
                identifier_or_source_candidates=set(),
                reviewed_evidence_candidates=set(),
                name_only_candidates=set(),
                has_historical_positive_evidence=True,
            )
        )

        self.assertEqual(
            category,
            "create_new_entity_candidate",
        )
        self.assertEqual(candidates, set())

    def test_unsupported_row_is_preserved(self) -> None:
        category, candidates, _ = (
            decide_reconciliation_category(
                identifier_or_source_candidates=set(),
                reviewed_evidence_candidates=set(),
                name_only_candidates=set(),
                has_historical_positive_evidence=False,
            )
        )

        self.assertEqual(
            category,
            "unsupported_aggregate_only",
        )
        self.assertEqual(candidates, set())

    def test_checksum_is_deterministic(self) -> None:
        first = checksum_directory(self.store)
        second = checksum_directory(self.store)

        self.assertEqual(first, second)
        self.assertEqual(
            checksum_manifest_digest(first),
            checksum_manifest_digest(second),
        )

    def test_full_row_and_spend_reconciliation(self) -> None:
        reconciliation = self.reconciliation_frame()

        validate_reconciliation(
            reconciliation,
            expected_rows=2,
            expected_spend_eur=301.00,
        )

    def test_manifest_rejects_store_change(self) -> None:
        reconciliation = self.reconciliation_frame()
        before = checksum_directory(self.store)

        (
            self.store / "supplier_entity_links.parquet"
        ).write_bytes(b"changed links")

        after = checksum_directory(self.store)

        with self.assertRaisesRegex(
            RuntimeError,
            "checksums changed",
        ):
            build_manifest_summary(
                reconciliation,
                expanded_source_rows=3,
                unique_source_record_ids=3,
                expected_rows=2,
                expected_spend_eur=301.00,
                canonical_checksums_before=before,
                canonical_checksums_after=after,
            )

    def test_manifest_is_deterministic(self) -> None:
        reconciliation = self.reconciliation_frame()
        checksums = checksum_directory(self.store)

        first = build_manifest_summary(
            reconciliation,
            expanded_source_rows=3,
            unique_source_record_ids=3,
            expected_rows=2,
            expected_spend_eur=301.00,
            canonical_checksums_before=checksums,
            canonical_checksums_after=checksums,
        )
        second = build_manifest_summary(
            reconciliation.sample(
                frac=1,
                random_state=42,
            ).reset_index(drop=True),
            expanded_source_rows=3,
            unique_source_record_ids=3,
            expected_rows=2,
            expected_spend_eur=301.00,
            canonical_checksums_before=checksums,
            canonical_checksums_after=checksums,
        )

        self.assertEqual(first, second)


class HistoricalContinuityTests(unittest.TestCase):
    """Test conservative historical evidence retrieval."""

    def historical_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "0": "IE",
                    "1": "Synthetic Cooperative",
                    "2": "IE123",
                    "3": "1 Test Street",
                    "4": "Facilities",
                    "_supplier_raw_name": (
                        "Synthetic Cooperative"
                    ),
                    "supplier_country": "IE",
                    "matched_register": "Synthetic Register",
                    "matched_entity_name": (
                        "Synthetic Cooperative CLG"
                    ),
                    "match_type": "tax_id_exact",
                    "match_score": 100,
                    "match_country": "IE",
                    "match_region": "",
                },
                {
                    "0": "IE",
                    "1": "Synthetic Cooperative",
                    "2": "IE456",
                    "3": "2 Test Street",
                    "4": "Facilities",
                    "_supplier_raw_name": (
                        "Synthetic Cooperative"
                    ),
                    "supplier_country": "IE",
                    "matched_register": "Second Register",
                    "matched_entity_name": (
                        "Synthetic Cooperative Limited"
                    ),
                    "match_type": "name_fuzzy",
                    "match_score": 99,
                    "match_country": "IE",
                    "match_region": "",
                },
                {
                    "0": "FR",
                    "1": "Different Foundation",
                    "2": "",
                    "3": "2 Test Rue",
                    "4": "Marketing",
                    "_supplier_raw_name": (
                        "Different Foundation"
                    ),
                    "supplier_country": "FR",
                    "matched_register": "",
                    "matched_entity_name": "",
                    "match_type": "",
                    "match_score": "",
                    "match_country": "",
                    "match_region": "",
                },
            ]
        )

    def resolved_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "supplier_baseline_id": "SB1",
                    "source_record_id": "ROW1",
                    "source_row_number": 2,
                    "supplier_name": "Synthetic Cooperative",
                    "original_supplier_name": (
                        "Synthetic Cooperative"
                    ),
                    "erp_supplier_name": "",
                    "country_code": "IE",
                },
                {
                    "supplier_baseline_id": "SB2",
                    "source_record_id": "ROW2",
                    "source_row_number": 3,
                    "supplier_name": "Synthetic Foundation",
                    "original_supplier_name": (
                        "Synthetic Foundation"
                    ),
                    "erp_supplier_name": "",
                    "country_code": "FR",
                },
            ]
        )

    def test_exact_name_country_retrieves_all_evidence(
        self,
    ) -> None:
        from build_aggregate_supplier_reconciliation import (
            attach_historical_rows,
        )

        attached = attach_historical_rows(
            self.resolved_frame(),
            self.historical_frame(),
            source_row_offset=2,
        )

        cooperative = attached.loc[
            attached["source_record_id"].eq("ROW1")
        ]

        self.assertEqual(len(cooperative), 2)
        self.assertEqual(
            set(cooperative["match_type"]),
            {"tax_id_exact", "name_fuzzy"},
        )
        self.assertTrue(
            cooperative[
                "_historical_candidate_found"
            ].all()
        )

    def test_unmatched_source_row_is_preserved(self) -> None:
        from build_aggregate_supplier_reconciliation import (
            attach_historical_rows,
        )

        attached = attach_historical_rows(
            self.resolved_frame(),
            self.historical_frame(),
            source_row_offset=999,
        )

        unmatched = attached.loc[
            attached["source_record_id"].eq("ROW2")
        ]

        self.assertEqual(len(unmatched), 1)
        self.assertFalse(
            bool(
                unmatched.iloc[0][
                    "_historical_candidate_found"
                ]
            )
        )
        self.assertEqual(
            unmatched.iloc[0]["match_type"],
            "",
        )

    def test_country_prevents_cross_border_attachment(
        self,
    ) -> None:
        from build_aggregate_supplier_reconciliation import (
            attach_historical_rows,
        )

        resolved = self.resolved_frame()
        resolved.loc[
            resolved["source_record_id"].eq("ROW1"),
            "country_code",
        ] = "DE"

        attached = attach_historical_rows(
            resolved,
            self.historical_frame(),
        )

        row = attached.loc[
            attached["source_record_id"].eq("ROW1")
        ].iloc[0]

        self.assertFalse(
            bool(row["_historical_candidate_found"])
        )


if __name__ == "__main__":
    unittest.main()
