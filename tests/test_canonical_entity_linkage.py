from __future__ import annotations

import unittest

import pandas as pd

from canonical_entity_linkage import (
    build_alias_lookup,
    build_canonical_name_lookup,
    build_identifier_lookup,
    build_source_record_lookup,
    resolve_accepted_match,
)


ENTITY_A = "sko_ent_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
ENTITY_B = "sko_ent_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
ENTITY_NEW = "sko_ent_cccccccc-cccc-4ccc-8ccc-cccccccccccc"
SOURCE_A = "sko_src_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"


class CanonicalEntityLinkageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.entities = pd.DataFrame(
            [
                {
                    "entity_id": ENTITY_A,
                    "canonical_name_norm": "example inclusion enterprise",
                    "country": "DE",
                    "record_status": "active",
                },
                {
                    "entity_id": ENTITY_B,
                    "canonical_name_norm": "example inclusion enterprise",
                    "country": "DE",
                    "record_status": "active",
                },
            ]
        )

        self.source_records = pd.DataFrame(
            [
                {
                    "source_record_id": SOURCE_A,
                    "entity_id": ENTITY_A,
                }
            ]
        )

        self.identifiers = pd.DataFrame(
            [
                {
                    "entity_id": ENTITY_A,
                    "country": "DE",
                    "identifier_type": "DE_HRB",
                    "identifier_value_normalized": "hrb1001",
                    "verification_status": "human_verified",
                }
            ]
        )

        self.aliases = pd.DataFrame(
            [
                {
                    "entity_id": ENTITY_A,
                    "alias_name_norm": "example social trading",
                    "country": "DE",
                    "verification_status": "human_verified",
                    "review_status": "accepted",
                }
            ]
        )

        self.source_lookup = build_source_record_lookup(
            self.source_records
        )
        self.identifier_lookup = build_identifier_lookup(
            self.identifiers
        )
        self.alias_lookup = build_alias_lookup(self.aliases)
        self.name_lookup = build_canonical_name_lookup(
            self.entities
        )

    def supplier(self, **overrides):
        row = {
            "supplier_record_key": "CLIENT-A|SUPPLIER-1",
            "supplier_name_original": "New Supplier Name",
            "supplier_name_norm": "new supplier name",
            "supplier_country": "DE",
            "supplier_identifier_type": "",
            "supplier_identifier_value": "",
            "acceptance_status": "accepted",
            "matched_source_record_id": "",
            "relationship_type": "legal_entity",
            "related_entity_id": "",
        }
        row.update(overrides)
        return row

    def test_same_identifier_reuses_existing_entity(self) -> None:
        decision = resolve_accepted_match(
            self.supplier(
                supplier_identifier_type="DE_HRB",
                supplier_identifier_value="HRB 1001",
            ),
            source_record_lookup=self.source_lookup,
            identifier_lookup=self.identifier_lookup,
            alias_lookup=self.alias_lookup,
            canonical_name_lookup=self.name_lookup,
        )

        self.assertEqual(decision.resolution_status, "reused")
        self.assertEqual(decision.resolution_method, "accepted_identifier")
        self.assertEqual(decision.matched_entity_id, ENTITY_A)
        self.assertFalse(decision.allocate_new_entity)

    def test_persisted_source_record_reuses_existing_entity(self) -> None:
        decision = resolve_accepted_match(
            self.supplier(
                matched_source_record_id=SOURCE_A,
            ),
            source_record_lookup=self.source_lookup,
            identifier_lookup=self.identifier_lookup,
            alias_lookup=self.alias_lookup,
            canonical_name_lookup=self.name_lookup,
        )

        self.assertEqual(decision.resolution_status, "reused")
        self.assertEqual(
            decision.resolution_method,
            "persisted_source_record",
        )
        self.assertEqual(decision.matched_entity_id, ENTITY_A)

    def test_reviewed_alias_reuses_existing_entity(self) -> None:
        decision = resolve_accepted_match(
            self.supplier(
                supplier_name_original="Example Social Trading",
                supplier_name_norm="Example Social Trading",
            ),
            source_record_lookup=self.source_lookup,
            identifier_lookup=self.identifier_lookup,
            alias_lookup=self.alias_lookup,
            canonical_name_lookup=self.name_lookup,
        )

        self.assertEqual(decision.resolution_status, "reused")
        self.assertEqual(decision.resolution_method, "reviewed_alias")
        self.assertEqual(decision.matched_entity_id, ENTITY_A)

    def test_equal_names_without_identifier_enter_review(self) -> None:
        decision = resolve_accepted_match(
            self.supplier(
                supplier_name_original="Example Inclusion Enterprise",
                supplier_name_norm="Example Inclusion Enterprise",
            ),
            source_record_lookup=self.source_lookup,
            identifier_lookup=self.identifier_lookup,
            alias_lookup=self.alias_lookup,
            canonical_name_lookup=self.name_lookup,
        )

        self.assertEqual(decision.resolution_status, "review")
        self.assertEqual(
            decision.resolution_method,
            "name_candidate_only",
        )
        self.assertTrue(decision.review_required)
        self.assertFalse(decision.allocate_new_entity)
        self.assertIn(ENTITY_A, decision.candidate_entity_ids)
        self.assertIn(ENTITY_B, decision.candidate_entity_ids)

    def test_new_identifier_does_not_merge_equal_names(self) -> None:
        decision = resolve_accepted_match(
            self.supplier(
                supplier_name_original="Example Inclusion Enterprise",
                supplier_name_norm="Example Inclusion Enterprise",
                supplier_identifier_type="DE_HRB",
                supplier_identifier_value="HRB 9999",
            ),
            source_record_lookup=self.source_lookup,
            identifier_lookup=self.identifier_lookup,
            alias_lookup=self.alias_lookup,
            canonical_name_lookup=self.name_lookup,
            id_factory=lambda: ENTITY_NEW,
        )

        self.assertEqual(decision.resolution_status, "new")
        self.assertEqual(decision.matched_entity_id, ENTITY_NEW)
        self.assertTrue(decision.allocate_new_entity)
        self.assertNotEqual(decision.matched_entity_id, ENTITY_A)
        self.assertNotEqual(decision.matched_entity_id, ENTITY_B)

    def test_conflicting_identity_signals_enter_review(self) -> None:
        conflicting_aliases = pd.DataFrame(
            [
                {
                    "entity_id": ENTITY_B,
                    "alias_name_norm": "example social trading",
                    "country": "DE",
                    "verification_status": "human_verified",
                    "review_status": "accepted",
                }
            ]
        )

        alias_lookup = build_alias_lookup(conflicting_aliases)

        decision = resolve_accepted_match(
            self.supplier(
                supplier_name_original="Example Social Trading",
                supplier_name_norm="Example Social Trading",
                supplier_identifier_type="DE_HRB",
                supplier_identifier_value="HRB 1001",
            ),
            source_record_lookup=self.source_lookup,
            identifier_lookup=self.identifier_lookup,
            alias_lookup=alias_lookup,
            canonical_name_lookup=self.name_lookup,
        )

        self.assertEqual(decision.resolution_status, "review")
        self.assertEqual(
            decision.resolution_method,
            "conflicting_identity_signals",
        )
        self.assertTrue(decision.review_required)
        self.assertFalse(decision.allocate_new_entity)

    def test_genuinely_new_accepted_supplier_gets_new_id(self) -> None:
        decision = resolve_accepted_match(
            self.supplier(
                supplier_name_original="Completely New Cooperative",
                supplier_name_norm="Completely New Cooperative",
            ),
            source_record_lookup=self.source_lookup,
            identifier_lookup=self.identifier_lookup,
            alias_lookup=self.alias_lookup,
            canonical_name_lookup=self.name_lookup,
            id_factory=lambda: ENTITY_NEW,
        )

        self.assertEqual(decision.resolution_status, "new")
        self.assertEqual(decision.matched_entity_id, ENTITY_NEW)
        self.assertTrue(decision.allocate_new_entity)
        self.assertFalse(decision.review_required)

    def test_group_relationship_does_not_reuse_related_entity(self) -> None:
        decision = resolve_accepted_match(
            self.supplier(
                supplier_name_original="Example Subsidiary GmbH",
                supplier_name_norm="Example Subsidiary GmbH",
                relationship_type="subsidiary",
                related_entity_id=ENTITY_A,
                supplier_identifier_type="DE_HRB",
                supplier_identifier_value="HRB 3003",
            ),
            source_record_lookup=self.source_lookup,
            identifier_lookup=self.identifier_lookup,
            alias_lookup=self.alias_lookup,
            canonical_name_lookup=self.name_lookup,
            id_factory=lambda: ENTITY_NEW,
        )

        self.assertEqual(decision.resolution_status, "new")
        self.assertEqual(decision.matched_entity_id, ENTITY_NEW)
        self.assertNotEqual(
            decision.matched_entity_id,
            decision.related_entity_id,
        )
        self.assertEqual(decision.relationship_type, "subsidiary")


if __name__ == "__main__":
    unittest.main()
