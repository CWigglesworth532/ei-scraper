from __future__ import annotations

import unittest

import pandas as pd

from canonical_matcher_safe_list import (
    CanonicalSafeListResolver,
)


ENTITY_A = "sko_ent_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
ENTITY_B = "sko_ent_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
SOURCE_A = "sko_src_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"


class CanonicalMatcherSafeListTests(unittest.TestCase):
    def row(self, **overrides):
        row = {
            "entity_id": ENTITY_A,
            "source_record_id": SOURCE_A,
            "term_type": "reviewed_alias",
            "term_raw": "Example Social Trading",
            "term_normalized": "examplesocialtrading",
            "country": "DE",
            "identifier_type": "",
            "identifier_value_normalized": "",
            "verification_status": "human_verified",
            "review_status": "accepted",
            "approved_for_matching": True,
        }
        row.update(overrides)
        return row

    def test_exact_country_aware_alias_resolves(self) -> None:
        resolver = CanonicalSafeListResolver(
            pd.DataFrame([self.row()])
        )

        match = resolver.resolve_supplier(
            supplier_name="Example Social Trading",
            country="DE",
        )

        self.assertIsNotNone(match)
        self.assertEqual(match.entity_id, ENTITY_A)
        self.assertEqual(
            match.match_method,
            "canonical_safe_name_exact",
        )

    def test_same_name_in_two_entities_is_ambiguous(self) -> None:
        resolver = CanonicalSafeListResolver(
            pd.DataFrame(
                [
                    self.row(),
                    self.row(
                        entity_id=ENTITY_B,
                        source_record_id="",
                    ),
                ]
            )
        )

        match = resolver.resolve_supplier(
            supplier_name="Example Social Trading",
            country="DE",
        )

        self.assertIsNone(match)

    def test_country_prevents_cross_border_name_resolution(
        self,
    ) -> None:
        resolver = CanonicalSafeListResolver(
            pd.DataFrame([self.row()])
        )

        match = resolver.resolve_supplier(
            supplier_name="Example Social Trading",
            country="FR",
        )

        self.assertIsNone(match)

    def test_unknown_country_requires_global_uniqueness(self) -> None:
        resolver = CanonicalSafeListResolver(
            pd.DataFrame(
                [
                    self.row(),
                    self.row(
                        entity_id=ENTITY_B,
                        country="FR",
                    ),
                ]
            )
        )

        match = resolver.resolve_supplier(
            supplier_name="Example Social Trading",
            country="",
        )

        self.assertIsNone(match)

    def test_exact_identifier_has_precedence(self) -> None:
        resolver = CanonicalSafeListResolver(
            pd.DataFrame(
                [
                    self.row(),
                    self.row(
                        term_type="identifier",
                        term_raw="HRB 1001",
                        term_normalized="hrb1001",
                        identifier_type="DE_HRB",
                        identifier_value_normalized="hrb1001",
                    ),
                ]
            )
        )

        match = resolver.resolve_supplier(
            supplier_name="Unrelated Client Name",
            country="DE",
            identifier_type="DE_HRB",
            identifier_value="HRB 1001",
        )

        self.assertIsNotNone(match)
        self.assertEqual(match.entity_id, ENTITY_A)
        self.assertEqual(
            match.match_method,
            "canonical_safe_identifier",
        )

    def test_brand_phrase_resolves_inside_supplier_name(
        self,
    ) -> None:
        resolver = CanonicalSafeListResolver(
            pd.DataFrame(
                [
                    self.row(
                        term_type="brand",
                        term_raw="Auticon",
                        term_normalized="auticon",
                    )
                ]
            )
        )

        match = resolver.resolve_supplier(
            supplier_name="Auticon Deutschland GmbH",
            country="DE",
        )

        self.assertIsNotNone(match)
        self.assertEqual(match.entity_id, ENTITY_A)
        self.assertEqual(
            match.match_method,
            "canonical_safe_brand",
        )

    def test_brand_does_not_match_inside_larger_word(self) -> None:
        resolver = CanonicalSafeListResolver(
            pd.DataFrame(
                [
                    self.row(
                        term_type="brand",
                        term_raw="Afb",
                        term_normalized="afb",
                    )
                ]
            )
        )

        match = resolver.resolve_supplier(
            supplier_name="Example Afbeyond Limited",
            country="DE",
        )

        self.assertIsNone(match)

    def test_unapproved_and_conflicted_rows_are_excluded(
        self,
    ) -> None:
        resolver = CanonicalSafeListResolver(
            pd.DataFrame(
                [
                    self.row(
                        approved_for_matching=False,
                    ),
                    self.row(
                        term_raw="Conflicted Name",
                        term_normalized="conflictedname",
                        verification_status="conflicted",
                    ),
                ]
            )
        )

        self.assertIsNone(
            resolver.resolve_supplier(
                supplier_name="Example Social Trading",
                country="DE",
            )
        )
        self.assertIsNone(
            resolver.resolve_supplier(
                supplier_name="Conflicted Name",
                country="DE",
            )
        )


if __name__ == "__main__":
    unittest.main()
