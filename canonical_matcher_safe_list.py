#!/usr/bin/env python3
"""Conservative canonical safe-list resolution for supplier matching.

The resolver:

- consumes the derived trusted_match_terms.csv export;
- resolves accepted identifiers exactly;
- resolves approved names and aliases exactly;
- resolves approved brands only as bounded phrases;
- applies country scoping where supplier country is known;
- requires one unique entity;
- never performs fuzzy identity resolution;
- does not classify social-economy status independently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from canonical_entity_linkage import (
    clean_text,
    normalize_identifier,
    normalize_name,
)


NAME_TERM_TYPES = {
    "legal_name",
    "reviewed_alias",
    "client_variant",
    "brand",
}

DISALLOWED_STATUSES = {
    "rejected",
    "conflicted",
    "superseded",
}


@dataclass(frozen=True)
class SafeListMatch:
    """One uniquely resolved canonical safe-list match."""

    entity_id: str
    source_record_id: str
    term_type: str
    term_raw: str
    country: str
    match_method: str


def _is_true(value: Any) -> bool:
    return clean_text(value).lower() in {
        "true",
        "yes",
        "y",
        "1",
    }


def _country_key(country: Any) -> str:
    return clean_text(country).upper()


def _name_key(country: Any, normalized_name: Any) -> str:
    return "|".join(
        [
            _country_key(country),
            normalize_name(normalized_name),
        ]
    )


def _identifier_key(
    country: Any,
    identifier_type: Any,
    identifier_value: Any,
) -> str:
    country_norm = _country_key(country)
    identifier_type_norm = clean_text(
        identifier_type
    ).upper()
    identifier_value_norm = normalize_identifier(
        identifier_value
    )

    if (
        not country_norm
        or not identifier_type_norm
        or not identifier_value_norm
    ):
        return ""

    return "|".join(
        [
            country_norm,
            identifier_type_norm,
            identifier_value_norm,
        ]
    )


def _bounded_phrase_regex(term: str) -> re.Pattern[str]:
    """Compile a phrase boundary regex for an approved brand."""
    escaped = re.escape(clean_text(term).lower())
    return re.compile(
        rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"
    )


class CanonicalSafeListResolver:
    """Resolve suppliers against approved canonical safe-list terms."""

    def __init__(self, trusted_terms: pd.DataFrame):
        required = {
            "entity_id",
            "source_record_id",
            "term_type",
            "term_raw",
            "term_normalized",
            "country",
            "identifier_type",
            "identifier_value_normalized",
            "verification_status",
            "review_status",
            "approved_for_matching",
        }
        missing = required - set(trusted_terms.columns)
        if missing:
            raise ValueError(
                "Canonical safe list is missing columns: "
                + ", ".join(sorted(missing))
            )

        accepted = trusted_terms.loc[
            trusted_terms["entity_id"]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            & trusted_terms["approved_for_matching"]
            .map(_is_true)
        ].copy()

        accepted = accepted.loc[
            ~accepted["verification_status"]
            .fillna("")
            .astype(str)
            .str.lower()
            .isin(DISALLOWED_STATUSES)
            & ~accepted["review_status"]
            .fillna("")
            .astype(str)
            .str.lower()
            .isin(DISALLOWED_STATUSES)
        ].copy()

        self.name_lookup: dict[
            str,
            list[dict[str, str]],
        ] = {}
        self.global_name_lookup: dict[
            str,
            list[dict[str, str]],
        ] = {}
        self.identifier_lookup: dict[
            str,
            list[dict[str, str]],
        ] = {}
        self.identifier_value_lookup: dict[
            str,
            list[dict[str, str]],
        ] = {}
        self.brand_rows: list[
            tuple[re.Pattern[str], dict[str, str]]
        ] = []

        for row in accepted.to_dict("records"):
            payload = {
                "entity_id": clean_text(
                    row.get("entity_id")
                ),
                "source_record_id": clean_text(
                    row.get("source_record_id")
                ),
                "term_type": clean_text(
                    row.get("term_type")
                ).lower(),
                "term_raw": clean_text(
                    row.get("term_raw")
                ),
                "country": _country_key(
                    row.get("country")
                ),
            }

            term_type = payload["term_type"]

            if term_type == "identifier":
                key = _identifier_key(
                    row.get("country"),
                    row.get("identifier_type"),
                    row.get(
                        "identifier_value_normalized"
                    ),
                )
                if key:
                    self.identifier_lookup.setdefault(
                        key,
                        [],
                    ).append(payload)

                    value_key = "|".join(
                        [
                            payload["country"],
                            normalize_identifier(
                                row.get(
                                    "identifier_value_normalized"
                                )
                            ),
                        ]
                    )
                    self.identifier_value_lookup.setdefault(
                        value_key,
                        [],
                    ).append(payload)
                continue

            if term_type not in NAME_TERM_TYPES:
                continue

            term_norm = normalize_name(
                row.get("term_normalized")
                or row.get("term_raw")
            )
            if not term_norm:
                continue

            country_key = _name_key(
                row.get("country"),
                term_norm,
            )
            self.name_lookup.setdefault(
                country_key,
                [],
            ).append(payload)

            self.global_name_lookup.setdefault(
                term_norm,
                [],
            ).append(payload)

            if (
                term_type == "brand"
                and len(term_norm) >= 4
                and payload["term_raw"]
            ):
                self.brand_rows.append(
                    (
                        _bounded_phrase_regex(
                            payload["term_raw"]
                        ),
                        payload,
                    )
                )

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
    ) -> "CanonicalSafeListResolver":
        """Load a canonical safe-list CSV."""
        frame = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            low_memory=False,
        )
        return cls(frame)

    @staticmethod
    def _unique_match(
        candidates: list[dict[str, str]],
        method: str,
    ) -> SafeListMatch | None:
        entity_ids = {
            candidate["entity_id"]
            for candidate in candidates
            if candidate["entity_id"]
        }

        if len(entity_ids) != 1:
            return None

        entity_id = next(iter(entity_ids))
        selected = next(
            candidate
            for candidate in candidates
            if candidate["entity_id"] == entity_id
        )

        return SafeListMatch(
            entity_id=entity_id,
            source_record_id=selected[
                "source_record_id"
            ],
            term_type=selected["term_type"],
            term_raw=selected["term_raw"],
            country=selected["country"],
            match_method=method,
        )

    def resolve_identifier(
        self,
        *,
        country: Any,
        identifier_type: Any,
        identifier_value: Any,
    ) -> SafeListMatch | None:
        """Resolve one exact country/type/value identifier."""
        key = _identifier_key(
            country,
            identifier_type,
            identifier_value,
        )
        if not key:
            return None

        return self._unique_match(
            self.identifier_lookup.get(key, []),
            "canonical_safe_identifier",
        )

    def resolve_identifier_value(
        self,
        *,
        country: Any,
        identifier_value: Any,
    ) -> SafeListMatch | None:
        """Resolve a value where the source lacks identifier type.

        Resolution succeeds only when country and normalized value identify
        one unique canonical entity across all accepted identifier types.
        """
        country_norm = _country_key(country)
        value_norm = normalize_identifier(identifier_value)

        if not country_norm or not value_norm:
            return None

        return self._unique_match(
            self.identifier_value_lookup.get(
                f"{country_norm}|{value_norm}",
                [],
            ),
            "canonical_safe_identifier",
        )

    def resolve_exact_name(
        self,
        *,
        supplier_name: Any,
        country: Any,
    ) -> SafeListMatch | None:
        """Resolve one exact normalized approved name."""
        name_norm = normalize_name(supplier_name)
        if not name_norm:
            return None

        country_norm = _country_key(country)

        if country_norm:
            candidates = self.name_lookup.get(
                f"{country_norm}|{name_norm}",
                [],
            )
        else:
            candidates = self.global_name_lookup.get(
                name_norm,
                [],
            )

        return self._unique_match(
            candidates,
            "canonical_safe_name_exact",
        )

    def resolve_brand(
        self,
        *,
        supplier_name: Any,
        country: Any,
    ) -> SafeListMatch | None:
        """Resolve one approved brand as a bounded phrase."""
        raw_name = clean_text(supplier_name).lower()
        if not raw_name:
            return None

        country_norm = _country_key(country)
        candidates = []

        for pattern, payload in self.brand_rows:
            if (
                country_norm
                and payload["country"]
                and payload["country"] != country_norm
            ):
                continue

            if pattern.search(raw_name):
                candidates.append(payload)

        return self._unique_match(
            candidates,
            "canonical_safe_brand",
        )

    def resolve_supplier(
        self,
        *,
        supplier_name: Any,
        country: Any,
        identifier_type: Any = "",
        identifier_value: Any = "",
    ) -> SafeListMatch | None:
        """Resolve using identifier, exact name, then approved brand."""
        if clean_text(identifier_type):
            identifier_match = self.resolve_identifier(
                country=country,
                identifier_type=identifier_type,
                identifier_value=identifier_value,
            )
        else:
            identifier_match = self.resolve_identifier_value(
                country=country,
                identifier_value=identifier_value,
            )

        if identifier_match is not None:
            return identifier_match

        name_match = self.resolve_exact_name(
            supplier_name=supplier_name,
            country=country,
        )
        if name_match is not None:
            return name_match

        return self.resolve_brand(
            supplier_name=supplier_name,
            country=country,
        )

    def approved_brand_names(self) -> list[str]:
        """Return approved canonical brand terms for legacy veto bypass."""
        names = {
            payload["term_raw"]
            for _, payload in self.brand_rows
            if payload["term_raw"]
        }
        return sorted(names)
