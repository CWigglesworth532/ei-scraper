#!/usr/bin/env python3
"""Resolve accepted supplier matches against persisted canonical entities.

E1.4 responsibilities in this module:

- reuse persisted source-record and identifier links;
- reuse reviewed aliases;
- never resolve identity from normalized-name equality alone;
- send ambiguous or conflicting candidates to review;
- permit a new opaque entity ID only when no unresolved identity candidate
  remains;
- keep legal-entity identity separate from group and establishment
  relationships.

This module initially contains the pure identity-decision layer. Persistence,
safe-list export and matcher integration are added in subsequent increments.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Callable

import pandas as pd


ENTITY_ID_PREFIX = "sko_ent_"

ACCEPTED_STATUSES = {
    "accepted",
    "verified",
    "reviewed_confirmed",
    "directory_approved",
    "reporting_approved",
}

ACCEPTED_ALIAS_STATUSES = {
    "accepted",
    "reviewed",
    "human_verified",
    "source_verified",
}

IDENTITY_RELATIONSHIP_TYPES = {
    "",
    "legal_entity",
}

NON_IDENTITY_RELATIONSHIP_TYPES = {
    "parent",
    "subsidiary",
    "group_member",
    "brand_of",
    "trading_name_of",
    "establishment_of",
    "branch_of",
    "successor",
    "predecessor",
}


@dataclass(frozen=True)
class LinkageDecision:
    """One inspectable canonical-linkage decision."""

    supplier_record_key: str
    resolution_status: str
    resolution_method: str
    matched_entity_id: str
    matched_source_record_id: str
    allocate_new_entity: bool
    review_required: bool
    candidate_entity_ids: str
    resolution_reason: str
    relationship_type: str
    related_entity_id: str

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable representation."""
        return asdict(self)


def clean_text(value: Any) -> str:
    """Return a stripped string and normalise missing values to blank."""
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def normalize_name(value: Any) -> str:
    """Create a conservative normalized-name lookup value."""
    text = clean_text(value).lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def normalize_identifier(value: Any) -> str:
    """Normalize an identifier for country/type-scoped lookup."""
    text = clean_text(value).lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def allocate_entity_id() -> str:
    """Allocate an opaque canonical entity ID."""
    return f"{ENTITY_ID_PREFIX}{uuid.uuid4()}"


def identifier_key(
    country: Any,
    identifier_type: Any,
    identifier_value: Any,
) -> str:
    """Create a country/type/value identifier lookup key."""
    country_norm = clean_text(country).upper()
    type_norm = clean_text(identifier_type).upper()
    value_norm = normalize_identifier(identifier_value)

    if not country_norm or not type_norm or not value_norm:
        return ""

    return "|".join(
        [
            country_norm,
            type_norm,
            value_norm,
        ]
    )


def build_source_record_lookup(
    source_records: pd.DataFrame,
) -> dict[str, set[str]]:
    """Map persisted source-record IDs to canonical entity IDs."""
    lookup: dict[str, set[str]] = {}

    if source_records.empty:
        return lookup

    required = {"source_record_id", "entity_id"}
    if not required.issubset(source_records.columns):
        return lookup

    for row in source_records.to_dict("records"):
        source_record_id = clean_text(row.get("source_record_id"))
        entity_id = clean_text(row.get("entity_id"))

        if source_record_id and entity_id:
            lookup.setdefault(source_record_id, set()).add(entity_id)

    return lookup


def build_identifier_lookup(
    identifiers: pd.DataFrame,
) -> dict[str, set[str]]:
    """Map accepted country-scoped identifiers to entity IDs."""
    lookup: dict[str, set[str]] = {}

    if identifiers.empty:
        return lookup

    required = {
        "entity_id",
        "country",
        "identifier_type",
        "identifier_value_normalized",
    }
    if not required.issubset(identifiers.columns):
        return lookup

    for row in identifiers.to_dict("records"):
        verification_status = clean_text(
            row.get("verification_status")
        ).lower()

        if verification_status in {
            "rejected",
            "conflicted",
            "superseded",
        }:
            continue

        key = identifier_key(
            row.get("country"),
            row.get("identifier_type"),
            row.get("identifier_value_normalized"),
        )
        entity_id = clean_text(row.get("entity_id"))

        if key and entity_id:
            lookup.setdefault(key, set()).add(entity_id)

    return lookup


def build_alias_lookup(
    aliases: pd.DataFrame,
) -> dict[str, set[str]]:
    """Map reviewed country-scoped aliases to entity IDs."""
    lookup: dict[str, set[str]] = {}

    if aliases.empty:
        return lookup

    required = {
        "entity_id",
        "alias_name_norm",
    }
    if not required.issubset(aliases.columns):
        return lookup

    for row in aliases.to_dict("records"):
        verification_status = clean_text(
            row.get("verification_status")
        ).lower()
        review_status = clean_text(
            row.get("review_status")
        ).lower()

        accepted = (
            verification_status in ACCEPTED_ALIAS_STATUSES
            or review_status in ACCEPTED_ALIAS_STATUSES
        )
        if not accepted:
            continue

        country = clean_text(row.get("country")).upper()
        alias_norm = normalize_name(row.get("alias_name_norm"))
        entity_id = clean_text(row.get("entity_id"))

        if not country or not alias_norm or not entity_id:
            continue

        key = f"{country}|{alias_norm}"
        lookup.setdefault(key, set()).add(entity_id)

    return lookup


def build_canonical_name_lookup(
    canonical_entities: pd.DataFrame,
) -> dict[str, set[str]]:
    """Build a candidate-only canonical-name lookup.

    This lookup is deliberately not sufficient to approve an identity link.
    """
    lookup: dict[str, set[str]] = {}

    if canonical_entities.empty:
        return lookup

    required = {
        "entity_id",
        "country",
        "canonical_name_norm",
    }
    if not required.issubset(canonical_entities.columns):
        return lookup

    for row in canonical_entities.to_dict("records"):
        if clean_text(row.get("record_status")).lower() in {
            "merged",
            "deprecated",
            "quarantined",
        }:
            continue

        country = clean_text(row.get("country")).upper()
        name_norm = normalize_name(row.get("canonical_name_norm"))
        entity_id = clean_text(row.get("entity_id"))

        if country and name_norm and entity_id:
            key = f"{country}|{name_norm}"
            lookup.setdefault(key, set()).add(entity_id)

    return lookup


def _candidate_string(candidate_ids: set[str]) -> str:
    """Return stable pipe-separated candidate IDs."""
    return "|".join(sorted(candidate_ids))


def resolve_accepted_match(
    supplier: dict[str, Any] | pd.Series,
    *,
    source_record_lookup: dict[str, set[str]],
    identifier_lookup: dict[str, set[str]],
    alias_lookup: dict[str, set[str]],
    canonical_name_lookup: dict[str, set[str]],
    id_factory: Callable[[], str] = allocate_entity_id,
) -> LinkageDecision:
    """Resolve one accepted supplier against persisted canonical identity."""
    row = dict(supplier)

    supplier_record_key = clean_text(
        row.get("supplier_record_key")
    )
    acceptance_status = clean_text(
        row.get("acceptance_status")
    ).lower()
    country = clean_text(
        row.get("supplier_country")
    ).upper()
    supplier_name = clean_text(
        row.get("supplier_name_original")
    )
    supplier_name_norm = normalize_name(
        row.get("supplier_name_norm") or supplier_name
    )

    matched_source_record_id = clean_text(
        row.get("matched_source_record_id")
    )
    identifier_type = clean_text(
        row.get("supplier_identifier_type")
    )
    identifier_value = clean_text(
        row.get("supplier_identifier_value")
    )

    relationship_type = clean_text(
        row.get("relationship_type")
    ).lower()
    related_entity_id = clean_text(
        row.get("related_entity_id")
    )

    if acceptance_status not in ACCEPTED_STATUSES:
        return LinkageDecision(
            supplier_record_key=supplier_record_key,
            resolution_status="not_eligible",
            resolution_method="acceptance_status",
            matched_entity_id="",
            matched_source_record_id=matched_source_record_id,
            allocate_new_entity=False,
            review_required=False,
            candidate_entity_ids="",
            resolution_reason=(
                "Supplier match is not in an accepted status."
            ),
            relationship_type=relationship_type,
            related_entity_id=related_entity_id,
        )

    if not supplier_record_key or not country or not supplier_name:
        return LinkageDecision(
            supplier_record_key=supplier_record_key,
            resolution_status="review",
            resolution_method="required_field_validation",
            matched_entity_id="",
            matched_source_record_id=matched_source_record_id,
            allocate_new_entity=False,
            review_required=True,
            candidate_entity_ids="",
            resolution_reason=(
                "Accepted supplier is missing its record key, country "
                "or original supplier name."
            ),
            relationship_type=relationship_type,
            related_entity_id=related_entity_id,
        )

    if (
        relationship_type
        and relationship_type
        not in IDENTITY_RELATIONSHIP_TYPES
        and relationship_type
        not in NON_IDENTITY_RELATIONSHIP_TYPES
    ):
        return LinkageDecision(
            supplier_record_key=supplier_record_key,
            resolution_status="review",
            resolution_method="unknown_relationship_type",
            matched_entity_id="",
            matched_source_record_id=matched_source_record_id,
            allocate_new_entity=False,
            review_required=True,
            candidate_entity_ids="",
            resolution_reason=(
                "Relationship type is not recognised by the "
                "canonical identity policy."
            ),
            relationship_type=relationship_type,
            related_entity_id=related_entity_id,
        )

    source_candidates = set()
    if matched_source_record_id:
        source_candidates = source_record_lookup.get(
            matched_source_record_id,
            set(),
        )

    identifier_candidates = set()
    supplier_identifier_key = identifier_key(
        country,
        identifier_type,
        identifier_value,
    )
    if supplier_identifier_key:
        identifier_candidates = identifier_lookup.get(
            supplier_identifier_key,
            set(),
        )

    alias_candidates = set()
    if supplier_name_norm:
        alias_candidates = alias_lookup.get(
            f"{country}|{supplier_name_norm}",
            set(),
        )

    evidence_sets = [
        candidates
        for candidates in [
            source_candidates,
            identifier_candidates,
            alias_candidates,
        ]
        if candidates
    ]
    combined_candidates = set().union(*evidence_sets) if evidence_sets else set()

    if any(len(candidates) > 1 for candidates in evidence_sets):
        return LinkageDecision(
            supplier_record_key=supplier_record_key,
            resolution_status="review",
            resolution_method="ambiguous_identity_candidate",
            matched_entity_id="",
            matched_source_record_id=matched_source_record_id,
            allocate_new_entity=False,
            review_required=True,
            candidate_entity_ids=_candidate_string(
                combined_candidates
            ),
            resolution_reason=(
                "At least one accepted identity signal resolves to "
                "multiple canonical entities."
            ),
            relationship_type=relationship_type,
            related_entity_id=related_entity_id,
        )

    if len(combined_candidates) > 1:
        return LinkageDecision(
            supplier_record_key=supplier_record_key,
            resolution_status="review",
            resolution_method="conflicting_identity_signals",
            matched_entity_id="",
            matched_source_record_id=matched_source_record_id,
            allocate_new_entity=False,
            review_required=True,
            candidate_entity_ids=_candidate_string(
                combined_candidates
            ),
            resolution_reason=(
                "Accepted source-record, identifier or alias signals "
                "point to different canonical entities."
            ),
            relationship_type=relationship_type,
            related_entity_id=related_entity_id,
        )

    if len(combined_candidates) == 1:
        entity_id = next(iter(combined_candidates))

        if source_candidates:
            method = "persisted_source_record"
        elif identifier_candidates:
            method = "accepted_identifier"
        else:
            method = "reviewed_alias"

        return LinkageDecision(
            supplier_record_key=supplier_record_key,
            resolution_status="reused",
            resolution_method=method,
            matched_entity_id=entity_id,
            matched_source_record_id=matched_source_record_id,
            allocate_new_entity=False,
            review_required=False,
            candidate_entity_ids=entity_id,
            resolution_reason=(
                "Reused one uniquely resolved persisted canonical entity."
            ),
            relationship_type=relationship_type,
            related_entity_id=related_entity_id,
        )

    name_candidates = canonical_name_lookup.get(
        f"{country}|{supplier_name_norm}",
        set(),
    )

    if name_candidates and not supplier_identifier_key:
        return LinkageDecision(
            supplier_record_key=supplier_record_key,
            resolution_status="review",
            resolution_method="name_candidate_only",
            matched_entity_id="",
            matched_source_record_id=matched_source_record_id,
            allocate_new_entity=False,
            review_required=True,
            candidate_entity_ids=_candidate_string(
                name_candidates
            ),
            resolution_reason=(
                "Normalized-name equality created identity candidates "
                "but is not sufficient to reuse or merge an entity."
            ),
            relationship_type=relationship_type,
            related_entity_id=related_entity_id,
        )

    new_entity_id = id_factory()

    return LinkageDecision(
        supplier_record_key=supplier_record_key,
        resolution_status="new",
        resolution_method=(
            "new_accepted_identifier_entity"
            if supplier_identifier_key
            else "new_accepted_singleton_entity"
        ),
        matched_entity_id=new_entity_id,
        matched_source_record_id=matched_source_record_id,
        allocate_new_entity=True,
        review_required=False,
        candidate_entity_ids="",
        resolution_reason=(
            "No persisted accepted identity link or unresolved "
            "identity candidate was found."
        ),
        relationship_type=relationship_type,
        related_entity_id=related_entity_id,
    )
