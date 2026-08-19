"""SKO-034 Germany-first offline source laboratory.

This module deliberately has no production integration side effects. It validates
disabled candidate configuration, normalizes laboratory identifiers/legal forms,
and parses small offline BAG IF and ZER fixtures while preserving rejected rows.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from bs4 import BeautifulSoup


SEMANTIC_LAYERS = {"Core", "Associated", "Review-only", "Reject"}
SOURCE_FORMATS = {"CSV", "XLSX", "HTML", "PDF", "XML", "ZIP", "local-build", "JSON"}
IDENTIFIER_TYPES = {
    "DE_HRB", "DE_HRA", "DE_GNR", "DE_VR", "DE_PR", "DE_GSR",
    "DE_EUID", "DE_VAT", "DE_ZER_INTERNAL",
}
COURT_SCOPED_TYPES = {"DE_HRB", "DE_HRA", "DE_GNR", "DE_VR", "DE_PR", "DE_GSR"}


class ContractError(ValueError):
    """Raised when a candidate source violates the laboratory contract."""


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\xa0", " ")).strip()


def validate_candidate_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    candidates = payload.get("candidate_sources")
    if not isinstance(candidates, list) or not candidates:
        raise ContractError("candidate_sources must be a non-empty list")

    required = {
        "source_id", "enabled", "acceptance_state", "country", "jurisdiction_level",
        "jurisdiction", "source_name", "publisher", "source_family", "access_method",
        "source_format", "landing_page", "native_record_key", "provenance",
        "access_terms", "semantic_evidence_layer", "source_evidence",
        "policy_classification", "rejected_unresolved_policy",
    }
    seen: set[str] = set()
    for index, candidate in enumerate(candidates):
        missing = sorted(required - set(candidate))
        if missing:
            raise ContractError(f"candidate {index} missing fields: {missing}")
        source_id = clean_text(candidate["source_id"])
        if not source_id or source_id in seen:
            raise ContractError(f"invalid or duplicate source_id: {source_id!r}")
        seen.add(source_id)
        if candidate["enabled"] is not False:
            raise ContractError(f"candidate {source_id} must remain disabled")
        if candidate["semantic_evidence_layer"] not in SEMANTIC_LAYERS:
            raise ContractError(f"candidate {source_id} has invalid semantic layer")
        if candidate["source_format"] not in SOURCE_FORMATS:
            raise ContractError(f"candidate {source_id} has unsupported source format")
        if candidate["policy_classification"] != "none":
            raise ContractError(f"candidate {source_id} must not assert policy classification")
        terms = candidate["access_terms"]
        for field in ("terms_url", "assessed_on", "systematic_reuse"):
            if not clean_text(terms.get(field)):
                raise ContractError(f"candidate {source_id} missing access_terms.{field}")
        provenance = candidate["provenance"]
        for field in ("preserve_raw_record", "preserve_source_url", "preserve_retrieved_at"):
            if provenance.get(field) is not True:
                raise ContractError(f"candidate {source_id} must set provenance.{field}=true")
    return payload


def normalize_identifier(
    identifier_type: str,
    raw_value: Any,
    *,
    court: Any = "",
) -> dict[str, str]:
    kind = clean_text(identifier_type).upper()
    if kind not in IDENTIFIER_TYPES:
        raise ValueError(f"unsupported German identifier type: {kind}")
    raw = clean_text(raw_value).upper()
    if not raw:
        raise ValueError("identifier value is required")

    if kind in COURT_SCOPED_TYPES:
        verified_court = clean_text(court).upper()
        if not verified_court:
            raise ValueError(f"verified register court required for {kind}")
        register_code = kind.removeprefix("DE_")
        number = re.sub(rf"^({register_code})\s*", "", raw, flags=re.IGNORECASE)
        number = re.sub(r"[^A-Z0-9]", "", number)
        court_norm = re.sub(r"[^A-Z0-9]", "", verified_court)
        if not number or not court_norm:
            raise ValueError("court and register number must normalize to non-empty values")
        normalized = f"{court_norm}|{number}"
        scope = "register_entry"
    elif kind == "DE_EUID":
        normalized = re.sub(r"[^A-Z0-9.]", "", raw)
        if not normalized.startswith("DE"):
            raise ValueError("German EUID must start with DE")
        scope = "legal_entity"
    elif kind == "DE_VAT":
        normalized = re.sub(r"[^A-Z0-9]", "", raw)
        if re.fullmatch(r"\d{9}", normalized):
            normalized = "DE" + normalized
        if not re.fullmatch(r"DE\d{9}", normalized):
            raise ValueError("German VAT identifier must be DE followed by 9 digits")
        scope = "tax_registration"
    else:
        normalized = re.sub(r"[^A-Z0-9_-]", "", raw)
        scope = "internal_source"

    return {
        "country": "DE",
        "identifier_type": kind,
        "identifier_value_raw": clean_text(raw_value),
        "identifier_value_normalized": normalized,
        "identifier_scope": scope,
        "issuing_authority": clean_text(court) if kind in COURT_SCOPED_TYPES else "",
        "reusable_for_identity_matching": kind != "DE_ZER_INTERNAL",
    }


LEGAL_FORM_PATTERNS = (
    ("GGMBH", "COMPANY_LIMITED_NONPROFIT", re.compile(r"(?<![A-Z])G\s*GMBH(?![A-Z])", re.I)),
    ("EINGETRAGENER_VEREIN", "ASSOCIATION", re.compile(r"(?<![A-Z])E\s*\.\s*V\s*\.?(?![A-Z])", re.I)),
    ("EINGETRAGENE_GENOSSENSCHAFT", "COOPERATIVE", re.compile(r"(?<![A-Z.])E\s*\.?\s*G\s*\.?(?![A-Z.])", re.I)),
    ("GMBH", "COMPANY_LIMITED", re.compile(r"(?<![A-Z])GMBH(?![A-Z])", re.I)),
    ("STIFTUNG", "FOUNDATION", re.compile(r"(?<![A-Z])STIFTUNG(?![A-Z])", re.I)),
)


def normalize_german_legal_form(name: Any) -> dict[str, str] | None:
    text = clean_text(name)
    if not text:
        return None
    for code, family, pattern in LEGAL_FORM_PATTERNS:
        match = pattern.search(text)
        if match:
            return {
                "legal_form_local": match.group(0),
                "base_legal_form_code": code,
                "base_legal_form_family": family,
                "evidence_only": True,
                "policy_classification": "none",
            }
    return None


@dataclass(frozen=True)
class ParseResult:
    accepted: list[dict[str, Any]]
    rejected: list[dict[str, Any]]
    metrics: dict[str, int]


def _deduplicate(rows: list[dict[str, Any]], key_fields: tuple[str, ...]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for row in rows:
        key = tuple(clean_text(row.get(field)).casefold() for field in key_fields)
        if key in seen:
            rejected.append({**row, "rejection_reason": "duplicate_record"})
        else:
            seen.add(key)
            accepted.append(row)
    return accepted, rejected


def parse_bag_if_html(
    html: str,
    *,
    source_url: str,
    retrieved_at: str,
) -> ParseResult:
    soup = BeautifulSoup(html, "html.parser")
    container = soup.select_one(".entry-content") or soup.body or soup
    raw_lines = [clean_text(line) for line in container.get_text("\n").splitlines()]
    lines = [line for line in raw_lines if line]
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    pending_name: list[str] = []
    pending_address = ""
    postcode_re = re.compile(r"^(?P<street>.*?)\s*(?:[-–]\s*)?(?P<postcode>\d{5})\s+(?P<city>.+)$")

    def flush_incomplete(reason: str) -> None:
        nonlocal pending_name, pending_address
        if pending_name or pending_address:
            rejected.append({
                "entity_name_raw": " ".join(pending_name),
                "address_raw": pending_address,
                "source_url": source_url,
                "retrieved_at": retrieved_at,
                "rejection_reason": reason,
            })
        pending_name = []
        pending_address = ""

    for line in lines:
        if line.startswith("#") or line.casefold() in {"unternehmen", "unternehmensübersicht"}:
            continue
        if line.startswith("http") or line.casefold().startswith("www."):
            if rows:
                rows[-1]["website"] = line
            continue
        match = postcode_re.match(line)
        if match:
            name = clean_text(" ".join(pending_name))
            street = clean_text(match.group("street") or pending_address)
            if name and not re.fullmatch(r"[-–—/]+", name):
                rows.append({
                    "country": "DE",
                    "entity_name_raw": name,
                    "entity_name_norm": name.casefold(),
                    "address_raw": street,
                    "postcode": match.group("postcode"),
                    "city": clean_text(match.group("city")),
                    "website": "",
                    "source_id": "de_bag_if_inclusion_enterprises",
                    "source_url": source_url,
                    "retrieved_at": retrieved_at,
                    "raw_record": " | ".join([name, line]),
                    "semantic_evidence_layer": "Associated",
                    "source_evidence": "BAG IF governed specialist directory listing",
                    "policy_classification": "none",
                    "legal_form": normalize_german_legal_form(name),
                })
            else:
                flush_incomplete("null_or_invalid_entity_name")
            pending_name = []
            pending_address = ""
            continue
        if any(char.isdigit() for char in line):
            pending_address = line
            continue
        if pending_address:
            flush_incomplete("incomplete_address")
        if re.fullmatch(r"[-–—/]+", line):
            pending_name = [line]
            continue
        if line.endswith(('.', ':')) and len(line.split()) > 6:
            continue
        pending_name.append(line)

    flush_incomplete("incomplete_address")
    deduped, duplicates = _deduplicate(rows, ("entity_name_norm", "postcode", "city"))
    rejected.extend(duplicates)
    metrics = {
        "input_nonempty_lines": len(lines),
        "accepted_rows": len(deduped),
        "rejected_rows": len(rejected),
        "duplicate_rows": sum(r["rejection_reason"] == "duplicate_record" for r in rejected),
        "null_or_invalid_name_rows": sum(r["rejection_reason"] == "null_or_invalid_entity_name" for r in rejected),
        "incomplete_rows": sum(r["rejection_reason"] == "incomplete_address" for r in rejected),
    }
    return ParseResult(deduped, rejected, metrics)


def parse_zer_fixture(
    payload: str | bytes | list[dict[str, Any]],
    *,
    source_url: str,
    retrieved_at: str,
) -> ParseResult:
    records = json.loads(payload) if isinstance(payload, (str, bytes)) else payload
    if not isinstance(records, list):
        raise ValueError("ZER fixture must contain a list of records")
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in records:
        native_id = clean_text(raw.get("zer_internal_id"))
        name = clean_text(raw.get("organisation"))
        if not native_id:
            rejected.append({**raw, "source_url": source_url, "retrieved_at": retrieved_at, "rejection_reason": "missing_native_identifier"})
            continue
        if not name:
            rejected.append({**raw, "source_url": source_url, "retrieved_at": retrieved_at, "rejection_reason": "null_or_invalid_entity_name"})
            continue
        if native_id in seen:
            rejected.append({**raw, "source_url": source_url, "retrieved_at": retrieved_at, "rejection_reason": "duplicate_native_identifier"})
            continue
        seen.add(native_id)
        identifier = normalize_identifier("DE_ZER_INTERNAL", native_id)
        accepted.append({
            "country": "DE",
            "entity_name_raw": name,
            "entity_name_norm": name.casefold(),
            "native_record_key": native_id,
            "source_local_identifier": identifier,
            "tax_id": "",
            "source_id": "de_bzst_zer",
            "source_url": source_url,
            "retrieved_at": retrieved_at,
            "raw_record": dict(raw),
            "semantic_evidence_layer": "Core",
            "se_recognition_type": "tax_designation",
            "source_evidence": "Official ZER tax-designation listing",
            "policy_classification": "none",
            "resolution_status": "unresolved",
        })
    reasons = Counter(row["rejection_reason"] for row in rejected)
    metrics = {
        "input_rows": len(records),
        "accepted_rows": len(accepted),
        "rejected_rows": len(rejected),
        "duplicate_rows": reasons["duplicate_native_identifier"],
        "missing_identifier_rows": reasons["missing_native_identifier"],
        "null_or_invalid_name_rows": reasons["null_or_invalid_entity_name"],
    }
    return ParseResult(accepted, rejected, metrics)


def normalize_ch_uid(raw_value: Any) -> str:
    value = re.sub(r"[^A-Z0-9]", "", clean_text(raw_value).upper())
    if re.fullmatch(r"\d{9}", value):
        value = "CHE" + value
    if not re.fullmatch(r"CHE\d{9}", value):
        raise ValueError("Swiss UID must contain CHE and 9 digits")
    return f"CHE-{value[3:6]}.{value[6:9]}.{value[9:12]}"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
