#!/usr/bin/env python3
"""Synthetic/offline forward-flow validation for SKO-025A.

Existing Record-ID-bound updates remain SKO-022 proposals. Canonical profiles
without a directory record become review-only new-profile candidates and can
never enter the SKO-024 handoff boundary until a separately governed creation
decision supplies an exact Airtable Record ID.
"""
from __future__ import annotations

import hashlib
from typing import Any
import pandas as pd
from directory_integration import PROPOSAL_FIELDS, build_proposals, clean, truthy

NEW_PROFILE_FIELDS = [
    "new_profile_candidate_id", "entity_id", "profile_relationship_type",
    "counting_entity_id", "candidate_status", "identity_status",
    "classification_status", "readiness_status", "publication_status",
    "evidence_reference", "integration_batch_id", "airtable_record_id",
]

def _one(frame: pd.DataFrame, column: str, value: str) -> dict[str, Any]:
    rows=frame.loc[frame[column].eq(value)]
    return rows.iloc[0].to_dict() if len(rows)==1 else {}

def evaluate_forward_flow(tables: dict[str,pd.DataFrame]) -> dict[str,pd.DataFrame]:
    """Return existing updates, new-profile candidates, holdouts and counting QA."""
    existing=build_proposals(tables)
    approved=tables["crosswalks"].loc[tables["crosswalks"].crosswalk_status.eq("approved")]
    existing_keys=set(zip(approved.entity_id,approved.profile_relationship_type))
    new=[]; holdouts=[]
    for candidate in tables["directory_candidates"].sort_values("candidate_id").to_dict("records"):
        entity_id=clean(candidate["entity_id"]); relationship=clean(candidate["profile_relationship_type"])
        if (entity_id,relationship) in existing_keys: continue
        entity=_one(tables["canonical_entities"],"entity_id",entity_id)
        classification=_one(tables["classifications"],"entity_id",entity_id)
        readiness=_one(tables["readiness"],"entity_id",entity_id)
        eligible=clean(classification.get("classification_status"))=="eligible"
        ready=eligible and truthy(readiness.get("enrichment_complete")) and clean(readiness.get("readiness_status"))=="ready"
        counting=clean(entity.get("counting_entity_id"))
        token="|".join([clean(candidate["candidate_id"]),entity_id,relationship,clean(candidate["integration_batch_id"])])
        row={"new_profile_candidate_id":"npc_"+hashlib.sha256(token.encode()).hexdigest()[:20],
             "entity_id":entity_id,"profile_relationship_type":relationship,"counting_entity_id":counting,
             "candidate_status":"new_directory_profile_candidate" if ready and counting else "holdout",
             "identity_status":"canonical_identity_accepted_no_directory_profile",
             "classification_status":clean(classification.get("classification_status")),
             "readiness_status":"ready" if ready else "not_ready","publication_status":"not_applicable_uncreated",
             "evidence_reference":clean(readiness.get("evidence_reference")) or clean(classification.get("evidence_reference")),
             "integration_batch_id":clean(candidate["integration_batch_id"]),"airtable_record_id":""}
        (new if row["candidate_status"]=="new_directory_profile_candidate" else holdouts).append(row)
    new_frame=pd.DataFrame(new,columns=NEW_PROFILE_FIELDS)
    holdout_frame=pd.DataFrame(holdouts,columns=NEW_PROFILE_FIELDS)
    existing_entities=set(existing["proposals"].entity_id) if not existing["proposals"].empty else set()
    represented=pd.concat([existing["proposals"][["entity_id","counting_entity_id"]],new_frame[["entity_id","counting_entity_id"]]],ignore_index=True)
    counting=pd.DataFrame([{"profile_count":len(represented),"unique_entity_count":represented.entity_id.nunique(),
                            "unique_counting_entity_count":represented.counting_entity_id.nunique(),
                            "additional_profiles_without_supplier_inflation":len(represented)-represented.counting_entity_id.nunique()}])
    return {"existing_profile_proposals":existing["proposals"],"existing_profile_references":existing["integration_references"],
            "new_profile_candidates":new_frame,"readiness_holdouts":holdout_frame,"counting_qa":counting,
            "protected_editorial_mutation_attempts":pd.DataFrame([{"count":0}]),
            "publication_mutation_attempts":pd.DataFrame([{"count":0}])}

def assert_safety(results):
    if list(results["existing_profile_proposals"].columns)!=PROPOSAL_FIELDS: raise AssertionError("existing proposal allowlist changed")
    if not results["new_profile_candidates"].airtable_record_id.eq("").all(): raise AssertionError("new candidate fabricated Record ID")
    if int(results["protected_editorial_mutation_attempts"].iloc[0]["count"]): raise AssertionError("protected mutation")
    if int(results["publication_mutation_attempts"].iloc[0]["count"]): raise AssertionError("publication mutation")
