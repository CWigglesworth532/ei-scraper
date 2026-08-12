"""Synthetic behavioural tests for SKO-024."""
import copy
import inspect
import json
import unittest
from pathlib import Path
import pandas as pd
import directory_integration as engine
import directory_integration_handoff as gate

ROOT=Path(__file__).resolve().parents[1]
FIX=ROOT/"tests"/"fixtures"/"directory"
CONFIG=ROOT/"config"/"directory_integration_handoff_no_write.yaml"

class HandoffTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tables=engine.load_fixture_pack(FIX); cls.results=engine.build_proposals(cls.tables); cls.config=gate.load_config(CONFIG)

 def build(self, **kw):
  args=dict(proposals=self.results["proposals"],integration_references=self.results["integration_references"],decisions=self.results["decisions"],airtable_profiles=self.tables["airtable_profiles"],crosswalks=self.tables["crosswalks"],canonical_entities=self.tables["canonical_entities"],config=self.config)
  args.update(kw)
  return gate.build_candidate_batch(**args,batch_id="batch-synthetic",created_at="2026-08-12T12:00:00Z",created_by="operator")

 def single(self):
  p=self.results["proposals"]; r=self.results["integration_references"]; d=self.results["decisions"]
  return self.build(proposals=p[p.entity_id=="ent_reuse"],integration_references=r[r.entity_id=="ent_reuse"],decisions=d[d.candidate_id=="c_reuse"])

 def package(self,batch):
  a=gate.record_approval(batch,decision="approved",approver="owner",approved_at="2026-08-12T12:30:00Z")
  return gate.create_handoff_package(batch,a)

 def test_determinism_and_shuffle(self):
  a=self.build(); b=self.build(proposals=self.results["proposals"].sample(frac=1,random_state=1),integration_references=self.results["integration_references"].sample(frac=1,random_state=2),decisions=self.results["decisions"].sample(frac=1,random_state=3),airtable_profiles=self.tables["airtable_profiles"].sample(frac=1,random_state=4),crosswalks=self.tables["crosswalks"].sample(frac=1,random_state=5))
  self.assertEqual(a["items"],b["items"]); self.assertEqual(a["manifest"]["batch_content_hash"],b["manifest"]["batch_content_hash"])

 def test_approved_handoff_and_mutation_invalidates(self):
  b=self.single(); a=gate.record_approval(b,decision="approved",approver="owner",approved_at="2026-08-12T12:30:00Z")
  self.assertTrue(gate.create_handoff_package(b,a)["manifest"]["handoff_ready"])
  b["items"][0]["proposed_values"]["entity_id"]="changed"
  with self.assertRaises(gate.HandoffGateViolation): gate.create_handoff_package(b,a)

 def test_stale_and_missing_record(self):
  profiles=self.tables["airtable_profiles"].copy(); profiles.loc[profiles.airtable_record_id=="rec_reuse","record_fingerprint"]="changed"
  self.assertEqual(self.build(airtable_profiles=profiles)["manifest"]["stale_fingerprint_count"],1)
  cross=self.tables["crosswalks"]; cross=cross[cross.entity_id!="ent_reuse"]
  self.assertEqual(self.build(crosswalks=cross)["manifest"]["missing_record_id_count"],1)

 def test_duplicates_conflicts_and_injections(self):
  p=self.results["proposals"]; one=p[p.entity_id=="ent_reuse"]; duplicate=pd.concat([one,one],ignore_index=True)
  self.assertGreater(self.build(proposals=duplicate)["manifest"]["duplicate_count"],0)
  conflict=duplicate.copy(); conflict.loc[1,"skopia_readiness_status"]="not_ready"
  self.assertGreater(self.build(proposals=conflict)["manifest"]["conflict_count"],0)
  for field,counter in [("unexpected","allowlist_violations"),("Organisation","protected_field_mutation_attempts"),("publication_status","publication_mutation_attempts")]:
   injected=one.copy(); injected[field]="forbidden"
   self.assertGreater(self.build(proposals=injected)["manifest"][counter],0)

 def test_classification_readiness_and_counting_gates(self):
  d=self.results["decisions"].copy(); d.loc[d.candidate_id=="c_reuse","classification_status"]="ineligible"
  self.assertGreater(self.build(decisions=d)["manifest"]["classification_gate_blocks"],0)
  d=self.results["decisions"].copy(); d.loc[d.candidate_id=="c_reuse","readiness_status"]="not_ready"
  self.assertGreater(self.build(decisions=d)["manifest"]["readiness_gate_blocks"],0)
  p=self.results["proposals"].copy(); p.loc[p.entity_id=="ent_reuse","counting_entity_id"]=""
  self.assertGreater(self.build(proposals=p)["manifest"]["invalid_counting_entity_id_count"],0)

 def test_noop_retry_idempotency_partial_failure_and_rollback(self):
  b=self.single(); item=b["items"][0]; profiles=self.tables["airtable_profiles"].copy()
  for field,value in item["proposed_values"].items():
   if field not in profiles: profiles[field]=""
   profiles.loc[profiles.airtable_record_id==item["airtable_record_id"],field]=value
  p=self.results["proposals"]; self.assertEqual(self.build(proposals=p[p.entity_id=="ent_reuse"],airtable_profiles=profiles)["manifest"]["noop_count"],1)
  package=self.package(b); op=package["application_contract"]["operations"][0]; failed=[{"proposal_id":op["proposal_id"],"outcome":"failed"}]
  retry=gate.derive_retry_package(package,failed,{op["airtable_record_id"]:op["current_fingerprint"]}); self.assertEqual(len(retry["eligible_operations"]),1)
  done=gate.derive_retry_package(package,[{"proposal_id":op["proposal_id"],"outcome":"applied"}],{}); self.assertFalse(done["eligible_operations"])
  stale=gate.derive_retry_package(package,failed,{op["airtable_record_id"]:"changed"}); self.assertEqual(stale["stale_proposal_ids"],[op["proposal_id"]])
  rollback=gate.build_rollback_candidate(package,[{"proposal_id":op["proposal_id"],"outcome":"applied"}],application_reference="run-1")
  self.assertTrue(rollback["requires_validation"] and rollback["requires_owner_approval"]); self.assertIn("reviewed_before_values",rollback["operations"][0])

 def test_one_to_many_and_safety_boundary(self):
  multi=[x for x in self.build()["items"] if x["entity_id"]=="ent_multi"]
  self.assertEqual(len({x["airtable_record_id"] for x in multi}),3); self.assertEqual(len({x["counting_entity_id"] for x in multi}),1)
  self.assertNotIn("APPLIED",gate.LIFECYCLE)
  source=inspect.getsource(gate).casefold(); forbidden=["import requests","import urllib","httpx","pyairtable","os.environ","webhook","--apply"]
  self.assertFalse([x for x in forbidden if x in source])

 def test_approval_identity_metadata_and_rejection(self):
  batch=self.single(); approval=gate.record_approval(batch,decision="approved",approver="owner",approved_at="2026-08-12T12:30:00Z")
  self.assertEqual(approval["batch_id"],batch["manifest"]["batch_id"])
  wrong=dict(approval,batch_id="wrong-batch")
  with self.assertRaises(gate.HandoffGateViolation): gate.create_handoff_package(batch,wrong)
  rejected=dict(approval,decision="rejected")
  with self.assertRaises(gate.HandoffGateViolation): gate.create_handoff_package(batch,rejected)
  for approver,timestamp in [("","2026-08-12T12:30:00Z"),("owner","bad")]:
   with self.assertRaises(gate.HandoffGateViolation): gate.record_approval(batch,decision="approved",approver=approver,approved_at=timestamp)

 def test_unhashed_views_cannot_inject_future_operations(self):
  batch=self.single(); approval=gate.record_approval(batch,decision="approved",approver="owner",approved_at="2026-08-12T12:30:00Z")
  batch["actionable"].append({"proposal_id":"injected","publication_status":"published","unexpected":"value"})
  operations=gate.create_handoff_package(batch,approval)["application_contract"]["operations"]
  self.assertNotIn("injected",{row["proposal_id"] for row in operations})

 def test_counting_relationship_batch_and_holdout_rules(self):
  proposals=self.results["proposals"].copy(); proposals.loc[(proposals.entity_id=="ent_multi")&(proposals.profile_relationship_type=="division"),"counting_entity_id"]="arbitrary_new_count"
  mixed=self.build(proposals=proposals)
  self.assertGreater(mixed["manifest"]["invalid_counting_entity_id_count"],0)
  self.assertEqual(mixed["manifest"]["unique_counting_entity_id_count"],len({row["counting_entity_id"] for row in mixed["items"] if row["gate_outcomes"]["counting_entity_id"]}))
  approval=gate.record_approval(mixed,decision="approved",approver="owner",approved_at="2026-08-12T12:30:00Z")
  with self.assertRaises(gate.HandoffGateViolation): gate.create_handoff_package(mixed,approval)
  invalid=self.results["proposals"].copy(); invalid.loc[invalid.entity_id=="ent_reuse","profile_relationship_type"]="unknown"
  self.assertGreater(self.build(proposals=invalid)["manifest"]["invalid_relationship_type_count"],0)
  mismatch=self.results["proposals"].copy(); mismatch.loc[mismatch.entity_id=="ent_reuse","integration_batch_id"]="other"
  normalized=self.build(proposals=mismatch); row=next(row for row in normalized["items"] if row["entity_id"]=="ent_reuse")
  self.assertEqual(row["integration_batch_id"],"batch-synthetic"); self.assertEqual(row["proposed_values"]["integration_batch_id"],"batch-synthetic"); self.assertEqual(row["source_integration_batch_id"],"other")

 def test_duplicate_binding_keys_block_deterministically(self):
  for name,key in [("crosswalks","duplicate_crosswalk_key_count"),("airtable_profiles","duplicate_profile_record_id_count")]:
   frame=self.tables[name]; duplicate=pd.concat([frame,frame.iloc[[0]]],ignore_index=True)
   first=self.build(**{name:duplicate}); shuffled=self.build(**{name:duplicate.sample(frac=1,random_state=24).reset_index(drop=True)})
   self.assertGreater(first["manifest"][key],0); self.assertEqual(first["manifest"][key],shuffled["manifest"][key]); self.assertEqual(first["manifest"]["handoff_ready"],shuffled["manifest"]["handoff_ready"])

 def test_order_insensitive_hash_outcomes_and_rollback_state(self):
  first=self.build(); shuffled=self.build(proposals=self.results["proposals"].sample(frac=1,random_state=7))
  self.assertEqual(first["manifest"]["proposal_input_hash"],shuffled["manifest"]["proposal_input_hash"])
  package=self.package(self.single()); op=package["application_contract"]["operations"][0]; current={op["airtable_record_id"]:op["current_fingerprint"]}
  for outcome in ("not_attempted","failed"):
   self.assertEqual(len(gate.derive_retry_package(package,[{"proposal_id":op["proposal_id"],"outcome":outcome}],current)["eligible_operations"]),1)
  self.assertEqual(gate.derive_retry_package(package,[{"proposal_id":op["proposal_id"],"outcome":"already_in_desired_state"}],current)["reconciled_proposal_ids"],[op["proposal_id"]])
  self.assertEqual(gate.derive_retry_package(package,[{"proposal_id":op["proposal_id"],"outcome":"stale_at_application"}],current)["stale_proposal_ids"],[op["proposal_id"]])
  planned=gate.build_rollback_candidate(package,[{"proposal_id":op["proposal_id"],"outcome":"applied"}],application_reference="run")
  self.assertFalse(planned["operations"][0]["application_time_state_supplied"]); self.assertIsNone(planned["operations"][0]["application_time_before_values"])
  supplied={field:"before" for field in engine.PROPOSAL_FIELDS}; actual=gate.build_rollback_candidate(package,[{"proposal_id":op["proposal_id"],"outcome":"applied","application_time_before_values":supplied}],application_reference="run")
  self.assertTrue(actual["operations"][0]["application_time_state_supplied"]); self.assertEqual(set(actual["operations"][0]["application_time_before_values"]),set(engine.PROPOSAL_FIELDS))

 def test_config_and_batch_validation(self):
  self.assertEqual(set(self.config["relationship_types"]),{"group","division","service_line"})
  schema=json.loads((ROOT/"schemas"/"directory_integration_batch_manifest.schema.json").read_text())
  self.assertFalse(schema["additionalProperties"]); self.assertTrue(set(self.single()["manifest"]) <= set(schema["properties"]))
  with self.assertRaises(gate.HandoffGateViolation): gate.build_candidate_batch([],[],[],[],[],[],batch_id="",created_at="2026-08-12T12:00:00Z",created_by="operator",config=self.config)

if __name__=="__main__": unittest.main()
