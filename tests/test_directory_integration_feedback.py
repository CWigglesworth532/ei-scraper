import copy, inspect, json, random, unittest
from pathlib import Path
import directory_integration_feedback as feedback

ROOT=Path(__file__).resolve().parents[1]; FIXTURE=ROOT/"tests/fixtures/directory_feedback"
class Tests(unittest.TestCase):
 def setUp(self):
  self.context=json.loads((FIXTURE/"context.json").read_text()); self.rows=json.loads((FIXTURE/"feedback.json").read_text())
 def row(self,fid,target_type,target_id,decision,**kw):
  row={"schema_version":feedback.SCHEMA_VERSION,"feedback_id":fid,"feedback_batch_id":"batch_test","target_type":target_type,"target_id":target_id,"decision":decision,"reviewer":"synthetic-owner","decision_at":"2026-08-17T13:00:00Z","reason":"synthetic","source_review_file":"synthetic.csv","source_review_file_hash":"b"*64,"related_batch_id":"","related_proposal_id":"","crosswalk_id":"","airtable_record_id":"","entity_id":"","relationship_type":"","counting_entity_id":"","fingerprint_version":"","prior_state_hash":"","supersedes_feedback_id":""}; row.update(kw); return row
 def reasons(self,result): return [r["rejection_reason"] for r in result["rejected"]]
 def test_fixture_exact_crosswalk_readiness_new_profile(self):
  r=feedback.import_feedback(self.rows,context=self.context); self.assertEqual(len(r["accepted"]),3); self.assertEqual(r["evidence"]["sko024_ineligible_new_profile_candidates"],1); self.assertEqual(next(x for x in r["current_state"] if x["target_type"]=="new_profile_candidate")["airtable_record_id"],"")
 def test_rejected_crosswalk_is_governed(self):
  row=dict(self.rows[0],feedback_id="fb_cross_reject",decision="reject"); self.assertEqual(feedback.import_feedback([row],context=self.context)["current_state"][0]["decision"],"reject")
 def test_stale_version_and_record_binding_blocks(self):
  stale=dict(self.rows[0],feedback_id="fb_stale",target_id="xw_stale0000000000000000000",crosswalk_id="xw_stale0000000000000000000",airtable_record_id="recStale123",relationship_type="brand")
  version=dict(self.rows[0],feedback_id="fb_version",fingerprint_version="v2"); malformed=dict(self.rows[0],feedback_id="fb_bad_record",airtable_record_id="bad"); mismatch=dict(self.rows[0],feedback_id="fb_wrong_record",airtable_record_id="recOther")
  self.assertCountEqual(self.reasons(feedback.import_feedback([stale,version,malformed,mismatch],context=self.context)),["stale_fingerprint","wrong_fingerprint_version","malformed_airtable_record_id","airtable_record_id_mismatch"])
 def test_relationship_and_unsupported_type(self):
  good=self.row("fb_rel","relationship","rel_a","approve",entity_id="ent_a",relationship_type="division",airtable_record_id="recExact123",fingerprint_version=feedback.FINGERPRINT_VERSION); bad=dict(good,feedback_id="fb_rel_bad",relationship_type="subsidiary")
  r=feedback.import_feedback([good,bad],context=self.context); self.assertEqual([x["feedback_id"] for x in r["accepted"]],["fb_rel"]); self.assertIn("unsupported_relationship_type",self.reasons(r))
 def test_counting_default_nondefault_and_invalid(self):
  default=self.row("fb_count_self","counting_mapping","cm_self","approve",entity_id="ent_a",counting_entity_id="ent_a"); grouped=self.row("fb_count_group","counting_mapping","cm_group","approve",entity_id="ent_a",counting_entity_id="ent_group"); invalid=self.row("fb_count_bad","counting_mapping","cm_group","approve",entity_id="ent_a",counting_entity_id="ent_new")
  r=feedback.import_feedback([default,grouped,invalid],context=self.context); self.assertEqual(len(r["accepted"]),2); self.assertIn("invalid_non_default_counting_mapping",self.reasons(r))
 def test_nondefault_must_be_approved(self):
  row=self.row("fb_count_reject","counting_mapping","cm_group","reject",entity_id="ent_a",counting_entity_id="ent_group"); self.assertIn("non_default_counting_mapping_not_approved",self.reasons(feedback.import_feedback([row],context=self.context)))
 def test_readiness_hold_research_supersession(self):
  hold=self.row("fb_ready_hold","readiness","ent_a","hold",entity_id="ent_a"); first=feedback.import_feedback([hold],context=self.context)
  research=self.row("fb_ready_research","readiness","ent_a","research_needed",entity_id="ent_a",supersedes_feedback_id="fb_ready_hold",prior_state_hash=feedback.current_state_hash(first["current_state"][0])); r=feedback.import_feedback([research],context=self.context,existing_ledger=first["ledger"])
  self.assertEqual(r["current_state"][0]["decision"],"research_needed"); self.assertEqual(r["evidence"]["protected_field_mutation_attempts"],0); self.assertEqual(r["evidence"]["publication_mutation_attempts"],0)
 def test_new_profile_rejected_and_fabricated_id_blocked(self):
  reject=self.row("fb_new_reject","new_profile_candidate","npc_new","reject",entity_id="ent_new"); bad=self.row("fb_new_bad","new_profile_candidate","npc_new","approve_future_creation",entity_id="ent_new",airtable_record_id="recInvented",feedback_batch_id="batch_fabricated")
  r=feedback.import_feedback([reject,bad],context=self.context); self.assertEqual(r["accepted"][0]["decision"],"reject"); self.assertIn("fabricated_airtable_record_id",self.reasons(r))
 def test_idempotent_duplicate_and_changed_conflict(self):
  first=feedback.import_feedback([self.rows[0]],context=self.context); rerun=feedback.import_feedback([self.rows[0]],context=self.context,existing_ledger=first["ledger"]); self.assertEqual(len(rerun["ledger"]),1); self.assertEqual(rerun["evidence"]["idempotent_duplicates"],1)
  conflict=feedback.import_feedback([dict(self.rows[0],reason="changed")],context=self.context,existing_ledger=first["ledger"]); self.assertIn("feedback_id_payload_conflict",self.reasons(conflict))
 def test_same_batch_conflict_blocks_all(self):
  a=self.rows[0]; b=dict(a,feedback_id="fb_other",decision="reject"); r=feedback.import_feedback([a,b],context=self.context); self.assertEqual(len(r["accepted"]),0); self.assertEqual(self.reasons(r),["conflicting_same_batch_feedback"]*2)
 def test_explicit_supersession_retains_history_one_current(self):
  first=feedback.import_feedback([self.rows[0]],context=self.context); replacement=dict(self.rows[0],feedback_id="fb_cross_revoke",feedback_batch_id="batch_revoke",decision="revoke")
  self.assertIn("contradictory_decision_requires_supersession",self.reasons(feedback.import_feedback([replacement],context=self.context,existing_ledger=first["ledger"])))
  replacement.update(supersedes_feedback_id="fb_cross_approve",prior_state_hash=feedback.current_state_hash(first["current_state"][0])); r=feedback.import_feedback([replacement],context=self.context,existing_ledger=first["ledger"]); self.assertEqual((len(r["ledger"]),len(r["current_state"])),(2,1)); self.assertEqual(r["current_state"][0]["decision"],"revoke")
 def test_bad_supersession_and_prior_hash(self):
  first=feedback.import_feedback([self.rows[0]],context=self.context); row=dict(self.rows[0],feedback_id="fb_replace",feedback_batch_id="batch2",decision="revoke",supersedes_feedback_id="fb_missing")
  self.assertIn("unknown_superseded_feedback_id",self.reasons(feedback.import_feedback([row],context=self.context,existing_ledger=first["ledger"]))); row.update(supersedes_feedback_id="fb_cross_approve",prior_state_hash="0"*64); self.assertIn("prior_state_hash_mismatch",self.reasons(feedback.import_feedback([row],context=self.context,existing_ledger=first["ledger"])))
 def test_malformed_metadata_and_unknown_target(self):
  rows=[self.row("fb_unknown","readiness","missing","hold"),self.row("fb_reviewer","readiness","ent_a","hold",reviewer=""),self.row("fb_time","readiness","ent_a","hold",decision_at="yesterday"),self.row("fb_schema","readiness","ent_a","hold",schema_version="v0")]
  self.assertCountEqual(self.reasons(feedback.import_feedback(rows,context=self.context)),["unknown_target_id","missing_reviewer","invalid_decision_timestamp","unsupported_schema_version"])
 def test_deterministic_rerun_and_shuffle(self):
  a=feedback.import_feedback(self.rows,context=self.context); b=feedback.import_feedback(copy.deepcopy(self.rows),context=self.context); shuffled=copy.deepcopy(self.rows); random.Random(26).shuffle(shuffled); c=feedback.import_feedback(shuffled,context=self.context); self.assertEqual(a,b); self.assertEqual(a,c)
 def test_schema_and_no_network_write_apply(self):
  schema=json.loads((ROOT/"schemas/directory_integration_feedback.schema.json").read_text()); self.assertEqual(schema["properties"]["schema_version"]["const"],feedback.SCHEMA_VERSION); self.assertTrue(set(feedback.FIELDS).issubset(schema["properties"]))
  source=inspect.getsource(feedback).casefold(); self.assertFalse([x for x in ["requests","httpx","pyairtable","urllib.request","--write","--apply","create_record","publication_status"] if x in source]); evidence=feedback.import_feedback(self.rows,context=self.context)["evidence"]; self.assertEqual((evidence["network_calls"],evidence["write_capability"],evidence["apply_capability"]),(0,False,False))
if __name__=="__main__": unittest.main()
