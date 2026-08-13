import copy,hashlib,inspect,sys,unittest
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"scripts"))
import directory_integration as engine
import sko025a_forward_flow as flow

def fixture():
 def df(rows,cols): return pd.DataFrame(rows,columns=cols)
 entities=df([
  ["ent_existing","Existing Supplier","active","ent_existing","",""],
  ["ent_new","New Supplier","active","ent_new","",""],
  ["ent_hold","Held Supplier","active","ent_hold","",""],
  ["ent_group","Group Supplier","active","ent_group","",""],
  ["ent_division","Division Supplier","active","ent_group","",""],
 ],["entity_id","canonical_name","entity_status","counting_entity_id","redirect_to","split_status"])
 candidates=df([
  ["c_existing","ent_existing","group","batch-forward"],
  ["c_new","ent_new","group","batch-forward"],
  ["c_hold","ent_hold","group","batch-forward"],
  ["c_group","ent_group","group","batch-forward"],
  ["c_division","ent_division","division","batch-forward"],
 ],["candidate_id","entity_id","profile_relationship_type","integration_batch_id"])
 cross=df([
  ["ent_existing","recExisting","group","approved","fp-existing"],
  ["ent_group","recGroup","group","approved","fp-group"],
 ],["entity_id","airtable_record_id","profile_relationship_type","crosswalk_status","approved_fingerprint"])
 profiles=df([
  ["recExisting","Protected Existing","unpublished","active","fp-existing"],
  ["recGroup","Protected Group","published","active","fp-group"],
 ],["airtable_record_id","Organisation","publication_status","profile_status","record_fingerprint"])
 classes=df([[e,"eligible",f"class-{e}"] for e in entities.entity_id],["entity_id","classification_status","evidence_reference"])
 ready=df([["ent_existing","true","ready","2026-08-13T00:00:00Z","ready-existing"],["ent_new","true","ready","2026-08-13T00:00:00Z","ready-new"],["ent_hold","false","not_ready","2026-08-13T00:00:00Z","hold"],["ent_group","true","ready","2026-08-13T00:00:00Z","ready-group"],["ent_division","true","ready","2026-08-13T00:00:00Z","ready-division"]],["entity_id","enrichment_complete","readiness_status","assessed_at","evidence_reference"])
 return {"canonical_entities":entities,"directory_candidates":candidates,"airtable_profiles":profiles,"crosswalks":cross,"classifications":classes,"readiness":ready,"activity_metrics":df([], ["activity_id","entity_id","supplier_count","spend_eur","impact_count"]),"lifecycle_events":df([], ["event_id","event_type","entity_id","target_entity_id","reason","evidence_reference"]),"integration_history":df([], ["history_id","entity_id","airtable_record_id","prior_integration_values","editorial_fingerprint","transition_reason","evidence_reference"]),"expected_behaviours":df([], ["test_id","expected_outcome"])}

class Tests(unittest.TestCase):
 def setUp(self): self.tables=fixture(); self.result=flow.evaluate_forward_flow(self.tables)
 def test_existing_profile_enrichment_reuses_exact_record_without_duplicate(self):
  rows=self.result["existing_profile_proposals"].loc[lambda d:d.entity_id.eq("ent_existing")]; self.assertEqual(len(rows),1)
  refs=self.result["existing_profile_references"].loc[lambda d:d.entity_id.eq("ent_existing")]; self.assertEqual(refs.iloc[0].airtable_record_id,"recExisting")
  self.assertFalse(self.result["new_profile_candidates"].entity_id.eq("ent_existing").any())
 def test_absent_ready_supplier_is_new_profile_candidate_without_record_id(self):
  row=self.result["new_profile_candidates"].loc[lambda d:d.entity_id.eq("ent_new")].iloc[0]
  self.assertEqual(row.candidate_status,"new_directory_profile_candidate"); self.assertEqual(row.airtable_record_id,""); self.assertEqual(row.counting_entity_id,"ent_new")
 def test_not_ready_supplier_is_holdout(self): self.assertEqual(self.result["readiness_holdouts"].entity_id.tolist(),["ent_hold"])
 def test_one_to_many_additional_profile_and_counting(self):
  row=self.result["new_profile_candidates"].loc[lambda d:d.entity_id.eq("ent_division")].iloc[0]
  self.assertEqual(row.profile_relationship_type,"division"); self.assertEqual(row.counting_entity_id,"ent_group")
  self.assertGreater(int(self.result["counting_qa"].iloc[0].additional_profiles_without_supplier_inflation),0)
 def test_safety_and_determinism(self):
  flow.assert_safety(self.result); again=flow.evaluate_forward_flow(copy.deepcopy(self.tables))
  for k in self.result: pd.testing.assert_frame_equal(self.result[k],again[k])
  self.assertNotIn("publication_status",self.result["existing_profile_proposals"].columns)
 def test_no_network_write_or_apply_capability(self):
  source=inspect.getsource(flow).casefold(); self.assertFalse([x for x in ["requests","urllib.request","httpx","pyairtable","--write","--apply"] if x in source])
if __name__=="__main__":unittest.main()
