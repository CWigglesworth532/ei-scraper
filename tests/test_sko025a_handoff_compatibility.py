import unittest
from pathlib import Path
import pandas as pd
import directory_integration as engine
import directory_integration_handoff as gate
ROOT=Path(__file__).resolve().parents[1]
class Tests(unittest.TestCase):
 def test_same_entity_relationship_can_bind_multiple_exact_records(self):
  tables=engine.load_fixture_pack(ROOT/"tests/fixtures/directory"); results=engine.build_proposals(tables); config=gate.load_config(ROOT/"config/directory_integration_handoff_no_write.yaml")
  base=results["proposals"].loc[(results["proposals"].entity_id=="ent_multi")&(results["proposals"].profile_relationship_type=="division")]
  proposals=pd.concat([results["proposals"],base],ignore_index=True)
  refs=pd.concat([results["integration_references"],pd.DataFrame([{"candidate_id":"c_extra","entity_id":"ent_multi","airtable_record_id":"rec_multi_division_2","proposal_operation":"review_update"}])],ignore_index=True)
  decisions=pd.concat([results["decisions"],pd.DataFrame([{"candidate_id":"c_extra","entity_id":"ent_multi","airtable_record_id":"rec_multi_division_2","identity_status":"approved_crosswalk","classification_status":"eligible","readiness_status":"ready"}])],ignore_index=True)
  cross=pd.concat([tables["crosswalks"],pd.DataFrame([{"entity_id":"ent_multi","airtable_record_id":"rec_multi_division_2","profile_relationship_type":"division","crosswalk_status":"approved","approved_fingerprint":"fp_extra"}])],ignore_index=True)
  profiles=pd.concat([tables["airtable_profiles"],pd.DataFrame([{"airtable_record_id":"rec_multi_division_2","Organisation":"Synthetic Extra","publication_status":"unpublished","profile_status":"active","record_fingerprint":"fp_extra"}])],ignore_index=True)
  batch=gate.build_candidate_batch(proposals,refs,decisions,profiles,cross,tables["canonical_entities"],batch_id="batch-compat",created_at="2026-08-13T12:00:00Z",created_by="test",config=config)
  rows=[x for x in batch["items"] if x["entity_id"]=="ent_multi" and x["profile_relationship_type"]=="division"]
  self.assertEqual(len(rows),2); self.assertEqual(len({x["airtable_record_id"] for x in rows}),2)
 def test_approved_vocabulary(self):
  config=gate.load_config(ROOT/"config/directory_integration_handoff_no_write.yaml")
  self.assertEqual(set(config["relationship_types"]),{"group","division","service_line","brand","establishment","operating_unit"})
if __name__=="__main__": unittest.main()
