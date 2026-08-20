"""SKO-035 governed purchased-activity/NACE behaviours T01-T35."""
from __future__ import annotations
import copy, inspect, json, subprocess, sys, tempfile, unittest
from decimal import Decimal
from pathlib import Path
import purchased_activity_nace as pa

ROOT=Path(__file__).resolve().parents[1]; FIX=ROOT/"tests/fixtures/purchased_activity_nace"; CFG=FIX/"test_config.yaml"; PRODUCTION_CFG=ROOT/"config/purchased_activity_nace.yaml"; NOW="2026-08-20T10:00:00Z"
class PurchasedActivityNaceTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.cfg=pa.load_config(CFG); cls.obs=pa.read_csv(FIX/"observations.csv"); cls.ev=pa.read_csv(FIX/"evidence.csv"); cls.dec=pa.read_csv(FIX/"decisions.csv"); cls.result=pa.materialise(cls.obs,cls.ev,cls.dec,config=cls.cfg,generated_at=NOW); cls.rows=cls.result["assertions"]
 def rows_for(self,sid): return [r for r in self.rows if r["subject_id"]==sid]
 def run_changed(self, *, obs=None, ev=None, dec=None, cfg=None): return pa.materialise(obs or self.obs,ev or self.ev,dec or self.dec,config=cfg or self.cfg,generated_at=NOW)
 def test_t01_selected_only(self): self.assertFalse(self.rows_for("obs-unselected")); self.assertEqual(self.result["qa"]["selected_observations"],5)
 def test_t02_entity_passes_through(self): self.assertEqual(self.rows_for("obs-clean")[0]["entity_id"],"entity-001")
 def test_t03_observation_without_entity(self): self.assertEqual(self.rows_for("obs-split")[0]["entity_id"],"")
 def test_t04_no_canonical_mutation(self): self.assertNotIn("create_entity",inspect.getsource(pa)); self.assertNotIn("update_entity",inspect.getsource(pa))
 def test_t05_contract_supports_high(self): self.assertEqual(self.rows_for("obs-clean")[0]["assignment_confidence"],"High")
 def test_t06_service_line_distinct(self): self.assertIn("supplier_service_line",{r["evidence_type"] for r in self.result["evidence"]})
 def test_t07_official_distinct(self): self.assertIn("official_activity",{r["evidence_type"] for r in self.result["evidence"]})
 def test_t08_procurement_supporting(self): self.assertEqual(next(r for r in self.result["evidence"] if r["evidence_type"]=="procurement_category")["evidence_authority"],"contextual")
 def test_t09_weak_not_high(self):
  d=copy.deepcopy(self.dec); next(x for x in d if x["decision_key"]=="d-weak")["assignment_confidence"]="High"
  with self.assertRaisesRegex(ValueError,"Weak evidence"): self.run_changed(dec=d)
 def test_t10_nace_coherence(self):
  d=copy.deepcopy(self.dec); d[0]["nace_description"]="Wrong"
  with self.assertRaisesRegex(ValueError,"Invalid NACE"): self.run_changed(dec=d)
 def test_t11_national_not_relabelled(self): self.assertEqual(next(r for r in self.result["evidence"] if r["activity_scheme"]=="CNAE")["activity_scheme"],"CNAE")
 def test_t12_explicit_crosswalk(self):
  d=copy.deepcopy(self.dec); next(x for x in d if x["decision_key"]=="d-crosswalk")["crosswalk_id"]=""
  with self.assertRaisesRegex(ValueError,"crosswalk"): self.run_changed(dec=d)
  production=pa.load_config(PRODUCTION_CFG)
  self.assertEqual(production["crosswalks"],{})
  with self.assertRaisesRegex(ValueError,"crosswalk"): self.run_changed(cfg=production)
 def test_t13_conflict_escalates(self): self.assertEqual((self.rows_for("obs-weak")[0]["conflicting_evidence"],self.rows_for("obs-weak")[0]["review_escalation_state"]),("true","required"))
 def test_t14_single_row(self): self.assertEqual(len(self.rows_for("obs-clean")),1)
 def test_t15_multi_rows(self): self.assertEqual(len(self.rows_for("obs-split")),2)
 def test_t16_shares_reconcile(self): self.assertEqual(sum(Decimal(r["allocation_share"]) for r in self.rows_for("obs-split")),1)
 def test_t17_bad_shares_rejected(self):
  for value in ("0","-1","1.1"):
   d=copy.deepcopy(self.dec); d[0]["allocation_share"]=value
   with self.assertRaises(ValueError): self.run_changed(dec=d)
 def test_t18_no_invented_split(self):
  d=copy.deepcopy(self.dec); next(x for x in d if x["decision_key"]=="d-split-clean")["spend_split"]="false"
  with self.assertRaisesRegex(ValueError,"require evidenced"): self.run_changed(dec=d)
 def test_t19_unresolved_allowed(self): self.assertEqual(self.rows_for("obs-unresolved")[0]["nace_code"],"")
 def test_t20_broader_allowed(self): self.assertEqual((self.rows_for("obs-weak")[0]["nace_level"],self.result["qa"]["broader_level_assignment"]),("2",1))
 def test_t21_contract_flag_requires_evidence(self):
  d=copy.deepcopy(self.dec); next(x for x in d if x["decision_key"]=="d-weak")["contract_specific"]="true"
  with self.assertRaisesRegex(ValueError,"contract_specific"): self.run_changed(dec=d)
 def test_t22_materiality_escalation(self): self.assertEqual(self.result["qa"]["materiality_escalated"],1)
 def test_t23_original_values_unchanged(self): self.assertEqual((self.rows_for("obs-weak")[0]["spend"],self.rows_for("obs-weak")[0]["spend_year"],self.rows_for("obs-weak")[0]["supplier_country"]),("9000.00","2025","GB"))
 def test_t24_allocated_spend_reconciles(self): self.assertEqual(sum(Decimal(r["allocated_spend"]) for r in self.rows_for("obs-split")),Decimal("600.00"))
 def test_t25_provenance_retained(self): self.assertTrue(all(r["evidence_ids"] and r["evidence_dates"] and r["assignment_method"] and r["rationale"] and r["reviewer_decision_reference"] for r in self.rows))
 def test_t26_deterministic(self): self.assertEqual(self.result,self.run_changed())
 def test_t27_shuffle_logical_input(self): self.assertEqual(self.result,self.run_changed(obs=list(reversed(self.obs)),ev=list(reversed(self.ev)),dec=list(reversed(self.dec))))
 def test_t28_identical_duplicate_safe(self): self.assertEqual(self.result,self.run_changed(dec=self.dec+[copy.deepcopy(self.dec[0])]))
 def test_t29_conflicting_duplicate_rejected(self):
  d=copy.deepcopy(self.dec[0]); d["rationale"]="different"
  with self.assertRaisesRegex(ValueError,"Conflicting decision"): self.run_changed(dec=self.dec+[d])
 def test_t30_schema_contracts(self):
  es=json.loads((ROOT/"schemas/purchased_activity_evidence.schema.json").read_text()); ass=json.loads((ROOT/"schemas/purchased_activity_assertion.schema.json").read_text()); self.assertEqual(es["required"],pa.EVIDENCE_FIELDS); self.assertEqual(ass["required"],pa.ASSERTION_FIELDS); self.assertTrue(all(set(r)==set(pa.ASSERTION_FIELDS) for r in self.rows))
 def test_t31_synthetic_only(self): self.assertTrue(all("synthetic://" in r["evidence_reference"] for r in self.result["evidence"]))
 def test_t32_no_impact_logic(self): self.assertFalse({"coefficient","gva","labour_income","employment","production_tax","ghg"}&set(pa.ASSERTION_FIELDS))
 def test_t33_no_network(self): self.assertFalse(any(x in inspect.getsource(pa).casefold() for x in ("requests","urllib","airtable","socket")))
 def test_t34_no_policy_or_publication_mutation(self): self.assertFalse(any(x in inspect.getsource(pa).casefold() for x in ("social_economy","directory_publication","publish(")))
 def test_t35_cli_deterministic(self):
  hashes=[]
  for _ in range(2):
   with tempfile.TemporaryDirectory() as td:
    p=Path(td); args=[sys.executable,str(ROOT/"purchased_activity_nace.py"),"--config",str(CFG),"--observations",str(FIX/"observations.csv"),"--evidence",str(FIX/"evidence.csv"),"--decisions",str(FIX/"decisions.csv"),"--evidence-output",str(p/"e.csv"),"--assertion-output",str(p/"a.csv"),"--qa-output",str(p/"q.json"),"--generated-at",NOW]; subprocess.run(args,check=True); hashes.append(((p/"e.csv").read_bytes(),(p/"a.csv").read_bytes(),(p/"q.json").read_bytes()))
  self.assertEqual(hashes[0],hashes[1])
if __name__=="__main__": unittest.main()
