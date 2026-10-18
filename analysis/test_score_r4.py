"""Scorer contract tests with explicitly synthetic fixtures, never R4 answers."""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCORER = Path(__file__).with_name("score_r4.py")
if not SCORER.exists():
    SCORER = Path(__file__).with_name("score_answer_sheet.py")
FIELDS = ["question", "choice_A", "choice_B", "choice_C", "choice_D", "choice_E"]

def save(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

class ScorerContractTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="synthetic-test-", dir=SCORER.parent)
        self.root = Path(self.tmp.name)
        self.answers, self.probes, self.manifest, self.mapping = [], [], [], []
        for tier in range(1, 6):
            cid = f"SYNTHETIC-{tier}"
            self.manifest.append({"case_id": cid, "risk_tier": f"L{tier}"})
            for kind in ["original", "lr_flip", "roi_only", "roi_masked", "tcf", "negation", "specificity_drop", "knowledge_only", "halluc_trap"]:
                text = f"Synthetic case {tier} " + ("shared image question" if kind in ("original", "lr_flip", "roi_only", "roi_masked") else kind)
                pid = cid + "__" + kind
                gold = "E" if kind in ("halluc_trap", "roi_masked") else "A"
                choices = {f"choice_{letter}": f"Synthetic option {letter}" for letter in "ABCDE"}
                seq = str(len(self.answers)+1)
                self.probes.append({"case_id": cid, "probe_id": pid, "probe_kind": kind, "question": text, "correct_letter": gold, **choices})
                self.answers.append({"seq": seq, "image": f"synthetic/{seq}.png", "question": text, "answer": gold, "data_origin": "synthetic_test_fixture", **choices})
                self.mapping.append({"seq": seq, "probe_id": pid})
        self.persist()

    def tearDown(self):
        if self.root.resolve().parent != SCORER.parent.resolve():
            raise RuntimeError("Temporary fixture path escaped the scorer workspace")
        self.tmp.cleanup()

    def persist(self):
        save(self.root/"answers.csv", self.answers, ["seq", "image", *FIELDS, "answer", "data_origin"])
        save(self.root/"probes.csv", self.probes, ["case_id", "probe_id", "probe_kind", *FIELDS, "correct_letter"])
        save(self.root/"manifest.csv", self.manifest, ["case_id", "risk_tier"])
        save(self.root/"mapping.csv", self.mapping, ["seq", "probe_id"])

    def run_cli(self, *extra):
        return subprocess.run([sys.executable, str(SCORER), "--answers", str(self.root/"answers.csv"), "--probes", str(self.root/"probes.csv"), "--manifest", str(self.root/"manifest.csv"), "--output-dir", str(self.root/"out"), *extra], capture_output=True, text=True)

    def test_ambiguous_image_identity_does_not_produce_mcs(self):
        run = self.run_cli()
        self.assertEqual(run.returncode, 0, run.stderr)
        summary = json.loads((self.root/"out/summary.json").read_text())
        self.assertIsNone(summary["metrics"]["MCS"])
        self.assertIsNone(summary["metrics"]["Original"]["pct"])
        self.assertEqual(summary["metrics"]["PR"]["n_correct"], 5)
        self.assertEqual(summary["metrics"]["SFR_pct_exact"], "0/1")

    def test_verified_mapping_resolves_all_metrics(self):
        run = self.run_cli("--seq-map", str(self.root/"mapping.csv"), "--require-complete")
        self.assertEqual(run.returncode, 0, run.stderr)
        summary = json.loads((self.root/"out/summary.json").read_text())
        self.assertTrue(summary["identity_resolved"])
        self.assertEqual(summary["metrics"]["MCS_exact"], "90/1")
        self.assertEqual(summary["metrics"]["MCS_1dp"], "90.0")
        self.assertTrue(all(t["MCS"]["one_decimal"] == "90.0" for t in summary["metrics"]["composites_by_tier"].values()))

    def test_full_reference_scope_is_not_labeled_public(self):
        run = self.run_cli("--seq-map", str(self.root/"mapping.csv"), "--require-complete", "--reference-scope", "full")
        self.assertEqual(run.returncode, 0, run.stderr)
        summary = json.loads((self.root/"out/summary.json").read_text())
        self.assertEqual(summary["reference_scope"], "full")
        self.assertIsNone(summary["n_public_probes"])
        self.assertEqual(summary["n_reference_probes"], 45)

    def test_require_complete_rejects_missing_mapping(self):
        run = self.run_cli("--require-complete")
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("verified seq-to-probe mapping", run.stderr)
        self.assertFalse((self.root/"out/summary.json").exists())

    def test_mapping_wrong_question_rejected(self):
        self.mapping[0]["probe_id"], self.mapping[4]["probe_id"] = self.mapping[4]["probe_id"], self.mapping[0]["probe_id"]
        self.persist()
        run = self.run_cli("--seq-map", str(self.root/"mapping.csv"))
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("specified probe", run.stderr)

    def test_signature_multiplicity_rejected(self):
        self.answers.pop()
        self.persist()
        run = self.run_cli()
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("multiplicity mismatch", run.stderr)

    def test_same_gold_tcf_duplicate_aggregate_identified(self):
        first = next(p for p in self.probes if p["probe_kind"] == "tcf")
        second = next(p for p in self.probes if p["probe_kind"] == "tcf" and p["case_id"] != first["case_id"])
        ar = next(a for a in self.answers if a["question"] == second["question"])
        second["question"] = first["question"]
        ar["question"] = first["question"]
        ar["answer"] = "B"
        self.persist()
        run = self.run_cli()
        self.assertEqual(run.returncode, 0, run.stderr)
        summary = json.loads((self.root/"out/summary.json").read_text())
        self.assertEqual(summary["metrics"]["PR"]["n_correct"], 4)
        with (self.root/"out/public_family_tier.csv").open(encoding="utf-8", newline="") as handle:
            tiers = list(csv.DictReader(handle))
        ambiguous = [r for r in tiers if r["probe_kind"] == "tcf" and r["risk_tier"] in ("L1", "L2")]
        self.assertTrue(all(r["status"] != "identified" for r in ambiguous))

    def test_half_up_tie(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("r4_scorer", SCORER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.one_decimal(module.Fraction(25, 4)), "6.3")

if __name__ == "__main__":
    unittest.main()
