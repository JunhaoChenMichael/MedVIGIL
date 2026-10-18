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

    def cli_args(self, *extra, scope="provided"):
        scope_args = ["--reference-scope", scope] if scope is not None and "--reference-scope" not in extra else []
        return [sys.executable, str(SCORER), "--answers", str(self.root/"answers.csv"), "--probes", str(self.root/"probes.csv"), "--manifest", str(self.root/"manifest.csv"), "--output-dir", str(self.root/"out"), *scope_args, *extra]

    def run_cli(self, *extra, scope="provided"):
        return subprocess.run(self.cli_args(*extra, scope=scope), capture_output=True, text=True)

    def test_reference_scope_is_mandatory(self):
        run = self.run_cli(scope=None)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("--reference-scope", run.stderr)
        self.assertFalse((self.root/"out").exists())

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

    def test_missing_family_is_unavailable_not_zero(self):
        removed = {p["probe_id"] for p in self.probes if p["probe_kind"] == "roi_only"}
        removed_seqs = {m["seq"] for m in self.mapping if m["probe_id"] in removed}
        self.probes = [p for p in self.probes if p["probe_id"] not in removed]
        self.mapping = [m for m in self.mapping if m["probe_id"] not in removed]
        self.answers = [a for a in self.answers if a["seq"] not in removed_seqs]
        self.persist()
        run = self.run_cli("--seq-map", str(self.root/"mapping.csv"), "--require-complete")
        self.assertEqual(run.returncode, 0, run.stderr)
        summary = json.loads((self.root/"out/summary.json").read_text())
        self.assertEqual(summary["metrics"]["ROI_only"]["n"], 0)
        self.assertIsNone(summary["metrics"]["ROI_only"]["n_correct"])
        self.assertIsNone(summary["metrics"]["ROI_only"]["pct"])
        self.assertIsNone(summary["metrics"]["MCS"])
        self.assertTrue(any("absent" in message for message in summary["limitations"]))

    def test_missing_tier_is_unavailable_not_zero(self):
        removed = {p["probe_id"] for p in self.probes if p["case_id"] == "SYNTHETIC-5"}
        removed_seqs = {m["seq"] for m in self.mapping if m["probe_id"] in removed}
        self.probes = [p for p in self.probes if p["probe_id"] not in removed]
        self.mapping = [m for m in self.mapping if m["probe_id"] not in removed]
        self.answers = [a for a in self.answers if a["seq"] not in removed_seqs]
        self.manifest = [m for m in self.manifest if m["risk_tier"] != "L5"]
        self.persist()
        run = self.run_cli("--seq-map", str(self.root/"mapping.csv"), "--require-complete")
        self.assertEqual(run.returncode, 0, run.stderr)
        metrics = json.loads((self.root/"out/summary.json").read_text())["metrics"]
        self.assertIsNone(metrics["SFR_by_tier"]["L5"]["n_silent_failures"])
        self.assertIsNone(metrics["SFR_w_pct"])
        self.assertIsNone(metrics["composites_by_tier"]["L5"])
        self.assertIsNone(metrics["MCS"])

    def test_formula_prefixed_exported_image_is_rejected(self):
        self.answers[0]["image"] = ' \t=HYPERLINK("https://example.invalid")'
        self.persist()
        run = self.run_cli()
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("Formula-prefixed CSV cell", run.stderr)
        self.assertFalse((self.root/"out").exists())

    def test_successful_report_cannot_be_reused(self):
        first = self.run_cli("--seq-map", str(self.root/"mapping.csv"))
        self.assertEqual(first.returncode, 0, first.stderr)
        before = {f.name: f.read_bytes() for f in (self.root/"out").iterdir()}
        second = self.run_cli("--require-complete")
        self.assertNotEqual(second.returncode, 0)
        self.assertIn("Output directory already exists", second.stderr)
        self.assertEqual(before, {f.name: f.read_bytes() for f in (self.root/"out").iterdir()})

    def test_existing_output_directory_is_rejected(self):
        (self.root/"out").mkdir()
        (self.root/"out/old.txt").write_text("old report")
        run = self.run_cli()
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("Output directory already exists", run.stderr)
        self.assertEqual([f.name for f in (self.root/"out").iterdir()], ["old.txt"])

    def test_failed_publication_leaves_no_partial_report(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("r4_scorer", SCORER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with self.assertRaisesRegex(OSError, "simulated write failure"):
            with module.new_output_directory(self.root/"out") as staged:
                (staged/"partial.txt").write_text("partial")
                raise OSError("simulated write failure")
        self.assertFalse((self.root/"out").exists())
        self.assertEqual(list(self.root.glob(".out*")), [])

    def test_concurrent_report_writers_do_not_mix_outputs(self):
        command = self.cli_args("--seq-map", str(self.root/"mapping.csv"))
        first = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        second = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        first.communicate(timeout=30)
        second.communicate(timeout=30)
        self.assertEqual(sorted([first.returncode, second.returncode]), [0, 1])
        summary = json.loads((self.root/"out/summary.json").read_text())
        self.assertEqual(summary["metrics"]["MCS_exact"], "90/1")
        self.assertEqual(len(list((self.root/"out").iterdir())), 7)
        self.assertEqual(list(self.root.glob(".out*")), [])

if __name__ == "__main__":
    unittest.main()
