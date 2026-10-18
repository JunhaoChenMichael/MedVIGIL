"""Synthetic-only tests for native key adaptation and provenance validation."""
import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("prepare_r4_inputs", Path(__file__).with_name("prepare_r4_inputs.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class PrepareR4Test(unittest.TestCase):
    def setUp(self):
        fields={k: 'synthetic '+k for k in module.FIELDS}
        self.answers=[{'seq':'1','image':'images/0001.png','answer':'A',**fields},{'seq':'2','image':'images/0002.png','answer':'E',**fields}]
        self.key=[{'seq':'1','probe_id':'SYNTHETIC1__orig','source_image':'images/SYNTHETIC1.png','r4_image':'images/0001.png'},{'seq':'2','probe_id':'SYNTHETIC2__trap1','source_image':'images/SYNTHETIC2.png','r4_image':'images/0002.png'}]
        self.public=[{'probe_id':'SYNTHETIC1__orig','case_id':'SYNTHETIC1','probe_kind':'original','image_file':'SYNTHETIC1.png','correct_letter':'A',**fields}]
        self.manifest=[{'case_id':'SYNTHETIC1','risk_tier':'L1'}]
        self.gold=[{'probe_id':'SYNTHETIC1__orig','case_id':'SYNTHETIC1','probe_kind':'original','risk_tier':'L1','correct_letter':'A'},{'probe_id':'SYNTHETIC2__trap1','case_id':'SYNTHETIC2','probe_kind':'halluc_trap','risk_tier':'L5','correct_letter':'E'}]

    def run_prepare(self,scope='public',gold=None):
        return module.prepare(copy.deepcopy(self.answers),copy.deepcopy(self.key),self.public,self.manifest,scope,gold)

    def test_explicit_public_scope_records_exclusion(self):
        answers,probes,key,manifest,report=self.run_prepare()
        self.assertEqual(len(answers),1)
        self.assertEqual(report['n_excluded_answers'],1)
        self.assertEqual(key[0]['reference_image_file'],'SYNTHETIC1.png')

    def test_full_requires_external_gold(self):
        with self.assertRaisesRegex(ValueError,'separate complete gold'):
            self.run_prepare('full')

    def test_full_joins_external_gold_and_preserves_letters(self):
        answers,probes,key,manifest,report=self.run_prepare('full',self.gold)
        self.assertEqual([r['answer'] for r in answers],['A','E'])
        self.assertEqual(probes[1]['correct_letter'],'E')
        self.assertTrue(report['separate_full_gold_validated'])

    def test_wrong_reader_image_rejected(self):
        self.key[0]['r4_image']='images/incorrect.png'
        with self.assertRaisesRegex(ValueError,'r4_image'):
            self.run_prepare()

    def test_wrong_public_variant_rejected(self):
        self.key[0]['source_image']='images_perturbed/SYNTHETIC1_lr_flip.png'
        with self.assertRaisesRegex(ValueError,'released image variant'):
            self.run_prepare()

    def test_public_gold_conflict_rejected(self):
        self.gold[0]['correct_letter']='B'
        with self.assertRaisesRegex(ValueError,'released public gold'):
            self.run_prepare('full',self.gold)

    def test_gold_missing_private_reference_rejected(self):
        with self.assertRaisesRegex(ValueError,'every key probe'):
            self.run_prepare('full',self.gold[:1])

    def test_mismatched_public_text_rejected(self):
        self.answers[0]['question']='different question'
        with self.assertRaisesRegex(ValueError,'question/options'):
            self.run_prepare()

    def test_unknown_scope_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Scope must explicitly"):
            self.run_prepare("unspecified", self.gold)

    def test_full_scope_rejects_formula_text_without_rewriting(self):
        self.answers[1]["question"] = ' \t=HYPERLINK("https://example.invalid")'
        original = copy.deepcopy(self.answers)
        with self.assertRaisesRegex(ValueError, "Formula-prefixed CSV cell"):
            self.run_prepare("full", self.gold)
        self.assertEqual(self.answers, original)

    def test_missing_public_manifest_case_is_rejected_clearly(self):
        self.manifest[0]["case_id"] = "SYNTHETIC_OTHER"
        with self.assertRaisesRegex(ValueError, "Public probe lacks manifest"):
            self.run_prepare()

    def test_failed_publication_leaves_no_partial_inputs(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            output = Path(temp)/"out"
            with self.assertRaisesRegex(OSError, "simulated write failure"):
                with module.new_output_directory(output) as staged:
                    (staged/"partial.csv").write_text("partial")
                    raise OSError("simulated write failure")
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_existing_output_directory_is_never_reused(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            output = Path(temp)/"out"
            output.mkdir()
            (output/"old.csv").write_text("old input set")
            with self.assertRaisesRegex(ValueError, "Output directory already exists"):
                with module.new_output_directory(output):
                    self.fail("Existing output was accepted")
            self.assertEqual((output/"old.csv").read_text(), "old input set")

    def test_in_progress_output_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            output = Path(temp)/"out"
            with module.new_output_directory(output) as staged:
                with self.assertRaisesRegex(ValueError, "already in progress"):
                    with module.new_output_directory(output):
                        self.fail("Concurrent writer was accepted")
                (staged/"complete.csv").write_text("complete input set")
            self.assertEqual((output/"complete.csv").read_text(), "complete input set")
            self.assertEqual([p.name for p in Path(temp).iterdir()], ["out"])

    def test_preparation_cli_publishes_one_complete_input_set(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp)
            for name, rows in (("answers", self.answers), ("key", self.key), ("probes", self.public), ("manifest", self.manifest)):
                module.write(root/(name+".csv"), rows, list(rows[0]))
            command = [sys.executable, str(Path(__file__).with_name("prepare_r4_inputs.py")), "--answers", str(root/"answers.csv"), "--seq-key", str(root/"key.csv"), "--public-probes", str(root/"probes.csv"), "--public-manifest", str(root/"manifest.csv"), "--scope", "public", "--output-dir", str(root/"out")]
            first = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(len(list((root/"out").iterdir())), 5)
            self.assertEqual(json.loads((root/"out/preparation_report.json").read_text())["n_selected_answers"], 1)
            before = {f.name: f.read_bytes() for f in (root/"out").iterdir()}
            second = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("Output directory already exists", second.stderr)
            self.assertEqual(before, {f.name: f.read_bytes() for f in (root/"out").iterdir()})

if __name__=='__main__':
    unittest.main()
