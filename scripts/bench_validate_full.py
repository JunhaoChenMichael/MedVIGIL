"""End-to-end integrity validator for medvigil_v1.

Run anytime to get a full quality report on the dataset state. Does NOT
modify anything; reports only.

Checks:
  L1  Manifest schema; 240 cases (public release) or 300 (with the
      credentialed chest-radiograph arm, see CXR_ACCESS.md on Hugging Face)
  L2  All images present and openable (chest radiographs not on disk are
      reported, not failed)
  L3  Grounding coverage and bbox sanity
  L4  Probe count consistency between probes_open and manifest
  L5  Triplet invariants (anchor != T-CF != V-CF; gold equalities/inequalities)
  L7  MCQ coverage (text probes have MCQ; image variants resolved)
  L8  MCQ choice quality (no empty / no duplicate / correct_letter rules)

Output:
  prints summary to stdout
  writes data/medvigil_v1/validation_report.md (machine-readable)
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data/medvigil_v1"
REPORT = BENCH / "validation_report.md"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower()).rstrip(".? ")


def load_csv(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


class Report:
    def __init__(self):
        self.checks = []  # list of (name, status, detail)
        self.fail = 0; self.warn = 0; self.ok = 0

    def add(self, name, status, detail=""):
        self.checks.append((name, status, detail))
        if status == "FAIL":
            self.fail += 1
        elif status == "WARN":
            self.warn += 1
        else:
            self.ok += 1

    def print(self):
        print(f"\n{'='*72}\nMedVIGIL Validation Report\n{'='*72}")
        for name, status, detail in self.checks:
            mark = {"OK": "✓", "WARN": "!", "FAIL": "✗"}[status]
            print(f"  [{mark}] {name:<55} {status}")
            if detail:
                for line in detail.splitlines():
                    print(f"        {line}")
        print(f"\nSummary: {self.ok} ok / {self.warn} warn / {self.fail} fail")

    def write(self, path: Path):
        lines = ["# MedVIGIL Validation Report\n"]
        lines.append(f"- ok: **{self.ok}**\n- warn: **{self.warn}**\n- fail: **{self.fail}**\n\n")
        for name, status, detail in self.checks:
            lines.append(f"## [{status}] {name}\n")
            if detail:
                lines.append("```\n" + detail + "\n```\n")
        path.write_text("".join(lines))


def main():
    R = Report()

    # ---- L1: manifest schema ----
    manifest = load_csv(BENCH / "manifest.csv")
    expected_manifest_cols = {"case_id", "source_dataset", "modality",
                               "image_file", "question", "gold_answer",
                               "risk_tier", "text_only_answerable",
                               "annotator_notes"}
    if not manifest:
        R.add("L1 manifest exists", "FAIL", "manifest.csv not found")
    elif set(manifest[0].keys()) != expected_manifest_cols:
        missing = expected_manifest_cols - set(manifest[0].keys())
        extra = set(manifest[0].keys()) - expected_manifest_cols
        R.add("L1 manifest schema", "FAIL",
              f"missing {missing}, extra {extra}")
    elif len(manifest) not in (240, 300):
        R.add("L1 manifest count", "FAIL", f"got {len(manifest)}; expected 240 (public) or 300 (full)")
    else:
        R.add(f"L1 manifest {len(manifest)}×9-cols", "OK")

    # ---- L2: images openable ----
    from PIL import Image
    bad_images = []
    credentialed = 0
    for r in manifest:
        p = BENCH / "images" / r["image_file"]
        if not p.exists() and r["source_dataset"] == "cxr":
            credentialed += 1      # MIMIC-CXR / CheXpert image: supplied by a credentialed user
            continue
        try:
            Image.open(p).verify()
        except Exception as e:
            bad_images.append((r["case_id"], str(e)[:50]))
    if bad_images:
        R.add("L2 images valid", "FAIL",
              "\n".join(f"{cid}: {e}" for cid, e in bad_images[:5]))
    else:
        R.add("L2 images valid", "OK",
              f"{len(manifest) - credentialed} present" + (f"; {credentialed} chest radiographs not on disk" if credentialed else ""))

    # ---- L3: grounding coverage + bbox sanity ----
    grounding = load_csv(BENCH / "grounding.csv")
    gd = {r["case_id"]: r for r in grounding}
    n_with_grounding = sum(1 for r in manifest if r["case_id"] in gd)
    if n_with_grounding < len(manifest):
        R.add("L3 grounding coverage", "FAIL",
              f"only {n_with_grounding}/{len(manifest)} cases have grounding")
    else:
        # bbox sanity
        bad_bbox = []
        for r in grounding:
            try:
                bb = json.loads(r["roi_bbox_norm"])
                if len(bb) != 4: raise ValueError("not 4 coords")
                x0, y0, x1, y1 = bb
                if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
                    raise ValueError(f"bad coords {bb}")
            except Exception as e:
                bad_bbox.append((r["case_id"], str(e)))
        if bad_bbox:
            R.add("L3 grounding bbox sanity", "WARN",
                  f"{len(bad_bbox)} bboxes invalid (kept as-is, scorer will skip ROI variants)")
        else:
            R.add(f"L3 grounding {n_with_grounding}/{len(manifest)} + bbox", "OK")

    # ---- L4: probes_open ----
    probes_open = load_csv(BENCH / "probes_open.csv")
    cnt_by_kind = Counter(p["probe_kind"] for p in probes_open)
    detail = "  ".join(f"{k}={v}" for k, v in sorted(cnt_by_kind.items()))
    if len(probes_open) < 1500:
        R.add("L4 probes_open ≥1500 rows", "FAIL", f"got {len(probes_open)}: {detail}")
    elif cnt_by_kind.get("original") != len(manifest):
        R.add("L4 probes_open one original per case", "FAIL", detail)
    else:
        R.add(f"L4 probes_open ({len(probes_open)} rows)", "OK", detail)

    # ---- L5: triplet invariants ----
    triplets = load_csv(BENCH / "triplets.csv")
    invariants_failed = 0
    for t in triplets:
        if norm(t["anchor_question"]) == norm(t["tcf_question"]):
            invariants_failed += 1
        elif norm(t["anchor_gold"]) != norm(t["tcf_gold"]):
            invariants_failed += 1
        elif norm(t["anchor_gold"]) == norm(t["vcf_gold"]):
            invariants_failed += 1
    if invariants_failed:
        R.add(f"L5 triplets ({len(triplets)}) invariants",
              "FAIL", f"{invariants_failed} triplets fail one or more of I1/I2/I3")
    else:
        R.add(f"L5 triplets ({len(triplets)}) invariants", "OK",
              f"all {len(triplets)} triplets satisfy anchor!=tcf, gold==tcf, gold!=vcf")

    # ---- L7: MCQ coverage ----
    mcq = load_csv(BENCH / "probes_mcq.csv")
    if not mcq:
        R.add("L7 probes_mcq exists", "WARN", "MCQ not generated yet")
    else:
        ids_open = {p["probe_id"] for p in probes_open}
        ids_mcq = {p["probe_id"] for p in mcq}
        gap = ids_open - ids_mcq
        if gap:
            R.add("L7 MCQ coverage", "WARN",
                  f"{len(gap)} open probes have no MCQ row "
                  f"(first 5: {sorted(gap)[:5]})")
        else:
            R.add(f"L7 MCQ coverage ({len(mcq)} rows)", "OK")

        # ---- L8: MCQ choice quality ----
        bad = defaultdict(int)
        for p in mcq:
            choices = {L: p[f"choice_{L}"] for L in "ABCDE"}
            non_empty = {L: c for L, c in choices.items() if c.strip()}
            if len(non_empty) < 3:
                bad["too_few_options"] += 1
            seen = set()
            for L, c in non_empty.items():
                if norm(c) in seen:
                    bad["dup_options"] += 1; break
                seen.add(norm(c))
            if p["correct_letter"] not in non_empty:
                bad["correct_invalid"] += 1
            if p["probe_kind"] == "halluc_trap" and p["correct_letter"] != "E":
                bad["trap_correct_not_E"] += 1
            elif p["probe_kind"] in {"original","tcf","negation","specificity_drop","knowledge_only"} and p["correct_letter"] != "A":
                bad["text_correct_not_A"] += 1
            elif p["probe_kind"] == "roi_masked" and p["correct_letter"] != "E":
                bad["roi_masked_correct_not_E"] += 1
        if bad:
            R.add("L8 MCQ choice quality gates", "WARN",
                  "  ".join(f"{k}={v}" for k, v in bad.items()))
        else:
            R.add("L8 MCQ choice quality gates", "OK")

    R.print()
    R.write(REPORT)
    print(f"\nFull report -> {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
