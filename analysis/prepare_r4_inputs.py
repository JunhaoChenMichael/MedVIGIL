"""Prepare explicitly scoped R4 scorer inputs from an author-supplied native key.

The originals are never edited. Public text, image paths, gold and risk tiers
are checked against an independent published release. Full scoring additionally
requires a complete author-supplied gold table. For nonpublic rows, question and
choices are copied from the supplied answer sheet, while probe identity, gold
and risk tier come from the separate key/gold tables. No missing labels or risk
tiers are inferred. This program does not publish or download any data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath

FIELDS = ("question", "choice_A", "choice_B", "choice_C", "choice_D", "choice_E")
SUFFIX_KIND = {"orig": "original", "tcf": "tcf", "negation": "negation", "specificity_drop": "specificity_drop", "knowledge_only": "knowledge_only", "trap1": "halluc_trap", "trap2": "halluc_trap", "lr_flip": "lr_flip", "roi_only": "roi_only", "roi_masked": "roi_masked"}

def require(condition, message):
    if not condition:
        raise ValueError(message)

def read(path, columns):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(rows and set(columns).issubset(rows[0]), f"Missing/empty required columns in {path}")
    return rows

def write(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def normalize_seqs(rows, label):
    require(all(r["seq"].isdigit() and int(r["seq"]) > 0 for r in rows), f"Invalid {label} seq")
    for row in rows:
        row["seq"] = str(int(row["seq"]))
    require(len({r["seq"] for r in rows}) == len(rows), f"Duplicate {label} seq")

def released_image(probe):
    image = probe["image_file"].replace("\\", "/")
    return "images/" + image if image and "/" not in image else image

def safe_image_path(path):
    return not path or ("\\" not in path and not PurePosixPath(path).is_absolute() and ":" not in path and ".." not in PurePosixPath(path).parts)

def prepare(answers, key, public_probes, public_manifest, scope, gold=None):
    normalize_seqs(answers, "answer")
    normalize_seqs(key, "mapping")
    by_seq = {r["seq"]: r for r in answers}
    require(set(by_seq) == {r["seq"] for r in key}, "Mapping must cover every answer seq exactly once")
    require(len({r["probe_id"] for r in key}) == len(key), "Duplicate mapped probe id")
    require(all(r["answer"] in "ABCDE" and len(r["answer"]) == 1 for r in answers), "Invalid answer letter")
    require(all(safe_image_path(r[c]) for r in key for c in ("source_image", "r4_image")), "Unsafe or nonportable image path")
    require(all(r["r4_image"] == by_seq[r["seq"]]["image"] for r in key), "Mapping r4_image differs from answer-sheet image")
    public = {r["probe_id"]: r for r in public_probes}
    manifest = {r["case_id"]: r for r in public_manifest}
    require(len(public) == len(public_probes) and len(manifest) == len(public_manifest), "Duplicate public reference id")
    require(set(public).issubset({r["probe_id"] for r in key}), "Key lacks released public probes")
    for row in key:
        parts = row["probe_id"].split("__")
        require(len(parts) == 2 and parts[1] in SUFFIX_KIND, "Unknown probe id suffix")
        if row["probe_id"] in public:
            pr, ar = public[row["probe_id"]], by_seq[row["seq"]]
            require(pr["case_id"] in manifest, "Public probe lacks manifest")
            require(all(ar[k] == pr[k] for k in FIELDS), "Public question/options mismatch")
            require(row["source_image"] == released_image(pr), "Public source_image differs from released image variant")

    gold_by_id = {}
    if gold is not None:
        gold_by_id = {r["probe_id"]: r for r in gold}
        require(len(gold_by_id) == len(gold), "Duplicate gold probe id")
        require(set(gold_by_id) == {r["probe_id"] for r in key}, "Gold must cover every key probe exactly once")
        case_tiers = {}
        for row in gold:
            case_id, suffix = row["probe_id"].split("__")
            require(row["case_id"] == case_id and row["probe_kind"] == SUFFIX_KIND[suffix], "Gold probe case/kind conflicts with identity")
            require(row["risk_tier"] in ("L1", "L2", "L3", "L4", "L5"), "Invalid gold risk tier")
            require(row["correct_letter"] in "ABCDE" and len(row["correct_letter"]) == 1, "Invalid gold answer letter")
            require(case_id not in case_tiers or case_tiers[case_id] == row["risk_tier"], "Conflicting case risk tiers")
            case_tiers[case_id] = row["risk_tier"]
            if row["probe_id"] in public:
                pr = public[row["probe_id"]]
                require(all(row[k] == pr[k] for k in ("case_id", "probe_kind", "correct_letter")), "Supplied gold differs from released public gold")
                require(row["risk_tier"] == manifest[pr["case_id"]]["risk_tier"], "Supplied risk tier differs from public manifest")
    require(scope == "public" or gold_by_id, "Full scope requires a separate complete gold table")
    selected = [r for r in key if scope == "full" or r["probe_id"] in public]
    output_answers, output_probes, output_mapping, output_manifest = [], [], [], {}
    for row in selected:
        pid, ar = row["probe_id"], by_seq[row["seq"]]
        if pid in public:
            pr = {k: public[pid][k] for k in ("probe_id", "case_id", "probe_kind", "correct_letter", "image_file", *FIELDS)}
            mr = manifest[pr["case_id"]]
        else:
            gr = gold_by_id[pid]
            pr = {**{k: gr[k] for k in ("probe_id", "case_id", "probe_kind", "correct_letter")}, "image_file": row["source_image"], **{k: ar[k] for k in FIELDS}}
            mr = gr
        output_answers.append(ar.copy())
        output_probes.append(pr)
        output_mapping.append({"seq": row["seq"], "probe_id": pid, "image": row["r4_image"], "reference_image_file": pr["image_file"]})
        output_manifest[pr["case_id"]] = {"case_id": pr["case_id"], "risk_tier": mr["risk_tier"]}
    report = {"reference_scope": scope, "n_source_answers": len(answers), "n_source_key_rows": len(key), "n_selected_answers": len(selected), "n_excluded_answers": len(answers)-len(selected), "n_reference_cases": len(output_manifest), "n_public_probes_validated": len(public), "n_public_cases_validated": len(manifest), "seq_and_image_bijection": True, "exact_public_text_choice_and_source_image_validation": True, "separate_full_gold_validated": bool(gold_by_id), "private_reference_text_origin": "copied from supplied answer sheet; identity/gold/tier from separate key/gold" if scope == "full" and len(selected) > len(public) else None, "risk_tier_case_counts": dict(Counter(r["risk_tier"] for r in output_manifest.values())), "limitation": "Image associations are declared by the supplied key; no pixel comparison is performed."}
    report["n_nonpublic_reference_probes"] = sum(r["probe_id"] not in public for r in selected)
    report["gold_provenance_limit"] = "Nonpublic labels/tiers come from the separate supplied local gold table and pass internal consistency checks; they are not independently checked against a separate published reference." if report["n_nonpublic_reference_probes"] else None
    return output_answers, output_probes, output_mapping, list(output_manifest.values()), report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--seq-key", type=Path, required=True)
    parser.add_argument("--public-probes", type=Path, required=True)
    parser.add_argument("--public-manifest", type=Path, required=True)
    parser.add_argument("--scope", choices=("public", "full"), required=True, help="Explicit reference population; public excludes unmapped-to-release rows")
    parser.add_argument("--gold", type=Path, help="Separate author gold table; required for full scope")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    answers = read(args.answers, ("seq", "image", "answer", *FIELDS))
    key = read(args.seq_key, ("seq", "probe_id", "source_image", "r4_image"))
    probes = read(args.public_probes, ("probe_id", "case_id", "probe_kind", "correct_letter", "image_file", *FIELDS))
    manifest = read(args.public_manifest, ("case_id", "risk_tier"))
    gold = read(args.gold, ("probe_id", "case_id", "probe_kind", "risk_tier", "correct_letter")) if args.gold else None
    a, p, m, c, report = prepare(answers, key, probes, manifest, args.scope, gold)
    paths = {"answers": args.answers, "seq_key": args.seq_key, "public_probes": args.public_probes, "public_manifest": args.public_manifest}
    if args.gold:
        paths["gold"] = args.gold
    report["input_sha256"] = {k: hashlib.sha256(path.read_bytes()).hexdigest() for k,path in paths.items()}
    require(all(args.output_dir.resolve() != path.resolve().parent for path in paths.values()), "Keep derived inputs in a separate directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write(args.output_dir/"answers.csv", a, list(a[0]))
    write(args.output_dir/"probes_mcq.csv", p, ("probe_id", "case_id", "probe_kind", "correct_letter", "image_file", *FIELDS))
    write(args.output_dir/"seq_to_probe.csv", m, ("seq", "probe_id", "image", "reference_image_file"))
    write(args.output_dir/"manifest.csv", c, ("case_id", "risk_tier"))
    (args.output_dir/"preparation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
