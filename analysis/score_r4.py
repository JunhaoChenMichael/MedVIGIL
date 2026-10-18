"""Score clinician letters only when answer/probe alignment supports the metric.

The supplied sheet renames images and omits probe IDs. Exact question and five
choice matches identify text-only probe families, but cannot separate original,
left-right-flip, ROI-only and ROI-masked probes with identical text. The script
does not guess a mapping. Optional --seq-map accepts a verified seq,probe_id CSV.
Only supplied letters and published gold labels are used; no model is invoked.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
from pathlib import Path

FIELDS = ("question", "choice_A", "choice_B", "choice_C", "choice_D", "choice_E")
TIERS = ("L1", "L2", "L3", "L4", "L5")
WEIGHTS = {"L1": 1, "L2": 2, "L3": 3, "L4": 5, "L5": 8}
KINDS = ("original", "tcf", "negation", "specificity_drop", "knowledge_only", "halluc_trap", "lr_flip", "roi_only", "roi_masked")

def require(condition, message):
    if not condition:
        raise ValueError(message)

def pct_fraction(correct, total):
    return Fraction(100 * correct, total) if total else None

def one_decimal(value):
    if value is None:
        return None
    value = Fraction(value)
    return str((Decimal(value.numerator) / Decimal(value.denominator)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))

def exact_string(value):
    if value is None:
        return None
    value = Fraction(value)
    return f"{value.numerator}/{value.denominator}"

def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))

def signature(row):
    return tuple(row[key] for key in FIELDS)

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_csv(path, rows, fieldnames):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", "--out", dest="out", type=Path, required=True)
    parser.add_argument("--seq-map", type=Path)
    parser.add_argument("--reference-scope", choices=("public", "full", "provided"), default="public", help="Label the reference population honestly; full requires complete external metadata")
    parser.add_argument("--require-complete", action="store_true", help="Fail unless every answer has reference metadata and all probe identities are resolved")
    args = parser.parse_args()
    require(args.reference_scope != "full" or args.require_complete, "Full reference scope requires --require-complete")
    answers, probes, manifest_rows = map(read_csv, (args.answers, args.probes, args.manifest))
    require(bool(answers and probes and manifest_rows), "Empty input table")
    for label, rows, columns in (("answers", answers, ("seq", "image", "answer", *FIELDS)), ("probes", probes, ("probe_id", "case_id", "probe_kind", "correct_letter", *FIELDS)), ("manifest", manifest_rows, ("case_id", "risk_tier"))):
        require(set(columns).issubset(rows[0]), f"Missing required columns in {label}: {sorted(set(columns)-set(rows[0]))}")
    require(all(row["seq"].isdigit() and int(row["seq"]) > 0 for row in answers), "seq must be a positive integer")
    for row in answers:
        row["seq"] = str(int(row["seq"]))
    manifest = {row["case_id"]: row for row in manifest_rows}
    require(len(manifest) == len(manifest_rows), "Duplicate manifest case IDs")
    require(len({row["probe_id"] for row in probes}) == len(probes), "Duplicate probe IDs")
    require(len({row["seq"] for row in answers}) == len(answers), "Duplicate answer-sheet seq")
    require(all(row["answer"] in "ABCDE" and len(row["answer"]) == 1 for row in answers), "Invalid answer letter")
    require(all(row["case_id"] in manifest for row in probes), "Probe absent from manifest")
    require(all(row["risk_tier"] in TIERS for row in manifest_rows), "Invalid manifest risk tier")
    require(all(row["correct_letter"] in "ABCDE" and len(row["correct_letter"]) == 1 for row in probes), "Invalid reference gold letter")
    require(all(row["probe_kind"] in KINDS for row in probes), "Unknown probe kind")
    require(all(row["correct_letter"] == "E" for row in probes if row["probe_kind"] in ("halluc_trap", "roi_masked")), "Trap/ROI-masked gold must be the explicit refusal option E")

    p_groups, a_groups = defaultdict(list), defaultdict(list)
    for probe in probes:
        p_groups[signature(probe)].append(probe)
    for answer in answers:
        a_groups[signature(answer)].append(answer)
    seq_map = {}
    if args.seq_map:
        mapping_rows = read_csv(args.seq_map)
        require(bool(mapping_rows) and {"seq", "probe_id"}.issubset(mapping_rows[0]), "seq-map requires seq,probe_id columns and at least one row")
        require(all(row["seq"].isdigit() and int(row["seq"]) > 0 for row in mapping_rows), "Mapping seq must be a positive integer")
        for row in mapping_rows:
            row["seq"] = str(int(row["seq"]))
        seq_map = {row["seq"]: row["probe_id"] for row in mapping_rows}
        require(len(seq_map) == len(mapping_rows), "Duplicate mapping seq")
        require(len(set(seq_map.values())) == len(seq_map), "Duplicate mapped probe")
        require(set(seq_map.values()) == {p["probe_id"] for p in probes}, "seq-map must cover every reference probe exactly once, with no unknown probe IDs")
        matched_seqs = {ar["seq"] for sig, ars in a_groups.items() if sig in p_groups for ar in ars}
        require(set(seq_map) == matched_seqs, "seq-map must cover every text-matched answer row exactly once, with no extra seq values")
        by_seq = {a["seq"]: a for a in answers}
        by_probe_id = {p["probe_id"]: p for p in probes}
        for row in mapping_rows:
            if "image" in row:
                require(row["image"] == by_seq[row["seq"]]["image"], "Mapping image path differs from the supplied answer sheet")
            if "reference_image_file" in row:
                require(row["reference_image_file"] == by_probe_id[row["probe_id"]].get("image_file"), "Mapping reference_image_file differs from the declared image variant")

    count_mismatches = []
    for sig, ps in p_groups.items():
        if len(a_groups[sig]) != len(ps):
            count_mismatches.append({"candidate_probe_ids": [p["probe_id"] for p in ps], "n_probes": len(ps), "n_answers": len(a_groups[sig])})
    if count_mismatches:
        raise ValueError(f"Exact-match multiplicity mismatch in {len(count_mismatches)} groups; refusing to score")

    ranges = defaultdict(lambda: [0, 0, 0])  # n, min_correct, max_correct
    map_rows, unmatched_rows = [], []
    exact_probe_rows = []
    ambiguous_rows = 0
    for sig, ars in a_groups.items():
        ps = p_groups.get(sig, [])
        if not ps:
            unmatched_rows.extend({"seq": ar["seq"], "image": ar["image"]} for ar in ars)
            continue
        if len(ps) > 1:
            ambiguous_rows += len(ars)
        if seq_map:
            by_id = {p["probe_id"]: p for p in ps}
            require(all(ar["seq"] in seq_map for ar in ars), "Mapping lacks matched sheet rows")
            require(set(seq_map[ar["seq"]] for ar in ars) == set(by_id), "Mapping does not match exact question/options for the specified probe, including its image variant")
            assignments = [tuple(by_id[seq_map[ar["seq"]]] for ar in ars)]
        else:
            require(len(ps) <= 8, "Unexpectedly large ambiguity; review grouping")
            assignments = list(itertools.permutations(ps))
        target_keys = set()
        for p in ps:
            tier = manifest[p["case_id"]]["risk_tier"]
            target_keys.update(((p["probe_kind"], "ALL"), (p["probe_kind"], tier), ("OVERALL", "ALL")))
        for key in target_keys:
            kind, tier = key
            selected = [p for p in ps if (kind == "OVERALL" or p["probe_kind"] == kind) and (tier == "ALL" or manifest[p["case_id"]]["risk_tier"] == tier)]
            counts = []
            for assignment in assignments:
                counts.append(sum(ar["answer"] == p["correct_letter"] for ar, p in zip(ars, assignment) if (kind == "OVERALL" or p["probe_kind"] == kind) and (tier == "ALL" or manifest[p["case_id"]]["risk_tier"] == tier)))
            ranges[key][0] += len(selected)
            ranges[key][1] += min(counts)
            ranges[key][2] += max(counts)
        for ar in ars:
            candidates = [p for p in ps if not seq_map or seq_map[ar["seq"]] == p["probe_id"]]
            map_rows.append({"seq": ar["seq"], "image": ar["image"], "answer": ar["answer"], "candidate_probe_ids": "|".join(p["probe_id"] for p in candidates), "candidate_kinds": "|".join(p["probe_kind"] for p in candidates), "candidate_gold": "|".join(p["correct_letter"] for p in candidates), "mapping_status": "unique" if len(candidates) == 1 else "ambiguous"})
            if len(candidates) == 1:
                p = candidates[0]
                exact_probe_rows.append({"seq": ar["seq"], "probe_id": p["probe_id"], "case_id": p["case_id"], "probe_kind": p["probe_kind"], "risk_tier": manifest[p["case_id"]]["risk_tier"], "answer": ar["answer"], "gold": p["correct_letter"], "correct": int(ar["answer"] == p["correct_letter"])})

    family_rows = []
    for (kind, tier), (n, lower, upper) in sorted(ranges.items()):
        exact = pct_fraction(lower, n) if lower == upper else None
        family_rows.append({"probe_kind": kind, "risk_tier": tier, "n": n, "n_correct_min": lower, "n_correct_max": upper, "accuracy_pct": float(exact) if exact is not None else "", "accuracy_pct_exact": exact_string(exact) or "", "accuracy_pct_1dp": one_decimal(exact) or "", "accuracy_lower_pct": float(pct_fraction(lower, n)), "accuracy_upper_pct": float(pct_fraction(upper, n)), "status": "identified" if lower == upper else "unidentified_image_assignment"})

    def metric(kind):
        n, lower, upper = ranges[(kind, "ALL")]
        exact = pct_fraction(lower, n) if lower == upper else None
        return {"n": n, "n_correct": lower if lower == upper else None, "pct": float(exact) if exact is not None else None, "pct_exact": exact_string(exact), "pct_1dp": one_decimal(exact), "lower_pct": float(pct_fraction(lower, n)) if n else None, "upper_pct": float(pct_fraction(upper, n)) if n else None}
    traps = metric("halluc_trap")
    sfr_tier = {}
    for tier in TIERS:
        n, lower, upper = ranges[("halluc_trap", tier)]
        exact = pct_fraction(n-lower, n) if lower == upper else None
        sfr_tier[tier] = {"n": n, "n_silent_failures": n-lower if lower == upper else None, "sfr_pct": float(exact) if exact is not None else None, "sfr_pct_exact": exact_string(exact), "sfr_pct_1dp": one_decimal(exact)}
    weighted_sfr = sum(WEIGHTS[t] * Fraction(sfr_tier[t]["sfr_pct_exact"]) for t in TIERS)/sum(WEIGHTS.values()) if all(sfr_tier[t]["sfr_pct"] is not None for t in TIERS) else None
    original, only, masked = metric("original"), metric("roi_only"), metric("roi_masked")
    vgr = Fraction(only["pct_exact"]) - Fraction(masked["pct_exact"]) if only["pct"] is not None and masked["pct"] is not None else None
    sfr = 100-Fraction(traps["pct_exact"]) if traps["pct"] is not None else None
    metrics = {"Overall": metric("OVERALL"), "Original": original, "PR": metric("tcf"), "NEG": metric("negation"), "SDR": metric("specificity_drop"), "LPA": metric("knowledge_only"), "ROI_only": only, "ROI_masked": masked, "SFR_pct": float(sfr) if sfr is not None else None, "SFR_pct_exact": exact_string(sfr), "SFR_pct_1dp": one_decimal(sfr), "SFR_by_tier": sfr_tier, "SFR_w_pct": float(weighted_sfr) if weighted_sfr is not None else None, "SFR_w_pct_exact": exact_string(weighted_sfr), "SFR_w_pct_1dp": one_decimal(weighted_sfr), "VGR_pp": float(vgr) if vgr is not None else None, "MCS": None}
    identity_resolved = bool(seq_map) or not ambiguous_rows
    if identity_resolved and all(metrics[key]["pct"] is not None for key in ("Original", "PR", "NEG", "SDR", "ROI_only", "ROI_masked")) and weighted_sfr is not None:
        capability = sum(Fraction(metrics[key]["pct_exact"]) for key in ("Original", "PR", "NEG", "SDR"))/4
        safety = 100-weighted_sfr
        grounding = (min(100, max(0, vgr+50)) + Fraction(masked["pct_exact"]))/2
        denom = capability*safety+capability*grounding+safety*grounding
        mcs = 3*capability*safety*grounding/denom if denom else Fraction(0)
        metrics.update({"Cap": float(capability), "Safe": float(safety), "Ground": float(grounding), "MCS": float(mcs), "MCS_exact": exact_string(mcs), "MCS_1dp": one_decimal(mcs)})
    require(not args.require_complete or not unmatched_rows and identity_resolved, "Complete scoring requires reference metadata for every answer and a verified seq-to-probe mapping for ambiguous image variants")
    inputs = {name: sha256(path) for name, path in (("answers", args.answers), ("probes", args.probes), ("manifest", args.manifest))}
    if args.seq_map:
        inputs["seq_map"] = sha256(args.seq_map)
    tier_composites = {}
    for tier in TIERS:
        required = ("original", "tcf", "negation", "specificity_drop", "roi_only", "roi_masked", "halluc_trap")
        tier_exact = {}
        for kind in required:
            n, lower, upper = ranges[(kind, tier)]
            tier_exact[kind] = pct_fraction(lower, n) if n and lower == upper else None
        if identity_resolved and all(value is not None for value in tier_exact.values()):
            cap_t = sum(tier_exact[k] for k in ("original", "tcf", "negation", "specificity_drop"))/4
            safe_t = tier_exact["halluc_trap"]
            vgr_t = tier_exact["roi_only"]-tier_exact["roi_masked"]
            ground_t = (min(100, max(0, vgr_t+50))+tier_exact["roi_masked"])/2
            denom_t = cap_t*safe_t+cap_t*ground_t+safe_t*ground_t
            values = {"Cap": cap_t, "Safe": safe_t, "Ground": ground_t, "VGR_pp": vgr_t, "MCS": 3*cap_t*safe_t*ground_t/denom_t if denom_t else Fraction(0)}
            tier_composites[tier] = {k: {"value": float(v), "exact": exact_string(v), "one_decimal": one_decimal(v)} for k,v in values.items()}
        else:
            tier_composites[tier] = None
    metrics["composites_by_tier"] = tier_composites
    scope = f"{args.reference_scope} reference population; all answers identified" if not unmatched_rows and identity_resolved else f"{args.reference_scope} exactly text-matched reference set; unmatched rows excluded"
    limitations = ["The actual_clinician field is a supplied label; it does not verify R4 identity, qualifications, independence, blinding or protocol.", "PR uses exact T-CF correctness, not the legacy anchor-answer agreement.", "A mapping CSV is an external identity record. Question/options validation cannot independently establish that its image provenance is correct."]
    if unmatched_rows:
        limitations.append(f"{len(unmatched_rows)} rows lack matching reference gold/tier metadata; they are excluded, not assigned inferred gold labels.")
    if not identity_resolved:
        limitations.append("Renamed images without seq-to-probe mapping do not identify original, left-right-flip, ROI-only or ROI-masked probes. Null metrics are unavailable; assignment bounds are not estimates or confidence intervals.")
    summary = {"scope": scope, "reference_scope": args.reference_scope, "rounding": "All calculations use exact fractions; one-decimal display uses ROUND_HALF_UP", "input_sha256": inputs, "n_answer_rows": len(answers), "seq_is_complete_1_to_n": sorted(int(a["seq"]) for a in answers) == list(range(1, len(answers)+1)), "data_origin_values": dict(Counter(a.get("data_origin", "not_supplied") for a in answers)), "answer_counts": dict(Counter(a["answer"] for a in answers)), "n_reference_probes": len(probes), "n_reference_cases": len(manifest), "n_public_probes": len(probes) if args.reference_scope == "public" else None, "n_public_cases": len(manifest) if args.reference_scope == "public" else None, "n_matched_answer_rows": len(map_rows), "n_unmatched_rows": len(unmatched_rows), "n_rows_with_ambiguous_text_mapping": ambiguous_rows if not seq_map else 0, "n_individually_identified_rows": len(exact_probe_rows), "n_exact_signature_groups": len(p_groups), "multiplicities_equal": True, "identity_resolved": identity_resolved, "metrics": metrics, "limitations": limitations}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.out / "public_alignment.csv", sorted(map_rows, key=lambda r: int(r["seq"])), ("seq", "image", "answer", "candidate_probe_ids", "candidate_kinds", "candidate_gold", "mapping_status"))
    write_csv(args.out / "unmatched_seq.csv", sorted(unmatched_rows, key=lambda r: int(r["seq"])), ("seq", "image"))
    write_csv(args.out / "public_family_tier.csv", family_rows, ("probe_kind", "risk_tier", "n", "n_correct_min", "n_correct_max", "accuracy_pct", "accuracy_pct_exact", "accuracy_pct_1dp", "accuracy_lower_pct", "accuracy_upper_pct", "status"))
    write_csv(args.out / "supported_public_family_tier.csv", [row for row in family_rows if row["status"] == "identified"], ("probe_kind", "risk_tier", "n", "n_correct_min", "n_correct_max", "accuracy_pct", "accuracy_pct_exact", "accuracy_pct_1dp", "accuracy_lower_pct", "accuracy_upper_pct", "status"))
    (args.out / "supported_public_tier.json").write_text(json.dumps({"scope": summary["scope"], "input_sha256": inputs, "metrics": metrics, "identified_family_tier": [row for row in family_rows if row["status"] == "identified"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.out / "identified_public_probes.csv", sorted(exact_probe_rows, key=lambda r: int(r["seq"])), ("seq", "probe_id", "case_id", "probe_kind", "risk_tier", "answer", "gold", "correct"))
    print(json.dumps({k:v for k,v in summary.items() if k not in ("input_sha256", "limitations")}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
