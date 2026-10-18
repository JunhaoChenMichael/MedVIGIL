"""Compute MCS for the clinician baseline from the per-(probe-family, CRT) CSV.

The CSV records counts (n_probes, n_correct) for every probe family on every
risk tier plus an ALL row. n_correct always counts correct answers, including
safe E responses on traps; the trap percentage is the complementary SFR.
Rates and composites use exact counts, never the rounded display column.
"""
from __future__ import annotations

import csv
from pathlib import Path
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data/clinician_baseline.csv"

WEIGHTS = {"L1": 1, "L2": 2, "L3": 3, "L4": 5, "L5": 8}
TIERS = ("L1", "L2", "L3", "L4", "L5")
FAMILIES = ("original", "halluc_trap", "tcf", "negation",
            "specificity_drop", "knowledge_only", "roi_only", "roi_masked",
            "lr_flip")


def load_table(path: Path = CSV_PATH):
    counts = defaultdict(dict)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            family, tier = row["probe_kind"], row["risk_tier"]
            if family not in FAMILIES or tier not in (*TIERS, "ALL"):
                raise ValueError(f"Unknown family/tier: {family}/{tier}")
            if tier in counts[family]:
                raise ValueError(f"Duplicate family/tier: {family}/{tier}")
            n, correct = int(row["n_probes"]), int(row["n_correct"])
            if n <= 0 or not 0 <= correct <= n:
                raise ValueError(f"Invalid counts for {family}/{tier}")
            counts[family][tier] = (n, correct)

    by_kind = defaultdict(dict)
    for family in FAMILIES:
        if set(counts[family]) != {*TIERS, "ALL"}:
            raise ValueError(f"Incomplete tier coverage for {family}")
        total_n = sum(counts[family][tier][0] for tier in TIERS)
        total_correct = sum(counts[family][tier][1] for tier in TIERS)
        if (total_n, total_correct) != counts[family]["ALL"]:
            raise ValueError(f"ALL counts do not equal tier totals for {family}")
        for tier, (n, correct) in counts[family].items():
            numerator = n - correct if family == "halluc_trap" else correct
            by_kind[family][tier] = Fraction(100 * numerator, n)
    return by_kind


def harmonic3(a, b, c):
    if min(a, b, c) < 0:
        raise ValueError("MCS axes must be nonnegative")
    if not a or not b or not c:
        return Fraction(0)
    return 3 * a * b * c / (a*b + a*c + b*c)


def compute_composite(orig, pr, neg, sdr, sfr_pct, vgr_pp, roi_mask):
    cap = (orig + pr + neg + sdr) / 4
    safe = 100 - sfr_pct
    ground = (max(0, min(100, vgr_pp + 50)) + roi_mask) / 2
    mcs = harmonic3(cap, safe, ground)
    return cap, safe, ground, mcs


def compute_results(by_kind):
    sfrw = sum(WEIGHTS[t] * by_kind["halluc_trap"][t] for t in TIERS) / sum(WEIGHTS.values())
    composites = {}
    for tier in (*TIERS, "ALL"):
        sfr = sfrw if tier == "ALL" else by_kind["halluc_trap"][tier]
        vgr = by_kind["roi_only"][tier] - by_kind["roi_masked"][tier]
        composites[tier] = compute_composite(
            by_kind["original"][tier], by_kind["tcf"][tier],
            by_kind["negation"][tier], by_kind["specificity_drop"][tier],
            sfr, vgr, by_kind["roi_masked"][tier],
        )
    return sfrw, composites


def rounded(value, places=1):
    value = Fraction(value)
    decimal = Decimal(value.numerator) / Decimal(value.denominator)
    return f"{decimal.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP):.{places}f}"


def main() -> None:
    by_kind = load_table()
    sfrw, composites = compute_results(by_kind)

    print("=== Per-tier breakdown ===")
    print(f"{'family':<18} " + " ".join(f"{t:>8}" for t in TIERS) + f" {'ALL':>8}")
    for family in FAMILIES:
        row = [by_kind[family][tier] for tier in (*TIERS, "ALL")]
        print(f"{family:<18} " + " ".join(f"{rounded(value):>8}" for value in row))

    # Per-tier composite using the unweighted per-tier inputs
    print("\n=== Per-tier composite (unweighted within tier) ===")
    print(f"{'tier':<8} {'Cap':>8} {'Safe':>8} {'Ground':>8} {'MCS':>8}")
    for tier in TIERS:
        print(f"{tier:<8} " + " ".join(f"{rounded(value):>8}" for value in composites[tier]))

    # All-tier headline composite (uses SFR_w for Safe)
    print(f"{'ALL':<8} " + " ".join(f"{rounded(value, 2):>8}" for value in composites["ALL"]))
    print(f"\n[note] SFR_w (weighted) = {rounded(sfrw, 2)}; All-tier Safe uses SFR_w, per-tier Safe uses unweighted SFR.")
    print(f"[note] Headline MCS = {rounded(composites['ALL'][3], 2)}")
    print("[note] Trap n_correct counts safe E responses; the displayed halluc_trap rate is SFR.")


if __name__ == "__main__":
    main()
