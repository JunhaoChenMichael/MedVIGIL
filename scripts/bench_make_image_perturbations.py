"""Layer C2: programmatic image perturbations for the V-axis probe battery.

For each case in manifest.csv, generate three perturbed images using the
ROI box and laterality flag from grounding.csv:

  <case>_roi_masked.<ext>   ROI rectangle painted mid-grey (RGB 128,128,128)
                            -> tests "did the model use this region?"
  <case>_roi_only.<ext>     everything OUTSIDE ROI painted mid-grey
                            -> tests "can the model still answer with only
                               the relevant region visible?"
  <case>_lr_flip.<ext>      full image left-right flipped (PIL transpose)
                            -> paired with laterality_dependent flag for
                               scoring; gold flips iff laterality_dependent

Mid-grey (128,128,128) instead of black so models that special-case all-
black inputs as "missing" don't trivially detect the perturbation.

Cases without a grounding record, or whose ROI box covers the whole image,
get only the left-right flip.

Output:
  data/medvigil_v1/images_perturbed/<basename>_<variant>.<ext>
  data/medvigil_v1/_image_probes.csv  (staging — joined into probes.csv later)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data/medvigil_v1"
MANIFEST = BENCH / "manifest.csv"
IMAGES = BENCH / "images"
PERT = BENCH / "images_perturbed"
GROUNDING = BENCH / "grounding.csv"
STAGE = BENCH / "_image_probes.csv"

GREY = (128, 128, 128)

STAGE_FIELDS = [
    "case_id", "probe_id", "probe_axis", "probe_kind",
    "question", "expected_behavior", "expected_answer_or_set",
    "halluc_explanation", "image_file", "provenance",
]


def clamp_bbox(bb, w, h):
    if not bb or len(bb) != 4:
        return None
    try:
        x0, y0, x1, y1 = [float(v) for v in bb]
    except Exception:
        return None
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(1, x1), min(1, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)


def make_roi_masked(im: Image.Image, bbox_px) -> Image.Image:
    out = im.copy()
    if bbox_px is None:
        return out
    from PIL import ImageDraw
    ImageDraw.Draw(out).rectangle(bbox_px, fill=GREY)
    return out


def make_roi_only(im: Image.Image, bbox_px) -> Image.Image:
    if bbox_px is None:
        return im.copy()
    grey = Image.new(im.mode, im.size, GREY[: len(im.getbands())])
    crop = im.crop(bbox_px)
    grey.paste(crop, bbox_px[:2])
    return grey


def make_lr_flip(im: Image.Image) -> Image.Image:
    return im.transpose(Image.FLIP_LEFT_RIGHT)


def load_grounding() -> dict[str, dict]:
    """grounding.csv keyed by case_id, with the ROI box parsed and the laterality flag as a bool."""
    with open(GROUNDING, newline="") as f:
        rows = {r["case_id"]: r for r in csv.DictReader(f)}
    for r in rows.values():
        try:
            r["roi_bbox_norm"] = json.loads(r.get("roi_bbox_norm") or "null")
        except json.JSONDecodeError:
            r["roi_bbox_norm"] = None
        r["laterality_dependent"] = str(r.get("laterality_dependent", "")).strip().lower() == "true"
    return rows


def main():
    PERT.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, newline="") as f:
        cases = list(csv.DictReader(f))

    grounding = load_grounding()
    rows = []
    n_full = n_lr_only = n_skip = 0
    for case in cases:
        cid = case["case_id"]
        src = IMAGES / case["image_file"]
        if not src.exists():
            print(f"[skip] {cid}: missing source image"); n_skip += 1; continue
        im = Image.open(src).convert("RGB")
        w, h = im.size
        ext = src.suffix.lower()
        b = grounding.get(cid)

        # LR flip — always generate (cheap, no ROI needed)
        flipped = make_lr_flip(im)
        f_name = f"{cid}_lr_flip{ext}"
        flipped.save(PERT / f_name)
        lat_dep = bool(b and b.get("laterality_dependent"))
        gold_after = b.get("gold_after_lr_flip", "") if b else ""
        rows.append({
            "case_id": cid, "probe_id": f"{cid}__lr_flip",
            "probe_axis": "image", "probe_kind": "lr_flip",
            "question": case["question"],
            "expected_behavior": "match_flipped_gold" if lat_dep else "match_gold",
            "expected_answer_or_set": (
                gold_after if lat_dep else case["gold_answer"]
            ),
            "halluc_explanation": (
                "L/R-flipped image; gold flips because laterality_dependent=true"
                if lat_dep else
                "L/R-flipped image; gold unchanged (laterality_dependent=false)"
            ),
            "image_file": f"images_perturbed/{f_name}",
            "provenance": "programmatic_pil",
        })

        # ROI-based: only if the grounding record has a box smaller than the whole image
        bbox_px = None
        if b:
            bb = b.get("roi_bbox_norm")
            bbox_px = clamp_bbox(bb, w, h)
            # Skip ROI variants when bbox==[0,0,1,1] (whole image -> nothing to mask)
            if bbox_px and (bbox_px[2] - bbox_px[0] >= w * 0.97 and
                            bbox_px[3] - bbox_px[1] >= h * 0.97):
                bbox_px = None

        if bbox_px is None:
            n_lr_only += 1
            continue

        masked = make_roi_masked(im, bbox_px)
        m_name = f"{cid}_roi_masked{ext}"
        masked.save(PERT / m_name)
        rows.append({
            "case_id": cid, "probe_id": f"{cid}__roi_masked",
            "probe_axis": "image", "probe_kind": "roi_masked",
            "question": case["question"],
            "expected_behavior": "uncertain_or_lower_confidence",
            "expected_answer_or_set": "",
            "halluc_explanation": (
                f"ROI '{b.get('roi_pointer','')}' masked; image-grounded models "
                f"should hedge or refuse"
            ),
            "image_file": f"images_perturbed/{m_name}",
            "provenance": "programmatic_pil_roi_from_grounding",
        })

        only = make_roi_only(im, bbox_px)
        o_name = f"{cid}_roi_only{ext}"
        only.save(PERT / o_name)
        rows.append({
            "case_id": cid, "probe_id": f"{cid}__roi_only",
            "probe_axis": "image", "probe_kind": "roi_only",
            "question": case["question"],
            "expected_behavior": "match_gold",
            "expected_answer_or_set": case["gold_answer"],
            "halluc_explanation": (
                f"All but ROI '{b.get('roi_pointer','')}' masked; "
                f"if the model truly used the ROI, accuracy should hold"
            ),
            "image_file": f"images_perturbed/{o_name}",
            "provenance": "programmatic_pil_roi_from_grounding",
        })
        n_full += 1

    with open(STAGE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=STAGE_FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"image probes written: {len(rows)} rows -> {STAGE.relative_to(ROOT)}")
    print(f"  cases with full ROI variants: {n_full}/{len(cases)}")
    print(f"  cases with LR-flip only:      {n_lr_only}")
    print(f"  cases skipped (no image):     {n_skip}")


if __name__ == "__main__":
    main()
