# R4 clinician responses: public subset

This release contains the supplied R4 responses for all **2,002 public probes
from 240 cases**. It preserves the original answer letters and reading-order
`seq` values. The mapping is taken from the author-supplied reader key; it was
not inferred by optimizing agreement with gold labels.

The public reference is pinned to
[MedVIGIL revision 5373ecdc8e20c467f1933b314b3a3164d2cba6d5](https://huggingface.co/datasets/jhq0709/MedVIGIL/tree/5373ecdc8e20c467f1933b314b3a3164d2cba6d5).
Every released question, option, reference image path, gold letter and risk tier
was checked against that reference. This directory contains metadata and
responses; obtain the public reference images from the dataset release.

## Files

| File | Contents |
| --- | --- |
| `answers.csv` | Original public-subset answer-sheet rows, including the original `seq`, question/options and response letter. |
| `seq_to_probe.csv` | One-to-one mapping from original `seq` to public `probe_id` and reference image variant. |
| `probes_mcq.csv` | Public reference questions, options, image paths and gold letters, in reading-key order. |
| `manifest.csv` | Case IDs and risk tiers for the 240 public cases. |
| `preparation_report.json` | Scope, validation counts and original input SHA-256 digests. |
| `release_validation.json` | Release checks and provenance limitations. |
| `score/identified_public_probes.csv` | Convenient per-probe table: `seq`, `probe_id`, case, family, tier, R4 response, gold and correctness. |
| `score/summary.json` | Exact counts and recomputed public-subset metrics. |
| `score/public_family_tier.csv` | Public-subset counts and rates by probe family and risk tier. |

The answer-sheet `image` column contains renamed reading-session image paths.
Those renamed files are not bundled. Use `reference_image_file` in
`seq_to_probe.csv` to locate the corresponding image in the public dataset;
bare filenames refer to its `images/` directory. `seq` has gaps because the
original full-benchmark order is preserved after exclusion of nonpublic rows.

## Reproduce the public-subset result

Run from the repository root with Python 3.10 or later; only the standard
library is required.

```sh
python analysis/score_r4.py --answers data/r4_public/answers.csv --probes data/r4_public/probes_mcq.csv --manifest data/r4_public/manifest.csv --seq-map data/r4_public/seq_to_probe.csv --require-complete --reference-scope public --output-dir r4_public_recomputed
```

The public-subset result is **1,837/2,002 correct (91.8%)**, Original
**229/240 (95.4%)**, trap SFR **30/480 (6.3%)**, weighted SFR **9.6%**, and
MCS **82.7**. Computation uses exact fractions; displayed percentages round
half up to one decimal, so 6.25% is shown as 6.3%.

## Relation to the full-benchmark paper result

The paper's full R4 result uses **2,556 probes from 300 cases**. Its exact
family/tier aggregate counts are released separately as
[`../clinician_baseline.csv`](../clinician_baseline.csv), with no per-case
records from the credentialed CXR subset. Recompute the full aggregate using:

```sh
python analysis/clinician_baseline_mcs.py
```

That aggregate gives Original **286/300 (95.3%)**, trap SFR **35/600 (5.8%)**,
ROI-only **175/192 (91.1%)**, ROI-masked **166/192 (86.5%)**, weighted SFR
**9.3%**, and MCS **83.3**. For `halluc_trap`, `n_correct` counts safe E
responses; `acc_or_sfr_pct` is the complementary silent failure rate.
Public-subset and full-benchmark scores use different populations.

The complete 60-case credentialed CXR subset, comprising 554 probes, is excluded
from the released per-probe answers, mapping, questions, gold labels and image
paths. Full-benchmark recomputation from aggregate counts does not constitute
public reproduction of its withheld per-probe source records.

## Provenance and license

The original supplied answer sheet, key and separate full gold table were
checked without editing them. Every full-benchmark aggregate row was checked
against a fresh direct join of these original records. No responses or gold
labels were generated or adjusted to match the paper.

`actual_clinician` is a supplied source label; it does not independently verify
reader identity, credentials, blinding or protocol. Image identity follows the
supplied key; image pixels were not independently compared. Nonpublic gold and
risk tiers have no separate public reference for independent verification.
No reader names, credentials, contact details or local absolute paths are
included in this release.

The repository [LICENSE](../../LICENSE) is the authoritative license statement.
Upstream data rights and access restrictions continue to apply.
