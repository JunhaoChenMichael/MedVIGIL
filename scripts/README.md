# `scripts/`

Every script runs from the repository root as `python scripts/<name>.py [--help]` and reads the dataset from `data/medvigil_v1/` (download it from <https://huggingface.co/datasets/jhq0709/MedVIGIL>). Results go to `results/`.

## Running models

| Script | Purpose |
|---|---|
| `bench_run_baseline.py` | One API model (OpenAI-compatible, Anthropic, Gemini, Together AI, DeepSeek text-only) on `probes_mcq.csv` or `probes_open.csv`. Async, resumable, retries with backoff on rate limits. `--no-image` runs the matched no-image control. |
| `bench_run_huatuo.py` | HuatuoGPT-Vision-7B locally, with `--shard i/n` for multi-GPU runs. Needs the official repository in `HuatuoGPT-Vision/` and weights in `checkpoints/huatuogpt-vision-7b/`. |
| `bench_run_llava_med.py` | LLaVA-Med v1.5 (Mistral-7B) locally through `inference.py`. |
| `bench_clean_errors.py` | Removes failed rows from a results file so the runner re-queries them. |

API keys come from the environment or a `.env` file at the repository root (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `TOGETHER_API_KEY`, `DEEPSEEK_API_KEY`). A probe whose image is not on disk is skipped, never sent without its image.

## Scoring

| Script | Purpose |
|---|---|
| `bench_aggregate.py` | Reads `results/baselines/<model>__mcq.jsonl` and writes per-model metrics by probe kind, risk tier and source, plus LaTeX tables. MCQ scoring is exact letter match against `correct_letter`; no judge model. |
| `bench_judge_open.py` | Scores the open-ended variant: regex rules for clear refusals and gold matches, an LLM judge (gpt-5.5) for ambiguous rows. |

## Data checks and image variants

| Script | Purpose |
|---|---|
| `bench_validate_full.py` | Integrity checks on the downloaded release: schema, images, grounding boxes, probe counts, triplet invariants, MCQ option rules. |
| `bench_make_image_perturbations.py` | Renders ROI-masked, ROI-only and left–right-flipped images from `grounding.csv`. Reproduces the released `images_perturbed/` pixel for pixel; credentialed users run it for the chest-radiograph cases. |
| `bench_visualize_roi.py` | Four-panel ROI overlays (original, box, masked, ROI-only) for inspection. |

## Ablations

| Script | Purpose |
|---|---|
| `bench_blur_sweep_make.py` | Gaussian-blurred copies of the original-probe images at sigma 0, 2, 4, 8, 16, 32 and 64, and one probe file per level. |
| `bench_blur_sweep_run.py`, `run_blur_sweep_one_model.sh` | One API model across all blur levels, plus the no-image level (`inf`). |
| `bench_blur_sweep_run_huatuo.py` | The blur sweep for HuatuoGPT-Vision-7B. |
| `bench_visual_token_ablation.py` | Progressive ROI masking (0, 33, 67 and 100% of the box) on a stratified case sample. |
