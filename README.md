# PvP Bot Training Scaffold

This repository now has a starter layout for training a PvP bot AI model.

## Structure

- `src/bot_training/` - reusable Python package code
  - `data/` - data loading and preprocessing helpers
  - `features/` - feature engineering helpers
  - `training/` - training entry points and logic
  - `evaluation/` - validation and metrics code
  - `inference/` - prediction helpers
- `data/raw/` - original match datasets
- `data/interim/` - intermediate files during cleaning
- `data/processed/` - cleaned and transformed training data
- `data/splits/` - train/validation/test splits
- `models/checkpoints/` - saved checkpoints
- `models/exports/` - exported model artifacts
- `reports/metrics/` - evaluation outputs and summaries
- `reports/figures/` - charts and plots
- `scripts/` - one-off runnable utilities
- `tests/` - smoke tests for the scaffold

## Quick checks

Run the smoke test:

```bash
python -m unittest
```

Run pytest suites (including feature engineering tests):

```bash
python -m pytest
```

Prepare raw data inventory:

```bash
python scripts/prepare_data.py
```

## Phase 1 data cleaning

This project now includes a chunked pandas pipeline that groups rows into matches, filters them by quality, and writes a clean sequential CSV.

Note: many PvP logs alternate `playerName` every tick. Because of that, player-change splitting is **off by default** so rows are not split into one-frame matches. You can enable strict player boundary splitting with `--split-on-player-change`.

Per-file output example (one clean CSV per input file):

```bash
python3 scripts/prepare_data.py \
  --input-dir data/raw \
  --output-mode per-file \
  --output-dir data/processed/phase1_clean_matches_per_file \
  --progress \
  --min-frames 400 \
  --max-damage-taken 60 \
  --min-attack-accuracy 0.20 \
  --min-sprint-uptime 0.15
```

## Automatic threshold sweep

Use the sweep utility to evaluate a grid of threshold combinations and rank them by keep-rate/quality tradeoff.

```bash
python3 scripts/sweep_thresholds.py \
  --input-dir data/raw \
  --sample-fraction 0.1 \
  --sample-seed 42 \
  --min-frames-grid 400,700,1000 \
  --max-damage-grid 40,50,60 \
  --min-attack-accuracy-grid 0.20,0.30,0.40 \
  --min-sprint-uptime-grid 0.15,0.30,0.60 \
  --score-weights 0.25,0.25,0.25,0.25 \
  --top-k 10
```

The command writes a ranked report to `reports/metrics/phase1_threshold_sweep.csv`.
Use `--sample-fraction 0.1` to run on roughly 1/10 of files for faster iteration.

## Phase 2 feature engineering

Convert cleaned phase 1 rows into normalized frame tensors and sequence windows. Phase 2 now writes one NPZ per cleaned match CSV under `data/processed/phase2_feature_tensors_per_file/`:

Continuous features use fixed Minecraft-aware scaling (no fitted scaler artifact):
- `health`, `targetHealth` / 20
- `yaw`, `targetYaw` / 180
- `pitch`, `targetPitch` / 90
- spatial terms / 50 (upper-clipped to `1.0`)
- velocity terms / 4 (upper-clipped to `1.0`)

```bash
python scripts/build_features.py \
  --input-file data/processed/phase1_clean_matches_per_file/example_ai_clean.csv \
  --output-file data/processed/phase2_feature_tensors.npz \
  --vocabulary-file models/exports/phase2_item_vocabulary.json
```

Run the full batch pipeline over every per-file clean CSV:

```bash
python3 scripts/build_features.py \
  --input-dir data/processed/phase1_clean_matches_per_file \
  --input-pattern "*_clean.csv" \
  --output-dir data/processed/phase2_feature_tensors_per_file \
  --manifest-file data/processed/phase2_feature_manifest.json \
  --vocabulary-file models/exports/phase2_item_vocabulary.json \
  --window-size 20
```

Optional: add `--max-files 100` for a quick subset dry run.

Saved NPZ fields:

- `inputs`: normalized frame-level input matrix
- `targets`: frame-level action + slot + deltaYaw/deltaPitch targets
- `input_windows`: overlapping windows with shape `[num_windows, 20, feature_count]`
- `sequence_targets`: target rows aligned to the end of each input window
- `window_match_ids`: match IDs aligned to each input window

## Phase 4 training

Train on a real Phase 2 artifact with the MLX sequence model:

```bash
python3 scripts/train_model.py \
  --dataset data/processed/phase2_feature_tensors_per_file \
  --checkpoint models/checkpoints/phase4_best_weights.npz \
  --epochs 50 \
  --batch-size 256 \
  --learning-rate 0.001
```

The trainer splits windows by `match_id`, uses an 80/20 train/validation split, and saves the best checkpoint only when validation loss improves.

## Phase 4 scenario tests

Run scenario-based checks against a trained checkpoint (dual input: continuous windows + mock inventory windows):

```bash
python3 scripts/assert_phase4_scenarios.py \
  --checkpoint models/checkpoints/phase4_best_weights.npz \
  --item-vocab models/exports/phase2_item_vocabulary.json \
  --allow-failures
```

Useful options:

- `--allow-failures`: print all scenario results and exit with code `0` even if checks fail.
- `--high-prob`, `--drop-prob`, `--rise-prob`, `--very-large-positive-pitch`: tune assertion thresholds.
- `--drink-slot`, `--splash-slot`, `--food-slot`, `--golden-apple-slot`: override expected hotbar slot indices.

Quick run that never fails CI locally:

```bash
python3 scripts/assert_phase4_scenarios.py --allow-failures
```

Output format:

- Each scenario prints one line like `[PASS] Step X - ...` or `[FAIL] Step X - ...`.
- The script ends with `Completed <N> scenario checks with <M> failure(s).`.
- Without `--allow-failures`, any failed scenario raises an assertion and returns a non-zero exit code.

