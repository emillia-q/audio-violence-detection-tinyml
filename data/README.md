# Data Management & Preprocessing

This directory contains the dataset for audio classification, split into raw recordings and segments ready for training.

## Layout
- `data/raw/` - long source recordings, organized into category folders: `ambient/`, `speech/`, `violence/`. Each category may contain arbitrary nested subfolders grouping clips by source/topic (see per-category READMEs).
- `data/processed/` - fixed-length segments produced by `scripts/preprocess.py`, organized as `{index}_{category}/`:
  - `0_ambient/`
  - `1_speech/`
  - `2_violence/`

  The numeric prefix is assigned in the order categories are iterated under `data/raw/` and is used directly as the class label by the training pipeline.

## Data Flow
1. **Raw Data** (`data/raw/<category>/...`): place long recordings under one of the category folders. Nested subfolders are walked recursively.
2. **Preprocessing**: run `scripts/preprocess.py` to convert raw files into training segments.
3. **Processed Data** (`data/processed/{index}_{category}/`): fixed-length segments consumed by the model.

## Supported Input Formats
- `*.wav`, `*.mp3`, `*.mp4`, `*.m4a`, `*.ogg`

## What Preprocessing Does
When you run `scripts/preprocess.py`, every input audio file is:
- Loaded and resampled to `target_sr = 16000 Hz`
- Peak-normalized (consistent amplitude across recording devices)
- Split into overlapping segments using a sliding window:
  - window length: `duration` seconds (default `2`)
  - hop size: `step` seconds (default `1`, i.e. 50% overlap)
- Saved to: `data/processed/{index}_{category}/<original_filename>_seg{i}.wav` as 16-bit PCM WAV at 16 kHz.

Already-processed source files are skipped (detected by the presence of `<original_filename>_seg0.wav`), so re-runs are incremental.
