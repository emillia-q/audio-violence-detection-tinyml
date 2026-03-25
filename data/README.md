# Data Management & Preprocessing

This directory contains the dataset for audio classification, split into raw recordings and segments ready for training.

## Data Flow
1. **Raw Data** (`data/raw/`): Place your long recordings here in appropriate subfolders (`ambient`, `speech`, `violence`).
2. **Preprocessing**: Run `scripts/preprocess.py` to convert raw files into training segments.
3. **Processed Data** (`data/processed/`): Final 2-second segments used by the Machine Learning model.

## Supported Input Formats
- `*.wav`, `*.mp3`, `*.mp4`, `*.m4a`, `*.ogg`

## What Preprocessing Does
When you run `scripts/preprocess.py`, every input audio file is:
- Loaded and resampled to `16000 Hz`
- Peak-normalized
- Split into segments of `duration` seconds (default: `2`)
- Saved to: `data/processed/<category>/<original_filename>_seg{i}.wav`