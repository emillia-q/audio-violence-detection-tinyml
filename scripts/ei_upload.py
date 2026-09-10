from __future__ import annotations

import os
from pathlib import Path

import edgeimpulse as ei
import pandas as pd
from dotenv import load_dotenv

from scripts.splitting import get_source_name, grouped_balanced_split

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed"
API_KEY = os.environ["EI_API_KEY"]

ei.API_KEY = API_KEY


def collect_metadata(processed_dir: Path) -> pd.DataFrame:
    """Builds the same table (path, label, source) as in the training notebook."""
    rows = []
    for category_dir in sorted(processed_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        # Strip the numeric prefix "0_ambient" -> "ambient" (label without index)
        label_name = category_dir.name.split("_", 1)[1] if "_" in category_dir.name else category_dir.name

        for file_path in category_dir.glob("*.wav"):
            rows.append(
                {
                    "path": file_path,
                    "label": label_name,
                    "source": get_source_name(str(file_path)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    meta = collect_metadata(DATA_PATH)
    print(f"Found {len(meta)} segments in {meta['source'].nunique()} source recordings.")

    # Same split as in train_model.ipynb - identical random_state = same split
    train_sources, val_sources, test_sources = grouped_balanced_split(
        meta.rename(columns={"label": "label"}),  # Expects "source" and "label" columns
        val_size=0.16,
        test_size=0.20,
        random_state=42,
    )

    # EI only knows two categories for upload: training / testing.
    # Treat "val" as part of "training" - the actual train/val split
    # will execute EI internally, but GROUPED by the "source" metadata
    def category_for(source: str) -> str:
        if source in test_sources:
            return "testing"
        return "training"  # train_sources and val_sources both go here

    # Prepare a list of all records
    meta_records = list(meta.itertuples(index=False))

    # Set batch size (500 files at a time)
    batch_size = 500
    total_success = 0
    total_fails = 0
    total_batches = (len(meta_records) // batch_size) + 1

    print(
        f"Starting upload of {len(meta_records)} files in {total_batches} batches (max {batch_size} items per batch)...\n")

    for i in range(0, len(meta_records), batch_size):
        batch = meta_records[i:i + batch_size]
        samples = []

        for row in batch:
            samples.append(
                ei.experimental.data.Sample(
                    data=row.path.read_bytes(),  # Read file bytes and release resources immediately
                    filename=row.path.name,
                    category=category_for(row.source),
                    label=row.label,
                    metadata={"source": row.source},
                )
            )

        # Send a single batch to Edge Impulse
        current_batch_num = (i // batch_size) + 1
        print(f"Uploading batch {current_batch_num}/{total_batches} ({len(samples)} files)")

        response = ei.experimental.data.upload_samples(samples)

        # Count statistics
        successes = len(getattr(response, "successes", []) or [])
        fails = len(getattr(response, "fails", []) or [])

        total_success += successes
        total_fails += fails

        print(f"   -> Finished batch {current_batch_num}. Success: {successes}, Fails: {fails}")

    print("\n" + "=" * 50)
    print("UPLOAD COMPLETED")
    print(f"Total successes: {total_success}")
    print(f"Total fails: {total_fails}")
    print("=" * 50)

if __name__ == "__main__":
    main()