from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def get_source_name(file_path: str) -> str:
    """Extracts the original filename from a segmented audio file path.
    Example: 'path/to/yell.wav_seg12.wav' -> 'yell.wav'.
    """
    base_name = Path(file_path).name
    return base_name.split("_seg")[0]


def grouped_balanced_split(
    meta: pd.DataFrame,
    *,
    val_size: float = 0.16,
    test_size: float = 0.20,
    random_state: int | None = 42,
) -> tuple[set[str], set[str], set[str]]:
    """Source-level split that balances *segment* counts per class across train/val/test.

    train_test_split(..., stratify=source.label) only stratifies the *sources*
    per class and ignores how many segments each source contributes. When source segment
    counts vary widely (e.g. one long ambient recording vs many short violence shouts),
    the per-split *segment* proportions end up wildly off the requested fractions even
    though no source is leaked across splits.

    Parameters
    ----------
    meta : DataFrame with at least columns `source` and `label` (one row per segment).
    val_size, test_size : target fraction of *segments* per class for val and test.
    random_state : seed for tie-breaking shuffles within each class.

    Returns
    -------
    (train_sources, val_sources, test_sources) : disjoint sets of source names.
    """
    if val_size < 0 or test_size < 0:
        raise ValueError("val_size and test_size must be non-negative")
    if val_size + test_size >= 1.0:
        raise ValueError("val_size + test_size must be < 1.0")

    rng = np.random.default_rng(random_state)
    train_size = 1.0 - val_size - test_size

    train_sources: set[str] = set()
    val_sources: set[str] = set()
    test_sources: set[str] = set()
    bins = {"train": train_sources, "val": val_sources, "test": test_sources}

    for label in sorted(meta["label"].unique()):
        class_sizes = (
            meta.loc[meta["label"] == label]
            .groupby("source", sort=False)
            .size()
            .reset_index(name="n")
        )
        if class_sizes.empty:
            continue

        # Shuffle first so equal-sized sources don't always land in alphabetical order,
        # then stable-sort largest-first (LPT heuristic).
        class_sizes = class_sizes.sample(
            frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))
        )
        class_sizes = class_sizes.sort_values("n", ascending=False, kind="stable")

        total = int(class_sizes["n"].sum())
        targets = {
            "train": total * train_size,
            "val": total * val_size,
            "test": total * test_size,
        }
        current = {"train": 0, "val": 0, "test": 0}

        for src, n in zip(class_sizes["source"], class_sizes["n"]):
            # Assign to the split most behind its target (lowest completion fraction).
            # When completions tie, dict order keeps train -> val -> test, which means
            # the very first source of a class goes to train (the largest budget).
            def completion(k: str) -> float:
                return current[k] / targets[k] if targets[k] > 0 else float("inf")

            chosen = min(("train", "val", "test"), key=completion)
            bins[chosen].add(src)
            current[chosen] += int(n)

    overlaps = (
        (train_sources & val_sources)
        | (train_sources & test_sources)
        | (val_sources & test_sources)
    )
    if overlaps:
        raise RuntimeError(f"internal error: source(s) leaked across splits: {overlaps}")

    return train_sources, val_sources, test_sources
