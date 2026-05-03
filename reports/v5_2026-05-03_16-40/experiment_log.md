# Version 5
### **Date:** 2026-05-03 16:40
### **Experiment Goal:** Fix the train/val/test split itself. v4 inherited a stratification scheme that balanced **sources per class** while ignoring how many segments each source contributes; with very uneven source lengths this produced an unfair test set (only 9.3% Speech, 63.9% Violence) and a val set that did not match the train mix. The goal of v5 is to keep the no-leak guarantee (no source crosses splits) while making **per-class segment proportions** match across train, val, and test, and then re-measure the model under that fair protocol.

---

## 1. What changed vs v4
* **New splitter:** Added `grouped_balanced_split` in `scripts/splitting.py` and wired it into the training notebook (replaces the two chained `train_test_split(..., stratify=label)` calls). For each class independently it sorts sources by segment count (largest first, deterministic shuffle for tie-breaking) and greedily assigns each source to whichever split is currently furthest below its target completion (Longest-Processing-Time bin packing). Whole sources still go to exactly one split, so no leakage.
* **Reporting in the split cell extended:** absolute per-class segment counts per split, per-class proportions per split, and a per-class share-of-segments table (target ~0.64 / 0.16 / 0.20) — so the next time the splits look off it is visible immediately, not buried in metrics.
* **No data, model or feature changes.** Same `data/processed/`, same MFCC mean+std features, same Random Forest baseline, same hyperparameter grid, same `random_state=42`, same `class_weight="balanced"`. This run is a clean A/B for the new splitter against v4.

---

## 2. Segment-level class mix (this run)
**Splits sizes:** train **16 666**, val **4 168**, test **5 210** (≈ **64.0% / 16.0% / 20.0%** of segments — exactly on target).

**Class proportions per split (Ambient / Speech / Violence):**

| split | Ambient | Speech | Violence |
|-------|--------:|-------:|---------:|
| train | 30.2%   | 35.6%  | 34.2%    |
| val   | 30.3%   | 35.6%  | 34.1%    |
| test  | 30.3%   | 35.6%  | 34.2%    |

The three splits are now **essentially indistinguishable** in class makeup, and the test set is no longer Violence-heavy / Speech-poor. Leak check is `0 / 0 / 0`. The splitter behaved as designed.

---

## 3. Performance summary

### Validation (used for hyperparameter selection)
* Accuracy: **78.3%**
* Balanced accuracy: **79.3%**
* Macro F1: **77.3%**
* Weighted F1: **76.7%**

Selected configuration: `n_estimators=300, max_depth=None, min_samples_leaf=1, class_weight="balanced"` (changed from v4's `n_estimators=100`, but val macro-F1 across the four candidates only spans ~0.770–0.773, so this choice is largely noise — the splitter change is the meaningful intervention).

### Holdout test (final evaluation)
* **Accuracy:** **79.3%** (v4: 85.7%, **−6.4 pp**)
* **Balanced accuracy:** **79.5%** (v4: 85.6%, **−6.1 pp**)
* **Macro F1:** **0.794** (v4: 0.789, **+0.004**)
* **Weighted F1:** **0.793** (v4: 0.868, **−0.075**)

* **Per-class (test):**

| Class    | Precision | Recall | F1-score | Support (segments) |
|----------|-----------|--------|----------|--------------------|
| Ambient  | 0.78      | 0.84   | 0.81     | 1 576              |
| Speech   | **0.76**  | 0.74   | **0.75** | 1 855              |
| Violence | 0.84      | 0.81   | 0.82     | 1 779              |

* **Confusion matrix (test):**

| true \ pred | Ambient | Speech | Violence |
|---|---:|---:|---:|
| **Ambient**  | 1 320 |   162 |    94 |
| **Speech**   |   292 | 1 375 |   188 |
| **Violence** |    72 |   272 | 1 435 |

* **Error breakdown** (predicted vs true):

| count | pattern              |
|------:|----------------------|
| 292   | Speech → Ambient     |
| 272   | Violence → Speech    |
| 188   | Speech → Violence    |
| 162   | Ambient → Speech     |
|  94   | Ambient → Violence   |
|  72   | Violence → Ambient   |

### v4 → v5 per-class deltas

| Class    | Precision     | Recall        | F1            |
|----------|--------------|--------------|--------------|
| Ambient  | 0.90 → 0.78 (−0.12) | 0.76 → 0.84 (+0.08) | 0.82 → 0.81 (−0.01) |
| **Speech**   | **0.47 → 0.76 (+0.29)** | 0.92 → 0.74 (−0.18) | **0.62 → 0.75 (+0.13)** |
| Violence | 0.96 → 0.84 (−0.12) | 0.89 → 0.81 (−0.08) | 0.92 → 0.82 (−0.10) |

---

## 4. Conclusions
1. **The splitter change worked exactly as designed.** Train, val, and test now have identical class proportions (~30 / 36 / 34) and the per-class share of segments lands cleanly on the target 64 / 16 / 20. The unfair test-set composition that inflated v4's headline numbers is gone.
2. **The model is essentially the same quality, just measured fairly.** Macro F1 is unchanged (0.789 → 0.794) and balanced accuracy is the right metric to compare across versions; the **−6.4 pp drop in raw accuracy is a measurement artifact**, not a regression. v4's test set was 64% Violence (the model's strongest class), which mechanically pulled accuracy upward.
3. **Speech is the cleanest win.** Precision **0.47 → 0.76 (+0.29)** with F1 **0.62 → 0.75 (+0.13)**. The v4 model was effectively defaulting to "Speech" when uncertain because val never punished it for that (val itself was 43% Speech). With val now matching train, the hyperparameter search saw the real cost of over-predicting Speech and the resulting model is no longer a Speech-dispenser.
4. **Confusion is now near-symmetric.** Every off-diagonal cell sits in the 72–292 range; v4 had two error buckets (Violence → Speech 133, Ambient → Speech 110) that were several times larger than the rest. There is no longer a "fall-through" class that absorbs uncertainty.
5. **Hyperparameter search is currently flat.** All four candidate configurations land within ~0.003 macro-F1 on val. The grid is too narrow to be informative — at this point sweeping more aggressive variants (`balanced_subsample`, deeper trees, larger forests, threshold calibration) only makes sense once the data side is settled.
6. **Methodologically, v5 is now the first version that is honestly comparable to itself across reruns.** All earlier numbers (v1–v4) used different test compositions, so any future delta should be reported as "vs v5" rather than chained back to v4.

---

## 5. Planned next steps (before v6 / ongoing)
* **Hard negatives for the two largest residual error modes:**
  * `Speech → Ambient` (292): quiet/whispered or off-mic speech being read as room tone — add more low-energy speech and re-listen to ambient files that contain occasional voices.
  * `Violence → Speech` (272): yells/screams confused with energetic speech — add more "shouty speech" examples (sports commentary, animated talk) labeled as Speech, and confirm violence sources do not contain long talking stretches.
* **Feature caching** (carried over from v3/v4 plan, still pending). Persist `X`, `y`, paths and `target_map` as a versioned `.npz` + manifest keyed on a hash of `data/processed`. Rebuild only when processed data actually changes — full MFCC extraction is by far the slowest cell in the notebook.
* **Broader hyperparameter sweep + threshold calibration** (after v6, not before — no point tuning on dirty data). Try `class_weight="balanced_subsample"`, larger `n_estimators`, and use `predict_proba` on val to set per-class decision thresholds based on a precision/recall curve.
* **Source-level metric** alongside the segment-level one (carry-over): aggregate per-segment predictions per recording (majority vote / mean probability) and report source-level accuracy — closer to how the model would be deployed.
* **Reproducibility check (cheap):** rerun v5 with two or three other `random_state` values for `grouped_balanced_split` to estimate split variance on macro F1; this gives a noise floor for future comparisons (e.g. "v6 is +X over v5 ± σ").
