# Version 3 (CNN → TFLite)
### **Date:** 2026-05-17 21:17
### **Experiment Goal:** Measure whether **targeted data cleanup** on the worst v2 error sources — without changing model architecture or training hyperparameters — improves segment-level metrics and reduces Speech→Ambient confusion. Same MFCC CNN pipeline, grouped-by-source splits (`random_state=42`), and TFLite export as v2.

---

## 1. What changed vs v2

* **Data only (A/B):** The three heaviest v2 error contributors among Speech-labelled sources — `talk_female2.wav`, `talk_child2.wav`, and `talk2.wav` — were re-listened in full. Long stretches that were effectively **ambient silence or non-speech room tone** (but still inside Speech-labelled files) were **trimmed out** in the raw recordings, then affected sources were **re-segmented** via `scripts/preprocess.py` (old `*_seg*.wav` removed first so preprocessing did not skip them).
* **Model & training:** Identical CNN (`Conv2D` 16→32, `Dense` 64, dropout 0.5), Adam, batch size **32**, up to **50** epochs, **EarlyStopping** on `val_loss` with **patience 5** and **`restore_best_weights=True`**.
* **Split:** Same `grouped_balanced_split` with `random_state=42`; leak check all zeros. Segment inventory is almost the same as v2; Speech train count is slightly lower (**5 604** vs **5 933**) after trimming, Ambient and Violence segment totals unchanged.

---

## 2. Segment-level class mix (this run)

**Split sizes:** train **16 337**, val **4 085**, test **5 106** (v2: 16 666 / 4 168 / 5 210).

**Class proportions per split (Ambient / Speech / Violence):**

| split | Ambient | Speech | Violence |
|-------|--------:|-------:|---------:|
| train | 30.9%   | 34.3%  | 34.8%    |
| val   | 30.9%   | 34.3%  | 34.8%    |
| test  | 30.9%   | 34.3%  | 34.8%    |

Per-class segment shares to train / val / test remain **~64% / 16% / 20%** for each class.

---

## 3. Performance summary

### Validation (EarlyStopping monitor)
* Accuracy: **82.5%**
* Balanced accuracy: **82.9%**
* Macro F1: **0.825**
* Weighted F1: **0.823**

### Holdout test (final evaluation)
* **Accuracy:** **80.6%** (v2: 75.6%)
* **Balanced accuracy:** **80.9%** (v2: 76.0%)
* **Macro F1:** **0.805** (v2: 0.754)
* **Weighted F1:** **0.806** (v2: 0.754)

* **Per-class (test):**

| Class    | Precision | Recall | F1-score | Support (segments) | v2 F1 (ref.) |
|----------|-----------|--------|----------|--------------------:|-------------:|
| Ambient  | 0.72      | 0.91   | 0.81     | 1 576               | 0.73         |
| Speech   | 0.79      | 0.70   | 0.75     | 1 751               | 0.68         |
| Violence | 0.92      | 0.81   | 0.87     | 1 779               | 0.86         |

* **Confusion matrix (test), segments:**

| true \\ pred | Ambient | Speech | Violence |
|-------------|--------:|-------:|---------:|
| **Ambient**  | 1 433 |    83 |     60 |
| **Speech**   |   459 | 1 233 |     59 |
| **Violence** |    89 |   243 |  1 447 |

* **Error breakdown** (true → predicted), **993** misclassified segments (v2: **1 271**):

| count | pattern              | v2 count |
|------:|----------------------|---------:|
| 459   | Speech → Ambient     | 560      |
| 243   | Violence → Speech    | 158      |
|  89   | Violence → Ambient   | 86       |
|  83   | Ambient → Speech     | 197      |
|  60   | Ambient → Violence   | 107      |
|  59   | Speech → Violence    | 163      |

The dominant v2 failure mode (**Speech → Ambient**) dropped by **101** errors; total test errors fell by **278** (~22%).

---

## 4. Training dynamics and where errors concentrate

### Overfitting and EarlyStopping
Training ran for **16 epochs** before EarlyStopping fired (v2: **9**). **Best `val_loss` occurred at epoch 11** (~0.477); exported weights use `restore_best_weights=True` from that checkpoint. In v2, validation loss tended to worsen from roughly **epoch 4** onward while train metrics kept improving; in v3 the model sustained useful val improvement longer (through epoch 11), which is consistent with **cleaner Speech segments** rather than the model latching onto silence-as-Ambient cues in the top problematic files.

Train accuracy still exceeds val by the end of the run — some overfitting remains — but the **best checkpoint is later and val metrics are higher** than v2 at export time.

### Effect on the trimmed sources (test errors by source)
| source file        | test errors (v3) | v2 errors (approx.) | Notes |
|--------------------|-----------------:|--------------------:|-------|
| `talk_female2.wav` | **11**            | 289                 | Large reduction after silence trim |
| `talk_child2.wav`  | **35**            | 197                 | Still some confusion, much lower |
| `talk2.wav`        | **0**             | 120                 | No test errors in this run |

New top contributors (not targeted in this pass) include `talk1.wav` (**266** errors), `convo1.mp3` (**107**), and `yell_female5.wav` (**81**, Violence→Speech). Data cleanup **shifted** the error budget rather than eliminating hard sources entirely.

---

## 5. Conclusions

1. **Data cleanup on three v2 outlier files was effective:** test accuracy **+5.0 pp**, macro F1 **+0.051**, with **no architecture or training-rule changes**. This supports the v2 hypothesis that label/borderline-audio quality drove much of the Speech→Ambient confusion.
2. **Speech improved most** (F1 **0.68 → 0.75**, recall **0.61 → 0.70**); **Ambient** precision/recall balance improved (F1 **0.73 → 0.81**). **Violence** F1 is essentially unchanged (**~0.87**), still acceptable for the primary detection target.
3. **Training behaved better:** best validation checkpoint at **epoch 11** vs clear val degradation from ~epoch 4 in v2 — worth documenting alongside metrics when arguing that cleaner data eases optimization, not only inference.
4. **Gap to RF v6 (~91% segment accuracy) remains**; the next gains likely need additional source audits (`talk1`, `convo1`, borderline Violence) and/or class-weighting / augmentation, not only more epochs.
5. **Next steps:** optional `error_analysis` on v3 is already in `error_source_report.csv` / `top_errors_chart.png`; consider a second cleanup pass on the new top files, TFLite-vs-Keras check on test, then ESP pipeline (MFCC 1:1 + multi-hit Violence alarm policy).

---

## 6. Artifacts in this folder
* `metrics.json`, `confusion_matrix.png`, `classification_errors.csv`
* `error_source_report.csv`, `top_errors_chart.png` (from `notebooks/tflite/error_analysis.ipynb`)
