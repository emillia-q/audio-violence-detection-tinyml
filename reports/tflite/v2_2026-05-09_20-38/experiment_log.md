# Version 2 (CNN → TFLite)
### **Date:** 2026-05-09 20:38
### **Experiment Goal:** Train a compact CNN on **MFCC** tensors (same 2 s / 16 kHz segmenting pipeline as the RF notebooks) for 3-class segment-level classification (Ambient / Speech / Violence), evaluate on the same grouped-by-source splits used for the RF baselines, and export a TFLite artifact suitable for TinyML deployment. Document validation vs test behaviour, per-class performance, training dynamics (overfitting), and error concentration by source file.

---

## 1. What changed vs earlier work
* **Model family:** Convolutional network trained end-to-end on **MFCC** features (TinyML-oriented footprint), replacing the **MFCC + Random Forest** pipeline used in `reports/rf/*` while keeping a comparable front-end feature extraction.
* **Splits and segment counts:** The run uses the **same segment inventory and split sizes** as the recent RF experiments: train **16 666**, val **4 168**, test **5 210** (see `metrics.json`). Class proportions are ~30% Ambient, ~36% Speech, ~34% Violence in each split.
* **Training setup:** Up to **50** epochs, batch size **32**, **EarlyStopping** on `val_loss` with **patience 5** and **`restore_best_weights=True`**. Training **stopped after 9 epochs** when validation loss stopped improving.

---

## 2. Segment-level class mix (this run)
**Split sizes:** train **16 666**, val **4 168**, test **5 210**.

**Class proportions per split (Ambient / Speech / Violence):**

| split | Ambient | Speech | Violence |
|-------|--------:|-------:|---------:|
| train | 30.2%   | 35.6%  | 34.2%    |
| val   | 30.3%   | 35.6%  | 34.1%    |
| test  | 30.3%   | 35.6%  | 34.2%    |

---

## 3. Performance summary

### Validation (monitor used by EarlyStopping)
* Accuracy: **71.4%**
* Balanced accuracy: **71.5%**
* Macro F1: **0.705**
* Weighted F1: **0.703**

Validation is **noticeably below** the holdout test headline metrics — the gap is consistent with early stopping on `val_loss` and the usual variance between val and test when the model is regularized by stopping at the best val checkpoint.

### Holdout test (final evaluation)
* **Accuracy:** **75.6%**
* **Balanced accuracy:** **76.0%**
* **Macro F1:** **0.754**
* **Weighted F1:** **0.754**

For a **lightweight MFCC-CNN** on noisy real-world audio, these aggregates are **solid**; they sit below the best RF segment-level numbers in `reports/rf/v6_*` but remain **competitive for an on-device model** where latency and size matter.

* **Per-class (test):**

| Class    | Precision | Recall | F1-score | Support (segments) |
|----------|-----------|--------|----------|--------------------|
| Ambient  | 0.66      | 0.81   | 0.73     | 1 576              |
| Speech   | 0.76      | 0.61   | 0.68     | 1 855              |
| Violence | 0.85      | 0.86   | 0.86     | 1 779              |

* **Confusion matrix (test), segments:**

| true \\ pred | Ambient | Speech | Violence |
|-------------|--------:|-------:|---------:|
| **Ambient**  | 1 272 |   197 |    107 |
| **Speech**   |   560 | 1 132 |   163 |
| **Violence** |    86 |   158 |  1 535 |

* **Error breakdown** (true → predicted), **1 271** misclassified segments:

| count | pattern              |
|------:|----------------------|
| 560   | Speech → Ambient     |
| 197   | Ambient → Speech     |
| 163   | Speech → Violence    |
| 158   | Violence → Speech    |
| 107   | Ambient → Violence   |
|  86   | Violence → Ambient   |

**Violence** is the strongest class (highest F1), which is favourable for a **violence-detection** use case: the model both finds most violence segments and avoids flooding other classes with false “violence” predictions at the rates seen for Ambient/Speech confusion.

**Speech** trades **recall for precision** (many Speech segments are missed and end up as Ambient — 560 of the errors). **Ambient** has the opposite profile: **high recall** but **lower precision**, i.e. other classes are sometimes folded into Ambient.

---

## 4. Training dynamics and where errors concentrate

### Overfitting and EarlyStopping
Training **loss on the train set decreased** while **train accuracy increased** across epochs. **Validation loss improved initially, then rose** (worsening generalisation from roughly **epoch 4 onward** in the manual review of the `history` curves), while **validation accuracy fluctuated** rather than tracking train accuracy — a **classic overfitting signature**.

**EarlyStopping behaved as intended:** training halted at **epoch 9**, and with **`restore_best_weights=True`** the exported weights correspond to the **best `val_loss` checkpoint**, limiting damage from later epochs where the model memorised training idiosyncrasies.

### Error concentration by source file
Errors are **not uniform** across recordings. Aggregating `classification_errors.csv` by original source (file stem before `_seg*.wav`), the **heaviest contributors** include:

| source file        | errors (approx.) | true label | dominant confusion |
|--------------------|-----------------:|------------|---------------------|
| `talk_female2.wav` | **289**          | Speech     | → Ambient / Violence |
| `talk_child2.wav`  | **197**          | Speech     | → Ambient / Violence |
| `talk2.wav`        | **120**          | Speech     | → Ambient / Violence |
| `computer.wav`     | **78**           | Ambient    | → Speech / Violence |
| `kid_playing.wav`  | **56**           | Speech     | → Ambient / Violence |
| `yell_female5.wav` | **51**           | Violence   | → Speech            |
| `fem_sobbing.wav`  | **51**           | Ambient    | → Violence / Speech |

This pattern mirrors the RF runs: **a small set of borderline or mis-annotated sources** absorbs a large share of the error budget; improving labels, trimming non-target audio, or splitting mixed files will likely move metrics more than marginal architecture tweaks alone.

---

## 5. Conclusions
1. **The CNN TFLite v2 model is a credible TinyML candidate:** ~**75.6%** accuracy and **~0.754** macro F1 on the holdout test, with **Violence F1 ≈ 0.86** — the priority class is well served.
2. **Validation underperforms test** on this split draw; always report **both** when comparing to RF baselines. The gap does not invalidate the run but shows sensitivity to which segments land in val vs test.
3. **Overfitting is visible in the learning curves**; EarlyStopping with best-weight restore **mitigated** late-epoch regression. Further gains should combine **data-side fixes** with **regularisation / augmentation**, not only more epochs.
4. **Speech vs Ambient is the dominant failure mode** (560 Speech → Ambient errors). **Ambient** is occasionally a “sink” class (lower precision, high recall). Addressing class balance via **loss weighting** or **targeted hard negatives** may help without exploding model size.
5. **Error mass is concentrated in known difficult sources** (`talk_female2`, `talk_child2`, `yell_female5`, `computer`, `fem_sobbing`, …) — consistent with **data quality** rather than a mysterious model bug.

---

## 6. Planned next steps (v3 / ongoing)
* **Regularisation & augmentation:** time / pitch shifts, light noise, mixup or SpecAugment-style masking; slightly **higher dropout** or **L2** on conv/dense kernels if not already tuned; consider **label smoothing** for the softmax head.
* **Class imbalance handling:** **class weights** or **focal loss** to lift Speech recall and reduce Ambient false positives, while monitoring Violence F1 (do not trade away the main detection objective).
* **Data cleanup (high ROI):** manual pass on the **top error sources** above — same playbook as RF v6/v7 (re-listen, re-trim, relabel or split files).
* **Baseline comparison:** report deltas against **RF v6** (segment-level) and this **TFLite v2** side-by-side for deployment-oriented decisions (accuracy vs latency vs model size).
* **Optional:** confusion-matrix PNG and TFLite size/latency numbers in the same folder once benchmarked on the target MCU / NPU.
