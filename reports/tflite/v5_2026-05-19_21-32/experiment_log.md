# Version 5 (CNN → TFLite)
### **Date:** 2026-05-19 21:32
### **Experiment Goal:** Re-run the smaller **Dense(32)** model from v4 with **higher EarlyStopping patience** (10 instead of 5) so validation has more time to improve before weights are frozen; measure segment-level test metrics vs v3 (`Dense(64)`) and document whether the capacity cut is acceptable for TinyML.

---

## 1. What changed vs v4 / v3

* **Architecture:** Same as v4 — **`Dense(32)`** (`Conv2D` 16→32, dropout 0.5, 3-class softmax). v3 used **`Dense(64)`**.
* **Training:** **EarlyStopping patience 10** on `val_loss` (v4 first run: **5**), `restore_best_weights=True`, Adam, batch **32**, up to **50** epochs.
* **Data & split:** Unchanged from v3/v4 (`grouped_balanced_split`, `random_state=42`).

---

## 2. Segment-level class mix (this run)

**Split sizes:** train **16 337**, val **4 085**, test **5 106** (same as v3).

**Class proportions per split (Ambient / Speech / Violence):**

| split | Ambient | Speech | Violence |
|-------|--------:|-------:|---------:|
| train | 30.9%   | 34.3%  | 34.9%    |
| val   | 30.9%   | 34.3%  | 34.8%    |
| test  | 30.9%   | 34.3%  | 34.8%    |

---

## 3. Performance summary

### Validation (EarlyStopping monitor)
* Accuracy: **70.1%**
* Balanced accuracy: **70.8%**
* Macro F1: **0.681**
* Weighted F1: **0.680**

### Holdout test (final evaluation)
* **Accuracy:** **71.7%** (v3: 80.6%, **−8.9 pp**)
* **Balanced accuracy:** **72.3%** (v3: 80.9%, **−8.6 pp**)
* **Macro F1:** **0.709** (v3: 0.805, **−0.096**)
* **Weighted F1:** **0.710** (v3: 0.806, **−0.096**)

* **Per-class (test):**

| Class    | Precision | Recall | F1-score | Support (segments) | v3 F1 (ref.) |
|----------|-----------|--------|----------|--------------------:|-------------:|
| Ambient  | 0.57      | 0.91   | 0.70     | 1 576               | 0.81         |
| Speech   | 0.77      | 0.45   | 0.57     | 1 751               | 0.75         |
| Violence | 0.91      | 0.81   | 0.86     | 1 779               | 0.87         |

* **Confusion matrix (test), segments:**

| true \\ pred | Ambient | Speech | Violence |
|-------------|--------:|-------:|---------:|
| **Ambient**  | 1 436 |    55 |     85 |
| **Speech**   |   910 |   789 |     52 |
| **Violence** |   169 |   175 |  1 435 |

* **Error breakdown** (true → predicted), **1 446** misclassified segments (v3: **993**):

| count | pattern              | v3 count |
|------:|----------------------|---------:|
| 910   | Speech → Ambient     | 459      |
| 175   | Violence → Speech    | 243      |
| 169   | Violence → Ambient   | 89       |
| 85    | Ambient → Violence   | 60       |
| 55    | Ambient → Speech     | 83       |
| 52    | Speech → Violence    | 59       |

**Speech → Ambient** nearly doubled (**+451** errors) and is again the dominant failure mode. Total test errors rose by **453** (~46%).

---

## 4. Training dynamics and where errors concentrate

### Overfitting and EarlyStopping
Training ran for **17 epochs** before EarlyStopping fired (v4 dense 32 / patience 5: stop **10**, best **5**; v3 dense 64 / patience 5: stop **16**, best **11**). Higher patience allowed a longer run than v4, but metrics still sit well below v3 — the bottleneck is mainly **model capacity**, not only stopping too early.

Train metrics continue to climb while val/test lag, consistent with overfitting on the smaller head.

### Top error sources (test)
| source file        | test errors (v5) | v3 errors | Notes |
|--------------------|-----------------:|----------:|-------|
| `convo1.mp3`       | **356**           | 107       | Speech → Ambient; largest single source |
| `talk1.wav`        | **334**           | 266       | Still dominant; worse than v3 |
| `yell_female5.wav` | **84**            | 81        | Violence ↔ Speech/Ambient |
| `talk_child2.wav`  | **80**            | 35        | Regression vs post-trim v3 |
| `talk_children.wav`| **73**            | —         | Speech confused with Ambient/Violence |
| `talk_female2.wav` | **56**            | 11        | Regression vs v3 cleanup gains |

Hard Speech sources (`convo1`, `talk1`) drive most of the macro-F1 drop; Violence detection remains relatively stable (F1 **~0.86**).

---

## 5. Conclusions

1. **Patience 10 helped training length** (17 epochs vs v4’s 10) but **did not close the gap to v3**: balanced accuracy **72.3%** vs **80.9%** (−8.6 pp). The **Dense(32)** cut costs mainly **Speech recall** (0.70 → **0.45**) and **Ambient precision** (0.72 → **0.57**), while **Violence** is nearly unchanged.
2. **Speech → Ambient** regressed sharply (**459 → 910**), including on files that improved in v3 (`talk_female2`, `talk_child2`) — the smaller dense layer appears to default to Ambient under uncertainty.
3. **v5 is not a suitable TinyML substitute for v3 on accuracy** without further changes; smaller weights come with a large segment-level penalty on Speech.
4. **Next steps (from v4 plan):** try **`Dense(48)`** as a middle ground; if still below v3, keep **Dense(64)** for deployment or add regularisation/augmentation before shrinking further. Optional: second data pass on `convo1` / `talk1`, class weights, TFLite size/latency comparison.

---

## 6. Artifacts in this folder
* `metrics.json`, `confusion_matrix.png`, `classification_errors.csv`
* `error_source_report.csv`, `top_errors_chart.png` (from `notebooks/tflite/error_analysis.ipynb`)
