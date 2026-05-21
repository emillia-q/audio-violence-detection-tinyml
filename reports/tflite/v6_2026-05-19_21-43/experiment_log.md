# Version 6 (CNN → TFLite)
### **Date:** 2026-05-19 21:43
### **Experiment Goal:** Test whether switching the optimizer from **Adam** to **AdamW** (with weight decay) improves training of the **Dense(32)** model from v5, without changing architecture, data, or EarlyStopping patience.

---

## 1. What changed vs v5

| Setting | v5 | v6 |
|---------|----|----|
| `Dense` units | 32 | 32 (unchanged) |
| Optimizer | **Adam** | **AdamW** |
| EarlyStopping patience | 10 | 10 |
| Data & split | v3 pipeline | unchanged |

* **Architecture:** `Dense(32)`, `Conv2D` 16→32, dropout 0.5 — ~**51k** trainable params (vs ~97k for v3 `Dense(64)`).
* **Training:** AdamW, batch **32**, up to **50** epochs, **EarlyStopping** on `val_loss`, **patience 10**, `restore_best_weights=True`.

---

## 2. Segment-level class mix (this run)

**Split sizes:** train **16 337**, val **4 085**, test **5 106** (same as v3/v5).

---

## 3. Performance summary

### Validation (EarlyStopping monitor)
* Accuracy: **83.3%** (v5: 70.1%)
* Balanced accuracy: **83.6%** (v5: 70.8%)
* Macro F1: **0.831** (v5: 0.681)

### Holdout test (final evaluation)
* **Accuracy:** **79.8%** (v5: 71.7%, v3: 80.6%)
* **Balanced accuracy:** **80.1%** (v5: 72.3%, v3: 80.9%, **−0.8 pp vs v3**)
* **Macro F1:** **0.797** (v5: 0.709, v3: 0.805, **−0.008 vs v3**)
* **Weighted F1:** **0.798** (v5: 0.710, v3: 0.806)

* **Per-class (test):**

| Class    | Precision | Recall | F1-score | Support | v5 F1 | v3 F1 |
|----------|-----------|--------|----------|--------:|------:|------:|
| Ambient  | 0.72      | 0.88   | 0.79     | 1 576   | 0.70  | 0.81  |
| Speech   | 0.78      | 0.69   | 0.73     | 1 751   | 0.57  | 0.75  |
| Violence | 0.91      | 0.83   | 0.87     | 1 779   | 0.86  | 0.87  |

* **Confusion matrix (test), segments:**

| true \\ pred | Ambient | Speech | Violence |
|-------------|--------:|-------:|---------:|
| **Ambient**  | 1 391 |   119 |     66 |
| **Speech**   |   468 | 1 202 |     81 |
| **Violence** |    86 |   212 |  1 481 |

* **Error breakdown**, **1 032** misclassified segments (v5: **1 446**, v3: **993**):

| count | pattern              | v5 count | v3 count |
|------:|----------------------|---------:|---------:|
| 468   | Speech → Ambient     | 910      | 459      |
| 212   | Violence → Speech    | 175      | 243      |
| 119   | Ambient → Speech     | 55       | 83       |
| 86    | Violence → Ambient   | 169      | 89       |
| 81    | Speech → Violence    | 52       | 59       |
| 66    | Ambient → Violence   | 85       | 60       |

**Speech → Ambient** halved vs v5 (**−442**); total errors **−414** (~29%). Remaining gap to v3 is mostly **Speech recall** (0.69 vs 0.70) and **Ambient precision** (0.72 vs 0.72 — similar, but more Ambient→Speech confusion than v3).

---

## 4. Training dynamics and where errors concentrate

### Overfitting and EarlyStopping
Training ran for **24 epochs** (v5: **17**). **Best `val_loss` at epoch 14** (~0.527); stop at epoch 24 matches patience 10. Val metrics at export (**~83.6%** balanced accuracy) are well above test (**80.1%**), so some generalization gap remains, but far less train–val divergence than v5.

### Top error sources (test)
| source file        | v6 errors | v5 errors | v3 errors |
|--------------------|----------:|----------:|----------:|
| `talk1.wav`        | **272**   | 334       | 266       |
| `convo1.mp3`       | **117**   | 356       | 107       |
| `yell_female5.wav` | **76**    | 84        | 81        |
| `talk_children.wav`| **51**    | 73        | —         |
| `fem_sobbing.wav`  | **46**    | —         | —         |

`convo1` and `talk1` still dominate, but **`convo1` recovered strongly** from the v5 collapse (356 → 117 errors).

---

## 5. Observations

1. **Optimizer mattered more than expected:** Same **Dense(32)** and patience as v5, but **AdamW** lifted balanced accuracy by **+7.8 pp** on test — nearly matching **v3 (`Dense(64)` + Adam)** with ~**half** the dense-layer parameters.
2. **v5’s poor result was largely an optimization issue**, not proof that 32 units are unusable. The model had been collapsing Speech into Ambient under Adam; AdamW restored Speech recall (**0.45 → 0.69**).
3. **Still ~0.8 pp below v3** on balanced accuracy; Speech F1 **0.73 vs 0.75**. Violence is on par. Further gains likely need **data work** (`talk1`, `convo1`) or light tuning, not necessarily reverting to `Dense(64)`.
4. **Trade-off:** **Ambient → Speech** errors rose (**55 → 119**) vs v5 — slightly more than v3 (**83**). Worth watching if tuning AdamW `weight_decay` / learning rate.

---

## 6. Proposed next optimizations

1. **Lock in v6 as the TinyML baseline** — export TFLite, record model size and latency vs v3; confirm Keras vs TFLite parity on test.
2. **Tune AdamW** — try explicit `learning_rate` (e.g. 1e-3 → 5e-4) and `weight_decay` (e.g. 1e-4, 1e-3) to reduce val–test gap and Ambient→Speech slips.
3. **Optional architecture sweep** — **`Dense(48)` + AdamW** as a single run: may close the small v3 gap without returning to 64 units.
4. **Data-side (highest leverage for `talk1` / `convo1`)** — second listening pass on borderline Speech segments; same approach as v3 trim on `talk_female2` / `talk_child2`.
5. **Training hygiene** — save `optimizer`, `dense_units`, and real `patience` in `metrics.json` (v5/v6 JSON still lists `early_stopping_patience: 5`).
6. **If deployment needs max accuracy** — keep v3 weights as fallback; use v6 when size budget is tight and ~80% balanced accuracy is acceptable.

---

## 7. Artifacts in this folder
* `metrics.json`, `confusion_matrix.png`, `classification_errors.csv`
* `error_source_report.csv`, `top_errors_chart.png` (from `notebooks/tflite/error_analysis.ipynb`)
