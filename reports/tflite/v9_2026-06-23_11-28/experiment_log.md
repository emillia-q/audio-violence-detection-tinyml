# Version 9 (CNN -> TFLite)
### **Date:** 2026-06-23 11:28
### **Experiment Goal:** Test the architecture change proposed after v8: move learning capacity from the dense classifier head into the convolutional feature extractor. v9 replaces the `Flatten -> Dense` bottleneck with `GlobalAveragePooling2D` and increases the convolutional filters, so the model learns stronger time-frequency features while staying within the TinyML budget.

---

## 1. What changed vs v8

| Setting | v8 | v9 |
|---------|----|----|
| Dataset | cleaned/enriched v8 dataset | unchanged |
| Preprocess normalization | Peak norm on each **2 s segment** | unchanged |
| Conv stack | `Conv2D(16)` -> `Conv2D(32)` | `Conv2D(32)` -> `Conv2D(64)` |
| Feature-to-classifier bridge | `Flatten()` | `GlobalAveragePooling2D()` |
| Dense head | `Dense(32)`, dropout 0.5 | `Dense(64)`, dropout 0.4 |
| Optimizer | **AdamW** | **AdamW** |
| Training | 18 epochs trained | **34** epochs trained |
| Trainable params | ~**51k** | **23 171** |
| TFLite size | **208 148 B** | **96 140 B** |

### Architecture motivation

v8 had most of its parameters in the first dense layer: **46 112** parameters out of about **51k** total. That meant the convolutional stack, which should learn local audio patterns across time and frequency, was very small, while the classifier head carried most of the model memory.

v9 fixes that imbalance:

* `Conv2D(32)` has **320** parameters.
* `Conv2D(64)` has **18 496** parameters.
* `GlobalAveragePooling2D` removes the large flattened vector.
* `Dense(64)` now has only **4 160** parameters.

This is a better TinyML allocation: the model is smaller overall, but more of its capacity is spent before classification, where audio-specific features are learned.

---

## 2. Segment-level class mix (this run)

**Split sizes:** train **17 279**, val **4 321**, test **5 402**.

**Class proportions per split (Ambient / Speech / Violence):**

| split | Ambient | Speech | Violence |
|-------|--------:|-------:|---------:|
| train | 32.0%   | 34.8%  | 33.2%    |
| val   | 32.0%   | 34.8%  | 33.2%    |
| test  | 32.0%   | 34.8%  | 33.3%    |

The dataset and split sizes match v8, so v8/v9 is a clean architecture comparison.

---

## 3. Performance summary

### Validation (EarlyStopping monitor)
* Accuracy: **87.4%** (v8: 80.7%)
* Balanced accuracy: **87.5%** (v8: 81.1%)
* Macro F1: **0.875** (v8: 0.808)

Validation improved by about **+6.5 pp** balanced accuracy. The model also trained longer (**34** epochs vs **18**), suggesting the new architecture optimized more steadily instead of stopping early.

### Holdout test (final evaluation)
* **Accuracy:** **84.2%** (v8: 75.5%, v7: 80.0%)
* **Balanced accuracy:** **84.3%** (v8: 75.9%, v7: 80.2%)
* **Macro F1:** **0.843** (v8: 0.752, v7: 0.797)
* **Weighted F1:** **0.841** (v8: 0.749, v7: 0.797)

v9 not only recovers the v8 regression, it also beats the previous v7 baseline on the same overall task direction. The gain is large enough to treat the architecture change as a real improvement, not metric noise.

* **Per-class (test):**

| Class    | Precision | Recall | F1-score | Support | v8 F1 | v8 recall |
|----------|-----------|--------|----------|--------:|------:|----------:|
| Ambient  | 0.916     | 0.925  | 0.921    | 1 728   | 0.826 | 0.866     |
| Speech   | 0.793     | 0.802  | 0.797    | 1 877   | 0.667 | 0.572     |
| Violence | 0.820     | 0.802  | 0.811    | 1 797   | 0.761 | 0.840     |

* **Confusion matrix (test), segments:**

| true \ pred | Ambient | Speech | Violence |
|-------------|--------:|-------:|---------:|
| **Ambient**  | 1 598 |   108 |     22 |
| **Speech**   |    77 | 1 506 |    294 |
| **Violence** |    69 |   286 |  1 442 |

* **Error breakdown**, **856** misclassified segments (v8: **1 323**):

| count | pattern              | v8 count |
|------:|----------------------|---------:|
| 294   | Speech -> Violence   | 503      |
| 286   | Violence -> Speech   | 191      |
| 108   | Ambient -> Speech    | 74       |
| 77    | Speech -> Ambient    | 301      |
| 69    | Violence -> Ambient  | 97       |
| 22    | Ambient -> Violence  | 157      |

The most important correction is the large drop in false violence detections:

* **Speech -> Violence:** **503 -> 294**
* **Ambient -> Violence:** **157 -> 22**

This directly addresses the v8 failure mode where high-pitched or expressive non-violent audio was too often classified as Violence. The trade-off is more **Violence -> Speech** errors (**191 -> 286**), so Violence recall drops from **0.840** to **0.802**. In practical terms, v9 is less trigger-happy and more precise, but slightly less sensitive to true Violence.

---

## 4. Training dynamics and where errors concentrate

### Generalization gap
Validation balanced accuracy is **87.5%**, while test balanced accuracy is **84.3%**, a gap of about **3.2 pp**. This is much healthier than v8's ~**5.1 pp** gap. The architecture is not just better on validation; it transfers better to the holdout test.

### Top error sources (test)
| source file             | v9 errors | v8 errors | dominant true label | predicted as |
|-------------------------|----------:|----------:|---------------------|--------------|
| `conversation_m&f.wav`  | **286**   | 436       | Speech              | Violence, Ambient |
| `yell_female5.wav`      | **96**    | 50        | Violence            | Speech, Ambient |
| `bg.wav`                | **61**    | 14        | Ambient             | Speech |
| `angry_201.wav`         | **46**    | 17        | Violence            | Speech, Ambient |
| `yell_violence0.2_h_piano.wav` | **43** | 15 | Violence | Speech, Ambient |

`conversation_m&f.wav` is still the largest error source, but it improved strongly (**436 -> 286**). The remaining top errors shifted toward true Violence files being predicted as Speech, which matches the aggregate confusion matrix.

---

## 5. Observations

1. **The architecture change worked.** Balanced accuracy jumped from **75.9%** to **84.3%**, and total test errors dropped by **467** segments.
2. **GlobalAveragePooling2D solved the dense bottleneck.** The model is smaller (**96 KB** TFLite vs **208 KB**) while performing much better. Removing `Flatten` prevented the dense head from dominating the parameter budget.
3. **More convolutional capacity helped audio discrimination.** The model now has enough filters to learn more useful local time-frequency cues, which is exactly what v8 lacked.
4. **False Violence rate improved substantially.** Both Speech -> Violence and Ambient -> Violence decreased. This is important for deployment because false alarms are likely the most disruptive user-facing failure.
5. **The new weakness is Violence -> Speech.** Violence precision improved (**0.696 -> 0.820**), but Violence recall dropped (**0.840 -> 0.802**). v9 is more conservative about predicting Violence.
6. **Ambient became very strong.** Ambient F1 reached **0.921**, and Ambient -> Violence nearly disappeared (**157 -> 22**). This is a major practical improvement over v8.

---

## 6. Proposed next optimizations

1. **Adopt v9 as the new TFLite baseline** for the cleaned/enriched dataset. It is both smaller and more accurate than v8.
2. **Tune the Violence/Speech boundary** instead of changing the whole architecture again. The next run should target **Violence -> Speech** without reopening the Speech -> Violence problem.
3. **Review top Violence misses:** `yell_female5.wav`, `angry_201.wav`, `yell_violence0.2_h_piano.wav`, and `argument2.4.wav`. Check whether these files contain lower-energy argument/speech-like regions that are correctly hard, mislabeled, or need more similar training examples.
4. **Consider a thresholding pass for deployment.** If false positives are more costly than missed Violence, v9 is already better. If recall is more important, tune class thresholds or loss weighting before changing the architecture.
5. **Run TFLite/Keras parity and ESP golden-vector checks** before accepting v9 for firmware, because the model architecture changed and the TFLite file is much smaller.

---

## 7. Artifacts in this folder
* `metrics.json`, `confusion_matrix.png`, `classification_errors.csv`
* `error_source_report.csv`, `top_errors_chart.png` (from `notebooks/tflite/error_analysis.ipynb`)
