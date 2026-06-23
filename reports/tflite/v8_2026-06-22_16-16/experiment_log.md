# Version 8 (CNN -> TFLite)
### **Date:** 2026-06-22 16:16
### **Experiment Goal:** Re-train the **v7** ESP-aligned CNN after a targeted dataset cleanup and enrichment pass. The main goal was to reduce false violence detections caused by problematic Speech sources: silence inside Speech files being interpreted as Ambient, and high-pitched / non-violent human vocal sounds being treated as Violence.

---

## 1. What changed vs v7

| Setting | v7 | v8 |
|---------|----|----|
| Preprocess normalization | Peak norm on each **2 s segment** | unchanged |
| Model | `Dense(32)`, Conv2D 16->32, dropout 0.5, **AdamW** | unchanged |
| Dataset cleanup | v7 processed corpus | manual review of problematic files; silence removed from Speech recordings where silent regions behaved like Ambient |
| Dataset enrichment | previous raw set | added hard negative / borderline human sounds: coughs, sneezes, laughter, and high-pitched female voices |
| Split | `grouped_balanced_split`, `random_state=42` | unchanged logic, but changed source inventory and segment counts |
| Training | batch 32, max 50 epochs, EarlyStopping on `val_loss` | unchanged; **18** epochs trained |

### Data-side motivation

The previous error analysis showed that many mistakes were concentrated in a small number of Speech files. The cleanup focused on two failure modes:

1. **Speech files with long silence:** silent fragments in Speech-labeled files were sometimes learned as Ambient-like examples, weakening the Speech class boundary.
2. **Over-sensitive Violence response:** the model often reacted to high-pitched female voices and expressive but non-violent human sounds as Violence. v8 adds more non-violent vocal variety so these sounds are represented in the negative classes.

This makes v8 a dataset robustness experiment rather than an architecture experiment. The CNN architecture and optimizer were intentionally kept fixed to isolate the effect of the data changes.

---

## 2. Segment-level class mix (this run)

**Split sizes:** train **17 279**, val **4 321**, test **5 402**.

**Class proportions per split (Ambient / Speech / Violence):**

| split | Ambient | Speech | Violence |
|-------|--------:|-------:|---------:|
| train | 32.0%   | 34.8%  | 33.2%    |
| val   | 32.0%   | 34.8%  | 33.2%    |
| test  | 32.0%   | 34.8%  | 33.3%    |

The dataset grew vs v7 (**25 528 -> 27 002** total segments). Because source inventory changed, v7/v8 metrics are not a pure same-test-set comparison, but the direction of the regression is still important.

---

## 3. Performance summary

### Validation (EarlyStopping monitor)
* Accuracy: **80.7%** (v7: 73.9%)
* Balanced accuracy: **81.1%** (v7: 74.4%)
* Macro F1: **0.808** (v7: 0.716)

Validation improved strongly vs v7, which suggests that the updated split is easier for the validation partition or that the cleaned examples align better with the validation distribution. This did **not** transfer to the holdout test.

### Holdout test (final evaluation)
* **Accuracy:** **75.5%** (v7: 80.0%)
* **Balanced accuracy:** **75.9%** (v7: 80.2%)
* **Macro F1:** **0.752** (v7: 0.797)
* **Weighted F1:** **0.749** (v7: 0.797)

* **Per-class (test):**

| Class    | Precision | Recall | F1-score | Support | v7 F1 | v7 recall |
|----------|-----------|--------|----------|--------:|------:|----------:|
| Ambient  | 0.790     | 0.866  | 0.826    | 1 728   | 0.80  | 0.87      |
| Speech   | 0.802     | 0.572  | 0.667    | 1 877   | 0.73  | 0.66      |
| Violence | 0.696     | 0.840  | 0.761    | 1 797   | 0.86  | 0.88      |

* **Confusion matrix (test), segments:**

| true \ pred | Ambient | Speech | Violence |
|-------------|--------:|-------:|---------:|
| **Ambient**  | 1 497 |    74 |    157 |
| **Speech**   |   301 | 1 073 |    503 |
| **Violence** |    97 |   191 |  1 509 |

* **Error breakdown**, **1 323** misclassified segments (v7: **1 021**):

| count | pattern              | v7 count |
|------:|----------------------|---------:|
| 503   | Speech -> Violence   | 181      |
| 301   | Speech -> Ambient    | 419      |
| 191   | Violence -> Speech   | 150      |
| 157   | Ambient -> Violence  | 122      |
| 97    | Violence -> Ambient  | 67       |
| 74    | Ambient -> Speech    | 82       |

The intended cleanup reduced **Speech -> Ambient** errors by count despite a larger test set (**419 -> 301**), which is consistent with removing silence from Speech sources. The main regression is **Speech -> Violence** (**181 -> 503**): the model is still over-triggering on some Speech-like human vocal content, and the current architecture may not have enough convolutional capacity to learn the finer time-frequency differences.

---

## 4. Training dynamics and where errors concentrate

### Overfitting / distribution gap
Training ran for **18 epochs**. Validation balanced accuracy is **81.1%**, while test balanced accuracy is **75.9%**, a gap of about **5.1 pp**. This is a larger warning sign than the raw accuracy drop alone: the model appears to fit the updated validation distribution but generalizes worse to the holdout files.

### Top error sources (test)
| source file             | v8 errors | dominant true label | predicted as |
|-------------------------|----------:|---------------------|--------------|
| `conversation_m&f.wav`  | **436**   | Speech              | Violence, Ambient |
| `convo1.mp3`            | **152**   | Speech              | Ambient, Violence |
| `talk_female2.wav`      | **82**    | Speech              | Ambient, Violence |
| `talk1.wav`             | **68**    | Speech              | Ambient, Violence |
| `yell_female5.wav`      | **50**    | Violence            | Speech, Ambient |

`conversation_m&f.wav` is now the dominant failure source. This should be listened to before interpreting the global metric drop as purely architectural; it may contain mixed speech, overlap, pauses, high voices, or label boundaries that expose the current model weakness.

---

## 5. Observations

1. **Data cleanup helped one target problem:** Speech segments are less often collapsed into Ambient, which matches the manual silence-removal goal.
2. **The Violence detector is still too broad:** the largest failure class is now Speech -> Violence. High-energy, high-pitched, emotional, or mixed human speech still lands too close to the Violence decision region.
3. **Accuracy dropped after adding harder cases:** this is expected when the test set becomes more realistic and contains more borderline non-violent vocal events. v8 is a harder, more diagnostic dataset than v7, not simply a worse training run.
4. **Current parameter allocation is likely inefficient:** the first `Dense(32)` layer has **46 112** parameters out of about **51k** total. The convolutional stack has only `Conv2D(16)` and `Conv2D(32)`, so most of the model memory sits after `Flatten`, while the feature extractor remains small.
5. **Human audio needs richer local features:** speech vs scream/violence differs in short time-frequency details. The current CNN may be asking a small convolutional front-end to produce features that are too crude, then relying on the dense layer to memorize them.

---

## 6. Proposed next optimizations

### Next experiment: redistribute parameters from Dense to Conv2D

Keep the TinyML budget, but move capacity into convolutional feature extraction:

| Variant | Conv stack | Classifier head | Why |
|---------|------------|-----------------|-----|
| v9-a | `Conv2D(24)` -> `Conv2D(48)` | smaller `Dense(16)` after `Flatten` | more filters for time-frequency patterns while keeping dense params below v8 |
| v9-b | `Conv2D(16)` -> `Conv2D(32)` -> `Conv2D(48)` | `GlobalAveragePooling2D` + `Dense(16)` | removes the huge Flatten->Dense bottleneck and forces compact convolutional features |
| v9-c | `Conv2D(24)` -> `Conv2D(48)` | `GlobalAveragePooling2D` + `Dense(16)` | strongest shift of capacity toward convolution with smallest classifier head |

Primary acceptance criteria:

1. Recover test balanced accuracy toward **~80%** without exceeding the TinyML memory budget.
2. Reduce **Speech -> Violence** substantially from **503** errors.
3. Preserve Violence recall near the current level, but improve Violence precision above **0.70**.
4. Re-check top sources, especially `conversation_m&f.wav`, `convo1.mp3`, and `talk_female2.wav`.

### Data audit before/after architecture run

Before accepting v9, do a short listening pass on the largest new error source:

* `conversation_m&f.wav` - verify whether Speech label is clean across all segments.
* `talk_female2.wav` - confirm that trimmed silence did not leave borderline emotional / high-pitch parts that should be separate hard examples.
* cough/sneeze/laughter additions - verify target class placement and whether some clips are acoustically closer to Violence than intended.

---

## 7. Artifacts in this folder
* `metrics.json`, `confusion_matrix.png`, `classification_errors.csv`
* `error_source_report.csv`, `top_errors_chart.png` (from `notebooks/tflite/error_analysis.ipynb`)
