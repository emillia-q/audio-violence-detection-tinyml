# Version 7 (CNN → TFLite)
### **Date:** 2026-06-01 20:14
### **Experiment Goal:** Re-train the **v6** CNN (`Dense(32)` + **AdamW**) on a **re-preprocessed** dataset whose amplitude handling matches the **ESP32 streaming pipeline**: peak normalization on each **2 s window**, not on the full source recording before segmentation. Measure whether test metrics stay in the same ballpark as v6 and whether error patterns shift in ways that are acceptable for on-device deployment.

---

## 1. What changed vs v6

| Setting | v6 | v7 |
|---------|----|----|
| **Preprocess normalization** | Peak norm on **entire file**, then slice into 2 s segments | Peak norm on **each 2 s segment** after slicing (ESP-aligned) |
| `scripts/preprocess.py` | `y = librosa.util.normalize(y)` before the sliding window | `segment = librosa.util.normalize(segment)` inside the loop |
| Processed corpus | v6-era `data/processed/` | **Full re-run** of `scripts/preprocess.py` on `data/raw/` (new segment waveforms) |
| Model | `Dense(32)`, Conv2D 16→32, dropout 0.5, **AdamW** | unchanged |
| Split | `grouped_balanced_split`, `random_state=42` | unchanged (same segment **counts**) |
| EarlyStopping | patience **10** (logged) | **20** epochs trained; `metrics.json` lists patience **5** (export field — verify against notebook if auditing) |

### Preprocessing change (deployment motivation)

From **v7 onward**, training data must match how audio is handled on the **ESP**: the device sees one **2 s buffer at a time** and can only normalize that buffer, not the whole file. Earlier versions (v2–v6) normalized the **full recording** first, then cut overlapping 2 s segments. That meant quiet tail segments from a loud file could be boosted differently than the same physical audio would be on-device, and segment-to-segment level could drift within one source. **v7** applies `librosa.util.normalize` per segment so offline MFCCs and the embedded inference path share the same amplitude contract.

* **Segment geometry unchanged:** 16 kHz, **2 s** window, **1 s** hop (50% overlap), same folder layout under `data/processed/{index}_{category}/`.
* **Split inventory unchanged:** train **16 337**, val **4 085**, test **5 106** — same sources and segment counts as v3–v6; only **waveform amplitudes** per segment differ.

---

## 2. Segment-level class mix (this run)

**Split sizes:** train **16 337**, val **4 085**, test **5 106**.

**Class proportions per split (Ambient / Speech / Violence):**

| split | Ambient | Speech | Violence |
|-------|--------:|-------:|---------:|
| train | 30.9%   | 34.3%  | 34.9%    |
| val   | 30.9%   | 34.3%  | 34.8%    |
| test  | 30.9%   | 34.3%  | 34.8%    |

---

## 3. Performance summary

### Validation (EarlyStopping monitor)
* Accuracy: **73.9%** (v6: 83.3%)
* Balanced accuracy: **74.4%** (v6: 83.6%)
* Macro F1: **0.716** (v6: 0.831)

Validation dropped sharply vs v6 while **test** stayed flat — likely a mix of **different segment statistics**, **shorter training** (20 vs 24 epochs), and the usual val/test sensitivity on this split. Test remains the primary comparison for deployment.

### Holdout test (final evaluation)
* **Accuracy:** **80.0%** (v6: 79.8%, v3: 80.6%)
* **Balanced accuracy:** **80.2%** (v6: 80.1%, v3: 80.9%)
* **Macro F1:** **0.797** (v6: 0.797, v3: 0.805)
* **Weighted F1:** **0.797** (v6: 0.798, v3: 0.806)

* **Per-class (test):**

| Class    | Precision | Recall | F1-score | Support | v6 F1 | v6 recall |
|----------|-----------|--------|----------|--------:|------:|----------:|
| Ambient  | 0.74      | 0.87   | 0.80     | 1 576   | 0.79  | 0.88      |
| Speech   | 0.83      | 0.66   | 0.73     | 1 751   | 0.73  | 0.69      |
| Violence | 0.84      | 0.88   | 0.86     | 1 779   | 0.87  | 0.83      |

* **Confusion matrix (test), segments:**

| true \\ pred | Ambient | Speech | Violence |
|-------------|--------:|-------:|---------:|
| **Ambient**  | 1 372 |    82 |    122 |
| **Speech**   |   419 | 1 151 |    181 |
| **Violence** |    67 |   150 |  1 562 |

* **Error breakdown**, **1 021** misclassified segments (v6: **1 032**):

| count | pattern              | v6 count |
|------:|----------------------|---------:|
| 419   | Speech → Ambient     | 468      |
| 181   | Speech → Violence    | 81       |
| 150   | Violence → Speech    | 212      |
| 122   | Ambient → Violence   | 66       |
| 82    | Ambient → Speech     | 119      |
| 67    | Violence → Ambient   | 86       |

**Speech → Ambient** improved (**−49**). **Violence → Speech** improved (**−62**). Trade-offs: more **Speech → Violence** (**+100**) and **Ambient → Violence** (**+56**). Aggregate test scores are essentially **on par with v6** despite the preprocessing shift.

---

## 4. Training dynamics and where errors concentrate

### Overfitting and EarlyStopping
Training ran for **20 epochs** (v6: **24**). Validation at export is much lower than test (~74% vs ~80% balanced accuracy), so the val monitor is a weak guide for this run; prefer test + ESP golden-vector checks when accepting v7 for deployment.

### Top error sources (test)
| source file        | v7 errors | v6 errors |
|--------------------|----------:|----------:|
| `talk1.wav`        | **263**   | 272       |
| `talk_child2.wav`  | **102**   | —         |
| `convo1.mp3`       | **69**    | 117       |
| `talk_female2.wav` | **56**    | —         |
| `talk_children.wav`| **50**    | 51        |

`talk1` remains the dominant failure mode. **`convo1`** improved vs v6 (117 → 69). **`talk_child2`** is a new top contributor (102 errors) — worth a listening pass after the amplitude change.

---

## 5. Observations

1. **ESP-aligned preprocessing did not collapse accuracy:** Test balanced accuracy **80.2%** vs v6 **80.1%** — within noise. The pipeline change is **safe to treat as the new training default** for anything that must match firmware.
2. **Confusion profile shifted, totals did not:** Fewer Speech→Ambient and Violence→Speech errors, but more Speech→Violence and Ambient→Violence. Violence **recall** rose (**0.83 → 0.88**) at the cost of **precision** (**0.91 → 0.84**); Speech **precision** rose (**0.78 → 0.83**) while **recall** dipped slightly (**0.69 → 0.66**).
3. **Per-window normalization changes relative loudness across segments** from the same file (e.g. a loud shout vs a quiet breath in one recording now normalize independently). That is intentional for ESP but can re-open borderline confusions on mixed-content Speech sources.
4. **Validation gap vs v6** is large; before tuning further, confirm **TFLite vs Keras parity** on test and run **on-device golden MFCC** checks with the new normalized segments.
5. **v6 weights are not interchangeable with v7 data** — any comparison across versions must use the preprocessing that model was trained on.

---

## 6. Proposed next optimizations

1. **Adopt v7 as the deployment training baseline** — keep per-segment normalization in `scripts/preprocess.py` and document the same rule in the ESP capture path.
2. **Data review on `talk1`, `talk_child2`, `talk_female2`** — same manual trim strategy as v3; errors are still concentrated on a handful of Speech-labelled sources.

---

## 7. Artifacts in this folder
* `metrics.json`, `confusion_matrix.png`, `classification_errors.csv`
* `error_source_report.csv`, `top_errors_chart.png` (from `notebooks/tflite/error_analysis.ipynb`)
