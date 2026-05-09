# Version 3
### **Date:** 2026-05-02 14:10
### **Experiment Goal:** Re-balance the raw dataset by adding more **Violence** and **Ambient** recordings so that, after preprocessing into 2 s segments, class counts move closer to **Speech** (which had dominated segment counts in earlier iterations). Evaluate whether this data-focused intervention improves learning under the same **source-level train/test split** used in v2.

---

## 1. What changed vs v2
* **Dataset growth & balance:** Additional raw audio was added for **Violence** and **Ambient** so the project is less skewed toward **Speech** at the segment level in training.
* **Methodology unchanged:** Same pipeline as v2 — MFCC features, Random Forest baseline, **grouped split by recording `source`** with `stratify` on the source’s class label (`random_state=42`, ~80% sources train / ~20% test). No segment from a source appears in both splits (`Common sources: 0`).
* **Scale:** Roughly **31.3k** training segments and **3.1k** test segments in this run (order of magnitude larger than v2’s logged split).

---

## 2. Segment-level class mix (this run)
**Train (proportion by label):**
* Label 0 (**Ambient**): ~38.3%
* Label 1 (**Speech**): ~38.2%
* Label 2 (**Violence**): ~23.5%

**Test (proportion by label):**
* **Ambient:** ~28.1%
* **Speech:** ~12.4%
* **Violence:** ~59.5%

**Note:** Stratification balances **sources per class**, not **segments per class**. Because Violence sources tend to yield many segments per file, the **test** set can still be **segment-heavy on Violence** even when train looks more balanced — this is expected under a source-level split and affects how raw **accuracy** should be read.

---

## 3. Performance summary (holdout test)
* **Overall accuracy:** 75.02%
* **Per-class (test):**

| Class    | Precision | Recall | F1-score | Support (segments) |
|----------|-----------|--------|----------|--------------------|
| Ambient  | 0.76      | 0.75   | 0.76     | 880                |
| Speech   | 0.42      | 0.88   | 0.56     | 387                |
| Violence | 0.93      | 0.72   | 0.81     | 1863               |

* **Macro avg F1:** ~0.71  
* **Weighted avg F1:** ~0.77  

**Direct numeric comparison to v2 is only qualitative:** v2 reported **~92% accuracy** on a different corpus size and a different test segment distribution. v3’s lower headline accuracy is **not** automatically a regression — it may reflect a **harder, larger, and more diverse** mix plus a **test set dominated by Violence segments**, where mistakes cost more in the global accuracy numerator.

---

## 4. Conclusions
1. **Balancing train segments across Ambient / Speech / Violence largely worked in training** (~38% / ~38% / ~24%), which was the main data goal for this iteration.
2. **Speech remains the brittle class:** high **recall** (0.88) but low **precision** (0.42) — the model **over-predicts Speech** (many false positives). This aligns with v2’s observation that Speech vs Violence / noisy ambience is the hardest boundary.
3. **Violence** shows **strong precision** (0.93) but **moderate recall** (0.72): when the model says Violence it is usually right, but it **misses** a non-trivial fraction of Violence segments — still partly a **confusion with Speech** pattern.
4. **Evaluation:** For the next write-up, prioritize **macro-F1**, **balanced accuracy**, and the **confusion matrix** alongside accuracy, especially when test segment priors differ strongly from train.

---

## 5. Planned next steps (v4 / ongoing)
* **Feature caching:** Save `X`, `y`, and paths (e.g. compressed `.npz` + manifest) after `load_dataset`, and rebuild only when `data/processed` changes — to avoid repeating full MFCC extraction on every notebook run.
* **Speech precision:** Add or curate **hard negatives** (speech-like TV, computer ambience, overlapping voices) and review borderline labels to reduce false Speech.
* **Violence recall:** Continue expanding **clean, representative** Violence sources; spot-check files that v2 flagged (e.g. yell-heavy recordings confused with Speech).
* **Model / decision layer (still lightweight):** Tune Random Forest hyperparameters; calibrate **decision thresholds** using `predict_proba` on a validation fold **by source** to trade precision vs recall per class.
* **Reporting:** Optionally add **source-level** accuracy (vote or majority over segments) next to segment-level metrics for a deployment-relevant view.
