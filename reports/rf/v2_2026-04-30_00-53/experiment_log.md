# Version 2
### **Date:** 2026-04-30 00:53
### **Experiment Goal:** Re-evaluate the Random Forest baseline after fixing train/test leakage by splitting on recording `source` (original file), not on individual 2 s segments.

---

## 1. What changed vs v1
* **Split methodology:** Train/test assignment is now **grouped by `source`** (all segments from one recording stay in either train or test). This removes the main source of optimistic bias from v1, where segments from the same file could appear in both sets.
* **Reproducible imports:** Project is installable as a package (`pyproject.toml`, editable install) so notebooks can import `scripts.splitting` consistently.
* **Segment distribution:** With a source-level split, **segment-level class proportions** in train vs test can diverge strongly from v1, even when source-level stratification is used. That is expected.

---

## 2. Performance summary (holdout test)
* **Overall accuracy:** 91.98%
* **Detailed metrics (test):**
    * **Ambient:** Precision 0.98, Recall 0.98 (5666 support segments in this run’s test split)
    * **Speech:** Precision 0.64, Recall 0.92 (840 support)
    * **Violence:** Precision 0.87, Recall **0.49** (742 support) — **main regression vs v1**

---

## 3. Why the headline metrics moved (expected)
* **Lower accuracy / lower Violence recall vs v1 was expected** after removing leakage: v1 could partially “memorize” recording-specific cues shared across segments from the same file.
* **Speech precision dropped** while recall stayed high: the model is **over-predicting Speech** (many false positives), which is consistent with confusion patterns in noisy / overlapping audio.

---

## 4. Error analysis snapshot (from `error_source_report.csv`)
Top problematic sources in v2 (by error count) include:
* **`yell3.wav`** — many Violence segments misclassified as **Speech**
* **`yell&aggression.wav`** — Violence confused with **Speech** and **Ambient**
* **`Ambience&computer.wav`** — Ambient confused with **Speech/Violence** (likely overlapping non-speech vs speech-like texture)

This reinforces that the bottleneck is **class boundary quality + dataset balance**, not only the classifier.

---

## 5. What to improve next (v3 priorities)
* **Violence data quantity & cleanliness:** add more representative Violence recordings; audit borderline files (yells vs speech-like aggression).
* **Reduce Speech false positives:** collect harder negatives (speech-like ambient / TV / computer ambience) and verify labels.
* **Model-side quick wins (still RF):** try decision thresholds using `predict_proba`, tune RF hyperparameters, and/or revisit MFCC settings (still cheap vs jumping to deep learning).
* **Evaluation hygiene:** keep the **source-level split**; optionally report **source-level metrics** in addition to segment-level metrics for a clearer picture.
