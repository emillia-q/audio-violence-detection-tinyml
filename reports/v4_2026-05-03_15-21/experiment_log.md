# Version 4
### **Date:** 2026-05-03 15:21
### **Experiment Goal:** Continue the data-quality push from v3 — manually review every raw file and trim fragments that do not match the assigned class, push the per-class **segment** balance closer to uniform, and introduce a proper **validation split** so hyperparameter search no longer touches the test set. Sanity-check the corpus for duplicates / cross-class leakage before training.

---

## 1. What changed vs v3
* **Per-file manual curation:** Every raw recording was listened to and segments that did not actually represent the labeled class (e.g. silence-only stretches, a few seconds of speech inside an ambient clip, intro/outro dead air in a violence clip) were cut out at the **raw** level. As a result the corpus shrank in segment count but became cleaner per label.
* **More balanced raw dataset:** Additional curation on **Ambient** and **Violence** brought the **train segment proportions** much closer to uniform than v3 (~32% / 38% / 30% vs v3’s ~38% / 38% / 24%).
* **Validation split introduced:** v4 now carves a separate **VAL** set out of the train/val pool **at the source level** (no source appears in two splits). All hyperparameter search runs on `train → val`, the **test** holdout is touched only at the very end. v3 used only train/test, which conflated tuning and reporting.
* **Hyperparameter search added:** A small grid over `n_estimators / max_depth / min_samples_leaf` is evaluated on val; the configuration with the best **macro-F1** is selected for the final test evaluation. Selected: `n_estimators=100, max_depth=None, min_samples_leaf=1`, `class_weight="balanced"`.
* **Duplicate / leakage audit:** Added `scripts/find_duplicates.py` (3-stage: MD5 → MFCC summary cosine → frame-level alignment) and produced `reports/duplicates_2026-05-03_11-12/` with byte-identical, cross-class and long-overlap pairs. **Findings have not yet been acted on** — see §5.
* **Pipeline / methodology unchanged:** Same MFCC mean+std features, same Random Forest baseline, same source-level grouping, same `random_state=42` for the split.

---

## 2. Segment-level class mix (this run)
**Train (proportion by label):**
* Label 0 (**Ambient**): ~32.0%
* Label 1 (**Speech**): ~37.6%
* Label 2 (**Violence**): ~30.5%

**Val (proportion by label):**
* **Ambient:** ~23.7%
* **Speech:** ~43.0%
* **Violence:** ~33.3%

**Test (proportion by label):**
* **Ambient:** ~26.9%
* **Speech:** ~9.3%
* **Violence:** ~63.9%

**Split sizes:** train **19 662**, val **3 833**, test **2 549** segments (~26k total — smaller than v3’s ~34k, as expected after manual trimming).

**Note:** Stratification still balances **sources per class**, not **segments per class**. With curation the train mix is now genuinely close to balanced, but the **test** set remains heavy on Violence (~64%) and **light on Speech** (only 236 segments). Per-class test metrics — especially Speech — should be read with that small support in mind.

---

## 3. Performance summary

### Validation (used for hyperparameter selection)
* Accuracy: **76.4%**
* Balanced accuracy: **78.7%**
* Macro F1: **76.8%**
* Weighted F1: **76.2%**

### Holdout test (final evaluation)
* **Overall accuracy:** **85.7%** (v3: 75.0%, **+10.7 pp**)
* **Balanced accuracy:** **85.6%**
* **Macro F1:** **78.9%** (v3: ~0.71, **+~0.08**)
* **Weighted F1:** **86.8%** (v3: ~0.77)

* **Per-class (test):**

| Class    | Precision | Recall | F1-score | Support (segments) |
|----------|-----------|--------|----------|--------------------|
| Ambient  | 0.90      | 0.76   | 0.82     | 685                |
| Speech   | 0.47      | 0.92   | 0.62     | 236                |
| Violence | 0.96      | 0.89   | 0.92     | 1 628              |

* **Confusion matrix (test):**

| true \ pred | Ambient | Speech | Violence |
|---|---:|---:|---:|
| **Ambient**  | 521 | 110 |  54 |
| **Speech**   |  12 | 217 |   7 |
| **Violence** |  49 | 133 | 1 446 |

* **Error breakdown** (predicted vs true):

| count | pattern              |
|------:|----------------------|
| 133   | Violence → Speech    |
| 110   | Ambient → Speech     |
|  54   | Ambient → Violence   |
|  49   | Violence → Ambient   |
|  12   | Speech → Ambient     |
|   7   | Speech → Violence    |

The headline accuracy jump is large but partially driven by the Violence-heavy test mix (Violence is the model’s strongest class). **Macro F1 (+~0.08)** and **balanced accuracy (~0.86)** are the cleaner improvement signal — they confirm that the model genuinely got better on the under-represented classes too, not only on the dominant one.

---

## 4. Conclusions
1. **Manual per-file curation paid off.** Cleaner labels + a more balanced train mix moved every aggregate metric noticeably upward (test accuracy +10.7 pp, macro F1 +~0.08, balanced accuracy ~0.86). The cost — fewer total segments — was clearly worth it.
2. **Violence is now the strongest class** (precision 0.96, recall 0.89). Recall improved most vs v3 (0.72 → 0.89), suggesting that trimming non-violent stretches out of violence sources removed a lot of confusing material.
3. **Speech is still the brittle class** — recall went up (0.88 → 0.92) but **precision is essentially unchanged (0.42 → 0.47)**. The model continues to **over-predict Speech**: by far the two largest error buckets are **Violence → Speech (133)** and **Ambient → Speech (110)**. Most “mistakes” on the test set are *“something not-speech got called speech”*.
4. **Speech support is small.** Only 236 Speech segments in test (and 9.3% of test). A few-percent change in Speech metrics here can flip on a handful of sources; treat its numbers as noisy.
5. **Methodology is now defensible.** Train / val / test are source-grouped, hyperparameter search runs on val only, test is touched once. Reports include the confusion matrix and per-class breakdown alongside accuracy — the v3 “next steps for evaluation” item is closed.
6. **Open data-quality risk:** the duplicate-detection script ran but the **report has not been processed yet** — any byte-identical or long cross-class overlapping pairs may still be skewing both train and test.

---

## 5. Planned next steps (before v5)
* **Speech precision (the dominant remaining error mode).** Add or curate **hard negatives** that are speech-adjacent but not speech: shouts inside violence (currently 133 → Speech), TV / radio chatter inside ambient (110 → Speech). If a clip really is speech-on-top-of-something-else, decide one label and apply consistently.
* **Speech support in val / test.** With only 236 Speech segments in test, per-class numbers wobble. Add more **independent Speech sources** so val/test contain ≥ a few hundred Speech segments from several distinct recordings.
* **Decision-threshold calibration.** Use `predict_proba` on **val** to push the Speech decision boundary up (or Violence/Ambient down) to trade some Speech recall for precision; report the precision–recall curve per class on val before committing.
* **Try `class_weight="balanced_subsample"`** in the next hyperparameter sweep — with curated, more balanced data it often behaves better than `"balanced"` because the weighting is recomputed inside each tree’s bootstrap.
* **Feature caching (carry-over from v3).** Persist `X`, `y`, paths and the `target_map` as a versioned `.npz` + manifest after `load_dataset`, keyed on a hash of `data/processed`. Rebuild only when the processed data actually changes — full MFCC extraction over ~26k segments on every notebook run is the slowest cell by far.
* **Source-level reporting.** Add a complementary metric that aggregates segment predictions per recording (majority vote / mean probability) and reports accuracy at the **source** level, which is closer to how the model would actually be used downstream.
* **Re-run v4 evaluation after the duplicate cleanup**, *before* changing the model, so any metric movement at v5 can be cleanly attributed to data hygiene rather than to a new model configuration.
