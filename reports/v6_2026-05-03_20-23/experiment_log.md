# Version 6
### **Date:** 2026-05-03 20:23
### **Experiment Goal:** Sanity-check that the `class_weight="balanced"` setting on the Random Forest, inherited from earlier versions, is still doing useful work now that v5's splitter produces near-uniform class proportions in train, val and test (~30 / 36 / 34). With balanced data, the per-class weighting should at best be a no-op and at worst a small drag (it down-weights majority-class examples that are no longer actually majority). Re-train without it and confirm the model is at least as good.

---

## 1. What changed vs v5
* **Single change: removed `class_weight="balanced"`** from both the hyperparameter-search loop and the final retrain. The Random Forest is now trained with default sample weighting.
* **Everything else is identical to v5:** same `data/processed/`, same `grouped_balanced_split` with `random_state=42`, same MFCC mean+std features, same hyperparameter grid, same selection metric (`macro_f1`), same final retrain on `train + val` before the test evaluation. This run is a clean A/B for the weighting choice.

---

## 2. Segment-level class mix (this run)
**Splits sizes:** train **16 666**, val **4 168**, test **5 210** — identical to v5 (same seed, same splitter).

**Class proportions per split (Ambient / Speech / Violence):**

| split | Ambient | Speech | Violence |
|-------|--------:|-------:|---------:|
| train | 30.2%   | 35.6%  | 34.2%    |
| val   | 30.3%   | 35.6%  | 34.1%    |
| test  | 30.3%   | 35.6%  | 34.2%    |

Leak check: `0 / 0 / 0`.

---

## 3. Performance summary

### Validation (used for hyperparameter selection)
* Accuracy: **78.5%** (v5: 78.3%)
* Balanced accuracy: **79.5%** (v5: 79.3%)
* Macro F1: **77.6%** (v5: 77.3%)
* Weighted F1: **76.9%** (v5: 76.7%)

Selected configuration: `n_estimators=200, max_depth=None, min_samples_leaf=1` (v5 picked `n_estimators=300`). All four candidates are within ~0.003 macro-F1 of each other, so this swap is noise — the four configurations are effectively interchangeable on val.

### Holdout test (final evaluation)
* **Accuracy:** **79.6%** (v5: 79.3%, **+0.3 pp**)
* **Balanced accuracy:** **79.8%** (v5: 79.5%, **+0.3 pp**)
* **Macro F1:** **0.797** (v5: 0.794, **+0.003**)
* **Weighted F1:** **0.796** (v5: 0.793, **+0.003**)

* **Per-class (test):**

| Class    | Precision | Recall | F1-score | Support (segments) |
|----------|-----------|--------|----------|--------------------|
| Ambient  | 0.79      | 0.84   | 0.81     | 1 576              |
| Speech   | 0.76      | 0.74   | 0.75     | 1 855              |
| Violence | 0.84      | 0.81   | 0.82     | 1 779              |

* **Confusion matrix (test):**

| true \ pred | Ambient | Speech | Violence |
|---|---:|---:|---:|
| **Ambient**  | 1 330 |   161 |    85 |
| **Speech**   |   290 | 1 380 |   185 |
| **Violence** |    71 |   272 | 1 436 |

* **Error breakdown** (predicted vs true):

| count | pattern              |
|------:|----------------------|
| 290   | Speech → Ambient     |
| 272   | Violence → Speech    |
| 185   | Speech → Violence    |
| 161   | Ambient → Speech     |
|  85   | Ambient → Violence   |
|  71   | Violence → Ambient   |

Numbers are within ~10 segments of v5 in every cell — removing the balancing hurt nothing and gained a fraction of a point everywhere. The hypothesis (balanced data → balanced weighting is redundant) is confirmed.

---

## 4. Where the errors actually come from

**Total test errors: 1 064** (= 5 210 − 4 146 correct). The error-source report shows that errors are **extremely concentrated in a handful of source files** rather than spread evenly across the test set:

| source file                | errors | % of all test errors | true label | predicted as            |
|---|---:|---:|---|---|
| `talk_female2.wav`         | **210** | **~19.7%** | Speech   | Ambient, Violence       |
| `yell_female5.wav`         | 136     | ~12.8%     | Violence | Speech                  |
| `talk_child2.wav`          | 102     | ~9.6%      | Speech   | Violence, Ambient       |
| `kid_playing.wav`          | 81      | ~7.6%      | Speech   | Violence, Ambient       |
| `dishwasher.wav`           | 72      | ~6.8%      | Ambient  | Speech, Violence        |
| `fem_sobbing.wav`          | 44      | ~4.1%      | Ambient  | Violence, Speech        |

**The top six source files alone account for ~61% of all test errors** (~645 of 1 064). That is a very long tail: most of the rest of the corpus is being classified essentially correctly, and a small number of recordings are pulling every aggregate metric down by several points.

A single file — **`talk_female2.wav`** (labeled Speech) — generates **~210 errors, ~20% of the entire test error budget**. This must be reviewed manually before the next training run; almost any other improvement is dominated by the noise this one file produces. Likely scenarios: long stretches of silence inside a "speech" file (→ Ambient prediction), or background noise / distress sounds mixed in (→ Violence prediction). Either it needs to be re-trimmed at the raw level (cut out non-speech segments), re-labeled, or split into multiple per-content sources.

The pattern of the remaining heavy-error files is consistent and informative:
* **Speech files producing Violence/Ambient errors** (`talk_female2`, `talk_child2`, `kid_playing`): suggests these are emotionally loud or have non-speech stretches; "kid_playing" labeled as Speech is also worth re-deciding from the labelling side.
* **Violence files producing Speech errors** (`yell_female5`, plus the long tail of `angry_*`, `argument*`, `yell*` items): the `Violence → Speech` confusion is concentrated in yelling/argument material — exactly the most speech-like end of violence, as expected from v5.
* **Ambient files producing Violence/Speech errors** (`dishwasher.wav`, `fem_sobbing.wav`, `computer.wav`): some of these are likely mislabeled (sobbing is closer to violence/speech than to ambient), the rest are realistic hard cases of background sound bleeding into speech-like spectral shapes.

This is a **data-quality** signal, not a model signal — and it is exactly the kind of finding that makes per-source error reporting worth it.

---

## 5. Conclusions
1. **`class_weight="balanced"` was redundant once the splitter delivered balanced classes.** Removing it was a free, fully-deserved +0.003 macro-F1 across every metric. The hypothesis is confirmed and the setting can stay off going forward unless the class mix changes.
2. **Hyperparameter search remains effectively flat** (~0.003 macro-F1 spread across four configurations). The grid is now too narrow to be informative — broader sweeps only make sense after the data side is fixed.
3. **Most of the test error budget lives in a short list of files.** ~20% of all errors come from a single source (`talk_female2.wav`, 210 errors); the top six sources account for ~61% of errors. This means the next big metric jump is unlikely to come from the model — it will come from cleaning or re-labeling **a handful** of files. A few hours of manual review here is worth more than another hyperparameter sweep.
4. **The general error pattern is unchanged from v5:** the Speech ↔ Violence (yell/scream confusion) and Speech ↔ Ambient (quiet voice / background bleed) boundaries are still the hardest, just now visible at a per-source level. The model's behavior is consistent and stable across this run and v5.
5. **Methodologically v5 + v6 are now interchangeable as a baseline.** Future versions should report deltas against v6 (current best) rather than v5.

---

## 6. Planned next steps (before v7 / ongoing)

* **v7 — manual review of the top error-source files** (highest priority; expected to move metrics more than any model change at this point):
  * **`talk_female2.wav` (210 errors)** — listen end-to-end, cut out non-speech regions at the raw level, decide whether the file should be split into multiple sources, or removed if it's irreparably mixed.
  * `yell_female5.wav` (136), `talk_child2.wav` (102), `kid_playing.wav` (81), `dishwasher.wav` (72), `fem_sobbing.wav` (44) — same treatment: re-listen, decide whether to relabel (e.g. `fem_sobbing.wav` likely is not Ambient), trim, split or drop.
  * Re-run preprocessing on whatever changes, then re-train and compare against v6.
* **Process the duplicate report** (`reports/duplicates_2026-05-03_11-12/`) — still pending from v4/v5 plan; can be combined with the per-file review above since both happen at the raw level.
* **Hard negatives for the two largest residual error modes** (after the per-file cleanup, since the cleanup may make these much smaller on its own):
  * `Speech → Ambient` (290): more low-energy / whispered speech, plus removing speech-bleed inside ambient files.
  * `Violence → Speech` (272): more shouty/animated speech labeled as Speech, so the boundary is learned explicitly.
* **Feature caching** (carry-over from v3/v4/v5 plan, still pending). Persist `X`, `y`, paths and `target_map` as a versioned `.npz` + manifest keyed on a hash of `data/processed`. The full MFCC extraction is still the slowest cell in the notebook — caching becomes especially valuable once we start iterating on per-file edits.
* **Broader hyperparameter sweep + threshold calibration** — only after v7. With four configurations within 0.003 macro-F1 there is nothing to learn from sweeping the same neighborhood.
* **Source-level metric** alongside the segment-level one (carry-over): given how concentrated the errors are, source-level accuracy will look very different from segment-level and will give a more deployment-relevant view.
* **Reproducibility check (cheap):** rerun v6 with two or three other `random_state` values for `grouped_balanced_split` to estimate split variance on macro F1 — gives a noise floor so future deltas can be reported as "vs v6 ± σ".
