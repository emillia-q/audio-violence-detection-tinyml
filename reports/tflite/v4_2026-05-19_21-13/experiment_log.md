# Version 4 (CNN → TFLite)
### **Date:** 2026-05-19 21:13
### **Experiment Goal:** Reduce model size (smaller `Dense` layer) while keeping the same MFCC pipeline and splits as v3; check whether the smaller network still trains sensibly under the current EarlyStopping setup.

---

## 1. What changed vs v3

* **Architecture:** `Dense(64)` → **`Dense(32)`** (everything else unchanged: `Conv2D` 16→32, dropout 0.5, 3-class softmax).
* **Data & split:** Unchanged from v3 (`grouped_balanced_split`, `random_state=42`).
* **Training (this run):** Adam, batch **32**, up to **50** epochs, **EarlyStopping** on `val_loss`, **patience 5**, `restore_best_weights=True`.

---

## 2. Training dynamics (first run, dense 32)

* Training stopped at **epoch 10** (no `val_loss` improvement for 5 epochs).
* **Best checkpoint:** **epoch 5** — exported weights restored from that epoch (`restore_best_weights=True`).
* Observation: the model **learns more slowly** than v3 (v3 best val ~epoch 11, stop ~epoch 16); with patience 5, EarlyStopping may halt training before `val_loss` has finished improving.

*Test/val metrics and error analysis — to be filled in after `error_analysis`.*

---

## 3. Preliminary conclusions

1. Smaller `Dense(32)` needs more time on val — current **patience 5** may be too aggressive.
2. The first run stopped early (best **epoch 5**, stop **epoch 10**); worth re-running with higher patience before comparing to v3.

---

## 4. Next steps

1. **Dense 32 + patience 10** (instead of 5) — allow more epochs without val improvement before stopping; then test metrics and comparison vs v3.
2. **Dense 48** — middle ground between 32 and 64 (v3); only after evaluating dense 32 with `patience=10`.
3. Optional: TFLite size/latency vs v3 once the architecture is chosen.
