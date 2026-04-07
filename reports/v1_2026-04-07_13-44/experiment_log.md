# Version 1
### **Date:** 2026-04-07 13:44
### **Experiment Goal:** Establish a baseline model using Random Forest and default MFCC extraction settings.

---

## 1. Performance Summary
* **Overall Accuracy:** 94.82%
* **Detailed Metrics:**
    * **Ambient:** Precision 0.97, Recall 0.97
    * **Speech:** Precision 0.94, Recall 0.98
    * **Violence:** Precision 0.93, Recall 0.79 (Lowest recall)



## 2. Error Analysis & Observations
Based on the `classification_errors.csv` and `error_source_report.csv`:

* **Top Problematic File:** `yell3.wav` (30 errors) and `yell_test.wav` (26 errors) remain the primary sources of confusion.
* **Class Confusion:** Despite high overall accuracy, 21% of **Violence** segments are being missed (Recall 0.79), mostly being misclassified as **Speech**.



## 3. Action Plan for v2
* **Data Auditing:** Review the segments of `yell3.wav` to see if the labeling is consistent with the audio energy.