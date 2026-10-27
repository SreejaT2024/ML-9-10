# EfficientNet-B0 ScopeGuide Shortcut Diagnostic Audit Report

**Task:** Diagnostic Evaluation of ScopeGuide Indicator Reliance on Frozen EfficientNet-B0 Baseline
**Target Split:** Validation Set (N = 320 images; 160 polyp, 160 non-polyp)
**Test Set Quarantine:** Strictly Preserved (400 test images untouched)
**Audit Status:** COMPLETE

---

## 1. Executive Summary & Core Findings

To test whether the pretrained EfficientNet-B0 model learned a Clever Hans shortcut based on the Olympus ScopeGuide graphic in the bottom-left corner of HyperKvasir images, a controlled occlusion experiment was conducted:

1. **Indicator Presence:** The ScopeGuide sub-window is present in **91.9% (147/160)** of validation non-polyp images (98% in cecum, 100% in BBPS 2-3) and in **only 1.2% (2/160)** of polyp images.
2. **Controlled Corner Occlusion:** Neutral black masking was applied to the bottom-left corner `[y: 0.65*h to h, x: 0 to 0.35*w]` on separate diagnostic copies of all 320 validation images without modifying the original datasets or model weights.
3. **Measured Impact:**
   - **Accuracy:** Shifted from **97.81%** to **89.69%** (Delta: -8.12%)
   - **Specificity (Non-Polyp Accuracy):** Shifted from **98.75%** to **81.87%** (Delta: -16.88%)
   - **Sensitivity (Polyp Recall):** Shifted from **96.88%** to **97.50%** (Delta: +0.62%)
   - **ROC-AUC:** Shifted from **0.9984** to **0.9837**

---

## 2. Quantitative Performance Comparison (N = 320)

| Metric | Original Validation Set | Masked Validation Set (Corner Occluded) | Absolute Delta |
| :--- | :--- | :--- | :--- |
| **Accuracy** | **97.81%** (313/320) | **89.69%** (287/320) | **-8.12%** |
| **Precision** | **98.73%** | **84.32%** | **-14.41%** |
| **Recall (Sensitivity)** | **96.88%** (155/160) | **97.50%** (156/160) | **+0.62%** |
| **Specificity** | **98.75%** (158/160) | **81.87%** (131/160) | **-16.88%** |
| **F1-Score** | **0.9779** | **0.9043** | **-0.0736** |
| **ROC-AUC** | **0.9984** | **0.9837** | **-0.0147** |

---

## 3. Confusion Matrix Breakdown

### Original Validation Confusion Matrix
- True Positives (TP): **155**
- False Positives (FP): **2**
- True Negatives (TN): **158**
- False Negatives (FN): **5**

### Masked Validation Confusion Matrix (Corner Occluded)
- True Positives (TP): **156**
- False Positives (FP): **29**
- True Negatives (TN): **131**
- False Negatives (FN): **4**

---

## 4. Subgroup Analysis on Non-Polyp Categories

- **Frames with ScopeGuide (N = 147):**
  - Original Specificity: **100.00%**
  - Masked Specificity: **80.27%**
- **Frames without ScopeGuide (N = 13):**
  - Original Specificity: **84.62%**
  - Masked Specificity: **100.00%**

---

## 5. Limitations of this Diagnostic Experiment

1. **Occlusion as Out-of-Distribution Shift:** Injecting an artificial zero/black bounding block into the bottom-left corner alters the input distribution that the neural network was trained on. A portion of any observed performance drop can arise from the unexpected black block rather than the absence of ScopeGuide information.
2. **Correlated Mucosal Confounders:** HyperKvasir non-polyp images (such as clear bowel preparation BBPS 2-3 and cecum) also have distinctive anatomical landmarks (e.g. appendiceal orifice, ileocecal valve, uniform colonic folds) that differ organically from polyp morphology. Occlusion alone cannot completely isolate anatomical differences from HUD telemetry.
3. **Frozen ImageNet Backbone Features:** Because the feature extractor was trained on ImageNet-1k and only the classification head was trained, the convolutional filters represent general natural image textures. The classification head weighted all spatial pooled features, of which the high-contrast ScopeGuide sub-window is a prominent component.

---

## 6. Generated Diagnostic Artifacts

- **Diagnostic Masked Copies Directory:** `results/scopeguide_audit/masked_val_images/` (320 images)
- **Diagnostic Predictions CSV:** `results/scopeguide_audit/diagnostic_predictions.csv`
- **Metrics Comparison JSON:** `results/scopeguide_audit/metrics_comparison.json`
- **Metrics Comparison CSV:** `results/scopeguide_audit/metrics_comparison.csv`
- **Audit Report:** `results/scopeguide_audit/SCOPEGUIDE_AUDIT_REPORT.md`
