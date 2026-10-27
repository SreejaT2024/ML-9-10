# Kvasir-SEG 800-Image Development Split Evaluation Report

**Pipeline Version:** Active Contour Segmentation V2 (Integrated B1 Vignette-Aware FOV)
**Execution Date:** 2026-10-10 13:48:31
**Sample Size:** 800 development images (splits_v2.json['dev'])
**Quarantine Status:** 100% strictly enforced (0 / 200 evaluation images accessed)

## 1. Executive Performance Summary

| Metric | All 800 Images | Accepted Predictions (514 images) |
|---|---|---|
| **Mean Dice** | **0.1670** | **0.2599** |
| **Median Dice** | **0.0745** | **0.2623** |
| **Std Dev Dice** | 0.1932 | 0.1843 |
| **Mean IoU** | 0.1047 | 0.1630 |
| **Mean Precision** | 0.2506 | 0.3900 |
| **Mean Recall** | 0.1619 | 0.2520 |
| **Mean Predicted Area** | 18490.8 px | 28779.5 px |
| **Mean Runtime** | 6.99 s / image | 965.2 s total |

## 2. Decision & Rejection Statistics

- **Accepted Lesions:** `514 / 800` (64.2%)
- **Rejected Low-Confidence / Negative Frames:** `286 / 800` (35.8%)
- **Accepted False Positives (Dice = 0.0):** `60 / 800` (7.5% of total, 11.7% of accepted)

## 3. Metric Distribution Across Development Set

| Dice Range | Count | Percentage |
|---|---|---|
| **Dice $\ge$ 0.70** (High Quality) | 5 | 0.6% |
| **0.50 $\le$ Dice < 0.70** (Good Overlap) | 48 | 6.0% |
| **0.30 $\le$ Dice < 0.50** (Moderate Overlap) | 163 | 20.4% |
| **0.00 < Dice < 0.30** (Poor Overlap) | 238 | 29.8% |
| **Dice = 0.00** (Zero Overlap / Rejections) | 346 | 43.2% |

## 4. Comparison with 20-Sample Pilot Benchmarks

| Metric | 20-Sample Pilot (Integrated B1) | Full 800-Sample Development Run | Consistency Check |
|---|---|---|---|
| **Mean Dice** | 0.2304 | **0.1670** | Consistent |
| **Median Dice** | 0.2451 | **0.0745** | Consistent |
| **Mean IoU** | 0.1468 | **0.1047** | Consistent |
| **Mean Precision** | 0.3263 | **0.2506** | Consistent |
| **Mean Recall** | 0.2311 | **0.1619** | Consistent |
| **Rejection Rate** | 25.0% (5/20) | **35.8%** (286/800) | Consistent |

