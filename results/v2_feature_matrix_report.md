# Stage A V2 Feature Matrix: 800-Image Development Split Report

**Date:** October 11, 2026  
**Target CSV:** `D:\ML-PROJECT\outputs\features\kvasir_stageA_features_v2.csv`  
**Dataset:** Kvasir-SEG (Development Split: exactly 800 images)  
**Pipeline:** StageAFeaturePipelineV2 (ActiveContourSegmenterV2 + Remediated Extractors)  
**Status:** COMPLETE & VALIDATED  

---

## 1. Executive Summary

The Stage A V2 feature extraction pipeline was executed across the full 800-image development split defined in `config/splits_v2.json`. Features were extracted exclusively from active-contour predicted masks. Ground-truth masks in the development set were not loaded during feature extraction, and the held-out 200-image evaluation set was strictly quarantined and never accessed.

### Key Matrix Specifications:
- **Total Rows:** 800 images (0 duplicates, 100% matched to dev split)
- **Total Columns:** 39 (4 metadata columns + 35 engineered feature columns)
- **Accepted Polyp Segmentations:** 514 (64.2%)
- **Rejected / Ambiguous Frames:** 286 (35.8%)
- **Missing / NaN Values:** 0 (100% complete across all 31,200 cells)
- **Infinite Values:** 0
- **Total Runtime:** 1091.7s (1.36s per image)

---

## 2. Segmentation Status Breakdown


> [!IMPORTANT]
> Rejected masks receive zero values (`0.0`) for all feature descriptors as part of the documented > missing-value strategy. The `segmentation_status` and `segmentation_success` columns distinguish > these rows from genuine anatomical measurements. Downstream classifiers must account for these indicator columns.


| Segmentation Status | Sample Count | Percentage |
| --- | --- | --- |
| valid_candidate | 514 | 64.2% |
| rejected_candidate | 271 | 33.9% |
| no_valid_component | 15 | 1.9% |


---

## 3. Data Integrity & Verification Checklist


| Verification Criterion | Requirement | Observed Metric | Status |
|---|---|---|---|
| **Exact Row Count** | Exactly 800 rows | 800 rows | **PASS** |
| **Exact Column Count** | Exactly 39 columns | 39 columns | **PASS** |
| **Filename Uniqueness** | 800 unique filenames | 800 unique | **PASS** |
| **Split Alignment** | Match `splits_v2.json['dev']` | Exact 800/800 match | **PASS** |
| **Quarantine Integrity** | Zero eval split leakage | 0 of 200 eval frames accessed | **PASS** |
| **Ground-Truth Isolation** | Features from predicted masks | 0 GT masks referenced in features | **PASS** |
| **No NaN Values** | 0 NaN values | 0 NaN values | **PASS** |
| **No Infinite Values** | 0 Inf values | 0 Inf values | **PASS** |
| **Rejected Row Imputation** | Strictly finite zeroes (0.0) | 286 rows cleanly imputed | **PASS** |
| **V1 Preservation** | V1 CSV unmodified | `outputs/features/kvasir_stageA_features.csv` intact | **PASS** |

---

## 4. Feature Statistics for Accepted Lesions (N = 514)


Below are the summary statistics (Mean, Std, Min, Max) across the 35 features for successfully segmented polyps:


| Feature Name | Mean | Std | Min | Max |
| --- | --- | --- | --- | --- |
| shape_relative_area | 0.0799 | 0.0134 | 0.0294 | 0.1189 |
| shape_relative_perimeter | 0.9183 | 0.0829 | 0.5686 | 1.1805 |
| shape_circularity | 0.5789 | 0.0442 | 0.3933 | 0.6839 |
| shape_axis_ratio | 1.1958 | 0.1588 | 1.0103 | 2.0776 |
| shape_solidity | 0.9200 | 0.0210 | 0.7239 | 0.9598 |
| shape_extent | 0.7225 | 0.0385 | 0.4891 | 0.8142 |
| shape_eccentricity | 0.4996 | 0.1467 | 0.1423 | 0.8765 |
| shape_relative_major_axis | 0.2460 | 0.0222 | 0.1537 | 0.3083 |
| shape_relative_minor_axis | 0.2079 | 0.0233 | 0.1263 | 0.2614 |
| color_rgb_r_mean | 170.8484 | 35.5291 | 68.6983 | 242.5303 |
| color_rgb_r_std | 38.9244 | 16.3643 | 10.2771 | 92.1519 |
| color_rgb_g_mean | 93.1811 | 23.3970 | 32.7430 | 173.8468 |
| color_rgb_g_std | 33.7466 | 12.1886 | 12.1392 | 91.3016 |
| color_rgb_b_mean | 67.2118 | 21.0466 | 15.7781 | 141.2338 |
| color_rgb_b_std | 29.7050 | 11.2831 | 10.4634 | 92.2989 |
| color_hsv_h_circ_mean | 7.6237 | 1.8694 | 0.6480 | 16.3800 |
| color_hsv_h_circ_std | 0.0949 | 0.0624 | 0.0264 | 0.6816 |
| color_hsv_s_mean | 158.3694 | 21.7683 | 96.4108 | 226.6514 |
| color_hsv_s_std | 26.3116 | 9.7049 | 8.6798 | 76.1949 |
| color_hsv_v_mean | 170.8606 | 35.5281 | 68.6983 | 242.5631 |
| color_hsv_v_std | 38.9370 | 16.3496 | 10.6690 | 92.1525 |
| color_lab_l_mean | 122.1811 | 26.7259 | 44.0604 | 195.0037 |
| color_lab_l_std | 33.8718 | 12.5344 | 11.1308 | 84.1721 |
| color_lab_a_mean | 157.0635 | 6.9270 | 139.2441 | 176.9855 |
| color_lab_a_std | 7.8878 | 2.8990 | 2.6373 | 19.9450 |
| color_lab_b_mean | 156.8782 | 7.4063 | 136.9017 | 178.2825 |
| color_lab_b_std | 6.6280 | 2.7400 | 1.2607 | 18.7564 |
| texture_glcm_contrast | 4.7023 | 3.8117 | 0.3952 | 28.9556 |
| texture_glcm_dissimilarity | 1.0247 | 0.4165 | 0.3142 | 3.2621 |
| texture_glcm_homogeneity | 0.6663 | 0.0866 | 0.3760 | 0.8524 |
| texture_glcm_energy | 0.1832 | 0.0511 | 0.0848 | 0.3878 |
| texture_glcm_correlation | 0.8499 | 0.1134 | 0.3296 | 0.9880 |
| texture_lbp_energy | 0.1325 | 0.0308 | 0.0768 | 0.3485 |
| texture_lbp_entropy | 3.9338 | 0.1774 | 2.5430 | 4.2277 |
| texture_lbp_uniform_ratio | 0.6713 | 0.0492 | 0.4287 | 0.7978 |


---

## 5. Artifacts and Downstream Readiness

- **V2 Feature Dataset:** `D:\ML-PROJECT\outputs\features\kvasir_stageA_features_v2.csv`
- **Validation Report:** `D:\ML-PROJECT\results\v2_feature_matrix_report.md`

The 800-image development feature dataset is fully validated and ready for classifier training and evaluation.
