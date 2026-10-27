# Stage A V2 Feature Pipeline: 5-Image Development Verification Report

**Date:** October 11, 2026  
**Pipeline:** StageAFeaturePipelineV2 (MorphGAC + Vascular Saliency + Remediated Features)  
**Dataset:** Kvasir-SEG (Development Split: first 5 established samples)  
**Status:** VERIFIED & VALIDATED  

---

## 1. Executive Summary

The Version 2 feature extraction pipeline (`src/features/feature_pipeline_v2.py`) was integrated and executed on the first 5 images of the established development split (`config/splits_v2.json`). All pipeline components operated end to end with zero data leakage, consistent output dimensions, and strict mathematical validity.

### Key Verification Metrics:
- **Total Images Verified:** 5
- **Exact Feature Count:** 35 engineered descriptors (9 Shape, 18 Color, 8 Texture)
- **Total CSV Columns:** 39 (4 metadata columns + 35 feature columns)
- **Duplicate Filenames:** 0
- **Missing / NaN Values:** 0
- **Infinite Values:** 0
- **Ground-Truth Isolation:** 100% (Strictly zero GT mask input during prediction/feature extraction)
- **Missing-Value Strategy:** Explicit zero-imputation (0.0) with diagnostic segmentation status

---

## 2. Sample-by-Sample Execution Diagnostics

| index | filename | resolution | segmentation_status | segmentation_success | confidence | roi_valid | fg_ratio_pct | audit_dice | audit_iou | runtime_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | cju0qkwl35piu0993l0dewei2.jpg | 622x529 | no_valid_component | False | 0.0000 | False | 0.0000 | 0.0000 | 0.0000 | 4.7320 |
| 2 | cju0qx73cjw570799j4n5cjze.jpg | 619x529 | valid_candidate | True | 0.3884 | True | 6.2617 | 0.3164 | 0.1879 | 3.6820 |
| 3 | cju0roawvklrq0799vmjorwfv.jpg | 622x528 | valid_candidate | True | 0.3456 | True | 7.4314 | 0.0948 | 0.0497 | 2.7530 |
| 4 | cju0s2a9ekvms080138tjjpxr.jpg | 626x547 | valid_candidate | True | 0.3900 | True | 8.1040 | 0.2249 | 0.1267 | 3.1820 |
| 5 | cju0s690hkp960855tjuaqvv0.jpg | 621x530 | rejected_candidate | False | 0.1432 | False | 0.0000 | 0.0000 | 0.0000 | 2.6800 |


---

## 3. Verification Criteria Checklist


| Verification Criterion | Required Status | Observed Result | Pass/Fail |
|---|---|---|---|
| **Pipeline runs end-to-end** | Complete execution on all samples | All 5 samples processed without uncaught errors | **PASS** |
| **Feature names & count consistent** | Exactly 35 V2 features, 39 columns | 35 feature columns matching schema exactly | **PASS** |
| **No duplicate filenames** | 5 unique filenames | Exactly 5 distinct filenames | **PASS** |
| **No NaN or Infinite values** | 0 NaN, 0 Inf | 0 NaN, 0 Inf across all 175 values | **PASS** |
| **Rejected mask handling** | Documented finite zero-imputation | Clean default features (0.0) on rejected/empty frames | **PASS** |
| **Ground-truth isolation** | GT strictly isolated from features | Features generated strictly from predicted masks | **PASS** |
| **Preservation of V1** | V1 files untouched | `src/features/feature_pipeline.py` & V1 CSVs preserved | **PASS** |

---

## 4. Extracted Feature Preview (Selected Descriptors)


| filename | segmentation_status | confidence | shape_relative_area | shape_circularity | shape_axis_ratio | color_rgb_r_mean | color_hsv_h_circ_mean | texture_glcm_contrast | texture_lbp_entropy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cju0qkwl35piu0993l0dewei2.jpg | no_valid_component | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| cju0qx73cjw570799j4n5cjze.jpg | valid_candidate | 0.3884 | 0.0626 | 0.5740 | 1.2147 | 210.9691 | 5.3972 | 6.6779 | 4.0818 |
| cju0roawvklrq0799vmjorwfv.jpg | valid_candidate | 0.3456 | 0.0743 | 0.5342 | 1.2406 | 136.1166 | 5.3969 | 1.8267 | 4.1546 |
| cju0s2a9ekvms080138tjjpxr.jpg | valid_candidate | 0.3900 | 0.0810 | 0.5370 | 1.4873 | 155.2642 | 6.3709 | 2.1343 | 4.0281 |
| cju0s690hkp960855tjuaqvv0.jpg | rejected_candidate | 0.1432 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |


---

## 5. Canonical V2 Feature Schema (35 Engineered Features)


### Shape Descriptors (9 Features)
Resolution-normalized and rotation-invariant geometric properties derived exclusively from the predicted binary mask:
- `shape_relative_area`
- `shape_relative_perimeter`
- `shape_circularity`
- `shape_axis_ratio`
- `shape_solidity`
- `shape_extent`
- `shape_eccentricity`
- `shape_relative_major_axis`
- `shape_relative_minor_axis`

### Color Descriptors (18 Features)
First- and second-order color statistics with circular Hue directional statistics across RGB, HSV, and LAB spaces:
- `color_rgb_r_mean`
- `color_rgb_r_std`
- `color_rgb_g_mean`
- `color_rgb_g_std`
- `color_rgb_b_mean`
- `color_rgb_b_std`
- `color_hsv_h_circ_mean`
- `color_hsv_h_circ_std`
- `color_hsv_s_mean`
- `color_hsv_s_std`
- `color_hsv_v_mean`
- `color_hsv_v_std`
- `color_lab_l_mean`
- `color_lab_l_std`
- `color_lab_a_mean`
- `color_lab_a_std`
- `color_lab_b_mean`
- `color_lab_b_std`

### Texture Descriptors (8 Features)
Valid-pair GLCM descriptors (no zero-padding step-edge bias) and normalized LBP distribution properties:
- `texture_glcm_contrast`
- `texture_glcm_dissimilarity`
- `texture_glcm_homogeneity`
- `texture_glcm_energy`
- `texture_glcm_correlation`
- `texture_lbp_energy`
- `texture_lbp_entropy`
- `texture_lbp_uniform_ratio`

---

## 6. Output Artifacts

- **Verification CSV:** `D:\ML-PROJECT\results\v2_feature_verification_5images.csv`
- **Validation Report:** `D:\ML-PROJECT\results\v2_feature_verification_report.md`
