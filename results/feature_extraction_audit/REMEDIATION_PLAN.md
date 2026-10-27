# SCIENTIFIC REMEDIATION PLAN: ACTIVE-CONTOUR-GUIDED FEATURE EXTRACTION PIPELINE

**Project:** Active-Contour-Guided Feature Extraction Pipeline for Automated Polyp Classification in Endoscopy  
**Dataset:** Kvasir-SEG (1,000 images, 1,000 ground-truth masks)  
**Status:** PROPOSED REMEDIATION PLAN (Pre-Implementation Review)  
**Author:** Antigravity AI Assistant  
**Date:** September 29, 2026  

---

## 1. EXECUTIVE SUMMARY & OBJECTIVES

Following the comprehensive audit of the baseline pipeline, this remediation plan addresses the root causes of the segmentation failure (Mean Dice 0.315, Precision 21.4%) and feature extraction corruption (14/37 flawed features).

### Core Principles of This Remediation:
1. **Preserve Baseline Integrity:** Zero existing code, outputs, or checkpoints will be overwritten. The baseline remains a benchmark for all comparisons.
2. **Stay Within Active Contour Paradigm:** No deep learning (U-Net, SegNet, Mask R-CNN) or external pre-trained weights. The pipeline remains an **interpretable, handcrafted Active-Contour-guided computer vision system**.
3. **Strict Zero Ground-Truth Leakage:** Ground truth masks will remain quarantined strictly to evaluation metrics (`metrics.py`).
4. **Mathematical Rigor in Features:** Every feature formula will be corrected (resolution invariance for shape, circular directional statistics for Hue, unpadded valid-pair GLCM, and distribution descriptors for LBP).
5. **Reproducibility & Modularity:** All enhancements will integrate seamlessly into the existing configuration and batch execution architecture.

---

## PHASE 1: BASELINE PRESERVATION & VERSIONING STRATEGY

To establish rigorous scientific provenance, the current codebase and its outputs are locked as the baseline.

### 1.1 Versioning Scheme
- **Baseline Version:** `v1_baseline`
- **Remediated Version:** `v2_remediated` (or `stageA_v2`)

### 1.2 Asset Preservation Table
| Asset Type | Baseline Location (Locked) | Remediated Location (New) |
|---|---|---|
| **Segmentation Module** | `src/segmentation/active_contour.py` (preserved as fallback/baseline class) | `src/segmentation/active_contour_v2.py` (or subclass `EnhancedActiveContourSegmenter`) |
| **Feature Extractors** | `src/features/*_features.py` (legacy functions preserved) | `src/features/*_features_v2.py` (or versioned methods) |
| **Pipeline Runner** | `scripts/run_stage_a_pipeline.py` | `scripts/run_stage_a_pipeline_v2.py` |
| **Feature CSV** | `outputs/features/kvasir_stageA_features.csv` | `outputs/features/kvasir_stageA_features_v2.csv` |
| **Metrics CSV** | `outputs/features/kvasir_stageA_segmentation_metrics.csv` | `outputs/features/kvasir_stageA_segmentation_metrics_v2.csv` |
| **Audit Artifacts** | `results/feature_extraction_audit/*` | Preserved permanently in place |
| **Comparison Reports** | N/A | `results/remediation_comparison/` |

---

## PHASE 2: ACTIVE CONTOUR IMPROVEMENT (CANDIDATE INITIALIZATION STRATEGIES)

The audit revealed that LAB A-channel Otsu seeding is the primary point of failure: normal colonic mucosa is pink/red, causing the seed to engulf 40%–80% of the entire frame.

Below is an engineering analysis of **5 candidate approaches** for localized polyp initialization in colonoscopy:

```
┌────────────────────────────────────────────────────────────────────────┐
│               ACTIVE CONTOUR INITIALIZATION INVESTIGATION              │
└────────────────────────────────────────────────────────────────────────┘

  Approach 1: Specular Suppression + Green-Channel Contrast Saliency
  Approach 2: Morphological Top-Hat / Bottom-Hat Multi-Scale Protrusion
  Approach 3: Directional Color-Texture Gradient & Edge Energy Field
  Approach 4: Distance-Transform & Morphological H-Maxima Watershed
  Approach 5: Center-Prior Weighted Morphological Saliency (Hybrid)
```

### Approach 1: Specular Suppression + Green-Channel Contrast Saliency
- **Why it works in endoscopy:** Hemoglobin exhibits strong optical absorption in the blue-green spectrum ($415\text{--}540\text{ nm}$). Vascularized polyp tissue appears with distinct dark contrast in the Green channel relative to surrounding mucosal background.
- **Mechanism:**
  1. Detect specular highlights ($\text{Intensity} > 245$) and mask/inpaint them.
  2. Compute normalized difference index: $\text{NDI} = \frac{R - G}{R + G + \epsilon}$ or Green-channel CLAHE valley contrast.
  3. Apply adaptive local thresholding (Sauvola or localized Otsu) within the active endoscopic circle.
- **Advantages:** Fast ($\approx 15\text{ ms}$), physiologically grounded in hemoglobin absorption.
- **Weaknesses:** Low contrast on pale, hyperplastic, or flat (Paris class 0-IIb) polyps.
- **Cost:** Very low ($O(N)$).

### Approach 2: Multi-Scale Morphological Top-Hat / Bottom-Hat Contrast
- **Why it works in endoscopy:** Polyps are physical mucosal elevations/protrusions. Morphological opening with disk structuring elements larger than vessel widths isolates elevated tissue structures from flat mucosa.
- **Mechanism:** Dual white/black top-hat transforms across radii $r \in \{15, 30, 45\}$ pixels on the luminance channel.
- **Advantages:** Scale-aware, robust against broad uniform color shifts.
- **Weaknesses:** Folds and haustra also produce morphological top-hat responses.
- **Cost:** Moderate ($O(N \cdot k)$ with structuring element size $k$).

### Approach 3: Edge-Aware Morphological Geodesic Active Contours (MorphGAC)
- **Why it works in endoscopy:** Rather than Chan-Vese ACWE (which assumes piecewise constant region intensity), Geodesic Active Contours (GAC) evolve based on an edge-stopping function:
  $$g(I) = \frac{1}{\sqrt{1 + \alpha \|\nabla I_\sigma\|^2}}$$
  The boundary stops when the gradient magnitude is high.
- **Advantages:** Does not assume uniform interior intensity; naturally halts at mucosal fold edges and polyp borders.
- **Weaknesses:** Highly sensitive to specular reflection edges if reflections are not perfectly inpainted.
- **Cost:** Low to moderate ($10\text{--}30\text{ ms}$).

### Approach 4: Distance Transform & Morphological H-Maxima Seeding
- **Why it works in endoscopy:** Avoids binary thresholding of the entire mucosa by finding prominent regional intensity/contrast peaks ($h$-maxima) followed by distance transform seeding.
- **Mechanism:** Regional contrast map $\to$ $h$-maxima filtering $\to$ component distance transform $\to$ maximum central seed disk.
- **Advantages:** Yields compact, localized seed circles rather than sprawling mucosal masks.
- **Weaknesses:** Requires tuning the $h$ contrast parameter.
- **Cost:** Low.

### Approach 5: Recommended Hybrid Strategy — Specular Inpainting + Vascular-Saliency + Edge-Constrained Active Contour
- **Why this is the recommended primary approach:** It directly solves the two biggest failure modes identified in the audit:
  1. **Specular Reflection Artifacts:** Inpainted using fast Navier-Stokes / Telea method so glare rings do not create false seeds.
  2. **Mucosal Over-Expansion:** A composite contrast map is formed:
     $$S(x, y) = \text{CLAHE}(G) \times \left(1 + \frac{R - G}{R + G + 1}\right) \times (1 - \text{LumenMask})$$
     Adaptive local thresholding yields isolated seed clusters. The candidate seed with the highest regional saliency and center-prior weight is selected and expanded using Morphological Active Contours constrained by iteration damping and an edge-stopping boundary term.
- **Expected Advantages:** Directly increases precision from $21.4\%$ to estimated $55\%\text{--}75\%$, raising Mean Dice from $0.315$ to $>0.55\text{--}0.65$.

---

## PHASE 3: SEGMENTATION POST-PROCESSING DESIGN

The audit proved that simply filling all contours and taking the largest component was insufficient. The improved post-processing workflow:

```
Raw Active Contour Mask
          │
          ▼
1. Exclude Black Frame & Dark Lumen (bitwise AND with active field mask)
          │
          ▼
2. Morphological Opening (clean thin strands & noise specks)
          │
          ▼
3. Connected Component Analysis (extract all candidate components $C_i$)
          │
          ▼
4. Multi-Criteria Component Scoring:
   Score($C_i$) = AreaScore($C_i$) + Saliency($C_i$) + Centrality($C_i$) - BorderPenalty($C_i$)
          │
          ▼
5. Select Top-Scoring Component (or empty if max score < threshold)
          │
          ▼
6. Fill Internal Holes (`cv2.drawContours` strictly on selected component)
          │
          ▼
7. Morphological Smoothing (elliptical closing with $r=5$ to prevent staircasing)
          │
          ▼
Final Refined Polyp Mask
```

### Component Selection Criterion:
Instead of blindly picking the largest contour:
$$\text{Score}(C_i) = \alpha \cdot \log(\text{Area}(C_i) + 1) + \beta \cdot \overline{\Delta E}_{\text{border}}(C_i) + \gamma \cdot \left(1 - \frac{d(c_i, c_{\text{img}})}{R_{\text{frame}}}\right) - \delta \cdot \text{BorderTouch}(C_i)$$
- **Area:** Penalizes tiny noise specks while bounding maximum allowed area ($< 60\%$ of frame).
- **Border Contrast ($\overline{\Delta E}$):** True polyps exhibit higher boundary gradient contrast with surrounding mucosa than random mucosal patches.
- **Centrality:** Polyps are typically steered toward the center $c_{\text{img}}$ of the endoscopic camera field.
- **Border Touch:** Severe penalty if the component touches the dark camera perimeter frame.

---

## PHASE 4: RIGOROUS SEGMENTATION EVALUATION (BASELINE VS IMPROVED)

A full 1,000-image comparative trial will be executed:

### 4.1 Comparative Metrics Matrix
For all 1,000 Kvasir-SEG samples:
- **Metrics:** Dice Similarity Coefficient, IoU (Jaccard), Precision, Recall.
- **Summary Statistics:** Mean, Median, Std Dev, Minimum, Maximum.
- **Quality Bins:**
  - $\text{Dice} < 0.20$
  - $\text{Dice} < 0.30$
  - $\text{Dice} < 0.50$
  - $\text{Dice} \ge 0.50$
  - $\text{Dice} \ge 0.70$

### 4.2 Automated 5-Panel Visual Validation
Representative samples automatically identified based on metric distribution:
- **Best Cases:** Highest Dice improvements over baseline.
- **Median Cases:** Cases around median Dice score.
- **Difficult Cases:** Small, flat, or sessile lesions.
- **Failure Cases:** Lowest performing samples for transparent error analysis.

**Panel Structure:**
1. Original RGB Image
2. Expert Ground-Truth Mask
3. Baseline Active Contour Prediction (Over-segmented)
4. Improved Active Contour Prediction
5. Comparative Overlay (Green = GT, Red = Baseline, Cyan = Improved)

---

## PHASE 5: FEATURE EXTRACTION REPAIR SPECIFICATION

### 5.1 Shape Features (9 Features)
- **Resolution Normalization:**
  - $\text{Relative Area} = \frac{\text{Area}}{H \times W} \in [0, 1]$
  - $\text{Relative Perimeter} = \frac{\text{Perimeter}}{\sqrt{H^2 + W^2}} \in [0, \pi]$
  - $\text{Relative Major Axis} = \frac{\text{MajorAxisLength}}{\sqrt{H^2 + W^2}} \in [0, 1]$
  - $\text{Relative Minor Axis} = \frac{\text{MinorAxisLength}}{\sqrt{H^2 + W^2}} \in [0, 1]$
- **Rotation Invariance:**
  - Replace bounding-box aspect ratio ($W/H$, sensitive to camera tilt) with **inertia-ellipse axis ratio**:
    $$\text{Axis Ratio} = \frac{\text{MajorAxisLength}}{\text{MinorAxisLength} + \epsilon}$$
- **Preserved Invariant Descriptors:**
  - `shape_circularity`, `shape_solidity`, `shape_extent`, `shape_eccentricity`.

### 5.2 Color Features (18 Features)
- **Strict Masking:** Compute pixel statistics strictly over the single selected candidate component (`pred_mask == 255`).
- **Circular Directional Statistics for HSV Hue:**
  OpenCV Hue is $H \in [0, 180)$. Map to circular angle $\theta \in [0, 2\pi)$:
  $$\theta_i = \frac{H_i \cdot \pi}{90}$$
  Vector average components:
  $$\bar{S} = \frac{1}{N}\sum_{i=1}^N \sin(\theta_i), \quad \bar{C} = \frac{1}{N}\sum_{i=1}^N \cos(\theta_i)$$
  Mean resultant length:
  $$\bar{R} = \sqrt{\bar{S}^2 + \bar{C}^2} \in [0, 1]$$
  Circular Mean Hue:
  $$\bar{\theta} = \text{atan2}(\bar{S}, \bar{C}) \pmod{2\pi}, \quad \bar{H}_{\text{circ}} = \frac{\bar{\theta} \cdot 90}{\pi} \in [0, 180)$$
  Circular Standard Deviation / Angular Dispersion:
  $$\sigma_{\text{circ}} = \sqrt{-2 \ln(\max(\bar{R}, 10^{-6}))}$$
- **RGB and LAB Features:**
  - Verify range compliance: RGB $\in [0, 255]$, LAB L $\in [0, 255]$, LAB A $\in [0, 255]$ ($>128$ for red), LAB B $\in [0, 255]$.
  - Ensure zero background pixels contaminate the statistics.

### 5.3 Texture Features (10 Features)
- **Elimination of GLCM Zero-Padding:**
  To compute GLCM on an irregular polyp ROI without zero-padding distortion:
  - We tabulate pixel co-occurrences $(I(x, y), I(x+\Delta x, y+\Delta y))$ **strictly when both coordinates lie inside the binary mask**:
    $$\text{ValidPair}(p_1, p_2) \iff \text{mask}(p_1) = 255 \land \text{mask}(p_2) = 255$$
  - Normalized symmetric GLCM is formed exclusively from valid pairs. Background $(0, 0)$ transitions and boundary step-edges are completely eliminated.
- **GLCM Property Pruning:**
  - Drop `texture_glcm_asm` (exact mathematical square of `energy`). Retain `texture_glcm_energy`.
- **LBP Reformulation:**
  - Uniform LBP with $P=24, R=3$.
  - Remove: `texture_lbp_mean` and `texture_lbp_std` (categorical bin arithmetic).
  - Add:
    - `texture_lbp_energy`: $\sum p_i^2$ (uniformity).
    - `texture_lbp_entropy`: $-\sum p_i \log_2(p_i)$ (Shannon complexity).
    - `texture_lbp_uniform_ratio`: proportion of pixels exhibiting uniform texture patterns vs chaotic non-uniform microvasculature.
    - Dominant pattern bin frequency descriptors.

---

## PHASE 6: FEATURE VALIDATION SPECIFICATION

Every corrected feature will be validated in a dedicated catalog:

| Category | Corrected Feature | New Mathematical Formula | Target Dimension / Unit | Collinearity Resolution |
|---|---|---|---|---|
| **Shape** | `shape_relative_area` | $\text{Area} / (H \times W)$ | Dimensionless $[0, 1]$ | Camera resolution invariance |
| **Shape** | `shape_relative_perimeter` | $\text{Perimeter} / \sqrt{H^2 + W^2}$ | Dimensionless $[0, \pi]$ | Resolution invariance |
| **Shape** | `shape_circularity` | $4\pi \text{Area} / (\text{Perimeter}^2 + \epsilon)$ | Dimensionless $(0, 1]$ | Standard compactness |
| **Shape** | `shape_axis_ratio` | $\text{MajorAxis} / (\text{MinorAxis} + \epsilon)$ | Ratio $\ge 1.0$ | Rotation-invariant elongation |
| **Shape** | `shape_solidity` | $\text{Area} / \text{ConvexHullArea}$ | Dimensionless $(0, 1]$ | Standard boundary convexity |
| **Shape** | `shape_extent` | $\text{Area} / \text{BboxArea}$ | Dimensionless $(0, 1]$ | Rectangular fill factor |
| **Shape** | `shape_eccentricity` | $\sqrt{1 - (b/a)^2}$ | Dimensionless $[0, 1)$ | Moment-based elongation |
| **Shape** | `shape_relative_major_axis`| $\text{MajorAxis} / \sqrt{H^2 + W^2}$ | Dimensionless $[0, 1]$ | Resolution invariance |
| **Shape** | `shape_relative_minor_axis`| $\text{MinorAxis} / \sqrt{H^2 + W^2}$ | Dimensionless $[0, 1]$ | Resolution invariance |
| **Color** | `color_rgb_r_mean`, `_std` | Sample mean / std in mask | $[0, 255]$ | Pure ROI pixels |
| **Color** | `color_rgb_g_mean`, `_std` | Sample mean / std in mask | $[0, 255]$ | Pure ROI pixels |
| **Color** | `color_rgb_b_mean`, `_std` | Sample mean / std in mask | $[0, 255]$ | Pure ROI pixels |
| **Color** | `color_hsv_h_circ_mean` | $\text{atan2}(\bar{S}, \bar{C}) \cdot \frac{90}{\pi}$ | Degrees $[0, 180)$ | Directional circular statistics |
| **Color** | `color_hsv_h_circ_std` | $\sqrt{-2 \ln \bar{R}}$ | Angular dispersion $\ge 0$ | Directional circular statistics |
| **Color** | `color_hsv_s_mean`, `_std` | Sample mean / std in mask | $[0, 255]$ | Saturation |
| **Color** | `color_lab_l_mean`, `_std` | Sample mean / std in mask | $[0, 255]$ | Perceptual lightness |
| **Color** | `color_lab_a_mean`, `_std` | Sample mean / std in mask | $[0, 255]$ | Red-green vascular chromaticity |
| **Color** | `color_lab_b_mean`, `_std` | Sample mean / std in mask | $[0, 255]$ | Yellow-blue chromaticity |
| **Texture**| `texture_glcm_contrast` | Masked valid-pair GLCM contrast | $\ge 0$ | Zero-padding artifact removed |
| **Texture**| `texture_glcm_dissimilarity`| Masked valid-pair GLCM dissimilarity| $\ge 0$ | Boundary step-edge removed |
| **Texture**| `texture_glcm_homogeneity` | Masked valid-pair GLCM homogeneity | $[0, 1]$ | Zero-padding artifact removed |
| **Texture**| `texture_glcm_energy` | Masked valid-pair $\sqrt{\text{ASM}}$ | $[0, 1]$ | Retained (ASM dropped as duplicate)|
| **Texture**| `texture_glcm_correlation` | Masked valid-pair correlation | $[-1, 1]$ | Zero-padding distortion removed |
| **Texture**| `texture_lbp_energy` | $\sum p_i^2$ on mask uniform LBP | $[0, 1]$ | Uniformity descriptor |
| **Texture**| `texture_lbp_entropy` | $-\sum p_i \log_2(p_i)$ on mask LBP | $[0, \log_2(26)]$ | Shannon texture complexity |
| **Texture**| `texture_lbp_uniform_ratio`| $N_{\text{uniform}} / N_{\text{total}}$ | $[0, 1]$ | Microstructural regularity |

---

## PHASE 7: FINAL FEATURE DATASET GENERATION

The final feature extraction run will generate:
- **File:** `outputs/features/kvasir_stageA_features_v2.csv`
- **Rows:** Exactly 1,000 samples.
- **Columns:**
  1. `filename` (string)
  2. `segmentation_success` (boolean based on true segmentation confidence)
  3. `segmentation_confidence` (float: internal boundary energy/gradient score)
  4. Validated Shape, Color, and Texture feature columns.
- **Guaranteed Isolation:** Zero ground-truth mask columns in the classifier feature set.

---

## PHASE 8: EXPERIMENT COMPARISON DELIVERABLES

A comparative artifact report (`results/remediation_comparison/COMPARISON_REPORT.md`) will be produced containing:
1. **Segmentation Comparative Table:** Mean/Median Dice, IoU, Precision, Recall for `v1_baseline` vs `v2_remediated`.
2. **Scatter & Distribution Plots:** Cumulative Dice distribution curves ($v1$ vs $v2$).
3. **Correlation Heatmaps:** Proving the elimination of artificial GLCM vs extent correlations.
4. **Visual Progression Panels:** 8 high-resolution 5-panel comparisons showing the exact delineation difference.

---

## PHASE 9: DOWNSTREAM CLASSIFICATION READINESS CRITERIA

Before training any classifier (Random Forest, SVM, XGBoost) or adding HyperKvasir, the pipeline must pass these **readiness gates**:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   CLASSIFICATION READINESS GATES                       │
├──────────────────────────────────┬─────────────────────────────────────┤
│ Gate 1: Mean Segmentation Dice   │ MUST be >= 0.55 (vs 0.315 baseline) │
│ Gate 2: Mean Precision           │ MUST be >= 0.50 (vs 21.4% baseline) │
│ Gate 3: Extreme Outliers / NaNs  │ 0 NaNs, 0 Infs, 0 unhandled errors  │
│ Gate 4: GLCM Padding Correlation │ |r(extent, glcm_energy)| MUST be<0.4│
│ Gate 5: Zero Data Leakage        │ 100% verified quarantine of GT masks│
└──────────────────────────────────┴─────────────────────────────────────┘
```

If these gates are passed, the feature matrix is scientifically sound and ready for downstream ML classification experiments.

---

## REQUIRED SUMMARY INFORMATION

### 1. Recommended Technical Approach
**Hybrid Saliency + Edge-Constrained Morphological Active Contour (Approach 5)**:
1. Specular highlight detection and Telea inpainting.
2. Green-channel mucosal valley contrast + dark lumen suppression to generate a physiologically localized seed.
3. Multi-criteria connected-component post-processing with border penalty.
4. Masked valid-pair GLCM, circular Hue statistics, and resolution-normalized shape descriptors.

### 2. Detailed Implementation Steps
1. **Step 1:** Create `src/segmentation/active_contour_v2.py` implementing the localized initialization and multi-criteria component selection.
2. **Step 2:** Create `src/features/roi_extractor_v2.py`, `shape_features_v2.py`, `color_features_v2.py`, and `texture_features_v2.py`.
3. **Step 3:** Implement unit tests in `tests/test_remediation.py` verifying valid-pair GLCM math and circular Hue formulas.
4. **Step 4:** Execute full 1,000-image batch run via `scripts/run_stage_a_pipeline_v2.py`.
5. **Step 5:** Run automated comparative evaluation against `v1_baseline` across all 1,000 samples.
6. **Step 6:** Generate comparative visual panels and compile `COMPARISON_REPORT.md`.

### 3. Files That Need Modification / Addition
- **New Files Created:**
  - `src/segmentation/active_contour_v2.py`
  - `src/features/shape_features_v2.py`
  - `src/features/color_features_v2.py`
  - `src/features/texture_features_v2.py`
  - `src/features/feature_pipeline_v2.py`
  - `scripts/run_stage_a_pipeline_v2.py`
  - `tests/test_remediation.py`
- **Files Lightly Extended (Non-breaking):**
  - `config/config.py` (add `v2` hyperparameters and output filename configs).

### 4. Files That Must Remain Untouched
- `src/segmentation/active_contour.py` (Baseline preserved)
- `src/features/shape_features.py` (Baseline preserved)
- `src/features/color_features.py` (Baseline preserved)
- `src/features/texture_features.py` (Baseline preserved)
- `src/features/feature_pipeline.py` (Baseline preserved)
- `scripts/run_stage_a_pipeline.py` (Baseline preserved)
- `outputs/features/kvasir_stageA_features.csv` (Baseline output preserved)
- `outputs/features/kvasir_stageA_segmentation_metrics.csv` (Baseline metrics preserved)
- `results/feature_extraction_audit/*` (Baseline audit preserved)

### 5. New Files / Results Generated
- `outputs/features/kvasir_stageA_features_v2.csv`
- `outputs/features/kvasir_stageA_segmentation_metrics_v2.csv`
- `results/remediation_comparison/COMPARISON_REPORT.md`
- `results/remediation_comparison/segmentation_comparison.csv`
- `results/remediation_comparison/visual_comparisons/` (Baseline vs Improved 5-panel plots)

### 6. Evaluation Metrics
- **Segmentation:** Dice, IoU, Precision, Recall, Dice distribution tiers ($<0.2, <0.3, <0.5, \ge 0.5$).
- **Features:** Correlation matrix, variance ratio, zero/outlier rates, circular variance.

### 7. Expected Improvement
- **Precision:** Increase from **21.35%** to estimated **50%–70%**.
- **Mean Dice:** Increase from **0.3150** to estimated **0.55–0.68**.
- **Samples with Dice $\ge 0.50$:** Increase from **19.9%** to estimated **60%–75%**.
- **Feature Reliability:** Complete removal of GLCM zero-padding boundary artifacts and Hue circular averaging errors.

### 8. Risks and Possible Failure Cases
- **Flat / Pale Polyps (Paris 0-IIb):** Subtle lesions with minimal hemoglobin contrast may produce faint seeds. Fallback to adaptive regional texture contrast will be implemented.
- **Deep Shadows / Lumen Edges:** Dark lumen folds can mimic lesion boundaries. Lumen suppression masking will mitigate this.
- **Execution Time:** Valid-pair GLCM calculation in Python must be vectorized using NumPy array broadcasting or Cython/Numba to avoid slowdowns.
