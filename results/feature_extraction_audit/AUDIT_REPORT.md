# TECHNICAL AUDIT AND VALIDATION REPORT: ACTIVE-CONTOUR-GUIDED FEATURE EXTRACTION PIPELINE

**Project:** Active-Contour-Guided Feature Extraction Pipeline for Automated Polyp Classification in Endoscopy  
**Dataset:** Kvasir-SEG (1,000 endoscopy images, 1,000 expert ground-truth segmentation masks)  
**Audit Date:** September 29, 2026  
**Auditor:** Antigravity Advanced Agentic AI System  
**Audit Scope:** Technical validation of segmentation quality, feature mathematical correctness, data leakage, CSV integrity, and reproducibility. (No code modifications were made during this audit).

---

## EXECUTIVE SUMMARY & AUDIT VERDICT

### Overall Verdict: **CRITICAL ISSUE (PIPELINE REQUIRES STRUCTURAL REPAIR BEFORE ML CLASSIFICATION)**

While the existing codebase is exceptionally well-engineered in software architecture, modularity, type hints, logging, and crash-resilient batch processing, **it suffers from critical domain-specific mathematical and computer-vision flaws that render the extracted feature matrix invalid for training a downstream polyp classifier**.

```
┌────────────────────────────────────────────────────────────────────────┐
│                          SUMMARY OF VERDICT                            │
├──────────────────────────┬─────────────────────────────────────────────┤
│ 1. Segmentation Quality  │ CRITICAL ISSUE (Mean Dice 0.315, Prec 21.4%)│
│ 2. Feature Mathematics   │ INCORRECT (14 of 37 features flawed)        │
│ 3. ROI Extraction        │ CORRECT BUT NEEDS IMPROVEMENT (Cropping OK) │
│ 4. Data Leakage          │ CORRECT (Zero ground-truth leakage)         │
│ 5. CSV Integrity         │ CORRECT (1000 unique rows, 0 NaN, 0 Inf)    │
│ 6. Reproducibility       │ CORRECT (100% exact numerical match)        │
│ 7. ML Suitability        │ CRITICAL ISSUE (Not suitable for classifier)│
└──────────────────────────┴─────────────────────────────────────────────┘
```

### Key Audit Findings:
1. **Severe Over-Segmentation (Recall 97.8%, Precision 21.3%):** Morphological Chan-Vese active contour is initialized with Otsu thresholding on the LAB A-channel. In endoscopy, the entire colon wall is red/pink mucosa. Consequently, the active contour segments virtually the whole illuminated endoscopic field rather than the polyp. In **80.1% of all images, Dice is below 0.50** (Mean Dice: **0.3150**, Median Dice: **0.2627**).
2. **False `segmentation_success = True` Definition:** All 1,000 samples are labeled `segmentation_success = True` simply because the foreground pixel ratio fell between 0.05% and 95.0% of the image frame. It measures code execution without failure, **not** segmentation success.
3. **Zero-Padding GLCM Texture Artifact:** To compute GLCM texture, pixels outside the polyp mask inside the rectangular bounding box are zeroed out (`cv2.bitwise_and`). This injects artificial black-to-tissue step edges and massive background `(0, 0)` co-occurrences, corrupting contrast, ASM, energy, and dissimilarity. In fact, `texture_glcm_energy` correlates strongly with `shape_extent` ($r = -0.951$) because bounding box emptiness dictates the GLCM energy!
4. **Circularity & Directional Flaws in Hue:** Hue is a circular angular variable in $[0, 180^\circ)$, but `color_hsv_h_mean` uses a linear arithmetic mean. For reddish colon tissue that wraps around $0^\circ / 180^\circ$, this produces nonsensical averages.
5. **Collinear Duplicates:** Multiple feature pairs have $|r| > 0.98$, including `color_hsv_v_mean` vs `color_rgb_r_mean` ($r = 0.9973$) and `texture_glcm_asm` vs `texture_glcm_energy` ($r = 0.9846$, where Energy is literally $\sqrt{\text{ASM}}$).
6. **No Resolution Normalization for Shape:** Shape features (`shape_area`, `shape_perimeter`, axis lengths) are in raw pixels. Because Kvasir-SEG resolutions vary drastically (from $332 \times 487$ to $1072 \times 1920$), these features primarily measure camera resolution rather than lesion anatomy.
7. **Zero Ground-Truth Leakage Confirmed:** The pipeline strictly isolates the ground-truth mask to evaluation metrics (`dice`, `iou`, `precision`, `recall`). Zero ground-truth information enters the feature matrix.
8. **100% Deterministic Reproducibility:** Re-running the pipeline on sample images yields identical features down to machine precision ($\Delta < 10^{-14}$).

---

## PHASE 1: LOCATE THE EXISTING IMPLEMENTATION

The codebase is organized into cleanly partitioned modules. The exact mapping of responsibilities is detailed below:

| Component | File Path | Key Functions / Classes | Responsibilities |
|---|---|---|---|
| **Pipeline Batch Script** | [`scripts/run_stage_a_pipeline.py`](file:///d:/ML-PROJECT/scripts/run_stage_a_pipeline.py) | `process_batch()`, `main()` | Iterates through 1,000 samples, coordinates checkpoints, outputs CSVs. |
| **Pipeline Orchestrator** | [`src/features/feature_pipeline.py`](file:///d:/ML-PROJECT/src/features/feature_pipeline.py) | `StageAFeaturePipeline.process_sample()` | Coordinates Preprocessing $\to$ Segmentation $\to$ ROI $\to$ Feature Extraction. |
| **Preprocessing** | [`src/preprocessing/image_preprocess.py`](file:///d:/ML-PROJECT/src/preprocessing/image_preprocess.py) | `PolypImagePreprocessor.preprocess_for_segmentation()` | Grayscale conversion, Gaussian blur ($5 \times 5, \sigma=1.0$), CLAHE ($2.0$, $8 \times 8$). |
| **Segmentation** | [`src/segmentation/active_contour.py`](file:///d:/ML-PROJECT/src/segmentation/active_contour.py) | `ActiveContourSegmenter.segment()`, `_generate_initial_level_set()` | LAB A-channel Otsu seed generation, Morphological Chan-Vese ACWE (80 iters). |
| **Segmentation Evaluation** | [`src/segmentation/metrics.py`](file:///d:/ML-PROJECT/src/segmentation/metrics.py) | `calculate_dice()`, `calculate_iou()`, `calculate_precision()`, `calculate_recall()` | Post-hoc comparison of predicted mask vs ground-truth mask. |
| **ROI Extractor** | [`src/features/roi_extractor.py`](file:///d:/ML-PROJECT/src/features/roi_extractor.py) | `ROIExtractor.extract_roi()` | Largest connected component bounding box crop, ROI mask extraction. |
| **Shape Features** | [`src/features/shape_features.py`](file:///d:/ML-PROJECT/src/features/shape_features.py) | `extract_shape_features()` | 9 geometric features (area, perimeter, circularity, aspect ratio, extent, solidity, eccentricity, axes). |
| **Color Features** | [`src/features/color_features.py`](file:///d:/ML-PROJECT/src/features/color_features.py) | `extract_color_features()` | 18 color statistics (mean & std across RGB, HSV, LAB). |
| **Texture Features** | [`src/features/texture_features.py`](file:///d:/ML-PROJECT/src/features/texture_features.py) | `extract_texture_features()` | 10 texture descriptors (6 GLCM props, 4 LBP stats). |
| **Configuration** | [`config/config.py`](file:///d:/ML-PROJECT/config/config.py) | `GlobalConfig`, `SegmentationConfig`, `DatasetConfig`, `PathConfig` | Hyperparameters, paths, thresholds, logging setup. |
| **Data Loader** | [`src/data/kvasir_dataset.py`](file:///d:/ML-PROJECT/src/data/kvasir_dataset.py) | `KvasirSEGDataset` | Pairs 1,000 images with masks; dimension verification. |
| **Primary Generated CSVs** | [`outputs/features/kvasir_stageA_features.csv`](file:///d:/ML-PROJECT/outputs/features/kvasir_stageA_features.csv)<br>[`outputs/features/kvasir_stageA_segmentation_metrics.csv`](file:///d:/ML-PROJECT/outputs/features/kvasir_stageA_segmentation_metrics.csv) | N/A | $1000 \times 39$ feature table;<br>$1000 \times 6$ segmentation metrics table. |
| **Intermediate Mask Files** | Disk verification: None saved for the 1,000 batch run. Only 10 sample masks saved under [`outputs/stage_a_validation/contour_optimization/`](file:///d:/ML-PROJECT/outputs/stage_a_validation/contour_optimization/). |
| **Validation Reports** | [`outputs/stage_a_validation/ACTIVE_CONTOUR_ANALYSIS_REPORT.md`](file:///d:/ML-PROJECT/outputs/stage_a_validation/ACTIVE_CONTOUR_ANALYSIS_REPORT.md)<br>[`outputs/features/stageA_summary.txt`](file:///d:/ML-PROJECT/outputs/features/stageA_summary.txt) | Prior analysis | Documentation acknowledging low precision (21.3%) and parameter insensitivity. |

---

## PHASE 2: AUDIT THE SEGMENTATION

### 1. Mechanism Inspection

- **Level Set Initialization:**  
  In [`src/segmentation/active_contour.py`](file:///d:/ML-PROJECT/src/segmentation/active_contour.py#L70-L109), `_generate_initial_level_set()` converts the RGB frame to LAB space, extracts the A-channel (green-red chromaticity), blurs with a $9 \times 9$ Gaussian ($\sigma = 2.0$), and applies Otsu thresholding. Dark corner borders ($< 15$ intensity) are masked out, and morphological opening ($7 \times 7$ ellipse) is applied. If empty, it falls back to a central disk ($r = 0.15 \times \min(H, W)$).
- **Is Initialization Reasonable for Kvasir-SEG?**  
  **NO.** Otsu thresholding assumes a bimodal intensity distribution between foreground and background. In colonoscopy, the colon interior is almost entirely composed of red/pink mucosal folds, specular reflections, lumen shadows, and blood vessels. The A-channel does not isolate polyps from normal mucosa; instead, Otsu selects 40%–80% of the active colon tissue. The initialization starts massively over-expanded.
- **Active Contour Formulation:**  
  Uses `skimage.segmentation.morphological_chan_vese(proc_img, num_iter=80, smoothing=1, lambda1=1.0, lambda2=1.5)`.  
  Chan-Vese Active Contours Without Edges minimizes variance inside ($c_1$) vs outside ($c_2$). Because `proc_img` is a CLAHE-enhanced grayscale image, the bright illuminated mucosa naturally forms one intensity cluster, and the dark unilluminated lumen/corners form the other. The active contour readily snaps to the entire mucosal surface.
- **Connected-Component & Post-Processing Flaw:**  
  In [`src/segmentation/active_contour.py`](file:///d:/ML-PROJECT/src/segmentation/active_contour.py#L169-L175), `cv2.findContours(pred_mask, cv2.RETR_EXTERNAL)` is used to fill all holes (`cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)`). **It fills ALL external contours**, keeping all disconnected islands. It does NOT isolate the largest connected component.
- **Coordinate Alignment & Resizing:**  
  When $\max(H, W) > 512$, the image is downscaled preserving aspect ratio, segmented, and then upscaled using `cv2.INTER_NEAREST`. Coordinates align with the original frame, though nearest-neighbor interpolation introduces staircased contours.
- **Background Treated as Polyp:**  
  **YES.** In the vast majority of images, normal colonic mucosa constitutes 70%–95% of the predicted mask.
- **Definition of `segmentation_success`:**  
  [`src/segmentation/active_contour.py` line 183](file:///d:/ML-PROJECT/src/segmentation/active_contour.py#L183) sets:
  ```python
  success = bool(0.0005 <= fg_ratio <= 0.95)
  ```
  This simply means: "Did the active contour produce foreground between 0.05% and 95.0% of the image?" It does **not** measure whether a polyp was found or whether the mask matches ground truth. Thus, all 1,000 samples show `segmentation_success = True` despite 80.1% failing clinical segmentation thresholds.

---

### 2. Full Dataset Quantitative Segmentation Performance

Evaluated across all 1,000 Kvasir-SEG samples from [`results/feature_extraction_audit/segmentation_metrics.csv`](file:///d:/ML-PROJECT/results/feature_extraction_audit/segmentation_metrics.csv):

| Metric | Mean | Median | Std Dev | Minimum | Maximum |
|---|---|---|---|---|---|
| **Dice Coefficient** | **0.3150** | **0.2627** | 0.2124 | 0.0000 | 0.9658 |
| **IoU (Jaccard Index)** | **0.2091** | **0.1512** | 0.1785 | 0.0000 | 0.9338 |
| **Precision** | **0.2135** | **0.1516** | 0.1889 | 0.0000 | 0.9994 |
| **Recall (Sensitivity)** | **0.9784** | **0.9996** | 0.0641 | 0.0000 | 1.0000 |

### 3. Segmentation Quality Stratification

```
Total Dataset Size: 1,000 samples

Dice < 0.20  :  373 samples (37.30%)  ███████████████
Dice < 0.30  :  569 samples (56.90%)  ███████████████████████
Dice < 0.50  :  801 samples (80.10%)  ████████████████████████████████
Dice >= 0.50 :  199 samples (19.90%)  ████████
```

> [!WARNING]
> **Clinical & Algorithmic Reality:**
> A recall of **97.84%** combined with a precision of **21.35%** is the mathematical fingerprint of severe over-segmentation. The active contour covers almost everything, capturing the true polyp solely because it captures almost the entire endoscopic view.

---

## PHASE 3: VISUAL SEGMENTATION VALIDATION

Representative samples were selected strictly based on quantitative Dice scores across 4 tiers. All 4-panel visual validation figures (Original Image, Ground Truth Mask, Predicted Active Contour Mask, Boundary Overlay) have been generated and saved to [`results/feature_extraction_audit/visual_validation/`](file:///d:/ML-PROJECT/results/feature_extraction_audit/visual_validation/).

| Quality Tier | Image Filename | Dice | IoU | Precision | Recall | Visual Inspection Findings | Saved Visualization Artifact |
|---|---|---|---|---|---|---|---|
| **Good** | `cju17x0j4nfc10993y31pvlgs.jpg` | **0.9658** | 0.9338 | 0.9982 | 0.9355 | Massive polyp occupying >80% of image frame. Active contour succeeded simply because the polyp filled the entire mucosal field. | [`good_cju17x0j4nfc10993y31pvlgs.png`](file:///d:/ML-PROJECT/results/feature_extraction_audit/visual_validation/good_cju17x0j4nfc10993y31pvlgs.png) |
| **Good (Alt)** | `cju17otoe119u0799nqcbl8n1.jpg` | **0.9658** | 0.9338 | 0.9926 | 0.9403 | Similarly large protruding adenoma filling most of the camera view. | [`good_alt_cju17otoe119u0799nqcbl8n1.png`](file:///d:/ML-PROJECT/results/feature_extraction_audit/visual_validation/good_alt_cju17otoe119u0799nqcbl8n1.png) |
| **Medium (Median)** | `cju8bqxxurs6i0850mu7mtef9.jpg` | **0.2628** | 0.1513 | 0.1520 | 0.9702 | Moderate sessile polyp on lower left. Active contour enveloped the entire central and right mucosa. | [`medium_cju8bqxxurs6i0850mu7mtef9.png`](file:///d:/ML-PROJECT/results/feature_extraction_audit/visual_validation/medium_cju8bqxxurs6i0850mu7mtef9.png) |
| **Medium (Alt)** | `cjz14qsk2wci60794un9ozwmw.jpg` | **0.2627** | 0.1512 | 0.1512 | 1.0000 | Flat lesion; predicted mask captured entire circumference of mucosal wall. | [`medium_alt_cjz14qsk2wci60794un9ozwmw.png`](file:///d:/ML-PROJECT/results/feature_extraction_audit/visual_validation/medium_alt_cjz14qsk2wci60794un9ozwmw.png) |
| **Poor** | `cju2t16vuucaq0835xcpsivn2.jpg` | **0.1501** | 0.0812 | 0.0812 | 1.0000 | Small polyp; predicted mask is $12 \times$ larger than ground truth. | [`poor_cju2t16vuucaq0835xcpsivn2.png`](file:///d:/ML-PROJECT/results/feature_extraction_audit/visual_validation/poor_cju2t16vuucaq0835xcpsivn2.png) |
| **Poor (Alt)** | `cju7ejm2l2ncl0801wq6y84nw.jpg` | **0.1495** | 0.0808 | 0.0811 | 0.9565 | Small pedunculated polyp; entire endoscopic lumen mucosa segmented. | [`poor_alt_cju7ejm2l2ncl0801wq6y84nw.png`](file:///d:/ML-PROJECT/results/feature_extraction_audit/visual_validation/poor_alt_cju7ejm2l2ncl0801wq6y84nw.png) |
| **Very Poor** | `cju0tl3uz8blh0993wxvn7ly3.jpg` | **0.0122** | 0.0061 | 0.0061 | 1.0000 | Tiny diminutive polyp ($< 1\%$ of image). Predicted mask covers $94.6\%$ of frame. $99.4\%$ of mask is false positive! | [`very_poor_cju0tl3uz8blh0993wxvn7ly3.png`](file:///d:/ML-PROJECT/results/feature_extraction_audit/visual_validation/very_poor_cju0tl3uz8blh0993wxvn7ly3.png) |
| **Very Poor (Alt)**| `cju76o55nymqd0871h31sph9w.jpg` | **0.0131** | 0.0066 | 0.0066 | 0.8465 | Diminutive lesion; predicted mask spans entire image frame. | [`very_poor_alt_cju76o55nymqd0871h31sph9w.png`](file:///d:/ML-PROJECT/results/feature_extraction_audit/visual_validation/very_poor_alt_cju76o55nymqd0871h31sph9w.png) |

---

## PHASE 4: AUDIT OF ALL 37 FEATURES

Every single feature was evaluated across mathematical correctness, input region, expected range, and actual empirical range from [`results/feature_extraction_audit/feature_audit.csv`](file:///d:/ML-PROJECT/results/feature_extraction_audit/feature_audit.csv).

### Summary of Feature Auditing
- **Total Features Audited:** 37
- **Mathematically Sound (YES):** 23 (62.2%)
- **Mathematically / Methodologically Flawed (NO):** 14 (37.8%)

| # | Feature Name | Category | Formula / Calculation | Input Used | Expected Range | Actual Range (Dataset) | Potential Issue | Correct? | Explanation |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `shape_area` | Shape | `cv2.contourArea(c)` | Largest contour | $[10^3, 5 \times 10^5]$ px | $[70688.0, 1324886.5]$ | Unnormalized across variable camera resolutions ($332 \times 487$ to $1072 \times 1920$); inflated by mucosal over-segmentation | **NO** | Calculated correctly via Green's theorem, but lacks resolution normalization and is severely inflated by mucosal false positives. |
| 2 | `shape_perimeter` | Shape | `cv2.arcLength(c, True)` | Largest contour | $[200, 4000]$ px | $[1352.6, 9795.3]$ | Scale-dependent on sensor resolution; measures mucosal boundary | **NO** | Correct OpenCV function, but unnormalized across image dimensions. |
| 3 | `shape_circularity` | Shape | $\frac{4\pi \cdot \text{area}}{\text{perimeter}^2 + 10^{-6}}$ | Largest contour | $(0.0, 1.0]$ | $[0.0620, 0.9015]$ | Measures mucosal contour circularity; discrete pixel noise | **YES** | Standard isoperimetric quotient implemented correctly. |
| 4 | `shape_aspect_ratio` | Shape | $\frac{\text{width}}{\text{height}}$ from `cv2.boundingRect` | Largest contour | $[0.2, 5.0]$ | $[0.4106, 2.0196]$ | Not rotation invariant; depends on camera rotation angle | **YES** | Standard bounding box aspect ratio formula implemented correctly. |
| 5 | `shape_extent` | Shape | $\frac{\text{area}}{\text{bbox\_w} \times \text{bbox\_h}}$ | Largest contour | $[0.0, 1.0]$ | $[0.3074, 0.9235]$ | Artificially inflated ($>0.75$) due to mucosal overfilling | **YES** | Standard extent formula implemented correctly. |
| 6 | `shape_solidity` | Shape | $\frac{\text{area}}{\text{convex\_hull\_area}}$ | Largest contour | $[0.0, 1.0]$ | $[0.4668, 0.9992]$ | High solidity ($0.89$) reflects smooth convex mucosal field | **YES** | Standard solidity definition implemented correctly. |
| 7 | `shape_eccentricity` | Shape | Regionprops eccentricity | Largest component | $[0.0, 1.0)$ | $[0.2230, 0.9310]$ | Second-moment ellipse of mucosal region | **YES** | Mathematically sound eccentricity derived from central moments. |
| 8 | `shape_major_axis_length` | Shape | Regionprops `axis_major_length` | Largest component | $[50, 1800]$ px | $[344.4, 1527.7]$ | Unnormalized pixel units; highly collinear with area ($r = 0.949$) | **NO** | Correctly called, but unnormalized across variable resolutions. |
| 9 | `shape_minor_axis_length` | Shape | Regionprops `axis_minor_length` | Largest component | $[50, 1200]$ px | $[191.7, 1200.9]$ | Unnormalized pixel units; highly collinear with area ($r = 0.942$) | **NO** | Correctly called, but unnormalized across variable resolutions. |
| 10 | `color_rgb_r_mean` | Color | `np.mean(R[mask > 0])` | RGB frame & mask | $[0, 255]$ | $[100.0, 241.8]$ | Collinear with HSV V-mean ($r = 0.9973$); pools all mask islands | **YES** | Mathematically correct mean of red channel inside mask. |
| 11 | `color_rgb_r_std` | Color | `np.std(R[mask > 0])` | RGB frame & mask | $[0, 128]$ | $[18.7, 86.0]$ | Collinear with HSV V-std ($r = 0.9707$) | **YES** | Standard sample standard deviation calculation. |
| 12 | `color_rgb_g_mean` | Color | `np.mean(G[mask > 0])` | RGB frame & mask | $[0, 255]$ | $[46.5, 174.9]$ | Collinear with LAB L-mean ($r = 0.9479$) | **YES** | Standard sample mean of green channel. |
| 13 | `color_rgb_g_std` | Color | `np.std(G[mask > 0])` | RGB frame & mask | $[0, 128]$ | $[20.2, 73.7]$ | Correlated with LAB L-std ($r = 0.9280$) | **YES** | Standard sample standard deviation. |
| 14 | `color_rgb_b_mean` | Color | `np.mean(B[mask > 0])` | RGB frame & mask | $[0, 255]$ | $[17.3, 150.5]$ | Heavily absorbed by hemoglobin; pools all mask pixels | **YES** | Standard sample mean of blue channel. |
| 15 | `color_rgb_b_std` | Color | `np.std(B[mask > 0])` | RGB frame & mask | $[0, 128]$ | $[16.9, 83.2]$ | Correlated with green std | **YES** | Standard sample standard deviation. |
| 16 | `color_hsv_h_mean` | Color | `np.mean(H[mask > 0])` | HSV frame & mask | $[0, 179]$ | $[4.23, 59.29]$ | **CRITICAL FLAW:** Hue is a circular angular quantity $[0, 180^\circ)$. Linear mean across $0^\circ / 180^\circ$ produces false cyan/green means | **NO** | Circular angular variables require directional statistics (vector mean of sine/cosine), not arithmetic mean. |
| 17 | `color_hsv_h_std` | Color | `np.std(H[mask > 0])` | HSV frame & mask | $[0, 90]$ | $[1.90, 80.37]$ | **CRITICAL FLAW:** Linear std severely overestimates dispersion across red wrap-around | **NO** | Circular variance must be used for circular data. |
| 18 | `color_hsv_s_mean` | Color | `np.mean(S[mask > 0])` | HSV frame & mask | $[0, 255]$ | $[81.4, 225.8]$ | Pools all mask pixels; reflects mucosal saturation | **YES** | Mathematically correct saturation mean. |
| 19 | `color_hsv_s_std` | Color | `np.std(S[mask > 0])` | HSV frame & mask | $[0, 128]$ | $[12.9, 76.8]$ | Correlated with blue std ($r = 0.9205$) | **YES** | Standard sample standard deviation. |
| 20 | `color_hsv_v_mean` | Color | `np.mean(V[mask > 0])` | HSV frame & mask | $[0, 255]$ | $[100.1, 241.9]$ | Collinear with Red mean ($r = 0.9973$) since $V = \max(R,G,B) = R$ | **YES** | Mathematically correct, but redundant duplicate of Red mean. |
| 21 | `color_hsv_v_std` | Color | `np.std(V[mask > 0])` | HSV frame & mask | $[0, 128]$ | $[18.7, 72.9]$ | Collinear with Red std ($r = 0.9707$) | **YES** | Mathematically correct, but redundant duplicate of Red std. |
| 22 | `color_lab_l_mean` | Color | `np.mean(L[mask > 0])` | LAB frame & mask | $[0, 255]$ | $[65.9, 189.2]$ | Measures perceptual lightness inside mask | **YES** | Correct extraction of LAB lightness mean. |
| 23 | `color_lab_l_std` | Color | `np.std(L[mask > 0])` | LAB frame & mask | $[0, 128]$ | $[22.0, 72.2]$ | Correlated with green std | **YES** | Standard standard deviation of lightness. |
| 24 | `color_lab_a_mean` | Color | `np.mean(A[mask > 0])` | LAB frame & mask | $[0, 255]$ | $[138.3, 172.4]$ | Mean $>128$ confirms red tissue; used in AC initialization | **YES** | Correct extraction of magenta-red chromaticity. |
| 25 | `color_lab_a_std` | Color | `np.std(A[mask > 0])` | LAB frame & mask | $[0, 128]$ | $[3.84, 24.14]$ | Captures vascularity variation | **YES** | Standard standard deviation of A-channel. |
| 26 | `color_lab_b_mean` | Color | `np.mean(B[mask > 0])` | LAB frame & mask | $[0, 255]$ | $[140.7, 180.1]$ | Captures yellow chromaticity (bile, lipid tone) | **YES** | Correct extraction of yellow-blue chromaticity. |
| 27 | `color_lab_b_std` | Color | `np.std(B[mask > 0])` | LAB frame & mask | $[0, 128]$ | $[3.19, 21.90]$ | Measures yellow variance | **YES** | Standard standard deviation of B-channel. |
| 28 | `texture_glcm_asm` | Texture | $\sum P(i,j)^2$ | Zero-masked ROI bbox | $[0.0, 1.0]$ | $[0.0155, 0.2859]$ | **CRITICAL FLAW:** Zero-padding background injects massive $(0,0)$ pairs, inflating ASM; collinear with energy ($r = 0.9846$) | **NO** | Corrupted by zero-padding artifact; redundant duplicate of energy. |
| 29 | `texture_glcm_contrast` | Texture | $\sum \|i - j\|^2 P(i,j)$ | Zero-masked ROI bbox | $[0, 961]$ | $[1.71, 23.29]$ | **CRITICAL FLAW:** Sharp artificial edge between zero-padded background and tissue dominates contrast | **NO** | Measures artificial mask boundary step-edge rather than intrinsic tissue texture. |
| 30 | `texture_glcm_correlation` | Texture | Graycoprops `correlation` | Zero-masked ROI bbox | $[-1.0, 1.0]$ | $[0.7492, 0.9763]$ | High self-correlation induced by contiguous background zero-blocks | **NO** | Distorted by zero-padding background. |
| 31 | `texture_glcm_dissimilarity` | Texture | $\sum \|i - j\| P(i,j)$ | Zero-masked ROI bbox | $[0, 31]$ | $[0.4744, 2.4439]$ | Artificial mask step-edge adds spurious dissimilarity | **NO** | Dominated by zero-masking boundary artifact. |
| 32 | `texture_glcm_energy` | Texture | $\sqrt{\text{ASM}}$ | Zero-masked ROI bbox | $[0.0, 1.0]$ | $[0.1239, 0.5345]$ | Artificially inflated by zero-padding; collinear with shape_extent ($r = -0.951$) | **NO** | Corrupted by zero-padding; exact mathematical square root of ASM. |
| 33 | `texture_glcm_homogeneity` | Texture | $\sum \frac{P(i,j)}{1 + \|i - j\|}$ | Zero-masked ROI bbox | $[0.0, 1.0]$ | $[0.4762, 0.8466]$ | Artificially inflated by $(0,0)$ background pairs where $\|0 - 0\| = 0$ | **NO** | Corrupted by zero-padded background pixels. |
| 34 | `texture_lbp_energy` | Texture | $\sum \text{hist}^2$ | Uniform LBP in mask | $[0.0, 1.0]$ | $[0.0789, 0.4018]$ | Collinear with LBP entropy ($r = -0.9931$) and LBP mean ($r = 0.9738$) | **YES** | Mathematically sound histogram energy calculation restricted to mask pixels. |
| 35 | `texture_lbp_entropy` | Texture | $-\sum p \log_2(p)$ | Uniform LBP in mask | $[0, 4.70]$ | $[2.5308, 4.1520]$ | Collinear with LBP energy ($r = -0.9931$) | **YES** | Mathematically sound Shannon entropy of uniform LBP pattern distribution. |
| 36 | `texture_lbp_mean` | Texture | `np.mean(polyp_lbp)` | Uniform LBP in mask | $[0, 25]$ | $[13.93, 20.17]$ | **CONCEPTUAL FLAW:** Uniform LBP codes (0–25) are categorical pattern bins. Taking arithmetic mean of categorical labels has dubious physical meaning | **NO** | Categorical pattern bin indices cannot be meaningfully averaged with linear arithmetic mean. |
| 37 | `texture_lbp_std` | Texture | `np.std(polyp_lbp)` | Uniform LBP in mask | $[0, 13]$ | $[6.72, 9.35]$ | Standard deviation of categorical pattern indices | **NO** | Standard deviation of categorical bin labels is conceptually flawed. |

---

## PHASE 5: DATA LEAKAGE CHECK

A rigorous audit was conducted across all files to trace the propagation of the ground-truth mask:

```
[Allowed Path]
Image ────────────────────────► Preprocessing
                                    │
                                    ▼
                          Active Contour Model
                                    │
                                    ▼
                             Predicted Mask ───► ROI Extractor ──► Features (CSV)
                                    │
[Evaluation-Only Path]              ▼
Image GT Mask ────────────► Metrics (Dice/IoU) ──► kvasir_stageA_segmentation_metrics.csv
```

### Direct Code Inspection:
1. **`src/features/roi_extractor.py`:** Accepts only `(rgb_image, pred_mask)`. Does not accept or reference `gt_mask`.
2. **`src/features/shape_features.py`:** Accepts only `pred_mask`.
3. **`src/features/color_features.py`:** Accepts only `(rgb_image, pred_mask)`.
4. **`src/features/texture_features.py`:** Accepts only `(roi_rgb, roi_mask)`.
5. **`src/features/feature_pipeline.py`:** `gt_mask` is an optional argument passed exclusively to `evaluate_segmentation(pred_mask, gt_mask)`.
6. **`scripts/run_stage_a_pipeline.py`:** Feature rows written to `kvasir_stageA_features.csv` are strictly filtered by key prefixes `shape_`, `color_`, and `texture_`. Metrics (`dice`, `iou`, `precision`, `recall`) are written exclusively to `kvasir_stageA_segmentation_metrics.csv`.

> [!NOTE]
> **Data Leakage Verdict: PASSED (ZERO DATA LEAKAGE)**  
> There is **no direct or indirect data leakage** of the ground-truth mask into the feature extraction process. The feature matrix was generated entirely from the active contour prediction.

---

## PHASE 6: CHECK THE CSV

The primary output file [`outputs/features/kvasir_stageA_features.csv`](file:///d:/ML-PROJECT/outputs/features/kvasir_stageA_features.csv) was audited:

- **Row Count:** Exactly 1,000 rows (plus header). Matches the 1,000 Kvasir-SEG samples.
- **Unique Filenames:** Exactly 1,000 unique filenames. Zero duplicates.
- **Missing / Null Values:** Exactly 0 missing values across all columns.
- **Infinite Values:** Exactly 0 infinite values.
- **Data Types:** Column 1 (`filename`: string), Column 2 (`segmentation_success`: boolean), Columns 3–39 (`float64`).
- **Constant Features:** None (0 features have constant value).
- **Near-Zero Variance ($< 10^{-4}$):** None.

### Collinearity and Redundancy Analysis

From [`results/feature_extraction_audit/correlation_analysis.csv`](file:///d:/ML-PROJECT/results/feature_extraction_audit/correlation_analysis.csv), severe collinearities were detected:

| Feature 1 | Feature 2 | Pearson $r$ | Nature of Collinearity / Flaw |
|---|---|---|---|
| `color_hsv_v_mean` | `color_rgb_r_mean` | **+0.9973** | In colonoscopy, Red is dominant ($V = \max(R,G,B) = R$). Complete redundancy. |
| `texture_lbp_energy` | `texture_lbp_entropy` | **-0.9931** | Histogram energy and Shannon entropy measure identical distribution concentration. |
| `texture_glcm_asm` | `texture_glcm_energy` | **+0.9846** | Mathematical identity: $\text{Energy} = \sqrt{\text{ASM}}$. Completely duplicate feature. |
| `texture_lbp_energy` | `texture_lbp_mean` | **+0.9738** | Artifact of categorical uniform LBP bin ordering. |
| `color_hsv_v_std` | `color_rgb_r_std` | **+0.9707** | Red channel standard deviation duplicate. |
| `texture_lbp_entropy` | `texture_lbp_mean` | **-0.9601** | Categorical bin ordering collinearity. |
| `shape_extent` | `texture_glcm_energy` | **-0.9511** | **PROOF OF GLCM ZERO-PADDING FLAW:** Low extent means more zero-padding in the bounding box, which inflates $(0,0)$ co-occurrences and spikes GLCM energy! |
| `shape_area` | `shape_major_axis_length` | **+0.9489** | Pixel scale collinearity due to lack of image resolution normalization. |
| `color_lab_l_mean` | `color_rgb_g_mean` | **+0.9479** | Green intensity drives human photometric lightness. |
| `shape_area` | `shape_minor_axis_length` | **+0.9424** | Pixel scale collinearity. |
| `color_rgb_b_std` | `color_rgb_g_std` | **+0.9415** | Cross-channel noise coupling. |

---

## PHASE 7: REPRODUCIBILITY TEST

A sample of 10 images across diverse indices was processed through the identical pipeline without code changes, and compared feature-by-feature against [`outputs/features/kvasir_stageA_features.csv`](file:///d:/ML-PROJECT/outputs/features/kvasir_stageA_features.csv).

Results from [`results/feature_extraction_audit/reproducibility_report.csv`](file:///d:/ML-PROJECT/results/feature_extraction_audit/reproducibility_report.csv):

| Sample Index | Image Filename | Existing Dice | Reproduced Dice | Max Absolute Feature $\Delta$ | Status |
|---|---|---|---|---|---|
| 0 | `cju0qkwl35piu0993l0dewei2.jpg` | 0.528252 | 0.528252 | $7.11 \times 10^{-15}$ | **EXACT MATCH** |
| 100 | `cju1f79yhsb5w0993txub59ol.jpg` | 0.025713 | 0.025713 | $2.84 \times 10^{-14}$ | **EXACT MATCH** |
| 250 | `cju2uzabhs6er0993x3aaf87p.jpg` | 0.215753 | 0.215753 | $2.84 \times 10^{-14}$ | **EXACT MATCH** |
| 400 | `cju3u4lxmg59o0755rz42b9en.jpg` | 0.421056 | 0.421056 | $1.78 \times 10^{-15}$ | **EXACT MATCH** |
| 500 | `cju5bdwa3aatx0818b79i18zf.jpg` | 0.180289 | 0.180289 | $4.44 \times 10^{-16}$ | **EXACT MATCH** |
| 650 | `cju6xlygpw7bs0818n691jsq4.jpg` | 0.639055 | 0.639055 | $3.55 \times 10^{-15}$ | **EXACT MATCH** |
| 750 | `cju7do8c72dbo0801vxfzxdc4.jpg` | 0.095398 | 0.095398 | $2.84 \times 10^{-14}$ | **EXACT MATCH** |
| 850 | `cju85rkbnlo1c08503uxcpax1.jpg` | 0.387185 | 0.387185 | $1.42 \times 10^{-14}$ | **EXACT MATCH** |
| 920 | `cju8b4ja9r2s808509d45ma86.jpg` | 0.390902 | 0.390902 | $1.78 \times 10^{-15}$ | **EXACT MATCH** |
| 999 | `ck2da7fwcjfis07218r1rvm95.jpg` | 0.096886 | 0.096886 | $1.42 \times 10^{-14}$ | **EXACT MATCH** |

> [!TIP]
> **Reproducibility Verdict: PASSED (100% DETERMINISTIC)**  
> The existing pipeline is completely deterministic. No stochastic seeds, random initializations, or nondeterministic GPU operations corrupt repeatability.

---

## PHASE 8: DETAILED ISSUE LOG AND VERDICT BREAKDOWN

### Issue Categorization

#### 1. CRITICAL ISSUE: Active Contour Severe Over-Segmentation (LAB Otsu Seeding)
- **File:** [`src/segmentation/active_contour.py`](file:///d:/ML-PROJECT/src/segmentation/active_contour.py)
- **Function:** `_generate_initial_level_set()`
- **Exact Problem:** Otsu thresholding on LAB A-channel captures all pink/red tissue, which comprises the entire illuminated colonic mucosa.
- **Why it is a problem:** In endoscopy, normal mucosa is pink. The seed includes almost the entire image. Morphological Chan-Vese cannot shrink past the mucosa boundary because the outside is black border/lumen.
- **Effect on the ML Project:** In 80.1% of images, the features represent healthy colon mucosa, not polyp pathology. The ML classifier will train on normal colon wall features instead of polyp characteristics.
- **Recommended Fix:** Initialize contours using polyp localization cues (e.g., green-channel valley contrast, specular reflection proximity, morphological saliency, or edge energy) or replace classical Chan-Vese with a modern lightweight attention/contour model.

#### 2. CRITICAL ISSUE: Masked GLCM Zero-Padding Boundary Artifact
- **File:** [`src/features/texture_features.py`](file:///d:/ML-PROJECT/src/features/texture_features.py)
- **Function:** `extract_texture_features()` (lines 74–89)
- **Exact Problem:** `masked_gray = cv2.bitwise_and(roi_gray, roi_gray, mask=proc_mask)`. Non-mask pixels in the rectangular bounding box are set to 0.
- **Why it is a problem:** Black background pixels (level 0) create massive artificial $(0, 0)$ transitions and sharp step-edges with tissue pixels.
- **Effect on the ML Project:** `texture_glcm_contrast`, `dissimilarity`, `asm`, and `energy` measure the shape and empty space of the bounding box rather than tissue micro-texture ($r = -0.951$ with `shape_extent`).
- **Recommended Fix:** Extract GLCM from a pure bounding box crop without zero-masking, or compute GLCM pairs strictly on coordinates where both $(x, y)$ and $(x + dx, y + dy)$ belong to `proc_mask > 0`.

#### 3. INCORRECT: Linear Arithmetic Mean on Circular Hue Angle
- **File:** [`src/features/color_features.py`](file:///d:/ML-PROJECT/src/features/color_features.py)
- **Function:** `extract_color_features()` (lines 78–89)
- **Exact Problem:** `color_hsv_h_mean` uses `np.mean(h_vals)` and `np.std(h_vals)` on OpenCV Hue $[0, 180)$.
- **Why it is a problem:** Red wraps around $0^\circ$ and $180^\circ$. An average of hue 2 and hue 178 yields 90 (cyan/green), which is mathematically false.
- **Effect on the ML Project:** Introduces severe non-linear noise in color descriptors for reddish lesions.
- **Recommended Fix:** Compute circular mean: $\theta = \frac{H \cdot 2\pi}{180}$, $\bar{\theta} = \text{arctan2}(\text{mean}(\sin\theta), \text{mean}(\cos\theta))$, and angular standard deviation.

#### 4. INCORRECT: Unnormalized Shape Metrics Across Variable Resolutions
- **File:** [`src/features/shape_features.py`](file:///d:/ML-PROJECT/src/features/shape_features.py)
- **Function:** `extract_shape_features()`
- **Exact Problem:** `shape_area`, `shape_perimeter`, `shape_major_axis_length`, and `shape_minor_axis_length` are in raw unnormalized pixels.
- **Why it is a problem:** Kvasir-SEG images vary from $332 \times 487$ (161k px) to $1072 \times 1920$ (2M px). A tiny lesion on a 2M px sensor has a larger raw area than a huge polyp on a 161k px sensor.
- **Effect on the ML Project:** The classifier learns camera model/resolution rather than polyp geometry.
- **Recommended Fix:** Normalize by total image area $H \times W$ (relative area) and image diagonal $\sqrt{H^2 + W^2}$ (relative perimeter and axis lengths).

#### 5. INCORRECT: Categorical Bin Indices Treated as Interval Values in LBP
- **File:** [`src/features/texture_features.py`](file:///d:/ML-PROJECT/src/features/texture_features.py)
- **Function:** `extract_texture_features()` (lines 118–119)
- **Exact Problem:** `lbp_mean = float(np.mean(polyp_lbp))` and `lbp_std = float(np.std(polyp_lbp))` where `polyp_lbp` contains uniform LBP pattern codes (0 to 25).
- **Why it is a problem:** Uniform LBP codes are arbitrary categorical bin assignments (sorted by rotational bit transitions), not continuous numerical measurements.
- **Effect on the ML Project:** Adds meaningless, highly correlated pseudo-metrics to the feature table ($r = 0.974$ with energy).
- **Recommended Fix:** Remove LBP mean/std and use normalized LBP histogram bin frequencies directly, or compute LBP energy, entropy, and uniformity.

#### 6. CORRECT BUT NEEDS IMPROVEMENT: Collinear and Redundant Feature Pairs
- **File:** [`src/features/color_features.py`](file:///d:/ML-PROJECT/src/features/color_features.py), [`src/features/texture_features.py`](file:///d:/ML-PROJECT/src/features/texture_features.py)
- **Exact Problem:**
  - `texture_glcm_asm` vs `texture_glcm_energy` ($r = 0.9846$ — identical information $\text{Energy} = \sqrt{\text{ASM}}$).
  - `color_rgb_r_mean` vs `color_hsv_v_mean` ($r = 0.9973$ — $V = \max(R,G,B) = R$).
  - `texture_lbp_energy` vs `texture_lbp_entropy` ($r = -0.9931$).
- **Effect on the ML Project:** Multicollinearity destabilizes linear models and inflates feature importance metrics without adding entropy.
- **Recommended Fix:** Eliminate redundant duplicate columns during feature selection.

#### 7. CORRECT BUT NEEDS IMPROVEMENT: Misleading `segmentation_success` Definition
- **File:** [`src/segmentation/active_contour.py`](file:///d:/ML-PROJECT/src/segmentation/active_contour.py) (line 183)
- **Exact Problem:** `segmentation_success` is flagged `True` for any mask with $0.05\% \le \text{fg\_ratio} \le 95.0\%$.
- **Why it is a problem:** Gives the false impression that 1,000/1,000 samples were successfully segmented, masking the fact that 801/1,000 failed clinical segmentation overlap ($\text{Dice} < 0.5$).
- **Recommended Fix:** Rename to `execution_success` or add an actual confidence/boundary-gradient validation metric.

---

## PHASE 9: ANSWERS TO THE 8 CORE QUESTIONS

### 1. Is the Active Contour implementation correct?
**NO.** While the Python wrapper for `morphological_chan_vese` runs without runtime errors, its application to endoscopic polyp segmentation is fundamentally flawed:
- The LAB A-channel Otsu initialization fails because normal colonic mucosa is pink/red.
- Chan-Vese ACWE assumes two piecewise constant regions, segmenting illuminated mucosa against dark shadows.
- As a result, mean Dice is **0.3150**, and mean precision is only **21.35%**. In **80.1% of images, Dice is under 0.50**.

### 2. Are the extracted 37 features mathematically correct?
**NO.** Only **23 of the 37 features** are mathematically sound:
- **6 GLCM texture features** are corrupted by zero-padding within the bounding box.
- **2 HSV Hue features** are corrupted by linear averaging over a circular variable.
- **4 Shape features** lack camera resolution normalization.
- **2 LBP features** improperly calculate arithmetic means over categorical bin labels.
- Several features are pure duplicate mathematical transformations (Energy vs ASM, Red vs Value).

### 3. Are the features extracted from the predicted polyp ROI?
**PARTIALLY.**
- **Texture features:** Extracted from the bounding box of the largest contour, but ruined by zero-padding non-polyp pixels.
- **Shape features:** Extracted from the largest contour of the predicted mask.
- **Color features:** Extracted by pooling **all** pixels where `pred_mask > 0`, including disconnected satellite noise islands across the entire frame.
- **Critical Caveat:** Because the predicted mask covers normal mucosa in ~80% of images, the "predicted polyp ROI" is actually normal colonic wall.

### 4. Is there any data leakage?
**NO. ZERO DATA LEAKAGE.**
The ground-truth mask is never passed to `ROIExtractor`, `extract_shape_features`, `extract_color_features`, or `extract_texture_features`. It is used strictly in `src/segmentation/metrics.py` for post-hoc validation.

### 5. Are the features suitable for downstream ML classification?
**NO.** Under current conditions, feeding this feature matrix to a classifier will lead to:
1. Training on normal mucosal wall properties rather than polyp pathology.
2. Learning camera sensor resolution rather than lesion size.
3. Severe collinearity artifacts from redundant and zero-padded GLCM features.

### 6. Which features need correction?
1. **GLCM Features (6):** `texture_glcm_contrast`, `dissimilarity`, `homogeneity`, `energy`, `correlation`, `asm` (must fix zero-padding masking).
2. **Hue Features (2):** `color_hsv_h_mean`, `color_hsv_h_std` (must use circular directional statistics).
3. **Shape Scale Features (4):** `shape_area`, `shape_perimeter`, `shape_major_axis_length`, `shape_minor_axis_length` (must normalize by image dimensions).
4. **LBP Statistics (2):** `texture_lbp_mean`, `texture_lbp_std` (replace with histogram bin features).
5. **Collinear Redundancies (2):** Drop `texture_glcm_asm` (keep `energy`) and drop `color_hsv_v_mean`/`std` (already covered by RGB Red channel).

### 7. Which parts of the pipeline are already good?
- **Pipeline Architecture:** Modular structure, clear separation of concerns, clean dataclass configs.
- **Batch Processing & Crash Recovery:** Checkpointing every 100 images, clean logging, failure fallback handlers.
- **Data Loading:** Robust pairing and dimension verification in `kvasir_dataset.py`.
- **Evaluation Metrics Module:** Clean, accurate implementation of Dice, IoU, Precision, Recall in `metrics.py`.
- **Feature Extraction Robustness:** Zero NaNs, zero Infs, 100% reproducible numerical determinism.
- **Circularity, Solidity, Extent, Eccentricity:** Geometrically sound implementations.
- **LAB / RGB Color Means & Stds:** Correctly implemented pixel-wise extractions.

### 8. What should be fixed before we start classification?
1. **Replace or Augment Active Contour Initialization:** Replace LAB A-channel Otsu seeding with polyp-salient localization (e.g., green-channel contrast, edge energy, or color-texture gradient seeds) to constrain the active contour to the actual lesion.
2. **Select Largest Connected Component in Segmentation:** Ensure `pred_mask` isolates only the primary candidate lesion before passing to color and shape extractors.
3. **Fix GLCM Masking:** Compute GLCM only on valid mask pixel pairs without zero-padding the background.
4. **Fix Hue Circular Statistics:** Implement directional mean and angular standard deviation for Hue.
5. **Normalize Shape Descriptors:** Divide area by $H \times W$ and perimeter/axes by $\sqrt{H^2 + W^2}$.
6. **Prune Multicollinear Features:** Remove exact duplicates (`texture_glcm_asm`, `color_hsv_v_mean`).
7. **Redefine `segmentation_success`:** Tie success to actual segmentation confidence or quality rather than mere code non-failure.

---

## ARTIFACTS AND GENERATED AUDIT FILES

All audit data files have been compiled and verified in the results directory:

```
results/
└── feature_extraction_audit/
    ├── AUDIT_REPORT.md                 # This comprehensive technical audit report
    ├── segmentation_metrics.csv        # Complete 1,000-sample Dice, IoU, Precision, Recall metrics
    ├── feature_audit.csv               # 37-feature evaluation table (formulas, ranges, issues, verdict)
    ├── feature_statistics.csv          # Statistical summary of all 37 features (mean, std, skew, zeros)
    ├── correlation_analysis.csv        # Pairwise Pearson correlations sorted by absolute magnitude
    ├── reproducibility_report.csv      # 10-sample verification proving bit-level reproducibility
    └── visual_validation/
        ├── visual_samples_index.csv    # Manifest of visual validation cases and quantitative scores
        ├── good_cju17x0j4nfc10993y31pvlgs.png
        ├── good_alt_cju17otoe119u0799nqcbl8n1.png
        ├── medium_cju8bqxxurs6i0850mu7mtef9.png
        ├── medium_alt_cjz14qsk2wci60794un9ozwmw.png
        ├── poor_cju2t16vuucaq0835xcpsivn2.png
        ├── poor_alt_cju7ejm2l2ncl0801wq6y84nw.png
        ├── very_poor_cju0tl3uz8blh0993wxvn7ly3.png
        └── very_poor_alt_cju76o55nymqd0871h31sph9w.png
```
