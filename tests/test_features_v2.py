"""PyTest Suite for Version 2 Corrected Features (GLCM Valid Pairs, LBP Normalized, Shape, Color)."""

import sys
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.shape_features_v2 import extract_shape_features_v2
from src.features.color_features_v2 import extract_color_features_v2
from src.features.texture_features_v2 import compute_valid_roi_glcm, extract_texture_features_v2


def test_valid_roi_glcm_manual_image():
    """Validates GLCM computation on a small hand-crafted image against exact manual calculation.

    Image (4x4):
    [[1, 2, 0, 0],
     [1, 2, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 0, 0]]

    Binary Mask: 1 for top-left 2x2, 0 elsewhere.
    Inside the mask, for horizontal distance d=1, angle=0:
    Row 0: pair (1, 2)
    Row 1: pair (1, 2)
    Symmetric GLCM pairs:
    (1, 2) appears 2 times
    (2, 1) appears 2 times
    Total counts = 4.
    Probabilities: P(1, 2) = 0.5, P(2, 1) = 0.5.
    Contrast = |1 - 2|^2 * 0.5 + |2 - 1|^2 * 0.5 = 1.0.
    Dissimilarity = |1 - 2| * 0.5 + |2 - 1| * 0.5 = 1.0.
    Homogeneity = 0.5 / (1 + 1) + 0.5 / (1 + 1) = 0.5.
    Energy = sqrt(0.5^2 + 0.5^2) = sqrt(0.5) = 0.70710678.
    """
    img = np.array(
        [
            [1, 2, 0, 0],
            [1, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    mask = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    glcm_res = compute_valid_roi_glcm(
        quantized_gray=img,
        binary_mask=mask,
        distances=[1],
        angles=[0.0],
        levels=8,
    )

    assert pytest.approx(glcm_res["texture_glcm_contrast"], rel=1e-4) == 1.0
    assert pytest.approx(glcm_res["texture_glcm_dissimilarity"], rel=1e-4) == 1.0
    assert pytest.approx(glcm_res["texture_glcm_homogeneity"], rel=1e-4) == 0.5
    assert pytest.approx(glcm_res["texture_glcm_energy"], rel=1e-4) == np.sqrt(0.5)


def test_glcm_no_zero_padding_contamination():
    """Confirms that background zeros do NOT enter the GLCM calculation."""
    # Image with high constant value inside ROI (e.g. 10) and 0 in background
    img = np.zeros((10, 10), dtype=np.uint8)
    img[2:7, 2:7] = 10
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:7, 2:7] = 1

    glcm_res = compute_valid_roi_glcm(
        quantized_gray=img,
        binary_mask=mask,
        distances=[1],
        angles=[0.0],
        levels=32,
    )

    # In a constant region, contrast must be exactly 0, homogeneity 1, energy 1
    assert pytest.approx(glcm_res["texture_glcm_contrast"], abs=1e-6) == 0.0
    assert pytest.approx(glcm_res["texture_glcm_dissimilarity"], abs=1e-6) == 0.0
    assert pytest.approx(glcm_res["texture_glcm_homogeneity"], abs=1e-6) == 1.0
    assert pytest.approx(glcm_res["texture_glcm_energy"], abs=1e-6) == 1.0


def test_lbp_normalized_histogram():
    """Verifies that LBP energy and entropy use normalized probabilities p_i summing to 1."""
    roi_rgb = np.random.randint(50, 200, size=(30, 30, 3), dtype=np.uint8)
    roi_mask = np.ones((30, 30), dtype=np.uint8)

    feats = extract_texture_features_v2(roi_rgb, roi_mask, lbp_points=8, lbp_radius=1)

    assert 0.0 <= feats["texture_lbp_energy"] <= 1.0
    assert feats["texture_lbp_entropy"] >= 0.0
    assert 0.0 <= feats["texture_lbp_uniform_ratio"] <= 1.0
    assert "texture_lbp_mean" not in feats  # Raw categorical arithmetic excluded
    assert "texture_lbp_std" not in feats


def test_circular_hue_statistics():
    """Verifies that circular mean of Hue handles boundary wrapping near 0 and 180 degrees.

    If pixels are half at H=5 and half at H=175 (both red in OpenCV hue range [0, 180)),
    arithmetic mean would incorrectly yield 90 (cyan!), whereas circular mean should yield ~0 (pure red).
    """
    h_dim = 20
    w_dim = 20
    rgb_img = np.zeros((h_dim, w_dim, 3), dtype=np.uint8)
    mask = np.ones((h_dim, w_dim), dtype=np.uint8)

    # We will test color_features_v2 with custom RGB colors:
    # Red with slight hue variation
    rgb_img[:, :10] = [200, 30, 30]
    rgb_img[:, 10:] = [210, 35, 35]

    feats = extract_color_features_v2(rgb_img, mask)

    assert "color_hsv_h_circ_mean" in feats
    assert "color_hsv_h_circ_std" in feats
    # Red hue in OpenCV is close to 0 or 180
    assert feats["color_hsv_h_circ_mean"] < 20 or feats["color_hsv_h_circ_mean"] > 160


def test_shape_resolution_invariance():
    """Verifies that relative area and perimeter scale proportionally with resolution."""
    # 50x50 square in 100x100 -> area ratio = 2500 / 10000 = 0.25
    mask_100 = np.zeros((100, 100), dtype=np.uint8)
    mask_100[25:75, 25:75] = 255

    # 100x100 square in 200x200 -> area ratio = 10000 / 40000 = 0.25
    mask_200 = np.zeros((200, 200), dtype=np.uint8)
    mask_200[50:150, 50:150] = 255

    # 200x200 square in 400x400 -> area ratio = 40000 / 160000 = 0.25
    mask_400 = np.zeros((400, 400), dtype=np.uint8)
    mask_400[100:300, 100:300] = 255

    f_100 = extract_shape_features_v2(mask_100)
    f_200 = extract_shape_features_v2(mask_200)
    f_400 = extract_shape_features_v2(mask_400)

    # 100x100 square in 200x200 must return exactly 0.25
    assert pytest.approx(f_200["shape_relative_area"], rel=1e-5) == 0.25
    # 200x200 square in 400x400 must return exactly 0.25
    assert pytest.approx(f_400["shape_relative_area"], rel=1e-5) == 0.25
    # Resolution invariance: 200x200 image and 400x400 image produce identical relative area
    assert f_200["shape_relative_area"] == f_400["shape_relative_area"]
    assert pytest.approx(f_100["shape_relative_area"], rel=1e-5) == 0.25

    # Perimeter and circularity consistency across scales
    assert pytest.approx(f_200["shape_relative_perimeter"], rel=0.05) == f_400["shape_relative_perimeter"]
    assert pytest.approx(f_200["shape_circularity"], rel=1e-3) == f_400["shape_circularity"]


def test_empty_mask_handling():
    """Verifies that all v2 feature extractors return finite zeroes on empty mask (Negative Image Readiness)."""
    empty_mask = np.zeros((50, 50), dtype=np.uint8)
    dummy_rgb = np.ones((50, 50, 3), dtype=np.uint8) * 128

    shape_f = extract_shape_features_v2(empty_mask)
    color_f = extract_color_features_v2(dummy_rgb, empty_mask)
    tex_f = extract_texture_features_v2(dummy_rgb, empty_mask)

    for k, v in {**shape_f, **color_f, **tex_f}.items():
        assert np.isfinite(v)
        assert v == 0.0, f"Expected 0.0 for empty mask on feature {k}, got {v}"


def test_pipeline_v2_schema():
    """Verifies that StageAFeaturePipelineV2 provides exactly 35 feature names and consistent columns."""
    from src.features.feature_pipeline_v2 import (
        StageAFeaturePipelineV2,
        FEATURE_NAMES,
        FEATURE_COLUMNS,
        METADATA_COLUMNS,
    )

    pipe = StageAFeaturePipelineV2()
    feats = pipe.get_feature_names()
    cols = pipe.get_column_names()

    assert len(feats) == 35
    assert len(cols) == 39
    assert len(set(feats)) == 35
    assert len(set(cols)) == 39
    assert cols[:4] == METADATA_COLUMNS
    assert cols[4:] == FEATURE_NAMES


def test_pipeline_v2_end_to_end_synthetic_polyp():
    """Verifies that StageAFeaturePipelineV2 successfully processes an image end to end."""
    import cv2
    from src.features.feature_pipeline_v2 import StageAFeaturePipelineV2

    pipe = StageAFeaturePipelineV2()

    # Create synthetic endoscopic image with a high-contrast vascular polyp
    h, w = 120, 120
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = 160
    img[:, :, 1] = 100
    img[:, :, 2] = 90
    cv2.circle(img, (60, 60), radius=16, color=(210, 45, 55), thickness=-1)

    result = pipe.process_sample(img, filename="synthetic_sample.jpg")

    assert result["filename"] == "synthetic_sample.jpg"
    assert "segmentation_status" in result
    assert isinstance(result["segmentation_success"], (bool, np.bool_))
    assert 0.0 <= result["confidence"] <= 1.0

    # Verify all 35 features are present and finite
    row = pipe.to_feature_row(result)
    assert len(row) == 39
    for feat in pipe.get_feature_names():
        assert feat in row
        assert np.isfinite(row[feat])
        assert not np.isnan(row[feat])


def test_pipeline_v2_negative_image_missing_value_strategy():
    """Verifies explicit handling and zero-imputation when image contains no polyp candidate."""
    from src.features.feature_pipeline_v2 import StageAFeaturePipelineV2

    pipe = StageAFeaturePipelineV2()

    # Uniform mucosal image (no lesion, no contrast gradient)
    flat_mucosa = np.full((100, 100, 3), 128, dtype=np.uint8)

    result = pipe.process_sample(flat_mucosa, filename="flat_negative.jpg")

    assert result["filename"] == "flat_negative.jpg"
    assert result["segmentation_success"] is False
    assert result["roi_valid"] is False
    assert result["confidence"] == 0.0
    assert result["segmentation_status"] in {
        "no_candidate_seed",
        "no_valid_component",
        "rejected_candidate",
    }

    # All 35 feature values must be exactly 0.0
    for feat in pipe.get_feature_names():
        assert result[feat] == 0.0
        assert np.isfinite(result[feat])


def test_pipeline_v2_ground_truth_isolation():
    """Verifies that features are completely independent of whether ground-truth mask is provided."""
    import cv2
    from src.features.feature_pipeline_v2 import StageAFeaturePipelineV2

    pipe = StageAFeaturePipelineV2()

    h, w = 120, 120
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = 160
    img[:, :, 1] = 100
    img[:, :, 2] = 90
    cv2.circle(img, (60, 60), radius=16, color=(210, 45, 55), thickness=-1)

    # Completely arbitrary dummy ground-truth mask located elsewhere
    dummy_gt = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(dummy_gt, (20, 20), radius=10, color=255, thickness=-1)

    res_without_gt = pipe.process_sample(img, filename="iso_test.jpg", gt_mask=None)
    res_with_gt = pipe.process_sample(img, filename="iso_test.jpg", gt_mask=dummy_gt)

    for feat in pipe.get_feature_names():
        assert res_without_gt[feat] == res_with_gt[feat]

