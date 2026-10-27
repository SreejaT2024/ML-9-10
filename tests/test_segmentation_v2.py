"""PyTest Suite for Version 2 Active Contour Segmentation Module.

Validates:
1. Specular highlight detection
2. Specular inpainting fidelity
3. Green-channel vascular saliency output properties
4. Lumen cavity suppression
5. Localized seed generation (no sprawling Otsu/central disk)
6. MorphGAC edge-stopping boundary localization
7. Strict binary mask output values {0, 255}
8. Multi-criteria component scoring
9. Weak centrality constraint verification (does not disqualify off-center polyps)
10. No-polyp empty mask readiness on uniform mucosa
11. Low-confidence rejection behavior
12. Output metadata diagnostics
13. Ground-truth isolation (never referenced or accepted during prediction)
14. Deterministic segmentation output
15. Split isolation (80/20 train/eval mutual exclusivity)
"""

import sys
import inspect
from pathlib import Path
import numpy as np
import cv2
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.active_contour_v2 import (
    ActiveContourSegmenterV2,
    compute_endoscopic_fov_mask,
)
from src.data.splits import get_splits


@pytest.fixture
def segmenter_v2():
    """Fixture returning default ActiveContourSegmenterV2."""
    return ActiveContourSegmenterV2()


@pytest.fixture
def synthetic_mucosa_with_polyp():
    """Generates synthetic endoscopic frame containing mucosa background and localized polyp."""
    h, w = 120, 120
    # Mucosa background: reddish/pink (R=160, G=100, B=90)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = 160
    img[:, :, 1] = 100
    img[:, :, 2] = 90

    # Localized polyp at (60, 60), radius 16: vascular pattern has darker green (higher absorption)
    cv2.circle(img, (60, 60), radius=16, color=(200, 50, 60), thickness=-1)

    # Slight specular highlight near edge of polyp
    cv2.circle(img, (68, 54), radius=3, color=(255, 255, 255), thickness=-1)

    return img


def test_specular_detection(segmenter_v2):
    """Verifies that specular highlight detection accurately identifies bright glare spots."""
    img = np.full((80, 80, 3), 120, dtype=np.uint8)
    # Place a 6x6 saturated glare spot
    img[30:36, 30:36] = 255

    mask = segmenter_v2.detect_specular_highlights(img)

    assert isinstance(mask, np.ndarray)
    assert mask.shape == (80, 80)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})
    # Glare region must be detected
    assert np.all(mask[32:35, 32:35] == 255)
    # Non-glare region must be zero
    assert mask[10, 10] == 0


def test_specular_inpainting(segmenter_v2):
    """Verifies that inpainting replaces glare spots without altering dimensions or creating NaNs."""
    img = np.full((80, 80, 3), 120, dtype=np.uint8)
    img[30:36, 30:36] = 255

    inpainted = segmenter_v2.inpaint_specular_highlights(img)

    assert inpainted.shape == img.shape
    assert inpainted.dtype == np.uint8
    assert np.all(np.isfinite(inpainted))
    # Inpainted glare spot must be restored closer to surrounding mucosal level
    assert np.mean(inpainted[32:35, 32:35]) < 200

    # Clean image without highlights returns unchanged
    clean_img = np.full((40, 40, 3), 100, dtype=np.uint8)
    inpainted_clean = segmenter_v2.inpaint_specular_highlights(clean_img)
    assert np.array_equal(clean_img, inpainted_clean)


def test_saliency_output(segmenter_v2, synthetic_mucosa_with_polyp):
    """Verifies that saliency map produces normalized [0, 1] float32 output with finite values."""
    saliency = segmenter_v2.compute_saliency_map(synthetic_mucosa_with_polyp)

    assert isinstance(saliency, np.ndarray)
    assert saliency.shape == synthetic_mucosa_with_polyp.shape[:2]
    assert saliency.dtype == np.float32
    assert np.all(np.isfinite(saliency))
    assert 0.0 <= float(np.min(saliency))
    assert float(np.max(saliency)) <= 1.0
    # Peak saliency should be located in the vascular lesion region (60, 60)
    assert saliency[60, 60] > saliency[10, 10]


def test_lumen_suppression(segmenter_v2):
    """Verifies that dark lumen cavity regions are suppressed in the saliency map."""
    h, w = 120, 120
    img = np.full((h, w, 3), 130, dtype=np.uint8)
    # Dark lumen cavity in top-left corner
    img[10:35, 10:35] = 15

    # Vascular lesion in center
    cv2.circle(img, (60, 60), radius=15, color=(190, 60, 70), thickness=-1)

    saliency = segmenter_v2.compute_saliency_map(img)

    lumen_mean_sal = float(np.mean(saliency[15:30, 15:30]))
    polyp_mean_sal = float(np.mean(saliency[55:65, 55:65]))

    # Polyp saliency must be significantly higher than lumen saliency
    assert polyp_mean_sal > lumen_mean_sal * 2.0


def test_seed_generation(segmenter_v2, synthetic_mucosa_with_polyp):
    """Verifies that seed generation creates compact, localized disks rather than sprawling masks."""
    saliency = segmenter_v2.compute_saliency_map(synthetic_mucosa_with_polyp)
    seed = segmenter_v2.generate_seeds(saliency, rgb_image=synthetic_mucosa_with_polyp)

    assert isinstance(seed, np.ndarray)
    assert seed.shape == saliency.shape
    assert set(np.unique(seed)).issubset({0, 1})

    seed_pixels = np.count_nonzero(seed)
    total_pixels = saliency.size
    seed_ratio = seed_pixels / float(total_pixels)

    # Seed must be compact and localized (between 0.05% and 8% of frame, never whole image)
    assert 0.0005 <= seed_ratio <= 0.08
    # Seed must be positioned near the synthetic polyp center (60, 60)
    assert seed[60, 60] == 1

    # Flat saliency with no contrast must return empty seed
    flat_sal = np.zeros_like(saliency)
    empty_seed = segmenter_v2.generate_seeds(flat_sal)
    assert np.count_nonzero(empty_seed) == 0


def test_morphgac_output(segmenter_v2):
    """Verifies that MorphGAC evolves binary level set and respects boundaries."""
    h, w = 80, 80
    # Synthetic circular lesion on contrasting background
    gray = np.full((h, w), 80, dtype=np.uint8)
    cv2.circle(gray, (40, 40), radius=18, color=190, thickness=-1)

    # Initial seed inside circle
    seed = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(seed, (40, 40), radius=5, color=1, thickness=-1)

    res_mask = segmenter_v2.evolve_morphgac(gray, seed)

    assert isinstance(res_mask, np.ndarray)
    assert res_mask.shape == (h, w)
    assert set(np.unique(res_mask)).issubset({0, 255})
    # Contour must have expanded beyond seed
    assert np.count_nonzero(res_mask) > np.count_nonzero(seed)
    # Contour must remain localized inside/near circle boundary (not engulf entire image)
    assert np.count_nonzero(res_mask) < (h * w * 0.40)


def test_binary_mask_values(segmenter_v2, synthetic_mucosa_with_polyp):
    """Verifies that the segmentation result produces strictly binary masks {0, 255}."""
    gray = cv2.cvtColor(synthetic_mucosa_with_polyp, cv2.COLOR_RGB2GRAY)
    pred_mask, success, info = segmenter_v2.segment(gray, rgb_image=synthetic_mucosa_with_polyp)

    assert isinstance(pred_mask, np.ndarray)
    assert pred_mask.shape == gray.shape
    assert set(np.unique(pred_mask)).issubset({0, 255})
    assert isinstance(success, (bool, np.bool_))


def test_component_scoring(segmenter_v2):
    """Verifies that multi-criteria scoring ranks high-contrast, non-border components above edge artifacts."""
    h, w = 100, 100
    rgb = np.full((h, w, 3), 120, dtype=np.uint8)
    saliency = np.full((h, w), 0.2, dtype=np.float32)

    # Component 1 (central, high contrast, clean)
    cv2.circle(rgb, (50, 50), radius=12, color=(210, 50, 50), thickness=-1)
    saliency[40:60, 40:60] = 0.9

    # Component 2 (border touching)
    rgb[0:15, 0:15] = (200, 60, 60)
    saliency[0:15, 0:15] = 0.8

    bin_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(bin_mask, (50, 50), radius=12, color=255, thickness=-1)
    bin_mask[0:15, 0:15] = 255

    candidates = segmenter_v2.score_components(bin_mask, rgb, saliency)

    assert len(candidates) >= 2
    # Central, high-contrast component must rank first with higher score
    assert candidates[0]["touches_border"] is False
    assert candidates[0]["score"] > candidates[1]["score"]
    assert candidates[1]["touches_border"] is True


def test_centrality_is_weak(segmenter_v2):
    """Verifies that centrality is strictly a weak heuristic and does not disqualify off-center lesions."""
    h, w = 140, 140
    rgb = np.full((h, w, 3), 120, dtype=np.uint8)
    saliency = np.full((h, w), 0.3, dtype=np.float32)

    # Component 1: near center (70, 70)
    cv2.circle(rgb, (70, 70), radius=12, color=(200, 50, 50), thickness=-1)
    saliency[60:80, 60:80] = 0.8

    # Component 2: off-center at (30, 70), identical size and contrast, well within active field
    cv2.circle(rgb, (30, 70), radius=12, color=(200, 50, 50), thickness=-1)
    saliency[20:40, 60:80] = 0.8

    # Score center component
    bin_mask_center = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(bin_mask_center, (70, 70), radius=12, color=255, thickness=-1)
    c_center = segmenter_v2.score_components(bin_mask_center, rgb, saliency)[0]

    # Score peripheral component
    bin_mask_periph = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(bin_mask_periph, (30, 70), radius=12, color=255, thickness=-1)
    c_periph = segmenter_v2.score_components(bin_mask_periph, rgb, saliency)[0]

    # Peripheral candidate must still receive a strong, valid score above acceptance threshold
    assert c_periph["score"] >= segmenter_v2.acceptance_threshold
    # Score difference between center and off-center must be bounded by weight_centrality
    score_diff = c_center["score"] - c_periph["score"]
    assert 0.0 <= score_diff <= segmenter_v2.weight_centrality


def test_no_polyp_empty_mask(segmenter_v2):
    """Verifies that negative/flat images with no polyp return an all-zero mask with success=False."""
    # Completely flat, uniform mucosa
    flat_mucosa = np.full((100, 100, 3), (150, 95, 85), dtype=np.uint8)
    flat_gray = cv2.cvtColor(flat_mucosa, cv2.COLOR_RGB2GRAY)

    pred_mask, success, info = segmenter_v2.segment(flat_gray, rgb_image=flat_mucosa)

    assert np.count_nonzero(pred_mask) == 0
    assert success is False
    assert info["confidence"] == 0.0
    assert info["status"] in {"no_candidate_seed", "rejected_candidate", "no_valid_component"}


def test_low_confidence_rejection():
    """Verifies that candidates below the acceptance threshold are cleanly rejected."""
    # Segmenter with high acceptance threshold
    strict_segmenter = ActiveContourSegmenterV2(acceptance_threshold=0.95)

    img = np.full((100, 100, 3), 130, dtype=np.uint8)
    # Weak contrast lesion that yields moderate score (~0.4 - 0.6)
    cv2.circle(img, (50, 50), radius=12, color=(160, 90, 80), thickness=-1)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    pred_mask, success, info = strict_segmenter.segment(gray, rgb_image=img)

    # Must reject and return empty mask
    assert np.count_nonzero(pred_mask) == 0
    assert success is False
    assert info["status"] in {"rejected_candidate", "no_candidate_seed"}


def test_output_metadata(segmenter_v2, synthetic_mucosa_with_polyp):
    """Verifies that output info dictionary contains all required diagnostic metadata keys."""
    gray = cv2.cvtColor(synthetic_mucosa_with_polyp, cv2.COLOR_RGB2GRAY)
    pred_mask, success, info = segmenter_v2.segment(gray, rgb_image=synthetic_mucosa_with_polyp)

    required_keys = {
        "status",
        "confidence",
        "selected_component_score",
        "num_candidates",
        "fg_pixels",
        "fg_ratio_pct",
        "success",
    }
    assert required_keys.issubset(set(info.keys()))
    assert info["status"] in {
        "valid_candidate",
        "rejected_candidate",
        "no_candidate_seed",
        "no_valid_component",
        "rejected_overexpansion",
        "segmentation_failure",
    }


def test_ground_truth_not_required(segmenter_v2):
    """Verifies that the segmentation method never requires, accepts, or imports ground truth masks."""
    sig = inspect.signature(segmenter_v2.segment)
    param_names = list(sig.parameters.keys())

    # Ground truth parameters must NOT exist in segmentation interface
    assert "gt_mask" not in param_names
    assert "mask" not in param_names
    assert "ground_truth" not in param_names
    assert param_names == ["preprocessed_image", "rgb_image"]


def test_deterministic_output(segmenter_v2, synthetic_mucosa_with_polyp):
    """Verifies that running segmentation twice on the same image produces identical results."""
    gray = cv2.cvtColor(synthetic_mucosa_with_polyp, cv2.COLOR_RGB2GRAY)

    mask1, success1, info1 = segmenter_v2.segment(gray, rgb_image=synthetic_mucosa_with_polyp)
    mask2, success2, info2 = segmenter_v2.segment(gray, rgb_image=synthetic_mucosa_with_polyp)

    assert np.array_equal(mask1, mask2)
    assert success1 == success2
    assert pytest.approx(info1["confidence"], rel=1e-5) == info2["confidence"]


def test_split_isolation():
    """Verifies that the development and evaluation splits are mutually exclusive and total 1,000 images."""
    dev_files, eval_files = get_splits()

    assert len(dev_files) == 800
    assert len(eval_files) == 200

    dev_set = set(dev_files)
    eval_set = set(eval_files)

    # Strict mutual exclusivity: 0 overlap
    assert len(dev_set.intersection(eval_set)) == 0
    # Complete partition: 1000 unique images
    assert len(dev_set.union(eval_set)) == 1000


# ==============================================================================
# B1 Vignette-Aware Integration Tests
# ==============================================================================


@pytest.fixture
def synthetic_vignette_frame():
    """Generates synthetic endoscopic frame with dark circular camera vignette and internal cavity."""
    h, w = 160, 160
    img = np.zeros((h, w, 3), dtype=np.uint8)
    center = (w // 2, h // 2)
    radius = 70

    # Illuminated tissue aperture (bright mucosa: R=160, G=100, B=90)
    cv2.circle(img, center, radius, color=(160, 100, 90), thickness=-1)

    # Internal dark mucosal lumen cavity / shadow inside illuminated field (gray < 15)
    cv2.circle(img, (center[0] - 20, center[1] - 20), radius=10, color=(5, 5, 5), thickness=-1)

    return img


def test_fov_mask_dimensions():
    """1. Verifies that FOV mask matches input dimensions and returns uint8."""
    # 3-channel RGB image
    img_rgb = np.full((120, 150, 3), 100, dtype=np.uint8)
    fov_rgb = compute_endoscopic_fov_mask(img_rgb)
    assert fov_rgb.shape == (120, 150)
    assert fov_rgb.dtype == np.uint8

    # 2D grayscale image
    img_gray = np.full((90, 80), 100, dtype=np.uint8)
    fov_gray = compute_endoscopic_fov_mask(img_gray)
    assert fov_gray.shape == (90, 80)
    assert fov_gray.dtype == np.uint8


def test_fov_mask_binary_values(synthetic_vignette_frame):
    """2. Verifies that FOV mask values are strictly binary {0, 255}."""
    fov_mask = compute_endoscopic_fov_mask(synthetic_vignette_frame)
    assert isinstance(fov_mask, np.ndarray)
    assert set(np.unique(fov_mask)).issubset({0, 255})


def test_fov_mask_does_not_require_ground_truth():
    """3. Verifies that compute_endoscopic_fov_mask does not accept or require ground truth masks."""
    sig = inspect.signature(compute_endoscopic_fov_mask)
    param_names = list(sig.parameters.keys())
    assert "gt_mask" not in param_names
    assert "ground_truth" not in param_names
    assert "mask" not in param_names
    assert "image" in param_names


def test_fov_mask_preserves_internal_dark_regions(synthetic_vignette_frame):
    """4. Verifies that external contour filling preserves internal dark lumen/cavity inside FOV mask."""
    fov_mask = compute_endoscopic_fov_mask(synthetic_vignette_frame)
    # The internal dark shadow was placed at (center_x - 20, center_y - 20) = (60, 60)
    # It must be preserved as valid anatomical field (255)
    assert fov_mask[60, 60] == 255
    # The surrounding illuminated tissue must also be 255
    assert fov_mask[80, 80] == 255


def test_fov_mask_excludes_external_vignette(synthetic_vignette_frame):
    """5. Verifies that FOV mask strictly excludes external dark camera vignette."""
    fov_mask = compute_endoscopic_fov_mask(synthetic_vignette_frame)
    # 4 outer corners are part of camera vignette
    h, w = synthetic_vignette_frame.shape[:2]
    assert fov_mask[2, 2] == 0
    assert fov_mask[2, w - 3] == 0
    assert fov_mask[h - 3, 2] == 0
    assert fov_mask[h - 3, w - 3] == 0


def test_border_penalty_distinguishes_internal_dark_shadow_from_vignette(
    segmenter_v2, synthetic_vignette_frame
):
    """6. Verifies border penalty penalizes vignette touching but NOT internal dark mucosal shadows."""
    h, w = synthetic_vignette_frame.shape[:2]
    fov_mask = segmenter_v2.compute_endoscopic_fov_mask(synthetic_vignette_frame)

    saliency = np.full((h, w), 0.2, dtype=np.float32)

    # Component A: Centered near internal shadow, placed at (78, 60), radius 8
    # Within illuminated aperture, adjacent to dark cavity, but well away from vignette
    bin_mask_a = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(bin_mask_a, (78, 60), radius=8, color=255, thickness=-1)
    saliency[bin_mask_a > 0] = 0.85

    # Component B: Placed near the circular lens boundary touching the camera vignette
    bin_mask_b = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(bin_mask_b, (80, 15), radius=8, color=255, thickness=-1)
    saliency[bin_mask_b > 0] = 0.85

    c_a = segmenter_v2.score_components(
        bin_mask_a, synthetic_vignette_frame, saliency, fov_mask=fov_mask
    )
    c_b = segmenter_v2.score_components(
        bin_mask_b, synthetic_vignette_frame, saliency, fov_mask=fov_mask
    )

    assert len(c_a) == 1
    assert len(c_b) == 1

    # Component A (near internal cavity) must NOT receive border penalty
    assert c_a[0]["touches_border"] is False
    assert c_a[0]["touches_canvas_edge"] is False
    assert c_a[0]["touches_vignette"] is False
    assert c_a[0]["p_border"] == 0.0

    # Component B (touching vignette boundary) MUST receive border penalty
    assert c_b[0]["touches_border"] is True
    assert c_b[0]["touches_vignette"] is True
    assert c_b[0]["p_border"] == 1.0

    # Score of Component A must be strictly higher than Component B
    assert c_a[0]["score"] > c_b[0]["score"]


def test_component_scoring_remains_deterministic(segmenter_v2, synthetic_vignette_frame):
    """7. Verifies that component scoring produces identical outputs on repeated runs."""
    h, w = synthetic_vignette_frame.shape[:2]
    bin_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(bin_mask, (80, 80), radius=12, color=255, thickness=-1)
    saliency = np.full((h, w), 0.5, dtype=np.float32)

    c1 = segmenter_v2.score_components(bin_mask, synthetic_vignette_frame, saliency)
    c2 = segmenter_v2.score_components(bin_mask, synthetic_vignette_frame, saliency)

    assert len(c1) == len(c2)
    assert c1[0]["score"] == c2[0]["score"]
    assert c1[0]["p_border"] == c2[0]["p_border"]
    assert c1[0]["touches_border"] == c2[0]["touches_border"]
    assert c1[0]["centroid"] == c2[0]["centroid"]


def test_no_regression_in_existing_v2_behavior(segmenter_v2, synthetic_mucosa_with_polyp):
    """8. Verifies that existing successful segmentation behavior is preserved with B1 integration."""
    gray = cv2.cvtColor(synthetic_mucosa_with_polyp, cv2.COLOR_RGB2GRAY)
    pred_mask, success, info = segmenter_v2.segment(gray, rgb_image=synthetic_mucosa_with_polyp)

    assert success is True
    assert pred_mask.shape == gray.shape
    assert set(np.unique(pred_mask)).issubset({0, 255})
    assert np.count_nonzero(pred_mask) > 0
    assert info["status"] == "valid_candidate"
    assert info["confidence"] >= segmenter_v2.acceptance_threshold


def test_existing_rejection_behavior_remains_valid(segmenter_v2):
    """9. Verifies that low-confidence and negative frames continue to be properly rejected."""
    # Flat mucosa image
    flat = np.full((120, 120, 3), 110, dtype=np.uint8)
    gray = cv2.cvtColor(flat, cv2.COLOR_RGB2GRAY)
    pred_mask, success, info = segmenter_v2.segment(gray, rgb_image=flat)

    assert success is False
    assert np.count_nonzero(pred_mask) == 0
    assert info["status"] in {"no_candidate_seed", "rejected_candidate", "no_valid_component"}

    # Strict threshold rejection
    strict_seg = ActiveContourSegmenterV2(acceptance_threshold=0.95)
    pred_strict, succ_strict, info_strict = strict_seg.segment(gray, rgb_image=flat)
    assert succ_strict is False
    assert np.count_nonzero(pred_strict) == 0


def test_output_metadata_remains_unchanged(segmenter_v2, synthetic_mucosa_with_polyp):
    """10. Verifies that output metadata dictionary keys and structure remain backward-compatible."""
    gray = cv2.cvtColor(synthetic_mucosa_with_polyp, cv2.COLOR_RGB2GRAY)
    pred_mask, success, info = segmenter_v2.segment(gray, rgb_image=synthetic_mucosa_with_polyp)

    expected_keys = {
        "status",
        "confidence",
        "selected_component_score",
        "num_candidates",
        "fg_pixels",
        "fg_ratio_pct",
        "success",
        "centroid",
        "s_area",
        "s_contrast",
        "s_centrality",
        "p_border",
    }
    assert expected_keys.issubset(set(info.keys()))
    assert isinstance(info["confidence"], float)
    assert isinstance(info["p_border"], float)
