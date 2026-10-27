"""Texture Feature Extraction Module - Version 2.

Calculates Gray-Level Co-occurrence Matrix (GLCM) descriptors restricted strictly
to valid intra-ROI pixel pairs (no zero-padding distortion) and Local Binary Pattern (LBP)
probability distribution metrics (no arithmetic mean/std of categorical codes).
"""

from typing import Dict, Tuple, Sequence
import cv2
import numpy as np
from skimage.feature import local_binary_pattern


def compute_valid_roi_glcm(
    quantized_gray: np.ndarray,
    binary_mask: np.ndarray,
    distances: Sequence[int] = (1, 3, 5),
    angles: Sequence[float] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    levels: int = 32,
) -> Dict[str, float]:
    """Computes GLCM features strictly for pairs where BOTH pixels lie inside the mask.

    Eliminates zero-padding artifacts and boundary step-edge contamination.

    Args:
        quantized_gray: (H, W) uint8 image quantized into [0, levels - 1].
        binary_mask: (H, W) uint8 binary mask (values > 0 inside ROI).
        distances: Sequence of integer pixel offsets.
        angles: Sequence of angles in radians.
        levels: Number of discrete gray levels.

    Returns:
        Dict mapping GLCM metric names to float values.
    """
    h, w = quantized_gray.shape
    I_grid, J_grid = np.meshgrid(np.arange(levels), np.arange(levels), indexing="ij")
    diff = I_grid - J_grid
    diff_sq = diff**2
    abs_diff = np.abs(diff)
    homo_denom = 1.0 + diff_sq

    contrast_list = []
    dissimilarity_list = []
    homogeneity_list = []
    energy_list = []
    correlation_list = []

    for d in distances:
        for theta in angles:
            dy = int(np.round(-d * np.sin(theta)))
            dx = int(np.round(d * np.cos(theta)))

            y1_min = max(0, -dy)
            y1_max = min(h, h - dy)
            x1_min = max(0, -dx)
            x1_max = min(w, w - dx)

            if y1_min >= y1_max or x1_min >= x1_max:
                continue

            y2_min = y1_min + dy
            y2_max = y1_max + dy
            x2_min = x1_min + dx
            x2_max = x1_max + dx

            m1 = binary_mask[y1_min:y1_max, x1_min:x1_max] > 0
            m2 = binary_mask[y2_min:y2_max, x2_min:x2_max] > 0
            valid_pairs = m1 & m2

            if not np.any(valid_pairs):
                continue

            q1 = quantized_gray[y1_min:y1_max, x1_min:x1_max][valid_pairs]
            q2 = quantized_gray[y2_min:y2_max, x2_min:x2_max][valid_pairs]

            # Fast 2D histogram via bincount
            p12 = q1.astype(np.int64) * levels + q2.astype(np.int64)
            p21 = q2.astype(np.int64) * levels + q1.astype(np.int64)

            glcm_flat = np.bincount(p12, minlength=levels * levels) + np.bincount(
                p21, minlength=levels * levels
            )
            total_counts = np.sum(glcm_flat)

            if total_counts == 0:
                continue

            P = (glcm_flat.reshape(levels, levels) / total_counts).astype(np.float64)

            contrast = np.sum(diff_sq * P)
            dissimilarity = np.sum(abs_diff * P)
            homogeneity = np.sum(P / homo_denom)
            energy = np.sqrt(np.sum(P**2))

            # Correlation
            mu_i = np.sum(I_grid * P)
            mu_j = np.sum(J_grid * P)
            sigma_i = np.sqrt(np.sum(((I_grid - mu_i) ** 2) * P))
            sigma_j = np.sqrt(np.sum(((J_grid - mu_j) ** 2) * P))

            if sigma_i * sigma_j > 1e-12:
                corr = np.sum((I_grid - mu_i) * (J_grid - mu_j) * P) / (sigma_i * sigma_j)
            else:
                corr = 1.0

            contrast_list.append(float(contrast))
            dissimilarity_list.append(float(dissimilarity))
            homogeneity_list.append(float(homogeneity))
            energy_list.append(float(energy))
            correlation_list.append(float(corr))

    if not contrast_list:
        return {
            "texture_glcm_contrast": 0.0,
            "texture_glcm_dissimilarity": 0.0,
            "texture_glcm_homogeneity": 0.0,
            "texture_glcm_energy": 0.0,
            "texture_glcm_correlation": 0.0,
        }

    return {
        "texture_glcm_contrast": float(np.mean(contrast_list)),
        "texture_glcm_dissimilarity": float(np.mean(dissimilarity_list)),
        "texture_glcm_homogeneity": float(np.mean(homogeneity_list)),
        "texture_glcm_energy": float(np.mean(energy_list)),
        "texture_glcm_correlation": float(np.mean(correlation_list)),
    }


def extract_texture_features_v2(
    roi_rgb: np.ndarray,
    roi_mask: np.ndarray,
    distances: Tuple[int, ...] = (1, 3, 5),
    angles: Tuple[float, ...] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    lbp_points: int = 24,
    lbp_radius: int = 3,
) -> Dict[str, float]:
    """Extracts valid-pair GLCM and normalized LBP distribution features from polyp ROI.

    Args:
        roi_rgb: (h_roi, w_roi, 3) uint8 cropped RGB frame.
        roi_mask: (h_roi, w_roi) uint8 cropped binary mask.
        distances: Sequence of pixel distance offsets for GLCM computation.
        angles: Sequence of angles in radians for GLCM computation.
        lbp_points: Number of circular neighbor set points for LBP.
        lbp_radius: Radius of circle for LBP.

    Returns:
        Dict mapping texture feature names to float values.
    """
    default_features = {
        "texture_glcm_contrast": 0.0,
        "texture_glcm_dissimilarity": 0.0,
        "texture_glcm_homogeneity": 0.0,
        "texture_glcm_energy": 0.0,
        "texture_glcm_correlation": 0.0,
        "texture_lbp_energy": 0.0,
        "texture_lbp_entropy": 0.0,
        "texture_lbp_uniform_ratio": 0.0,
    }

    if (
        roi_rgb is None
        or roi_mask is None
        or roi_mask.size == 0
        or np.count_nonzero(roi_mask) == 0
        or roi_rgb.shape[0] < 4
        or roi_rgb.shape[1] < 4
    ):
        return default_features

    # 1. Grayscale conversion and optional spatial downsampling for texture
    if roi_rgb.ndim == 3:
        roi_gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
    else:
        roi_gray = roi_rgb.copy()

    cur_h, cur_w = roi_gray.shape
    max_tex_dim = 256
    if max(cur_h, cur_w) > max_tex_dim:
        scale = max_tex_dim / float(max(cur_h, cur_w))
        t_w, t_h = max(1, int(cur_w * scale)), max(1, int(cur_h * scale))
        proc_gray = cv2.resize(roi_gray, (t_w, t_h), interpolation=cv2.INTER_AREA)
        proc_mask = cv2.resize(roi_mask, (t_w, t_h), interpolation=cv2.INTER_NEAREST)
    else:
        proc_gray = roi_gray
        proc_mask = roi_mask

    # 2. GLCM Features (strictly intra-ROI valid pairs, no zero padding)
    gray_levels = 32
    quantized_gray = np.clip(proc_gray // (256 // gray_levels), 0, gray_levels - 1).astype(np.uint8)
    glcm_feats = compute_valid_roi_glcm(
        quantized_gray=quantized_gray,
        binary_mask=proc_mask,
        distances=distances,
        angles=angles,
        levels=gray_levels,
    )

    # 3. LBP Feature Extraction (Normalized Histogram Probabilities)
    try:
        lbp = local_binary_pattern(
            proc_gray, P=lbp_points, R=lbp_radius, method="uniform"
        )
        polyp_lbp = lbp[proc_mask > 0]
        n_roi_pixels = len(polyp_lbp)

        if n_roi_pixels > 0:
            n_bins = int(lbp_points + 2)  # For uniform LBP, labels are 0..P+1
            hist, _ = np.histogram(polyp_lbp, bins=n_bins, range=(0, n_bins))
            p_i = hist.astype(np.float64) / float(n_roi_pixels)

            # Energy: sum(p_i^2)
            lbp_energy = float(np.sum(p_i**2))

            # Shannon Entropy: -sum(p_i * log2(p_i)) for p_i > 0
            p_nonzero = p_i[p_i > 0]
            lbp_entropy = float(-np.sum(p_nonzero * np.log2(p_nonzero)))

            # Uniform ratio: fraction of pixels with uniform patterns (codes 0..P)
            # In skimage uniform LBP, codes 0..P are uniform; code P+1 is non-uniform
            uniform_pixels = np.sum(polyp_lbp < (lbp_points + 1))
            lbp_uniform_ratio = float(uniform_pixels / float(n_roi_pixels))
        else:
            lbp_energy = 0.0
            lbp_entropy = 0.0
            lbp_uniform_ratio = 0.0

    except Exception:
        lbp_energy = 0.0
        lbp_entropy = 0.0
        lbp_uniform_ratio = 0.0

    features = {
        **glcm_feats,
        "texture_lbp_energy": lbp_energy,
        "texture_lbp_entropy": lbp_entropy,
        "texture_lbp_uniform_ratio": lbp_uniform_ratio,
    }

    for k, v in features.items():
        if not np.isfinite(v):
            features[k] = 0.0

    return features
