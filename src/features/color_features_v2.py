"""Color Feature Extraction Module - Version 2.

Extracts statistical color descriptors across RGB, HSV, and LAB color spaces
exclusively from pixels within the predicted polyp region. HSV Hue is computed using
circular directional statistics to eliminate 0/180 wraparound distortion.
"""

from typing import Dict
import cv2
import numpy as np


def extract_color_features_v2(rgb_image: np.ndarray, pred_mask: np.ndarray) -> Dict[str, float]:
    """Extracts first- and second-order color statistics with circular Hue statistics.

    Args:
        rgb_image: (H, W, 3) uint8 original RGB endoscopy frame.
        pred_mask: (H, W) uint8 binary predicted segmentation mask.

    Returns:
        Dict mapping color feature names to float values.
    """
    default_features = {
        "color_rgb_r_mean": 0.0,
        "color_rgb_r_std": 0.0,
        "color_rgb_g_mean": 0.0,
        "color_rgb_g_std": 0.0,
        "color_rgb_b_mean": 0.0,
        "color_rgb_b_std": 0.0,
        "color_hsv_h_circ_mean": 0.0,
        "color_hsv_h_circ_std": 0.0,
        "color_hsv_s_mean": 0.0,
        "color_hsv_s_std": 0.0,
        "color_hsv_v_mean": 0.0,
        "color_hsv_v_std": 0.0,
        "color_lab_l_mean": 0.0,
        "color_lab_l_std": 0.0,
        "color_lab_a_mean": 0.0,
        "color_lab_a_std": 0.0,
        "color_lab_b_mean": 0.0,
        "color_lab_b_std": 0.0,
    }

    if (
        rgb_image is None
        or pred_mask is None
        or pred_mask.size == 0
        or np.count_nonzero(pred_mask) == 0
    ):
        return default_features

    valid_pixel_indices = pred_mask > 0

    # 1. RGB Features
    rgb_polyp_pixels = rgb_image[valid_pixel_indices]
    if len(rgb_polyp_pixels) == 0:
        return default_features

    r_vals = rgb_polyp_pixels[:, 0].astype(np.float64)
    g_vals = rgb_polyp_pixels[:, 1].astype(np.float64)
    b_vals = rgb_polyp_pixels[:, 2].astype(np.float64)

    # 2. HSV Features with Circular Directional Statistics
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    hsv_polyp_pixels = hsv_image[valid_pixel_indices]
    h_vals = hsv_polyp_pixels[:, 0].astype(np.float64)  # OpenCV Hue in [0, 180)
    s_vals = hsv_polyp_pixels[:, 1].astype(np.float64)
    v_vals = hsv_polyp_pixels[:, 2].astype(np.float64)

    # Convert Hue degrees in [0, 180) to radians in [0, 2*pi)
    theta = h_vals * (np.pi / 90.0)
    s_mean = np.mean(np.sin(theta))
    c_mean = np.mean(np.cos(theta))
    r_bar = np.sqrt(s_mean**2 + c_mean**2)

    # Circular mean angle mapped back to [0, 180)
    theta_bar = np.arctan2(s_mean, c_mean) % (2.0 * np.pi)
    h_circ_mean = theta_bar * (90.0 / np.pi)

    # Circular standard deviation / angular dispersion
    r_bar_clamped = np.clip(r_bar, 1e-6, 1.0)
    h_circ_std = np.sqrt(-2.0 * np.log(r_bar_clamped))

    # 3. LAB Features
    lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    lab_polyp_pixels = lab_image[valid_pixel_indices]
    l_vals = lab_polyp_pixels[:, 0].astype(np.float64)
    a_vals = lab_polyp_pixels[:, 1].astype(np.float64)
    lab_b_vals = lab_polyp_pixels[:, 2].astype(np.float64)

    features = {
        "color_rgb_r_mean": float(np.mean(r_vals)),
        "color_rgb_r_std": float(np.std(r_vals)),
        "color_rgb_g_mean": float(np.mean(g_vals)),
        "color_rgb_g_std": float(np.std(g_vals)),
        "color_rgb_b_mean": float(np.mean(b_vals)),
        "color_rgb_b_std": float(np.std(b_vals)),
        "color_hsv_h_circ_mean": float(h_circ_mean),
        "color_hsv_h_circ_std": float(h_circ_std),
        "color_hsv_s_mean": float(np.mean(s_vals)),
        "color_hsv_s_std": float(np.std(s_vals)),
        "color_hsv_v_mean": float(np.mean(v_vals)),
        "color_hsv_v_std": float(np.std(v_vals)),
        "color_lab_l_mean": float(np.mean(l_vals)),
        "color_lab_l_std": float(np.std(l_vals)),
        "color_lab_a_mean": float(np.mean(a_vals)),
        "color_lab_a_std": float(np.std(a_vals)),
        "color_lab_b_mean": float(np.mean(lab_b_vals)),
        "color_lab_b_std": float(np.std(lab_b_vals)),
    }

    for k, v in features.items():
        if not np.isfinite(v):
            features[k] = 0.0

    return features
