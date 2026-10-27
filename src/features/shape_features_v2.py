"""Shape Feature Extraction Module - Version 2.

Extracts geometric and topological shape descriptors with resolution normalization
and rotation invariance from predicted polyp binary masks.
"""

from typing import Dict
import cv2
import numpy as np
from skimage.measure import regionprops, label


def extract_shape_features_v2(pred_mask: np.ndarray) -> Dict[str, float]:
    """Calculates resolution-normalized geometric shape feature descriptors from a predicted binary polyp mask.

    Args:
        pred_mask: (H, W) uint8 binary NumPy array (values 0 or 255).

    Returns:
        Dict mapping feature names to float values.
    """
    default_features = {
        "shape_relative_area": 0.0,
        "shape_relative_perimeter": 0.0,
        "shape_circularity": 0.0,
        "shape_axis_ratio": 0.0,
        "shape_solidity": 0.0,
        "shape_extent": 0.0,
        "shape_eccentricity": 0.0,
        "shape_relative_major_axis": 0.0,
        "shape_relative_minor_axis": 0.0,
    }

    if pred_mask is None or pred_mask.size == 0 or np.count_nonzero(pred_mask) == 0:
        return default_features

    h, w = pred_mask.shape[:2]
    total_image_area = float(h * w)
    image_diagonal = float(np.sqrt(h**2 + w**2))

    bin_mask = (pred_mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(
        bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return default_features

    largest_contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest_contour))

    if area == 0.0:
        return default_features

    perimeter = float(cv2.arcLength(largest_contour, True))
    circularity = (4.0 * np.pi * area) / ((perimeter**2) + 1e-6)

    # Resolution-normalized dimensions: fraction of image pixels occupied by ROI
    pixel_area = float(np.count_nonzero(bin_mask))
    relative_area = pixel_area / total_image_area if total_image_area > 0 else 0.0
    relative_perimeter = perimeter / image_diagonal if image_diagonal > 0 else 0.0

    # Bounding box extent
    _, _, bw, bh = cv2.boundingRect(largest_contour)
    bbox_area = float(bw * bh)
    extent = area / bbox_area if bbox_area > 0 else 0.0

    # Convex hull solidity
    hull = cv2.convexHull(largest_contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 0 else 0.0

    # Inertia ellipse properties via skimage regionprops
    labeled_mask = label(bin_mask)
    props = regionprops(labeled_mask)

    eccentricity = 0.0
    major_axis = 0.0
    minor_axis = 0.0

    if props:
        largest_prop = max(props, key=lambda r: r.area)
        eccentricity = float(getattr(largest_prop, "eccentricity", 0.0))
        if hasattr(largest_prop, "axis_major_length"):
            major_axis = float(largest_prop.axis_major_length)
        else:
            major_axis = float(getattr(largest_prop, "major_axis_length", 0.0))
        if hasattr(largest_prop, "axis_minor_length"):
            minor_axis = float(largest_prop.axis_minor_length)
        else:
            minor_axis = float(getattr(largest_prop, "minor_axis_length", 0.0))

    # Rotation-invariant axis ratio (replaces bounding-box aspect ratio)
    axis_ratio = major_axis / (minor_axis + 1e-6) if minor_axis > 0 else 1.0

    relative_major_axis = major_axis / image_diagonal if image_diagonal > 0 else 0.0
    relative_minor_axis = minor_axis / image_diagonal if image_diagonal > 0 else 0.0

    features = {
        "shape_relative_area": float(relative_area),
        "shape_relative_perimeter": float(relative_perimeter),
        "shape_circularity": float(circularity),
        "shape_axis_ratio": float(axis_ratio),
        "shape_solidity": float(solidity),
        "shape_extent": float(extent),
        "shape_eccentricity": float(eccentricity),
        "shape_relative_major_axis": float(relative_major_axis),
        "shape_relative_minor_axis": float(relative_minor_axis),
    }

    for k, v in features.items():
        if not np.isfinite(v):
            features[k] = 0.0

    return features
