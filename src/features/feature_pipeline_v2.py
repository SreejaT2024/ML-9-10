"""Stage A Feature Extraction Pipeline - Version 2.

Orchestrates the complete end-to-end V2 feature extraction workflow:
Image -> Preprocessing -> ActiveContourSegmenterV2 (MorphGAC + Vascular Saliency)
-> Predicted Mask -> ROIExtractor -> Shape V2 / Color V2 / Texture V2 Features.

Ground-truth masks are strictly isolated: NEVER used for active contour segmentation,
ROI extraction, or feature extraction. They may only be passed for optional evaluation
metrics (Dice, IoU, Precision, Recall).
"""

from typing import Dict, Any, List, Optional
import numpy as np

from config.config import GlobalConfig, get_config
from src.preprocessing.image_preprocess import PolypImagePreprocessor
from src.segmentation.active_contour_v2 import ActiveContourSegmenterV2
from src.segmentation.metrics import evaluate_segmentation
from src.features.roi_extractor import ROIExtractor
from src.features.shape_features_v2 import extract_shape_features_v2
from src.features.color_features_v2 import extract_color_features_v2
from src.features.texture_features_v2 import extract_texture_features_v2
from src.utils.logger import get_logger

logger = get_logger("feature_pipeline_v2")

# Canonical feature definitions
SHAPE_FEATURE_NAMES: List[str] = [
    "shape_relative_area",
    "shape_relative_perimeter",
    "shape_circularity",
    "shape_axis_ratio",
    "shape_solidity",
    "shape_extent",
    "shape_eccentricity",
    "shape_relative_major_axis",
    "shape_relative_minor_axis",
]

COLOR_FEATURE_NAMES: List[str] = [
    "color_rgb_r_mean",
    "color_rgb_r_std",
    "color_rgb_g_mean",
    "color_rgb_g_std",
    "color_rgb_b_mean",
    "color_rgb_b_std",
    "color_hsv_h_circ_mean",
    "color_hsv_h_circ_std",
    "color_hsv_s_mean",
    "color_hsv_s_std",
    "color_hsv_v_mean",
    "color_hsv_v_std",
    "color_lab_l_mean",
    "color_lab_l_std",
    "color_lab_a_mean",
    "color_lab_a_std",
    "color_lab_b_mean",
    "color_lab_b_std",
]

TEXTURE_FEATURE_NAMES: List[str] = [
    "texture_glcm_contrast",
    "texture_glcm_dissimilarity",
    "texture_glcm_homogeneity",
    "texture_glcm_energy",
    "texture_glcm_correlation",
    "texture_lbp_energy",
    "texture_lbp_entropy",
    "texture_lbp_uniform_ratio",
]

FEATURE_NAMES: List[str] = SHAPE_FEATURE_NAMES + COLOR_FEATURE_NAMES + TEXTURE_FEATURE_NAMES
METADATA_COLUMNS: List[str] = ["filename", "segmentation_status", "segmentation_success", "confidence"]
FEATURE_COLUMNS: List[str] = METADATA_COLUMNS + FEATURE_NAMES


class StageAFeaturePipelineV2:
    """End-to-end V2 pipeline: preprocessing -> segmentation V2 -> ROI -> V2 features."""

    def __init__(self, config: Optional[GlobalConfig] = None) -> None:
        """Initializes pipeline components with production V2 defaults.

        Args:
            config: Optional GlobalConfig instance.
        """
        self.config = config or get_config()
        self.preprocessor = PolypImagePreprocessor()
        self.segmenter = ActiveContourSegmenterV2(config=self.config)
        self.roi_extractor = ROIExtractor()
        self.feature_names = list(FEATURE_NAMES)

    @staticmethod
    def get_feature_names() -> List[str]:
        """Returns the complete list of 35 engineered V2 feature names."""
        return list(FEATURE_NAMES)

    @staticmethod
    def get_column_names() -> List[str]:
        """Returns the canonical CSV column schema (4 metadata + 35 features = 39 columns)."""
        return list(FEATURE_COLUMNS)

    @staticmethod
    def get_default_features() -> Dict[str, float]:
        """Returns the documented default zero-imputation dictionary for all 35 features."""
        return {k: 0.0 for k in FEATURE_NAMES}

    def process_sample(
        self,
        rgb_image: np.ndarray,
        filename: str,
        gt_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Processes a single endoscopic image through the complete Stage A V2 pipeline.

        Strict Ground-Truth Isolation:
        gt_mask is NEVER provided to preprocessing, active contour segmentation,
        ROI extraction, or any feature extractor. It is only evaluated against pred_mask
        if explicitly provided for auditing purposes.

        Missing-Value Strategy:
        If preprocessing fails, or segmentation rejects the frame / returns an empty mask,
        or ROI extraction yields no valid connected component, all 35 features are imputed
        with deterministic 0.0 values, and segmentation_status explicitly records the cause.

        Args:
            rgb_image: (H, W, 3) uint8 RGB endoscopy frame.
            filename: Sample identifier string.
            gt_mask: Optional (H, W) uint8 binary ground-truth mask for evaluation only.

        Returns:
            Dict containing:
                - filename: sample identifier
                - segmentation_status: diagnostic status string
                - segmentation_success: bool
                - confidence: float confidence score in [0.0, 1.0]
                - roi_valid: bool indicating whether valid ROI was extracted
                - fg_ratio_pct: foreground pixel ratio in percent
                - All 35 extracted shape, color, and texture features
                - Optional evaluation metrics (dice, iou, precision, recall) if gt_mask given
                - Internal intermediate arrays (_pred_mask, _roi_rgb, _roi_mask)
        """
        result: Dict[str, Any] = {"filename": filename}

        if rgb_image is None or not isinstance(rgb_image, np.ndarray) or rgb_image.ndim != 3:
            logger.error(f"Invalid image array provided for {filename}.")
            result["segmentation_status"] = "invalid_input_image"
            result["segmentation_success"] = False
            result["confidence"] = 0.0
            result["roi_valid"] = False
            result["fg_ratio_pct"] = 0.0
            result.update(self.get_default_features())
            return self._attach_evaluation_and_intermediates(
                result, None, None, None, gt_mask
            )

        h, w = rgb_image.shape[:2]

        # ── Step 1: Preprocessing ──
        try:
            preprocessed = self.preprocessor.preprocess_for_segmentation(rgb_image)
        except Exception as e:
            logger.error(f"Preprocessing failed for {filename}: {e}")
            result["segmentation_status"] = f"preprocessing_failure: {e}"
            result["segmentation_success"] = False
            result["confidence"] = 0.0
            result["roi_valid"] = False
            result["fg_ratio_pct"] = 0.0
            result.update(self.get_default_features())
            return self._attach_evaluation_and_intermediates(
                result, np.zeros((h, w), dtype=np.uint8), None, None, gt_mask
            )

        # ── Step 2: Active Contour V2 Segmentation (Strictly No Ground-Truth) ──
        try:
            pred_mask, seg_success, seg_info = self.segmenter.segment(
                preprocessed, rgb_image=rgb_image
            )
        except Exception as e:
            logger.error(f"Segmentation failed for {filename}: {e}")
            result["segmentation_status"] = f"segmentation_failure: {e}"
            result["segmentation_success"] = False
            result["confidence"] = 0.0
            result["roi_valid"] = False
            result["fg_ratio_pct"] = 0.0
            result.update(self.get_default_features())
            return self._attach_evaluation_and_intermediates(
                result, np.zeros((h, w), dtype=np.uint8), None, None, gt_mask
            )

        # Record segmentation metadata
        seg_status = str(seg_info.get("status", "valid_candidate" if seg_success else "rejected_candidate"))
        confidence = float(seg_info.get("confidence", 0.0))
        fg_ratio_pct = float(seg_info.get("fg_ratio_pct", 0.0))

        result["segmentation_status"] = seg_status
        result["segmentation_success"] = bool(seg_success)
        result["confidence"] = confidence
        result["fg_ratio_pct"] = fg_ratio_pct

        # ── Step 3: Explicit Rejection & Empty Mask Handling ──
        # If segmentation was rejected or empty, apply documented missing-value defaults
        if not seg_success or pred_mask is None or np.count_nonzero(pred_mask) == 0:
            clean_empty_mask = np.zeros((h, w), dtype=np.uint8)
            result["roi_valid"] = False
            result["roi_bbox_x"] = 0
            result["roi_bbox_y"] = 0
            result["roi_bbox_w"] = 0
            result["roi_bbox_h"] = 0
            result.update(self.get_default_features())
            return self._attach_evaluation_and_intermediates(
                result, clean_empty_mask, None, None, gt_mask
            )

        # ── Step 4: ROI Extraction from Predicted Mask ──
        try:
            roi_rgb, roi_mask, _, bbox, valid_roi = self.roi_extractor.extract_roi(
                rgb_image, pred_mask
            )
        except Exception as e:
            logger.error(f"ROI extraction failed for {filename}: {e}")
            result["segmentation_status"] = f"roi_extraction_failure: {e}"
            result["roi_valid"] = False
            result["roi_bbox_x"] = 0
            result["roi_bbox_y"] = 0
            result["roi_bbox_w"] = 0
            result["roi_bbox_h"] = 0
            result.update(self.get_default_features())
            return self._attach_evaluation_and_intermediates(
                result, pred_mask, None, None, gt_mask
            )

        result["roi_valid"] = bool(valid_roi)
        result["roi_bbox_x"] = int(bbox[0])
        result["roi_bbox_y"] = int(bbox[1])
        result["roi_bbox_w"] = int(bbox[2])
        result["roi_bbox_h"] = int(bbox[3])

        if not valid_roi or np.count_nonzero(roi_mask) == 0:
            result.update(self.get_default_features())
            return self._attach_evaluation_and_intermediates(
                result, pred_mask, roi_rgb, roi_mask, gt_mask
            )

        # ── Step 5: Shape Features V2 from Predicted Mask ──
        try:
            shape_feats = extract_shape_features_v2(pred_mask)
        except Exception as e:
            logger.error(f"Shape feature extraction failed for {filename}: {e}")
            shape_feats = {k: 0.0 for k in SHAPE_FEATURE_NAMES}

        # ── Step 6: Color Features V2 from Original RGB + Predicted Mask ──
        try:
            color_feats = extract_color_features_v2(rgb_image, pred_mask)
        except Exception as e:
            logger.error(f"Color feature extraction failed for {filename}: {e}")
            color_feats = {k: 0.0 for k in COLOR_FEATURE_NAMES}

        # ── Step 7: Texture Features V2 from Cropped ROI RGB + ROI Mask ──
        try:
            texture_feats = extract_texture_features_v2(roi_rgb, roi_mask)
        except Exception as e:
            logger.error(f"Texture feature extraction failed for {filename}: {e}")
            texture_feats = {k: 0.0 for k in TEXTURE_FEATURE_NAMES}

        # Clean all features: guarantee finite numerical floats
        all_feats = {**shape_feats, **color_feats, **texture_feats}
        for k in FEATURE_NAMES:
            val = float(all_feats.get(k, 0.0))
            result[k] = val if np.isfinite(val) else 0.0

        return self._attach_evaluation_and_intermediates(
            result, pred_mask, roi_rgb, roi_mask, gt_mask
        )

    def _attach_evaluation_and_intermediates(
        self,
        result: Dict[str, Any],
        pred_mask: Optional[np.ndarray],
        roi_rgb: Optional[np.ndarray],
        roi_mask: Optional[np.ndarray],
        gt_mask: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """Attaches optional ground-truth evaluation metrics and intermediate visualization arrays."""
        if gt_mask is not None and pred_mask is not None:
            metrics = evaluate_segmentation(pred_mask, gt_mask)
            result["dice"] = float(metrics["dice"])
            result["iou"] = float(metrics["iou"])
            result["precision"] = float(metrics["precision"])
            result["recall"] = float(metrics["recall"])
        else:
            result["dice"] = None
            result["iou"] = None
            result["precision"] = None
            result["recall"] = None

        result["_pred_mask"] = pred_mask
        result["_roi_rgb"] = roi_rgb
        result["_roi_mask"] = roi_mask

        return result

    @classmethod
    def to_feature_row(cls, sample_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts strictly the canonical CSV feature row (4 metadata + 35 features).

        Args:
            sample_result: Result dictionary returned by process_sample().

        Returns:
            Dict ordered strictly matching FEATURE_COLUMNS.
        """
        row: Dict[str, Any] = {}
        row["filename"] = sample_result.get("filename", "")
        row["segmentation_status"] = sample_result.get("segmentation_status", "unknown")
        row["segmentation_success"] = bool(sample_result.get("segmentation_success", False))
        row["confidence"] = float(sample_result.get("confidence", 0.0))

        for feat in FEATURE_NAMES:
            v = sample_result.get(feat, 0.0)
            row[feat] = float(v) if (v is not None and np.isfinite(v)) else 0.0

        return row


# Alias for compatibility
FeaturePipelineV2 = StageAFeaturePipelineV2
__all__ = [
    "StageAFeaturePipelineV2",
    "FeaturePipelineV2",
    "FEATURE_NAMES",
    "SHAPE_FEATURE_NAMES",
    "COLOR_FEATURE_NAMES",
    "TEXTURE_FEATURE_NAMES",
    "FEATURE_COLUMNS",
    "METADATA_COLUMNS",
]
