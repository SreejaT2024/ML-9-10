"""Active Contour Segmentation Module - Version 2.

Implements an enhanced Active-Contour-guided segmentation pipeline integrating:
1. Specular highlight detection and Navier-Stokes/Telea inpainting.
2. Green-channel vascular and tissue contrast saliency with lumen suppression.
3. Localized candidate seed generation (avoiding full-frame Otsu or large central discs).
4. Morphological Geodesic Active Contour (MorphGAC) guided by edge-stopping gradients.
5. Multi-criteria connected component scoring (area, boundary contrast, weak centrality, border penalty).
6. Mandatory low-confidence rejection returning clean empty masks for negative or ambiguous frames.
7. Strict ground-truth isolation (ground truth is never loaded or accessed during prediction).
"""

from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import cv2
from skimage.segmentation import (
    morphological_geodesic_active_contour,
    inverse_gaussian_gradient,
)

from config.config import GlobalConfig, get_config
from src.utils.logger import get_logger

logger = get_logger("active_contour_v2")

__all__ = ["ActiveContourSegmenterV2", "compute_endoscopic_fov_mask"]


def compute_endoscopic_fov_mask(
    image: np.ndarray,
    dark_threshold: int = 15,
    morph_kernel_size: int = 15,
) -> np.ndarray:
    """Computes a robust, vignette-aware Field-of-View (FOV) mask for endoscopic frames.

    Distinguishes the dark circular/barrel camera vignette outside the active lens view
    from internal dark mucosal cavities, lumen, or shadows within the illuminated field.
    Ground-truth masks are strictly never accessed or required.

    Args:
        image: (H, W) or (H, W, 3) uint8 image.
        dark_threshold: Intensity threshold below which dark pixels are detected (default: 15).
        morph_kernel_size: Size of elliptical structuring element for closing (default: 15).

    Returns:
        (H, W) uint8 binary mask (255 inside active endoscopic FOV, 0 in camera vignette).
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    h, w = gray.shape

    # Threshold bright tissue vs dark background
    _, bin_mask = cv2.threshold(gray, dark_threshold, 255, cv2.THRESH_BINARY)

    # Morphological close to bridge internal specular inpainting or thin vessels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    closed = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)

    # Find largest illuminated component (the endoscopic view aperture)
    n_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    if n_lbl <= 1:
        return np.ones((h, w), dtype=np.uint8) * 255

    main_lbl = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    tissue_comp = (lbls == main_lbl).astype(np.uint8) * 255

    # Fill external contour to ensure all internal lumen/shadows remain valid anatomical field
    cnts, _ = cv2.findContours(tissue_comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.ones((h, w), dtype=np.uint8) * 255

    c = max(cnts, key=cv2.contourArea)
    fov_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(fov_mask, [c], -1, 255, thickness=-1)

    return fov_mask


class ActiveContourSegmenterV2:
    """Enhanced active contour segmenter utilizing MorphGAC, vascular saliency, and component selection."""

    def __init__(
        self,
        config: Optional[GlobalConfig] = None,
        specular_threshold: Optional[int] = None,
        specular_inpaint_radius: Optional[int] = None,
        specular_inpaint_method: Optional[str] = None,
        clahe_clip_limit: Optional[float] = None,
        clahe_grid_size: Optional[Tuple[int, int]] = None,
        lumen_dark_threshold: Optional[int] = None,
        seed_saliency_percentile: Optional[float] = None,
        seed_min_area: Optional[int] = None,
        seed_max_area: Optional[int] = None,
        seed_contrast_floor: Optional[float] = None,
        gac_iterations: Optional[int] = None,
        gac_smoothing: Optional[int] = None,
        gac_balloon: Optional[float] = None,
        gac_threshold: Optional[float] = None,
        gac_sigma: Optional[float] = None,
        gac_alpha: Optional[float] = None,
        min_component_area_ratio: Optional[float] = None,
        max_component_area_ratio: Optional[float] = None,
        weight_log_area: Optional[float] = None,
        weight_contrast: Optional[float] = None,
        weight_centrality: Optional[float] = None,
        weight_border_penalty: Optional[float] = None,
        acceptance_threshold: Optional[float] = None,
    ) -> None:
        """Initializes ActiveContourSegmenterV2 with configurable hyperparameters.

        Explicit arguments take precedence over GlobalConfig values.
        """
        self.config = config or get_config()
        cfg = getattr(self.config, "segmentation_v2", None)

        self.specular_threshold = (
            specular_threshold
            if specular_threshold is not None
            else (getattr(cfg, "specular_threshold", 245) if cfg else 245)
        )
        self.specular_inpaint_radius = (
            specular_inpaint_radius
            if specular_inpaint_radius is not None
            else (getattr(cfg, "specular_inpaint_radius", 3) if cfg else 3)
        )
        self.specular_inpaint_method = (
            specular_inpaint_method
            if specular_inpaint_method is not None
            else (getattr(cfg, "specular_inpaint_method", "telea") if cfg else "telea")
        )

        self.clahe_clip_limit = (
            clahe_clip_limit
            if clahe_clip_limit is not None
            else (getattr(cfg, "clahe_clip_limit", 2.0) if cfg else 2.0)
        )
        self.clahe_grid_size = (
            clahe_grid_size
            if clahe_grid_size is not None
            else (getattr(cfg, "clahe_grid_size", (8, 8)) if cfg else (8, 8))
        )
        self.lumen_dark_threshold = (
            lumen_dark_threshold
            if lumen_dark_threshold is not None
            else (getattr(cfg, "lumen_dark_threshold", 30) if cfg else 30)
        )

        self.seed_saliency_percentile = (
            seed_saliency_percentile
            if seed_saliency_percentile is not None
            else (getattr(cfg, "seed_saliency_percentile", 90.0) if cfg else 90.0)
        )
        self.seed_min_area = (
            seed_min_area
            if seed_min_area is not None
            else (getattr(cfg, "seed_min_area", 20) if cfg else 20)
        )
        self.seed_max_area = (
            seed_max_area
            if seed_max_area is not None
            else (getattr(cfg, "seed_max_area", 6000) if cfg else 6000)
        )
        self.seed_contrast_floor = (
            seed_contrast_floor
            if seed_contrast_floor is not None
            else (getattr(cfg, "seed_contrast_floor", 0.15) if cfg else 0.15)
        )

        self.gac_iterations = (
            gac_iterations
            if gac_iterations is not None
            else (getattr(cfg, "gac_iterations", 80) if cfg else 80)
        )
        self.gac_smoothing = (
            gac_smoothing
            if gac_smoothing is not None
            else (getattr(cfg, "gac_smoothing", 1) if cfg else 1)
        )
        self.gac_balloon = (
            gac_balloon
            if gac_balloon is not None
            else (getattr(cfg, "gac_balloon", 1.0) if cfg else 1.0)
        )
        self.gac_threshold = (
            gac_threshold
            if gac_threshold is not None
            else (getattr(cfg, "gac_threshold", 0.30) if cfg else 0.30)
        )
        self.gac_sigma = (
            gac_sigma
            if gac_sigma is not None
            else (getattr(cfg, "gac_sigma", 1.5) if cfg else 1.5)
        )
        self.gac_alpha = (
            gac_alpha
            if gac_alpha is not None
            else (getattr(cfg, "gac_alpha", 100.0) if cfg else 100.0)
        )

        self.min_component_area_ratio = (
            min_component_area_ratio
            if min_component_area_ratio is not None
            else (getattr(cfg, "min_component_area_ratio", 0.001) if cfg else 0.001)
        )
        self.max_component_area_ratio = (
            max_component_area_ratio
            if max_component_area_ratio is not None
            else (getattr(cfg, "max_component_area_ratio", 0.50) if cfg else 0.50)
        )
        self.weight_log_area = (
            weight_log_area
            if weight_log_area is not None
            else (getattr(cfg, "weight_log_area", 0.25) if cfg else 0.25)
        )
        self.weight_contrast = (
            weight_contrast
            if weight_contrast is not None
            else (getattr(cfg, "weight_contrast", 0.35) if cfg else 0.35)
        )
        self.weight_centrality = (
            weight_centrality
            if weight_centrality is not None
            else (getattr(cfg, "weight_centrality", 0.15) if cfg else 0.15)
        )
        self.weight_border_penalty = (
            weight_border_penalty
            if weight_border_penalty is not None
            else (getattr(cfg, "weight_border_penalty", 0.25) if cfg else 0.25)
        )
        self.acceptance_threshold = (
            acceptance_threshold
            if acceptance_threshold is not None
            else (getattr(cfg, "acceptance_threshold", 0.30) if cfg else 0.30)
        )

    @staticmethod
    def compute_endoscopic_fov_mask(
        image: np.ndarray,
        dark_threshold: int = 15,
        morph_kernel_size: int = 15,
    ) -> np.ndarray:
        """Computes a robust, vignette-aware Field-of-View (FOV) mask for endoscopic frames.

        Delegates to module-level compute_endoscopic_fov_mask.
        """
        return compute_endoscopic_fov_mask(
            image=image,
            dark_threshold=dark_threshold,
            morph_kernel_size=morph_kernel_size,
        )

    @staticmethod
    def get_black_border_mask(image: np.ndarray, dark_threshold: int = 15) -> np.ndarray:
        """Generates a binary mask of the active camera view excluding dark endoscopy borders.

        Args:
            image: (H, W) or (H, W, 3) uint8 image.
            dark_threshold: Pixel intensity threshold below which corners are marked background.

        Returns:
            (H, W) uint8 binary mask (255 inside active view, 0 in dark corners).
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        _, border_mask = cv2.threshold(gray, dark_threshold, 255, cv2.THRESH_BINARY)
        return border_mask

    def detect_specular_highlights(self, rgb_image: np.ndarray) -> np.ndarray:
        """Detects bright specular glare reflections on mucosal surfaces.

        Args:
            rgb_image: (H, W, 3) uint8 image.

        Returns:
            (H, W) uint8 binary mask (255 at specular locations, 0 elsewhere).
        """
        if rgb_image.ndim != 3:
            raise ValueError(f"Expected 3-channel RGB image, got shape {rgb_image.shape}.")

        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        max_channel = np.max(rgb_image, axis=2)

        # Highlight occurs where brightness exceeds specular threshold
        specular_raw = (max_channel >= self.specular_threshold) | (gray >= self.specular_threshold)
        specular_mask = (specular_raw.astype(np.uint8)) * 255

        # Restrict to active endoscopy field
        border_mask = self.get_black_border_mask(rgb_image)
        specular_mask = cv2.bitwise_and(specular_mask, border_mask)

        # Mild dilation (3x3 ellipse) to cover specular halo transition
        if np.count_nonzero(specular_mask) > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            specular_mask = cv2.dilate(specular_mask, kernel, iterations=1)

        return specular_mask

    def inpaint_specular_highlights(
        self, rgb_image: np.ndarray, highlight_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Inpaints specular glare spots using Telea or Navier-Stokes interpolation.

        Args:
            rgb_image: (H, W, 3) uint8 image.
            highlight_mask: Optional (H, W) uint8 binary highlight mask.

        Returns:
            (H, W, 3) uint8 glare-suppressed image.
        """
        if highlight_mask is None:
            highlight_mask = self.detect_specular_highlights(rgb_image)

        if np.count_nonzero(highlight_mask) == 0:
            return rgb_image.copy()

        method = (
            cv2.INPAINT_NS
            if self.specular_inpaint_method.lower() == "ns"
            else cv2.INPAINT_TELEA
        )

        inpainted = cv2.inpaint(
            rgb_image,
            highlight_mask,
            inpaintRadius=self.specular_inpaint_radius,
            flags=method,
        )
        return inpainted

    def compute_saliency_map(self, rgb_image: np.ndarray) -> np.ndarray:
        """Computes Green-channel vascular and mucosal tissue contrast saliency with lumen suppression.

        Args:
            rgb_image: (H, W, 3) uint8 image.

        Returns:
            (H, W) float32 saliency map normalized to [0.0, 1.0].
        """
        h, w = rgb_image.shape[:2]
        r = rgb_image[:, :, 0].astype(np.float32)
        g = rgb_image[:, :, 1].astype(np.float32)
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

        # 1. Hemoglobin absorption in Green channel:
        # Vascular tissue strongly absorbs green; inverted Green CLAHE enhances vessel/polyp peaks.
        inv_g = (255 - rgb_image[:, :, 1]).astype(np.uint8)
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_grid_size
        )
        g_clahe = clahe.apply(inv_g).astype(np.float32) / 255.0

        # 2. Normalized difference index (NDI): (R - G) / (R + G + 1)
        ndi = np.clip((r - g) / (r + g + 1.0), 0.0, 1.0)

        # Composite raw saliency
        saliency_raw = g_clahe * (1.0 + ndi)

        # 3. Lumen suppression:
        # Deep intestinal cavity (lumen) appears as a dark void across all channels.
        lumen_mask = (gray < self.lumen_dark_threshold).astype(np.float32)
        lumen_blur = cv2.GaussianBlur(lumen_mask, (15, 15), 5.0)
        lumen_weight = np.clip(1.0 - lumen_blur, 0.0, 1.0)

        # 4. Active border mask suppression (erode to avoid edge artifacts)
        border_mask = self.get_black_border_mask(rgb_image)
        border_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        border_eroded = cv2.erode(border_mask, border_kernel).astype(np.float32) / 255.0

        saliency = saliency_raw * lumen_weight * border_eroded

        # 5. Normalization to [0.0, 1.0]
        min_val, max_val = float(np.min(saliency)), float(np.max(saliency))
        if max_val - min_val > 1e-6:
            saliency_norm = (saliency - min_val) / (max_val - min_val)
        else:
            saliency_norm = np.zeros((h, w), dtype=np.float32)

        return saliency_norm.astype(np.float32)

    def generate_seeds(
        self, saliency_map: np.ndarray, rgb_image: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Generates localized candidate seed level set from the saliency map.

        DOES NOT return sprawling full-frame Otsu masks or large fallback central discs.
        Returns an all-zero mask if no meaningful candidate exists.

        Args:
            saliency_map: (H, W) float32 saliency map in [0, 1].
            rgb_image: Optional (H, W, 3) RGB image for border masking.

        Returns:
            (H, W) uint8 binary level set (1 inside seed disk, 0 outside).
        """
        h, w = saliency_map.shape
        seed_mask = np.zeros((h, w), dtype=np.uint8)

        # Mask valid pixels within active endoscopy view
        if rgb_image is not None:
            border_mask = self.get_black_border_mask(rgb_image)
            border_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            active_field = cv2.erode(border_mask, border_kernel) > 0
        else:
            active_field = np.ones((h, w), dtype=bool)

        valid_vals = saliency_map[active_field]
        if valid_vals.size == 0:
            return seed_mask

        med_sal = float(np.median(valid_vals))
        max_sal = float(np.max(valid_vals))

        # If peak contrast above mucosal background is weak (< seed_contrast_floor), no polyp candidate
        if (max_sal - med_sal) < self.seed_contrast_floor:
            return seed_mask

        # Prominent elevation above background
        tau_seed = med_sal + 0.35 * (max_sal - med_sal)
        candidates_bin = ((saliency_map >= tau_seed) & active_field).astype(np.uint8)

        # Morphological opening to clean small noise specks
        clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        candidates_bin = cv2.morphologyEx(candidates_bin, cv2.MORPH_OPEN, clean_kernel)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            candidates_bin, connectivity=8
        )

        if num_labels <= 1:
            return seed_mask

        # Select candidate with highest mean saliency and valid area
        best_label = 0
        best_seed_score = -1.0

        for lbl in range(1, num_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if not (self.seed_min_area <= area <= self.seed_max_area):
                continue

            comp_mask = labels == lbl
            mean_sal = float(np.mean(saliency_map[comp_mask]))

            # Weak centrality prior for seed ranking: center distance penalty is mild (at most 10%)
            cx, cy = centroids[lbl]
            center_dist = np.hypot(cx - (w / 2.0), cy - (h / 2.0))
            max_dist = 0.5 * np.hypot(w, h)
            centrality_factor = 1.0 - 0.10 * (center_dist / max(max_dist, 1.0))

            seed_score = mean_sal * centrality_factor
            if seed_score > best_seed_score:
                best_seed_score = seed_score
                best_label = lbl

        if best_label == 0:
            return seed_mask

        # Generate a compact, localized seed circle positioned at the component's distance peak
        best_comp_mask = (labels == best_label).astype(np.uint8)
        dist_transform = cv2.distanceTransform(best_comp_mask, cv2.DIST_L2, 5)
        max_dist = float(np.max(dist_transform))

        if max_dist < 2.0:
            return seed_mask

        y_peak, x_peak = np.unravel_index(np.argmax(dist_transform), dist_transform.shape)
        # Compact localized disk radius (capped at 25px to prevent sprawling)
        seed_radius = int(np.clip(max_dist * 0.75, 4, 25))

        cv2.circle(seed_mask, (int(x_peak), int(y_peak)), radius=seed_radius, color=1, thickness=-1)
        return seed_mask

    def evolve_morphgac(
        self, image_gray: np.ndarray, seed_mask: np.ndarray
    ) -> np.ndarray:
        """Evolves Morphological Geodesic Active Contour constrained by image gradients.

        Args:
            image_gray: (H, W) uint8 grayscale image.
            seed_mask: (H, W) uint8 binary initial level set mask (1 inside, 0 outside).

        Returns:
            (H, W) uint8 binary mask (values 0 or 255).
        """
        if np.count_nonzero(seed_mask) == 0:
            return np.zeros_like(image_gray, dtype=np.uint8)

        # Normalize image to [0, 1] for inverse gaussian gradient calculation
        img_flt = image_gray.astype(np.float64) / 255.0
        gimage = inverse_gaussian_gradient(
            img_flt, alpha=self.gac_alpha, sigma=self.gac_sigma
        )

        # Evolve MorphGAC
        cv_res = morphological_geodesic_active_contour(
            gimage=gimage,
            num_iter=self.gac_iterations,
            init_level_set=seed_mask,
            smoothing=self.gac_smoothing,
            threshold=self.gac_threshold,
            balloon=self.gac_balloon,
        )

        return (cv_res.astype(np.uint8)) * 255

    def score_components(
        self,
        binary_mask: np.ndarray,
        rgb_image: np.ndarray,
        saliency_map: np.ndarray,
        fov_mask: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """Extracts and evaluates connected components using multi-criteria scoring.

        Criteria:
        1. Log Area: S_area in [0, 1]
        2. Boundary Contrast: S_contrast in [0, 1] (outer rim evaluated relative to valid FOV)
        3. Weak Centrality: S_centrality in [0, 1] (weak heuristic, never disqualifies off-center lesions)
        4. Border-Touch Penalty: P_border in {0, 1} (distinguishes camera vignette / canvas edge from internal shadows)

        Args:
            binary_mask: (H, W) uint8 active contour binary mask (0 or 255).
            rgb_image: (H, W, 3) uint8 original or inpainted RGB image.
            saliency_map: (H, W) float32 saliency map.
            fov_mask: Optional (H, W) uint8 binary FOV mask. If None, computed from rgb_image.

        Returns:
            List of candidate component dictionaries sorted by score descending.
        """
        h, w = binary_mask.shape
        total_pixels = h * w
        max_dist = 0.5 * np.hypot(w, h)
        center_x, center_y = w / 2.0, h / 2.0

        if fov_mask is None:
            fov_mask = self.compute_endoscopic_fov_mask(rgb_image)

        # Candidate components are clipped/interpreted relative to valid FOV
        clipped_mask = cv2.bitwise_and(binary_mask, fov_mask)

        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            (clipped_mask > 0).astype(np.uint8), connectivity=8
        )

        candidates = []
        kernel_boundary = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_adj = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        for lbl in range(1, num_labels):
            area = int(stats[lbl, cv2.CC_STAT_AREA])
            area_ratio = area / float(total_pixels)

            # Reject components outside plausible polyp size bounds
            if not (self.min_component_area_ratio <= area_ratio <= self.max_component_area_ratio):
                continue

            comp_bin = (labels == lbl).astype(np.uint8)
            cx, cy = float(centroids[lbl][0]), float(centroids[lbl][1])

            # 1. Log Area Score: Normalized log of area
            # Reaches 1.0 at ~15% of frame area
            s_area = float(
                np.clip(
                    np.log(area + 1) / np.log(0.15 * total_pixels + 1),
                    0.0,
                    1.0,
                )
            )

            # 2. Boundary Contrast:
            # Contrast between inner mucosal edge and outer surrounding mucosa strip within valid FOV
            dilated = cv2.dilate(comp_bin, kernel_boundary, iterations=1)
            eroded = cv2.erode(comp_bin, kernel_boundary, iterations=1)
            outer_rim = (dilated > 0) & (comp_bin == 0) & (fov_mask > 0)
            inner_rim = (comp_bin > 0) & (eroded == 0)

            if np.any(inner_rim) and np.any(outer_rim):
                inner_mean = float(np.mean(gray[inner_rim]))
                outer_mean = float(np.mean(gray[outer_rim]))
                contrast_val = abs(inner_mean - outer_mean)
                # Saliency contrast boost
                sal_inner = float(np.mean(saliency_map[comp_bin > 0]))
                sal_outer = float(np.mean(saliency_map[outer_rim]))
                sal_diff = max(0.0, sal_inner - sal_outer)
            else:
                contrast_val = 0.0
                sal_diff = 0.0

            # Scale boundary contrast smoothly to [0, 1]
            s_contrast = float(
                np.clip((contrast_val / 35.0) * 0.7 + (sal_diff / 0.5) * 0.3, 0.0, 1.0)
            )

            # 3. Weak Centrality:
            # Center of camera receives 1.0; far boundary receives ~0.15.
            dist_to_center = np.hypot(cx - center_x, cy - center_y)
            s_centrality = float(np.clip(1.0 - (dist_to_center / max_dist), 0.0, 1.0))

            # 4. Border Touching Penalty (B1: Vignette-Aware):
            # Genuine rectangular canvas-edge touching remains penalized
            touches_canvas_edge = bool(
                np.any(comp_bin[0, :])
                or np.any(comp_bin[h - 1, :])
                or np.any(comp_bin[:, 0])
                or np.any(comp_bin[:, w - 1])
            )

            # Camera vignette touching (dilated component touches fov_mask == 0 or overlap)
            comp_dilated = cv2.dilate(comp_bin, kernel_adj, iterations=1)
            vignette_overlap_px = int(np.count_nonzero((comp_bin > 0) & (fov_mask == 0)))
            touches_vignette_boundary = bool(
                vignette_overlap_px > 0 or np.any((comp_dilated > 0) & (fov_mask == 0))
            )

            # Internal dark mucosal shadows inside fov_mask do NOT trigger border penalty
            touches_edge = bool(touches_canvas_edge or touches_vignette_boundary)
            p_border = 1.0 if touches_edge else 0.0

            # Multi-criteria composite score
            score = (
                self.weight_log_area * s_area
                + self.weight_contrast * s_contrast
                + self.weight_centrality * s_centrality
                - self.weight_border_penalty * p_border
            )
            score = float(np.clip(score, 0.0, 1.0))

            candidates.append(
                {
                    "label": lbl,
                    "area": area,
                    "area_ratio": area_ratio,
                    "centroid": (cx, cy),
                    "score": score,
                    "s_area": s_area,
                    "s_contrast": s_contrast,
                    "s_centrality": s_centrality,
                    "p_border": p_border,
                    "touches_border": touches_edge,
                    "touches_canvas_edge": touches_canvas_edge,
                    "touches_vignette": touches_vignette_boundary,
                    "vignette_overlap_px": vignette_overlap_px,
                    "comp_mask": comp_bin,
                }
            )

        candidates.sort(key=lambda c: c["score"], reverse=True)
        return candidates

    def segment(
        self, preprocessed_image: np.ndarray, rgb_image: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        """Evolves active contour v2 pipeline to segment endoscopic polyp.

        DOES NOT access or accept ground-truth masks.
        Returns empty mask with success=False when no confident lesion is detected.

        Args:
            preprocessed_image: (H, W) uint8 single-channel preprocessed image array.
            rgb_image: Optional (H, W, 3) uint8 original RGB image. If None,
                       synthesizes 3-channel image from preprocessed image.

        Returns:
            Tuple of:
                - pred_mask: (H, W) uint8 binary mask with values 0 or 255.
                - success: bool indicating whether a valid confident polyp was segmented.
                - info: dict containing segmentation diagnostics and confidence metadata.
        """
        if preprocessed_image.ndim != 2:
            raise ValueError(
                f"Expected 2D single-channel image, got shape {preprocessed_image.shape}."
            )

        h, w = preprocessed_image.shape
        max_dim = 512

        # Ensure valid 3-channel RGB image
        if rgb_image is None or rgb_image.ndim != 3:
            rgb_work = cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2RGB)
        else:
            rgb_work = rgb_image

        # Scale down large frames for fast MorphGAC computation
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            proc_sub = cv2.resize(
                preprocessed_image, (new_w, new_h), interpolation=cv2.INTER_AREA
            )
            rgb_sub = cv2.resize(
                rgb_work, (new_w, new_h), interpolation=cv2.INTER_AREA
            )
        else:
            new_h, new_w = h, w
            proc_sub = preprocessed_image
            rgb_sub = rgb_work

        try:
            # 1. Specular Highlight Inpainting
            inpainted_rgb = self.inpaint_specular_highlights(rgb_sub)

            # 2. Vascular / Polyp Candidate Saliency Map
            saliency_map = self.compute_saliency_map(inpainted_rgb)

            # 3. Localized Candidate Seed Generation
            seed_mask = self.generate_seeds(saliency_map, rgb_image=inpainted_rgb)

            if np.count_nonzero(seed_mask) == 0:
                empty_mask = np.zeros((h, w), dtype=np.uint8)
                info = {
                    "status": "no_candidate_seed",
                    "confidence": 0.0,
                    "selected_component_score": 0.0,
                    "num_candidates": 0,
                    "fg_pixels": 0,
                    "fg_ratio_pct": 0.0,
                    "success": False,
                }
                return empty_mask, False, info

            # 4. MorphGAC Evolution
            inpainted_gray = cv2.cvtColor(inpainted_rgb, cv2.COLOR_RGB2GRAY)
            gac_mask = self.evolve_morphgac(inpainted_gray, seed_mask)

            # Compute vignette-aware FOV mask for working resolution
            fov_mask_sub = self.compute_endoscopic_fov_mask(rgb_sub)

            # 5. Connected Component Analysis and Multi-Criteria Scoring (B1)
            # Candidate pixels outside valid FOV are excluded before scoring
            eval_mask = cv2.bitwise_and(gac_mask, fov_mask_sub)
            candidates = self.score_components(
                eval_mask, inpainted_rgb, saliency_map, fov_mask=fov_mask_sub
            )

            # 6. Low-Confidence Rejection
            if not candidates or candidates[0]["score"] < self.acceptance_threshold:
                empty_mask = np.zeros((h, w), dtype=np.uint8)
                best_score = float(candidates[0]["score"]) if candidates else 0.0
                info = {
                    "status": "rejected_candidate" if candidates else "no_valid_component",
                    "confidence": best_score,
                    "selected_component_score": best_score,
                    "num_candidates": len(candidates),
                    "fg_pixels": 0,
                    "fg_ratio_pct": 0.0,
                    "success": False,
                }
                return empty_mask, False, info

            # 7. Refine Accepted Top Candidate Component
            best_comp = candidates[0]
            selected_mask = (best_comp["comp_mask"] * 255).astype(np.uint8)

            # Fill internal holes of selected component strictly
            contours, _ = cv2.findContours(
                selected_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            filled_mask = np.zeros_like(selected_mask)
            if contours:
                cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)
                selected_mask = filled_mask

            # Morphological smoothing to eliminate contour staircasing
            smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            selected_mask = cv2.morphologyEx(selected_mask, cv2.MORPH_CLOSE, smooth_kernel)

            # Upscale back to original resolution if downscaled
            if (new_h, new_w) != (h, w):
                pred_mask = cv2.resize(
                    selected_mask, (w, h), interpolation=cv2.INTER_NEAREST
                )
                fov_mask_full = cv2.resize(
                    fov_mask_sub, (w, h), interpolation=cv2.INTER_NEAREST
                )
            else:
                pred_mask = selected_mask
                fov_mask_full = fov_mask_sub

            # Exclude dark camera vignette using FOV mask (B1)
            pred_mask = cv2.bitwise_and(pred_mask, fov_mask_full)

            fg_pixels = int(np.count_nonzero(pred_mask == 255))
            fg_ratio = fg_pixels / float(h * w)

            # Final sanity check: valid if non-empty and <= 55% of frame
            success = bool(fg_pixels > 0 and fg_ratio <= 0.55)
            if not success:
                pred_mask = np.zeros((h, w), dtype=np.uint8)
                status = "rejected_overexpansion" if fg_ratio > 0.55 else "empty_post_mask"
            else:
                status = "valid_candidate"

            info = {
                "status": status,
                "confidence": float(best_comp["score"]),
                "selected_component_score": float(best_comp["score"]),
                "num_candidates": len(candidates),
                "fg_pixels": fg_pixels if success else 0,
                "fg_ratio_pct": float(fg_ratio * 100.0) if success else 0.0,
                "success": success,
                "centroid": best_comp["centroid"],
                "s_area": best_comp["s_area"],
                "s_contrast": best_comp["s_contrast"],
                "s_centrality": best_comp["s_centrality"],
                "p_border": best_comp["p_border"],
            }

            return pred_mask, success, info

        except Exception as e:
            logger.error(f"Active contour v2 segmentation failed: {str(e)}")
            empty_mask = np.zeros((h, w), dtype=np.uint8)
            info = {
                "status": "segmentation_failure",
                "error": str(e),
                "confidence": 0.0,
                "selected_component_score": 0.0,
                "num_candidates": 0,
                "fg_pixels": 0,
                "fg_ratio_pct": 0.0,
                "success": False,
            }
            return empty_mask, False, info
