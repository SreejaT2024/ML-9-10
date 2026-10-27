"""Controlled Experiment for Border/Vignette Handling in ActiveContourSegmenterV2.

Compares:
  B0: Current generic border penalty (unclipped scoring, touches edge or gray<15)
  B1: Vignette-aware FOV mask, pixels outside FOV excluded before scoring,
      border penalty applied to vignette boundary and canvas edge
  B2: Vignette-aware FOV mask + border penalty applied ONLY to true canvas edge

Fixed MorphGAC (A1_iter80) parameters:
  gac_iterations = 80
  gac_smoothing = 1
  gac_balloon = 1.0
  gac_threshold = 0.30
  gac_sigma = 1.5
  seed_contrast_floor = 0.15
  acceptance_threshold = 0.30

Evaluated strictly on the 20-image development pilot.
Output saved to results/v2_vignette_analysis/
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import get_config
from src.utils.io import load_image, load_mask
from src.preprocessing.image_preprocess import PolypImagePreprocessor
from src.segmentation.active_contour_v2 import ActiveContourSegmenterV2
from src.segmentation.metrics import evaluate_segmentation


def compute_endoscopic_fov_mask(
    image: np.ndarray,
    dark_threshold: int = 15,
    morph_kernel_size: int = 15,
) -> np.ndarray:
    """Computes a robust, vignette-aware Field-of-View (FOV) mask for endoscopic frames.

    Distinguishes the dark circular/barrel camera vignette outside the active lens view
    from internal dark mucosal cavities, lumen, or shadows within the illuminated field.

    Args:
        image: (H, W) or (H, W, 3) uint8 image.
        dark_threshold: Intensity threshold below which dark pixels are detected.
        morph_kernel_size: Size of elliptical structuring element for closing.

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


def score_components_variant(
    binary_mask: np.ndarray,
    rgb_image: np.ndarray,
    saliency_map: np.ndarray,
    segmenter: ActiveContourSegmenterV2,
    variant: str,
    fov_mask: np.ndarray,
    raw_gac_mask: np.ndarray,
) -> List[Dict[str, Any]]:
    """Evaluates connected components under variant B0, B1, or B2 border handling.

    Args:
        binary_mask: (H, W) uint8 binary mask to extract components from.
        rgb_image: (H, W, 3) uint8 image.
        saliency_map: (H, W) float32 saliency map.
        segmenter: ActiveContourSegmenterV2 instance.
        variant: 'B0', 'B1', or 'B2'.
        fov_mask: (H, W) uint8 FOV mask.
        raw_gac_mask: (H, W) uint8 unclipped MorphGAC mask.

    Returns:
        List of scored candidate components sorted descending by score.
    """
    h, w = binary_mask.shape
    total_pixels = h * w
    max_dist = 0.5 * np.hypot(w, h)
    center_x, center_y = w / 2.0, h / 2.0

    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    border_mask_crude = segmenter.get_black_border_mask(rgb_image)  # gray > 15

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (binary_mask > 0).astype(np.uint8), connectivity=8
    )

    candidates = []
    kernel_boundary = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_adj = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for lbl in range(1, num_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        area_ratio = area / float(total_pixels)

        if not (segmenter.min_component_area_ratio <= area_ratio <= segmenter.max_component_area_ratio):
            continue

        comp_bin = (labels == lbl).astype(np.uint8)
        cx, cy = float(centroids[lbl][0]), float(centroids[lbl][1])

        # 1. Log Area Score
        s_area = float(
            np.clip(
                np.log(area + 1) / np.log(0.15 * total_pixels + 1),
                0.0,
                1.0,
            )
        )

        # 2. Boundary Contrast
        dilated = cv2.dilate(comp_bin, kernel_boundary, iterations=1)
        eroded = cv2.erode(comp_bin, kernel_boundary, iterations=1)

        # For outer rim, use active region
        active_valid = fov_mask > 0 if variant in ("B1", "B2") else border_mask_crude > 0
        outer_rim = (dilated > 0) & (comp_bin == 0) & active_valid
        inner_rim = (comp_bin > 0) & (eroded == 0)

        if np.any(inner_rim) and np.any(outer_rim):
            inner_mean = float(np.mean(gray[inner_rim]))
            outer_mean = float(np.mean(gray[outer_rim]))
            contrast_val = abs(inner_mean - outer_mean)
            sal_inner = float(np.mean(saliency_map[comp_bin > 0]))
            sal_outer = float(np.mean(saliency_map[outer_rim]))
            sal_diff = max(0.0, sal_inner - sal_outer)
        else:
            contrast_val = 0.0
            sal_diff = 0.0

        s_contrast = float(
            np.clip((contrast_val / 35.0) * 0.7 + (sal_diff / 0.5) * 0.3, 0.0, 1.0)
        )

        # 3. Centrality
        dist_to_center = np.hypot(cx - center_x, cy - center_y)
        s_centrality = float(np.clip(1.0 - (dist_to_center / max_dist), 0.0, 1.0))

        # 4. Border Penalty & Touching Detection
        touches_canvas_edge = bool(
            np.any(comp_bin[0, :])
            or np.any(comp_bin[h - 1, :])
            or np.any(comp_bin[:, 0])
            or np.any(comp_bin[:, w - 1])
        )

        # Vignette overlap / adjacency
        # Check if dilated component touches vignette (fov_mask == 0) or unclipped had pixels in vignette
        comp_dilated = cv2.dilate(comp_bin, kernel_adj, iterations=1)
        vignette_overlap_px = int(np.count_nonzero((comp_bin > 0) & (fov_mask == 0)))
        touches_vignette_boundary = bool(
            vignette_overlap_px > 0 or np.any((comp_dilated > 0) & (fov_mask == 0))
        )

        if variant == "B0":
            # Current generic implementation: touches canvas edge OR any pixel where gray < 15
            touches_edge = bool(
                touches_canvas_edge
                or np.any(comp_bin[border_mask_crude == 0])
            )
            p_border = 1.0 if touches_edge else 0.0

        elif variant == "B1":
            # Vignette-aware: candidate clipped to FOV. Border penalty applied if touching
            # canvas edge OR touching vignette boundary
            touches_edge = bool(touches_canvas_edge or touches_vignette_boundary)
            p_border = 1.0 if touches_edge else 0.0

        elif variant == "B2":
            # Vignette-aware: candidate clipped to FOV. Border penalty applied ONLY to
            # TRUE valid image boundary (canvas edge: row 0, h-1, col 0, w-1)
            touches_edge = touches_canvas_edge
            p_border = 1.0 if touches_edge else 0.0
        else:
            raise ValueError(f"Unknown variant: {variant}")

        # Multi-criteria score
        score = (
            segmenter.weight_log_area * s_area
            + segmenter.weight_contrast * s_contrast
            + segmenter.weight_centrality * s_centrality
            - segmenter.weight_border_penalty * p_border
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


def run_pipeline_for_variant(
    variant: str,
    cached_data: List[Dict[str, Any]],
    segmenter: ActiveContourSegmenterV2,
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    """Runs segmentation pipeline across all samples for a specific border variant."""
    results = []
    masks = {}

    for idx, item in enumerate(cached_data):
        fn = item["filename"]
        rgb_img = item["rgb"]
        prep_img = item["prep"]
        gt_mask = item["gt"]
        h, w = rgb_img.shape[:2]

        max_dim = 512
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            proc_sub = cv2.resize(prep_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            rgb_sub = cv2.resize(rgb_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            new_h, new_w = h, w
            proc_sub, rgb_sub = prep_img, rgb_img

        inpainted_rgb = segmenter.inpaint_specular_highlights(rgb_sub)
        saliency_map = segmenter.compute_saliency_map(inpainted_rgb)

        # Seed generation
        seed_mask = segmenter.generate_seeds(saliency_map, rgb_image=inpainted_rgb)

        # Compute vignette-aware FOV mask for the frame
        fov_mask_sub = compute_endoscopic_fov_mask(rgb_sub, dark_threshold=15)

        if np.count_nonzero(seed_mask) == 0:
            pred_mask = np.zeros((h, w), dtype=np.uint8)
            masks[fn] = pred_mask
            results.append({
                "variant": variant,
                "index": idx + 1,
                "filename": fn,
                "component_score": 0.0,
                "border_penalty": 0.0,
                "vignette_overlap": 0,
                "status": "No Seed",
                "dice": 0.0,
                "iou": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "predicted_area": 0,
                "gt_area": int(np.count_nonzero(gt_mask > 127)),
                "s_area": 0.0,
                "s_contrast": 0.0,
                "s_centrality": 0.0,
            })
            continue

        inpainted_gray = cv2.cvtColor(inpainted_rgb, cv2.COLOR_RGB2GRAY)
        raw_gac_mask = segmenter.evolve_morphgac(inpainted_gray, seed_mask)

        # Prepare mask for component scoring
        if variant == "B0":
            # Unclipped MorphGAC mask evaluated
            eval_mask = raw_gac_mask
        else:
            # B1 and B2: candidate pixels outside valid FOV excluded before component scoring
            eval_mask = cv2.bitwise_and(raw_gac_mask, fov_mask_sub)

        candidates = score_components_variant(
            binary_mask=eval_mask,
            rgb_image=inpainted_rgb,
            saliency_map=saliency_map,
            segmenter=segmenter,
            variant=variant,
            fov_mask=fov_mask_sub,
            raw_gac_mask=raw_gac_mask,
        )

        if not candidates or candidates[0]["score"] < segmenter.acceptance_threshold:
            pred_mask = np.zeros((h, w), dtype=np.uint8)
            masks[fn] = pred_mask
            best_c = candidates[0] if candidates else None
            results.append({
                "variant": variant,
                "index": idx + 1,
                "filename": fn,
                "component_score": float(best_c["score"]) if best_c else 0.0,
                "border_penalty": float(best_c["p_border"]) if best_c else 0.0,
                "vignette_overlap": int(best_c["vignette_overlap_px"]) if best_c else 0,
                "status": "Rejected",
                "dice": 0.0,
                "iou": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "predicted_area": 0,
                "gt_area": int(np.count_nonzero(gt_mask > 127)),
                "s_area": float(best_c["s_area"]) if best_c else 0.0,
                "s_contrast": float(best_c["s_contrast"]) if best_c else 0.0,
                "s_centrality": float(best_c["s_centrality"]) if best_c else 0.0,
            })
            continue

        # Accepted candidate refinement
        best_comp = candidates[0]
        selected_mask = (best_comp["comp_mask"] * 255).astype(np.uint8)

        # Fill internal holes strictly
        contours, _ = cv2.findContours(selected_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_mask = np.zeros_like(selected_mask)
        if contours:
            cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)
            selected_mask = filled_mask

        smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        selected_mask = cv2.morphologyEx(selected_mask, cv2.MORPH_CLOSE, smooth_kernel)

        if (new_h, new_w) != (h, w):
            pred_mask = cv2.resize(selected_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            fov_mask_full = cv2.resize(fov_mask_sub, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            pred_mask = selected_mask
            fov_mask_full = fov_mask_sub

        # Final FOV clipping
        if variant == "B0":
            border_mask_full = segmenter.get_black_border_mask(rgb_img)
            pred_mask = cv2.bitwise_and(pred_mask, border_mask_full)
        else:
            pred_mask = cv2.bitwise_and(pred_mask, fov_mask_full)

        fg_pixels = int(np.count_nonzero(pred_mask == 255))
        fg_ratio = fg_pixels / float(h * w)

        if fg_pixels > 0 and fg_ratio <= 0.55:
            eval_metrics = evaluate_segmentation(pred_mask, gt_mask)
            status = "Accepted"
        else:
            pred_mask = np.zeros((h, w), dtype=np.uint8)
            eval_metrics = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
            status = "Rejected"

        masks[fn] = pred_mask
        results.append({
            "variant": variant,
            "index": idx + 1,
            "filename": fn,
            "component_score": float(best_comp["score"]),
            "border_penalty": float(best_comp["p_border"]),
            "vignette_overlap": int(best_comp["vignette_overlap_px"]),
            "status": status,
            "dice": eval_metrics["dice"],
            "iou": eval_metrics["iou"],
            "precision": eval_metrics["precision"],
            "recall": eval_metrics["recall"],
            "predicted_area": fg_pixels if status == "Accepted" else 0,
            "gt_area": int(np.count_nonzero(gt_mask > 127)),
            "s_area": float(best_comp["s_area"]),
            "s_contrast": float(best_comp["s_contrast"]),
            "s_centrality": float(best_comp["s_centrality"]),
        })

    return results, masks


def draw_boundary_overlay(
    rgb_image: np.ndarray,
    gt_mask: np.ndarray,
    b0_mask: np.ndarray,
    b_new_mask: np.ndarray,
) -> np.ndarray:
    """Draws Ground Truth (Green), B0 Mask (Red), and B_new Mask (Cyan) contours."""
    overlay = rgb_image.copy()

    # Ground Truth: Green (0, 255, 0)
    gt_cnt, _ = cv2.findContours(
        (gt_mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, gt_cnt, -1, (0, 255, 0), 2)

    # B0 Mask: Red (255, 0, 0)
    if np.count_nonzero(b0_mask) > 0:
        b0_cnt, _ = cv2.findContours(
            (b0_mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, b0_cnt, -1, (255, 0, 0), 2)

    # B_new Mask: Cyan (0, 200, 255)
    if np.count_nonzero(b_new_mask) > 0:
        bn_cnt, _ = cv2.findContours(
            (b_new_mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, bn_cnt, -1, (0, 200, 255), 2)

    return overlay


def main():
    print("=" * 80)
    print("V2 BORDER/VIGNETTE HANDLING CONTROLLED EXPERIMENT (B0 vs B1 vs B2)")
    print("=" * 80)

    config = get_config()
    images_dir = Path(config.paths.kvasir_images_dir)
    masks_dir = Path(config.paths.kvasir_masks_dir)

    output_dir = PROJECT_ROOT / "results" / "v2_vignette_analysis"
    vis_dir = output_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    pilot_fn_file = PROJECT_ROOT / "results" / "v2_pilot_20" / "pilot_20_filenames.txt"
    with open(pilot_fn_file, "r", encoding="utf-8") as f:
        pilot_filenames = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(pilot_filenames)} development pilot samples.")

    preprocessor = PolypImagePreprocessor()
    cached_data = []
    for fn in pilot_filenames:
        rgb_img = load_image(images_dir / fn)
        gt_mask = load_mask(masks_dir / fn)
        prep_img = preprocessor.preprocess_for_segmentation(rgb_img)
        cached_data.append({"filename": fn, "rgb": rgb_img, "prep": prep_img, "gt": gt_mask})

    # Fixed MorphGAC (A1_iter80) parameters
    fixed_params = {
        "gac_iterations": 80,
        "gac_smoothing": 1,
        "gac_balloon": 1.0,
        "gac_threshold": 0.30,
        "gac_sigma": 1.5,
        "seed_contrast_floor": 0.15,
        "acceptance_threshold": 0.30,
    }
    segmenter = ActiveContourSegmenterV2(**fixed_params)

    # 1. Run B0
    print("\n[1/3] Running B0 (Current Generic Border Penalty)...")
    res_b0, masks_b0 = run_pipeline_for_variant("B0", cached_data, segmenter)

    # 2. Run B1
    print("\n[2/3] Running B1 (Vignette-Aware FOV Mask, Pixels Outside Excluded Before Scoring)...")
    res_b1, masks_b1 = run_pipeline_for_variant("B1", cached_data, segmenter)

    # 3. Run B2
    print("\n[3/3] Running B2 (Vignette-Aware FOV Mask + Penalty Only on True Canvas Edge)...")
    res_b2, masks_b2 = run_pipeline_for_variant("B2", cached_data, segmenter)

    df_b0 = pd.DataFrame(res_b0)
    df_b1 = pd.DataFrame(res_b1)
    df_b2 = pd.DataFrame(res_b2)

    df_all = pd.concat([df_b0, df_b1, df_b2], ignore_index=True)
    df_all.to_csv(output_dir / "vignette_experiment_all_metrics.csv", index=False)

    # Summary table per variant
    summary_rows = []
    for var, df in [("B0", df_b0), ("B1", df_b1), ("B2", df_b2)]:
        acc_df = df[df["status"] == "Accepted"]
        summary_rows.append({
            "Variant": var,
            "Accepted Cases": f"{len(acc_df)} / {len(df)} ({len(acc_df)/len(df)*100:.1f}%)",
            "Rejected Cases": f"{len(df) - len(acc_df)} / {len(df)}",
            "Mean Dice (All 20)": f"{df['dice'].mean():.4f}",
            "Std Dice (All 20)": f"{df['dice'].std():.4f}",
            "Mean IoU (All 20)": f"{df['iou'].mean():.4f}",
            "Mean Precision (All 20)": f"{df['precision'].mean():.4f}",
            "Mean Recall (All 20)": f"{df['recall'].mean():.4f}",
            "Mean Dice (Accepted)": f"{acc_df['dice'].mean():.4f}" if len(acc_df) > 0 else "N/A",
            "Accepted Dice=0 Count": int(np.count_nonzero((df["status"] == "Accepted") & (df["dice"] == 0.0))),
        })

    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY TABLE:")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(output_dir / "vignette_experiment_summary.csv", index=False)

    # Comparison Table across 20 images
    comp_table_rows = []
    for i in range(len(cached_data)):
        r0 = df_b0.iloc[i]
        r1 = df_b1.iloc[i]
        r2 = df_b2.iloc[i]
        fn = r0["filename"]
        comp_table_rows.append({
            "idx": i + 1,
            "filename": fn,
            "B0_Score": r0["component_score"],
            "B0_BorderPen": r0["border_penalty"],
            "B0_Status": r0["status"],
            "B0_Dice": r0["dice"],
            "B1_Score": r1["component_score"],
            "B1_BorderPen": r1["border_penalty"],
            "B1_Status": r1["status"],
            "B1_Dice": r1["dice"],
            "B2_Score": r2["component_score"],
            "B2_BorderPen": r2["border_penalty"],
            "B2_Status": r2["status"],
            "B2_Dice": r2["dice"],
            "GT_Area": r0["gt_area"],
        })

    comp_df = pd.DataFrame(comp_table_rows)
    comp_df.to_csv(output_dir / "per_image_comparison.csv", index=False)

    # Save visual comparison panels for all 20 images
    print("\nGenerating visualizations under results/v2_vignette_analysis/visualizations/ ...")
    vis_cards = []
    for i, item in enumerate(cached_data):
        fn = item["filename"]
        rgb = item["rgb"]
        gt = item["gt"]
        m0 = masks_b0[fn]
        m1 = masks_b1[fn]
        m2 = masks_b2[fn]

        r0 = df_b0.iloc[i]
        r1 = df_b1.iloc[i]
        r2 = df_b2.iloc[i]

        overlay_b1 = draw_boundary_overlay(rgb, gt, m0, m1)
        overlay_b2 = draw_boundary_overlay(rgb, gt, m0, m2)

        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        axes[0].imshow(rgb)
        axes[0].set_title(f"Original: {fn[:16]}...", fontsize=10)
        axes[0].axis("off")

        axes[1].imshow(gt, cmap="gray")
        axes[1].set_title(f"Ground Truth\nArea: {r0['gt_area']} px", fontsize=10)
        axes[1].axis("off")

        axes[2].imshow(m0, cmap="gray")
        axes[2].set_title(
            f"B0 Mask ({r0['status']})\nScore: {r0['component_score']:.3f} | Dice: {r0['dice']:.3f}",
            fontsize=10,
        )
        axes[2].axis("off")

        axes[3].imshow(m2, cmap="gray")
        axes[3].set_title(
            f"B2 Mask ({r2['status']})\nScore: {r2['component_score']:.3f} | Dice: {r2['dice']:.3f}",
            fontsize=10,
        )
        axes[3].axis("off")

        axes[4].imshow(overlay_b2)
        axes[4].set_title(
            "Overlay: GT(Green)\nB0(Red) | B2(Cyan)",
            fontsize=10,
        )
        axes[4].axis("off")

        plt.tight_layout()
        stem = Path(fn).stem
        save_path = vis_dir / f"{i+1:02d}_{stem}_vignette_cmp.png"
        fig.savefig(save_path, dpi=120)
        plt.close(fig)

    print(f"Visualizations saved to {vis_dir}")


if __name__ == "__main__":
    main()
