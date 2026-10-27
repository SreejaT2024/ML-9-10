"""Diagnostic script to inspect border and vignette touching in ActiveContourSegmenterV2."""

import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import get_config
from src.utils.io import load_image, load_mask
from src.preprocessing.image_preprocess import PolypImagePreprocessor
from src.segmentation.active_contour_v2 import ActiveContourSegmenterV2
from src.segmentation.metrics import evaluate_segmentation


def main():
    cfg = get_config()
    img_dir = Path(cfg.paths.kvasir_images_dir)
    mask_dir = Path(cfg.paths.kvasir_masks_dir)
    prep = PolypImagePreprocessor()

    pilot_file = PROJECT_ROOT / "results" / "v2_pilot_20" / "pilot_20_filenames.txt"
    with open(pilot_file) as f:
        fns = [line.strip() for line in f if line.strip()]

    segmenter = ActiveContourSegmenterV2(
        gac_iterations=80,
        gac_smoothing=1,
        gac_balloon=1.0,
        gac_threshold=0.30,
        gac_sigma=1.5,
        seed_contrast_floor=0.15,
        acceptance_threshold=0.30,
    )

    rows = []
    for i, fn in enumerate(fns):
        rgb = load_image(str(img_dir / fn))
        gt = load_mask(str(mask_dir / fn))
        prep_img = prep.preprocess_for_segmentation(rgb)
        h, w = prep_img.shape
        max_dim = 512
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            proc_sub = cv2.resize(prep_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            rgb_sub = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            new_h, new_w = h, w
            proc_sub, rgb_sub = prep_img, rgb

        inp_rgb = segmenter.inpaint_specular_highlights(rgb_sub)
        sal = segmenter.compute_saliency_map(inp_rgb)
        seeds = segmenter.generate_seeds(sal, rgb_image=inp_rgb)

        if np.count_nonzero(seeds) == 0:
            rows.append({
                "idx": i + 1,
                "fn": fn,
                "has_seed": False,
                "num_cands": 0,
                "score": 0.0,
                "p_border": 0.0,
                "t_top": False,
                "t_bot": False,
                "t_left": False,
                "t_right": False,
                "t_vignette": False,
                "vignette_px": 0,
                "cand_area": 0,
                "gt_area": int(np.count_nonzero(gt)),
                "dice_if_accepted": 0.0,
            })
            continue

        gray_sub = cv2.cvtColor(inp_rgb, cv2.COLOR_RGB2GRAY)
        gac = segmenter.evolve_morphgac(gray_sub, seeds)
        cands = segmenter.score_components(gac, inp_rgb, sal)
        b_mask = segmenter.get_black_border_mask(rgb_sub)

        if not cands:
            rows.append({
                "idx": i + 1,
                "fn": fn,
                "has_seed": True,
                "num_cands": 0,
                "score": 0.0,
                "p_border": 0.0,
                "t_top": False,
                "t_bot": False,
                "t_left": False,
                "t_right": False,
                "t_vignette": False,
                "vignette_px": 0,
                "cand_area": 0,
                "gt_area": int(np.count_nonzero(gt)),
                "dice_if_accepted": 0.0,
            })
            continue

        c = cands[0]
        m = c["comp_mask"]
        t_top = bool(np.any(m[0, :]))
        t_bot = bool(np.any(m[new_h - 1, :]))
        t_left = bool(np.any(m[:, 0]))
        t_right = bool(np.any(m[:, new_w - 1]))
        vignette_overlap = int(np.count_nonzero((m > 0) & (b_mask == 0)))
        t_vignette = bool(vignette_overlap > 0)

        # Analyze dark regions in rgb_sub
        gray_orig = cv2.cvtColor(rgb_sub, cv2.COLOR_RGB2GRAY)
        _, b_thresh = cv2.threshold(gray_orig, 15, 255, cv2.THRESH_BINARY)
        black = (b_thresh == 0).astype(np.uint8)
        n_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(black, 8)
        corner_pixels = [(0, 0), (0, new_w - 1), (new_h - 1, 0), (new_h - 1, new_w - 1)]
        corner_labels = set(lbls[r, c] for (r, c) in corner_pixels if lbls[r, c] > 0)
        vignette_mask = np.isin(lbls, list(corner_labels))
        internal_dark_mask = (black > 0) & (~vignette_mask)

        vig_overlap = int(np.count_nonzero((m > 0) & vignette_mask))
        int_dark_overlap = int(np.count_nonzero((m > 0) & internal_dark_mask))

        # evaluate dice if this component were used
        c_mask_full = cv2.resize(m.astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)
        eval_res = evaluate_segmentation(c_mask_full, gt)

        rows.append({
            "idx": i + 1,
            "fn": fn,
            "has_seed": True,
            "num_cands": len(cands),
            "score": c["score"],
            "s_area": c["s_area"],
            "s_contrast": c["s_contrast"],
            "s_centrality": c["s_centrality"],
            "p_border": c["p_border"],
            "t_canvas_edge": bool(t_top or t_bot or t_left or t_right),
            "vig_px": vig_overlap,
            "int_dark_px": int_dark_overlap,
            "cand_area": c["area"],
            "gt_area": int(np.count_nonzero(gt)),
            "dice": eval_res["dice"],
            "prec": eval_res["precision"],
            "rec": eval_res["recall"],
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
