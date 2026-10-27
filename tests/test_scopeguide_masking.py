"""PyTest Suite for ScopeGuide Occlusion Masking.

Validates:
1. Deterministic behavior (repeated applications on identical inputs yield bitwise identical outputs).
2. Class-agnostic consistency (identical masking applied to both polyp and non-polyp frames).
3. Relative coordinate precision across diverse resolutions and aspect ratios.
4. Non-masked mucosal regions remain unaltered.
5. Integration with PolypClassificationDataset (applied prior to resizing and normalization).
6. Strict preservation of test set quarantine.
"""

import sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.scopeguide_mask import ScopeGuideMask
from src.data.polyp_classification_dataset import (
    PolypClassificationDataset,
    get_default_transforms,
    IMAGENET_MEAN,
    IMAGENET_STD
)


def test_scopeguide_mask_determinism():
    """Confirms that ScopeGuideMask produces bitwise identical results across repeated runs."""
    masker = ScopeGuideMask()
    np.random.seed(42)
    raw_img = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)

    out1 = masker(raw_img)
    out2 = masker(raw_img)

    assert np.array_equal(out1, out2), "Masking output must be bitwise deterministic"

    # Also test with PIL Image
    pil_img = Image.fromarray(raw_img)
    pil_out1 = masker(pil_img)
    pil_out2 = masker(pil_img)

    assert np.array_equal(np.array(pil_out1), np.array(pil_out2)), "PIL masking must be deterministic"


def test_scopeguide_mask_coordinates_relative_scaling():
    """Confirms that mask coordinates scale proportionally to image height and width."""
    masker = ScopeGuideMask(y_min_ratio=0.65, x_max_ratio=0.35)

    test_shapes = [
        (100, 100),
        (288, 384),
        (576, 720),
        (1080, 1920)
    ]

    for h, w in test_shapes:
        y_start, y_end, x_start, x_end = masker.get_mask_coordinates(h, w)
        assert y_start == int(0.65 * h)
        assert y_end == h
        assert x_start == 0
        assert x_end == int(0.35 * w)

        img = np.ones((h, w, 3), dtype=np.uint8) * 255
        masked = masker(img)

        # Bottom-left quadrant must be exactly 0
        assert np.all(masked[y_start:y_end, x_start:x_end] == 0), (
            f"Expected masked region [y:{y_start}:{y_end}, x:{x_start}:{x_end}] to be 0 for shape ({h}, {w})"
        )

        # Region strictly outside must be preserved at 255
        # Top portion (above y_start)
        assert np.all(masked[:y_start, :] == 255), "Top image portion must remain unaltered"
        # Right portion (to the right of x_end)
        assert np.all(masked[:, x_end:] == 255), "Right image portion must remain unaltered"


def test_class_agnostic_consistency():
    """Confirms that the exact same masking rules apply to both polyp (class 1) and non-polyp (class 0)."""
    masker = ScopeGuideMask()
    h, w = 200, 300

    # Synthetic polyp frame (bright reddish vascular mucosa)
    polyp_frame = np.full((h, w, 3), fill_value=[180, 60, 70], dtype=np.uint8)
    # Synthetic non-polyp frame (pinkish smooth mucosa)
    non_polyp_frame = np.full((h, w, 3), fill_value=[150, 100, 110], dtype=np.uint8)

    masked_polyp = masker(polyp_frame)
    masked_non_polyp = masker(non_polyp_frame)

    y_s, y_e, x_s, x_e = masker.get_mask_coordinates(h, w)

    # Both classes must have the exact same region masked to 0
    assert np.all(masked_polyp[y_s:y_e, x_s:x_e] == 0)
    assert np.all(masked_non_polyp[y_s:y_e, x_s:x_e] == 0)

    # Outside regions for both must retain their respective original colors
    assert np.array_equal(masked_polyp[:y_s, :], polyp_frame[:y_s, :])
    assert np.array_equal(masked_polyp[:, x_e:], polyp_frame[:, x_e:])

    assert np.array_equal(masked_non_polyp[:y_s, :], non_polyp_frame[:y_s, :])
    assert np.array_equal(masked_non_polyp[:, x_e:], non_polyp_frame[:, x_e:])


def test_dataset_mask_scopeguide_integration():
    """Confirms that PolypClassificationDataset correctly applies ScopeGuideMask before resizing."""
    val_ds_unmasked = PolypClassificationDataset(split="val", max_samples=4, mask_scopeguide=False)
    val_ds_masked = PolypClassificationDataset(split="val", max_samples=4, mask_scopeguide=True)

    assert len(val_ds_unmasked) == 4
    assert len(val_ds_masked) == 4

    # Both datasets should yield (tensor, label, meta)
    tensor_unmasked, label_u, meta_u = val_ds_unmasked[0]
    tensor_masked, label_m, meta_m = val_ds_masked[0]

    assert meta_u["filename"] == meta_m["filename"]
    assert label_u == label_m
    assert tensor_unmasked.shape == torch.Size([3, 224, 224])
    assert tensor_masked.shape == torch.Size([3, 224, 224])

    # The masked and unmasked tensors must differ in their bottom-left corner
    # In normalized space: normalized_zero = (0 - mean) / std
    expected_norm_val = [(-m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)]

    # Check bottom-left corner of masked tensor in 224x224 space
    # The bottom 35% is rows ~145:224, and left 35% is cols ~0:78
    bl_masked_slice = tensor_masked[:, 160:220, 5:70]
    for c in range(3):
        # All pixel values in this interior masked region should match the normalized zero value closely
        assert torch.allclose(bl_masked_slice[c], torch.tensor(expected_norm_val[c]), atol=1e-3), (
            f"Channel {c} in masked corner must match ImageNet-normalized zero"
        )


def test_test_set_quarantine_enforced():
    """Confirms that attempting to access the quarantined test set raises ValueError."""
    with pytest.raises(ValueError, match="Access to quarantined test split is prohibited"):
        PolypClassificationDataset(split="test", mask_scopeguide=True)
