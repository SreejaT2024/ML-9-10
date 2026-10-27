"""ScopeGuide Occlusion Masking Module.

Provides deterministic masking of the bottom-left Olympus ScopeGuide telemetry
region based on relative image coordinates identified during the diagnostic audit.
"""

from typing import Tuple, Union
import numpy as np
from PIL import Image


class ScopeGuideMask:
    """
    Applies deterministic neutral black masking to the bottom-left ScopeGuide region.

    Empirical ScopeGuide Footprint (from diagnostic audit):
        Relative Y (rows): [0.65 * height, height] (bottom 35% of the frame)
        Relative X (cols): [0.0 * width, 0.35 * width] (leftmost 35% of the frame)
        Fill Value: 0 (neutral black for all color channels)

    Design Safeguards:
        - Deterministic: Always applies identical masking for identical inputs.
        - Class-Agnostic: Applied identically across all classes (polyp and non-polyp).
        - Split-Consistent: Applied to both train and validation splits before resizing.
        - Safe: Does not mutate the input image in-place.
    """

    def __init__(
        self,
        y_min_ratio: float = 0.65,
        x_max_ratio: float = 0.35,
        fill_val: int = 0
    ) -> None:
        self.y_min_ratio = y_min_ratio
        self.x_max_ratio = x_max_ratio
        self.fill_val = fill_val

    def get_mask_coordinates(self, height: int, width: int) -> Tuple[int, int, int, int]:
        """
        Calculates absolute integer pixel coordinates for the mask given image dimensions.
        Returns (y_start, y_end, x_start, x_end).
        """
        y_start = int(self.y_min_ratio * height)
        y_end = height
        x_start = 0
        x_end = int(self.x_max_ratio * width)
        return y_start, y_end, x_start, x_end

    def __call__(self, img: Union[Image.Image, np.ndarray]) -> Union[Image.Image, np.ndarray]:
        """
        Applies ScopeGuide masking to a PIL Image or NumPy array.
        Returns a new instance with the bottom-left quadrant zeroed out.
        """
        is_pil = isinstance(img, Image.Image)
        if is_pil:
            arr = np.array(img, copy=True)
        else:
            arr = img.copy()

        h, w = arr.shape[:2]
        y_start, y_end, x_start, x_end = self.get_mask_coordinates(h, w)

        arr[y_start:y_end, x_start:x_end] = self.fill_val

        if is_pil:
            return Image.fromarray(arr)
        return arr
