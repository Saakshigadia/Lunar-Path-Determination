"""Per-cell terrain features used by the hazard model."""
import numpy as np
from scipy import ndimage

from .terrain import slope_degrees

FEATURE_NAMES = ["slope", "roughness", "curvature", "local_relief", "tpi", "slope_max_3x3"]


def compute_features(dem: np.ndarray, cell_size: float) -> np.ndarray:
    """Return an array of shape (H, W, n_features)."""
    slope = slope_degrees(dem, cell_size)
    resid = dem - ndimage.uniform_filter(dem, 5)
    # std in a 5x5 window = sqrt(E[x^2] - E[x]^2): small-scale bumpiness (boulders)
    roughness = np.sqrt(np.maximum(ndimage.uniform_filter(resid ** 2, 5) - ndimage.uniform_filter(resid, 5) ** 2, 0))
    curvature = ndimage.laplace(dem) / cell_size ** 2                        # bowls vs peaks
    local_relief = ndimage.maximum_filter(dem, 7) - ndimage.minimum_filter(dem, 7)
    tpi = dem - ndimage.uniform_filter(dem, 9)                               # topographic position index
    slope_max = ndimage.maximum_filter(slope, 3)
    return np.stack([slope, roughness, curvature, local_relief, tpi, slope_max], axis=-1)
