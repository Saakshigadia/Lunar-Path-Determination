"""Realistic-looking renders of lunar terrain and rover paths."""
import numpy as np
from matplotlib.colors import LightSource
from scipy import ndimage


def lunar_surface(dem, cell_size, scale=4, seed=0):
    """Return an RGB image (0-1) that looks like an orbital photo of the Moon.

    The elevation map is upsampled for smooth shading, lit by a low sun (like
    real lunar images, which makes craters pop), and given fine regolith grain.
    """
    rng = np.random.default_rng(seed)
    hi = ndimage.zoom(dem, scale, order=3)
    grain = ndimage.gaussian_filter(rng.normal(size=hi.shape), 0.8) * 0.06   # small pebbles / dust texture
    hi = hi + grain
    ls = LightSource(azdeg=285, altdeg=16)
    shade = ls.hillshade(hi, vert_exag=3.0, dx=cell_size / scale, dy=cell_size / scale)
    # albedo: brighter fresh crater rims, slightly darker flat regolith, large-scale mottling
    mottling = ndimage.gaussian_filter(rng.normal(size=hi.shape), 18)
    mottling = (mottling - mottling.min()) / (np.ptp(mottling) + 1e-9)
    albedo = 0.55 + 0.25 * mottling
    img = np.clip(shade * albedo * 1.45, 0, 1) ** 1.15
    rgb = np.stack([img * 1.0, img * 0.985, img * 0.96], axis=-1)            # very slight warm grey, like LRO photos
    return np.clip(rgb, 0, 1)


def smooth_path(path, window=5):
    """Moving-average smoothing so the drawn route looks like a real drive, not grid steps."""
    p = np.asarray(path, dtype=float)
    if len(p) < window:
        return p
    k = np.ones(window) / window
    pad = np.pad(p, ((window // 2, window // 2), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, i], k, mode="valid") for i in range(2)], axis=1)
