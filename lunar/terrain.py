"""Synthetic lunar terrain generation.

Real lunar elevation data (e.g. NASA LRO LOLA DEMs) can be loaded with `load_dem`,
but for a fully offline, reproducible demo we generate terrain that has the main
features a rover has to deal with: rolling regolith, craters with raised rims,
and scattered boulders.
"""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass
class Terrain:
    dem: np.ndarray          # elevation in metres, shape (H, W)
    hazard: np.ndarray       # ground-truth hazard mask (bool), shape (H, W)
    cell_size: float         # metres per grid cell


def fractal_noise(size: int, rng: np.random.Generator, octaves: int = 5) -> np.ndarray:
    """Smooth multi-scale noise: coarse hills plus finer bumps."""
    out = np.zeros((size, size))
    amp = 1.0
    for o in range(octaves):
        n = 2 ** (o + 2)
        grid = rng.normal(size=(n, n))
        out += amp * ndimage.zoom(grid, size / n, order=3)[:size, :size]
        amp *= 0.5
    return out


def add_craters(dem, rng, n, cell_size, hazard):
    size = dem.shape[0]
    yy, xx = np.mgrid[0:size, 0:size]
    for _ in range(n):
        cy, cx = rng.integers(0, size, 2)
        r = rng.uniform(4, size / 7)                     # radius in cells
        depth = r * cell_size * rng.uniform(0.15, 0.25)  # real simple craters are ~0.1-0.2 deep per diameter
        d = np.hypot(yy - cy, xx - cx) / r
        bowl = np.where(d < 1, -depth * (1 - d ** 2), 0.0)
        rim = 0.25 * depth * np.exp(-((d - 1) ** 2) / 0.02)
        dem += bowl + rim
        hazard |= (d < 0.7) & (r > 6)                    # steep inner walls of larger craters
    return dem


def add_boulders(dem, rng, n, hazard):
    size = dem.shape[0]
    for _ in range(n):
        cy, cx = rng.integers(2, size - 2, 2)
        h = rng.uniform(0.8, 2.0)                        # boulder height in metres
        dem[cy - 1:cy + 2, cx - 1:cx + 2] += h * np.array([[.4, .7, .4], [.7, 1, .7], [.4, .7, .4]])
        hazard[cy - 1:cy + 2, cx - 1:cx + 2] = True
    return dem


def generate_terrain(size: int = 128, cell_size: float = 5.0, seed: int = 0,
                     n_craters: int = 10, n_boulders: int = 60) -> Terrain:
    rng = np.random.default_rng(seed)
    dem = fractal_noise(size, rng) * 3.5                 # gentle hills, a few metres high
    hazard = np.zeros((size, size), dtype=bool)
    dem = add_craters(dem, rng, n_craters, cell_size, hazard)
    dem = add_boulders(dem, rng, n_boulders, hazard)
    hazard |= slope_degrees(dem, cell_size) > 20         # anything steeper than 20 degrees is unsafe
    return Terrain(dem=dem, hazard=hazard, cell_size=cell_size)


def slope_degrees(dem: np.ndarray, cell_size: float) -> np.ndarray:
    gy, gx = np.gradient(dem, cell_size)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def load_dem(path: str) -> np.ndarray:
    """Load a real DEM saved as .npy (convert GeoTIFFs with rasterio first)."""
    return np.load(path).astype(float)
