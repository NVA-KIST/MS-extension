"""Distance math (no Markups / slicer UI)."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def euclidean(p1: Sequence[float], p2: Sequence[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(p1, p2)))


def ras_distance_to_voxels(
    p1_ras: Sequence[float],
    p2_ras: Sequence[float],
    ras_to_ijk_4x4,
) -> float:
    """
    Convert two RAS points via RAS→IJK and return Euclidean voxel distance.

    ``ras_to_ijk_4x4`` is a 4×4 matrix (numpy or nested list).
    Column vector is [R, A, S, 1]^T.
    """
    mat = np.asarray(ras_to_ijk_4x4, dtype=float)
    if mat.shape != (4, 4):
        raise ValueError(f"ras_to_ijk must be 4x4, got {mat.shape}")

    def to_ijk(p):
        h = mat @ np.array([float(p[0]), float(p[1]), float(p[2]), 1.0])
        return h[0], h[1], h[2]

    i1, j1, k1 = to_ijk(p1_ras)
    i2, j2, k2 = to_ijk(p2_ras)
    return euclidean((i1, j1, k1), (i2, j2, k2))


def format_distance(
    length_mm: float,
    unit: str,
    voxel_length: float | None = None,
) -> str:
    if unit == "mm":
        return f"{length_mm:.2f} mm"
    if unit == "cm":
        return f"{length_mm / 10.0:.3f} cm"
    if unit == "voxels":
        if voxel_length is None:
            return "- vox (no ref)"
        return f"{voxel_length:.2f} vox"
    raise ValueError(f"Unknown unit: {unit}")
