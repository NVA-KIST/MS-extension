"""
PET hotspot finding — pure array math (no slicer / qt).

Used by: slicer_modules/PETHotspotNavigator/PETHotspotNavigatorLogic.py
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def find_hottest_voxels(
    pet_arr: np.ndarray,
    label_arr: np.ndarray,
    ijk_to_ras: np.ndarray | Sequence[Sequence[float]],
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """
    Return up to ``top_n`` hottest voxels inside the label mask.

    Parameters
    ----------
    pet_arr :
        PET volume as (Z, Y, X) = (K, J, I), same as ``slicer.util.arrayFromVolume``.
    label_arr :
        Binary / label mask, same shape as ``pet_arr``. Non-zero = inside segment.
    ijk_to_ras :
        4×4 IJK→RAS matrix (numpy or nested list). Column vector is [I, J, K, 1].
    top_n :
        How many hottest voxels to return (sorted by SUV descending).

    Returns
    -------
    list of dicts with keys: suv, ras_x, ras_y, ras_z
    """
    if pet_arr.shape != label_arr.shape:
        raise ValueError(
            f"Shape mismatch: pet {pet_arr.shape} vs label {label_arr.shape}"
        )

    mask_idx = np.argwhere(label_arr > 0)  # (N, 3): [z, y, x]
    if len(mask_idx) == 0:
        return []

    pet_vals = pet_arr[mask_idx[:, 0], mask_idx[:, 1], mask_idx[:, 2]]
    order = np.argsort(pet_vals)[::-1]
    top_idx = order[: max(1, int(top_n))]

    mat = np.asarray(ijk_to_ras, dtype=float)
    if mat.shape != (4, 4):
        raise ValueError(f"ijk_to_ras must be 4x4, got {mat.shape}")

    hotspots: list[dict[str, Any]] = []
    for idx in top_idx:
        z, y, x = mask_idx[idx]
        # IJK column vector → RAS
        ijk_h = np.array([float(x), float(y), float(z), 1.0])
        ras_h = mat @ ijk_h
        hotspots.append(
            {
                "suv": float(pet_vals[idx]),
                "ras_x": float(ras_h[0]),
                "ras_y": float(ras_h[1]),
                "ras_z": float(ras_h[2]),
            }
        )
    return hotspots
