"""Morphological dilate + resample + subtract (numpy/scipy only)."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import ndimage
from scipy.ndimage import map_coordinates


def dilate_mask(arr, affine, dilate_mm: float) -> np.ndarray:
    """Ellipsoidal morphological dilation scaled to physical mm per axis."""
    affine = np.asarray(affine, dtype=float)
    vox = np.array([abs(affine[0, 0]), abs(affine[1, 1]), abs(affine[2, 2])])
    rx = max(1, int(round(dilate_mm / max(vox[0], 1e-6))))
    ry = max(1, int(round(dilate_mm / max(vox[1], 1e-6))))
    rz = max(1, int(round(dilate_mm / max(vox[2], 1e-6))))
    zz, yy, xx = np.ogrid[-rz : rz + 1, -ry : ry + 1, -rx : rx + 1]
    struct = (
        (zz / max(rz, 1)) ** 2
        + (yy / max(ry, 1)) ** 2
        + (xx / max(rx, 1)) ** 2
    ) <= 1.0
    return ndimage.binary_dilation(arr > 0, structure=struct).astype(np.uint8)


def resample_to_target(src_arr, src_affine, tgt_shape, tgt_affine) -> np.ndarray:
    """Nearest-neighbour resample src into tgt voxel space."""
    src_affine = np.asarray(src_affine, dtype=float)
    tgt_affine = np.asarray(tgt_affine, dtype=float)
    z_idx, y_idx, x_idx = np.meshgrid(
        np.arange(tgt_shape[0]),
        np.arange(tgt_shape[1]),
        np.arange(tgt_shape[2]),
        indexing="ij",
    )
    ijk_hom = np.stack(
        [x_idx.ravel(), y_idx.ravel(), z_idx.ravel(), np.ones(x_idx.size)],
        axis=1,
    ).astype(np.float32)
    ras_pts = (tgt_affine @ ijk_hom.T).T
    src_ijk = (np.linalg.inv(src_affine) @ ras_pts.T).T[:, :3]
    coords = np.array([src_ijk[:, 2], src_ijk[:, 1], src_ijk[:, 0]])
    resampled = map_coordinates(
        src_arr.astype(np.float32), coords, order=0, mode="constant", cval=0.0
    )
    return resampled.reshape(tgt_shape)


def subtract_dilated_union(
    tgt_arr: np.ndarray,
    tgt_affine,
    dilated_items: Sequence[tuple],
) -> np.ndarray:
    """
    Subtract union of dilated sources (each (arr, affine)) from target mask.
    Returns a copy of tgt with overlapping voxels zeroed.
    """
    tgt_arr = np.asarray(tgt_arr)
    union = np.zeros(tgt_arr.shape, dtype=np.uint8)
    for src_arr, src_affine in dilated_items:
        resampled = resample_to_target(src_arr, src_affine, tgt_arr.shape, tgt_affine)
        union = np.maximum(union, (resampled > 0).astype(np.uint8))
    result = tgt_arr.copy()
    result[(result > 0) & (union > 0)] = 0
    return result


def file_stem(filename: str) -> str:
    for ext in (".nii.gz", ".nii"):
        if filename.endswith(ext):
            return filename[: -len(ext)]
    return filename


def bulk_outputs_exist(seg_dir: str, src_file_configs, tgt_file_configs) -> bool:
    import os

    for cfg in src_file_configs:
        stem = file_stem(cfg["filename"])
        if not os.path.exists(os.path.join(seg_dir, f"{stem}_dilated.nii.gz")):
            return False
    for cfg in tgt_file_configs:
        stem = file_stem(cfg["filename"])
        if not os.path.exists(os.path.join(seg_dir, f"{stem}_subtracted.nii.gz")):
            return False
    return True
