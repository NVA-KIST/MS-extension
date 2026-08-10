"""Ureter mask + organ clip/clean (numpy/scipy only)."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_fill_holes

from lib.processing.dilate import dilate_mask, resample_to_target


def connect_ureter_path(
    mask_arr: np.ndarray,
    vox_size: Sequence[float],
    max_gap_mm: float = 35.0,
    tube_radius_vox: int = 3,
) -> np.ndarray:
    """Bridge gaps between adjacent disconnected ureter fragments (vertical tube)."""
    labeled, n = ndimage.label(mask_arr)
    if n <= 1:
        return mask_arr.copy()

    z_mm_per_vox = abs(float(vox_size[2]))
    components = []
    for comp_id in range(1, n + 1):
        comp_vox = np.argwhere(labeled == comp_id)
        z_min = int(comp_vox[:, 0].min())
        z_max = int(comp_vox[:, 0].max())
        bot_face = comp_vox[comp_vox[:, 0] == z_min]
        top_face = comp_vox[comp_vox[:, 0] == z_max]
        components.append(
            {
                "z_min": z_min,
                "z_max": z_max,
                "z_mid": (z_min + z_max) / 2.0,
                "bot": bot_face.mean(axis=0).astype(np.float64),
                "top": top_face.mean(axis=0).astype(np.float64),
            }
        )

    components.sort(key=lambda c: c["z_mid"])
    result = mask_arr.copy()

    for i in range(len(components) - 1):
        c_lo = components[i]
        c_hi = components[i + 1]
        z_gap_vox = c_hi["z_min"] - c_lo["z_max"]
        z_gap_mm = float(z_gap_vox) * z_mm_per_vox
        if z_gap_mm > max_gap_mm:
            continue

        cy = (c_lo["top"][1] + c_hi["bot"][1]) / 2.0
        cx = (c_lo["top"][2] + c_hi["bot"][2]) / 2.0
        z_lo = c_lo["z_max"]
        z_hi = c_hi["z_min"]
        r = int(tube_radius_vox)
        y0 = max(0, int(cy) - r - 1)
        y1 = min(mask_arr.shape[1] - 1, int(cy) + r + 1)
        x0 = max(0, int(cx) - r - 1)
        x1 = min(mask_arr.shape[2] - 1, int(cx) + r + 1)
        yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
        disk = ((yy - cy) ** 2 + (xx - cx) ** 2) <= float(r) ** 2
        for z in range(z_lo, z_hi + 1):
            result[z, y0 : y1 + 1, x0 : x1 + 1][disk] = 1

    return result.astype(np.uint8)


def build_ureter_mask_from_pet(
    pet_arr: np.ndarray,
    pet_affine,
    vox_size: Sequence[float],
    z_inferior: float,
    z_superior: float,
    suv_thresh: float,
    dilate_mm: float,
    torso_center_xy: Tuple[float, float],
    *,
    ureter_z_inf: Optional[float] = None,
    ureter_ext_inf_mm: float = 90.0,
    torso_radius_mm: float = 220.0,
    connect_path: bool = True,
    max_gap_mm: float = 35.0,
    fill_holes: bool = True,
) -> np.ndarray:
    """
    Pure array ureter mask (no Slicer nodes).

    ``torso_center_xy`` = (cx, cy) in RAS mm (usually vertebrae XY centroid).
    """
    if z_inferior is None or z_superior is None:
        raise ValueError("z_inferior and z_superior are required")

    ureter_z_sup = float(z_superior)
    if ureter_z_inf is None:
        ureter_z_inf = float(z_inferior) - float(ureter_ext_inf_mm)
    else:
        ureter_z_inf = float(ureter_z_inf)

    pet_affine = np.asarray(pet_affine, dtype=float)
    vox_size = np.asarray(vox_size, dtype=float)
    shape = pet_arr.shape
    z_idx, y_idx, x_idx = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
    )
    ijk_hom = np.stack(
        [x_idx.ravel(), y_idx.ravel(), z_idx.ravel(), np.ones(x_idx.size)],
        axis=1,
    ).astype(np.float32)
    ras_coords = (pet_affine @ ijk_hom.T).T
    ras_z = ras_coords[:, 2].reshape(shape)
    ras_x = ras_coords[:, 0].reshape(shape)
    ras_y = ras_coords[:, 1].reshape(shape)

    z_mask = ((ras_z >= ureter_z_inf) & (ras_z <= ureter_z_sup)).astype(np.uint8)
    cx, cy = torso_center_xy
    dist_xy = np.sqrt((ras_x - cx) ** 2 + (ras_y - cy) ** 2)
    torso_mask = (dist_xy <= torso_radius_mm).astype(np.uint8)
    anat_mask = (z_mask & torso_mask).astype(np.uint8)

    hot = (pet_arr > suv_thresh).astype(np.uint8)
    labeled, n = ndimage.label(hot)
    if n > 0:
        sizes = ndimage.sum(hot, labeled, range(1, n + 1))
        bladder_label = int(np.argmax(sizes)) + 1
        hot_no_bladder = ((labeled != bladder_label) & (labeled > 0)).astype(np.uint8)
    else:
        hot_no_bladder = hot

    hot_clipped = (hot_no_bladder & anat_mask).astype(np.uint8)
    ureter_mask = dilate_mask(hot_clipped, pet_affine, dilate_mm)

    if connect_path:
        tube_r = max(3, int(round(dilate_mm / max(float(vox_size.max()), 1e-6) * 0.5)))
        ureter_mask = connect_ureter_path(
            ureter_mask, vox_size, max_gap_mm=max_gap_mm, tube_radius_vox=tube_r
        )

    if fill_holes:
        for z in range(ureter_mask.shape[0]):
            if ureter_mask[z].any():
                ureter_mask[z] = binary_fill_holes(ureter_mask[z]).astype(np.uint8)

    ureter_mask = (ureter_mask & z_mask).astype(np.uint8)
    return ureter_mask


def clip_organ_to_z(
    organ_arr: np.ndarray,
    organ_affine,
    z_inferior: float,
    z_superior: float,
) -> np.ndarray:
    """Zero voxels whose RAS Z is outside [z_inferior, z_superior]."""
    organ_arr = organ_arr.copy()
    organ_affine = np.asarray(organ_affine, dtype=float)
    shape = organ_arr.shape
    z_idx, y_idx, x_idx = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
    )
    ijk_hom = np.stack(
        [x_idx.ravel(), y_idx.ravel(), z_idx.ravel(), np.ones(x_idx.size)],
        axis=1,
    ).astype(np.float32)
    ras_z = (organ_affine @ ijk_hom.T).T[:, 2].reshape(shape)
    outside = (ras_z < z_inferior) | (ras_z > z_superior)
    organ_arr[outside] = 0
    return organ_arr


def clean_organ_with_ureter(
    organ_arr: np.ndarray,
    organ_affine,
    ureter_arr,
    ureter_affine,
    pet_arr,
    pet_affine,
    suv_clean_thresh: float,
) -> np.ndarray:
    """Remove organ voxels overlapping ureter where PET > threshold."""
    organ_arr = organ_arr.copy()
    ureter_in = resample_to_target(ureter_arr, ureter_affine, organ_arr.shape, organ_affine)
    pet_in = resample_to_target(pet_arr, pet_affine, organ_arr.shape, organ_affine)
    remove = (organ_arr > 0) & (ureter_in > 0) & (pet_in > suv_clean_thresh)
    organ_arr[remove] = 0
    return organ_arr


def apply_exclusion_mask(
    organ_arr: np.ndarray,
    organ_affine,
    excl_arr,
    excl_affine,
    pet_arr,
    pet_affine,
    suv_thresh: float,
) -> np.ndarray:
    organ_arr = organ_arr.copy()
    excl_in = resample_to_target(excl_arr, excl_affine, organ_arr.shape, organ_affine)
    pet_in = resample_to_target(pet_arr, pet_affine, organ_arr.shape, organ_affine)
    remove = (organ_arr > 0) & (excl_in > 0) & (pet_in > suv_thresh)
    organ_arr[remove] = 0
    return organ_arr


def apply_organ_processing(
    organ_arr: np.ndarray,
    organ_affine,
    mode: str,
    ureter_arr=None,
    ureter_affine=None,
    pet_arr=None,
    pet_affine=None,
    suv_clean_thresh: float = 1.2,
    z_inferior: Optional[float] = None,
    z_superior: Optional[float] = None,
) -> np.ndarray:
    """Port of UreterPostProcessLogic._apply_processing."""
    out = organ_arr.copy()
    if mode in ("Clip only", "Clip + Clean"):
        if z_inferior is not None and z_superior is not None:
            out = clip_organ_to_z(out, organ_affine, z_inferior, z_superior)
    if mode in ("Clean only", "Clip + Clean"):
        if ureter_arr is not None and pet_arr is not None:
            out = clean_organ_with_ureter(
                out,
                organ_affine,
                ureter_arr,
                ureter_affine,
                pet_arr,
                pet_affine,
                suv_clean_thresh,
            )
    return out


def z_bounds_from_mask(mask_arr: np.ndarray, affine) -> Tuple[float, float]:
    """RAS Z min/max of non-zero voxels in a binary/label mask."""
    affine = np.asarray(affine, dtype=float)
    idx = np.argwhere(mask_arr > 0)
    if len(idx) == 0:
        raise ValueError("Empty mask — cannot compute Z bounds")
    ijk = np.column_stack(
        [idx[:, 2], idx[:, 1], idx[:, 0], np.ones(len(idx))]
    ).astype(np.float64)
    ras = (affine @ ijk.T).T
    return float(ras[:, 2].min()), float(ras[:, 2].max())
