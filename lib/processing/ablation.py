"""
Array-level ROI ablation / urinary-mask helpers from PETBiomarkerStudioLogic.

Pure numpy + scipy.ndimage (no Slicer / Qt).
Masks use Slicer ``arrayFromVolume`` order: (K, J, I).
Spacing is the (I, J, K) mm tuple from ``GetSpacing``.
Affine maps homogeneous IJK (I, J, K, 1) to RAS.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

ArrayLike = np.ndarray
InfoDict = Dict[str, Any]


def physical_dilation_structure(
    spacing: Sequence[float],
    radius_mm: float,
) -> np.ndarray:
    """Ball structuring element in physical mm (I, J, K spacing)."""
    if radius_mm <= 0:
        return np.ones((1, 1, 1), dtype=bool)

    sx, sy, sz = spacing  # I, J, K spacing in mm

    rx = int(math.ceil(radius_mm / sx))
    ry = int(math.ceil(radius_mm / sy))
    rz = int(math.ceil(radius_mm / sz))

    z, y, x = np.meshgrid(
        np.arange(-rz, rz + 1),
        np.arange(-ry, ry + 1),
        np.arange(-rx, rx + 1),
        indexing="ij",
    )

    distance_mm = np.sqrt(
        (x * sx) ** 2 +
        (y * sy) ** 2 +
        (z * sz) ** 2
    )

    return distance_mm <= radius_mm


def clip_binary_mask_by_ras_z(
    mask: ArrayLike,
    affine_4x4: ArrayLike,
    z_inferior: float,
    z_superior: float,
) -> Tuple[np.ndarray, InfoDict]:
    """Keep foreground voxels whose RAS Z is in [z_inferior, z_superior]."""
    mask_bool = np.asarray(mask) > 0
    voxels = np.argwhere(mask_bool)
    original_voxel_count = int(len(voxels))

    clipped = mask_bool.copy()
    if original_voxel_count == 0:
        return clipped, {
            "originalVoxelCount": 0,
            "remainingVoxelCount": 0,
            "removedVoxelCount": 0,
        }

    affine = np.asarray(affine_4x4, dtype=float)
    ijk_hom = np.column_stack([
        voxels[:, 2].astype(float),  # I
        voxels[:, 1].astype(float),  # J
        voxels[:, 0].astype(float),  # K
        np.ones(len(voxels)),
    ])
    ras = (affine @ ijk_hom.T).T
    ras_z = ras[:, 2]
    keep = (ras_z >= float(z_inferior)) & (ras_z <= float(z_superior))
    remove_voxels = voxels[~keep]

    if len(remove_voxels) > 0:
        clipped[
            remove_voxels[:, 0],
            remove_voxels[:, 1],
            remove_voxels[:, 2],
        ] = False

    remaining_voxel_count = int(np.count_nonzero(clipped))
    return clipped, {
        "originalVoxelCount": original_voxel_count,
        "remainingVoxelCount": remaining_voxel_count,
        "removedVoxelCount": original_voxel_count - remaining_voxel_count,
    }


def exclude_dilated_structure(
    roi_mask: ArrayLike,
    structure_mask: ArrayLike,
    spacing: Sequence[float],
    dilation_radius_mm: float,
) -> Tuple[np.ndarray, InfoDict]:
    """Dilate structure with a physical ball and remove overlap from ROI."""
    roi = np.asarray(roi_mask)
    structure = np.asarray(structure_mask) > 0
    roi_bool = roi > 0

    original_voxel_count = int(np.count_nonzero(roi_bool))
    if original_voxel_count == 0:
        raise ValueError("ROI segment contains no foreground voxels")

    if np.count_nonzero(structure) == 0:
        raise ValueError("Structure segmentation contains no foreground voxels")

    struct_elem = physical_dilation_structure(
        spacing=spacing,
        radius_mm=float(dilation_radius_mm),
    )
    dilated = ndimage.binary_dilation(structure, structure=struct_elem)
    remove_mask = roi_bool & dilated

    cleaned = roi.copy()
    cleaned[remove_mask] = 0 if np.issubdtype(cleaned.dtype, np.number) else False

    remaining_voxel_count = int(np.count_nonzero(cleaned > 0))
    removed_voxel_count = original_voxel_count - remaining_voxel_count

    if remaining_voxel_count == 0:
        raise ValueError(
            "Kidney exclusion removed the entire ROI. "
            "Try a smaller dilation radius or check the selected masks."
        )

    info: InfoDict = {
        "dilationRadiusMm": float(dilation_radius_mm),
        "originalVoxelCount": original_voxel_count,
        "remainingVoxelCount": remaining_voxel_count,
        "removedVoxelCount": removed_voxel_count,
    }
    return cleaned, info


def build_pet_urinary_mask(
    pet_arr: ArrayLike,
    affine_4x4: ArrayLike,
    spacing: Sequence[float],
    z_inferior: float,
    z_superior: float,
    suv_threshold: float,
    dilation_radius_mm: float,
) -> Tuple[np.ndarray, InfoDict]:
    """
    BiomarkerStudio PET-derived urinary activity mask:

    hot > thresh → Z clip → remove largest CC (bladder) → physical dilate.
    """
    pet = np.asarray(pet_arr)
    hot_mask = pet > float(suv_threshold)
    initial_hot_voxel_count = int(np.count_nonzero(hot_mask))

    if initial_hot_voxel_count == 0:
        raise ValueError(
            f"No PET voxel found above SUV threshold: {suv_threshold}"
        )

    z_clipped_hot_mask, clip_info = clip_binary_mask_by_ras_z(
        hot_mask,
        affine_4x4,
        z_inferior=float(z_inferior),
        z_superior=float(z_superior),
    )
    z_clipped_hot_voxel_count = int(clip_info["remainingVoxelCount"])

    if z_clipped_hot_voxel_count == 0:
        raise ValueError(
            "No PET hot voxel remains after spine-range Z clipping. "
            "Try a lower SUV threshold or check the selected spine range."
        )

    labeled, n_components = ndimage.label(z_clipped_hot_mask)

    if n_components > 0:
        sizes = ndimage.sum(
            z_clipped_hot_mask,
            labeled,
            range(1, n_components + 1),
        )
        bladder_label = int(np.argmax(sizes)) + 1
        hot_without_bladder = z_clipped_hot_mask & (labeled != bladder_label)
    else:
        hot_without_bladder = z_clipped_hot_mask

    after_bladder_removal_voxel_count = int(np.count_nonzero(hot_without_bladder))

    if after_bladder_removal_voxel_count == 0:
        raise ValueError(
            "All hot voxels were removed as the largest component. "
            "The selected region may contain only bladder-like activity."
        )

    struct_elem = physical_dilation_structure(
        spacing=spacing,
        radius_mm=float(dilation_radius_mm),
    )
    final_mask = ndimage.binary_dilation(
        hot_without_bladder,
        structure=struct_elem,
    )
    final_mask_voxel_count = int(np.count_nonzero(final_mask))

    info: InfoDict = {
        "zInferiorUsed": float(z_inferior),
        "zSuperiorUsed": float(z_superior),
        "suvThreshold": float(suv_threshold),
        "dilationRadiusMm": float(dilation_radius_mm),
        "initialHotVoxelCount": initial_hot_voxel_count,
        "zClippedHotVoxelCount": z_clipped_hot_voxel_count,
        "afterBladderRemovalVoxelCount": after_bladder_removal_voxel_count,
        "finalMaskVoxelCount": final_mask_voxel_count,
    }
    return final_mask.astype(np.uint8), info


def apply_urinary_cleanup(
    roi_mask: ArrayLike,
    urinary_mask: ArrayLike,
    pet_arr: ArrayLike,
    clean_suv_threshold: float,
) -> Tuple[np.ndarray, InfoDict]:
    """Remove ROI voxels that overlap urinary mask and exceed SUV threshold."""
    roi = np.asarray(roi_mask)
    urinary = np.asarray(urinary_mask)
    pet = np.asarray(pet_arr)

    if roi.shape != urinary.shape or roi.shape != pet.shape:
        raise RuntimeError(
            "Array shape mismatch.\n"
            f"ROI: {roi.shape}, urinary: {urinary.shape}, PET: {pet.shape}"
        )

    roi_bool = roi > 0
    urinary_bool = urinary > 0
    pet_hot = pet > float(clean_suv_threshold)

    original_voxel_count = int(np.count_nonzero(roi_bool))
    if original_voxel_count == 0:
        raise ValueError("ROI segment contains no foreground voxels")

    overlap_mask = roi_bool & urinary_bool
    remove_mask = overlap_mask & pet_hot

    cleaned = roi.copy()
    cleaned[remove_mask] = 0 if np.issubdtype(cleaned.dtype, np.number) else False

    remaining_voxel_count = int(np.count_nonzero(cleaned > 0))
    if remaining_voxel_count == 0:
        raise ValueError(
            "Urinary activity cleanup removed the entire ROI. "
            "Try a higher cleanup SUV threshold or check the generated urinary mask."
        )

    info: InfoDict = {
        "cleanSUVThreshold": float(clean_suv_threshold),
        "originalVoxelCount": original_voxel_count,
        "overlapVoxelCount": int(np.count_nonzero(overlap_mask)),
        "removedVoxelCount": int(np.count_nonzero(remove_mask)),
        "remainingVoxelCount": remaining_voxel_count,
    }
    return cleaned, info


def qc_suv_stats(pet_arr: ArrayLike, mask: ArrayLike) -> dict:
    """SUVmean / max / peak (3x3x3 neighbourhood) and max/mean ratio for QC."""
    pet = np.asarray(pet_arr)
    m = np.asarray(mask) > 0

    if not np.any(m):
        return {
            "suv_mean": None,
            "suv_max": None,
            "suv_peak": None,
            "ratio": None,
            "max_ijk": None,
        }

    vals = pet[m]
    suv_mean = float(np.mean(vals))
    suv_max = float(np.max(vals))

    masked = np.where(m, pet, -np.inf)
    max_idx = np.unravel_index(int(np.argmax(masked)), masked.shape)
    kz, ky, kx = max_idx

    z0, z1 = max(0, kz - 1), min(pet.shape[0], kz + 2)
    y0, y1 = max(0, ky - 1), min(pet.shape[1], ky + 2)
    x0, x1 = max(0, kx - 1), min(pet.shape[2], kx + 2)
    suv_peak = float(np.mean(pet[z0:z1, y0:y1, x0:x1]))

    return {
        "suv_mean": suv_mean,
        "suv_max": suv_max,
        "suv_peak": suv_peak,
        "ratio": (suv_max / suv_mean) if suv_mean else None,
        "max_ijk": (int(kz), int(ky), int(kx)),
    }


def flag_qc_outliers(
    qc_rows: list,
    mad_k: float = 3.5,
    ratio_thresh: float = 4.0,
) -> list:
    """Flag rows when SUVmax is a MAD outlier or max/mean exceeds ratio_thresh."""
    suv_max_values = [r["suv_max"] for r in qc_rows if r["suv_max"] is not None]

    upper_bound: Optional[float] = None
    if len(suv_max_values) >= 3:
        arr = np.array(suv_max_values, dtype=float)
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median)))
        sigma = 1.4826 * mad
        if sigma > 0:
            upper_bound = median + mad_k * sigma

    for row in qc_rows:
        reasons = []
        if row["suv_max"] is None:
            row["flag"] = ""
            continue
        if upper_bound is not None and row["suv_max"] > upper_bound:
            reasons.append("SUVmax")
        if row["ratio"] is not None and row["ratio"] > ratio_thresh:
            reasons.append("ratio")
        row["flag"] = ("OUTLIER:" + "+".join(reasons)) if reasons else ""

    return qc_rows
