"""
PET blood-pool vessel growing (array-level, no Slicer).

Port of VesselSegmenterLogic core helpers.
Arrays are (Z, Y, X) like slicer.util.arrayFromVolume.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage


def apply_distance_constraint(
    comp_mask: np.ndarray,
    seed_zyx: Tuple[int, int, int],
    vox_mm: Sequence[float],
    max_extent_mm: float,
) -> np.ndarray:
    """Keep only the seed-connected piece within max_extent_mm."""
    sz, sy, sx = seed_zyx
    nz, ny, nx = comp_mask.shape
    dz = (np.arange(nz) - sz) * float(vox_mm[0])
    dy = (np.arange(ny) - sy) * float(vox_mm[1])
    dx = (np.arange(nx) - sx) * float(vox_mm[2])
    DZ, DY, DX = np.meshgrid(dz, dy, dx, indexing="ij")
    dist = np.sqrt(DZ**2 + DY**2 + DX**2)
    clipped = (comp_mask & (dist <= max_extent_mm).astype(np.uint8)).astype(np.uint8)
    labeled_local, _ = ndimage.label(clipped)
    local_lbl = int(labeled_local[sz, sy, sx])
    if local_lbl == 0:
        return clipped
    return (labeled_local == local_lbl).astype(np.uint8)


def _draw_bridge(mask: np.ndarray, p1, p2, radius_vox: int) -> np.ndarray:
    result = mask.copy()
    nz, ny, nx = result.shape
    vec = np.asarray(p2, dtype=float) - np.asarray(p1, dtype=float)
    dist = float(np.linalg.norm(vec))
    if dist < 1e-3:
        return result
    n_steps = max(3, int(np.ceil(dist)) * 2)
    r2 = radius_vox**2
    for t in np.linspace(0.0, 1.0, n_steps):
        pt = np.asarray(p1, dtype=float) + t * vec
        cz, cy, cx = [int(round(float(v))) for v in pt]
        z0, z1 = max(0, cz - radius_vox), min(nz, cz + radius_vox + 1)
        y0, y1 = max(0, cy - radius_vox), min(ny, cy + radius_vox + 1)
        x0, x1 = max(0, cx - radius_vox), min(nx, cx + radius_vox + 1)
        zz = np.arange(z0, z1) - cz
        yy = np.arange(y0, y1) - cy
        xx = np.arange(x0, x1) - cx
        ZZ, YY, XX = np.meshgrid(zz, yy, xx, indexing="ij")
        sphere = (ZZ**2 + YY**2 + XX**2) <= r2
        result[z0:z1, y0:y1, x0:x1] |= sphere
    return result


def stitch_fragments(
    mask_arr: np.ndarray,
    vox_mm: Sequence[float],
    max_gap_mm: float,
    bridge_radius_mm: float,
) -> np.ndarray:
    """Bridge nearest disconnected fragments until gaps exceed max_gap_mm."""
    result = mask_arr.copy()
    bridge_r_vox = max(1, int(round(bridge_radius_mm / max(min(vox_mm), 1e-6))))
    for _ in range(200):
        labeled, n = ndimage.label(result)
        if n <= 1:
            break
        centroids = [
            np.asarray(c, dtype=float)
            for c in ndimage.center_of_mass(result, labeled, range(1, n + 1))
        ]
        best_dist = float("inf")
        best_i, best_j = 0, 1
        for i in range(n):
            for j in range(i + 1, n):
                d = float(
                    np.sqrt(
                        ((centroids[i][0] - centroids[j][0]) * vox_mm[0]) ** 2
                        + ((centroids[i][1] - centroids[j][1]) * vox_mm[1]) ** 2
                        + ((centroids[i][2] - centroids[j][2]) * vox_mm[2]) ** 2
                    )
                )
                if d < best_dist:
                    best_dist = d
                    best_i, best_j = i, j
        if best_dist > max_gap_mm:
            break
        result = _draw_bridge(result, centroids[best_i], centroids[best_j], bridge_r_vox)
    return result.astype(np.uint8)


def ct_vesselness_mask(
    ct_arr: np.ndarray,
    spacing_mm: Sequence[float],
    sigma_min_mm: float = 2.0,
    sigma_max_mm: float = 8.0,
    threshold: float = 0.10,
) -> np.ndarray:
    """
    Multi-scale Frangi-style vesselness via SimpleITK (ZYX array in/out).
    """
    import SimpleITK as sitk

    img = sitk.GetImageFromArray(ct_arr.astype(np.float32))
    img.SetSpacing([float(spacing_mm[2]), float(spacing_mm[1]), float(spacing_mm[0])])
    # ObjectnessMeasureImageFilter: bright tubular structures
    filt = sitk.ObjectnessMeasureImageFilter()
    filt.SetBrightObject(True)
    filt.SetScaleObjectnessMeasure(True)
    filt.SetAlpha(0.5)
    filt.SetBeta(0.5)
    filt.SetGamma(5.0)
    # Multi-scale: run a few sigmas and take max
    sigmas = np.linspace(sigma_min_mm, sigma_max_mm, num=4)
    acc = None
    for s in sigmas:
        filt.SetObjectDimension(1)  # lines / vessels
        # Gaussian sigma via SmoothRecursiveGaussian then objectness on smoothed?
        # Simpler: SetScaleObjectnessMeasure uses internal scales — use Hessian
        try:
            gauss = sitk.SmoothingRecursiveGaussian(img, float(s))
            resp = filt.Execute(gauss)
        except Exception:
            continue
        arr = sitk.GetArrayFromImage(resp)
        acc = arr if acc is None else np.maximum(acc, arr)
    if acc is None:
        return np.zeros_like(ct_arr, dtype=np.uint8)
    mx = float(acc.max()) if acc.size else 0.0
    if mx <= 0:
        return np.zeros_like(ct_arr, dtype=np.uint8)
    return (acc / mx >= threshold).astype(np.uint8)


def grow_vessels_from_seeds(
    pet_arr: np.ndarray,
    spacing_mm: Sequence[float],
    seeds_zyx: Sequence[Tuple[int, int, int]],
    suv_min: float = 0.8,
    suv_max: float = 4.0,
    max_extent_mm: float = 150.0,
    closing_radius_mm: float = 2.0,
    min_volume_ml: float = 5.0,
    stitch: bool = True,
    stitch_gap_mm: float = 25.0,
    bridge_radius_mm: float = 5.0,
    ct_vesselness: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Grow one vessel mask per seed from PET blood-pool.

    Returns ``{ "Vessel_1": mask_zyx, ... }``.
    """
    pet_arr = np.asarray(pet_arr)
    vox_mm = [float(v) for v in spacing_mm]
    pet_mask = ((pet_arr >= suv_min) & (pet_arr <= suv_max)).astype(np.uint8)
    combined = pet_mask.copy()
    if ct_vesselness is not None:
        vm = ct_vesselness
        if vm.shape != combined.shape:
            raise ValueError(f"vesselness shape {vm.shape} != PET {combined.shape}")
        combined = (combined & (vm > 0)).astype(np.uint8)

    if closing_radius_mm > 0:
        r = max(1, int(round(closing_radius_mm / max(min(vox_mm), 1e-6))))
        struct = ndimage.generate_binary_structure(3, 1)
        struct = ndimage.iterate_structure(struct, r)
        combined = ndimage.binary_closing(combined, structure=struct).astype(np.uint8)

    labeled, _n = ndimage.label(combined)
    vox_vol_ml = (vox_mm[0] * vox_mm[1] * vox_mm[2]) / 1000.0

    out: Dict[str, np.ndarray] = {}
    kept: List[np.ndarray] = []

    for seed_idx, (z, y, x) in enumerate(seeds_zyx):
        name = f"Vessel_{seed_idx + 1}"
        z = int(np.clip(z, 0, labeled.shape[0] - 1))
        y = int(np.clip(y, 0, labeled.shape[1] - 1))
        x = int(np.clip(x, 0, labeled.shape[2] - 1))
        comp_lbl = int(labeled[z, y, x])
        if comp_lbl == 0:
            continue
        comp_mask = (labeled == comp_lbl).astype(np.uint8)
        if max_extent_mm > 0:
            comp_mask = apply_distance_constraint(comp_mask, (z, y, x), vox_mm, max_extent_mm)
        vol_ml = float(comp_mask.sum()) * vox_vol_ml
        if vol_ml < min_volume_ml:
            continue
        if stitch:
            comp_mask = stitch_fragments(comp_mask, vox_mm, stitch_gap_mm, bridge_radius_mm)
        duplicate = False
        for prev in kept:
            overlap = float((comp_mask & prev).sum()) / max(float(comp_mask.sum()), 1.0)
            if overlap > 0.9:
                duplicate = True
                break
        if duplicate:
            continue
        kept.append(comp_mask)
        out[name] = comp_mask

    return out
