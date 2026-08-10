"""
Orientation helpers for VF prediction vs reference masks.

Key bug this fixes
------------------
MONAI Orientationd runs inference in RAS, but older code saved the RAS
voxel array with the *original* CT affine → L/R or A/P mismatch vs
torso_fat / CT in Slicer.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


def dice_binary(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool).ravel()
    b = b.astype(bool).ravel()
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    if denom == 0:
        return 0.0
    return float(2.0 * inter / denom)


def mutual_information_binary(a: np.ndarray, b: np.ndarray, bins: int = 2) -> float:
    """Simple joint-histogram MI for binary (or low-cardinality) masks."""
    a = (a > 0).astype(np.uint8).ravel()
    b = (b > 0).astype(np.uint8).ravel()
    hist_2d, _, _ = np.histogram2d(a, b, bins=bins, range=[[0, 1], [0, 1]])
    pxy = hist_2d / hist_2d.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    px_py = np.outer(px, py)
    nz = pxy > 0
    return float(np.sum(pxy[nz] * np.log(pxy[nz] / px_py[nz])))


def apply_xy_flips(arr: np.ndarray, flip_x: bool, flip_y: bool) -> np.ndarray:
    """
    Flip along array axes 0 / 1 (nibabel XYZ: axis0≈X, axis1≈Y).
    Matches the manual Mirror QC flips used in the pipeline.
    """
    out = np.asarray(arr)
    if flip_x:
        out = np.flip(out, axis=0)
    if flip_y:
        out = np.flip(out, axis=1)
    return out.copy()


def best_xy_flip(
    pred: np.ndarray,
    reference: np.ndarray,
    *,
    metric: str = "dice",
) -> Tuple[np.ndarray, dict]:
    """
    Try the 4 combinations: identity, flipX, flipY, flipX+flipY.
    Pick the one that maximises overlap with ``reference`` (e.g. torso_fat).

    Returns (best_array, info_dict).
    """
    if pred.shape != reference.shape:
        raise ValueError(f"Shape mismatch pred {pred.shape} vs ref {reference.shape}")

    ref = (reference > 0).astype(np.uint8)
    best_score = -1.0
    best = pred
    best_flags = (False, False)
    scores = {}

    for fx in (False, True):
        for fy in (False, True):
            cand = apply_xy_flips(pred, fx, fy)
            if metric == "mi":
                score = mutual_information_binary(cand, ref)
            else:
                score = dice_binary(cand, ref)
            scores[(fx, fy)] = score
            if score > best_score:
                best_score = score
                best = cand
                best_flags = (fx, fy)

    info = {
        "metric": metric,
        "best_flip_x": best_flags[0],
        "best_flip_y": best_flags[1],
        "best_score": best_score,
        "scores": {
            f"flipX={fx}_flipY={fy}": sc for (fx, fy), sc in scores.items()
        },
    }
    return best, info


def reorient_ras_pred_to_reference(
    pred_ras: np.ndarray,
    ras_affine: np.ndarray,
    ref_shape: Sequence[int],
    ref_affine: np.ndarray,
) -> np.ndarray:
    """
    Resample a RAS-space prediction onto the reference CT grid (nearest neighbour).

    Uses nibabel when available; falls back to a scipy map_coordinates path.
    """
    import nibabel as nib
    from nibabel.processing import resample_from_to

    src = nib.Nifti1Image(np.asarray(pred_ras, dtype=np.uint8), np.asarray(ras_affine))
    # Dummy target geometry
    tgt = nib.Nifti1Image(
        np.zeros(tuple(ref_shape), dtype=np.uint8), np.asarray(ref_affine)
    )
    out = resample_from_to(src, tgt, order=0)
    return np.asanyarray(out.dataobj, dtype=np.uint8)


def metatensor_affine(img) -> Optional[np.ndarray]:
    """Extract 4x4 affine from a MONAI MetaTensor / dict meta if present."""
    aff = getattr(img, "affine", None)
    if aff is not None:
        return np.asarray(aff.detach().cpu() if hasattr(aff, "detach") else aff, dtype=float)
    meta = getattr(img, "meta", None) or {}
    if "affine" in meta:
        a = meta["affine"]
        return np.asarray(a.detach().cpu() if hasattr(a, "detach") else a, dtype=float)
    return None
