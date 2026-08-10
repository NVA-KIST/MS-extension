"""Left/right or A/P flip of NIfTI / arrays (no Slicer UI)."""
from __future__ import annotations

from typing import Optional

import numpy as np


def flip_volume_axis(arr: np.ndarray, axis: int) -> np.ndarray:
    """Flip a volume along the given array axis (0=X/I, 1=Y/J, 2=Z/K for XYZ nibabel)."""
    return np.flip(np.asarray(arr), axis=axis).copy()


def flip_nifti_file(
    nii_path: str,
    ref_ct: str,
    axis: int,
    *,
    axis_name: Optional[str] = None,
    log=None,
) -> None:
    """
    Flip a NIfTI on disk in-place, using ``ref_ct`` affine/header.
    Port of PETCTSegmentationModuleLogic._flipNiftiAxis / mirroring.py.
    """
    import nibabel as nib

    def _info(msg: str):
        if log:
            if hasattr(log, "info"):
                log.info(msg)
            else:
                log("info", msg)

    ct_img = nib.load(ref_ct)
    arr = np.asarray(nib.load(nii_path).dataobj, dtype=np.uint8)
    before_nz = int(np.count_nonzero(arr))
    flipped = flip_volume_axis(arr, axis)
    after_nz = int(np.count_nonzero(flipped))
    _info(
        f"flip axis={axis_name or axis}: nonzero {before_nz} → {after_nz} ({nii_path})"
    )
    out = nib.Nifti1Image(flipped, ct_img.affine, ct_img.header)
    out.header.set_data_dtype(np.uint8)
    try:
        out.header.set_intent("label", (), "")
    except Exception:
        pass
    out.header["scl_slope"] = 0
    out.header["scl_inter"] = 0
    nib.save(out, nii_path)
