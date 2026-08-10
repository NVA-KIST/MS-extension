"""
NIfTI helpers (no Slicer UI).

Integrity checks + DICOM→NIfTI via SimpleITK.
Flip delegates to lib.processing.mirroring.
"""
from __future__ import annotations

import os
import time
from typing import Optional


def _warn(log, msg: str) -> None:
    if log is None:
        return
    if hasattr(log, "warn"):
        log.warn(msg)
    elif callable(log):
        log("warn", msg)


def _info(log, msg: str) -> None:
    if log is None:
        return
    if hasattr(log, "info"):
        log.info(msg)
    elif callable(log):
        log("info", msg)


def _ok(log, msg: str) -> None:
    if log is None:
        return
    if hasattr(log, "ok"):
        log.ok(msg)
    elif hasattr(log, "info"):
        log.info(msg)


def mb_str(path: str) -> str:
    try:
        return f"{os.path.getsize(path) / 1e6:.1f} MB"
    except OSError:
        return "not found"


def ok_nii(
    path: str,
    min_kb: int = 5,
    need_nonzero: bool = False,
    log=None,
) -> bool:
    """
    Return True only if path is a readable, non-corrupted NIfTI.

    Checks (cheapest first): exists → size → nibabel header/shape → optional nonzero.
    """
    if not os.path.isfile(path):
        return False

    size_kb = os.path.getsize(path) / 1024.0
    if size_kb < min_kb:
        _warn(
            log,
            f"  Integrity FAIL — file too small ({size_kb:.1f} KB < {min_kb} KB): "
            f"{os.path.basename(path)}",
        )
        _warn(log, "  Will re-run this stage to regenerate the file.")
        return False

    try:
        import nibabel as nib
    except ImportError:
        return True  # size check only

    try:
        img = nib.load(path)
        shp = img.header.get_data_shape()
        if len(shp) < 3 or any(s == 0 for s in shp):
            _warn(log, f"  Integrity FAIL — invalid shape {shp}: {os.path.basename(path)}")
            return False
    except Exception as e:
        _warn(
            log,
            f"  Integrity FAIL — nibabel cannot open file ({e}): "
            f"{os.path.basename(path)}",
        )
        return False

    if need_nonzero:
        try:
            import numpy as np

            arr = np.asarray(nib.load(path).dataobj)
            if not arr.any():
                _warn(
                    log,
                    f"  Integrity FAIL — array is all zeros: {os.path.basename(path)}",
                )
                return False
        except Exception as e:
            _warn(log, f"  Integrity FAIL — cannot read array ({e}): {os.path.basename(path)}")
            return False

    return True


def flip_nifti_axis(
    nii_path: str,
    ref_ct: str,
    axis: int,
    axis_name: str = "",
    log=None,
) -> None:
    """Flip a NIfTI along axis (0=X, 1=Y)."""
    from lib.processing.mirroring import flip_nifti_file

    flip_nifti_file(nii_path, ref_ct, axis, axis_name=axis_name or None, log=log)


def convert_dicom_to_nifti(dicom_dir: str, out_nii: str, log=None) -> None:
    """
    Convert a DICOM series folder to NIfTI with SimpleITK.
    Skips when ``out_nii`` already passes integrity (≥ 5 MB).
    """
    if hasattr(log, "sep"):
        log.sep("convertDicom")
    _info(log, f"Source DICOM dir : {dicom_dir}")
    _info(log, f"Target NIfTI     : {out_nii}")

    if not os.path.isdir(dicom_dir):
        raise FileNotFoundError(f"CT DICOM directory not found: {dicom_dir}")

    dcm_files = [f for f in os.listdir(dicom_dir) if not f.startswith(".")]
    _info(log, f"Files in source  : {len(dcm_files)}")

    if ok_nii(out_nii, min_kb=5_000, need_nonzero=False, log=log):
        _ok(log, f"CT NIfTI valid ({mb_str(out_nii)}) — SKIPPING conversion.")
        return
    if os.path.isfile(out_nii):
        _warn(log, "CT NIfTI exists but failed integrity check — regenerating.")

    import SimpleITK as sitk

    _info(log, "Running SimpleITK ImageSeriesReader…")
    t0 = time.perf_counter()

    reader = sitk.ImageSeriesReader()
    ids = reader.GetGDCMSeriesIDs(dicom_dir)
    _info(log, f"DICOM series IDs found: {list(ids)}")
    if not ids:
        raise ValueError(f"No DICOM series found in {dicom_dir}")

    fnames = reader.GetGDCMSeriesFileNames(dicom_dir, ids[0])
    _info(log, f"Files in series  : {len(fnames)}")
    reader.SetFileNames(fnames)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()

    image = reader.Execute()
    size = image.GetSize()
    sp = image.GetSpacing()
    _info(log, f"Image size       : {size[0]}×{size[1]}×{size[2]}")
    _info(log, f"Voxel spacing    : {sp[0]:.3f}×{sp[1]:.3f}×{sp[2]:.3f} mm")

    os.makedirs(os.path.dirname(os.path.abspath(out_nii)) or ".", exist_ok=True)
    sitk.WriteImage(image, str(out_nii))
    elapsed = time.perf_counter() - t0
    _ok(log, f"CT NIfTI saved: {mb_str(out_nii)} in {elapsed:.1f}s")
