"""
File-based organ post-processing (no Slicer).

Typical KU pipeline after VF / TotalSeg:
  1. Build PET-derived ureter / urinary mask (L1–L5 Z range)
  2. Clip + clean target organs (VF, psoas, spleen, …)
  3. Optional: dilate kidneys and subtract from targets
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from lib.processing.dilate import dilate_mask, subtract_dilated_union
from lib.processing.ureter import (
    apply_organ_processing,
    build_ureter_mask_from_pet,
    z_bounds_from_mask,
)

DEFAULT_ORGANS = (
    "visceral_fat.nii.gz",
    "iliopsoas_left.nii.gz",
    "iliopsoas_right.nii.gz",
    "spleen.nii.gz",
)

ORGAN_MODES = (
    "Clip only",
    "Clean only",
    "Clip + Clean",
    "Skip",
)

# Filenames / tokens that are usually not target quantification organs
AUX_ORGAN_TOKENS = (
    "combined",
    "torso_fat",
    "body_trunc",
    "body_extremities",
    "ureter",
    "urinary",
    "bladder",
    "vertebra",
    "rib_",
    "heart",
    "liver",
    "kidney",
    "skin",
    "ct",
    "dilated",
    "subtracted",
    "processed",
    "totalseg",
    "total_seg",
)

DEFAULT_VERTEBRAE = (
    "vertebrae_L1.nii.gz",
    "vertebrae_L2.nii.gz",
    "vertebrae_L3.nii.gz",
    "vertebrae_L4.nii.gz",
    "vertebrae_L5.nii.gz",
)

DEFAULT_KIDNEYS = (
    "kidney_left.nii.gz",
    "kidney_right.nii.gz",
)


def _is_aux_organ_name(name: str) -> bool:
    nl = name.lower()
    return any(tok in nl for tok in AUX_ORGAN_TOKENS)


def list_candidate_organs(
    root: str | Path,
    *,
    include_aux: bool = False,
    sample_only: bool = True,
) -> list[dict]:
    """
    Discover organ NIfTI filenames under Segments/*_Seg/.

    Returns list of dicts:
      {filename, stem, present_in, n_subjects_with_file, is_default, is_aux}
    """
    from lib.io.paths import find_bulk_subjects

    root = Path(root)
    subjects = find_bulk_subjects(str(root))
    if not subjects:
        return []

    sample_dirs = [Path(subjects[0]["seg_dir"])]
    if not sample_only:
        sample_dirs = [Path(s["seg_dir"]) for s in subjects]

    counts: dict[str, int] = {}
    for seg_dir in sample_dirs:
        if not seg_dir.is_dir():
            continue
        seen = set()
        for f in seg_dir.iterdir():
            name = f.name
            if not (name.endswith(".nii.gz") or name.endswith(".nii")):
                continue
            if name in seen:
                continue
            seen.add(name)
            counts[name] = counts.get(name, 0) + 1

    # If sample_only, also count how many subjects have each sample file
    if sample_only and subjects:
        full_counts: dict[str, int] = {k: 0 for k in counts}
        for s in subjects:
            seg_dir = Path(s["seg_dir"])
            for name in list(full_counts):
                if (seg_dir / name).is_file():
                    full_counts[name] += 1
        counts = full_counts

    default_set = set(DEFAULT_ORGANS)
    rows = []
    for name in sorted(counts):
        is_aux = _is_aux_organ_name(name)
        if is_aux and not include_aux:
            continue
        stem = name
        for ext in (".nii.gz", ".nii"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        rows.append(
            {
                "filename": name,
                "stem": stem,
                "n_subjects_with_file": counts[name],
                "is_default": name in default_set,
                "is_aux": is_aux,
            }
        )
    return rows


def format_organ_catalog(rows: Sequence[dict], *, n_subjects: int = 0) -> str:
    """Pretty table for --list-organs / interactive picker."""
    lines = []
    lines.append("Available target organs (NIfTI in Segments/*_Seg/):")
    lines.append(
        f"  {'#':>3}  {'filename':<40}  {'stem':<28}  "
        f"{'subjects':>8}  default"
    )
    lines.append("  " + "-" * 90)
    for i, r in enumerate(rows, 1):
        flag = "yes" if r.get("is_default") else ""
        n = r.get("n_subjects_with_file", 0)
        subj = f"{n}/{n_subjects}" if n_subjects else str(n)
        lines.append(
            f"  {i:3d}  {r['filename']:<40}  {r['stem']:<28}  {subj:>8}  {flag}"
        )
    if not rows:
        lines.append("  (none found)")
    lines.append("")
    lines.append("Processing modes per organ:")
    for i, m in enumerate(ORGAN_MODES, 1):
        lines.append(f"  {i}. {m}")
    return "\n".join(lines)



def _load_nii(path: str | Path):
    import nibabel as nib

    img = nib.load(str(path))
    # Slicer/SimpleITK array order is often ZYX; nibabel is XYZ.
    # Our ureter helpers expect ZYX-like arrays matching PET from Slicer.
    # For file-based NIfTI we keep nibabel XYZ and use consistent order
    # within this module: treat as (X,Y,Z) → transpose to (Z,Y,X) for processing.
    arr = np.asarray(img.dataobj)
    if arr.ndim == 3:
        arr_zyx = np.transpose(arr, (2, 1, 0))
    else:
        arr_zyx = arr
    # Affine maps IJK (I,J,K)=(X,Y,Z) → RAS; for ZYX voxel indexing we still
    # pass the same affine used with IJK = (x,y,z) in helpers that build
    # ijk_hom as [x,y,z,1] from (z,y,x) arrays — matching ureter.py.
    return arr_zyx, img.affine, img


def _save_nii(arr_zyx: np.ndarray, ref_img, out_path: str | Path) -> None:
    import nibabel as nib

    arr_xyz = np.transpose(np.asarray(arr_zyx), (2, 1, 0)).astype(np.uint8)
    out = nib.Nifti1Image(arr_xyz, ref_img.affine, ref_img.header)
    out.header.set_data_dtype(np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    nib.save(out, str(out_path))


def _find_first(seg_dir: Path, names: Sequence[str]) -> Optional[Path]:
    for n in names:
        p = seg_dir / n
        if p.is_file():
            return p
    return None


def load_pet_array(
    pet_path: str | Path | None = None,
    pet_dicom_dir: str | Path | None = None,
    pet_nii_out: str | Path | None = None,
    log=None,
) -> tuple[np.ndarray, np.ndarray, object]:
    """
    Load PET as (arr_zyx, affine, ref_img).
    Prefers NIfTI; can convert DICOM → NIfTI via SimpleITK.
    """
    if pet_path and Path(pet_path).is_file():
        return _load_nii(pet_path)

    if pet_dicom_dir and Path(pet_dicom_dir).is_dir():
        from lib.io.nifti import convert_dicom_to_nifti

        out = Path(pet_nii_out) if pet_nii_out else (
            Path(pet_dicom_dir).parent.parent / "PET_NIfTI" /
            (Path(pet_dicom_dir).name.replace("_PET", "") + "_PET.nii.gz")
        )
        convert_dicom_to_nifti(str(pet_dicom_dir), str(out), log=log)
        return _load_nii(out)

    raise FileNotFoundError("No PET NIfTI or DICOM folder available")


def vertebrae_z_bounds(seg_dir: Path, vertebrae_files: Sequence[str] = DEFAULT_VERTEBRAE):
    """Union L1–L5 masks and return RAS Z min/max."""
    union = None
    affine = None
    for name in vertebrae_files:
        fp = seg_dir / name
        if not fp.is_file():
            continue
        arr, aff, _ = _load_nii(fp)
        union = arr.astype(np.uint8) if union is None else np.maximum(union, (arr > 0).astype(np.uint8))
        affine = aff
    if union is None or affine is None:
        raise FileNotFoundError(
            f"No vertebrae files found in {seg_dir} "
            f"(tried {list(vertebrae_files)[:3]}…)"
        )
    return z_bounds_from_mask(union, affine)


def torso_center_xy(seg_dir: Path, vertebrae_files: Sequence[str] = DEFAULT_VERTEBRAE):
    """RAS XY centroid of L-spine for ureter torso radius."""
    pts = []
    for name in vertebrae_files:
        fp = seg_dir / name
        if not fp.is_file():
            continue
        arr, aff, _ = _load_nii(fp)
        idx = np.argwhere(arr > 0)
        if len(idx) == 0:
            continue
        ijk = np.column_stack(
            [idx[:, 2], idx[:, 1], idx[:, 0], np.ones(len(idx))]
        ).astype(np.float64)
        ras = (np.asarray(aff) @ ijk.T).T
        pts.append(ras[:, :2].mean(axis=0))
    if not pts:
        raise FileNotFoundError(f"No vertebrae voxels for centroid in {seg_dir}")
    mean = np.mean(np.stack(pts, axis=0), axis=0)
    return float(mean[0]), float(mean[1])


def process_subject_seg_dir(
    seg_dir: str | Path,
    pet_arr: np.ndarray,
    pet_affine,
    *,
    organs: Sequence[str] = DEFAULT_ORGANS,
    mode: str = "Clip + Clean",
    organ_modes: Optional[dict[str, str]] = None,
    suv_thresh: float = 4.0,
    suv_clean_thresh: float = 1.2,
    dilate_mm: float = 5.0,
    kidney_dilate_mm: float = 3.0,
    subtract_kidneys: bool = False,
    connect_path: bool = True,
    skip_done: bool = True,
    write_ureter: bool = True,
    log=None,
) -> dict:
    """
    Process one patient's Segments/<id>_Seg folder.

    ``organ_modes`` maps filename → mode (``Clip only`` / ``Clean only`` /
    ``Clip + Clean`` / ``Skip``). When omitted, every entry in ``organs`` uses
    ``mode``.

    Writes ``ureter_from_pet.nii.gz`` and ``<stem>_processed.nii.gz`` files.
    """
    seg_dir = Path(seg_dir)
    info = {"seg_dir": str(seg_dir), "organs": [], "skipped": [], "errors": []}

    def _info(msg: str):
        print(msg)
        if log and hasattr(log, "info"):
            log.info(msg)

    # Resolve organ → mode map
    modes: dict[str, str] = {}
    if organ_modes:
        modes = {str(k): str(v) for k, v in organ_modes.items()}
    else:
        modes = {str(o): mode for o in organs}

    organ_list = list(modes.keys()) if organ_modes else list(organs)
    # Drop explicit Skip entries from work list but keep them logged
    work = []
    for name in organ_list:
        m = modes.get(name, mode)
        if m == "Skip":
            info["skipped"].append(name)
            _info(f"  [SKIP] user skipped {name}")
            continue
        if m not in ORGAN_MODES:
            raise ValueError(f"Unknown mode {m!r} for {name}; choose from {ORGAN_MODES}")
        work.append((name, m))

    needs_ureter = any(m in ("Clean only", "Clip + Clean") for _, m in work)
    needs_clip = any(m in ("Clip only", "Clip + Clean") for _, m in work)
    needs_z = needs_ureter or needs_clip

    z_inf = z_sup = None
    cx = cy = 0.0
    if needs_z:
        z_inf, z_sup = vertebrae_z_bounds(seg_dir)
    if needs_ureter:
        cx, cy = torso_center_xy(seg_dir)

    vox = np.abs(np.diag(np.asarray(pet_affine))[:3])
    spacing_zyx = (float(vox[2]), float(vox[1]), float(vox[0]))

    ureter_arr = ureter_aff = None
    if needs_ureter:
        ureter_path = seg_dir / "ureter_from_pet.nii.gz"
        if skip_done and ureter_path.is_file() and write_ureter:
            ureter_arr, ureter_aff, _ = _load_nii(ureter_path)
            _info(f"  [SKIP] ureter exists: {ureter_path.name}")
        else:
            ureter_arr = build_ureter_mask_from_pet(
                pet_arr,
                pet_affine,
                spacing_zyx,
                z_inferior=float(z_inf),
                z_superior=float(z_sup),
                suv_thresh=suv_thresh,
                dilate_mm=dilate_mm,
                torso_center_xy=(cx, cy),
                connect_path=connect_path,
            )
            import nibabel as nib

            pet_xyz = np.transpose(ureter_arr, (2, 1, 0)).astype(np.uint8)
            ureter_img = nib.Nifti1Image(pet_xyz, pet_affine)
            ureter_aff = pet_affine
            if write_ureter:
                _save_nii(ureter_arr, ureter_img, ureter_path)
                _info(
                    f"  [OK] ureter → {ureter_path.name}  nz={int(ureter_arr.sum()):,}"
                )
    else:
        _info("  [INFO] no Clean mode selected — ureter mask not built")

    kidney_items = []
    if subtract_kidneys:
        for kname in DEFAULT_KIDNEYS:
            kpath = seg_dir / kname
            if not kpath.is_file():
                continue
            karr, kaff, _ = _load_nii(kpath)
            kidney_items.append((dilate_mask(karr, kaff, kidney_dilate_mm), kaff))

    for organ_name, organ_mode in work:
        opath = seg_dir / organ_name
        if not opath.is_file():
            info["skipped"].append(organ_name)
            _info(f"  [SKIP] missing organ {organ_name}")
            continue
        stem = organ_name
        for ext in (".nii.gz", ".nii"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        out_path = seg_dir / f"{stem}_processed.nii.gz"
        if skip_done and out_path.is_file():
            info["skipped"].append(organ_name)
            _info(f"  [SKIP] exists {out_path.name}")
            continue
        try:
            oarr, oaff, oimg = _load_nii(opath)
            processed = apply_organ_processing(
                oarr,
                oaff,
                mode=organ_mode,
                ureter_arr=ureter_arr,
                ureter_affine=ureter_aff,
                pet_arr=pet_arr,
                pet_affine=pet_affine,
                suv_clean_thresh=suv_clean_thresh,
                z_inferior=z_inf,
                z_superior=z_sup,
            )
            if kidney_items:
                processed = subtract_dilated_union(processed, oaff, kidney_items)
            _save_nii(processed, oimg, out_path)
            info["organs"].append(str(out_path.name))
            _info(
                f"  [OK] {out_path.name}  mode={organ_mode!r}  "
                f"nz {int((oarr > 0).sum()):,} → {int((processed > 0).sum()):,}"
            )
        except Exception as e:
            info["errors"].append(f"{organ_name}: {e}")
            _info(f"  [ERR] {organ_name}: {e}")

    return info
