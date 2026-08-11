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

from lib.processing.dilate import dilate_mask, resample_to_target, subtract_dilated_union
from lib.processing.ureter import (
    apply_organ_processing,
    build_ureter_mask_from_pet,
    clip_organ_to_z,
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

# ── KU protocol defaults (ureter + group subtract/clean) ─────────────────────
URETER_SUV_THRESH = 2.5
URETER_DILATE_MM = 18.0
URETER_EXT_INF_MM = 50.0
URETER_TORSO_RADIUS_MM = 220.0
GROUP_SUBTRACT_DILATE_MM = 5.0
CLEAN_EXCLUDE_DILATE_MM = 13.0
SUV_CLEAN_FAT = 1.2
SUV_CLEAN_PSOAS = 1.6

GROUP_SEGNRRD_FILES = (
    "abdomen.seg.nrrd",
    "vessels.seg.nrrd",
    "spine.seg.nrrd",
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


def _sitk_affine_ras(img) -> np.ndarray:
    """SimpleITK image → 4x4 affine mapping index (i,j,k) → physical RAS-ish."""
    spacing = np.asarray(img.GetSpacing(), dtype=np.float64)
    origin = np.asarray(img.GetOrigin(), dtype=np.float64)
    direction = np.asarray(img.GetDirection(), dtype=np.float64).reshape(3, 3)
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = direction @ np.diag(spacing)
    affine[:3, 3] = origin
    return affine


def _guess_pet_nii(seg_dir: Path) -> Optional[Path]:
    """Locate ``<subject>_PET.nii.gz`` next to ``Segments/<subject>_Seg``."""
    stem = seg_dir.name.replace("_Seg", "")
    root = seg_dir.parent.parent
    for sub in ("PET_NIfTI", "PET"):
        p = root / sub / f"{stem}_PET.nii.gz"
        if p.is_file():
            return p
    return None


def _resolve_ref_nii(seg_dir: Path, organ_list: Sequence[str]) -> Path:
    """CT-grid reference NIfTI (prefer visceral fat) for group / exclusion alignment."""
    candidates = ["visceral_fat.nii.gz"]
    for name in organ_list:
        n = str(name)
        if not (n.endswith(".nii.gz") or n.endswith(".nii")):
            n = n + ".nii.gz"
        if n not in candidates:
            candidates.append(n)
    for name in candidates:
        p = seg_dir / name
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"No reference organ NIfTI in {seg_dir} (tried {', '.join(candidates)})"
    )


def _resample_image_to_nifti_ref(moving_path: str | Path, ref_nii_path: str | Path) -> np.ndarray:
    """Resample any SimpleITK-readable mask onto a NIfTI reference grid (ZYX uint8)."""
    import SimpleITK as sitk

    ref = sitk.ReadImage(str(ref_nii_path))
    moving = sitk.ReadImage(str(moving_path))
    out = sitk.Resample(
        moving,
        ref,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        moving.GetPixelID(),
    )
    return (sitk.GetArrayFromImage(out) > 0).astype(np.uint8)


def _segnrrd_union_on_ref_excluding(
    segnrrd_path: str | Path,
    ref_nii_path: str | Path,
    exclude_names: Sequence[str],
) -> np.ndarray:
    """Load a ``.seg.nrrd``, drop named segments, resample union onto ref NIfTI grid."""
    import SimpleITK as sitk

    exclude = {str(n).strip() for n in exclude_names}
    img = sitk.ReadImage(str(segnrrd_path))
    arr = sitk.GetArrayFromImage(img)
    drop_labels: list[int] = []
    i = 0
    while img.HasMetaDataKey(f"Segment{i}_Name"):
        name = str(img.GetMetaData(f"Segment{i}_Name")).strip()
        if name in exclude:
            if img.HasMetaDataKey(f"Segment{i}_LabelValue"):
                drop_labels.append(int(float(img.GetMetaData(f"Segment{i}_LabelValue"))))
            else:
                drop_labels.append(i + 1)
        i += 1
    for lab in drop_labels:
        arr = np.where(arr == lab, 0, arr)
    tmp = sitk.GetImageFromArray(arr.astype(np.uint8))
    tmp.CopyInformation(img)
    ref = sitk.ReadImage(str(ref_nii_path))
    out = sitk.Resample(
        tmp,
        ref,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )
    return (sitk.GetArrayFromImage(out) > 0).astype(np.uint8)


# Organs that also appear inside abdomen.seg.nrrd — exclude from self-subtraction
SELF_EXCLUDE_FROM_ABDOMEN = {
    "spleen": ("spleen",),
}


def _group_dilated_for_subtract(
    seg_dir: Path,
    ref_nii_path: Path,
    ref_aff,
    dilate_mm: float,
    organ_stem: str,
) -> list[np.ndarray]:
    """5 mm dilated abdomen/vessels/spine masks for hard-subtract (skip self for spleen)."""
    exclude = SELF_EXCLUDE_FROM_ABDOMEN.get(organ_stem.lower(), ())
    dilated: list[np.ndarray] = []
    for name in GROUP_SEGNRRD_FILES:
        path = seg_dir / name
        if not path.is_file():
            continue
        if name == "abdomen.seg.nrrd" and exclude:
            union = _segnrrd_union_on_ref_excluding(path, ref_nii_path, exclude)
        else:
            union = _resample_image_to_nifti_ref(path, ref_nii_path)
        if union.any():
            dilated.append(dilate_mask(union, ref_aff, float(dilate_mm)))
    return dilated


def _resample_mask_zyx_to_nifti_ref(
    arr_zyx: np.ndarray,
    template_nii_path: str | Path,
    ref_nii_path: str | Path,
) -> np.ndarray:
    """Resample an in-memory ZYX mask (geometry = template NIfTI) onto ref NIfTI grid."""
    import SimpleITK as sitk

    tmpl = sitk.ReadImage(str(template_nii_path))
    ref = sitk.ReadImage(str(ref_nii_path))
    img = sitk.GetImageFromArray(np.asarray(arr_zyx).astype(np.uint8))
    img.CopyInformation(tmpl)
    out = sitk.Resample(
        img,
        ref,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )
    return (sitk.GetArrayFromImage(out) > 0).astype(np.uint8)


def load_segnrrd_union_zyx(path: str | Path):
    """
    Load a multilabel ``.seg.nrrd`` as a binary union (ZYX) + affine.

    Returns ``(union_zyx, affine)`` or ``(None, None)`` if missing/empty.
    """
    path = Path(path)
    if not path.is_file():
        return None, None
    try:
        import SimpleITK as sitk
    except ImportError:
        return None, None

    img = sitk.ReadImage(str(path))
    arr_zyx = sitk.GetArrayFromImage(img)
    union = (np.asarray(arr_zyx) > 0).astype(np.uint8)
    if not union.any():
        return None, None
    return union, _sitk_affine_ras(img)


def organ_suv_clean_threshold(organ_name: str) -> Optional[float]:
    """Per-organ PET SUV threshold for the 13 mm exclusion clean step."""
    nl = organ_name.lower()
    if "visceral_fat" in nl or nl.startswith("fat"):
        return SUV_CLEAN_FAT
    if "iliopsoas" in nl or "psoas" in nl:
        return SUV_CLEAN_PSOAS
    # spleen / others: hard subtract only (no SUV clean unless set)
    return None


def _load_group_unions(
    seg_dir: Path,
    ref_nii_path: str | Path,
    log_fn=None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Load abdomen / vessels / spine unions resampled onto ``ref_nii_path`` grid.

    SimpleITK resampling avoids nibabel vs .seg.nrrd affine mismatches and keeps
    full CT resolution (do not resample groups onto the lower-res PET grid).
    """
    ref_nii_path = Path(ref_nii_path)
    _, ref_aff, _ = _load_nii(ref_nii_path)
    items: list[np.ndarray] = []
    for name in GROUP_SEGNRRD_FILES:
        path = seg_dir / name
        if not path.is_file():
            if log_fn:
                log_fn(f"  [WARN] missing group {name}")
            continue
        try:
            union = _resample_image_to_nifti_ref(path, ref_nii_path)
        except Exception as e:
            if log_fn:
                log_fn(f"  [WARN] failed to load {name}: {e}")
            continue
        if not union.any():
            if log_fn:
                log_fn(f"  [WARN] empty group {name}")
            continue
        items.append(union)
        if log_fn:
            log_fn(f"  [group] {name} on CT ref  nz={int(union.sum()):,}")
    return items, ref_aff


def clean_organ_with_exclusion(
    organ_arr: np.ndarray,
    organ_affine,
    excl_arr,
    excl_affine,
    pet_arr,
    pet_affine,
    suv_clean_thresh: float,
) -> np.ndarray:
    """Remove organ voxels overlapping exclusion where PET > threshold."""
    organ_arr = organ_arr.copy()
    excl_in = resample_to_target(excl_arr, excl_affine, organ_arr.shape, organ_affine)
    pet_in = resample_to_target(pet_arr, pet_affine, organ_arr.shape, organ_affine)
    remove = (organ_arr > 0) & (excl_in > 0) & (pet_in > float(suv_clean_thresh))
    organ_arr[remove] = 0
    return organ_arr


def process_subject_ku_protocol(
    seg_dir: str | Path,
    pet_arr: np.ndarray,
    pet_affine,
    *,
    organs: Sequence[str] = DEFAULT_ORGANS,
    pet_path: str | Path | None = None,
    ref_organ_path: str | Path | None = None,
    ureter_suv_thresh: float = URETER_SUV_THRESH,
    ureter_dilate_mm: float = URETER_DILATE_MM,
    ureter_ext_inf_mm: float = URETER_EXT_INF_MM,
    torso_radius_mm: float = URETER_TORSO_RADIUS_MM,
    group_subtract_dilate_mm: float = GROUP_SUBTRACT_DILATE_MM,
    clean_exclude_dilate_mm: float = CLEAN_EXCLUDE_DILATE_MM,
    suv_clean_fat: float = SUV_CLEAN_FAT,
    suv_clean_psoas: float = SUV_CLEAN_PSOAS,
    connect_path: bool = True,
    fill_holes: bool = True,
    skip_done: bool = True,
    write_ureter: bool = True,
    log=None,
) -> dict:
    """
    KU post-processing protocol:

      1. Build PET ureter mask (SUV thresh / dilate / L5 extend / torso cylinder;
         fill holes + bridge gaps on).
      2. Dilate abdomen + vessels + spine by ``group_subtract_dilate_mm`` (5 mm)
         and hard-subtract from each target organ.
      3. Dilate ureter + abdomen + vessels + spine by ``clean_exclude_dilate_mm``
         (13 mm); in overlap with fat/psoas, remove voxels with PET above
         organ-specific SUV (fat 1.2, psoas 1.6).
      4. Visceral fat is also clipped to L1–L5 Z.

    Writes ``ureter_from_pet.nii.gz`` and ``<stem>_processed.nii.gz``.
    """
    seg_dir = Path(seg_dir)
    info = {"seg_dir": str(seg_dir), "organs": [], "skipped": [], "errors": []}

    def _info(msg: str):
        print(msg)
        if log and hasattr(log, "info"):
            log.info(msg)

    organ_list = []
    for name in organs:
        n = str(name)
        if not (n.endswith(".nii.gz") or n.endswith(".nii")):
            n = n + ".nii.gz"
        organ_list.append(n)

    ref_nii = Path(ref_organ_path) if ref_organ_path else _resolve_ref_nii(seg_dir, organ_list)
    ref_arr, ref_aff, _ = _load_nii(ref_nii)
    pet_nii = Path(pet_path) if pet_path else _guess_pet_nii(seg_dir)
    _info(f"  [ref] CT grid: {ref_nii.name}  shape={ref_arr.shape}")

    z_inf, z_sup = vertebrae_z_bounds(seg_dir)
    cx, cy = torso_center_xy(seg_dir)
    _info(f"  [Z] L1-L5 bounds: {z_inf:.1f} .. {z_sup:.1f} mm")

    vox = np.abs(np.diag(np.asarray(pet_affine))[:3])
    spacing_zyx = (float(vox[2]), float(vox[1]), float(vox[0]))

    # ── 1. Ureter mask ───────────────────────────────────────────────────────
    ureter_path = seg_dir / "ureter_from_pet.nii.gz"
    if skip_done and ureter_path.is_file() and write_ureter:
        ureter_arr, ureter_aff, _ = _load_nii(ureter_path)
        _info(f"  [SKIP] ureter exists: {ureter_path.name}")
    else:
        _info(
            f"  [ureter] SUV>{ureter_suv_thresh} dilate={ureter_dilate_mm}mm "
            f"extend_below_L5={ureter_ext_inf_mm}mm "
            f"connect={connect_path} fill_holes={fill_holes}"
        )
        ureter_arr = build_ureter_mask_from_pet(
            pet_arr,
            pet_affine,
            spacing_zyx,
            z_inferior=float(z_inf),
            z_superior=float(z_sup),
            suv_thresh=float(ureter_suv_thresh),
            dilate_mm=float(ureter_dilate_mm),
            torso_center_xy=(cx, cy),
            ureter_ext_inf_mm=float(ureter_ext_inf_mm),
            torso_radius_mm=float(torso_radius_mm),
            connect_path=connect_path,
            fill_holes=fill_holes,
        )
        ureter_aff = pet_affine
        if write_ureter:
            import nibabel as nib

            # reuse PET affine header via a lightweight ref image
            pet_xyz = np.transpose(ureter_arr, (2, 1, 0)).astype(np.uint8)
            ref = nib.Nifti1Image(pet_xyz, pet_affine)
            _save_nii(ureter_arr, ref, ureter_path)
            _info(f"  [OK] ureter → {ureter_path.name}  nz={int(ureter_arr.sum()):,}")

    # ── Group unions (abdomen / vessels / spine) on CT reference grid ─────────
    group_masks, group_aff = _load_group_unions(seg_dir, ref_nii, log_fn=_info)
    if not group_masks:
        _info("  [WARN] no abdomen/vessels/spine .seg.nrrd groups found")

    # Pre-dilate groups for step 2 logging / step 3 exclusion (full abdomen union)
    group_dilated_5 = [
        dilate_mask(g, group_aff, float(group_subtract_dilate_mm)) for g in group_masks
    ]

    # Exclusion for step 3: ureter ∪ groups on CT grid, then dilate 13 mm
    if pet_nii and pet_nii.is_file():
        ureter_on_ct = _resample_mask_zyx_to_nifti_ref(ureter_arr, pet_nii, ref_nii)
    else:
        _info("  [WARN] PET NIfTI path unknown — affine resample ureter → CT ref")
        ureter_on_ct = resample_to_target(
            ureter_arr, ureter_aff, ref_arr.shape, ref_aff
        )
        ureter_on_ct = (ureter_on_ct > 0).astype(np.uint8)

    excl_base = (ureter_on_ct > 0).astype(np.uint8)
    for g in group_masks:
        excl_base = np.maximum(excl_base, (g > 0).astype(np.uint8))
    excl_dilated_13 = dilate_mask(excl_base, ref_aff, float(clean_exclude_dilate_mm))
    _info(
        f"  [excl] CT grid 5mm subtract groups={len(group_dilated_5)}  "
        f"13mm clean excl nz={int(excl_dilated_13.sum()):,}"
    )

    # ── Per-target organ ─────────────────────────────────────────────────────
    for organ_name in organ_list:
        opath = seg_dir / organ_name
        if not opath.is_file():
            info["skipped"].append(organ_name)
            _info(f"  [SKIP] missing {organ_name}")
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
            nz0 = int((oarr > 0).sum())
            processed = oarr.copy()

            # VF: always clip to L1–L5
            is_vf = "visceral_fat" in stem.lower()
            if is_vf:
                processed = clip_organ_to_z(processed, oaff, float(z_inf), float(z_sup))
                _info(
                    f"  [clip] {stem} → L1-L5  "
                    f"nz {nz0:,} → {int((processed > 0).sum()):,}"
                )
                nz0 = int((processed > 0).sum())

            # Step 2: hard-subtract 5 mm dilated abdomen/vessels/spine
            group_sub_5 = _group_dilated_for_subtract(
                seg_dir,
                ref_nii,
                ref_aff,
                group_subtract_dilate_mm,
                stem,
            )
            if group_sub_5:
                before = int((processed > 0).sum())
                if processed.shape == ref_arr.shape and np.allclose(oaff, ref_aff):
                    processed = subtract_dilated_union(
                        processed,
                        oaff,
                        [],
                        same_grid_items=group_sub_5,
                    )
                else:
                    processed = subtract_dilated_union(
                        processed,
                        oaff,
                        [(g, ref_aff) for g in group_sub_5],
                    )
                _info(
                    f"  [sub5] {stem}  "
                    f"nz {before:,} → {int((processed > 0).sum()):,}"
                )

            # Step 3: SUV-clean in 13 mm dilated ureter∪groups (fat / psoas)
            nl = stem.lower()
            if "visceral_fat" in nl:
                thresh = float(suv_clean_fat)
            elif "iliopsoas" in nl or "psoas" in nl:
                thresh = float(suv_clean_psoas)
            else:
                thresh = None

            if thresh is not None:
                before = int((processed > 0).sum())
                if processed.shape == ref_arr.shape and np.allclose(oaff, ref_aff):
                    excl_for_organ = excl_dilated_13
                else:
                    excl_for_organ = resample_to_target(
                        excl_dilated_13, ref_aff, processed.shape, oaff
                    )
                    excl_for_organ = (excl_for_organ > 0).astype(np.uint8)
                processed = clean_organ_with_exclusion(
                    processed,
                    oaff,
                    excl_for_organ,
                    oaff,
                    pet_arr,
                    pet_affine,
                    float(thresh),
                )
                _info(
                    f"  [clean13] {stem} SUV>{thresh}  "
                    f"nz {before:,} → {int((processed > 0).sum()):,}"
                )

            _save_nii(processed, oimg, out_path)
            info["organs"].append(str(out_path.name))
            _info(
                f"  [OK] {out_path.name}  "
                f"nz {int((oarr > 0).sum()):,} → {int((processed > 0).sum()):,}"
            )
        except Exception as e:
            info["errors"].append(f"{organ_name}: {e}")
            _info(f"  [ERR] {organ_name}: {e}")

    return info


def _lumbar_union_from_spine_segnrrd(seg_dir: Path):
    """
    Load L1–L5 from ``spine.seg.nrrd`` (ZYX uint8 union + affine).

    Returns (union_zyx, affine) or (None, None) if unavailable.
    """
    spine_path = seg_dir / "spine.seg.nrrd"
    if not spine_path.is_file():
        return None, None
    try:
        import SimpleITK as sitk
    except ImportError:
        return None, None

    img = sitk.ReadImage(str(spine_path))
    arr_zyx = sitk.GetArrayFromImage(img)
    wanted = {"L1", "L2", "L3", "L4", "L5",
              "vertebrae_L1", "vertebrae_L2", "vertebrae_L3",
              "vertebrae_L4", "vertebrae_L5"}
    labels = []
    # Slicer-style metadata written by write_seg_nrrd
    i = 0
    while True:
        name_key = f"Segment{i}_Name"
        lab_key = f"Segment{i}_LabelValue"
        if not img.HasMetaDataKey(name_key):
            break
        name = str(img.GetMetaData(name_key)).strip()
        try:
            lab = int(float(img.GetMetaData(lab_key))) if img.HasMetaDataKey(lab_key) else i + 1
        except Exception:
            lab = i + 1
        if name in wanted:
            labels.append(lab)
        i += 1

    # Fallback: if metadata missing, use all non-zero labels (still better than fail)
    if not labels:
        uniq = [int(v) for v in np.unique(arr_zyx) if int(v) != 0]
        # Prefer first 5 labels if names unavailable (L1-L5 usually packed first)
        labels = uniq[:5] if uniq else []

    if not labels:
        return None, None

    union = np.zeros(arr_zyx.shape, dtype=np.uint8)
    for lab in labels:
        union |= (arr_zyx == lab).astype(np.uint8)
    if not union.any():
        return None, None
    return union, _sitk_affine_ras(img)


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
        union, affine = _lumbar_union_from_spine_segnrrd(Path(seg_dir))
    if union is None or affine is None:
        raise FileNotFoundError(
            f"No lumbar vertebrae found in {seg_dir} "
            f"(loose L1–L5 NIfTI or spine.seg.nrrd)"
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
        union, aff = _lumbar_union_from_spine_segnrrd(Path(seg_dir))
        if union is None or aff is None:
            raise FileNotFoundError(f"No vertebrae voxels for centroid in {seg_dir}")
        idx = np.argwhere(union > 0)
        ijk = np.column_stack(
            [idx[:, 2], idx[:, 1], idx[:, 0], np.ones(len(idx))]
        ).astype(np.float64)
        ras = (np.asarray(aff) @ ijk.T).T
        pts.append(ras[:, :2].mean(axis=0))
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
                ureter_ext_inf_mm=URETER_EXT_INF_MM,
                torso_radius_mm=URETER_TORSO_RADIUS_MM,
                connect_path=connect_path,
                fill_holes=True,
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
