"""
Path / dataset discovery helpers (no slicer / qt).

Quantification discovery + segmentation patient / bulk subject helpers.
"""
from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Tuple


_FOLDER_RE = re.compile(r"^(.+)_(\d{4}-\d{2}-\d{2})_(CT|PET|Seg)$")


def parse_folder_name(name: str) -> Optional[Tuple[str, str, str]]:
    """Parse '{SubjectID}_{YYYY-MM-DD}_{CT|PET|Seg}' → (subj, date, kind)."""
    m = _FOLDER_RE.match(name)
    return (m.group(1), m.group(2), m.group(3)) if m else None


def detect_scans(root_folder: str, log=None) -> list[dict[str, Any]]:
    """
    Walk root/PET/ and build scan dicts for quantification.

    Each dict: subject_id, scan_date, pet_path, ct_path|None, seg_path|None
    """
    def _log(level, msg):
        if log:
            log(level, msg)

    pet_root = os.path.join(root_folder, "PET")
    ct_root = os.path.join(root_folder, "CT")
    seg_root = os.path.join(root_folder, "Segments")

    if not os.path.isdir(pet_root):
        raise ValueError(f"No PET/ folder found in {root_folder}")

    scans: list[dict[str, Any]] = []
    for folder in sorted(os.listdir(pet_root)):
        fpath = os.path.join(pet_root, folder)
        if not os.path.isdir(fpath):
            continue
        parsed = parse_folder_name(folder)
        if not parsed:
            _log("warning", f"detect_scans: skipping unrecognised folder: {folder}")
            continue
        subj, date, _ = parsed
        ct_p = os.path.join(ct_root, f"{subj}_{date}_CT")
        seg_p = os.path.join(seg_root, f"{subj}_{date}_Seg")
        ct_found = os.path.isdir(ct_p)
        seg_found = os.path.isdir(seg_p)
        scans.append(
            {
                "subject_id": subj,
                "scan_date": date,
                "pet_path": fpath,
                "ct_path": ct_p if ct_found else None,
                "seg_path": seg_p if seg_found else None,
            }
        )
    _log("info", f"detect_scans: {len(scans)} scan(s)")
    return scans


def detect_segmentations(
    root_folder: str,
    scans: list[dict],
    log=None,
) -> dict[str, Any]:
    """
    Collect .nii.gz stems across Seg folders.

    Returns stem → {count, total} plus "__total__".
    """
    def _log(level, msg):
        if log:
            log(level, msg)

    stem_counts: dict[str, int] = defaultdict(int)
    total = len(scans)
    for scan in scans:
        seg_path = scan.get("seg_path")
        if not seg_path or not os.path.isdir(seg_path):
            continue
        for fname in os.listdir(seg_path):
            if fname.endswith(".nii.gz"):
                stem_counts[fname[: -len(".nii.gz")]] += 1

    result = {stem: {"count": cnt, "total": total} for stem, cnt in stem_counts.items()}
    result["__total__"] = total
    _log("info", f"detect_segmentations: {len(result) - 1} unique stem(s)")
    return result


def collect_ct_scans(root: Path | str) -> list[Path]:
    """Find CT .nii.gz files under folders containing 'CT' in the path."""
    root = Path(root)
    return [p for p in root.rglob("*.nii.gz") if "CT" in p.parts]


def collect_ct_folders(root: Path | str) -> list[Path]:
    """Collect immediate CT scan folders that contain any files (DICOM etc.)."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"CT root not found: {root}")
    ct_dirs: list[Path] = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        if any(fp.is_file() for fp in folder.rglob("*")):
            ct_dirs.append(folder)
    return ct_dirs


def choose_ct_file(ct_files: list[Path], index: int) -> Path:
    if not ct_files:
        raise FileNotFoundError("No CT .nii.gz files found under dataset root.")
    if index < 0 or index >= len(ct_files):
        raise IndexError(f"CT index {index} out of range (0..{len(ct_files) - 1})")
    return ct_files[index]


def map_ct_to_seg_folder(ct_folder: Path | str, seg_root: Path | str) -> Path:
    """Map MSP0001_2025-07-09_CT → MSP0001_2025-07-09_Seg under seg_root."""
    from lib.segmentation.spleen import map_ct_to_seg_folder as _map

    return _map(ct_folder, seg_root)


def discover_patients(root: str) -> list[dict[str, Any]]:
    """
    Walk dataset root with CT/ PET/ Segments/ layout.
    Returns list of {subject_id, scan_date, ct_path, pet_path, seg_path, ct_nii?}.
    """
    scans = []
    # Prefer PET-driven discovery (same as detect_scans) but tolerate missing PET
    ct_root = os.path.join(root, "CT")
    pet_root = os.path.join(root, "PET")
    seg_root = os.path.join(root, "Segments")
    source = pet_root if os.path.isdir(pet_root) else ct_root
    if not os.path.isdir(source):
        raise FileNotFoundError(f"No PET/ or CT/ under {root}")

    for folder in sorted(os.listdir(source)):
        fpath = os.path.join(source, folder)
        if not os.path.isdir(fpath):
            continue
        parsed = parse_folder_name(folder)
        if not parsed:
            continue
        subj, date, _ = parsed
        ct_p = os.path.join(ct_root, f"{subj}_{date}_CT")
        pet_p = os.path.join(pet_root, f"{subj}_{date}_PET")
        seg_p = os.path.join(seg_root, f"{subj}_{date}_Seg")
        ct_nii = os.path.join(root, "CT_NIfTI", f"{subj}_{date}_CT.nii.gz")
        if not os.path.isfile(ct_nii):
            # also accept a single .nii.gz inside CT folder
            if os.path.isdir(ct_p):
                hits = list(Path(ct_p).glob("*.nii.gz"))
                ct_nii = str(hits[0]) if hits else ct_nii
        scans.append(
            {
                "subject_id": subj,
                "scan_date": date,
                "ct_path": ct_p if os.path.isdir(ct_p) else None,
                "pet_path": pet_p if os.path.isdir(pet_p) else None,
                "seg_path": seg_p,
                "ct_nii": ct_nii if os.path.isfile(ct_nii) else None,
            }
        )
    return scans


def find_bulk_subjects(dataset_root: str) -> list[dict[str, Any]]:
    """Subjects with Segments/<ID>_Seg folders."""
    seg_root = os.path.join(dataset_root, "Segments")
    if not os.path.isdir(seg_root):
        return []
    out = []
    for d in sorted(os.listdir(seg_root)):
        if d.endswith("_Seg") and os.path.isdir(os.path.join(seg_root, d)):
            key = d[: -len("_Seg")]
            out.append({"subject_id": key, "seg_dir": os.path.join(seg_root, d)})
    return out
