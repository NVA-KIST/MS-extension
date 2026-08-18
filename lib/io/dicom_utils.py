"""Shared DICOM header helpers (no pixels, no Slicer)."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Optional

_DCM_EXTS = {".dcm", ".ima", ".dicom"}
_SKIP_NAMES = {"metacache.mim", "dicomdir"}


def ds_get(ds, tag: str, default: Any = "") -> Any:
    val = getattr(ds, tag, default)
    if val is None or val == "":
        return default
    return val


def to_float(x) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def parse_age_years(age_str) -> Optional[float]:
    """DICOM PatientAge ``064Y`` / ``011M`` → years (float)."""
    if not age_str:
        return None
    m = re.match(r"(\d+)\s*([YMWD]?)", str(age_str).strip(), re.I)
    if not m:
        return None
    n = int(m.group(1))
    unit = (m.group(2) or "Y").upper()
    return {"Y": float(n), "M": n / 12.0, "W": n / 52.0, "D": n / 365.25}.get(unit, float(n))


def fmt_study_date(yyyymmdd: str) -> str:
    s = re.sub(r"\D", "", str(yyyymmdd or ""))
    if len(s) >= 8:
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return str(yyyymmdd or "")


def looks_like_dicom(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.name.lower() in _SKIP_NAMES:
        return False
    if path.suffix.lower() in _DCM_EXTS or path.suffix == "":
        return True
    return False


def iter_dicom_files(folder: Path, *, max_files: int = 0):
    """Yield candidate DICOM files under *folder* (files first, then recurse)."""
    folder = Path(folder)
    n = 0
    if folder.is_file():
        yield folder
        return
    if not folder.is_dir():
        return
    for p in sorted(folder.iterdir()):
        if p.is_file() and looks_like_dicom(p):
            yield p
            n += 1
            if max_files and n >= max_files:
                return
    for p in sorted(folder.rglob("*")):
        if p.is_file() and looks_like_dicom(p):
            yield p
            n += 1
            if max_files and n >= max_files:
                return


def read_dicom_header(path: str | Path):
    """Read one file header (stop_before_pixels). Returns dataset or None."""
    import pydicom

    path = Path(path)
    try:
        return pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except Exception:
        return None


def first_dicom_header(folder: str | Path):
    """First readable DICOM header in a folder (or the file itself)."""
    for fp in iter_dicom_files(Path(folder), max_files=80):
        ds = read_dicom_header(fp)
        if ds is not None and (ds_get(ds, "SOPInstanceUID") or ds_get(ds, "Modality")):
            return ds, fp
    return None, None


def folder_has_dicom(folder: Path) -> bool:
    ds, _ = first_dicom_header(folder)
    return ds is not None


def count_files(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for p in folder.iterdir() if p.is_file())


def normalize_modality(raw: str) -> Optional[str]:
    m = str(raw or "").strip().upper()
    if m in ("CT",):
        return "CT"
    if m in ("PT", "PET"):
        return "PET"
    return None


def radiopharma_fields(ds) -> dict[str, Any]:
    out: dict[str, Any] = {
        "radiopharmaceutical": "",
        "injected_dose_MBq": None,
        "radionuclide_half_life_s": None,
        "radiopharm_start_time": "",
        "pet_units": str(ds_get(ds, "Units")),
        "decay_correction": str(ds_get(ds, "DecayCorrection")),
    }
    try:
        seq = ds[0x0054, 0x0016].value[0]
    except (KeyError, IndexError, TypeError):
        seq = None
    if seq is None:
        return out
    out["radiopharmaceutical"] = str(getattr(seq, "Radiopharmaceutical", "") or "")
    dose_bq = to_float(getattr(seq, "RadionuclideTotalDose", None))
    out["injected_dose_MBq"] = round(dose_bq / 1e6, 3) if dose_bq else None
    out["radionuclide_half_life_s"] = to_float(getattr(seq, "RadionuclideHalfLife", None))
    start = getattr(seq, "RadiopharmaceuticalStartDateTime", None) or getattr(
        seq, "RadiopharmaceuticalStartTime", None
    )
    out["radiopharm_start_time"] = str(start or "").split(".")[0]
    return out
