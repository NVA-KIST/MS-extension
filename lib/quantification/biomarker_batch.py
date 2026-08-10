"""
Pure batch Excel / derived-metric helpers from PETBiomarkerStudioLogic.

No Slicer imports. openpyxl is required for Excel I/O (ImportError if missing).
"""
from __future__ import annotations

import datetime
import math
import os
import re
from typing import Any, Dict, Iterable, Optional, Sequence, Set, Tuple

import numpy as np

from lib.quantification.radiomics import radiomics_config_signature

AUXILIARY_STEM_TOKENS = (
    "combined_seg", "totalseg", "total_seg", "body_trunc", "body_extremities",
    "body", "skin", "ct", "vertebra", "vertebrae", "kidney", "ureter",
    "urinary", "bladder", "spine",
)


def is_auxiliary_segment_stem(stem: str) -> bool:
    """Files that are usually masks/aux rather than target organs."""
    nl = stem.lower()
    return any(tok in nl for tok in AUXILIARY_STEM_TOKENS)


def default_excel_label(stem: str) -> str:
    label = stem
    for suffix in (".seg",):
        if label.lower().endswith(suffix):
            label = label[: -len(suffix)]
    return label


def parse_batch_base_name(base: str) -> Tuple[str, str]:
    """'MSP0001_2025-07-09' -> ('MSP0001', '2025-07-09'). Falls back to (base, '')."""
    m = re.match(r"(?P<sid>.+?)_(?P<date>\d{4}-\d{2}-\d{2})", base)
    if m:
        return m.group("sid"), m.group("date")
    return base, ""


def scan_batch_dataset(root: str) -> dict:
    """Discover subjects (Segments/<base>_Seg) and candidate segment stems."""
    if not root or not os.path.isdir(root):
        raise ValueError(f"Root folder not found: {root}")

    seg_root = os.path.join(root, "Segments")
    if not os.path.isdir(seg_root):
        raise ValueError(
            f"Expected a 'Segments' subfolder under: {root}\n"
            "Layout: <root>/PET, <root>/CT, <root>/Segments/<base>_Seg/"
        )

    subject_dirs = sorted(
        d for d in os.listdir(seg_root)
        if d.endswith("_Seg") and os.path.isdir(os.path.join(seg_root, d))
    )
    if not subject_dirs:
        raise ValueError(f"No '*_Seg' subject folders found in: {seg_root}")

    first_dir = os.path.join(seg_root, subject_dirs[0])
    segment_stems = []
    for f in sorted(os.listdir(first_dir)):
        stem = None
        if f.endswith(".seg.nrrd"):
            stem = f[: -len(".seg.nrrd")]
        elif f.endswith(".nii.gz"):
            stem = f[: -len(".nii.gz")]
        elif f.endswith(".nii"):
            stem = f[: -len(".nii")]
        elif f.endswith(".nrrd"):
            stem = f[: -len(".nrrd")]
        if stem is not None:
            segment_stems.append(stem)

    return {
        "subjectCount": len(subject_dirs),
        "subjectDirs": subject_dirs,
        "segmentStems": segment_stems,
    }


def find_batch_segment_file(seg_dir: str, stem: str) -> Optional[str]:
    for ext in (".seg.nrrd", ".nii.gz", ".nii", ".nrrd"):
        candidate = os.path.join(seg_dir, stem + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def existing_batch_keys(
    output_file: str,
    required_segments: Optional[Sequence[str]] = None,
    required_signature: Optional[str] = None,
) -> Set[Tuple[str, str]]:
    """Return subjects that are complete for this exact computation."""
    keys: Set[Tuple[str, str]] = set()
    if not output_file or not os.path.isfile(output_file):
        return keys
    try:
        import openpyxl
    except ModuleNotFoundError:
        return keys
    try:
        wb = openpyxl.load_workbook(output_file, read_only=True)
    except Exception:
        return keys
    if "Data" not in wb.sheetnames:
        return keys

    requested = {
        str(segment).strip() for segment in (required_segments or [])
        if str(segment).strip()
    }
    completed: Dict[Tuple[str, str], Set[str]] = {}
    ws = wb["Data"]
    header = None
    for row in ws.iter_rows(values_only=True):
        if header is None:
            header = [str(c).strip() if c is not None else "" for c in row]
            continue
        rec = dict(zip(header, row))
        if str(rec.get("status", "")).strip().lower() != "done":
            continue
        if required_signature is not None:
            found_signature = str(
                rec.get("computation_signature", "")
            ).strip()
            if found_signature != required_signature:
                continue
        key = (
            str(rec.get("subject_id", "")).strip(),
            str(rec.get("scan_date", "")).strip(),
        )
        segment = str(rec.get("segment", "")).strip()
        completed.setdefault(key, set()).add(segment)

    for key, segments in completed.items():
        if not requested or requested.issubset(segments):
            keys.add(key)
    return keys


def _normalise_segment_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(label).lower())


def segment_role(label: str) -> str:
    """Infer only the roles needed for optional cross-ROI derived metrics."""
    raw = str(label).lower()
    compact = _normalise_segment_label(raw)
    tokens = set(re.findall(r"[a-z0-9]+", raw))

    if any(token in compact for token in (
        "bloodpool", "blood", "femoralvessel", "femoralvein",
        "femoralartery", "aorta", "vessel"
    )):
        return "blood_pool"

    if "psoas" in compact:
        is_left = (
            bool(tokens.intersection({"left", "lt", "l"}))
            or compact.endswith("left")
            or compact.endswith("lt")
            or compact.endswith("l")
        )
        is_right = (
            bool(tokens.intersection({"right", "rt", "r"}))
            or compact.endswith("right")
            or compact.endswith("rt")
            or compact.endswith("r")
        )
        if is_left and not is_right:
            return "psoas_left"
        if is_right and not is_left:
            return "psoas_right"

    return "other"


def safe_asymmetry(left: Any, right: Any) -> Optional[float]:
    if left in (None, "") or right in (None, ""):
        return None
    try:
        left_f = float(left)
        right_f = float(right)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(left_f) or not math.isfinite(right_f):
        return None
    denominator = (abs(left_f) + abs(right_f)) / 2.0
    if denominator <= 1e-12:
        return None
    return abs(left_f - right_f) / denominator


def cross_roi_derived_by_subject(all_data: Sequence[dict]) -> dict:
    """Return Summary-only features that require two or more ROI rows."""
    grouped: Dict[tuple, list] = {}
    for rec in all_data:
        if str(rec.get("status", "")).strip().lower() != "done":
            continue
        key = (
            rec.get("subject_id", ""),
            rec.get("patient_id", ""),
            rec.get("scan_date", ""),
        )
        grouped.setdefault(key, []).append(rec)

    output = {}
    for key, records in grouped.items():
        derived = {}

        blood_medians = []
        for rec in records:
            if segment_role(rec.get("segment", "")) != "blood_pool":
                continue
            value = rec.get("rad_firstorder_Median")
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and abs(value) > 1e-12:
                blood_medians.append(value)

        if blood_medians:
            blood_median = float(np.mean(blood_medians))
            for rec in records:
                if segment_role(rec.get("segment", "")) == "blood_pool":
                    continue
                p90 = rec.get("rad_firstorder_90Percentile")
                try:
                    p90 = float(p90)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(p90):
                    continue
                segment = str(rec.get("segment", "")).strip()
                if segment:
                    derived[
                        f"{segment}_derived_P90ToBloodPoolMedian"
                    ] = p90 / blood_median

        psoas = {}
        for rec in records:
            role = segment_role(rec.get("segment", ""))
            if role in ("psoas_left", "psoas_right"):
                psoas[role] = rec

        left = psoas.get("psoas_left")
        right = psoas.get("psoas_right")
        if left is not None and right is not None:
            feature_map = {
                "P90": "rad_firstorder_90Percentile",
                "Entropy": "rad_firstorder_Entropy",
                "SAHGLE": "rad_glszm_SmallAreaHighGrayLevelEmphasis",
            }
            for short_name, feature_name in feature_map.items():
                value = safe_asymmetry(
                    left.get(feature_name), right.get(feature_name)
                )
                if value is not None:
                    derived[
                        f"derived_Psoas_{short_name}_Asymmetry"
                    ] = value

        output[key] = derived

    return output


def batch_error_row(
    subject_id: str,
    scan_date: str,
    segment: str,
    status: str,
    stem: str,
) -> dict:
    return {
        "subject_id": subject_id,
        "patient_id": "",
        "scan_date": scan_date,
        "segment": segment,
        "source_file": stem,
        "status": status,
    }


def computation_signature(
    metrics_options: Optional[dict],
    radiomics_options: Optional[dict],
) -> str:
    opts = metrics_options or {}
    metrics = ",".join(
        key for key in ("mean", "max", "peak", "tlg", "volume")
        if opts.get(key, False)
    ) or "none"
    return f"metrics={metrics}|{radiomics_config_signature(radiomics_options)}"


def parse_quantitative_indices_results(
    name_value_pairs: Iterable[Tuple[Any, Any]],
) -> dict:
    """Parse QuantitativeIndicesCLI-style (rawName, rawValue) pairs."""
    name_map = {
        "Mean": "suv_mean",
        "Max": "suv_max",
        "Peak": "suv_peak",
        "TLG": "tlg",
        "Volume": "volume_mL",
    }

    results = {}
    for raw_name, raw_value in name_value_pairs:
        if raw_value in (None, "", "--"):
            continue

        clean_name = str(raw_name).replace("_s", "").replace("_", " ").strip()
        if clean_name not in name_map:
            continue

        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue

        if math.isnan(value):
            continue

        results[name_map[clean_name]] = value

    return results


def save_batch_rows_to_excel(
    rows: Sequence[dict],
    output_file: str,
    append: bool = True,
) -> str:
    """Write Data + rebuilt Summary sheets (schema-merged on append)."""
    if not rows:
        raise ValueError("No batch rows to save.")
    output_file = os.path.abspath(output_file)
    if not output_file.lower().endswith(".xlsx"):
        output_file += ".xlsx"

    try:
        import openpyxl
    except ModuleNotFoundError as e:
        raise ImportError(
            "openpyxl is required to save batch Excel files. "
            "Install with: pip install openpyxl"
        ) from e

    base_cols = [
        "subject_id", "patient_id", "scan_date", "segment", "source_file",
        "volume_mL", "suv_mean", "suv_max", "suv_peak", "tlg",
        "radiomics_status", "computation_signature", "status", "computed_at",
    ]
    extra_cols = []
    for row in rows:
        for key in row.keys():
            if key not in base_cols and key not in extra_cols:
                extra_cols.append(key)
    current_cols = base_cols[:-2] + sorted(extra_cols) + base_cols[-2:]

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for row in rows:
        row.setdefault("computed_at", now)

    if append and os.path.isfile(output_file):
        wb = openpyxl.load_workbook(output_file)
        for stale in ("Sheet", "Sheet1", "Results"):
            if stale in wb.sheetnames:
                del wb[stale]
    else:
        wb = openpyxl.Workbook()

    if "Data" in wb.sheetnames:
        ws = wb["Data"]
        existing_header = [
            ws.cell(row=1, column=c).value for c in range(1, ws.max_column + 1)
        ]
        existing_header = [
            str(h).strip() for h in existing_header
            if h is not None and str(h).strip() != ""
        ]
        data_cols = list(dict.fromkeys(existing_header + current_cols))
    else:
        if wb.active and wb.active.title in ("Sheet", "Results"):
            ws = wb.active
            ws.title = "Data"
        else:
            ws = wb.create_sheet("Data", 0)
        data_cols = list(dict.fromkeys(current_cols))

    for ci, col in enumerate(data_cols, 1):
        ws.cell(row=1, column=ci, value=col)
    for row in rows:
        ws.append([row.get(c, "") for c in data_cols])

    header = [ws.cell(row=1, column=c).value for c in range(1, len(data_cols) + 1)]
    all_data = []
    for r in range(2, ws.max_row + 1):
        vals = [ws.cell(row=r, column=c).value for c in range(1, len(data_cols) + 1)]
        if any(v is not None and v != "" for v in vals):
            all_data.append(dict(zip(header, vals)))

    def _is_done(rec):
        return str(rec.get("status", "")).strip().lower() == "done"

    metric_cols = [
        c for c in data_cols
        if c in ("volume_mL", "suv_mean", "suv_max", "suv_peak", "tlg")
        or str(c).startswith("rad_")
    ]
    seen_segs = list(dict.fromkeys(
        str(r.get("segment", "")).strip()
        for r in all_data if _is_done(r) and str(r.get("segment", "")).strip()
    ))

    pivot = {}
    for r in all_data:
        if not _is_done(r):
            continue
        key = (r.get("subject_id", ""), r.get("patient_id", ""), r.get("scan_date", ""))
        pivot.setdefault(key, {})
        seg = str(r.get("segment", "")).strip()
        for m in metric_cols:
            val = r.get(m)
            if val not in (None, ""):
                pivot[key][f"{seg}_{m}"] = val

    cross_roi_derived = cross_roi_derived_by_subject(all_data)
    for key, values in cross_roi_derived.items():
        pivot.setdefault(key, {}).update(values)

    id_cols = ["subject_id", "patient_id", "scan_date"]
    metric_order = [f"{seg}_{m}" for seg in seen_segs for m in metric_cols]
    derived_order = sorted({
        col
        for values in pivot.values()
        for col in values.keys()
        if col.startswith("derived_") or "_derived_" in col
    })
    metric_order.extend(
        col for col in derived_order if col not in metric_order
    )
    sum_cols = id_cols + metric_order

    if "Summary" in wb.sheetnames:
        del wb["Summary"]
    ws_s = wb.create_sheet("Summary")
    for ci, col in enumerate(sum_cols, 1):
        ws_s.cell(row=1, column=ci, value=col)
    for ri, (subj, pid, date) in enumerate(sorted(pivot), start=2):
        ws_s.cell(ri, 1, subj)
        ws_s.cell(ri, 2, pid)
        ws_s.cell(ri, 3, date)
        data = pivot[(subj, pid, date)]
        for ci, col in enumerate(metric_order, start=4):
            ws_s.cell(ri, ci, data.get(col, ""))

    wb.save(output_file)
    return output_file


def save_qc_rows_to_excel(
    qc_rows: Sequence[dict],
    output_file: str,
    baseline_suv_max: Optional[float] = None,
) -> str:
    """Write the QC table to a 'QC' sheet (append/overwrite that one sheet only)."""
    if not qc_rows:
        raise ValueError("No QC rows to save.")
    if not output_file:
        raise ValueError("No output Excel file selected.")

    output_file = os.path.abspath(output_file)
    if not output_file.lower().endswith(".xlsx"):
        output_file += ".xlsx"

    try:
        import openpyxl
    except ModuleNotFoundError as e:
        raise ImportError(
            "openpyxl is required to save QC Excel files. "
            "Install with: pip install openpyxl"
        ) from e

    if os.path.exists(output_file):
        wb = openpyxl.load_workbook(output_file)
    else:
        wb = openpyxl.Workbook()
        default = wb.active
        if default is not None and default.title == "Sheet" and default.max_row == 1:
            wb.remove(default)

    if "QC" in wb.sheetnames:
        del wb["QC"]
    ws = wb.create_sheet("QC")

    columns = [
        "roi_node", "segment", "suv_mean", "suv_max", "suv_peak",
        "max_over_mean", "baseline_suv_max", "delta_suv_max_pct",
        "ras_r", "ras_a", "ras_s", "flag", "computed_at",
    ]
    for ci, col in enumerate(columns, start=1):
        ws.cell(row=1, column=ci, value=col)

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for ri, row in enumerate(qc_rows, start=2):
        ras = row.get("rasHotspot") or [None, None, None]
        ws.cell(ri, 1, row.get("roiNodeName", ""))
        ws.cell(ri, 2, row.get("segmentName", ""))
        ws.cell(ri, 3, row.get("suv_mean"))
        ws.cell(ri, 4, row.get("suv_max"))
        ws.cell(ri, 5, row.get("suv_peak"))
        ws.cell(ri, 6, row.get("ratio"))
        ws.cell(ri, 7, baseline_suv_max)
        ws.cell(ri, 8, row.get("deltaMaxPct"))
        ws.cell(ri, 9, ras[0])
        ws.cell(ri, 10, ras[1])
        ws.cell(ri, 11, ras[2])
        ws.cell(ri, 12, row.get("flag", ""))
        ws.cell(ri, 13, now)

    wb.save(output_file)
    return output_file
