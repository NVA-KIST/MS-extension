"""
Fast Slicer-compatible .seg.nrrd packaging via SimpleITK (+ optional pynrrd).

Avoids slow Slicer MRML round-trips for building multi-segment files.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# TotalSegmentator stems used for grouped exports
VESSEL_STEMS = [
    "aorta",
    "iliac_artery_left",
    "iliac_artery_right",
    "iliac_vena_left",
    "iliac_vena_right",
    "inferior_vena_cava",
    "portal_vein_and_splenic_vein",
]

SPINE_STEMS = [
    "vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4", "vertebrae_L5",
    "vertebrae_T1", "vertebrae_T2", "vertebrae_T3", "vertebrae_T4", "vertebrae_T5",
    "vertebrae_T6", "vertebrae_T7", "vertebrae_T8", "vertebrae_T9",
    "vertebrae_T10", "vertebrae_T11", "vertebrae_T12",
    "sacrum",
]

# Display names inside spine.seg.nrrd (individual selectable segments)
SPINE_SEGMENT_NAMES = {
    "vertebrae_L1": "L1",
    "vertebrae_L2": "L2",
    "vertebrae_L3": "L3",
    "vertebrae_L4": "L4",
    "vertebrae_L5": "L5",
    "vertebrae_T1": "T1",
    "vertebrae_T2": "T2",
    "vertebrae_T3": "T3",
    "vertebrae_T4": "T4",
    "vertebrae_T5": "T5",
    "vertebrae_T6": "T6",
    "vertebrae_T7": "T7",
    "vertebrae_T8": "T8",
    "vertebrae_T9": "T9",
    "vertebrae_T10": "T10",
    "vertebrae_T11": "T11",
    "vertebrae_T12": "T12",
    "sacrum": "sacrum",
}

ABDOMEN_STEMS = [
    "liver",
    "spleen",
    "pancreas",
    "kidney_right",
    "kidney_left",
    "gallbladder",
    "adrenal_gland_right",
    "adrenal_gland_left",
    # Hollow / GI
    "urinary_bladder",
    "small_bowel",
    "colon",
    "duodenum",
    "stomach",
]

# Kept as individual .nii.gz only (NOT packaged into a .seg.nrrd group)
TARGET_ORGANS = [
    ("iliopsoas_left", "psoas_left"),
    ("iliopsoas_right", "psoas_right"),
    ("spleen", "spleen"),
    ("visceral_fat", "visceral_fat"),
]

# Loose NIfTIs that are only intermediates for groups / combined_mask.
# After packaging they are deleted (targets + combined_mask.nii.gz are kept).
LOOSE_ANATOMY_STEMS = list(dict.fromkeys(
    [
        *VESSEL_STEMS,
        *SPINE_STEMS,
        *ABDOMEN_STEMS,
        "liver",
        "heart",
        "body_trunc",
        "body",
        "body_extremities",
        "skin",
        "torso_fat",
        "subcutaneous_fat",
        "skeletal_muscle",
        *[f"rib_left_{i}" for i in range(1, 13)],
        *[f"rib_right_{i}" for i in range(1, 13)],
    ]
))

KEEP_AFTER_CLEANUP_STEMS = {
    "visceral_fat",
    "spleen",
    "iliopsoas_left",
    "iliopsoas_right",
    "combined_mask",
    "ureter_from_pet",
}

# combined_mask flat labels → segment names
COMBINED_LABEL_NAMES = {
    1: "body_trunc",
    2: "torso_fat",
    3: "liver",
    4: "heart",
    5: "vertebrae_L",
    6: "ribs",
    7: "vertebrae_T",
}

_COLORS = [
    (0.90, 0.20, 0.20),
    (0.20, 0.70, 0.95),
    (0.95, 0.75, 0.15),
    (0.25, 0.90, 0.40),
    (0.95, 0.50, 0.15),
    (0.70, 0.25, 0.95),
    (0.95, 0.40, 0.75),
    (0.30, 0.85, 0.85),
]


@dataclass
class SegmentSpec:
    name: str
    array_xyz: np.ndarray  # binary/uint8 in nibabel (X,Y,Z) order
    label_value: int
    color: Tuple[float, float, float] = (0.5, 0.5, 0.5)


@dataclass
class SegNrrdPackage:
    """One .seg.nrrd multi-label file."""
    name: str
    segments: List[SegmentSpec] = field(default_factory=list)


def _sitk_from_xyz(arr_xyz: np.ndarray, ref_sitk):
    """nibabel (X,Y,Z) → SimpleITK image matching ref geometry."""
    import SimpleITK as sitk

    arr_zyx = np.transpose(np.asarray(arr_xyz), (2, 1, 0))
    img = sitk.GetImageFromArray(arr_zyx.astype(np.uint8))
    img.CopyInformation(ref_sitk)
    return img


def load_ref_sitk(path: str):
    import SimpleITK as sitk
    return sitk.ReadImage(path)


def load_mask_xyz(path: str, ref_sitk=None) -> np.ndarray:
    """Load mask as nibabel-order (X,Y,Z) uint8, optionally resampled to ref."""
    import SimpleITK as sitk

    img = sitk.ReadImage(path)
    if ref_sitk is not None:
        img = sitk.Resample(
            img,
            ref_sitk,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8,
        )
    arr_zyx = sitk.GetArrayFromImage(img).astype(np.uint8)
    return np.transpose(arr_zyx, (2, 1, 0))


def multilabel_from_segments(segments: Sequence[SegmentSpec]) -> np.ndarray:
    if not segments:
        raise ValueError("No segments")
    shape = segments[0].array_xyz.shape
    out = np.zeros(shape, dtype=np.uint8)
    for seg in segments:
        if seg.array_xyz.shape != shape:
            raise ValueError(f"{seg.name} shape {seg.array_xyz.shape} != {shape}")
        out[seg.array_xyz > 0] = np.uint8(seg.label_value)
    return out


def write_seg_nrrd(
    out_path: str,
    segments: Sequence[SegmentSpec],
    ref_sitk,
) -> str:
    """
    Write a Slicer-compatible multi-label .seg.nrrd (single layer).

    Uses SimpleITK for fast IO; injects Segment_* header fields Slicer expects.
    """
    import SimpleITK as sitk

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    label_xyz = multilabel_from_segments(segments)
    label_img = _sitk_from_xyz(label_xyz, ref_sitk)

    # Extent string for Slicer
    size = label_img.GetSize()  # (X,Y,Z)
    extent = f"0 {size[0]-1} 0 {size[1]-1} 0 {size[2]-1}"

    for i, seg in enumerate(segments):
        label_img.SetMetaData(f"Segment{i}_ID", f"Segment_{i+1}")
        label_img.SetMetaData(f"Segment{i}_Name", seg.name)
        label_img.SetMetaData(
            f"Segment{i}_Color",
            f"{seg.color[0]:.6f} {seg.color[1]:.6f} {seg.color[2]:.6f}",
        )
        label_img.SetMetaData(f"Segment{i}_LabelValue", str(int(seg.label_value)))
        label_img.SetMetaData(f"Segment{i}_Layer", "0")
        label_img.SetMetaData(f"Segment{i}_Extent", extent)
        label_img.SetMetaData(f"Segment{i}_ColorAutoGenerated", "0")
        label_img.SetMetaData(f"Segment{i}_NameAutoGenerated", "0")

    label_img.SetMetaData("Segmentation_MasterRepresentation", "Binary labelmap")
    label_img.SetMetaData("Segmentation_ContainedRepresentationNames", "Binary labelmap")

    sitk.WriteImage(label_img, out_path)
    return out_path


def _pick_existing(seg_dir: str, stems: Iterable[str]) -> List[Tuple[str, str]]:
    found = []
    for stem in stems:
        for ext in (".nii.gz", ".nii", ".nrrd"):
            fp = os.path.join(seg_dir, stem + ext)
            if os.path.isfile(fp):
                found.append((stem, fp))
                break
    return found


def package_patient_segmentations(
    seg_dir: str,
    ref_ct: str,
    out_dir: Optional[str] = None,
    *,
    include_combined: bool = True,
    include_targets: bool = False,
    include_vessels: bool = True,
    include_spine: bool = True,
    include_abdomen: bool = True,
    cleanup_loose: bool = True,
) -> Dict[str, str]:
    """
    Build the .seg.nrrd set for one patient Segments folder.

    Outputs (by default into ``seg_dir``):
      - combined_mask.seg.nrrd   (subsegments from combined label map)
      - vessels.seg.nrrd
      - spine.seg.nrrd           (individual L1..L5, T1..T12, sacrum)
      - abdomen.seg.nrrd

    Target organs (visceral_fat, spleen, iliopsoas L/R) stay as individual
    ``.nii.gz`` files and are NOT bundled into a group (``include_targets``
    defaults to False).

    When ``cleanup_loose`` is True (default), intermediate anatomy / body /
    tissue NIfTIs that were packed into groups are deleted afterward.
    """
    import SimpleITK as sitk

    out_dir = out_dir or seg_dir
    os.makedirs(out_dir, exist_ok=True)
    ref = sitk.ReadImage(ref_ct)
    written: Dict[str, str] = {}

    def _pack(name: str, items: List[Tuple[str, np.ndarray]]) -> Optional[str]:
        if not items:
            return None
        segs = []
        for i, (seg_name, arr) in enumerate(items, start=1):
            color = _COLORS[(i - 1) % len(_COLORS)]
            segs.append(
                SegmentSpec(
                    name=seg_name,
                    array_xyz=(arr > 0).astype(np.uint8),
                    label_value=i,
                    color=color,
                )
            )
        out_path = os.path.join(out_dir, f"{name}.seg.nrrd")
        write_seg_nrrd(out_path, segs, ref)
        return out_path

    # ── Target organs: optional only (default off — keep as flat NIfTI) ──────
    if include_targets:
        items = []
        for file_stem, seg_name in TARGET_ORGANS:
            hits = _pick_existing(seg_dir, [file_stem])
            if hits:
                items.append((seg_name, load_mask_xyz(hits[0][1], ref)))
        path = _pack("target_organs", items)
        if path:
            written["target_organs"] = path

    # ── Combined mask subsegments ────────────────────────────────────────────
    if include_combined:
        comb = None
        for cand in ("combined_mask.nii.gz", "combined_seg.nii.gz", "combined_mask.nrrd"):
            fp = os.path.join(seg_dir, cand)
            if os.path.isfile(fp):
                comb = load_mask_xyz(fp, ref)
                break
        if comb is not None:
            items = []
            for lab, name in COMBINED_LABEL_NAMES.items():
                m = (comb == lab).astype(np.uint8)
                if m.any():
                    items.append((name, m))
            path = _pack("combined_mask", items)
            if path:
                written["combined_mask"] = path

    # ── Vessel / spine / abdomen groups ──────────────────────────────────────
    if include_vessels:
        items = [
            (stem, load_mask_xyz(fp, ref))
            for stem, fp in _pick_existing(seg_dir, VESSEL_STEMS)
        ]
        path = _pack("vessels", items)
        if path:
            written["vessels"] = path

    if include_spine:
        # One selectable segment per vertebra (L1, L2, …) — never merged.
        items = []
        for stem, fp in _pick_existing(seg_dir, SPINE_STEMS):
            seg_name = SPINE_SEGMENT_NAMES.get(stem, stem)
            items.append((seg_name, load_mask_xyz(fp, ref)))
        path = _pack("spine", items)
        if path:
            written["spine"] = path

    if include_abdomen:
        items = [
            (stem, load_mask_xyz(fp, ref))
            for stem, fp in _pick_existing(seg_dir, ABDOMEN_STEMS)
        ]
        path = _pack("abdomen", items)
        if path:
            written["abdomen"] = path

    if cleanup_loose:
        removed = cleanup_loose_anatomy_files(seg_dir)
        if removed:
            written["cleaned_loose_files"] = str(len(removed))

    return written


def cleanup_loose_anatomy_files(seg_dir: str) -> List[str]:
    """
    Delete intermediate single-organ NIfTIs after .seg.nrrd packaging.

    Keeps target organs, combined_mask, ureter, any ``*_processed``, and all
    ``.seg.nrrd`` group files.
    """
    removed: List[str] = []
    for stem in LOOSE_ANATOMY_STEMS:
        if stem in KEEP_AFTER_CLEANUP_STEMS:
            continue
        for ext in (".nii.gz", ".nii"):
            fp = os.path.join(seg_dir, stem + ext)
            if os.path.isfile(fp):
                try:
                    os.remove(fp)
                    removed.append(os.path.basename(fp))
                except OSError:
                    pass
                break

    if removed:
        print(f"[seg.nrrd] cleaned {len(removed)} loose intermediate NIfTI(s)")

    # Stale group from older runs (targets stay as individual NIfTIs)
    stale_group = os.path.join(seg_dir, "target_organs.seg.nrrd")
    if os.path.isfile(stale_group):
        try:
            os.remove(stale_group)
            removed.append("target_organs.seg.nrrd")
            print("[seg.nrrd] removed stale target_organs.seg.nrrd")
        except OSError:
            pass

    return removed
