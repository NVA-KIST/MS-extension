"""Spleen TotalSegmentator ROI helper."""
from __future__ import annotations

from pathlib import Path

from lib.segmentation.totalseg import run_totalsegmentator_api


def map_ct_to_seg_folder(ct_folder: Path | str, seg_root: Path | str) -> Path:
    """Map MSP0001_2025-07-09_CT → MSP0001_2025-07-09_Seg under seg_root."""
    ct_folder = Path(ct_folder)
    seg_root = Path(seg_root)
    name = ct_folder.name
    seg_name = f"{name[:-3]}_Seg" if name.endswith("_CT") else f"{name}_Seg"
    return seg_root / seg_name


def run_spleen_segmentation(
    ct_input: Path | str,
    output_dir: Path | str | None = None,
    *,
    seg_root: Path | str | None = None,
    device: str = "gpu",
) -> Path:
    """
    Segment spleen from a CT NIfTI or DICOM folder.

    If ``output_dir`` is None and ``seg_root`` is set, uses map_ct_to_seg_folder.
    """
    ct_input = Path(ct_input)
    if output_dir is None:
        if seg_root is None:
            raise ValueError("Provide output_dir or seg_root")
        output_dir = map_ct_to_seg_folder(ct_input, seg_root)
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[spleen] CT: {ct_input}")
    print(f"[spleen] out: {output_dir}")
    run_totalsegmentator_api(
        ct_input,
        output_dir,
        task="total",
        roi_subset=["spleen"],
        device=device,
        nr_thr_resamp=1,
        nr_thr_saving=1,
    )
    return output_dir
