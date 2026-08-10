"""Iliopsoas (psoas) TotalSegmentator ROI helper."""
from __future__ import annotations

from pathlib import Path

from lib.segmentation.totalseg import run_totalsegmentator_api


def run_iliopsoas_segmentation(
    ct_path: Path | str,
    output_dir: Path | str | None = None,
    *,
    device: str = "gpu",
) -> Path:
    """
    Segment iliopsoas_left / iliopsoas_right into ``output_dir``.

    If ``output_dir`` is None, writes next to the CT as
    ``segmentation_<stem>_iliopsoas/``.
    """
    ct_path = Path(ct_path)
    if output_dir is None:
        ct_stem = Path(ct_path.stem).stem  # strip .nii.gz
        output_dir = ct_path.parent / f"segmentation_{ct_stem}_iliopsoas"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[psoas] CT: {ct_path}")
    print(f"[psoas] out: {output_dir}")
    run_totalsegmentator_api(
        ct_path,
        output_dir,
        task="total",
        roi_subset=["iliopsoas_left", "iliopsoas_right"],
        device=device,
        nr_thr_resamp=1,
        nr_thr_saving=1,
    )
    return output_dir
