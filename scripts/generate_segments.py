"""
generate_segments.py
====================
Python-IDE / CLI entry for segmentation generation (same lib as Slicer
PETCTSegmentationModule).

Inject a dataset root — no Slicer required::

    cd extension_new
    python scripts/generate_segments.py --root E:\\KUPETCTMS\\new_data_clean

What it runs (per patient under ROOT):
  1. CT DICOM → CT_NIfTI (if needed)
  2. TotalSegmentator (targets used by VF + organs/vessels)
  3. combined_mask.nii.gz
  4. visceral_fat.nii.gz (SegResNet; optional auto L/R·A/P fix)
  5. optional .seg.nrrd packaging

Slicer equivalent: PETCTSegmentationModule Steps 1–2 (Mirror QC stays human/UI).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_sibling(name: str):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(f"_kupet_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str] | None = None) -> int:
    return _load_sibling("run_visceralfat.py").main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
