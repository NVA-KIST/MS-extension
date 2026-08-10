"""
postprocessing.py
=================
Python-IDE / CLI entry for organ post-processing (same lib as Slicer
UreterPostProcess array cores).

Inject a dataset root::

    cd extension_new
    python scripts/postprocessing.py --root E:\\KUPETCTMS\\new_data_clean --list-organs
    python scripts/postprocessing.py --root E:\\KUPETCTMS\\new_data_clean --interactive

    python scripts/postprocessing.py --root E:\\KUPETCTMS\\new_data_clean ^
        --organ-mode "visceral_fat.nii.gz:Clip + Clean,spleen.nii.gz:Clip only"

Writes per Segments/<ID>_Seg/:
  ureter_from_pet.nii.gz
  <stem>_processed.nii.gz
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
    return _load_sibling("run_postprocessing.py").main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
