"""
postprocessing.py
=================
Python-IDE / CLI entry for KU organ post-processing.

    cd extension_new
    python scripts/postprocessing.py --root "E:\\KUPETCTMS\\temp sample" --limit 1
    python scripts/postprocessing.py --root "E:\\KUPETCTMS\\temp sample" --no-skip-done

Default KU protocol:
  1. PET ureter mask (SUV 2.5, dilate 18 mm, extend 50 mm below L5,
     fill holes + bridge gaps)
  2. Dilate abdomen/vessels/spine 5 mm → hard-subtract from targets
  3. Dilate ureter∪groups 13 mm → remove overlap voxels with PET >
     1.2 (visceral fat) or 1.6 (psoas)
  4. Visceral fat also clipped to L1–L5

Writes ``ureter_from_pet.nii.gz`` and ``*_processed.nii.gz``.
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
