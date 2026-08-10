"""One-shot: port PETCTQuantAnalysis Logic/Widget into extension_new."""
from __future__ import annotations

import ast
import re
from pathlib import Path

SRC = Path(r"e:\KUPETCTMS\extension\PETCTQuantAnalysis_v2.py")
OUT = Path(r"e:\KUPETCTMS\extension\extension_new\slicer_modules\PETCTQuantAnalysis")

src = SRC.read_text(encoding="utf-8")
tree = ast.parse(src)
lines = src.splitlines(keepends=True)


def extract(name: str) -> str:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return "".join(lines[node.lineno - 1 : node.end_lineno])
    raise SystemExit(f"missing {name}")


for node in tree.body:
    if isinstance(node, ast.ClassDef) and node.name == "PETCTQuantAnalysis_v2":
        preamble = "".join(lines[: node.lineno - 1])
        break
else:
    raise SystemExit("module class not found")

module_cls = extract("PETCTQuantAnalysis_v2")
widget_cls = extract("PETCTQuantAnalysis_v2Widget")
logic_cls = extract("PETCTQuantAnalysis_v2Logic")


def replace_method(class_src: str, method_name: str, new_body: str) -> str:
    pattern = rf"(    def {re.escape(method_name)}\(.*?(?=\n    def |\nclass |\Z))"
    m = re.search(pattern, class_src, flags=re.S)
    if not m:
        raise SystemExit(f"method {method_name} not found")
    return class_src[: m.start()] + new_body.rstrip() + "\n\n" + class_src[m.end() :]


logic = logic_cls
logic = replace_method(
    logic,
    "_parseFolderName",
    "    def _parseFolderName(self, name):\n        return lib_parse_folder_name(name)\n",
)
logic = replace_method(
    logic,
    "detectScans",
    "    def detectScans(self, root_folder):\n"
    "        return lib_detect_scans(root_folder, log=self._log)\n",
)
logic = replace_method(
    logic,
    "detectSegmentations",
    "    def detectSegmentations(self, root_folder, scans):\n"
    "        return lib_detect_segmentations(root_folder, scans, log=self._log)\n",
)
logic = replace_method(
    logic,
    "_errorRow",
    "    def _errorRow(self, subject_id, scan_date, segment, status, patient_id=\"\"):\n"
    "        return lib_error_row(subject_id, scan_date, segment, status, patient_id)\n",
)
logic = replace_method(
    logic,
    "_saveExcel",
    "    def _saveExcel(self, rows, output_file, append):\n"
    "        lib_save_excel(rows, output_file, append=append, log=self._log)\n",
)

pattern = (
    r"        def _hhmmss_to_s\(t\):.*?slicer\.util\.updateVolumeFromArray\(pet_node, arr_suv\)"
)
replacement = """\
        suv_factor = compute_suvbw_factor(
            weight_kg=weight_kg,
            dose_bq=dose_bq,
            injection_time=inj_str,
            acquisition_time=acq_t_raw,
            half_life_s=half_life_s,
            decay_correction=decay_corr,
        )
        self._log("info",
            f"    _applySuvConversion: SUV factor = {suv_factor:.6f}")

        arr = slicer.util.arrayFromVolume(pet_node)
        max_before = float(arr.max())
        arr_suv    = (arr * suv_factor).astype(np.float32)
        slicer.util.updateVolumeFromArray(pet_node, arr_suv)"""
logic2, n = re.subn(pattern, replacement, logic, count=1, flags=re.S)
print("SUV core replace count:", n)
if n == 1:
    logic = logic2

logic_header = '''\
"""
PETCTQuantAnalysis_v2Logic — Slicer adapter.

lib: paths discovery, excel export, SUVbw factor math
Logic (Slicer): DICOM load, registration, QuantitativeIndicesCLI, volume update
"""
from __future__ import annotations

import os
import sys
import logging
import math
import vtk
import slicer
import numpy as np
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleLogic

LOG = logging.getLogger("PETCTQuant")

_EXT_NEW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _EXT_NEW_ROOT not in sys.path:
    sys.path.insert(0, _EXT_NEW_ROOT)

from lib.io.paths import detect_scans as lib_detect_scans
from lib.io.paths import detect_segmentations as lib_detect_segmentations
from lib.io.paths import parse_folder_name as lib_parse_folder_name
from lib.quantification.pet_metrics import (
    save_excel as lib_save_excel,
    error_row as lib_error_row,
    compute_suvbw_factor,
)

'''

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "PETCTQuantAnalysis_v2Logic.py").write_text(logic_header + "\n" + logic + "\n", encoding="utf-8")

widget_file = f'''\
"""
PETCTQuantAnalysis_v2 — Module + Widget.

  Widget → Logic (Slicer I/O + CLI) → lib (paths / excel / SUV factor)
"""
{preamble}
try:
    from PETCTQuantAnalysis_v2Logic import PETCTQuantAnalysis_v2Logic
except ImportError:
    import importlib.util, os as _os
    _p = _os.path.join(_os.path.dirname(__file__), "PETCTQuantAnalysis_v2Logic.py")
    _spec = importlib.util.spec_from_file_location("PETCTQuantAnalysis_v2Logic", _p)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    PETCTQuantAnalysis_v2Logic = getattr(_mod, "PETCTQuantAnalysis_v2Logic")

{module_cls}

{widget_cls}
'''
(OUT / "PETCTQuantAnalysis_v2.py").write_text(widget_file, encoding="utf-8")
(OUT / "__init__.py").write_text('"""PET-CT Quantification module package."""\n', encoding="utf-8")
print("OK")
