"""
One-shot scaffold: Widgets (UI+triggers) + Logic stubs + lib placeholders.
Run from extension_new:  python _scaffold.py
"""
from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT.parent  # extension/


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.lstrip("\n") if content.startswith("\n") else content, encoding="utf-8")
    print(f"  wrote {path.relative_to(ROOT)}")


def extract_class_source(py_path: Path, class_name: str) -> str:
    """Return source text of a top-level class including decorators."""
    src = py_path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            start = node.lineno - 1
            # include decorators
            if node.decorator_list:
                start = min(d.lineno for d in node.decorator_list) - 1
            end = node.end_lineno
            return "".join(lines[start:end])
    raise ValueError(f"{class_name} not found in {py_path}")


def extract_module_preamble(py_path: Path, stop_at_class: str) -> str:
    """Imports + constants before the first target class."""
    src = py_path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == stop_at_class:
            return "".join(lines[: node.lineno - 1])
    return src


# ── lib stubs ────────────────────────────────────────────────────────────────

LIB_FILES = {
    "lib/__init__.py": '"""Pure logic package — no qt/slicer UI here."""\n',
    "lib/io/__init__.py": "",
    "lib/io/paths.py": '''\
"""Path / dataset discovery helpers."""
from __future__ import annotations
from pathlib import Path
from typing import Any


def collect_ct_scans(root: Path) -> list[Path]:
    """Find CT .nii.gz files under folders containing 'CT'."""
    raise NotImplementedError("Implement in lib/io/paths.py — see extension/psoas_segmentation.py")


def collect_ct_folders(root: Path) -> list[Path]:
    """Collect immediate CT scan folders that contain DICOM files."""
    raise NotImplementedError("Implement in lib/io/paths.py — see extension/spleen_segmentation.py")


def choose_ct_file(ct_files: list[Path], index: int) -> Path:
    raise NotImplementedError("Implement in lib/io/paths.py")


def map_ct_to_seg_folder(ct_folder: Path, seg_root: Path) -> Path:
    """Map MSP0001_2025-07-09_CT -> MSP0001_2025-07-09_Seg under seg_root."""
    raise NotImplementedError("Implement in lib/io/paths.py")


def discover_patients(root: str) -> list[dict[str, Any]]:
    """
    Walk a dataset root and return patient-scan dicts.
    Expected keys: subject_id, scan_date (and any paths you need).
    Port from PETCTSegmentationModuleLogic.discoverPatients.
    """
    raise NotImplementedError("Implement in lib/io/paths.py")


def detect_scans(root: str) -> list[dict[str, Any]]:
    """Port from PETCTQuantAnalysis_v2Logic.detectScans."""
    raise NotImplementedError("Implement in lib/io/paths.py")


def detect_segmentations(root: str, scans: list[dict]) -> dict[str, Any]:
    """Port from PETCTQuantAnalysis_v2Logic.detectSegmentations."""
    raise NotImplementedError("Implement in lib/io/paths.py")


def find_bulk_subjects(dataset_root: str) -> list[dict[str, Any]]:
    """Port from VesselSegmenterLogic.find_bulk_subjects."""
    raise NotImplementedError("Implement in lib/io/paths.py")
''',
    "lib/io/nifti.py": '''\
"""NIfTI integrity / load / save helpers (no Slicer UI)."""
from __future__ import annotations
from typing import Any, Optional


def ok_nii(path: str, min_kb: int = 5, need_nonzero: bool = False, log=None) -> bool:
    """Return True if path is a readable, non-corrupted NIfTI. Port _ok_nii."""
    raise NotImplementedError("Implement in lib/io/nifti.py — see PETCTSegmentationModule._ok_nii")


def flip_nifti_axis(nii_path: str, ref_ct: str, axis: int, axis_name: str, log=None) -> None:
    """Flip a NIfTI along axis (0=X, 1=Y). Port flipNiftiXAxis / YAxis."""
    raise NotImplementedError("Implement in lib/io/nifti.py — see mirroring.py / Logic._flipNiftiAxis")


def convert_dicom_to_nifti(dicom_dir: str, out_nii: str, log=None) -> None:
    """CT DICOM folder -> NIfTI. Port PETCTSegmentationModuleLogic.convertDicom."""
    raise NotImplementedError("Implement in lib/io/nifti.py")
''',
    "lib/io/logging_utils.py": '''\
"""Patient pipeline logger (file + console). Port PLog from PETCTSegmentationModule."""
from __future__ import annotations


class PatientLog:
    def __init__(self, log_dir: str, key: str):
        raise NotImplementedError("Implement PatientLog — see PETCTSegmentationModule.PLog")

    def info(self, msg: str): ...
    def ok(self, msg: str): ...
    def warn(self, msg: str): ...
    def error(self, msg: str): ...
    def sep(self, label: str = ""): ...
    def close(self): ...
''',
    "lib/io/batch_patient.py": '''\
"""Batch patient record + steps.json persistence."""
from __future__ import annotations
from typing import Optional


class BatchPatient:
    """Port BatchPatient + _save_steps / _load_steps from PETCTSegmentationModule."""

    def __init__(self, subject_id: str, scan_date: str, root: str):
        raise NotImplementedError("Implement BatchPatient")

    @property
    def key(self) -> str: ...
    def ct_dcm(self) -> str: ...
    def pet_dcm(self) -> str: ...
    def ct_nii(self) -> str: ...
    def seg_dir(self) -> str: ...
    def log_dir(self) -> str: ...
    def seg(self, fname: str) -> str: ...
    def set(self, step: str, status: str, msg: str = "", pct: int = -1): ...
    def is_done(self, step: str) -> bool: ...


def save_steps(rec: BatchPatient) -> None:
    raise NotImplementedError("Implement save_steps")


def load_steps(rec: BatchPatient) -> bool:
    raise NotImplementedError("Implement load_steps")
''',
    "lib/segmentation/__init__.py": "",
    "lib/segmentation/totalseg.py": '''\
"""TotalSegmentator wrappers."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Optional, Sequence


def run_totalsegmentator_api(input_path: Path, output_dir: Path, **kwargs: Any) -> None:
    """Prefer GPU; fall back if device kw unsupported. Port from psoas/spleen scripts."""
    raise NotImplementedError("Implement — see psoas_segmentation.run_totalsegmentator_api")


def run_totalsegmentator_cli(
    ct_nii: str,
    out_dir: str,
    task: str,
    ts_cmd: str,
    gpu: str,
    ts_fast: bool,
    check_files: Optional[Sequence[str]] = None,
    log=None,
) -> None:
    """CLI-based TS used by the Slicer pipeline. Port Logic.runTS."""
    raise NotImplementedError("Implement — see PETCTSegmentationModuleLogic.runTS")


def run_iliopsoas_segmentation(ct_path: Path) -> Path:
    """ROI subset iliopsoas_left/right. Port psoas_segmentation.run_iliopsoas_segmentation."""
    raise NotImplementedError("Implement in lib/segmentation/totalseg.py or psoas.py")


def run_spleen_segmentation(ct_folder: Path, seg_root: Path) -> Path:
    """Port spleen_segmentation.run_spleen_segmentation."""
    raise NotImplementedError("Implement in lib/segmentation/spleen.py")
''',
    "lib/segmentation/psoas.py": '''\
from __future__ import annotations
from pathlib import Path


def run_iliopsoas_segmentation(ct_path: Path) -> Path:
    raise NotImplementedError("Port from extension/psoas_segmentation.py")
''',
    "lib/segmentation/spleen.py": '''\
from __future__ import annotations
from pathlib import Path


def run_spleen_segmentation(ct_folder: Path, seg_root: Path) -> Path:
    raise NotImplementedError("Port from extension/spleen_segmentation.py")
''',
    "lib/segmentation/visceral_fat.py": '''\
"""Combined mask + SegResNet VF prediction."""
from __future__ import annotations
from typing import Optional


def build_combined_mask(seg_dir: str, out_path: str, log=None, label_map: Optional[dict] = None) -> None:
    """Port Logic.buildCombinedMask / visceral_fat_segmentations.py."""
    raise NotImplementedError("Implement — see PETCTSegmentationModuleLogic.buildCombinedMask")


def run_vf_prediction(
    combined: str,
    ct_nii: str,
    out: str,
    ckpt: str,
    device: str = "gpu",
    overlap: float = 0.10,
    log=None,
) -> None:
    """Port Logic.runVFPrediction / VFInference.py."""
    raise NotImplementedError("Implement — see PETCTSegmentationModuleLogic.runVFPrediction")
''',
    "lib/segmentation/vessels.py": '''\
"""PET blood-pool vessel growing."""
from __future__ import annotations
from typing import Any, Optional, Sequence


def grow_vessels_from_seeds(
    pet_arr,
    spacing_mm: Sequence[float],
    seeds_zyx: Sequence[tuple],
    suv_min: float,
    suv_max: float,
    max_extent_mm: float,
    closing_radius_mm: float,
    min_volume_ml: float,
    stitch: bool = True,
    stitch_gap_mm: float = 25.0,
    bridge_radius_mm: float = 2.0,
    ct_vesselness_mask=None,
) -> dict[str, Any]:
    """
    Core algorithm from VesselSegmenterLogic.run (without Slicer nodes).
    Return dict of segment_name -> binary mask array.
    """
    raise NotImplementedError("Port from VesselSegmenterLogic.run and helpers")


def apply_distance_constraint(comp_mask, seed_zyx, vox_mm, max_extent_mm):
    raise NotImplementedError("Port _apply_distance_constraint")


def stitch_fragments(mask_arr, vox_mm, max_gap_mm, bridge_radius_mm):
    raise NotImplementedError("Port _stitch_fragments")


def ct_vesselness_mask(ct_arr, spacing_mm, sigma_min_mm, sigma_max_mm, threshold):
    raise NotImplementedError("Port _ct_vesselness_mask")
''',
    "lib/segmentation/hotspots.py": '''\
"""PET hotspot finding (array-level)."""
from __future__ import annotations
from typing import Any


def find_hottest_voxels(pet_arr, label_arr, ijk_to_ras_4x4, top_n: int = 10) -> list[dict[str, Any]]:
    """
    Return up to top_n dicts: {suv, ras_x, ras_y, ras_z}.
    Port PETHotspotNavigatorLogic.findHottestVoxels (numpy only).
    """
    raise NotImplementedError("Port from PETHotspotNavigatorLogic.findHottestVoxels")
''',
    "lib/processing/__init__.py": "",
    "lib/processing/dilate.py": '''\
"""Morphological dilate + subtract."""
from __future__ import annotations
from typing import Any, Sequence


def dilate_mask(arr, affine, dilate_mm: float):
    """Port SegmentDilatorLogic._dilate."""
    raise NotImplementedError("Implement — see SegmentDilatorLogic._dilate")


def dilate_and_subtract_scene(src_configs: list[dict], tgt_configs: list[dict]) -> dict[str, Any]:
    """
    High-level scene mode. You may keep Slicer I/O in Logic and only put
    array ops here — then implement dilate_mask + resample instead.
    """
    raise NotImplementedError("Port SegmentDilatorLogic.run (array core)")


def dilate_and_subtract_bulk(
    dataset_root: str,
    src_file_configs: list[dict],
    tgt_file_configs: list[dict],
    skip_done: bool = True,
    progress_cb=None,
    log_cb=None,
) -> dict[str, Any]:
    """Port SegmentDilatorLogic.run_bulk core."""
    raise NotImplementedError("Port SegmentDilatorLogic.run_bulk")


def resample_to_target(src_arr, src_affine, tgt_shape, tgt_affine):
    raise NotImplementedError("Port _resample_to_target")
''',
    "lib/processing/ureter.py": '''\
"""Ureter mask + organ clip/clean."""
from __future__ import annotations
from typing import Any, Optional, Sequence


def build_ureter_mask(
    pet_arr,
    pet_affine,
    vox_size,
    suv_thresh: float,
    z_bounds,
    dilate_mm: float,
    connect_path: bool = True,
    max_gap_mm: float = 35.0,
    fill_holes: bool = True,
    inferior_z: Optional[float] = None,
):
    """Port UreterPostProcessLogic._build_ureter_mask (+ _connect_ureter_path)."""
    raise NotImplementedError("Port from UreterPostProcessLogic")


def apply_organ_processing(
    organ_arr,
    organ_affine,
    mode: str,
    ureter_mask,
    l1l5_z_bounds,
    excl_masks: list,
    suv_clean_thresh: float,
    pet_arr=None,
    pet_affine=None,
):
    """Port _apply_processing / _process_organ_*."""
    raise NotImplementedError("Port from UreterPostProcessLogic")


def get_l1l5_z_bounds(vertebra_masks: dict, affines: dict):
    raise NotImplementedError("Port _get_l1l5_z_bounds")
''',
    "lib/processing/mirroring.py": '''\
"""Left/right or A/P flip of VF masks."""
from __future__ import annotations


def flip_volume_axis(arr, axis: int):
    """Numpy flip along axis. Port mirroring.py / Logic._flipNiftiAxis core."""
    raise NotImplementedError("Port from extension/mirroring.py")
''',
    "lib/quantification/__init__.py": "",
    "lib/quantification/distance.py": '''\
"""Distance math (no Markups UI)."""
from __future__ import annotations
import math
from typing import Sequence


def euclidean_mm(p1: Sequence[float], p2: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def ras_distance_to_voxels(p1_ras, p2_ras, ras_to_ijk_4x4) -> float:
    """Convert two RAS points via RAS→IJK and return Euclidean voxel distance."""
    raise NotImplementedError(
        "Port DistanceMeasurerWidget._mm_to_vox (matrix multiply + sqrt)"
    )


def format_distance(length_mm: float, unit: str, voxel_length: float | None = None) -> str:
    if unit == "mm":
        return f"{length_mm:.2f} mm"
    if unit == "cm":
        return f"{length_mm / 10.0:.3f} cm"
    if voxel_length is None:
        return "— vox (no ref)"
    return f"{voxel_length:.2f} vox"
''',
    "lib/quantification/pet_metrics.py": '''\
"""Batch PET quantification → Excel."""
from __future__ import annotations
from typing import Any, Callable, Optional


def run_batch_quantification(
    root_folder: str,
    scans: list[dict],
    seg_name_map: dict[str, str],
    output_file: str,
    metrics: dict[str, bool],
    suv_type: str = "bw",
    append: bool = True,
    progress_cb: Optional[Callable] = None,
    status_cb: Optional[Callable] = None,
    cancel_check: Optional[Callable] = None,
) -> None:
    """Port PETCTQuantAnalysis_v2Logic.runBatch."""
    raise NotImplementedError("Port from PETCTQuantAnalysis_v2Logic")


def compute_segment_metrics(pet_arr, mask_arr, spacing_mm, metrics: dict, suv_type: str) -> dict[str, Any]:
    """SUVmean/max/peak, TLG, volume for one mask."""
    raise NotImplementedError("Extract metric math from PETCTQuantAnalysis_v2Logic")
''',
    "lib/models/__init__.py": "",
    "lib/models/segresnet.py": '''\
"""
Place SegResNet / SPADESegResNet here (or re-export).
Port from extension/segresnet.py — large file; copy when ready.
"""
raise NotImplementedError(
    "Copy extension/segresnet.py into lib/models/segresnet.py when you implement VF inference"
)
''',
}


def write_lib() -> None:
    print("Writing lib stubs…")
    for rel, body in LIB_FILES.items():
        write(ROOT / rel, body)


def write_scripts() -> None:
    print("Writing scripts…")
    write(
        ROOT / "scripts" / "run_psoas.py",
        '''\
"""CLI entry — wire config here; call lib when implemented."""
from __future__ import annotations
import os
from pathlib import Path
from multiprocessing import freeze_support

# TODO: set your paths
DATASET_ROOT = Path(r"D:\\segmentation\\data_organising\\datasetA_nii")
CUDA_DEVICE = "0"
CT_INDEX = 1


def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE
    # from lib.io.paths import collect_ct_scans, choose_ct_file
    # from lib.segmentation.psoas import run_iliopsoas_segmentation
    # ct = choose_ct_file(collect_ct_scans(DATASET_ROOT), CT_INDEX)
    # out = run_iliopsoas_segmentation(ct)
    # print("Done:", out)
    raise NotImplementedError("Implement lib/io/paths + lib/segmentation/psoas first")


if __name__ == "__main__":
    freeze_support()
    main()
''',
    )
    write(
        ROOT / "scripts" / "run_spleen.py",
        '''\
"""CLI entry for spleen TotalSegmentator batch."""
from __future__ import annotations
import os
from pathlib import Path
from multiprocessing import freeze_support

CT_ROOT = Path(r"E:\\Claude-Ishita\\dataset_clean\\CT")
SEG_ROOT = Path(r"E:\\Claude-Ishita\\dataset_clean\\Segments")
CUDA_DEVICE = "0"


def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE
    raise NotImplementedError("Implement lib/io/paths + lib/segmentation/spleen first")


if __name__ == "__main__":
    freeze_support()
    main()
''',
    )
    write(ROOT / "scripts" / "__init__.py", "")


# ── Widget extraction helpers ────────────────────────────────────────────────

WIDGET_EXTRA_HELPERS = {
    # class names that live outside Widget but Widget needs them
    "PETCTSegmentationModule": [
        "_ok_nii",
        "PLog",
        "BatchPatient",
        "_steps_file",
        "_save_steps",
        "_load_steps",
        "_ts_task_supports_fast",
    ],
}


def method_stubs_from_logic(py_path: Path, logic_class: str) -> str:
    """Generate Logic class with methods that raise NotImplementedError pointing to lib."""
    src = py_path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    methods = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == logic_class:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name.startswith("__") and item.name.endswith("__"):
                        continue
                    args = [a.arg for a in item.args.args]
                    # keep signature loosely
                    sig_args = ", ".join(args)
                    methods.append(
                        f"    def {item.name}({sig_args}, *a, **kw):\n"
                        f'        raise NotImplementedError(\n'
                        f'            "{logic_class}.{item.name}: implement underlying lib/* '
                        f'and wire this Logic method to it"\n'
                        f"        )\n"
                    )
            break
    body = "\n".join(methods) if methods else "    pass\n"
    return (
        f"class {logic_class}(ScriptedLoadableModuleLogic):\n"
        f'    """Thin adapter: Slicer nodes ↔ lib. Fill in after lib is ready."""\n\n'
        f"{body}"
    )


MODULES = [
    {
        "folder": "DistanceMeasurer",
        "src": "DistanceMeasurer.py",
        "module": "DistanceMeasurer",
        "widget": "DistanceMeasurerWidget",
        "logic": "DistanceMeasurerLogic",
        "title": "6. Distance Measurer",
        "category": "Metabolic Syndrome Toolkit",
        "lib_notes": "lib/quantification/distance.py",
    },
    {
        "folder": "PETHotspotNavigator",
        "src": "PETHotspotNavigator.py",
        "module": "PETHotspotNavigator",
        "widget": "PETHotspotNavigatorWidget",
        "logic": "PETHotspotNavigatorLogic",
        "title": "5. PET Hotspot Navigator",
        "category": "Metabolic Syndrome Toolkit",
        "lib_notes": "lib/segmentation/hotspots.py",
    },
    {
        "folder": "ModuleLauncher",
        "src": "ModuleLauncher.py",
        "module": "ModuleLauncher",
        "widget": "ModuleLauncherWidget",
        "logic": "ModuleLauncherLogic",
        "title": "Module Launcher",
        "category": "Utilities",
        "lib_notes": "(UI only — no lib)",
    },
    {
        "folder": "ScribbleTool",
        "src": "ScribbleTool.py",
        "module": "ScribbleTool",
        "widget": "ScribbleToolWidget",
        "logic": "ScribbleToolLogic",
        "title": "Scribble Tool",
        "category": "Utilities",
        "lib_notes": "(UI/Markups only — optional helpers later)",
        "extra_classes": ["_SliceEventFilter"],
    },
    {
        "folder": "SegmentDilator",
        "src": "SegmentDilator.py",
        "module": "SegmentDilator",
        "widget": "SegmentDilatorWidget",
        "logic": "SegmentDilatorLogic",
        "title": "3. Segment Dilator",
        "category": "Metabolic Syndrome Toolkit",
        "lib_notes": "lib/processing/dilate.py",
    },
    {
        "folder": "VesselSegmenter",
        "src": "VesselSegmenter.py",
        "module": "VesselSegmenter",
        "widget": "VesselSegmenterWidget",
        "logic": "VesselSegmenterLogic",
        "title": "2. Vessel Segmenter",
        "category": "Metabolic Syndrome Toolkit",
        "lib_notes": "lib/segmentation/vessels.py + lib/io/paths.find_bulk_subjects",
    },
    {
        "folder": "UreterPostProcess",
        "src": "UreterPostProcess.py",
        "module": "UreterPostProcess",
        "widget": "UreterPostProcessWidget",
        "logic": "UreterPostProcessLogic",
        "title": "4. Ureter Post-Process",
        "category": "Metabolic Syndrome Toolkit",
        "lib_notes": "lib/processing/ureter.py",
        "extra_funcs": ["_default_mode"],
    },
    {
        "folder": "PETCTSegmentationModule",
        "src": "PETCTSegmentationModule.py",
        "module": "PETCTSegmentationModule",
        "widget": "PETCTSegmentationModuleWidget",
        "logic": "PETCTSegmentationModuleLogic",
        "title": "1. PETCT Segmentation Module",
        "category": "Metabolic Syndrome Toolkit",
        "lib_notes": "lib/io/*, lib/segmentation/totalseg.py, visceral_fat.py, mirroring",
        "keep_preamble_constants": True,
    },
    {
        "folder": "PETCTQuantAnalysis",
        "src": "PETCTQuantAnalysis_v2.py",
        "module": "PETCTQuantAnalysis_v2",
        "widget": "PETCTQuantAnalysis_v2Widget",
        "logic": "PETCTQuantAnalysis_v2Logic",
        "title": "7. PET-CT Quantification",
        "category": "Metabolic Syndrome Toolkit",
        "lib_notes": "lib/quantification/pet_metrics.py + lib/io/paths",
    },
]


def scaffold_module(cfg: dict) -> None:
    src_path = SRC / cfg["src"]
    if not src_path.exists():
        print(f"  SKIP missing {src_path}")
        return

    out_dir = ROOT / "slicer_modules" / cfg["folder"]
    # remove .gitkeep if present
    gk = out_dir / ".gitkeep"
    if gk.exists():
        gk.unlink()

    # preamble: imports + constants before Module class
    preamble = extract_module_preamble(src_path, cfg["module"])
    # strip huge docstring at top optionally keep
    module_cls = extract_class_source(src_path, cfg["module"])
    widget_cls = extract_class_source(src_path, cfg["widget"])

    extras = ""
    for extra in cfg.get("extra_classes", []):
        try:
            extras += "\n\n" + extract_class_source(src_path, extra)
        except ValueError:
            pass
    for extra in cfg.get("extra_funcs", []):
        # pull function by regex
        text = src_path.read_text(encoding="utf-8")
        m = re.search(
            rf"(^def {re.escape(extra)}\(.*?(?=^def |^class |\Z))",
            text,
            flags=re.M | re.S,
        )
        if m:
            extras = m.group(1).rstrip() + "\n\n" + extras

    # For PETCTSegmentationModule keep helper classes used by Widget
    if cfg["folder"] == "PETCTSegmentationModule":
        for name in ("PLog", "BatchPatient"):
            extras += "\n\n" + extract_class_source(src_path, name)
        text = src_path.read_text(encoding="utf-8")
        for fname in ("_ok_nii", "_steps_file", "_save_steps", "_load_steps", "_ts_task_supports_fast"):
            m = re.search(
                rf"(^def {re.escape(fname)}\(.*?(?=^def |^class |\Z))",
                text,
                flags=re.M | re.S,
            )
            if m:
                extras = m.group(1).rstrip() + "\n\n" + extras

    # Widget file: preamble bits + widget + extras that widget needs
    # Keep original imports from preamble but add relative logic import note
    widget_header = f'''\
"""
{cfg["widget"]} — UI + triggers only.
Calls {cfg["logic"]} methods; those should delegate to lib/ once you implement them.
Source ported from extension/{cfg["src"]}.
"""
'''
    # Use original preamble for imports/constants, then widget
    widget_body = (
        widget_header
        + preamble
        + f"\nfrom {cfg['folder']}.{cfg['logic'].replace(cfg['folder'], cfg['logic']) if False else cfg['logic']} "
        # simpler: import Logic from sibling via same package name pattern
    )

    # Simpler packaging for Slicer drag-drop: single entry file that includes
    # Module + Widget, Logic in separate file imported if possible.
    # Many Slicer users load by path — use flat imports within the folder via
    # injecting Logic into the same module namespace from entry file.

    logic_src = method_stubs_from_logic(src_path, cfg["logic"])
    write(
        out_dir / f"{cfg['logic']}.py",
        f'''\
"""
{cfg["logic"]} — thin stubs.
Implement bodies by calling lib/ ({cfg["lib_notes"]}).
"""
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleLogic

# When ready, import from lib, e.g.:
#   from lib.segmentation.hotspots import find_hottest_voxels

{logic_src}
''',
    )

    # Entry file: Module + Widget (+ helpers) + import Logic
    entry = f'''\
"""
{cfg["module"]} — Slicer entry (Module metadata + Widget UI/triggers).
Logic lives in {cfg["logic"]}.py and should call lib/ once implemented.
"""
{preamble}
from {cfg["logic"]} import {cfg["logic"]}

{extras}

{module_cls}

{widget_cls}
'''
    # Fix: relative import may fail when Slicer loads file by path.
    # Use same-directory import hack:
    entry = entry.replace(
        f"from {cfg['logic']} import {cfg['logic']}",
        textwrap.dedent(
            f"""\
            try:
                from {cfg['logic']} import {cfg['logic']}
            except ImportError:
                import importlib.util, os as _os
                _p = _os.path.join(_os.path.dirname(__file__), "{cfg['logic']}.py")
                _spec = importlib.util.spec_from_file_location("{cfg['logic']}", _p)
                _mod = importlib.util.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)
                {cfg['logic']} = getattr(_mod, "{cfg['logic']}")
            """
        ),
    )

    write(out_dir / f"{cfg['module']}.py", entry)

    # Package marker
    write(out_dir / "__init__.py", f'"""Slicer module package: {cfg["folder"]}."""\n')


def write_petbiomarker_stub() -> None:
    """PETBiomarkerStudio is huge — stub UI shell only."""
    out = ROOT / "slicer_modules" / "PETBiomarkerStudio"
    gk = out / ".gitkeep"
    if gk.exists():
        gk.unlink()
    write(
        out / "PETBiomarkerStudio.py",
        '''\
"""
PETBiomarkerStudio — UI shell (original is monolithic ~3k lines).
Triggers call Logic stubs → implement lib later, or split the old file step by step.
"""
import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
)


class PETBiomarkerStudio(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "PET Biomarker Studio"
        self.parent.categories = ["Metabolic Syndrome Toolkit"]
        self.parent.contributors = ["IshitaSinghFaujdar"]
        self.parent.helpText = (
            "Scaffold only. Port UI sections from extension/PETBiomarkerStudio.py "
            "and move algorithms into lib/."
        )


class PETBiomarkerStudioWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = PETBiomarkerStudioLogic()

        box = ctk.ctkCollapsibleButton()
        box.text = "Scaffold"
        self.layout.addWidget(box)
        lay = qt.QVBoxLayout(box)

        info = qt.QLabel(
            "UI not fully ported yet.\\n"
            "Open extension/PETBiomarkerStudio.py and move each collapsible "
            "section here; wire buttons to self.logic.* stubs."
        )
        info.setWordWrap(True)
        lay.addWidget(info)

        btn = qt.QPushButton("Example trigger → logic.placeholder()")
        btn.clicked.connect(self._on_example)
        lay.addWidget(btn)
        self.status = qt.QLabel("")
        lay.addWidget(self.status)
        self.layout.addStretch(1)

    def _on_example(self):
        try:
            self.logic.placeholder()
        except NotImplementedError as e:
            self.status.setText(str(e))
            slicer.util.warningDisplay(str(e))


class PETBiomarkerStudioLogic(ScriptedLoadableModuleLogic):
    def placeholder(self):
        raise NotImplementedError(
            "Split PETBiomarkerStudio.py into lib/ modules, then implement this."
        )
'''
    )
    write(out / "__init__.py", "")


def main() -> None:
    write_lib()
    write_scripts()
    print("Writing slicer_modules…")
    for cfg in MODULES:
        print(f"  module {cfg['folder']}")
        scaffold_module(cfg)
    write_petbiomarker_stub()
    print("Done.")


if __name__ == "__main__":
    main()
