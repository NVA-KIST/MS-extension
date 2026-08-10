"""Port SegmentDilator + UreterPostProcess into extension_new with lib wiring."""
from __future__ import annotations

import ast
import re
from pathlib import Path

EXT = Path(r"e:\KUPETCTMS\extension")
OUT = Path(r"e:\KUPETCTMS\extension\extension_new\slicer_modules")


def extract_classes(src_path: Path, names: list[str]):
    src = src_path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    found = {}
    preamble = ""
    first = names[0]
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == first:
            preamble = "".join(lines[: node.lineno - 1])
            break
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in names:
            found[node.name] = "".join(lines[node.lineno - 1 : node.end_lineno])
    # helper funcs before first class
    extras = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            extras.append("".join(lines[node.lineno - 1 : node.end_lineno]))
        if isinstance(node, ast.ClassDef) and node.name == first:
            break
    return preamble, found, extras


def replace_method(class_src: str, method_name: str, new_body: str) -> str:
    pattern = rf"(    def {re.escape(method_name)}\(.*?(?=\n    def |\nclass |\Z))"
    m = re.search(pattern, class_src, flags=re.S)
    if not m:
        raise SystemExit(f"method {method_name} not found")
    return class_src[: m.start()] + new_body.rstrip() + "\n\n" + class_src[m.end() :]


def write_module(folder, module, widget, logic, src_file, logic_header, patches: dict):
    preamble, classes, extras = extract_classes(EXT / src_file, [module, widget, logic])
    logic_src = classes[logic]
    for name, body in patches.items():
        logic_src = replace_method(logic_src, name, body)

    out = OUT / folder
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{logic}.py").write_text(logic_header + "\n" + logic_src + "\n", encoding="utf-8")

    extras_txt = ("\n\n".join(extras) + "\n\n") if extras else ""
    entry = f'''\
"""
{module} — Module + Widget.

  Widget → {logic} → lib/processing/*
"""
{preamble}
try:
    from {logic} import {logic}
except ImportError:
    import importlib.util, os as _os
    _p = _os.path.join(_os.path.dirname(__file__), "{logic}.py")
    _spec = importlib.util.spec_from_file_location("{logic}", _p)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    {logic} = getattr(_mod, "{logic}")

{extras_txt}{classes[module]}

{classes[widget]}
'''
    (out / f"{module}.py").write_text(entry, encoding="utf-8")
    (out / "__init__.py").write_text(f'"""{folder} package."""\n', encoding="utf-8")
    print("ported", folder)


DILATE_HEADER = '''\
"""
SegmentDilatorLogic — Slicer adapter.

lib: dilate_mask, resample_to_target, subtract_dilated_union, bulk_outputs_exist
Logic: export/import Slicer nodes, bulk folder orchestration
"""
from __future__ import annotations

import os
import sys
import vtk
import slicer
import numpy as np
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleLogic

_EXT_NEW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _EXT_NEW_ROOT not in sys.path:
    sys.path.insert(0, _EXT_NEW_ROOT)

from lib.processing.dilate import (
    dilate_mask as lib_dilate_mask,
    resample_to_target as lib_resample_to_target,
    subtract_dilated_union,
    bulk_outputs_exist as lib_bulk_outputs_exist,
    file_stem as lib_file_stem,
)

'''

URETER_HEADER = '''\
"""
UreterPostProcessLogic — Slicer adapter.

lib: build_ureter_mask_from_pet, connect_ureter_path, apply_organ_processing,
     apply_exclusion_mask, dilate_mask, resample_to_target
Logic: Slicer node I/O, DICOM load, vertebra Z bounds from scene
"""
from __future__ import annotations

import os
import sys
import vtk
import slicer
import numpy as np
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleLogic

_EXT_NEW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _EXT_NEW_ROOT not in sys.path:
    sys.path.insert(0, _EXT_NEW_ROOT)

from lib.processing.dilate import dilate_mask as lib_dilate_mask
from lib.processing.dilate import resample_to_target as lib_resample_to_target
from lib.processing.ureter import (
    build_ureter_mask_from_pet,
    connect_ureter_path as lib_connect_ureter_path,
    apply_organ_processing as lib_apply_organ_processing,
    apply_exclusion_mask as lib_apply_exclusion_mask,
)

'''

write_module(
    "SegmentDilator",
    "SegmentDilator",
    "SegmentDilatorWidget",
    "SegmentDilatorLogic",
    "SegmentDilator.py",
    DILATE_HEADER,
    {
        "_dilate": '''\
    def _dilate(self, arr, affine, dilate_mm):
        return lib_dilate_mask(arr, affine, dilate_mm)
''',
        "_resample_to_target": '''\
    def _resample_to_target(self, src_arr, src_affine, tgt_shape, tgt_affine):
        return lib_resample_to_target(src_arr, src_affine, tgt_shape, tgt_affine)
''',
        "_stem": '''\
    @staticmethod
    def _stem(filename):
        return lib_file_stem(filename)
''',
        "_bulk_outputs_exist": '''\
    def _bulk_outputs_exist(self, seg_dir, src_file_configs, tgt_file_configs):
        return lib_bulk_outputs_exist(seg_dir, src_file_configs, tgt_file_configs)
''',
    },
)

# Also patch run() subtract loop to use subtract_dilated_union - optional, current uses _resample which is wired
# Good enough.

write_module(
    "UreterPostProcess",
    "UreterPostProcess",
    "UreterPostProcessWidget",
    "UreterPostProcessLogic",
    "UreterPostProcess.py",
    URETER_HEADER,
    {
        "_connect_ureter_path": '''\
    def _connect_ureter_path(self, mask_arr, vox_size, max_gap_mm=35.0, tube_radius_vox=3):
        return lib_connect_ureter_path(mask_arr, vox_size, max_gap_mm, tube_radius_vox)
''',
        "_apply_processing": '''\
    def _apply_processing(self, organ_arr, organ_affine, mode,
                           ureter_arr, ureter_affine,
                           pet_arr, pet_affine, suv_clean_thresh,
                           z_inferior, z_superior):
        return lib_apply_organ_processing(
            organ_arr, organ_affine, mode,
            ureter_arr=ureter_arr, ureter_affine=ureter_affine,
            pet_arr=pet_arr, pet_affine=pet_affine,
            suv_clean_thresh=suv_clean_thresh,
            z_inferior=z_inferior, z_superior=z_superior,
        )
''',
        "_resample_to_target": '''\
    def _resample_to_target(self, src_arr, src_affine, tgt_shape, tgt_affine):
        return lib_resample_to_target(src_arr, src_affine, tgt_shape, tgt_affine)
''',
    },
)

# Patch _build_ureter_mask array core via regex on the written Logic file
ureter_logic_path = OUT / "UreterPostProcess" / "UreterPostProcessLogic.py"
text = ureter_logic_path.read_text(encoding="utf-8")

# Replace from "# Combined anatomical mask" through fill holes + z clip, before lm_name scene save
# Better: replace the whole method body after resolving ureter_z_inf / printing, calling build_ureter_mask_from_pet

pattern = r"    def _build_ureter_mask\(.*?(?=\n    # ── Ureter connectivity|\n    def _connect_ureter_path)"
new_build = '''\
    def _build_ureter_mask(self, pet_arr, pet_affine, pet_mat, vox_size,
                            z_inferior, z_superior, suv_thresh, dilate_mm,
                            seg_node_name, totalseg_node_name=None,
                            connect_path=True, max_gap_mm=35.0, fill_holes=True,
                            inf_bound_seg_name=None,
                            ureter_ext_inf_mm=90.0, torso_radius_mm=220.0):
        """Build ureter exclusion mask; array core in lib, scene nodes here."""
        if z_inferior is None or z_superior is None:
            raise ValueError(
                "_build_ureter_mask called without Z bounds. "
                "Ensure L1-L5 vertebrae are selected before running.")

        ureter_z_inf = None
        if inf_bound_seg_name and totalseg_node_name:
            try:
                seg_z_min, _ = self._segment_ras_z_bounds(
                    totalseg_node_name, inf_bound_seg_name)
                ureter_z_inf = seg_z_min
                print(f"\\n[URETER] Inferior boundary from '{inf_bound_seg_name}': "
                      f"Z_inf set to {ureter_z_inf:.1f} mm")
            except Exception as e:
                print(f"[URETER] WARNING: could not use '{inf_bound_seg_name}' "
                      f"({e}) — falling back to fixed offset.")

        if totalseg_node_name is not None:
            cx, cy = self._vertebrae_xy_centroid(totalseg_node_name)
        else:
            # Fallback: image XY centre in RAS
            shape = pet_arr.shape
            z_idx, y_idx, x_idx = np.meshgrid(
                np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]),
                indexing='ij')
            ijk_hom = np.stack([
                x_idx.ravel(), y_idx.ravel(), z_idx.ravel(),
                np.ones(x_idx.size)], axis=1).astype(np.float32)
            ras = (np.asarray(pet_affine) @ ijk_hom.T).T
            cx = float(np.mean(ras[:, 0]))
            cy = float(np.mean(ras[:, 1]))

        print(f"\\n[URETER] SUV>{suv_thresh}  dilation={dilate_mm}mm  "
              f"torso XY=({cx:.1f},{cy:.1f})")

        ureter_mask = build_ureter_mask_from_pet(
            pet_arr, pet_affine, vox_size,
            z_inferior, z_superior, suv_thresh, dilate_mm,
            torso_center_xy=(cx, cy),
            ureter_z_inf=ureter_z_inf,
            ureter_ext_inf_mm=ureter_ext_inf_mm,
            torso_radius_mm=torso_radius_mm,
            connect_path=connect_path,
            max_gap_mm=max_gap_mm,
            fill_holes=fill_holes,
        )
        print(f"[URETER] mask voxels: {int(ureter_mask.sum())}")

        lm_name = seg_node_name + '_lm'
        self._remove_existing(lm_name)
        lm = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode', lm_name)
        slicer.util.updateVolumeFromArray(lm, ureter_mask)
        lm.SetIJKToRASMatrix(pet_mat)

        self._remove_existing(seg_node_name)
        seg = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode', seg_node_name)
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(lm, seg)
        seg.CreateClosedSurfaceRepresentation()
        slicer.mrmlScene.RemoveNode(lm)
        print(f"[URETER] '{seg_node_name}' added to scene.")

        return ureter_mask, pet_affine

'''

text2, n = re.subn(pattern, new_build, text, count=1, flags=re.S)
print("ureter _build_ureter_mask replace:", n)
if n != 1:
    raise SystemExit("failed to patch _build_ureter_mask")
ureter_logic_path.write_text(text2, encoding="utf-8")

# Patch exclusion loops in scene/bulk to use lib_apply_exclusion_mask optionally -
# they already call _resample_to_target which is wired. OK.

# Also patch SegmentDilator run() to use subtract_dilated_union for clarity
sd_path = OUT / "SegmentDilator" / "SegmentDilatorLogic.py"
sd = sd_path.read_text(encoding="utf-8")
old = '''\
            # Build union of all dilated sources resampled to target space
            union = np.zeros(tgt_arr.shape, dtype=np.uint8)
            for src_arr, src_affine in dilated_items:
                resampled = self._resample_to_target(
                    src_arr, src_affine, tgt_arr.shape, tgt_affine)
                union = np.maximum(union, (resampled > 0).astype(np.uint8))

            result = tgt_arr.copy()
            removed = int(((result > 0) & (union > 0)).sum())
            result[(result > 0) & (union > 0)] = 0
            after = int((result > 0).sum())
'''
new = '''\
            before_overlap = int((tgt_arr > 0).sum())
            result = subtract_dilated_union(tgt_arr, tgt_affine, dilated_items)
            after = int((result > 0).sum())
            removed = before_overlap - after
'''
if old in sd:
    sd_path.write_text(sd.replace(old, new), encoding="utf-8")
    print("SegmentDilator run() uses subtract_dilated_union")
else:
    print("WARN: could not patch SegmentDilator subtract block")

print("done")
