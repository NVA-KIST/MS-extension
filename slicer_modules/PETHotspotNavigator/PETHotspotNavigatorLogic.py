"""
PETHotspotNavigatorLogic — Slicer adapter (nodes ↔ lib).

Pattern for other modules
-------------------------
1. Read Slicer nodes → numpy / plain Python values
2. Call lib.* (no qt/slicer inside lib)
3. Return plain results for the Widget to display
"""
from __future__ import annotations

import os
import sys

import numpy as np
import slicer
import vtk
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleLogic

# Make extension_new/ importable as the package root for `lib.*`
_EXT_NEW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _EXT_NEW_ROOT not in sys.path:
    sys.path.insert(0, _EXT_NEW_ROOT)

from lib.segmentation.hotspots import find_hottest_voxels  # noqa: E402


def _vtk_matrix_to_numpy(mat: vtk.vtkMatrix4x4) -> np.ndarray:
    out = np.eye(4, dtype=float)
    for r in range(4):
        for c in range(4):
            out[r, c] = mat.GetElement(r, c)
    return out


class PETHotspotNavigatorLogic(ScriptedLoadableModuleLogic):
    """Thin bridge: Slicer MRML nodes → lib.segmentation.hotspots."""

    def findHottestVoxels(self, pet_node, label_node, top_n=10):
        """
        Widget-facing API (same signature as the old monolithic module).

        Returns
        -------
        list[dict]: {suv, ras_x, ras_y, ras_z}
        """
        pet_arr = slicer.util.arrayFromVolume(pet_node)
        label_arr = slicer.util.arrayFromVolume(label_node)

        ijk_to_ras_vtk = vtk.vtkMatrix4x4()
        pet_node.GetIJKToRASMatrix(ijk_to_ras_vtk)
        ijk_to_ras = _vtk_matrix_to_numpy(ijk_to_ras_vtk)

        return find_hottest_voxels(
            pet_arr=pet_arr,
            label_arr=label_arr,
            ijk_to_ras=ijk_to_ras,
            top_n=top_n,
        )
