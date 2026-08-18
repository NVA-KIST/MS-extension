"""
DistanceMeasurerLogic — Slicer adapter (nodes ↔ lib).

Widget → Logic.formatDistance / Logic.voxelDistance → lib.quantification.distance
"""
from __future__ import annotations

import os
import sys

import numpy as np
import vtk
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleLogic

_EXT_NEW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _EXT_NEW_ROOT not in sys.path:
    sys.path.insert(0, _EXT_NEW_ROOT)

from lib.quantification.distance import format_distance, ras_distance_to_voxels


def _vtk_matrix_to_numpy(mat: vtk.vtkMatrix4x4) -> np.ndarray:
    out = np.eye(4, dtype=float)
    for r in range(4):
        for c in range(4):
            out[r, c] = mat.GetElement(r, c)
    return out


class DistanceMeasurerLogic(ScriptedLoadableModuleLogic):

    def voxelDistance(self, line_node, ref_vol) -> float:
        """RAS endpoints of a Markups line → Euclidean distance in voxels."""
        p1 = [0.0, 0.0, 0.0]
        p2 = [0.0, 0.0, 0.0]
        line_node.GetNthControlPointPositionWorld(0, p1)
        line_node.GetNthControlPointPositionWorld(1, p2)

        mat = vtk.vtkMatrix4x4()
        ref_vol.GetRASToIJKMatrix(mat)
        return ras_distance_to_voxels(p1, p2, _vtk_matrix_to_numpy(mat))

    def formatDistance(self, length_mm: float, unit: str, voxel_length=None) -> str:
        return format_distance(length_mm, unit, voxel_length)
