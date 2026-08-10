"""
PET Hotspot Navigator — Slicer entry (Module + Widget UI/triggers).

Layer map (study this, then copy the pattern)
---------------------------------------------
  PETHotspotNavigator.py          ← YOU ARE HERE: UI + triggers only
       │  button / table click
       ▼
  PETHotspotNavigatorLogic.py     ← Slicer nodes ↔ plain data
       │  numpy arrays + 4x4 matrix
       ▼
  lib/segmentation/hotspots.py    ← pure algorithm (no slicer/qt)

Load this folder (or this .py) via Slicer Additional Module Paths.
"""

import logging
import os
import sys

import vtk
import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
)

LOG = logging.getLogger("PETHotspotNav")

# Import sibling Logic (works when Slicer loads this file by path)
try:
    from PETHotspotNavigatorLogic import PETHotspotNavigatorLogic
except ImportError:
    import importlib.util

    _p = os.path.join(os.path.dirname(__file__), "PETHotspotNavigatorLogic.py")
    _spec = importlib.util.spec_from_file_location("PETHotspotNavigatorLogic", _p)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    PETHotspotNavigatorLogic = getattr(_mod, "PETHotspotNavigatorLogic")


# ── Module metadata ───────────────────────────────────────────────────────────

class PETHotspotNavigator(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "5. PET Hotspot Navigator"
        parent.categories = ["Metabolic Syndrome Toolkit"]
        parent.dependencies = []
        parent.contributors = ["Research Lab"]
        parent.helpText = (
            "Select a PET volume and a segmentation node already in the scene. "
            "Click 'Find Hotspots' to locate the highest-SUV voxel in every segment. "
            "Click any row in the results table to jump the slice views to that location."
        )


# ── Widget (UI + triggers) ────────────────────────────────────────────────────

class PETHotspotNavigatorWidget(ScriptedLoadableModuleWidget):
    """
    Owns: dropdowns, buttons, table, status text, jump-to-slice UX.
    Does NOT own: hottest-voxel math (that is Logic → lib).
    """

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = PETHotspotNavigatorLogic()
        self._row_coords = []
        self._marker_node = None

        # ── 1. Selectors ──────────────────────────────────────────────────────
        col1 = ctk.ctkCollapsibleButton()
        col1.text = "1. Select data (already loaded in scene)"
        self.layout.addWidget(col1)
        lay1 = qt.QFormLayout(col1)

        self.petSelector = slicer.qMRMLNodeComboBox()
        self.petSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.petSelector.selectNodeUponCreation = True
        self.petSelector.addEnabled = False
        self.petSelector.removeEnabled = False
        self.petSelector.noneEnabled = True
        self.petSelector.showHidden = False
        self.petSelector.setMRMLScene(slicer.mrmlScene)
        self.petSelector.setToolTip("Choose the PET volume from the scene")
        lay1.addRow("PET volume:", self.petSelector)

        self.segSelector = slicer.qMRMLNodeComboBox()
        self.segSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
        self.segSelector.selectNodeUponCreation = True
        self.segSelector.addEnabled = False
        self.segSelector.removeEnabled = False
        self.segSelector.noneEnabled = True
        self.segSelector.showHidden = False
        self.segSelector.setMRMLScene(slicer.mrmlScene)
        self.segSelector.setToolTip("Choose the segmentation node from the scene")
        lay1.addRow("Segmentation:", self.segSelector)

        self.topNSpin = qt.QSpinBox()
        self.topNSpin.setRange(1, 50)
        self.topNSpin.setValue(1)
        self.topNSpin.setToolTip(
            "How many hottest voxels to list per segment "
            "(1 = show only the single hottest)"
        )
        lay1.addRow("Top N per segment:", self.topNSpin)

        # ── Find button ───────────────────────────────────────────────────────
        self.findBtn = qt.QPushButton("Find Hotspots")
        self.findBtn.setStyleSheet(
            "QPushButton{background:#4CAF50;color:white;font-weight:bold;"
            "padding:8px;border-radius:4px}"
            "QPushButton:hover{background:#388E3C}"
            "QPushButton:disabled{background:#BDBDBD}"
        )
        self.layout.addWidget(self.findBtn)

        self.statusLabel = qt.QLabel(
            "Select a PET volume and a segmentation node, then click Find Hotspots."
        )
        self.statusLabel.setWordWrap(True)
        self.layout.addWidget(self.statusLabel)

        # ── Results table ─────────────────────────────────────────────────────
        self.resultsTable = qt.QTableWidget(0, 6)
        self.resultsTable.setHorizontalHeaderLabels(
            ["Segment", "Rank", "SUV", "RAS X (mm)", "RAS Y (mm)", "RAS Z (mm)"]
        )
        hdr = self.resultsTable.horizontalHeader()
        hdr.setSectionResizeMode(0, qt.QHeaderView.Stretch)
        for col in range(1, 6):
            hdr.setSectionResizeMode(col, qt.QHeaderView.ResizeToContents)
        self.resultsTable.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.resultsTable.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self.resultsTable.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.resultsTable.setAlternatingRowColors(True)
        self.layout.addWidget(self.resultsTable)

        hint = qt.QLabel("Click a row to jump all slice views to that hotspot.")
        hint.setStyleSheet("color: gray; font-style: italic;")
        self.layout.addWidget(hint)

        self.layout.addStretch(1)

        # ── Triggers ──────────────────────────────────────────────────────────
        self.findBtn.connect("clicked(bool)", self.onFind)
        self.resultsTable.connect("cellClicked(int,int)", self.onRowClicked)

    # ── Handlers (UI orchestration only) ──────────────────────────────────────

    def onFind(self):
        """Validate UI → export each segment → call Logic → fill table."""
        pet_node = self.petSelector.currentNode()
        seg_node = self.segSelector.currentNode()

        if not pet_node:
            slicer.util.errorDisplay("Please select a PET volume from the dropdown.")
            return
        if not seg_node:
            slicer.util.errorDisplay("Please select a segmentation node from the dropdown.")
            return

        segmentation = seg_node.GetSegmentation()
        n_segs = segmentation.GetNumberOfSegments()
        if n_segs == 0:
            slicer.util.errorDisplay("The selected segmentation node has no segments.")
            return

        top_n = self.topNSpin.value
        self.resultsTable.setRowCount(0)
        self._row_coords.clear()
        self.findBtn.setEnabled(False)
        self.statusLabel.setText(f"Analysing {n_segs} segment(s)…")
        slicer.app.processEvents()

        errors = []
        for seg_idx in range(n_segs):
            segment_id = segmentation.GetNthSegmentID(seg_idx)
            segment_name = segmentation.GetSegment(segment_id).GetName()

            self.statusLabel.setText(
                f"Segment {seg_idx + 1}/{n_segs}: {segment_name}…"
            )
            slicer.app.processEvents()

            try:
                # Slicer-only I/O: one segment → temporary labelmap on PET grid
                label_node = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLLabelMapVolumeNode", f"_hs_tmp_{seg_idx}"
                )
                seg_ids = vtk.vtkStringArray()
                seg_ids.InsertNextValue(segment_id)
                slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
                    seg_node,
                    seg_ids,
                    label_node,
                    pet_node,
                    slicer.vtkSegmentation.EXTENT_REFERENCE_GEOMETRY,
                )

                # ★ THE CONNECTION: Widget → Logic → lib
                hotspots = self.logic.findHottestVoxels(pet_node, label_node, top_n)
                slicer.mrmlScene.RemoveNode(label_node)

                for rank, h in enumerate(hotspots, start=1):
                    row = self.resultsTable.rowCount
                    self.resultsTable.insertRow(row)
                    values = [
                        segment_name,
                        str(rank),
                        f"{h['suv']:.4f}",
                        f"{h['ras_x']:.1f}",
                        f"{h['ras_y']:.1f}",
                        f"{h['ras_z']:.1f}",
                    ]
                    for col, val in enumerate(values):
                        item = qt.QTableWidgetItem(val)
                        item.setTextAlignment(qt.Qt.AlignCenter)
                        self.resultsTable.setItem(row, col, item)
                    self._row_coords.append((h["ras_x"], h["ras_y"], h["ras_z"]))

            except Exception as e:
                LOG.warning(f"[PETHotspotNav] {segment_name}: {e}")
                errors.append(segment_name)

        self.findBtn.setEnabled(True)
        n_rows = self.resultsTable.rowCount
        summary = (
            f"Done — {n_segs} segment(s), {n_rows} row(s). "
            "Click a row to jump there."
        )
        if errors:
            summary += f"  Errors on: {', '.join(errors)}."
        self.statusLabel.setText(summary)

        if n_rows > 0:
            self.onRowClicked(0, 0)
            self.resultsTable.selectRow(0)

    def onRowClicked(self, row, col):
        """UI-only: jump slices + move marker. No lib call needed."""
        if row < 0 or row >= len(self._row_coords):
            return

        ras_x, ras_y, ras_z = self._row_coords[row]
        slicer.modules.markups.logic().JumpSlicesToLocation(ras_x, ras_y, ras_z, True)

        if self._marker_node is None or not slicer.mrmlScene.IsNodePresent(self._marker_node):
            self._marker_node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsFiducialNode", "HotspotMarker"
            )
            disp = self._marker_node.GetDisplayNode()
            if disp:
                disp.SetColor(1.0, 0.0, 0.0)
                disp.SetSelectedColor(1.0, 0.0, 0.0)
                disp.SetActiveColor(1.0, 0.3, 0.3)
                disp.SetGlyphScale(4.0)
                disp.SetTextScale(0.0)
                disp.SetSliceProjection(False)

        try:
            self._marker_node.RemoveAllControlPoints()
            self._marker_node.AddControlPoint(vtk.vtkVector3d(ras_x, ras_y, ras_z))
        except AttributeError:
            self._marker_node.RemoveAllMarkups()
            self._marker_node.AddFiducial(ras_x, ras_y, ras_z)

        seg_name = (
            self.resultsTable.item(row, 0).text()
            if self.resultsTable.item(row, 0)
            else "?"
        )
        self.statusLabel.setText(
            f"[{seg_name}] RAS = ({ras_x:.1f}, {ras_y:.1f}, {ras_z:.1f}) mm"
        )
