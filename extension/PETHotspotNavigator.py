"""
PETHotspotNavigator.py
======================
3D Slicer Scripted Module — PET Hotspot Navigator

Works entirely with nodes already loaded in the Slicer scene.
Supports any number of patients loaded at once: just pick the right
PET volume + segmentation pair from the dropdowns.

For each segment in the selected segmentation node, the module finds
the voxel(s) with the highest PET value and lists them in a table.
Clicking any row recentres all three slice views on that RAS coordinate.
"""

import logging
import vtk
import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import *

LOG = logging.getLogger("PETHotspotNav")


# ── MODULE REGISTRATION ────────────────────────────────────────────────────

class PETHotspotNavigator(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title       = "PET Hotspot Navigator"
        parent.categories  = ["Quantification"]
        parent.dependencies = []
        parent.contributors = ["Research Lab"]
        parent.helpText = (
            "Select a PET volume and a segmentation node already in the scene. "
            "Click 'Find Hotspots' to locate the highest-SUV voxel in every segment. "
            "Click any row in the results table to jump the slice views to that location."
        )


# ── WIDGET ─────────────────────────────────────────────────────────────────

class PETHotspotNavigatorWidget(ScriptedLoadableModuleWidget):

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic        = PETHotspotNavigatorLogic()
        self._row_coords  = []    # (ras_x, ras_y, ras_z) per table row
        self._marker_node = None  # single red markup point, repositioned on each click

        # ── 1. Volume / segmentation selectors ────────────────────────
        col1 = ctk.ctkCollapsibleButton()
        col1.text = "1. Select data (already loaded in scene)"
        self.layout.addWidget(col1)
        lay1 = qt.QFormLayout(col1)

        self.petSelector = slicer.qMRMLNodeComboBox()
        self.petSelector.nodeTypes            = ["vtkMRMLScalarVolumeNode"]
        self.petSelector.selectNodeUponCreation = True
        self.petSelector.addEnabled           = False
        self.petSelector.removeEnabled        = False
        self.petSelector.noneEnabled          = True
        self.petSelector.showHidden           = False
        self.petSelector.setMRMLScene(slicer.mrmlScene)
        self.petSelector.setToolTip("Choose the PET volume from the scene")
        lay1.addRow("PET volume:", self.petSelector)

        self.segSelector = slicer.qMRMLNodeComboBox()
        self.segSelector.nodeTypes            = ["vtkMRMLSegmentationNode"]
        self.segSelector.selectNodeUponCreation = True
        self.segSelector.addEnabled           = False
        self.segSelector.removeEnabled        = False
        self.segSelector.noneEnabled          = True
        self.segSelector.showHidden           = False
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

        # ── Find button ───────────────────────────────────────────────
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

        # ── Results table ─────────────────────────────────────────────
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

        # connections
        self.findBtn.connect('clicked(bool)', self.onFind)
        self.resultsTable.connect('cellClicked(int,int)', self.onRowClicked)

    # ── Handlers ──────────────────────────────────────────────────────────

    def onFind(self):
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
            segment_id   = segmentation.GetNthSegmentID(seg_idx)
            segment_name = segmentation.GetSegment(segment_id).GetName()

            self.statusLabel.setText(
                f"Segment {seg_idx + 1}/{n_segs}: {segment_name}…"
            )
            slicer.app.processEvents()

            try:
                label_node = slicer.mrmlScene.AddNewNodeByClass(
                    'vtkMRMLLabelMapVolumeNode', f'_hs_tmp_{seg_idx}'
                )
                seg_ids = vtk.vtkStringArray()
                seg_ids.InsertNextValue(segment_id)
                slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
                    seg_node, seg_ids, label_node, pet_node,
                    slicer.vtkSegmentation.EXTENT_REFERENCE_GEOMETRY,
                )

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

                    self._row_coords.append(
                        (h["ras_x"], h["ras_y"], h["ras_z"])
                    )

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

        # Auto-jump to the top row if there is one
        if n_rows > 0:
            self.onRowClicked(0, 0)
            self.resultsTable.selectRow(0)

    def onRowClicked(self, row, col):
        if row < 0 or row >= len(self._row_coords):
            return

        ras_x, ras_y, ras_z = self._row_coords[row]

        # Centre all three slice views on this coordinate
        slicer.modules.markups.logic().JumpSlicesToLocation(
            ras_x, ras_y, ras_z, True   # True = centred view
        )

        # Create the red marker node once; reuse it for every subsequent click
        if self._marker_node is None or not slicer.mrmlScene.IsNodePresent(self._marker_node):
            self._marker_node = slicer.mrmlScene.AddNewNodeByClass(
                'vtkMRMLMarkupsFiducialNode', 'HotspotMarker'
            )
            disp = self._marker_node.GetDisplayNode()
            if disp:
                disp.SetColor(1.0, 0.0, 0.0)           # red (unselected)
                disp.SetSelectedColor(1.0, 0.0, 0.0)   # red (selected)
                disp.SetActiveColor(1.0, 0.3, 0.3)     # lighter red (hover)
                disp.SetGlyphScale(4.0)
                disp.SetTextScale(0.0)                  # no floating label on the dot
                disp.SetSliceProjection(False)          # only show on the exact slice — disappears when scrolling past

        # Move the marker to the new position
        try:
            self._marker_node.RemoveAllControlPoints()
            self._marker_node.AddControlPoint(vtk.vtkVector3d(ras_x, ras_y, ras_z))
        except AttributeError:
            self._marker_node.RemoveAllMarkups()
            self._marker_node.AddFiducial(ras_x, ras_y, ras_z)

        seg_name = self.resultsTable.item(row, 0).text() if self.resultsTable.item(row, 0) else "?"
        self.statusLabel.setText(
            f"[{seg_name}] RAS = ({ras_x:.1f}, {ras_y:.1f}, {ras_z:.1f}) mm"
        )


# ── LOGIC ──────────────────────────────────────────────────────────────────

class PETHotspotNavigatorLogic(ScriptedLoadableModuleLogic):

    def findHottestVoxels(self, pet_node, label_node, top_n=10):
        """
        Return up to top_n dicts sorted descending by SUV.
        Each dict: {suv, ras_x, ras_y, ras_z}

        arrayFromVolume shape is (Z, Y, X) = (K, J, I).
        GetIJKToRASMatrix expects the column vector [I, J, K, 1]^T.
        """
        import numpy as np

        pet_arr   = slicer.util.arrayFromVolume(pet_node)
        label_arr = slicer.util.arrayFromVolume(label_node)

        mask_idx = np.argwhere(label_arr > 0)   # shape (N, 3): [z, y, x]
        if len(mask_idx) == 0:
            return []

        pet_vals = pet_arr[mask_idx[:, 0], mask_idx[:, 1], mask_idx[:, 2]]

        order   = np.argsort(pet_vals)[::-1]
        top_idx = order[:top_n]

        ijk_to_ras = vtk.vtkMatrix4x4()
        pet_node.GetIJKToRASMatrix(ijk_to_ras)

        hotspots = []
        for idx in top_idx:
            z, y, x = mask_idx[idx]
            ras_h   = ijk_to_ras.MultiplyPoint([float(x), float(y), float(z), 1.0])
            hotspots.append({
                "suv":   float(pet_vals[idx]),
                "ras_x": float(ras_h[0]),
                "ras_y": float(ras_h[1]),
                "ras_z": float(ras_h[2]),
            })

        return hotspots


# ── TEST ───────────────────────────────────────────────────────────────────

class PETHotspotNavigatorTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        self.delayDisplay("PETHotspotNavigator — no automated test defined")
