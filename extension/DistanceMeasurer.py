"""
DistanceMeasurer — 3D Slicer scripted module
============================================

Drag-and-drop this file into 3D Slicer, then find it under:
    Modules → Quantification → Distance Measurer

Usage
-----
1. Click "New Measurement".
2. Click once in any slice or 3D view to place the first point, then click
   again (or click-drag) to place the second point.  The line snaps into
   place and the distance appears instantly.
3. Drag either endpoint at any time — the distance updates live.
4. Choose mm / cm / voxels from the unit selector at the top.
   (Voxels requires a reference volume to be selected.)
5. Click the colour swatch on any row to change that line's colour.
6. Hit × to delete a measurement.
"""

import math
import vtk
import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
)

# Colours cycled automatically for successive measurements
_PALETTE = [
    (1.00, 0.27, 0.27),   # red
    (0.20, 0.72, 1.00),   # cyan
    (1.00, 0.80, 0.00),   # yellow
    (0.20, 1.00, 0.40),   # green
    (1.00, 0.55, 0.00),   # orange
    (0.78, 0.20, 1.00),   # purple
    (1.00, 0.40, 0.80),   # pink
    (0.40, 0.90, 0.90),   # teal
]


# ── Module metadata ────────────────────────────────────────────────────────────

class DistanceMeasurer(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title        = "Distance Measurer"
        self.parent.categories   = ["Quantification"]
        self.parent.dependencies = []
        self.parent.contributors = ["IshitaSinghFaujdar"]
        self.parent.helpText = (
            "Place ruler lines in any view and read straight-line distances "
            "in mm, cm, or voxels.  All measurements update live as you drag "
            "the endpoints."
        )
        self.parent.acknowledgementText = ""


# ── Widget ─────────────────────────────────────────────────────────────────────

class DistanceMeasurerWidget(ScriptedLoadableModuleWidget):

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        self._rows        = []   # list of row dicts
        self._meas_count  = 0   # used to auto-name measurements

        # ── Settings ──────────────────────────────────────────────────────────
        settingsBox = ctk.ctkCollapsibleButton()
        settingsBox.text = "Settings"
        self.layout.addWidget(settingsBox)
        settingsForm = qt.QFormLayout(settingsBox)

        self.unitCombo = qt.QComboBox()
        for u in ("mm", "cm", "voxels"):
            self.unitCombo.addItem(u)
        self.unitCombo.setCurrentIndex(0)
        self.unitCombo.setToolTip(
            "Unit for all distance readouts.\n"
            "'voxels' requires a reference volume to be selected below.")
        self.unitCombo.currentIndexChanged.connect(self._refresh_all_distances)
        settingsForm.addRow("Unit:", self.unitCombo)

        self.refVolSelector = slicer.qMRMLNodeComboBox()
        self.refVolSelector.nodeTypes     = ['vtkMRMLScalarVolumeNode']
        self.refVolSelector.addEnabled    = False
        self.refVolSelector.removeEnabled = False
        self.refVolSelector.noneEnabled   = True
        self.refVolSelector.setMRMLScene(slicer.mrmlScene)
        self.refVolSelector.setToolTip(
            "Reference volume used only when unit = voxels.\n"
            "The RAS→IJK matrix of this volume is used for the conversion.")
        self.refVolSelector.currentNodeChanged.connect(self._refresh_all_distances)
        settingsForm.addRow("Reference volume (voxels):", self.refVolSelector)

        # Auto-select the first scalar volume in the scene
        vols = slicer.util.getNodesByClass('vtkMRMLScalarVolumeNode')
        if vols:
            self.refVolSelector.setCurrentNode(vols[0])

        # ── Measurements panel ─────────────────────────────────────────────────
        measBox = ctk.ctkCollapsibleButton()
        measBox.text = "Measurements"
        self.layout.addWidget(measBox)
        measLayout = qt.QVBoxLayout(measBox)

        # Column header
        hdr = qt.QHBoxLayout()
        for txt, w in [("", 18), ("Name", 120), ("Distance", 110), ("", 28)]:
            lbl = qt.QLabel(txt)
            lbl.setStyleSheet("font-weight:bold;")
            if w:
                lbl.setFixedWidth(w)
            hdr.addWidget(lbl)
        hdr.addStretch(1)
        measLayout.addLayout(hdr)

        # Scrollable container for measurement rows
        self._measContainer = qt.QWidget()
        self._measContainerLayout = qt.QVBoxLayout(self._measContainer)
        self._measContainerLayout.setContentsMargins(0, 0, 0, 0)
        self._measContainerLayout.setSpacing(4)
        measLayout.addWidget(self._measContainer)

        self._noMeasLabel = qt.QLabel(
            'Click "New Measurement" to start placing a ruler line.')
        self._noMeasLabel.setStyleSheet("color:#666; font-style:italic;")
        measLayout.addWidget(self._noMeasLabel)

        # Buttons
        btnRow = qt.QHBoxLayout()
        newBtn = qt.QPushButton("⊕  New Measurement")
        newBtn.setStyleSheet(
            "QPushButton{background:#1565c0;color:white;font-weight:bold;"
            "padding:6px 12px;border-radius:4px;}"
            "QPushButton:hover{background:#0d47a1;}")
        newBtn.clicked.connect(self._on_new_measurement)
        btnRow.addWidget(newBtn)

        clearBtn = qt.QPushButton("Clear All")
        clearBtn.setStyleSheet(
            "QPushButton{background:#b71c1c;color:white;font-weight:bold;"
            "padding:6px 12px;border-radius:4px;}"
            "QPushButton:hover{background:#7f0000;}")
        clearBtn.clicked.connect(self._on_clear_all)
        btnRow.addWidget(clearBtn)
        measLayout.addLayout(btnRow)

        self.layout.addStretch(1)

        # Observe scene node-removed events so rows stay in sync if the user
        # deletes a Markups node from the scene directly.
        self._sceneObserverTag = slicer.mrmlScene.AddObserver(
            slicer.vtkMRMLScene.NodeRemovedEvent, self._on_scene_node_removed)

    def cleanup(self):
        # Reset interaction mode so closing this window never leaves placement stuck
        try:
            interactionNode = slicer.app.applicationLogic().GetInteractionNode()
            interactionNode.SetCurrentInteractionMode(
                slicer.vtkMRMLInteractionNode.ViewTransform)
        except Exception:
            pass
        slicer.mrmlScene.RemoveObserver(self._sceneObserverTag)
        for rd in list(self._rows):
            self._delete_row(rd, remove_node=False)

    # ── Measurement creation ──────────────────────────────────────────────────

    def _on_new_measurement(self):
        self._meas_count += 1
        name  = f"Measurement {self._meas_count}"
        color = _PALETTE[(self._meas_count - 1) % len(_PALETTE)]

        # Create the line node
        line_node = slicer.mrmlScene.AddNewNodeByClass(
            'vtkMRMLMarkupsLineNode', name)

        dn = line_node.GetDisplayNode()
        if dn:
            dn.SetSelectedColor(*color)
            dn.SetColor(*color)
            dn.SetLineThickness(0.5)
            dn.SetTextScale(3.0)
            dn.SetVisibility(True)

        # Build the UI row before activating placement so the label is ready
        rd = self._add_row(line_node, name, color)

        # Watch for point moves / additions → update distance
        obs1 = line_node.AddObserver(
            slicer.vtkMRMLMarkupsNode.PointModifiedEvent,
            lambda c, e, r=rd: self._update_distance(r))
        obs2 = line_node.AddObserver(
            slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent,
            lambda c, e, r=rd: self._update_distance(r))
        rd['observers'] = [obs1, obs2]

        # Activate placement mode (click once → point 1, click again → point 2).
        # Cancel any active placement first so switching from another markup type
        # (e.g. fiducial seeds in Vessel Segmenter) is not a no-op.
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        interactionNode.SetCurrentInteractionMode(
            slicer.vtkMRMLInteractionNode.ViewTransform)
        slicer.modules.markups.logic().SetActiveListID(line_node)
        interactionNode.SetCurrentInteractionMode(
            slicer.vtkMRMLInteractionNode.Place)

        self._noMeasLabel.setVisible(False)

    # ── Row builder ───────────────────────────────────────────────────────────

    def _add_row(self, line_node, name, color):
        frame = qt.QFrame()
        frame.setFrameShape(qt.QFrame.StyledPanel)
        rowLayout = qt.QHBoxLayout(frame)
        rowLayout.setContentsMargins(4, 2, 4, 2)
        rowLayout.setSpacing(6)

        # Colour swatch button
        swatch = qt.QPushButton()
        swatch.setFixedSize(18, 18)
        r, g, b = [int(c * 255) for c in color]
        swatch.setStyleSheet(
            f"QPushButton{{background:rgb({r},{g},{b});"
            f"border:1px solid #333;border-radius:3px;}}")
        swatch.setToolTip("Click to change colour")

        # Name editor
        name_edit = qt.QLineEdit(name)
        name_edit.setFixedWidth(120)
        name_edit.setToolTip("Rename this measurement")
        name_edit.editingFinished.connect(
            lambda n=name_edit, node=line_node: node.SetName(n.text))

        # Distance display
        dist_lbl = qt.QLabel("—")
        dist_lbl.setFixedWidth(110)
        dist_lbl.setStyleSheet(
            "font-size:14px; font-weight:bold; color:#0d47a1;")
        dist_lbl.setToolTip("Straight-line distance between the two endpoints")

        # Delete button
        rm_btn = qt.QPushButton("×")
        rm_btn.setFixedWidth(28)
        rm_btn.setStyleSheet(
            "QPushButton{color:red;font-weight:bold;font-size:14px;}"
            "QPushButton:hover{background:#ffebee;}")
        rm_btn.setToolTip("Remove this measurement")

        rowLayout.addWidget(swatch)
        rowLayout.addWidget(name_edit)
        rowLayout.addWidget(dist_lbl)
        rowLayout.addStretch(1)
        rowLayout.addWidget(rm_btn)

        rd = {
            'widget':    frame,
            'node':      line_node,
            'name_edit': name_edit,
            'dist_lbl':  dist_lbl,
            'swatch':    swatch,
            'color':     list(color),
            'observers': [],
        }

        # Wire colour picker
        swatch.clicked.connect(lambda checked=False, r=rd: self._pick_color(r))
        # Wire delete
        rm_btn.clicked.connect(lambda checked=False, r=rd: self._delete_row(r))

        self._measContainerLayout.addWidget(frame)
        self._rows.append(rd)
        return rd

    # ── Distance computation ──────────────────────────────────────────────────

    def _update_distance(self, rd):
        node = rd['node']
        lbl  = rd['dist_lbl']

        if node.GetNumberOfDefinedControlPoints() < 2:
            lbl.setText("placing…")
            lbl.setStyleSheet("font-size:13px; font-weight:normal; color:#888;")
            return

        length_mm = node.GetLineLengthWorld()
        unit      = self.unitCombo.currentText

        if unit == 'mm':
            text = f"{length_mm:.2f} mm"
        elif unit == 'cm':
            text = f"{length_mm / 10.0:.3f} cm"
        else:  # voxels
            ref = self.refVolSelector.currentNode()
            if ref is None:
                text = "— vox (no ref)"
            else:
                text = f"{self._mm_to_vox(node, ref):.2f} vox"

        lbl.setText(text)
        lbl.setStyleSheet(
            "font-size:14px; font-weight:bold; color:#0d47a1;")

    def _mm_to_vox(self, line_node, ref_vol):
        """Convert the line's RAS endpoints to IJK and return Euclidean distance."""
        p1 = [0.0, 0.0, 0.0]
        p2 = [0.0, 0.0, 0.0]
        line_node.GetNthControlPointPositionWorld(0, p1)
        line_node.GetNthControlPointPositionWorld(1, p2)

        mat = vtk.vtkMatrix4x4()
        ref_vol.GetRASToIJKMatrix(mat)

        def to_ijk(p):
            h = mat.MultiplyPoint([p[0], p[1], p[2], 1.0])
            return h[0], h[1], h[2]

        i1, j1, k1 = to_ijk(p1)
        i2, j2, k2 = to_ijk(p2)
        return math.sqrt((i1-i2)**2 + (j1-j2)**2 + (k1-k2)**2)

    def _refresh_all_distances(self):
        for rd in self._rows:
            self._update_distance(rd)

    # ── Colour picker ─────────────────────────────────────────────────────────

    def _pick_color(self, rd):
        cur = rd['color']
        qcur = qt.QColor(int(cur[0]*255), int(cur[1]*255), int(cur[2]*255))
        chosen = qt.QColorDialog.getColor(qcur, None, "Choose line colour")
        if not chosen.isValid():
            return

        r, g, b = chosen.redF(), chosen.greenF(), chosen.blueF()
        rd['color'] = [r, g, b]

        # Update swatch button
        ri, gi, bi = chosen.red(), chosen.green(), chosen.blue()
        rd['swatch'].setStyleSheet(
            f"QPushButton{{background:rgb({ri},{gi},{bi});"
            f"border:1px solid #333;border-radius:3px;}}")

        # Update the Markups display node
        dn = rd['node'].GetDisplayNode()
        if dn:
            dn.SetSelectedColor(r, g, b)
            dn.SetColor(r, g, b)

    # ── Row deletion ──────────────────────────────────────────────────────────

    def _delete_row(self, rd, remove_node=True):
        # Remove observers
        for tag in rd.get('observers', []):
            try:
                rd['node'].RemoveObserver(tag)
            except Exception:
                pass

        # Remove scene node
        if remove_node:
            try:
                slicer.mrmlScene.RemoveNode(rd['node'])
            except Exception:
                pass

        # Remove widget
        rd['widget'].setParent(None)

        if rd in self._rows:
            self._rows.remove(rd)

        if not self._rows:
            self._noMeasLabel.setVisible(True)

    def _on_clear_all(self):
        for rd in list(self._rows):
            self._delete_row(rd)

    # ── Scene sync (handle external node deletion) ────────────────────────────

    def _on_scene_node_removed(self, caller, event):
        removed_ids = {rd['node'].GetID() for rd in self._rows
                       if not slicer.mrmlScene.GetNodeByID(rd['node'].GetID())}
        for rd in list(self._rows):
            if rd['node'].GetID() in removed_ids:
                self._delete_row(rd, remove_node=False)


# ── Logic (thin — all work is in the widget + Markups infrastructure) ──────────

class DistanceMeasurerLogic(ScriptedLoadableModuleLogic):
    pass
