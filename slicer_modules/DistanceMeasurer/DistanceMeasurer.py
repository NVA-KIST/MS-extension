"""
Distance Measurer — Module + Widget (UI + triggers).

  Widget  →  DistanceMeasurerLogic  →  lib/quantification/distance.py
"""
from __future__ import annotations

import os

import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
)

try:
    from DistanceMeasurerLogic import DistanceMeasurerLogic
except ImportError:
    import importlib.util

    _p = os.path.join(os.path.dirname(__file__), "DistanceMeasurerLogic.py")
    _spec = importlib.util.spec_from_file_location("DistanceMeasurerLogic", _p)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    DistanceMeasurerLogic = getattr(_mod, "DistanceMeasurerLogic")

_PALETTE = [
    (1.00, 0.27, 0.27),
    (0.20, 0.72, 1.00),
    (1.00, 0.80, 0.00),
    (0.20, 1.00, 0.40),
    (1.00, 0.55, 0.00),
    (0.78, 0.20, 1.00),
    (1.00, 0.40, 0.80),
    (0.40, 0.90, 0.90),
]


class DistanceMeasurer(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "6. Distance Measurer"
        self.parent.categories = ["Metabolic Syndrome Toolkit"]
        self.parent.dependencies = []
        self.parent.contributors = ["IshitaSinghFaujdar"]
        self.parent.helpText = (
            "Place ruler lines in any view and read straight-line distances "
            "in mm, cm, or voxels."
        )
        self.parent.acknowledgementText = ""


class DistanceMeasurerWidget(ScriptedLoadableModuleWidget):

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = DistanceMeasurerLogic()
        self._rows = []
        self._meas_count = 0

        settingsBox = ctk.ctkCollapsibleButton()
        settingsBox.text = "Settings"
        self.layout.addWidget(settingsBox)
        settingsForm = qt.QFormLayout(settingsBox)

        self.unitCombo = qt.QComboBox()
        for u in ("mm", "cm", "voxels"):
            self.unitCombo.addItem(u)
        self.unitCombo.setCurrentIndex(0)
        self.unitCombo.currentIndexChanged.connect(self._refresh_all_distances)
        settingsForm.addRow("Unit:", self.unitCombo)

        self.refVolSelector = slicer.qMRMLNodeComboBox()
        self.refVolSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.refVolSelector.addEnabled = False
        self.refVolSelector.removeEnabled = False
        self.refVolSelector.noneEnabled = True
        self.refVolSelector.setMRMLScene(slicer.mrmlScene)
        self.refVolSelector.currentNodeChanged.connect(self._refresh_all_distances)
        settingsForm.addRow("Reference volume (voxels):", self.refVolSelector)

        vols = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
        if vols:
            self.refVolSelector.setCurrentNode(vols[0])

        measBox = ctk.ctkCollapsibleButton()
        measBox.text = "Measurements"
        self.layout.addWidget(measBox)
        measLayout = qt.QVBoxLayout(measBox)

        hdr = qt.QHBoxLayout()
        for txt, w in [("", 18), ("Name", 120), ("Distance", 110), ("", 28)]:
            lbl = qt.QLabel(txt)
            lbl.setStyleSheet("font-weight:bold;")
            if w:
                lbl.setFixedWidth(w)
            hdr.addWidget(lbl)
        hdr.addStretch(1)
        measLayout.addLayout(hdr)

        self._measContainer = qt.QWidget()
        self._measContainerLayout = qt.QVBoxLayout(self._measContainer)
        self._measContainerLayout.setContentsMargins(0, 0, 0, 0)
        self._measContainerLayout.setSpacing(4)
        measLayout.addWidget(self._measContainer)

        self._noMeasLabel = qt.QLabel('Click "New Measurement" to start placing a ruler line.')
        self._noMeasLabel.setStyleSheet("color:#666; font-style:italic;")
        measLayout.addWidget(self._noMeasLabel)

        btnRow = qt.QHBoxLayout()
        newBtn = qt.QPushButton("New Measurement")
        newBtn.setStyleSheet(
            "QPushButton{background:#1565c0;color:white;font-weight:bold;"
            "padding:6px 12px;border-radius:4px;}"
            "QPushButton:hover{background:#0d47a1;}"
        )
        newBtn.clicked.connect(self._on_new_measurement)
        btnRow.addWidget(newBtn)

        clearBtn = qt.QPushButton("Clear All")
        clearBtn.setStyleSheet(
            "QPushButton{background:#b71c1c;color:white;font-weight:bold;"
            "padding:6px 12px;border-radius:4px;}"
            "QPushButton:hover{background:#7f0000;}"
        )
        clearBtn.clicked.connect(self._on_clear_all)
        btnRow.addWidget(clearBtn)
        measLayout.addLayout(btnRow)

        self.layout.addStretch(1)
        self._sceneObserverTag = slicer.mrmlScene.AddObserver(
            slicer.vtkMRMLScene.NodeRemovedEvent, self._on_scene_node_removed
        )

    def cleanup(self):
        try:
            interactionNode = slicer.app.applicationLogic().GetInteractionNode()
            interactionNode.SetCurrentInteractionMode(
                slicer.vtkMRMLInteractionNode.ViewTransform
            )
        except Exception:
            pass
        slicer.mrmlScene.RemoveObserver(self._sceneObserverTag)
        for rd in list(self._rows):
            self._delete_row(rd, remove_node=False)

    def _on_new_measurement(self):
        self._meas_count += 1
        name = f"Measurement {self._meas_count}"
        color = _PALETTE[(self._meas_count - 1) % len(_PALETTE)]

        line_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", name)
        dn = line_node.GetDisplayNode()
        if dn:
            dn.SetSelectedColor(*color)
            dn.SetColor(*color)
            dn.SetLineThickness(0.5)
            dn.SetTextScale(3.0)
            dn.SetVisibility(True)

        rd = self._add_row(line_node, name, color)
        obs1 = line_node.AddObserver(
            slicer.vtkMRMLMarkupsNode.PointModifiedEvent,
            lambda c, e, r=rd: self._update_distance(r),
        )
        obs2 = line_node.AddObserver(
            slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent,
            lambda c, e, r=rd: self._update_distance(r),
        )
        rd["observers"] = [obs1, obs2]

        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        interactionNode.SetCurrentInteractionMode(
            slicer.vtkMRMLInteractionNode.ViewTransform
        )
        slicer.modules.markups.logic().SetActiveListID(line_node)
        interactionNode.SetCurrentInteractionMode(slicer.vtkMRMLInteractionNode.Place)
        self._noMeasLabel.setVisible(False)

    def _add_row(self, line_node, name, color):
        frame = qt.QFrame()
        frame.setFrameShape(qt.QFrame.StyledPanel)
        rowLayout = qt.QHBoxLayout(frame)
        rowLayout.setContentsMargins(4, 2, 4, 2)
        rowLayout.setSpacing(6)

        swatch = qt.QPushButton()
        swatch.setFixedSize(18, 18)
        r, g, b = [int(c * 255) for c in color]
        swatch.setStyleSheet(
            f"QPushButton{{background:rgb({r},{g},{b});"
            f"border:1px solid #333;border-radius:3px;}}"
        )

        name_edit = qt.QLineEdit(name)
        name_edit.setFixedWidth(120)
        name_edit.editingFinished.connect(
            lambda n=name_edit, node=line_node: node.SetName(n.text)
        )

        dist_lbl = qt.QLabel("-")
        dist_lbl.setFixedWidth(110)
        dist_lbl.setStyleSheet("font-size:14px; font-weight:bold; color:#0d47a1;")

        rm_btn = qt.QPushButton("x")
        rm_btn.setFixedWidth(28)
        rm_btn.setStyleSheet(
            "QPushButton{color:red;font-weight:bold;font-size:14px;}"
            "QPushButton:hover{background:#ffebee;}"
        )

        rowLayout.addWidget(swatch)
        rowLayout.addWidget(name_edit)
        rowLayout.addWidget(dist_lbl)
        rowLayout.addStretch(1)
        rowLayout.addWidget(rm_btn)

        rd = {
            "widget": frame,
            "node": line_node,
            "name_edit": name_edit,
            "dist_lbl": dist_lbl,
            "swatch": swatch,
            "color": list(color),
            "observers": [],
        }
        swatch.clicked.connect(lambda checked=False, r=rd: self._pick_color(r))
        rm_btn.clicked.connect(lambda checked=False, r=rd: self._delete_row(r))
        self._measContainerLayout.addWidget(frame)
        self._rows.append(rd)
        return rd

    def _update_distance(self, rd):
        """UI readout — math goes through Logic → lib."""
        node = rd["node"]
        lbl = rd["dist_lbl"]

        if node.GetNumberOfDefinedControlPoints() < 2:
            lbl.setText("placing...")
            lbl.setStyleSheet("font-size:13px; font-weight:normal; color:#888;")
            return

        length_mm = node.GetLineLengthWorld()
        unit = self.unitCombo.currentText
        voxel_length = None
        if unit == "voxels":
            ref = self.refVolSelector.currentNode()
            if ref is not None:
                voxel_length = self.logic.voxelDistance(node, ref)

        text = self.logic.formatDistance(length_mm, unit, voxel_length)
        lbl.setText(text)
        lbl.setStyleSheet("font-size:14px; font-weight:bold; color:#0d47a1;")

    def _refresh_all_distances(self):
        for rd in self._rows:
            self._update_distance(rd)

    def _pick_color(self, rd):
        cur = rd["color"]
        qcur = qt.QColor(int(cur[0] * 255), int(cur[1] * 255), int(cur[2] * 255))
        chosen = qt.QColorDialog.getColor(qcur, None, "Choose line colour")
        if not chosen.isValid():
            return
        r, g, b = chosen.redF(), chosen.greenF(), chosen.blueF()
        rd["color"] = [r, g, b]
        ri, gi, bi = chosen.red(), chosen.green(), chosen.blue()
        rd["swatch"].setStyleSheet(
            f"QPushButton{{background:rgb({ri},{gi},{bi});"
            f"border:1px solid #333;border-radius:3px;}}"
        )
        dn = rd["node"].GetDisplayNode()
        if dn:
            dn.SetSelectedColor(r, g, b)
            dn.SetColor(r, g, b)

    def _delete_row(self, rd, remove_node=True):
        for tag in rd.get("observers", []):
            try:
                rd["node"].RemoveObserver(tag)
            except Exception:
                pass
        if remove_node:
            try:
                slicer.mrmlScene.RemoveNode(rd["node"])
            except Exception:
                pass
        rd["widget"].setParent(None)
        if rd in self._rows:
            self._rows.remove(rd)
        if not self._rows:
            self._noMeasLabel.setVisible(True)

    def _on_clear_all(self):
        for rd in list(self._rows):
            self._delete_row(rd)

    def _on_scene_node_removed(self, caller, event):
        removed_ids = {
            rd["node"].GetID()
            for rd in self._rows
            if not slicer.mrmlScene.GetNodeByID(rd["node"].GetID())
        }
        for rd in list(self._rows):
            if rd["node"].GetID() in removed_ids:
                self._delete_row(rd, remove_node=False)
