"""
PETCTQuantAnalysis_v2 — Module + Widget.

  Widget → Logic (Slicer I/O + CLI) → lib (paths / excel / SUV factor)
"""
"""
PETCTQuantAnalysis_v2.py
========================
3D Slicer Scripted Module — PET/CT Quantitative Analysis  (Version 2)

CHANGES FROM v1:
  - Segmentations are auto-discovered from the Segments folder (no fixed organ list).
  - After detection, a table shows every .nii.gz found across all patients,
    how many scans have each file, and lets the user tick/untick and rename
    each one for the Excel header.
  - runBatch iterates over the user-selected segmentation files dynamically.
  - Excel columns are named after the user-supplied display labels.
  - Full verbose logging throughout (same depth as v1).

DIRECTORY STRUCTURE (same as v1):
    root/
      CT/       {SubjectID}_{YYYY-MM-DD}_CT/   *.dcm
      PET/      {SubjectID}_{YYYY-MM-DD}_PET/  *.dcm
      Segments/ {SubjectID}_{YYYY-MM-DD}_Seg/
                  *.nii.gz  ← any number of segmentation files
"""

import os
import re
import logging
import vtk
import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import *
import openpyxl
from datetime import datetime
from pathlib import Path
from collections import defaultdict

LOG = logging.getLogger("PETCTQuant")

# Files that are pre-unchecked by default (internal / non-segmentation files).
# Users can still enable them manually.
_DEFAULT_SKIP = {
    "combined_seg", "body_trunc", "body_extremities",
    "body", "skin", "ct",
}


# ── MODULE REGISTRATION ────────────────────────────────────────────────────


try:
    from PETCTQuantAnalysisLib.PETCTQuantAnalysisLogic import PETCTQuantAnalysisLogic
except ImportError:
    import importlib.util, os as _os
    _p = _os.path.join(_os.path.dirname(__file__), "PETCTQuantAnalysisLib", "PETCTQuantAnalysisLogic.py")
    _spec = importlib.util.spec_from_file_location("PETCTQuantAnalysisLogic", _p)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    PETCTQuantAnalysisLogic = getattr(_mod, "PETCTQuantAnalysisLogic")

class PETCTQuantAnalysis(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title       = "7. Quantitative Analysis"
        parent.categories  = ["Metabolic Syndrome Toolkit"]
        parent.dependencies = ["QuantitativeIndicesCLI"]
        parent.contributors = ["Research Lab"]
        parent.helpText = (
            "Batch PET/CT quantification. "
            "Automatically discovers all segmentation files and lets you "
            "rename them before writing to Excel."
        )


class PETCTQuantAnalysisWidget(ScriptedLoadableModuleWidget):

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = PETCTQuantAnalysisLogic()
        self._scans = []          # list of scan dicts from detectScans
        self._seg_rows = []       # list of (stem, cb_widget, name_edit)

        # ── 1. DATA INPUT ──────────────────────────────────────────────
        col1 = ctk.ctkCollapsibleButton()
        col1.text = "1. Data Input"
        self.layout.addWidget(col1)
        lay1 = qt.QFormLayout(col1)

        self.rootEdit = ctk.ctkPathLineEdit()
        self.rootEdit.filters = ctk.ctkPathLineEdit.Dirs
        lay1.addRow("Root folder:", self.rootEdit)

        self.detectBtn = qt.QPushButton("Detect patients + segmentations")
        lay1.addRow("", self.detectBtn)

        self.scansLabel = qt.QLabel("No folder selected")
        lay1.addRow("Scans found:", self.scansLabel)

        # ── 2. SEGMENTATIONS FOUND ────────────────────────────────────
        col2 = ctk.ctkCollapsibleButton()
        col2.text = "2. Segmentations Found"
        self.layout.addWidget(col2)
        lay2 = qt.QVBoxLayout(col2)

        helpLabel = qt.QLabel(
            "Tick the files you want to analyse.\n"
            "Edit 'Excel label' to rename the column header "
            "(e.g. VF instead of visceral_fat)."
        )
        helpLabel.setWordWrap(True)
        lay2.addWidget(helpLabel)

        # Table: checkbox | filename | coverage | excel label
        self.segTable = qt.QTableWidget(0, 4)
        self.segTable.setHorizontalHeaderLabels(
            ["Include", "File (stem)", "Patients", "Excel label"]
        )
        self.segTable.horizontalHeader().setStretchLastSection(True)
        self.segTable.setSelectionMode(qt.QAbstractItemView.NoSelection)
        self.segTable.setFixedHeight(180)
        lay2.addWidget(self.segTable)

        # ── 3. COMPUTATION ────────────────────────────────────────────
        col3 = ctk.ctkCollapsibleButton()
        col3.text = "3. Computation"
        self.layout.addWidget(col3)
        lay3 = qt.QFormLayout(col3)

        suvWidget = qt.QWidget()
        suvBox = qt.QHBoxLayout(suvWidget)
        self.suvBwRadio  = qt.QRadioButton("SUVbw (body weight)")
        self.suvLbmRadio = qt.QRadioButton("SUVlbm (lean body mass)")
        self.suvBwRadio.setChecked(True)
        for r in (self.suvBwRadio, self.suvLbmRadio):
            suvBox.addWidget(r)
        lay3.addRow("SUV type:", suvWidget)

        mWidget = qt.QWidget()
        mBox = qt.QHBoxLayout(mWidget)
        self.cbMean   = qt.QCheckBox("SUVmean")
        self.cbMax    = qt.QCheckBox("SUVmax")
        self.cbPeak   = qt.QCheckBox("SUVpeak")
        self.cbTLG    = qt.QCheckBox("TLG")
        self.cbVolume = qt.QCheckBox("Volume")
        for cb in (self.cbMean, self.cbMax, self.cbPeak, self.cbTLG, self.cbVolume):
            cb.setChecked(True)
            mBox.addWidget(cb)
        lay3.addRow("Metrics:", mWidget)

        # ── 4. EXPORT ─────────────────────────────────────────────────
        col4 = ctk.ctkCollapsibleButton()
        col4.text = "4. Export"
        self.layout.addWidget(col4)
        lay4 = qt.QFormLayout(col4)

        self.outputEdit = ctk.ctkPathLineEdit()
        self.outputEdit.filters   = ctk.ctkPathLineEdit.Files
        self.outputEdit.nameFilters = ["Excel files (*.xlsx)"]
        lay4.addRow("Output Excel:", self.outputEdit)

        self.appendCb = qt.QCheckBox("Append to existing file")
        self.appendCb.setChecked(True)
        lay4.addRow("", self.appendCb)

        # ── 5. PREVIEW ────────────────────────────────────────────────
        # ── RUN CONTROLS ──────────────────────────────────────────────
        self.startBtn = qt.QPushButton("START — run all patients")
        self.startBtn.setStyleSheet(
            "QPushButton{background:#2196F3;color:white;font-weight:bold;"
            "padding:8px;border-radius:4px}"
            "QPushButton:hover{background:#1976D2}"
            "QPushButton:disabled{background:#BDBDBD}"
        )
        self.layout.addWidget(self.startBtn)

        self.cancelBtn = qt.QPushButton("Cancel")
        self.cancelBtn.setEnabled(False)
        self.layout.addWidget(self.cancelBtn)

        self.progressBar = qt.QProgressBar()
        self.progressBar.setRange(0, 100)
        self.layout.addWidget(self.progressBar)

        self.statusLabel = qt.QLabel("Ready")
        self.statusLabel.setWordWrap(True)
        self.layout.addWidget(self.statusLabel)

        self.layout.addStretch(1)

        # connections
        self.detectBtn.connect('clicked(bool)', self.onDetect)
        self.startBtn.connect('clicked(bool)', self.onStart)
        self.cancelBtn.connect('clicked(bool)', self.onCancel)
        self._cancel = False

    # ── Detection ─────────────────────────────────────────────────────────

    def onDetect(self):
        root = self.rootEdit.currentPath
        if not root or not os.path.isdir(root):
            self.scansLabel.setText("Invalid folder")
            return

        self._scans = self.logic.detectScans(root)
        n_subj = len(set(s["subject_id"] for s in self._scans))
        self.scansLabel.setText(
            f"{len(self._scans)} scan(s) across {n_subj} subject(s)"
        )

        seg_info = self.logic.detectSegmentations(root, self._scans)
        self._buildSegTable(seg_info)

    def _buildSegTable(self, seg_info):
        """
        seg_info: dict  stem → {"count": int, "total": int}
        Fills the table with one row per discovered file stem.
        """
        self._seg_rows.clear()
        self.segTable.setRowCount(0)
        total = seg_info.get("__total__", 1)

        for stem, info in sorted(seg_info.items()):
            if stem == "__total__":
                continue

            row_idx = self.segTable.rowCount
            self.segTable.insertRow(row_idx)

            # Checkbox
            cb = qt.QCheckBox()
            cb.setChecked(stem not in _DEFAULT_SKIP)
            cb_widget = qt.QWidget()
            cb_layout = qt.QHBoxLayout(cb_widget)
            cb_layout.addWidget(cb)
            cb_layout.setAlignment(qt.Qt.AlignCenter)
            cb_layout.setContentsMargins(0, 0, 0, 0)
            self.segTable.setCellWidget(row_idx, 0, cb_widget)

            # Filename stem (read-only)
            item_name = qt.QTableWidgetItem(stem)
            item_name.setFlags(qt.Qt.ItemIsEnabled)
            self.segTable.setItem(row_idx, 1, item_name)

            # Coverage
            coverage = f"{info['count']}/{info['total']}"
            item_cov = qt.QTableWidgetItem(coverage)
            item_cov.setFlags(qt.Qt.ItemIsEnabled)
            self.segTable.setItem(row_idx, 2, item_cov)

            # Editable display name
            name_edit = qt.QLineEdit(stem)
            self.segTable.setCellWidget(row_idx, 3, name_edit)

            self._seg_rows.append((stem, cb, name_edit))

        self.segTable.resizeColumnsToContents()

    def _getSegNameMap(self):
        """
        Returns {stem: display_name} for all checked rows.
        display_name defaults to stem if the edit is blank.
        """
        name_map = {}
        for stem, cb, name_edit in self._seg_rows:
            if cb.isChecked():
                label = name_edit.text.strip() or stem
                name_map[stem] = label
        return name_map

    # ── Run ───────────────────────────────────────────────────────────────

    def onStart(self):
        root        = self.rootEdit.currentPath
        output_file = self.outputEdit.currentPath

        if not root or not os.path.isdir(root):
            slicer.util.errorDisplay("Please select a valid root folder.")
            return
        if not output_file:
            slicer.util.errorDisplay("Please select an output Excel file.")
            return
        if not self._scans:
            slicer.util.errorDisplay("Click 'Detect' first.")
            return

        seg_name_map = self._getSegNameMap()
        if not seg_name_map:
            slicer.util.errorDisplay(
                "No segmentations selected. "
                "Run Detect and tick at least one segmentation file."
            )
            return

        metrics = {
            "mean":   self.cbMean.isChecked(),
            "max":    self.cbMax.isChecked(),
            "peak":   self.cbPeak.isChecked(),
            "tlg":    self.cbTLG.isChecked(),
            "volume": self.cbVolume.isChecked(),
        }
        suv_type = "bw" if self.suvBwRadio.isChecked() else "lbm"

        self.startBtn.setEnabled(False)
        self.cancelBtn.setEnabled(True)
        self._cancel = False

        try:
            self.logic.runBatch(
                root_folder     = root,
                scans           = self._scans,
                seg_name_map    = seg_name_map,
                output_file     = output_file,
                metrics         = metrics,
                suv_type        = suv_type,
                append          = self.appendCb.isChecked(),
                progress_cb     = lambda v: (self.progressBar.setValue(int(v)),
                                             slicer.app.processEvents()),
                status_cb       = lambda m: (self.statusLabel.setText(m),
                                             slicer.app.processEvents()),
                cancel_check    = lambda: self._cancel,
            )
            slicer.util.infoDisplay(f"Done!\nResults saved to:\n{output_file}")
        except Exception as e:
            slicer.util.errorDisplay(f"Error: {e}")
            LOG.error(f"[PETCTQuant-v2] top-level error: {e}", exc_info=True)
        finally:
            self.startBtn.setEnabled(True)
            self.cancelBtn.setEnabled(False)

    def onCancel(self):
        self._cancel = True
        self.statusLabel.setText("Cancelling after current scan …")

