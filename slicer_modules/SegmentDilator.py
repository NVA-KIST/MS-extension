"""
SegmentDilator — Module + Widget.

  Widget → SegmentDilatorLogic → lib/processing/*
"""
"""
SegmentDilator — 3D Slicer scripted module
==========================================

Drag-and-drop this file into 3D Slicer, then find it under:
    Modules → Segmentation → Segment Dilator

Modes
-----
Scene  (sample-patient setup, unchanged)
    1. Add one or more *Dilation Sources* — each with its own dilation
       radius.  The dilated masks are saved to the scene as
       <name>_<seg>_dilated  so you can inspect them.
    2. Add one or more *Target Segments* — the union of all dilated source
       masks is subtracted (voxel-by-voxel) from each target.  Results are
       saved as  <name>_<seg>_subtracted.
    All original nodes are never modified.

Bulk  (apply the sample setup to every patient in a folder)
    Configure the Dilation Sources / Target Segments above on ONE sample
    patient, click "Copy sample setup from above" to derive a list of
    per-patient NIfTI filenames (TotalSegmentator naming convention,
    editable), then run.  For each patient under
    <dataset_root>/Segments/<ID>_Seg/, every source/target file is loaded,
    SegmentDilatorLogic.run() is called UNCHANGED, and the resulting
    '<stem>_dilated.nii.gz' / '<stem>_subtracted.nii.gz' files are written
    back into that patient's Segments folder.  Per-patient errors are
    logged (console + pipeline_logs/bulk_log.txt) and do not stop the run.
"""

import os
import vtk
import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
)


# ── Module metadata ────────────────────────────────────────────────────────────


try:
    from SegmentDilatorLib.SegmentDilatorLogic import SegmentDilatorLogic
except ImportError:
    import importlib.util, os as _os
    _p = _os.path.join(_os.path.dirname(__file__), "SegmentDilatorLib", "SegmentDilatorLogic.py")
    _spec = importlib.util.spec_from_file_location("SegmentDilatorLogic", _p)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    SegmentDilatorLogic = getattr(_mod, "SegmentDilatorLogic")

class SegmentDilator(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title        = "3. Segment Dilator"
        self.parent.categories   = ["Metabolic Syndrome Toolkit"]
        self.parent.dependencies = []
        self.parent.contributors = ["IshitaSinghFaujdar"]
        self.parent.helpText = (
            "Dilate any number of segmentations by a per-source radius (mm), "
            "then subtract the union of those dilated masks from any number of "
            "target segments.  All originals are untouched."
        )
        self.parent.acknowledgementText = ""


class SegmentDilatorWidget(ScriptedLoadableModuleWidget):

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = SegmentDilatorLogic()

        # ── Mode toggle ───────────────────────────────────────────────────────
        modeGroup = qt.QGroupBox("Mode")
        modeRow   = qt.QHBoxLayout(modeGroup)
        self.sceneRadio = qt.QRadioButton("Run on Scene (sample setup)")
        self.bulkRadio  = qt.QRadioButton("Run on Folder (Bulk)")
        self.sceneRadio.setChecked(True)
        modeRow.addWidget(self.sceneRadio)
        modeRow.addWidget(self.bulkRadio)
        self.sceneRadio.toggled.connect(self.onModeToggled)
        self.layout.addWidget(modeGroup)

        # ── Scene panel — sample-patient setup (unchanged) ──────────────────────
        self.scenePanel = qt.QFrame()
        sceneLayout = qt.QVBoxLayout(self.scenePanel)
        sceneLayout.setContentsMargins(0, 0, 0, 0)

        # ══════════════════════════════════════════════════════════════════════
        # Section 1 — Dilation Sources
        # ══════════════════════════════════════════════════════════════════════
        srcBox = ctk.ctkCollapsibleButton()
        srcBox.text = "Dilation Sources  (dilate → save _dilated node)"
        sceneLayout.addWidget(srcBox)
        srcLayout = qt.QVBoxLayout(srcBox)

        # Column header
        srcHdr = qt.QHBoxLayout()
        for txt, stretch in [("Segmentation", 3), ("Sub-segment", 2),
                              ("Dilation mm", 2), ("", 0)]:
            lbl = qt.QLabel(txt)
            lbl.setStyleSheet("font-weight:bold;")
            srcHdr.addWidget(lbl, stretch)
        srcHdr.addSpacing(30)
        srcLayout.addLayout(srcHdr)

        self._srcRows = []
        self._srcContainer = qt.QWidget()
        self._srcContainerLayout = qt.QVBoxLayout(self._srcContainer)
        self._srcContainerLayout.setContentsMargins(0, 0, 0, 0)
        self._srcContainerLayout.setSpacing(3)
        srcLayout.addWidget(self._srcContainer)

        addSrcBtn = qt.QPushButton("+ Add dilation source")
        addSrcBtn.clicked.connect(lambda: self._add_src_row())
        srcLayout.addWidget(addSrcBtn)

        # ══════════════════════════════════════════════════════════════════════
        # Section 2 — Target Segments
        # ══════════════════════════════════════════════════════════════════════
        tgtBox = ctk.ctkCollapsibleButton()
        tgtBox.text = "Target Segments  (receive subtraction → save _subtracted node)"
        sceneLayout.addWidget(tgtBox)
        tgtLayout = qt.QVBoxLayout(tgtBox)

        # Column header
        tgtHdr = qt.QHBoxLayout()
        for txt, stretch in [("Segmentation", 3), ("Sub-segment", 2), ("", 0)]:
            lbl = qt.QLabel(txt)
            lbl.setStyleSheet("font-weight:bold;")
            tgtHdr.addWidget(lbl, stretch)
        tgtHdr.addSpacing(30)
        tgtLayout.addLayout(tgtHdr)

        self._tgtRows = []
        self._tgtContainer = qt.QWidget()
        self._tgtContainerLayout = qt.QVBoxLayout(self._tgtContainer)
        self._tgtContainerLayout.setContentsMargins(0, 0, 0, 0)
        self._tgtContainerLayout.setSpacing(3)
        tgtLayout.addWidget(self._tgtContainer)

        addTgtBtn = qt.QPushButton("+ Add target segment")
        addTgtBtn.clicked.connect(lambda: self._add_tgt_row())
        tgtLayout.addWidget(addTgtBtn)

        # ── Status + Run ──────────────────────────────────────────────────────
        self.statusLabel = qt.QLabel("Status: ready")
        self.statusLabel.setWordWrap(True)
        sceneLayout.addWidget(self.statusLabel)

        self.runButton = qt.QPushButton("Run Dilation + Subtract")
        self.runButton.setStyleSheet(
            "QPushButton{background:#1565c0;color:white;font-weight:bold;"
            "padding:8px;border-radius:4px;font-size:13px;}"
            "QPushButton:hover{background:#0d47a1;}"
            "QPushButton:disabled{background:#888;}"
        )
        self.runButton.clicked.connect(self.onRun)
        sceneLayout.addWidget(self.runButton)

        self.layout.addWidget(self.scenePanel)

        # ══════════════════════════════════════════════════════════════════════
        # Bulk panel — apply the sample setup above to every patient in a folder
        # ══════════════════════════════════════════════════════════════════════
        self.bulkPanel = qt.QFrame()
        bulkLayout = qt.QVBoxLayout(self.bulkPanel)
        bulkLayout.setContentsMargins(0, 4, 0, 0)

        # Filenames detected in the first patient's Segments/<ID>_Seg/ folder —
        # used to populate the source/target filename dropdowns below.
        self._bulkSegFiles = []

        # ── Folder picker ────────────────────────────────────────────────────
        folderRow = qt.QHBoxLayout()
        self.folderEdit = qt.QLineEdit()
        self.folderEdit.setPlaceholderText(
            "Path to dataset root (contains Segments/<ID>_Seg/ folders) …")
        self.folderEdit.textChanged.connect(self.onFolderChanged)
        folderRow.addWidget(self.folderEdit)
        browseBtn = qt.QPushButton("Browse…")
        browseBtn.setFixedWidth(80)
        browseBtn.clicked.connect(self.onBrowse)
        folderRow.addWidget(browseBtn)
        folderForm = qt.QFormLayout()
        folderForm.addRow("Folder:", folderRow)
        bulkLayout.addLayout(folderForm)

        self.skipDoneCheck = qt.QCheckBox("Skip subjects whose outputs already exist")
        self.skipDoneCheck.setChecked(True)
        bulkLayout.addWidget(self.skipDoneCheck)

        self.scanResultLabel = qt.QLabel("No folder selected.")
        self.scanResultLabel.setWordWrap(True)
        self.scanResultLabel.setStyleSheet("color:#555; font-style:italic;")
        bulkLayout.addWidget(self.scanResultLabel)

        # ── Copy sample setup ────────────────────────────────────────────────
        copyBtn = qt.QPushButton("↓ Copy sample setup from Scene tab above")
        copyBtn.setToolTip(
            "Configure Dilation Sources / Target Segments above on ONE sample "
            "patient's loaded scene first, then click here to derive the file "
            "lists below.  Filenames follow the TotalSegmentator naming "
            "convention ('All segments' rows -> '<segmentation node>.nii.gz', "
            "specific sub-segment rows -> '<sub-segment>.nii.gz') and can be "
            "edited per row before running.")
        copyBtn.clicked.connect(self._copy_from_scene)
        bulkLayout.addWidget(copyBtn)

        self.copyResultLabel = qt.QLabel("")
        self.copyResultLabel.setWordWrap(True)
        bulkLayout.addWidget(self.copyResultLabel)

        # ── Bulk dilation source files ───────────────────────────────────────
        bulkLayout.addWidget(qt.QLabel(
            "Dilation source files  (each path relative to a patient's "
            "Segments/<ID>_Seg/ folder):"))

        bSrcHdr = qt.QHBoxLayout()
        for txt, stretch in [("Filename", 4), ("Dilation mm", 2), ("", 0)]:
            lbl = qt.QLabel(txt)
            lbl.setStyleSheet("font-weight:bold;")
            bSrcHdr.addWidget(lbl, stretch)
        bSrcHdr.addSpacing(30)
        bulkLayout.addLayout(bSrcHdr)

        self._bulkSrcRows = []
        self._bulkSrcContainer = qt.QWidget()
        self._bulkSrcContainerLayout = qt.QVBoxLayout(self._bulkSrcContainer)
        self._bulkSrcContainerLayout.setContentsMargins(0, 0, 0, 0)
        self._bulkSrcContainerLayout.setSpacing(3)
        bulkLayout.addWidget(self._bulkSrcContainer)

        addBulkSrcBtn = qt.QPushButton("+ Add source file")
        addBulkSrcBtn.clicked.connect(lambda: self._add_bulk_src_row())
        bulkLayout.addWidget(addBulkSrcBtn)

        # ── Bulk target files ─────────────────────────────────────────────────
        bulkLayout.addWidget(qt.QLabel(
            "Target files  (union of dilated sources is subtracted from each):"))

        bTgtHdr = qt.QHBoxLayout()
        for txt, stretch in [("Filename", 4), ("", 0)]:
            lbl = qt.QLabel(txt)
            lbl.setStyleSheet("font-weight:bold;")
            bTgtHdr.addWidget(lbl, stretch)
        bTgtHdr.addSpacing(30)
        bulkLayout.addLayout(bTgtHdr)

        self._bulkTgtRows = []
        self._bulkTgtContainer = qt.QWidget()
        self._bulkTgtContainerLayout = qt.QVBoxLayout(self._bulkTgtContainer)
        self._bulkTgtContainerLayout.setContentsMargins(0, 0, 0, 0)
        self._bulkTgtContainerLayout.setSpacing(3)
        bulkLayout.addWidget(self._bulkTgtContainer)

        addBulkTgtBtn = qt.QPushButton("+ Add target file")
        addBulkTgtBtn.clicked.connect(lambda: self._add_bulk_tgt_row())
        bulkLayout.addWidget(addBulkTgtBtn)

        # ── Progress + Run ────────────────────────────────────────────────────
        self.bulkProgressBar = qt.QProgressBar()
        self.bulkProgressBar.setVisible(False)
        bulkLayout.addWidget(self.bulkProgressBar)

        self.bulkStatusLabel = qt.QLabel("Status: ready")
        self.bulkStatusLabel.setWordWrap(True)
        bulkLayout.addWidget(self.bulkStatusLabel)

        self.bulkRunButton = qt.QPushButton("Run Bulk Dilation + Subtract")
        self.bulkRunButton.setStyleSheet(
            "QPushButton{background:#1565c0;color:white;font-weight:bold;"
            "padding:8px;border-radius:4px;font-size:13px;}"
            "QPushButton:hover{background:#0d47a1;}"
            "QPushButton:disabled{background:#888;}"
        )
        self.bulkRunButton.clicked.connect(self._run_bulk)
        bulkLayout.addWidget(self.bulkRunButton)

        self.bulkPanel.setVisible(False)
        self.layout.addWidget(self.bulkPanel)

        self.layout.addStretch(1)

    # ── Mode toggle ───────────────────────────────────────────────────────────

    def onModeToggled(self, scene_checked):
        self.scenePanel.setVisible(scene_checked)
        self.bulkPanel.setVisible(not scene_checked)

    # ── Source rows ───────────────────────────────────────────────────────────

    def _add_src_row(self, dilate_mm=5.0):
        frame = qt.QFrame()
        row   = qt.QHBoxLayout(frame)
        row.setContentsMargins(0, 1, 0, 1)

        seg_combo = self._make_seg_combo()
        sub_combo = qt.QComboBox()
        sub_combo.addItem("All segments")
        sub_combo.setToolTip("Which sub-segment to dilate, or 'All segments' to merge all")

        dil_spin = qt.QDoubleSpinBox()
        dil_spin.setRange(0.5, 100.0)
        dil_spin.setSingleStep(1.0)
        dil_spin.setDecimals(1)
        dil_spin.setValue(dilate_mm)
        dil_spin.setSuffix(" mm")

        rm_btn = qt.QPushButton("×")
        rm_btn.setFixedWidth(28)
        rm_btn.setStyleSheet("QPushButton{color:red;font-weight:bold;}")
        rm_btn.clicked.connect(lambda: self._remove_row(frame, self._srcRows))

        row.addWidget(seg_combo, 3)
        row.addWidget(sub_combo, 2)
        row.addWidget(dil_spin, 2)
        row.addWidget(rm_btn)

        rd = {'widget': frame, 'seg': seg_combo,
              'sub': sub_combo, 'dilate': dil_spin}
        seg_combo.currentNodeChanged.connect(
            lambda node, r=rd: self._refresh_sub_combo(node, r['sub']))

        self._srcContainerLayout.addWidget(frame)
        self._srcRows.append(rd)

    # ── Target rows ───────────────────────────────────────────────────────────

    def _add_tgt_row(self):
        frame = qt.QFrame()
        row   = qt.QHBoxLayout(frame)
        row.setContentsMargins(0, 1, 0, 1)

        seg_combo = self._make_seg_combo()
        sub_combo = qt.QComboBox()
        sub_combo.addItem("All segments")
        sub_combo.setToolTip(
            "Which sub-segment to subtract from, or 'All segments' to merge all")

        rm_btn = qt.QPushButton("×")
        rm_btn.setFixedWidth(28)
        rm_btn.setStyleSheet("QPushButton{color:red;font-weight:bold;}")
        rm_btn.clicked.connect(lambda: self._remove_row(frame, self._tgtRows))

        row.addWidget(seg_combo, 3)
        row.addWidget(sub_combo, 2)
        row.addWidget(rm_btn)

        rd = {'widget': frame, 'seg': seg_combo, 'sub': sub_combo}
        seg_combo.currentNodeChanged.connect(
            lambda node, r=rd: self._refresh_sub_combo(node, r['sub']))

        self._tgtContainerLayout.addWidget(frame)
        self._tgtRows.append(rd)

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _make_seg_combo(self):
        c = slicer.qMRMLNodeComboBox()
        c.nodeTypes     = ['vtkMRMLSegmentationNode']
        c.addEnabled    = False
        c.removeEnabled = False
        c.noneEnabled   = True
        c.setMRMLScene(slicer.mrmlScene)
        return c

    def _refresh_sub_combo(self, node, combo):
        combo.clear()
        combo.addItem("All segments")
        if node is None:
            return
        seg = node.GetSegmentation()
        for i in range(seg.GetNumberOfSegments()):
            combo.addItem(seg.GetNthSegment(i).GetName())

    def _remove_row(self, frame, row_list):
        for i, rd in enumerate(row_list):
            if rd['widget'] is frame:
                row_list.pop(i)
                break
        frame.setParent(None)

    # ── Run ───────────────────────────────────────────────────────────────────

    def onRun(self):
        # Collect source configs
        src_configs = []
        for rd in self._srcRows:
            node = rd['seg'].currentNode()
            if node is None:
                continue
            sub_text = rd['sub'].currentText
            src_configs.append({
                'seg_node':  node,
                'seg_name':  None if sub_text == "All segments" else sub_text,
                'dilate_mm': rd['dilate'].value,
            })

        # Collect target configs
        tgt_configs = []
        for rd in self._tgtRows:
            node = rd['seg'].currentNode()
            if node is None:
                continue
            sub_text = rd['sub'].currentText
            tgt_configs.append({
                'seg_node': node,
                'seg_name': None if sub_text == "All segments" else sub_text,
            })

        if not src_configs:
            self.statusLabel.setText("Status: add at least one dilation source.")
            return

        self.runButton.setEnabled(False)
        self.statusLabel.setText("Status: running…")
        slicer.app.processEvents()

        try:
            dilated_names, subtracted_names = self.logic.run(src_configs, tgt_configs)
            parts = []
            if dilated_names:
                parts.append(f"Dilated: {', '.join(dilated_names)}")
            if subtracted_names:
                parts.append(f"Subtracted: {', '.join(subtracted_names)}")
            self.statusLabel.setText("Status: done — " + "  |  ".join(parts))
        except Exception:
            import traceback
            self.statusLabel.setText("Status: ERROR — see Python console.")
            print(traceback.format_exc())
        finally:
            self.runButton.setEnabled(True)

    # ── Bulk: folder picker ──────────────────────────────────────────────────

    def onBrowse(self):
        folder = qt.QFileDialog.getExistingDirectory(
            None, "Select dataset root folder", self.folderEdit.text or "")
        if folder:
            self.folderEdit.setText(folder)

    def onFolderChanged(self, folder):
        if not folder or not os.path.isdir(folder):
            self.scanResultLabel.setText("Folder not found.")
            self.scanResultLabel.setStyleSheet("color:#b71c1c; font-style:italic;")
            self._bulkSegFiles = []
            self._refresh_bulk_filename_combos()
            return

        n, seg_files = self._scan_bulk_folder(folder)
        self.scanResultLabel.setText(
            f"{n} patient Seg folder(s) found in 'Segments/'  "
            f"({len(seg_files)} segment file(s) detected for the dropdowns below).")
        self.scanResultLabel.setStyleSheet(
            "color:#1b5e20; font-style:normal; font-weight:bold;")
        print(f"[SCAN] {folder}: {n} patient Seg folder(s) found, "
              f"{len(seg_files)} segment file(s)")

        self._bulkSegFiles = seg_files
        self._refresh_bulk_filename_combos()

    def _scan_bulk_folder(self, root):
        """Return (n_seg_folders, seg_files) where seg_files is the sorted
        list of segment files found in the FIRST patient's
        Segments/<ID>_Seg/ folder."""
        seg_root = os.path.join(root, 'Segments')
        seg_dirs = sorted(
            d for d in os.listdir(seg_root)
            if d.endswith('_Seg') and os.path.isdir(os.path.join(seg_root, d))
        ) if os.path.isdir(seg_root) else []

        seg_files = []
        if seg_dirs:
            first = os.path.join(seg_root, seg_dirs[0])
            seg_files = sorted(
                f for f in os.listdir(first)
                if f.endswith('.seg.nrrd') or f.endswith('.nii.gz') or f.endswith('.nii')
            )
        return len(seg_dirs), seg_files

    def _refresh_bulk_filename_combos(self):
        """Repopulate the dropdown items in every existing bulk source/target
        row with the segment files just detected, preserving each row's
        current selection/typed text."""
        for rd in self._bulkSrcRows + self._bulkTgtRows:
            combo = rd['fname']
            current = combo.currentText
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(self._bulkSegFiles)
            combo.setCurrentText(current)
            combo.blockSignals(False)

    # ── Bulk: copy sample setup from Scene tab ───────────────────────────────

    def _copy_from_scene(self):
        """
        Derive bulk file rows from the Dilation Sources / Target Segments
        configured on the Scene tab for the currently-loaded sample patient.

        Mapping (TotalSegmentator file-naming convention):
          - row set to a specific sub-segment -> '<sub-segment>.nii.gz'
          - row set to 'All segments'         -> '<segmentation node name>.nii.gz'

        The resulting filenames are pre-filled but fully editable below
        before running bulk — e.g. if a previous bulk module wrote its
        output under a different stem.
        """
        for rd in list(self._bulkSrcRows):
            self._remove_row(rd['widget'], self._bulkSrcRows)
        for rd in list(self._bulkTgtRows):
            self._remove_row(rd['widget'], self._bulkTgtRows)

        n_src = n_tgt = 0
        for rd in self._srcRows:
            node = rd['seg'].currentNode()
            if node is None:
                continue
            sub_text = rd['sub'].currentText
            fname = (f"{node.GetName()}.nii.gz" if sub_text == "All segments"
                     else f"{sub_text}.nii.gz")
            self._add_bulk_src_row(fname, rd['dilate'].value)
            n_src += 1

        for rd in self._tgtRows:
            node = rd['seg'].currentNode()
            if node is None:
                continue
            sub_text = rd['sub'].currentText
            fname = (f"{node.GetName()}.nii.gz" if sub_text == "All segments"
                     else f"{sub_text}.nii.gz")
            self._add_bulk_tgt_row(fname)
            n_tgt += 1

        if n_src == 0 and n_tgt == 0:
            self.copyResultLabel.setText(
                "Nothing to copy — set up Dilation Sources / Target Segments "
                "above on a sample patient first, then click this button.")
            self.copyResultLabel.setStyleSheet("color:#b71c1c; font-style:italic;")
        else:
            self.copyResultLabel.setText(
                f"Copied {n_src} source file(s) and {n_tgt} target file(s) — "
                f"review/edit filenames below before running.")
            self.copyResultLabel.setStyleSheet("color:#1b5e20; font-weight:bold;")

    # ── Bulk: source/target file rows ────────────────────────────────────────

    def _make_filename_combo(self, filename="", placeholder=""):
        """Editable combo box listing the segment files detected in the
        sample patient's Segments/<ID>_Seg/ folder.  Stays editable so a
        filename that doesn't exist yet (e.g. an output produced by a
        previous bulk module) can still be typed in."""
        combo = qt.QComboBox()
        combo.setEditable(True)
        combo.setInsertPolicy(qt.QComboBox.NoInsert)
        combo.addItems(self._bulkSegFiles)
        le = combo.lineEdit()
        if le is not None and placeholder:
            le.setPlaceholderText(placeholder)
        combo.setCurrentText(filename)
        return combo

    def _add_bulk_src_row(self, filename="", dilate_mm=5.0):
        frame = qt.QFrame()
        row   = qt.QHBoxLayout(frame)
        row.setContentsMargins(0, 1, 0, 1)

        fname_combo = self._make_filename_combo(filename, "e.g. aorta.nii.gz")

        dil_spin = qt.QDoubleSpinBox()
        dil_spin.setRange(0.5, 100.0)
        dil_spin.setSingleStep(1.0)
        dil_spin.setDecimals(1)
        dil_spin.setValue(dilate_mm)
        dil_spin.setSuffix(" mm")

        rm_btn = qt.QPushButton("×")
        rm_btn.setFixedWidth(28)
        rm_btn.setStyleSheet("QPushButton{color:red;font-weight:bold;}")
        rm_btn.clicked.connect(lambda: self._remove_row(frame, self._bulkSrcRows))

        row.addWidget(fname_combo, 4)
        row.addWidget(dil_spin, 2)
        row.addWidget(rm_btn)

        rd = {'widget': frame, 'fname': fname_combo, 'dilate': dil_spin}
        self._bulkSrcContainerLayout.addWidget(frame)
        self._bulkSrcRows.append(rd)

    def _add_bulk_tgt_row(self, filename=""):
        frame = qt.QFrame()
        row   = qt.QHBoxLayout(frame)
        row.setContentsMargins(0, 1, 0, 1)

        fname_combo = self._make_filename_combo(filename, "e.g. visceral_fat.nii.gz")

        rm_btn = qt.QPushButton("×")
        rm_btn.setFixedWidth(28)
        rm_btn.setStyleSheet("QPushButton{color:red;font-weight:bold;}")
        rm_btn.clicked.connect(lambda: self._remove_row(frame, self._bulkTgtRows))

        row.addWidget(fname_combo, 4)
        row.addWidget(rm_btn)

        rd = {'widget': frame, 'fname': fname_combo}
        self._bulkTgtContainerLayout.addWidget(frame)
        self._bulkTgtRows.append(rd)

    def _get_bulk_src_configs(self):
        out = []
        for rd in self._bulkSrcRows:
            fname = rd['fname'].currentText.strip()
            if fname:
                out.append({'filename': fname, 'dilate_mm': rd['dilate'].value})
        return out

    def _get_bulk_tgt_configs(self):
        out = []
        for rd in self._bulkTgtRows:
            fname = rd['fname'].currentText.strip()
            if fname:
                out.append({'filename': fname})
        return out

    # ── Bulk: run ─────────────────────────────────────────────────────────────

    def _run_bulk(self):
        folder = self.folderEdit.text.strip()
        if not folder or not os.path.isdir(folder):
            self.bulkStatusLabel.setText("Status: choose a valid dataset root folder.")
            return

        src_file_configs = self._get_bulk_src_configs()
        tgt_file_configs = self._get_bulk_tgt_configs()

        if not src_file_configs:
            self.bulkStatusLabel.setText("Status: add at least one dilation source file.")
            return

        print(f"[BULK] Source files:")
        for cfg in src_file_configs:
            print(f"[BULK]   {cfg['filename']!r:50s} dilate={cfg['dilate_mm']} mm")
        print(f"[BULK] Target files:")
        for cfg in tgt_file_configs:
            print(f"[BULK]   {cfg['filename']!r:50s}")

        self.bulkRunButton.setEnabled(False)
        self.bulkProgressBar.setVisible(True)
        self.bulkProgressBar.setValue(0)
        self.bulkStatusLabel.setText("Status: starting bulk run…")
        slicer.app.processEvents()

        def progress_cb(done, total, subject):
            self.bulkProgressBar.setMaximum(total)
            self.bulkProgressBar.setValue(done)
            self.bulkStatusLabel.setText(f"Status: [{done}/{total}] {subject}")
            slicer.app.processEvents()

        try:
            self.logic.run_bulk(
                dataset_root      = folder,
                src_file_configs  = src_file_configs,
                tgt_file_configs  = tgt_file_configs,
                skip_done         = self.skipDoneCheck.isChecked(),
                progress_cb       = progress_cb,
            )
            self.bulkStatusLabel.setText("Status: bulk run complete.")
        except Exception:
            import traceback
            self.bulkStatusLabel.setText("Status: ERROR — see Python console.")
            print(traceback.format_exc())
        finally:
            self.bulkRunButton.setEnabled(True)
            self.bulkProgressBar.setVisible(False)

