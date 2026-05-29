"""
UreterPostProcess — 3D Slicer scripted module

Drag-and-drop this file into 3D Slicer to load, then find it under
Modules → Segmentation → Ureter Post-Process.

Modes
-----
Scene  Run the pipeline on nodes already loaded in the current scene.
Bulk   Iterate over every subject in a dataset_clean folder tree.

Per-organ processing modes
--------------------------
Skip          No processing.
Clip only     Zero voxels outside the L1-L5 Z range.
Clean only    Zero voxels that overlap the ureter mask AND exceed SUV_CLEAN_THRESH.
Clip + Clean  Clip first, then clean.

Output: one *_processed.nii.gz per organ (bulk) or *_processed seg node (scene).
"""

import os
import re
import vtk
import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
)

PROCESS_MODES = ['Skip', 'Clip only', 'Clean only', 'Clip + Clean']


def _default_mode(filename):
    nl = filename.lower()
    if any(k in nl for k in ('fat', 'iliopsoas', 'psoas')):
        return 'Clip + Clean'
    return 'Skip'


# ── Module metadata ───────────────────────────────────────────────────────────

class UreterPostProcess(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title        = "Ureter Post-Process"
        self.parent.categories   = ["Segmentation"]
        self.parent.dependencies = []
        self.parent.contributors = ["IshitaSinghFaujdar"]
        self.parent.helpText = (
            "PET-guided ureter mask, per-organ L1-L5 clipping and/or ureter cleanup.\n"
            "For each organ select: Skip / Clip only / Clean only / Clip + Clean."
        )
        self.parent.acknowledgementText = ""


# ── Widget ────────────────────────────────────────────────────────────────────

class UreterPostProcessWidget(ScriptedLoadableModuleWidget):

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = UreterPostProcessLogic()

        # ── Parameters ──────────────────────────────────────────────────────
        paramsBox = ctk.ctkCollapsibleButton()
        paramsBox.text = "Parameters"
        self.layout.addWidget(paramsBox)
        form = qt.QFormLayout(paramsBox)

        self.suvThreshSpin = qt.QDoubleSpinBox()
        self.suvThreshSpin.setRange(0.1, 20.0)
        self.suvThreshSpin.setSingleStep(0.5)
        self.suvThreshSpin.setValue(2.0)
        form.addRow("SUV threshold (ureter):", self.suvThreshSpin)

        self.suvCleanSpin = qt.QDoubleSpinBox()
        self.suvCleanSpin.setRange(0.1, 20.0)
        self.suvCleanSpin.setSingleStep(0.5)
        self.suvCleanSpin.setValue(1.2)
        form.addRow("SUV threshold (organ clean):", self.suvCleanSpin)

        self.dilateSpin = qt.QDoubleSpinBox()
        self.dilateSpin.setRange(1.0, 50.0)
        self.dilateSpin.setSingleStep(1.0)
        self.dilateSpin.setValue(18.0)
        form.addRow("Dilation radius (mm):", self.dilateSpin)

        self.connectUreterCheck = qt.QCheckBox("Connect ureter path (link fragments)")
        self.connectUreterCheck.setChecked(True)
        self.connectUreterCheck.setToolTip(
            "After dilation, draw a vertical tube between adjacent disconnected "
            "ureter fragments sorted by Z position."
        )
        form.addRow("", self.connectUreterCheck)

        self.maxGapSpin = qt.QDoubleSpinBox()
        self.maxGapSpin.setRange(10.0, 300.0)
        self.maxGapSpin.setSingleStep(10.0)
        self.maxGapSpin.setValue(35.0)
        self.maxGapSpin.setSuffix(" mm")
        self.maxGapSpin.setToolTip(
            "Maximum Z gap between fragment borders to bridge. "
            "Pairs farther apart than this are left disconnected."
        )
        form.addRow("Max connection gap:", self.maxGapSpin)

        self.fillHolesCheck = qt.QCheckBox("Fill holes (per-slice)")
        self.fillHolesCheck.setChecked(True)
        self.fillHolesCheck.setToolTip(
            "After connecting fragments, fill enclosed holes inside the ureter mask "
            "on each axial slice using binary_fill_holes."
        )
        form.addRow("", self.fillHolesCheck)

        # ── Mode radio buttons ────────────────────────────────────────────────
        modeGroup = qt.QGroupBox("Mode")
        modeRow   = qt.QHBoxLayout(modeGroup)
        self.sceneRadio = qt.QRadioButton("Run on Scene")
        self.bulkRadio  = qt.QRadioButton("Run on Uploaded File (Bulk)")
        self.sceneRadio.setChecked(True)
        modeRow.addWidget(self.sceneRadio)
        modeRow.addWidget(self.bulkRadio)
        self.sceneRadio.toggled.connect(self.onModeToggled)
        self.layout.addWidget(modeGroup)

        # ── Scene panel ───────────────────────────────────────────────────────
        self.scenePanel = qt.QFrame()
        sceneLayout = qt.QVBoxLayout(self.scenePanel)
        sceneLayout.setContentsMargins(0, 0, 0, 0)

        refreshBtn = qt.QPushButton("Refresh from Scene")
        refreshBtn.setToolTip("Scan the scene and auto-populate all pickers below")
        refreshBtn.clicked.connect(self.onRefreshMapping)
        sceneLayout.addWidget(refreshBtn)

        sceneForm = qt.QFormLayout()
        sceneLayout.addLayout(sceneForm)

        self.petSelector = slicer.qMRMLNodeComboBox()
        self.petSelector.nodeTypes = ['vtkMRMLScalarVolumeNode']
        self.petSelector.addEnabled = False
        self.petSelector.removeEnabled = False
        self.petSelector.noneEnabled = True
        self.petSelector.setMRMLScene(slicer.mrmlScene)
        self.petSelector.setToolTip("SUVbw PET volume node")
        sceneForm.addRow("PET volume:", self.petSelector)

        # ── Ureter mask on/off switch ──────────────────────────────────────
        self.generateUreterCheck = qt.QCheckBox(
            "Generate ureter mask  (uncheck to apply exclusion masks / clipping only)")
        self.generateUreterCheck.setChecked(True)
        self.generateUreterCheck.setStyleSheet(
            "font-weight:bold; padding:4px; color:#1b5e20;")
        self.generateUreterCheck.setToolTip(
            "When unchecked:\n"
            "  • No ureter mask is built (PET threshold step is skipped entirely)\n"
            "  • 'Clean only' and 'Clip + Clean' organ modes still run, but the\n"
            "    ureter-overlap removal step is skipped — only clipping applies\n"
            "  • Extra exclusion masks still work normally\n"
            "  • TotalSeg node is only required if you have clip-mode organs\n\n"
            "Use this when you already have a good mask and just want to apply\n"
            "exclusion masks or Z-clipping without rebuilding the ureter mask.")
        self.generateUreterCheck.toggled.connect(self._on_ureter_toggle)
        sceneLayout.addWidget(self.generateUreterCheck)

        self.totalSegSelector = slicer.qMRMLNodeComboBox()
        self.totalSegSelector.nodeTypes = ['vtkMRMLSegmentationNode']
        self.totalSegSelector.addEnabled = False
        self.totalSegSelector.removeEnabled = False
        self.totalSegSelector.noneEnabled = True
        self.totalSegSelector.setMRMLScene(slicer.mrmlScene)
        self.totalSegSelector.setToolTip("TotalSeg segmentation node containing vertebrae")
        self.totalSegSelector.currentNodeChanged.connect(self.onTotalSegChanged)
        sceneForm.addRow("TotalSeg node:", self.totalSegSelector)

        # Vertebrae multi-select checklist
        sceneLayout.addWidget(qt.QLabel(
            "L1-L5 vertebrae segments (check all — union Z range used):"))
        self.vertebraeList = qt.QListWidget()
        self.vertebraeList.setFixedHeight(120)
        sceneLayout.addWidget(self.vertebraeList)

        # Inferior ureter boundary
        infBoundLbl = qt.QLabel(
            "Inferior ureter boundary segment (e.g. sacrum — "
            "its inferior Z border caps the ureter mask downward):")
        infBoundLbl.setWordWrap(True)
        sceneLayout.addWidget(infBoundLbl)
        self.inferiorSegCombo = qt.QComboBox()
        self.inferiorSegCombo.addItem("None (use fixed 90 mm offset below L5)")
        self.inferiorSegCombo.setToolTip(
            "Pick a segment from the TotalSeg node whose INFERIOR Z border "
            "becomes the hard lower limit of the ureter mask.\n"
            "Recommended: 'sacrum' — stops the mask from going into the thighs.")
        sceneLayout.addWidget(self.inferiorSegCombo)

        # Per-organ processing rows
        sceneLayout.addWidget(qt.QLabel("Organs to process (set mode per organ):"))
        self._sceneOrganRows = []
        self._sceneOrganContainer = qt.QWidget()
        self._sceneOrganContainerLayout = qt.QVBoxLayout(self._sceneOrganContainer)
        self._sceneOrganContainerLayout.setContentsMargins(0, 0, 0, 0)
        self._sceneOrganContainerLayout.setSpacing(3)
        sceneLayout.addWidget(self._sceneOrganContainer)

        addSceneOrganBtn = qt.QPushButton("+ Add organ from scene")
        addSceneOrganBtn.clicked.connect(lambda: self._add_scene_organ_row())
        sceneLayout.addWidget(addSceneOrganBtn)

        # ── Extra exclusion masks ─────────────────────────────────────────
        sceneLayout.addWidget(qt.QLabel(
            "Extra exclusion masks (dilate, then remove overlapping voxels above SUV threshold):"))

        exclHdr = qt.QHBoxLayout()
        eh1 = qt.QLabel("Segmentation"); eh1.setStyleSheet("font-weight:bold;")
        eh2 = qt.QLabel("Sub-segment");  eh2.setStyleSheet("font-weight:bold;")
        eh3 = qt.QLabel("Dilation (mm)"); eh3.setStyleSheet("font-weight:bold;")
        eh4 = qt.QLabel("SUV threshold"); eh4.setStyleSheet("font-weight:bold;")
        exclHdr.addWidget(eh1, 3)
        exclHdr.addWidget(eh2, 2)
        exclHdr.addWidget(eh3, 2)
        exclHdr.addWidget(eh4, 2)
        exclHdr.addSpacing(30)
        sceneLayout.addLayout(exclHdr)

        self._exclRows = []
        self._exclContainer = qt.QWidget()
        self._exclContainerLayout = qt.QVBoxLayout(self._exclContainer)
        self._exclContainerLayout.setContentsMargins(0, 0, 0, 0)
        self._exclContainerLayout.setSpacing(3)
        sceneLayout.addWidget(self._exclContainer)

        addExclBtn = qt.QPushButton("+ Add exclusion mask")
        addExclBtn.clicked.connect(lambda: self._add_excl_row())
        sceneLayout.addWidget(addExclBtn)

        self.layout.addWidget(self.scenePanel)

        # ── Bulk panel with sub-tabs ──────────────────────────────────────────
        self.bulkPanel = qt.QFrame()
        bulkOuterLayout = qt.QVBoxLayout(self.bulkPanel)
        bulkOuterLayout.setContentsMargins(0, 4, 0, 0)

        self.bulkTabs = qt.QTabWidget()
        bulkOuterLayout.addWidget(self.bulkTabs)

        # ── Sub-tab 1: Data ───────────────────────────────────────────────────
        dataTab = qt.QWidget()
        dataLayout = qt.QVBoxLayout(dataTab)

        folderRow = qt.QHBoxLayout()
        self.folderEdit = qt.QLineEdit()
        self.folderEdit.setPlaceholderText("Path to dataset_clean …")
        self.folderEdit.textChanged.connect(self.onFolderChanged)
        folderRow.addWidget(self.folderEdit)
        browseBtn = qt.QPushButton("Browse…")
        browseBtn.setFixedWidth(80)
        browseBtn.clicked.connect(self.onBrowse)
        folderRow.addWidget(browseBtn)
        dataFolderRow = qt.QFormLayout()
        dataFolderRow.addRow("Folder:", folderRow)
        dataLayout.addLayout(dataFolderRow)

        self.skipDoneCheck = qt.QCheckBox("Skip subjects whose outputs already exist")
        self.skipDoneCheck.setChecked(True)
        dataLayout.addWidget(self.skipDoneCheck)

        self.scanResultLabel = qt.QLabel("No folder selected.")
        self.scanResultLabel.setWordWrap(True)
        self.scanResultLabel.setStyleSheet("color: #555; font-style: italic;")
        dataLayout.addWidget(self.scanResultLabel)

        dataLayout.addStretch(1)
        self.bulkTabs.addTab(dataTab, "Data")

        # ── Sub-tab 2: Slicer (vertebrae segment picker) ──────────────────────
        slicerTab = qt.QWidget()
        slicerLayout = qt.QVBoxLayout(slicerTab)

        slicerForm = qt.QFormLayout()
        slicerLayout.addLayout(slicerForm)

        self.parentSegCombo = qt.QComboBox()
        self.parentSegCombo.setToolTip(
            "Segmentation file from which vertebrae segments will be read")
        self.parentSegCombo.currentIndexChanged.connect(self.onParentSegChanged)
        slicerForm.addRow("Select segment:", self.parentSegCombo)

        slicerLayout.addWidget(qt.QLabel(
            "Sub-segments (L1-L5 vertebrae — add/remove as needed):"))

        self._subSegRows = []
        self.subSegContainer = qt.QWidget()
        self.subSegContainerLayout = qt.QVBoxLayout(self.subSegContainer)
        self.subSegContainerLayout.setContentsMargins(0, 0, 0, 0)
        self.subSegContainerLayout.setSpacing(3)
        slicerLayout.addWidget(self.subSegContainer)

        self._addSubSegBtn = qt.QPushButton("+ Add segment")
        self._addSubSegBtn.clicked.connect(lambda: self._add_sub_seg_row())
        slicerLayout.addWidget(self._addSubSegBtn)

        self._singleMaskInfoLabel = qt.QLabel(
            "ℹ  This is a single-mask NIfTI file — the entire mask Z range will be "
            "used as the vertebrae region automatically. No sub-segment selection is needed.")
        self._singleMaskInfoLabel.setWordWrap(True)
        self._singleMaskInfoLabel.setStyleSheet(
            "color:#0d47a1; background:#e3f2fd; padding:6px; border-radius:3px;")
        self._singleMaskInfoLabel.setVisible(False)
        slicerLayout.addWidget(self._singleMaskInfoLabel)

        self.continuityWarning = qt.QLabel("")
        self.continuityWarning.setWordWrap(True)
        self.continuityWarning.setStyleSheet(
            "color:#b71c1c; background:#ffebee; padding:5px; border-radius:3px;")
        self.continuityWarning.setVisible(False)
        slicerLayout.addWidget(self.continuityWarning)

        slicerLayout.addStretch(1)
        self.bulkTabs.addTab(slicerTab, "Slicer")

        # ── Sub-tab 3: Organs ─────────────────────────────────────────────────
        organsTab = qt.QWidget()
        organsLayout = qt.QVBoxLayout(organsTab)
        organsLayout.addWidget(qt.QLabel(
            "For each organ file choose what processing to apply:"))

        # Column header
        hdr = qt.QHBoxLayout()
        h1 = qt.QLabel("File"); h1.setStyleSheet("font-weight:bold;")
        h2 = qt.QLabel("Processing"); h2.setStyleSheet("font-weight:bold;")
        hdr.addWidget(h1, 3)
        hdr.addWidget(h2, 2)
        hdr.addSpacing(30)
        organsLayout.addLayout(hdr)

        self._bulkOrganRows = []
        self._bulkOrganContainer = qt.QWidget()
        self._bulkOrganContainerLayout = qt.QVBoxLayout(self._bulkOrganContainer)
        self._bulkOrganContainerLayout.setContentsMargins(0, 0, 0, 0)
        self._bulkOrganContainerLayout.setSpacing(3)
        organsLayout.addWidget(self._bulkOrganContainer)

        self._bulkOrganNoFilesLabel = qt.QLabel(
            "Scan a folder in the Data tab to populate organ files.")
        self._bulkOrganNoFilesLabel.setStyleSheet("color:#555; font-style:italic;")
        organsLayout.addWidget(self._bulkOrganNoFilesLabel)

        organsLayout.addStretch(1)
        self.bulkTabs.addTab(organsTab, "Organs")

        self.bulkPanel.setVisible(False)
        self.layout.addWidget(self.bulkPanel)

        # ── Progress + status ─────────────────────────────────────────────────
        self.progressBar = qt.QProgressBar()
        self.progressBar.setVisible(False)
        self.layout.addWidget(self.progressBar)

        self.statusLabel = qt.QLabel("Status: ready")
        self.statusLabel.setWordWrap(True)
        self.layout.addWidget(self.statusLabel)

        # ── Run button ────────────────────────────────────────────────────────
        self.runButton = qt.QPushButton("Run")
        self.runButton.setStyleSheet(
            "QPushButton{background:#2e7d32;color:white;font-weight:bold;"
            "padding:8px;border-radius:4px;font-size:13px;}"
            "QPushButton:hover{background:#1b5e20;}"
            "QPushButton:disabled{background:#888;}"
        )
        self.runButton.clicked.connect(self.onRun)
        self.layout.addWidget(self.runButton)

        self.layout.addStretch(1)

    # ── Mode toggle ───────────────────────────────────────────────────────────

    def onModeToggled(self, scene_checked):
        self.scenePanel.setVisible(scene_checked)
        self.bulkPanel.setVisible(not scene_checked)

    # ── Ureter on/off toggle ──────────────────────────────────────────────────

    def _on_ureter_toggle(self, enabled):
        """Grey out all ureter-specific controls when the switch is off."""
        ureter_widgets = [
            self.totalSegSelector,
            self.vertebraeList,
            self.inferiorSegCombo,
            self.connectUreterCheck,
            self.maxGapSpin,
            self.fillHolesCheck,
            self.suvThreshSpin,
        ]
        for w in ureter_widgets:
            w.setEnabled(enabled)
        color = "#1b5e20" if enabled else "#b71c1c"
        label = ("Generate ureter mask  (uncheck to apply exclusion masks / clipping only)"
                 if enabled else
                 "⊘  Ureter mask OFF — exclusion masks and clipping still active")
        self.generateUreterCheck.setText(label)
        self.generateUreterCheck.setStyleSheet(
            f"font-weight:bold; padding:4px; color:{color};")

    # ── Bulk: folder scanning ─────────────────────────────────────────────────

    def onBrowse(self):
        folder = qt.QFileDialog.getExistingDirectory(
            None, "Select dataset_clean folder", self.folderEdit.text or "")
        if folder:
            self.folderEdit.setText(folder)

    def onFolderChanged(self, folder):
        if not folder or not os.path.isdir(folder):
            self.scanResultLabel.setText("Folder not found.")
            self.scanResultLabel.setStyleSheet("color:#b71c1c; font-style:italic;")
            return

        pet_count, ct_count, seg_count, seg_files = self._scan_bulk_folder(folder)
        self.scanResultLabel.setText(
            f"{pet_count} PET   {ct_count} CT   {seg_count} Seg folder(s) found")
        self.scanResultLabel.setStyleSheet(
            "color:#1b5e20; font-style:normal; font-weight:bold;")
        print(f"[SCAN] {folder}: PET={pet_count} CT={ct_count} Seg={seg_count}")

        # Populate vertebrae combo (Slicer tab)
        self.parentSegCombo.blockSignals(True)
        self.parentSegCombo.clear()
        for name in seg_files:
            self.parentSegCombo.addItem(name)
        self.parentSegCombo.blockSignals(False)
        if self.parentSegCombo.count > 0:
            self.onParentSegChanged()

        # Populate organ rows (Organs tab) — all .nii.gz files except vertebrae file
        vert_file = self.parentSegCombo.currentText
        organ_files = [f for f in seg_files
                       if f != vert_file and
                       (f.endswith('.nii.gz') or f.endswith('.nii'))]
        self._rebuild_bulk_organ_rows(organ_files)

    def _scan_bulk_folder(self, root):
        pet_root = os.path.join(root, 'PET')
        ct_root  = os.path.join(root, 'CT')
        seg_root = os.path.join(root, 'Segments')

        pet_count = len([d for d in os.listdir(pet_root)
                         if os.path.isdir(os.path.join(pet_root, d))]) \
                    if os.path.isdir(pet_root) else 0
        ct_count  = len([d for d in os.listdir(ct_root)
                         if os.path.isdir(os.path.join(ct_root, d))]) \
                    if os.path.isdir(ct_root) else 0
        seg_dirs  = sorted([d for d in os.listdir(seg_root)
                            if d.endswith('_Seg') and
                            os.path.isdir(os.path.join(seg_root, d))]) \
                    if os.path.isdir(seg_root) else []

        seg_files = []
        if seg_dirs:
            first = os.path.join(seg_root, seg_dirs[0])
            seg_files = sorted(
                f for f in os.listdir(first)
                if f.endswith('.seg.nrrd') or f.endswith('.nii.gz') or f.endswith('.nii')
            )
        return pet_count, ct_count, len(seg_dirs), seg_files

    # ── Bulk: organ rows (Organs tab) ─────────────────────────────────────────

    def _rebuild_bulk_organ_rows(self, organ_files):
        for _, _, frame in self._bulkOrganRows:
            frame.setParent(None)
        self._bulkOrganRows = []

        if not organ_files:
            self._bulkOrganNoFilesLabel.setVisible(True)
            return
        self._bulkOrganNoFilesLabel.setVisible(False)
        for fname in organ_files:
            self._add_bulk_organ_row(fname)

    def _add_bulk_organ_row(self, filename, mode=None):
        if mode is None:
            mode = _default_mode(filename)
        frame = qt.QFrame()
        row = qt.QHBoxLayout(frame)
        row.setContentsMargins(0, 0, 0, 0)

        lbl = qt.QLabel(filename)
        lbl.setToolTip(filename)
        row.addWidget(lbl, 3)

        mode_combo = qt.QComboBox()
        for m in PROCESS_MODES:
            mode_combo.addItem(m)
        idx = PROCESS_MODES.index(mode) if mode in PROCESS_MODES else 0
        mode_combo.setCurrentIndex(idx)
        row.addWidget(mode_combo, 2)

        rm = qt.QPushButton("×")
        rm.setFixedWidth(28)
        rm.setStyleSheet("color:red; font-weight:bold;")
        rm.clicked.connect(lambda: self._remove_bulk_organ_row(frame))
        row.addWidget(rm)

        self._bulkOrganRows.append((filename, mode_combo, frame))
        self._bulkOrganContainerLayout.addWidget(frame)

    def _remove_bulk_organ_row(self, frame):
        self._bulkOrganRows = [
            (f, c, fr) for f, c, fr in self._bulkOrganRows if fr is not frame]
        frame.setParent(None)

    def _get_bulk_organ_configs(self):
        return [
            (fname, combo.currentText)
            for fname, combo, _ in self._bulkOrganRows
            if combo.currentText != 'Skip'
        ]

    # ── Scene: organ rows ─────────────────────────────────────────────────────

    def _add_scene_organ_row(self, node=None, mode='Clip + Clean'):
        frame = qt.QFrame()
        row = qt.QHBoxLayout(frame)
        row.setContentsMargins(0, 0, 0, 0)

        node_combo = slicer.qMRMLNodeComboBox()
        node_combo.nodeTypes = ['vtkMRMLSegmentationNode']
        node_combo.addEnabled = False
        node_combo.removeEnabled = False
        node_combo.noneEnabled = True
        node_combo.setMRMLScene(slicer.mrmlScene)
        if node:
            node_combo.setCurrentNode(node)
        row.addWidget(node_combo, 3)

        mode_combo = qt.QComboBox()
        for m in PROCESS_MODES:
            mode_combo.addItem(m)
        idx = PROCESS_MODES.index(mode) if mode in PROCESS_MODES else 3
        mode_combo.setCurrentIndex(idx)
        row.addWidget(mode_combo, 2)

        rm = qt.QPushButton("×")
        rm.setFixedWidth(28)
        rm.setStyleSheet("color:red; font-weight:bold;")
        rm.clicked.connect(lambda: self._remove_scene_organ_row(frame))
        row.addWidget(rm)

        self._sceneOrganRows.append((node_combo, mode_combo, frame))
        self._sceneOrganContainerLayout.addWidget(frame)

    def _remove_scene_organ_row(self, frame):
        self._sceneOrganRows = [
            (nc, mc, fr) for nc, mc, fr in self._sceneOrganRows if fr is not frame]
        frame.setParent(None)

    def _get_scene_organ_configs(self):
        configs = []
        for node_combo, mode_combo, _ in self._sceneOrganRows:
            node = node_combo.currentNode()
            mode = mode_combo.currentText
            if node and mode != 'Skip':
                configs.append((node.GetName(), mode))
        return configs

    # ── Scene: exclusion mask rows ────────────────────────────────────────────

    def _add_excl_row(self, dilate_mm=5.0, suv_thresh=1.2):
        frame = qt.QFrame()
        frame.setFrameShape(qt.QFrame.StyledPanel)
        outer = qt.QVBoxLayout(frame)
        outer.setContentsMargins(4, 2, 4, 2)
        outer.setSpacing(2)

        line1 = qt.QHBoxLayout()
        seg_combo = slicer.qMRMLNodeComboBox()
        seg_combo.nodeTypes = ['vtkMRMLSegmentationNode']
        seg_combo.addEnabled = False
        seg_combo.removeEnabled = False
        seg_combo.noneEnabled = True
        seg_combo.setMRMLScene(slicer.mrmlScene)
        seg_combo.setToolTip("Segmentation node to use as exclusion mask")

        seg_name_combo = qt.QComboBox()
        seg_name_combo.addItem("All segments")
        seg_name_combo.setToolTip(
            "Choose a specific sub-segment or 'All segments' to merge all")
        line1.addWidget(seg_combo, 3)
        line1.addWidget(seg_name_combo, 2)
        outer.addLayout(line1)

        line2 = qt.QHBoxLayout()
        dil_spin = qt.QDoubleSpinBox()
        dil_spin.setRange(0.0, 50.0)
        dil_spin.setSingleStep(1.0)
        dil_spin.setValue(dilate_mm)
        dil_spin.setSuffix(" mm")

        suv_spin = qt.QDoubleSpinBox()
        suv_spin.setRange(0.1, 20.0)
        suv_spin.setSingleStep(0.5)
        suv_spin.setDecimals(1)
        suv_spin.setValue(suv_thresh)

        rm_btn = qt.QPushButton("×")
        rm_btn.setFixedWidth(28)
        rm_btn.setStyleSheet("QPushButton{color:red;font-weight:bold;}")
        rm_btn.clicked.connect(lambda: self._remove_excl_row(frame))

        line2.addWidget(qt.QLabel("Dilation:"))
        line2.addWidget(dil_spin)
        line2.addSpacing(8)
        line2.addWidget(qt.QLabel("SUV >"))
        line2.addWidget(suv_spin)
        line2.addStretch(1)
        line2.addWidget(rm_btn)
        outer.addLayout(line2)

        row_dict = {
            'widget': frame, 'seg': seg_combo, 'seg_name': seg_name_combo,
            'dilate': dil_spin, 'suv': suv_spin,
        }
        seg_combo.currentNodeChanged.connect(
            lambda node, r=row_dict: self._refresh_excl_seg_names(node, r))
        self._exclContainerLayout.addWidget(frame)
        self._exclRows.append(row_dict)

    def _refresh_excl_seg_names(self, seg_node, row_dict):
        combo = row_dict['seg_name']
        combo.clear()
        combo.addItem("All segments")
        if seg_node is None:
            return
        seg = seg_node.GetSegmentation()
        for i in range(seg.GetNumberOfSegments()):
            combo.addItem(seg.GetNthSegment(i).GetName())

    def _remove_excl_row(self, frame):
        self._exclRows = [r for r in self._exclRows if r['widget'] is not frame]
        frame.setParent(None)

    def _get_excl_configs(self):
        configs = []
        for r in self._exclRows:
            seg_node = r['seg'].currentNode()
            if seg_node is None:
                continue
            seg_name_text = r['seg_name'].currentText
            seg_name = None if seg_name_text == "All segments" else seg_name_text
            configs.append({
                'seg_node':   seg_node,
                'seg_name':   seg_name,
                'dilate_mm':  r['dilate'].value,
                'suv_thresh': r['suv'].value,
            })
        return configs

    # ── Bulk: vertebrae segment picker (Slicer tab) ───────────────────────────

    def onParentSegChanged(self):
        seg_file = self.parentSegCombo.currentText
        if not seg_file:
            return

        is_nii = seg_file.endswith('.nii.gz') or seg_file.endswith('.nii')
        if is_nii:
            self._clear_sub_seg_rows()
            self._addSubSegBtn.setVisible(False)
            self._singleMaskInfoLabel.setVisible(True)
            self.continuityWarning.setVisible(False)
            print(f"[SLICER] '{seg_file}' is NIfTI mask — full Z extent will be used")
            return

        self._addSubSegBtn.setVisible(True)
        self._singleMaskInfoLabel.setVisible(False)

        folder = self.folderEdit.text.strip()
        if not folder:
            return
        seg_root = os.path.join(folder, 'Segments')
        if not os.path.isdir(seg_root):
            return
        seg_dirs = sorted(d for d in os.listdir(seg_root)
                          if d.endswith('_Seg') and
                          os.path.isdir(os.path.join(seg_root, d)))
        if not seg_dirs:
            return

        file_path = os.path.join(seg_root, seg_dirs[0], seg_file)
        if not os.path.exists(file_path):
            return

        names = self._read_nrrd_segment_names(file_path)
        print(f"[SLICER] '{seg_file}' — {len(names)} segments: {names}")

        self._clear_sub_seg_rows()
        l1l5 = [n for n in names if self._looks_like_l1l5(n)]
        defaults = l1l5 if l1l5 else names[:5]
        for name in (defaults or ['']):
            self._add_sub_seg_row(all_names=names, selected=name)
        self._check_contiguity()

    def _looks_like_l1l5(self, name):
        nl = name.lower()
        return any(f'l{i}' in nl for i in range(1, 6)) and \
               any(k in nl for k in ('vertebra', 'vertebrae', 'segment', 'spine'))

    def _read_nrrd_segment_names(self, path):
        names = []
        try:
            with open(path, 'rb') as f:
                for raw in f:
                    try:
                        line = raw.decode('utf-8').strip()
                    except UnicodeDecodeError:
                        break
                    if line == '':
                        break
                    if '_Name:=' in line:
                        names.append(line.split(':=', 1)[1].strip())
        except Exception as e:
            print(f"[SLICER] Could not parse {path}: {e}")
        return names

    def _add_sub_seg_row(self, all_names=None, selected=None):
        row_frame = qt.QFrame()
        row_layout = qt.QHBoxLayout(row_frame)
        row_layout.setContentsMargins(0, 0, 0, 0)

        combo = qt.QComboBox()
        if all_names:
            for n in all_names:
                combo.addItem(n)
            if selected and selected in all_names:
                combo.setCurrentIndex(all_names.index(selected))
        combo.currentIndexChanged.connect(self._check_contiguity)
        row_layout.addWidget(combo)

        rm = qt.QPushButton("×")
        rm.setFixedWidth(28)
        rm.setStyleSheet("color:red; font-weight:bold;")
        rm.clicked.connect(lambda: self._remove_sub_seg_row(row_frame))
        row_layout.addWidget(rm)

        self._subSegRows.append((combo, rm, row_frame))
        self.subSegContainerLayout.addWidget(row_frame)
        self._check_contiguity()

    def _remove_sub_seg_row(self, row_frame):
        self._subSegRows = [(c, b, f) for c, b, f in self._subSegRows if f is not row_frame]
        row_frame.setParent(None)
        self._check_contiguity()

    def _clear_sub_seg_rows(self):
        for _, _, frame in self._subSegRows:
            frame.setParent(None)
        self._subSegRows = []

    def _get_selected_sub_segs(self):
        return [combo.currentText for combo, _, _ in self._subSegRows if combo.currentText]

    def _check_contiguity(self):
        selected = self._get_selected_sub_segs()
        nums = []
        for name in selected:
            m = re.search(r'[Ll](\d+)', name)
            if m:
                nums.append(int(m.group(1)))
        if len(nums) < 2:
            self.continuityWarning.setVisible(False)
            return
        nums_sorted = sorted(set(nums))
        gaps = [nums_sorted[i] for i in range(len(nums_sorted) - 1)
                if nums_sorted[i + 1] - nums_sorted[i] > 1]
        if gaps:
            missing = []
            for g in gaps:
                idx = nums_sorted.index(g)
                for mv in range(g + 1, nums_sorted[idx + 1]):
                    missing.append(f"L{mv}")
            self.continuityWarning.setText(
                f"⚠  Non-contiguous: missing {', '.join(missing)}. "
                f"Z range will span the gap — verify this is intended.")
            self.continuityWarning.setVisible(True)
        else:
            self.continuityWarning.setVisible(False)

    # ── Scene mapping helpers ─────────────────────────────────────────────────

    def onRefreshMapping(self):
        best_pet = next(
            (n for n in slicer.util.getNodesByClass('vtkMRMLScalarVolumeNode')
             if 'suvbw' in n.GetName().lower()), None)
        if best_pet is None:
            nodes = slicer.util.getNodesByClass('vtkMRMLScalarVolumeNode')
            best_pet = nodes[0] if nodes else None
        if best_pet:
            self.petSelector.setCurrentNode(best_pet)

        best_ts = next(
            (n for n in slicer.util.getNodesByClass('vtkMRMLSegmentationNode')
             if 'totalseg' in n.GetName().lower() or 'abdomen' in n.GetName().lower()),
            None)
        if best_ts:
            self.totalSegSelector.setCurrentNode(best_ts)
            self._populate_vertebrae_list(best_ts)

        self._rebuild_scene_organ_rows()
        print("[MAPPING] Refreshed from scene.")
        self._print_mapping_summary()

    def _rebuild_scene_organ_rows(self):
        for _, _, frame in self._sceneOrganRows:
            frame.setParent(None)
        self._sceneOrganRows = []
        ts_node = self.totalSegSelector.currentNode()
        ts_name = ts_node.GetName() if ts_node else ''
        for n in slicer.util.getNodesByClass('vtkMRMLSegmentationNode'):
            if n.GetName() == ts_name:
                continue
            nl = n.GetName().lower()
            mode = 'Clip + Clean' if any(k in nl for k in ('fat', 'iliopsoas', 'psoas')) \
                   else 'Skip'
            self._add_scene_organ_row(node=n, mode=mode)

    def onTotalSegChanged(self, node):
        self._populate_vertebrae_list(node)
        self._populate_inferior_seg_combo(node)

    def _populate_inferior_seg_combo(self, seg_node):
        current = self.inferiorSegCombo.currentText
        self.inferiorSegCombo.clear()
        self.inferiorSegCombo.addItem("None (use fixed 90 mm offset below L5)")
        if seg_node is None:
            return
        seg = seg_node.GetSegmentation()
        for i in range(seg.GetNumberOfSegments()):
            name = seg.GetNthSegment(i).GetName()
            self.inferiorSegCombo.addItem(name)
            # Auto-select sacrum if present
            if name.lower() == 'sacrum':
                self.inferiorSegCombo.setCurrentIndex(
                    self.inferiorSegCombo.count - 1)
        # Restore previous selection if still present
        if current and current != "None (use fixed 90 mm offset below L5)":
            idx = self.inferiorSegCombo.findText(current)
            if idx >= 0:
                self.inferiorSegCombo.setCurrentIndex(idx)

    def _populate_vertebrae_list(self, seg_node):
        self.vertebraeList.clear()
        if seg_node is None:
            return
        seg = seg_node.GetSegmentation()
        for i in range(seg.GetNumberOfSegments()):
            name = seg.GetNthSegment(i).GetName()
            item = qt.QListWidgetItem(name)
            item.setFlags(item.flags() | qt.Qt.ItemIsUserCheckable)
            nl = name.lower()
            is_l1l5 = (
                any(f'l{v}' in nl for v in range(1, 6)) and
                any(k in nl for k in ('vertebra', 'vertebrae', 'segment'))
            ) or any(f'segment_{v}' == nl for v in range(1, 6))
            item.setCheckState(qt.Qt.Checked if is_l1l5 else qt.Qt.Unchecked)
            self.vertebraeList.addItem(item)

    def _get_checked_vertebrae(self):
        return [
            self.vertebraeList.item(i).text()
            for i in range(self.vertebraeList.count)
            if self.vertebraeList.item(i).checkState() == qt.Qt.Checked
        ]

    def _print_mapping_summary(self):
        pet  = self.petSelector.currentNode()
        ts   = self.totalSegSelector.currentNode()
        vert = self._get_checked_vertebrae()
        organ_configs = self._get_scene_organ_configs()
        print(f"[MAPPING] PET      : {pet.GetName() if pet else 'None'}")
        print(f"[MAPPING] TotalSeg : {ts.GetName()  if ts  else 'None'}")
        print(f"[MAPPING] Vertebrae: {vert}")
        print(f"[MAPPING] Organs   :")
        for name, mode in organ_configs:
            print(f"[MAPPING]   {name!r:40s} -> {mode}")

    # ── Run ───────────────────────────────────────────────────────────────────

    def onRun(self):
        if self.sceneRadio.isChecked():
            self._run_scene()
        else:
            self._run_bulk()

    def _run_scene(self):
        pet_node        = self.petSelector.currentNode()
        generate_ureter = self.generateUreterCheck.isChecked()
        organ_configs   = self._get_scene_organ_configs()
        excl_configs    = self._get_excl_configs()

        if pet_node is None:
            self.statusLabel.setText("Status: pick a PET volume (Refresh from Scene).")
            return

        # TotalSeg + vertebrae are required only when clipping OR ureter generation is on
        needs_clip    = any(m in ('Clip only', 'Clip + Clean') for _, m in organ_configs)
        needs_totalseg = generate_ureter or needs_clip

        ts_node   = self.totalSegSelector.currentNode()
        vert_segs = self._get_checked_vertebrae()

        if needs_totalseg and ts_node is None:
            self.statusLabel.setText(
                "Status: pick a TotalSeg node "
                "(required for ureter generation and/or Z-clipping).")
            return
        if needs_totalseg and not vert_segs:
            self.statusLabel.setText(
                "Status: check at least one L1-L5 vertebrae segment "
                "(required for ureter generation and/or Z-clipping).")
            return

        if not organ_configs and not excl_configs:
            self.statusLabel.setText(
                "Status: add at least one organ (mode ≠ Skip) or exclusion mask.")
            return

        print("\n[SCENE] Mapping:")
        self._print_mapping_summary()

        self.runButton.setEnabled(False)
        self.statusLabel.setText("Status: running on scene…")
        slicer.app.processEvents()

        inf_seg_text = self.inferiorSegCombo.currentText
        inf_seg_name = (None if inf_seg_text.startswith("None")
                        else inf_seg_text)

        try:
            self.logic.run_scene(
                suv_thresh          = self.suvThreshSpin.value,
                suv_clean_thresh    = self.suvCleanSpin.value,
                dilate_mm           = self.dilateSpin.value,
                connect_path        = self.connectUreterCheck.isChecked(),
                max_gap_mm          = self.maxGapSpin.value,
                fill_holes          = self.fillHolesCheck.isChecked(),
                pet_node            = pet_node,
                totalseg_node_name  = ts_node.GetName() if ts_node else None,
                vertebrae_seg_names = vert_segs,
                organ_configs       = organ_configs,
                excl_configs        = excl_configs,
                inf_bound_seg_name  = inf_seg_name,
                generate_ureter     = generate_ureter,
            )
            self.statusLabel.setText("Status: done — check Segmentations module.")
        except Exception:
            import traceback
            self.statusLabel.setText("Status: ERROR — see Python console.")
            print(traceback.format_exc())
        finally:
            self.runButton.setEnabled(True)

    def _run_bulk(self):
        folder = self.folderEdit.text.strip()
        if not folder or not os.path.isdir(folder):
            self.statusLabel.setText("Status: choose a valid dataset_clean folder.")
            return

        parent_seg_file = self.parentSegCombo.currentText
        is_nii = parent_seg_file.endswith('.nii.gz') or parent_seg_file.endswith('.nii')

        if is_nii:
            vert_segs = []
        else:
            vert_segs = self._get_selected_sub_segs()
            if not vert_segs:
                self.statusLabel.setText(
                    "Status: select at least one vertebrae segment in the Slicer tab.")
                return

        organ_file_configs = self._get_bulk_organ_configs()
        if not organ_file_configs:
            self.statusLabel.setText(
                "Status: set at least one organ mode (not Skip) in the Organs tab.")
            return

        print(f"[BULK] Vertebrae file: {parent_seg_file}")
        print(f"[BULK] Organ configs:")
        for fname, mode in organ_file_configs:
            print(f"[BULK]   {fname!r:50s} -> {mode}")

        self.runButton.setEnabled(False)
        self.progressBar.setVisible(True)
        self.progressBar.setValue(0)
        self.statusLabel.setText("Status: starting bulk run…")
        slicer.app.processEvents()

        def progress_cb(done, total, subject):
            self.progressBar.setMaximum(total)
            self.progressBar.setValue(done)
            self.statusLabel.setText(f"Status: [{done}/{total}] {subject}")
            slicer.app.processEvents()

        try:
            self.logic.run_bulk(
                dataset_clean_root  = folder,
                suv_thresh          = self.suvThreshSpin.value,
                suv_clean_thresh    = self.suvCleanSpin.value,
                dilate_mm           = self.dilateSpin.value,
                connect_path        = self.connectUreterCheck.isChecked(),
                max_gap_mm          = self.maxGapSpin.value,
                fill_holes          = self.fillHolesCheck.isChecked(),
                skip_done           = self.skipDoneCheck.isChecked(),
                vertebrae_seg_names = vert_segs,
                totalseg_file       = parent_seg_file,
                organ_file_configs  = organ_file_configs,
                progress_cb         = progress_cb,
            )
            self.statusLabel.setText("Status: bulk run complete.")
        except Exception:
            import traceback
            self.statusLabel.setText("Status: ERROR — see Python console.")
            print(traceback.format_exc())
        finally:
            self.runButton.setEnabled(True)
            self.progressBar.setVisible(False)


# ── Logic ─────────────────────────────────────────────────────────────────────

class UreterPostProcessLogic(ScriptedLoadableModuleLogic):

    # ── Scene entry point ─────────────────────────────────────────────────────

    def run_scene(self, suv_thresh, suv_clean_thresh, dilate_mm,
                  pet_node, totalseg_node_name, vertebrae_seg_names, organ_configs,
                  connect_path=True, max_gap_mm=35.0, fill_holes=True,
                  excl_configs=None, inf_bound_seg_name=None,
                  generate_ureter=True):
        """
        organ_configs:   list of (node_name, mode)
        excl_configs:    list of dicts {'seg_node','seg_name','dilate_mm','suv_thresh'}
        generate_ureter: False → skip ureter mask entirely; clean steps are no-ops,
                         clipping and exclusion masks still work normally.
        Outputs one '<node>_processed' segmentation node per organ.
        """
        print("\n" + "="*60)
        print("SCENE MODE" + ("" if generate_ureter else "  [ureter mask OFF]"))
        print("="*60)

        pet_arr, pet_affine, pet_mat, vox_size = self._get_pet(pet_node)

        # Z bounds — only needed for clipping or ureter mask construction
        needs_clip   = any(m in ('Clip only', 'Clip + Clean') for _, m in organ_configs)
        needs_ureter = generate_ureter and any(
            m in ('Clean only', 'Clip + Clean') for _, m in organ_configs)
        needs_z      = needs_clip or needs_ureter

        z_inferior = z_superior = None
        if needs_z and totalseg_node_name and vertebrae_seg_names:
            z_inferior, z_superior = self._get_l1l5_z_bounds(
                vertebrae_seg_names, totalseg_node_name=totalseg_node_name)

        ureter_arr = ureter_affine = None
        if needs_ureter:
            ureter_arr, ureter_affine = self._build_ureter_mask(
                pet_arr, pet_affine, pet_mat, vox_size,
                z_inferior, z_superior, suv_thresh, dilate_mm,
                seg_node_name='ureter_from_pet',
                totalseg_node_name=totalseg_node_name,
                connect_path=connect_path,
                max_gap_mm=max_gap_mm,
                fill_holes=fill_holes,
                inf_bound_seg_name=inf_bound_seg_name,
            )
        elif not generate_ureter:
            print("[URETER] Ureter mask generation disabled — clean steps will be skipped.")

        for organ_name, mode in organ_configs:
            self._process_organ_scene(
                organ_name, mode,
                ureter_arr, ureter_affine,
                pet_arr, pet_affine, suv_clean_thresh,
                z_inferior, z_superior,
                excl_configs=excl_configs,
            )

        print("\n" + "="*60)
        print("SCENE MODE COMPLETE")
        print("="*60)

    # ── Bulk entry point ──────────────────────────────────────────────────────

    def run_bulk(self, dataset_clean_root, suv_thresh, suv_clean_thresh,
                 dilate_mm, skip_done,
                 connect_path=True, max_gap_mm=35.0, fill_holes=True,
                 vertebrae_seg_names=None,
                 totalseg_file='TotalSeg_abdomen.seg.nrrd',
                 organ_file_configs=None,
                 progress_cb=None):
        """
        organ_file_configs: list of (filename, mode)
        Outputs one '<stem>_processed.nii.gz' per organ per subject.
        """
        seg_root = os.path.join(dataset_clean_root, 'Segments')
        pet_root = os.path.join(dataset_clean_root, 'PET')

        if vertebrae_seg_names is None:
            vertebrae_seg_names = ['vertebrae_L1', 'vertebrae_L2', 'vertebrae_L3',
                                   'vertebrae_L4', 'vertebrae_L5']
        if organ_file_configs is None:
            organ_file_configs = []

        needs_ureter = any(m in ('Clean only', 'Clip + Clean') for _, m in organ_file_configs)

        subjects = sorted(
            d for d in os.listdir(seg_root)
            if d.endswith('_Seg') and os.path.isdir(os.path.join(seg_root, d))
        )
        total = len(subjects)
        print(f"\nBulk run: {total} subjects  ureter_needed={needs_ureter}")

        for idx, seg_folder in enumerate(subjects):
            subject_id = seg_folder[:-4]
            seg_dir    = os.path.join(seg_root, seg_folder)
            pet_dir    = os.path.join(pet_root, subject_id + '_PET')

            if progress_cb:
                progress_cb(idx, total, subject_id)

            print(f"\n{'='*60}")
            print(f"[{idx+1}/{total}] {subject_id}")
            print(f"{'='*60}")

            if skip_done and self._bulk_outputs_exist(seg_dir, organ_file_configs):
                print("  [SKIP] Outputs already exist.")
                continue

            if needs_ureter and not os.path.isdir(pet_dir):
                print("  [SKIP] PET folder not found (required for clean mode).")
                continue

            print("  [SCENE] Clearing Slicer scene…")
            slicer.mrmlScene.Clear(0)

            try:
                # Load PET only if needed
                pet_arr = pet_affine = pet_mat = vox_size = None
                if needs_ureter:
                    pet_node = self._load_pet_dicom(pet_dir, subject_id)
                    if pet_node is None:
                        print("  [ERROR] Could not load PET — skipping.")
                        continue
                    pet_arr, pet_affine, pet_mat, vox_size = self._get_pet(pet_node)

                # Compute Z bounds
                totalseg_path = os.path.join(seg_dir, totalseg_file)
                if not os.path.exists(totalseg_path):
                    print(f"  [ERROR] '{totalseg_file}' not found — skipping.")
                    continue

                totalseg_is_nii = (totalseg_file.endswith('.nii.gz') or
                                   totalseg_file.endswith('.nii'))
                if totalseg_is_nii:
                    lm = slicer.util.loadLabelVolume(totalseg_path)
                    z_inferior, z_superior = self._get_z_bounds_from_label_volume(lm)
                    slicer.mrmlScene.RemoveNode(lm)
                else:
                    totalseg_node_name = os.path.splitext(
                        os.path.splitext(totalseg_file)[0])[0]
                    ts_node = slicer.util.loadSegmentation(totalseg_path)
                    ts_node.SetName(totalseg_node_name)
                    n_segs = ts_node.GetSegmentation().GetNumberOfSegments()
                    print(f"  [SEG] '{totalseg_node_name}' — {n_segs} segments")
                    z_inferior, z_superior = self._get_l1l5_z_bounds(
                        vertebrae_seg_names, totalseg_node_name=totalseg_node_name)

                print(f"  [SEG] Z range: {z_inferior:.1f} -> {z_superior:.1f} mm RAS")

                # Build ureter mask
                ureter_arr = ureter_affine = None
                loaded_totalseg_name = None if totalseg_is_nii else totalseg_node_name
                if needs_ureter:
                    ureter_arr, ureter_affine = self._build_ureter_mask(
                        pet_arr, pet_affine, pet_mat, vox_size,
                        z_inferior, z_superior, suv_thresh, dilate_mm,
                        seg_node_name='ureter_from_pet',
                        totalseg_node_name=loaded_totalseg_name,
                        connect_path=connect_path,
                        max_gap_mm=max_gap_mm,
                        fill_holes=fill_holes,
                    )
                    ureter_out = os.path.join(seg_dir, 'ureter_from_pet.nii.gz')
                    self._export_seg_to_nifti('ureter_from_pet', ureter_out)

                # Process each organ
                for fname, mode in organ_file_configs:
                    fpath = os.path.join(seg_dir, fname)
                    if not os.path.exists(fpath):
                        print(f"  [ORGAN] '{fname}' not found — skipping.")
                        continue
                    stem = fname.replace('.nii.gz', '').replace('.nii', '')
                    out_path = os.path.join(seg_dir, stem + '_processed.nii.gz')
                    self._process_organ_bulk(
                        fpath, stem, mode,
                        ureter_arr, ureter_affine,
                        pet_arr, pet_affine, suv_clean_thresh,
                        z_inferior, z_superior,
                        out_path,
                    )

                print(f"\n  [DONE] {subject_id}")

            except Exception:
                import traceback
                print(f"\n  [ERROR] {subject_id}:")
                print(traceback.format_exc())

        if progress_cb:
            progress_cb(total, total, 'done')
        print(f"\n{'='*60}\nBULK RUN FINISHED\n{'='*60}")

    # ── Organ processing ──────────────────────────────────────────────────────

    def _process_organ_scene(self, organ_name, mode,
                              ureter_arr, ureter_affine,
                              pet_arr, pet_affine, suv_clean_thresh,
                              z_inferior, z_superior,
                              excl_configs=None):
        import numpy as np
        print(f"\n[ORGAN] '{organ_name}' mode={mode}")
        try:
            organ_seg = slicer.util.getNode(organ_name)
        except Exception:
            print(f"[ORGAN]   NOT FOUND — skipping.")
            return

        lm = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode', 'tmp_organ')
        slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(organ_seg, lm)
        arr    = slicer.util.arrayFromVolume(lm).copy()
        mat    = vtk.vtkMatrix4x4()
        lm.GetIJKToRASMatrix(mat)
        affine = self._mat_to_np(mat)
        before = int((arr > 0).sum())

        arr = self._apply_processing(arr, affine, mode,
                                     ureter_arr, ureter_affine,
                                     pet_arr, pet_affine, suv_clean_thresh,
                                     z_inferior, z_superior)

        # ── Apply extra exclusion masks ───────────────────────────────────────
        if excl_configs:
            for excl in excl_configs:
                try:
                    excl_arr, excl_affine = self._build_mask_from_seg(
                        excl['seg_node'], excl['dilate_mm'], excl.get('seg_name'),
                        save_to_scene=True)
                    excl_in = self._resample_to_target(
                        excl_arr, excl_affine, arr.shape, affine)
                    pet_in  = self._resample_to_target(
                        pet_arr, pet_affine, arr.shape, affine)
                    remove = ((arr > 0) & (excl_in > 0) & (pet_in > excl['suv_thresh']))
                    arr[remove] = 0
                    label = (excl.get('seg_name') or 'all')
                    print(f"[ORGAN]   Excl '{excl['seg_node'].GetName()}/{label}' "
                          f"removed {int(remove.sum())} voxels "
                          f"(SUV>{excl['suv_thresh']}, dil={excl['dilate_mm']}mm)")
                except Exception as e:
                    print(f"[ORGAN]   Excl error: {e}")

        after = int((arr > 0).sum())
        print(f"[ORGAN]   before={before}  after={after}  removed={before-after}")

        slicer.util.updateVolumeFromArray(lm, arr)
        result_name = organ_name + '_processed'
        self._remove_existing(result_name)
        result_seg = slicer.mrmlScene.AddNewNodeByClass(
            'vtkMRMLSegmentationNode', result_name)
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(lm, result_seg)
        result_seg.CreateClosedSurfaceRepresentation()
        slicer.mrmlScene.RemoveNode(lm)
        print(f"[ORGAN]   -> '{result_name}' added to scene.")
        return result_seg

    def _process_organ_bulk(self, fpath, stem, mode,
                             ureter_arr, ureter_affine,
                             pet_arr, pet_affine, suv_clean_thresh,
                             z_inferior, z_superior, out_path):
        print(f"\n[ORGAN] '{stem}' mode={mode}")
        lm = slicer.util.loadLabelVolume(fpath)
        arr    = slicer.util.arrayFromVolume(lm).copy()
        mat    = vtk.vtkMatrix4x4()
        lm.GetIJKToRASMatrix(mat)
        affine = self._mat_to_np(mat)
        before = int((arr > 0).sum())
        print(f"[ORGAN]   shape={arr.shape}  foreground={before}")

        arr = self._apply_processing(arr, affine, mode,
                                     ureter_arr, ureter_affine,
                                     pet_arr, pet_affine, suv_clean_thresh,
                                     z_inferior, z_superior)
        after = int((arr > 0).sum())
        print(f"[ORGAN]   before={before}  after={after}  removed={before-after}")

        slicer.util.updateVolumeFromArray(lm, arr)
        ok = slicer.util.saveNode(lm, out_path)
        slicer.mrmlScene.RemoveNode(lm)
        if ok:
            print(f"[ORGAN]   -> {os.path.basename(out_path)} "
                  f"({os.path.getsize(out_path)//1024} KB)")
        else:
            print(f"[ORGAN]   FAILED saving {out_path}")

    def _apply_processing(self, organ_arr, organ_affine, mode,
                           ureter_arr, ureter_affine,
                           pet_arr, pet_affine, suv_clean_thresh,
                           z_inferior, z_superior):
        import numpy as np

        if mode in ('Clip only', 'Clip + Clean'):
            if z_inferior is None or z_superior is None:
                print("[ORGAN]   Z bounds not available — clip step skipped")
            else:
                print(f"[ORGAN]   Clipping to L1-L5 Z ({z_inferior:.1f}–{z_superior:.1f} mm)…")
                shape = organ_arr.shape
                z_idx, y_idx, x_idx = np.meshgrid(
                    np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]),
                    indexing='ij')
                ijk_hom = np.stack([
                    x_idx.ravel(), y_idx.ravel(), z_idx.ravel(),
                    np.ones(x_idx.size)], axis=1).astype(np.float32)
                ras_z = (organ_affine @ ijk_hom.T).T[:, 2].reshape(shape)
                outside = (ras_z < z_inferior) | (ras_z > z_superior)
                clipped = int(((organ_arr > 0) & outside).sum())
                organ_arr[outside] = 0
                print(f"[ORGAN]   Clipped {clipped} voxels outside L1-L5")

        if mode in ('Clean only', 'Clip + Clean'):
            if ureter_arr is None:
                print("[ORGAN]   Ureter mask not available — ureter clean step skipped")
            else:
                print(f"[ORGAN]   Removing ureter-overlapping voxels "
                      f"(SUV>{suv_clean_thresh})…")
                ureter_in = self._resample_to_target(
                    ureter_arr, ureter_affine, organ_arr.shape, organ_affine)
                pet_in = self._resample_to_target(
                    pet_arr, pet_affine, organ_arr.shape, organ_affine)
                remove = ((organ_arr > 0) & (ureter_in > 0) & (pet_in > suv_clean_thresh))
                cleaned = int(remove.sum())
                organ_arr[remove] = 0
                print(f"[ORGAN]   Removed {cleaned} ureter-overlap voxels")

        return organ_arr

    # ── Core pipeline ─────────────────────────────────────────────────────────

    def _get_pet(self, pet_node):
        import numpy as np
        print(f"\n[PET] Node: '{pet_node.GetName()}'")
        pet_arr = slicer.util.arrayFromVolume(pet_node)
        pet_mat = vtk.vtkMatrix4x4()
        pet_node.GetIJKToRASMatrix(pet_mat)
        pet_affine = self._mat_to_np(pet_mat)
        vox_size = np.array([abs(pet_affine[0, 0]),
                              abs(pet_affine[1, 1]),
                              abs(pet_affine[2, 2])])
        print(f"[PET] Shape (Z,Y,X): {pet_arr.shape}  "
              f"voxel mm: {np.round(vox_size, 3)}  "
              f"SUV range: [{float(pet_arr.min()):.2f}, {float(pet_arr.max()):.2f}]")
        return pet_arr, pet_affine, pet_mat, vox_size

    def _get_l1l5_z_bounds(self, vertebrae_seg_names,
                            totalseg_node_name='TotalSeg_abdomen'):
        if isinstance(vertebrae_seg_names, str):
            vertebrae_seg_names = [vertebrae_seg_names]
        print(f"\n[L1-L5] '{totalseg_node_name}'  requested: {vertebrae_seg_names}")
        self._log_all_segments(totalseg_node_name)

        seg_node  = slicer.util.getNode(totalseg_node_name)
        seg       = seg_node.GetSegmentation()
        available = {seg.GetNthSegment(i).GetName()
                     for i in range(seg.GetNumberOfSegments())}
        found   = [n for n in vertebrae_seg_names if n in available]
        missing = [n for n in vertebrae_seg_names if n not in available]
        if missing:
            print(f"[L1-L5] WARNING: not found (skipped): {missing}")
        if not found:
            raise ValueError(
                f"None of {vertebrae_seg_names} exist in '{totalseg_node_name}'.\n"
                f"Available: {sorted(available)}")

        z_mins, z_maxs = [], []
        for name in found:
            z_min, z_max = self._segment_ras_z_bounds(totalseg_node_name, name)
            z_mins.append(z_min); z_maxs.append(z_max)

        z_inferior, z_superior = min(z_mins), max(z_maxs)
        print(f"[L1-L5] Union of {len(found)} segments: "
              f"{z_inferior:.1f} -> {z_superior:.1f} mm")
        return z_inferior, z_superior

    def _build_ureter_mask(self, pet_arr, pet_affine, pet_mat, vox_size,
                            z_inferior, z_superior, suv_thresh, dilate_mm,
                            seg_node_name, totalseg_node_name=None,
                            connect_path=True, max_gap_mm=35.0, fill_holes=True,
                            inf_bound_seg_name=None,
                            ureter_ext_inf_mm=90.0, torso_radius_mm=220.0):
        """
        Build the ureter exclusion mask from PET.

        Z range:  from (z_superior + 50 mm) down to (z_inferior - ureter_ext_inf_mm).
                  This covers kidneys → full pelvis, well past L5 where
                  the psoas/iliopsoas still runs alongside the ureter.

        XY range: cylinder of radius torso_radius_mm centred on the vertebrae
                  centroid (derived from totalseg_node_name when available).
                  This removes arms and head without any extra segmentation.
        """
        import numpy as np
        from scipy import ndimage

        # Upper Z bound: top of L1 (no upward extension beyond the vertebrae)
        ureter_z_sup = z_superior

        if inf_bound_seg_name and totalseg_node_name:
            try:
                seg_z_min, seg_z_max = self._segment_ras_z_bounds(
                    totalseg_node_name, inf_bound_seg_name)
                ureter_z_inf = seg_z_min   # inferior border of sacrum
                print(f"\n[URETER] Inferior boundary from '{inf_bound_seg_name}': "
                      f"Z_inf set to {ureter_z_inf:.1f} mm (inferior border of segment)")
            except Exception as e:
                print(f"[URETER] WARNING: could not use '{inf_bound_seg_name}' as "
                      f"inferior boundary ({e}) — falling back to fixed offset.")
                ureter_z_inf = z_inferior - ureter_ext_inf_mm
        else:
            ureter_z_inf = z_inferior - ureter_ext_inf_mm

        print(f"\n[URETER] SUV>{suv_thresh}  dilation={dilate_mm}mm")
        print(f"[URETER] Z range: {ureter_z_inf:.1f} -> {ureter_z_sup:.1f} mm RAS "
              f"(L1-L5 was {z_inferior:.1f} -> {z_superior:.1f})")

        shape = pet_arr.shape
        z_idx, y_idx, x_idx = np.meshgrid(
            np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]),
            indexing='ij')
        ijk_hom = np.stack([
            x_idx.ravel(), y_idx.ravel(), z_idx.ravel(),
            np.ones(x_idx.size)], axis=1).astype(np.float32)
        ras_coords = (pet_affine @ ijk_hom.T).T          # (N, 4)
        ras_z = ras_coords[:, 2].reshape(shape)
        ras_x = ras_coords[:, 0].reshape(shape)
        ras_y = ras_coords[:, 1].reshape(shape)

        # Z clip: kidney-to-bladder range
        z_mask = ((ras_z >= ureter_z_inf) & (ras_z <= ureter_z_sup)).astype(np.uint8)
        print(f"[URETER] Z-mask voxels: {int(z_mask.sum())}")

        # XY torso cylinder: centred on vertebrae centroid
        if totalseg_node_name is not None:
            cx, cy = self._vertebrae_xy_centroid(totalseg_node_name)
        else:
            # Fallback: use image XY centre (rough but better than nothing)
            cx = float(np.mean(ras_x))
            cy = float(np.mean(ras_y))
        dist_xy = np.sqrt((ras_x - cx)**2 + (ras_y - cy)**2)
        torso_mask = (dist_xy <= torso_radius_mm).astype(np.uint8)
        print(f"[URETER] Torso cylinder centre XY=({cx:.1f},{cy:.1f}) "
              f"r={torso_radius_mm} mm  voxels: {int(torso_mask.sum())}")

        # Combined anatomical mask
        anat_mask = (z_mask & torso_mask).astype(np.uint8)

        # Threshold PET
        hot = (pet_arr > suv_thresh).astype(np.uint8)
        print(f"[URETER] Hot voxels (SUV>{suv_thresh}): {int(hot.sum())}")

        # Remove bladder (largest CC in full hot image)
        labeled, n = ndimage.label(hot)
        print(f"[URETER] Connected components: {n}")
        sizes = ndimage.sum(hot, labeled, range(1, n + 1))
        bladder_label  = int(np.argmax(sizes)) + 1
        hot_no_bladder = ((labeled != bladder_label) & (labeled > 0)).astype(np.uint8)
        print(f"[URETER] Bladder CC #{bladder_label} ({int(max(sizes))} vox) removed")

        # Apply anatomical mask (cuts out head/arms, keeps torso kidney→pelvis)
        hot_clipped = (hot_no_bladder & anat_mask).astype(np.uint8)
        print(f"[URETER] After torso+Z mask: {int(hot_clipped.sum())} voxels")

        rx = int(round(dilate_mm / vox_size[0]))
        ry = int(round(dilate_mm / vox_size[1]))
        rz = int(round(dilate_mm / vox_size[2]))
        zz, yy, xx = np.ogrid[-rz:rz+1, -ry:ry+1, -rx:rx+1]
        struct_3d = ((zz/max(rz,1))**2 + (yy/max(ry,1))**2 + (xx/max(rx,1))**2) <= 1.0
        ureter_mask = ndimage.binary_dilation(
            hot_clipped, structure=struct_3d).astype(np.uint8)
        print(f"[URETER] After {dilate_mm}mm dilation: {int(ureter_mask.sum())} voxels")

        # ── Connect disconnected ureter fragments (vertical cylinder only) ─────
        if connect_path:
            tube_r = max(3, int(round(dilate_mm / max(float(vox_size.max()), 1e-6) * 0.5)))
            ureter_mask = self._connect_ureter_path(
                ureter_mask, vox_size,
                max_gap_mm=max_gap_mm,
                tube_radius_vox=tube_r,
            )

        # ── Fill per-slice holes ───────────────────────────────────────────────
        if fill_holes:
            from scipy.ndimage import binary_fill_holes
            before_fh = int(ureter_mask.sum())
            for z in range(ureter_mask.shape[0]):
                if ureter_mask[z].any():
                    ureter_mask[z] = binary_fill_holes(ureter_mask[z]).astype(np.uint8)
            print(f"[URETER] Fill holes: {before_fh} → {int(ureter_mask.sum())} voxels")

        # ── Hard Z clip AFTER all processing ──────────────────────────────────
        # Dilation (and gap-bridging) can push voxels past the Z bounds set
        # above.  Re-apply the same Z mask as a strict post-processing clamp
        # so the final mask never exceeds the L1 top / sacrum bottom limits.
        before_clip = int(ureter_mask.sum())
        ureter_mask = (ureter_mask & z_mask).astype(np.uint8)
        print(f"[URETER] Post-dilation Z clip: {before_clip} → "
              f"{int(ureter_mask.sum())} voxels "
              f"(Z {ureter_z_inf:.1f} → {ureter_z_sup:.1f} mm enforced)")

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

    # ── Ureter connectivity (pure vertical cylinder) ──────────────────────────

    def _connect_ureter_path(self, mask_arr, vox_size, max_gap_mm=35.0, tube_radius_vox=3):
        """
        Bridge gaps between adjacent disconnected ureter fragments.

        For each consecutive pair of fragments (sorted by Z mid-point):
        - Gap is measured purely in Z (voxels × mm/vox).
        - The bridge is a vertical cylinder with a FIXED XY centre derived
          from the average of the two face centroids — no lateral drift.
        - The cylinder is stamped as a disk at every Z slice in the gap.
        """
        import numpy as np
        from scipy import ndimage

        labeled, n = ndimage.label(mask_arr)
        if n <= 1:
            return mask_arr.copy()

        z_mm_per_vox = abs(float(vox_size[2]))

        components = []
        for comp_id in range(1, n + 1):
            comp_vox = np.argwhere(labeled == comp_id)
            z_min = int(comp_vox[:, 0].min())
            z_max = int(comp_vox[:, 0].max())
            bot_face = comp_vox[comp_vox[:, 0] == z_min]
            top_face = comp_vox[comp_vox[:, 0] == z_max]
            components.append({
                'z_min': z_min,
                'z_max': z_max,
                'z_mid': (z_min + z_max) / 2.0,
                'bot':   bot_face.mean(axis=0).astype(np.float64),  # (Z,Y,X)
                'top':   top_face.mean(axis=0).astype(np.float64),
            })

        components.sort(key=lambda c: c['z_mid'])
        result = mask_arr.copy()

        for i in range(len(components) - 1):
            c_lo = components[i]
            c_hi = components[i + 1]

            z_gap_vox = c_hi['z_min'] - c_lo['z_max']
            z_gap_mm  = float(z_gap_vox) * z_mm_per_vox
            if z_gap_mm > max_gap_mm:
                print(f"[CONNECT]   Z gap {z_gap_mm:.1f} mm > {max_gap_mm} mm — skipping")
                continue

            # Fixed XY centre — average of the two facing centroid faces
            cy = (c_lo['top'][1] + c_hi['bot'][1]) / 2.0
            cx = (c_lo['top'][2] + c_hi['bot'][2]) / 2.0
            z_lo = c_lo['z_max']
            z_hi = c_hi['z_min']

            r  = tube_radius_vox
            y0 = max(0, int(cy) - r - 1)
            y1 = min(mask_arr.shape[1] - 1, int(cy) + r + 1)
            x0 = max(0, int(cx) - r - 1)
            x1 = min(mask_arr.shape[2] - 1, int(cx) + r + 1)

            yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
            disk = ((yy - cy) ** 2 + (xx - cx) ** 2) <= float(r) ** 2

            for z in range(z_lo, z_hi + 1):
                result[z, y0:y1 + 1, x0:x1 + 1][disk] = 1

            print(f"[CONNECT]   Bridged comp {i}→{i+1}: "
                  f"Z {z_lo}→{z_hi} ({z_gap_mm:.1f} mm), "
                  f"XY centre ({cy:.1f},{cx:.1f}), r={r}vox")

        after  = int((result > 0).sum())
        before = int((mask_arr > 0).sum())
        print(f"[CONNECT] Done: {before} → {after} voxels (+{after - before})")
        return result.astype(np.uint8)

    # ── Bulk helpers ──────────────────────────────────────────────────────────

    def _bulk_outputs_exist(self, seg_dir, organ_file_configs):
        for fname, mode in organ_file_configs:
            stem = fname.replace('.nii.gz', '').replace('.nii', '')
            if not os.path.exists(os.path.join(seg_dir, stem + '_processed.nii.gz')):
                return False
        return True

    def _load_pet_dicom(self, pet_dir, subject_id):
        from DICOMLib import DICOMUtils
        db = slicer.dicomDatabase

        def _all_series():
            uids = set()
            for pat in db.patients():
                for study in db.studiesForPatient(pat):
                    for series in db.seriesForStudy(study):
                        uids.add(series)
            return uids

        before_series   = _all_series()
        before_node_ids = {n.GetID() for n in
                           slicer.util.getNodesByClass('vtkMRMLScalarVolumeNode')}
        print(f"  [DICOM] Importing: {pet_dir}")
        ok = DICOMUtils.importDicom(pet_dir)
        print(f"  [DICOM] importDicom={ok}")

        after_series = _all_series()
        new_series   = after_series - before_series
        print(f"  [DICOM] New series: {len(new_series)}")

        if not new_series:
            pet_dir_norm = os.path.normpath(pet_dir)
            for series in after_series:
                files = db.filesForSeries(series)
                if files and os.path.normpath(os.path.dirname(files[0])) == pet_dir_norm:
                    new_series.add(series)
            print(f"  [DICOM] Series matched by path: {len(new_series)}")

        pt_series = []
        for uid in new_series:
            files = db.filesForSeries(uid)
            desc  = db.fileValue(files[0], '0008,103e') if files else '?'
            mod   = db.fileValue(files[0], '0008,0060') if files else '?'
            print(f"  [DICOM]   mod={mod!r}  desc={desc!r}")
            if mod.strip().upper() in ('PT', 'NM'):
                pt_series.append((uid, mod, desc))

        if not pt_series:
            pt_series = [(uid, '?', '?') for uid in new_series]

        for uid, mod, desc in pt_series:
            print(f"  [DICOM] Loading: mod={mod!r}  desc={desc!r}")
            try:
                node_ids = DICOMUtils.loadSeriesByUID([uid])
                print(f"  [DICOM]   -> {len(node_ids)} node(s)")
            except Exception as e:
                print(f"  [DICOM]   -> failed: {e}")

        after_nodes = slicer.util.getNodesByClass('vtkMRMLScalarVolumeNode')
        new_nodes   = [n for n in after_nodes if n.GetID() not in before_node_ids]
        print(f"  [DICOM] New volume nodes: {[n.GetName() for n in new_nodes]}")

        if not new_nodes:
            return None

        pet_node = next(
            (n for n in new_nodes if 'suv' in n.GetName().lower()), new_nodes[0])
        if 'SUVbw' not in pet_node.GetName():
            old = pet_node.GetName()
            pet_node.SetName(f'SUVbw_{subject_id}')
            print(f"  [DICOM] Renamed '{old}' -> '{pet_node.GetName()}'")
        return pet_node

    def _export_seg_to_nifti(self, seg_node_name, out_path):
        print(f"[SAVE] '{seg_node_name}' -> {os.path.basename(out_path)}")
        try:
            seg_node = slicer.util.getNode(seg_node_name)
        except Exception:
            print(f"[SAVE]   WARNING: node not found.")
            return
        lm = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode', 'tmp_export')
        slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(seg_node, lm)
        ok = slicer.util.saveNode(lm, out_path)
        slicer.mrmlScene.RemoveNode(lm)
        if ok:
            print(f"[SAVE]   OK ({os.path.getsize(out_path)//1024} KB)")
        else:
            print(f"[SAVE]   FAILED")

    # ── Shared helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _mat_to_np(mat):
        import numpy as np
        return np.array([[mat.GetElement(r, c) for c in range(4)] for r in range(4)])

    @staticmethod
    def _remove_existing(node_name):
        try:
            slicer.mrmlScene.RemoveNode(slicer.util.getNode(node_name))
        except Exception:
            pass

    def _log_all_segments(self, seg_node_name):
        try:
            seg_node = slicer.util.getNode(seg_node_name)
            seg      = seg_node.GetSegmentation()
            n        = seg.GetNumberOfSegments()
            print(f"[L1-L5] '{seg_node_name}' has {n} segment(s):")
            for i in range(n):
                s = seg.GetNthSegment(i)
                print(f"[L1-L5]   [{i}] {s.GetName()!r}")
        except Exception as e:
            print(f"[L1-L5] Could not list segments of '{seg_node_name}': {e}")

    def _segment_ras_z_bounds(self, seg_node_name, segment_name):
        import numpy as np
        print(f"[L1-L5]   Z bounds for '{segment_name}'…")
        seg_node = slicer.util.getNode(seg_node_name)
        seg      = seg_node.GetSegmentation()
        seg_id   = seg.GetSegmentIdBySegmentName(segment_name)
        if not seg_id:
            raise ValueError(f"Segment '{segment_name}' not found in {seg_node_name}")

        lm  = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode', 'tmp_vert')
        ids = vtk.vtkStringArray()
        ids.InsertNextValue(seg_id)
        slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
            seg_node, ids, lm)
        arr    = slicer.util.arrayFromVolume(lm)
        mat    = vtk.vtkMatrix4x4()
        lm.GetIJKToRASMatrix(mat)
        affine = self._mat_to_np(mat)
        slicer.mrmlScene.RemoveNode(lm)

        vox   = np.argwhere(arr > 0)
        print(f"[L1-L5]   '{segment_name}' voxels: {len(vox)}")
        ras_z = [
            (affine @ np.array([float(v[2]), float(v[1]), float(v[0]), 1.0]))[2]
            for v in vox[::3]
        ]
        z_min, z_max = float(min(ras_z)), float(max(ras_z))
        print(f"[L1-L5]   RAS Z: {z_min:.1f} -> {z_max:.1f}")
        return z_min, z_max

    def _get_z_bounds_from_label_volume(self, lm_node):
        import numpy as np
        arr = slicer.util.arrayFromVolume(lm_node)
        mat = vtk.vtkMatrix4x4()
        lm_node.GetIJKToRASMatrix(mat)
        affine = self._mat_to_np(mat)
        vox = np.argwhere(arr > 0)
        print(f"  [SEG] Non-zero voxels in NIfTI mask: {len(vox)}")
        if len(vox) == 0:
            raise ValueError("NIfTI mask has no non-zero voxels")
        vox_sample = vox[::max(1, len(vox) // 5000)]
        ras_z = (affine @ np.column_stack([
            vox_sample[:, 2].astype(float),
            vox_sample[:, 1].astype(float),
            vox_sample[:, 0].astype(float),
            np.ones(len(vox_sample))
        ]).T)[2]
        z_min, z_max = float(ras_z.min()), float(ras_z.max())
        print(f"  [SEG] NIfTI mask RAS Z: {z_min:.1f} -> {z_max:.1f}")
        return z_min, z_max

    def _vertebrae_xy_centroid(self, totalseg_node_name):
        """Return (cx, cy) RAS centroid of all vertebrae segments — used to centre the torso cylinder."""
        import numpy as np
        try:
            seg_node = slicer.util.getNode(totalseg_node_name)
            lm = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode', 'tmp_vert_xy')
            slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(seg_node, lm)
            arr    = slicer.util.arrayFromVolume(lm)
            mat    = vtk.vtkMatrix4x4()
            lm.GetIJKToRASMatrix(mat)
            affine = self._mat_to_np(mat)
            slicer.mrmlScene.RemoveNode(lm)

            vox = np.argwhere(arr > 0)
            if len(vox) == 0:
                raise ValueError("Empty vertebrae mask")
            sample = vox[::max(1, len(vox) // 2000)]
            ras = (affine @ np.column_stack([
                sample[:, 2].astype(float),
                sample[:, 1].astype(float),
                sample[:, 0].astype(float),
                np.ones(len(sample))
            ]).T)
            cx, cy = float(ras[0].mean()), float(ras[1].mean())
            print(f"[URETER] Vertebrae centroid: X={cx:.1f}  Y={cy:.1f} mm RAS")
            return cx, cy
        except Exception as e:
            print(f"[URETER] Could not compute vertebrae centroid ({e}) — using image centre")
            return 0.0, 0.0

    def _build_mask_from_seg(self, seg_node, dilate_mm, seg_name=None,
                             save_to_scene=False):
        """
        Export a segmentation node (or one of its sub-segments) to a binary
        numpy array, optionally dilate it by *dilate_mm* millimetres, and
        return (arr, affine).

        If *save_to_scene* is True the dilated mask is also written back into
        a new SegmentationNode named  <seg_node>_<seg_label>_dilated  so the
        user can inspect how far the dilation extends.
        """
        import numpy as np
        from scipy import ndimage

        lm = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode', '_excl_tmp')

        if seg_name:
            seg    = seg_node.GetSegmentation()
            seg_id = None
            for i in range(seg.GetNumberOfSegments()):
                if seg.GetNthSegment(i).GetName() == seg_name:
                    seg_id = seg.GetNthSegmentID(i)
                    break
            if seg_id is None:
                slicer.mrmlScene.RemoveNode(lm)
                raise ValueError(f"Segment '{seg_name}' not found in "
                                 f"'{seg_node.GetName()}'")
            seg_ids = vtk.vtkStringArray()
            seg_ids.InsertNextValue(seg_id)
            slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
                seg_node, seg_ids, lm, None,
                slicer.vtkSegmentation.EXTENT_UNION_OF_SEGMENTS)
        else:
            slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(
                seg_node, lm)

        arr = slicer.util.arrayFromVolume(lm).copy().astype(np.uint8)
        mat = vtk.vtkMatrix4x4()
        lm.GetIJKToRASMatrix(mat)
        affine = self._mat_to_np(mat)

        seg_label = seg_name or "all"
        if dilate_mm > 0:
            vox_size   = np.array([abs(affine[0, 0]), abs(affine[1, 1]), abs(affine[2, 2])])
            mean_vox   = float(vox_size.mean())
            iterations = max(1, int(round(dilate_mm / mean_vox)))
            struct     = ndimage.generate_binary_structure(3, 1)
            arr = ndimage.binary_dilation(
                arr > 0, structure=struct, iterations=iterations).astype(np.uint8)
            print(f"[EXCL] '{seg_node.GetName()}/{seg_label}': "
                  f"dilated {dilate_mm} mm ({iterations} iter) "
                  f"→ {int(arr.sum())} voxels")

        # ── Optionally persist the dilated mask as a visible scene node ───────
        if save_to_scene:
            dilated_name = f"{seg_node.GetName()}_{seg_label}_dilated"
            self._remove_existing(dilated_name)
            slicer.util.updateVolumeFromArray(lm, arr)
            dilated_seg = slicer.mrmlScene.AddNewNodeByClass(
                'vtkMRMLSegmentationNode', dilated_name)
            slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
                lm, dilated_seg)
            dilated_seg.CreateClosedSurfaceRepresentation()
            print(f"[EXCL] '{dilated_name}' added to scene.")

        slicer.mrmlScene.RemoveNode(lm)
        return arr, affine

    def _resample_to_target(self, src_arr, src_affine, tgt_shape, tgt_affine):
        import numpy as np
        from scipy.ndimage import map_coordinates

        z_idx, y_idx, x_idx = np.meshgrid(
            np.arange(tgt_shape[0]), np.arange(tgt_shape[1]), np.arange(tgt_shape[2]),
            indexing='ij')
        ijk_hom = np.stack([
            x_idx.ravel(), y_idx.ravel(), z_idx.ravel(),
            np.ones(x_idx.size)], axis=1).astype(np.float32)
        ras_pts = (tgt_affine @ ijk_hom.T).T
        src_ijk = (np.linalg.inv(src_affine) @ ras_pts.T).T[:, :3]
        coords  = np.array([src_ijk[:, 2], src_ijk[:, 1], src_ijk[:, 0]])
        resampled = map_coordinates(
            src_arr.astype(np.float32), coords, order=0, mode='constant', cval=0.0)
        return resampled.reshape(tgt_shape)
