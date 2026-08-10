"""
UreterPostProcess — Module + Widget.

  Widget → UreterPostProcessLogic → lib/processing/*
"""
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


try:
    from UreterPostProcessLogic import UreterPostProcessLogic
except ImportError:
    import importlib.util, os as _os
    _p = _os.path.join(_os.path.dirname(__file__), "UreterPostProcessLogic.py")
    _spec = importlib.util.spec_from_file_location("UreterPostProcessLogic", _p)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    UreterPostProcessLogic = getattr(_mod, "UreterPostProcessLogic")

def _default_mode(filename):
    nl = filename.lower()
    if any(k in nl for k in ('fat', 'iliopsoas', 'psoas')):
        return 'Clip + Clean'
    return 'Skip'


class UreterPostProcess(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title        = "4. Ureter Post-Process"
        self.parent.categories   = ["Metabolic Syndrome Toolkit"]
        self.parent.dependencies = []
        self.parent.contributors = ["IshitaSinghFaujdar"]
        self.parent.helpText = (
            "PET-guided ureter mask, per-organ L1-L5 clipping and/or ureter cleanup.\n"
            "For each organ select: Skip / Clip only / Clean only / Clip + Clean."
        )
        self.parent.acknowledgementText = ""


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

        # Warning shown when the selected TotalSeg node has no L1-L5 vertebrae
        self._vertebraeWarning = qt.QLabel()
        self._vertebraeWarning.setWordWrap(True)
        self._vertebraeWarning.setStyleSheet(
            "color:#b71c1c; background:#ffebee; padding:5px; border-radius:3px;")
        self._vertebraeWarning.setVisible(False)
        sceneLayout.addWidget(self._vertebraeWarning)

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

        # Filenames detected in the first patient's Segments/<ID>_Seg/ folder —
        # used to populate the exclusion-mask filename dropdowns below.
        self._bulkSegFiles = []

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

        addBulkOrganBtn = qt.QPushButton("+ Add organ")
        addBulkOrganBtn.setToolTip(
            "Add an organ row manually (choose or type a segment filename). "
            "Use this to process an organ that wasn't auto-detected in the scan.")
        addBulkOrganBtn.clicked.connect(self._on_add_bulk_organ)
        organsLayout.addWidget(addBulkOrganBtn)

        organsLayout.addStretch(1)
        self.bulkTabs.addTab(organsTab, "Organs")

        # ── Sub-tab 4: Exclusion Masks ─────────────────────────────────────────
        exclTab = qt.QWidget()
        exclTabLayout = qt.QVBoxLayout(exclTab)
        exclTabLayout.addWidget(qt.QLabel(
            "Extra exclusion masks (file-based equivalent of the Scene tab's "
            "'Extra exclusion masks'): each file is dilated, then voxels of "
            "every organ that overlap it AND exceed the SUV threshold are removed."))

        copyExclBtn = qt.QPushButton("↓ Copy from Scene Exclusion Masks above")
        copyExclBtn.setToolTip(
            "Derive exclusion-mask file rows from the 'Extra exclusion masks' "
            "configured on the Scene tab for the currently-loaded sample patient "
            "(TotalSegmentator naming convention).")
        copyExclBtn.clicked.connect(self._copy_excl_to_bulk)
        exclTabLayout.addWidget(copyExclBtn)

        self.copyExclResultLabel = qt.QLabel("")
        self.copyExclResultLabel.setWordWrap(True)
        exclTabLayout.addWidget(self.copyExclResultLabel)

        exclHdr2 = qt.QHBoxLayout()
        for txt, stretch in [("Filename", 4), ("Dilation mm", 2), ("SUV >", 2), ("", 0)]:
            lbl = qt.QLabel(txt)
            lbl.setStyleSheet("font-weight:bold;")
            exclHdr2.addWidget(lbl, stretch)
        exclHdr2.addSpacing(30)
        exclTabLayout.addLayout(exclHdr2)

        self._bulkExclRows = []
        self._bulkExclContainer = qt.QWidget()
        self._bulkExclContainerLayout = qt.QVBoxLayout(self._bulkExclContainer)
        self._bulkExclContainerLayout.setContentsMargins(0, 0, 0, 0)
        self._bulkExclContainerLayout.setSpacing(3)
        exclTabLayout.addWidget(self._bulkExclContainer)

        addBulkExclBtn = qt.QPushButton("+ Add exclusion mask file")
        addBulkExclBtn.clicked.connect(lambda: self._add_bulk_excl_row())
        exclTabLayout.addWidget(addBulkExclBtn)

        exclTabLayout.addStretch(1)
        self.bulkTabs.addTab(exclTab, "Exclusion Masks")

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
            self._bulkSegFiles = []
            self._refresh_bulk_filename_combos()
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

        # Populate exclusion-mask filename dropdowns (Exclusion Masks tab)
        self._bulkSegFiles = seg_files
        self._refresh_bulk_filename_combos()

    def _refresh_bulk_filename_combos(self):
        """Repopulate the dropdown items in every existing exclusion-mask row
        with the segment files just detected, preserving each row's current
        selection/typed text."""
        for rd in self._bulkExclRows:
            combo = rd['fname']
            current = combo.currentText
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(self._bulkSegFiles)
            combo.setCurrentText(current)
            combo.blockSignals(False)

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

    def _on_add_bulk_organ(self):
        """Add a manual organ row with an editable filename dropdown."""
        self._bulkOrganNoFilesLabel.setVisible(False)
        self._add_bulk_organ_row("", editable=True)

    def _add_bulk_organ_row(self, filename, mode=None, editable=False):
        if mode is None:
            mode = _default_mode(filename)
        frame = qt.QFrame()
        row = qt.QHBoxLayout(frame)
        row.setContentsMargins(0, 0, 0, 0)

        if editable:
            # User-added organ: editable dropdown of detected segment files
            # (still typable for files not present in the sample patient).
            file_widget = self._make_filename_combo(filename, "e.g. spleen.nii.gz")
            fname_get = lambda w=file_widget: w.currentText.strip()
        else:
            # Auto-discovered organ file: fixed read-only label.
            file_widget = qt.QLabel(filename)
            file_widget.setToolTip(filename)
            fname_get = lambda f=filename: f
        row.addWidget(file_widget, 3)

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

        self._bulkOrganRows.append((fname_get, mode_combo, frame))
        self._bulkOrganContainerLayout.addWidget(frame)

    def _remove_bulk_organ_row(self, frame):
        self._bulkOrganRows = [
            (f, c, fr) for f, c, fr in self._bulkOrganRows if fr is not frame]
        frame.setParent(None)

    def _get_bulk_organ_configs(self):
        configs = []
        for fname_get, combo, _ in self._bulkOrganRows:
            if combo.currentText == 'Skip':
                continue
            fname = fname_get()
            if not fname:
                continue
            configs.append((fname, combo.currentText))
        return configs

    # ── Bulk: exclusion mask rows (Exclusion Masks tab) ───────────────────────

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

    def _add_bulk_excl_row(self, filename="", dilate_mm=5.0, suv_thresh=1.2):
        frame = qt.QFrame()
        row   = qt.QHBoxLayout(frame)
        row.setContentsMargins(0, 1, 0, 1)

        fname_combo = self._make_filename_combo(filename, "e.g. femoral_vessels.nii.gz")

        dil_spin = qt.QDoubleSpinBox()
        dil_spin.setRange(0.0, 100.0)
        dil_spin.setSingleStep(1.0)
        dil_spin.setDecimals(1)
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
        rm_btn.clicked.connect(lambda: self._remove_bulk_excl_row(frame))

        row.addWidget(fname_combo, 4)
        row.addWidget(dil_spin, 2)
        row.addWidget(suv_spin, 2)
        row.addWidget(rm_btn)

        rd = {'widget': frame, 'fname': fname_combo, 'dilate': dil_spin, 'suv': suv_spin}
        self._bulkExclContainerLayout.addWidget(frame)
        self._bulkExclRows.append(rd)

    def _remove_bulk_excl_row(self, frame):
        self._bulkExclRows = [r for r in self._bulkExclRows if r['widget'] is not frame]
        frame.setParent(None)

    def _get_bulk_excl_configs(self):
        out = []
        for rd in self._bulkExclRows:
            fname = rd['fname'].currentText.strip()
            if fname:
                out.append({
                    'filename':   fname,
                    'dilate_mm':  rd['dilate'].value,
                    'suv_thresh': rd['suv'].value,
                })
        return out

    def _copy_excl_to_bulk(self):
        """
        Derive bulk exclusion-mask file rows from the "Extra exclusion masks"
        configured on the Scene tab for the currently-loaded sample patient.

        Mapping (TotalSegmentator file-naming convention):
          - row set to a specific sub-segment -> '<sub-segment>.nii.gz'
          - row set to 'All segments'         -> '<segmentation node name>.nii.gz'

        The resulting filenames are pre-filled but fully editable below
        before running bulk.
        """
        for rd in list(self._bulkExclRows):
            self._remove_bulk_excl_row(rd['widget'])

        n = 0
        for r in self._exclRows:
            seg_node = r['seg'].currentNode()
            if seg_node is None:
                continue
            seg_name_text = r['seg_name'].currentText
            fname = (f"{seg_node.GetName()}.nii.gz" if seg_name_text == "All segments"
                     else f"{seg_name_text}.nii.gz")
            self._add_bulk_excl_row(fname, r['dilate'].value, r['suv'].value)
            n += 1

        if n == 0:
            self.copyExclResultLabel.setText(
                "Nothing to copy — set up 'Extra exclusion masks' on the Scene "
                "tab above for a sample patient first, then click this button.")
            self.copyExclResultLabel.setStyleSheet("color:#b71c1c; font-style:italic;")
        else:
            self.copyExclResultLabel.setText(
                f"Copied {n} exclusion mask(s) — review/edit filenames below "
                f"before running.")
            self.copyExclResultLabel.setStyleSheet("color:#1b5e20; font-weight:bold;")

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
        self._vertebraeWarning.setVisible(False)
        if seg_node is None:
            return
        seg = seg_node.GetSegmentation()
        n_segs = seg.GetNumberOfSegments()
        if n_segs == 0:
            self._vertebraeWarning.setText(
                "⚠  This segmentation node has no segments — "
                "select a different TotalSeg node.")
            self._vertebraeWarning.setVisible(True)
            return
        auto_checked = 0
        for i in range(n_segs):
            name = seg.GetNthSegment(i).GetName()
            item = qt.QListWidgetItem(name)
            item.setFlags(item.flags() | qt.Qt.ItemIsUserCheckable)
            nl = name.lower()
            is_l1l5 = (
                any(f'l{v}' in nl for v in range(1, 6)) and
                any(k in nl for k in ('vertebra', 'vertebrae', 'segment'))
            ) or any(f'segment_{v}' == nl for v in range(1, 6))
            item.setCheckState(qt.Qt.Checked if is_l1l5 else qt.Qt.Unchecked)
            if is_l1l5:
                auto_checked += 1
            self.vertebraeList.addItem(item)
        if auto_checked == 0:
            self._vertebraeWarning.setText(
                f"⚠  No L1-L5 vertebrae detected in '{seg_node.GetName()}' "
                f"({n_segs} segment(s) found, none match the vertebra pattern). "
                f"This node likely has no vertebrae — select the original "
                f"TotalSegmentator output node instead, or check the correct "
                f"segments manually.")
            self._vertebraeWarning.setVisible(True)

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

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _set_status_error(self, msg: str):
        """Show a validation/runtime error in the status label (red, bold)."""
        self.statusLabel.setText(f"⚠  {msg}")
        self.statusLabel.setStyleSheet(
            "color:#b71c1c; font-weight:bold; "
            "background:#ffebee; padding:4px; border-radius:3px;")

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
            self._set_status_error("Pick a PET volume (Refresh from Scene).")
            return

        # Vertebrae are required for both Z-clipping AND ureter generation.
        # Ureter without Z bounds catches all bowel uptake — always blocked.
        needs_clip    = any(m in ('Clip only', 'Clip + Clean') for _, m in organ_configs)

        ts_node   = self.totalSegSelector.currentNode()
        vert_segs = self._get_checked_vertebrae()

        if needs_clip and ts_node is None:
            self._set_status_error(
                "Pick a TotalSeg node — required for Z-clipping (Clip only / Clip + Clean).")
            return
        if needs_clip and not vert_segs:
            self._set_status_error(
                "Check at least one L1-L5 vertebrae segment — "
                "required for Z-clipping. The selected TotalSeg node may not contain vertebrae.")
            return

        # Ureter generation without vertebrae produces a whole-body mask
        # (bowel / stomach uptake drowns out the real ureter). Block it.
        if generate_ureter and not vert_segs:
            self._set_status_error(
                "L1-L5 vertebrae are required to generate the ureter mask — "
                "without them the Z range is unrestricted and the mask will "
                "capture all bowel/stomach uptake. "
                "Select a TotalSeg node that contains the original vertebrae segments.")
            return

        if not organ_configs and not excl_configs and not generate_ureter:
            self._set_status_error(
                "Add at least one organ (mode ≠ Skip), exclusion mask, "
                "or enable 'Generate ureter mask'.")
            return

        print("\n[SCENE] Mapping:")
        self._print_mapping_summary()

        self.runButton.setEnabled(False)
        self.statusLabel.setText("Status: running on scene…")
        self.statusLabel.setStyleSheet("")
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
            self.statusLabel.setStyleSheet("color:#2e7d32; font-weight:bold;")
        except Exception:
            import traceback
            self._set_status_error("ERROR — see Python console.")
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

        excl_file_configs = self._get_bulk_excl_configs()

        print(f"[BULK] Vertebrae file: {parent_seg_file}")
        print(f"[BULK] Organ configs:")
        for fname, mode in organ_file_configs:
            print(f"[BULK]   {fname!r:50s} -> {mode}")
        print(f"[BULK] Exclusion mask configs:")
        for cfg in excl_file_configs:
            print(f"[BULK]   {cfg['filename']!r:50s} dilate={cfg['dilate_mm']} mm "
                  f"SUV>{cfg['suv_thresh']}")

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
                excl_file_configs   = excl_file_configs,
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

