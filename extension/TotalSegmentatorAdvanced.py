"""
TotalSegmentatorAdvanced.py
============================
3D Slicer Scripted Module — TotalSegmentator Advanced Runner

Wraps the totalsegmentator Python API with full task + sub-structure control:
  • Pick any task (total, abdominal, body, lung_vessels, etc.)
  • Checklist shows every structure available for that task
  • "Select All / None" buttons and a live search/filter box
  • Fast mode, GPU, and multilabel-output toggles
  • Output lands directly in the Slicer scene as a segmentation node

Requires the totalsegmentator package to be installed in Slicer's Python:
    pip_install("totalsegmentator")
"""

import os
import logging
import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import *

LOG = logging.getLogger("TotalSegAdv")

# ── Structure catalogue ────────────────────────────────────────────────────────
# Maps task name → ordered list of structure names that the task can produce.
# Used to populate the checklist; also forwarded as roi_subset to the API.

TASK_STRUCTURES = {
    "total": [
        "spleen", "kidney_right", "kidney_left", "gallbladder", "liver",
        "stomach", "aorta", "inferior_vena_cava", "portal_vein_and_splenic_vein",
        "pancreas", "adrenal_gland_right", "adrenal_gland_left",
        "lung_upper_lobe_left", "lung_lower_lobe_left",
        "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right",
        "vertebrae_L5", "vertebrae_L4", "vertebrae_L3", "vertebrae_L2", "vertebrae_L1",
        "vertebrae_T12", "vertebrae_T11", "vertebrae_T10", "vertebrae_T9", "vertebrae_T8",
        "vertebrae_T7", "vertebrae_T6", "vertebrae_T5", "vertebrae_T4", "vertebrae_T3",
        "vertebrae_T2", "vertebrae_T1",
        "vertebrae_C7", "vertebrae_C6", "vertebrae_C5", "vertebrae_C4",
        "vertebrae_C3", "vertebrae_C2", "vertebrae_C1",
        "esophagus", "trachea",
        "heart_myocardium", "heart_atrium_left", "heart_ventricle_left",
        "heart_atrium_right", "heart_ventricle_right",
        "pulmonary_artery", "brain",
        "iliac_artery_left", "iliac_artery_right",
        "iliac_vena_left", "iliac_vena_right",
        "small_bowel", "duodenum", "colon",
        "rib_left_1", "rib_left_2", "rib_left_3", "rib_left_4", "rib_left_5",
        "rib_left_6", "rib_left_7", "rib_left_8", "rib_left_9", "rib_left_10",
        "rib_left_11", "rib_left_12",
        "rib_right_1", "rib_right_2", "rib_right_3", "rib_right_4", "rib_right_5",
        "rib_right_6", "rib_right_7", "rib_right_8", "rib_right_9", "rib_right_10",
        "rib_right_11", "rib_right_12",
        "humerus_left", "humerus_right", "scapula_left", "scapula_right",
        "clavicula_left", "clavicula_right", "femur_left", "femur_right",
        "hip_left", "hip_right", "sacrum", "face",
        "gluteus_maximus_left", "gluteus_maximus_right",
        "gluteus_medius_left", "gluteus_medius_right",
        "gluteus_minimus_left", "gluteus_minimus_right",
        "autochthon_left", "autochthon_right",
        "iliopsoas_left", "iliopsoas_right",
        "urinary_bladder",
    ],
    "total_mr": [
        "spleen", "kidney_right", "kidney_left", "gallbladder", "liver",
        "stomach", "aorta", "inferior_vena_cava", "pancreas",
        "adrenal_gland_right", "adrenal_gland_left",
        "lung_upper_lobe_left", "lung_lower_lobe_left",
        "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right",
        "vertebrae_L5", "vertebrae_L4", "vertebrae_L3", "vertebrae_L2", "vertebrae_L1",
        "vertebrae_T12", "vertebrae_T11", "vertebrae_T10", "vertebrae_T9", "vertebrae_T8",
        "vertebrae_T7", "vertebrae_T6", "vertebrae_T5", "vertebrae_T4", "vertebrae_T3",
        "vertebrae_T2", "vertebrae_T1",
        "esophagus", "trachea",
        "heart_myocardium", "heart_atrium_left", "heart_ventricle_left",
        "heart_atrium_right", "heart_ventricle_right",
        "brain", "iliac_artery_left", "iliac_artery_right",
        "small_bowel", "duodenum", "colon",
        "femur_left", "femur_right", "hip_left", "hip_right", "sacrum",
        "gluteus_maximus_left", "gluteus_maximus_right",
        "gluteus_medius_left", "gluteus_medius_right",
        "gluteus_minimus_left", "gluteus_minimus_right",
        "autochthon_left", "autochthon_right",
        "iliopsoas_left", "iliopsoas_right", "urinary_bladder",
    ],
    "abdominal": [
        "colon", "duodenum", "esophagus", "gallbladder", "liver",
        "kidney_left", "kidney_right", "pancreas",
        "small_bowel", "spleen", "stomach", "urinary_bladder",
        "adrenal_gland_left", "adrenal_gland_right",
    ],
    "body": [
        "body_trunc", "body_extremities",
    ],
    "lung_vessels": [
        "lung_vessels", "lung_trachea_bronchia",
    ],
    "pleural_pericard_effusion": [
        "pleural_effusion", "pericardial_effusion",
    ],
    "liver_vessels": [
        "liver_vessels", "liver_tumor",
    ],
    "vertebrae_body": [
        "vertebrae_L5", "vertebrae_L4", "vertebrae_L3", "vertebrae_L2", "vertebrae_L1",
        "vertebrae_T12", "vertebrae_T11", "vertebrae_T10", "vertebrae_T9", "vertebrae_T8",
        "vertebrae_T7", "vertebrae_T6", "vertebrae_T5", "vertebrae_T4", "vertebrae_T3",
        "vertebrae_T2", "vertebrae_T1",
        "vertebrae_C7", "vertebrae_C6", "vertebrae_C5", "vertebrae_C4",
        "vertebrae_C3", "vertebrae_C2", "vertebrae_C1",
        "intervertebral_discs",
    ],
    "tissue_types": [
        "subcutaneous_fat", "torso_fat", "skeletal_muscle",
    ],
    "heartchambers_highres": [
        "heart_myocardium", "heart_atrium_left", "heart_ventricle_left",
        "heart_atrium_right", "heart_ventricle_right",
        "aorta_ascending",
    ],
    "aortic_branches": [
        "aorta", "brachiocephalic_trunk",
        "subclavian_artery_right", "subclavian_artery_left",
        "common_carotid_artery_right", "common_carotid_artery_left",
        "brachiocephalic_vein_left", "brachiocephalic_vein_right",
        "atrial_appendage_left", "superior_vena_cava",
        "renal_vein_left", "renal_vein_right",
        "heart_myocardium", "heart_atrium_left", "heart_ventricle_left",
        "heart_atrium_right", "heart_ventricle_right",
        "pulmonary_vein_right_upper", "pulmonary_vein_right_lower",
        "pulmonary_vein_left_upper", "pulmonary_vein_left_lower",
    ],
    "appendicular_bones": [
        "patella", "tibia", "fibula", "tarsal", "metatarsal", "phalanges_feet",
        "ulna", "radius", "carpal", "metacarpal", "phalanges_hand", "sternum",
    ],
    "bones_extremities": [
        "femur_left", "femur_right",
        "tibia_left", "tibia_right", "fibula_left", "fibula_right",
        "patella_left", "patella_right",
        "tarsal_left", "tarsal_right",
        "metatarsal_left", "metatarsal_right",
        "phalanges_feet_left", "phalanges_feet_right",
        "humerus_left", "humerus_right",
        "ulna_left", "ulna_right",
        "radius_left", "radius_right",
        "carpal_left", "carpal_right",
        "metacarpal_left", "metacarpal_right",
        "phalanges_hand_left", "phalanges_hand_right",
    ],
    "head_glands_cavities": [
        "eye_left", "eye_right", "eye_lens_left", "eye_lens_right",
        "optic_nerve_left", "optic_nerve_right",
        "parotid_gland_left", "parotid_gland_right",
        "submandibular_gland_right", "submandibular_gland_left",
        "lacrimal_glands_left", "lacrimal_glands_right",
        "cochlea_left", "cochlea_right",
        "thyroid_gland", "thymus", "pituitary_gland",
        "nasal_cavity_right", "nasal_cavity_left",
        "sinuses_sphenoid",
    ],
    "head_muscles": [
        "masseter_right", "masseter_left",
        "temporalis_right", "temporalis_left",
        "lateral_pterygoid_right", "lateral_pterygoid_left",
        "medial_pterygoid_right", "medial_pterygoid_left",
        "tongue", "digastric_right", "digastric_left",
    ],
    "headneck_bones_vessels": [
        "vertebrae_C1", "vertebrae_C2", "vertebrae_C3", "vertebrae_C4",
        "vertebrae_C5", "vertebrae_C6", "vertebrae_C7",
        "mandible", "skull", "hyoid",
        "carotid_artery_left", "carotid_artery_right",
        "jugular_vein_left", "jugular_vein_right",
    ],
    "headneck_muscles": [
        "sternocleidomastoid_right", "sternocleidomastoid_left",
        "trapezius_right", "trapezius_left",
        "levator_scapulae_right", "levator_scapulae_left",
        "anterior_scalene_right", "anterior_scalene_left",
        "middle_scalene_right", "middle_scalene_left",
        "posterior_scalene_right", "posterior_scalene_left",
        "splenius_cervicis_right", "splenius_cervicis_left",
        "semispinalis_capitis_right", "semispinalis_capitis_left",
    ],
    "female_pelvis": [
        "urinary_bladder", "uterus", "vagina",
        "femur_left", "femur_right", "hip_left", "hip_right", "sacrum",
    ],
    "cerebral_bleed": [
        "intracerebral_hemorrhage",
    ],
    "hip_implant": [
        "hip_implant_right", "hip_implant_left",
    ],
    "coronary_arteries": [
        "coronary_arteries",
    ],
    "face": [
        "face_region",
    ],
}

TASK_DESCRIPTIONS = {
    "total":                    "Full-body CT — ~100 structures",
    "total_mr":                 "Full-body MRI — ~60 structures",
    "abdominal":                "Abdominal organs (CT)",
    "body":                     "Body contour: trunk + extremities",
    "lung_vessels":             "Lung vessels + trachea/bronchia",
    "pleural_pericard_effusion":"Pleural & pericardial effusion",
    "liver_vessels":            "Liver vessels + liver tumour",
    "vertebrae_body":           "Vertebrae + intervertebral discs",
    "tissue_types":             "Subcutaneous fat, torso fat, muscle",
    "heartchambers_highres":    "Heart chambers (high-res)",
    "aortic_branches":          "Aorta + major branches + heart",
    "appendicular_bones":       "Appendicular skeleton",
    "bones_extremities":        "Extremity bones (left + right)",
    "head_glands_cavities":     "Head glands, orbits, cochlea…",
    "head_muscles":             "Mastication & tongue muscles",
    "headneck_bones_vessels":   "Head/neck bones + carotids",
    "headneck_muscles":         "Head/neck muscles",
    "female_pelvis":            "Female pelvis organs + bones",
    "cerebral_bleed":           "Intracerebral haemorrhage",
    "hip_implant":              "Hip implant detection",
    "coronary_arteries":        "Coronary arteries",
    "face":                     "Face region",
}


# ── MODULE REGISTRATION ────────────────────────────────────────────────────────

class TotalSegmentatorAdvanced(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title       = "TotalSegmentator Advanced"
        parent.categories  = ["Segmentation"]
        parent.dependencies = []
        parent.contributors = ["Ishita Singh Faujdar"]
        parent.helpText = (
            "Run TotalSegmentator with full control over which task and which "
            "individual structures to segment. Requires the totalsegmentator "
            "Python package (pip install totalsegmentator)."
        )


# ── WIDGET ─────────────────────────────────────────────────────────────────────

class TotalSegmentatorAdvancedWidget(ScriptedLoadableModuleWidget):

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = TotalSegmentatorAdvancedLogic()

        # ── 1. Input ──────────────────────────────────────────────────────────
        col1 = ctk.ctkCollapsibleButton()
        col1.text = "1. Input volume"
        self.layout.addWidget(col1)
        lay1 = qt.QFormLayout(col1)

        self.volumeSelector = slicer.qMRMLNodeComboBox()
        self.volumeSelector.nodeTypes            = ["vtkMRMLScalarVolumeNode"]
        self.volumeSelector.selectNodeUponCreation = True
        self.volumeSelector.addEnabled           = False
        self.volumeSelector.removeEnabled        = False
        self.volumeSelector.noneEnabled          = True
        self.volumeSelector.showHidden           = False
        self.volumeSelector.setMRMLScene(slicer.mrmlScene)
        self.volumeSelector.setToolTip("CT (or MR) volume to segment")
        lay1.addRow("Input volume:", self.volumeSelector)

        # ── 2. Task selection ─────────────────────────────────────────────────
        col2 = ctk.ctkCollapsibleButton()
        col2.text = "2. Task"
        self.layout.addWidget(col2)
        lay2 = qt.QVBoxLayout(col2)

        taskRow = qt.QHBoxLayout()
        taskRow.addWidget(qt.QLabel("Task:"))
        self.taskCombo = qt.QComboBox()
        for task in TASK_STRUCTURES:
            desc = TASK_DESCRIPTIONS.get(task, "")
            self.taskCombo.addItem(f"{task}   —   {desc}", task)
        self.taskCombo.setCurrentIndex(0)
        taskRow.addWidget(self.taskCombo, 1)
        lay2.addLayout(taskRow)

        self.taskInfoLabel = qt.QLabel()
        self.taskInfoLabel.setWordWrap(True)
        self.taskInfoLabel.setStyleSheet(
            "color:#0d47a1; background:#e3f2fd; padding:5px; border-radius:3px;")
        lay2.addWidget(self.taskInfoLabel)

        # ── 3. Structure selection ─────────────────────────────────────────────
        col3 = ctk.ctkCollapsibleButton()
        col3.text = "3. Structures (select which to segment)"
        self.layout.addWidget(col3)
        lay3 = qt.QVBoxLayout(col3)

        # Search bar
        searchRow = qt.QHBoxLayout()
        searchRow.addWidget(qt.QLabel("Filter:"))
        self.searchEdit = qt.QLineEdit()
        self.searchEdit.setPlaceholderText("type to filter structures…")
        self.searchEdit.textChanged.connect(self._applyFilter)
        searchRow.addWidget(self.searchEdit, 1)
        lay3.addLayout(searchRow)

        # Select All / None buttons
        btnRow = qt.QHBoxLayout()
        selAllBtn  = qt.QPushButton("Select All")
        selNoneBtn = qt.QPushButton("Select None")
        selAllBtn.clicked.connect(self._selectAll)
        selNoneBtn.clicked.connect(self._selectNone)
        btnRow.addWidget(selAllBtn)
        btnRow.addWidget(selNoneBtn)
        btnRow.addStretch(1)
        self.selCountLabel = qt.QLabel("0 / 0 selected")
        self.selCountLabel.setStyleSheet("color:#555; font-style:italic;")
        btnRow.addWidget(self.selCountLabel)
        lay3.addLayout(btnRow)

        # Checklist
        self.structList = qt.QListWidget()
        self.structList.setSelectionMode(qt.QAbstractItemView.NoSelection)
        self.structList.setAlternatingRowColors(True)
        self.structList.setFixedHeight(240)
        self.structList.itemChanged.connect(self._updateCount)
        lay3.addWidget(self.structList)

        # ── 4. Options ────────────────────────────────────────────────────────
        col4 = ctk.ctkCollapsibleButton()
        col4.text = "4. Options"
        col4.collapsed = True
        self.layout.addWidget(col4)
        lay4 = qt.QFormLayout(col4)

        self.fastCheck = qt.QCheckBox("Fast mode  (lower accuracy, much faster)")
        self.fastCheck.setChecked(False)
        lay4.addRow("", self.fastCheck)

        self.gpuCheck = qt.QCheckBox("Use GPU  (requires CUDA)")
        self.gpuCheck.setChecked(True)
        lay4.addRow("", self.gpuCheck)

        self.multilabelCheck = qt.QCheckBox(
            "Multilabel output  (single NIfTI with integer labels instead of one-per-structure)")
        self.multilabelCheck.setChecked(False)
        lay4.addRow("", self.multilabelCheck)

        self.outputNameEdit = qt.QLineEdit()
        self.outputNameEdit.setPlaceholderText("leave blank → use task name")
        lay4.addRow("Output seg name:", self.outputNameEdit)

        # ── Run ───────────────────────────────────────────────────────────────
        self.runBtn = qt.QPushButton("Run TotalSegmentator")
        self.runBtn.setStyleSheet(
            "QPushButton{background:#1565C0;color:white;font-weight:bold;"
            "padding:10px;border-radius:4px;font-size:13px;}"
            "QPushButton:hover{background:#0D47A1;}"
            "QPushButton:disabled{background:#888;}"
        )
        self.layout.addWidget(self.runBtn)

        self.progressBar = qt.QProgressBar()
        self.progressBar.setVisible(False)
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(0)   # indeterminate
        self.layout.addWidget(self.progressBar)

        self.statusLabel = qt.QLabel("Ready.")
        self.statusLabel.setWordWrap(True)
        self.layout.addWidget(self.statusLabel)

        self.layout.addStretch(1)

        # connections
        self.taskCombo.currentIndexChanged.connect(self._onTaskChanged)
        self.runBtn.clicked.connect(self.onRun)

        # initialise checklist for the first task
        self._onTaskChanged()

    # ── Task changed ──────────────────────────────────────────────────────────

    def _onTaskChanged(self):
        task = self.taskCombo.currentData
        structures = TASK_STRUCTURES.get(task, [])
        desc = TASK_DESCRIPTIONS.get(task, "")
        n = len(structures)
        self.taskInfoLabel.setText(
            f"<b>{task}</b>: {desc} — {n} structure{'s' if n != 1 else ''} available")

        self.structList.blockSignals(True)
        self.structList.clear()
        self.searchEdit.clear()
        for name in structures:
            item = qt.QListWidgetItem(name)
            item.setFlags(item.flags() | qt.Qt.ItemIsUserCheckable)
            item.setCheckState(qt.Qt.Checked)
            self.structList.addItem(item)
        self.structList.blockSignals(False)
        self._updateCount()

    # ── Filter / select helpers ───────────────────────────────────────────────

    def _applyFilter(self, text):
        text = text.lower()
        for i in range(self.structList.count):
            item = self.structList.item(i)
            item.setHidden(text not in item.text().lower())
        self._updateCount()

    def _selectAll(self):
        self.structList.blockSignals(True)
        for i in range(self.structList.count):
            item = self.structList.item(i)
            if not item.isHidden():
                item.setCheckState(qt.Qt.Checked)
        self.structList.blockSignals(False)
        self._updateCount()

    def _selectNone(self):
        self.structList.blockSignals(True)
        for i in range(self.structList.count):
            item = self.structList.item(i)
            if not item.isHidden():
                item.setCheckState(qt.Qt.Unchecked)
        self.structList.blockSignals(False)
        self._updateCount()

    def _updateCount(self):
        checked = sum(
            1 for i in range(self.structList.count)
            if self.structList.item(i).checkState() == qt.Qt.Checked
        )
        total = self.structList.count
        self.selCountLabel.setText(f"{checked} / {total} selected")

    def _getSelectedStructures(self):
        """Return list of checked structure names, or None if all are checked (= no subset filter)."""
        all_names = [
            self.structList.item(i).text()
            for i in range(self.structList.count)
        ]
        checked = [
            self.structList.item(i).text()
            for i in range(self.structList.count)
            if self.structList.item(i).checkState() == qt.Qt.Checked
        ]
        if len(checked) == len(all_names):
            return None   # all selected — don't pass roi_subset, run full task
        return checked if checked else None

    # ── Run ───────────────────────────────────────────────────────────────────

    def onRun(self):
        vol_node = self.volumeSelector.currentNode()
        if vol_node is None:
            slicer.util.errorDisplay("Please select an input volume.")
            return

        task       = self.taskCombo.currentData
        roi_subset = self._getSelectedStructures()

        if roi_subset is not None and len(roi_subset) == 0:
            slicer.util.errorDisplay("No structures selected — check at least one.")
            return

        out_name = self.outputNameEdit.text.strip() or task
        fast     = self.fastCheck.isChecked()
        gpu      = self.gpuCheck.isChecked()
        multilabel = self.multilabelCheck.isChecked()

        n_str = len(roi_subset) if roi_subset else len(TASK_STRUCTURES.get(task, []))
        self.statusLabel.setText(
            f"Running '{task}' — {n_str} structure(s)…  "
            f"({'fast' if fast else 'full'}, {'GPU' if gpu else 'CPU'})"
        )
        self.runBtn.setEnabled(False)
        self.progressBar.setVisible(True)

        # Warn immediately if GPU was requested but may not be available
        if gpu:
            try:
                import torch
                if not torch.cuda.is_available():
                    self.statusLabel.setText(
                        "GPU requested but CUDA not available in Slicer's PyTorch "
                        "— running on CPU. See Python console for upgrade instructions."
                    )
            except ImportError:
                pass
        slicer.app.processEvents()

        try:
            seg_node = self.logic.run(
                vol_node    = vol_node,
                task        = task,
                roi_subset  = roi_subset,
                fast        = fast,
                use_gpu     = gpu,
                multilabel  = multilabel,
                output_name = out_name,
            )
            self.statusLabel.setText(
                f"Done — '{seg_node.GetName()}' added to scene.")
        except Exception as e:
            import traceback
            msg = str(e)
            LOG.error(traceback.format_exc())
            self.statusLabel.setText(f"ERROR: {msg}")
            slicer.util.errorDisplay(f"TotalSegmentator failed:\n{msg}")
        finally:
            self.runBtn.setEnabled(True)
            self.progressBar.setVisible(False)


# ── LOGIC ──────────────────────────────────────────────────────────────────────

class TotalSegmentatorAdvancedLogic(ScriptedLoadableModuleLogic):

    def run(self, vol_node, task, roi_subset, fast, use_gpu,
            multilabel, output_name):
        """
        Export vol_node to a temp NIfTI, run totalsegmentator, load result.
        Returns the output vtkMRMLSegmentationNode.
        """
        import tempfile

        # ── 1. Fix pydicom version conflict ────────────────────────────────
        self._patch_pydicom_compat()

        # ── 2. Resolve device (checks actual CUDA availability) ────────────
        device = self._resolve_device(use_gpu)

        # ── 3. Import API (after patches so imports are clean) ─────────────
        try:
            import totalsegmentator.python_api as ts_api
        except ImportError:
            raise RuntimeError(
                "totalsegmentator is not installed.\n"
                "Open the Python console and run:\n"
                "  import subprocess, sys\n"
                "  subprocess.check_call([sys.executable, '-m', 'pip', "
                "'install', 'totalsegmentator'])"
            )

        with tempfile.TemporaryDirectory(prefix="tsadv_") as tmp:
            input_path = os.path.join(tmp, "input.nii.gz")
            output_dir = os.path.join(tmp, "output")
            os.makedirs(output_dir, exist_ok=True)

            # ── export input ───────────────────────────────────────────────
            print(f"[TSAdv] Exporting input to {input_path} …")
            slicer.util.saveNode(vol_node, input_path)

            # ── build kwargs, adapting to installed API version ────────────
            import inspect
            ts_params = set(inspect.signature(ts_api.totalsegmentator).parameters)

            print(f"[TSAdv] task={task}  fast={fast}  device={device}  "
                  f"roi_subset={roi_subset}  multilabel={multilabel}")
            slicer.app.processEvents()

            kwargs = dict(
                input  = input_path,
                output = output_dir,
                task   = task,
                fast   = fast,
                device = device,
            )
            if roi_subset:
                kwargs["roi_subset"] = roi_subset

            # Multilabel flag name varies: 'ml' in v2+, 'multilabel' in v1
            if multilabel:
                if "ml" in ts_params:
                    kwargs["ml"] = True
                elif "multilabel" in ts_params:
                    kwargs["multilabel"] = True
                else:
                    print("[TSAdv] WARNING: multilabel not supported — ignored")

            ts_api.totalsegmentator(**kwargs)
            slicer.app.processEvents()

            # ── load result ────────────────────────────────────────────────
            seg_node = self._load_output(output_dir, multilabel, output_name)
            print(f"[TSAdv] Done — '{seg_node.GetName()}' in scene.")
            return seg_node

    # ── Dependency / device helpers ───────────────────────────────────────────

    def _patch_pydicom_compat(self):
        """
        TotalSegmentator's import chain goes:
          nnunet.py → dicom_io.py → dicom2nifti → pydicom.pixels
        pydicom.pixels.decoders requires pydicom >= 2.4.0 (adds get_frame).
        Slicer ships an older pydicom → upgrade it in-place before importing.
        """
        import sys, subprocess

        # Quick probe: if get_frame already exists, nothing to do
        try:
            from pydicom.encaps import get_frame  # noqa
            return
        except (ImportError, AttributeError):
            pass

        import pydicom
        print(f"[TSAdv] pydicom {pydicom.__version__} is incompatible with "
              f"dicom2nifti — upgrading…")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--quiet",
                "--upgrade", "pydicom", "dicom2nifti"
            ])
        except Exception as e:
            raise RuntimeError(
                f"Auto-upgrade of pydicom failed: {e}\n\n"
                "Please fix manually in Slicer's Python console:\n"
                "  import subprocess, sys\n"
                "  subprocess.check_call([sys.executable, '-m', 'pip',\n"
                "    'install', '--upgrade', 'pydicom', 'dicom2nifti'])\n"
                "Then restart Slicer."
            )

        # Flush cached modules so re-imports pick up the new versions
        stale = [k for k in sys.modules
                 if k.startswith(("pydicom", "dicom2nifti",
                                  "totalsegmentator", "nnunet"))]
        for k in stale:
            del sys.modules[k]
        print("[TSAdv] pydicom upgraded and module cache cleared.")

    def _resolve_device(self, use_gpu):
        """
        Return 'gpu' only when PyTorch actually sees a CUDA device.
        If GPU was requested but is unavailable, print installation hint and
        fall back to CPU.
        """
        if not use_gpu:
            return "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                print(f"[TSAdv] GPU: {name}")
                return "gpu"
            else:
                print(
                    "[TSAdv] WARNING: GPU requested but torch.cuda.is_available() "
                    "= False.\n"
                    "[TSAdv] Slicer ships with a CPU-only PyTorch build. "
                    "To enable your GPU:\n"
                    "[TSAdv]   1. Open Slicer's Python console\n"
                    "[TSAdv]   2. Run:\n"
                    "[TSAdv]      import subprocess, sys\n"
                    "[TSAdv]      subprocess.check_call([\n"
                    "[TSAdv]        sys.executable, '-m', 'pip', 'install',\n"
                    "[TSAdv]        'torch', 'torchvision', 'torchaudio',\n"
                    "[TSAdv]        '--index-url',\n"
                    "[TSAdv]        'https://download.pytorch.org/whl/cu121'])\n"
                    "[TSAdv]   3. Restart Slicer\n"
                    "[TSAdv] Falling back to CPU for this run."
                )
                return "cpu"
        except ImportError:
            print("[TSAdv] torch not importable — using CPU")
            return "cpu"

    # ── helpers ───────────────────────────────────────────────────────────────

    def _load_output(self, output_dir, multilabel, output_name):
        """
        Load the segmentation result from output_dir into a scene segmentation node.
        Multilabel: single .nii.gz with integer labels.
        Non-multilabel: one .nii.gz per structure.
        """
        import glob

        # Remove any pre-existing node with the same name
        try:
            existing = slicer.util.getNode(output_name)
            slicer.mrmlScene.RemoveNode(existing)
        except Exception:
            pass

        seg_node = slicer.mrmlScene.AddNewNodeByClass(
            'vtkMRMLSegmentationNode', output_name)

        if multilabel:
            # Single file with integer labels — load as label map then import
            nii_files = glob.glob(os.path.join(output_dir, "*.nii.gz"))
            if not nii_files:
                raise RuntimeError(f"No output NIfTI found in {output_dir}")
            lm = slicer.util.loadLabelVolume(nii_files[0])
            slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
                lm, seg_node)
            slicer.mrmlScene.RemoveNode(lm)
        else:
            # One file per structure — load each as label map and merge
            nii_files = sorted(glob.glob(os.path.join(output_dir, "*.nii.gz")))
            if not nii_files:
                raise RuntimeError(f"No output NIfTI files found in {output_dir}")

            for fpath in nii_files:
                struct_name = os.path.basename(fpath).replace(".nii.gz", "")
                lm = slicer.util.loadLabelVolume(fpath)

                # Import as a new segment in seg_node
                import vtk
                seg_ids_before = set()
                seg = seg_node.GetSegmentation()
                for i in range(seg.GetNumberOfSegments()):
                    seg_ids_before.add(seg.GetNthSegmentID(i))

                slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
                    lm, seg_node)
                slicer.mrmlScene.RemoveNode(lm)

                # Rename the newly added segment to the structure name
                for i in range(seg.GetNumberOfSegments()):
                    sid = seg.GetNthSegmentID(i)
                    if sid not in seg_ids_before:
                        seg.GetSegment(sid).SetName(struct_name)
                        break

                slicer.app.processEvents()

        seg_node.CreateClosedSurfaceRepresentation()
        return seg_node


# ── TEST ───────────────────────────────────────────────────────────────────────

class TotalSegmentatorAdvancedTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        self.delayDisplay("TotalSegmentatorAdvanced — no automated test defined")
