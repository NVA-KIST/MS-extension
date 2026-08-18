"""
PETCTSegmentationModule — Slicer entry (Module metadata + Widget UI/triggers).

Logic: PETCTSegmentationModuleLogic.py → lib/io + lib/segmentation + mirroring.
PLog / BatchPatient / steps persistence: lib/io/logging_utils + batch_patient.

Semi-automatic, batch-oriented KU protocol PET-CT pre-processing &
segmentation module.
Modules → Metabolic Syndrome Toolkit → PETCT Segmentation Module

This module covers ONLY the first 3 steps of the old wizard (CT
pre-processing, TotalSegmentator, visceral-fat prediction, and mirroring
QC). The remaining steps now live in their own modules:

  - Segment Dilator       — dilate/subtract segmentations (bulk mode)
  - Vessel Segmentator     — seed-guided vessel growing (bulk mode, per
                              patient human seed placement)
  - Ureter Post Process    — PET-guided ureter mask, L1-L5 clip, organ
                              cleaning, exclusion masks (bulk mode)
  - Quantification         — PET metrics → Excel (batch)

Detect a root folder once, then walk through the steps below — each step's
"Run for All Patients" button processes every patient that is ready for it.

Steps:
  1. SEGMENTATION   [bulk]   CT DICOM → NIfTI, then TotalSegmentator
                              "total" + "body" + "tissue_types" tasks,
                              for every patient in the batch.
  2. VISCERAL FAT   [bulk]   Build the combined multi-label mask and run the
                              SegResNet (ASGSNet) visceral-fat prediction for
                              every patient.
  3. MIRROR QC      [human]  Step through patients one at a time. An axial
                              view overlays the TotalSegmentator torso-fat
                              segmentation with the predicted visceral-fat
                              segmentation. If the VF mask looks mirrored
                              left/right, click "Mirror (Flip X-axis)"; if it
                              looks mirrored front/back, click "Mirror (Flip
                              Y-axis)". Either can be applied repeatedly -
                              re-preview after each flip, then confirm.

Each patient's progress per step is persisted to
  root/pipeline_logs/{patient_key}.steps.json
so the queue table reflects the correct state after a Slicer restart.

ASGSNet note
------------
The ASGSNet script and model weights live anywhere on your filesystem.
Configure the checkpoint path in "1. Setup".  If not configured, step 2 logs
a warning and waits for you to drop a visceral_fat.nii.gz into the Seg folder
manually.

Logging
-------
Every method logs to:
  • Python console  (visible in Slicer Application Log)
  • root/pipeline_logs/{patient_key}.log  (one file per patient, appended)
Log fields: timestamp · level · patient key · message
"""
from __future__ import annotations

import os, re, json, threading, traceback, logging, subprocess, time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule, ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic, ScriptedLoadableModuleTest,
)

import sys as _sys
_EXT_NEW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _EXT_NEW_ROOT not in _sys.path:
    _sys.path.insert(0, _EXT_NEW_ROOT)

from lib.io.logging_utils import PatientLog as PLog
from lib.io.batch_patient import (
    BatchPatient,
    save_steps as _save_steps,
    load_steps as _load_steps,
    steps_file as _steps_file,
)
from lib.io.nifti import ok_nii as _ok_nii

try:
    import nibabel as nib
    _HAS_NIB = True
except ImportError:
    _HAS_NIB = False

_MODULE_LOG = logging.getLogger("PETCTSegmentationModule")


# ── Wizard step keys ──────────────────────────────────────────────────────────
# Each patient carries one status value per step (see ST_* below).
STEP_SEG     = "seg"      # 1. TotalSegmentator segmentation        [bulk]
STEP_VF      = "vf"       # 2. Visceral-fat (SegResNet) prediction   [bulk]
STEP_MIRROR  = "mirror"   # 3. Mirroring QC                          [human]

STEP_KEYS   = [STEP_SEG, STEP_VF, STEP_MIRROR]
HUMAN_STEPS = {STEP_MIRROR}

# Each step requires the previous step to be DONE (or SKIPPED) for that
# patient before the batch runner / human-step UI will process it.
STEP_PREREQ: Dict[str, Optional[str]] = {
    STEP_SEG:    None,
    STEP_VF:     STEP_SEG,
    STEP_MIRROR: STEP_VF,
}

STEP_TITLES: Dict[str, str] = {
    STEP_SEG:    "1. Segmentation",
    STEP_VF:     "2. Visceral Fat",
    STEP_MIRROR: "3. Mirror QC",
}

# ── Per-step status values ────────────────────────────────────────────────────
ST_PENDING = "PENDING"
ST_RUNNING = "RUNNING"
ST_DONE    = "DONE"
ST_ERROR   = "ERROR"
ST_SKIP    = "SKIPPED"

# Status → (display label, hex colour)
STATUS_STYLE: Dict[str, Tuple[str, str]] = {
    ST_PENDING: ("—",         "#9E9E9E"),
    ST_RUNNING: ("Running…",  "#1565C0"),
    ST_DONE:    ("Done ✓",    "#2E7D32"),
    ST_ERROR:   ("Error ✗",   "#B71C1C"),
    ST_SKIP:    ("Skipped",   "#757575"),
}

# TotalSegmentator output groups
VESSEL_FILES = [
    "aorta.nii.gz",
    "iliac_artery_left.nii.gz", "iliac_artery_right.nii.gz",
    "iliac_vena_left.nii.gz",   "iliac_vena_right.nii.gz",
    "inferior_vena_cava.nii.gz",
    "portal_vein_and_splenic_vein.nii.gz",
]
ABDOMINAL_FILES = [
    "spleen.nii.gz", "kidney_left.nii.gz", "kidney_right.nii.gz",
    "gallbladder.nii.gz", "liver.nii.gz", "stomach.nii.gz",
    "pancreas.nii.gz", "adrenal_gland_right.nii.gz", "adrenal_gland_left.nii.gz",
    "small_bowel.nii.gz", "duodenum.nii.gz", "colon.nii.gz",
    "urinary_bladder.nii.gz",
    "vertebrae_L1.nii.gz","vertebrae_L2.nii.gz","vertebrae_L3.nii.gz",
    "vertebrae_L4.nii.gz","vertebrae_L5.nii.gz",
]
PSOAS_FILES  = ["iliopsoas_left.nii.gz", "iliopsoas_right.nii.gz"]

# Per-TS-task "completeness" checklists used by runTS() / _autodetect_progress
# to decide whether a task needs to (re-)run. TotalSegmentator has no "fill in
# only the missing labels" mode - if ANY file in a task's checklist is
# missing/invalid, the WHOLE task is re-run, regenerating all of that task's
# output files (including ones that were already fine). These lists are kept
# to files referenced by name elsewhere in the pipeline (so they're
# unambiguous across TS versions) - other "total" outputs (ribs, T-vertebrae,
# heart, ...) are nice-to-have context for the combined mask and are handled
# gracefully (logged as missing, but not required) by buildCombinedMask().
TS_TOTAL_CHECK_FILES  = sorted(set(VESSEL_FILES + ABDOMINAL_FILES + PSOAS_FILES))
TS_BODY_CHECK_FILES   = ["body_trunc.nii.gz"]
TS_TISSUE_CHECK_FILES = ["torso_fat.nii.gz"]

# Combined-mask label scheme — MUST match visceral_fat_segmentations.py exactly.
# Label 0 = background (implicit, always).
# All L1-L5 vertebrae get label 5; all ribs get label 6; all T1-T12 get label 7.
# This 7-label flat scheme is what the SegResNet was trained on (label_nc=8).
#
# Lower labels form a base layer; higher labels overwrite (same priority order
# as create_combined_segmentation in visceral_fat_segmentations.py).
COMBINED_LABELS: Dict[str, int] = {
    # Label 1 — body trunk (base layer).
    #   Source: TotalSegmentator "body" task → body_trunc.nii.gz
    #   This is the outer body boundary that anchors all other labels spatially.
    "body_trunc": 1,
    # Label 2 — torso fat (total fat inside the torso).
    #   Source: TotalSegmentator "tissue_types" task → torso_fat.nii.gz
    #   Requires TotalSegmentator academic licence.
    #   Does NOT support --fast (the pipeline handles this automatically).
    #   This label gives the VF model fat-distribution context; keep it for best accuracy.
    "torso_fat":  2,
    # Label 3 — liver
    "liver":      3,
    # Label 4 — heart.
    # TotalSegmentator total task outputs "heart.nii.gz" (whole heart, not individual chambers).
    # "heart_myocardium" is kept as a fallback in case a different TS task or version is used.
    "heart":            4,
    "heart_myocardium": 4,
    # Label 5 — all L-vertebrae (L1-L5 all assigned the same label)
    "vertebrae_L1": 5, "vertebrae_L2": 5, "vertebrae_L3": 5,
    "vertebrae_L4": 5, "vertebrae_L5": 5,
    # Label 6 — all ribs (left and right, 1-12 all assigned the same label)
    "rib_left_1":  6, "rib_left_2":  6, "rib_left_3":  6,
    "rib_left_4":  6, "rib_left_5":  6, "rib_left_6":  6,
    "rib_left_7":  6, "rib_left_8":  6, "rib_left_9":  6,
    "rib_left_10": 6, "rib_left_11": 6, "rib_left_12": 6,
    "rib_right_1": 6, "rib_right_2": 6, "rib_right_3": 6,
    "rib_right_4": 6, "rib_right_5": 6, "rib_right_6": 6,
    "rib_right_7": 6, "rib_right_8": 6, "rib_right_9": 6,
    "rib_right_10":6, "rib_right_11":6, "rib_right_12":6,
    # Label 7 — all T-vertebrae (T1-T12 all assigned the same label)
    "vertebrae_T1": 7, "vertebrae_T2": 7, "vertebrae_T3": 7,
    "vertebrae_T4": 7, "vertebrae_T5": 7, "vertebrae_T6": 7,
    "vertebrae_T7": 7, "vertebrae_T8": 7, "vertebrae_T9": 7,
    "vertebrae_T10":7, "vertebrae_T11":7, "vertebrae_T12":7,
}


# ══════════════════════════════════════════════════════════════════════════════
# Per-patient logger
# ══════════════════════════════════════════════════════════════════════════════

def _mb(path: str) -> str:
    """Return file size as a human string, or 'not found'."""
    try:
        return f"{os.path.getsize(path)/1e6:.1f} MB"
    except OSError:
        return "not found"


def _nz(arr: np.ndarray) -> int:
    """Non-zero voxel count."""
    return int((arr > 0).sum())


# ══════════════════════════════════════════════════════════════════════════════
# Batch patient record
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# Module registration
# ══════════════════════════════════════════════════════════════════════════════


try:
    from PETCTSegmentationModuleLib.PETCTSegmentationModuleLogic import PETCTSegmentationModuleLogic
except ImportError:
    import importlib.util, os as _os
    _p = _os.path.join(_os.path.dirname(__file__), "PETCTSegmentationModuleLib", "PETCTSegmentationModuleLogic.py")
    _spec = importlib.util.spec_from_file_location("PETCTSegmentationModuleLogic", _p)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    PETCTSegmentationModuleLogic = getattr(_mod, "PETCTSegmentationModuleLogic")


_TS_FAST_UNSUPPORTED: set = set()
_TS_FAST_CHECKED: bool = False


def _ts_task_supports_fast(task: str, log=None) -> bool:
    """Return True if TotalSegmentator supports --fast for this task."""
    global _TS_FAST_UNSUPPORTED, _TS_FAST_CHECKED

    if not _TS_FAST_CHECKED:
        _TS_FAST_CHECKED = True
        try:
            import re as _re, inspect as _inspect
            from totalsegmentator import python_api as _ts_api
            src = _inspect.getsource(_ts_api.totalsegmentator)
            # Every unsupported task has a line like:
            #   if fast: raise ValueError("task XYZ does not work with option --fast")
            hits = _re.findall(
                r'if fast: raise ValueError\("task (\S+) does not work', src)
            _TS_FAST_UNSUPPORTED = set(hits)
            if log:
                log.info(f"TS fast-unsupported tasks detected: "
                         f"{len(_TS_FAST_UNSUPPORTED)} tasks")
        except Exception as e:
            # Conservative fallback — known unsupported tasks as of TS 2.x
            _TS_FAST_UNSUPPORTED = {
                "tissue_types", "tissue_types_mr", "tissue_4_types",
                "lung_vessels_LEGACY", "cerebral_bleed", "hip_implant",
                "pleural_pericard_effusion", "liver_vessels", "vertebrae_body",
                "heartchambers_highres", "appendicular_bones",
                "appendicular_bones_mr", "head_glands_cavities",
                "headneck_bones_vessels", "head_muscles", "headneck_muscles",
                "oculomotor_muscles", "lung_nodules", "kidney_cysts",
                "breasts", "ventricle_parts", "liver_segments",
                "liver_segments_mr", "liver_lesions", "liver_lesions_mr",
                "craniofacial_structures", "abdominal_muscles", "teeth",
                "trunk_cavities", "brain_aneurysm", "face", "face_mr",
                "brain_structures", "thigh_shoulder_muscles",
                "thigh_shoulder_muscles_mr", "coronary_arteries",
                "coronary_arteries_LEGACY", "aortic_sinuses",
            }
            if log:
                log.warn(f"Could not read TS source for fast-check ({e}) "
                         f"— using hardcoded fallback list "
                         f"({len(_TS_FAST_UNSUPPORTED)} tasks)")

    supported = task not in _TS_FAST_UNSUPPORTED
    return supported


class PETCTSegmentationModule(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title       = "1. PETCT Segmentation Module"
        parent.categories  = ["Metabolic Syndrome Toolkit"]
        parent.dependencies= []
        parent.contributors= ["IshitaSinghFaujdar"]
        parent.helpText    = (
            "Semi-automatic, batch KU protocol PET-CT pre-processing & "
            "segmentation. "
            "Step 1: TotalSegmentator (bulk). Step 2: visceral-fat prediction (bulk). "
            "Step 3: mirroring QC (human, per patient). "
            "Vessel growing, ureter post-processing and quantification have "
            "moved to their own modules (Vessel Segmentator, Ureter Post "
            "Process, Quantification)."
        )
        parent.acknowledgementText = ""


class PETCTSegmentationModuleWidget(ScriptedLoadableModuleWidget):

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self._logic = PETCTSegmentationModuleLogic()
        self._patients: Dict[str, BatchPatient] = {}
        self._order: List[str] = []

        self._bulk_thread: Optional[threading.Thread] = None
        self._bulk_stop  = threading.Event()
        self._bulk_step: Optional[str] = None

        self._mirror_idx = 0
        self._mirror_nodes: list = []

        self._build_ui()
        self._poll_timer = qt.QTimer()
        self._poll_timer.timeout.connect(self._poll)
        self._poll_timer.start(800)

    # ── UI helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _btn(text, bg="#1565C0", hover="#0d47a1"):
        b = qt.QPushButton(text)
        b.setStyleSheet(
            f"QPushButton{{background:{bg};color:white;font-weight:bold;"
            f"padding:6px 14px;border-radius:3px}}"
            f"QPushButton:hover{{background:{hover}}}")
        return b

    @staticmethod
    def _info(text):
        lbl = qt.QLabel(text)
        lbl.setWordWrap(True)
        lbl.setStyleSheet("padding:6px;background:#ECEFF1;border-radius:3px;")
        return lbl

    @staticmethod
    def _spin(lo, hi, val, dec=1, sfx=""):
        s = qt.QDoubleSpinBox()
        s.setRange(lo, hi); s.setValue(val); s.setDecimals(dec)
        if sfx: s.setSuffix(sfx)
        return s

    def _uiLog(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.logBox.appendPlainText(f"[{ts}] {msg}")

    # ── UI build ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        # 1. SETUP ─────────────────────────────────────────────────────────────
        s1 = ctk.ctkCollapsibleButton(); s1.text = "1. Setup"
        self.layout.addWidget(s1); f1 = qt.QFormLayout(s1)

        self.rootEdit = ctk.ctkPathLineEdit()
        self.rootEdit.filters = ctk.ctkPathLineEdit.Dirs
        f1.addRow("Root folder:", self.rootEdit)

        self.tsCmd = qt.QLineEdit("TotalSegmentator")
        self.tsCmd.setToolTip(
            "TotalSegmentator command — either just 'TotalSegmentator' if it is on "
            "PATH, or a full path like C:/envs/ts/Scripts/TotalSegmentator.exe")
        f1.addRow("TotalSegmentator:", self.tsCmd)

        self.ckptEdit = ctk.ctkPathLineEdit()
        self.ckptEdit.filters = ctk.ctkPathLineEdit.Files
        self.ckptEdit.nameFilters = ["Checkpoint files (*.ckpt *.pth *.h5)"]
        self.ckptEdit.setToolTip(
            "SegResNet checkpoint file for visceral-fat prediction (Step 2).\n"
            "Bundled checkpoints are in the extension's models/ folder:\n"
            "  epoch=399-step=8800.ckpt  <- recommended (most trained)\n"
            "  epoch=299-step=6600.ckpt\n"
            "  epoch=199-step=4400.ckpt\n\n"
            "Leave blank to skip VF prediction - drop visceral_fat.nii.gz\n"
            "into the patient Seg folder manually before Step 3.\n\n"
            "Note: first run installs torch + monai into Slicer's Python (~2 GB).")
        f1.addRow("VF model checkpoint:", self.ckptEdit)

        # Auto-populate with the best bundled checkpoint if it exists
        _ext_dir  = os.path.dirname(os.path.abspath(__file__))
        _best_ckpt = os.path.join(_ext_dir, "models", "epoch=399-step=8800.ckpt")
        if os.path.isfile(_best_ckpt):
            self.ckptEdit.setCurrentPath(_best_ckpt)

        self.gpuEdit = qt.QLineEdit("gpu")
        self.gpuEdit.setToolTip("'gpu', 'cpu', or device index like '0'.")
        f1.addRow("GPU device:", self.gpuEdit)

        detectBtn = self._btn("Detect Patients", "#1565C0", "#0d47a1")
        detectBtn.clicked.connect(self._on_detect)
        f1.addRow("", detectBtn)

        self.detectLabel = qt.QLabel("No folder selected.")
        f1.addRow("", self.detectLabel)

        f1.addRow("", self._info(
            "On detect, any step whose expected output files already exist on "
            "disk (e.g. segmentations / VF run previously, or copied in "
            "manually) is auto-marked Done so you can start at a later step. "
            "If a bulk step's required inputs are missing, it will fail with "
            "a clear message naming the step to run first instead of running "
            "for that patient."))

        # 2. PATIENT QUEUE ────────────────────────────────────────────────────
        s2 = ctk.ctkCollapsibleButton(); s2.text = "2. Patient Queue"
        self.layout.addWidget(s2); v2 = qt.QVBoxLayout(s2)

        self.table = qt.QTableWidget(0, 2 + len(STEP_KEYS))
        self.table.setHorizontalHeaderLabels(
            ["Patient ID", "Date"] + [STEP_TITLES[k] for k in STEP_KEYS])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionMode(qt.QAbstractItemView.NoSelection)
        self.table.setFixedHeight(200)
        v2.addWidget(self.table)

        btn_row = qt.QHBoxLayout()
        stopBtn = self._btn("■  Stop Current Batch Step", "#B71C1C", "#7f0000")
        stopBtn.clicked.connect(self._on_stop_bulk)
        btn_row.addWidget(stopBtn)

        openLogBtn = qt.QPushButton("Open Log Folder")
        openLogBtn.clicked.connect(self._on_open_log_folder)
        btn_row.addWidget(openLogBtn)
        v2.addLayout(btn_row)

        self.logBox = qt.QPlainTextEdit()
        self.logBox.setReadOnly(True); self.logBox.setFixedHeight(110)
        self.logBox.setStyleSheet("font-family:monospace;font-size:11px;")
        v2.addWidget(self.logBox)

        # STEP 1 — SEGMENTATION (bulk) ──────────────────────────────────────────
        s3 = ctk.ctkCollapsibleButton(); s3.text = "Step 1 — Segmentation (TotalSegmentator)"
        self.layout.addWidget(s3); v3 = qt.QVBoxLayout(s3)
        v3.addWidget(self._info(
            "Runs for every patient: CT DICOM -> NIfTI conversion, then "
            "TotalSegmentator 'total' + 'body' + 'tissue_types' tasks. "
            "Already-completed patients are skipped automatically."))
        self.seg1Btn = self._btn("▶  Run Segmentation for All Patients", "#2E7D32", "#1b5e20")
        self.seg1Btn.clicked.connect(self._on_run_seg)
        v3.addWidget(self.seg1Btn)
        self.seg1Status = qt.QLabel("No patients detected.")
        v3.addWidget(self.seg1Status)

        # STEP 2 — VISCERAL FAT (bulk) ──────────────────────────────────────────
        s4 = ctk.ctkCollapsibleButton(); s4.text = "Step 2 — Visceral Fat Segmentation"
        self.layout.addWidget(s4); v4 = qt.QVBoxLayout(s4)
        v4.addWidget(self._info(
            "Runs for every patient with Step 1 done: builds the combined "
            "multi-label mask and runs the SegResNet visceral-fat prediction."))
        self.seg2Btn = self._btn("▶  Run Visceral Fat for All Patients", "#2E7D32", "#1b5e20")
        self.seg2Btn.clicked.connect(self._on_run_vf)
        v4.addWidget(self.seg2Btn)
        self.seg2Status = qt.QLabel("No patients detected.")
        v4.addWidget(self.seg2Status)

        # STEP 3 — MIRROR QC (human, per-patient) ───────────────────────────────
        s5 = ctk.ctkCollapsibleButton(); s5.text = "Step 3 — Mirror QC (per patient)"
        self.layout.addWidget(s5); v5 = qt.QVBoxLayout(s5)
        v5.addWidget(self._info(
            "Step through patients one at a time. The axial view overlays the "
            "TotalSegmentator torso-fat segmentation (blue) with the predicted "
            "visceral-fat segmentation (orange). If the orange mask is on the "
            "wrong side of the body (left/right swapped), click 'Mirror (Flip "
            "X-axis)'. If it is shifted front-to-back (anterior/posterior "
            "swapped), click 'Mirror (Flip Y-axis)'. Either flip can be applied "
            "more than once if needed - check the preview again after each flip "
            "before confirming."))

        navRow = qt.QHBoxLayout()
        prevBtn = qt.QPushButton("◀ Prev")
        prevBtn.clicked.connect(lambda: self._on_mirror_nav(-1))
        navRow.addWidget(prevBtn)
        self.mirrorPatientLabel = qt.QLabel("No patients detected.")
        self.mirrorPatientLabel.setAlignment(qt.Qt.AlignCenter)
        navRow.addWidget(self.mirrorPatientLabel, 1)
        nextBtn = qt.QPushButton("Next ▶")
        nextBtn.clicked.connect(lambda: self._on_mirror_nav(1))
        navRow.addWidget(nextBtn)
        v5.addLayout(navRow)

        self.mirrorLoadBtn = self._btn("Load Preview (Axial)", "#1565C0", "#0d47a1")
        self.mirrorLoadBtn.clicked.connect(self._on_mirror_load_preview)
        v5.addWidget(self.mirrorLoadBtn)

        mirrorBtnRow = qt.QHBoxLayout()
        self.mirrorFlipXBtn = self._btn("Mirror (Flip X-axis)", "#E65100", "#bf360c")
        self.mirrorFlipXBtn.clicked.connect(lambda: self._on_mirror_flip("x"))
        mirrorBtnRow.addWidget(self.mirrorFlipXBtn)
        self.mirrorFlipYBtn = self._btn("Mirror (Flip Y-axis)", "#E65100", "#bf360c")
        self.mirrorFlipYBtn.clicked.connect(lambda: self._on_mirror_flip("y"))
        mirrorBtnRow.addWidget(self.mirrorFlipYBtn)
        v5.addLayout(mirrorBtnRow)

        confirmRow = qt.QHBoxLayout()
        self.mirrorConfirmBtn = self._btn("✓  Confirm — looks correct", "#2E7D32", "#1b5e20")
        self.mirrorConfirmBtn.clicked.connect(self._on_mirror_confirm)
        confirmRow.addWidget(self.mirrorConfirmBtn)
        v5.addLayout(confirmRow)

        self.mirrorStatus = qt.QLabel("")
        self.mirrorStatus.setWordWrap(True)
        v5.addWidget(self.mirrorStatus)

        # ADVANCED — TOTALSEGMENTATOR / VF INFERENCE PARAMETERS ─────────────────
        s6 = ctk.ctkCollapsibleButton()
        s6.text = "Advanced — TotalSegmentator / VF inference settings"
        s6.collapsed = True
        self.layout.addWidget(s6); f6 = qt.QFormLayout(s6)

        self.tsFast = qt.QCheckBox(
            "TotalSegmentator --fast  (lower resolution, ~2x faster - recommended)")
        self.tsFast.setChecked(True)
        f6.addRow(self.tsFast)

        self.tsOverlap = self._spin(0.0, 1.0, 0.10, dec=2)
        self.tsOverlap.setToolTip(
            "VF inference sliding-window overlap (0.10 = fast, 0.25 = accurate).\n"
            "0.10 is ~2.5x faster than 0.25 with minimal quality loss for VF.")
        f6.addRow("VF inference overlap:", self.tsOverlap)

        self.layout.addStretch(1)

    # ── Detection ────────────────────────────────────────────────────────────

    def _on_detect(self):
        root = self.rootEdit.currentPath
        if not root or not os.path.isdir(root):
            slicer.util.errorDisplay("Select a valid root folder first.")
            return
        try:
            scans = self._logic.discoverPatients(root)
        except Exception as e:
            slicer.util.errorDisplay(str(e)); return

        self._patients.clear()
        self._order = []
        restored = 0
        autodetected = 0
        for s in scans:
            rec = BatchPatient(s["subject_id"], s["scan_date"], root)
            if _load_steps(rec):
                restored += 1
            if self._autodetect_progress(rec):
                autodetected += 1
            self._patients[rec.key] = rec
            self._order.append(rec.key)

        self._mirror_idx = 0
        self._clear_mirror_preview()

        n = len(self._patients)
        msg = f"{n} patient-scan(s) found."
        if restored:
            msg += f"  {restored} with saved progress."
        if autodetected:
            msg += f"  {autodetected} with auto-detected existing outputs."
        self.detectLabel.setText(msg)
        self._uiLog(f"Detected {n} patient(s)  |  {restored} with saved progress"
                     + (f"  |  {autodetected} auto-detected from existing files on disk"
                        if autodetected else ""))
        self._rebuild_table()
        self._refresh_mirror_panel()

    def _autodetect_progress(self, rec: "BatchPatient") -> bool:
        """
        Allow resuming mid-pipeline when earlier steps were already completed
        outside this module (e.g. an older pipeline version, manual
        TotalSegmentator runs, or segmentations copied in from elsewhere).

        For any step still at its default ST_PENDING (i.e. nothing was
        restored from {key}.steps.json), check whether that step's expected
        output file(s) already exist on disk and look valid; if so, mark the
        step DONE so STEP_PREREQ no longer blocks the steps that follow it.

        Step 3 (Mirror QC) is a human-review gate and is not normally
        auto-marked done from files alone — EXCEPT that an existing, valid
        femoral_vessels.nii.gz (produced downstream by the Vessel Segmentator
        module, which requires Step 3 done first) is treated as proof Step 3
        was already completed for this patient. Re-open Step 3 by hand if you
        want to re-review a specific patient.

        Returns True if anything changed (and persists via _save_steps).
        """
        changed = False

        # Step 1 - Segmentation: CT NIfTI + every file referenced by name from
        # all 3 TS tasks (same checklists runTS() uses to decide whether a
        # task needs to (re-)run — see TS_*_CHECK_FILES).
        if rec.status[STEP_SEG] == ST_PENDING:
            check_files = TS_TOTAL_CHECK_FILES + TS_BODY_CHECK_FILES + TS_TISSUE_CHECK_FILES
            if _ok_nii(rec.ct_nii, min_kb=5_000) and all(
                    _ok_nii(rec.seg(f), min_kb=50) for f in check_files):
                rec.set(STEP_SEG, ST_DONE,
                        "Auto-detected: segmentation outputs already exist on disk.", 100)
                changed = True

        # Step 2 - Visceral fat: the predicted mask itself.
        if rec.status[STEP_VF] == ST_PENDING and rec.is_done(STEP_SEG):
            if _ok_nii(rec.seg("visceral_fat.nii.gz"), min_kb=50, need_nonzero=True):
                rec.set(STEP_VF, ST_DONE,
                        "Auto-detected: visceral_fat.nii.gz already exists on disk.", 100)
                changed = True

        # Step 3 - Mirror QC: only auto-complete when a downstream deliverable
        # (grown vessels, produced by the Vessel Segmentator module after this
        # step) already exists.
        if (rec.status[STEP_MIRROR] == ST_PENDING
                and _ok_nii(rec.seg("femoral_vessels.nii.gz"), min_kb=10, need_nonzero=True)):
            rec.set(STEP_MIRROR, ST_DONE,
                    "Auto-detected: pipeline outputs already exist on disk "
                    "(re-open this step to review if needed).", 100)
            changed = True

        if changed:
            _save_steps(rec)
        return changed

    # ── Table ────────────────────────────────────────────────────────────────

    def _rebuild_table(self):
        t = self.table; t.setRowCount(0)
        for key in self._order:
            rec = self._patients[key]
            row = t.rowCount; t.insertRow(row)
            t.setItem(row, 0, qt.QTableWidgetItem(rec.subject_id))
            t.setItem(row, 1, qt.QTableWidgetItem(rec.scan_date))
            for ci, step in enumerate(STEP_KEYS, start=2):
                status = rec.status[step]
                lbl, col = STATUS_STYLE.get(status, (status, "#9E9E9E"))
                text = f"  {lbl}  "
                if status == ST_RUNNING:
                    text = f"  {lbl} {rec.pct}%  "
                si = qt.QTableWidgetItem(text)
                si.setBackground(qt.QColor(col)); si.setForeground(qt.QColor("#FFFFFF"))
                t.setItem(row, ci, si)
        t.resizeColumnsToContents()

    def _on_open_log_folder(self):
        root = self.rootEdit.currentPath
        if root:
            ld = os.path.join(root, "pipeline_logs")
            os.makedirs(ld, exist_ok=True)
            qt.QDesktopServices.openUrl(qt.QUrl.fromLocalFile(ld))

    # ── Parameter collection ─────────────────────────────────────────────────

    def _collect_params(self) -> dict:
        return {
            "ts_cmd":       self.tsCmd.text.strip() or "TotalSegmentator",
            "ts_fast":      self.tsFast.isChecked(),
            "gpu":          self.gpuEdit.text.strip() or "gpu",
            "ckpt_path":    self.ckptEdit.currentPath,
            "vf_overlap":   self.tsOverlap.value,
        }

    # ── Patient navigation (Step 3) ─────────────────────────────────────────────

    def _nav_patient(self, idx_attr: str):
        if not self._order:
            return None
        idx = getattr(self, idx_attr)
        idx = max(0, min(idx, len(self._order) - 1))
        setattr(self, idx_attr, idx)
        return self._patients[self._order[idx]]

    def _advance_index(self, idx_attr: str, step: str):
        if not self._order:
            return
        n = len(self._order)
        cur = getattr(self, idx_attr)
        for off in range(1, n + 1):
            cand = (cur + off) % n
            if self._patients[self._order[cand]].status[step] == ST_PENDING:
                setattr(self, idx_attr, cand)
                return
        setattr(self, idx_attr, (cur + 1) % n)

    # ── Bulk-step infrastructure (Steps 1, 2) ───────────────────────────────────

    def _on_stop_bulk(self):
        self._bulk_stop.set()
        self._uiLog("Stop requested - the current batch step will halt after this patient.")

    def _step_ready(self, rec: "BatchPatient", step: str) -> bool:
        pre = STEP_PREREQ.get(step)
        if pre is None:
            return True
        return rec.is_done(pre)

    def _start_bulk(self, step: str, worker_fn):
        if not self._patients:
            slicer.util.errorDisplay("No patients detected. Click 'Detect Patients' first.")
            return
        if self._bulk_thread and self._bulk_thread.is_alive():
            slicer.util.errorDisplay(
                "Another batch step is already running.\n"
                "Click 'Stop Current Batch Step' or wait for it to finish.")
            return
        self._bulk_stop.clear()
        self._bulk_step = step
        p = self._collect_params()
        t = threading.Thread(target=self._bulk_runner, args=(step, worker_fn, p), daemon=True)
        self._bulk_thread = t
        t.start()
        self._uiLog(f"Started '{STEP_TITLES[step]}' for the batch.")

    def _bulk_runner(self, step: str, worker_fn, p: dict):
        for key in list(self._order):
            if self._bulk_stop.is_set():
                self._uiLog("Batch step stopped by user.")
                break
            rec = self._patients[key]
            if rec.status[step] in (ST_DONE, ST_SKIP):
                continue
            if not self._step_ready(rec, step):
                rec.set(step, ST_SKIP, "Prerequisite step not completed yet.", 0)
                _save_steps(rec)
                continue

            for d in [os.path.join(rec.root, "CT_NIfTI"), rec.seg_dir, rec.log_dir]:
                os.makedirs(d, exist_ok=True)
            log = PLog(rec.log_dir, rec.key)
            rec.set(step, ST_RUNNING, "Starting...", 0)
            try:
                worker_fn(rec, p, log, self._bulk_stop)
                if self._bulk_stop.is_set() and rec.status[step] == ST_RUNNING:
                    rec.set(step, ST_PENDING, "Stopped before completion.", rec.pct)
                else:
                    rec.set(step, ST_DONE, "Done.", 100)
                    log.ok(f"STEP '{step}' DONE for {rec.key}")
            except Exception as exc:
                rec.set(step, ST_ERROR, str(exc), rec.pct)
                log.error(f"STEP '{step}' FAILED for {rec.key}: {exc}")
                log.error(traceback.format_exc())
            finally:
                _save_steps(rec)
                log.close()
        self._bulk_step = None
        self._uiLog(f"Batch step '{STEP_TITLES.get(step, step)}' finished.")

    # ── Poll (UI timer, main thread) ────────────────────────────────────────────

    def _poll(self):
        if self._patients:
            self._rebuild_table()
            self._update_bulk_status()

    def _update_bulk_status(self):
        status_labels = {
            STEP_SEG:   self.seg1Status,
            STEP_VF:    self.seg2Status,
        }
        for step, lbl in status_labels.items():
            if self._bulk_step == step:
                running = next((r for r in self._patients.values()
                                 if r.status[step] == ST_RUNNING), None)
                if running:
                    lbl.setText(
                        f"Running: {running.key} - "
                        f"{running.message[step]} ({running.pct}%)")
                    continue
            n    = len(self._patients)
            done = sum(1 for r in self._patients.values() if r.status[step] == ST_DONE)
            err  = sum(1 for r in self._patients.values() if r.status[step] == ST_ERROR)
            skip = sum(1 for r in self._patients.values() if r.status[step] == ST_SKIP)
            txt = f"{done}/{n} done"
            if err:  txt += f"  -  {err} error(s)"
            if skip: txt += f"  -  {skip} skipped"
            lbl.setText(txt)

    # ── Step 1 — Segmentation (bulk) ─────────────────────────────────────────────

    def _on_run_seg(self):
        self._start_bulk(STEP_SEG, self._wf_segmentation)

    def _wf_segmentation(self, rec: "BatchPatient", p: dict, log: PLog, stop_evt: threading.Event):
        L = self._logic
        rec.set(STEP_SEG, ST_RUNNING, "CT DICOM -> NIfTI", 5)
        L.convertDicom(rec.ct_dcm, rec.ct_nii, log)

        if stop_evt.is_set(): return
        rec.set(STEP_SEG, ST_RUNNING, "TotalSegmentator - total", 30)
        L.runTS(rec.ct_nii, rec.seg_dir, "total",
                p["ts_cmd"], p["gpu"], p["ts_fast"],
                check_files=TS_TOTAL_CHECK_FILES, log=log)

        if stop_evt.is_set(): return
        rec.set(STEP_SEG, ST_RUNNING, "TotalSegmentator - body", 65)
        L.runTS(rec.ct_nii, rec.seg_dir, "body",
                p["ts_cmd"], p["gpu"], p["ts_fast"],
                check_files=TS_BODY_CHECK_FILES, log=log)

        if stop_evt.is_set(): return
        rec.set(STEP_SEG, ST_RUNNING, "TotalSegmentator - tissue_types", 90)
        L.runTS(rec.ct_nii, rec.seg_dir, "tissue_types",
                p["ts_cmd"], p["gpu"], p["ts_fast"],
                check_files=TS_TISSUE_CHECK_FILES, log=log)

    # ── Step 2 — Visceral Fat (bulk) ─────────────────────────────────────────────

    def _on_run_vf(self):
        self._start_bulk(STEP_VF, self._wf_visceral_fat)

    def _wf_visceral_fat(self, rec: "BatchPatient", p: dict, log: PLog, stop_evt: threading.Event):
        L = self._logic

        # Pre-flight: Step 2 needs the CT NIfTI + TotalSegmentator outputs
        # from Step 1. Fail fast with an actionable message instead of a deep
        # traceback from buildCombinedMask if the segmentation folder is
        # empty/missing (e.g. Step 1 was skipped/never run for this patient).
        if not _ok_nii(rec.ct_nii, min_kb=5_000):
            raise RuntimeError(
                f"Cannot run Step 2 (Visceral Fat): CT NIfTI not found/invalid "
                f"({rec.ct_nii}). Run Step 1 (Segmentation) first.")
        if not os.path.isdir(rec.seg_dir) or not any(
                f.endswith(".nii.gz") for f in os.listdir(rec.seg_dir)):
            raise RuntimeError(
                f"Cannot run Step 2 (Visceral Fat): no segmentation files found "
                f"in {rec.seg_dir}. Run Step 1 (Segmentation) first, or copy "
                f"existing TotalSegmentator outputs into that folder.")

        rec.set(STEP_VF, ST_RUNNING, "Building combined mask", 20)
        L.buildCombinedMask(rec.seg_dir, rec.seg("combined_mask.nii.gz"), log)

        if stop_evt.is_set(): return
        rec.set(STEP_VF, ST_RUNNING, "Visceral-fat prediction (SegResNet)", 60)
        L.runVFPrediction(
            combined=rec.seg("combined_mask.nii.gz"),
            ct_nii=rec.ct_nii,
            out=rec.seg("visceral_fat.nii.gz"),
            ckpt=p["ckpt_path"],
            device=p["gpu"],
            overlap=p.get("vf_overlap", 0.10),
            log=log)

    # ── Step 3 — Mirror QC (per patient, human) ─────────────────────────────────

    def _on_mirror_nav(self, delta: int):
        if not self._order: return
        self._mirror_idx = (self._mirror_idx + delta) % len(self._order)
        self._clear_mirror_preview()
        self._refresh_mirror_panel()

    def _refresh_mirror_panel(self):
        rec = self._nav_patient("_mirror_idx")
        if rec is None:
            self.mirrorPatientLabel.setText("No patients detected.")
            self.mirrorLoadBtn.setEnabled(False)
            return
        lbl, _col = STATUS_STYLE[rec.status[STEP_MIRROR]]
        self.mirrorPatientLabel.setText(
            f"Patient {self._mirror_idx + 1}/{len(self._order)}:  "
            f"{rec.subject_id}  [{rec.scan_date}]   -   {lbl}")
        ready = rec.is_done(STEP_VF)
        self.mirrorLoadBtn.setEnabled(ready)
        if not ready:
            self.mirrorStatus.setText(
                "Step 2 (Visceral Fat) is not done for this patient yet.")
        else:
            self.mirrorStatus.setText(
                "Click 'Load Preview' to compare the predicted visceral-fat mask "
                "against the TotalSegmentator torso-fat reference.")

    def _style_segment(self, seg_node, rgb, display_name):
        segmentation = seg_node.GetSegmentation()
        for i in range(segmentation.GetNumberOfSegments()):
            seg = segmentation.GetNthSegment(i)
            seg.SetName(display_name if i == 0 else f"{display_name} {i + 1}")
            seg.SetColor(*rgb)
        disp = seg_node.GetDisplayNode()
        if disp:
            disp.SetAllSegmentsVisibility(True)
            disp.SetOpacity(0.5)

    def _reset_slice_views(self):
        """
        Drop the background/foreground/label volume references on every slice
        view, then force a re-render. Without this, removing a volume node
        from the scene can leave the slice view still showing the last
        rendered frame (the composite node's stale reference isn't always
        repainted automatically), so the preview appears to "stick" even
        after the nodes backing it are gone.
        """
        try:
            slicer.util.setSliceViewerLayers(background=None, foreground=None,
                                              label=None, fit=False)
        except Exception:
            pass
        try:
            lm = slicer.app.layoutManager()
            for name in lm.sliceViewNames():
                sw = lm.sliceWidget(name)
                if sw:
                    sw.sliceView().forceRender()
        except Exception:
            pass

    def _clear_mirror_preview(self):
        self._reset_slice_views()
        for n in getattr(self, "_mirror_nodes", []):
            try:
                if n and slicer.mrmlScene.IsNodePresent(n):
                    slicer.mrmlScene.RemoveNode(n)
            except Exception:
                pass
        self._mirror_nodes = []

    def _on_mirror_load_preview(self):
        rec = self._nav_patient("_mirror_idx")
        if rec is None: return
        if not rec.is_done(STEP_VF):
            slicer.util.errorDisplay(
                "Step 2 (Visceral Fat) must be completed for this patient first.")
            return

        self._clear_mirror_preview()
        try:
            ct = slicer.util.loadVolume(rec.ct_nii)
            self._mirror_nodes.append(ct)

            lm = slicer.app.layoutManager()
            lm.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpRedSliceView)
            sw = lm.sliceWidget("Red"); sl = sw.sliceLogic()
            sl.GetSliceNode().SetOrientationToAxial()
            slicer.util.setSliceViewerLayers(background=ct)

            tf_path = rec.seg("torso_fat.nii.gz")
            if os.path.isfile(tf_path):
                tf_seg = slicer.util.loadSegmentation(tf_path)
                tf_seg.SetName(f"TS_torso_fat_{rec.key}")
                self._style_segment(tf_seg, (0.2, 0.6, 1.0), "TS torso fat")
                self._mirror_nodes.append(tf_seg)
            else:
                self._uiLog(f"  torso_fat.nii.gz not found for {rec.key} "
                             "- skipping reference overlay.")

            vf_path = rec.seg("visceral_fat.nii.gz")
            vf_seg = slicer.util.loadSegmentation(vf_path)
            vf_seg.SetName(f"VF_predicted_{rec.key}")
            self._style_segment(vf_seg, (1.0, 0.3, 0.0), "Predicted visceral fat")
            self._mirror_nodes.append(vf_seg)

            sl.FitSliceToAll()
            self.mirrorStatus.setText(
                "Compare the orange (predicted VF) mask against the blue (TS torso "
                "fat) reference. If the orange mask sits on the wrong side "
                "left/right, click 'Mirror (Flip X-axis)'. If it's shifted "
                "front/back, click 'Mirror (Flip Y-axis)'. Otherwise click "
                "'Confirm'.")
        except Exception as e:
            slicer.util.errorDisplay(f"Preview load failed: {e}")

    def _on_mirror_flip(self, axis: str = "x"):
        rec = self._nav_patient("_mirror_idx")
        if rec is None: return
        log = PLog(rec.log_dir, rec.key)
        try:
            if axis == "y":
                self._logic.flipNiftiYAxis(rec.seg("visceral_fat.nii.gz"), rec.ct_nii, log)
                rec.vf_flip_y = not rec.vf_flip_y
            else:
                self._logic.flipNiftiXAxis(rec.seg("visceral_fat.nii.gz"), rec.ct_nii, log)
                rec.vf_was_flipped = not rec.vf_was_flipped
            _save_steps(rec)
            self._uiLog(f"[{rec.key}] visceral_fat.nii.gz flipped on {axis.upper()}-axis.")
            self._on_mirror_load_preview()
        except Exception as e:
            slicer.util.errorDisplay(f"Flip failed: {e}")
        finally:
            log.close()

    def _on_mirror_confirm(self):
        rec = self._nav_patient("_mirror_idx")
        if rec is None: return
        flips = []
        if rec.vf_was_flipped: flips.append("X")
        if rec.vf_flip_y:      flips.append("Y")
        msg = f"Confirmed (flipped {'+'.join(flips)}-axis)" if flips else "Confirmed (no flip needed)"
        rec.set(STEP_MIRROR, ST_DONE, msg, 100)
        _save_steps(rec)
        self._uiLog(f"[{rec.key}] Mirror QC confirmed.")
        self._clear_mirror_preview()
        self._advance_index("_mirror_idx", STEP_MIRROR)
        self._refresh_mirror_panel()
        self._rebuild_table()

    # ── Cleanup ──────────────────────────────────────────────────────────────────

    def cleanup(self):
        self._poll_timer.stop()
        self._bulk_stop.set()
        self._clear_mirror_preview()

