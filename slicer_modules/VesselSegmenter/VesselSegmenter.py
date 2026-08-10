"""
VesselSegmenter — Slicer entry (Module metadata + Widget UI/triggers).
Logic lives in VesselSegmenterLogic.py and should call lib/ once implemented.
"""
"""
VesselSegmenter — 3D Slicer scripted module
==========================================

Segments large vessels (femoral veins/arteries, iliac vessels, etc.) that show
up as PET blood-pool uptake but are not included in TotalSegmentator.

Scene mode (single patient)
----------------------------
1. Select the PET volume (SUV units).
2. Set the SUV window that captures blood-pool signal (default 0.8 – 4.0).
3. Click "Place Seeds" and click once inside each vessel you want to extract.
4. Click "Run" — each seed extracts the connected blood-pool region it belongs to.
5. Optional: enable CT vesselness to restrict the mask to tubular structures.

Bulk mode (folder, per-patient seeds)
--------------------------------------
Unlike Segment Dilator / Ureter Post Process, vessel growing REQUIRES a human
to place seeds for every patient — there is no "sample setup -> apply to all".
The Bulk panel instead provides a Prev/Next stepper over
<dataset_root>/Segments/<ID>_Seg/ subjects:
  1. "Load PET (coronal view)" — loads that patient's PET DICOM from
     <dataset_root>/PET/<ID>_PET/, switches to a one-up coronal Red slice view
     with the PET-Rainbow colour map.
  2. "Place Seeds" / "Clear Seeds" — same seed markup workflow as Scene mode.
  3. "Grow Vessels" — calls VesselSegmenterLogic.run() UNCHANGED, using the
     SUV / Region / Post-Processing / Stitching / CT Vesselness parameters
     configured in the Scene tab above (those DO carry over to every patient),
     producing a green preview segmentation named 'femoral_vessels'.
  4. "Confirm & Save -> Next" — exports 'femoral_vessels' to
     Segments/<ID>_Seg/femoral_vessels.nii.gz, logs to pipeline_logs/bulk_log.txt,
     clears the preview/seeds and advances to the next patient.
"""

import os
import numpy as np
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
    from VesselSegmenterLogic import VesselSegmenterLogic
except ImportError:
    import importlib.util, os as _os
    _p = _os.path.join(_os.path.dirname(__file__), "VesselSegmenterLogic.py")
    _spec = importlib.util.spec_from_file_location("VesselSegmenterLogic", _p)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    VesselSegmenterLogic = getattr(_mod, "VesselSegmenterLogic")




class VesselSegmenter(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title        = "2. Vessel Segmenter"
        self.parent.categories   = ["Metabolic Syndrome Toolkit"]
        self.parent.dependencies = []
        self.parent.contributors = ["IshitaSinghFaujdar"]
        self.parent.helpText = (
            "Segment large vessels visible as PET blood-pool uptake.\n"
            "Place seed points inside vessels of interest and run segmentation.\n"
            "Suitable for femoral, iliac, and other large vessels."
        )
        self.parent.acknowledgementText = ""


class VesselSegmenterWidget(ScriptedLoadableModuleWidget):

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self._logic     = VesselSegmenterLogic()
        self._seed_node = None
        self._seed_obs  = None

        # ── Bulk-mode state ───────────────────────────────────────────────────
        self._bulk_subjects   = []
        self._bulk_idx        = -1
        self._bulk_root       = None
        self._bulk_pet_node   = None
        self._bulk_ct_node    = None
        self._bulk_seed_node  = None
        self._bulk_seed_obs   = None
        self._bulk_result_seg = None

        # ── Mode toggle ───────────────────────────────────────────────────────
        modeGroup  = qt.QGroupBox("Mode")
        modeLayout = qt.QHBoxLayout(modeGroup)
        self.sceneRadio = qt.QRadioButton("Scene (single patient)")
        self.bulkRadio  = qt.QRadioButton("Bulk (folder, per-patient seeds)")
        self.sceneRadio.setChecked(True)
        modeLayout.addWidget(self.sceneRadio)
        modeLayout.addWidget(self.bulkRadio)
        self.layout.addWidget(modeGroup)

        # ── Scene panel (wraps the original single-patient UI, unchanged) ──────
        self.scenePanel = qt.QFrame()
        sceneLayout = qt.QVBoxLayout(self.scenePanel)
        sceneLayout.setContentsMargins(0, 0, 0, 0)

        # ── Input volumes ─────────────────────────────────────────────────────
        inputsBox = ctk.ctkCollapsibleButton()
        inputsBox.text = "Input Volumes"
        sceneLayout.addWidget(inputsBox)
        inputsForm = qt.QFormLayout(inputsBox)

        self.petSelector = slicer.qMRMLNodeComboBox()
        self.petSelector.nodeTypes     = ['vtkMRMLScalarVolumeNode']
        self.petSelector.addEnabled    = False
        self.petSelector.removeEnabled = False
        self.petSelector.noneEnabled   = True
        self.petSelector.setMRMLScene(slicer.mrmlScene)
        self.petSelector.setToolTip("PET volume in SUV units")
        inputsForm.addRow("PET (SUV):", self.petSelector)

        self.ctSelector = slicer.qMRMLNodeComboBox()
        self.ctSelector.nodeTypes     = ['vtkMRMLScalarVolumeNode']
        self.ctSelector.addEnabled    = False
        self.ctSelector.removeEnabled = False
        self.ctSelector.noneEnabled   = True
        self.ctSelector.setMRMLScene(slicer.mrmlScene)
        self.ctSelector.setToolTip(
            "CT volume — optional, used only when CT vesselness is enabled.")
        inputsForm.addRow("CT (optional):", self.ctSelector)

        # Auto-select likely PET / CT
        for v in slicer.util.getNodesByClass('vtkMRMLScalarVolumeNode'):
            n = v.GetName().lower()
            if any(k in n for k in ('pet', 'suv', 'fdg')):
                self.petSelector.setCurrentNode(v)
            elif any(k in n for k in ('ct', 'ac', 'attn')):
                self.ctSelector.setCurrentNode(v)

        # ── PET threshold ─────────────────────────────────────────────────────
        petBox = ctk.ctkCollapsibleButton()
        petBox.text = "PET Blood-Pool Threshold"
        sceneLayout.addWidget(petBox)
        petForm = qt.QFormLayout(petBox)

        self.suvMinSpin = qt.QDoubleSpinBox()
        self.suvMinSpin.setRange(0.0, 20.0)
        self.suvMinSpin.setSingleStep(0.1)
        self.suvMinSpin.setDecimals(2)
        self.suvMinSpin.setValue(0.8)
        self.suvMinSpin.setToolTip(
            "Minimum SUV for blood-pool mask.\n"
            "Large vessels typically show SUV 0.8–1.8 on FDG-PET.")
        petForm.addRow("SUV min:", self.suvMinSpin)

        self.suvMaxSpin = qt.QDoubleSpinBox()
        self.suvMaxSpin.setRange(0.0, 100.0)
        self.suvMaxSpin.setSingleStep(0.1)
        self.suvMaxSpin.setDecimals(2)
        self.suvMaxSpin.setValue(4.0)
        self.suvMaxSpin.setToolTip(
            "Maximum SUV for blood-pool mask.\n"
            "Keeps hot tumour voxels out of the vessel mask.")
        petForm.addRow("SUV max:", self.suvMaxSpin)

        # ── Seeds ─────────────────────────────────────────────────────────────
        seedBox = ctk.ctkCollapsibleButton()
        seedBox.text = "Seed Points"
        sceneLayout.addWidget(seedBox)
        seedLayout = qt.QVBoxLayout(seedBox)

        seedInfo = qt.QLabel(
            "Place one seed inside each vessel you want to extract.\n"
            "The connected blood-pool region containing each seed becomes\n"
            "one segment in the output.")
        seedInfo.setWordWrap(True)
        seedInfo.setStyleSheet(
            "color:#37474f; padding:4px; background:#eceff1; border-radius:3px;")
        seedLayout.addWidget(seedInfo)

        seedBtnRow = qt.QHBoxLayout()
        self.placeSeedsBtn = qt.QPushButton("⊕  Place Seeds")
        self.placeSeedsBtn.setStyleSheet(
            "QPushButton{background:#1565c0;color:white;font-weight:bold;"
            "padding:6px 12px;border-radius:4px;}"
            "QPushButton:hover{background:#0d47a1;}")
        self.placeSeedsBtn.clicked.connect(self._on_place_seeds)
        seedBtnRow.addWidget(self.placeSeedsBtn)

        clearSeedsBtn = qt.QPushButton("Clear Seeds")
        clearSeedsBtn.setStyleSheet(
            "QPushButton{background:#546e7a;color:white;font-weight:bold;"
            "padding:6px 12px;border-radius:4px;}"
            "QPushButton:hover{background:#37474f;}")
        clearSeedsBtn.clicked.connect(self._on_clear_seeds)
        seedBtnRow.addWidget(clearSeedsBtn)
        seedLayout.addLayout(seedBtnRow)

        self._seedCountLabel = qt.QLabel("No seeds placed.")
        self._seedCountLabel.setStyleSheet("color:#666; font-style:italic;")
        seedLayout.addWidget(self._seedCountLabel)

        # ── Region constraint ─────────────────────────────────────────────────
        regionBox = ctk.ctkCollapsibleButton()
        regionBox.text = "Region Constraint"
        regionBox.collapsed = False
        sceneLayout.addWidget(regionBox)
        regionForm = qt.QFormLayout(regionBox)

        self.maxExtentSpin = qt.QDoubleSpinBox()
        self.maxExtentSpin.setRange(0.0, 500.0)
        self.maxExtentSpin.setSingleStep(10.0)
        self.maxExtentSpin.setValue(150.0)
        self.maxExtentSpin.setToolTip(
            "Maximum distance (mm) from each seed that growth is allowed.\n"
            "Prevents a seed in the femoral vein from capturing the entire\n"
            "aorta–heart blood pool (which may be connected).\n"
            "Set to 0 to disable (grow without distance limit).")
        regionForm.addRow("Max extent from seed (mm):", self.maxExtentSpin)

        # ── Post-processing ───────────────────────────────────────────────────
        morphBox = ctk.ctkCollapsibleButton()
        morphBox.text = "Post-Processing"
        morphBox.collapsed = True
        sceneLayout.addWidget(morphBox)
        morphForm = qt.QFormLayout(morphBox)

        self.closingRadiusSpin = qt.QDoubleSpinBox()
        self.closingRadiusSpin.setRange(0.0, 10.0)
        self.closingRadiusSpin.setSingleStep(0.5)
        self.closingRadiusSpin.setValue(2.0)
        self.closingRadiusSpin.setToolTip(
            "Morphological closing radius in mm to fill partial-volume gaps\n"
            "in the blood-pool signal.")
        morphForm.addRow("Closing radius (mm):", self.closingRadiusSpin)

        self.minVolSpin = qt.QDoubleSpinBox()
        self.minVolSpin.setRange(0.0, 1000.0)
        self.minVolSpin.setSingleStep(5.0)
        self.minVolSpin.setValue(5.0)
        self.minVolSpin.setToolTip(
            "Minimum connected component volume in mL.\n"
            "Smaller fragments are silently discarded.")
        morphForm.addRow("Min component volume (mL):", self.minVolSpin)

        # ── Vessel continuity (stitching) ─────────────────────────────────────
        stitchBox = ctk.ctkCollapsibleButton()
        stitchBox.text = "Vessel Continuity (stitch gaps)"
        stitchBox.collapsed = False
        sceneLayout.addWidget(stitchBox)
        stitchLayout = qt.QVBoxLayout(stitchBox)

        stitchInfo = qt.QLabel(
            "PET blood-pool signal has gaps between slices due to partial-volume\n"
            "effects.  This step finds all disconnected fragments inside the\n"
            "seed region and bridges any two that are closer than Max gap.")
        stitchInfo.setWordWrap(True)
        stitchInfo.setStyleSheet(
            "color:#37474f; padding:4px; background:#eceff1; border-radius:3px;")
        stitchLayout.addWidget(stitchInfo)

        stitchForm = qt.QFormLayout()
        stitchLayout.addLayout(stitchForm)

        self.stitchCheck = qt.QCheckBox("Stitch vessel gaps")
        self.stitchCheck.setChecked(True)
        stitchForm.addRow(self.stitchCheck)

        self.stitchGapSpin = qt.QDoubleSpinBox()
        self.stitchGapSpin.setRange(1.0, 100.0)
        self.stitchGapSpin.setSingleStep(2.0)
        self.stitchGapSpin.setValue(25.0)
        self.stitchGapSpin.setToolTip(
            "Maximum centroid-to-centroid gap in mm between two fragments\n"
            "for them to be bridged.  Raise if vessel has large PET gaps.")
        stitchForm.addRow("Max gap (mm):", self.stitchGapSpin)

        self.stitchRadiusSpin = qt.QDoubleSpinBox()
        self.stitchRadiusSpin.setRange(1.0, 20.0)
        self.stitchRadiusSpin.setSingleStep(1.0)
        self.stitchRadiusSpin.setValue(5.0)
        self.stitchRadiusSpin.setToolTip(
            "Radius in mm of the tube drawn to bridge each gap.\n"
            "Should roughly match the true vessel radius.")
        stitchForm.addRow("Bridge radius (mm):", self.stitchRadiusSpin)

        # ── CT Vesselness (optional) ──────────────────────────────────────────
        ctBox = ctk.ctkCollapsibleButton()
        ctBox.text = "CT Vesselness Filter (optional, ~30 s)"
        ctBox.collapsed = True
        sceneLayout.addWidget(ctBox)
        ctLayout = qt.QVBoxLayout(ctBox)

        ctInfo = qt.QLabel(
            "Applies a multi-scale Frangi vesselness filter to the CT to keep\n"
            "only tubular structures.  AND-ed with the PET mask to reduce\n"
            "false positives from non-vascular blood pool (bladder, heart, etc.).\n\n"
            "Works best for vessels surrounded by fat (thigh / pelvis).\n"
            "Requires SimpleITK (included in standard 3D Slicer).")
        ctInfo.setWordWrap(True)
        ctInfo.setStyleSheet("color:#37474f; padding:4px;")
        ctLayout.addWidget(ctInfo)

        ctForm = qt.QFormLayout()
        ctLayout.addLayout(ctForm)

        self.useVesselnessCheck = qt.QCheckBox("Enable CT vesselness masking")
        self.useVesselnessCheck.setChecked(False)
        ctForm.addRow(self.useVesselnessCheck)

        self.sigmaMinSpin = qt.QDoubleSpinBox()
        self.sigmaMinSpin.setRange(0.5, 15.0)
        self.sigmaMinSpin.setSingleStep(0.5)
        self.sigmaMinSpin.setValue(2.0)
        self.sigmaMinSpin.setToolTip("Minimum vessel radius in mm")
        ctForm.addRow("Min vessel radius (mm):", self.sigmaMinSpin)

        self.sigmaMaxSpin = qt.QDoubleSpinBox()
        self.sigmaMaxSpin.setRange(1.0, 30.0)
        self.sigmaMaxSpin.setSingleStep(0.5)
        self.sigmaMaxSpin.setValue(8.0)
        self.sigmaMaxSpin.setToolTip("Maximum vessel radius in mm")
        ctForm.addRow("Max vessel radius (mm):", self.sigmaMaxSpin)

        self.vesselThreshSpin = qt.QDoubleSpinBox()
        self.vesselThreshSpin.setRange(0.0, 1.0)
        self.vesselThreshSpin.setSingleStep(0.05)
        self.vesselThreshSpin.setDecimals(2)
        self.vesselThreshSpin.setValue(0.10)
        self.vesselThreshSpin.setToolTip(
            "Normalised vesselness response threshold (0–1).\n"
            "Lower = more sensitive but more noise.")
        ctForm.addRow("Vesselness threshold:", self.vesselThreshSpin)

        # ── Output ────────────────────────────────────────────────────────────
        outBox = ctk.ctkCollapsibleButton()
        outBox.text = "Output"
        sceneLayout.addWidget(outBox)
        outForm = qt.QFormLayout(outBox)

        self.outNameEdit = qt.QLineEdit("Vessels")
        self.outNameEdit.setToolTip(
            "Name of the output segmentation node.\n"
            "Individual segments are named Vessel_1, Vessel_2, …")
        outForm.addRow("Output name:", self.outNameEdit)

        self.saveProbMapCheck = qt.QCheckBox("Save blood-pool mask as volume")
        self.saveProbMapCheck.setChecked(False)
        self.saveProbMapCheck.setToolTip(
            "Saves the raw PET blood-pool binary mask as a scalar volume\n"
            "so you can inspect it before seeds are applied.")
        outForm.addRow(self.saveProbMapCheck)

        # ── Run ───────────────────────────────────────────────────────────────
        self.runBtn = qt.QPushButton("▶  Run Vessel Segmentation")
        self.runBtn.setStyleSheet(
            "QPushButton{background:#2e7d32;color:white;font-weight:bold;"
            "font-size:14px;padding:10px;border-radius:4px;}"
            "QPushButton:hover{background:#1b5e20;}")
        self.runBtn.clicked.connect(self._on_run)
        sceneLayout.addWidget(self.runBtn)

        self._statusLabel = qt.QLabel("")
        self._statusLabel.setWordWrap(True)
        self._statusLabel.setStyleSheet("padding:4px;")
        sceneLayout.addWidget(self._statusLabel)

        sceneLayout.addStretch(1)
        self.layout.addWidget(self.scenePanel)

        # ── Bulk panel (folder, per-patient seed placement) ──────────────────
        self.bulkPanel = qt.QFrame()
        bulkLayout = qt.QVBoxLayout(self.bulkPanel)
        bulkLayout.setContentsMargins(0, 0, 0, 0)

        # Dataset folder
        folderBox = ctk.ctkCollapsibleButton()
        folderBox.text = "Dataset Folder"
        bulkLayout.addWidget(folderBox)
        folderForm = qt.QFormLayout(folderBox)

        folderRow = qt.QHBoxLayout()
        self.bulkFolderEdit = qt.QLineEdit()
        self.bulkFolderEdit.setPlaceholderText("Path to dataset_clean …")
        folderRow.addWidget(self.bulkFolderEdit)
        bulkBrowseBtn = qt.QPushButton("Browse…")
        bulkBrowseBtn.setFixedWidth(80)
        bulkBrowseBtn.clicked.connect(self._on_bulk_browse)
        folderRow.addWidget(bulkBrowseBtn)
        folderForm.addRow("Folder:", folderRow)

        self.bulkDetectBtn = qt.QPushButton("Detect Subjects")
        self.bulkDetectBtn.clicked.connect(self._on_bulk_detect)
        folderForm.addRow(self.bulkDetectBtn)

        self.bulkScanLabel = qt.QLabel("No folder selected.")
        self.bulkScanLabel.setWordWrap(True)
        self.bulkScanLabel.setStyleSheet("color:#555; font-style:italic;")
        folderForm.addRow(self.bulkScanLabel)

        # Patient stepper
        patientBox = ctk.ctkCollapsibleButton()
        patientBox.text = "Patient (Prev / Next)"
        patientBox.collapsed = False
        bulkLayout.addWidget(patientBox)
        patientLayout = qt.QVBoxLayout(patientBox)

        patientInfo = qt.QLabel(
            "Vessel growing requires a human to place seeds for every patient — "
            "there is no 'sample setup -> apply to all'.  The SUV / Region / "
            "Post-Processing / Stitching / CT Vesselness parameters configured "
            "in the Scene tab above DO carry over to every patient.")
        patientInfo.setWordWrap(True)
        patientInfo.setStyleSheet(
            "color:#37474f; padding:4px; background:#eceff1; border-radius:3px;")
        patientLayout.addWidget(patientInfo)

        navRow = qt.QHBoxLayout()
        self.bulkPrevBtn = qt.QPushButton("<<  Prev")
        self.bulkPrevBtn.clicked.connect(lambda: self._on_bulk_nav(-1))
        navRow.addWidget(self.bulkPrevBtn)
        self.bulkPatientLabel = qt.QLabel("No subjects detected.")
        self.bulkPatientLabel.setAlignment(qt.Qt.AlignCenter)
        self.bulkPatientLabel.setStyleSheet("font-weight:bold;")
        navRow.addWidget(self.bulkPatientLabel, 1)
        self.bulkNextBtn = qt.QPushButton("Next  >>")
        self.bulkNextBtn.clicked.connect(lambda: self._on_bulk_nav(1))
        navRow.addWidget(self.bulkNextBtn)
        patientLayout.addLayout(navRow)

        # Step 1
        self.bulkLoadPetBtn = qt.QPushButton("1.  Load PET (coronal view)")
        self.bulkLoadPetBtn.setStyleSheet(
            "QPushButton{background:#1565c0;color:white;font-weight:bold;"
            "padding:6px;border-radius:4px;}"
            "QPushButton:hover{background:#0d47a1;}")
        self.bulkLoadPetBtn.clicked.connect(self._on_bulk_load_pet)
        patientLayout.addWidget(self.bulkLoadPetBtn)

        # Step 2
        seedRow2 = qt.QHBoxLayout()
        self.bulkPlaceSeedsBtn = qt.QPushButton("2.  Place Seeds")
        self.bulkPlaceSeedsBtn.setStyleSheet(
            "QPushButton{background:#1565c0;color:white;font-weight:bold;"
            "padding:6px;border-radius:4px;}"
            "QPushButton:hover{background:#0d47a1;}")
        self.bulkPlaceSeedsBtn.clicked.connect(self._on_bulk_place_seeds)
        seedRow2.addWidget(self.bulkPlaceSeedsBtn)
        self.bulkClearSeedsBtn = qt.QPushButton("Clear Seeds")
        self.bulkClearSeedsBtn.setStyleSheet(
            "QPushButton{background:#546e7a;color:white;font-weight:bold;"
            "padding:6px;border-radius:4px;}"
            "QPushButton:hover{background:#37474f;}")
        self.bulkClearSeedsBtn.clicked.connect(self._on_bulk_clear_seeds)
        seedRow2.addWidget(self.bulkClearSeedsBtn)
        patientLayout.addLayout(seedRow2)

        self.bulkSeedLabel = qt.QLabel("No seeds placed.")
        self.bulkSeedLabel.setStyleSheet("color:#666; font-style:italic;")
        patientLayout.addWidget(self.bulkSeedLabel)

        # Step 3
        self.bulkGrowBtn = qt.QPushButton("3.  Grow Vessels  (Scene-tab parameters)")
        self.bulkGrowBtn.setStyleSheet(
            "QPushButton{background:#2e7d32;color:white;font-weight:bold;"
            "padding:8px;border-radius:4px;}"
            "QPushButton:hover{background:#1b5e20;}")
        self.bulkGrowBtn.clicked.connect(self._on_bulk_grow)
        patientLayout.addWidget(self.bulkGrowBtn)

        # Step 4
        self.bulkConfirmBtn = qt.QPushButton("4.  Confirm && Save  ->  Next")
        self.bulkConfirmBtn.setStyleSheet(
            "QPushButton{background:#ef6c00;color:white;font-weight:bold;"
            "padding:8px;border-radius:4px;}"
            "QPushButton:hover{background:#e65100;}")
        self.bulkConfirmBtn.clicked.connect(self._on_bulk_confirm)
        patientLayout.addWidget(self.bulkConfirmBtn)

        self.bulkStatusLabel = qt.QLabel("")
        self.bulkStatusLabel.setWordWrap(True)
        self.bulkStatusLabel.setStyleSheet("padding:4px;")
        patientLayout.addWidget(self.bulkStatusLabel)

        bulkLayout.addStretch(1)
        self._refresh_bulk_panel()

        self.bulkPanel.setVisible(False)
        self.layout.addWidget(self.bulkPanel)

        self.sceneRadio.toggled.connect(self._on_mode_toggled)

        self.layout.addStretch(1)

    def cleanup(self):
        # Reset interaction mode so closing this window never leaves placement stuck
        try:
            interactionNode = slicer.app.applicationLogic().GetInteractionNode()
            interactionNode.SetCurrentInteractionMode(
                slicer.vtkMRMLInteractionNode.ViewTransform)
        except Exception:
            pass
        if self._seed_obs and self._seed_node:
            try:
                self._seed_node.RemoveObserver(self._seed_obs)
            except Exception:
                pass
        if self._bulk_seed_obs and self._bulk_seed_node:
            try:
                self._bulk_seed_node.RemoveObserver(self._bulk_seed_obs)
            except Exception:
                pass

    # ── Seeds ─────────────────────────────────────────────────────────────────

    def _on_place_seeds(self):
        if self._seed_node is None:
            self._seed_node = slicer.mrmlScene.AddNewNodeByClass(
                'vtkMRMLMarkupsFiducialNode', 'VesselSeeds')
            dn = self._seed_node.GetDisplayNode()
            if dn:
                dn.SetSelectedColor(1.0, 0.5, 0.0)
                dn.SetColor(1.0, 0.5, 0.0)
                dn.SetGlyphScale(3.0)
                dn.SetTextScale(3.0)
            self._seed_obs = self._seed_node.AddObserver(
                slicer.vtkMRMLMarkupsNode.PointAddedEvent,
                lambda c, e: self._update_seed_count())

        interactionNode = slicer.app.applicationLogic().GetInteractionNode()

        # Cancel any ongoing placement from another module first.
        # If we skip this and are already in Place mode (e.g. because Distance
        # Measurer left it there), calling Place again is a no-op and the old
        # markup node stays active.
        interactionNode.SetCurrentInteractionMode(
            slicer.vtkMRMLInteractionNode.ViewTransform)

        # Set active markup BEFORE re-entering Place mode
        slicer.modules.markups.logic().SetActiveListID(self._seed_node)

        interactionNode.SetCurrentInteractionMode(
            slicer.vtkMRMLInteractionNode.Place)
        self._update_seed_count()

    def _on_clear_seeds(self):
        if self._seed_node:
            self._seed_node.RemoveAllControlPoints()
        self._update_seed_count()

    def _update_seed_count(self):
        n = self._seed_node.GetNumberOfControlPoints() if self._seed_node else 0
        if n == 0:
            self._seedCountLabel.setText("No seeds placed.")
            self._seedCountLabel.setStyleSheet("color:#666; font-style:italic;")
        else:
            self._seedCountLabel.setText(f"{n} seed(s) placed.")
            self._seedCountLabel.setStyleSheet("color:#1b5e20; font-weight:bold;")

    # ── Run ───────────────────────────────────────────────────────────────────

    def _on_run(self):
        pet_node = self.petSelector.currentNode()
        if pet_node is None:
            slicer.util.errorDisplay(
                "Please select a PET volume.", windowTitle="Vessel Segmenter")
            return

        seeds = []
        if self._seed_node:
            for i in range(self._seed_node.GetNumberOfControlPoints()):
                p = [0.0, 0.0, 0.0]
                self._seed_node.GetNthControlPointPositionWorld(i, p)
                seeds.append(list(p))
        if not seeds:
            slicer.util.errorDisplay(
                "Please place at least one seed point inside the vessel.",
                windowTitle="Vessel Segmenter")
            return

        self._set_status("Running…", "#1565c0")
        self.runBtn.setEnabled(False)
        slicer.app.processEvents()

        try:
            seg_node = self._logic.run(
                pet_node        = pet_node,
                ct_node         = self.ctSelector.currentNode(),
                seeds_ras       = seeds,
                suv_min         = self.suvMinSpin.value,
                suv_max         = self.suvMaxSpin.value,
                max_extent_mm   = self.maxExtentSpin.value,
                closing_mm      = self.closingRadiusSpin.value,
                min_vol_ml      = self.minVolSpin.value,
                stitch_gaps     = self.stitchCheck.isChecked(),
                stitch_gap_mm   = self.stitchGapSpin.value,
                stitch_radius_mm= self.stitchRadiusSpin.value,
                use_vesselness  = self.useVesselnessCheck.isChecked(),
                sigma_min_mm    = self.sigmaMinSpin.value,
                sigma_max_mm    = self.sigmaMaxSpin.value,
                vessel_thresh   = self.vesselThreshSpin.value,
                out_name        = self.outNameEdit.text.strip() or "Vessels",
                save_mask_vol   = self.saveProbMapCheck.isChecked(),
            )
            n = seg_node.GetSegmentation().GetNumberOfSegments()
            self._set_status(
                f"✓ Done — {n} vessel segment(s) in '{seg_node.GetName()}'.",
                "#1b5e20")
        except Exception as exc:
            import traceback
            self._set_status(f"Error: {exc}", "#b71c1c")
            slicer.util.errorDisplay(
                traceback.format_exc(), windowTitle="Vessel Segmenter")
        finally:
            self.runBtn.setEnabled(True)

    def _set_status(self, msg, color="#333"):
        self._statusLabel.setText(msg)
        self._statusLabel.setStyleSheet(f"color:{color}; padding:4px;")

    # ── Mode toggle ───────────────────────────────────────────────────────────

    def _on_mode_toggled(self, scene_checked):
        self.scenePanel.setVisible(scene_checked)
        self.bulkPanel.setVisible(not scene_checked)

    # ── Bulk: folder / subject detection ────────────────────────────────────────

    def _on_bulk_browse(self):
        folder = qt.QFileDialog.getExistingDirectory(
            None, "Select dataset folder", self.bulkFolderEdit.text or "")
        if folder:
            self.bulkFolderEdit.setText(folder)

    def _on_bulk_detect(self):
        root = self.bulkFolderEdit.text.strip()
        if not root or not os.path.isdir(root):
            self.bulkScanLabel.setText("Folder not found.")
            self.bulkScanLabel.setStyleSheet("color:#b71c1c; font-style:italic;")
            self._bulk_root     = None
            self._bulk_subjects = []
            self._bulk_idx      = -1
            self._refresh_bulk_panel()
            return

        self._bulk_root     = root
        self._bulk_subjects = self._logic.find_bulk_subjects(root)
        if not self._bulk_subjects:
            self.bulkScanLabel.setText(
                "No '<ID>_Seg' folders found under Segments/.")
            self.bulkScanLabel.setStyleSheet("color:#b71c1c; font-style:italic;")
            self._bulk_idx = -1
        else:
            self.bulkScanLabel.setText(
                f"{len(self._bulk_subjects)} subject(s) found.")
            self.bulkScanLabel.setStyleSheet("color:#1b5e20; font-weight:bold;")
            self._bulk_idx = 0
        self._refresh_bulk_panel()

    def _refresh_bulk_panel(self):
        has_subjects = bool(self._bulk_subjects) and self._bulk_idx >= 0
        for b in (self.bulkLoadPetBtn, self.bulkPlaceSeedsBtn,
                  self.bulkClearSeedsBtn, self.bulkGrowBtn,
                  self.bulkConfirmBtn, self.bulkPrevBtn, self.bulkNextBtn):
            b.setEnabled(has_subjects)

        if not has_subjects:
            self.bulkPatientLabel.setText("No subjects detected.")
            self.bulkStatusLabel.setText("")
            self._update_bulk_seed_label()
            return

        subject_id = self._bulk_subjects[self._bulk_idx]
        n          = len(self._bulk_subjects)
        out_path   = os.path.join(
            self._bulk_root, 'Segments', f"{subject_id}_Seg", "femoral_vessels.nii.gz")
        done = os.path.isfile(out_path)
        self.bulkPatientLabel.setText(
            f"Patient {self._bulk_idx + 1}/{n}:  {subject_id}" +
            ("   [done]" if done else ""))
        self.bulkStatusLabel.setText(
            "Step 1: Load PET (coronal view), then place seed(s) and grow vessels.")
        self._update_bulk_seed_label()

    # ── Bulk: patient navigation ─────────────────────────────────────────────────

    def _on_bulk_nav(self, delta):
        if not self._bulk_subjects:
            return
        self._bulk_cleanup_patient_nodes()
        self._bulk_idx = (self._bulk_idx + delta) % len(self._bulk_subjects)
        self._refresh_bulk_panel()

    def _bulk_cleanup_patient_nodes(self):
        self._clear_bulk_preview()
        self._bulk_clear_seed_node()
        for attr in ('_bulk_pet_node', '_bulk_ct_node'):
            node = getattr(self, attr, None)
            if node:
                try:
                    if slicer.mrmlScene.IsNodePresent(node):
                        slicer.mrmlScene.RemoveNode(node)
                except Exception:
                    pass
            setattr(self, attr, None)

    # ── Bulk: Step 1 — load PET (coronal view) ───────────────────────────────────

    def _on_bulk_load_pet(self):
        if not self._bulk_subjects:
            return
        subject_id = self._bulk_subjects[self._bulk_idx]
        pet_dir    = os.path.join(self._bulk_root, 'PET', f"{subject_id}_PET")

        self.bulkStatusLabel.setText("Loading PET…")
        self.bulkLoadPetBtn.setEnabled(False)
        slicer.app.processEvents()
        try:
            pet_node = self._logic.load_pet_dicom(pet_dir, subject_id)
            if pet_node is None:
                raise RuntimeError(f"Could not load PET DICOM from:\n{pet_dir}")
            self._bulk_pet_node = pet_node

            # Optional CT for vesselness — read the Scene-tab checkbox.
            self._bulk_ct_node = None
            if self.useVesselnessCheck.isChecked():
                ct_path = os.path.join(
                    self._bulk_root, 'CT_NIfTI', f"{subject_id}_CT.nii.gz")
                if os.path.isfile(ct_path):
                    ct_node = slicer.util.loadVolume(ct_path)
                    ct_node.SetName(f"CT_{subject_id}")
                    self._bulk_ct_node = ct_node
                else:
                    print(f"[VESSEL][BULK] CT not found at {ct_path} "
                          "— vesselness will be skipped for this patient")

            self._logic.setup_coronal_pet_view(pet_node)
            self.bulkStatusLabel.setText(
                "PET loaded. Step 2: place seed(s) inside the vessel(s).")
        except Exception as exc:
            import traceback
            self.bulkStatusLabel.setText(f"Error: {exc}")
            slicer.util.errorDisplay(
                traceback.format_exc(), windowTitle="Vessel Segmenter")
        finally:
            self.bulkLoadPetBtn.setEnabled(True)

    # ── Bulk: Step 2 — seeds ──────────────────────────────────────────────────────

    def _on_bulk_place_seeds(self):
        if self._bulk_seed_node is None:
            self._bulk_seed_node = slicer.mrmlScene.AddNewNodeByClass(
                'vtkMRMLMarkupsFiducialNode', 'VesselSeeds_Bulk')
            dn = self._bulk_seed_node.GetDisplayNode()
            if dn:
                dn.SetSelectedColor(1.0, 0.5, 0.0)
                dn.SetColor(1.0, 0.5, 0.0)
                dn.SetGlyphScale(3.0)
                dn.SetTextScale(3.0)
            self._bulk_seed_obs = self._bulk_seed_node.AddObserver(
                slicer.vtkMRMLMarkupsNode.PointAddedEvent,
                lambda c, e: self._update_bulk_seed_label())

        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        interactionNode.SetCurrentInteractionMode(
            slicer.vtkMRMLInteractionNode.ViewTransform)
        slicer.modules.markups.logic().SetActiveListID(self._bulk_seed_node)
        interactionNode.SetCurrentInteractionMode(
            slicer.vtkMRMLInteractionNode.Place)
        self._update_bulk_seed_label()

    def _on_bulk_clear_seeds(self):
        if self._bulk_seed_node:
            self._bulk_seed_node.RemoveAllControlPoints()
        self._update_bulk_seed_label()

    def _update_bulk_seed_label(self):
        n = (self._bulk_seed_node.GetNumberOfControlPoints()
             if self._bulk_seed_node else 0)
        if n == 0:
            self.bulkSeedLabel.setText("No seeds placed.")
            self.bulkSeedLabel.setStyleSheet("color:#666; font-style:italic;")
        else:
            self.bulkSeedLabel.setText(f"{n} seed(s) placed.")
            self.bulkSeedLabel.setStyleSheet("color:#1b5e20; font-weight:bold;")

    def _bulk_clear_seed_node(self):
        if self._bulk_seed_obs and self._bulk_seed_node:
            try:
                self._bulk_seed_node.RemoveObserver(self._bulk_seed_obs)
            except Exception:
                pass
        if self._bulk_seed_node:
            try:
                if slicer.mrmlScene.IsNodePresent(self._bulk_seed_node):
                    slicer.mrmlScene.RemoveNode(self._bulk_seed_node)
            except Exception:
                pass
        self._bulk_seed_node = None
        self._bulk_seed_obs  = None
        self._update_bulk_seed_label()

    # ── Bulk: Step 3 — grow vessels (Scene-tab parameters, logic UNCHANGED) ──────

    def _clear_bulk_preview(self):
        if self._bulk_result_seg:
            try:
                if slicer.mrmlScene.IsNodePresent(self._bulk_result_seg):
                    slicer.mrmlScene.RemoveNode(self._bulk_result_seg)
            except Exception:
                pass
        self._bulk_result_seg = None

    def _on_bulk_grow(self):
        if self._bulk_pet_node is None:
            slicer.util.errorDisplay(
                "Load PET first (Step 1).", windowTitle="Vessel Segmenter")
            return

        seeds = []
        if self._bulk_seed_node:
            for i in range(self._bulk_seed_node.GetNumberOfControlPoints()):
                p = [0.0, 0.0, 0.0]
                self._bulk_seed_node.GetNthControlPointPositionWorld(i, p)
                seeds.append(list(p))
        if not seeds:
            slicer.util.errorDisplay(
                "Place at least one seed point (Step 2).",
                windowTitle="Vessel Segmenter")
            return

        self.bulkStatusLabel.setText("Growing vessels…")
        self.bulkGrowBtn.setEnabled(False)
        slicer.app.processEvents()
        try:
            self._clear_bulk_preview()
            seg_node = self._logic.run(
                pet_node        = self._bulk_pet_node,
                ct_node         = self._bulk_ct_node,
                seeds_ras       = seeds,
                suv_min         = self.suvMinSpin.value,
                suv_max         = self.suvMaxSpin.value,
                max_extent_mm   = self.maxExtentSpin.value,
                closing_mm      = self.closingRadiusSpin.value,
                min_vol_ml      = self.minVolSpin.value,
                stitch_gaps     = self.stitchCheck.isChecked(),
                stitch_gap_mm   = self.stitchGapSpin.value,
                stitch_radius_mm= self.stitchRadiusSpin.value,
                use_vesselness  = self.useVesselnessCheck.isChecked(),
                sigma_min_mm    = self.sigmaMinSpin.value,
                sigma_max_mm    = self.sigmaMaxSpin.value,
                vessel_thresh   = self.vesselThreshSpin.value,
                out_name        = "femoral_vessels",
                save_mask_vol   = False,
            )
            self._bulk_result_seg = seg_node
            n = seg_node.GetSegmentation().GetNumberOfSegments()
            self.bulkStatusLabel.setText(
                f"✓ Grown {n} segment(s) as 'femoral_vessels'. Review the preview, "
                "then Step 4: Confirm & Save -> Next.")
        except Exception as exc:
            import traceback
            self.bulkStatusLabel.setText(f"Error: {exc}")
            slicer.util.errorDisplay(
                traceback.format_exc(), windowTitle="Vessel Segmenter")
        finally:
            self.bulkGrowBtn.setEnabled(True)

    # ── Bulk: Step 4 — confirm, save, advance ────────────────────────────────────

    def _on_bulk_confirm(self):
        if self._bulk_result_seg is None:
            slicer.util.errorDisplay(
                "Grow vessels first (Step 3).", windowTitle="Vessel Segmenter")
            return

        subject_id = self._bulk_subjects[self._bulk_idx]
        seg_dir    = os.path.join(self._bulk_root, 'Segments', f"{subject_id}_Seg")
        os.makedirs(seg_dir, exist_ok=True)
        out_path   = os.path.join(seg_dir, "femoral_vessels.nii.gz")

        try:
            ok = self._logic.export_segmentation_to_nifti(
                self._bulk_result_seg, out_path)
            if not ok:
                raise RuntimeError("saveNode returned False")
            self._logic.log_bulk(self._bulk_root, subject_id, "OK")
            self.bulkStatusLabel.setText(
                f"✓ Saved femoral_vessels.nii.gz for {subject_id}.")
        except Exception as exc:
            import traceback
            print(traceback.format_exc())
            self._logic.log_bulk(self._bulk_root, subject_id, f"ERROR - {exc}")
            slicer.util.errorDisplay(f"Save failed: {exc}", windowTitle="Vessel Segmenter")
            return

        self._on_bulk_nav(1)

