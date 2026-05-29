"""
VesselSegmenter — 3D Slicer scripted module
==========================================

Segments large vessels (femoral veins/arteries, iliac vessels, etc.) that show
up as PET blood-pool uptake but are not included in TotalSegmentator.

Workflow
--------
1. Select the PET volume (SUV units).
2. Set the SUV window that captures blood-pool signal (default 0.8 – 2.5).
3. Click "Place Seeds" and click once inside each vessel you want to extract.
4. Click "Run" — each seed extracts the connected blood-pool region it belongs to.
5. Optional: enable CT vesselness to restrict the mask to tubular structures.
"""

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

class VesselSegmenter(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title        = "Vessel Segmenter"
        self.parent.categories   = ["Segmentation"]
        self.parent.dependencies = []
        self.parent.contributors = ["IshitaSinghFaujdar"]
        self.parent.helpText = (
            "Segment large vessels visible as PET blood-pool uptake.\n"
            "Place seed points inside vessels of interest and run segmentation.\n"
            "Suitable for femoral, iliac, and other large vessels."
        )
        self.parent.acknowledgementText = ""


# ── Widget ─────────────────────────────────────────────────────────────────────

class VesselSegmenterWidget(ScriptedLoadableModuleWidget):

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self._logic     = VesselSegmenterLogic()
        self._seed_node = None
        self._seed_obs  = None

        # ── Input volumes ─────────────────────────────────────────────────────
        inputsBox = ctk.ctkCollapsibleButton()
        inputsBox.text = "Input Volumes"
        self.layout.addWidget(inputsBox)
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
        self.layout.addWidget(petBox)
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
        self.suvMaxSpin.setValue(2.5)
        self.suvMaxSpin.setToolTip(
            "Maximum SUV for blood-pool mask.\n"
            "Keeps hot tumour voxels out of the vessel mask.")
        petForm.addRow("SUV max:", self.suvMaxSpin)

        # ── Seeds ─────────────────────────────────────────────────────────────
        seedBox = ctk.ctkCollapsibleButton()
        seedBox.text = "Seed Points"
        self.layout.addWidget(seedBox)
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
        self.layout.addWidget(regionBox)
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
        self.layout.addWidget(morphBox)
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
        self.layout.addWidget(stitchBox)
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
        self.layout.addWidget(ctBox)
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
        self.layout.addWidget(outBox)
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
        self.layout.addWidget(self.runBtn)

        self._statusLabel = qt.QLabel("")
        self._statusLabel.setWordWrap(True)
        self._statusLabel.setStyleSheet("padding:4px;")
        self.layout.addWidget(self._statusLabel)

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


# ── Logic ──────────────────────────────────────────────────────────────────────

class VesselSegmenterLogic(ScriptedLoadableModuleLogic):

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self, pet_node, ct_node, seeds_ras,
            suv_min=0.8, suv_max=2.5,
            max_extent_mm=150.0,
            closing_mm=2.0, min_vol_ml=5.0,
            stitch_gaps=True, stitch_gap_mm=25.0, stitch_radius_mm=5.0,
            use_vesselness=False,
            sigma_min_mm=2.0, sigma_max_mm=8.0, vessel_thresh=0.10,
            out_name="Vessels", save_mask_vol=False):
        """
        Main segmentation pipeline.  Returns the output vtkMRMLSegmentationNode.
        """
        from scipy import ndimage

        vox_mm = self._spacing(pet_node)   # (dz, dy, dx)

        # ── Step 1: PET blood-pool mask ───────────────────────────────────────
        print(f"[VESSEL] Step 1: PET blood-pool mask  SUV {suv_min:.2f}–{suv_max:.2f}")
        pet_arr  = slicer.util.arrayFromVolume(pet_node).astype(np.float32)
        pet_mask = ((pet_arr >= suv_min) & (pet_arr <= suv_max)).astype(np.uint8)
        print(f"[VESSEL]   → {int(pet_mask.sum()):,} voxels")

        # ── Step 2: Optional CT vesselness ────────────────────────────────────
        combined = pet_mask.copy()
        if use_vesselness and ct_node is not None:
            print("[VESSEL] Step 2: CT vesselness filter (may take ~30 s)…")
            slicer.app.processEvents()
            vessel_mask = self._ct_vesselness_mask(
                ct_node, sigma_min_mm, sigma_max_mm, vessel_thresh)
            if vessel_mask.shape != pet_mask.shape:
                print("[VESSEL]   Resampling vesselness mask to PET grid…")
                vessel_mask = self._resample_nn(vessel_mask, ct_node, pet_node)
            combined = (combined & vessel_mask).astype(np.uint8)
            print(f"[VESSEL]   → {int(combined.sum()):,} voxels after vesselness AND")
        else:
            print("[VESSEL] Step 2: CT vesselness disabled")

        # ── Step 3: Optionally save the binary mask as a volume ───────────────
        if save_mask_vol:
            self._save_as_volume(
                combined.astype(np.float32), pet_node,
                f"{out_name}_BloodPoolMask")

        # ── Step 4: Morphological closing ─────────────────────────────────────
        if closing_mm > 0:
            r = max(1, int(round(closing_mm / min(vox_mm))))
            struct = ndimage.generate_binary_structure(3, 1)
            struct = ndimage.iterate_structure(struct, r)
            print(f"[VESSEL] Step 4: Morphological closing  r={r} vox ({closing_mm:.1f} mm)")
            combined = ndimage.binary_closing(combined, structure=struct).astype(np.uint8)

        # ── Step 5: Connected-component labelling ─────────────────────────────
        print("[VESSEL] Step 5: Connected-component labelling…")
        labeled, n_comp = ndimage.label(combined)
        vox_vol_ml = (vox_mm[0] * vox_mm[1] * vox_mm[2]) / 1000.0
        print(f"[VESSEL]   → {n_comp:,} components found")

        # ── Step 6: Extract component for each seed ───────────────────────────
        print("[VESSEL] Step 6: Mapping seeds to components…")
        seg_masks = []   # list of (name, mask_array, vol_ml)

        for seed_idx, ras in enumerate(seeds_ras):
            seed_label = f"Vessel_{seed_idx + 1}"

            z, y, x = self._ras_to_zyx(pet_node, ras)
            z = int(np.clip(z, 0, labeled.shape[0] - 1))
            y = int(np.clip(y, 0, labeled.shape[1] - 1))
            x = int(np.clip(x, 0, labeled.shape[2] - 1))

            comp_lbl = int(labeled[z, y, x])
            if comp_lbl == 0:
                print(f"[VESSEL]   Seed {seed_idx+1} RAS {ras} → background "
                      "(not inside blood-pool mask — try lowering SUV min or "
                      "repositioning the seed)")
                continue

            # Full component mask
            comp_mask = (labeled == comp_lbl).astype(np.uint8)

            # Distance constraint: keep only voxels within max_extent_mm of seed
            if max_extent_mm > 0:
                comp_mask = self._apply_distance_constraint(
                    comp_mask, (z, y, x), vox_mm, max_extent_mm, ndimage)

            vol_ml = float(comp_mask.sum()) * vox_vol_ml
            print(f"[VESSEL]   Seed {seed_idx+1} → component {comp_lbl}, "
                  f"{vol_ml:.1f} mL (after extent clip)")

            if vol_ml < min_vol_ml:
                print(f"[VESSEL]   Component too small ({vol_ml:.1f} < {min_vol_ml} mL), skipped")
                continue

            # ── Stitch disconnected fragments inside this seed region ──────────
            if stitch_gaps:
                comp_mask = self._stitch_fragments(
                    comp_mask, vox_mm, stitch_gap_mm, stitch_radius_mm, ndimage)
                vol_ml = float(comp_mask.sum()) * vox_vol_ml
                print(f"[VESSEL]   After stitching: {vol_ml:.1f} mL")

            # Check for duplicate (another seed captured the same region)
            duplicate = False
            for _, prev_mask, _ in seg_masks:
                overlap = float((comp_mask & prev_mask).sum()) / max(comp_mask.sum(), 1)
                if overlap > 0.9:
                    duplicate = True
                    break
            if duplicate:
                print(f"[VESSEL]   Seed {seed_idx+1} overlaps an existing segment, skipped")
                continue

            seg_masks.append((seed_label, comp_mask, vol_ml))

        if not seg_masks:
            raise RuntimeError(
                "No seeds landed in a usable blood-pool region.\n\n"
                f"  • Current SUV range: {suv_min:.2f}–{suv_max:.2f}\n"
                f"  • Min component volume: {min_vol_ml:.0f} mL\n"
                f"  • Max extent from seed: {max_extent_mm:.0f} mm\n\n"
                "Try:\n"
                "  – Lowering SUV min or raising SUV max\n"
                "  – Reducing the minimum component volume\n"
                "  – Repositioning seeds to the centre of the vessel lumen\n"
                "  – Checking that the PET volume is selected correctly")

        # ── Step 7: Build segmentation node ───────────────────────────────────
        print(f"[VESSEL] Step 7: Building segmentation node "
              f"({len(seg_masks)} segment(s))…")
        seg_node = self._build_segmentation(pet_node, seg_masks, out_name)
        print(f"[VESSEL] Done — '{seg_node.GetName()}' created.")
        return seg_node

    # ── Distance constraint ───────────────────────────────────────────────────

    def _apply_distance_constraint(self, comp_mask, seed_zyx, vox_mm,
                                   max_extent_mm, ndimage):
        """
        Keep only the sub-component of comp_mask that is connected to seed_zyx
        AND within max_extent_mm Euclidean distance of it.
        """
        sz, sy, sx = seed_zyx
        nz, ny, nx = comp_mask.shape

        # Build distance map from seed in mm
        dz = (np.arange(nz) - sz) * vox_mm[0]
        dy = (np.arange(ny) - sy) * vox_mm[1]
        dx = (np.arange(nx) - sx) * vox_mm[2]
        DZ, DY, DX = np.meshgrid(dz, dy, dx, indexing='ij')
        dist = np.sqrt(DZ**2 + DY**2 + DX**2)

        # Sphere-clipped component
        sphere = (dist <= max_extent_mm).astype(np.uint8)
        clipped = (comp_mask & sphere).astype(np.uint8)

        # Re-label clipped mask and keep only the piece the seed is in
        labeled_local, _ = ndimage.label(clipped)
        local_lbl = int(labeled_local[sz, sy, sx])
        if local_lbl == 0:
            return clipped   # seed voxel itself was clipped; return sphere-clipped mask
        return (labeled_local == local_lbl).astype(np.uint8)

    # ── Fragment stitching ────────────────────────────────────────────────────

    def _stitch_fragments(self, mask_arr, vox_mm, max_gap_mm, bridge_radius_mm,
                          ndimage):
        """
        Iteratively find the closest pair of disconnected sub-fragments inside
        mask_arr and bridge them with a tube until no pair is within max_gap_mm.

        Uses centroid-to-centroid distance as the proximity metric — fast and
        sufficient for roughly symmetric vessel fragments.
        """
        result = mask_arr.copy()
        bridge_r_vox = max(1, int(round(bridge_radius_mm / min(vox_mm))))

        for iteration in range(200):   # safety cap
            labeled, n = ndimage.label(result)
            if n <= 1:
                break

            centroids = ndimage.center_of_mass(result, labeled, range(1, n + 1))
            centroids = [np.asarray(c, dtype=float) for c in centroids]

            # Find closest pair within max_gap_mm
            best_dist = float('inf')
            best_i, best_j = 0, 1
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.sqrt(
                        ((centroids[i][0] - centroids[j][0]) * vox_mm[0]) ** 2 +
                        ((centroids[i][1] - centroids[j][1]) * vox_mm[1]) ** 2 +
                        ((centroids[i][2] - centroids[j][2]) * vox_mm[2]) ** 2)
                    if d < best_dist:
                        best_dist = d
                        best_i, best_j = i, j

            if best_dist > max_gap_mm:
                break   # no more bridgeable pairs

            result = self._draw_bridge(
                result, centroids[best_i], centroids[best_j], bridge_r_vox)
            print(f"[VESSEL]   Stitch iter {iteration+1}: "
                  f"bridged gap {best_dist:.1f} mm  (r={bridge_r_vox} vox)")

        return result

    def _draw_bridge(self, mask, p1, p2, radius_vox):
        """
        Stamp a sphere of radius_vox at every voxel along the straight line
        from p1 to p2, returning the updated mask.
        """
        result  = mask.copy()
        nz, ny, nx = result.shape
        vec     = p2 - p1
        dist    = float(np.linalg.norm(vec))
        if dist < 1e-3:
            return result

        n_steps = max(3, int(np.ceil(dist)) * 2)
        r2      = radius_vox ** 2

        for t in np.linspace(0.0, 1.0, n_steps):
            pt = p1 + t * vec
            cz = int(round(float(pt[0])))
            cy = int(round(float(pt[1])))
            cx = int(round(float(pt[2])))

            z0 = max(0, cz - radius_vox);  z1 = min(nz, cz + radius_vox + 1)
            y0 = max(0, cy - radius_vox);  y1 = min(ny, cy + radius_vox + 1)
            x0 = max(0, cx - radius_vox);  x1 = min(nx, cx + radius_vox + 1)

            zz = np.arange(z0, z1) - cz
            yy = np.arange(y0, y1) - cy
            xx = np.arange(x0, x1) - cx
            ZZ, YY, XX = np.meshgrid(zz, yy, xx, indexing='ij')
            sphere = (ZZ ** 2 + YY ** 2 + XX ** 2) <= r2
            result[z0:z1, y0:y1, x0:x1] |= sphere

        return result

    # ── CT Vesselness ─────────────────────────────────────────────────────────

    def _ct_vesselness_mask(self, ct_node, sigma_min_mm, sigma_max_mm, threshold):
        """
        Multi-scale Frangi-style vesselness via SimpleITK ObjectnessMeasureImageFilter.
        Returns a binary uint8 array in CT voxel space (Z,Y,X).
        BrightObject=True: works for vessels in fat (femoral / pelvic region).
        """
        import SimpleITK as sitk

        ct_arr  = slicer.util.arrayFromVolume(ct_node).astype(np.float32)
        sp      = self._spacing(ct_node)   # (dz, dy, dx)

        ct_img = sitk.GetImageFromArray(ct_arr)
        ct_img.SetSpacing([float(sp[2]), float(sp[1]), float(sp[0])])

        n_scales   = 4
        sigmas     = np.linspace(sigma_min_mm, sigma_max_mm, n_scales)
        resp_max   = np.zeros_like(ct_arr)

        objFilter = sitk.ObjectnessMeasureImageFilter()
        objFilter.SetObjectDimension(1)          # 1 = tubular
        objFilter.SetScaleObjectnessMeasure(True)
        objFilter.SetBrightObject(True)          # vessels bright vs. fat
        objFilter.SetAlpha(0.5)
        objFilter.SetBeta(0.5)
        objFilter.SetGamma(5.0)

        for sigma in sigmas:
            smooth = sitk.SmoothingRecursiveGaussian(ct_img, sigma=float(sigma))
            resp   = sitk.GetArrayFromImage(objFilter.Execute(smooth))
            resp_max = np.maximum(resp_max, resp)
            slicer.app.processEvents()

        mx = resp_max.max()
        if mx > 0:
            resp_max /= mx

        print(f"[VESSEL]   Vesselness max={mx:.4f}, "
              f"voxels ≥ threshold: {int((resp_max >= threshold).sum()):,}")
        return (resp_max >= threshold).astype(np.uint8)

    # ── Build segmentation node ───────────────────────────────────────────────

    def _build_segmentation(self, ref_node, seg_masks, out_name):
        """
        seg_masks : list of (name, uint8_array_ZYX, vol_ml)
        Returns vtkMRMLSegmentationNode.
        """
        palette = [
            (0.20, 0.60, 1.00),   # blue
            (1.00, 0.35, 0.35),   # red
            (0.20, 0.85, 0.40),   # green
            (1.00, 0.75, 0.15),   # amber
            (0.80, 0.35, 1.00),   # purple
            (1.00, 0.55, 0.15),   # orange
        ]

        existing = slicer.mrmlScene.GetFirstNodeByName(out_name)
        if existing:
            slicer.mrmlScene.RemoveNode(existing)

        seg_node = slicer.mrmlScene.AddNewNodeByClass(
            'vtkMRMLSegmentationNode', out_name)
        seg_node.CreateDefaultDisplayNodes()

        # Temporary labelmap node with PET geometry
        tmp_lm = slicer.mrmlScene.AddNewNodeByClass(
            'vtkMRMLLabelMapVolumeNode', '__vessel_tmp_lm__')
        self._init_lm_geometry(ref_node, tmp_lm)

        for idx, (name, mask_arr, vol_ml) in enumerate(seg_masks):
            slicer.util.updateVolumeFromArray(tmp_lm, mask_arr.astype(np.int16))
            slicer.modules.segmentations.logic() \
                .ImportLabelmapToSegmentationNode(tmp_lm, seg_node)

            seg_obj  = seg_node.GetSegmentation()
            last_seg = seg_obj.GetNthSegment(seg_obj.GetNumberOfSegments() - 1)
            if last_seg:
                last_seg.SetName(f"{name}  ({vol_ml:.0f} mL)")
                r, g, b = palette[idx % len(palette)]
                last_seg.SetColor(r, g, b)

        slicer.mrmlScene.RemoveNode(tmp_lm)
        seg_node.CreateClosedSurfaceRepresentation()
        return seg_node

    # ── Volume helpers ────────────────────────────────────────────────────────

    def _spacing(self, vol_node):
        """Returns (dz, dy, dx) in mm — matches arrayFromVolume axis order."""
        sx, sy, sz = vol_node.GetSpacing()
        return (sz, sy, sx)

    def _ras_to_zyx(self, vol_node, ras):
        """Convert RAS mm → (z, y, x) array indices (for arrayFromVolume order)."""
        mat = vtk.vtkMatrix4x4()
        vol_node.GetRASToIJKMatrix(mat)
        h = mat.MultiplyPoint([ras[0], ras[1], ras[2], 1.0])
        i, j, k = h[0], h[1], h[2]   # IJK = (col=X, row=Y, slice=Z)
        return k, j, i                # array order (Z, Y, X)

    def _init_lm_geometry(self, ref_node, lm_node):
        """Initialise lm_node with the same geometry (IJK→RAS, dims) as ref_node."""
        ijk_to_ras = vtk.vtkMatrix4x4()
        ref_node.GetIJKToRASMatrix(ijk_to_ras)
        lm_node.SetIJKToRASMatrix(ijk_to_ras)

        dims = ref_node.GetImageData().GetDimensions()
        img  = vtk.vtkImageData()
        img.SetDimensions(dims)
        img.AllocateScalars(vtk.VTK_SHORT, 1)
        lm_node.SetAndObserveImageData(img)

    def _resample_nn(self, arr, from_node, to_node):
        """Nearest-neighbour resample arr (from_node space) → to_node space."""
        from scipy.ndimage import map_coordinates

        to_arr      = slicer.util.arrayFromVolume(to_node)
        nz, ny, nx  = to_arr.shape

        def mat_np(m):
            return np.array([[m.GetElement(r, c) for c in range(4)]
                             for r in range(4)], dtype=np.float64)

        M_to   = mat_np(self._get_ijk_to_ras(to_node))
        M_from = mat_np(self._get_ras_to_ijk(from_node))

        zz, yy, xx = np.meshgrid(
            np.arange(nz), np.arange(ny), np.arange(nx), indexing='ij')
        ijk_to  = np.vstack([xx.ravel(), yy.ravel(), zz.ravel(), np.ones(xx.size)])
        ras     = M_to   @ ijk_to
        ijk_fr  = M_from @ ras

        coords = [ijk_fr[2].reshape(nz, ny, nx),
                  ijk_fr[1].reshape(nz, ny, nx),
                  ijk_fr[0].reshape(nz, ny, nx)]
        return map_coordinates(arr.astype(float), coords,
                               order=0, mode='constant', cval=0).astype(np.uint8)

    def _get_ijk_to_ras(self, node):
        m = vtk.vtkMatrix4x4(); node.GetIJKToRASMatrix(m); return m

    def _get_ras_to_ijk(self, node):
        m = vtk.vtkMatrix4x4(); node.GetRASToIJKMatrix(m); return m

    def _save_as_volume(self, arr, ref_node, name):
        """Save float32 array as a scalar volume node with ref_node geometry."""
        existing = slicer.mrmlScene.GetFirstNodeByName(name)
        if existing:
            slicer.mrmlScene.RemoveNode(existing)
        vol = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode', name)
        ijk_to_ras = vtk.vtkMatrix4x4()
        ref_node.GetIJKToRASMatrix(ijk_to_ras)
        vol.SetIJKToRASMatrix(ijk_to_ras)
        dims = ref_node.GetImageData().GetDimensions()
        img  = vtk.vtkImageData()
        img.SetDimensions(dims)
        img.AllocateScalars(vtk.VTK_FLOAT, 1)
        vol.SetAndObserveImageData(img)
        slicer.util.updateVolumeFromArray(vol, arr.astype(np.float32))
        print(f"[VESSEL] Blood-pool mask saved as volume '{name}'")
        return vol
