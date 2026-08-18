"""
VesselSegmenterLogic — Slicer adapter.

Array algorithms: lib/segmentation/vessels.py
Bulk discovery:   lib/io/paths.find_bulk_subjects
Scene / DICOM / export helpers stay here.
"""
from __future__ import annotations

import os
import sys
import datetime

import numpy as np
import vtk
import slicer
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleLogic

_EXT_NEW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _EXT_NEW_ROOT not in sys.path:
    sys.path.insert(0, _EXT_NEW_ROOT)

from lib.io.paths import find_bulk_subjects as lib_find_bulk_subjects
from lib.segmentation.vessels import (
    apply_distance_constraint as lib_apply_distance_constraint,
    ct_vesselness_mask as lib_ct_vesselness_mask,
    grow_vessels_from_seeds as lib_grow_vessels_from_seeds,
    stitch_fragments as lib_stitch_fragments,
    _draw_bridge as lib_draw_bridge,
)

class VesselSegmenterLogic(ScriptedLoadableModuleLogic):

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self, pet_node, ct_node, seeds_ras,
            suv_min=0.8, suv_max=4.0,
            max_extent_mm=150.0,
            closing_mm=2.0, min_vol_ml=5.0,
            stitch_gaps=True, stitch_gap_mm=25.0, stitch_radius_mm=5.0,
            use_vesselness=False,
            sigma_min_mm=2.0, sigma_max_mm=8.0, vessel_thresh=0.10,
            out_name="Vessels", save_mask_vol=False):
        """
        Main segmentation pipeline.  Returns the output vtkMRMLSegmentationNode.
        Array core: lib.segmentation.vessels.grow_vessels_from_seeds.
        """
        vox_mm = self._spacing(pet_node)   # (dz, dy, dx)
        pet_arr = slicer.util.arrayFromVolume(pet_node).astype(np.float32)

        vesselness = None
        if use_vesselness and ct_node is not None:
            print("[VESSEL] CT vesselness filter…")
            slicer.app.processEvents()
            vesselness = self._ct_vesselness_mask(
                ct_node, sigma_min_mm, sigma_max_mm, vessel_thresh)
            if vesselness.shape != pet_arr.shape:
                print("[VESSEL]   Resampling vesselness mask to PET grid…")
                vesselness = self._resample_nn(vesselness, ct_node, pet_node)

        if save_mask_vol:
            pet_mask = ((pet_arr >= suv_min) & (pet_arr <= suv_max)).astype(np.uint8)
            combined = pet_mask.copy()
            if vesselness is not None:
                combined = (combined & vesselness).astype(np.uint8)
            self._save_as_volume(
                combined.astype(np.float32), pet_node,
                f"{out_name}_BloodPoolMask")

        seeds_zyx = []
        for ras in seeds_ras:
            z, y, x = self._ras_to_zyx(pet_node, ras)
            z = int(np.clip(z, 0, pet_arr.shape[0] - 1))
            y = int(np.clip(y, 0, pet_arr.shape[1] - 1))
            x = int(np.clip(x, 0, pet_arr.shape[2] - 1))
            seeds_zyx.append((z, y, x))

        print(f"[VESSEL] Growing from {len(seeds_zyx)} seed(s) via lib…")
        grown = lib_grow_vessels_from_seeds(
            pet_arr,
            spacing_mm=vox_mm,
            seeds_zyx=seeds_zyx,
            suv_min=float(suv_min),
            suv_max=float(suv_max),
            max_extent_mm=float(max_extent_mm),
            closing_radius_mm=float(closing_mm),
            min_volume_ml=float(min_vol_ml),
            stitch=bool(stitch_gaps),
            stitch_gap_mm=float(stitch_gap_mm),
            bridge_radius_mm=float(stitch_radius_mm),
            ct_vesselness=vesselness,
        )

        if not grown:
            raise RuntimeError(
                "No seeds landed in a usable blood-pool region.\n\n"
                f"  • Current SUV range: {suv_min:.2f}–{suv_max:.2f}\n"
                f"  • Min component volume: {min_vol_ml:.0f} mL\n"
                f"  • Max extent from seed: {max_extent_mm:.0f} mm\n\n"
                "Try lowering SUV min, reducing min volume, or repositioning seeds."
            )

        vox_vol_ml = (vox_mm[0] * vox_mm[1] * vox_mm[2]) / 1000.0
        seg_masks = []
        for name, mask in grown.items():
            vol_ml = float(mask.sum()) * vox_vol_ml
            print(f"[VESSEL]   {name}: {vol_ml:.1f} mL")
            seg_masks.append((name, mask, vol_ml))

        print(f"[VESSEL] Building segmentation node ({len(seg_masks)} segment(s))…")
        seg_node = self._build_segmentation(pet_node, seg_masks, out_name)
        print(f"[VESSEL] Done — '{seg_node.GetName()}' created.")
        return seg_node

    # ── Distance constraint ───────────────────────────────────────────────────

    def _apply_distance_constraint(self, comp_mask, seed_zyx, vox_mm,
                                   max_extent_mm, ndimage=None):
        return lib_apply_distance_constraint(
            comp_mask, seed_zyx, vox_mm, max_extent_mm
        )

    # ── Fragment stitching ────────────────────────────────────────────────────

    def _stitch_fragments(self, mask_arr, vox_mm, max_gap_mm, bridge_radius_mm,
                          ndimage=None):
        return lib_stitch_fragments(
            mask_arr, vox_mm, max_gap_mm, bridge_radius_mm
        )


    def _draw_bridge(self, mask, p1, p2, radius_vox):
        return lib_draw_bridge(mask, p1, p2, radius_vox)

    # ── CT Vesselness ─────────────────────────────────────────────────────────

    def _ct_vesselness_mask(self, ct_node, sigma_min_mm, sigma_max_mm, threshold):
        ct_arr = slicer.util.arrayFromVolume(ct_node).astype(np.float32)
        sp = self._spacing(ct_node)
        slicer.app.processEvents()
        return lib_ct_vesselness_mask(
            ct_arr, sp,
            sigma_min_mm=float(sigma_min_mm),
            sigma_max_mm=float(sigma_max_mm),
            threshold=float(threshold),
        )


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

    # ── Bulk-mode helpers (folder I/O, PET loading, export, logging) ──────────
    #
    # These are pure I/O / glue helpers added for bulk mode.  They do NOT
    # change run() or any of the segmentation algorithm above.

    def find_bulk_subjects(self, dataset_root):
        """Return sorted subject IDs that have a Segments/<ID>_Seg folder."""
        return [r["subject_id"] for r in lib_find_bulk_subjects(dataset_root)]

    def setup_coronal_pet_view(self, pet_node):
        """Switch to a one-up coronal Red slice view with the PET-Rainbow colour map."""
        lm = slicer.app.layoutManager()
        lm.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpRedSliceView)
        sw = lm.sliceWidget("Red")
        sl = sw.sliceLogic()
        sl.GetSliceNode().SetOrientationToCoronal()
        slicer.util.setSliceViewerLayers(background=pet_node)
        display = pet_node.GetDisplayNode()
        if display:
            display.SetAndObserveColorNodeID('vtkMRMLPETProceduralColorNodePET-Rainbow')
        sl.FitSliceToAll()

    def load_pet_dicom(self, pet_dir, subject_id):
        """
        Import + load a patient's PET DICOM series, convert Bq/mL -> SUVbw if
        needed, and return the resulting vtkMRMLScalarVolumeNode (or None).
        """
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
        print(f"[VESSEL][DICOM] Importing: {pet_dir}")
        ok = DICOMUtils.importDicom(pet_dir)
        print(f"[VESSEL][DICOM] importDicom={ok}")

        after_series = _all_series()
        new_series   = after_series - before_series
        print(f"[VESSEL][DICOM] New series: {len(new_series)}")

        if not new_series:
            pet_dir_norm = os.path.normpath(pet_dir)
            for series in after_series:
                files = db.filesForSeries(series)
                if files and os.path.normpath(os.path.dirname(files[0])) == pet_dir_norm:
                    new_series.add(series)
            print(f"[VESSEL][DICOM] Series matched by path: {len(new_series)}")

        pt_series = []
        for uid in new_series:
            files = db.filesForSeries(uid)
            desc  = db.fileValue(files[0], '0008,103e') if files else '?'
            mod   = db.fileValue(files[0], '0008,0060') if files else '?'
            print(f"[VESSEL][DICOM]   mod={mod!r}  desc={desc!r}")
            if mod.strip().upper() in ('PT', 'NM'):
                pt_series.append((uid, mod, desc))

        if not pt_series:
            pt_series = [(uid, '?', '?') for uid in new_series]

        for uid, mod, desc in pt_series:
            print(f"[VESSEL][DICOM] Loading: mod={mod!r}  desc={desc!r}")
            try:
                node_ids = DICOMUtils.loadSeriesByUID([uid])
                print(f"[VESSEL][DICOM]   -> {len(node_ids)} node(s)")
            except Exception as e:
                print(f"[VESSEL][DICOM]   -> failed: {e}")

        after_nodes = slicer.util.getNodesByClass('vtkMRMLScalarVolumeNode')
        new_nodes   = [n for n in after_nodes if n.GetID() not in before_node_ids]
        print(f"[VESSEL][DICOM] New volume nodes: {[n.GetName() for n in new_nodes]}")

        if not new_nodes:
            return None

        pet_node = next(
            (n for n in new_nodes if 'suv' in n.GetName().lower()), new_nodes[0])

        try:
            self._apply_suv_conversion(pet_node, pet_dir)
        except Exception as e:
            print(f"[VESSEL][DICOM]   SUV conversion skipped/failed: {e}")

        if 'SUVbw' not in pet_node.GetName():
            old = pet_node.GetName()
            pet_node.SetName(f'SUVbw_{subject_id}')
            print(f"[VESSEL][DICOM] Renamed '{old}' -> '{pet_node.GetName()}'")
        return pet_node

    def _apply_suv_conversion(self, pet_node, dicom_folder):
        """
        Convert the PET volume from Bq/mL -> SUVbw in-place.
        SUVbw = (Bq/mL x weight_g) / decay_corrected_dose_Bq
        If DICOM Units tag is already 'SUV', conversion is skipped.
        """
        import pydicom

        first_dcm = None
        for root, _, files in os.walk(dicom_folder):
            for f in sorted(files):
                fp = os.path.join(root, f)
                try:
                    pydicom.dcmread(fp, stop_before_pixels=True)
                    first_dcm = fp
                    break
                except Exception:
                    continue
            if first_dcm:
                break

        if not first_dcm:
            raise ValueError(f"No readable DICOM file found in {dicom_folder}")

        ds    = pydicom.dcmread(first_dcm, stop_before_pixels=True)
        units = str(getattr(ds, "Units", "BQML")).strip().upper()
        print(f"[VESSEL][SUV] Units tag = '{units}'")

        if units == "SUV":
            print("[VESSEL][SUV] already SUV - skipping conversion")
            return

        if units not in ("BQML", "BQ/ML", ""):
            print(f"[VESSEL][SUV] unexpected units '{units}' - "
                  f"attempting conversion anyway (assuming Bq/mL)")

        # Patient weight
        try:
            weight_kg = float(str(getattr(ds, "PatientWeight", None)))
            if __import__("math").isnan(weight_kg):
                raise ValueError("PatientWeight is NaN")
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"PatientWeight missing or unreadable in DICOM header: {e}")
        weight_g = weight_kg * 1000
        print(f"[VESSEL][SUV] PatientWeight = {weight_kg} kg")

        # Injected dose
        try:
            radio_seq = ds[0x0054, 0x0016][0]
        except (KeyError, IndexError):
            raise ValueError(
                "RadiopharmaceuticalInformationSequence (0054,0016) missing - "
                "cannot compute SUV")

        try:
            dose_bq = float(str(radio_seq[0x0018, 0x1074].value))
        except (KeyError, ValueError):
            raise ValueError(
                "RadionuclideTotalDose (0018,1074) missing from "
                "RadiopharmaceuticalInformationSequence")

        # Injection time
        inj_dt = getattr(radio_seq, "RadiopharmaceuticalStartDateTime", None)
        inj_t  = getattr(radio_seq, "RadiopharmaceuticalStartTime", None)
        if inj_dt:
            inj_str = str(inj_dt).split(".")[0]
        elif inj_t:
            inj_str = str(inj_t).split(".")[0]
        else:
            raise ValueError(
                "No injection time found in RadiopharmaceuticalInformationSequence "
                "(tried RadiopharmaceuticalStartDateTime and RadiopharmaceuticalStartTime)")

        # Radionuclide half-life (standard tag, then GE private fallback, then F-18 default)
        try:
            half_life_s = float(str(radio_seq[0x0018, 0x1075].value))
            print(f"[VESSEL][SUV] half-life from standard tag = {half_life_s}s")
        except (KeyError, ValueError):
            try:
                half_life_s = float(str(ds[0x0009, 0x103F].value))
                print(f"[VESSEL][SUV] half-life from GE private tag = {half_life_s}s")
            except (KeyError, ValueError):
                half_life_s = 6586.2  # F-18 default
                print(f"[VESSEL][SUV] half-life not in DICOM - "
                      f"using F-18 default = {half_life_s}s")

        # Acquisition time and decay correction type
        acq_t_raw  = str(getattr(ds, "AcquisitionTime", "") or
                         getattr(ds, "SeriesTime", "")).split(".")[0].strip()
        decay_corr = str(getattr(ds, "DecayCorrection", "START")).strip().upper()

        print(f"[VESSEL][SUV] dose={dose_bq:.0f} Bq  inj={inj_str}  acq={acq_t_raw}  "
              f"half_life={half_life_s}s  decay_correction={decay_corr}")

        def _hhmmss_to_s(t):
            t = t.strip()[-6:]
            try:
                return int(t[0:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:6])
            except (ValueError, IndexError):
                return 0

        if decay_corr == "ADMIN":
            decay_corrected_dose_bq = dose_bq
            print("[VESSEL][SUV] ADMIN correction - using injected dose directly")
        else:
            inj_s = _hhmmss_to_s(inj_str)
            acq_s = _hhmmss_to_s(acq_t_raw)
            if acq_s < inj_s:
                acq_s += 86400          # midnight crossing
            decay_time_s = acq_s - inj_s
            decay_corrected_dose_bq = dose_bq * (2.0 ** (-decay_time_s / half_life_s))
            print(f"[VESSEL][SUV] dt={decay_time_s}s  "
                  f"decay_corrected_dose={decay_corrected_dose_bq:.0f} Bq "
                  f"(={decay_corrected_dose_bq/1e6:.3f} MBq)")

        if decay_corrected_dose_bq <= 0:
            raise ValueError(
                f"decay_corrected_dose={decay_corrected_dose_bq} - "
                "check injection time and dose values in DICOM header")

        suv_factor = weight_g / decay_corrected_dose_bq
        print(f"[VESSEL][SUV] SUV factor = {suv_factor:.6f}  "
              f"(weight={weight_g}g / decay_dose={decay_corrected_dose_bq:.0f} Bq)")

        arr        = slicer.util.arrayFromVolume(pet_node)
        max_before = float(arr.max())
        arr_suv    = (arr * suv_factor).astype(np.float32)
        slicer.util.updateVolumeFromArray(pet_node, arr_suv)
        max_after  = float(arr_suv.max())
        print(f"[VESSEL][SUV] DONE - max: {max_before:.1f} Bq/mL -> {max_after:.4f} SUV")

    def export_segmentation_to_nifti(self, seg_node, out_path):
        """Export every segment in seg_node to a single labelmap NIfTI file."""
        print(f"[VESSEL][SAVE] '{seg_node.GetName()}' -> {os.path.basename(out_path)}")
        lm = slicer.mrmlScene.AddNewNodeByClass(
            'vtkMRMLLabelMapVolumeNode', '__vessel_export_lm__')
        slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(seg_node, lm)
        ok = slicer.util.saveNode(lm, out_path)
        slicer.mrmlScene.RemoveNode(lm)
        if ok:
            print(f"[VESSEL][SAVE]   OK ({os.path.getsize(out_path)//1024} KB)")
        else:
            print(f"[VESSEL][SAVE]   FAILED")
        return ok

    def log_bulk(self, dataset_root, subject_id, status):
        """Append one line to dataset_root/pipeline_logs/bulk_log.txt."""
        import datetime
        log_dir  = os.path.join(dataset_root, 'pipeline_logs')
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, 'bulk_log.txt')
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f"[{ts}] [VesselSegmenter] {subject_id}: {status}\n")
        except Exception as e:
            print(f"  WARNING: could not write bulk_log.txt: {e}")
