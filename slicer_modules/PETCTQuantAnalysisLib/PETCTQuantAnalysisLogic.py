"""
PETCTQuantAnalysisLogic — Slicer adapter.

lib: paths discovery, excel export, SUVbw factor math
Logic (Slicer): DICOM load, registration, QuantitativeIndicesCLI, volume update
"""
from __future__ import annotations

import os
import sys
import logging
import math
import vtk
import slicer
import numpy as np
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleLogic

LOG = logging.getLogger("PETCTQuant")

_EXT_NEW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _EXT_NEW_ROOT not in sys.path:
    sys.path.insert(0, _EXT_NEW_ROOT)

from lib.io.paths import detect_scans as lib_detect_scans
from lib.io.paths import detect_segmentations as lib_detect_segmentations
from lib.io.paths import parse_folder_name as lib_parse_folder_name
from lib.quantification.pet_metrics import (
    save_excel as lib_save_excel,
    error_row as lib_error_row,
    compute_suvbw_factor,
)


class PETCTQuantAnalysisLogic(ScriptedLoadableModuleLogic):

    # ── Logging helper ────────────────────────────────────────────────────

    def _log(self, level, msg, status_cb=None):
        """
        Unified logger: writes to the module logger (visible in Slicer's
        application log via Help → Report a Bug) AND prints to the Python
        console so you see it live without opening the log file.

        level: 'info' | 'warning' | 'error'
        """
        tagged = f"[PETCTQuant-v2] {msg}"
        if level == "error":
            LOG.error(tagged)
        elif level == "warning":
            LOG.warning(tagged)
        else:
            LOG.info(tagged)
        print(tagged)
        if status_cb:
            status_cb(msg)

    def _parseFolderName(self, name):
        return lib_parse_folder_name(name)


    def detectScans(self, root_folder):
        return lib_detect_scans(root_folder, log=self._log)


    def detectSegmentations(self, root_folder, scans):
        return lib_detect_segmentations(root_folder, scans, log=self._log)


    def runBatch(self, root_folder, scans, seg_name_map,
                 output_file, metrics, suv_type, append,
                 progress_cb, status_cb, cancel_check):
        """
        seg_name_map: {stem: display_name}  — only these files are processed.
        """
        total    = len(scans)
        all_rows = []

        self._log("info",
            f"runBatch: starting  scans={total}  "
            f"segs={list(seg_name_map.keys())}  "
            f"suv_type={suv_type}  append={append}",
            status_cb)

        if total == 0:
            self._log("error", "runBatch: no valid scans found — aborting", status_cb)
            raise ValueError("No valid scans found in root folder.")

        for i, scan in enumerate(scans):
            if cancel_check():
                self._log("warning", "runBatch: cancelled by user", status_cb)
                break

            subj     = scan["subject_id"]
            date     = scan["scan_date"]
            pet_path = scan["pet_path"]
            seg_path = scan["seg_path"]

            self._log("info",
                f"── Patient {i+1}/{total}: {subj}  {date} ──────",
                status_cb)
            progress_cb(i / total * 100)

            # ── Step 1: Load DICOM PET ─────────────────────────────────
            self._log("info",
                f"  Step 1/4 — loading PET DICOM from: {pet_path}", status_cb)
            try:
                pet_node = self._loadDicomSeries(pet_path, subj)
                self._log("info",
                    f"  Step 1/4 — PET DICOM loaded OK  "
                    f"node={pet_node.GetName()}  id={pet_node.GetID()}")
            except Exception as e:
                self._log("error",
                    f"  Step 1/4 FAILED — could not load PET DICOM: {e}", status_cb)
                for stem in seg_name_map:
                    all_rows.append(
                        self._errorRow(subj, date, seg_name_map[stem], "missing_pet"))
                continue

            patient_id = self._getPatientIdFromNode(pet_node)
            self._log("info", f"  patient_id extracted: {patient_id}")

            # ── Step 1b: Convert Bq/mL → SUVbw if needed ──────────────
            self._log("info", "  Step 1b — applying SUV conversion...", status_cb)
            try:
                self._applySuvConversion(pet_node, pet_path)
            except Exception as e:
                self._log("error",
                    f"  Step 1b FAILED — SUV conversion error: {e}  "
                    f"Results will be in raw DICOM units (Bq/mL), not SUV",
                    status_cb)

            # ── Step 1c: CT→PET registration (always on) ───────────────
            # The segmentations are defined in CT space; align CT→PET and bake
            # the transform onto each segment before measuring so the masks sit
            # correctly on the PET grid. Rigid + affine (rigid initialises
            # affine). If the CT folder is missing we log and continue without
            # registration rather than failing the patient.
            ct_node             = None
            ct_to_pet_transform = None
            ct_path = scan.get("ct_path")
            if ct_path:
                self._log("info",
                    f"  Step 1c — loading CT DICOM for registration: {ct_path}",
                    status_cb)
                try:
                    ct_node = self._loadDicomSeries(ct_path, subj)
                    ct_node.SetName(f"CT_{subj}")
                    self._log("info",
                        f"  Step 1c — CT loaded OK  node={ct_node.GetName()}")
                    ct_to_pet_transform = self._registerCtToPet(ct_node, pet_node)
                except Exception as e:
                    self._log("error",
                        f"  Step 1c FAILED — registration skipped: {e}", status_cb)
                    if ct_node:
                        slicer.mrmlScene.RemoveNode(ct_node)
                    ct_node             = None
                    ct_to_pet_transform = None
            else:
                self._log("warning",
                    "  Step 1c — no CT folder found for this patient "
                    "— computing without registration", status_cb)

            # ── Step 2: (TotalSegmentator skipped in v2 — segs assumed present)
            self._log("info", "  Step 2/4 — skipped (using existing segments)")

            # ── Step 3–4: For each selected segmentation ───────────────
            self._log("info",
                f"  Step 3–4/4 — computing metrics for: "
                f"{list(seg_name_map.values())}")

            for stem, display_name in seg_name_map.items():
                seg_file = os.path.join(seg_path or "", f"{stem}.nii.gz") \
                           if seg_path else None

                self._log("info",
                    f"    [{display_name}] looking for: {seg_file}")

                if not seg_file or not os.path.isfile(seg_file):
                    self._log("warning",
                        f"    [{display_name}] MISSING segment file — recording error row",
                        status_cb)
                    all_rows.append(
                        self._errorRow(subj, date, display_name,
                                       "missing_seg", patient_id))
                    continue

                self._log("info",
                    f"    [{display_name}] segment file found — loading...",
                    status_cb)

                try:
                    seg_node = self._loadSegmentation(seg_file)
                    self._log("info",
                        f"    [{display_name}] segmentation loaded OK  "
                        f"node={seg_node.GetName()}")

                    if ct_to_pet_transform is not None:
                        self._log("info",
                            f"    [{display_name}] applying CT→PET transform...",
                            status_cb)
                        self._applyTransformToSeg(seg_node, ct_to_pet_transform)

                    self._log("info",
                        f"    [{display_name}] running pet-indic CLI...", status_cb)
                    results = self._runPetIndic(pet_node, seg_node, metrics, suv_type)
                    slicer.mrmlScene.RemoveNode(seg_node)
                    self._log("info",
                        f"    [{display_name}] seg node removed from scene")

                    row = {
                        "subject_id":  subj,
                        "patient_id":  patient_id,
                        "scan_date":   date,
                        "segment":     display_name,
                        "source_file": f"{stem}.nii.gz",
                        "status":      "done",
                    }
                    row.update(results)
                    all_rows.append(row)

                    self._log("info",
                        f"    [{display_name}] DONE  "
                        f"SUVmean={results.get('suv_mean','?')}  "
                        f"SUVmax={results.get('suv_max','?')}  "
                        f"SUVpeak={results.get('suv_peak','?')}  "
                        f"TLG={results.get('tlg','?')}  "
                        f"vol={results.get('volume_mL','?')} mL")

                except Exception as e:
                    self._log("error",
                        f"    [{display_name}] FAILED — {e}", status_cb)
                    all_rows.append(
                        self._errorRow(subj, date, display_name,
                                       f"error: {str(e)[:80]}", patient_id))

            if ct_to_pet_transform is not None:
                slicer.mrmlScene.RemoveNode(ct_to_pet_transform)
                self._log("info", "  CT→PET transform node removed from scene")
            if ct_node is not None:
                slicer.mrmlScene.RemoveNode(ct_node)
                self._log("info", "  CT node removed from scene")
            slicer.mrmlScene.RemoveNode(pet_node)
            self._log("info",
                f"  PET node removed from scene — memory freed for next patient")
            progress_cb((i + 1) / total * 100)

        done_count  = sum(1 for r in all_rows if r.get("status") == "done")
        error_count = len(all_rows) - done_count
        self._log("info",
            f"runBatch: all patients processed — "
            f"{done_count} OK  {error_count} errors  "
            f"saving to {output_file}",
            status_cb)
        self._saveExcel(all_rows, output_file, append)
        self._log("info",
            f"runBatch: COMPLETE — {len(all_rows)} rows written to {output_file}",
            status_cb)

    # ── Slicer helpers ────────────────────────────────────────────────────

    def _loadDicomSeries(self, dicom_folder, subject_id):
        """
        Load a DICOM series from folder into Slicer.
        Always purges any stale DB entry for this folder first so that
        every run behaves like a clean-DB run (avoids ITK/GDCM re-open errors).
        """
        from DICOMLib import DICOMUtils

        dicom_files = []
        for root, _, files in os.walk(dicom_folder):
            for f in files:
                dicom_files.append(os.path.join(root, f))

        self._log("info",
            f"    _loadDicomSeries: {len(dicom_files)} file(s) found in {dicom_folder}")

        if not dicom_files:
            raise ValueError(f"No DICOM files in {dicom_folder}")

        db = slicer.dicomDatabase
        if not db.isOpen:
            default_db_path = os.path.join(slicer.app.temporaryPath, "CtkDicomDatabase")
            db.openDatabase(default_db_path)
            self._log("info", f"    _loadDicomSeries: opened DICOM DB at {default_db_path}")
        
        
        def _all_series_uids():
            uids = set()
            for patient in db.patients():
                for study in db.studiesForPatient(patient):
                    for series in db.seriesForStudy(study):
                        uids.add(series)
            return uids

        # Purge stale DB entries for this folder
        folder_variants = set([
            os.path.normpath(dicom_folder).lower(),
            os.path.normpath(os.path.realpath(dicom_folder)).lower(),
        ])
        purged = []
        for uid in list(_all_series_uids()):
            files_in_db = db.filesForSeries(uid)
            if not files_in_db:
                continue
            file_norm = os.path.normpath(files_in_db[0]).lower()
            if any(file_norm.startswith(v) for v in folder_variants):
                try:
                    db.removeSeries(uid)
                    purged.append(uid)
                except Exception as e:
                    self._log("warning",
                        f"    _loadDicomSeries: could not remove UID={uid}: {e}")

        if purged:
            self._log("info",
                f"    _loadDicomSeries: purged {len(purged)} stale DB entry/entries: {purged}")
        else:
            self._log("info",
                "    _loadDicomSeries: no existing DB entries for this folder — clean import")

        # Fresh import
        uids_before = _all_series_uids()
        self._log("info", "    _loadDicomSeries: importing into DICOM database...")
        DICOMUtils.importDicom(dicom_folder)

        uids_after = _all_series_uids()
        new_uids   = list(uids_after - uids_before)
        self._log("info",
            f"    _loadDicomSeries: import done — "
            f"{len(new_uids)} new series UID(s): {new_uids}")

        if not new_uids:
            raise ValueError(
                f"importDicom produced no new series UIDs from {dicom_folder}. "
                f"Folder may be empty, contain no valid DICOM files, "
                f"or all files failed to parse."
            )

        # Load into scene
        self._log("info",
            f"    _loadDicomSeries: loading series into scene: {new_uids}...")
        loaded_node_ids = DICOMUtils.loadSeriesByUID(new_uids)
        self._log("info",
            f"    _loadDicomSeries: loadSeriesByUID returned "
            f"{len(loaded_node_ids)} node id(s): {loaded_node_ids}")

        if not loaded_node_ids:
            raise ValueError(
                f"loadSeriesByUID returned no nodes for UIDs {new_uids}. "
                f"Check the VTK/ITK errors above for the root cause."
            )

        node = slicer.mrmlScene.GetNodeByID(loaded_node_ids[0])
        if not node:
            raise ValueError(
                f"Node '{loaded_node_ids[0]}' not found in scene — "
                "DICOM may have loaded as a non-volume type (check series modality)"
            )

        node.SetName(f"PET_{subject_id}")
        self._log("info",
            f"    _loadDicomSeries: volume node ready — "
            f"name={node.GetName()}  id={node.GetID()}")
        return node

    def _getPatientIdFromNode(self, volume_node):
        """Read patient ID from DICOM tag 0010,0020. Falls back to 'UNKNOWN'."""
        self._log("info", "    _getPatientIdFromNode: reading DICOM tag 0010,0020...")
        try:
            sh_node = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(
                slicer.mrmlScene)
            item_id = sh_node.GetItemByDataNode(volume_node)
            uid     = sh_node.GetItemUID(item_id, "DICOM")
            self._log("info", f"    _getPatientIdFromNode: subject hierarchy UID={uid}")
            db    = slicer.dicomDatabase
            files = db.filesForSeries(uid)
            self._log("info",
                f"    _getPatientIdFromNode: "
                f"{len(files)} DICOM file(s) in DB for this series")
            if files:
                import pydicom
                ds         = pydicom.dcmread(files[0], stop_before_pixels=True)
                patient_id = str(ds[0x0010, 0x0020].value).strip()
                self._log("info",
                    f"    _getPatientIdFromNode: extracted patient_id={patient_id}")
                return patient_id
            else:
                self._log("warning",
                    "    _getPatientIdFromNode: no files in DICOM DB for series "
                    "— falling back to UNKNOWN")
        except Exception as e:
            self._log("warning",
                f"    _getPatientIdFromNode: exception reading DICOM tag: {e} "
                "— falling back to UNKNOWN")
        return "UNKNOWN"

    def _loadSegmentation(self, seg_file):
        """Load a NIfTI segmentation file into Slicer as a segmentation node."""
        self._log("info", f"    _loadSegmentation: loading {seg_file}")
        node = slicer.util.loadSegmentation(seg_file)
        if not node:
            raise ValueError(f"Failed to load segmentation: {seg_file}")
        n_segs = node.GetSegmentation().GetNumberOfSegments()
        self._log("info",
            f"    _loadSegmentation: OK — node={node.GetName()}  segments={n_segs}")
        return node

    def _registerCtToPet(self, ct_node, pet_node):
        """
        BRAINSFit registration: CT (moving) → PET (fixed), always applied.
        Runs rigid then affine (rigid initialises affine) so the final linear
        transform may include small scale/shear in addition to rotation+
        translation. Metric: Mattes Mutual Information, sampling 2%.
        Returns a vtkMRMLLinearTransformNode with the CT→PET transform.
        """
        self._log("info", "    _registerCtToPet: creating output transform node...")
        transform_node = slicer.mrmlScene.AddNewNodeByClass(
            'vtkMRMLLinearTransformNode', 'CT_to_PET'
        )

        params = {
            'fixedVolume':             pet_node.GetID(),
            'movingVolume':            ct_node.GetID(),
            'linearTransform':         transform_node.GetID(),
            'useRigid':                True,
            'useAffine':               True,
            'costMetric':              'MMI',
            'samplingPercentage':      0.02,
            'initializeTransformMode': 'useMomentsAlign',
        }

        self._log("info", f"    _registerCtToPet: BRAINSFit params — {params}")
        self._log("info",
            "    _registerCtToPet: running BRAINSFit (rigid+affine, "
            "wait_for_completion=True)...")

        cli_node = slicer.cli.run(
            slicer.modules.brainsfit, None, params, wait_for_completion=True
        )
        status = cli_node.GetStatusString()
        self._log("info", f"    _registerCtToPet: BRAINSFit finished — status={status}")
        slicer.mrmlScene.RemoveNode(cli_node)

        matrix = vtk.vtkMatrix4x4()
        transform_node.GetMatrixTransformToParent(matrix)
        mat_rows = []
        for r in range(4):
            row = [f"{matrix.GetElement(r, c):+.4f}" for c in range(4)]
            mat_rows.append("[" + "  ".join(row) + "]")
        self._log("info",
            "    _registerCtToPet: CT→PET transform matrix:\n"
            + "\n".join(f"      {r}" for r in mat_rows))

        return transform_node

    def _applyTransformToSeg(self, seg_node, transform_node):
        """Harden transform_node onto seg_node (bakes the transform in-place)."""
        self._log("info",
            f"    _applyTransformToSeg: setting transform {transform_node.GetID()} "
            f"on '{seg_node.GetName()}' and hardening...")
        seg_node.SetAndObserveTransformNodeID(transform_node.GetID())
        slicer.vtkSlicerTransformLogic().hardenTransform(seg_node)
        self._log("info", "    _applyTransformToSeg: transform hardened successfully")

    def _runPetIndic(self, pet_node, seg_node, metrics, suv_type):
        """
        Run QuantitativeIndicesTool CLI on pet_node + seg_node.
        Returns dict of metric results keyed by: suv_mean, suv_max, suv_peak,
        tlg, volume_mL (only keys where CLI returned a numeric value).
        """
        segmentation = seg_node.GetSegmentation()
        n_segs = segmentation.GetNumberOfSegments()
        self._log("info", f"    _runPetIndic: segmentation has {n_segs} segment(s)")

        if n_segs == 0:
            raise ValueError("Segmentation has no segments")

        segment_id = segmentation.GetNthSegmentID(0)
        self._log("info", f"    _runPetIndic: using segment_id={segment_id}")

        self._log("info", "    _runPetIndic: checking QuantitativeIndicesCLI module...")
        qi_module = slicer.modules.quantitativeindicescli
        self._log("info", f"    _runPetIndic: module OK — {qi_module}")

        # Export segment to temporary label map aligned to PET geometry
        self._log("info", "    _runPetIndic: exporting segment to label map...")
        label_node = slicer.mrmlScene.AddNewNodeByClass(
            'vtkMRMLLabelMapVolumeNode', 'temp_label'
        )
        segment_ids = vtk.vtkStringArray()
        segment_ids.InsertNextValue(segment_id)
        slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
            seg_node, segment_ids, label_node, pet_node,
            slicer.vtkSegmentation.EXTENT_REFERENCE_GEOMETRY,
        )
        self._log("info",
            f"    _runPetIndic: label map created — id={label_node.GetID()}")

        try:
            parameters = {
                'Grayscale_Image': pet_node.GetID(),
                'Label_Image':     label_node.GetID(),
                'Label_Value':     '1',
            }
            if metrics.get("mean"):   parameters['Mean']   = 'true'
            if metrics.get("max"):    parameters['Max']    = 'true'
            if metrics.get("peak"):   parameters['Peak']   = 'true'
            if metrics.get("tlg"):    parameters['TLG']    = 'true'
            if metrics.get("volume"): parameters['Volume'] = 'true'

            self._log("info",
                f"    _runPetIndic: CLI parameters — {parameters}")
            self._log("info",
                "    _runPetIndic: running CLI (wait_for_completion=True)...")

            cli_node = slicer.cli.run(
                qi_module, None, parameters, wait_for_completion=True
            )

            cli_status = cli_node.GetStatusString()
            self._log("info", f"    _runPetIndic: CLI finished — status={cli_status}")

            # Parse results
            name_map = {
                'Mean':   'suv_mean',
                'Max':    'suv_max',
                'Peak':   'suv_peak',
                'TLG':    'tlg',
                'Volume': 'volume_mL',
            }
            results  = {}
            import math
            n_params = cli_node.GetNumberOfParametersInGroup(3)
            self._log("info",
                f"    _runPetIndic: parsing {n_params} output parameter(s) from group 3...")

            for i in range(n_params):
                raw  = cli_node.GetParameterDefault(3, i)
                name = cli_node.GetParameterName(3, i)
                self._log("info",
                    f"    _runPetIndic:   raw param [{name}] = {raw!r}")
                if raw == '--':
                    continue
                name_clean = name.replace('_s', '').replace('_', ' ').strip()
                if name_clean in name_map:
                    try:
                        val = float(raw)
                        if not math.isnan(val):
                            results[name_map[name_clean]] = val
                    except ValueError:
                        self._log("warning",
                            f"    _runPetIndic: could not convert "
                            f"{raw!r} to float for param '{name}'")

            slicer.mrmlScene.RemoveNode(cli_node)
            self._log("info", f"    _runPetIndic: results parsed — {results}")

            if not results:
                self._log("warning",
                    "    _runPetIndic: results dict is EMPTY — "
                    "CLI may have run but returned no numeric values. "
                    "Check that PET units are correct and label map is non-empty.")

            if "volume_mL" in results:
                self._log("info",
                    f"    _runPetIndic: volume={results['volume_mL']:.3f} mL  "
                    f"(measured in PET voxel space — ~1-2% smaller than CT-resolution "
                    f"Segment Stats due to resampling, this is expected)")

            return results

        finally:
            slicer.mrmlScene.RemoveNode(label_node)
            self._log("info", "    _runPetIndic: temp label map removed")

    def _applySuvConversion(self, pet_node, dicom_folder):
        """
        Convert the PET volume from Bq/mL → SUVbw in-place.
        SUVbw = (Bq/mL × weight_g) / decay_corrected_dose_Bq
        If DICOM Units tag is already 'SUV', conversion is skipped.
        """
        import pydicom
        import numpy as np

        # Find first readable DICOM file
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
        self._log("info", f"    _applySuvConversion: DICOM Units tag = '{units}'")

        if units == "SUV":
            self._log("info",
                "    _applySuvConversion: already SUV — skipping conversion")
            return

        if units not in ("BQML", "BQ/ML", ""):
            self._log("warning",
                f"    _applySuvConversion: unexpected units '{units}' — "
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
        self._log("info",
            f"    _applySuvConversion: PatientWeight = {weight_kg} kg")

        # Injected dose
        try:
            radio_seq = ds[0x0054, 0x0016][0]
        except (KeyError, IndexError):
            raise ValueError(
                "RadiopharmaceuticalInformationSequence (0054,0016) missing — "
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
            self._log("info",
                f"    _applySuvConversion: half-life from standard tag = {half_life_s}s")
        except (KeyError, ValueError):
            try:
                half_life_s = float(str(ds[0x0009, 0x103F].value))
                self._log("info",
                    f"    _applySuvConversion: half-life from GE private tag = {half_life_s}s")
            except (KeyError, ValueError):
                half_life_s = 6586.2  # F-18 default
                self._log("warning",
                    f"    _applySuvConversion: half-life not in DICOM — "
                    f"using F-18 default = {half_life_s}s")

        # Acquisition time and decay correction type
        acq_t_raw  = str(getattr(ds, "AcquisitionTime", "") or
                         getattr(ds, "SeriesTime", "")).split(".")[0].strip()
        decay_corr = str(getattr(ds, "DecayCorrection", "START")).strip().upper()

        self._log("info",
            f"    _applySuvConversion: dose={dose_bq:.0f} Bq  "
            f"inj={inj_str}  acq={acq_t_raw}  "
            f"half_life={half_life_s}s  decay_correction={decay_corr}")

        suv_factor = compute_suvbw_factor(
            weight_kg=weight_kg,
            dose_bq=dose_bq,
            injection_time=inj_str,
            acquisition_time=acq_t_raw,
            half_life_s=half_life_s,
            decay_correction=decay_corr,
        )
        self._log("info",
            f"    _applySuvConversion: SUV factor = {suv_factor:.6f}")

        arr = slicer.util.arrayFromVolume(pet_node)
        max_before = float(arr.max())
        arr_suv    = (arr * suv_factor).astype(np.float32)
        slicer.util.updateVolumeFromArray(pet_node, arr_suv)
        max_after  = float(arr_suv.max())

        self._log("info",
            f"    _applySuvConversion: DONE — "
            f"max: {max_before:.1f} Bq/mL → {max_after:.4f} SUV")

    # ── Excel ─────────────────────────────────────────────────────────────

    def _saveExcel(self, rows, output_file, append):
        lib_save_excel(rows, output_file, append=append, log=self._log)


    def _errorRow(self, subject_id, scan_date, segment, status, patient_id=""):
        return lib_error_row(subject_id, scan_date, segment, status, patient_id)


