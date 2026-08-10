"""
UreterPostProcessLogic — Slicer adapter.

lib: build_ureter_mask_from_pet, connect_ureter_path, apply_organ_processing,
     apply_exclusion_mask, dilate_mask, resample_to_target
Logic: Slicer node I/O, DICOM load, vertebra Z bounds from scene
"""
from __future__ import annotations

import os
import sys
import vtk
import slicer
import numpy as np
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleLogic

_EXT_NEW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _EXT_NEW_ROOT not in sys.path:
    sys.path.insert(0, _EXT_NEW_ROOT)

from lib.processing.dilate import dilate_mask as lib_dilate_mask
from lib.processing.dilate import resample_to_target as lib_resample_to_target
from lib.processing.ureter import (
    build_ureter_mask_from_pet,
    connect_ureter_path as lib_connect_ureter_path,
    apply_organ_processing as lib_apply_organ_processing,
    apply_exclusion_mask as lib_apply_exclusion_mask,
)


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
        # Build ureter mask whenever generate_ureter is on — regardless of organ list
        # (covers ureter-only mode where the user wants just the mask, no organs).
        needs_ureter = generate_ureter
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
                 excl_file_configs=None,
                 progress_cb=None):
        """
        organ_file_configs: list of (filename, mode)
        excl_file_configs:  list of {'filename', 'dilate_mm', 'suv_thresh'} —
                             file-based equivalent of the Scene tab's
                             "extra exclusion masks" (_exclRows/excl_configs).
                             Each is dilated once per subject and then applied
                             to every organ exactly like _process_organ_scene's
                             excl_configs loop (independent SUV threshold per
                             mask). A missing file is skipped with a warning —
                             it does not abort the subject.
        Outputs one '<stem>_processed.nii.gz' per organ per subject.
        """
        seg_root = os.path.join(dataset_clean_root, 'Segments')
        pet_root = os.path.join(dataset_clean_root, 'PET')

        if vertebrae_seg_names is None:
            vertebrae_seg_names = ['vertebrae_L1', 'vertebrae_L2', 'vertebrae_L3',
                                   'vertebrae_L4', 'vertebrae_L5']
        if organ_file_configs is None:
            organ_file_configs = []
        if excl_file_configs is None:
            excl_file_configs = []

        needs_ureter = any(m in ('Clean only', 'Clip + Clean') for _, m in organ_file_configs)
        # Exclusion masks are cleaned against PET too, so PET must be loaded
        # whenever either the ureter mask or any exclusion mask needs it.
        needs_pet    = needs_ureter or bool(excl_file_configs)

        import datetime
        log_dir  = os.path.join(dataset_clean_root, 'pipeline_logs')
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, 'bulk_log.txt')

        def _log(line):
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(f"[{ts}] [UreterPostProcess] {line}\n")
            except Exception as e:
                print(f"  WARNING: could not write bulk_log.txt: {e}")

        subjects = sorted(
            d for d in os.listdir(seg_root)
            if d.endswith('_Seg') and os.path.isdir(os.path.join(seg_root, d))
        )
        total = len(subjects)
        print(f"\nBulk run: {total} subjects  ureter_needed={needs_ureter}  "
              f"pet_needed={needs_pet}  excl_masks={len(excl_file_configs)}")

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
                _log(f"{subject_id}: SKIPPED (outputs already exist)")
                continue

            if needs_pet and not os.path.isdir(pet_dir):
                print("  [SKIP] PET folder not found "
                      "(required for clean mode / exclusion masks).")
                _log(f"{subject_id}: SKIPPED (PET folder not found)")
                continue

            print("  [SCENE] Clearing Slicer scene…")
            slicer.mrmlScene.Clear(0)

            try:
                # Load PET only if needed
                pet_arr = pet_affine = pet_mat = vox_size = None
                if needs_pet:
                    pet_node = self._load_pet_dicom(pet_dir, subject_id)
                    if pet_node is None:
                        print("  [ERROR] Could not load PET — skipping.")
                        _log(f"{subject_id}: SKIPPED (could not load PET)")
                        continue
                    pet_arr, pet_affine, pet_mat, vox_size = self._get_pet(pet_node)

                # Compute Z bounds
                totalseg_path = os.path.join(seg_dir, totalseg_file)
                if not os.path.exists(totalseg_path):
                    print(f"  [ERROR] '{totalseg_file}' not found — skipping.")
                    _log(f"{subject_id}: SKIPPED (missing '{totalseg_file}')")
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

                # Build extra exclusion masks (file-based equivalent of the
                # Scene tab's "extra exclusion masks" / excl_configs).
                excl_masks = []
                for excl_cfg in excl_file_configs:
                    excl_fname = excl_cfg['filename']
                    excl_fpath = os.path.join(seg_dir, excl_fname)
                    if not os.path.exists(excl_fpath):
                        print(f"  [EXCL] '{excl_fname}' not found — skipping this mask.")
                        continue
                    excl_arr, excl_affine = self._build_mask_from_file(
                        excl_fpath, excl_cfg.get('dilate_mm', 0), label=excl_fname
                    )
                    excl_masks.append((excl_fname, excl_arr, excl_affine,
                                        excl_cfg.get('suv_thresh', 1.5)))

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
                        excl_masks=excl_masks,
                    )

                print(f"\n  [DONE] {subject_id}")
                _log(f"{subject_id}: OK")

            except Exception:
                import traceback
                tb = traceback.format_exc()
                print(f"\n  [ERROR] {subject_id}:")
                print(tb)
                last_line = tb.strip().splitlines()[-1] if tb.strip() else "unknown error"
                _log(f"{subject_id}: ERROR - {last_line}")

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
                             z_inferior, z_superior, out_path,
                             excl_masks=None):
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

        # ── Apply extra exclusion masks (file-based) ──────────────────────────
        if excl_masks:
            for label, excl_arr, excl_affine, excl_suv in excl_masks:
                try:
                    excl_in = self._resample_to_target(
                        excl_arr, excl_affine, arr.shape, affine)
                    pet_in  = self._resample_to_target(
                        pet_arr, pet_affine, arr.shape, affine)
                    remove = ((arr > 0) & (excl_in > 0) & (pet_in > excl_suv))
                    n_removed = int(remove.sum())
                    arr[remove] = 0
                    print(f"[ORGAN]   Excl '{label}' removed {n_removed} voxels "
                          f"(SUV>{excl_suv})")
                except Exception as e:
                    print(f"[ORGAN]   Excl error: {e}")

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
        return lib_apply_organ_processing(
            organ_arr, organ_affine, mode,
            ureter_arr=ureter_arr, ureter_affine=ureter_affine,
            pet_arr=pet_arr, pet_affine=pet_affine,
            suv_clean_thresh=suv_clean_thresh,
            z_inferior=z_inferior, z_superior=z_superior,
        )


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
        """Build ureter exclusion mask; array core in lib, scene nodes here."""
        if z_inferior is None or z_superior is None:
            raise ValueError(
                "_build_ureter_mask called without Z bounds. "
                "Ensure L1-L5 vertebrae are selected before running.")

        ureter_z_inf = None
        if inf_bound_seg_name and totalseg_node_name:
            try:
                seg_z_min, _ = self._segment_ras_z_bounds(
                    totalseg_node_name, inf_bound_seg_name)
                ureter_z_inf = seg_z_min
                print(
                    f"[URETER] Inferior boundary from '{inf_bound_seg_name}': "
                    f"Z_inf set to {ureter_z_inf:.1f} mm"
                )
            except Exception as e:
                print(
                    f"[URETER] WARNING: could not use '{inf_bound_seg_name}' "
                    f"({e}) - falling back to fixed offset."
                )

        if totalseg_node_name is not None:
            cx, cy = self._vertebrae_xy_centroid(totalseg_node_name)
        else:
            # Fallback: image XY centre in RAS
            shape = pet_arr.shape
            z_idx, y_idx, x_idx = np.meshgrid(
                np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]),
                indexing='ij')
            ijk_hom = np.stack([
                x_idx.ravel(), y_idx.ravel(), z_idx.ravel(),
                np.ones(x_idx.size)], axis=1).astype(np.float32)
            ras = (np.asarray(pet_affine) @ ijk_hom.T).T
            cx = float(np.mean(ras[:, 0]))
            cy = float(np.mean(ras[:, 1]))

        print(
            f"[URETER] SUV>{suv_thresh}  dilation={dilate_mm}mm  "
            f"torso XY=({cx:.1f},{cy:.1f})"
        )

        ureter_mask = build_ureter_mask_from_pet(
            pet_arr, pet_affine, vox_size,
            z_inferior, z_superior, suv_thresh, dilate_mm,
            torso_center_xy=(cx, cy),
            ureter_z_inf=ureter_z_inf,
            ureter_ext_inf_mm=ureter_ext_inf_mm,
            torso_radius_mm=torso_radius_mm,
            connect_path=connect_path,
            max_gap_mm=max_gap_mm,
            fill_holes=fill_holes,
        )
        print(f"[URETER] mask voxels: {int(ureter_mask.sum())}")

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
        return lib_connect_ureter_path(mask_arr, vox_size, max_gap_mm, tube_radius_vox)


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

    def _build_mask_from_file(self, fpath, dilate_mm, label=None):
        """
        File-based equivalent of _build_mask_from_seg, for bulk mode.

        Loads a NIfTI mask from *fpath*, optionally dilates it by *dilate_mm*
        millimetres using the SAME dilation math as _build_mask_from_seg, and
        returns (arr, affine). Does not write anything back to the scene.
        """
        import numpy as np
        from scipy import ndimage

        lm = slicer.util.loadLabelVolume(fpath)
        arr = slicer.util.arrayFromVolume(lm).copy().astype(np.uint8)
        mat = vtk.vtkMatrix4x4()
        lm.GetIJKToRASMatrix(mat)
        affine = self._mat_to_np(mat)
        slicer.mrmlScene.RemoveNode(lm)

        if dilate_mm > 0:
            vox_size   = np.array([abs(affine[0, 0]), abs(affine[1, 1]), abs(affine[2, 2])])
            mean_vox   = float(vox_size.mean())
            iterations = max(1, int(round(dilate_mm / mean_vox)))
            struct     = ndimage.generate_binary_structure(3, 1)
            arr = ndimage.binary_dilation(
                arr > 0, structure=struct, iterations=iterations).astype(np.uint8)
            print(f"[EXCL] '{label}': dilated {dilate_mm} mm ({iterations} iter) "
                  f"→ {int(arr.sum())} voxels")

        return arr, affine

    def _resample_to_target(self, src_arr, src_affine, tgt_shape, tgt_affine):
        return lib_resample_to_target(src_arr, src_affine, tgt_shape, tgt_affine)


