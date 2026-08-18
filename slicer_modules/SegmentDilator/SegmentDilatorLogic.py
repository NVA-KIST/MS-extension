"""
SegmentDilatorLogic — Slicer adapter.

lib: dilate_mask, resample_to_target, subtract_dilated_union, bulk_outputs_exist
Logic: export/import Slicer nodes, bulk folder orchestration
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

from lib.processing.dilate import (
    dilate_mask as lib_dilate_mask,
    resample_to_target as lib_resample_to_target,
    subtract_dilated_union,
    bulk_outputs_exist as lib_bulk_outputs_exist,
    file_stem as lib_file_stem,
)


class SegmentDilatorLogic(ScriptedLoadableModuleLogic):

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self, src_configs, tgt_configs):
        """
        src_configs : list of {'seg_node', 'seg_name', 'dilate_mm'}
        tgt_configs : list of {'seg_node', 'seg_name'}

        For every source:
          - export, dilate, save as <name>_<seg>_dilated in scene

        For every target:
          - export, subtract union of all dilated sources (resampled to target
            space), save as <name>_<seg>_subtracted in scene

        Returns (list_of_dilated_names, list_of_subtracted_names).
        """
        import numpy as np

        print("\n" + "=" * 60)
        print("SEGMENT DILATOR")
        print("=" * 60)

        # ── Step 1: dilate every source, collect (arr, affine) pairs ──────────
        dilated_items  = []   # list of (arr, affine) in source voxel space
        dilated_names  = []

        for i, cfg in enumerate(src_configs):
            node      = cfg['seg_node']
            seg_name  = cfg.get('seg_name')
            dilate_mm = cfg['dilate_mm']
            label     = seg_name or "all"
            print(f"\n[SRC {i+1}] '{node.GetName()}' / '{label}'  "
                  f"dilation={dilate_mm} mm")

            arr, affine, lm = self._export_to_array(node, seg_name)
            before = int((arr > 0).sum())

            dilated = self._dilate(arr, affine, dilate_mm)
            after   = int(dilated.sum())
            print(f"[SRC {i+1}]   voxels: {before} → {after}  (+{after - before})")

            # Save dilated as scene node
            dname = f"{node.GetName()}_{label}_sub_dilated"
            self._save_array_as_seg(dilated, lm, dname)
            dilated_names.append(dname)

            dilated_items.append((dilated, affine))
            slicer.mrmlScene.RemoveNode(lm)

        # ── Step 2: subtract union of dilated sources from every target ────────
        subtracted_names = []

        for j, cfg in enumerate(tgt_configs):
            node     = cfg['seg_node']
            seg_name = cfg.get('seg_name')
            label    = seg_name or "all"
            print(f"\n[TGT {j+1}] '{node.GetName()}' / '{label}'")

            tgt_arr, tgt_affine, tgt_lm = self._export_to_array(node, seg_name)
            before = int((tgt_arr > 0).sum())

            before_overlap = int((tgt_arr > 0).sum())
            result = subtract_dilated_union(tgt_arr, tgt_affine, dilated_items)
            after = int((result > 0).sum())
            removed = before_overlap - after
            print(f"[TGT {j+1}]   voxels: {before} → {after}  "
                  f"(removed {removed} overlapping)")

            # Save subtracted result
            sname = f"{node.GetName()}_{label}_subtracted"
            self._save_array_as_seg(result, tgt_lm, sname)
            subtracted_names.append(sname)

            slicer.mrmlScene.RemoveNode(tgt_lm)

        print("\n" + "=" * 60)
        print("DONE")
        print("=" * 60)
        return dilated_names, subtracted_names

    # ── Bulk entry point ─────────────────────────────────────────────────────

    def run_bulk(self, dataset_root, src_file_configs, tgt_file_configs,
                  skip_done=True, progress_cb=None):
        """
        Bulk/folder mode — apply a sample-patient setup to every patient.

        src_file_configs : list of {'filename', 'dilate_mm'}
        tgt_file_configs : list of {'filename'}

        For each patient under dataset_root/Segments/<ID>_Seg/:
          - load each source/target file as a one-segment SegmentationNode
            (named after its file stem, so the 'All segments' / seg_name=None
            path of self.run() applies)
          - call self.run() UNCHANGED — same per-scene dilate+subtract logic
          - export the resulting '<stem>_..._sub_dilated' / '..._subtracted'
            nodes back to '<stem>_dilated.nii.gz' / '<stem>_subtracted.nii.gz'
            in that patient's Segments/<ID>_Seg/ folder

        Per-patient errors are logged (console + dataset_root/pipeline_logs/
        bulk_log.txt) and do not stop the run.
        """
        import datetime

        seg_root = os.path.join(dataset_root, 'Segments')
        if not os.path.isdir(seg_root):
            raise RuntimeError(f"'{seg_root}' not found.")

        log_dir = os.path.join(dataset_root, 'pipeline_logs')
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, 'bulk_log.txt')

        def _log(line):
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(f"[{ts}] [SegmentDilator] {line}\n")
            except Exception as e:
                print(f"[BULK]   WARNING: could not write bulk_log.txt: {e}")

        subjects = sorted(
            d for d in os.listdir(seg_root)
            if d.endswith('_Seg') and os.path.isdir(os.path.join(seg_root, d))
        )
        total = len(subjects)
        print(f"\nSegment Dilator — bulk run: {total} subject(s)")

        for idx, seg_folder in enumerate(subjects):
            subject_id = seg_folder[:-4]
            seg_dir    = os.path.join(seg_root, seg_folder)

            if progress_cb:
                progress_cb(idx, total, subject_id)

            print(f"\n{'='*60}")
            print(f"[{idx+1}/{total}] {subject_id}")
            print(f"{'='*60}")

            if skip_done and self._bulk_outputs_exist(
                    seg_dir, src_file_configs, tgt_file_configs):
                print("  [SKIP] Outputs already exist.")
                continue

            try:
                slicer.mrmlScene.Clear(0)

                # Load every dilation source. If any is missing, the union
                # of dilated sources would be incomplete for ALL targets, so
                # skip this patient entirely.
                src_configs = []
                missing_src = None
                for cfg in src_file_configs:
                    fpath = os.path.join(seg_dir, cfg['filename'])
                    if not os.path.exists(fpath):
                        missing_src = cfg['filename']
                        break
                    src_configs.append({
                        'seg_node':  self._load_nifti_as_seg(fpath),
                        'seg_name':  None,
                        'dilate_mm': cfg['dilate_mm'],
                    })
                if missing_src:
                    print(f"  [ERROR] dilation source '{missing_src}' not found — "
                          f"skipping patient.")
                    _log(f"{subject_id}: SKIPPED (missing source '{missing_src}')")
                    continue

                # Load every target that exists; missing targets are skipped
                # individually (they just won't get a '_subtracted' output).
                tgt_configs = []
                tgt_stems   = []
                for cfg in tgt_file_configs:
                    fpath = os.path.join(seg_dir, cfg['filename'])
                    if not os.path.exists(fpath):
                        print(f"  [WARN] target '{cfg['filename']}' not found — skipping.")
                        continue
                    tgt_configs.append({'seg_node': self._load_nifti_as_seg(fpath),
                                         'seg_name': None})
                    tgt_stems.append(self._stem(cfg['filename']))

                src_stems = [self._stem(cfg['filename']) for cfg in src_file_configs]

                # ── Unchanged core algorithm ──────────────────────────────────
                dilated_names, subtracted_names = self.run(src_configs, tgt_configs)

                for stem, dname in zip(src_stems, dilated_names):
                    out_path = os.path.join(seg_dir, f"{stem}_dilated.nii.gz")
                    self._export_seg_to_nifti(dname, out_path)

                for stem, sname in zip(tgt_stems, subtracted_names):
                    out_path = os.path.join(seg_dir, f"{stem}_subtracted.nii.gz")
                    self._export_seg_to_nifti(sname, out_path)

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

    # ── Bulk helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _stem(filename):
        return lib_file_stem(filename)


    def _bulk_outputs_exist(self, seg_dir, src_file_configs, tgt_file_configs):
        return lib_bulk_outputs_exist(seg_dir, src_file_configs, tgt_file_configs)


    def _load_nifti_as_seg(self, fpath):
        """Load a single-structure NIfTI mask as a one-segment
        SegmentationNode named after the file stem, so self.run()'s
        seg_name=None ('All segments') path applies to it."""
        stem = self._stem(os.path.basename(fpath))
        lm = slicer.util.loadLabelVolume(fpath)
        seg_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode', stem)
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(lm, seg_node)
        slicer.mrmlScene.RemoveNode(lm)
        return seg_node

    def _export_seg_to_nifti(self, seg_node_name, out_path):
        print(f"[SAVE] '{seg_node_name}' -> {os.path.basename(out_path)}")
        try:
            seg_node = slicer.util.getNode(seg_node_name)
        except Exception:
            print(f"[SAVE]   WARNING: node not found.")
            return
        lm = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode', '_sd_export_tmp')
        slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(seg_node, lm)
        ok = slicer.util.saveNode(lm, out_path)
        slicer.mrmlScene.RemoveNode(lm)
        if ok:
            print(f"[SAVE]   OK ({os.path.getsize(out_path)//1024} KB)")
        else:
            print(f"[SAVE]   FAILED")

    # ── Internals ─────────────────────────────────────────────────────────────

    def _export_to_array(self, seg_node, seg_name):
        """
        Export seg_node (or one sub-segment) to a labelmap volume.
        Returns (array_uint8, affine_4x4, lm_node).
        The caller is responsible for removing lm_node when done.
        """
        import numpy as np

        lm = slicer.mrmlScene.AddNewNodeByClass(
            'vtkMRMLLabelMapVolumeNode', '_sd_tmp')

        if seg_name:
            seg    = seg_node.GetSegmentation()
            seg_id = None
            for i in range(seg.GetNumberOfSegments()):
                if seg.GetNthSegment(i).GetName() == seg_name:
                    seg_id = seg.GetNthSegmentID(i)
                    break
            if seg_id is None:
                slicer.mrmlScene.RemoveNode(lm)
                raise ValueError(
                    f"Sub-segment '{seg_name}' not found in "
                    f"'{seg_node.GetName()}'")
            ids = vtk.vtkStringArray()
            ids.InsertNextValue(seg_id)
            slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
                seg_node, ids, lm, None,
                slicer.vtkSegmentation.EXTENT_UNION_OF_SEGMENTS)
        else:
            slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(
                seg_node, lm)

        arr = slicer.util.arrayFromVolume(lm).copy().astype(np.uint8)
        mat = vtk.vtkMatrix4x4()
        lm.GetIJKToRASMatrix(mat)
        affine = self._mat_to_np(mat)
        return arr, affine, lm   # lm intentionally NOT removed here

    def _dilate(self, arr, affine, dilate_mm):
        return lib_dilate_mask(arr, affine, dilate_mm)


    def _save_array_as_seg(self, arr, reference_lm, name):
        """
        Write *arr* into *reference_lm* (reusing its geometry), then import
        as a new SegmentationNode named *name*.  Any existing node with that
        name is replaced.
        """
        # Remove pre-existing node
        try:
            slicer.mrmlScene.RemoveNode(slicer.util.getNode(name))
        except Exception:
            pass

        slicer.util.updateVolumeFromArray(reference_lm, arr)
        seg = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode', name)
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
            reference_lm, seg)
        seg.CreateClosedSurfaceRepresentation()
        print(f"[DILATOR]   → '{name}' added to scene.")

    def _resample_to_target(self, src_arr, src_affine, tgt_shape, tgt_affine):
        return lib_resample_to_target(src_arr, src_affine, tgt_shape, tgt_affine)


    @staticmethod
    def _mat_to_np(mat):
        import numpy as np
        return np.array([[mat.GetElement(r, c) for c in range(4)]
                          for r in range(4)])

