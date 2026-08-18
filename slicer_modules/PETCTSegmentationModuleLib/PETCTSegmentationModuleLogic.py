"""
PETCTSegmentationModuleLogic — Slicer adapter.

Pure work lives in lib/io + lib/segmentation + lib/processing.mirroring.
"""
from __future__ import annotations

import os
import sys
import time

from slicer.ScriptedLoadableModule import ScriptedLoadableModuleLogic

_EXT_NEW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _EXT_NEW_ROOT not in sys.path:
    sys.path.insert(0, _EXT_NEW_ROOT)

from lib.io.nifti import (  # noqa: E402
    convert_dicom_to_nifti,
    mb_str,
    ok_nii,
)
from lib.io.paths import discover_patients as lib_discover_patients  # noqa: E402
from lib.io.paths import parse_folder_name  # noqa: E402
from lib.processing.mirroring import flip_nifti_file  # noqa: E402
from lib.segmentation.totalseg import run_totalsegmentator_cli  # noqa: E402
from lib.segmentation.visceral_fat import (  # noqa: E402
    build_combined_mask,
    predict_visceral_fat,
)


class PETCTSegmentationModuleLogic(ScriptedLoadableModuleLogic):
    """Thin adapter: Widget → lib. Thread-safe for bulk steps (no MRML)."""

    def discoverPatients(self, root, *a, **kw):
        """
        Return ``[{subject_id, scan_date}, …]`` from PET/ folders
        (same contract as the original Widget).
        """
        pet_root = os.path.join(root, "PET")
        if not os.path.isdir(pet_root):
            # Fall back to richer discovery if PET/ missing
            rows = lib_discover_patients(root)
            if not rows:
                raise ValueError(f"No PET/ or CT/ subjects under: {root}")
            return [
                {"subject_id": r["subject_id"], "scan_date": r["scan_date"]}
                for r in rows
            ]

        scans = []
        for name in sorted(os.listdir(pet_root)):
            parsed = parse_folder_name(name)
            if not parsed:
                continue
            subj, date, kind = parsed
            if kind != "PET":
                continue
            if os.path.isdir(os.path.join(pet_root, name)):
                scans.append({"subject_id": subj, "scan_date": date})
        if not scans:
            raise ValueError(
                f"No '{{ID}}_{{YYYY-MM-DD}}_PET' folders found in {pet_root}"
            )
        return scans

    def convertDicom(self, dicom_dir, out_nii, log, *a, **kw):
        convert_dicom_to_nifti(dicom_dir, out_nii, log=log)

    def runTS(
        self,
        ct_nii,
        out_dir,
        task,
        ts_cmd,
        device,
        fast,
        check_files,
        log,
        *a,
        **kw,
    ):
        # Prefer ok_nii integrity over bare size checks inside CLI helper
        if check_files:
            missing = [
                f
                for f in check_files
                if not ok_nii(os.path.join(out_dir, f), min_kb=50, log=log)
            ]
            if not missing:
                if hasattr(log, "ok"):
                    log.ok(
                        f"All {len(check_files)} expected output(s) valid — "
                        f"SKIPPING task '{task}'."
                    )
                return
            if hasattr(log, "warn"):
                log.warn(
                    f"{len(missing)}/{len(check_files)} expected output(s) "
                    f"missing/invalid for task '{task}' — re-running."
                )

        run_totalsegmentator_cli(
            ct_nii=ct_nii,
            out_dir=out_dir,
            task=task,
            ts_cmd=ts_cmd or "TotalSegmentator",
            gpu=device or "gpu",
            ts_fast=bool(fast),
            check_files=None,  # already handled above
            log=log,
        )

        if check_files:
            still = [
                f
                for f in check_files
                if not ok_nii(os.path.join(out_dir, f), min_kb=50)
            ]
            if still and hasattr(log, "warn"):
                log.warn(
                    f"TS completed but {len(still)}/{len(check_files)} "
                    f"expected output(s) still missing/invalid: {still}"
                )
            elif hasattr(log, "ok"):
                log.ok(f"Task '{task}' outputs present.")

    def buildCombinedMask(self, seg_dir, out_path, log, label_map=None, *a, **kw):
        if hasattr(log, "sep"):
            log.sep("buildCombinedMask")
        if ok_nii(out_path, min_kb=500, need_nonzero=True, log=log):
            if hasattr(log, "ok"):
                log.ok(f"Combined mask valid ({mb_str(out_path)}) — SKIPPING.")
            return
        if os.path.isfile(out_path) and hasattr(log, "warn"):
            log.warn(
                "Combined mask exists but failed integrity check — regenerating."
            )
        build_combined_mask(seg_dir, out_path, label_map=label_map, log=log)

    def runVFPrediction(
        self,
        combined,
        ct_nii,
        out,
        ckpt,
        device,
        log,
        overlap=0.10,
        *a,
        **kw,
    ):
        if hasattr(log, "sep"):
            log.sep("VF prediction — SegResNet")
        if hasattr(log, "info"):
            log.info(f"Combined mask  : {combined} ({mb_str(combined)})")
            log.info(f"CT NIfTI       : {ct_nii} ({mb_str(ct_nii)})")
            log.info(f"Output         : {out}")
            log.info(f"Checkpoint     : {ckpt or '(not set)'}")
            log.info(f"Device         : {device}")

        if ok_nii(out, min_kb=50, need_nonzero=True, log=log):
            if hasattr(log, "ok"):
                log.ok(f"visceral_fat.nii.gz valid ({mb_str(out)}) — SKIPPING.")
            return
        if os.path.isfile(out) and hasattr(log, "warn"):
            log.warn(
                "visceral_fat.nii.gz exists but failed integrity check — regenerating."
            )

        if not ckpt or not os.path.isfile(ckpt):
            if hasattr(log, "warn"):
                log.warn("No checkpoint configured or file not found.")
                log.warn(
                    f"→ Place visceral_fat.nii.gz manually in: {os.path.dirname(out)}"
                )
                log.warn("  Waiting up to 30 min…")
            deadline = time.time() + 1800
            while not os.path.isfile(out):
                if time.time() > deadline:
                    raise FileNotFoundError(
                        f"visceral_fat.nii.gz not found after 30 min.\n"
                        f"Set a checkpoint in Setup or place the file at: {out}"
                    )
                time.sleep(10)
            if hasattr(log, "ok"):
                log.ok(f"visceral_fat.nii.gz placed manually ({mb_str(out)}).")
            return

        t0 = time.perf_counter()
        predict_visceral_fat(
            ct_path=ct_nii,
            combined_path=combined,
            ckpt_path=ckpt,
            out_path=out,
            device=device,
            overlap=float(overlap),
            log=log,
            auto_orient=True,
        )
        elapsed = time.perf_counter() - t0
        if not os.path.isfile(out):
            raise RuntimeError(
                f"predict_visceral_fat ran but output not found: {out}"
            )
        if hasattr(log, "ok"):
            log.ok(f"VF prediction saved: {mb_str(out)} in {elapsed:.0f}s")

    def _flipNiftiAxis(self, nii_path, ref_ct, axis, axis_name, log, *a, **kw):
        if hasattr(log, "sep"):
            log.sep(f"flipNifti{axis_name}Axis")
        if hasattr(log, "info"):
            log.info(f"File to flip: {nii_path} ({mb_str(nii_path)})")
        flip_nifti_file(
            nii_path, ref_ct, int(axis), axis_name=str(axis_name), log=log
        )
        if hasattr(log, "ok"):
            log.ok(f"Flipped on {axis_name}-axis and saved: {nii_path}")

    def flipNiftiXAxis(self, nii_path, ref_ct, log, *a, **kw):
        self._flipNiftiAxis(nii_path, ref_ct, axis=0, axis_name="X", log=log)

    def flipNiftiYAxis(self, nii_path, ref_ct, log, *a, **kw):
        self._flipNiftiAxis(nii_path, ref_ct, axis=1, axis_name="Y", log=log)
