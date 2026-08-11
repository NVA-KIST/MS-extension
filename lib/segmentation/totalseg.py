"""TotalSegmentator wrappers (Python API + CLI)."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional, Sequence

# ROIs needed for combined_mask / VF (from visceral_fat_segmentations.py)
VF_TOTAL_ROIS = [
    "liver",
    "heart",
    "vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4", "vertebrae_L5",
    "vertebrae_T1", "vertebrae_T2", "vertebrae_T3", "vertebrae_T4", "vertebrae_T5",
    "vertebrae_T6", "vertebrae_T7", "vertebrae_T8", "vertebrae_T9",
    "vertebrae_T10", "vertebrae_T11", "vertebrae_T12",
    "rib_left_1", "rib_left_2", "rib_left_3", "rib_left_4", "rib_left_5", "rib_left_6",
    "rib_left_7", "rib_left_8", "rib_left_9", "rib_left_10", "rib_left_11", "rib_left_12",
    "rib_right_1", "rib_right_2", "rib_right_3", "rib_right_4", "rib_right_5", "rib_right_6",
    "rib_right_7", "rib_right_8", "rib_right_9", "rib_right_10", "rib_right_11", "rib_right_12",
]

# Extra target organs we also want as individual files
TARGET_EXTRA_ROIS = [
    "iliopsoas_left",
    "iliopsoas_right",
    "spleen",
]

# Abdomen group organs (packed into abdomen.seg.nrrd)
ABDOMEN_ROIS = [
    "liver",
    "spleen",
    "pancreas",
    "kidney_right",
    "kidney_left",
    "gallbladder",
    "adrenal_gland_right",
    "adrenal_gland_left",
    # Hollow / GI
    "urinary_bladder",
    "small_bowel",
    "colon",
    "duodenum",
    "stomach",
]

VESSEL_ROIS = [
    "aorta",
    "iliac_artery_left", "iliac_artery_right",
    "iliac_vena_left", "iliac_vena_right",
    "inferior_vena_cava",
    "portal_vein_and_splenic_vein",
]

# Tasks that do NOT support --fast in typical TS installs
_NO_FAST_TASKS = {"tissue_types", "tissue_types_mr", "body"}


def run_totalsegmentator_api(input_path: Path | str, output_dir: Path | str, **kwargs: Any) -> None:
    """
    Run TotalSegmentator Python API with GPU preference.
    Falls back if installed API does not support ``device``.
    """
    from totalsegmentator.python_api import totalsegmentator

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    call_kwargs: dict[str, Any] = dict(kwargs)
    if "device" not in call_kwargs:
        call_kwargs["device"] = "gpu"
    try:
        totalsegmentator(str(input_path), str(output_dir), **call_kwargs)
    except TypeError as exc:
        if "unexpected keyword argument 'device'" not in str(exc):
            raise
        call_kwargs.pop("device", None)
        totalsegmentator(str(input_path), str(output_dir), **call_kwargs)


def run_totalsegmentator_cli(
    ct_nii: str,
    out_dir: str,
    task: str,
    ts_cmd: str = "TotalSegmentator",
    gpu: str = "gpu",
    ts_fast: bool = True,
    check_files: Optional[Sequence[str]] = None,
    log=None,
) -> None:
    """CLI-based TS used by the Slicer pipeline. Port of Logic.runTS."""

    def _info(msg: str):
        if log is not None and hasattr(log, "info"):
            log.info(msg)
        else:
            print(f"[TS] {msg}")

    def _warn(msg: str):
        if log is not None and hasattr(log, "warn"):
            log.warn(msg)
        else:
            print(f"[TS][WARN] {msg}")

    os.makedirs(out_dir, exist_ok=True)

    if check_files:
        try:
            from lib.io.nifti import ok_nii as _ok_nii
        except Exception:
            _ok_nii = None
        missing = []
        for f in check_files:
            fp = os.path.join(out_dir, f)
            if _ok_nii is not None:
                if not _ok_nii(fp, min_kb=50):
                    missing.append(f)
            elif (
                not os.path.isfile(fp)
                or os.path.getsize(fp) < 50 * 1024
            ):
                missing.append(f)
        if not missing:
            _info(f"All {len(check_files)} outputs present — skip task '{task}'")
            return
        _warn(f"Re-running '{task}': missing/invalid {missing[:8]}")

    cmd = [ts_cmd, "-i", ct_nii, "-o", out_dir, "--task", task, "--device", gpu]
    if ts_fast and task not in _NO_FAST_TASKS:
        cmd.append("--fast")
    elif ts_fast and task in _NO_FAST_TASKS:
        _warn(f"--fast ignored for task '{task}'")

    _info(" ".join(cmd))
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    lines = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            _info(f"  {line}")
            lines.append(line)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(
            f"TotalSegmentator '{task}' failed (exit {proc.returncode})\n"
            + "\n".join(lines[-30:])
        )


def run_totalseg_for_visceral_fat(
    ct_path: Path | str,
    out_dir: Path | str,
    *,
    device: str = "gpu",
    include_targets: bool = True,
    include_vessels: bool = True,
    include_abdomen: bool = True,
    use_api: bool = True,
    log=None,
) -> Path:
    """
    Run the TS tasks needed for VF + target organs + abdomen + optional vessels.

    Tasks:
      1. total  (ROI subset: VF anatomy + psoas/spleen + abdomen + vessels)
      2. body   → body_trunc
      3. tissue_types → torso_fat  (needs academic licence)
    """
    ct_path = Path(ct_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _info(msg: str):
        print(msg) if log is None else (
            log.info(msg) if hasattr(log, "info") else print(msg)
        )

    rois = list(VF_TOTAL_ROIS)
    if include_targets:
        rois.extend(TARGET_EXTRA_ROIS)
    if include_abdomen:
        rois.extend(ABDOMEN_ROIS)
    if include_vessels:
        rois.extend(VESSEL_ROIS)
    # unique preserve order
    seen = set()
    rois = [r for r in rois if not (r in seen or seen.add(r))]

    _info(f"[TS] total task → {out_dir}  ({len(rois)} ROIs)")
    if use_api:
        run_totalsegmentator_api(
            ct_path,
            out_dir,
            task="total",
            roi_subset=rois,
            device=device,
            nr_thr_resamp=1,
            nr_thr_saving=1,
        )
    else:
        run_totalsegmentator_cli(
            str(ct_path), str(out_dir), "total", gpu=device, ts_fast=True, log=log
        )

    for task in ("body", "tissue_types"):
        task_tmp = out_dir / f"_temp_{task}"
        task_tmp.mkdir(parents=True, exist_ok=True)
        _info(f"[TS] {task} task…")
        try:
            if use_api:
                run_totalsegmentator_api(ct_path, task_tmp, task=task, device=device)
            else:
                run_totalsegmentator_cli(
                    str(ct_path), str(task_tmp), task, gpu=device, ts_fast=False, log=log
                )
            for fname in os.listdir(task_tmp):
                if fname.endswith(".nii.gz"):
                    src = task_tmp / fname
                    dst = out_dir / fname
                    if dst.exists():
                        dst.unlink()
                    shutil.move(str(src), str(dst))
                    _info(f"  [{task}] {fname}")
        except SystemExit as e:
            _info(f"[TS][WARN] {task} exited (code={getattr(e, 'code', e)}) — licence?")
        except Exception as e:
            _info(f"[TS][WARN] {task} failed: {e}")
        finally:
            if task_tmp.exists():
                shutil.rmtree(task_tmp, ignore_errors=True)

    return out_dir
