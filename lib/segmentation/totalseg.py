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
    # Match CLI default: --fast for the total task unless caller overrides.
    if "fast" not in call_kwargs and call_kwargs.get("task", "total") == "total":
        call_kwargs["fast"] = True
    try:
        print(f"[TS] API start task={call_kwargs.get('task')} device={call_kwargs.get('device')} "
              f"fast={call_kwargs.get('fast')}", flush=True)
        totalsegmentator(str(input_path), str(output_dir), **call_kwargs)
        print(f"[TS] API done task={call_kwargs.get('task')}", flush=True)
    except TypeError as exc:
        msg = str(exc)
        if "unexpected keyword argument 'device'" in msg:
            call_kwargs.pop("device", None)
            totalsegmentator(str(input_path), str(output_dir), **call_kwargs)
            return
        if "unexpected keyword argument 'fast'" in msg:
            call_kwargs.pop("fast", None)
            totalsegmentator(str(input_path), str(output_dir), **call_kwargs)
            return
        raise


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
    import sys

    def _info(msg: str):
        if log is not None and hasattr(log, "info"):
            log.info(msg)
        else:
            print(f"[TS] {msg}", flush=True)

    def _warn(msg: str):
        if log is not None and hasattr(log, "warn"):
            log.warn(msg)
        else:
            print(f"[TS][WARN] {msg}", flush=True)

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

    args = ["-i", ct_nii, "-o", out_dir, "--task", task, "--device", str(gpu)]
    if ts_fast and task not in _NO_FAST_TASKS:
        args.append("--fast")
    elif ts_fast and task in _NO_FAST_TASKS:
        _warn(f"--fast ignored for task '{task}'")

    # Windows conda TotalSegmentator.exe is often a broken stub (exit 1, no
    # stdout/stderr). Prefer the same interpreter's Python entry point.
    use_py_entry = False
    try:
        import totalsegmentator  # noqa: F401
        use_py_entry = Path(sys.executable).is_file()
    except Exception:
        use_py_entry = False

    if use_py_entry:
        bootstrap = (
            "import sys; "
            "from totalsegmentator.bin.TotalSegmentator import main; "
            "sys.argv=['TotalSegmentator']+sys.argv[1:]; "
            "raise SystemExit(main() or 0)"
        )
        cmd = [sys.executable, "-u", "-c", bootstrap, *args]
    else:
        cmd = [ts_cmd, *args]

    _info(" ".join(cmd))
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
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
        tail = "\n".join(lines[-40:]) if lines else "(no output captured)"
        raise RuntimeError(
            f"TotalSegmentator '{task}' failed (exit {proc.returncode})\n{tail}"
        )


def run_totalseg_for_visceral_fat(
    ct_path: Path | str,
    out_dir: Path | str,
    *,
    device: str = "gpu",
    include_targets: bool = True,
    include_vessels: bool = True,
    include_abdomen: bool = True,
    use_api: bool | None = None,
    ts_cmd: str = "TotalSegmentator",
    log=None,
) -> Path:
    """
    Run the TS tasks needed for VF + target organs + abdomen + optional vessels.

    Tasks:
      1. total  (ROI subset via API, or full total via CLI --fast)
      2. body   → body_trunc
      3. tissue_types → torso_fat  (needs academic licence)

    Prefers the TotalSegmentator CLI when available (streams progress, uses --fast).
    Set use_api=True to force the Python API (still uses fast=True for total).
    """
    import shutil

    ct_path = Path(ct_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _info(msg: str):
        line = msg if isinstance(msg, str) else str(msg)
        if log is None:
            print(line, flush=True)
        elif hasattr(log, "info"):
            log.info(line)
        else:
            print(line, flush=True)

    if use_api is None:
        # Prefer subprocess CLI via current Python entrypoint (not the often-broken
        # Windows TotalSegmentator.exe stub). Fall back to in-process API only if
        # totalsegmentator cannot be imported here.
        try:
            import totalsegmentator  # noqa: F401
            use_api = False
            _info("[TS] mode=CLI (python -c totalsegmentator.bin.TotalSegmentator)")
        except Exception:
            if shutil.which(ts_cmd):
                use_api = False
                _info(f"[TS] mode=CLI (executable {ts_cmd})")
            else:
                use_api = True
                _info("[TS] mode=API (TotalSegmentator not importable / not on PATH)")

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
            fast=True,
            nr_thr_resamp=1,
            nr_thr_saving=1,
        )
    else:
        run_totalsegmentator_cli(
            str(ct_path), str(out_dir), "total",
            ts_cmd=ts_cmd, gpu=device, ts_fast=True, log=log,
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
                    str(ct_path), str(task_tmp), task,
                    ts_cmd=ts_cmd, gpu=device, ts_fast=False, log=log,
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
