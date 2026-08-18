# -*- coding: utf-8 -*-
"""
run_pipeline.py
===============
Master script: (optional organize) -> generate_segments -> postprocessing -> quantification.

Equivalent to running these three commands in sequence:

    python scripts/generate_segments.py
        --root ROOT --ckpt CKPT --device gpu --limit N

    python scripts/postprocessing.py
        --root ROOT
        --organs "visceral_fat.nii.gz,iliopsoas_left.nii.gz,iliopsoas_right.nii.gz,spleen.nii.gz"
        --suv-thresh 2.5 --ureter-dilate-mm 18 --ureter-ext-inf-mm 50
        --group-subtract-mm 5 --clean-exclude-mm 13
        --suv-clean-fat 1.2 --suv-clean-psoas 1.6
        --limit N [--no-skip-done]

    python scripts/quantification.py
        --root ROOT --out OUT
        --segments visceral_fat,spleen,iliopsoas_left,iliopsoas_right
        --radiomics --bin-width 0.25
        --limit N [--no-append]

Usage
-----
    cd extension_new

    # Full run, single subject, with radiomics
    python scripts/run_pipeline.py
        --root "E:\\KUPETCTMS\\temp sample"
        --ckpt "E:\\KUPETCTMS\\extension\\models\\epoch=399-step=8800.ckpt"
        --out  "E:\\KUPETCTMS\\temp sample\\metrics.xlsx"
        --limit 1 --radiomics

    # Skip segmentation (already done), force re-run post + quant
    python scripts/run_pipeline.py
        --root "E:\\KUPETCTMS\\temp sample"
        --ckpt placeholder
        --out  metrics.xlsx
        --skip-seg --no-skip-done --no-append --limit 1

Stage skip flags
----------------
  --src PATH       Stage 0: organize inbound folder into --root first
  --skip-seg       Skip generate_segments
  --skip-post      Skip postprocessing
  --skip-quant     Skip quantification
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

# Fix Windows cp949 console encoding
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import importlib.util


def _load_script(name: str):
    """Load a sibling script as a module and return it."""
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(f"_kupet_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _banner(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def _elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{int(s // 60)}m {int(s % 60)}s"


# ── Stage runners ─────────────────────────────────────────────────────────────

def run_organize(args) -> int:
    _banner("STAGE 0  Organize inbound data  (organize)")
    t0 = time.time()
    argv = ["--src", str(args.src), "--dest", str(args.root)]
    if args.existing_map:
        argv += ["--existing-map", str(args.existing_map)]
    if args.no_skip_done:
        argv.append("--no-skip-existing")
    print(f"  args: {' '.join(argv)}")
    rc = _load_script("organize.py").main(argv)
    print(f"\nStage 0 finished in {_elapsed(t0)}  exit={rc}")
    return rc


def run_segmentation(args) -> int:
    _banner("STAGE 1  Segmentation  (generate_segments)")
    t0 = time.time()

    argv = [
        "--root", str(args.root),
        "--ckpt", str(args.ckpt),
        "--device", args.device,
    ]
    if args.limit:
        argv += ["--limit", str(args.limit)]
    if args.cuda:
        argv += ["--cuda", str(args.cuda)]
    if args.skip_ts:
        argv.append("--skip-ts")
    if args.skip_vf:
        argv.append("--skip-vf")
    if args.no_nrrd:
        argv.append("--no-nrrd")
    if args.keep_loose:
        argv.append("--keep-loose")
    if args.no_auto_orient:
        argv.append("--no-auto-orient")
    if args.no_skip_done:
        argv.append("--no-skip-done")

    print(f"  args: {' '.join(argv)}")
    rc = _load_script("generate_segments.py").main(argv)
    print(f"\nStage 1 finished in {_elapsed(t0)}  exit={rc}")
    return rc


def run_postprocessing(args) -> int:
    _banner("STAGE 2  Post-processing  (postprocessing)")
    t0 = time.time()

    argv = [
        "--root",            str(args.root),
        "--organs",          args.organs,
        "--suv-thresh",      str(args.suv_thresh),
        "--ureter-dilate-mm",  str(args.ureter_dilate_mm),
        "--ureter-ext-inf-mm", str(args.ureter_ext_inf_mm),
        "--group-subtract-mm", str(args.group_subtract_mm),
        "--clean-exclude-mm",  str(args.clean_exclude_mm),
        "--suv-clean-fat",   str(args.suv_clean_fat),
        "--suv-clean-psoas", str(args.suv_clean_psoas),
    ]
    if args.limit:
        argv += ["--limit", str(args.limit)]
    if args.no_skip_done:
        argv.append("--no-skip-done")

    print(f"  args: {' '.join(argv)}")
    rc = _load_script("postprocessing.py").main(argv)
    print(f"\nStage 2 finished in {_elapsed(t0)}  exit={rc}")
    return rc


def run_quantification(args) -> int:
    _banner("STAGE 3  Quantification  (quantification)")
    t0 = time.time()

    argv = [
        "--root",     str(args.root),
        "--out",      str(args.out),
        "--segments", args.segments,
        "--bin-width", str(args.bin_width),
    ]
    if args.limit:
        argv += ["--limit", str(args.limit)]
    if args.radiomics:
        argv.append("--radiomics")
    if args.no_append:
        argv.append("--no-append")
    if args.no_skip_done:
        argv.append("--no-skip-done")
    if args.no_prefer_processed:
        argv.append("--no-prefer-processed")

    print(f"  args: {' '.join(argv)}")
    rc = _load_script("quantification.py").main(argv)
    print(f"\nStage 3 finished in {_elapsed(t0)}  exit={rc}")
    return rc


# ── Argument parser ───────────────────────────────────────────────────────────

def _build_parser():
    import argparse

    p = argparse.ArgumentParser(
        description="KUPETCTMS master pipeline: segmentation -> postprocessing -> quantification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    p.add_argument("--root", type=Path, required=True,
                   help="Dataset root (CT/ PET/ Segments/)")
    p.add_argument("--out", type=Path, required=True,
                   help="Output Excel path for quantification")
    p.add_argument("--ckpt", type=Path, default=None,
                   help="SegResNet checkpoint (.ckpt). Required unless --skip-seg")

    # Stage control
    g = p.add_argument_group("Stage control")
    g.add_argument("--src", type=Path, default=None,
                   help="Stage 0: inbound raw folder (*__Studies or any DICOM tree). "
                        "Copied into --root CT/ PET/ before segmentation.")
    g.add_argument("--existing-map", type=Path, default=None,
                   help="CSV patient_id,subject_id to reuse MSP codes during organize")
    g.add_argument("--skip-seg",   action="store_true", help="Skip Stage 1 (generate_segments)")
    g.add_argument("--skip-post",  action="store_true", help="Skip Stage 2 (postprocessing)")
    g.add_argument("--skip-quant", action="store_true", help="Skip Stage 3 (quantification)")

    # Common
    p.add_argument("--limit",        type=int,   default=0,     help="Max subjects (0=all)")
    p.add_argument("--no-skip-done", action="store_true",        help="Force re-run all stages")
    p.add_argument("--no-append",    action="store_true",        help="Overwrite Excel")

    # Segmentation flags (forwarded verbatim to generate_segments.py)
    g2 = p.add_argument_group("Segmentation (Stage 1)")
    g2.add_argument("--device",          default="gpu",  help="gpu | cpu | 0 (default: gpu)")
    g2.add_argument("--cuda",            default=None,   help="Set CUDA_VISIBLE_DEVICES")
    g2.add_argument("--skip-ts",         action="store_true", help="Skip TotalSegmentator")
    g2.add_argument("--skip-vf",         action="store_true", help="Skip VF inference")
    g2.add_argument("--no-nrrd",         action="store_true", help="Skip .seg.nrrd packaging")
    g2.add_argument("--keep-loose",      action="store_true", help="Keep intermediate NIfTIs")
    g2.add_argument("--no-auto-orient",  action="store_true", help="Disable 4-flip search")

    # Post-processing flags (forwarded verbatim to postprocessing.py)
    g3 = p.add_argument_group("Post-processing (Stage 2)")
    g3.add_argument("--organs",
                    default="visceral_fat.nii.gz,iliopsoas_left.nii.gz,"
                            "iliopsoas_right.nii.gz,spleen.nii.gz",
                    help="Comma-separated organ filenames (default: VF+psoas+spleen)")
    g3.add_argument("--suv-thresh",        type=float, default=2.5,
                    help="PET SUV threshold for ureter mask (default: 2.5)")
    g3.add_argument("--ureter-dilate-mm",  type=float, default=18.0,
                    help="Ureter dilation radius mm (default: 18)")
    g3.add_argument("--ureter-ext-inf-mm", type=float, default=50.0,
                    help="Extend ureter below L5 mm (default: 50)")
    g3.add_argument("--group-subtract-mm", type=float, default=5.0,
                    help="Abdomen/vessels/spine subtract dilation mm (default: 5)")
    g3.add_argument("--clean-exclude-mm",  type=float, default=13.0,
                    help="Exclusion zone dilation for SUV clean mm (default: 13)")
    g3.add_argument("--suv-clean-fat",     type=float, default=1.2,
                    help="SUV threshold for fat clean (default: 1.2)")
    g3.add_argument("--suv-clean-psoas",   type=float, default=1.6,
                    help="SUV threshold for psoas clean (default: 1.6)")

    # Quantification flags (forwarded verbatim to quantification.py)
    g4 = p.add_argument_group("Quantification (Stage 3)")
    g4.add_argument("--segments",
                    default="visceral_fat,spleen,iliopsoas_left,iliopsoas_right",
                    help="Comma-separated stems to quantify (default: VF+spleen+psoas)")
    g4.add_argument("--radiomics",          action="store_true",
                    help="Extract PyRadiomics features (adds Radiomics sheet)")
    g4.add_argument("--bin-width",          type=float, default=0.25,
                    help="PyRadiomics bin width (default: 0.25)")
    g4.add_argument("--no-prefer-processed", action="store_true",
                    help="Use raw masks even when *_processed.nii.gz exists")

    return p


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    import os

    p = _build_parser()
    args = p.parse_args(argv)

    # Validate checkpoint
    if not args.skip_seg:
        if args.ckpt is None:
            try:
                from lib.models.segresnet import default_vf_checkpoint
                args.ckpt = default_vf_checkpoint()
            except Exception:
                pass
        if args.ckpt is None or not Path(args.ckpt).is_file():
            print("[ERROR] --ckpt not provided or file not found. "
                  "Pass --ckpt PATH or use --skip-seg.")
            return 1

    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    t_total = time.time()
    _banner("KUPETCTMS  Master Pipeline")
    print(f"  root    : {args.root}")
    print(f"  out     : {args.out}")
    print(f"  src     : {args.src or '(already organized)'}")
    print(f"  stages  : "
          + ("ORG " if args.src else "")
          + ("SEG " if not args.skip_seg  else "")
          + ("POST " if not args.skip_post else "")
          + ("QUANT" if not args.skip_quant else ""))
    print(f"  limit   : {args.limit or 'all'}")

    total_rc = 0

    if args.src:
        rc = run_organize(args)
        total_rc = max(total_rc, rc)
    else:
        print("\n[Stage 0 SKIPPED]  (no --src; assuming --root is already organized)")

    if not args.skip_seg:
        rc = run_segmentation(args)
        total_rc = max(total_rc, rc)
        if rc != 0:
            print("\n[Stage 1 FAILED] Skipping post + quant. Fix TotalSegmentator / masks first.")
            _banner("Pipeline complete")
            print(f"  Total wall time : {_elapsed(t_total)}")
            print(f"  Exit code       : {total_rc}")
            print()
            return total_rc
    else:
        print("\n[Stage 1 SKIPPED]")

    if not args.skip_post:
        rc = run_postprocessing(args)
        total_rc = max(total_rc, rc)
    else:
        print("\n[Stage 2 SKIPPED]")

    if not args.skip_quant:
        rc = run_quantification(args)
        total_rc = max(total_rc, rc)
    else:
        print("\n[Stage 3 SKIPPED]")

    _banner("Pipeline complete")
    print(f"  Total wall time : {_elapsed(t_total)}")
    print(f"  Exit code       : {total_rc}")
    print()
    return total_rc


if __name__ == "__main__":
    raise SystemExit(main())
