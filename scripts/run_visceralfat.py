"""
run_visceralfat.py
==================
Batch CLI: TotalSegmentator → combined_mask → SegResNet VF → .seg.nrrd packages.

Expected dataset layout (same as the Slicer pipeline)::

    ROOT/
      CT/            {Subject}_{YYYY-MM-DD}_CT/   (DICOM)  OR
      CT_NIfTI/      {Subject}_{YYYY-MM-DD}_CT.nii.gz
      Segments/      {Subject}_{YYYY-MM-DD}_Seg/   (outputs written here)

Usage
-----
    cd extension_new
    python scripts/run_visceralfat.py --root D:\\data\\dataset_clean ^
        --ckpt E:\\KUPETCTMS\\extension\\models\\epoch=399-step=8800.ckpt

    # single CT NIfTI smoke test
    python scripts/run_visceralfat.py --ct path\\to\\ct.nii.gz --seg-dir path\\to\\out \\
        --ckpt path\\to\\ckpt
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make extension_new importable as package root
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib.io.paths import discover_patients
from lib.io.seg_nrrd import package_patient_segmentations
from lib.segmentation.totalseg import run_totalseg_for_visceral_fat
from lib.segmentation.visceral_fat import build_combined_mask, predict_visceral_fat


def _ensure_ct_nii(patient: dict, root: Path, log_print) -> Path | None:
    """Return path to CT NIfTI, converting DICOM with SimpleITK if needed."""
    if patient.get("ct_nii") and Path(patient["ct_nii"]).is_file():
        return Path(patient["ct_nii"])

    ct_path = patient.get("ct_path")
    if not ct_path or not Path(ct_path).is_dir():
        log_print(f"  [SKIP] no CT for {patient['subject_id']}")
        return None

    # Prefer existing nii inside CT folder
    hits = list(Path(ct_path).rglob("*.nii.gz"))
    if hits:
        return hits[0]

    out_dir = root / "CT_NIfTI"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_nii = out_dir / f"{patient['subject_id']}_{patient['scan_date']}_CT.nii.gz"
    if out_nii.is_file():
        return out_nii

    try:
        import SimpleITK as sitk

        log_print(f"  Converting DICOM → NIfTI: {ct_path}")
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(str(ct_path))
        if not series_ids:
            # nested folders
            for sub in Path(ct_path).rglob("*"):
                if sub.is_dir():
                    series_ids = reader.GetGDCMSeriesIDs(str(sub))
                    if series_ids:
                        ct_path = str(sub)
                        break
        if not series_ids:
            log_print("  [SKIP] no DICOM series found")
            return None
        files = reader.GetGDCMSeriesFileNames(str(ct_path), series_ids[0])
        reader.SetFileNames(files)
        img = reader.Execute()
        sitk.WriteImage(img, str(out_nii))
        return out_nii
    except Exception as e:
        log_print(f"  [SKIP] DICOM convert failed: {e}")
        return None


def process_one(
    ct_nii: Path,
    seg_dir: Path,
    ckpt: Path,
    *,
    device: str = "gpu",
    skip_ts: bool = False,
    skip_vf: bool = False,
    package_nrrd: bool = True,
    cleanup_loose: bool = True,
    auto_orient: bool = True,
) -> dict:
    seg_dir.mkdir(parents=True, exist_ok=True)
    result = {"ct": str(ct_nii), "seg_dir": str(seg_dir)}

    if not skip_ts:
        print(f"\n=== TotalSegmentator → {seg_dir} ===")
        run_totalseg_for_visceral_fat(
            ct_nii,
            seg_dir,
            device=device,
            include_targets=True,
            include_vessels=True,
        )
    else:
        print("[TS] skipped")

    combined = seg_dir / "combined_mask.nii.gz"
    print(f"\n=== Combined mask → {combined} ===")
    build_combined_mask(str(seg_dir), str(combined))
    result["combined"] = str(combined)

    vf_out = seg_dir / "visceral_fat.nii.gz"
    if not skip_vf:
        print(f"\n=== VF prediction → {vf_out} ===")
        info = predict_visceral_fat(
            ct_path=str(ct_nii),
            combined_path=str(combined),
            ckpt_path=str(ckpt),
            out_path=str(vf_out),
            device=device,
            auto_orient=auto_orient,
        )
        result["vf"] = info
    else:
        print("[VF] skipped")

    if package_nrrd:
        print("\n=== Packaging .seg.nrrd ===")
        written = package_patient_segmentations(
            str(seg_dir), str(ct_nii), cleanup_loose=cleanup_loose
        )
        result["seg_nrrd"] = written
        for k, v in written.items():
            if k == "cleaned_loose_files":
                print(f"  cleaned loose intermediates: {v}")
            else:
                print(f"  {k}: {v}")

    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run TotalSeg + VF + seg.nrrd packaging")
    p.add_argument("--root", type=Path, help="Dataset root (CT/ PET/ Segments/)")
    p.add_argument("--ct", type=Path, help="Single CT NIfTI (overrides --root batch)")
    p.add_argument("--seg-dir", type=Path, help="Output Segments folder for --ct mode")
    p.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="SegResNet checkpoint (default: extension/models/epoch=399-step=8800.ckpt)",
    )
    p.add_argument("--device", default="gpu", help="gpu | cpu | 0")
    p.add_argument("--cuda", default=None, help="Set CUDA_VISIBLE_DEVICES")
    p.add_argument("--skip-ts", action="store_true", help="Skip TotalSegmentator")
    p.add_argument("--skip-vf", action="store_true", help="Skip VF inference")
    p.add_argument("--no-nrrd", action="store_true", help="Skip .seg.nrrd packaging")
    p.add_argument(
        "--keep-loose",
        action="store_true",
        help="Keep intermediate anatomy/body NIfTIs after packaging (default: delete them)",
    )
    p.add_argument("--no-auto-orient", action="store_true", help="Disable 4-flip search")
    p.add_argument("--limit", type=int, default=0, help="Process at most N patients")
    args = p.parse_args(argv)

    if args.cuda is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    if args.ckpt is None:
        from lib.models.segresnet import default_vf_checkpoint

        args.ckpt = default_vf_checkpoint()
    if not args.ckpt.is_file():
        print(f"Checkpoint not found: {args.ckpt}")
        return 1

    if args.ct:
        if not args.seg_dir:
            print("--seg-dir required with --ct")
            return 1
        process_one(
            args.ct,
            args.seg_dir,
            args.ckpt,
            device=args.device,
            skip_ts=args.skip_ts,
            skip_vf=args.skip_vf,
            package_nrrd=not args.no_nrrd,
            cleanup_loose=not args.keep_loose,
            auto_orient=not args.no_auto_orient,
        )
        print("\nDone.")
        return 0

    if not args.root:
        print("Provide --root or --ct/--seg-dir")
        return 1

    patients = discover_patients(str(args.root))
    print(f"Found {len(patients)} patient-scan(s) under {args.root}")
    if args.limit > 0:
        patients = patients[: args.limit]

    for i, pat in enumerate(patients, 1):
        key = f"{pat['subject_id']}_{pat['scan_date']}"
        print(f"\n######## [{i}/{len(patients)}] {key} ########")
        ct_nii = _ensure_ct_nii(pat, args.root, print)
        if ct_nii is None:
            continue
        seg_dir = Path(pat["seg_path"])
        try:
            process_one(
                ct_nii,
                seg_dir,
                args.ckpt,
                device=args.device,
                skip_ts=args.skip_ts,
                skip_vf=args.skip_vf,
                package_nrrd=not args.no_nrrd,
                cleanup_loose=not args.keep_loose,
                auto_orient=not args.no_auto_orient,
            )
        except Exception as e:
            print(f"  [ERROR] {key}: {e}")
            import traceback

            traceback.print_exc()

    print("\nBatch finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
