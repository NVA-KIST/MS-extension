"""
run_postprocessing.py
=====================
Batch CLI (no Slicer): KU post-processing protocol.

Protocol
--------
1. Build PET ureter mask::
     SUV thresh 2.5, dilate 18 mm, extend 50 mm below L5, torso radius 220 mm,
     fill holes + bridge vertical gaps ON.
2. Dilate abdomen + vessels + spine by 5 mm and hard-subtract from targets.
3. Dilate ureter ∪ abdomen ∪ vessels ∪ spine by 13 mm; in overlap with
   visceral fat / psoas, remove voxels with PET SUV > 1.2 (fat) or > 1.6 (psoas).
4. Visceral fat is also clipped to L1–L5 Z.

Writes per Segments/<ID>_Seg/:
  ureter_from_pet.nii.gz
  <stem>_processed.nii.gz

Usage
-----
    python scripts/postprocessing.py --root E:\\KUPETCTMS\\temp sample --limit 1
    python scripts/postprocessing.py --root E:\\KUPETCTMS\\temp sample --no-skip-done
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib.io.logging_utils import PatientLog
from lib.io.paths import discover_patients, find_bulk_subjects
from lib.processing.postprocess import (
    CLEAN_EXCLUDE_DILATE_MM,
    DEFAULT_ORGANS,
    GROUP_SUBTRACT_DILATE_MM,
    ORGAN_MODES,
    SUV_CLEAN_FAT,
    SUV_CLEAN_PSOAS,
    URETER_DILATE_MM,
    URETER_EXT_INF_MM,
    URETER_SUV_THRESH,
    URETER_TORSO_RADIUS_MM,
    format_organ_catalog,
    list_candidate_organs,
    load_pet_array,
    process_subject_ku_protocol,
    process_subject_seg_dir,
)
from lib.quantification.pet_metrics import suvbw_factor_from_dicom_folder


def _resolve_pet(root: Path, subject_key: str, log) -> tuple:
    import numpy as np

    nii = root / "PET_NIfTI" / f"{subject_key}_PET.nii.gz"
    if not nii.is_file():
        alt = root / "PET" / f"{subject_key}_PET.nii.gz"
        if alt.is_file():
            nii = alt
    dcm = root / "PET" / f"{subject_key}_PET"
    if nii.is_file():
        pet_arr, pet_aff, ref = load_pet_array(pet_path=nii, log=log)
    elif dcm.is_dir():
        out = root / "PET_NIfTI" / f"{subject_key}_PET.nii.gz"
        pet_arr, pet_aff, ref = load_pet_array(
            pet_dicom_dir=dcm, pet_nii_out=out, log=log
        )
    else:
        raise FileNotFoundError(f"No PET for {subject_key}")

    # Convert Bq/mL → SUVbw when DICOM headers are available
    if dcm.is_dir():
        try:
            factor, meta = suvbw_factor_from_dicom_folder(str(dcm))
            if not meta.get("skipped"):
                pet_arr = (
                    np.asarray(pet_arr, dtype=np.float32) * float(factor)
                ).astype(np.float32)
                msg = (
                    f"[SUV] factor={factor:.6g} weight={meta.get('weight_kg')} kg "
                    f"max={float(np.max(pet_arr)):.4g}"
                )
                print(f"  {msg}")
                if hasattr(log, "info"):
                    log.info(msg)
        except Exception as e:
            print(f"  [SUV] WARN {e}")
    return pet_arr, pet_aff, ref


def _subjects(root: Path) -> list[dict]:
    subjects = find_bulk_subjects(str(root))
    if subjects:
        return subjects
    try:
        rows = discover_patients(str(root))
        return [
            {
                "subject_id": f"{r['subject_id']}_{r['scan_date']}",
                "seg_dir": r["seg_path"],
            }
            for r in rows
        ]
    except Exception:
        return []


def parse_organ_mode_spec(spec: str) -> dict[str, str]:
    """Parse ``file:mode,file:mode`` for legacy mode-based path."""
    out: dict[str, str] = {}
    if not spec or not spec.strip():
        return out
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if ":" not in raw:
            raise ValueError(
                f"Organ mode entry must be 'filename:mode', got: {raw!r}"
            )
        fname, mode = raw.split(":", 1)
        fname = fname.strip()
        mode = mode.strip()
        if mode not in ORGAN_MODES:
            raise ValueError(
                f"Unknown mode {mode!r} for {fname}. Choose from {list(ORGAN_MODES)}"
            )
        if not (fname.endswith(".nii.gz") or fname.endswith(".nii")):
            fname = fname + ".nii.gz"
        out[fname] = mode
    return out


def process_root_ku(
    root: Path,
    organs: list[str],
    *,
    skip_done: bool,
    limit: int = 0,
    ureter_suv_thresh: float = URETER_SUV_THRESH,
    ureter_dilate_mm: float = URETER_DILATE_MM,
    ureter_ext_inf_mm: float = URETER_EXT_INF_MM,
    group_subtract_dilate_mm: float = GROUP_SUBTRACT_DILATE_MM,
    clean_exclude_dilate_mm: float = CLEAN_EXCLUDE_DILATE_MM,
    suv_clean_fat: float = SUV_CLEAN_FAT,
    suv_clean_psoas: float = SUV_CLEAN_PSOAS,
) -> int:
    subjects = _subjects(root)
    if not subjects:
        print(f"No subjects under {root}/Segments")
        return 1
    if limit > 0:
        subjects = subjects[:limit]

    print(f"KU post-processing {len(subjects)} subject(s)")
    print(
        f"  ureter: SUV>{ureter_suv_thresh} dilate={ureter_dilate_mm}mm "
        f"extend_below_L5={ureter_ext_inf_mm}mm"
    )
    print(
        f"  subtract groups@{group_subtract_dilate_mm}mm; "
        f"clean excl@{clean_exclude_dilate_mm}mm "
        f"(fat SUV>{suv_clean_fat}, psoas SUV>{suv_clean_psoas})"
    )
    print(f"  organs: {', '.join(organs)}")

    ok = err = 0
    for i, sub in enumerate(subjects, 1):
        key = sub["subject_id"]
        seg_dir = Path(sub["seg_dir"])
        print(f"\n[{i}/{len(subjects)}] {key}")
        log = PatientLog(str(root / "pipeline_logs"), key)
        try:
            pet_arr, pet_aff, _ = _resolve_pet(root, key, log)
            pet_nii = root / "PET_NIfTI" / f"{key}_PET.nii.gz"
            if not pet_nii.is_file():
                pet_nii = root / "PET" / f"{key}_PET.nii.gz"
            info = process_subject_ku_protocol(
                seg_dir,
                pet_arr,
                pet_aff,
                organs=organs,
                pet_path=pet_nii if pet_nii.is_file() else None,
                ureter_suv_thresh=ureter_suv_thresh,
                ureter_dilate_mm=ureter_dilate_mm,
                ureter_ext_inf_mm=ureter_ext_inf_mm,
                torso_radius_mm=URETER_TORSO_RADIUS_MM,
                group_subtract_dilate_mm=group_subtract_dilate_mm,
                clean_exclude_dilate_mm=clean_exclude_dilate_mm,
                suv_clean_fat=suv_clean_fat,
                suv_clean_psoas=suv_clean_psoas,
                connect_path=True,
                fill_holes=True,
                skip_done=skip_done,
                log=log,
            )
            if info["errors"]:
                err += 1
            else:
                ok += 1
        except Exception as e:
            err += 1
            log.error(str(e))
            print(f"  [ERR] {e}")
            import traceback

            traceback.print_exc()
        finally:
            log.close()

    print(f"\nDone. ok={ok} errors={err}")
    return 0 if err == 0 else 2


def process_root_legacy(
    root: Path,
    organ_modes: dict[str, str],
    suv_thresh: float,
    suv_clean: float,
    dilate_mm: float,
    kidney_dilate_mm: float,
    subtract_kidneys: bool,
    skip_done: bool,
    limit: int = 0,
) -> int:
    subjects = _subjects(root)
    if not subjects:
        print(f"No subjects under {root}/Segments")
        return 1
    if limit > 0:
        subjects = subjects[:limit]

    print(f"Legacy post-processing {len(subjects)} subject(s)")
    ok = err = 0
    for i, sub in enumerate(subjects, 1):
        key = sub["subject_id"]
        seg_dir = Path(sub["seg_dir"])
        print(f"\n[{i}/{len(subjects)}] {key}")
        log = PatientLog(str(root / "pipeline_logs"), key)
        try:
            pet_arr, pet_aff, _ = _resolve_pet(root, key, log)
            info = process_subject_seg_dir(
                seg_dir,
                pet_arr,
                pet_aff,
                organ_modes=organ_modes,
                suv_thresh=suv_thresh,
                suv_clean_thresh=suv_clean,
                dilate_mm=dilate_mm,
                kidney_dilate_mm=kidney_dilate_mm,
                subtract_kidneys=subtract_kidneys,
                skip_done=skip_done,
                log=log,
            )
            if info["errors"]:
                err += 1
            else:
                ok += 1
        except Exception as e:
            err += 1
            log.error(str(e))
            print(f"  [ERR] {e}")
        finally:
            log.close()

    print(f"\nDone. ok={ok} errors={err}")
    return 0 if err == 0 else 2


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="KU post-processing: ureter + group subtract/clean"
    )
    p.add_argument("--root", type=Path, required=True, help="Dataset root")
    p.add_argument(
        "--list-organs",
        action="store_true",
        help="List available organ NIfTIs and exit",
    )
    p.add_argument(
        "--include-aux",
        action="store_true",
        help="Include aux masks in --list-organs",
    )
    p.add_argument(
        "--organs",
        default=",".join(DEFAULT_ORGANS),
        help="Comma-separated target organ filenames",
    )
    p.add_argument(
        "--legacy",
        action="store_true",
        help="Use old per-organ Clip/Clean modes instead of KU protocol",
    )
    p.add_argument(
        "--organ-mode",
        default="",
        help='Legacy only: "visceral_fat.nii.gz:Clip + Clean,..."',
    )
    p.add_argument("--mode", default="Clip + Clean", choices=list(ORGAN_MODES))
    p.add_argument("--suv-thresh", type=float, default=URETER_SUV_THRESH)
    p.add_argument("--ureter-dilate-mm", type=float, default=URETER_DILATE_MM)
    p.add_argument("--ureter-ext-inf-mm", type=float, default=URETER_EXT_INF_MM)
    p.add_argument(
        "--group-subtract-mm", type=float, default=GROUP_SUBTRACT_DILATE_MM
    )
    p.add_argument(
        "--clean-exclude-mm", type=float, default=CLEAN_EXCLUDE_DILATE_MM
    )
    p.add_argument("--suv-clean-fat", type=float, default=SUV_CLEAN_FAT)
    p.add_argument("--suv-clean-psoas", type=float, default=SUV_CLEAN_PSOAS)
    p.add_argument("--suv-clean", type=float, default=SUV_CLEAN_FAT, help="Legacy")
    p.add_argument("--dilate-mm", type=float, default=URETER_DILATE_MM, help="Legacy")
    p.add_argument("--kidney-dilate-mm", type=float, default=3.0)
    p.add_argument("--subtract-kidneys", action="store_true")
    p.add_argument("--no-skip-done", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args(argv)

    if args.list_organs:
        subjects = _subjects(args.root)
        rows = list_candidate_organs(
            args.root, include_aux=args.include_aux, sample_only=True
        )
        print(format_organ_catalog(rows, n_subjects=len(subjects)))
        return 0

    organs = [
        o.strip() if (o.strip().endswith(".nii.gz") or o.strip().endswith(".nii"))
        else o.strip() + ".nii.gz"
        for o in args.organs.split(",")
        if o.strip()
    ]

    if args.legacy or args.organ_mode.strip():
        if args.organ_mode.strip():
            organ_modes = parse_organ_mode_spec(args.organ_mode)
        else:
            organ_modes = {o: args.mode for o in organs}
        return process_root_legacy(
            args.root,
            organ_modes=organ_modes,
            suv_thresh=args.suv_thresh,
            suv_clean=args.suv_clean,
            dilate_mm=args.dilate_mm,
            kidney_dilate_mm=args.kidney_dilate_mm,
            subtract_kidneys=args.subtract_kidneys,
            skip_done=not args.no_skip_done,
            limit=args.limit,
        )

    return process_root_ku(
        args.root,
        organs=organs,
        skip_done=not args.no_skip_done,
        limit=args.limit,
        ureter_suv_thresh=args.suv_thresh,
        ureter_dilate_mm=args.ureter_dilate_mm,
        ureter_ext_inf_mm=args.ureter_ext_inf_mm,
        group_subtract_dilate_mm=args.group_subtract_mm,
        clean_exclude_dilate_mm=args.clean_exclude_mm,
        suv_clean_fat=args.suv_clean_fat,
        suv_clean_psoas=args.suv_clean_psoas,
    )


if __name__ == "__main__":
    raise SystemExit(main())
