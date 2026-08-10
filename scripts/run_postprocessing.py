"""
run_postprocessing.py
=====================
Batch CLI (no Slicer): PET ureter mask + organ clip/clean (+ optional kidney subtract).

Per-organ modes are selectable (not one global mode for every organ).

Expected layout::

    ROOT/
      PET/          {Subject}_{YYYY-MM-DD}_PET/     (DICOM)  and/or
      PET_NIfTI/    {Subject}_{YYYY-MM-DD}_PET.nii.gz
      Segments/     {Subject}_{YYYY-MM-DD}_Seg/     (inputs + outputs)

Usage
-----
    # List available organs in the dataset
    python scripts/postprocessing.py --root E:\\KUPETCTMS\\new_data_clean --list-organs

    # Interactive picker: choose organs + mode for each
    python scripts/postprocessing.py --root E:\\KUPETCTMS\\new_data_clean --interactive

    # Non-interactive per-organ modes
    python scripts/postprocessing.py --root E:\\KUPETCTMS\\new_data_clean ^
        --organ-mode "visceral_fat.nii.gz:Clip + Clean,spleen.nii.gz:Clip only"

    # Legacy: same mode for a fixed organ list
    python scripts/postprocessing.py --root E:\\KUPETCTMS\\new_data_clean ^
        --organs visceral_fat.nii.gz,spleen.nii.gz --mode "Clip + Clean"
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
    DEFAULT_ORGANS,
    ORGAN_MODES,
    format_organ_catalog,
    list_candidate_organs,
    load_pet_array,
    process_subject_seg_dir,
)


def _resolve_pet(root: Path, subject_key: str, log) -> tuple:
    nii = root / "PET_NIfTI" / f"{subject_key}_PET.nii.gz"
    if not nii.is_file():
        alt = root / "PET" / f"{subject_key}_PET.nii.gz"
        if alt.is_file():
            nii = alt
    dcm = root / "PET" / f"{subject_key}_PET"
    if nii.is_file():
        return load_pet_array(pet_path=nii, log=log)
    if dcm.is_dir():
        out = root / "PET_NIfTI" / f"{subject_key}_PET.nii.gz"
        return load_pet_array(pet_dicom_dir=dcm, pet_nii_out=out, log=log)
    raise FileNotFoundError(f"No PET for {subject_key}")


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
    """
    Parse ``file:mode,file:mode`` (mode may contain spaces / '+').
    Example: ``visceral_fat.nii.gz:Clip + Clean,spleen.nii.gz:Clip only``
    """
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


def interactive_pick_organ_modes(root: Path, *, include_aux: bool = False) -> dict[str, str]:
    """
    Terminal click-style picker:
      1) show numbered organ list
      2) user selects which organs (numbers / 'all' / 'defaults')
      3) for each selected organ, choose a mode
    """
    subjects = _subjects(root)
    n_subj = len(subjects)
    rows = list_candidate_organs(root, include_aux=include_aux, sample_only=True)
    if not rows:
        raise RuntimeError(f"No organ NIfTIs found under {root}/Segments")

    print()
    print(format_organ_catalog(rows, n_subjects=n_subj))
    print()
    print("Select target organs:")
    print("  - numbers separated by space/comma  (e.g. 1 3 4)")
    print("  - 'all'      = every listed organ")
    print("  - 'defaults' = visceral_fat + both psoas + spleen (if present)")
    print("  - 'q'        = quit")
    choice = input("Organs> ").strip().lower()
    if choice in ("q", "quit", "exit"):
        raise SystemExit(0)

    selected: list[dict] = []
    if choice in ("all", "*"):
        selected = list(rows)
    elif choice in ("defaults", "default", "d"):
        default_names = set(DEFAULT_ORGANS)
        selected = [r for r in rows if r["filename"] in default_names]
        if not selected:
            selected = [r for r in rows if r.get("is_default")]
    else:
        tokens = choice.replace(",", " ").split()
        idxs = []
        for t in tokens:
            if not t.isdigit():
                raise ValueError(f"Invalid organ selection token: {t!r}")
            idxs.append(int(t))
        for i in idxs:
            if i < 1 or i > len(rows):
                raise ValueError(f"Organ index out of range: {i}")
            selected.append(rows[i - 1])

    if not selected:
        raise RuntimeError("No organs selected")

    print()
    print("Modes:")
    for i, m in enumerate(ORGAN_MODES, 1):
        print(f"  {i}. {m}")
    print("  Enter a mode number, or press Enter for default [3 = Clip + Clean]")
    print("  Or type 'same' after the first organ to reuse that mode for the rest")

    organ_modes: dict[str, str] = {}
    last_mode = "Clip + Clean"
    use_same = False
    for r in selected:
        fname = r["filename"]
        if use_same:
            organ_modes[fname] = last_mode
            print(f"  {fname} → {last_mode}")
            continue
        raw = input(f"Mode for {fname} [{last_mode}]> ").strip()
        if raw.lower() in ("same", "s") and organ_modes:
            use_same = True
            organ_modes[fname] = last_mode
            print(f"  (same) {fname} → {last_mode}")
            continue
        if not raw:
            mode = last_mode
        elif raw.isdigit():
            mi = int(raw)
            if mi < 1 or mi > len(ORGAN_MODES):
                raise ValueError(f"Mode index out of range: {mi}")
            mode = ORGAN_MODES[mi - 1]
        else:
            # allow typing the mode string
            if raw not in ORGAN_MODES:
                raise ValueError(f"Unknown mode {raw!r}")
            mode = raw
        organ_modes[fname] = mode
        last_mode = mode
        print(f"  {fname} → {mode}")

    print()
    print("Summary:")
    for k, v in organ_modes.items():
        print(f"  {k}: {v}")
    confirm = input("Proceed? [Y/n]> ").strip().lower()
    if confirm in ("n", "no"):
        raise SystemExit(0)
    return organ_modes


def process_root(
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

    print(f"Post-processing {len(subjects)} subject(s)")
    for k, v in organ_modes.items():
        print(f"  {k}: {v}")

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
        description="Ureter clip/clean with per-organ mode selection"
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
        help="Include aux masks (vertebrae, combined, kidneys, …) in the list",
    )
    p.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive picker: list organs, choose each organ's mode",
    )
    p.add_argument(
        "--organ-mode",
        default="",
        help='Per-organ modes: "visceral_fat.nii.gz:Clip + Clean,spleen.nii.gz:Clip only"',
    )
    p.add_argument(
        "--organs",
        default="",
        help="Comma-separated filenames (used with --mode when --organ-mode omitted)",
    )
    p.add_argument(
        "--mode",
        default="Clip + Clean",
        choices=[m for m in ORGAN_MODES if m != "Skip"],
        help="Single mode applied to all --organs (legacy)",
    )
    p.add_argument("--suv-thresh", type=float, default=4.0)
    p.add_argument("--suv-clean", type=float, default=1.2)
    p.add_argument("--dilate-mm", type=float, default=5.0)
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

    if args.interactive:
        organ_modes = interactive_pick_organ_modes(
            args.root, include_aux=args.include_aux
        )
    elif args.organ_mode.strip():
        organ_modes = parse_organ_mode_spec(args.organ_mode)
    else:
        organs = [
            o.strip()
            for o in (args.organs or ",".join(DEFAULT_ORGANS)).split(",")
            if o.strip()
        ]
        # normalize stems → filenames
        norm = []
        for o in organs:
            if not (o.endswith(".nii.gz") or o.endswith(".nii")):
                o = o + ".nii.gz"
            norm.append(o)
        organ_modes = {o: args.mode for o in norm}

    if not organ_modes:
        print("No organs selected. Use --interactive, --organ-mode, or --organs.")
        return 1

    return process_root(
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


if __name__ == "__main__":
    raise SystemExit(main())
