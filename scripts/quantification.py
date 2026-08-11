"""
quantification.py
=================
Python-IDE / CLI entry for PET metrics (+ optional radiomics).

Inject a dataset root and output Excel path — no Slicer required::

    cd extension_new
    python scripts/quantification.py --root E:\\KUPETCTMS\\new_data_clean ^
        --out E:\\KUPETCTMS\\new_data_clean\\metrics.xlsx

    python scripts/quantification.py --root E:\\KUPETCTMS\\new_data_clean ^
        --out metrics.xlsx ^
        --segments visceral_fat,spleen,iliopsoas_left,iliopsoas_right ^
        --radiomics

With ``--radiomics``, the workbook gets two content sheets:
  - Quantification  (SUV mean/max/peak, volume, TLG)
  - Radiomics       (selected PyRadiomics features)
plus a Summary pivot of Quantification metrics.

Prefers ``*_processed.nii.gz`` masks when present (postprocessing output).

Slicer equivalent: PETBiomarkerStudio / PETCTQuantAnalysis batch
(same Quantification + Radiomics sheet layout).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib.quantification.pet_metrics import run_batch_quantification
from lib.quantification.radiomics import SELECTED_RADIOMICS_FEATURE_ORDER


DEFAULT_SEGMENTS = [
    "visceral_fat",
    "spleen",
    "iliopsoas_left",
    "iliopsoas_right",
]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Batch PET quantification → Excel")
    p.add_argument("--root", type=Path, required=True, help="Dataset root")
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output .xlsx path (Quantification + Radiomics sheets when --radiomics)",
    )
    p.add_argument(
        "--segments",
        default=",".join(DEFAULT_SEGMENTS),
        help="Comma-separated segment stems (no .nii.gz)",
    )
    p.add_argument("--no-append", action="store_true", help="Overwrite Excel")
    p.add_argument("--no-skip-done", action="store_true")
    p.add_argument("--no-prefer-processed", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument(
        "--radiomics",
        action="store_true",
        help="Also extract radiomics into a separate 'Radiomics' sheet",
    )
    p.add_argument("--bin-width", type=float, default=0.25)
    args = p.parse_args(argv)

    stems = [s.strip() for s in args.segments.split(",") if s.strip()]
    rad_opts = {}
    if args.radiomics:
        rad_opts = {
            "selected_feature_keys": list(SELECTED_RADIOMICS_FEATURE_ORDER),
            "bin_width": args.bin_width,
            "derived": True,
            # Large ROIs (VF) OOM at native PET resolution without this
            "resample_isotropic": True,
            "resampled_spacing_mm": 4.0,
            "auto_resample_large": True,
        }

    summary = run_batch_quantification(
        str(args.root),
        stems,
        str(args.out),
        radiomics_options=rad_opts,
        append=not args.no_append,
        skip_done=not args.no_skip_done,
        prefer_processed=not args.no_prefer_processed,
        limit=args.limit,
    )
    print(
        f"\nDone. processed={summary['processed']} skipped={summary['skipped']} "
        f"errors={summary['errors']} rows={summary['rowCount']}"
    )
    print(f"Excel: {summary['savedPath']}")
    if args.radiomics:
        print("Sheets: Quantification + Radiomics (+ Summary pivot)")
    else:
        print("Sheets: Quantification (+ Summary pivot)")
    return 0 if summary["errors"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
