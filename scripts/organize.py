# -*- coding: utf-8 -*-
"""
organize.py
===========
Stage 0 — arrange inbound PET/CT folders into the pipeline layout.

Usual inbound (MIM / PACS)::

    2026-03__Studies/
      {uid}_CT_{YYYY-MM-DD}_{HHMMSS}_.../
      {uid}_PT_{YYYY-MM-DD}_{HHMMSS}_.../

Also accepted: a parent of several ``*__Studies`` batches, an already
organized ``CT/``+``PET/`` tree, or any nested DICOM folders.

Output::

    DEST/
      CT/   {MSPxxxx}_{YYYY-MM-DD}_CT/
      PET/  {MSPxxxx}_{YYYY-MM-DD}_PET/
      patient_id_mapping.csv
      scan_metadata.csv
      scan_mapping.xlsx     (Mapping + Metadata sheets)

Usage
-----
    python scripts/organize.py ^
        --src  "C:\\Users\\ishit\\Downloads\\2026-03__Studies" ^
        --dest "E:\\KUPETCTMS\\new_data_clean"

    # several monthly batches in one parent
    python scripts/organize.py ^
        --src  "C:\\Users\\ishit\\Downloads" ^
        --dest "E:\\KUPETCTMS\\new_data_clean"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib.io.organize import detect_input_layout, organize_dataset


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Organize raw PET/CT exports into CT/ PET/ pipeline folders"
    )
    p.add_argument("--src", type=Path, required=True,
                   help="Inbound folder (*__Studies, parent of batches, or any DICOM tree)")
    p.add_argument("--dest", type=Path, required=True,
                   help="Pipeline dataset root (CT/ PET/ written here)")
    p.add_argument("--existing-map", type=Path, default=None,
                   help="CSV with patient_id,subject_id to reuse MSP codes")
    p.add_argument("--prefix", default="MSP", help="Subject ID prefix (default: MSP)")
    p.add_argument("--no-skip-existing", action="store_true",
                   help="Re-copy even if dest series already has files")
    p.add_argument("--no-metadata", action="store_true",
                   help="Skip DICOM metadata CSV/Excel after copy")
    p.add_argument("--detect-only", action="store_true",
                   help="Print detected layout and exit (no copy)")
    args = p.parse_args(argv)

    if not args.src.is_dir():
        print(f"[ERROR] --src not found: {args.src}")
        return 1

    layout = detect_input_layout(args.src)
    print(f"Detected layout: {layout}")
    if args.detect_only:
        return 0

    result = organize_dataset(
        args.src,
        args.dest,
        existing_map=args.existing_map,
        subject_prefix=args.prefix,
        skip_existing=not args.no_skip_existing,
        write_metadata=not args.no_metadata,
    )
    print(f"\nStudies organized : {result['n_studies']}")
    print(f"Dest              : {result['dest']}")
    print(f"Subject map       : {result['map_csv']}")
    if result.get("metadata_csv"):
        print(f"Metadata CSV      : {result['metadata_csv']}")
    if result.get("mapping_xlsx"):
        print(f"Mapping Excel     : {result['mapping_xlsx']}")
    if result["warnings"]:
        print(f"Warnings          : {len(result['warnings'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
