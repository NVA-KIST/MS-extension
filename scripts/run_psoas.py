"""CLI: iliopsoas TotalSegmentator."""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import freeze_support
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from lib.io.paths import choose_ct_file, collect_ct_scans
from lib.segmentation.psoas import run_iliopsoas_segmentation


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--cuda", default="0")
    p.add_argument("--device", default="gpu")
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    cts = collect_ct_scans(args.root)
    print(f"Found {len(cts)} CT files")
    ct = choose_ct_file(cts, args.index)
    out = run_iliopsoas_segmentation(ct, device=args.device)
    print("Done:", out)
    for f in sorted(out.glob("*.nii.gz")):
        print(" ", f.name)


if __name__ == "__main__":
    freeze_support()
    main()
