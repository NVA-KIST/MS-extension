"""CLI: spleen TotalSegmentator batch."""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import freeze_support
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from lib.io.paths import collect_ct_folders
from lib.segmentation.spleen import run_spleen_segmentation


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ct-root", type=Path, required=True)
    p.add_argument("--seg-root", type=Path, required=True)
    p.add_argument("--cuda", default="0")
    p.add_argument("--device", default="gpu")
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    folders = collect_ct_folders(args.ct_root)
    print(f"Found {len(folders)} CT folders")
    for i, folder in enumerate(folders, 1):
        print(f"\n[{i}/{len(folders)}] {folder.name}")
        out = run_spleen_segmentation(folder, seg_root=args.seg_root, device=args.device)
        for f in sorted(out.glob("spleen*.nii.gz")):
            print(" ", f.name)


if __name__ == "__main__":
    freeze_support()
    main()
