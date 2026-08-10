"""
Test lib/segmentation/hotspots.py WITHOUT 3D Slicer.

Run from extension_new:
    python tests/test_hotspots.py

Optional real NIfTI (if you have files + nibabel):
    python tests/test_hotspots.py --pet path/to/pet.nii.gz --seg path/to/mask.nii.gz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# extension_new on path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib.segmentation.hotspots import find_hottest_voxels


def _approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


def test_empty_mask():
    pet = np.ones((3, 3, 3))
    lab = np.zeros((3, 3, 3))
    out = find_hottest_voxels(pet, lab, np.eye(4), top_n=5)
    assert out == [], f"expected [], got {out}"
    print("  PASS  empty mask → []")


def test_known_hotspots_identity_ras():
    """Plant two voxels; identity IJK→RAS means RAS == (x, y, z)."""
    pet = np.zeros((5, 5, 5), dtype=float)
    lab = np.zeros((5, 5, 5), dtype=float)

    # hottest: z=2, y=1, x=4  → SUV 12.0 → RAS (4, 1, 2)
    pet[2, 1, 4] = 12.0
    lab[2, 1, 4] = 1
    # second:  z=0, y=3, x=1  → SUV 7.5  → RAS (1, 3, 0)
    pet[0, 3, 1] = 7.5
    lab[0, 3, 1] = 1
    # outside mask — must be ignored even if hotter
    pet[4, 4, 4] = 99.0

    out = find_hottest_voxels(pet, lab, np.eye(4), top_n=2)
    assert len(out) == 2, out
    assert _approx(out[0]["suv"], 12.0), out[0]
    assert _approx(out[0]["ras_x"], 4.0) and _approx(out[0]["ras_y"], 1.0) and _approx(out[0]["ras_z"], 2.0)
    assert _approx(out[1]["suv"], 7.5), out[1]
    assert _approx(out[1]["ras_x"], 1.0) and _approx(out[1]["ras_y"], 3.0) and _approx(out[1]["ras_z"], 0.0)
    print("  PASS  known hotspots + identity RAS")
    print(f"        top1={out[0]}")
    print(f"        top2={out[1]}")


def test_top_n_clipping():
    pet = np.arange(8, dtype=float).reshape(2, 2, 2)  # 0..7
    lab = np.ones((2, 2, 2))
    out = find_hottest_voxels(pet, lab, np.eye(4), top_n=3)
    assert len(out) == 3
    assert [h["suv"] for h in out] == [7.0, 6.0, 5.0]
    print("  PASS  top_n clipping / sort order")


def test_shape_mismatch_raises():
    try:
        find_hottest_voxels(np.zeros((2, 2, 2)), np.zeros((3, 3, 3)), np.eye(4))
    except ValueError:
        print("  PASS  shape mismatch raises ValueError")
        return
    raise AssertionError("expected ValueError on shape mismatch")


def test_affine_scaling():
    """Non-identity IJK→RAS: scale I by 2 → ras_x = 2*x."""
    pet = np.zeros((2, 2, 2), dtype=float)
    lab = np.zeros((2, 2, 2), dtype=float)
    pet[0, 0, 1] = 5.0
    lab[0, 0, 1] = 1

    mat = np.eye(4)
    mat[0, 0] = 2.0  # I (x) spacing 2 mm
    out = find_hottest_voxels(pet, lab, mat, top_n=1)
    assert _approx(out[0]["ras_x"], 2.0), out[0]
    assert _approx(out[0]["ras_y"], 0.0) and _approx(out[0]["ras_z"], 0.0)
    print("  PASS  affine scaling IJK→RAS")


def test_real_nifti(pet_path: Path, seg_path: Path, top_n: int = 5):
    """Optional: push real files through the same lib function."""
    import nibabel as nib

    pet_img = nib.load(str(pet_path))
    seg_img = nib.load(str(seg_path))

    # nibabel is often (X,Y,Z); Slicer arrayFromVolume is (Z,Y,X).
    # Match Slicer convention for a fair pre-Slicer check:
    pet = np.asanyarray(pet_img.dataobj)
    seg = np.asanyarray(seg_img.dataobj)
    if pet.ndim != 3 or seg.ndim != 3:
        raise ValueError("Expect 3D NIfTI volumes")

    # If shapes match axis order already, use as-is; else try transpose to ZYX
    if pet.shape != seg.shape:
        raise ValueError(f"PET {pet.shape} vs seg {seg.shape} — resample/match first")

    # Prefer ZYX like Slicer if last dim is smallest (common for axial stacks)
    # User can force --zyx / --xyz; default: assume file is XYZ → transpose to ZYX
    pet_zyx = np.transpose(pet, (2, 1, 0))
    seg_zyx = np.transpose(seg, (2, 1, 0))

    # Build IJK→RAS from affine. nibabel affine maps voxel (i,j,k)=(x,y,z) → RAS.
    # After ZYX transpose, voxel index (z,y,x) corresponds to nibabel (x,y,z)=(i,j,k)
    # with i=x, j=y, k=z. Our lib expects mat @ [I,J,K,1] = [x,y,z,1] in RAS,
    # where I=x, J=y, K=z — so use the NIfTI affine directly.
    affine = np.asarray(pet_img.affine, dtype=float)

    # Mask: any positive label
    label = (seg_zyx > 0).astype(np.uint8)
    # pet still needs same axis order as label (ZYX). Values moved with transpose.
    out = find_hottest_voxels(pet_zyx, label, affine, top_n=top_n)

    print(f"  INFO  real NIfTI: {pet_path.name} + {seg_path.name}")
    print(f"        shape ZYX={pet_zyx.shape}, nonzero mask={int(label.sum())}")
    if not out:
        print("  WARN  no voxels in mask")
        return
    for i, h in enumerate(out, 1):
        print(
            f"        #{i}  SUV={h['suv']:.4f}  "
            f"RAS=({h['ras_x']:.1f}, {h['ras_y']:.1f}, {h['ras_z']:.1f})"
        )
    print("  PASS  real NIfTI ran without error (spot-check RAS in Slicer later)")


def main():
    parser = argparse.ArgumentParser(description="Test hotspot lib without Slicer")
    parser.add_argument("--pet", type=Path, help="Optional PET .nii.gz")
    parser.add_argument("--seg", type=Path, help="Optional segmentation/mask .nii.gz")
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    print("Testing lib.segmentation.hotspots (no Slicer)…\n")
    test_empty_mask()
    test_known_hotspots_identity_ras()
    test_top_n_clipping()
    test_shape_mismatch_raises()
    test_affine_scaling()

    if args.pet and args.seg:
        print()
        test_real_nifti(args.pet, args.seg, top_n=args.top_n)
    else:
        print("\n  (skip real NIfTI - pass --pet and --seg to try your own files)")

    print("\nAll synthetic checks passed.")
    print(
        "Note: Widget/Logic still need a quick click-test inside Slicer "
        "(node export + jump-to-slice)."
    )


if __name__ == "__main__":
    main()
