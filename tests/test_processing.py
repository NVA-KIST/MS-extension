"""
Offline tests for lib/processing (no Slicer).

Run:
    python tests/test_processing.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from lib.processing.dilate import dilate_mask, resample_to_target, subtract_dilated_union
from lib.processing.mirroring import flip_volume_axis
from lib.processing.ureter import (
    apply_organ_processing,
    build_ureter_mask_from_pet,
    clip_organ_to_z,
    connect_ureter_path,
)


def test_dilate_grows():
    arr = np.zeros((21, 21, 21), dtype=np.uint8)
    arr[10, 10, 10] = 1
    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = 1.0  # 1 mm voxels
    out = dilate_mask(arr, affine, dilate_mm=2.0)
    assert out.sum() > 1
    assert out[10, 10, 10] == 1
    print("  PASS  dilate_mask grows")


def test_subtract():
    tgt = np.ones((5, 5, 5), dtype=np.uint8)
    src = np.zeros((5, 5, 5), dtype=np.uint8)
    src[2, 2, 2] = 1
    aff = np.eye(4)
    # dilate src a bit then subtract
    dilated = dilate_mask(src, aff, 1.0)
    result = subtract_dilated_union(tgt, aff, [(dilated, aff)])
    assert result.sum() < tgt.sum()
    assert result[2, 2, 2] == 0
    print("  PASS  subtract_dilated_union")


def test_resample_identity():
    src = np.zeros((4, 4, 4), dtype=np.uint8)
    src[1, 2, 3] = 1
    aff = np.eye(4)
    out = resample_to_target(src, aff, (4, 4, 4), aff)
    assert out[1, 2, 3] == 1
    print("  PASS  resample identity")


def test_flip():
    a = np.arange(8).reshape(2, 2, 2)
    f = flip_volume_axis(a, 0)
    assert f[0, 0, 0] == a[1, 0, 0]
    print("  PASS  flip_volume_axis")


def test_clip_z():
    arr = np.ones((5, 5, 5), dtype=np.uint8)
    aff = np.eye(4)  # RAS z == k index
    # keep only z in [1, 3]
    out = clip_organ_to_z(arr, aff, z_inferior=1.0, z_superior=3.0)
    # voxels with ras_z=0 and 4 should be 0
    assert out[0].sum() == 0
    assert out[4].sum() == 0
    assert out[2].sum() > 0
    print("  PASS  clip_organ_to_z")


def test_connect_path():
    mask = np.zeros((20, 10, 10), dtype=np.uint8)
    mask[2:4, 5, 5] = 1
    mask[10:12, 5, 5] = 1
    out = connect_ureter_path(mask, vox_size=(1, 1, 1), max_gap_mm=20, tube_radius_vox=1)
    assert out[6, 5, 5] == 1  # bridged middle
    print("  PASS  connect_ureter_path")


def test_build_ureter_smoke():
    pet = np.zeros((16, 24, 24), dtype=float)
    # bladder-like hot blob (largest CC) near bottom
    pet[2:5, 10:14, 10:14] = 8.0
    # ureter-like hot streak
    pet[8:12, 12, 12] = 5.0
    aff = np.eye(4)
    vox = (1.0, 1.0, 1.0)
    mask = build_ureter_mask_from_pet(
        pet,
        aff,
        vox,
        z_inferior=0.0,
        z_superior=15.0,
        suv_thresh=2.0,
        dilate_mm=1.0,
        torso_center_xy=(12.0, 12.0),
        ureter_z_inf=0.0,
        torso_radius_mm=50.0,
        connect_path=True,
        fill_holes=False,
    )
    assert mask.dtype == np.uint8
    assert mask.sum() >= 0  # smoke: runs without error
    print(f"  PASS  build_ureter_mask_from_pet (voxels={int(mask.sum())})")


def test_apply_processing_clip_only():
    arr = np.ones((5, 5, 5), dtype=np.uint8)
    aff = np.eye(4)
    out = apply_organ_processing(
        arr, aff, "Clip only", z_inferior=1.0, z_superior=3.0
    )
    assert out[0].sum() == 0
    print("  PASS  apply_organ_processing clip")


def main():
    print("Testing lib.processing (no Slicer)...\n")
    test_dilate_grows()
    test_subtract()
    test_resample_identity()
    test_flip()
    test_clip_z()
    test_connect_path()
    test_build_ureter_smoke()
    test_apply_processing_clip_only()
    print("\nAll processing checks passed.")


if __name__ == "__main__":
    main()
