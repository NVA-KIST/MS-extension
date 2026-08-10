"""
Unit tests for lib.processing.ablation (no Slicer).

Run:
    python tests/test_ablation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from lib.processing.ablation import (
    apply_urinary_cleanup,
    clip_binary_mask_by_ras_z,
    flag_qc_outliers,
    physical_dilation_structure,
)


def test_physical_dilation_radius_zero():
    struct = physical_dilation_structure(spacing=(1.0, 1.0, 1.0), radius_mm=0.0)
    assert struct.shape == (1, 1, 1)
    assert struct.dtype == bool
    assert struct[0, 0, 0]
    print("  PASS  physical_dilation_structure radius 0")


def test_clip_by_z():
    mask = np.ones((5, 3, 3), dtype=bool)
    affine = np.eye(4)  # RAS z == K index
    clipped, info = clip_binary_mask_by_ras_z(
        mask, affine, z_inferior=1.0, z_superior=3.0
    )
    assert info["originalVoxelCount"] == 5 * 3 * 3
    assert clipped[0].sum() == 0
    assert clipped[4].sum() == 0
    assert clipped[2].sum() == 3 * 3
    assert info["remainingVoxelCount"] == 3 * 3 * 3
    assert info["removedVoxelCount"] == 2 * 3 * 3
    print("  PASS  clip_binary_mask_by_ras_z")


def test_urinary_cleanup():
    roi = np.zeros((4, 4, 4), dtype=np.uint8)
    roi[1:3, 1:3, 1:3] = 1
    urinary = np.zeros_like(roi)
    urinary[2, 1:3, 1:3] = 1
    pet = np.zeros((4, 4, 4), dtype=float)
    pet[2, 1:3, 1:3] = 5.0

    cleaned, info = apply_urinary_cleanup(
        roi, urinary, pet, clean_suv_threshold=2.0
    )
    assert info["removedVoxelCount"] == 4
    assert cleaned[2, 1:3, 1:3].sum() == 0
    assert cleaned[1, 1:3, 1:3].sum() == 4
    assert info["remainingVoxelCount"] == 4
    print("  PASS  apply_urinary_cleanup")


def test_flag_qc_outliers():
    rows = [
        {"suv_max": 2.0, "ratio": 1.0},
        {"suv_max": 2.1, "ratio": 1.1},
        {"suv_max": 2.2, "ratio": 1.0},
        {"suv_max": 20.0, "ratio": 1.2},  # SUVmax outlier
        {"suv_max": 2.0, "ratio": 5.0},   # ratio outlier
    ]
    out = flag_qc_outliers(rows, mad_k=3.5, ratio_thresh=4.0)
    assert "SUVmax" in out[3]["flag"]
    assert "ratio" in out[4]["flag"]
    assert out[0]["flag"] == ""
    print("  PASS  flag_qc_outliers")


def main():
    print("test_ablation")
    test_physical_dilation_radius_zero()
    test_clip_by_z()
    test_urinary_cleanup()
    test_flag_qc_outliers()
    print("ALL PASSED")


if __name__ == "__main__":
    main()
