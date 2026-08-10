"""
Offline tests for orientation helpers + seg.nrrd packaging (no Slicer / no GPU).

    python tests/test_vf_orientation_and_segnrrd.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from lib.segmentation.orientation import best_xy_flip, dice_binary, apply_xy_flips
from lib.io.seg_nrrd import SegmentSpec, multilabel_from_segments, write_seg_nrrd


def test_best_flip_finds_x():
    ref = np.zeros((8, 8, 8), dtype=np.uint8)
    # Asymmetric blob so a pure X flip does not still overlap much
    ref[1:3, 3:6, 3:6] = 1
    wrong = np.flip(ref, axis=0).copy()
    assert dice_binary(wrong, ref) < 0.2
    fixed, info = best_xy_flip(wrong, ref, metric="dice")
    assert info["best_flip_x"] is True
    assert info["best_flip_y"] is False
    assert dice_binary(fixed, ref) > 0.99
    print("  PASS  best_xy_flip recovers X flip")


def test_best_flip_finds_xy():
    ref = np.zeros((8, 8, 8), dtype=np.uint8)
    ref[1:3, 1:3, 2:5] = 1
    wrong = apply_xy_flips(ref, True, True)
    fixed, info = best_xy_flip(wrong, ref, metric="dice")
    assert info["best_flip_x"] and info["best_flip_y"]
    assert dice_binary(fixed, ref) > 0.99
    print("  PASS  best_xy_flip recovers X+Y")


def test_seg_nrrd_roundtrip():
    try:
        import SimpleITK as sitk
    except ImportError:
        print("  SKIP  seg.nrrd (SimpleITK not installed)")
        return

    # Synthetic ref CT geometry
    arr = np.zeros((16, 16, 12), dtype=np.int16)  # ZYX for sitk
    ref = sitk.GetImageFromArray(arr)
    ref.SetSpacing((1.0, 1.0, 1.5))
    ref.SetOrigin((0.0, 0.0, 0.0))

    # XYZ masks
    psoas = np.zeros((16, 16, 12), dtype=np.uint8)
    psoas[4:8, 4:8, 3:9] = 1
    spleen = np.zeros((16, 16, 12), dtype=np.uint8)
    spleen[10:14, 10:14, 3:9] = 1

    segs = [
        SegmentSpec("psoas_left", psoas, 1, (1, 0, 0)),
        SegmentSpec("spleen", spleen, 2, (0, 1, 0)),
    ]
    ml = multilabel_from_segments(segs)
    assert set(np.unique(ml)) == {0, 1, 2}

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "target_organs.seg.nrrd"
        write_seg_nrrd(str(out), segs, ref)
        assert out.is_file()
        loaded = sitk.ReadImage(str(out))
        assert loaded.GetMetaData("Segment0_Name") == "psoas_left"
        assert loaded.GetMetaData("Segment1_Name") == "spleen"
        print(f"  PASS  seg.nrrd write/read ({out.stat().st_size} bytes)")


def main():
    print("Testing VF orientation + seg.nrrd helpers...\n")
    test_best_flip_finds_x()
    test_best_flip_finds_xy()
    test_seg_nrrd_roundtrip()
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
