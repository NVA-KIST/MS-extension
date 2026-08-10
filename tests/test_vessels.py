"""Smoke tests for vessel array helpers used by VesselSegmenterLogic."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from lib.segmentation.vessels import (
    apply_distance_constraint,
    grow_vessels_from_seeds,
)


def test_grow_simple_blob():
    pet = np.zeros((20, 30, 30), dtype=np.float32)
    pet[5:15, 10:20, 10:20] = 1.5
    seeds = [(10, 15, 15)]
    out = grow_vessels_from_seeds(
        pet,
        spacing_mm=(2.0, 2.0, 2.0),
        seeds_zyx=seeds,
        suv_min=0.8,
        suv_max=4.0,
        max_extent_mm=50.0,
        closing_radius_mm=0.0,
        min_volume_ml=0.1,
        stitch=False,
    )
    assert "Vessel_1" in out
    assert out["Vessel_1"].sum() > 0


def test_distance_constraint_keeps_seed():
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask[2:8, 2:8, 2:8] = 1
    clipped = apply_distance_constraint(mask, (5, 5, 5), (1, 1, 1), max_extent_mm=2.0)
    assert clipped[5, 5, 5] == 1
    assert clipped.sum() < mask.sum()


if __name__ == "__main__":
    test_grow_simple_blob()
    test_distance_constraint_keeps_seed()
    print("ALL PASSED")
