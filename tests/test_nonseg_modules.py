"""
Offline tests for non-segmentation libs (no Slicer).

Run from extension_new:
    python tests/test_nonseg_modules.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from lib.quantification.distance import format_distance, ras_distance_to_voxels
from lib.quantification.pet_metrics import compute_suvbw_factor, error_row, save_excel
from lib.io.paths import detect_scans, detect_segmentations, parse_folder_name


def test_distance():
    mat = np.eye(4)
    d = ras_distance_to_voxels((0, 0, 0), (3, 4, 0), mat)
    assert abs(d - 5.0) < 1e-9, d
    assert format_distance(12.345, "mm") == "12.35 mm"
    assert format_distance(10.0, "cm") == "1.000 cm"
    assert "vox" in format_distance(1.0, "voxels", 2.5)
    print("  PASS  distance")


def test_suv_factor_admin():
    # ADMIN: no decay → factor = weight_g / dose
    f = compute_suvbw_factor(
        weight_kg=70.0,
        dose_bq=1e8,
        injection_time="120000",
        acquisition_time="123000",
        half_life_s=6586.2,
        decay_correction="ADMIN",
    )
    assert abs(f - (70000.0 / 1e8)) < 1e-12, f
    print("  PASS  suvbw factor (ADMIN)")


def test_parse_and_detect(tmp_path: Path):
    assert parse_folder_name("MSP0001_2025-07-09_PET") == ("MSP0001", "2025-07-09", "PET")
    assert parse_folder_name("badname") is None

    pet = tmp_path / "PET" / "MSP0001_2025-07-09_PET"
    seg = tmp_path / "Segments" / "MSP0001_2025-07-09_Seg"
    ct = tmp_path / "CT" / "MSP0001_2025-07-09_CT"
    pet.mkdir(parents=True)
    seg.mkdir(parents=True)
    ct.mkdir(parents=True)
    (seg / "liver.nii.gz").write_bytes(b"x")
    (seg / "spleen.nii.gz").write_bytes(b"x")

    scans = detect_scans(str(tmp_path))
    assert len(scans) == 1
    assert scans[0]["subject_id"] == "MSP0001"
    assert scans[0]["ct_path"] is not None
    assert scans[0]["seg_path"] is not None

    segs = detect_segmentations(str(tmp_path), scans)
    assert segs["liver"]["count"] == 1
    assert segs["spleen"]["count"] == 1
    assert segs["__total__"] == 1
    print("  PASS  detect_scans / detect_segmentations")


def test_excel(tmp_path: Path):
    out = tmp_path / "out.xlsx"
    rows = [
        {
            **error_row("MSP0001", "2025-07-09", "liver", "done", "P1"),
            "source_file": "liver.nii.gz",
            "suv_mean": 1.2,
            "suv_max": 3.4,
            "suv_peak": 2.1,
            "tlg": 5.0,
            "volume_mL": 10.0,
        }
    ]
    try:
        save_excel(rows, str(out), append=False)
    except ImportError:
        print("  SKIP  excel (openpyxl not installed)")
        return
    assert out.is_file()
    print("  PASS  save_excel")


def main():
    print("Testing non-segmentation libs (no Slicer)...\n")
    test_distance()
    test_suv_factor_admin()
    with tempfile.TemporaryDirectory() as td:
        test_parse_and_detect(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_excel(Path(td))
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
