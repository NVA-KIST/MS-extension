# KUPETCTMS API reference

Public functions intended for library users. Private helpers (`_name`) are omitted.

**Import prefix:** `from lib.<package> import <fn>`

**Array order:** ZYX unless noted. **Affine:** 4×4, index `(X,Y,Z)` → RAS mm.

---

## 1. Segmentation — `lib.segmentation`

### `run_totalseg_for_visceral_fat`

```python
from lib.segmentation.totalseg import run_totalseg_for_visceral_fat

run_totalseg_for_visceral_fat(
    ct_path,          # Path | str  CT NIfTI
    out_dir,          # Path | str  Segments/<ID>_Seg/
    *,
    device="gpu",     # "gpu" | "cpu" | "0"
    include_targets=True,   # psoas L/R + spleen
    include_vessels=True,
    include_abdomen=True,
    use_api=True,
    log=None,
) -> Path            # out_dir
```

**Does:** TotalSegmentator `total` (ROI subset) + `body` + `tissue_types`.  
**Writes:** organ NIfTIs in `out_dir` (`liver.nii.gz`, `spleen.nii.gz`, vertebrae, vessels, `body_trunc`, `torso_fat`, …).  
**Needs extra:** `totalsegmentator`, GPU optional. `tissue_types` needs TS academic licence.

---

### `build_combined_mask`

```python
from lib.segmentation.visceral_fat import build_combined_mask

build_combined_mask(seg_dir: str, out_path: str) -> None
```

**Input:** folder with TotalSeg NIfTIs (liver, heart, vertebrae, ribs, body, torso fat).  
**Output:** `combined_mask.nii.gz` (multi-label: organs / fat / liver channels used by VF).

---

### `predict_visceral_fat`

```python
from lib.segmentation.visceral_fat import predict_visceral_fat

predict_visceral_fat(
    ct_path: str,
    combined_path: str,
    ckpt_path: str,          # Lightning .ckpt
    out_path: str,           # visceral_fat.nii.gz
    device="gpu",
    auto_orient=True,        # 4-flip search vs torso_fat
    orient_metric="dice",
    sw_batch_size=4,
    overlap=0.10,
    patch_size=(96, 96, 96),
    log=None,
) -> dict
```

**Input:** CT NIfTI + combined mask + checkpoint.  
**Output file:** `visceral_fat.nii.gz` on the **original CT grid**.  
**Return dict:** `out_path`, flip/orientation diagnostics.  
**Needs extra:** `torch`, `monai`, `lightning`.

---

### `package_patient_segmentations`

```python
from lib.io.seg_nrrd import package_patient_segmentations

package_patient_segmentations(
    seg_dir: str,
    ref_ct: str,                 # CT NIfTI (geometry reference)
    out_dir=None,                # default: seg_dir
    include_combined=True,
    include_targets=False,       # VF / spleen / psoas stay as .nii.gz
    include_vessels=True,
    include_spine=True,
    include_abdomen=True,
    cleanup_loose=True,          # delete packed intermediate NIfTIs
) -> dict[str, str]              # group name → written path
```

**Writes:** `abdomen.seg.nrrd`, `vessels.seg.nrrd`, `spine.seg.nrrd`, `combined_mask.seg.nrrd`.

---

### Other segmentation helpers

| Function | Input | Output |
|----------|--------|--------|
| `run_totalsegmentator_api(ct, out, **kwargs)` | CT path, out dir | None (writes NIfTIs) |
| `best_xy_flip(pred, ref)` | two binary arrays | `(flip_x, flip_y, score)` |
| `dice_binary(a, b)` | two arrays | float Dice |
| `grow_vessels_from_seeds(...)` | CT + seed masks | grown vessel mask (ZYX) |
| `find_hottest_voxels(pet, mask, n=…)` | PET + mask | list of voxel indices |

---

## 2. Post-processing — `lib.processing`

### `load_pet_array`

```python
from lib.processing.postprocess import load_pet_array

arr_zyx, affine, ref_img = load_pet_array(
    pet_path=None,          # existing PET NIfTI
    pet_dicom_dir=None,     # or DICOM folder (converted via SimpleITK)
    pet_nii_out=None,       # where to write converted NIfTI
    log=None,
)
```

**Output:** `(np.ndarray ZYX, 4×4 affine, nibabel image)`.  
Units are whatever is in the file (often Bq/mL). Convert to SUVbw before the KU protocol.

---

### `suvbw_factor_from_dicom_folder`

```python
from lib.quantification.pet_metrics import suvbw_factor_from_dicom_folder

factor, meta = suvbw_factor_from_dicom_folder(dicom_folder: str)
# SUV = Bq/mL * factor
# meta: weight_kg, skipped, …
```

---

### `process_subject_ku_protocol`

```python
from lib.processing.postprocess import process_subject_ku_protocol

info = process_subject_ku_protocol(
    seg_dir,                 # Segments/<ID>_Seg
    pet_arr,                 # ZYX, preferably SUVbw
    pet_affine,
    *,
    organs=("visceral_fat.nii.gz", "iliopsoas_left.nii.gz",
            "iliopsoas_right.nii.gz", "spleen.nii.gz"),
    pet_path=None,           # PET NIfTI path (for SimpleITK resample)
    ureter_suv_thresh=2.5,
    ureter_dilate_mm=18.0,
    ureter_ext_inf_mm=50.0,
    torso_radius_mm=220.0,
    group_subtract_dilate_mm=5.0,
    clean_exclude_dilate_mm=13.0,
    suv_clean_fat=1.2,
    suv_clean_psoas=1.6,
    connect_path=True,
    fill_holes=True,
    skip_done=True,
    write_ureter=True,
    log=None,
) -> dict   # {seg_dir, organs, skipped, errors}
```

**Requires in `seg_dir`:** target organ NIfTIs + `abdomen.seg.nrrd` / `vessels.seg.nrrd` / `spine.seg.nrrd` (L1–L5).  
**Writes:** `ureter_from_pet.nii.gz`, `<stem>_processed.nii.gz`.

Protocol:

1. PET ureter mask (SUV, dilate, extend below L5, torso cylinder)
2. Hard-subtract 5 mm dilated abdomen ∪ vessels ∪ spine (spleen excluded from its own abdomen union)
3. 13 mm dilated exclusion; drop fat voxels with SUV > 1.2 and psoas > 1.6
4. VF clipped to L1–L5 Z

---

### Dilate / resample / subtract

```python
from lib.processing.dilate import dilate_mask, resample_to_target, subtract_dilated_union

dilate_mask(arr, affine, dilate_mm: float) -> np.ndarray
# Ellipsoidal binary dilation in mm. Input/output ZYX uint8.

resample_to_target(src_arr, src_affine, tgt_shape, tgt_affine) -> np.ndarray
# Nearest-neighbour resample onto target grid.

subtract_dilated_union(
    tgt_arr, tgt_affine,
    dilated_items,                 # list of (arr, affine)
    same_grid_items=None,          # list of arrays already on tgt grid
) -> np.ndarray
# Zeroes target voxels overlapping the union of sources.
```

---

### Ureter building blocks

```python
from lib.processing.ureter import (
    build_ureter_mask_from_pet,
    clip_organ_to_z,
    clean_organ_with_ureter,
    z_bounds_from_mask,
)

build_ureter_mask_from_pet(
    pet_arr, pet_affine, vox_size,   # vox_size = (sz, sy, sx) mm
    z_inferior, z_superior,          # RAS Z mm (L5..L1)
    suv_thresh, dilate_mm,
    torso_center_xy,                 # (cx, cy) RAS mm
    ureter_ext_inf_mm=50.0,
    torso_radius_mm=220.0,
    connect_path=True,
    fill_holes=True,
) -> np.ndarray   # ZYX uint8

clip_organ_to_z(organ_arr, affine, z_inf, z_sup) -> np.ndarray
z_bounds_from_mask(mask_arr, affine) -> (z_min, z_max)
```

---

### Discovery

```python
from lib.io.paths import discover_patients, find_bulk_subjects, detect_scans

discover_patients(root: str) -> list[dict]
# [{subject_id, scan_date, ct_path, pet_path, seg_path, ct_nii?}, ...]

find_bulk_subjects(dataset_root: str) -> list[dict]
# [{subject_id, seg_dir}, ...]   from ROOT/Segments/*_Seg

detect_scans(root_folder: str) -> list[dict]
# from ROOT/PET/  (+ matching CT/ and Segments/ if present)
```

---

## 3. Quantification — `lib.quantification`

### `compute_segment_metrics`

```python
from lib.quantification.pet_metrics import compute_segment_metrics

compute_segment_metrics(
    pet_arr, mask_arr, spacing_mm,   # same shape; spacing 3-tuple mm
    metrics=None,                    # default: mean, max, peak, tlg, volume
) -> dict
# volume_mL, suv_mean, suv_max, suv_peak, tlg
```

PET and mask must already be on the same grid. Peak = mean of 3×3×3 around max voxel. TLG = mean × volume_mL.

---

### `run_batch_quantification`

```python
from lib.quantification.pet_metrics import run_batch_quantification

run_batch_quantification(
    root: str,
    segment_stems: list[str],        # e.g. ["visceral_fat", "spleen", ...]
    output_file: str,                # .xlsx
    *,
    metrics=None,
    radiomics_options=None,          # {} or see below
    append=True,
    skip_done=True,
    prefer_processed=True,           # use *_processed.nii.gz if present
    limit=0,
    log=None,
) -> dict
# processed, skipped, errors, rowCount, savedPath
```

**Excel sheets:** `Quantification` (always), `Radiomics` (if options enabled), `Summary` pivot.

**`radiomics_options` example:**

```python
{
    "selected_feature_keys": ["p10", "p90", "entropy", "skewness",
                              "contrast", "sahgle", "lalgle", "zone_entropy"],
    "bin_width": 0.25,
    "resample_isotropic": True,
    "resampled_spacing_mm": 4.0,
    "auto_resample_large": True,
}
```

---

### `extract_radiomics_from_paths`

```python
from lib.quantification.radiomics import extract_radiomics_from_paths

extract_radiomics_from_paths(
    image_path: str,     # PET NIfTI (SUVbw recommended)
    mask_path: str,      # binary / label NIfTI
    radiomics_options: dict,
    label=1,
) -> dict               # rad_* keys + optional derived_* + metadata
```

**Needs extra:** `pyradiomics`. Large ROIs auto-resample to avoid MemoryError.

---

### `compute_suvbw_factor`

```python
compute_suvbw_factor(
    weight_kg, dose_bq, injection_time, acquisition_time, half_life_s,
    decay_correction="START",   # or "ADMIN"
) -> float
```

---

### Distance (optional)

```python
from lib.quantification.distance import euclidean, ras_distance_to_voxels, format_distance

euclidean(p1, p2) -> float
ras_distance_to_voxels(p1, p2, spacing_mm) -> (dx, dy, dz, dist_mm)
format_distance(...) -> str
```

---

## 4. Organize inbound data — `lib.io.organize`

Raw PACS/MIM dumps are **not** `CT/` + `PET/`. Call this first (or `python scripts/organize.py`).

### Accepted input layouts (auto-detected)

| Layout | Example |
|--------|---------|
| `studies` | `2026-03__Studies/{uid}_CT_2026-03-18_…/` and matching `_PT_` series |
| `studies_parent` | a folder that contains several `*__Studies` batches |
| `pipeline` | already `ROOT/CT/{ID}_{date}_CT/` + `ROOT/PET/…` |
| `dicom_tree` | any nested folders; grouped by DICOM `StudyInstanceUID` + modality |

### `organize_dataset`

```python
from lib.io.organize import organize_dataset, detect_input_layout

detect_input_layout(src) -> str   # "studies" | "studies_parent" | "pipeline" | "dicom_tree"

result = organize_dataset(
    src,                          # e.g. r"C:\Users\…\2026-03__Studies"
    dest,                         # pipeline root, e.g. r"E:\KUPETCTMS\new_data_clean"
    *,
    existing_map=None,            # CSV patient_id,subject_id to reuse MSP codes
    subject_prefix="MSP",
    skip_existing=True,           # do not re-copy if dest series already populated
    write_metadata=True,          # also write scan_metadata.csv + scan_mapping.xlsx
    log=None,
) -> dict
```

**Writes under `dest`:**

```
CT/{MSPxxxx}_{YYYY-MM-DD}_CT/
PET/{MSPxxxx}_{YYYY-MM-DD}_PET/
patient_id_mapping.csv
scan_metadata.csv
scan_mapping.xlsx          # sheets: Mapping, Metadata
```

**Return dict:** `layout`, `n_studies`, `mapping_rows`, `warnings`, `dest`, `map_csv`, `metadata_csv`, `mapping_xlsx`.

Files are **copied** (source left untouched). Same `PatientID` reuses the same `MSPxxxx` (from `dest/patient_id_mapping.csv` or `--existing-map`).

```bash
python scripts/organize.py \
  --src  "C:\Users\ishit\Downloads\2026-03__Studies" \
  --dest "E:\KUPETCTMS\new_data_clean"
```

---

## 5. Patient metadata — `lib.io.metadata`

### `extract_dicom_metadata`

```python
from lib.io.metadata import extract_dicom_metadata

extract_dicom_metadata(path) -> dict
# path = one .dcm file OR a series folder
```

**Fields (when present in the header):**

| Group | Keys |
|-------|------|
| Identity | `patient_id`, `patient_name`, `patient_sex`, `patient_age_y`, `patient_birth_date` |
| Body | `weight_kg`, `height_m`, `bmi` |
| Study | `study_date`, `study_time`, `accession_number`, `study_uid`, `study_description`, `series_description` |
| Scanner | `modality`, `manufacturer`, `model`, `station`, `institution` |
| Image | `rows`, `columns`, `slice_thickness_mm`, `pixel_spacing`, `kvp` |
| PET dose | `radiopharmaceutical`, `injected_dose_MBq`, `radionuclide_half_life_s`, `radiopharm_start_time`, `pet_units`, `decay_correction` |
| Times | `acquisition_date`, `acquisition_time` |

`patient_age_y` parses DICOM `064Y` / `011M`. `PatientSize` > 3 is treated as centimetres and converted to metres.

### `extract_dataset_metadata`

```python
from lib.io.metadata import extract_dataset_metadata, save_metadata

rows = extract_dataset_metadata(root, prefer_pet=True)
# one row per scan under ROOT/CT + ROOT/PET (PET header preferred)

save_metadata(rows, csv_path="scan_metadata.csv", xlsx_path="scan_mapping.xlsx")
```

Also adds longitudinal fields per subject: `scan_number_for_subject`, `n_scans_for_patient`, `weight_delta_prev_kg`, `days_since_prev`, `weight_delta_baseline_kg`, `weight_pct_change_baseline`.

---

## 6. Other I/O helpers — `lib.io`

| Function | Input | Output |
|----------|--------|--------|
| `convert_dicom_to_nifti(dicom_dir, out_nii)` | DICOM folder | writes NIfTI |
| `write_seg_nrrd(path, segments, ref_sitk)` | `SegmentSpec` list + SimpleITK ref | `.seg.nrrd` path |
| `load_mask_xyz(path, ref_sitk=None)` | mask file | XYZ uint8 array |
| `default_vf_checkpoint()` | — | `Path` to bundled `.ckpt` |

---

## 7. Default KU constants

```python
from lib.processing.postprocess import (
    URETER_SUV_THRESH,          # 2.5
    URETER_DILATE_MM,           # 18.0
    URETER_EXT_INF_MM,          # 50.0
    URETER_TORSO_RADIUS_MM,     # 220.0
    GROUP_SUBTRACT_DILATE_MM,   # 5.0
    CLEAN_EXCLUDE_DILATE_MM,    # 13.0
    SUV_CLEAN_FAT,              # 1.2
    SUV_CLEAN_PSOAS,            # 1.6
    DEFAULT_ORGANS,             # VF, psoas L/R, spleen
)
```

---

## 8. What is *not* part of the published library API

- `slicer_modules/*` — 3D Slicer UI (requires Slicer, qt, vtk)
- `scripts/*` — CLI wrappers; call them from a shell, do not import as API
- Any function whose name starts with `_`
