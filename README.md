# KUPETCTMS

Python library + CLI for a KU PET/CT cohort pipeline:

1. **Segmentation** — CT → TotalSegmentator → combined mask → visceral fat (SegResNet) → `.seg.nrrd` groups
2. **Post-processing** — PET ureter mask, 5 mm abdomen/vessels/spine subtract, 13 mm SUV clean
3. **Quantification** — SUVmean / max / peak, volume, TLG (+ optional PyRadiomics) → Excel

Slicer is **not** required for the library or CLI. Slicer modules under `slicer_modules/` are a separate GUI layer on the same `lib/`.

---

## Install and dependencies

When someone installs this package, **core dependencies are pulled automatically**. They do **not** need to run a separate `uv sync` after a published install.

| How they install | What happens |
|------------------|--------------|
| `pip install kupetctms` | pip reads `pyproject.toml` and installs numpy, scipy, nibabel, SimpleITK, pydicom, openpyxl |
| `uv add kupetctms` | same — uv resolves and installs declared deps |
| `pip install kupetctms[full]` | core + TotalSegmentator/torch/MONAI + pyradiomics |
| Local clone: `uv sync` | installs the project in editable mode from this folder |

`uv sync` is a **developer** command (lockfile + local env). End users of a published library just `pip install` / `uv add`.

### Local development (this repo)

This repo includes **`uv.lock`** + **`.python-version`** (3.11) so everyone gets the same resolved versions.

```bash
cd extension_new   # or repo root on the TEST branch
uv sync --frozen --extra full   # exact lockfile install (preferred after pull)
# first-time / refresh lock only when deps change:
#   uv lock
#   uv sync --extra full
# pip fallback (not locked):
#   pip install -e ".[full]"
```

### Optional extras

| Extra | Packages | Needed for |
|-------|----------|------------|
| *(none)* | numpy, scipy, nibabel, SimpleITK, pydicom, openpyxl | post-process + SUV metrics |
| `seg` | TotalSegmentator, torch, monai, lightning | Stage 1 segmentation + VF |
| `radiomics` | pyradiomics | Stage 3 radiomics sheet |
| `full` | `seg` + `radiomics` | entire pipeline |
| `dev` | pytest | tests |

```bash
pip install kupetctms[seg]
pip install kupetctms[radiomics]
pip install kupetctms[full]
```

### Not installed by pip (you still need these)

| Item | Why |
|------|-----|
| **VF checkpoint** `.ckpt` | SegResNet weights (pass `--ckpt`) |
| **TotalSegmentator licence** | `tissue_types` (torso fat) needs the academic licence |
| **CUDA / GPU** | optional; use `--device cpu` if none |
| **3D Slicer** | only if using the GUI modules |

---

## Dataset layout

**Pipeline root (after organize):**

```
ROOT/
  CT/          {Subject}_{YYYY-MM-DD}_CT/      DICOM
  PET/         {Subject}_{YYYY-MM-DD}_PET/     DICOM
  CT_NIfTI/    {Subject}_{YYYY-MM-DD}_CT.nii.gz   (created if missing)
  PET_NIfTI/   {Subject}_{YYYY-MM-DD}_PET.nii.gz  (created if missing)
  Segments/    {Subject}_{YYYY-MM-DD}_Seg/        outputs
  patient_id_mapping.csv
  scan_metadata.csv
  scan_mapping.xlsx
```

Folder names must match `{SubjectID}_{YYYY-MM-DD}_{CT|PET|Seg}`.

**Inbound (before organize)** is usually a MIM/PACS dump:

```
2026-03__Studies/
  {uid}_CT_{YYYY-MM-DD}_{HHMMSS}_Torso.PET-CT…/
  {uid}_PT_{YYYY-MM-DD}_{HHMMSS}_Torso.PET-CT…/
```

```bash
python scripts/organize.py \
  --src  /path/to/inbound \
  --dest /path/to/DATASET_ROOT
```

Also accepts: a parent of several `*__Studies` months, an already-organized `CT/`+`PET/` tree, or any nested DICOM folders (grouped by Study UID). Same hospital `PatientID` reuses the same `MSPxxxx`.

---

## CLI (batch)

Full flag-by-flag reference: **[docs/SCRIPTS.md](docs/SCRIPTS.md)**.

Run from the project root:

```bash
cd /path/to/extension_new
```

### Master pipeline (recommended)

**Already organized `CT/` + `PET/`:**

```bash
python scripts/run_pipeline.py \
  --root  /path/to/DATASET_ROOT \
  --ckpt  /path/to/epoch=399-step=8800.ckpt \
  --out   /path/to/DATASET_ROOT/metrics.xlsx \
  --device gpu \
  --radiomics
```

**Raw inbound dump → organize + full pipeline:**

```bash
python scripts/run_pipeline.py \
  --src   /path/to/inbound \
  --root  /path/to/DATASET_ROOT \
  --ckpt  /path/to/epoch=399-step=8800.ckpt \
  --out   /path/to/DATASET_ROOT/metrics.xlsx \
  --radiomics
```

| Flag | Meaning |
|------|---------|
| `--root` | Dataset folder with `CT/` and `PET/` |
| `--ckpt` | Visceral-fat SegResNet checkpoint |
| `--out` | Excel output path |
| `--src` | Optional inbound dump to organize into `--root` first |
| `--device` | `gpu`, `cpu`, or device index like `0` |
| `--radiomics` | Also extract PyRadiomics features |
| `--limit 1` | Process only the first subject (smoke test) |
| `--no-skip-done` | Force re-run even if outputs exist |
| `--no-append` | Overwrite Excel instead of appending |
| `--skip-seg` / `--skip-post` / `--skip-quant` | Skip a stage |

Equivalent to `organize.py` (if `--src`) → `generate_segments.py` → `postprocessing.py` → `quantification.py`.

Default checkpoint in this repo:

```text
/path/to/extension_new/lib/models/epoch=399-step=8800.ckpt
```

**Note:** TotalSegmentator can take a long time and may print little until it finishes. That is expected.

### Individual stages

```bash
# 0. Organize
python scripts/organize.py \
  --src  /path/to/inbound \
  --dest /path/to/DATASET_ROOT

# 1. Segmentation (TS + VF + nrrd)
python scripts/generate_segments.py \
  --root   /path/to/DATASET_ROOT \
  --ckpt   /path/to/epoch=399-step=8800.ckpt \
  --device gpu

# 2. Post-processing
python scripts/postprocessing.py \
  --root /path/to/DATASET_ROOT \
  --organs "visceral_fat.nii.gz,iliopsoas_left.nii.gz,iliopsoas_right.nii.gz,spleen.nii.gz"

# 3. Quantification
python scripts/quantification.py \
  --root     /path/to/DATASET_ROOT \
  --out      /path/to/DATASET_ROOT/metrics.xlsx \
  --segments visceral_fat,spleen,iliopsoas_left,iliopsoas_right \
  --radiomics
```

```bash
python scripts/run_pipeline.py --help
```

---

## Library usage (Python)

After install, import from `lib` (current package layout):

```python
from pathlib import Path
from lib.io.organize import organize_dataset
from lib.io.metadata import extract_dataset_metadata, extract_dicom_metadata, save_metadata
from lib.segmentation.totalseg import run_totalseg_for_visceral_fat
from lib.segmentation.visceral_fat import build_combined_mask, predict_visceral_fat
from lib.io.seg_nrrd import package_patient_segmentations
from lib.processing.postprocess import load_pet_array, process_subject_ku_protocol
from lib.quantification.pet_metrics import (
    suvbw_factor_from_dicom_folder,
    run_batch_quantification,
)

raw  = Path("/path/to/inbound")
root = Path("/path/to/DATASET_ROOT")
# Checkpoint path (from lib.models.segresnet.default_vf_checkpoint())
ckpt = Path("lib/models/epoch=399-step=8800.ckpt").resolve()

# 0. Organize inbound dump + extract demographics
organize_dataset(raw, root)
rows = extract_dataset_metadata(root)          # age, sex, weight, height, BMI, dose, …
save_metadata(rows, csv_path=root / "scan_metadata.csv")
# one series folder:
extract_dicom_metadata(root / "PET" / "SUBJECT_YYYY-MM-DD_PET")

seg  = root / "Segments" / "SUBJECT_YYYY-MM-DD_Seg"
ct   = root / "CT_NIfTI" / "SUBJECT_YYYY-MM-DD_CT.nii.gz"

# 1. Segmentation
run_totalseg_for_visceral_fat(ct, seg, device="gpu")
build_combined_mask(str(seg), str(seg / "combined_mask.nii.gz"))
predict_visceral_fat(str(ct), str(seg / "combined_mask.nii.gz"), str(ckpt),
                     str(seg / "visceral_fat.nii.gz"))
package_patient_segmentations(str(seg), str(ct))

# 2. Post-processing (PET as SUVbw)
pet_arr, pet_aff, _ = load_pet_array(pet_path=root / "PET_NIfTI" / "SUBJECT_YYYY-MM-DD_PET.nii.gz")
dcm = root / "PET" / "SUBJECT_YYYY-MM-DD_PET"
factor, meta = suvbw_factor_from_dicom_folder(str(dcm))
if not meta.get("skipped"):
    pet_arr = pet_arr.astype("float32") * factor

process_subject_ku_protocol(seg, pet_arr, pet_aff, skip_done=False)

# 3. Quantification
run_batch_quantification(
    str(root),
    ["visceral_fat", "spleen", "iliopsoas_left", "iliopsoas_right"],
    str(root / "metrics.xlsx"),
    radiomics_options={"selected_feature_keys": ["p10", "p90", "entropy"],
                       "bin_width": 0.25, "resample_isotropic": True},
    prefer_processed=True,
)
```

Full function reference: **[docs/API.md](docs/API.md)**.  
Full CLI / scripts reference: **[docs/SCRIPTS.md](docs/SCRIPTS.md)**.

---

## Array convention

Processing functions use **ZYX** voxel order (SimpleITK / Slicer style).  
NIfTI files on disk are **XYZ** (nibabel). Loaders in `lib.processing.postprocess` transpose automatically.

Affines are 4×4 matrices mapping index `(i, j, k) = (X, Y, Z)` → RAS mm.

---

## Package map

| Package | What it contains |
|---------|------------------|
| `lib.segmentation` | TotalSeg, VF inference, combined mask, orientation, vessels |
| `lib.processing` | Dilate/resample/subtract, ureter, KU protocol |
| `lib.quantification` | SUV metrics, radiomics, Excel batch export |
| `lib.io` | Organize inbound dumps, patient metadata, DICOM→NIfTI, `.seg.nrrd` |
| `lib.models` | SegResNet + default checkpoint path |
| `scripts/` | CLI entry points (not imported as a library) — see [docs/SCRIPTS.md](docs/SCRIPTS.md) |
| `slicer_modules/` | 3D Slicer GUI (optional) |

---

## Publishing note

Today internal imports are `from lib....`. After a public rename you would use `from kupetctms....` and keep a compatibility shim. Do not rename until you are ready to update every import and Slicer adapter.
