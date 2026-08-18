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

```bash
cd extension_new
uv sync --extra full          # or: pip install -e ".[full]"
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
  --src  "C:\Users\ishit\Downloads\2026-03__Studies" \
  --dest "E:\KUPETCTMS\new_data_clean"
```

Also accepts: a parent of several `*__Studies` months, an already-organized `CT/`+`PET/` tree, or any nested DICOM folders (grouped by Study UID). Same hospital `PatientID` reuses the same `MSPxxxx`.

---

## CLI (batch)

```bash
# inbound dump -> organize + full pipeline (uses default checkpoint)
python scripts/run_pipeline.py \
  --src   "C:/Users/ishit/Downloads/2026-03__Studies" \
  --root  "E:/data/new_data_clean" \
  --out   "E:/data/new_data_clean/metrics.xlsx" \
  --radiomics

# already organized CT/ PET/ (uses default checkpoint from lib/models/)
python scripts/run_pipeline.py \
  --root  "E:/data/new_data_clean" \
  --out   "E:/data/new_data_clean/metrics.xlsx" \
  --radiomics

# or specify custom checkpoint
python scripts/run_pipeline.py \
  --root  "E:/data/new_data_clean" \
  --ckpt  "path/to/custom/checkpoint.ckpt" \
  --out   "E:/data/new_data_clean/metrics.xlsx"
```

Equivalent to `organize.py` (if `--src`) → `generate_segments.py` → `postprocessing.py` → `quantification.py`.

Skip stages with `--skip-seg` / `--skip-post` / `--skip-quant`.  
Force re-run with `--no-skip-done`.  
One subject: `--limit 1`.

See `python scripts/run_pipeline.py --help` for all flags (same names as the three individual scripts).

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

raw  = Path(r"C:/Users/ishit/Downloads/2026-03__Studies")
root = Path(r"E:/data/new_data_clean")
# Checkpoint path (from lib.models.segresnet.default_vf_checkpoint())
ckpt = Path(r"lib/models/epoch=399-step=8800.ckpt").resolve()

# 0. Organize inbound dump + extract demographics
organize_dataset(raw, root)
rows = extract_dataset_metadata(root)          # age, sex, weight, height, BMI, dose, …
save_metadata(rows, csv_path=root / "scan_metadata.csv")
# one series folder:
extract_dicom_metadata(root / "PET" / "MSP0002_2026-05-20_PET")

seg  = root / "Segments" / "MSP0002_2026-05-20_Seg"
ct   = root / "CT_NIfTI" / "MSP0002_2026-05-20_CT.nii.gz"

# 1. Segmentation
run_totalseg_for_visceral_fat(ct, seg, device="gpu")
build_combined_mask(str(seg), str(seg / "combined_mask.nii.gz"))
predict_visceral_fat(str(ct), str(seg / "combined_mask.nii.gz"), str(ckpt),
                     str(seg / "visceral_fat.nii.gz"))
package_patient_segmentations(str(seg), str(ct))

# 2. Post-processing (PET as SUVbw)
pet_arr, pet_aff, _ = load_pet_array(pet_path=root / "PET_NIfTI" / "MSP0002_2026-05-20_PET.nii.gz")
dcm = root / "PET" / "MSP0002_2026-05-20_PET"
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
| `scripts/` | CLI entry points (not imported as a library) |
| `slicer_modules/` | 3D Slicer GUI (optional) |

---

## 3D Slicer setup (important)

Slicer only auto-loads **top-level `*.py` module files** in an Additional Module Path.
Use this exact folder (not `extension_new`, not the old flat `extension/` tree):

```
E:\KUPETCTMS\extension\extension_new\slicer_modules
```

Steps:
1. **Edit → Application Settings → Modules → Additional module paths → Add**
2. Select `...\extension_new\slicer_modules`
3. **Restart Slicer** (required after adding/changing paths)
4. Open **Modules → Metabolic Syndrome Toolkit**

You should see Module Launcher, numbered clinical modules, Scribble Tool, PET Biomarker Studio.

If a module is missing, check **Application Settings → Modules** for a “Failed to load” / ignored list and clear it, then restart.

---

## Publishing note

Today internal imports are `from lib....`. After a public rename you would use `from kupetctms....` and keep a compatibility shim. Do not rename until you are ready to update every import and Slicer adapter.
