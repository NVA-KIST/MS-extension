# CLI scripts reference

Batch entry points under `scripts/`. Run all commands from the `extension_new` project root (or any directory after `pip install -e .`).

```bash
cd /path/to/extension_new
```

Activate a Python environment that has the needed extras (`seg` for TotalSegmentator + VF, `radiomics` for PyRadiomics). See the main [README](../README.md) for install.

---

## Paths you will set

| Placeholder | Meaning |
|-------------|---------|
| `/path/to/extension_new` | This project folder |
| `/path/to/DATASET_ROOT` | Organized dataset with `CT/` and `PET/` |
| `/path/to/inbound` | Raw hospital dump (`*__Studies` or nested DICOM) |
| `/path/to/epoch=399-step=8800.ckpt` | SegResNet visceral-fat checkpoint |
| `/path/to/DATASET_ROOT/metrics.xlsx` | Quantification Excel output |

Default checkpoint location in this repo:

```text
/path/to/extension_new/lib/models/epoch=399-step=8800.ckpt
```

---

## Script map

| Script | Stage | What it does |
|--------|-------|----------------|
| `organize.py` | 0 | Inbound DICOM / `__Studies` → `DATASET_ROOT` (`CT/`, `PET/`, mapping CSVs) |
| `generate_segments.py` | 1 | CT→NIfTI, TotalSegmentator, combined mask, visceral fat, optional `.seg.nrrd` |
| `run_visceralfat.py` | 1 | Same as `generate_segments.py` (implementation entry) |
| `postprocessing.py` | 2 | Thin wrapper → `run_postprocessing.py` |
| `run_postprocessing.py` | 2 | Ureter mask, organ clip/clean (KU protocol) |
| `quantification.py` | 3 | SUV metrics (+ optional radiomics) → Excel |
| `run_pipeline.py` | 0–3 | Master runner: optional organize → seg → post → quant |

---

## 1. Master pipeline — `run_pipeline.py` (recommended)

Runs: optional organize → segmentation → post-processing → quantification.

### A. Already organized `CT/` + `PET/`

```bash
python scripts/run_pipeline.py \
  --root  /path/to/DATASET_ROOT \
  --ckpt  /path/to/epoch=399-step=8800.ckpt \
  --out   /path/to/DATASET_ROOT/metrics.xlsx \
  --device gpu \
  --radiomics
```

| Flag | Meaning |
|------|---------|
| `--root` | Dataset folder with `CT/` and `PET/` |
| `--ckpt` | Visceral-fat model checkpoint (required unless `--skip-seg`) |
| `--out` | Excel output path |
| `--device` | `gpu`, `cpu`, or device index like `0` |
| `--radiomics` | Also extract PyRadiomics features |

### B. Raw inbound dump (organize + pipeline)

```bash
python scripts/run_pipeline.py \
  --src   /path/to/inbound \
  --root  /path/to/DATASET_ROOT \
  --ckpt  /path/to/epoch=399-step=8800.ckpt \
  --out   /path/to/DATASET_ROOT/metrics.xlsx \
  --radiomics
```

`--src` is copied/organized into `--root`, then the full pipeline runs.

### Common options

```bash
--limit 1              # only first subject (smoke test)
--no-skip-done         # recompute even if outputs exist
--no-append            # overwrite Excel instead of appending
--skip-seg             # skip TotalSeg + VF (masks already exist)
--skip-post            # skip ureter/clean post-processing
--skip-quant           # skip Excel quantification
--skip-ts              # within seg: skip TotalSegmentator only
--skip-vf              # within seg: skip visceral-fat model only
```

### Smoke test (one patient)

```bash
python scripts/run_pipeline.py \
  --root  /path/to/DATASET_ROOT \
  --ckpt  /path/to/epoch=399-step=8800.ckpt \
  --out   /path/to/DATASET_ROOT/metrics.xlsx \
  --device gpu \
  --radiomics \
  --limit 1 \
  --no-append
```

### Typical workflows

| Situation | Flags to add |
|-----------|----------------|
| First full run on organized data | (defaults above) |
| Quick test on 1 patient | `--limit 1` |
| Segmentation done; only post + quant | `--skip-seg` |
| Force everything again | `--no-skip-done --no-append` |
| No GPU | `--device cpu` (much slower) |

**Note:** TotalSegmentator can take a long time and may print little until it finishes. That is expected.

```bash
python scripts/run_pipeline.py --help
```

---

## 2. Organize — `organize.py`

```bash
python scripts/organize.py \
  --src  /path/to/inbound \
  --dest /path/to/DATASET_ROOT
```

| Flag | Meaning |
|------|---------|
| `--src` | Inbound raw folder |
| `--dest` | Destination dataset root (`CT/`, `PET/`, …) |
| `--existing-map` | Optional CSV to reuse subject IDs |
| `--prefix` | Subject ID prefix (default `MSP`) |
| `--no-skip-existing` | Re-copy even if subject folders exist |
| `--detect-only` | Dry-run style detection without writing |
| `--no-metadata` | Skip writing scan metadata |

```bash
python scripts/organize.py --help
```

---

## 3. Segmentation — `generate_segments.py`

DICOM→NIfTI (if needed), TotalSegmentator, combined mask, visceral fat, optional `.seg.nrrd`.

```bash
python scripts/generate_segments.py \
  --root   /path/to/DATASET_ROOT \
  --ckpt   /path/to/epoch=399-step=8800.ckpt \
  --device gpu
```

| Flag | Meaning |
|------|---------|
| `--root` | Dataset root |
| `--ckpt` | VF checkpoint |
| `--device` | `gpu` / `cpu` / device index |
| `--limit N` | Process at most N subjects |
| `--skip-ts` | Skip TotalSegmentator |
| `--skip-vf` | Skip visceral-fat inference |
| `--no-nrrd` | Skip `.seg.nrrd` packaging |
| `--no-auto-orient` | Disable VF auto L/R·A/P flip search |
| `--keep-loose` | Keep intermediate NIfTIs |
| `--cuda` | Set `CUDA_VISIBLE_DEVICES` |

Single-CT mode (optional):

```bash
python scripts/generate_segments.py \
  --ct /path/to/CT.nii.gz \
  --seg-dir /path/to/output_Seg \
  --ckpt /path/to/epoch=399-step=8800.ckpt \
  --device gpu
```

```bash
python scripts/generate_segments.py --help
```

---

## 4. Post-processing — `postprocessing.py`

PET-guided ureter mask, L1–L5 clip, organ clean (writes `*_processed.nii.gz` under each Seg folder).

```bash
python scripts/postprocessing.py \
  --root /path/to/DATASET_ROOT \
  --organs "visceral_fat.nii.gz,iliopsoas_left.nii.gz,iliopsoas_right.nii.gz,spleen.nii.gz"
```

| Flag | Meaning |
|------|---------|
| `--root` | Dataset root |
| `--organs` | Comma-separated organ mask filenames |
| `--limit N` | Max subjects |
| `--no-skip-done` | Force re-run |
| `--suv-thresh` | PET SUV threshold for ureter (default `2.5`) |
| `--ureter-dilate-mm` | Ureter dilation mm (default `18`) |
| `--suv-clean-fat` | SUV clean threshold for fat (default `1.2`) |
| `--suv-clean-psoas` | SUV clean threshold for psoas (default `1.6`) |

```bash
python scripts/postprocessing.py --help
```

---

## 5. Quantification — `quantification.py`

Batch SUV metrics → Excel; optional PyRadiomics sheet.

```bash
python scripts/quantification.py \
  --root     /path/to/DATASET_ROOT \
  --out      /path/to/DATASET_ROOT/metrics.xlsx \
  --segments visceral_fat,spleen,iliopsoas_left,iliopsoas_right \
  --radiomics
```

| Flag | Meaning |
|------|---------|
| `--root` | Dataset root |
| `--out` | Excel path |
| `--segments` | Comma-separated mask stems (no `.nii.gz`) |
| `--radiomics` | Enable PyRadiomics |
| `--bin-width` | Radiomics bin width (default `0.25`) |
| `--no-append` | Overwrite Excel |
| `--no-skip-done` | Force re-run |
| `--no-prefer-processed` | Use raw masks even if `*_processed.nii.gz` exists |
| `--limit N` | Max subjects |

```bash
python scripts/quantification.py --help
```

---

## End-to-end examples

**Organized data, full run**

```bash
cd /path/to/extension_new

python scripts/run_pipeline.py \
  --root  /path/to/DATASET_ROOT \
  --ckpt  /path/to/extension_new/lib/models/epoch=399-step=8800.ckpt \
  --out   /path/to/DATASET_ROOT/metrics.xlsx \
  --device gpu \
  --radiomics
```

**Inbound dump → organize → full run**

```bash
python scripts/run_pipeline.py \
  --src   /path/to/inbound \
  --root  /path/to/DATASET_ROOT \
  --ckpt  /path/to/extension_new/lib/models/epoch=399-step=8800.ckpt \
  --out   /path/to/DATASET_ROOT/metrics.xlsx \
  --radiomics
```

**Seg already finished; post + quant only**

```bash
python scripts/run_pipeline.py \
  --root  /path/to/DATASET_ROOT \
  --ckpt  /path/to/extension_new/lib/models/epoch=399-step=8800.ckpt \
  --out   /path/to/DATASET_ROOT/metrics.xlsx \
  --skip-seg \
  --radiomics \
  --no-append
```

**Stage-by-stage**

```bash
python scripts/organize.py \
  --src  /path/to/inbound \
  --dest /path/to/DATASET_ROOT

python scripts/generate_segments.py \
  --root /path/to/DATASET_ROOT \
  --ckpt /path/to/epoch=399-step=8800.ckpt \
  --device gpu

python scripts/postprocessing.py \
  --root /path/to/DATASET_ROOT \
  --organs "visceral_fat.nii.gz,iliopsoas_left.nii.gz,iliopsoas_right.nii.gz,spleen.nii.gz"

python scripts/quantification.py \
  --root /path/to/DATASET_ROOT \
  --out  /path/to/DATASET_ROOT/metrics.xlsx \
  --segments visceral_fat,spleen,iliopsoas_left,iliopsoas_right \
  --radiomics
```
