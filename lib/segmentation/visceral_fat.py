"""
Visceral-fat SegResNet inference (lib) with correct orientation save.

Fixes
-----
1. Reorient RAS prediction back onto the original CT grid before saving
   (prevents L/R / A/P mismatch vs torso_fat and CT).
2. Optional auto X/Y flip search (4 combos) maximising Dice/MI vs torso_fat
   (combined label 2).
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

from lib.segmentation.orientation import (
    best_xy_flip,
    metatensor_affine,
    reorient_ras_pred_to_reference,
)


def _fallback_log():
    class _L:
        def info(self, m):  print(f"[VF][INFO ] {m}")
        def ok(self, m):    print(f"[VF][OK   ] {m}")
        def warn(self, m):  print(f"[VF][WARN ] {m}")
        def error(self, m): print(f"[VF][ERROR] {m}")
        def sep(self, lbl=""): print(f"[VF] --- {lbl}")
    return _L()


def _import_segresnet(log):
    """Import SegResNet from lib.models.segresnet (loads segresnet_full.py)."""
    try:
        from lib.models.segresnet import SegResNet, default_vf_checkpoint

        ckpt = default_vf_checkpoint()
        log.info(f"SegResNet from lib.models.segresnet (ckpt: {ckpt})")
        return SegResNet
    except Exception as e:
        log.warn(f"lib.models.segresnet unavailable ({e}); trying fallback")

    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "models" / "segresnet_full.py",  # lib/models/segresnet_full.py
        here.parents[3] / "segresnet.py",  # fallback: extension/segresnet.py (old location)
    ]
    for path in candidates:
        if path.is_file() and path.stat().st_size > 1000:
            import importlib.util

            spec = importlib.util.spec_from_file_location("segresnet_mod", str(path))
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)
                if hasattr(mod, "SegResNet"):
                    log.info(f"SegResNet from {path}")
                    return mod.SegResNet
            except Exception as err:
                log.warn(f"Could not load {path}: {err}")
    raise ImportError(
        "Could not import SegResNet. Expected lib/models/segresnet_full.py "
        "or extension/segresnet.py (wired via lib/models/segresnet.py)."
    )


def _liver_ceiling_ras(combined_path: str, log) -> int:
    import nibabel as nib

    img = nib.as_closest_canonical(nib.load(combined_path))
    arr = np.asanyarray(img.dataobj, dtype=np.uint8)
    liver_z = np.where((arr == 3).any(axis=(0, 1)))[0]
    if liver_z.size:
        z_max = int(liver_z.max())
        log.info(f"Liver Z range: {int(liver_z.min())}-{z_max} → ceiling Z={z_max}")
        return z_max
    heart_z = np.where((arr == 4).any(axis=(0, 1)))[0]
    if heart_z.size:
        z_max = int(heart_z.min())
        log.warn(f"Heart fallback ceiling Z={z_max}")
        return z_max
    log.warn("No liver/heart — no ceiling")
    return arr.shape[2]


def _torso_fat_on_grid(combined_path: str, ref_shape, ref_affine, log) -> Optional[np.ndarray]:
    """Load torso_fat (label 2) resampled onto reference CT grid."""
    import nibabel as nib
    from nibabel.processing import resample_from_to

    if not os.path.isfile(combined_path):
        return None
    comb = nib.load(combined_path)
    arr = np.asanyarray(comb.dataobj, dtype=np.uint8)
    torso = (arr == 2).astype(np.uint8)
    if not torso.any():
        log.warn("combined mask has no torso_fat (label 2)")
        return None
    src = nib.Nifti1Image(torso, comb.affine)
    tgt = nib.Nifti1Image(np.zeros(tuple(ref_shape), dtype=np.uint8), ref_affine)
    out = resample_from_to(src, tgt, order=0)
    return np.asanyarray(out.dataobj, dtype=np.uint8)


def predict_visceral_fat(
    ct_path: str,
    combined_path: str,
    ckpt_path: str,
    out_path: str,
    device: str = "gpu",
    log=None,
    sw_batch_size: int = 4,
    overlap: float = 0.10,
    patch_size: tuple = (96, 96, 96),
    auto_orient: bool = True,
    orient_metric: str = "dice",
) -> dict:
    """
    CT + combined_mask → visceral_fat.nii.gz on the **original CT grid**.

    Returns info dict including orientation / flip diagnostics.
    """
    import torch
    from monai.inferers import sliding_window_inference
    from monai.transforms import (
        AsDiscrete,
        AsDiscreted,
        Compose,
        EnsureChannelFirstd,
        EnsureType,
        KeepLargestConnectedComponent,
        LoadImaged,
        Orientationd,
        ScaleIntensityRanged,
        generate_spatial_bounding_box,
    )
    import nibabel as nib

    if log is None:
        log = _fallback_log()

    log.sep("predict_visceral_fat")
    info = {"out_path": out_path, "auto_orient": auto_orient}

    # Device
    if device.lower() in ("gpu", "cuda") and torch.cuda.is_available():
        torch_dev = torch.device("cuda:0")
    elif str(device).isdigit() and torch.cuda.is_available():
        torch_dev = torch.device(f"cuda:{device}")
    else:
        torch_dev = torch.device("cpu")
        if device.lower() in ("gpu", "cuda"):
            log.warn("CUDA unavailable — CPU")

    SegResNet = _import_segresnet(log)
    model = SegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        init_filters=16,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        dropout_prob=0.2,
        label_nc=8,
    )
    try:
        ckpt = torch.load(ckpt_path, map_location=torch_dev, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=torch_dev)
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    cleaned = {}
    for k, v in state_dict.items():
        for prefix in ("_model.", "model."):
            if k.startswith(prefix):
                k = k[len(prefix) :]
                break
        cleaned[k] = v
    model.load_state_dict(cleaned, strict=False)
    model.to(torch_dev).eval()

    pre = Compose(
        [
            LoadImaged(keys=["image", "seg"]),
            EnsureChannelFirstd(keys=["image", "seg"]),
            Orientationd(
                keys=["image", "seg"],
                axcodes="RAS",
                labels=(("L", "R"), ("P", "A"), ("I", "S")),
            ),
            ScaleIntensityRanged(
                keys=["image"], a_min=-200, a_max=100, b_min=0.0, b_max=1.0, clip=True
            ),
            AsDiscreted(keys=["seg"], to_onehot=8),
        ]
    )
    post = Compose(
        [
            EnsureType("tensor", device="cpu"),
            AsDiscrete(argmax=True, to_onehot=2),
            KeepLargestConnectedComponent(applied_labels=[1]),
        ]
    )

    original_ct = nib.load(ct_path)
    ref_affine = np.asarray(original_ct.affine, dtype=float)
    ref_shape = original_ct.shape

    t0 = time.perf_counter()
    d = pre({"image": str(ct_path), "seg": str(combined_path)})
    image_full = d["image"]
    seg_full = d["seg"]
    full_shape = tuple(image_full.shape[1:])
    ras_affine = metatensor_affine(image_full)
    if ras_affine is None:
        # Fallback: nibabel canonical affine
        ras_affine = np.asarray(nib.as_closest_canonical(original_ct).affine, dtype=float)
        log.warn("MetaTensor affine missing — using as_closest_canonical affine")

    log.info(f"RAS shape={full_shape}  original CT shape={ref_shape}")

    box_start, box_end = generate_spatial_bounding_box(
        image_full, select_fn=lambda x: x > 0, channel_indices=0
    )
    sl = tuple(slice(int(s), int(e)) for s, e in zip(box_start, box_end))
    image_c = image_full[:, sl[0], sl[1], sl[2]]
    seg_c = seg_full[:, sl[0], sl[1], sl[2]]

    combined_input = torch.cat(
        (image_c.unsqueeze(0).to(torch_dev), seg_c.unsqueeze(0).to(torch_dev)), dim=1
    )

    with torch.no_grad():
        prediction = sliding_window_inference(
            inputs=combined_input,
            roi_size=patch_size,
            sw_batch_size=sw_batch_size,
            predictor=lambda x: model(x[:, :1, ...], x[:, 1:, ...]),
            overlap=overlap,
            progress=True,
        )

    pred_c = post(prediction[0])[1].numpy().astype(np.uint8)
    pred_ras = np.zeros(full_shape, dtype=np.uint8)
    pred_ras[sl[0], sl[1], sl[2]] = pred_c

    z_max = _liver_ceiling_ras(combined_path, log)
    pred_ras[:, :, z_max:] = 0
    log.info(f"VF voxels (RAS) after ceiling: {int(pred_ras.sum()):,}")

    # ★ CRITICAL FIX: map RAS prediction → original CT voxel grid
    if tuple(full_shape) == tuple(ref_shape) and np.allclose(ras_affine, ref_affine, atol=1e-3):
        pred_orig = pred_ras
        log.info("RAS grid already matches original CT — no resample")
    else:
        log.info("Reorienting RAS prediction onto original CT grid…")
        pred_orig = reorient_ras_pred_to_reference(
            pred_ras, ras_affine, ref_shape, ref_affine
        )
        log.info(f"VF voxels (original grid): {int(pred_orig.sum()):,}")

    flip_info = None
    if auto_orient:
        torso = _torso_fat_on_grid(combined_path, ref_shape, ref_affine, log)
        if torso is not None and torso.any():
            before = int(pred_orig.sum())
            pred_orig, flip_info = best_xy_flip(pred_orig, torso, metric=orient_metric)
            log.info(
                f"Auto-orient ({orient_metric}): flipX={flip_info['best_flip_x']} "
                f"flipY={flip_info['best_flip_y']} score={flip_info['best_score']:.4f} "
                f"scores={flip_info['scores']}"
            )
            log.info(f"VF voxels after flip search: {before:,} → {int(pred_orig.sum()):,}")
        else:
            log.warn("Auto-orient skipped — no torso_fat reference")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    out_img = nib.Nifti1Image(pred_orig.astype(np.uint8), ref_affine, original_ct.header)
    out_img.header.set_data_dtype(np.uint8)
    out_img.header["scl_slope"] = 0
    out_img.header["scl_inter"] = 0
    nib.save(out_img, out_path)

    elapsed = time.perf_counter() - t0
    log.ok(f"Saved {out_path} in {elapsed:.1f}s")
    info.update(
        {
            "vf_voxels": int(pred_orig.sum()),
            "elapsed_s": elapsed,
            "flip": flip_info,
            "ras_shape": full_shape,
            "orig_shape": tuple(ref_shape),
        }
    )
    return info


def build_combined_mask(
    seg_dir: str,
    out_path: str,
    label_map: Optional[dict] = None,
    log=None,
) -> str:
    """
    Fast combined mask via SimpleITK reads (same label scheme as pipeline).
    """
    import SimpleITK as sitk
    import nibabel as nib

    if log is None:
        log = _fallback_log()

    if label_map is None:
        # Import lazily to avoid circular deps — inline default matching pipeline
        label_map = {
            "body_trunc": 1,
            "torso_fat": 2,
            "liver": 3,
            "heart": 4,
            "heart_myocardium": 4,
            "vertebrae_L1": 5, "vertebrae_L2": 5, "vertebrae_L3": 5,
            "vertebrae_L4": 5, "vertebrae_L5": 5,
            "rib_left_1": 6, "rib_left_2": 6, "rib_left_3": 6, "rib_left_4": 6,
            "rib_left_5": 6, "rib_left_6": 6, "rib_left_7": 6, "rib_left_8": 6,
            "rib_left_9": 6, "rib_left_10": 6, "rib_left_11": 6, "rib_left_12": 6,
            "rib_right_1": 6, "rib_right_2": 6, "rib_right_3": 6, "rib_right_4": 6,
            "rib_right_5": 6, "rib_right_6": 6, "rib_right_7": 6, "rib_right_8": 6,
            "rib_right_9": 6, "rib_right_10": 6, "rib_right_11": 6, "rib_right_12": 6,
            "vertebrae_T1": 7, "vertebrae_T2": 7, "vertebrae_T3": 7, "vertebrae_T4": 7,
            "vertebrae_T5": 7, "vertebrae_T6": 7, "vertebrae_T7": 7, "vertebrae_T8": 7,
            "vertebrae_T9": 7, "vertebrae_T10": 7, "vertebrae_T11": 7, "vertebrae_T12": 7,
        }

    combined = None
    ref_img = None
    for stem, label in label_map.items():
        fp = os.path.join(seg_dir, f"{stem}.nii.gz")
        if not os.path.isfile(fp):
            continue
        sitk_img = sitk.ReadImage(fp)
        arr = np.transpose(sitk.GetArrayFromImage(sitk_img).astype(np.uint8), (2, 1, 0))
        if combined is None:
            ref_img = nib.load(fp)
            combined = np.zeros(arr.shape, dtype=np.uint8)
        combined[arr > 0] = np.uint8(label)

    if combined is None:
        raise RuntimeError(f"No TotalSeg files matched in {seg_dir}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    out = nib.Nifti1Image(combined, ref_img.affine, ref_img.header)
    out.header.set_data_dtype(np.uint8)
    nib.save(out, out_path)
    log.ok(f"combined mask → {out_path}  nonzero={int(np.count_nonzero(combined)):,}")
    return out_path
