"""
PyRadiomics helpers ported from PETBiomarkerStudioLogic (no Slicer).

Slicer node I/O (ExportSegmentsToLabelmapNode, saveNode) stays in the module Logic;
call ``extract_radiomics_from_paths`` once PET + mask are on disk.
"""
from __future__ import annotations

import math
from typing import Any, Optional

# Short UI keys → (PyRadiomics class, feature name)
SELECTED_RADIOMICS_FEATURES: dict[str, tuple[str, str]] = {
    "p10": ("firstorder", "10Percentile"),
    "p90": ("firstorder", "90Percentile"),
    "entropy": ("firstorder", "Entropy"),
    "skewness": ("firstorder", "Skewness"),
    "contrast": ("glcm", "Contrast"),
    "sahgle": ("glszm", "SmallAreaHighGrayLevelEmphasis"),
    "lalgle": ("glszm", "LargeAreaLowGrayLevelEmphasis"),
    "zone_entropy": ("glszm", "ZoneEntropy"),
}

SELECTED_RADIOMICS_FEATURE_ORDER: tuple[str, ...] = (
    "p10",
    "p90",
    "entropy",
    "skewness",
    "contrast",
    "sahgle",
    "lalgle",
    "zone_entropy",
)

RADIOMICS_CLASS_KEYS: tuple[str, ...] = (
    "firstorder",
    "shape",
    "glcm",
    "glrlm",
    "glszm",
    "gldm",
    "ngtdm",
)


def ensure_radiomics_featureextractor():
    """Import pyradiomics featureextractor or raise a clear ImportError."""
    try:
        from radiomics import featureextractor

        return featureextractor
    except ModuleNotFoundError as e:
        raise ImportError(
            "pyradiomics is required for radiomics extraction. "
            "Install with: pip install pyradiomics"
        ) from e


def selected_radiomics_feature_keys(radiomics_options: Optional[dict]) -> list[str]:
    requested = set((radiomics_options or {}).get("selected_feature_keys", []) or [])
    return [k for k in SELECTED_RADIOMICS_FEATURE_ORDER if k in requested]


def selected_radiomics_features_by_class(
    radiomics_options: Optional[dict],
) -> dict[str, list[str]]:
    by_class: dict[str, list[str]] = {}
    for key in selected_radiomics_feature_keys(radiomics_options):
        feature_class, feature_name = SELECTED_RADIOMICS_FEATURES[key]
        by_class.setdefault(feature_class, []).append(feature_name)
    return by_class


def is_radiomics_enabled(radiomics_options: Optional[dict]) -> bool:
    if not radiomics_options:
        return False
    if selected_radiomics_feature_keys(radiomics_options):
        return True
    if bool(radiomics_options.get("core_panel", False)):
        return True
    return any(bool(radiomics_options.get(k, False)) for k in RADIOMICS_CLASS_KEYS)


def radiomics_config_signature(radiomics_options: Optional[dict]) -> str:
    opts = radiomics_options or {}
    if not is_radiomics_enabled(opts):
        return "radiomics=off"
    selected = ",".join(selected_radiomics_feature_keys(opts)) or "none"
    classes = ",".join(k for k in RADIOMICS_CLASS_KEYS if opts.get(k, False)) or "none"
    if opts.get("resample_isotropic", False):
        geometry = f"iso:{float(opts.get('resampled_spacing_mm', 4.0)):.6g}"
    else:
        geometry = "native"
    return (
        f"features={selected};"
        f"derived={int(bool(opts.get('derived', False)))};"
        f"bin={float(opts.get('bin_width', 0.25)):.6g};"
        f"geometry={geometry};classes={classes}"
    )


def make_radiomics_extractor(radiomics_options: Optional[dict]):
    """Build a configured PyRadiomics RadiomicsFeatureExtractor."""
    opts = dict(radiomics_options or {})
    featureextractor = ensure_radiomics_featureextractor()

    bin_width = float(opts.get("bin_width", 0.25))
    if not math.isfinite(bin_width) or bin_width <= 0:
        raise ValueError(f"Radiomics bin width must be positive: {bin_width}")

    settings: dict[str, Any] = {
        "binWidth": bin_width,
        "label": 1,
        "normalize": False,
        "additionalInfo": False,
    }

    if opts.get("resample_isotropic", False):
        spacing = float(opts.get("resampled_spacing_mm", 4.0))
        if not math.isfinite(spacing) or spacing <= 0:
            raise ValueError(f"Radiomics resampled spacing must be positive: {spacing}")
        settings["resampledPixelSpacing"] = [spacing, spacing, spacing]
        settings["interpolator"] = "sitkBSpline"

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()

    selected_by_class = selected_radiomics_features_by_class(opts)
    if selected_by_class:
        extractor.enableFeaturesByName(**selected_by_class)

    if opts.get("core_panel", False):
        all_selected = selected_radiomics_features_by_class(
            {"selected_feature_keys": list(SELECTED_RADIOMICS_FEATURE_ORDER)}
        )
        extractor.enableFeaturesByName(**all_selected)

    for key in RADIOMICS_CLASS_KEYS:
        if opts.get(key, False):
            extractor.enableFeatureClassByName(featureClass=key)

    return extractor


def derived_within_roi_features(features: dict[str, Any]) -> dict[str, float]:
    """Robust high-tail descriptors from selected first-order values."""
    derived: dict[str, float] = {}
    p10 = features.get("rad_firstorder_10Percentile")
    p90 = features.get("rad_firstorder_90Percentile")
    median = features.get("rad_firstorder_Median")

    if p10 is not None and p90 is not None:
        derived["rad_derived_P90MinusP10"] = float(p90 - p10)

    if p90 is not None and median is not None and abs(float(median)) > 1e-12:
        derived["rad_derived_P90ToMedian"] = float(p90 / median)

    return derived


def normalize_raw_radiomics(raw_features: dict) -> dict[str, float]:
    """Drop diagnostics_*; map original_* → rad_* floats."""
    out: dict[str, float] = {}
    for key, value in raw_features.items():
        if key.startswith("diagnostics_"):
            continue
        clean = "rad_" + key.replace("original_", "")
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(numeric):
            continue
        out[clean] = numeric
    return out


def radiomics_profile_label(radiomics_options: Optional[dict]) -> str:
    opts = radiomics_options or {}
    has_selected = bool(selected_radiomics_feature_keys(opts))
    has_legacy_core = bool(opts.get("core_panel", False))
    has_full = any(opts.get(k, False) for k in RADIOMICS_CLASS_KEYS)
    if (has_selected or has_legacy_core) and has_full:
        return "selected_features+full_classes"
    if has_selected or has_legacy_core:
        return "selected_features"
    return "full_classes"


def attach_radiomics_metadata(
    features: dict[str, Any],
    radiomics_options: Optional[dict],
) -> dict[str, Any]:
    opts = radiomics_options or {}
    features = dict(features)
    features["radiomics_profile"] = radiomics_profile_label(opts)
    features["radiomics_config_signature"] = radiomics_config_signature(opts)
    features["radiomics_bin_width"] = float(opts.get("bin_width", 0.25))
    if opts.get("resample_isotropic", False):
        features["radiomics_resampled_spacing_mm"] = float(
            opts.get("resampled_spacing_mm", 4.0)
        )
    else:
        features["radiomics_resampled_spacing_mm"] = "native"
    return features


def extract_radiomics_from_paths(
    image_path: str,
    mask_path: str,
    radiomics_options: Optional[dict] = None,
    *,
    label: int = 1,
) -> dict[str, Any]:
    """
    Run PyRadiomics on on-disk image + label mask.

    Returns feature dict with rad_* keys, optional derived_*, and metadata fields.
    """
    opts = radiomics_options or {}
    if not is_radiomics_enabled(opts):
        return {}

    extractor = make_radiomics_extractor(opts)
    raw = extractor.execute(image_path, mask_path, label=label)
    features = normalize_raw_radiomics(raw)

    if opts.get("derived", False):
        features.update(derived_within_roi_features(features))

    return attach_radiomics_metadata(features, opts)
