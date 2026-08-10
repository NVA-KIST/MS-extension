"""Unit tests for radiomics option helpers (no pyradiomics required)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.quantification.radiomics import (
    SELECTED_RADIOMICS_FEATURE_ORDER,
    derived_within_roi_features,
    is_radiomics_enabled,
    radiomics_config_signature,
    selected_radiomics_feature_keys,
    selected_radiomics_features_by_class,
)


def test_selected_keys_order_preserved():
    opts = {"selected_feature_keys": ["zone_entropy", "p10", "unknown"]}
    assert selected_radiomics_feature_keys(opts) == ["p10", "zone_entropy"]


def test_by_class_mapping():
    opts = {"selected_feature_keys": ["p10", "contrast", "sahgle"]}
    by = selected_radiomics_features_by_class(opts)
    assert by["firstorder"] == ["10Percentile"]
    assert by["glcm"] == ["Contrast"]
    assert by["glszm"] == ["SmallAreaHighGrayLevelEmphasis"]


def test_enabled_and_signature():
    assert not is_radiomics_enabled({})
    opts = {
        "selected_feature_keys": list(SELECTED_RADIOMICS_FEATURE_ORDER),
        "bin_width": 0.25,
        "derived": True,
    }
    assert is_radiomics_enabled(opts)
    sig = radiomics_config_signature(opts)
    assert "features=p10,p90" in sig
    assert "derived=1" in sig
    assert "geometry=native" in sig


def test_derived_features():
    feats = {
        "rad_firstorder_10Percentile": 1.0,
        "rad_firstorder_90Percentile": 4.0,
        "rad_firstorder_Median": 2.0,
    }
    d = derived_within_roi_features(feats)
    assert d["rad_derived_P90MinusP10"] == 3.0
    assert d["rad_derived_P90ToMedian"] == 2.0
