"""
SegResNet / SegResNetVAE / SPADESegResNet used by visceral-fat inference.

Re-exports the real network definitions from ``extension/segresnet.py``
(MONAI-based; same file VFInference.py and visceral_fat_segmentations.py load).

Checkpoint (Lightning):
  ``extension/models/epoch=399-step=8800.ckpt``

Torch/MONAI are imported lazily so checkpoint path helpers work without GPU deps.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

__all__ = [
    "SegResNet",
    "SegResNetVAE",
    "SPADESegResNet",
    "default_vf_checkpoint",
    "EXTENSION_ROOT",
]

# lib/models/segresnet.py → parents[3] == extension/
EXTENSION_ROOT = Path(__file__).resolve().parents[3]
_SRC = EXTENSION_ROOT / "segresnet.py"
_LOADED = False


def default_vf_checkpoint() -> Path:
    """Recommended VF weights used by VFInference / run_visceralfat."""
    return EXTENSION_ROOT / "models" / "epoch=399-step=8800.ckpt"


def _ensure_loaded() -> None:
    global _LOADED, SegResNet, SegResNetVAE, SPADESegResNet
    if _LOADED:
        return
    if not _SRC.is_file():
        raise ImportError(
            f"Missing SegResNet source at {_SRC}. "
            "Expected the MONAI network file next to VFInference.py."
        )
    spec = importlib.util.spec_from_file_location("_kupet_segresnet", str(_SRC))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    SegResNet = mod.SegResNet
    SegResNetVAE = mod.SegResNetVAE
    SPADESegResNet = mod.SPADESegResNet
    _LOADED = True


def __getattr__(name: str) -> Any:
    if name in ("SegResNet", "SegResNetVAE", "SPADESegResNet"):
        _ensure_loaded()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
