"""Model definitions and checkpoint path helpers."""
from lib.models.segresnet import EXTENSION_ROOT, default_vf_checkpoint

__all__ = [
    "EXTENSION_ROOT",
    "SegResNet",
    "SegResNetVAE",
    "SPADESegResNet",
    "default_vf_checkpoint",
]


def __getattr__(name: str):
    if name in ("SegResNet", "SegResNetVAE", "SPADESegResNet"):
        from lib.models import segresnet as _sr

        return getattr(_sr, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
