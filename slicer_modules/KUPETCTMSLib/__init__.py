"""
Shared helpers for 3D Slicer scripted modules in this extension.

Importing this package puts the extension root on ``sys.path`` so
``import lib`` works in both layouts:

* Development: Additional Module Path = ``<repo>/slicer_modules``
  (``lib`` lives in ``<repo>/lib``)
* Installed / CMake-built extension: ``lib`` and ``scripts`` are copied
  next to the scripted modules in ``qt-scripted-modules``
"""
from __future__ import annotations

import sys
from pathlib import Path

__all__ = ["extension_root"]


def extension_root() -> Path:
    """Directory that contains ``lib/`` (and usually ``scripts/``)."""
    here = Path(__file__).resolve().parent
    candidates = (
        here.parent,         # slicer_modules (dev) or qt-scripted-modules (install)
        here.parent.parent,  # repo root when this file is slicer_modules/KUPETCTMSLib
    )
    for candidate in candidates:
        if (candidate / "lib" / "__init__.py").is_file():
            return candidate
    return here.parent


def _ensure_lib_importable() -> Path:
    root = extension_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


_ROOT = _ensure_lib_importable()
