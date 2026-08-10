"""Batch patient record + steps.json persistence."""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

STEP_SEG = "seg"
STEP_VF = "vf"
STEP_MIRROR = "mirror"
STEP_KEYS = [STEP_SEG, STEP_VF, STEP_MIRROR]

ST_PENDING = "PENDING"
ST_RUNNING = "RUNNING"
ST_DONE = "DONE"
ST_ERROR = "ERROR"
ST_SKIP = "SKIPPED"


class BatchPatient:
    """Per-patient state for the PET-CT segmentation batch wizard."""

    def __init__(self, subject_id: str, scan_date: str, root: str):
        self.subject_id = subject_id
        self.scan_date = scan_date
        self.root = root

        self.status: Dict[str, str] = {k: ST_PENDING for k in STEP_KEYS}
        self.message: Dict[str, str] = {k: "" for k in STEP_KEYS}
        self.pct = 0

        self.vf_was_flipped: bool = False
        self.vf_flip_y: bool = False

    @property
    def key(self) -> str:
        return f"{self.subject_id}_{self.scan_date}"

    @property
    def ct_dcm(self) -> str:
        return os.path.join(self.root, "CT", f"{self.key}_CT")

    @property
    def pet_dcm(self) -> str:
        return os.path.join(self.root, "PET", f"{self.key}_PET")

    @property
    def ct_nii(self) -> str:
        return os.path.join(self.root, "CT_NIfTI", f"{self.key}_CT.nii.gz")

    @property
    def seg_dir(self) -> str:
        return os.path.join(self.root, "Segments", f"{self.key}_Seg")

    @property
    def log_dir(self) -> str:
        return os.path.join(self.root, "pipeline_logs")

    def seg(self, fname: str) -> str:
        return os.path.join(self.seg_dir, fname)

    def set(self, step: str, status: str, msg: str = "", pct: int = -1) -> None:
        self.status[step] = status
        if msg:
            self.message[step] = msg
        if pct >= 0:
            self.pct = pct

    def is_done(self, step: str) -> bool:
        return self.status.get(step) in (ST_DONE, ST_SKIP)


def steps_file(rec: BatchPatient) -> str:
    return os.path.join(rec.log_dir, f"{rec.key}.steps.json")


def save_steps(rec: BatchPatient) -> None:
    """Persist per-step status + mirror flip flags (best-effort)."""
    try:
        os.makedirs(rec.log_dir, exist_ok=True)
        data = {
            "status": rec.status,
            "message": rec.message,
            "vf_was_flipped": rec.vf_was_flipped,
            "vf_flip_y": rec.vf_flip_y,
        }
        with open(steps_file(rec), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def load_steps(rec: BatchPatient) -> bool:
    """
    Restore status from disk. RUNNING → PENDING after restart.
    Returns True if a steps file was found.
    """
    sf = steps_file(rec)
    if not os.path.isfile(sf):
        return False
    try:
        with open(sf, encoding="utf-8") as f:
            data = json.load(f)
        status = data.get("status", {})
        for k in STEP_KEYS:
            v = status.get(k, ST_PENDING)
            rec.status[k] = ST_PENDING if v == ST_RUNNING else v
        msg = data.get("message", {})
        for k in STEP_KEYS:
            if k in msg:
                rec.message[k] = msg[k]
        rec.vf_was_flipped = bool(data.get("vf_was_flipped", False))
        rec.vf_flip_y = bool(data.get("vf_flip_y", False))
        return True
    except Exception:
        return False
