"""Patient pipeline logger (file + console). Port of PETCTSegmentationModule.PLog."""
from __future__ import annotations

import logging
import os
from datetime import datetime

_MODULE_LOG = logging.getLogger("KU.PETCTSegmentation")


class PatientLog:
    """
    Logs to three sinks:
      1. Python logging (Slicer Application Log when available)
      2. stdout / console
      3. ``{log_dir}/{key}.log`` (append)
    """

    LEVELS = {"INFO": "INFO ", "OK": "OK   ", "WARN": "WARN ", "ERROR": "ERROR"}

    def __init__(self, log_dir: str, key: str):
        self._key = key
        os.makedirs(log_dir, exist_ok=True)
        self._path = os.path.join(log_dir, f"{key}.log")
        self._fh = open(self._path, "a", encoding="utf-8", buffering=1)
        self._write("INFO", f"{'─' * 60}")
        self._write(
            "INFO",
            f"Log opened  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        )

    def _write(self, level: str, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        lvl = self.LEVELS.get(level, level[:5].ljust(5))
        line = f"[{ts}] [{lvl}] [{self._key}] {msg}"
        print(line)
        if level == "ERROR":
            _MODULE_LOG.error(line)
        elif level == "WARN":
            _MODULE_LOG.warning(line)
        else:
            _MODULE_LOG.info(line)
        if self._fh and not self._fh.closed:
            self._fh.write(line + "\n")

    def info(self, msg: str) -> None:
        self._write("INFO", msg)

    def ok(self, msg: str) -> None:
        self._write("OK", msg)

    def warn(self, msg: str) -> None:
        self._write("WARN", msg)

    def error(self, msg: str) -> None:
        self._write("ERROR", msg)

    def sep(self, label: str = "") -> None:
        self._write("INFO", f"── {label} {'─' * (55 - len(label))}")

    def close(self) -> None:
        if self._fh and not self._fh.closed:
            self._write("INFO", "Log closed")
            self._fh.close()


# Alias used by the original module
PLog = PatientLog
