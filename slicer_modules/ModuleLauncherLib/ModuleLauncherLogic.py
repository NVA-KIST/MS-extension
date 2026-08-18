"""
ModuleLauncherLogic — thin / UI-support Logic.
Launcher UI + master pipeline runner (scripts/run_pipeline.py).
"""
from __future__ import annotations

import importlib.util
import io
import os
import sys
import threading
import traceback
from pathlib import Path

from slicer.ScriptedLoadableModule import ScriptedLoadableModuleLogic


def _extension_new_root() -> Path:
    # .../extension_new/slicer_modules/ModuleLauncher/ModuleLauncherLogic.py
    return Path(__file__).resolve().parents[2]


class ModuleLauncherLogic(ScriptedLoadableModuleLogic):

    def __init__(self):
        super().__init__()
        self._pipeline_thread = None
        self._pipeline_stop = threading.Event()

    @property
    def pipeline_running(self) -> bool:
        t = self._pipeline_thread
        return t is not None and t.is_alive()

    def default_vf_checkpoint(self) -> str:
        """Best-effort path to the bundled SegResNet checkpoint."""
        root = _extension_new_root()
        candidates = [
            root / "lib" / "models" / "epoch=399-step=8800.ckpt",
            root.parent / "models" / "epoch=399-step=8800.ckpt",
        ]
        try:
            sys.path.insert(0, str(root))
            from lib.models.segresnet import default_vf_checkpoint
            candidates.insert(0, Path(default_vf_checkpoint()))
        except Exception:
            pass
        for p in candidates:
            if p.is_file():
                return str(p)
        return str(candidates[0])

    def build_pipeline_argv(self, opts: dict) -> list[str]:
        """Build argv list for scripts/run_pipeline.py from a UI options dict."""
        root = opts["root"]
        out = opts["out"]
        argv = ["--root", root, "--out", out]

        ckpt = (opts.get("ckpt") or "").strip()
        if ckpt:
            argv += ["--ckpt", ckpt]

        src = (opts.get("src") or "").strip()
        if src:
            argv += ["--src", src]
            em = (opts.get("existing_map") or "").strip()
            if em:
                argv += ["--existing-map", em]

        if opts.get("skip_seg"):
            argv.append("--skip-seg")
        if opts.get("skip_post"):
            argv.append("--skip-post")
        if opts.get("skip_quant"):
            argv.append("--skip-quant")

        limit = int(opts.get("limit") or 0)
        if limit > 0:
            argv += ["--limit", str(limit)]

        if opts.get("no_skip_done"):
            argv.append("--no-skip-done")
        if opts.get("no_append"):
            argv.append("--no-append")

        device = (opts.get("device") or "gpu").strip() or "gpu"
        argv += ["--device", device]
        cuda = (opts.get("cuda") or "").strip()
        if cuda:
            argv += ["--cuda", cuda]

        if opts.get("skip_ts"):
            argv.append("--skip-ts")
        if opts.get("skip_vf"):
            argv.append("--skip-vf")
        if opts.get("radiomics"):
            argv.append("--radiomics")

        return argv

    def run_pipeline_async(self, opts: dict, log_cb=None, done_cb=None) -> bool:
        """
        Start scripts/run_pipeline.py in a background thread.
        log_cb(str) and done_cb(exit_code:int, error:str|None) are optional.
        Returns False if a run is already in progress.
        """
        if self.pipeline_running:
            return False

        argv = self.build_pipeline_argv(opts)
        self._pipeline_stop.clear()

        def _worker():
            rc = 1
            err = None
            try:
                rc = self._run_pipeline_blocking(argv, log_cb=log_cb)
            except Exception:
                err = traceback.format_exc()
                if log_cb:
                    log_cb(err)
                rc = 1
            if done_cb:
                done_cb(rc, err)

        self._pipeline_thread = threading.Thread(
            target=_worker, name="ModuleLauncherPipeline", daemon=True
        )
        self._pipeline_thread.start()
        return True

    def _run_pipeline_blocking(self, argv: list[str], log_cb=None) -> int:
        root = _extension_new_root()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        script = root / "scripts" / "run_pipeline.py"
        if not script.is_file():
            raise FileNotFoundError(f"run_pipeline.py not found: {script}")

        if log_cb:
            log_cb(f"extension_new root: {root}")
            log_cb(f"argv: {' '.join(argv)}")

        # Capture print() from the pipeline scripts into the UI log.
        class _Tee(io.TextIOBase):
            def __init__(self, original):
                self._original = original
                self._buf = ""

            def write(self, s):
                if not s:
                    return 0
                if self._original is not None:
                    try:
                        self._original.write(s)
                    except Exception:
                        pass
                if log_cb:
                    self._buf += s
                    while "\n" in self._buf:
                        line, self._buf = self._buf.split("\n", 1)
                        log_cb(line)
                return len(s)

            def flush(self):
                if self._original is not None:
                    try:
                        self._original.flush()
                    except Exception:
                        pass
                if log_cb and self._buf:
                    log_cb(self._buf)
                    self._buf = ""

        spec = importlib.util.spec_from_file_location("_kupet_run_pipeline", str(script))
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        # Ensure cwd-independent imports inside scripts
        os.environ.setdefault("KUPETCTMS_EXTENSION_NEW", str(root))
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = _Tee(old_out)
        sys.stderr = _Tee(old_err)
        try:
            spec.loader.exec_module(mod)
            return int(mod.main(argv))
        finally:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except Exception:
                pass
            sys.stdout, sys.stderr = old_out, old_err
