"""
ModuleLauncherLogic — thin / UI-support Logic.
Launcher UI + master pipeline runner (scripts/run_pipeline.py).

Runs the pipeline as an *external* Python subprocess (not Slicer's embedded
interpreter). Paths are auto-discovered from this file's location and the
environment — no machine-specific hardcoded roots.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import traceback
from pathlib import Path

from slicer.ScriptedLoadableModule import ScriptedLoadableModuleLogic


def _extension_new_root() -> Path:
    """
    Auto-root: walk up from this file until scripts/run_pipeline.py is found.
    Works for both ModuleLauncher/ and ModuleLauncherLib/ layouts.
    """
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "scripts" / "run_pipeline.py").is_file():
            return parent
    # Fallback: .../slicer_modules/<this>/ → extension_new
    return p.parents[2]


def _is_slicer_python(exe: Path | str) -> bool:
    s = str(exe).replace("\\", "/").lower()
    return "slicer.org" in s or "3d slicer" in s or "pythonslicer" in s


def _conda_base() -> Path | None:
    """Resolve conda base without hardcoding miniconda/anaconda paths."""
    for key in ("CONDA_ROOT", "CONDA_PREFIX_1"):
        v = (os.environ.get(key) or "").strip()
        if v:
            p = Path(v)
            if p.is_dir():
                return p
    prefix = (os.environ.get("CONDA_PREFIX") or "").strip()
    if prefix:
        p = Path(prefix)
        # active env: .../envs/name → base is parents[1]; base install: .../miniconda3
        if p.parent.name.lower() == "envs":
            return p.parent.parent
        return p
    conda = shutil.which("conda")
    if conda:
        # .../Scripts/conda.exe or .../condabin/conda.bat → base nearby
        c = Path(conda).resolve()
        for parent in c.parents:
            if (parent / "envs").is_dir() or (parent / "python.exe").is_file():
                return parent
    return None


def _python_from_prefix(prefix: Path) -> list[Path]:
    return [
        prefix / "python.exe",
        prefix / "bin" / "python",
        prefix / "bin" / "python3",
    ]


def _python_can_import_ts(python_exe: Path) -> bool:
    """True if this interpreter can import totalsegmentator (quick probe)."""
    if not python_exe.is_file() or _is_slicer_python(python_exe):
        return False
    try:
        r = subprocess.run(
            [str(python_exe), "-c", "import totalsegmentator"],
            capture_output=True,
            timeout=20,
            env=_clean_env_for_probe(python_exe),
        )
        return r.returncode == 0
    except Exception:
        return False


def _clean_env_for_probe(python_exe: Path) -> dict:
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONNOUSERSITE"):
        env.pop(key, None)
    py_dir = python_exe.resolve().parent
    env_root = py_dir.parent if py_dir.name.lower() == "scripts" else py_dir
    env["PYTHONHOME"] = str(env_root)
    return env


class ModuleLauncherLogic(ScriptedLoadableModuleLogic):

    def __init__(self):
        super().__init__()
        self._pipeline_thread = None
        self._pipeline_proc = None
        self._pipeline_stop = threading.Event()

    @property
    def pipeline_running(self) -> bool:
        t = self._pipeline_thread
        return t is not None and t.is_alive()

    def project_root(self) -> Path:
        """Auto-detected extension_new root (contains scripts/ + lib/)."""
        return _extension_new_root()

    def default_vf_checkpoint(self) -> str:
        """
        Best-effort checkpoint under the auto-root.
        Prefer lib.models.segresnet.default_vf_checkpoint(); else newest *.ckpt
        under lib/models/ (no hardcoded filename required).
        """
        root = self.project_root()
        candidates: list[Path] = []
        try:
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            from lib.models.segresnet import default_vf_checkpoint
            candidates.append(Path(default_vf_checkpoint()))
        except Exception:
            pass

        models_dir = root / "lib" / "models"
        if models_dir.is_dir():
            ckpts = sorted(
                models_dir.glob("*.ckpt"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            candidates.extend(ckpts)
        # Sibling extension/models/ (legacy layout next to extension_new)
        legacy = root.parent / "models"
        if legacy.is_dir():
            candidates.extend(
                sorted(legacy.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
            )

        seen = set()
        for p in candidates:
            try:
                rp = p.resolve()
            except Exception:
                continue
            if rp in seen or not rp.is_file():
                continue
            seen.add(rp)
            return str(rp)
        # Placeholder path for the UI (may not exist yet)
        return str(models_dir / "checkpoint.ckpt")

    def default_pipeline_python(self) -> str:
        """
        Auto-pick an external Python that has TotalSegmentator.
        Order (no hardcoded user paths):
          1. KUPETCTMS_PYTHON
          2. CONDA_PREFIX (active env)
          3. python next to TotalSegmentator on PATH
          4. any conda env under conda-base/envs/* that imports totalsegmentator
          5. PATH python / python3 (skip Slicer)
        """
        candidates: list[Path] = []

        env_py = (os.environ.get("KUPETCTMS_PYTHON") or "").strip()
        if env_py:
            candidates.append(Path(env_py))

        conda_prefix = (os.environ.get("CONDA_PREFIX") or "").strip()
        if conda_prefix:
            candidates.extend(_python_from_prefix(Path(conda_prefix)))

        # If TotalSegmentator is on PATH, its env's python is ideal
        ts = shutil.which("TotalSegmentator")
        if ts:
            ts_p = Path(ts).resolve()
            # .../env/Scripts/TotalSegmentator.exe → env root
            if ts_p.parent.name.lower() == "scripts":
                candidates.extend(_python_from_prefix(ts_p.parent.parent))
            else:
                candidates.extend(_python_from_prefix(ts_p.parent))

        base = _conda_base()
        if base is not None:
            envs = base / "envs"
            if envs.is_dir():
                for env_dir in sorted(envs.iterdir()):
                    if env_dir.is_dir():
                        candidates.extend(_python_from_prefix(env_dir))
            candidates.extend(_python_from_prefix(base))

        for name in ("python", "python3"):
            w = shutil.which(name)
            if w:
                candidates.append(Path(w))

        # Prefer interpreters that can import totalsegmentator
        seen: set[str] = set()
        fallback = ""
        for p in candidates:
            try:
                if not p.is_file():
                    continue
                rp = str(p.resolve())
            except Exception:
                continue
            if rp in seen or _is_slicer_python(rp):
                continue
            seen.add(rp)
            if not fallback:
                fallback = rp
            if _python_can_import_ts(p):
                return rp
        return fallback or ("" if _is_slicer_python(sys.executable) else sys.executable)

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
        Start scripts/run_pipeline.py in a background thread via subprocess.
        log_cb(str) and done_cb(exit_code:int, error:str|None) are optional.
        Returns False if a run is already in progress.
        """
        if self.pipeline_running:
            return False

        argv = self.build_pipeline_argv(opts)
        python_exe = (opts.get("python") or "").strip() or self.default_pipeline_python()
        self._pipeline_stop.clear()

        def _worker():
            rc = 1
            err = None
            try:
                rc = self._run_pipeline_blocking(argv, python_exe=python_exe, log_cb=log_cb)
            except Exception:
                err = traceback.format_exc()
                if log_cb:
                    log_cb(err)
                rc = 1
            finally:
                self._pipeline_proc = None
            if done_cb:
                done_cb(rc, err)

        self._pipeline_thread = threading.Thread(
            target=_worker, name="ModuleLauncherPipeline", daemon=True
        )
        self._pipeline_thread.start()
        return True

    def stop_pipeline(self) -> None:
        """Request stop; terminates the subprocess if running."""
        self._pipeline_stop.set()
        proc = self._pipeline_proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass

    def _run_pipeline_blocking(self, argv: list[str], python_exe: str, log_cb=None) -> int:
        root = self.project_root()
        script = root / "scripts" / "run_pipeline.py"
        if not script.is_file():
            raise FileNotFoundError(f"run_pipeline.py not found under auto-root: {root}")
        if not python_exe or not Path(python_exe).is_file():
            raise FileNotFoundError(
                "Pipeline Python not found.\n"
                "Set env KUPETCTMS_PYTHON to an interpreter with TotalSegmentator,\n"
                "or activate a conda env that has it, or put TotalSegmentator on PATH."
            )
        if _is_slicer_python(python_exe):
            raise RuntimeError(
                f"Refusing Slicer Python: {python_exe}\n"
                "Pick an external conda/venv Python (Pipeline Python field)."
            )

        cmd = [python_exe, "-u", str(script), *argv]
        if log_cb:
            log_cb(f"auto-root: {root}")
            log_cb(f"python: {python_exe}")
            log_cb(f"cmd: {' '.join(cmd)}")
            log_cb(
                "NOTE: TotalSegmentator runs in this external env (not Slicer Python). "
                "Progress lines should appear below; the 'total' task can still take several minutes."
            )

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["KUPETCTMS_EXTENSION_NEW"] = str(root)

        # Slicer injects PYTHONPATH / PYTHONHOME → SRE module mismatch if leaked.
        for key in (
            "PYTHONPATH",
            "PYTHONHOME",
            "PYTHONSTARTUP",
            "PYTHONNOUSERSITE",
            "PYTHONUSERBASE",
            "PYTHONSAFEPATH",
            "QT_PLUGIN_PATH",
            "VTK_SILENCE_GET_VOID_POINTER_WARNINGS",
        ):
            env.pop(key, None)

        py_dir = Path(python_exe).resolve().parent
        env_root = py_dir
        scripts_dir = py_dir / "Scripts"
        if py_dir.name.lower() == "scripts":
            env_root = py_dir.parent
            scripts_dir = py_dir
        env["PYTHONHOME"] = str(env_root)

        path_parts = []
        for part in (scripts_dir, env_root, *env.get("PATH", "").split(os.pathsep)):
            if not part:
                continue
            part_s = str(part)
            low = part_s.replace("\\", "/").lower()
            if "slicer.org" in low or "/3d slicer" in low:
                continue
            if part_s not in path_parts:
                path_parts.append(part_s)
        env["PATH"] = os.pathsep.join(path_parts)

        if log_cb:
            log_cb(f"PYTHONHOME={env.get('PYTHONHOME')}")
            log_cb("Cleared Slicer PYTHONPATH/PYTHONHOME for subprocess.")

        proc = subprocess.Popen(
            cmd,
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        self._pipeline_proc = proc
        assert proc.stdout is not None
        for line in proc.stdout:
            if self._pipeline_stop.is_set():
                try:
                    proc.terminate()
                except Exception:
                    pass
                break
            line = line.rstrip("\r\n")
            if log_cb:
                log_cb(line)
            else:
                print(line, flush=True)
        return int(proc.wait())
