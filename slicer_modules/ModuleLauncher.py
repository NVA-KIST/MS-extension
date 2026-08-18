"""
ModuleLauncher — Module + Widget (UI + triggers).
Launcher UI only — no lib algorithm required.
"""
"""
ModuleLauncher — 3D Slicer scripted module
==========================================

Drag-and-drop this file into 3D Slicer, then find it under:
    Modules → Metabolic Syndrome Toolkit → Module Launcher

Opens any of your custom modules as an independent floating window so you
can have multiple modules visible and usable at the same time.
Each window is fully functional — it is not a copy, it runs the real module.
"""

import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
)

# ── Registry of launchable modules ────────────────────────────────────────────
# Add or remove entries here to customise the launcher.
# 'import_name'  : the Python filename without .py
# 'class_name'   : the Widget class inside that file
# 'display_name' : label shown on the button and window title bar
# 'description'  : tooltip on the button
# 'color'        : button accent colour (hex)

_MODULES = [
    {
        'import_name':  'PETCTSegmentationModule',
        'class_name':   'PETCTSegmentationModuleWidget',
        'display_name': 'PETCT Segmentation Module',
        'description':  'Semi-automatic batch wizard: DICOM conversion + TotalSegmentator and '
                        'visceral-fat prediction run in bulk; Mirror QC is a per-patient '
                        'human step. Vessel growing, ureter post-processing and '
                        'quantification have their own modules.',
        'color':        '#1a237e',
    },
    {
        'import_name':  'UreterPostProcess',
        'class_name':   'UreterPostProcessWidget',
        'display_name': 'Ureter Post-Process',
        'description':  'PET-guided ureter mask, per-organ L1-L5 clipping and cleanup.',
        'color':        '#2e7d32',
    },
    {
        'import_name':  'SegmentDilator',
        'class_name':   'SegmentDilatorWidget',
        'display_name': 'Segment Dilator',
        'description':  'Dilate segments and subtract the dilated mask from targets.',
        'color':        '#6a1b9a',
    },
    {
        'import_name':  'DistanceMeasurer',
        'class_name':   'DistanceMeasurerWidget',
        'display_name': 'Distance Measurer',
        'description':  'Place ruler lines and measure distances in mm / cm / voxels.',
        'color':        '#e65100',
    },
    {
        'import_name':  'PETHotspotNavigator',
        'class_name':   'PETHotspotNavigatorWidget',
        'display_name': 'PET Hotspot Navigator',
        'description':  'Locate the highest-SUV voxel in every segment and jump to it.',
        'color':        '#ad1457',
    },
    {
        'import_name':  'ScribbleTool',
        'class_name':   'ScribbleToolWidget',
        'display_name': 'Scribble Tool',
        'description':  'Draw freehand strokes and place text labels on any slice view.',
        'color':        '#4527a0',
    },
    {
        'import_name':  'VesselSegmenter',
        'class_name':   'VesselSegmenterWidget',
        'display_name': 'Vessel Segmenter',
        'description':  'Segment large vessels (femoral/iliac) using PET blood-pool signal + seed points.',
        'color':        '#00695c',
    },
]


# ── Module metadata ────────────────────────────────────────────────────────────


try:
    from ModuleLauncherLib.ModuleLauncherLogic import ModuleLauncherLogic
except ImportError:
    import importlib.util, os as _os
    _p = _os.path.join(_os.path.dirname(__file__), "ModuleLauncherLib", "ModuleLauncherLogic.py")
    _spec = importlib.util.spec_from_file_location("ModuleLauncherLogic", _p)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    ModuleLauncherLogic = getattr(_mod, "ModuleLauncherLogic")

class ModuleLauncher(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title        = "Module Launcher"
        self.parent.categories   = ["Metabolic Syndrome Toolkit"]
        self.parent.dependencies = []
        self.parent.contributors = ["IshitaSinghFaujdar"]
        self.parent.helpText = (
            "Open any custom module as a floating window.\n"
            "Multiple modules can be open and used simultaneously.\n"
            "Also runs the master batch pipeline (scripts/run_pipeline.py) "
            "from a selected dataset folder."
        )
        self.parent.acknowledgementText = ""


class ModuleLauncherWidget(ScriptedLoadableModuleWidget):

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        self.logic = ModuleLauncherLogic()
        self._pipe_log_queue = []
        self._pipe_log_lock = __import__("threading").Lock()
        self._pipe_done_result = None  # (rc, err) when finished

        # Keep references so windows aren't garbage-collected when closed
        self._open_windows = {}   # import_name → list of open QWidget windows

        self._pipe_poll = qt.QTimer()
        self._pipe_poll.setInterval(250)
        self._pipe_poll.connect('timeout()', self._poll_pipeline)

        # ── Header ────────────────────────────────────────────────────────────
        hdrLbl = qt.QLabel(
            "Click a button to open that module in its own floating window.\n"
            "You can open the same module more than once if needed.")
        hdrLbl.setWordWrap(True)
        hdrLbl.setStyleSheet(
            "color:#37474f; padding:6px; background:#eceff1; border-radius:4px;")
        self.layout.addWidget(hdrLbl)

        # ── Module buttons ────────────────────────────────────────────────────
        for cfg in _MODULES:
            self.layout.addWidget(self._make_module_card(cfg))

        # ── Master pipeline (scripts/run_pipeline.py) ─────────────────────────
        self._build_pipeline_panel()

        # ── Open windows list ─────────────────────────────────────────────────
        winBox = ctk.ctkCollapsibleButton()
        winBox.text = "Open windows"
        winBox.collapsed = True
        self.layout.addWidget(winBox)
        winLayout = qt.QVBoxLayout(winBox)

        self._winListLabel = qt.QLabel("No windows open.")
        self._winListLabel.setStyleSheet("color:#666; font-style:italic;")
        winLayout.addWidget(self._winListLabel)

        closeAllBtn = qt.QPushButton("Close all windows")
        closeAllBtn.setStyleSheet(
            "QPushButton{background:#b71c1c;color:white;font-weight:bold;"
            "padding:5px;border-radius:4px;}"
            "QPushButton:hover{background:#7f0000;}")
        closeAllBtn.clicked.connect(self._close_all)
        winLayout.addWidget(closeAllBtn)

        self.layout.addStretch(1)

    def _build_pipeline_panel(self):
        pipeBox = ctk.ctkCollapsibleButton()
        pipeBox.text = "Run Pipeline (batch CLI)"
        pipeBox.collapsed = False
        self.layout.addWidget(pipeBox)
        form = qt.QFormLayout(pipeBox)

        info = qt.QLabel(
            "Runs scripts/run_pipeline.py: optional organize → generate_segments → "
            "postprocessing → quantification. Select a dataset root folder and output Excel.")
        info.setWordWrap(True)
        info.setStyleSheet("color:#455a64; font-size:11px; padding:4px;")
        form.addRow(info)

        self.pipeRootEdit = ctk.ctkPathLineEdit()
        self.pipeRootEdit.filters = ctk.ctkPathLineEdit.Dirs
        self.pipeRootEdit.setToolTip(
            "Dataset root with CT/ PET/ Segments/ (or destination after organize).")
        form.addRow("Root folder:", self.pipeRootEdit)

        self.pipeSrcEdit = ctk.ctkPathLineEdit()
        self.pipeSrcEdit.filters = ctk.ctkPathLineEdit.Dirs
        self.pipeSrcEdit.setToolTip(
            "Optional Stage 0: inbound raw DICOM / *__Studies folder to organize into Root.")
        form.addRow("Organize from (optional):", self.pipeSrcEdit)

        self.pipeOutEdit = ctk.ctkPathLineEdit()
        self.pipeOutEdit.filters = ctk.ctkPathLineEdit.Files
        self.pipeOutEdit.nameFilters = ["Excel (*.xlsx)", "All files (*)"]
        self.pipeOutEdit.setToolTip("Quantification Excel output path.")
        form.addRow("Output Excel:", self.pipeOutEdit)

        self.pipeCkptEdit = ctk.ctkPathLineEdit()
        self.pipeCkptEdit.filters = ctk.ctkPathLineEdit.Files
        self.pipeCkptEdit.nameFilters = ["Checkpoint (*.ckpt *.pth)", "All files (*)"]
        self.pipeCkptEdit.setToolTip("SegResNet VF checkpoint. Required unless Skip segmentation.")
        default_ckpt = self.logic.default_vf_checkpoint()
        if default_ckpt:
            self.pipeCkptEdit.setCurrentPath(default_ckpt)
        form.addRow("VF checkpoint:", self.pipeCkptEdit)

        self.pipePythonEdit = ctk.ctkPathLineEdit()
        self.pipePythonEdit.filters = ctk.ctkPathLineEdit.Files
        self.pipePythonEdit.nameFilters = ["Python (python.exe python*)", "All files (*)"]
        self.pipePythonEdit.setToolTip(
            "External Python that has TotalSegmentator + torch (NOT Slicer's Python).\n"
            "Example: .../miniconda3/envs/kupetct/python.exe")
        default_py = self.logic.default_pipeline_python()
        if default_py:
            self.pipePythonEdit.setCurrentPath(default_py)
        form.addRow("Pipeline Python:", self.pipePythonEdit)

        self.pipeDeviceEdit = qt.QLineEdit("gpu")
        self.pipeDeviceEdit.setToolTip("'gpu', 'cpu', or device index like '0'.")
        form.addRow("Device:", self.pipeDeviceEdit)

        self.pipeLimitSpin = qt.QSpinBox()
        self.pipeLimitSpin.setRange(0, 9999)
        self.pipeLimitSpin.setValue(0)
        self.pipeLimitSpin.setSpecialValueText("all")
        self.pipeLimitSpin.setToolTip("Max subjects (0 = all).")
        form.addRow("Limit subjects:", self.pipeLimitSpin)

        stageRow = qt.QHBoxLayout()
        self.pipeSkipSeg = qt.QCheckBox("Skip seg")
        self.pipeSkipPost = qt.QCheckBox("Skip post")
        self.pipeSkipQuant = qt.QCheckBox("Skip quant")
        for w in (self.pipeSkipSeg, self.pipeSkipPost, self.pipeSkipQuant):
            stageRow.addWidget(w)
        stageRow.addStretch(1)
        form.addRow("Stages:", stageRow)

        optRow = qt.QHBoxLayout()
        self.pipeRadiomics = qt.QCheckBox("Radiomics")
        self.pipeNoSkipDone = qt.QCheckBox("Force re-run")
        self.pipeNoAppend = qt.QCheckBox("Overwrite Excel")
        self.pipeRadiomics.setToolTip("Pass --radiomics to quantification.")
        self.pipeNoSkipDone.setToolTip("Pass --no-skip-done (recompute even if outputs exist).")
        self.pipeNoAppend.setToolTip("Pass --no-append (overwrite Excel instead of appending).")
        for w in (self.pipeRadiomics, self.pipeNoSkipDone, self.pipeNoAppend):
            optRow.addWidget(w)
        optRow.addStretch(1)
        form.addRow("Options:", optRow)

        self.pipeRunBtn = qt.QPushButton("Run Pipeline")
        self.pipeRunBtn.setStyleSheet(
            "QPushButton{background:#0d47a1;color:white;font-weight:bold;"
            "padding:8px;border-radius:4px;}"
            "QPushButton:hover{background:#1565c0;}"
            "QPushButton:disabled{background:#90a4ae;}")
        self.pipeRunBtn.clicked.connect(self._on_run_pipeline)

        self.pipeStopBtn = qt.QPushButton("Stop")
        self.pipeStopBtn.setEnabled(False)
        self.pipeStopBtn.setStyleSheet(
            "QPushButton{background:#b71c1c;color:white;font-weight:bold;"
            "padding:8px;border-radius:4px;}"
            "QPushButton:disabled{background:#90a4ae;}")
        self.pipeStopBtn.clicked.connect(self._on_stop_pipeline)

        btnRow = qt.QHBoxLayout()
        btnRow.addWidget(self.pipeRunBtn, 1)
        btnRow.addWidget(self.pipeStopBtn)
        form.addRow("", btnRow)

        self.pipeStatus = qt.QLabel("Idle.")
        self.pipeStatus.setStyleSheet("color:#666; font-style:italic;")
        form.addRow("Status:", self.pipeStatus)

        self.pipeLog = qt.QPlainTextEdit()
        self.pipeLog.setReadOnly(True)
        self.pipeLog.setMaximumBlockCount(5000)
        self.pipeLog.setFixedHeight(160)
        self.pipeLog.setStyleSheet(
            "QPlainTextEdit{font-family:Consolas,monospace;font-size:11px;"
            "background:#263238;color:#eceff1;}")
        form.addRow("Log:", self.pipeLog)

    # ── Card builder ──────────────────────────────────────────────────────────

    def _make_module_card(self, cfg):
        card = qt.QFrame()
        card.setFrameShape(qt.QFrame.StyledPanel)
        card.setStyleSheet(
            "QFrame{background:#fafafa; border:1px solid #ccc; border-radius:6px;"
            "margin:2px;}")
        layout = qt.QHBoxLayout(card)
        layout.setContentsMargins(10, 8, 10, 8)

        # Text block
        textBlock = qt.QVBoxLayout()
        titleLbl = qt.QLabel(cfg['display_name'])
        titleLbl.setStyleSheet("font-weight:bold; font-size:13px;")
        descLbl  = qt.QLabel(cfg['description'])
        descLbl.setWordWrap(True)
        descLbl.setStyleSheet("color:#555; font-size:11px;")
        textBlock.addWidget(titleLbl)
        textBlock.addWidget(descLbl)
        layout.addLayout(textBlock, 1)

        # Launch button
        btn = qt.QPushButton("Open ↗")
        btn.setFixedWidth(80)
        color = cfg.get('color', '#1565c0')
        btn.setStyleSheet(
            f"QPushButton{{background:{color};color:white;font-weight:bold;"
            f"padding:6px;border-radius:4px;}}"
            f"QPushButton:hover{{opacity:0.85;}}")
        btn.setToolTip(f"Open {cfg['display_name']} in a floating window")
        btn.clicked.connect(lambda checked=False, c=cfg: self._launch(c))
        layout.addWidget(btn)

        return card

    # ── Launch logic ──────────────────────────────────────────────────────────

    def _launch(self, cfg):
        import_name  = cfg['import_name']
        class_name   = cfg['class_name']
        display_name = cfg['display_name']

        # Import the module
        try:
            mod = __import__(import_name)
        except ImportError as e:
            slicer.util.errorDisplay(
                f"Could not import '{import_name}':\n{e}\n\n"
                f"Make sure {import_name}.py is in your Additional Module Paths.",
                windowTitle="Module Launcher")
            return

        # Get the widget class
        widget_cls = getattr(mod, class_name, None)
        if widget_cls is None:
            slicer.util.errorDisplay(
                f"Class '{class_name}' not found in '{import_name}'.",
                windowTitle="Module Launcher")
            return

        # Create the parent widget ourselves and pass it in.
        # This prevents ScriptedLoadableModuleWidget.__init__ from
        # auto-calling setup() (which some Slicer versions do when
        # parent=None), so setup() runs exactly once — when we call it.
        try:
            parent_widget = slicer.qMRMLWidget()
            parent_widget.setLayout(qt.QVBoxLayout())
            parent_widget.setMRMLScene(slicer.mrmlScene)
            instance = widget_cls(parent_widget)
            instance.setup()
        except Exception as e:
            import traceback
            slicer.util.errorDisplay(
                f"Error setting up '{display_name}':\n{traceback.format_exc()}",
                windowTitle="Module Launcher")
            return

        window = parent_widget
        window.setWindowTitle(display_name)
        window.setWindowFlags(qt.Qt.Window)   # independent, re-sizable window
        window.setMinimumWidth(460)
        window.resize(520, 700)

        # Bring to front and show
        window.show()
        window.raise_()
        window.activateWindow()

        # Track open windows
        if import_name not in self._open_windows:
            self._open_windows[import_name] = []
        self._open_windows[import_name].append(window)

        # Clean up closed windows from our list when this one closes
        window.destroyed.connect(
            lambda obj=None, k=import_name, w=window:
                self._on_window_closed(k, w))

        self._refresh_win_list()

    # ── Master pipeline ───────────────────────────────────────────────────────

    def _pipeline_opts_from_ui(self):
        return {
            "root": self.pipeRootEdit.currentPath,
            "src": self.pipeSrcEdit.currentPath,
            "out": self.pipeOutEdit.currentPath,
            "ckpt": self.pipeCkptEdit.currentPath,
            "python": self.pipePythonEdit.currentPath,
            "device": self.pipeDeviceEdit.text.strip(),
            "limit": int(self.pipeLimitSpin.value),
            "skip_seg": bool(self.pipeSkipSeg.checked),
            "skip_post": bool(self.pipeSkipPost.checked),
            "skip_quant": bool(self.pipeSkipQuant.checked),
            "radiomics": bool(self.pipeRadiomics.checked),
            "no_skip_done": bool(self.pipeNoSkipDone.checked),
            "no_append": bool(self.pipeNoAppend.checked),
        }

    def _on_stop_pipeline(self):
        self.logic.stop_pipeline()
        self.pipeStatus.setText("Stopping…")
        self.pipeStatus.setStyleSheet("color:#e65100; font-weight:bold;")

    def _on_run_pipeline(self):
        import os

        opts = self._pipeline_opts_from_ui()
        root = (opts["root"] or "").strip()
        out = (opts["out"] or "").strip()
        py = (opts["python"] or "").strip()

        if not root or not os.path.isdir(root):
            slicer.util.errorDisplay(
                "Select a valid Root folder (dataset with CT/ PET/ Segments/).",
                windowTitle="Run Pipeline")
            return
        if not out:
            slicer.util.errorDisplay(
                "Select an Output Excel path (.xlsx).",
                windowTitle="Run Pipeline")
            return
        if not py or not os.path.isfile(py):
            slicer.util.errorDisplay(
                "Select Pipeline Python (conda env with TotalSegmentator),\n"
                "e.g. .../miniconda3/envs/kupetct/python.exe\n"
                "Do not use Slicer's Python.",
                windowTitle="Run Pipeline")
            return
        if not opts["skip_seg"]:
            ckpt = (opts["ckpt"] or "").strip()
            if not ckpt or not os.path.isfile(ckpt):
                slicer.util.errorDisplay(
                    "VF checkpoint file not found. Pick a .ckpt or check Skip seg.",
                    windowTitle="Run Pipeline")
                return
        src = (opts["src"] or "").strip()
        if src and not os.path.isdir(src):
            slicer.util.errorDisplay(
                "Organize-from path is set but is not a valid folder.",
                windowTitle="Run Pipeline")
            return

        self.pipeLog.clear()
        self._pipe_log_queue = []
        self._pipe_done_result = None
        self.pipeStatus.setText("Running…")
        self.pipeStatus.setStyleSheet("color:#0d47a1; font-weight:bold;")
        self.pipeRunBtn.setEnabled(False)
        self.pipeStopBtn.setEnabled(True)

        def _log(msg):
            with self._pipe_log_lock:
                self._pipe_log_queue.append(str(msg))

        def _done(rc, err):
            self._pipe_done_result = (rc, err)

        ok = self.logic.run_pipeline_async(opts, log_cb=_log, done_cb=_done)
        if not ok:
            self.pipeRunBtn.setEnabled(True)
            self.pipeStopBtn.setEnabled(False)
            self.pipeStatus.setText("Already running.")
            slicer.util.warningDisplay(
                "A pipeline run is already in progress.",
                windowTitle="Run Pipeline")
            return
        self._pipe_poll.start()

    def _poll_pipeline(self):
        with self._pipe_log_lock:
            batch = self._pipe_log_queue
            self._pipe_log_queue = []
        for line in batch:
            self.pipeLog.appendPlainText(line)
        done = self._pipe_done_result
        if done is None:
            return
        self._pipe_done_result = None
        self._pipe_poll.stop()
        rc, err = done
        self.pipeRunBtn.setEnabled(True)
        self.pipeStopBtn.setEnabled(False)
        if err:
            self.pipeStatus.setText(f"Failed (exception). exit={rc}")
            self.pipeStatus.setStyleSheet("color:#b71c1c; font-weight:bold;")
            slicer.util.errorDisplay(
                f"Pipeline failed:\n{err}",
                windowTitle="Run Pipeline")
        elif rc == 0:
            self.pipeStatus.setText("Finished OK.")
            self.pipeStatus.setStyleSheet("color:#1b5e20; font-weight:bold;")
        else:
            self.pipeStatus.setText(f"Finished with errors (exit={rc}).")
            self.pipeStatus.setStyleSheet("color:#e65100; font-weight:bold;")

    # ── Window tracking ───────────────────────────────────────────────────────

    def _on_window_closed(self, key, window):
        if key in self._open_windows:
            try:
                self._open_windows[key].remove(window)
            except ValueError:
                pass
        self._refresh_win_list()

    def _refresh_win_list(self):
        lines = []
        for key, wins in self._open_windows.items():
            alive = [w for w in wins if w is not None]
            self._open_windows[key] = alive
            if alive:
                lines.append(f"• {key}: {len(alive)} window(s) open")
        if lines:
            self._winListLabel.setText("\n".join(lines))
            self._winListLabel.setStyleSheet("color:#1b5e20;")
        else:
            self._winListLabel.setText("No windows open.")
            self._winListLabel.setStyleSheet("color:#666; font-style:italic;")

    def _close_all(self):
        for wins in self._open_windows.values():
            for w in list(wins):
                try:
                    w.close()
                except Exception:
                    pass
        self._open_windows.clear()
        self._refresh_win_list()

