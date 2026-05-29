"""
ModuleLauncher — 3D Slicer scripted module
==========================================

Drag-and-drop this file into 3D Slicer, then find it under:
    Modules → Utilities → Module Launcher

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
        'import_name':  'UreterPostProcess',
        'class_name':   'UreterPostProcessWidget',
        'display_name': 'Ureter Post-Process',
        'description':  'PET-guided ureter mask, per-organ L1-L5 clipping and cleanup.',
        'color':        '#2e7d32',
    },
    {
        'import_name':  'TotalSegmentatorAdvanced',
        'class_name':   'TotalSegmentatorAdvancedWidget',
        'display_name': 'TotalSegmentator Advanced',
        'description':  'Run TotalSegmentator with task + structure selection.',
        'color':        '#1565c0',
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
]


# ── Module metadata ────────────────────────────────────────────────────────────

class ModuleLauncher(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title        = "Module Launcher"
        self.parent.categories   = ["Utilities"]
        self.parent.dependencies = []
        self.parent.contributors = ["IshitaSinghFaujdar"]
        self.parent.helpText = (
            "Open any custom module as a floating window.\n"
            "Multiple modules can be open and used simultaneously."
        )
        self.parent.acknowledgementText = ""


# ── Widget ─────────────────────────────────────────────────────────────────────

class ModuleLauncherWidget(ScriptedLoadableModuleWidget):

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        # Keep references so windows aren't garbage-collected when closed
        self._open_windows = {}   # import_name → list of open QWidget windows

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

        # Instantiate with parent=None so the widget creates its own
        # qMRMLWidget container, which we then dress up as a window.
        try:
            instance = widget_cls(None)
            instance.setup()
        except Exception as e:
            import traceback
            slicer.util.errorDisplay(
                f"Error setting up '{display_name}':\n{traceback.format_exc()}",
                windowTitle="Module Launcher")
            return

        window = instance.parent
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


# ── Logic ──────────────────────────────────────────────────────────────────────

class ModuleLauncherLogic(ScriptedLoadableModuleLogic):
    pass
