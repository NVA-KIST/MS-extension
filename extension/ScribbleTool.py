"""
ScribbleTool — 3D Slicer scripted module
=========================================

Drag-and-drop into 3D Slicer → Modules → Utilities → Scribble Tool

Modes
-----
Draw   Click and drag in any slice view to draw a freehand stroke.
Text   Click anywhere to drop a text label at that position.

All annotations are stored as Markups nodes so they persist with the scene
and are visible in all views.  Toggle visibility or delete per annotation,
or undo the last one, or clear everything at once.
"""

import vtk
import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
)

_PALETTE = [
    (1.00, 0.20, 0.20),   # red
    (1.00, 0.80, 0.00),   # yellow
    (0.20, 1.00, 0.20),   # green
    (0.20, 0.60, 1.00),   # blue
    (1.00, 0.55, 0.00),   # orange
    (1.00, 0.40, 0.80),   # pink
    (1.00, 1.00, 1.00),   # white
    (0.05, 0.05, 0.05),   # black
]


# ── Module metadata ────────────────────────────────────────────────────────────

class ScribbleTool(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title        = "Scribble Tool"
        self.parent.categories   = ["Utilities"]
        self.parent.dependencies = []
        self.parent.contributors = ["IshitaSinghFaujdar"]
        self.parent.helpText = (
            "Draw freehand strokes or drop text labels on any slice view.\n"
            "Draw: click + drag.  Text: click to place.\n"
            "Annotations are Markups nodes — they save with the scene."
        )
        self.parent.acknowledgementText = ""


# ── Widget ─────────────────────────────────────────────────────────────────────

class ScribbleToolWidget(ScriptedLoadableModuleWidget):

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        # State
        self._active               = False
        self._mode                 = 'Draw'
        self._color                = [1.0, 0.20, 0.20]
        self._line_width           = 2.0
        self._stroke_count         = 0
        self._current_stroke_node  = None
        self._event_filters        = []   # list of (view_widget, filter)
        self._annotations          = []   # list of row dicts

        # ── Drawing settings ──────────────────────────────────────────────────
        settingsBox = ctk.ctkCollapsibleButton()
        settingsBox.text = "Drawing Settings"
        self.layout.addWidget(settingsBox)
        settingsLayout = qt.QVBoxLayout(settingsBox)

        # Mode buttons
        modeRow = qt.QHBoxLayout()
        modeRow.addWidget(qt.QLabel("Mode:"))
        self._drawBtn = qt.QPushButton("✏  Draw")
        self._drawBtn.setCheckable(True)
        self._drawBtn.setChecked(True)
        self._drawBtn.setToolTip("Click and drag to draw freehand strokes")
        self._drawBtn.clicked.connect(lambda: self._set_mode('Draw'))
        self._textBtn = qt.QPushButton("T  Text")
        self._textBtn.setCheckable(True)
        self._textBtn.setToolTip("Click to place a text label")
        self._textBtn.clicked.connect(lambda: self._set_mode('Text'))
        modeRow.addWidget(self._drawBtn)
        modeRow.addWidget(self._textBtn)
        modeRow.addStretch(1)
        settingsLayout.addLayout(modeRow)

        # Colour palette
        paletteRow = qt.QHBoxLayout()
        paletteRow.addWidget(qt.QLabel("Colour:"))
        for r, g, b in _PALETTE:
            sw = qt.QPushButton()
            sw.setFixedSize(24, 24)
            ri, gi, bi = int(r * 255), int(g * 255), int(b * 255)
            border = "1px solid #333" if (r + g + b) > 0.3 else "1px solid #aaa"
            sw.setStyleSheet(
                f"QPushButton{{background:rgb({ri},{gi},{bi});"
                f"border:{border};border-radius:3px;}}"
                f"QPushButton:hover{{border:2px solid #000;}}")
            sw.clicked.connect(
                lambda checked=False, c=(r, g, b): self._set_color(c))
            paletteRow.addWidget(sw)
        customBtn = qt.QPushButton("…")
        customBtn.setFixedSize(24, 24)
        customBtn.setToolTip("Choose custom colour")
        customBtn.clicked.connect(self._pick_custom_color)
        paletteRow.addWidget(customBtn)
        paletteRow.addStretch(1)
        settingsLayout.addLayout(paletteRow)

        # Current colour preview + line width
        styleRow = qt.QHBoxLayout()
        styleRow.addWidget(qt.QLabel("Current colour:"))
        self._colorPreview = qt.QLabel("  ████  ")
        self._colorPreview.setStyleSheet("color:rgb(255,51,51); font-size:15px;")
        styleRow.addWidget(self._colorPreview)
        styleRow.addSpacing(20)
        styleRow.addWidget(qt.QLabel("Line width:"))
        self._widthSpin = qt.QDoubleSpinBox()
        self._widthSpin.setRange(0.5, 8.0)
        self._widthSpin.setSingleStep(0.5)
        self._widthSpin.setValue(2.0)
        self._widthSpin.setFixedWidth(65)
        self._widthSpin.setToolTip("Stroke thickness (relative units)")
        self._widthSpin.valueChanged.connect(
            lambda v: setattr(self, '_line_width', v))
        styleRow.addWidget(self._widthSpin)
        styleRow.addStretch(1)
        settingsLayout.addLayout(styleRow)

        # Text input (shown only in Text mode)
        self._textInputRow = qt.QHBoxLayout()
        self._textInputLbl = qt.QLabel("Label text:")
        self._textEdit = qt.QLineEdit("Label")
        self._textEdit.setToolTip("Text that will appear in the view when you click")
        self._textInputRow.addWidget(self._textInputLbl)
        self._textInputRow.addWidget(self._textEdit)
        settingsLayout.addLayout(self._textInputRow)

        # ── Activate toggle ───────────────────────────────────────────────────
        self._activeBtn = qt.QPushButton("✏  Start Drawing")
        self._activeBtn.setCheckable(True)
        self._activeBtn.setStyleSheet(
            "QPushButton{background:#1565c0;color:white;font-weight:bold;"
            "padding:9px;border-radius:5px;font-size:13px;}"
            "QPushButton:checked{background:#b71c1c;}"
            "QPushButton:hover{opacity:0.9;}")
        self._activeBtn.toggled.connect(self._on_toggle_active)
        self.layout.addWidget(self._activeBtn)

        self._hintLabel = qt.QLabel(
            "↑ Activate, then click & drag in any Red / Green / Yellow slice view.")
        self._hintLabel.setWordWrap(True)
        self._hintLabel.setStyleSheet("color:#555; font-style:italic;")
        self.layout.addWidget(self._hintLabel)

        # ── Annotations list ──────────────────────────────────────────────────
        annoBox = ctk.ctkCollapsibleButton()
        annoBox.text = "Annotations"
        self.layout.addWidget(annoBox)
        annoLayout = qt.QVBoxLayout(annoBox)

        self._annoContainer = qt.QWidget()
        self._annoContainerLayout = qt.QVBoxLayout(self._annoContainer)
        self._annoContainerLayout.setContentsMargins(0, 0, 0, 0)
        self._annoContainerLayout.setSpacing(2)
        annoLayout.addWidget(self._annoContainer)

        self._noAnnoLabel = qt.QLabel("No annotations yet.")
        self._noAnnoLabel.setStyleSheet("color:#666; font-style:italic;")
        annoLayout.addWidget(self._noAnnoLabel)

        actRow = qt.QHBoxLayout()
        undoBtn = qt.QPushButton("↩  Undo last")
        undoBtn.setStyleSheet(
            "QPushButton{padding:5px 10px;border-radius:3px;border:1px solid #aaa;}"
            "QPushButton:hover{background:#e3f2fd;}")
        undoBtn.clicked.connect(self._undo_last)
        actRow.addWidget(undoBtn)
        clearBtn = qt.QPushButton("🗑  Clear all")
        clearBtn.setStyleSheet(
            "QPushButton{background:#b71c1c;color:white;font-weight:bold;"
            "padding:5px 10px;border-radius:3px;}"
            "QPushButton:hover{background:#7f0000;}")
        clearBtn.clicked.connect(self._clear_all)
        actRow.addWidget(clearBtn)
        annoLayout.addLayout(actRow)

        self.layout.addStretch(1)

        # Initialise UI state
        self._set_mode('Draw')

        # Watch for layout changes so we can reinstall event filters
        slicer.app.layoutManager().connect(
            'layoutChanged(int)', self._on_layout_changed)

    def cleanup(self):
        try:
            slicer.app.layoutManager().disconnect(
                'layoutChanged(int)', self._on_layout_changed)
        except Exception:
            pass
        self._deactivate()

    # ── Mode ──────────────────────────────────────────────────────────────────

    def _set_mode(self, mode):
        self._mode = mode
        self._drawBtn.setChecked(mode == 'Draw')
        self._textBtn.setChecked(mode == 'Text')
        show_text = (mode == 'Text')
        self._textInputLbl.setVisible(show_text)
        self._textEdit.setVisible(show_text)

    # ── Colour ────────────────────────────────────────────────────────────────

    def _set_color(self, rgb):
        self._color = list(rgb)
        r, g, b = int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        self._colorPreview.setStyleSheet(
            f"color:rgb({r},{g},{b}); font-size:15px;")

    def _pick_custom_color(self):
        r, g, b = [int(c * 255) for c in self._color]
        chosen = qt.QColorDialog.getColor(qt.QColor(r, g, b), None, "Choose colour")
        if chosen.isValid():
            self._set_color([chosen.redF(), chosen.greenF(), chosen.blueF()])

    # ── Activate / Deactivate ─────────────────────────────────────────────────

    def _on_toggle_active(self, checked):
        if checked:
            self._activate()
            self._activeBtn.setText("⏹  Stop Drawing")
            self._hintLabel.setText(
                "ACTIVE — click & drag in any slice view.  "
                "Click Stop to resume normal navigation.")
            self._hintLabel.setStyleSheet(
                "color:#b71c1c; font-weight:bold;")
        else:
            self._deactivate()
            self._activeBtn.setText("✏  Start Drawing")
            self._hintLabel.setText(
                "↑ Activate, then click & drag in any slice view.")
            self._hintLabel.setStyleSheet("color:#555; font-style:italic;")

    def _activate(self):
        self._active = True
        self._install_filters()

    def _deactivate(self):
        self._active = False
        if self._current_stroke_node:
            self._finish_stroke()
        self._uninstall_filters()

    def _install_filters(self):
        self._uninstall_filters()
        lm = slicer.app.layoutManager()
        for name in lm.sliceViewNames():
            sw = lm.sliceWidget(name)
            if sw is None:
                continue
            view = sw.sliceView()
            f = _SliceEventFilter(self, name)
            view.installEventFilter(f)
            self._event_filters.append((view, f))

    def _uninstall_filters(self):
        for view, f in self._event_filters:
            try:
                view.removeEventFilter(f)
            except Exception:
                pass
        self._event_filters = []

    def _on_layout_changed(self, _layout_id):
        if self._active:
            self._install_filters()

    # ── Drawing callbacks (called by the event filter) ────────────────────────

    def _start_stroke(self, view_name, ras):
        self._stroke_count += 1
        node = slicer.mrmlScene.AddNewNodeByClass(
            'vtkMRMLMarkupsCurveNode', f"Stroke {self._stroke_count}")

        # Linear curve — no spline smoothing during live drawing
        try:
            node.SetCurveTypeToLinear()
        except Exception:
            pass

        dn = node.GetDisplayNode()
        r, g, b = self._color
        dn.SetSelectedColor(r, g, b)
        dn.SetColor(r, g, b)
        dn.SetTextScale(0)          # hide point labels
        dn.SetGlyphScale(0.01)      # make control-point markers nearly invisible
        # Line thickness: 0.0 (thin) – 1.0 (thick), map our 0.5-8 spin
        dn.SetLineThickness(min(self._line_width / 8.0, 1.0))
        dn.SetVisibility(True)

        node.AddControlPoint(vtk.vtkVector3d(*ras))
        self._current_stroke_node = node

    def _continue_stroke(self, ras):
        if self._current_stroke_node:
            self._current_stroke_node.AddControlPoint(vtk.vtkVector3d(*ras))

    def _finish_stroke(self):
        node = self._current_stroke_node
        self._current_stroke_node = None
        if node is None:
            return
        if node.GetNumberOfControlPoints() < 2:
            slicer.mrmlScene.RemoveNode(node)
            self._stroke_count -= 1
            return
        self._add_annotation_row(node, 'stroke')

    def _place_text(self, view_name, ras):
        text = self._textEdit.text.strip() or "Label"
        self._stroke_count += 1
        node = slicer.mrmlScene.AddNewNodeByClass(
            'vtkMRMLMarkupsFiducialNode', f"Text {self._stroke_count}")
        node.AddControlPoint(vtk.vtkVector3d(*ras))
        node.SetNthControlPointLabel(0, text)

        dn = node.GetDisplayNode()
        r, g, b = self._color
        dn.SetSelectedColor(r, g, b)
        dn.SetColor(r, g, b)
        dn.SetTextScale(4.5)
        dn.SetGlyphScale(1.0)
        dn.SetVisibility(True)

        self._add_annotation_row(node, 'text', display_label=text)

    # ── Annotation list ───────────────────────────────────────────────────────

    def _add_annotation_row(self, node, kind, display_label=None):
        frame = qt.QFrame()
        frame.setFrameShape(qt.QFrame.StyledPanel)
        row = qt.QHBoxLayout(frame)
        row.setContentsMargins(4, 2, 4, 2)
        row.setSpacing(5)

        # Type icon
        type_lbl = qt.QLabel("✏" if kind == 'stroke' else "T")
        type_lbl.setFixedWidth(16)

        # Colour dot
        r, g, b = [int(c * 255) for c in self._color]
        dot = qt.QLabel("●")
        dot.setStyleSheet(f"color:rgb({r},{g},{b}); font-size:14px;")
        dot.setFixedWidth(18)

        # Name
        name_lbl = qt.QLabel(display_label or node.GetName())
        name_lbl.setToolTip(node.GetName())

        # Visibility toggle
        vis_btn = qt.QPushButton("👁")
        vis_btn.setFixedWidth(28)
        vis_btn.setCheckable(True)
        vis_btn.setChecked(True)
        vis_btn.setToolTip("Toggle visibility")
        vis_btn.toggled.connect(
            lambda on, n=node: n.GetDisplayNode().SetVisibility(on))

        # Delete
        rm_btn = qt.QPushButton("×")
        rm_btn.setFixedWidth(28)
        rm_btn.setStyleSheet(
            "QPushButton{color:red;font-weight:bold;}"
            "QPushButton:hover{background:#ffebee;}")
        rm_btn.clicked.connect(
            lambda checked=False, n=node, f=frame: self._delete_annotation(n, f))

        row.addWidget(type_lbl)
        row.addWidget(dot)
        row.addWidget(name_lbl, 1)
        row.addWidget(vis_btn)
        row.addWidget(rm_btn)

        self._annoContainerLayout.addWidget(frame)
        self._annotations.append({'widget': frame, 'node': node})
        self._noAnnoLabel.setVisible(False)

    def _delete_annotation(self, node, frame):
        try:
            slicer.mrmlScene.RemoveNode(node)
        except Exception:
            pass
        frame.setParent(None)
        self._annotations = [
            a for a in self._annotations if a['widget'] is not frame]
        if not self._annotations:
            self._noAnnoLabel.setVisible(True)

    def _undo_last(self):
        if self._annotations:
            last = self._annotations[-1]
            self._delete_annotation(last['node'], last['widget'])

    def _clear_all(self):
        for a in list(self._annotations):
            self._delete_annotation(a['node'], a['widget'])


# ── Slice view event filter ────────────────────────────────────────────────────

class _SliceEventFilter(qt.QObject):
    """
    Qt event filter installed on each slice view widget.
    Captures left-button mouse events only when the tool is active.
    Normal navigation (right-button pan, scroll zoom) is unaffected.
    """

    _MIN_DIST_SQ = 25   # minimum squared pixel distance between sampled points

    def __init__(self, widget, view_name):
        super().__init__()
        self._w         = widget
        self._view_name = view_name
        self._pressing  = False
        self._last_xy   = None

    def eventFilter(self, obj, event):
        if not self._w._active:
            return False   # pass through — normal Slicer navigation works

        et = event.type()

        # ── Mouse press ───────────────────────────────────────────────────────
        if et == qt.QEvent.MouseButtonPress and event.button() == qt.Qt.LeftButton:
            ras = self._to_ras(event.pos())
            if ras is None:
                return False
            self._pressing = True
            self._last_xy  = (event.pos().x(), event.pos().y())
            if self._w._mode == 'Draw':
                self._w._start_stroke(self._view_name, ras)
            elif self._w._mode == 'Text':
                self._w._place_text(self._view_name, ras)
            return True

        # ── Mouse move (draw only) ────────────────────────────────────────────
        elif et == qt.QEvent.MouseMove and self._pressing \
                and self._w._mode == 'Draw':
            x, y = event.pos().x(), event.pos().y()
            if self._last_xy:
                dx = x - self._last_xy[0]
                dy = y - self._last_xy[1]
                if dx * dx + dy * dy < self._MIN_DIST_SQ:
                    return True   # too close — skip to avoid point overload
            ras = self._to_ras(event.pos())
            if ras:
                self._w._continue_stroke(ras)
            self._last_xy = (x, y)
            return True

        # ── Mouse release ─────────────────────────────────────────────────────
        elif et == qt.QEvent.MouseButtonRelease \
                and event.button() == qt.Qt.LeftButton:
            if self._pressing and self._w._mode == 'Draw':
                self._w._finish_stroke()
            self._pressing = False
            return True

        return False

    def _to_ras(self, qpos):
        """Convert Qt widget pixel position → RAS world coordinates."""
        try:
            sw = slicer.app.layoutManager().sliceWidget(self._view_name)
            if sw is None:
                return None
            sliceNode = sw.mrmlSliceNode()
            h = sw.sliceView().height
            # Qt Y=0 is at the top; VTK expects Y=0 at the bottom
            x = float(qpos.x())
            y = float(h - qpos.y())
            mat = sliceNode.GetXYToRAS()          # vtkMatrix4x4
            h4  = mat.MultiplyPoint([x, y, 0.0, 1.0])
            return [h4[0], h4[1], h4[2]]
        except Exception as e:
            print(f"[SCRIBBLE] Coordinate conversion error ({self._view_name}): {e}")
            return None


# ── Logic ──────────────────────────────────────────────────────────────────────

class ScribbleToolLogic(ScriptedLoadableModuleLogic):
    pass
