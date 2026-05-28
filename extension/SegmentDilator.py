"""
SegmentDilator — 3D Slicer scripted module
==========================================

Drag-and-drop this file into 3D Slicer, then find it under:
    Modules → Segmentation → Segment Dilator

Workflow
--------
1. Add one or more *Dilation Sources* — each with its own dilation radius.
   The dilated masks are saved to the scene as  <name>_<seg>_dilated  so you
   can inspect them.

2. Add one or more *Target Segments* — the union of all dilated source masks
   is subtracted (voxel-by-voxel) from each target.
   Results are saved as  <name>_<seg>_subtracted.

All original nodes are never modified.
"""

import os
import vtk
import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
)


# ── Module metadata ────────────────────────────────────────────────────────────

class SegmentDilator(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title        = "Segment Dilator"
        self.parent.categories   = ["Segmentation"]
        self.parent.dependencies = []
        self.parent.contributors = ["IshitaSinghFaujdar"]
        self.parent.helpText = (
            "Dilate any number of segmentations by a per-source radius (mm), "
            "then subtract the union of those dilated masks from any number of "
            "target segments.  All originals are untouched."
        )
        self.parent.acknowledgementText = ""


# ── Widget ─────────────────────────────────────────────────────────────────────

class SegmentDilatorWidget(ScriptedLoadableModuleWidget):

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = SegmentDilatorLogic()

        # ══════════════════════════════════════════════════════════════════════
        # Section 1 — Dilation Sources
        # ══════════════════════════════════════════════════════════════════════
        srcBox = ctk.ctkCollapsibleButton()
        srcBox.text = "Dilation Sources  (dilate → save _dilated node)"
        self.layout.addWidget(srcBox)
        srcLayout = qt.QVBoxLayout(srcBox)

        # Column header
        srcHdr = qt.QHBoxLayout()
        for txt, stretch in [("Segmentation", 3), ("Sub-segment", 2),
                              ("Dilation mm", 2), ("", 0)]:
            lbl = qt.QLabel(txt)
            lbl.setStyleSheet("font-weight:bold;")
            srcHdr.addWidget(lbl, stretch)
        srcHdr.addSpacing(30)
        srcLayout.addLayout(srcHdr)

        self._srcRows = []
        self._srcContainer = qt.QWidget()
        self._srcContainerLayout = qt.QVBoxLayout(self._srcContainer)
        self._srcContainerLayout.setContentsMargins(0, 0, 0, 0)
        self._srcContainerLayout.setSpacing(3)
        srcLayout.addWidget(self._srcContainer)

        addSrcBtn = qt.QPushButton("+ Add dilation source")
        addSrcBtn.clicked.connect(lambda: self._add_src_row())
        srcLayout.addWidget(addSrcBtn)

        # ══════════════════════════════════════════════════════════════════════
        # Section 2 — Target Segments
        # ══════════════════════════════════════════════════════════════════════
        tgtBox = ctk.ctkCollapsibleButton()
        tgtBox.text = "Target Segments  (receive subtraction → save _subtracted node)"
        self.layout.addWidget(tgtBox)
        tgtLayout = qt.QVBoxLayout(tgtBox)

        # Column header
        tgtHdr = qt.QHBoxLayout()
        for txt, stretch in [("Segmentation", 3), ("Sub-segment", 2), ("", 0)]:
            lbl = qt.QLabel(txt)
            lbl.setStyleSheet("font-weight:bold;")
            tgtHdr.addWidget(lbl, stretch)
        tgtHdr.addSpacing(30)
        tgtLayout.addLayout(tgtHdr)

        self._tgtRows = []
        self._tgtContainer = qt.QWidget()
        self._tgtContainerLayout = qt.QVBoxLayout(self._tgtContainer)
        self._tgtContainerLayout.setContentsMargins(0, 0, 0, 0)
        self._tgtContainerLayout.setSpacing(3)
        tgtLayout.addWidget(self._tgtContainer)

        addTgtBtn = qt.QPushButton("+ Add target segment")
        addTgtBtn.clicked.connect(lambda: self._add_tgt_row())
        tgtLayout.addWidget(addTgtBtn)

        # ── Status + Run ──────────────────────────────────────────────────────
        self.statusLabel = qt.QLabel("Status: ready")
        self.statusLabel.setWordWrap(True)
        self.layout.addWidget(self.statusLabel)

        self.runButton = qt.QPushButton("Run Dilation + Subtract")
        self.runButton.setStyleSheet(
            "QPushButton{background:#1565c0;color:white;font-weight:bold;"
            "padding:8px;border-radius:4px;font-size:13px;}"
            "QPushButton:hover{background:#0d47a1;}"
            "QPushButton:disabled{background:#888;}"
        )
        self.runButton.clicked.connect(self.onRun)
        self.layout.addWidget(self.runButton)

        self.layout.addStretch(1)

    # ── Source rows ───────────────────────────────────────────────────────────

    def _add_src_row(self, dilate_mm=5.0):
        frame = qt.QFrame()
        row   = qt.QHBoxLayout(frame)
        row.setContentsMargins(0, 1, 0, 1)

        seg_combo = self._make_seg_combo()
        sub_combo = qt.QComboBox()
        sub_combo.addItem("All segments")
        sub_combo.setToolTip("Which sub-segment to dilate, or 'All segments' to merge all")

        dil_spin = qt.QDoubleSpinBox()
        dil_spin.setRange(0.5, 100.0)
        dil_spin.setSingleStep(1.0)
        dil_spin.setDecimals(1)
        dil_spin.setValue(dilate_mm)
        dil_spin.setSuffix(" mm")

        rm_btn = qt.QPushButton("×")
        rm_btn.setFixedWidth(28)
        rm_btn.setStyleSheet("QPushButton{color:red;font-weight:bold;}")
        rm_btn.clicked.connect(lambda: self._remove_row(frame, self._srcRows))

        row.addWidget(seg_combo, 3)
        row.addWidget(sub_combo, 2)
        row.addWidget(dil_spin, 2)
        row.addWidget(rm_btn)

        rd = {'widget': frame, 'seg': seg_combo,
              'sub': sub_combo, 'dilate': dil_spin}
        seg_combo.currentNodeChanged.connect(
            lambda node, r=rd: self._refresh_sub_combo(node, r['sub']))

        self._srcContainerLayout.addWidget(frame)
        self._srcRows.append(rd)

    # ── Target rows ───────────────────────────────────────────────────────────

    def _add_tgt_row(self):
        frame = qt.QFrame()
        row   = qt.QHBoxLayout(frame)
        row.setContentsMargins(0, 1, 0, 1)

        seg_combo = self._make_seg_combo()
        sub_combo = qt.QComboBox()
        sub_combo.addItem("All segments")
        sub_combo.setToolTip(
            "Which sub-segment to subtract from, or 'All segments' to merge all")

        rm_btn = qt.QPushButton("×")
        rm_btn.setFixedWidth(28)
        rm_btn.setStyleSheet("QPushButton{color:red;font-weight:bold;}")
        rm_btn.clicked.connect(lambda: self._remove_row(frame, self._tgtRows))

        row.addWidget(seg_combo, 3)
        row.addWidget(sub_combo, 2)
        row.addWidget(rm_btn)

        rd = {'widget': frame, 'seg': seg_combo, 'sub': sub_combo}
        seg_combo.currentNodeChanged.connect(
            lambda node, r=rd: self._refresh_sub_combo(node, r['sub']))

        self._tgtContainerLayout.addWidget(frame)
        self._tgtRows.append(rd)

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _make_seg_combo(self):
        c = slicer.qMRMLNodeComboBox()
        c.nodeTypes     = ['vtkMRMLSegmentationNode']
        c.addEnabled    = False
        c.removeEnabled = False
        c.noneEnabled   = True
        c.setMRMLScene(slicer.mrmlScene)
        return c

    def _refresh_sub_combo(self, node, combo):
        combo.clear()
        combo.addItem("All segments")
        if node is None:
            return
        seg = node.GetSegmentation()
        for i in range(seg.GetNumberOfSegments()):
            combo.addItem(seg.GetNthSegment(i).GetName())

    def _remove_row(self, frame, row_list):
        for i, rd in enumerate(row_list):
            if rd['widget'] is frame:
                row_list.pop(i)
                break
        frame.setParent(None)

    # ── Run ───────────────────────────────────────────────────────────────────

    def onRun(self):
        # Collect source configs
        src_configs = []
        for rd in self._srcRows:
            node = rd['seg'].currentNode()
            if node is None:
                continue
            sub_text = rd['sub'].currentText
            src_configs.append({
                'seg_node':  node,
                'seg_name':  None if sub_text == "All segments" else sub_text,
                'dilate_mm': rd['dilate'].value,
            })

        # Collect target configs
        tgt_configs = []
        for rd in self._tgtRows:
            node = rd['seg'].currentNode()
            if node is None:
                continue
            sub_text = rd['sub'].currentText
            tgt_configs.append({
                'seg_node': node,
                'seg_name': None if sub_text == "All segments" else sub_text,
            })

        if not src_configs:
            self.statusLabel.setText("Status: add at least one dilation source.")
            return
        if not tgt_configs:
            self.statusLabel.setText("Status: add at least one target segment.")
            return

        self.runButton.setEnabled(False)
        self.statusLabel.setText("Status: running…")
        slicer.app.processEvents()

        try:
            dilated_names, subtracted_names = self.logic.run(src_configs, tgt_configs)
            parts = []
            if dilated_names:
                parts.append(f"Dilated: {', '.join(dilated_names)}")
            if subtracted_names:
                parts.append(f"Subtracted: {', '.join(subtracted_names)}")
            self.statusLabel.setText("Status: done — " + "  |  ".join(parts))
        except Exception:
            import traceback
            self.statusLabel.setText("Status: ERROR — see Python console.")
            print(traceback.format_exc())
        finally:
            self.runButton.setEnabled(True)


# ── Logic ──────────────────────────────────────────────────────────────────────

class SegmentDilatorLogic(ScriptedLoadableModuleLogic):

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self, src_configs, tgt_configs):
        """
        src_configs : list of {'seg_node', 'seg_name', 'dilate_mm'}
        tgt_configs : list of {'seg_node', 'seg_name'}

        For every source:
          - export, dilate, save as <name>_<seg>_dilated in scene

        For every target:
          - export, subtract union of all dilated sources (resampled to target
            space), save as <name>_<seg>_subtracted in scene

        Returns (list_of_dilated_names, list_of_subtracted_names).
        """
        import numpy as np

        print("\n" + "=" * 60)
        print("SEGMENT DILATOR")
        print("=" * 60)

        # ── Step 1: dilate every source, collect (arr, affine) pairs ──────────
        dilated_items  = []   # list of (arr, affine) in source voxel space
        dilated_names  = []

        for i, cfg in enumerate(src_configs):
            node      = cfg['seg_node']
            seg_name  = cfg.get('seg_name')
            dilate_mm = cfg['dilate_mm']
            label     = seg_name or "all"
            print(f"\n[SRC {i+1}] '{node.GetName()}' / '{label}'  "
                  f"dilation={dilate_mm} mm")

            arr, affine, lm = self._export_to_array(node, seg_name)
            before = int((arr > 0).sum())

            dilated = self._dilate(arr, affine, dilate_mm)
            after   = int(dilated.sum())
            print(f"[SRC {i+1}]   voxels: {before} → {after}  (+{after - before})")

            # Save dilated as scene node
            dname = f"{node.GetName()}_{label}_sub_dilated"
            self._save_array_as_seg(dilated, lm, dname)
            dilated_names.append(dname)

            dilated_items.append((dilated, affine))
            slicer.mrmlScene.RemoveNode(lm)

        # ── Step 2: subtract union of dilated sources from every target ────────
        subtracted_names = []

        for j, cfg in enumerate(tgt_configs):
            node     = cfg['seg_node']
            seg_name = cfg.get('seg_name')
            label    = seg_name or "all"
            print(f"\n[TGT {j+1}] '{node.GetName()}' / '{label}'")

            tgt_arr, tgt_affine, tgt_lm = self._export_to_array(node, seg_name)
            before = int((tgt_arr > 0).sum())

            # Build union of all dilated sources resampled to target space
            union = np.zeros(tgt_arr.shape, dtype=np.uint8)
            for src_arr, src_affine in dilated_items:
                resampled = self._resample_to_target(
                    src_arr, src_affine, tgt_arr.shape, tgt_affine)
                union = np.maximum(union, (resampled > 0).astype(np.uint8))

            result = tgt_arr.copy()
            removed = int(((result > 0) & (union > 0)).sum())
            result[(result > 0) & (union > 0)] = 0
            after = int((result > 0).sum())
            print(f"[TGT {j+1}]   voxels: {before} → {after}  "
                  f"(removed {removed} overlapping)")

            # Save subtracted result
            sname = f"{node.GetName()}_{label}_subtracted"
            self._save_array_as_seg(result, tgt_lm, sname)
            subtracted_names.append(sname)

            slicer.mrmlScene.RemoveNode(tgt_lm)

        print("\n" + "=" * 60)
        print("DONE")
        print("=" * 60)
        return dilated_names, subtracted_names

    # ── Internals ─────────────────────────────────────────────────────────────

    def _export_to_array(self, seg_node, seg_name):
        """
        Export seg_node (or one sub-segment) to a labelmap volume.
        Returns (array_uint8, affine_4x4, lm_node).
        The caller is responsible for removing lm_node when done.
        """
        import numpy as np

        lm = slicer.mrmlScene.AddNewNodeByClass(
            'vtkMRMLLabelMapVolumeNode', '_sd_tmp')

        if seg_name:
            seg    = seg_node.GetSegmentation()
            seg_id = None
            for i in range(seg.GetNumberOfSegments()):
                if seg.GetNthSegment(i).GetName() == seg_name:
                    seg_id = seg.GetNthSegmentID(i)
                    break
            if seg_id is None:
                slicer.mrmlScene.RemoveNode(lm)
                raise ValueError(
                    f"Sub-segment '{seg_name}' not found in "
                    f"'{seg_node.GetName()}'")
            ids = vtk.vtkStringArray()
            ids.InsertNextValue(seg_id)
            slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
                seg_node, ids, lm, None,
                slicer.vtkSegmentation.EXTENT_UNION_OF_SEGMENTS)
        else:
            slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(
                seg_node, lm)

        arr = slicer.util.arrayFromVolume(lm).copy().astype(np.uint8)
        mat = vtk.vtkMatrix4x4()
        lm.GetIJKToRASMatrix(mat)
        affine = self._mat_to_np(mat)
        return arr, affine, lm   # lm intentionally NOT removed here

    def _dilate(self, arr, affine, dilate_mm):
        """Ellipsoidal morphological dilation scaled to physical mm per axis."""
        import numpy as np
        from scipy import ndimage

        vox = np.array([abs(affine[0, 0]), abs(affine[1, 1]), abs(affine[2, 2])])
        rx  = max(1, int(round(dilate_mm / vox[0])))
        ry  = max(1, int(round(dilate_mm / vox[1])))
        rz  = max(1, int(round(dilate_mm / vox[2])))
        zz, yy, xx = np.ogrid[-rz:rz + 1, -ry:ry + 1, -rx:rx + 1]
        struct = ((zz / max(rz, 1)) ** 2 +
                  (yy / max(ry, 1)) ** 2 +
                  (xx / max(rx, 1)) ** 2) <= 1.0
        print(f"[DILATOR]   struct axes (vox): Z={rz} Y={ry} X={rx}  "
              f"vox size mm: {np.round(vox, 2)}")
        return ndimage.binary_dilation(arr > 0, structure=struct).astype(np.uint8)

    def _save_array_as_seg(self, arr, reference_lm, name):
        """
        Write *arr* into *reference_lm* (reusing its geometry), then import
        as a new SegmentationNode named *name*.  Any existing node with that
        name is replaced.
        """
        # Remove pre-existing node
        try:
            slicer.mrmlScene.RemoveNode(slicer.util.getNode(name))
        except Exception:
            pass

        slicer.util.updateVolumeFromArray(reference_lm, arr)
        seg = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode', name)
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
            reference_lm, seg)
        seg.CreateClosedSurfaceRepresentation()
        print(f"[DILATOR]   → '{name}' added to scene.")

    def _resample_to_target(self, src_arr, src_affine, tgt_shape, tgt_affine):
        """Nearest-neighbour resample src into tgt voxel space."""
        import numpy as np
        from scipy.ndimage import map_coordinates

        z_idx, y_idx, x_idx = np.meshgrid(
            np.arange(tgt_shape[0]),
            np.arange(tgt_shape[1]),
            np.arange(tgt_shape[2]),
            indexing='ij')
        ijk_hom = np.stack([
            x_idx.ravel(), y_idx.ravel(), z_idx.ravel(),
            np.ones(x_idx.size)], axis=1).astype(np.float32)
        ras_pts  = (tgt_affine  @ ijk_hom.T).T
        src_ijk  = (np.linalg.inv(src_affine) @ ras_pts.T).T[:, :3]
        coords   = np.array([src_ijk[:, 2], src_ijk[:, 1], src_ijk[:, 0]])
        resampled = map_coordinates(
            src_arr.astype(np.float32), coords,
            order=0, mode='constant', cval=0.0)
        return resampled.reshape(tgt_shape)

    @staticmethod
    def _mat_to_np(mat):
        import numpy as np
        return np.array([[mat.GetElement(r, c) for c in range(4)]
                          for r in range(4)])
