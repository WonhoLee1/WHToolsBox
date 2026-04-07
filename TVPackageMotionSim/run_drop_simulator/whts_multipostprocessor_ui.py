# -*- coding: utf-8 -*-
"""
[WHTOOLS] Integrated Multi-PostProcessor UI
사용자 인터페이스(Qt), 3D 시각화(PyVista), 2D 그래프(Matplotlib)를 담당하는 모듈입니다.
해석 엔진(whts_multipostprocessor_engine)과 연동하여 대시보드를 구성합니다.
"""

import os
import sys
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6 import QtWidgets, QtCore, QtGui

# [WHTOOLS] 현재 디렉토리를 경로에 추가 (부모 디렉토리에서 임포트 대비)
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.append(_current_dir)

# 엔진 모듈 임포트
from whts_multipostprocessor_engine import (
    PlateAssemblyManager, 
    ShellDeformationAnalyzer, 
    PlateConfig
)

@dataclass
class PlotSlotConfig:
    """[WHTOOLS] 2D 그래프 슬롯별 시각화 구성 설정"""
    part_idx: int = 0
    plot_type: str = 'contour'
    data_key: str = 'Displacement [mm]'

@dataclass
class DashboardConfig:
    """[WHTOOLS] 통합 대시보드 레이아웃 및 제어 설정"""
    layout_2d: str = '2x2'
    plots_2d: List[PlotSlotConfig] = field(default_factory=list)
    v_font_size: int = 9
    animation_step: int = 1
    animation_speed_ms: int = 30

class VisibilityToolWindow(QtWidgets.QWidget):
    """
    [WHTOOLS] 가시성 관리자 (Visibility Manager)
    트리 구조를 이용한 파트별 메쉬/마커 가시성 및 실시간 Info(Min/Max) 제공
    """
    def __init__(self, parent=None):
        super().__init__(parent, QtCore.Qt.Window | QtCore.Qt.WindowStaysOnTopHint)
        self.setWindowTitle("Visibility Manager")
        self.resize(400, 600)
        self.parent = parent
        
        layout = QtWidgets.QVBoxLayout(self)
        
        # Global Control
        global_group = QtWidgets.QGroupBox("Global Control")
        global_layout = QtWidgets.QVBoxLayout(global_group)
        for title, col_idx in [("Mesh", 1), ("Markers", 2)]:
            h_layout = QtWidgets.QHBoxLayout()
            h_layout.addWidget(QtWidgets.QLabel(f"{title}:"))
            btn_show = QtWidgets.QPushButton("Show All")
            btn_hide = QtWidgets.QPushButton("Hide All")
            btn_show.clicked.connect(partial(self._bulk_set, col_idx, True))
            btn_hide.clicked.connect(partial(self._bulk_set, col_idx, False))
            h_layout.addWidget(btn_show); h_layout.addWidget(btn_hide)
            global_layout.addLayout(h_layout)
        layout.addWidget(global_group)
        
        # Tree Widget
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Part", "Mesh", "Markers", "Info (Min / Max)"])
        self.tree.setColumnWidth(0, 150)
        self.tree.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.tree)
        
        self.groups = {}
        self.id_to_item = {}
        self._init_tree()

    def _init_tree(self):
        self.tree.blockSignals(True)
        if not self.parent or not self.parent.mgr: return
        
        for i, part in enumerate(self.parent.mgr.analyzers):
            prefix = part.name.split('_')[0] if "_" in part.name else part.name
            if prefix not in self.groups:
                self.groups[prefix] = QtWidgets.QTreeWidgetItem(self.tree, [prefix])
                self.groups[prefix].setExpanded(True)
                
            n_markers = part.m_raw.shape[1] if part.m_raw is not None else 0
            item = QtWidgets.QTreeWidgetItem(self.groups[prefix], [part.name])
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            item.setData(0, QtCore.Qt.UserRole, i)
            
            actor_data = self.parent.part_actors.get(i, {})
            m_v = QtCore.Qt.Checked if actor_data.get('visible', True) else QtCore.Qt.Unchecked
            mk_v = QtCore.Qt.Checked if actor_data.get('visible_markers', False) else QtCore.Qt.Unchecked
            
            item.setCheckState(1, m_v)
            item.setCheckState(2, mk_v)
            item.setText(2, f"Markers ({n_markers})")
            item.setText(3, "-")
            self.id_to_item[i] = item
        self.tree.blockSignals(False)
        self.update_info()

    def _bulk_set(self, column: int, state: bool):
        self.tree.blockSignals(True)
        cs = QtCore.Qt.Checked if state else QtCore.Qt.Unchecked
        for i in range(self.tree.topLevelItemCount()):
            g = self.tree.topLevelItem(i)
            g.setCheckState(column, cs)
            for j in range(g.childCount()): g.child(j).setCheckState(column, cs)
        self.tree.blockSignals(False)
        self._apply()

    def _on_item_changed(self, item, column):
        self.tree.blockSignals(True)
        if item.parent() is None:
            cs = item.checkState(column)
            for j in range(item.childCount()): item.child(j).setCheckState(column, cs)
        else:
            p = item.parent(); all_c = True
            for j in range(p.childCount()):
                if p.child(j).checkState(column) == QtCore.Qt.Unchecked: all_c = False; break
            p.setCheckState(column, QtCore.Qt.Checked if all_c else QtCore.Qt.Unchecked)
        self.tree.blockSignals(False); self._apply()

    def update_info(self):
        if not self.parent or not hasattr(self.parent, 'cmb_3d'): return
        f_idx, fk = self.parent.current_frame, self.parent.cmb_3d.currentText()
        self.tree.blockSignals(True)
        for i, item in self.id_to_item.items():
            ana = self.parent.mgr.analyzers[i]
            if fk in ana.results and fk not in ["Body Color", "Face Color"]:
                val = ana.results[fk][f_idx]
                item.setText(3, f"{val.min():.2e} / {val.max():.2e}")
            else: item.setText(3, "-")
        self.tree.blockSignals(False)

    def _apply(self):
        for i, item in self.id_to_item.items():
            if i in self.parent.part_actors:
                self.parent.part_actors[i]['visible'] = (item.checkState(1) == QtCore.Qt.Checked)
                self.parent.part_actors[i]['visible_markers'] = (item.checkState(2) == QtCore.Qt.Checked)
        self.parent.update_frame(self.parent.current_frame)

class AddPlotDialog(QtWidgets.QDialog):
    def __init__(self, slot_idx, parts, field_keys, stat_keys, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Add Plot to Slot {slot_idx + 1}")
        l = QtWidgets.QVBoxLayout(self); g = QtWidgets.QGridLayout(); l.addLayout(g)
        g.addWidget(QtWidgets.QLabel("Part:"), 0, 0); self.cmb_part = QtWidgets.QComboBox(); self.cmb_part.addItems(parts); g.addWidget(self.cmb_part, 0, 1)
        g.addWidget(QtWidgets.QLabel("Type:"), 1, 0); hb = QtWidgets.QHBoxLayout(); self.rb_c = QtWidgets.QRadioButton("Contour"); self.rb_cur = QtWidgets.QRadioButton("Curve"); self.rb_c.setChecked(True); hb.addWidget(self.rb_c); hb.addWidget(self.rb_cur); g.addLayout(hb, 1, 1)
        g.addWidget(QtWidgets.QLabel("Key:"), 2, 0); self.cmb_key = QtWidgets.QComboBox(); g.addWidget(self.cmb_key, 2, 1)
        self.f_keys, self.s_keys = field_keys, stat_keys
        self.rb_c.toggled.connect(self._update_keys); self.rb_cur.toggled.connect(self._update_keys); self._update_keys()
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject); l.addWidget(bb)
    def _update_keys(self):
        self.cmb_key.clear()
        if self.rb_c.isChecked(): self.cmb_key.addItems(self.f_keys)
        else: self.cmb_key.addItems(self.s_keys)
    def get_config(self) -> PlotSlotConfig:
        return PlotSlotConfig(part_idx=self.cmb_part.currentIndex(), plot_type="contour" if self.rb_c.isChecked() else "curve", data_key=self.cmb_key.currentText())

class AboutDialog(QtWidgets.QDialog):
    def __init__(self, logo_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About WHTOOLS Dashboard"); self.setFixedSize(550, 650)
        l = QtWidgets.QVBoxLayout(self); l.setContentsMargins(40,40,40,40); l.setSpacing(20)
        if os.path.exists(logo_path):
            il = QtWidgets.QLabel(); px = QtGui.QPixmap(logo_path).scaledToHeight(220, QtCore.Qt.SmoothTransformation); il.setPixmap(px); il.setAlignment(QtCore.Qt.AlignCenter); l.addWidget(il)
        t = QtWidgets.QLabel("WHTOOLS Structural Dashboard v5.9"); t.setStyleSheet("font-size: 20pt; font-weight: bold; color: #1A73E8;"); t.setAlignment(QtCore.Qt.AlignCenter); l.addWidget(t)
        st = QtWidgets.QLabel("Expert Structural Analysis & Digital Twin Solution"); st.setStyleSheet("font-size: 11pt; color: #5F6368; font-style: italic;"); st.setAlignment(QtCore.Qt.AlignCenter); l.addWidget(st)
        line = QtWidgets.QFrame(); line.setFrameShape(QtWidgets.QFrame.HLine); l.addWidget(line)
        f = QtWidgets.QLabel("<b>Advanced Computational Core:</b><br>• <b>Multi-Theory Shell Solver:</b> Kirchhoff / Mindlin / Von Karman<br>• <b>JAX-SSR Engine:</b> Ultra-fast surface reconstruction<br>• <b>Autonomous Alignment:</b> SVD-based plane fitting<br>• <b>Expert Visualization:</b> Multi-slot 3D/2D interaction")
        f.setStyleSheet("font-size: 11pt; line-height: 170%; color: #3C4043;"); f.setWordWrap(True); l.addWidget(f)
        l.addStretch()
        cl = QtWidgets.QLabel("© 2026 WHTOOLS. All Rights Reserved."); cl.setStyleSheet("font-size: 9pt; color: #9AA0A6;"); cl.setAlignment(QtCore.Qt.AlignCenter); l.addWidget(cl)
        bc = QtWidgets.QPushButton("Close"); bc.setFixedWidth(120); bc.setStyleSheet("padding: 10px; font-weight: bold;"); bc.clicked.connect(self.accept)
        hl = QtWidgets.QHBoxLayout(); hl.addStretch(); hl.addWidget(bc); hl.addStretch(); l.addLayout(hl)

class QtVisualizerV2(QtWidgets.QMainWindow):
    """
    [WHTOOLS] 차세대 구조 변형 분석 대시보드 (V2)
    VTK 기반 3D 뷰어와 Matplotlib 기반 2D 그래프를 결합한 통합 분석 플랫폼
    """
    def __init__(self, manager: PlateAssemblyManager, config: DashboardConfig = None, ground_size=(4000, 4000)):
        super().__init__()
        print(f"[WHTOOLS-UI] Initializing Dashboard with {len(manager.analyzers)} parts...")
        self.mgr, self.cfg, self.ground_size = manager, config or DashboardConfig(), ground_size
        self.current_frame, self.is_playing, self.active_slot = 0, False, 0
        self.anim_step = self.cfg.animation_step
        self.plot_slots: List[Optional[PlotSlotConfig]] = [None] * 6
        self.part_actors, self.v_font_size = {}, self.cfg.v_font_size
        self.ims, self.vls = [None]*6, [None]*6
        
        # Floor state
        self.floor_origin = [0, 0, 0]
        self.floor_normal = [0, 0, 1]
        self.floor_w, self.floor_h = ground_size
        
        if not manager.analyzers:
            print("❌ No analyzers provided to dashboard!")
            return
            
        p0 = manager.analyzers[0]; n_f = len(self.mgr.times); res_sq = p0.sol.res**2
        self.field_keys = [k for k in p0.results if p0.results[k].ndim == 3 and p0.results[k].size // n_f == res_sq]
        self.stat_keys = [k for k in p0.results if k not in self.field_keys and p0.results[k].ndim < 3] + ['Marker Local Disp. [mm]', 'Marker Global Disp. [mm]']
        
        self.res_dir = os.path.join(os.path.dirname(__file__), "resources")
        self.logo_path = os.path.join(self.res_dir, "logo.png")

        self.statusBar().showMessage("WHTOOLS Ready")
        self.visibility_tool = VisibilityToolWindow(self)
        self._init_ui(); self._init_3d_view(); self._init_2d_plots(); self.update_frame(0)

    def _init_ui(self):
        self.setWindowTitle("WHTOOLS Structural Dashboard v5.9"); self.resize(1700, 1020)
        # Menu Bar 삭제 (User Request)
        
        cw = QtWidgets.QWidget(); self.setCentralWidget(cw); ml = QtWidgets.QVBoxLayout(cw)
        ml.setContentsMargins(0, 0, 0, 0); ml.setSpacing(0)
        
        tc = QtWidgets.QWidget(); tl = QtWidgets.QHBoxLayout(tc); tl.setContentsMargins(0,0,0,0); tl.setSpacing(0)
        
        # Logo 크기 확대 (130px)
        self.lbl_logo = QtWidgets.QLabel()
        if os.path.exists(self.logo_path): self.lbl_logo.setPixmap(QtGui.QPixmap(self.logo_path).scaledToHeight(130, QtCore.Qt.SmoothTransformation))
        else: self.lbl_logo.setText("WHTOOLS"); self.lbl_logo.setStyleSheet("font-weight: bold; font-size: 22pt; color: #1A73E8; margin-left: 10px;")
        tl.addWidget(self.lbl_logo)
        
        # Tab Widget 높이 확장 및 스타일 최적화 (Slim Header)
        self.ct = QtWidgets.QTabWidget(); self.t3, self.t2, self.ts = QtWidgets.QWidget(), QtWidgets.QWidget(), QtWidgets.QWidget()
        self.ct.setStyleSheet("QTabWidget::pane { border: 0; top: -1px; } QTabBar::tab { height: 35px; padding: 0 20px; }")
        self.ct.addTab(self.t3, "🧊 3D Field")
        self.ct.addTab(self.t2, "📈 2D Field & Curves")
        self.ct.addTab(self.ts, "⚙️ Settings")
        tl.addWidget(self.ct, stretch=1); ml.addWidget(tc)
        
        self.split = QtWidgets.QSplitter(QtCore.Qt.Horizontal); ml.addWidget(self.split, stretch=1)
        self.p3d = QtWidgets.QWidget(); l3 = QtWidgets.QVBoxLayout(self.p3d); l3.setContentsMargins(0,0,0,0); self.v_int = QtInteractor(self.p3d); self.v_int.setContextMenuPolicy(QtCore.Qt.CustomContextMenu); self.v_int.customContextMenuRequested.connect(self._show_part_menu); l3.addWidget(self.v_int)
        
        self.p2d = QtWidgets.QWidget(); l2 = QtWidgets.QVBoxLayout(self.p2d); l2.setContentsMargins(0,0,0,0); 
        # 2D 패널 내부 레이아웃
        self._cw = QtWidgets.QWidget(); self._cl = QtWidgets.QVBoxLayout(self._cw); self._cl.setContentsMargins(0,0,0,0); l2.addWidget(self._cw, stretch=1)
        
        self.split.addWidget(self.p3d); self.split.addWidget(self.p2d)
        # 초기 화면 분할 비율 2:3 강제 설정 (setSizes)
        self.split.setSizes([680, 1020]) 
        self.split.setStretchFactor(0, 2); self.split.setStretchFactor(1, 3)
        
        self._init_3d_controls(self.t3)
        self._init_2d_controls(self.t2)
        self._init_settings_controls(self.ts)
        self._init_animation_dock()

    def _init_settings_controls(self, p):
        l = QtWidgets.QHBoxLayout(p)
        g1 = QtWidgets.QGroupBox("General Settings"); gl1 = QtWidgets.QGridLayout(g1)
        b_vis = QtWidgets.QPushButton("Visibility Manager"); b_vis.clicked.connect(self.visibility_tool.show); gl1.addWidget(b_vis, 0, 0)
        b_res = QtWidgets.QPushButton("Reset Camera (f)"); b_res.clicked.connect(lambda: self.v_int.reset_camera()); gl1.addWidget(b_res, 0, 1)
        b_abt = QtWidgets.QPushButton("About WHTOOLS"); b_abt.clicked.connect(self._show_about); gl1.addWidget(b_abt, 1, 0, 1, 2)
        l.addWidget(g1)
        
        g2 = QtWidgets.QGroupBox("Animation Settings"); gl2 = QtWidgets.QGridLayout(g2)
        gl2.addWidget(QtWidgets.QLabel("Skip Frames (Step):"), 0, 0); self.sp_step = QtWidgets.QSpinBox(); self.sp_step.setRange(1, 100); self.sp_step.setValue(self.anim_step); self.sp_step.valueChanged.connect(self._update_step); gl2.addWidget(self.sp_step, 0, 1)
        l.addWidget(g2); l.addStretch()

    def _update_step(self, v): self.anim_step = v

    def _init_3d_controls(self, p):
        l = QtWidgets.QHBoxLayout(p)
        g1 = QtWidgets.QGroupBox("View & Deformation"); gl1 = QtWidgets.QGridLayout(g1)
        gl1.addWidget(QtWidgets.QLabel("View:"), 0, 0); self.cmb_v = QtWidgets.QComboBox(); self.cmb_v.addItems(["Global", "Local"]); self.cmb_v.currentTextChanged.connect(lambda: self.update_frame(self.current_frame)); gl1.addWidget(self.cmb_v, 0, 1)
        gl1.addWidget(QtWidgets.QLabel("Scale:"), 0, 2); self.sp_sc = QtWidgets.QDoubleSpinBox(); self.sp_sc.setRange(1.0, 1000.0); self.sp_sc.setValue(1.0); self.sp_sc.valueChanged.connect(lambda: self.update_frame(self.current_frame)); gl1.addWidget(self.sp_sc, 0, 3)
        self.ch_per = QtWidgets.QCheckBox("Perspective"); self.ch_per.setChecked(True); self.ch_per.toggled.connect(self._on_persp_toggled); gl1.addWidget(self.ch_per, 1, 0, 1, 2)
        gl1.addWidget(QtWidgets.QLabel("Background:"), 1, 2); self.cmb_bg = QtWidgets.QComboBox(); self.cmb_bg.addItems(["White", "Grey Grad.", "Sky Grad."]); self.cmb_bg.currentTextChanged.connect(self._on_bg_changed); gl1.addWidget(self.cmb_bg, 1, 3)
        l.addWidget(g1)
        
        g2 = QtWidgets.QGroupBox("Field & Range"); gl2 = QtWidgets.QGridLayout(g2)
        gl2.addWidget(QtWidgets.QLabel("Field:"), 0, 0); self.cmb_3d = QtWidgets.QComboBox(); self.cmb_3d.addItems(["Body Color", "Face Color"] + self.field_keys); self.cmb_3d.currentTextChanged.connect(self._on_field_changed); gl2.addWidget(self.cmb_3d, 0, 1)
        gl2.addWidget(QtWidgets.QLabel("Range:"), 0, 2); self.cmb_l = QtWidgets.QComboBox(); self.cmb_l.addItems(["Dynamic", "Static"]); self.cmb_l.currentTextChanged.connect(self._on_legend_mode_changed); gl2.addWidget(self.cmb_l, 0, 3)
        
        # Min/Max Note를 Field & Range 내부로 이동하여 라벨 왼쪽에 체크박스 배치
        hmin = QtWidgets.QHBoxLayout(); self.ch_min = QtWidgets.QCheckBox(""); self.ch_min.setFixedWidth(20); self.ch_min.toggled.connect(lambda: self.update_frame(self.current_frame)); hmin.addWidget(self.ch_min); hmin.addWidget(QtWidgets.QLabel("Min:")); self.sp_min = QtWidgets.QDoubleSpinBox(); self.sp_min.setRange(-1e9, 1e9); self.sp_min.setDecimals(4); self.sp_min.valueChanged.connect(lambda: self.update_frame(self.current_frame)); hmin.addWidget(self.sp_min); gl2.addLayout(hmin, 1, 0, 1, 2)
        
        hmax = QtWidgets.QHBoxLayout(); self.ch_max = QtWidgets.QCheckBox(""); self.ch_max.setFixedWidth(20); self.ch_max.toggled.connect(lambda: self.update_frame(self.current_frame)); hmax.addWidget(self.ch_max); hmax.addWidget(QtWidgets.QLabel("Max:")); self.sp_max = QtWidgets.QDoubleSpinBox(); self.sp_max.setRange(-1e9, 1e9); self.sp_max.setDecimals(4); self.sp_max.valueChanged.connect(lambda: self.update_frame(self.current_frame)); hmax.addWidget(self.sp_max); gl2.addLayout(hmax, 1, 2, 1, 2)
        l.addWidget(g2); l.addStretch()

    def _init_2d_controls(self, p):
        l = QtWidgets.QHBoxLayout(p)
        g1 = QtWidgets.QGroupBox("2D Slot Layout"); gl1 = QtWidgets.QGridLayout(g1)
        self.cmb_lay = QtWidgets.QComboBox(); self.cmb_lay.addItems(["1x1", "1x2", "2x2", "3x2"]); self.cmb_lay.setCurrentText(self.cfg.layout_2d); self.cmb_lay.currentTextChanged.connect(self._init_2d_plots); gl1.addWidget(QtWidgets.QLabel("Grid:"), 0, 0); gl1.addWidget(self.cmb_lay, 0, 1)
        bt_add = QtWidgets.QPushButton("+ Add Plot"); bt_add.clicked.connect(self._show_add_plot_dialog); gl1.addWidget(bt_add, 1, 0, 1, 2)
        l.addWidget(g1)
        
        g2 = QtWidgets.QGroupBox("Display Options"); gl2 = QtWidgets.QVBoxLayout(g2)
        self.checks = {}
        for t, s in [("Sync Timeline", True), ("Interpolation", True)]:
            c = QtWidgets.QCheckBox(t); c.setChecked(s); c.toggled.connect(lambda: self.update_frame(self.current_frame)); gl2.addWidget(c); self.checks[t.split()[0]] = c
        bt_pop = QtWidgets.QPushButton("Pop-out View"); bt_pop.clicked.connect(self._pop_out_2d); gl2.addWidget(bt_pop); l.addWidget(g2); l.addStretch()

    def _init_animation_dock(self):
        self.ad = QtWidgets.QDockWidget("Animation Control"); self.ad.setAllowedAreas(QtCore.Qt.BottomDockWidgetArea)
        cn = QtWidgets.QWidget(); l = QtWidgets.QHBoxLayout(cn); l.setContentsMargins(10,5,10,5)
        for t, s in [("<<", 0), ("<", -1), (">", 1), (">>", 9999)]:
            b = QtWidgets.QPushButton(t); b.setFixedSize(35, 30); b.clicked.connect(partial(self._ctrl_slot, s)); l.addWidget(b)
        self.bp = QtWidgets.QPushButton("▶"); self.bp.setFixedSize(45, 30); self.bp.clicked.connect(lambda: self._ctrl_slot(-2)); l.addWidget(self.bp)
        n_frames = len(self.mgr.times) if self.mgr.times is not None else 1
        self.sld = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.sld.setRange(0, n_frames - 1); self.sld.valueChanged.connect(self.update_frame); l.addWidget(self.sld, stretch=1)
        self.lf = QtWidgets.QLabel(f"Frame: 0 / {n_frames-1}"); self.lf.setFixedWidth(150); l.addWidget(self.lf)
        l.addWidget(QtWidgets.QLabel(" Speed Control (ms):")); self.cs = QtWidgets.QComboBox(); self.cs.addItems(["0", "15", "30", "50", "100"]); self.cs.setCurrentText("30"); self.cs.currentTextChanged.connect(lambda v: self.timer.setInterval(int(v))); l.addWidget(self.cs)
        self.ad.setWidget(cn); self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.ad)

    def _init_3d_view(self):
        self.v_int.set_background("white"); self.v_int.add_axes()
        gp = pv.Plane(center=self.floor_origin, direction=self.floor_normal, i_size=self.floor_w, j_size=self.floor_h); self.ground = self.v_int.add_mesh(gp, color="blue", opacity=0.1)
        self.lut = pv.LookupTable(cmap="turbo"); self.lut.below_range_color = 'lightgrey'; self.lut.above_range_color = 'magenta'
        for i, ana in enumerate(self.mgr.analyzers):
            if ana.m_raw is None: continue
            poly = pv.Plane(i_size=ana.W, j_size=ana.H, i_resolution=ana.sol.res-1, j_resolution=ana.sol.res-1)
            ma = self.v_int.add_mesh(poly, scalars=None, cmap=self.lut, show_edges=True, edge_color="darkgray", show_scalar_bar=False)
            mp = pv.PolyData(np.array(ana.m_raw[0])); n_m = ana.m_raw.shape[1]; mp.point_data["names"] = [f"{ana.name}_M{j:02d}" for j in range(n_m)]
            mka = self.v_int.add_mesh(mp, render_points_as_spheres=True, point_size=10, color='skyblue')
            la = self.v_int.add_point_labels(mp, "names", font_size=self.v_font_size, text_color='black', always_visible=True, point_size=0, shadow=False)
            # [WHTOOLS] 해석 성공 여부 및 메쉬 속성 존재 확인 (AttributeError 방지)
            if not hasattr(ana.sol, 'X_mesh') or ana.sol.X_mesh is None:
                print(f"  > [Warning] {ana.name} has no mesh data. Skipping 3D visualization for this part.")
                self.part_actors[i] = {'mesh': None, 'visible': False}
                continue
                
            mka.SetVisibility(False); la.SetVisibility(False); self.part_actors[i] = {'mesh': ma, 'poly': poly, 'm_poly': mp, 'markers': mka, 'labels': la, 'visible': True, 'visible_markers': False, 'p_base': np.column_stack([ana.sol.X_mesh.ravel(), ana.sol.Y_mesh.ravel(), np.zeros(ana.sol.res**2)])}
        if self.part_actors:
            f_i = min(self.part_actors.keys()); 
            # 컬러바 스타일 복구: 정밀도 %.3f, 라벨 7개
            self.sb = self.v_int.add_scalar_bar("Field Analysis [mm]", position_x=0.15, position_y=0.05, width=0.7, mapper=self.part_actors[f_i]['mesh'].mapper, vertical=False, n_labels=7, fmt="%.3f", label_font_size=10)
        else: self.sb = self.v_int.add_scalar_bar("No Data", position_x=0.15); self.sb.SetVisibility(False)
        self.ov = self.v_int.add_text("-", position='upper_right', font_size=9, color='black')
        self.gui_txt = self.v_int.add_text("[Space]: Play/Pause | [R]: Reset | [W]: Wireframe", position='upper_left', font_size=9, color='black')
        self.v_int.view_isometric(); 
        # Perspective 모드 고정 (ParallelProjection Off)
        self.v_int.camera.ParallelProjectionOff()
        self.timer = QtCore.QTimer(); self.timer.setInterval(30); self.timer.timeout.connect(lambda: self._ctrl_slot(1))

    def _init_2d_plots(self):
        for i in reversed(range(self._cl.count())):
            item = self._cl.itemAt(i)
            if item.widget(): item.widget().setParent(None)
        plt.rcParams['font.family'] = 'Consolas'; plt.rcParams['font.size'] = 9
        self.fig = Figure(figsize=(8, 8)); self.can = FigureCanvas(self.fig); self._cl.addWidget(NavigationToolbar(self.can, self)); self._cl.addWidget(self.can)
        self.can.mpl_connect('button_press_event', self._on_axis_clicked)
        lm = {"1x1": (1, 1), "1x2": (1, 2), "2x2": (2, 2), "3x2": (3, 2)}; r, c = lm.get(self.cmb_lay.currentText(), (2, 2))
        self.axes, self.ims, self.vls = [], [None]*6, [None]*6; self.fig.clear(); self.fig.subplots_adjust(hspace=0.4, wspace=0.3)
        for i in range(r*c): ax = self.fig.add_subplot(r, c, i+1); self.axes.append(ax); ax.text(0.5, 0.5, f"Slot {i+1}", ha='center', transform=ax.transAxes)
        self._update_selection_ui(); self.can.draw_idle()

    def update_frame(self, f_i: int):
        self.current_frame = f_i; self.sld.blockSignals(True); self.sld.setValue(f_i); self.sld.blockSignals(False)
        n_frames_tot = len(self.mgr.times) if self.mgr.times is not None else 1
        self.lf.setText(f"Frame: {f_i} / {n_frames_tot-1}"); vm, fk, sc = self.cmb_v.currentText(), self.cmb_3d.currentText(), self.sp_sc.value(); dy = self.cmb_l.currentText()=="Dynamic"
        av = []
        for i, ana in enumerate(self.mgr.analyzers):
            if i not in self.part_actors: continue
            inf = self.part_actors[i]; mv, mkv = inf['visible'], inf['visible_markers']
            inf['mesh'].SetVisibility(mv); inf['markers'].SetVisibility(mkv); inf['labels'].SetVisibility(mkv)
            if not mv and not mkv: continue
            
            wd = ana.results.get('Displacement [mm]', np.zeros((n_frames_tot, ana.sol.res, ana.sol.res)))[f_i]
            pd = inf['p_base'].copy(); pd[:, 2] = wd.ravel() * sc
            
            R_hist = ana.results.get('R', np.repeat(np.eye(3)[None,...], n_frames_tot, axis=0))
            cq_hist = ana.results.get('c_Q', np.zeros((n_frames_tot, 3)))
            R, cq, b = R_hist[f_i], cq_hist[f_i], ana.ref_basis
            
            if vm == "Global":
                inf['poly'].points = (pd @ b) @ R + cq
                inf['m_poly'].points = np.array(ana.m_raw[f_i])
            else:
                inf['poly'].points = pd
                inf['m_poly'].points = np.array(ana.results.get('Q_local', np.zeros_like(ana.m_raw))[f_i])
                
            if fk in ["Body Color", "Face Color"]:
                inf['mesh'].mapper.scalar_visibility = False
                inf['mesh'].GetProperty().SetColor(plt.cm.tab20(i%20)[:3])
            else:
                inf['mesh'].mapper.scalar_visibility = True
                k = fk if fk in ana.results else 'Displacement [mm]'
                fv = ana.results.get(k, np.zeros((n_frames_tot, ana.sol.res, ana.sol.res)))[f_i]
                if fv.size == ana.sol.res**2:
                    inf['poly'].point_data["S"] = fv.ravel()
                    inf['poly'].set_active_scalars("S")
                    av.append(fv)
            inf['poly'].Modified(); inf['m_poly'].Modified()
            
        if av and fk not in ["Body Color", "Face Color"]:
            cb = np.concatenate([v.ravel() for v in av]); v_min, v_max = float(cb.min()), float(cb.max())
            clim = [v_min, v_max] if dy else [self.sp_min.value(), self.sp_max.value()]
            if clim[0]>=clim[1]: clim[1]=clim[0]+1e-6
            if dy:
                self.sp_min.blockSignals(True); self.sp_min.setValue(v_min); self.sp_min.blockSignals(False)
                self.sp_max.blockSignals(True); self.sp_max.setValue(v_max); self.sp_max.blockSignals(False)
            self.lut.scalar_range = (clim[0], clim[1]); self.sb.SetVisibility(True); self.sb.title = f"[{fk}] Analysis [mm]"
            for ai in self.part_actors.values(): ai['mesh'].mapper.SetScalarRange(clim[0], clim[1])
            self.v_int.add_text(f"[{fk}]\nMin: {v_min:.3e}\nMax: {v_max:.3e}", position='upper_left', font_size=9, color='black', name='st_ov')
        else:
            self.sb.SetVisibility(False); self.v_int.add_text("", position='upper_left', name='st_ov')
            
        self._update_2d_plots(f_i); self.v_int.render()

    def _update_2d_plots(self, f_i):
        if self.mgr.times is None: return
        ct = self.mgr.times[f_i]; interp = self.checks.get('Interp').isChecked() if 'Interp' in self.checks else True
        for i, ax in enumerate(self.axes):
            cfg = self.plot_slots[i]
            if not cfg: continue
            ana, key = self.mgr.analyzers[cfg.part_idx], cfg.data_key
            if cfg.plot_type == "contour":
                d2 = ana.results.get(key, np.zeros((len(self.mgr.times), ana.sol.res, ana.sol.res)))[f_i]
                if self.ims[i] is None:
                    ax.clear(); self.ims[i] = ax.imshow(d2, cmap='turbo', origin='lower')
                    self.fig.colorbar(self.ims[i], ax=ax, format="%.2e")
                self.ims[i].set_data(d2); self.ims[i].set_interpolation('bilinear' if interp else 'nearest')
                ax.set_title(f"[{ana.name}] {key}")
            else:
                if self.vls[i] is None:
                    ax.clear(); ax.grid(True, alpha=0.3)
                    sd = ana.results.get(key, np.zeros(len(self.mgr.times)))
                    if sd.ndim == 1: ax.plot(self.mgr.times, sd, color='#1A73E8')
                    else:
                        for m in range(min(sd.shape[1], 12)): ax.plot(self.mgr.times, sd[:, m], alpha=0.5, label=f"M{m}")
                    self.vls[i] = ax.axvline(ct, color='red', ls='--'); ax.set_ylabel(key); ax.set_xlabel("Time [s]")
                self.vls[i].set_xdata([ct]); ax.set_title(f"[{ana.name}] {key}")
        self.can.draw_idle()

    def _pop_out_2d(self):
        pw = QtWidgets.QMainWindow(self); pw.setWindowTitle("Analysis View"); pw.resize(1100, 850); c = QtWidgets.QWidget(); pw.setCentralWidget(c); l = QtWidgets.QVBoxLayout(c)
        f = Figure(figsize=(10,10)); cn = FigureCanvas(f); l.addWidget(cn); lm = {"1x1":(1,1),"1x2":(1,2),"2x2":(2,2),"3x2":(3,2)}; r, col = lm.get(self.cmb_lay.currentText(),(2,2))
        for i in range(r*col):
            ax = f.add_subplot(r, col, i+1); cfg = self.plot_slots[i]
            if cfg:
                ana, key = self.mgr.analyzers[cfg.part_idx], cfg.data_key
                if cfg.plot_type=="contour":
                    res_val = ana.results.get(key, np.zeros((len(self.mgr.times), ana.sol.res, ana.sol.res)))[self.current_frame]
                    im = ax.imshow(res_val, cmap='turbo', origin='lower'); f.colorbar(im, ax=ax)
                else:
                    vs = ana.results.get(key, np.zeros(len(self.mgr.times))); 
                    if vs.ndim>1:
                        for m in range(min(vs.shape[1], 10)): ax.plot(self.mgr.times, vs[:, m], alpha=0.5)
                    else: ax.plot(self.mgr.times, vs)
                    ax.axvline(self.mgr.times[self.current_frame], color='red')
            else: ax.text(0.5, 0.5, "Empty Slot", ha='center', transform=ax.transAxes)
        cn.draw(); pw.show()

    def _show_about(self): AboutDialog(self.logo_path, self).exec()
    def _on_persp_toggled(self, s): self.v_int.disable_parallel_projection() if s else self.v_int.enable_parallel_projection(); self.v_int.render()
    def _on_bg_changed(self, m):
        if m=="White": self.v_int.set_background("white")
        elif m=="Grey Grad.": self.v_int.set_background("white", top="grey")
        elif m=="Sky Grad.": self.v_int.set_background("white", top="lightskyblue")
        self.v_int.render()

    def _on_legend_mode_changed(self, m):
        if m=="Static":
            fk = self.cmb_3d.currentText()
            if fk not in ["Body Color", "Face Color"]:
                av = [a.results[fk] for a in self.mgr.analyzers if fk in a.results]
                if av: cb = np.concatenate([v.ravel() for v in av]); self.sp_min.setValue(float(cb.min())); self.sp_max.setValue(float(cb.max()))
        self.update_frame(self.current_frame)

    def _on_field_changed(self, fk):
        if self.cmb_l.currentText()=="Static" and fk not in ["Body Color", "Face Color"]:
            av = [a.results[fk] for a in self.mgr.analyzers if fk in a.results]
            if av: cb = np.concatenate([v.ravel() for v in av]); self.sp_min.setValue(float(cb.min())); self.sp_max.setValue(float(cb.max()))
        self.update_frame(self.current_frame)

    def keyPressEvent(self, e): (self._ctrl_slot(-2) if e.key()==QtCore.Qt.Key_Space else super().keyPressEvent(e))

    def _show_part_menu(self, pos=None):
        if pos is None: pos = self.v_int.mapFromGlobal(QtGui.QCursor.pos())
        m = QtWidgets.QMenu(self); m.addAction("Visibility Manager", self.visibility_tool.show); m.addSeparator()
        [m.addAction(n, f) for n, f in [("XY Plane", self.v_int.view_xy), ("YZ Plane", self.v_int.view_yz), ("ZX Plane", self.v_int.view_zx), ("Isometric", self.v_int.view_isometric)]]; m.addSeparator()
        af = m.addAction("Floor Visibility"); af.setCheckable(True); af.setChecked(self.ground.GetVisibility())
        
        # Floor Settings Sub-menu
        fs = m.addMenu("Floor Settings")
        fs.addAction("Change Origin", self._set_floor_origin)
        fs.addAction("Change Normal", self._set_floor_normal)
        fs.addAction("Change Size", self._set_floor_size)
        m.addSeparator()
        
        ame = m.addAction("Show Mesh Edges"); ame.setCheckable(True)
        if self.part_actors:
            f_idx = min(self.part_actors.keys())
            e_v = self.part_actors[f_idx]['mesh'].GetProperty().GetEdgeVisibility()
            ame.setChecked(e_v)
        else: ame.setChecked(True)
        
        ap = m.addAction("Perspective View"); ap.setCheckable(True); ap.setChecked(self.ch_per.isChecked()); m.addSeparator()
        
        # PyVista Actions 정제
        m.addAction("Wireframe Mode", lambda: self.v_int.set_representation_to_wireframe())
        m.addAction("Surface Mode", lambda: self.v_int.set_representation_to_surface())
        m.addAction("Reset Camera", lambda: self.v_int.reset_camera())
        m.addAction("Pick Mode", lambda: self.v_int.enable_point_picking())
        
        a = m.exec_(self.v_int.mapToGlobal(pos))
        if a==af: self.ground.SetVisibility(a.isChecked()); self.v_int.render()
        elif a==ame:
            for ai in self.part_actors.values(): ai['mesh'].GetProperty().SetEdgeVisibility(a.isChecked())
            self.v_int.render()
        elif a==ap: self.ch_per.setChecked(a.isChecked())

    def _set_floor_origin(self):
        v, ok = QtWidgets.QInputDialog.getText(self, "Floor Origin", "Origin (x,y,z):", text=",".join(map(str, self.floor_origin)))
        if ok:
            try: self.floor_origin = [float(x) for x in v.split(",")]; self._update_floor()
            except: pass

    def _set_floor_normal(self):
        v, ok = QtWidgets.QInputDialog.getText(self, "Floor Normal", "Normal (nx,ny,nz):", text=",".join(map(str, self.floor_normal)))
        if ok:
            try: self.floor_normal = [float(x) for x in v.split(",")]; self._update_floor()
            except: pass

    def _set_floor_size(self):
        v, ok = QtWidgets.QInputDialog.getText(self, "Floor Size", "Size (W, H):", text=f"{self.floor_w},{self.floor_h}")
        if ok:
            try: self.floor_w, self.floor_h = [float(x) for x in v.split(",")]; self._update_floor()
            except: pass

    def _update_floor(self):
        self.v_int.remove_actor(self.ground)
        gp = pv.Plane(center=self.floor_origin, direction=self.floor_normal, i_size=self.floor_w, j_size=self.floor_h)
        self.ground = self.v_int.add_mesh(gp, color="blue", opacity=0.1); self.v_int.render()

    def _on_axis_clicked(self, e):
        if e.inaxes:
            for i, ax in enumerate(self.axes):
                if e.inaxes == ax: self.active_slot = i; break
            self._update_selection_ui(); self.statusBar().showMessage(f"Active Slot: {self.active_slot + 1}")

    def _update_selection_ui(self):
        for i, ax in enumerate(self.axes):
            cl, w = ("#1A73E8", 2.0) if i == self.active_slot else ("#DADCE0", 0.5)
            for s in ax.spines.values(): s.set_color(cl); s.set_linewidth(w)
        self.can.draw_idle()

    def _show_add_plot_dialog(self):
        pn = [p.name for p in self.mgr.analyzers]; d = AddPlotDialog(self.active_slot, pn, self.field_keys, self.stat_keys, self)
        if d.exec(): self.plot_slots[self.active_slot] = d.get_config(); self.ims[self.active_slot] = self.vls[self.active_slot] = None; self.update_frame(self.current_frame)

    def _ctrl_slot(self, c):
        n_frames = len(self.mgr.times) if self.mgr.times is not None else 1
        if c == -2:
            if self.is_playing: self.timer.stop(); self.bp.setText("▶")
            else: self.timer.start(self.timer.interval()); self.bp.setText("⏸")
            self.is_playing = not self.is_playing
        elif c == 0: self.update_frame(0)
        elif c == 9999: self.update_frame(n_frames-1)
        else: self.update_frame(max(0, min(n_frames-1, self.current_frame + c*self.anim_step)))

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    print("Please run whts_multipostprocessor.py for full functionality.")
    sys.exit(0)
