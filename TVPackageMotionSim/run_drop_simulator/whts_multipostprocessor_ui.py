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
import koreanize_matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6 import QtWidgets, QtCore, QtGui

# ==============================================================================
# --- [Section 1] Global Configuration & Data Models ---
# ==============================================================================

# 전역 시각화 설정 (WHTOOLS Standard)
plt.rcParams['font.size'] = 9
FIGURE_WIDTH = 12
FIGURE_HEIGHT = 8

# [WHTOOLS] 현재 디렉토리를 경로에 추가 (모듈 관리 효율화)
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
    """[WHTOOLS] 2D 그래프 슬롯별 시각화 구성 설정 데이터 본체"""
    part_idx: int = 0
    plot_type: str = 'contour'
    data_key: str = 'Displacement [mm]'

@dataclass
class DashboardConfig:
    """[WHTOOLS] 통합 대시보드 레이아웃 및 제어 전략 설정"""
    layout_2d: str = '2x2'
    plots_2d: List[PlotSlotConfig] = field(default_factory=list)
    v_font_size: int = 9
    animation_step: int = 1
    animation_speed_ms: int = 30


# ==============================================================================
# --- [Section 2] Helper Windows & Dialogs ---
# ==============================================================================

class VisibilityToolWindow(QtWidgets.QWidget):
    """
    [WHTOOLS] 가시성 관리자 (Visibility Manager)
    트리 구조를 이용한 파트별 메쉬/마커 가시성 및 실시간 해석 정보(Min/Max) 모니터링 창
    """
    def __init__(self, parent=None):
        super().__init__(parent, QtCore.Qt.Window | QtCore.Qt.WindowStaysOnTopHint)
        self.setWindowTitle("Visibility Manager")
        self.resize(400, 600)
        self.parent = parent
        
        layout = QtWidgets.QVBoxLayout(self)
        
        # 1. Global Control Section
        global_group = QtWidgets.QGroupBox("Global Control")
        global_layout = QtWidgets.QVBoxLayout(global_group)
        for title, col_idx in [("Mesh", 1), ("Markers", 2)]:
            h_layout = QtWidgets.QHBoxLayout()
            h_layout.addWidget(QtWidgets.QLabel(f"{title}:"))
            
            btn_show = QtWidgets.QPushButton("Show All")
            btn_hide = QtWidgets.QPushButton("Hide All")
            
            btn_show.clicked.connect(partial(self._bulk_set, col_idx, True))
            btn_hide.clicked.connect(partial(self._bulk_set, col_idx, False))
            
            h_layout.addWidget(btn_show)
            h_layout.addWidget(btn_hide)
            global_layout.addLayout(h_layout)
        layout.addWidget(global_group)
        
        # 2. Hierarchy Tree Section
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Part", "Mesh", "Markers", "Info (Min / Max)"])
        self.tree.setColumnWidth(0, 150)
        self.tree.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.tree)
        
        self.groups = {}
        self.id_to_item = {}
        self._init_tree()

    def _init_tree(self):
        """데이터 소스(Manager)로부터 트리 아이템 초기 생성"""
        self.tree.blockSignals(True)
        if not self.parent or not self.parent.mgr:
            return
        
        for i, part in enumerate(self.parent.mgr.analyzers):
            # 그룹핑 (접두사 기준)
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
        """전체 항목 가시성 일괄 제어"""
        self.tree.blockSignals(True)
        cs = QtCore.Qt.Checked if state else QtCore.Qt.Unchecked
        for i in range(self.tree.topLevelItemCount()):
            g = self.tree.topLevelItem(i)
            g.setCheckState(column, cs)
            for j in range(g.childCount()):
                g.child(j).setCheckState(column, cs)
        self.tree.blockSignals(False)
        self._apply()

    def _on_item_changed(self, item, column):
        """아이템 체크 상태 변화 시 가시성 동기화"""
        self.tree.blockSignals(True)
        if item.parent() is None: # 그룹 선택 시 자식 포함
            cs = item.checkState(column)
            for j in range(item.childCount()):
                item.child(j).setCheckState(column, cs)
        else: # 자식 선택 시 부모 상태 갱신 (부분 체크 로직은 단순화됨)
            p = item.parent()
            all_c = True
            for j in range(p.childCount()):
                if p.child(j).checkState(column) == QtCore.Qt.Unchecked:
                    all_c = False
                    break
            p.setCheckState(column, QtCore.Qt.Checked if all_c else QtCore.Qt.Unchecked)
        self.tree.blockSignals(False)
        self._apply()

    def update_info(self):
        """현재 선택된 필드 데이터의 Min/Max 정보를 트리 행에 업데이트"""
        if not self.parent or not hasattr(self.parent, 'cmb_3d'):
            return
        
        f_idx = self.parent.current_frame
        fk = self.parent.cmb_3d.currentText()
        
        self.tree.blockSignals(True)
        for i, item in self.id_to_item.items():
            ana = self.parent.mgr.analyzers[i]
            if fk in ana.results and fk not in ["Body Color", "Face Color"]:
                val = ana.results[fk][f_idx]
                item.setText(3, f"{val.min():.2e} / {val.max():.2e}")
            else:
                item.setText(3, "-")
        self.tree.blockSignals(False)

    def _apply(self):
        """변경된 가시성 설정을 메인 렌더러에 즉각 반영"""
        for i, item in self.id_to_item.items():
            if i in self.parent.part_actors:
                self.parent.part_actors[i]['visible'] = (item.checkState(1) == QtCore.Qt.Checked)
                self.parent.part_actors[i]['visible_markers'] = (item.checkState(2) == QtCore.Qt.Checked)
        self.parent.update_frame(self.parent.current_frame)


class AddPlotDialog(QtWidgets.QDialog):
    """장면 내 Matplotlib 그래프 슬롯 추가/편집 대화상자"""
    def __init__(self, slot_idx, parts, field_keys, stat_keys, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Add Plot to Slot {slot_idx + 1}")
        
        layout = QtWidgets.QVBoxLayout(self)
        grid = QtWidgets.QGridLayout()
        layout.addLayout(grid)
        
        # Part Selection
        grid.addWidget(QtWidgets.QLabel("Part:"), 0, 0)
        self.cmb_part = QtWidgets.QComboBox()
        self.cmb_part.addItems(parts)
        grid.addWidget(self.cmb_part, 0, 1)
        
        # Plot Type
        grid.addWidget(QtWidgets.QLabel("Type:"), 1, 0)
        hb = QtWidgets.QHBoxLayout()
        self.rb_c = QtWidgets.QRadioButton("Contour")
        self.rb_cur = QtWidgets.QRadioButton("Curve")
        self.rb_c.setChecked(True)
        hb.addWidget(self.rb_c)
        hb.addWidget(self.rb_cur)
        grid.addLayout(hb, 1, 1)
        
        # Data Key
        grid.addWidget(QtWidgets.QLabel("Key:"), 2, 0)
        self.cmb_key = QtWidgets.QComboBox()
        grid.addWidget(self.cmb_key, 2, 1)
        
        self.f_keys = field_keys
        self.s_keys = stat_keys
        
        self.rb_c.toggled.connect(self._update_keys)
        self.rb_cur.toggled.connect(self._update_keys)
        self._update_keys()
        
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

    def _update_keys(self):
        """선택된 그래프 타입에 따른 필터링된 키 목록 제공"""
        self.cmb_key.clear()
        if self.rb_c.isChecked():
            self.cmb_key.addItems(self.f_keys)
        else:
            self.cmb_key.addItems(self.s_keys)

    def get_config(self) -> PlotSlotConfig:
        """대화상자에서 결정된 GUI 구성을 설정 객체로 반환"""
        return PlotSlotConfig(
            part_idx=self.cmb_part.currentIndex(), 
            plot_type="contour" if self.rb_c.isChecked() else "curve", 
            data_key=self.cmb_key.currentText()
        )


class AboutDialog(QtWidgets.QDialog):
    """WHTOOLS 소프트웨어 정보 및 기술 스펙 안내"""
    def __init__(self, logo_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About WHTOOLS Dashboard")
        self.setFixedSize(550, 650)
        
        l = QtWidgets.QVBoxLayout(self)
        l.setContentsMargins(40, 40, 40, 40)
        l.setSpacing(20)
        
        if os.path.exists(logo_path):
            il = QtWidgets.QLabel()
            px = QtGui.QPixmap(logo_path).scaledToHeight(220, QtCore.Qt.SmoothTransformation)
            il.setPixmap(px)
            il.setAlignment(QtCore.Qt.AlignCenter)
            l.addWidget(il)
            
        t = QtWidgets.QLabel("WHTOOLS Structural Dashboard v5.9")
        t.setStyleSheet("font-size: 20pt; font-weight: bold; color: #1A73E8;")
        t.setAlignment(QtCore.Qt.AlignCenter)
        l.addWidget(t)
        
        st = QtWidgets.QLabel("Expert Structural Analysis & Digital Twin Solution")
        st.setStyleSheet("font-size: 11pt; color: #5F6368; font-style: italic;")
        st.setAlignment(QtCore.Qt.AlignCenter)
        l.addWidget(st)
        
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        l.addWidget(line)
        
        features_text = (
            "<b>Advanced Computational Core:</b><br>"
            "• <b>Multi-Theory Shell Solver:</b> Kirchhoff / Mindlin / Von Karman<br>"
            "• <b>JAX-SSR Engine:</b> Ultra-fast surface reconstruction<br>"
            "• <b>Autonomous Alignment:</b> SVD-based plane fitting<br>"
            "• <b>Expert Visualization:</b> Multi-slot 3D/2D interaction"
        )
        f = QtWidgets.QLabel(features_text)
        f.setStyleSheet("font-size: 11pt; line-height: 170%; color: #3C4043;")
        f.setWordWrap(True)
        l.addWidget(f)
        
        l.addStretch()
        
        cl = QtWidgets.QLabel("© 2026 WHTOOLS. All Rights Reserved.")
        cl.setStyleSheet("font-size: 9pt; color: #9AA0A6;")
        cl.setAlignment(QtCore.Qt.AlignCenter)
        l.addWidget(cl)
        
        bc = QtWidgets.QPushButton("Close")
        bc.setFixedWidth(120)
        bc.setStyleSheet("padding: 10px; font-weight: bold;")
        bc.clicked.connect(self.accept)
        
        hl = QtWidgets.QHBoxLayout()
        hl.addStretch()
        hl.addWidget(bc)
        hl.addStretch()
        l.addLayout(hl)

# ==============================================================================
# --- [Section 3] Main Application: QtVisualizerV2 ---
# ==============================================================================

class QtVisualizerV2(QtWidgets.QMainWindow):
    """
    [WHTOOLS] 차세대 구조 변형 분석 대시보드 (V2)
    VTK 기반 3D 뷰어와 Matplotlib 기반 2D 그래프를 결합한 통합 분석 플랫폼.
    """
    
    def __init__(self, manager: PlateAssemblyManager, config: DashboardConfig = None, ground_size=(4000, 4000)):
        """
        대시보드 초기화 및 핵심 데이터 바인딩.
        
        Args:
            manager (PlateAssemblyManager): 해석 결과를 관리하는 어셈블리 매니저.
            config (DashboardConfig, optional): 사용자 정의 시각화 설정.
            ground_size (tuple, optional): 바닥 그리드의 물리적 크기 (mm).
        """
        super().__init__()
        print(f"[WHTOOLS-UI] Initializing Dashboard with {len(manager.analyzers)} parts...")
        
        # 1. 속성 초기화
        self.mgr = manager
        self.cfg = config or DashboardConfig()
        self.ground_size = ground_size
        
        self.current_frame = 0
        self.is_playing = False
        self.active_slot = 0
        self.anim_step = self.cfg.animation_step
        
        self.plot_slots: List[Optional[PlotSlotConfig]] = [None] * 6
        self.part_actors = {}
        self.v_font_size = self.cfg.v_font_size
        self.ims = [None] * 6
        self.vls = [None] * 6
        
        # 2. 바닥(Floor) 상태 설정
        self.floor_origin = [0, 0, 0]
        self.floor_normal = [0, 0, 1]
        self.floor_w, self.floor_h = ground_size
        
        # 3. 데이터 유효성 검사 (v7.5.3 패치 유지)
        if not manager.analyzers:
            print("❌ No analyzers provided to dashboard!")
            return
            
        # 해석에 성공한 첫 번째 파트를 기준으로 필드 키 검색
        valid_analyzers = [a for a in manager.analyzers if a.sol is not None and a.results]
        if not valid_analyzers:
            print("❌ No valid analysis results found. Dashboard cannot be initialized.")
            return

        # 필드 데이터 필터링 (3D 텐서/스칼라 필드 식별)
        p0 = valid_analyzers[0]
        n_f = len(self.mgr.times)
        res_sq = p0.sol.res**2
        
        self.field_keys = [
            k for k in p0.results 
            if p0.results[k].ndim == 3 and p0.results[k].size // n_f == res_sq
        ]
        self.stat_keys = [
            k for k in p0.results 
            if k not in self.field_keys and p0.results[k].ndim < 3
        ] + ['Marker Local Disp. [mm]', 'Marker Global Disp. [mm]']
        
        # 4. 리소스 경로 설정
        self.res_dir = os.path.join(os.path.dirname(__file__), "resources")
        self.logo_path = os.path.join(self.res_dir, "logo.png")

        # 5. UI 엔진 가동
        self.statusBar().showMessage("WHTOOLS Ready")
        self.visibility_tool = VisibilityToolWindow(self)
        
        # 메인 시퀀스 실행
        self._init_ui()
        self._init_3d_view()
        self._init_2d_plots()
        self.update_frame(0)

    # --------------------------------------------------------------------------
    # --- UI Layout & Component Setup ---
    # --------------------------------------------------------------------------

    def _init_ui(self):
        """메인 레이아웃 및 탭 구조 초기화"""
        self.setWindowTitle("WHTOOLS Structural Dashboard v5.9")
        self.resize(1700, 1020)
        
        # 중앙 위젯 및 메인 레이아웃
        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        ml = QtWidgets.QVBoxLayout(cw)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.setSpacing(0)
        
        # 상단 헤더 (로고 + 탭)
        tc = QtWidgets.QWidget()
        tl = QtWidgets.QHBoxLayout(tc)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(0)
        
        self.lbl_logo = QtWidgets.QLabel()
        if os.path.exists(self.logo_path):
            self.lbl_logo.setPixmap(QtGui.QPixmap(self.logo_path).scaledToHeight(130, QtCore.Qt.SmoothTransformation))
        else:
            self.lbl_logo.setText("WHTOOLS")
            self.lbl_logo.setStyleSheet("font-weight: bold; font-size: 22pt; color: #1A73E8; margin-left: 10px;")
        tl.addWidget(self.lbl_logo)
        
        # 탭 위젯 (3D / 2D / 설정)
        self.ct = QtWidgets.QTabWidget()
        self.t3 = QtWidgets.QWidget()
        self.t2 = QtWidgets.QWidget()
        self.ts = QtWidgets.QWidget()
        
        self.ct.setStyleSheet("QTabWidget::pane { border: 0; top: -1px; } QTabBar::tab { height: 35px; padding: 0 20px; }")
        self.ct.addTab(self.t3, "🧊 3D Field")
        self.ct.addTab(self.t2, "📈 2D Field & Curves")
        self.ct.addTab(self.ts, "⚙️ Settings")
        tl.addWidget(self.ct, stretch=1)
        ml.addWidget(tc)
        
        # 메인 콘텐츠 분할 (Splitter)
        self.split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        ml.addWidget(self.split, stretch=1)
        
        # 좌측: 3D 뷰 패널
        self.p3d = QtWidgets.QWidget()
        l3 = QtWidgets.QVBoxLayout(self.p3d)
        l3.setContentsMargins(0, 0, 0, 0)
        self.v_int = QtInteractor(self.p3d)
        self.v_int.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.v_int.customContextMenuRequested.connect(self._show_part_menu)
        l3.addWidget(self.v_int)
        
        # 우측: 2D 그래프 패널
        self.p2d = QtWidgets.QWidget()
        l2 = QtWidgets.QVBoxLayout(self.p2d)
        l2.setContentsMargins(0, 0, 0, 0)
        
        self._cw = QtWidgets.QWidget()
        self._cl = QtWidgets.QVBoxLayout(self._cw)
        self._cl.setContentsMargins(0, 0, 0, 0)
        l2.addWidget(self._cw, stretch=1)
        
        self.split.addWidget(self.p3d)
        self.split.addWidget(self.p2d)
        
        # 초기 화면 분할 비율 설정 (2:3)
        self.split.setSizes([680, 1020]) 
        self.split.setStretchFactor(0, 2)
        self.split.setStretchFactor(1, 3)
        
        # 각 컨트롤 패널 초기화
        self._init_3d_controls(self.t3)
        self._init_2d_controls(self.t2)
        self._init_settings_controls(self.ts)
        self._init_animation_dock()

    def _init_settings_controls(self, p):
        """환경 설정 패널 구성 (Ultra-Compact Horizontal Layout)"""
        main_l = QtWidgets.QHBoxLayout()
        main_l.setAlignment(QtCore.Qt.AlignLeft)
        main_l.setContentsMargins(2, 2, 2, 2)
        main_l.setSpacing(2)
        p.setLayout(main_l)
        
        # Group 1: General & Info
        g1 = QtWidgets.QGroupBox("General & Info")
        gl1 = QtWidgets.QGridLayout(g1)
        gl1.setSpacing(2); gl1.setContentsMargins(4, 8, 4, 4)
        
        b_vis = QtWidgets.QPushButton("Visibility Manager")
        b_vis.clicked.connect(self.visibility_tool.show)
        gl1.addWidget(b_vis, 0, 0)
        
        b_res = QtWidgets.QPushButton("Reset Camera (f)")
        b_res.clicked.connect(lambda: self.v_int.reset_camera())
        gl1.addWidget(b_res, 0, 1)
        
        b_abt = QtWidgets.QPushButton("About WHTOOLS")
        b_abt.clicked.connect(self._show_about)
        gl1.addWidget(b_abt, 1, 0)
        
        main_l.addWidget(g1)
        
        # Group 2: Animation & Step Settings
        g2 = QtWidgets.QGroupBox("Animation & Step Settings")
        gl2 = QtWidgets.QGridLayout(g2)
        
        gl2.addWidget(QtWidgets.QLabel("Skip Frames (Step):"), 0, 0)
        self.sp_step = QtWidgets.QSpinBox()
        self.sp_step.setRange(1, 100)
        self.sp_step.setValue(self.anim_step)
        self.sp_step.valueChanged.connect(self._update_step)
        gl2.addWidget(self.sp_step, 0, 1)
        
        main_l.addWidget(g2)
        main_l.addStretch()

    def _update_step(self, v): 
        """애니메이션 스킵 프레임 간격 업데이트"""
        self.anim_step = v

    def _init_3d_controls(self, p):
        """3D 뷰어 제어 패널 구성 (Ultra-Compact Horizontal Layout)"""
        main_l = QtWidgets.QHBoxLayout()
        main_l.setAlignment(QtCore.Qt.AlignLeft)
        main_l.setContentsMargins(2, 2, 2, 2)
        main_l.setSpacing(2)
        p.setLayout(main_l)
        
        # Group 1: View & Deformation
        g1 = QtWidgets.QGroupBox("View & Deformation")
        gl1 = QtWidgets.QGridLayout(g1)
        gl1.setSpacing(2); gl1.setContentsMargins(4, 8, 4, 4)
        
        gl1.addWidget(QtWidgets.QLabel("View:"), 0, 0)
        self.cmb_v = QtWidgets.QComboBox()
        self.cmb_v.addItems(["Global", "Local"])
        self.cmb_v.currentTextChanged.connect(lambda: self.update_frame(self.current_frame))
        gl1.addWidget(self.cmb_v, 0, 1)
        
        gl1.addWidget(QtWidgets.QLabel("Scale:"), 0, 2)
        self.sp_sc = QtWidgets.QDoubleSpinBox()
        self.sp_sc.setRange(1.0, 1000.0)
        self.sp_sc.setValue(1.0)
        self.sp_sc.valueChanged.connect(lambda: self.update_frame(self.current_frame))
        gl1.addWidget(self.sp_sc, 0, 3)
        
        self.ch_per = QtWidgets.QCheckBox("Perspective")
        self.ch_per.setChecked(True)
        self.ch_per.toggled.connect(self._on_persp_toggled)
        gl1.addWidget(self.ch_per, 1, 0)
        
        gl1.addWidget(QtWidgets.QLabel("Background:"), 1, 1)
        self.cmb_bg = QtWidgets.QComboBox()
        self.cmb_bg.addItems(["White", "Grey Grad.", "Sky Grad."])
        self.cmb_bg.currentTextChanged.connect(self._on_bg_changed)
        gl1.addWidget(self.cmb_bg, 1, 2, 1, 2) # Span 2 columns
        
        main_l.addWidget(g1)
        
        # Group 2: Field & Range Analysis
        g2 = QtWidgets.QGroupBox("Field & Range Analysis")
        gl2 = QtWidgets.QGridLayout(g2)
        gl2.setSpacing(2); gl2.setContentsMargins(4, 8, 4, 4)
        
        gl2.addWidget(QtWidgets.QLabel("Field:"), 0, 0)
        self.cmb_3d = QtWidgets.QComboBox()
        self.cmb_3d.addItems(["Body Color", "Face Color"] + self.field_keys)
        self.cmb_3d.currentTextChanged.connect(self._on_field_changed)
        gl2.addWidget(self.cmb_3d, 0, 1)
        
        gl2.addWidget(QtWidgets.QLabel("Range:"), 0, 2)
        self.cmb_l = QtWidgets.QComboBox()
        self.cmb_l.addItems(["Dynamic", "Static"])
        self.cmb_l.currentTextChanged.connect(self._on_legend_mode_changed)
        gl2.addWidget(self.cmb_l, 0, 3)
        
        gl2.addWidget(QtWidgets.QLabel("Min:"), 1, 0)
        self.sp_min = QtWidgets.QDoubleSpinBox()
        self.sp_min.setRange(-1e9, 1e9); self.sp_min.setDecimals(4)
        self.sp_min.valueChanged.connect(lambda: self.update_frame(self.current_frame))
        gl2.addWidget(self.sp_min, 1, 1)
        
        gl2.addWidget(QtWidgets.QLabel("Max:"), 1, 2)
        self.sp_max = QtWidgets.QDoubleSpinBox()
        self.sp_max.setRange(-1e9, 1e9); self.sp_max.setDecimals(4)
        self.sp_max.valueChanged.connect(lambda: self.update_frame(self.current_frame))
        gl2.addWidget(self.sp_max, 1, 3)
        
        main_l.addWidget(g2)
        main_l.addStretch()

    def _init_2d_controls(self, p):
        """2D 그래프 세션 제어 패널 구성 (Ultra-Compact Horizontal Layout)"""
        main_l = QtWidgets.QHBoxLayout()
        main_l.setAlignment(QtCore.Qt.AlignLeft)
        main_l.setContentsMargins(2, 2, 2, 2)
        main_l.setSpacing(2)
        p.setLayout(main_l)
        
        # Group 1: 2D Slot Layout
        g1 = QtWidgets.QGroupBox("2D Slot Layout")
        gl1 = QtWidgets.QGridLayout(g1)
        gl1.setSpacing(2); gl1.setContentsMargins(4, 8, 4, 4)
        
        gl1.addWidget(QtWidgets.QLabel("Grid:"), 0, 0)
        self.cmb_lay = QtWidgets.QComboBox()
        self.cmb_lay.addItems(["1x1", "1x2", "2x2", "3x2"])
        self.cmb_lay.setCurrentText(self.cfg.layout_2d)
        self.cmb_lay.currentTextChanged.connect(self._init_2d_plots)
        gl1.addWidget(self.cmb_lay, 0, 1)
        
        bt_add = QtWidgets.QPushButton("+ Add Plot")
        bt_add.clicked.connect(self._show_add_plot_dialog)
        gl1.addWidget(bt_add, 1, 0, 1, 2) # Span 2 columns
        
        main_l.addWidget(g1)
        
        # Group 2: Display & Tools
        g2 = QtWidgets.QGroupBox("Display & Tools")
        gl2 = QtWidgets.QGridLayout(g2)
        gl2.setSpacing(2); gl2.setContentsMargins(4, 8, 4, 4)
        
        self.checks = {}
        for i, (t, s) in enumerate([("Sync Timeline", True), ("Interpolation", True)]):
            c = QtWidgets.QCheckBox(t)
            c.setChecked(s)
            c.toggled.connect(lambda: self.update_frame(self.current_frame))
            gl2.addWidget(c, i, 0)
            self.checks[t.split()[0]] = c
            
        bt_pop = QtWidgets.QPushButton("Pop-out View")
        bt_pop.clicked.connect(self._pop_out_2d)
        gl2.addWidget(bt_pop, 0, 1, 2, 1) # Span 2 rows
        
        main_l.addWidget(g2)
        main_l.addStretch()

    def _init_animation_dock(self):
        """하단 애니메이션 타임라인 및 제어 도크 초기화"""
        self.ad = QtWidgets.QDockWidget("Animation Control")
        self.ad.setAllowedAreas(QtCore.Qt.BottomDockWidgetArea)
        
        cn = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(cn)
        layout.setContentsMargins(10, 5, 10, 5)
        
        # 탐색 버튼들
        for t, s in [("<<", 0), ("<", -1), (">", 1), (">>", 9999)]:
            b = QtWidgets.QPushButton(t)
            b.setFixedSize(35, 30)
            b.clicked.connect(partial(self._ctrl_slot, s))
            layout.addWidget(b)
            
        # 재생/일시정지
        self.bp = QtWidgets.QPushButton("▶")
        self.bp.setFixedSize(45, 30)
        self.bp.clicked.connect(lambda: self._ctrl_slot(-2))
        layout.addWidget(self.bp)
        
        # 슬라이더 (타임라인)
        n_frames = len(self.mgr.times) if self.mgr.times is not None else 1
        self.sld = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sld.setRange(0, n_frames - 1)
        self.sld.valueChanged.connect(self.update_frame)
        layout.addWidget(self.sld, stretch=1)
        
        # 상태 정보
        self.lf = QtWidgets.QLabel(f"Frame: 0 / {n_frames-1}")
        self.lf.setFixedWidth(150)
        layout.addWidget(self.lf)
        
        # 재생 속도 제어
        layout.addWidget(QtWidgets.QLabel(" Speed Control (ms):"))
        self.cs = QtWidgets.QComboBox()
        self.cs.addItems(["0", "15", "30", "50", "100"])
        self.cs.setCurrentText("30")
        self.cs.currentTextChanged.connect(lambda v: self.timer.setInterval(int(v)))
        layout.addWidget(self.cs)
        
        self.ad.setWidget(cn)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.ad)

    # --------------------------------------------------------------------------
    # --- 3D Visualization Engine ---
    # --------------------------------------------------------------------------

    def _init_3d_view(self):
        """3D 장면 초기화 및 어셈블리 파트별 메쉬 생성"""
        self.v_int.set_background("white")
        self.v_int.add_axes()
        
        # 바닥 그리드 평면 생성
        gp = pv.Plane(
            center=self.floor_origin, 
            direction=self.floor_normal, 
            i_size=self.floor_w, 
            j_size=self.floor_h
        )
        self.ground = self.v_int.add_mesh(gp, color="blue", opacity=0.1)
        
        # 컬러맵/룩업테이블 설정
        self.lut = pv.LookupTable(cmap="turbo", flip=True)
        self.lut.below_range_color = 'lightgrey'
        self.lut.above_range_color = 'magenta'
        
        for i, ana in enumerate(self.mgr.analyzers):
            # [WHTOOLS] v7.5.3: 데이터가 없는 파트는 시각화 인스턴스에서 안전하게 제외
            if ana.m_raw is None or ana.sol is None:
                self.part_actors[i] = {'mesh': None, 'visible': False}
                continue
                
            # 1. 쉘 메쉬(평면) 생성 (해석 해상도 기반)
            poly = pv.Plane(
                i_size=ana.W, 
                j_size=ana.H, 
                i_resolution=ana.sol.res - 1, 
                j_resolution=ana.sol.res - 1
            )
            ma = self.v_int.add_mesh(
                poly, 
                scalars=None, 
                cmap=self.lut, 
                show_edges=True, 
                edge_color="darkgray", 
                show_scalar_bar=False
            )
            
            # 2. 마커 시각화 (구체 포인트 데이터)
            mp = pv.PolyData(np.array(ana.m_raw[0]))
            n_m = ana.m_raw.shape[1]
            mp.point_data["names"] = [f"{ana.name}_M{j:02d}" for j in range(n_m)]
            mka = self.v_int.add_mesh(
                mp, 
                render_points_as_spheres=True, 
                point_size=10, 
                color='skyblue'
            )
            
            # 3. 포인트 라벨
            la = self.v_int.add_point_labels(
                mp, "names", 
                font_size=self.v_font_size, 
                text_color='black', 
                always_visible=True, 
                point_size=0, 
                shadow=False
            )
            
            # [WHTOOLS] v7.5.4: 추가 데이터 검증 (해석 격자 부재 시 예외 처리)
            if not hasattr(ana.sol, 'X_mesh') or ana.sol.X_mesh is None:
                ma.SetVisibility(False)
                self.part_actors[i] = {'mesh': ma, 'visible': False}
                continue
                
            # 4. 액터 컬렉션 저장
            mka.SetVisibility(False)
            la.SetVisibility(False)
            
            # 기본 좌표 격자(p_base) 미리 생성하여 성능 최적화
            p_base = np.column_stack([
                ana.sol.X_mesh.ravel(), 
                ana.sol.Y_mesh.ravel(), 
                np.zeros(ana.sol.res**2)
            ])
            
            self.part_actors[i] = {
                'mesh': ma, 
                'poly': poly, 
                'm_poly': mp, 
                'markers': mka, 
                'labels': la, 
                'visible': True, 
                'visible_markers': False, 
                'p_base': p_base
            }
            
        # 5. 통합 컬러바 설정 (성공한 첫 번째 파트 기준)
        if self.part_actors:
            f_i = min(self.part_actors.keys())
            # 컬러바 스타일: 정밀도 %.3f, 라벨 7개 (WHTOOLS Standard)
            self.sb = self.v_int.add_scalar_bar(
                "Field Analysis [mm]", 
                position_x=0.15, position_y=0.05, 
                width=0.7, 
                mapper=self.part_actors[f_i]['mesh'].mapper, 
                vertical=False, 
                n_labels=7, 
                fmt="%.3f", 
                label_font_size=10
            )
        else:
            self.sb = self.v_int.add_scalar_bar("No Data", position_x=0.15)
            self.sb.SetVisibility(False)
            
        # 6. 기타 오버레이 텍스트
        self.ov = self.v_int.add_text("-", position='upper_right', font_size=9, color='black')
        self.gui_txt = self.v_int.add_text(
            "[Space]: Play/Pause | [R]: Reset | [W]: Wireframe", 
            position='upper_right', 
            font_size=9, 
            color='black'
        )
        
        # 7. 기본 카메라 및 타이머 설정
        self.v_int.view_isometric()
        self.v_int.camera.ParallelProjectionOff()
        
        self.timer = QtCore.QTimer()
        self.timer.setInterval(30)
        self.timer.timeout.connect(lambda: self._ctrl_slot(1))

    def update_frame(self, f_i: int):
        """
        주어진 시점(f_i)으로 모든 파트의 변형 정보를 실시간 업데이트.
        
        이 메서드는 로컬 변위 필드를 기구학(R, Centroid) 정보와 결합하여 
        물리적인 3D 배치를 수행합니다.
        """
        self.current_frame = f_i
        
        # UI 슬라이더 동기화
        self.sld.blockSignals(True)
        self.sld.setValue(f_i)
        self.sld.blockSignals(False)
        
        n_frames_tot = len(self.mgr.times) if self.mgr.times is not None else 1
        self.lf.setText(f"Frame: {f_i} / {n_frames_tot - 1}")
        
        # 제어 바 상태 읽기
        vm = self.cmb_v.currentText()        # View Mode: Global / Local
        fk = self.cmb_3d.currentText()       # Field Key: Displacement, Stress, etc.
        sc = self.sp_sc.value()              # Deformation Scale
        dy = self.cmb_l.currentText() == "Dynamic" # Range Mode
        
        active_values = []
        
        for i, ana in enumerate(self.mgr.analyzers):
            if i not in self.part_actors:
                continue
                
            inf = self.part_actors[i]
            if inf['mesh'] is None:
                continue
                
            mv = inf['visible']
            mkv = inf['visible_markers']
            
            inf['mesh'].SetVisibility(mv)
            inf['markers'].SetVisibility(mkv)
            inf['labels'].SetVisibility(mkv)
            
            if not mv and not mkv:
                continue
            
            # [WHTOOLS] v7.5 안정화: 수치적 정적 변형 및 기구학 연동
            displacement_w = ana.results.get('Displacement [mm]', np.zeros((n_frames_tot, ana.sol.res, ana.sol.res)))[f_i]
            
            # 1. 로컬 좌표계 기반 변위 적용
            points_local = inf['p_base'].copy()
            points_local[:, 2] = displacement_w.ravel() * sc
            
            # 2. 기구학 정보 추출 (Rot / Trans)
            R_matrix = ana.results.get('R_matrix')[f_i]
            cur_centroid = ana.results.get('cur_centroid')[f_i]
            ref_centroid = ana.results.get('ref_centroid')[f_i]
            local_basis = np.array(ana.kin.local_basis_axes)
            local_cent_0 = np.array(ana.kin.local_centroid_0)
            
            if vm == "Global":
                # 로컬 -> 글로벌 좌표 변환 (Expert Kinematics Formula)
                # Formula: (P_local @ Basis.T + Centroid_0 - Ref_Centroid) @ R + Cur_Centroid
                inf['poly'].points = (
                    points_local @ local_basis.T + 
                    local_cent_0 - ref_centroid
                ) @ R_matrix + cur_centroid
                inf['m_poly'].points = np.array(ana.m_raw[f_i])
            else:
                # 로컬 뷰 모드
                inf['poly'].points = points_local
                inf['m_poly'].points = np.array(ana.results.get('local_markers')[f_i])
                
            # 3. 색상 맵핑 (Colorization)
            if fk in ["Body Color", "Face Color"]:
                inf['mesh'].mapper.scalar_visibility = False
                inf['mesh'].GetProperty().SetColor(plt.cm.tab20(i % 20)[:3])
            else:
                inf['mesh'].mapper.scalar_visibility = True
                active_key = fk if fk in ana.results else 'Displacement [mm]'
                
                if inf['visible'] and ana.sol is not None:
                    field_val = ana.results.get(active_key)[f_i]
                    if field_val.size == ana.sol.res**2:
                        inf['poly'].point_data["S"] = field_val.ravel()
                        inf['poly'].set_active_scalars("S")
                        active_values.append(field_val)
                        
            inf['poly'].Modified()
            inf['m_poly'].Modified()
            
        # 4. 컬러 범례(Legend) 및 통계 업데이트
        if active_values and fk not in ["Body Color", "Face Color"]:
            all_field_data = np.concatenate([v.ravel() for v in active_values])
            v_min, v_max = float(all_field_data.min()), float(all_field_data.max())
            
            # 범위 결정 (Dynamic vs Static)
            clim = [v_min, v_max] if dy else [self.sp_min.value(), self.sp_max.value()]
            if clim[0] >= clim[1]:
                clim[1] = clim[0] + 1e-6
            
            if dy:
                self.sp_min.blockSignals(True)
                self.sp_min.setValue(v_min)
                self.sp_min.blockSignals(False)
                self.sp_max.blockSignals(True)
                self.sp_max.setValue(v_max)
                self.sp_max.blockSignals(False)
            
            self.lut.scalar_range = (clim[0], clim[1])
            self.sb.SetVisibility(True)
            self.sb.title = f"[{fk}] Analysis [mm]"
            
            for ai in self.part_actors.values():
                if ai['mesh'] is not None:
                    ai['mesh'].mapper.SetScalarRange(clim[0], clim[1])
                    
            status_text = f"[{fk}]\nMin: {v_min:.3e}\nMax: {v_max:.3e}"
            self.v_int.add_text(status_text, position='upper_left', font_size=6, color='black', name='st_ov')
        else:
            self.sb.SetVisibility(False)
            self.v_int.add_text("", position='upper_left', name='st_ov')
            
        # 5. 후속 처리 (2D 연동 및 렌더링)
        self._update_2d_plots(f_i)
        self.v_int.render()

    # --------------------------------------------------------------------------
    # --- 2D Plotting Engine (Matplotlib) ---
    # --------------------------------------------------------------------------

    def _init_2d_plots(self):
        """2D 차트 영역(Grid Layout) 초기화 및 슬롯 생성"""
        # 기존 위젯 정리
        for i in reversed(range(self._cl.count())):
            item = self._cl.itemAt(i)
            if item.widget():
                item.widget().setParent(None)
                
        plt.rcParams['font.family'] = 'Consolas'
        plt.rcParams['font.size'] = 9
        
        # Figure 및 Canvas 생성
        self.fig = Figure(figsize=(8, 8))
        self.can = FigureCanvas(self.fig)
        self._cl.addWidget(NavigationToolbar(self.can, self))
        self._cl.addWidget(self.can)
        
        # 상호작용 이벤트 연결
        self.can.mpl_connect('button_press_event', self._on_axis_clicked)
        
        # 그리드 레이아웃 결정
        layout_map = {
            "1x1": (1, 1), 
            "1x2": (1, 2), 
            "2x2": (2, 2), 
            "3x2": (3, 2)
        }
        rows, cols = layout_map.get(self.cmb_lay.currentText(), (2, 2))
        
        self.axes = []
        self.ims = [None] * 6
        self.vls = [None] * 6
        self.cbs = [None] * 6
        
        self.fig.clear()
        self.fig.subplots_adjust(hspace=0.4, wspace=0.3)
        
        for i in range(rows * cols):
            ax = self.fig.add_subplot(rows, cols, i + 1)
            self.axes.append(ax)
            ax.text(0.5, 0.5, f"Slot {i + 1}", ha='center', transform=ax.transAxes)
            
        self._update_selection_ui()
        self.can.draw_idle()

    def _update_2d_plots(self, f_i):
        """현재 프레임에 맞춰 2D 슬롯 데이터 갱신"""
        if self.mgr.times is None or not self.axes:
            return
            
        current_time = self.mgr.times[f_i]
        use_interp = self.checks.get('Interp').isChecked() if 'Interp' in self.checks else True
        
        for i, ax in enumerate(self.axes):
            cfg = self.plot_slots[i]
            if not cfg:
                continue
                
            ana = self.mgr.analyzers[cfg.part_idx]
            key = cfg.data_key
            
            if cfg.plot_type == "contour":
                # 2D 콘토어 맵 (imshow)
                if ana.sol is None:
                    continue
                
                # 데이터 추출 (기본 Displacement 방어 로직)
                data_2d = ana.results.get(key, np.zeros((len(self.mgr.times), ana.sol.res, ana.sol.res)))[f_i]
                
                if self.ims[i] is None:
                    ax.clear()
                    # [WHTOOLS] 기존 컬러바 제거 (중복 생성 방지)
                    if self.cbs[i] is not None:
                        try: self.cbs[i].remove()
                        except: pass
                    
                    # [WHTOOLS] 실제 치수 비(W:H)에 맞게 extent 설정하여 Aspect Ratio 보정
                    self.ims[i] = ax.imshow(data_2d, cmap='turbo', origin='lower', extent=[0, ana.W, 0, ana.H])
                    self.cbs[i] = self.fig.colorbar(self.ims[i], ax=ax, format="%.2e")
                
                self.ims[i].set_data(data_2d)
                self.ims[i].set_interpolation('bilinear' if use_interp else 'nearest')
                ax.set_title(f"[{ana.name}] {key}")
                
            else:
                # 1D 시계열 곡선 (Curve)
                if self.vls[i] is None:
                    ax.clear()
                    # [WHTOOLS] 이전 타입이 Contour였을 경우 컬러바 제거
                    if self.cbs[i] is not None:
                        try: self.cbs[i].remove()
                        except: pass
                        self.cbs[i] = None
                    ax.grid(True, alpha=0.3)
                    stat_data = ana.results.get(key, np.zeros(len(self.mgr.times)))
                    
                    if stat_data.ndim == 1:
                        ax.plot(self.mgr.times, stat_data, color='#1A73E8')
                    else:
                        # 다중 마커 데이터 처리 (최대 12개)
                        for m in range(min(stat_data.shape[1], 12)):
                            ax.plot(self.mgr.times, stat_data[:, m], alpha=0.5, label=f"M{m}")
                            
                    self.vls[i] = ax.axvline(current_time, color='red', ls='--')
                    ax.set_ylabel(key)
                    ax.set_xlabel("Time [s]")
                    
                self.vls[i].set_xdata([current_time])
                ax.set_title(f"[{ana.name}] {key}")
                
        self.can.draw_idle()

    # --------------------------------------------------------------------------
    # --- Interaction & Event Handlers ---
    # --------------------------------------------------------------------------

    def _pop_out_2d(self):
        """개별 2D 분석 창 분리 (Pop-out View)"""
        pw = QtWidgets.QMainWindow(self)
        pw.setWindowTitle("Analysis View")
        pw.resize(1100, 850)
        
        cw = QtWidgets.QWidget()
        pw.setCentralWidget(cw)
        layout = QtWidgets.QVBoxLayout(cw)
        
        fig = Figure(figsize=(10, 10))
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        
        layout_map = {"1x1": (1,1), "1x2": (1,2), "2x2": (2,2), "3x2": (3,2)}
        rows, cols = layout_map.get(self.cmb_lay.currentText(), (2, 2))
        
        for i in range(rows * cols):
            ax = fig.add_subplot(rows, cols, i + 1)
            cfg = self.plot_slots[i]
            
            if cfg:
                ana = self.mgr.analyzers[cfg.part_idx]
                key = cfg.data_key
                
                if ana.results and ana.sol and cfg.plot_type == "contour":
                    res_val = ana.results.get(key, np.zeros((len(self.mgr.times), ana.sol.res, ana.sol.res)))[self.current_frame]
                    im = ax.imshow(res_val, cmap='turbo', origin='lower')
                    fig.colorbar(im, ax=ax)
                else:
                    vs = ana.results.get(key, np.zeros(len(self.mgr.times)))
                    if vs.ndim > 1:
                        for m in range(min(vs.shape[1], 10)):
                            ax.plot(self.mgr.times, vs[:, m], alpha=0.5)
                    else:
                        ax.plot(self.mgr.times, vs)
                    ax.axvline(self.mgr.times[self.current_frame], color='red')
            else:
                ax.text(0.5, 0.5, "Empty Slot", ha='center', transform=ax.transAxes)
                
        canvas.draw()
        pw.show()

    def _show_about(self):
        """WHTOOLS 정보 창 표시"""
        AboutDialog(self.logo_path, self).exec()

    def _on_persp_toggled(self, state):
        """원근법(Perspective) 모드 전환"""
        if state:
            self.v_int.disable_parallel_projection()
        else:
            self.v_int.enable_parallel_projection()
        self.v_int.render()

    def _on_bg_changed(self, mode):
        """3D 배경 테마 변경"""
        if mode == "White":
            self.v_int.set_background("white")
        elif mode == "Grey Grad.":
            self.v_int.set_background("white", top="grey")
        elif mode == "Sky Grad.":
            self.v_int.set_background("white", top="lightskyblue")
        self.v_int.render()

    def _on_legend_mode_changed(self, mode):
        """범례 범위 모드(Dynamic/Static) 변경 시 즉시 동기화"""
        if mode == "Static":
            field_key = self.cmb_3d.currentText()
            if field_key not in ["Body Color", "Face Color"]:
                # 모든 파트의 해당 필드 전체 타임라인 Min/Max 계산
                all_values = [
                    a.results[field_key] for a in self.mgr.analyzers 
                    if field_key in a.results
                ]
                if all_values:
                    concatenated = np.concatenate([v.ravel() for v in all_values])
                    self.sp_min.setValue(float(concatenated.min()))
                    self.sp_max.setValue(float(concatenated.max()))
        self.update_frame(self.current_frame)

    def _on_field_changed(self, field_key):
        """가시화 필드 키 변경 시 후속 처리"""
        if self.cmb_l.currentText() == "Static" and field_key not in ["Body Color", "Face Color"]:
            all_values = [
                a.results[field_key] for a in self.mgr.analyzers 
                if field_key in a.results
            ]
            if all_values:
                concatenated = np.concatenate([v.ravel() for v in all_values])
                self.sp_min.setValue(float(concatenated.min()))
                self.sp_max.setValue(float(concatenated.max()))
        self.update_frame(self.current_frame)

    def keyPressEvent(self, event):
        """키보드 단축키 핸들러 (Space: Play/Pause)"""
        if event.key() == QtCore.Qt.Key_Space:
            self._ctrl_slot(-2)
        else:
            super().keyPressEvent(event)

    def _show_part_menu(self, pos=None):
        """3D 뷰 우클릭 컨텍스트 메뉴 생성"""
        if pos is None:
            pos = self.v_int.mapFromGlobal(QtGui.QCursor.pos())
            
        menu = QtWidgets.QMenu(self)
        menu.addAction("Visibility Manager", self.visibility_tool.show)
        menu.addSeparator()
        
        # 뷰 정렬 액션
        view_actions = [
            ("XY Plane", self.v_int.view_xy), 
            ("YZ Plane", self.v_int.view_yz), 
            ("ZX Plane", self.v_int.view_zx), 
            ("Isometric", self.v_int.view_isometric)
        ]
        for name, func in view_actions:
            menu.addAction(name, func)
        menu.addSeparator()
        
        # 바닥 가시성 제어
        act_floor = menu.addAction("Floor Visibility")
        act_floor.setCheckable(True)
        act_floor.setChecked(self.ground.GetVisibility())
        
        # 바닥 세부 설정 서브메뉴
        fs = menu.addMenu("Floor Settings")
        fs.addAction("Change Origin", self._set_floor_origin)
        fs.addAction("Change Normal", self._set_floor_normal)
        fs.addAction("Change Size", self._set_floor_size)
        menu.addSeparator()
        
        # 메쉬 엣지 가시성
        act_edges = menu.addAction("Show Mesh Edges")
        act_edges.setCheckable(True)
        if self.part_actors:
            first_idx = min(self.part_actors.keys())
            if self.part_actors[first_idx]['mesh'] is not None:
                edge_v = self.part_actors[first_idx]['mesh'].GetProperty().GetEdgeVisibility()
                act_edges.setChecked(edge_v)
        else:
            act_edges.setChecked(True)
        
        act_perp = menu.addAction("Perspective View")
        act_perp.setCheckable(True)
        act_perp.setChecked(self.ch_per.isChecked())
        menu.addSeparator()
        
        # 표준 PyVista 액션
        menu.addAction("Wireframe Mode", lambda: self.v_int.set_representation_to_wireframe())
        menu.addAction("Surface Mode", lambda: self.v_int.set_representation_to_surface())
        menu.addAction("Reset Camera", lambda: self.v_int.reset_camera())
        menu.addAction("Pick Mode", lambda: self.v_int.enable_point_picking())
        
        selected_action = menu.exec_(self.v_int.mapToGlobal(pos))
        
        if selected_action == act_floor:
            self.ground.SetVisibility(selected_action.isChecked())
            self.v_int.render()
        elif selected_action == act_edges:
            for ai in self.part_actors.values():
                if ai['mesh'] is not None:
                    ai['mesh'].GetProperty().SetEdgeVisibility(selected_action.isChecked())
            self.v_int.render()
        elif selected_action == act_perp:
            self.ch_per.setChecked(selected_action.isChecked())

    # --------------------------------------------------------------------------
    # --- Floor & Plot Helper Methods ---
    # --------------------------------------------------------------------------

    def _set_floor_origin(self):
        v, ok = QtWidgets.QInputDialog.getText(self, "Floor Origin", "Origin (x,y,z):", text=",".join(map(str, self.floor_origin)))
        if ok:
            try:
                self.floor_origin = [float(x) for x in v.split(",")]
                self._update_floor()
            except: pass

    def _set_floor_normal(self):
        v, ok = QtWidgets.QInputDialog.getText(self, "Floor Normal", "Normal (nx,ny,nz):", text=",".join(map(str, self.floor_normal)))
        if ok:
            try:
                self.floor_normal = [float(x) for x in v.split(",")]
                self._update_floor()
            except: pass

    def _set_floor_size(self):
        v, ok = QtWidgets.QInputDialog.getText(self, "Floor Size", "Size (W, H):", text=f"{self.floor_w},{self.floor_h}")
        if ok:
            try:
                self.floor_w, self.floor_h = [float(x) for x in v.split(",")]
                self._update_floor()
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
        if d.exec(): 
            self.plot_slots[self.active_slot] = d.get_config()
            self.ims[self.active_slot] = self.vls[self.active_slot] = self.cbs[self.active_slot] = None
            self.update_frame(self.current_frame)

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
