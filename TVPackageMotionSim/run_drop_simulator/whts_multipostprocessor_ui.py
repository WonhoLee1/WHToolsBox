# -*- coding: utf-8 -*-
"""
[WHTOOLS] Integrated Multi-PostProcessor UI
사용자 인터페이스(Qt), 3D 시각화(PyVista), 2D 그래프(Matplotlib)를 담당하는 모듈입니다.
해석 엔진(whts_multipostprocessor_engine)과 연동하여 대시보드를 구성합니다.
"""

import os
import sys
import glob
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
from dataclasses import dataclass, field
from functools import partial
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

import pickle
# 엔진 모듈 임포트
from run_drop_simulator.whts_multipostprocessor_engine import (
    PlateAssemblyManager, 
    ShellDeformationAnalyzer, 
    PlateConfig,
    scale_result_to_mm
)
from run_drop_simulator.whts_mapping import get_assembly_data_from_sim

# [WHTOOLS] 공통 UI 컴포넌트 라이브러리 임포트
from run_drop_simulator.wht_ui_components import WHTRangeControlGroup, WHTRangeDialog

@dataclass
class PlotSlotConfig:
    """[WHTOOLS] 2D 그래프 슬롯별 시각화 구성 설정 데이터 본체"""
    part_indices: List[int] = field(default_factory=lambda: [0])
    plot_type: str = 'contour'
    data_key: str = 'Displacement [mm]'

    @property
    def part_idx(self) -> int:
        """Backward compatibility for single-part access"""
        return self.part_indices[0] if self.part_indices else 0

@dataclass
class DashboardConfig:
    """[WHTOOLS] 통합 대시보드 레이아웃 및 제어 전략 설정"""
    layout_2d: str = '2x1'
    plots_2d: List[PlotSlotConfig] = field(default_factory=list)
    v_font_size: int = 9
    animation_step: int = 1
    animation_speed_ms: int = 30


# ==============================================================================
# --- [Section 2] Helper Windows & Dialogs ---
# ==============================================================================

class PartManagerWindow(QtWidgets.QWidget):
    """
    [WHTOOLS] 파트 관리자 (Part Manager)
    트리 구조를 이용한 파트별 가시성, 렌더 모드 제어 및 실시간 해석 정보(Min/Max) 모니터링 창
    """
    def __init__(self, parent=None):
        super().__init__(parent, QtCore.Qt.Window | QtCore.Qt.WindowStaysOnTopHint)
        self.setWindowTitle("Part Manager")
        self.resize(500, 650)
        self.parent = parent
        
        layout = QtWidgets.QVBoxLayout(self)
        
        # 1. Global Control Section (Single Row)
        global_group = QtWidgets.QGroupBox("Global Control")
        global_layout = QtWidgets.QHBoxLayout(global_group)
        
        # Mesh Section
        global_layout.addWidget(QtWidgets.QLabel("Mesh:"))
        btn_m_show = QtWidgets.QPushButton("💡 Show All"); btn_m_show.clicked.connect(partial(self._bulk_set, 2, True))
        btn_m_hide = QtWidgets.QPushButton("🌑 Hide All"); btn_m_hide.clicked.connect(partial(self._bulk_set, 2, False))
        global_layout.addWidget(btn_m_show); global_layout.addWidget(btn_m_hide)
        
        # Separator Line
        v_line = QtWidgets.QFrame(); v_line.setFrameShape(QtWidgets.QFrame.VLine); v_line.setFrameShadow(QtWidgets.QFrame.Sunken)
        global_layout.addWidget(v_line)
        
        # Markers Section
        global_layout.addWidget(QtWidgets.QLabel("Markers:"))
        btn_k_show = QtWidgets.QPushButton("💡 Show All"); btn_k_show.clicked.connect(partial(self._bulk_set, 3, True))
        btn_k_hide = QtWidgets.QPushButton("🌑 Hide All"); btn_k_hide.clicked.connect(partial(self._bulk_set, 3, False))
        global_layout.addWidget(btn_k_show); global_layout.addWidget(btn_k_hide)
        
        global_layout.addStretch(1)
        layout.addWidget(global_group)
        
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection) # [WHT] Multi-selection
        self.tree.setHeaderLabels(["Part", "Clr", "Mesh", "Markers", "View Mode", "Info (Min / Max)"])
        self.tree.setColumnWidth(0, 160) # Part Name
        self.tree.setColumnWidth(1, 15)  # Clr (Color) - Ultra Slim
        self.tree.setColumnWidth(2, 50)  # Mesh (Narrow)
        self.tree.setColumnWidth(3, 100) # Markers
        self.tree.setColumnWidth(4, 180) # View Mode (Wide)
        
        self.tree.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.tree)
        
        # 3. Focus Mode Button
        self.btn_focus = QtWidgets.QPushButton("🎯 Focus Selection (Toggle Highlight)")
        self.btn_focus.setStyleSheet("font-weight: bold; padding: 5px;") # [WHT] Default BG color
        self.btn_focus.clicked.connect(self._on_focus_view)
        layout.addWidget(self.btn_focus)
        
        self.groups = {}
        self.id_to_item = {}
        self._init_tree()

    def _init_tree(self):
        """데이터 소스(Manager)로부터 트리 아이템 초기 생성"""
        self.tree.blockSignals(True)
        if not self.parent or not self.parent.mgr:
            print("[WHTOOLS-DEBUG] Part Manager: No parent manager found during init.")
            return
        
        n_analyzers = len(self.parent.mgr.analyzers)
        print(f"[WHTOOLS-DEBUG] Part Manager: Initializing tree with {n_analyzers} parts.")
        
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
            
            # [WHT] Color Picker Button (Ultra Slim version)
            btn_clr = QtWidgets.QPushButton()
            btn_clr.setFixedWidth(10)
            btn_clr.setFixedHeight(18)
            btn_clr.setCursor(QtCore.Qt.PointingHandCursor)
            c_hex = actor_data.get('body_color', "#888888")
            btn_clr.setStyleSheet(f"background-color: {c_hex}; border: 1px solid #555; margin: 0px; padding: 0px;")
            btn_clr.clicked.connect(partial(self._on_color_clicked, i))
            self.tree.setItemWidget(item, 1, btn_clr)
            
            item.setCheckState(2, m_v)
            item.setCheckState(3, mk_v)
            item.setText(3, f"Markers ({n_markers})")
            
            # View Mode ComboBox
            cmb = QtWidgets.QComboBox()
            cmb.addItems(["Surface w/ Edge", "Surface only", "Wireframe", "Outline"])
            cmb.setCurrentText(actor_data.get('view_mode', "Surface w/ Edge"))
            cmb.currentTextChanged.connect(partial(self._on_view_mode_changed, i))
            self.tree.setItemWidget(item, 4, cmb)
            
            item.setText(5, "-")
            self.id_to_item[i] = item
            
        self.tree.expandAll()
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

    def _on_view_mode_changed(self, part_idx, mode):
        # 1. Update the triggered item
        if part_idx in self.parent.part_actors:
            self.parent.part_actors[part_idx]['view_mode'] = mode
            
        # [WHT] 2. Batch Update: If multiple items are selected, update all of them
        selected_items = self.tree.selectedItems()
        target_item = self.id_to_item.get(part_idx)
        
        if target_item in selected_items:
            for item in selected_items:
                idx = item.data(0, QtCore.Qt.UserRole)
                if idx is not None and idx in self.parent.part_actors and idx != part_idx:
                    self.parent.part_actors[idx]['view_mode'] = mode
                    # Update the UI ComboBox without triggering signals recursively
                    cmb = self.tree.itemWidget(item, 3)
                    if isinstance(cmb, QtWidgets.QComboBox):
                        cmb.blockSignals(True)
                        cmb.setCurrentText(mode)
                        cmb.blockSignals(False)
        
        self.parent.update_frame(self.parent.current_frame)

    def _on_color_clicked(self, part_idx):
        """[WHTOOLS] 컬러 피커 다이얼로그 실행 및 멀티 선택 일괄 적용"""
        actor_data = self.parent.part_actors.get(part_idx)
        if not actor_data: return
        
        initial_qcolor = QtGui.QColor(actor_data.get('body_color', "#888888"))
        color = QtWidgets.QColorDialog.getColor(initial_qcolor, self, "Select Part Color")
        
        if color.isValid():
            new_hex = color.name()
            rgb_float = (color.redF(), color.greenF(), color.blueF())
            
            # 1. Identify targets (single vs batch)
            selected_items = self.tree.selectedItems()
            target_item = self.id_to_item.get(part_idx)
            
            target_indices = [part_idx]
            if target_item in selected_items:
                for item in selected_items:
                    idx = item.data(0, QtCore.Qt.UserRole)
                    if idx is not None and idx not in target_indices:
                        target_indices.append(idx)
            
            # 2. Apply to actors and UI
            for idx in target_indices:
                act_inf = self.parent.part_actors[idx]
                act_inf['body_color'] = new_hex
                # [WHT] 필드 데이터(Body/Face Color)도 즉시 업데이트하여 시각화 동기화
                ana = self.parent.mgr.analyzers[idx]
                if 'Body Color' in ana.results:
                    # 모든 프레임에 대해 색상 업데이트 (상수 색상 가정)
                    ana.results['Body Color'][:] = rgb_float[0] 
                    ana.results['Face Color'][:] = rgb_float[0] # [WHT] 기본적으로 동일하게 유지
                
                # UI 버튼 색상 업데이트
                item = self.id_to_item[idx]
                btn = self.tree.itemWidget(item, 1)
                if isinstance(btn, QtWidgets.QPushButton):
                    btn.setStyleSheet(f"background-color: {new_hex}; border: 1px solid #555;")
            
            self.parent.update_frame(self.parent.current_frame)

    def _on_focus_view(self):
        """[WHTOOLS] 선택한 파트들을 강조하고 나머지는 Wireframe으로 전환"""
        selected_items = self.tree.selectedItems()
        if not selected_items: return
        
        # 1. Get all selected indices
        target_ids = []
        for item in selected_items:
            idx = item.data(0, QtCore.Qt.UserRole)
            if idx is not None: target_ids.append(idx)
            
        if not target_ids: return
        
        self.tree.blockSignals(True)
        for i, act in self.parent.part_actors.items():
            if i in target_ids:
                act['visible'] = True
                act['view_mode'] = "Surface only"
                self.id_to_item[i].setCheckState(2, QtCore.Qt.Checked)
            else:
                act['view_mode'] = "Wireframe"
            
            # UI ComboBox 동기화
            widget = self.tree.itemWidget(self.id_to_item[i], 4)
            if isinstance(widget, QtWidgets.QComboBox):
                widget.setCurrentText(act['view_mode'])
                
        self.tree.blockSignals(False)
        self._on_focus() # [WHT] Also focus camera
        self.parent.update_frame(self.parent.current_frame)

    def update_info(self):
        """현재 선택된 필드 데이터의 Min/Max 정보를 트리 행에 업데이트"""
        # [WHT-DEBUG] cmb_comp 존재 여부 및 유효성 체크
        if not self.parent or not hasattr(self.parent, 'cmb_comp') or self.parent.cmb_comp is None:
            return
        
        try:
            f_idx = getattr(self.parent, 'current_frame', 0)
            # cmb_comp의 currentData() 또는 currentText() 사용
            fk = self.parent.cmb_comp.currentText()
        except (RuntimeError, AttributeError):
            return
        
        self.tree.blockSignals(True)
        for i, item in self.id_to_item.items():
            ana = self.parent.mgr.analyzers[i]
            if fk in ana.results and fk not in ["Body Color", "Face Color"]:
                val = ana.results[fk][f_idx]
                item.setText(4, f"{val.min():.2e} / {val.max():.2e}")
            else:
                item.setText(4, "-")
        self.tree.blockSignals(False)

    def _on_focus(self):
        """[WHTOOLS] 선택한 파트들의 바운딩 박스를 계산하여 카메라 초점을 맞춥니다."""
        selected_items = self.tree.selectedItems()
        if not selected_items: return
        
        bounds = []
        for item in selected_items:
            i = item.data(0, QtCore.Qt.UserRole)
            if i is not None and i in self.parent.part_actors:
                bounds.append(self.parent.part_actors[i]['poly'].bounds)
                
        if not bounds: return
        
        # 합집합 바운딩 박스 계산
        bounds = np.array(bounds)
        xmin, xmax = bounds[:, 0].min(), bounds[:, 1].max()
        ymin, ymax = bounds[:, 2].min(), bounds[:, 3].max()
        zmin, zmax = bounds[:, 4].min(), bounds[:, 5].max()
        
        self.parent.v_int.view_vector((1, 1, 1)) # Perspective view
        self.parent.v_int.reset_camera(bounds=[xmin, xmax, ymin, ymax, zmin, zmax])

    def _apply(self):
        """변경된 가시성 설정을 메인 렌더러에 즉각 반영"""
        for i, item in self.id_to_item.items():
            if i in self.parent.part_actors:
                # [WHT] Column index adjustment: 2: Mesh, 3: Markers
                self.parent.part_actors[i]['visible'] = (item.checkState(2) == QtCore.Qt.Checked)
                self.parent.part_actors[i]['visible_markers'] = (item.checkState(3) == QtCore.Qt.Checked)
        self.parent.update_frame(self.parent.current_frame)





class AddPlotDialog(QtWidgets.QDialog):
    """장면 내 Matplotlib 그래프 슬롯 추가/편집 대화상자 (동적 슬롯 선택 그리드 지원)"""
    def __init__(self, slot_idx, parts, field_keys, stat_keys, parent=None, rows=2, cols=1, plot_slots=None):
        super().__init__(parent)
        self.slot_idx = slot_idx
        self.parts = parts
        self.f_keys = field_keys
        self.s_keys = stat_keys
        self.plot_slots = plot_slots # [PlotSlotConfig, ...]
        
        self.setWindowTitle("Edit Plot Layout & Data")
        self.setMinimumWidth(450)
        
        layout = QtWidgets.QVBoxLayout(self)
        
        # --- 1. Slot Selection Grid (Dynamic Radio Buttons) ---
        slot_group = QtWidgets.QGroupBox("Target Slot Selection")
        slot_grid = QtWidgets.QGridLayout(slot_group)
        self.slot_buttons = []
        self.bg = QtWidgets.QButtonGroup(self)
        
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                rb = QtWidgets.QRadioButton(f"Slot {idx + 1}")
                slot_grid.addWidget(rb, r, c)
                self.bg.addButton(rb, idx)
                self.slot_buttons.append(rb)
                if idx == self.slot_idx:
                    rb.setChecked(True)
        
        self.bg.idClicked.connect(self._on_slot_switched)
        layout.addWidget(slot_group)
        
        # --- 2. Data Settings Grid ---
        data_group = QtWidgets.QGroupBox("Data Settings")
        grid = QtWidgets.QGridLayout(data_group)
        
        # Part Selection (Checklist)
        grid.addWidget(QtWidgets.QLabel("Part:"), 0, 0, QtCore.Qt.AlignTop)
        self.list_parts = QtWidgets.QListWidget()
        self.list_parts.setMinimumHeight(150)
        
        self.item_all = QtWidgets.QListWidgetItem("All Parts")
        self.item_all.setCheckState(QtCore.Qt.Unchecked)
        self.list_parts.addItem(self.item_all)
        
        self.item_main = QtWidgets.QListWidgetItem("All Main Parts")
        self.item_main.setCheckState(QtCore.Qt.Unchecked)
        self.list_parts.addItem(self.item_main)
        
        self.part_items = []
        for i, p in enumerate(parts):
            item = QtWidgets.QListWidgetItem(p)
            item.setCheckState(QtCore.Qt.Unchecked)
            item.setData(QtCore.Qt.UserRole, i)
            self.list_parts.addItem(item)
            self.part_items.append(item)
        
        grid.addWidget(self.list_parts, 0, 1)
        self.list_parts.itemChanged.connect(self._on_item_changed)
        
        # Plot Type
        grid.addWidget(QtWidgets.QLabel("Type:"), 1, 0)
        hb = QtWidgets.QHBoxLayout()
        self.rb_c = QtWidgets.QRadioButton("Contour")
        self.rb_cur = QtWidgets.QRadioButton("Curve")
        hb.addWidget(self.rb_c); hb.addWidget(self.rb_cur)
        grid.addLayout(hb, 1, 1)
        
        # Data Key
        grid.addWidget(QtWidgets.QLabel("Key:"), 2, 0)
        self.cmb_key = QtWidgets.QComboBox()
        grid.addWidget(self.cmb_key, 2, 1)
        
        layout.addWidget(data_group)
        
        # Signals
        self.rb_c.toggled.connect(self._update_keys)
        self.rb_cur.toggled.connect(self._update_keys)
        
        # OK/Cancel
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)
        
        # Initial Load
        self._load_slot_config(self.slot_idx)

    def _on_slot_switched(self, idx):
        """슬롯 라디오 버튼 클릭 시 해당 슬롯의 설정 로드"""
        self.slot_idx = idx
        self._load_slot_config(idx)

    def _load_slot_config(self, idx):
        """특정 슬롯의 설정을 UI에 반영"""
        config = self.plot_slots[idx] if (self.plot_slots and idx < len(self.plot_slots)) else None
        
        # Reset parts
        self.list_parts.blockSignals(True)
        self.item_all.setCheckState(QtCore.Qt.Unchecked)
        self.item_main.setCheckState(QtCore.Qt.Unchecked)
        for item in self.part_items:
            item.setCheckState(QtCore.Qt.Unchecked)
            
        if config:
            # Load parts
            if -1 in config.part_indices: self.item_all.setCheckState(QtCore.Qt.Checked)
            elif -2 in config.part_indices: self.item_main.setCheckState(QtCore.Qt.Checked)
            else:
                for p_idx in config.part_indices:
                    if 0 <= p_idx < len(self.part_items):
                        self.part_items[p_idx].setCheckState(QtCore.Qt.Checked)
            
            # Load type
            if config.plot_type == "contour": self.rb_c.setChecked(True)
            else: self.rb_cur.setChecked(True)
            
            # Update keys and select
            self._update_keys()
            k_idx = self.cmb_key.findText(config.data_key)
            if k_idx >= 0: self.cmb_key.setCurrentIndex(k_idx)
        else:
            # Default to Contour if no config
            self.rb_c.setChecked(True)
            self._update_keys()
            
        self.list_parts.blockSignals(False)

    def _update_keys(self):
        """선택된 그래프 타입에 따른 필터링된 키 목록 제공 (Max- 가상 키 포함)"""
        self.cmb_key.clear()
        if self.rb_c.isChecked():
            self.cmb_key.addItems(self.f_keys)
        else:
            # Curve인 경우 기본 통계량과 더불어 필드 데이터의 Max값을 추출하는 가상 키 추가
            keys = self.s_keys + [f"Max-{k}" for k in self.f_keys]
            self.cmb_key.addItems(keys)

    def _on_item_changed(self, item):
        self.list_parts.blockSignals(True)
        if item == self.item_all and item.checkState() == QtCore.Qt.Checked:
            self.item_main.setCheckState(QtCore.Qt.Unchecked)
            for pi in self.part_items: pi.setCheckState(QtCore.Qt.Unchecked)
        elif item == self.item_main and item.checkState() == QtCore.Qt.Checked:
            self.item_all.setCheckState(QtCore.Qt.Unchecked)
            for pi in self.part_items: pi.setCheckState(QtCore.Qt.Unchecked)
        elif item in self.part_items and item.checkState() == QtCore.Qt.Checked:
            self.item_all.setCheckState(QtCore.Qt.Unchecked)
            self.item_main.setCheckState(QtCore.Qt.Unchecked)
            
        self.list_parts.blockSignals(False)
        
        # 선택된 파트 개수에 따라 그래프 타입(Contour/Curve) 제한
        checked_count = 0
        if self.item_all.checkState() == QtCore.Qt.Checked: checked_count = 999
        elif self.item_main.checkState() == QtCore.Qt.Checked: checked_count = 999
        else:
            checked_count = sum(1 for pi in self.part_items if pi.checkState() == QtCore.Qt.Checked)
            
        if checked_count > 1 or checked_count == 0:
            self.rb_cur.setChecked(True)
            self.rb_c.setEnabled(False)
        else:
            self.rb_c.setEnabled(True)
            
        self._update_keys()

    def get_config(self) -> PlotSlotConfig:
        """대화상자에서 결정된 GUI 구성을 설정 객체로 반환"""
        indices = []
        if self.item_all.checkState() == QtCore.Qt.Checked:
            indices = [-1]
        elif self.item_main.checkState() == QtCore.Qt.Checked:
            indices = [-2]
        else:
            indices = [pi.data(QtCore.Qt.UserRole) for pi in self.part_items if pi.checkState() == QtCore.Qt.Checked]
            
        if not indices: indices = [0] # Default
            
        return PlotSlotConfig(
            part_indices=indices,
            plot_type="contour" if self.rb_c.isChecked() else "curve", 
            data_key=self.cmb_key.currentText()
        )


class OpenSettingsDialog(QtWidgets.QDialog):
    """
    [WHTOOLS] 결과 로드 전 해석 옵션 설정을 위한 대화상자
    데이터를 미리 스캔하여 파트별 마커 수와 추천 차수를 제안합니다.
    """
    def __init__(self, result_data, default_parts, parent=None):
        super().__init__(parent)
        self.setWindowTitle("✨ Fitting the plate deformation 🛠️")
        self.setMinimumWidth(550)
        self.result_data = result_data
        
        layout = QtWidgets.QVBoxLayout(self)
        
        # 1. 상단 기본 설정
        top_group = QtWidgets.QGroupBox("Global Settings")
        grid = QtWidgets.QGridLayout(top_group)
        layout.addWidget(top_group)
        
        grid.addWidget(QtWidgets.QLabel("Resolution (sol.res):"), 0, 0)
        self.sp_res = QtWidgets.QSpinBox()
        self.sp_res.setRange(5, 200); self.sp_res.setValue(20)
        grid.addWidget(self.sp_res, 0, 1)
        
        grid.addWidget(QtWidgets.QLabel("Target Parts (prefixes):"), 1, 0)
        self.le_parts = QtWidgets.QLineEdit(", ".join(default_parts))
        grid.addWidget(self.le_parts, 1, 1)
        
        self.btn_scan = QtWidgets.QPushButton("🔍 Scan Parts")
        self.btn_scan.clicked.connect(self._do_scan)
        grid.addWidget(self.btn_scan, 1, 2)
        
        # 2. 스마트 스캔 결과 테이블
        layout.addWidget(QtWidgets.QLabel("<b>Part Analysis & Degree Overrides:</b>"))
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Part/Face", "Markers", "Target Degree"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        layout.addWidget(self.table)
        
        # 3. 고급 옵션
        adv_group = QtWidgets.QGroupBox("Advanced Params")
        adv_grid = QtWidgets.QGridLayout(adv_group)
        layout.addWidget(adv_group)
        
        adv_grid.addWidget(QtWidgets.QLabel("Marker Mode:"), 0, 0)
        self.cmb_mode = QtWidgets.QComboBox()
        self.cmb_mode.addItems(["statistical", "direct"])
        adv_grid.addWidget(self.cmb_mode, 0, 1)
        
        adv_grid.addWidget(QtWidgets.QLabel("Default Reg. Lambda:"), 0, 2)
        self.le_lam = QtWidgets.QLineEdit("1e-4")
        adv_grid.addWidget(self.le_lam, 0, 3)

        # 4. 데이터 해석 모드 선택 (v5 vs v6 스타일 결정)
        mode_group = QtWidgets.QGroupBox("Data Interpretation Mode")
        mode_layout = QtWidgets.QVBoxLayout(mode_group)
        self.rb_v5 = QtWidgets.QRadioButton("마커 재구성 모드 (경계 근사형) - Marker Reconstruction")
        self.rb_v6 = QtWidgets.QRadioButton("정밀 형상 모드 (경계 확정형) - Verified Geometry (Exact)")
        
        # [WHT] 기능 설명을 툴팁으로 전환하여 UI를 깔끔하게 유지
        self.rb_v5.setToolTip("마커 위치를 추적하여 형상을 유추합니다.\n정확한 외곽 치수(박스 크기) 보장이 어려울 수 있습니다.")
        self.rb_v6.setToolTip("코너 및 정밀 메쉬 정보를 직접 사용하여\n실제 설계 치수와 완벽히 일치하는 결과를 보장합니다.")
        
        # [WHT] 사용자의 요청에 따라 '정밀 형상 모드'를 기본값으로 설정
        self.rb_v6.setChecked(True)
            
        mode_layout.addWidget(self.rb_v5)
        mode_layout.addWidget(self.rb_v6)
        layout.addWidget(mode_group)

        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)
        
        self._do_scan()

    def _do_scan(self):
        """데이터를 스캔하여 파트 목록과 마커 수를 추출합니다."""
        target_prefixes = [p.strip() for p in self.le_parts.text().split(",") if p.strip()]
        mode = self.cmb_mode.currentText()
        
        try:
            # 1. 일차적으로 입력된 접두사들을 기준으로 스캔 시도
            markers, _ = get_assembly_data_from_sim(self.result_data, target_prefixes, mode=mode)
            
            # [WHT-SMART] 만약 검색된 파트가 없다면, 데이터 내의 모든 컴포넌트를 자동으로 탐색
            # [WHT-FIX] any() 대신 명시적인 존재 여부 체크로 Ambiguous Truth Value 에러 방지
            has_data = any(len(faces) > 0 for faces in markers.values())
            if not has_data:
                print("[WHTOOLS-SCAN] Default prefixes failed. Attempting smart auto-scan of all components...")
                if hasattr(self.result_data, 'components'):
                    all_keys = list(self.result_data.components.keys())
                    markers, _ = get_assembly_data_from_sim(self.result_data, all_keys, mode=mode)
                    # UI에도 감지된 접두사들을 업데이트하여 사용자에게 알림
                    self.le_parts.setText(", ".join(all_keys))

            self.table.setRowCount(0)
            self.part_configs = []
            
            for p_name, faces in markers.items():
                for f_name, m_data in faces.items():
                    if m_data is None or len(m_data) == 0: continue
                    row = self.table.rowCount()
                    self.table.insertRow(row)
                    
                    full_name = f"{p_name}_{f_name}"
                    n_markers = len(m_data)
                    
                    # 엔진 로직에 따른 추천 차수 계산
                    rec_deg = 4
                    if n_markers < 16: rec_deg = 2
                    elif n_markers < 25: rec_deg = 3
                    
                    self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(full_name))
                    self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(n_markers)))
                    
                    sp_deg = QtWidgets.QSpinBox()
                    sp_deg.setRange(0, 8); sp_deg.setValue(rec_deg)
                    self.table.setCellWidget(row, 2, sp_deg)
                    
                    self.part_configs.append({'name': full_name, 'markers': m_data, 'sp_deg': sp_deg})
        except Exception as e:
            print(f"Scan Error: {e}")

    def get_settings(self):
        try: lam = float(self.le_lam.text())
        except: lam = 1e-4
        
        # 각 파트별 개별 차수 설정 추출
        overrides = {}
        for cfg in self.part_configs:
            overrides[cfg['name']] = cfg['sp_deg'].value()
            
        return {
            'res': self.sp_res.value(),
            'mode': self.cmb_mode.currentText(),
            'lambda': float(self.le_lam.text()),
            'parts_data': self.part_configs,
            'overrides': overrides,
            'interpretation_mode': 'v5' if self.rb_v5.isChecked() else 'v6'
        }


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
    
    def __init__(self, manager: PlateAssemblyManager = None, config: DashboardConfig = None, ground_size=(4000, 4000)):
        """
        대시보드 초기화 및 핵심 데이터 바인딩.
        """
        super().__init__()
        if manager:
            print(f"[WHTOOLS-UI] Initializing Dashboard with {len(manager.analyzers)} parts...")
        else:
            print(f"[WHTOOLS-UI] Initializing Dashboard in Standalone Mode (No data)...")
        
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
        self.cbs = [None] * 6
        
        # 2. 바닥(Floor) 상태 설정
        self.floor_origin = [0, 0, 0]
        self.floor_normal = [0, 0, 1]
        self.floor_w, self.floor_h = ground_size
        
        # [WHTOOLS] Matplotlib Font Setup (Noto Sans KR for Korean/Emoji)
        plt.rcParams['axes.unicode_minus'] = False
        
        # 3. 데이터 바인딩 (있을 경우에만)
        self.field_keys = []
        self.stat_keys = []
        if self.mgr:
            self._bind_manager_data()
        
        # 4. 리소스 경로 설정
        self.res_dir = os.path.join(os.path.dirname(__file__), "resources")
        self.logo_path = os.path.join(self.res_dir, "logo.png")
        # 5. UI 엔진 가동
        self.statusBar().showMessage("WHTOOLS Ready")
        
        # 메인 시퀀스 실행
        self.visibility_tool = PartManagerWindow(self)
        self._is_first_2d_update = True  # tight_layout 트리거용
        self._init_ui()
        self._init_3d_view()
        self._init_2d_plots()
        
        if self.mgr:
            self._apply_initial_preset()
            # [WHTOOLS] Ensure initial font settings are applied
            self._update_vtk_font(self.v_font_size)
            self.update_frame(0)
            # [WHT] 기본 배경 스타일 설정 (Light Grey Grad.)
            QtCore.QTimer.singleShot(100, lambda: self._on_bg_changed("Light Grey Grad."))

    def _bind_manager_data(self):
        """[WHTOOLS] 매니저 데이터에서 필드 키 및 통계 키 추출"""
        if not self.mgr or not self.mgr.analyzers:
            return
            
        valid_analyzers = [a for a in self.mgr.analyzers if a.results]
        if not valid_analyzers:
            return

        p0 = valid_analyzers[0]
        n_f = len(self.mgr.times)
        
        # sol이 없을 경우(v6 로딩 등) 결과 데이터 형상에서 res 유추
        res_val = 20
        if hasattr(p0, 'sol') and p0.sol: res_val = p0.sol.res
        elif 'Displacement [mm]' in p0.results:
            res_val = p0.results['Displacement [mm]'].shape[1]
            
        res_sq = res_val**2
        
        self.field_keys = [
            k for k in p0.results 
            if p0.results[k].ndim == 3 and p0.results[k].size // n_f == res_sq
        ]
        self.stat_keys = [
            k for k in p0.results 
            if k not in self.field_keys and p0.results[k].ndim < 3
        ] + ['Marker Local Disp. [mm]', 'Marker Global Disp. [mm]']
        
        # [WHTOOLS] Populate Field Combos
        self._populate_category_combos()

    # --------------------------------------------------------------------------
    # --- UI Layout & Component Setup ---
    # --------------------------------------------------------------------------

    def _init_ui(self):
        """메인 레이아웃 및 탭 구조 초기화"""
        # [WHTOOLS] 전역 스타일 시트 적용
        QtWidgets.QApplication.instance().setStyleSheet("QPushButton { padding-left: 8px; padding-right: 8px; padding-top: 4px; padding-bottom: 4px; min-width: 30px; }")
        
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
        
        # 탭 위젯 (Data / 3D / 2D / 설정)
        self.ct = QtWidgets.QTabWidget()
        self.t_data = QtWidgets.QWidget()
        self.t3 = QtWidgets.QWidget()
        self.t2 = QtWidgets.QWidget()
        self.ts = QtWidgets.QWidget()
        
        self.ct.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #CCC; } 
            QTabBar::tab { 
                padding: 6px 20px; 
                border: 1px solid #CCC;
                border-bottom: none;
                margin-right: 2px;
                font-size: 9pt;
            } 
            QTabBar::tab:selected { 
                background: #666; 
                font-weight: bold;
            }
        """)
        self.ct.addTab(self.t_data, "📂 Data")
        self.ct.addTab(self.t3, "🧊 3D Field")
        self.ct.addTab(self.t2, "📈 2D Field/Curves")
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
        
        # 초기 화면 분할 비율 설정
        self.split.setSizes([1130, 570]) 
        self.split.setStretchFactor(0, 2)
        self.split.setStretchFactor(1, 1)
        
        # 각 컨트롤 패널 초기화
        self._init_data_controls(self.t_data)
        self._init_3d_controls(self.t3)
        self._init_2d_controls(self.t2)
        self._init_settings_controls(self.ts)
        self._init_animation_dock()

    def _create_v_line(self):
        """MS Office 스타일의 수직 구분선 생성"""
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.VLine)
        line.setFrameShadow(QtWidgets.QFrame.Plain)
        line.setStyleSheet("color: #A5C7E9; margin: 5px 0px;")
        return line

    def _init_data_controls(self, tab):
        """데이터 관리(Data) 탭 초기화 - Ribbon Style"""
        layout = QtWidgets.QHBoxLayout(tab)
        layout.setContentsMargins(15, 2, 15, 2)
        layout.setSpacing(10)

        f_open = QtWidgets.QFrame()
        l_open = QtWidgets.QHBoxLayout(f_open)
        l_open.setContentsMargins(5, 5, 5, 5)
        
        self.btn_latest = QtWidgets.QPushButton("🆕 Open Latest")
        self.btn_latest.setMinimumHeight(32)
        self.btn_latest.clicked.connect(self._on_open_latest)
        l_open.addWidget(self.btn_latest)
        
        self.btn_open = QtWidgets.QPushButton("📁 Open File...")
        self.btn_open.setMinimumHeight(32)
        self.btn_open.clicked.connect(lambda: self._on_open_file())
        l_open.addWidget(self.btn_open)
        
        layout.addWidget(f_open)
        layout.addWidget(self._create_v_line())
        layout.addStretch(1)

    def _on_open_latest(self):
        """[WHTOOLS] results 폴더 내 가장 최근의 .pkl 파일을 찾아 로드합니다."""
        patterns = [
            "results/rds-*/simulation_result.pkl",
            "results/latest_results.pkl",
            "results/*.pkl"
        ]
        
        all_files = []
        for p in patterns:
            all_files.extend(glob.glob(p))
            
        if not all_files:
            self._show_warning("No Results", "No simulation result (.pkl) found in 'results' folder.")
            return
            
        all_files.sort(key=os.path.getmtime, reverse=True)
        latest_path = all_files[0]
        
        self.statusBar().showMessage(f"Auto-found latest: {latest_path}")
        self._on_open_file(latest_path)

    def _on_open_file(self, path=None):
        """[WHTOOLS] 결과 파일(.pkl) 로드 및 UI 핫-리로딩"""
        if not path:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Open Simulation Result", "", "Result Files (*.pkl);;All Files (*)"
            )
        if not path: return
        
        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            self.statusBar().showMessage(f"Loading {os.path.basename(path)}...")
            QtWidgets.QApplication.processEvents()

            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            data = scale_result_to_mm(data)
            
            while QtWidgets.QApplication.overrideCursor():
                QtWidgets.QApplication.restoreOverrideCursor()
            
            if hasattr(data, 'time_history'):
                default_parts = ['bpaperbox', 'bcushion', 'bchassis', 'bopencell']
                sett_dlg = OpenSettingsDialog(data, default_parts, self)
                if not sett_dlg.exec(): return
                sett = sett_dlg.get_settings()
                
                QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
                print(f"[WHTOOLS] Processing Raw Result (v5) with custom per-part settings...")
                times = np.array(data.time_history)
                
                n_parts_tot = len(sett['parts_data'])
                new_mgr = PlateAssemblyManager(times)
                
                for i, cfg in enumerate(sett['parts_data']):
                    full_name = cfg['name']
                    m_dict = cfg['markers']
                    deg = sett['overrides'][full_name]
                    
                    msg = f"📦 [{i+1}/{n_parts_tot}] Initializing {full_name}..."
                    self.statusBar().showMessage(msg)
                    QtWidgets.QApplication.processEvents()
                    
                    m_names = sorted(list(m_dict.keys()))
                    m_array = np.stack([m_dict[name] for name in m_names], axis=0).transpose(1, 0, 2)
                    
                    ana = ShellDeformationAnalyzer(name=full_name)
                    ana.m_raw = m_array
                    ana.cfg.mesh_resolution = sett['res']
                    ana.cfg.polynomial_degree = deg
                    ana.cfg.regularization_lambda = sett['lambda']
                    new_mgr.add_analyzer(ana)
                
                for i, ana in enumerate(new_mgr.analyzers):
                    msg = f"⏳ Analyzing {ana.name} ({i+1}/{n_parts_tot})..."
                    self.statusBar().showMessage(msg)
                    QtWidgets.QApplication.processEvents()
                    ana.run_analysis(sim_data=data)
                    
            elif isinstance(data, dict) and 'times' in data:
                QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
                new_mgr = PlateAssemblyManager(data['times'])
                for i, (name, res) in enumerate(data['analyzers'].items()):
                    self.statusBar().showMessage(f"🚚 Loading {name}...")
                    QtWidgets.QApplication.processEvents()
                    dummy = ShellDeformationAnalyzer(name=name)
                    dummy.results = res
                    dummy.W, dummy.H = 100.0, 100.0 
                    if 'W' in res: dummy.W = float(res['W'])
                    if 'H' in res: dummy.H = float(res['H'])
                    new_mgr.analyzers.append(dummy)
            else:
                raise ValueError("Unknown .pkl data format.")
            
            if not new_mgr.analyzers:
                raise ValueError("No parts found in the loaded data.")
                
            self.last_path = path
            self.load_new_manager(new_mgr)
            self.setWindowTitle(f"✨ WHTOOLS Multi-PostProcessor [V2] - {os.path.basename(path)}")
            self.statusBar().showMessage(f"✅ Successfully loaded: {os.path.basename(path)}")
            
        except Exception as e:
            while QtWidgets.QApplication.overrideCursor():
                QtWidgets.QApplication.restoreOverrideCursor()
            self._show_critical_error("Load Error", f"Failed to load file:\n{str(e)}")
        finally:
            while QtWidgets.QApplication.overrideCursor():
                QtWidgets.QApplication.restoreOverrideCursor()

    def load_new_manager(self, manager):
        """[WHTOOLS] 새로운 매니저로 대시보드 데이터 및 뷰 갱신"""
        self.mgr = manager
        
        # 1. 3D 뷰 및 액터 초기화
        self.v_int.clear()
        self.part_actors = {}
        self._init_3d_view()
        
        # 2. UI 상태 및 필드 선택 목록(cmb_3d) 먼저 초기화
        self.current_frame = 0
        self.is_playing = False
        
        if hasattr(self, 'timer') and self.timer is not None:
            try:
                self.timer.stop()
                self.timer.timeout.disconnect()
            except: pass
            self.timer.deleteLater()
            self.timer = None
        if hasattr(self, 'bp'): self.bp.setText("▶️")
        
        valid_analyzers = [a for a in manager.analyzers if a.results]
        if valid_analyzers:
            p0 = valid_analyzers[0]
            self.field_keys = [k for k in p0.results if p0.results[k].ndim == 3]
            self.stat_keys = [k for k in p0.results if k not in self.field_keys]
        else:
            self.field_keys = []; self.stat_keys = []
            
        if hasattr(self, 'cmb_cat'):
            self.cmb_cat.blockSignals(True)
            self.cmb_cat.clear()
            
            # [WHT] 카테고리 맵 구축: {Category: [(Display Name, Full Key), ...]}
            self._categories = {"Basic": [("Body Color", "Body Color"), ("Face Color", "Face Color")]}
            
            for k in self.field_keys:
                # 필드 이름의 첫 단어를 카테고리로 사용 (예: "Stress XX" -> "Stress")
                parts = k.split(' ')
                cat = parts[0] if len(parts) > 1 else "Other"
                comp = " ".join(parts[1:]) if len(parts) > 1 else k
                
                if cat not in self._categories:
                    self._categories[cat] = []
                self._categories[cat].append((comp, k))
            
            self.cmb_cat.addItems(sorted(list(self._categories.keys())))
            self.cmb_cat.blockSignals(False)
            
            # 4. 카테고리별 첫 항목 자동 선택 및 화면 갱신
            if self._categories and len(self._categories) > 0:
                first_cat = list(self._categories.keys())[0]
                idx = self.cmb_cat.findText(first_cat)
                if idx >= 0: self.cmb_cat.setCurrentIndex(idx)
        
        # [WHT] 2D 관련 위젯 동기화 (cmb_p2d, cmb_f2d)
        if hasattr(self, 'cmb_p2d'):
            self.cmb_p2d.blockSignals(True)
            self.cmb_p2d.clear()
            self.cmb_p2d.addItems(["[All Parts]"] + [a.name for a in self.mgr.analyzers])
            self.cmb_p2d.blockSignals(False)
            
        if hasattr(self, 'cmb_f2d'):
            self.cmb_f2d.blockSignals(True)
            self.cmb_f2d.clear()
            # 2D는 보통 Scalar 값 위주이므로 field_keys 전체를 표시
            self.cmb_f2d.addItems(self.field_keys)
            self.cmb_f2d.blockSignals(False)

        # [WHT-INITIAL] 4. 초기 플롯 구성을 위한 자동 설정
        self._on_clear_2d_plots() # [WHT] 기존 플롯 데이터 및 설정 완전 초기화
        self.plot_slots = [None] * 6
        
        # (1) 2D Grid Setup
        if hasattr(self, 'cmb_lay'):
            self.cmb_lay.blockSignals(True)
            self.cmb_lay.setCurrentText("2x1")
            self.cmb_lay.blockSignals(False)
            self._on_grid_layout_changed("2x1")

        # (2) Slot 1: Contour (Principal Strain)
        target_contour_key = next((k for k in self.field_keys if "Prin.Strain" in k or "Principal" in k), None)
        if not target_contour_key and self.field_keys: target_contour_key = self.field_keys[0]
        
        if target_contour_key:
            self.plot_slots[0] = PlotSlotConfig(
                part_indices=[-1], # All Parts
                plot_type="contour",
                data_key=target_contour_key
            )
            
        # (3) [WHTOOLS] Slot 2 Default: Curve (Max-Curvature Mean) - User Request
        self.plot_slots[1] = PlotSlotConfig(
            part_indices=[-2], # All Main Parts
            plot_type="curve",
            data_key="Max-Curvature Mean [1/mm]"
        )
        
        # (4) 3D Force Refresh (Colormap & Initial Frame)
        self._on_cmap_changed()  # [WHT] 컬러맵 설정
        self.update_frame(0)    # [WHT] 프레임 업데이트
        if hasattr(self, 'fig'):
            self.fig.tight_layout()
            self.can.draw_idle()
        
        # 3. 모든 데이터가 준비된 후 파트 매니저 동기화
        was_visible = False
        if hasattr(self, 'visibility_tool') and self.visibility_tool is not None:
            try:
                was_visible = self.visibility_tool.isVisible()
                self.visibility_tool.close()
                self.visibility_tool.deleteLater()
            except RuntimeError:
                pass
            self.visibility_tool = None
            
        self.visibility_tool = PartManagerWindow(self)
        if was_visible:
            self.visibility_tool.show()
        
        # [WHT] 파트 매니저가 열려 있다면 즉시 정보 갱신
        if hasattr(self, 'visibility_tool') and self.visibility_tool is not None:
            try: self.visibility_tool.update_info()
            except: pass
        
        if hasattr(self, 'sld'):    # [WHT] 슬라이더 초기화
            n_frames = len(self.mgr.times) if self.mgr.times is not None else 1
            self.sld.setRange(0, n_frames - 1); self.sld.setValue(0)
            
        self.plot_slots = [None] * 6; self.ims = [None] * 6
        self.vls = [None] * 6; self.cbs = [None] * 6
        self._init_2d_plots()
        
        fname = os.path.basename(getattr(self, 'last_path', 'Data Loaded'))
        self.setWindowTitle(f"WHTOOLS Structural Dashboard - {fname}")
        self._apply_initial_preset()
        self.update_frame(0)
        self.v_int.reset_camera()
        self.v_int.render()

    def _init_settings_controls(self, p):
        """환경 설정 패널 구성 - Ribbon Style (Colormap 설정 포함)"""
        layout = QtWidgets.QHBoxLayout(p)
        layout.setContentsMargins(15, 2, 15, 2); layout.setSpacing(10)
        
        f_vis = QtWidgets.QFrame(); l_vis = QtWidgets.QHBoxLayout(f_vis); l_vis.setContentsMargins(0, 0, 0, 0)
        b_res = QtWidgets.QPushButton("🔄 Reset View"); b_res.clicked.connect(lambda: self.v_int.reset_camera()); l_vis.addWidget(b_res)
        layout.addWidget(f_vis); layout.addWidget(self._create_v_line())
        
        f_ani = QtWidgets.QFrame(); l_ani = QtWidgets.QHBoxLayout(f_ani); l_ani.setContentsMargins(0, 0, 0, 0)
        l_ani.addWidget(QtWidgets.QLabel("Font size:")); self.sp_vtk_font = QtWidgets.QSpinBox()
        self.sp_vtk_font.setRange(6, 30); self.sp_vtk_font.setValue(9); self.sp_vtk_font.valueChanged.connect(self._update_vtk_font); l_ani.addWidget(self.sp_vtk_font)
        layout.addWidget(f_ani); layout.addWidget(self._create_v_line())
        
        b_abt = QtWidgets.QPushButton("ℹ️ About"); b_abt.clicked.connect(self._show_about); layout.addWidget(b_abt)
        layout.addStretch(1)

    def _update_vtk_font(self, v):
        """[WHTOOLS] 전역 폰트 크기 업데이트 (3D & 2D 통합)"""
        self.v_font_size = v
        # 1. PyVista Global Theme
        pv.global_theme.font.size = v
        pv.global_theme.font.label_size = max(6, v - 2)
        # 2. Matplotlib Global Params
        plt.rcParams['font.size'] = v
        # 3. UI Sync (2D Font ComboBox)
        if hasattr(self, 'cmb_font_2d'):
            self.cmb_font_2d.blockSignals(True)
            self.cmb_font_2d.setCurrentText(str(v))
            self.cmb_font_2d.blockSignals(False)
        # 4. Immediate Refresh (PyVista Text & Scalar Bar)
        if hasattr(self, 'ov'): 
            self.ov.prop.font_size = max(6, v - 2)
        if hasattr(self, 'gui_txt'): 
            self.gui_txt.prop.font_size = max(6, v - 2)
        
        if hasattr(self, 'sb') and self.sb is not None:
            try:
                # vtkScalarBarActor use LabelTextProperty and TitleTextProperty
                self.sb.GetLabelTextProperty().SetFontSize(v + 2)
                self.sb.GetTitleTextProperty().SetFontSize(v + 2)
                self.sb.SetUnconstrainedFontSize(True)
            except: pass
        
        # [WHT] Actor labels update
        if hasattr(self, 'part_actors'):
            for act in self.part_actors.values():
                if 'labels' in act and act['labels']:
                    try: 
                        # add_point_labels returns a vtkActor2D with a vtkTextMapper/vtkLabeledDataMapper
                        # The actor itself might have GetTextProperty()
                        if hasattr(act['labels'], 'GetTextProperty'):
                            act['labels'].GetTextProperty().SetFontSize(max(6, v - 2))
                    except: pass
                    
        self.update_frame(self.current_frame)

    def _on_2d_font_changed(self, text):
        """[WHTOOLS] 2D 그래프(Matplotlib)의 전역 폰트 크기 변경"""
        try:
            sz = int(text)
            plt.rcParams['font.size'] = sz
            self.update_frame(self.current_frame)
        except: pass

    def _apply_initial_preset(self):
        """[WHTOOLS] 파일 로딩 직후 최적의 시각화 상태(프리셋)를 자동으로 설정합니다."""
        target_name = "opencell_front"; target_idx = -1
        for i, ana in enumerate(self.mgr.analyzers):
            if target_name in ana.name.lower(): target_idx = i; break
        
        # [WHT] 데이터 카테고리에 따른 지능형 프리셋
        if hasattr(self, 'cmb_cat') and hasattr(self, '_categories'):
            if "Curvature" in self._categories:
                self.cmb_cat.setCurrentText("Curvature")
                self._on_category_changed("Curvature")
                self.cmb_comp.setCurrentText("Curvature Mean [1/mm]")
            elif self.cmb_cat.count() > 1:
                # "Basic" 이외의 첫 번째 실제 데이터 카테고리 선택
                self.cmb_cat.setCurrentIndex(1)
            elif self.cmb_cat.count() > 0:
                self.cmb_cat.setCurrentIndex(0)
                
        if hasattr(self, 'cmb_lay'):
            self.cmb_lay.setCurrentText("2x1")
            
        if target_idx != -1:
            # Opencell이 있으면 해당 파트를 1번 슬롯(Contour)에 자동 할당
            self.plot_slots[0] = PlotSlotConfig(part_indices=[target_idx], plot_type="contour", data_key=self.cmb_comp.currentText())
        
        # [WHTOOLS] Slot 2: [Multi] Max-Curvature Mean (User Request Default)
        # 파일 로딩 직후 Slot 2는 항상 주요 파트의 최대 곡률 추이를 보여주도록 설정합니다.
        self.plot_slots[1] = PlotSlotConfig(
            part_indices=[-2], # All Main Parts
            plot_type="curve", 
            data_key="Max-Curvature Mean [1/mm]"
        )
        
        # UI 슬롯 선택 콤보박스 업데이트 (필요 시)
        if hasattr(self, 'cmb_slot'):
            self.cmb_slot.blockSignals(True)
            self.cmb_slot.setCurrentIndex(1) # Slot 2 선택 상태로 시작 (선택 사항)
            self.cmb_slot.blockSignals(False)
            
        # [WHTOOLS] 초기 로딩 시 1회 정렬 (성능 최적화)
        self.fig.tight_layout(pad=3.0)
        self.can.draw_idle()

    def _update_step(self, v):
        self.anim_step = v
        if hasattr(self, 'sp_step_ui'):
            self.sp_step_ui.blockSignals(True); self.sp_step_ui.setValue(v); self.sp_step_ui.blockSignals(False)

    def _init_3d_controls(self, tab):
        """[WHTOOLS] 3D Field Control Panel - Professional Ribbon Style (WHT Inspector ported)"""
        layout = QtWidgets.QHBoxLayout(tab)
        layout.setContentsMargins(10, 2, 10, 2); layout.setSpacing(8)

        # [Group 1] Part & Deform
        g_part = QtWidgets.QGroupBox("Structure & Deform")
        l_part_main = QtWidgets.QVBoxLayout(g_part); l_part_main.setContentsMargins(5, 5, 5, 5); l_part_main.setSpacing(2)
        row_p1 = QtWidgets.QHBoxLayout(); row_p2 = QtWidgets.QHBoxLayout()
        
        self.btn_mgr = QtWidgets.QPushButton("🏢 Part Manager")
        self.btn_mgr.setStyleSheet("background-color: #666; color: white; font-weight: bold; min-width: 100px;")
        self.btn_mgr.clicked.connect(lambda: self.visibility_tool.show())
        
        self.chk_warp = QtWidgets.QCheckBox("Use Deform")
        self.chk_warp.setChecked(True); self.chk_warp.toggled.connect(lambda: self.update_frame(self.current_frame))
        
        row_p1.addWidget(self.btn_mgr); row_p1.addWidget(self.chk_warp); row_p1.addStretch(1)
        
        self.sp_sc = QtWidgets.QDoubleSpinBox(); self.sp_sc.setRange(-1000, 1000); self.sp_sc.setValue(1.0); self.sp_sc.setSingleStep(0.1)
        self.sp_sc.valueChanged.connect(lambda: self.update_frame(self.current_frame))
        row_p2.addWidget(QtWidgets.QLabel("  Scale:")); row_p2.addWidget(self.sp_sc); row_p2.addStretch(1)
        
        l_part_main.addLayout(row_p1); l_part_main.addLayout(row_p2)
        layout.addWidget(g_part)

        # [Group 2] Fields (Dual Combo System) - Ported from WHT Inspector for Professional Layout
        g_field = QtWidgets.QGroupBox("Fields")
        l_field = QtWidgets.QGridLayout(g_field)
        l_field.setSpacing(5)
        
        self.cmb_cat = QtWidgets.QComboBox(); self.cmb_cat.setFixedWidth(150)
        self.cmb_comp = QtWidgets.QComboBox(); self.cmb_comp.setFixedWidth(130)
        self.cmb_cat.currentTextChanged.connect(self._on_category_changed)
        self.cmb_comp.currentTextChanged.connect(self._on_component_changed)
        
        # [WHTOOLS] Unified Range Control Group
        self.rng_3d = WHTRangeControlGroup(show_robust=True)
        self.rng_3d.modeChanged.connect(lambda: self.update_frame(self.current_frame))
        self.rng_3d.rangeChanged.connect(lambda: self.update_frame(self.current_frame))
        self.rng_3d.fitRequested.connect(self._on_fit_range) # Use signal instead of direct click
        self.rng_3d.btn_adj.clicked.connect(self._show_range_dialog)

        # Row 1: Field Selection
        l_field.addWidget(QtWidgets.QLabel("Category:"), 0, 0)
        l_field.addWidget(self.cmb_cat, 0, 1)
        l_field.addWidget(QtWidgets.QLabel(" Comp:"), 0, 2)
        l_field.addWidget(self.cmb_comp, 0, 3)
        
        # [WHT] Repositioned Fit button
        btn_fit = QtWidgets.QPushButton("🎯 Fit")
        btn_fit.setFixedWidth(50)
        btn_fit.setToolTip("Fit Scalar Range to Data")
        btn_fit.clicked.connect(self._on_fit_range)
        l_field.addWidget(btn_fit, 0, 4)
        
        l_field.setColumnStretch(1, 0); l_field.setColumnStretch(3, 0); l_field.setColumnStretch(4, 0); l_field.setColumnStretch(5, 1) 
        
        # Row 2: Range & Robustness
        l_field.addWidget(QtWidgets.QLabel("Range:"), 1, 0)
        l_field.addWidget(self.rng_3d, 1, 1, 1, 5) 
        
        layout.addWidget(g_field)

        # [Group 3] Display
        g_disp = QtWidgets.QGroupBox("Display Style")
        l_disp_main = QtWidgets.QVBoxLayout(g_disp); l_disp_main.setContentsMargins(5, 5, 5, 5); l_disp_main.setSpacing(2)
        row_disp1 = QtWidgets.QHBoxLayout(); row_disp2 = QtWidgets.QHBoxLayout()
        
        self.cmb_cb_type = QtWidgets.QComboBox(); self.cmb_cb_type.addItems(["Continuous", "Discrete"])
        self.cmb_cb_lv = QtWidgets.QComboBox(); self.cmb_cb_lv.addItems([str(x) for x in [8, 10, 12, 16, 20, 24, 32, 64]]); self.cmb_cb_lv.setCurrentText("12")
        self.sp_cb_dec = QtWidgets.QSpinBox(); self.sp_cb_dec.setRange(0, 5); self.sp_cb_dec.setValue(3)
        for w in [self.cmb_cb_type, self.cmb_cb_lv, self.sp_cb_dec]:
            if isinstance(w, QtWidgets.QComboBox): w.currentTextChanged.connect(lambda: self.update_frame(self.current_frame))
            else: w.valueChanged.connect(lambda: self.update_frame(self.current_frame))
            
        row_disp1.addWidget(self.cmb_cb_type); row_disp1.addStretch(1)
        row_disp2.addWidget(QtWidgets.QLabel("Lv:")); row_disp2.addWidget(self.cmb_cb_lv)
        row_disp2.addWidget(QtWidgets.QLabel("Dec:")); row_disp2.addWidget(self.sp_cb_dec)
        
        l_disp_main.addLayout(row_disp1); l_disp_main.addLayout(row_disp2)
        layout.addWidget(g_disp)

        # [Group 4] Environment
        g_env = QtWidgets.QGroupBox("Environment")
        l_env_main = QtWidgets.QVBoxLayout(g_env); l_env_main.setContentsMargins(5, 5, 5, 5); l_env_main.setSpacing(2)
        row_env1 = QtWidgets.QHBoxLayout(); row_env2 = QtWidgets.QHBoxLayout()
        
        self.cmb_cmap = QtWidgets.QComboBox(); self.cmb_cmap.addItems(["jet", "viridis", "inferno", "plasma", "magma", "coolwarm", "bone", "gray"])
        self.ch_cmap_r = QtWidgets.QCheckBox("Rev"); self.ch_cmap_r.toggled.connect(self._on_cmap_changed)
        self.cmb_cmap.currentTextChanged.connect(self._on_cmap_changed)
        
        row_env1.addWidget(self.cmb_cmap); row_env1.addWidget(self.ch_cmap_r); row_env1.addStretch(1)
        
        # [WHT] BG 버튼을 QToolButton으로 변경하여 메뉴(Black/White/Grad) 지원
        self.btn_bg = QtWidgets.QToolButton()
        self.btn_bg.setText("🌓 BG")
        self.btn_bg.setFixedWidth(60)
        self.btn_bg.setPopupMode(QtWidgets.QToolButton.MenuButtonPopup)
        self.btn_bg.clicked.connect(self._toggle_bg)
        
        bg_menu = QtWidgets.QMenu(self)
        bg_list = ["Black", "White", "Dark Grey", "Light Grey", "Grey Grad.", "Light Grey Grad.", "Light Sky Grad."]
        for bg_name in bg_list:
            action = bg_menu.addAction(bg_name)
            action.triggered.connect(partial(self._on_bg_changed, bg_name))
        self.btn_bg.setMenu(bg_menu)
        
        self.ch_per = QtWidgets.QCheckBox("Persp")
        self.ch_per.setChecked(True)
        self.ch_per.toggled.connect(self._on_persp_toggled)
        
        row_env2.addWidget(self.btn_bg); row_env2.addWidget(self.ch_per); row_env2.addStretch(1)
        
        l_env_main.addLayout(row_env1); l_env_main.addLayout(row_env2)
        layout.addWidget(g_env)

        # [Group 5] View Orientation Control
        g_view = QtWidgets.QGroupBox("Camera View")
        l_view_main = QtWidgets.QVBoxLayout(g_view); l_view_main.setContentsMargins(5, 5, 5, 5); l_view_main.setSpacing(2)
        row_v1 = QtWidgets.QHBoxLayout()
        
        btn_px = QtWidgets.QPushButton("+X"); btn_px.setToolTip("View from +X Axis"); btn_px.clicked.connect(lambda: self.v_int.view_zy())
        btn_py = QtWidgets.QPushButton("+Y"); btn_py.setToolTip("View from +Y Axis"); btn_py.clicked.connect(lambda: self.v_int.view_xz())
        btn_pz = QtWidgets.QPushButton("+Z"); btn_pz.setToolTip("View from +Z Axis"); btn_pz.clicked.connect(lambda: self.v_int.view_xy())
        btn_iso = QtWidgets.QPushButton("Iso"); btn_iso.setToolTip("Isometric View"); btn_iso.clicked.connect(lambda: self.v_int.view_isometric())
        
        for b in [btn_px, btn_py, btn_pz, btn_iso]:
            b.setFixedWidth(38)
            row_v1.addWidget(b)
            
        l_view_main.addLayout(row_v1)
        layout.addWidget(g_view)

        layout.addStretch(1)

        # Hidden Spinboxes for range state management
        self.sp_min = QtWidgets.QDoubleSpinBox(); self.sp_max = QtWidgets.QDoubleSpinBox()
        self.sp_min.setRange(-1e15, 1e15); self.sp_max.setRange(-1e15, 1e15)
        self.sp_min.setValue(0.0); self.sp_max.setValue(1.0)

    def _init_2d_controls(self, p):
        """[WHTOOLS] 2D Graph Control Panel - Reorganized 2-row layout"""
        main_layout = QtWidgets.QVBoxLayout(p)
        main_layout.setContentsMargins(10, 2, 10, 2); main_layout.setSpacing(4)
        
        row1 = QtWidgets.QHBoxLayout(); row2 = QtWidgets.QHBoxLayout()
        main_layout.addLayout(row1); main_layout.addLayout(row2)
        
        # --- Row 1: Plot Management & Grid Settings ---
        f_lay = QtWidgets.QFrame(); l_lay = QtWidgets.QHBoxLayout(f_lay); l_lay.setContentsMargins(0, 0, 0, 0)
        l_lay.addWidget(QtWidgets.QLabel("Grid:")); self.cmb_lay = QtWidgets.QComboBox(); self.cmb_lay.addItems(["2x1", "1x1", "1x2", "2x2", "3x2"])
        self.cmb_lay.currentTextChanged.connect(self._on_grid_layout_changed); self.cmb_lay.setCurrentText("2x1")
        
        l_lay.addWidget(QtWidgets.QLabel("Slot:")); self.cmb_slot = QtWidgets.QComboBox(); self.cmb_slot.setFixedWidth(70)
        self.cmb_slot.currentIndexChanged.connect(self._on_slot_changed)
        
        bt_add = QtWidgets.QPushButton("➕ Add Plot"); bt_add.clicked.connect(self._show_add_plot_dialog)
        bt_clr = QtWidgets.QPushButton("🗑️ Clear Plots"); bt_clr.clicked.connect(self._on_clear_2d_plots)
        
        l_lay.addWidget(self.cmb_lay); l_lay.addWidget(self.cmb_slot); l_lay.addWidget(bt_add); l_lay.addWidget(bt_clr)
        row1.addWidget(f_lay); row1.addWidget(self._create_v_line())
        
        f_app = QtWidgets.QFrame(); l_app = QtWidgets.QHBoxLayout(f_app); l_app.setContentsMargins(0, 0, 0, 0)
        l_app.addWidget(QtWidgets.QLabel("Theme:")); self.cmb_theme = QtWidgets.QComboBox(); self.cmb_theme.addItems(['default', 'ggplot', 'bmh', 'dark_background'])
        self.cmb_theme.currentTextChanged.connect(lambda: self.update_frame(self.current_frame))
        l_app.addWidget(QtWidgets.QLabel("Font:")); self.cmb_font_2d = QtWidgets.QComboBox(); self.cmb_font_2d.addItems([str(i) for i in range(6, 31)]); self.cmb_font_2d.setCurrentText("9")
        self.cmb_font_2d.currentTextChanged.connect(lambda v: self.sp_vtk_font.setValue(int(v)))
        
        l_app.addWidget(self.cmb_theme); l_app.addWidget(self.cmb_font_2d)
        row1.addWidget(f_app); row1.addStretch(1)

        # --- Row 2: Data Range & Options ---
        f_rng = QtWidgets.QFrame(); l_rng = QtWidgets.QHBoxLayout(f_rng); l_rng.setContentsMargins(0, 0, 0, 0)
        # [WHTOOLS] Unified Range Control Group for 2D Plots
        self.rng_2d = WHTRangeControlGroup(show_robust=True)
        self.rng_2d.modeChanged.connect(lambda: self.update_frame(self.current_frame))
        self.rng_2d.rangeChanged.connect(lambda: self.update_frame(self.current_frame))
        self.rng_2d.fitRequested.connect(self._on_fit_range_2d) # Use signal
        self.rng_2d.btn_adj.clicked.connect(self._show_range_dialog_2d) # Enable common dialog for 2D

        l_rng.addWidget(self.rng_2d)
        row2.addWidget(f_rng); row2.addWidget(self._create_v_line())
        
        f_opt = QtWidgets.QFrame(); l_opt = QtWidgets.QHBoxLayout(f_opt); l_opt.setContentsMargins(0, 0, 0, 0)
        self.checks = {}
        # Use Full Names: Sync Time, Interpolation
        for t_full, t_short, s in [("Sync Time", "Sync", True), ("Interpolation", "Interp", True)]:
            c = QtWidgets.QCheckBox(t_full); c.setChecked(s); c.toggled.connect(lambda: self.update_frame(self.current_frame))
            l_opt.addWidget(c); self.checks[t_short] = c
            
        bt_pop = QtWidgets.QPushButton("📺 Pop-out View"); bt_pop.clicked.connect(self._pop_out_2d)
        l_opt.addWidget(bt_pop)
        row2.addWidget(f_opt); row2.addStretch(1)

    def _init_animation_dock(self):
        """
        [WHTOOLS] 하단 애니메이션 타임라인 및 제어 도크를 초기화합니다.
        """
        self.ad = QtWidgets.QDockWidget("Animation Control")
        self.ad.setAllowedAreas(QtCore.Qt.BottomDockWidgetArea)
        
        cn = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(cn)
        layout.setContentsMargins(10, 5, 10, 5)
        
        # 1. Navigation & Play Control
        controls = [
            ("⏮️", 0, "To Start"),
            ("⏪", -1, "Step Backward"),
            ("PLAY_PAUSE", -2, "Play/Pause"),
            ("⏩", 1, "Step Forward"),
            ("⏭️", 9999, "To End")
        ]
        
        for t, s, tooltip in controls:
            if t == "PLAY_PAUSE":
                self.bp = QtWidgets.QPushButton("▶️")
                self.bp.setFixedSize(55, 32)
                self.bp.setToolTip(tooltip)
                self.bp.clicked.connect(lambda: self._ctrl_slot(-2))
                layout.addWidget(self.bp)
            else:
                b = QtWidgets.QPushButton(t)
                b.setFixedSize(50, 32)
                b.setToolTip(tooltip)
                b.clicked.connect(partial(self._ctrl_slot, s))
                layout.addWidget(b)
        
        # 3. Timeline Slider
        n_frames = len(self.mgr.times) if (self.mgr is not None and self.mgr.times is not None) else 1
        self.sld = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sld.setRange(0, n_frames - 1)
        self.sld.valueChanged.connect(self.update_frame)
        layout.addWidget(self.sld, stretch=1)
        
        # 4. Status Information
        self.lf = QtWidgets.QLabel(f"Frame: 0 / {n_frames-1}")
        self.lf.setFixedWidth(150)
        layout.addWidget(self.lf)
        
        # 5. Playback Speed Control
        layout.addWidget(QtWidgets.QLabel(" Speed:"))
        self.cs = QtWidgets.QComboBox()
        self.cs.addItems(["Max", "High", "Mid", "Low"])
        self.cs.setCurrentText("High")
        self.cs.currentTextChanged.connect(self._on_speed_changed)
        layout.addWidget(self.cs)
        
        # 6. Step Frame Control
        layout.addWidget(QtWidgets.QLabel(" Step:"))
        self.sp_step_ui = QtWidgets.QSpinBox()
        self.sp_step_ui.setRange(1, 100)
        self.sp_step_ui.setValue(self.anim_step)
        self.sp_step_ui.valueChanged.connect(self._update_step)
        layout.addWidget(self.sp_step_ui)
        
        self.ad.setWidget(cn)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.ad)

    def _on_speed_changed(self, text):
        mapping = {"Max": 0, "High": 15, "Mid": 30, "Low": 100}
        self.timer.setInterval(mapping.get(text, 30))

    # --------------------------------------------------------------------------
    # --- 3D Visualization Engine ---
    # --------------------------------------------------------------------------

    def _init_3d_view(self):
        """3D 장면 초기화 및 어셈블리 파트별 메쉬 생성"""
        self._on_bg_changed("Grey Grad.")
        self.v_int.add_axes()
        
        gp = pv.Plane(
            center=self.floor_origin, 
            direction=self.floor_normal, 
            i_size=self.floor_w, 
            j_size=self.floor_h
        )
        self.ground = self.v_int.add_mesh(gp, color="#111111", opacity=0.3, show_edges=False)
        
        edges = gp.extract_feature_edges(boundary_edges=True, non_manifold_edges=False, feature_edges=False, manifold_edges=False)
        self.v_int.add_mesh(edges, color="darkgray", line_width=2)
        
        self.lut = pv.LookupTable(cmap="jet_r")
        self.lut.below_range_color = 'lightgrey'
        self.lut.above_range_color = 'magenta'
        
        analyzers = self.mgr.analyzers if self.mgr else []
        import matplotlib
        tab10 = matplotlib.colormaps.get_cmap('tab10')
        
        for i, ana in enumerate(analyzers):
            if ana.m_raw is None or ana.sol is None:
                self.part_actors[i] = {'mesh': None, 'visible': False}
                continue
                
            # [WHT] Use cyclic tab10 colormap for consistent part coloring
            c_rgb = tab10(i % 10)[:3]
            c_hex = matplotlib.colors.to_hex(c_rgb)
            
            poly = pv.Plane(
                i_size=ana.W, 
                j_size=ana.H, 
                i_resolution=ana.sol.res - 1, 
                j_resolution=ana.sol.res - 1
            )
            ma = self.v_int.add_mesh(
                poly, 
                scalars=None, 
                color=c_hex,
                cmap=self.lut, 
                show_edges=True, 
                edge_color="darkgray", 
                show_scalar_bar=False
            )
            
            mp = pv.PolyData(np.array(ana.m_raw[0]))
            n_m = ana.m_raw.shape[1]
            mp.point_data["names"] = [f"{ana.name}_M{j:02d}" for j in range(n_m)]
            mka = self.v_int.add_mesh(
                mp, 
                render_points_as_spheres=True, 
                point_size=10, 
                color=c_hex # [WHT] Match mesh color
            )
            
            la = self.v_int.add_point_labels(
                mp, "names", 
                font_size=max(6, self.v_font_size - 2), 
                text_color='black', 
                always_visible=False, 
                point_size=0, 
                shadow=False,
                pickable=False
            )
            
            if not hasattr(ana.sol, 'X_mesh') or ana.sol.X_mesh is None:
                ma.SetVisibility(False)
                self.part_actors[i] = {'mesh': ma, 'body_color': c_hex, 'visible': False}
                continue
                
            mka.SetVisibility(False)
            la.SetVisibility(False)
            
            p_base = np.column_stack([
                ana.sol.X_mesh.ravel(), 
                ana.sol.Y_mesh.ravel(), 
                np.zeros(ana.sol.res**2)
            ])
            
            self.part_actors[i] = {
                'mesh': ma, 
                'body_color': c_hex,
                'poly': poly, 
                'm_poly': mp, 
                'markers': mka, 
                'labels': la, 
                'visible': True, 
                'visible_markers': False, 
                'p_base': p_base
            }
            
        if self.part_actors:
            f_i = min(self.part_actors.keys())
            fsize = int(self.cmb_font_2d.currentText()) if hasattr(self, 'cmb_font_2d') else 9
            self.sb = self.v_int.add_scalar_bar(
                "Field Analysis", 
                position_x=0.88, position_y=0.1, 
                width=0.08, height=0.8,
                vertical=True,
                mapper=self.part_actors[f_i]['mesh'].mapper,
                title_font_size=fsize,
                label_font_size=max(6, fsize - 1),
                n_labels=11,
                fmt="%.3e"
            )
            self.sb.unconstrained_font_size = True
        else:
            self.sb = self.v_int.add_scalar_bar("No Data", vertical=True, position_x=0.88)
            self.sb.SetVisibility(False)
            
        self.ov = self.v_int.add_text("-", position='upper_right', font_size=9, color='black')
        self.gui_txt = self.v_int.add_text(
            "[Space]: Play/Pause | [R]: Reset | [W]: Wireframe", 
            position='upper_right', 
            font_size=9, 
            color='black'
        )
        
        self.v_int.view_isometric()
        self.v_int.camera.ParallelProjectionOff()
        
        self.timer = QtCore.QTimer()
        self.timer.setInterval(30)
        self.timer.timeout.connect(lambda: self._ctrl_slot(self.anim_step))
        
        self.v_int.add_key_event('space', self._on_toggle_play)
        self.v_int.add_key_event('r', self._on_reset_animation)

    # [WHT] Redundant _on_cmap_changed removed. Using the one at line 2298.


    def update_frame(self, f_i: int):
        """[WHTOOLS] Real-time frame update with professional field mapping."""
        if self.mgr is None or self.mgr.times is None or len(self.mgr.times) == 0: return
        self.current_frame = f_i
        
        # UI Sync
        if hasattr(self, 'sld'):
            self.sld.blockSignals(True); self.sld.setValue(f_i); self.sld.blockSignals(False)
        self.lf.setText(f"Frame: {f_i} / {len(self.mgr.times)-1}")
        
        # State
        # [WHTOOLS] 필드 키 추출 (듀얼 콤보 시스템 연동)
        cat = self.cmb_cat.currentText()
        fk = self.cmb_comp.currentData()
        if fk is None: fk = self.cmb_comp.currentText()
        
        # [WHT] Basic 모드 체크 로직 강화
        is_basic = (cat == "Basic" or str(fk) in ["Body Color", "Face Color"])
        
        vm = self.cmb_v.currentText() if hasattr(self, 'cmb_v') else "Global"
        sc = self.sp_sc.value(); mode = self.cmb_lay.currentText() if hasattr(self, 'cmb_lay') else "Surface w/ Edge"
        
        active_values = []
        n_frames_tot = len(self.mgr.times)
        
        for i, ana in enumerate(self.mgr.analyzers):
            if i not in self.part_actors: continue
            inf = self.part_actors[i]
            if not inf['mesh']: continue
            
            # Visibility
            mv = inf['visible']
            mkv = inf['visible_markers']
            
            inf['mesh'].SetVisibility(mv)
            
            # [WHT] View Mode Apply (Bug Fix)
            if inf['mesh'] is not None and mv:
                v_mode = inf.get('view_mode', "Surface w/ Edge")
                prop = inf['mesh'].GetProperty()
                if v_mode == "Surface w/ Edge":
                    prop.SetRepresentationToSurface()
                    prop.SetEdgeVisibility(True)
                    prop.SetOpacity(1.0)
                elif v_mode == "Surface only":
                    prop.SetRepresentationToSurface()
                    prop.SetEdgeVisibility(False)
                    prop.SetOpacity(1.0)
                elif v_mode == "Wireframe":
                    prop.SetRepresentationToWireframe()
                    prop.SetEdgeVisibility(True)
                    prop.SetOpacity(1.0)
                elif v_mode == "Outline":
                    prop.SetRepresentationToSurface()
                    prop.SetEdgeVisibility(True)
                    prop.SetOpacity(0.3) # Outline 대용으로 투명 처리
                else:
                    prop.SetRepresentationToSurface()
                    prop.SetOpacity(1.0)
            
            if inf['markers'] is not None:
                inf['markers'].SetVisibility(mkv)
                if 'body_color' in inf:
                    inf['markers'].prop.color = inf['body_color']
            if inf['labels'] is not None:
                inf['labels'].SetVisibility(mkv)
            if not mv and not mkv: continue
            
            # Deformation
            disp = ana.results.get('Displacement [mm]', np.zeros((n_frames_tot, ana.sol.res, ana.sol.res)))[f_i]
            pts = inf['p_base'].copy()
            if self.chk_warp.isChecked(): pts[:, 2] = disp.ravel() * sc
            
            # Basis Transform
            if vm == "Global":
                R = ana.results.get('R_matrix')[f_i]; cur_c = ana.results.get('cur_centroid')[f_i]
                ref_c = ana.results.get('ref_centroid')[f_i]
                l_basis = np.array(ana.kin.local_basis_axes if ana.kin else ana.results.get('local_basis_axes', np.eye(3)))
                l_c0 = np.array(ana.kin.local_centroid_0 if ana.kin else ana.results.get('local_centroid_0', np.zeros(3)))
                inf['poly'].points = (pts @ l_basis.T + l_c0 - ref_c) @ R + cur_c
                inf['m_poly'].points = np.array(ana.m_raw[f_i])
            else:
                inf['poly'].points = pts
                inf['m_poly'].points = np.array(ana.results.get('local_markers')[f_i])
            
            # Color Mapping
            if is_basic:
                inf['mesh'].mapper.scalar_visibility = False
                if 'body_color' in inf:
                    inf['mesh'].prop.color = inf['body_color']
            else:
                inf['mesh'].mapper.scalar_visibility = True
                if fk in ana.results:
                    val = ana.results[fk][f_i]
                    inf['poly'].point_data["S"] = val.ravel()
                    inf['poly'].set_active_scalars("S")
                    if mv: active_values.append(val)
                else:
                    inf['mesh'].mapper.scalar_visibility = False
            
            inf['poly'].Modified(); inf['m_poly'].Modified()
            if not is_basic and mv:
                try: inf['mesh'].mapper.Modified()
                except: pass
            
        # Scalar Bar Update
        if active_values and not is_basic:
            v_min, v_max = float(min(v.min() for v in active_values)), float(max(v.max() for v in active_values))
            
            mode = self.rng_3d.cmb_mode.currentText()
            if mode == "Dynamic":
                self.rng_3d.set_range(v_min, v_max)
            elif mode == "Robust":
                pct = self.rng_3d.get_robust_pct()
                r_min, r_max = self._calculate_robust_range(fk, p_low=(100-pct)/2, p_high=100-(100-pct)/2)
                self.rng_3d.set_range(r_min, r_max)
            
            cur_min, cur_max = self.rng_3d.get_range()
            self._update_scalar_bar(fk, cur_min, cur_max)
        else:
            # [WHT] 데이터가 없는 경우 스칼라 바 숨김
            if hasattr(self, 'sb') and self.sb is not None:
                try: self.sb.SetVisibility(False)
                except: pass
            self.v_int.add_text("", position='upper_left', name='st_ov')
            
        self._update_2d_plots(f_i)
        self.v_int.render()

    # --------------------------------------------------------------------------
    # --- 2D Plotting Engine (Matplotlib) ---
    # --------------------------------------------------------------------------

    def _on_grid_layout_changed(self, text):
        """그리드 레이아웃 변경 시 슬롯 선택 위젯 항목을 동기화합니다."""
        self._init_2d_plots()
        # [WHTOOLS] 레이아웃 변경 시 1회 정렬
        self.fig.tight_layout(pad=3.0)
        self.can.draw_idle()
        
        layout_map = {"1x1": (1,1), "1x2": (1,2), "2x2": (2,2), "3x2": (3,2), "2x1": (2,1)}
        rows, cols = layout_map.get(text, (2, 1))
        n_slots = rows * cols
        
        if hasattr(self, 'cmb_slot'):
            self.cmb_slot.blockSignals(True); self.cmb_slot.clear()
            self.cmb_slot.addItems([f"Slot {i+1}" for i in range(n_slots)])
            if self.active_slot < n_slots: self.cmb_slot.setCurrentIndex(self.active_slot)
            else: self.active_slot = 0; self.cmb_slot.setCurrentIndex(0)
            self.cmb_slot.blockSignals(False)
            
    def _on_slot_changed(self, index):
        if index < 0: return
        self.active_slot = index
        self.update_frame(self.current_frame)
        
    def _update_selection_ui(self):
        if hasattr(self, 'cmb_slot'):
            self.cmb_slot.blockSignals(True)
            self.cmb_slot.setCurrentIndex(self.active_slot)
            self.cmb_slot.blockSignals(False)
        self.update_frame(self.current_frame)

    def _init_2d_plots(self):
        """2D 차트 영역(Grid Layout) 초기화 및 슬롯 생성"""
        for i in reversed(range(self._cl.count())):
            item = self._cl.itemAt(i)
            if item.widget():
                w = item.widget()
                w.setParent(None)
                w.deleteLater()
                
        plt.rcParams['font.size'] = 9
        self._is_first_2d_update = True
        self.fig = Figure(figsize=(8, 8))
        self.can = FigureCanvas(self.fig)
        self._cl.addWidget(NavigationToolbar(self.can, self))
        self._cl.addWidget(self.can)
        
        self.can.mpl_connect('button_press_event', self._on_axis_clicked)
        
        layout_map = {"2x1": (2, 1), "1x1": (1, 1), "1x2": (1, 2), "2x2": (2, 2), "3x2": (3, 2)}
        rows, cols = layout_map.get(self.cmb_lay.currentText(), (2, 2))
        
        self.axes = []; self.ims = [None] * 6; self.vls = [None] * 6; self.cbs = [None] * 6
        self.fig.clear()
        self.fig.subplots_adjust(hspace=0.4, wspace=0.3)
        
        for i in range(rows * cols):
            ax = self.fig.add_subplot(rows, cols, i + 1)
            self.axes.append(ax)
            fsize = self.v_font_size
            ax.text(0.5, 0.5, f"Slot {i + 1}\n(Click to Add)", ha='center', va='center', transform=ax.transAxes, fontsize=fsize, color='gray', alpha=0.5)
            ax.set_xticks([]); ax.set_yticks([])
            
        theme = getattr(self, 'cmb_theme', None)
        if theme: plt.style.use(theme.currentText())
        else: plt.style.use('default')
        
        plt.rcParams['font.size'] = self.v_font_size
        self._update_selection_ui()
        self.fig.tight_layout()
        self.can.draw_idle()

    def _update_2d_plots(self, f_i):
        """현재 프레임에 맞춰 2D 슬롯 데이터 갱신"""
        if self.mgr is None or self.mgr.times is None or len(self.mgr.times) == 0 or not self.axes: return
            
        is_sync = self.checks.get('Sync').isChecked() if 'Sync' in self.checks else True
        if self.is_playing and not is_sync: return
            
        current_time = self.mgr.times[f_i]
        use_interp = self.checks.get('Interp').isChecked() if 'Interp' in self.checks else True
        fsize = self.v_font_size
        
        for i, ax in enumerate(self.axes):
            fsize = self.v_font_size
            cfg = self.plot_slots[i]
            ax.title.set_fontsize(fsize); ax.xaxis.label.set_fontsize(fsize); ax.yaxis.label.set_fontsize(fsize)
            for label in (ax.get_xticklabels() + ax.get_yticklabels()): label.set_fontsize(max(6, fsize - 1))

            if not cfg: continue
                
            p_idx_main = cfg.part_indices[0] if (cfg.part_indices is not None and len(cfg.part_indices) > 0) else 0
            if p_idx_main < 0: p_idx_main = 0
            ana = self.mgr.analyzers[p_idx_main]
            key = cfg.data_key
            
            # [WHT] 데이터 키 유효성 체크 및 폴백
            if key not in ana.results:
                if self.field_keys: key = self.field_keys[0] # 첫 번째 필드로 대체
                else: continue
            
            if cfg.plot_type == "contour":
                if ana.sol is None: continue
                data_2d = ana.results.get(key, np.zeros((len(self.mgr.times), ana.sol.res, ana.sol.res)))[f_i]
                dy_2d = self.rng_2d.cmb_mode.currentText() == "Dynamic"
                vmin, vmax = float(data_2d.min()), float(data_2d.max())
                
                if dy_2d:
                    self.rng_2d.set_range(vmin, vmax)
                    clim = [vmin, vmax]
                else: 
                    clim = list(self.rng_2d.get_range())
                
                if clim[0] >= clim[1]: clim[1] = clim[0] + 1e-6
                
                if self.ims[i] is None:
                    ax.clear()
                    if self.cbs[i] is not None:
                        try:
                            # [WHT] More robust colorbar removal to prevent layout corruption
                            if hasattr(self.cbs[i], 'ax'):
                                self.fig.delaxes(self.cbs[i].ax)
                            self.cbs[i] = None
                        except: pass
                    cmap_3d = self.cmb_cmap.currentText()
                    self.ims[i] = ax.imshow(data_2d, cmap=cmap_3d, origin='lower', extent=[0, ana.W, 0, ana.H])
                    self.cbs[i] = self.fig.colorbar(self.ims[i], ax=ax, format="%.3e")
                    self.cbs[i].ax.tick_params(labelsize=fsize - 1)
                    ax.set_title(f"[{ana.name}] {key}", fontsize=fsize)
                
                self.ims[i].set_data(data_2d)
                cmap_2d = self.cmb_cmap.currentText()
                if self.ch_cmap_r.isChecked(): cmap_2d += "_r"
                self.ims[i].set_cmap(cmap_2d); self.ims[i].set_clim(clim[0], clim[1])
                self.ims[i].set_interpolation('bilinear' if use_interp else 'nearest')
                
            else:
                if self.vls[i] is None:
                    ax.clear()
                    if self.cbs[i] is not None:
                        try: self.cbs[i].remove(); self.cbs[i] = None
                        except: pass
                    ax.grid(True, alpha=0.3)
                    
                    target_data_list = []
                    if -1 in cfg.part_indices:
                        for a in self.mgr.analyzers: target_data_list.append((a, a.name))
                    elif -2 in cfg.part_indices:
                        groups = {}
                        for a in self.mgr.analyzers:
                            prefix = a.name.split('_')[0] if "_" in a.name else a.name
                            if prefix not in groups: groups[prefix] = []
                            groups[prefix].append(a)
                        for prefix, members in groups.items(): target_data_list.append((members, prefix))
                    else:
                        for p_idx in cfg.part_indices:
                            if 0 <= p_idx < len(self.mgr.analyzers):
                                a = self.mgr.analyzers[p_idx]; target_data_list.append((a, a.name))
                                
                    for obj, name in target_data_list:
                        key = cfg.data_key
                        if isinstance(obj, list):
                            vals = []
                            for a in obj:
                                if key.startswith("Max-"):
                                    real_key = key.replace("Max-", "")
                                    v = np.max(a.results[real_key], axis=(1, 2)) if real_key in a.results else np.zeros(len(self.mgr.times))
                                else: v = a.results.get(key, np.zeros(len(self.mgr.times)))
                                vals.append(v)
                            y_data = np.max(np.array(vals), axis=0)
                        else:
                            if key.startswith("Max-"):
                                real_key = key.replace("Max-", "")
                                y_data = np.max(obj.results[real_key], axis=(1, 2)) if real_key in obj.results else np.zeros(len(self.mgr.times))
                            else: y_data = obj.results.get(key, np.zeros(len(self.mgr.times)))
                        
                        if y_data.ndim == 1: ax.plot(self.mgr.times, y_data, label=name)
                        else:
                            for m in range(min(y_data.shape[1], 8)): ax.plot(self.mgr.times, y_data[:, m], alpha=0.5, label=f"{name}-M{m}")
                                
                    if len(target_data_list) > 1: ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=fsize-2, borderaxespad=0.)
                    self.vls[i] = ax.axvline(current_time, color='red', ls='--')
                    ax.set_ylabel(key, fontsize=fsize); ax.set_xlabel("Time [s]", fontsize=fsize)
                    t_str = f"[Multi] {key}" if len(target_data_list) > 1 or cfg.part_indices[0] < 0 else f"[{target_data_list[0][1]}] {key}"
                    ax.set_title(t_str, fontsize=fsize)
                    
                self.vls[i].set_xdata([current_time])
        
        # [WHTOOLS] 매 프레임 업데이트 시에는 성능을 위해 tight_layout 호출을 지양합니다.
        self.can.draw_idle()

    def _on_clear_2d_plots(self):
        self.plot_slots = [None] * 6; self._init_2d_plots(); self.can.draw_idle()

    # --------------------------------------------------------------------------
    # --- Interaction & Event Handlers ---
    # --------------------------------------------------------------------------

    def _pop_out_2d(self):
        pw = QtWidgets.QMainWindow(self); pw.setWindowTitle("Analysis View"); pw.resize(1100, 850)
        cw = QtWidgets.QWidget(); pw.setCentralWidget(cw); layout = QtWidgets.QVBoxLayout(cw)
        fig = Figure(figsize=(10, 10)); canvas = FigureCanvas(fig); layout.addWidget(canvas)
        layout_map = {"1x1": (1,1), "1x2": (1,2), "2x2": (2,2), "3x2": (3,2)}
        rows, cols = layout_map.get(self.cmb_lay.currentText(), (2, 2))
        for i in range(rows * cols):
            ax = fig.add_subplot(rows, cols, i + 1)
            color, width = ("#1A73E8", 2.0) if i == self.active_slot else ("#DADCE0", 0.5)
            for spine in ax.spines.values(): spine.set_edgecolor(color); spine.set_linewidth(width)
            cfg = self.plot_slots[i]
            if cfg:
                p_idx = cfg.part_indices[0] if cfg.part_indices else 0
                ana = self.mgr.analyzers[p_idx]; key = cfg.data_key
                if ana.results and ana.sol and cfg.plot_type == "contour":
                    res_val = ana.results.get(key, np.zeros((len(self.mgr.times), ana.sol.res, ana.sol.res)))[self.current_frame]
                    cmap_base = self.cmb_cmap.currentText(); cmap_name = cmap_base + "_r" if self.ch_cmap_r.isChecked() else cmap_base
                    im = ax.imshow(res_val, cmap=cmap_name, origin='lower'); fig.colorbar(im, ax=ax); ax.set_title(f"[{ana.name}] {key}", fontsize=9)
                else:
                    vs = ana.results.get(key, np.zeros(len(self.mgr.times)))
                    if vs.ndim > 1:
                        for m in range(min(vs.shape[1], 10)): ax.plot(self.mgr.times, vs[:, m], alpha=0.5)
                    else: ax.plot(self.mgr.times, vs)
                    ax.set_title(f"[{ana.name}] {key}", fontsize=9); ax.axvline(self.mgr.times[self.current_frame], color='red')
            else: ax.text(0.5, 0.5, "Empty Slot", ha='center', transform=ax.transAxes)
        canvas.draw(); pw.show()

    def _show_about(self): AboutDialog(self.logo_path, self).exec()

    def _on_persp_toggled(self, state):
        if state: self.v_int.disable_parallel_projection()
        else: self.v_int.enable_parallel_projection()
        self.v_int.render()


    def _on_legend_mode_changed(self, mode):
        """기존 시그널 대응 (하위 호환성)"""
        self.update_frame(self.current_frame)

    def _on_fit_range(self):
        if self.mgr is None: return
        field_key = self.cmb_comp.currentData()
        if not field_key or field_key == "Body Color": return
        all_values = []
        for i, a in enumerate(self.mgr.analyzers):
            # [WHTOOLS] Only include data from VISIBLE parts
            if i in self.part_actors and self.part_actors[i]['visible']:
                if field_key in a.results: 
                    all_values.append(a.results[field_key])
        if all_values:
            merged = np.concatenate([v.ravel() for v in all_values])
            v_min, v_max = float(np.nanmin(merged)), float(np.nanmax(merged))
            self.rng_3d.set_range(v_min, v_max)
            self.update_frame(self.current_frame)

    def _on_field_changed(self, field_key):
        """기존 cmb_3d 시그널 대응 (하위 호환성)"""
        if self.mgr is None: return
        self.update_frame(self.current_frame)


    def _on_fit_range_2d(self):
        if self.mgr is None: return
        all_vals = []
        for cfg in self.plot_slots:
            if cfg and cfg.plot_type == "contour":
                for p_idx in cfg.part_indices:
                    # [WHTOOLS] Only include data from VISIBLE parts
                    if p_idx in self.part_actors and self.part_actors[p_idx]['visible']:
                        ana = self.mgr.analyzers[p_idx]
                        if cfg.data_key in ana.results: 
                            all_vals.append(ana.results[cfg.data_key])
        if all_vals:
            merged = np.concatenate([v.ravel() for v in all_vals])
            v_min, v_max = float(np.nanmin(merged)), float(np.nanmax(merged))
            self.rng_2d.set_range(v_min, v_max)
            self.update_frame(self.current_frame)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Space: self._ctrl_slot(-2)
        else: super().keyPressEvent(event)

    def _show_part_menu(self, pos=None):
        if pos is None: pos = self.v_int.mapFromGlobal(QtGui.QCursor.pos())
        menu = QtWidgets.QMenu(self)
        # [WHT] 픽킹된 파트 감지 및 매니저 연동
        picked_actor = self.v_int.picker.GetActor()
        target_part_idx = -1
        if picked_actor:
            for idx, act_inf in self.part_actors.items():
                if act_inf['mesh'] == picked_actor:
                    target_part_idx = idx; break
        
        if target_part_idx != -1:
            part_name = self.mgr.analyzers[target_part_idx].name
            menu.addAction(f"🎯 Focus on: {part_name}", lambda: self._focus_part_in_manager(target_part_idx))
            
            # [WHT] 회전 중심 설정 기능 추가
            pick_pos = self.v_int.picker.GetPickPosition()
            if pick_pos:
                menu.addAction("📍 Set Rotation Center", lambda: self._set_rotation_center(pick_pos))
            
            menu.addSeparator()

        act_floor = menu.addAction("Floor Visibility"); act_floor.setCheckable(True); act_floor.setChecked(self.ground.GetVisibility())
        fs = menu.addMenu("Floor Settings"); fs.addAction("Change Origin", self._set_floor_origin); fs.addAction("Change Normal", self._set_floor_normal); fs.addAction("Change Size", self._set_floor_size)
        menu.addSeparator()
        act_edges = menu.addAction("Show Mesh Edges"); act_edges.setCheckable(True)
        if self.part_actors:
            first_idx = min(self.part_actors.keys())
            if self.part_actors[first_idx]['mesh'] is not None:
                edge_v = self.part_actors[first_idx]['mesh'].GetProperty().GetEdgeVisibility()
                act_edges.setChecked(edge_v)
        else: act_edges.setChecked(True)
        act_perp = menu.addAction("Perspective View"); act_perp.setCheckable(True); act_perp.setChecked(self.ch_per.isChecked()); menu.addSeparator()
        def _set_repr(mode):
            for ai in self.part_actors.values():
                if ai['mesh'] is not None:
                    prop = ai['mesh'].GetProperty()
                    if mode == 'wireframe': prop.SetRepresentationToWireframe()
                    elif mode == 'surface': prop.SetRepresentationToSurface()
                    elif mode == 'points': prop.SetRepresentationToPoints()
            self.v_int.render()
        menu.addAction("Wireframe Mode", lambda: _set_repr('wireframe')); menu.addAction("Surface Mode", lambda: _set_repr('surface'))
        menu.addAction("📐 Fit View", lambda: self.v_int.reset_camera()); menu.addAction("Pick Mode", lambda: self.v_int.enable_point_picking())
        selected_action = menu.exec_(self.v_int.mapToGlobal(pos))
        if selected_action == act_floor: self.ground.SetVisibility(selected_action.isChecked()); self.v_int.render()
        elif selected_action == act_edges:
            for ai in self.part_actors.values():
                if ai['mesh'] is not None: ai['mesh'].GetProperty().SetEdgeVisibility(selected_action.isChecked())
            self.v_int.render()
        elif selected_action == act_perp: self.ch_per.setChecked(selected_action.isChecked())

    def _set_rotation_center(self, pos):
        """[WHTOOLS] 선택한 좌표를 카메라의 회전 중심(Focal Point)으로 설정합니다."""
        if pos is None: return
        self.v_int.set_focus(pos)
        # 시각적 피드백: 회전 중심 지점에 잠시 표시 (선택 사항)
        print(f"[WHT-DEBUG] Rotation center updated to: {pos}")
        self.v_int.render()

    def _focus_part_in_manager(self, idx):
        """3D에서 선택한 파트를 매니저 트리에서 강조하고 창을 띄웁니다."""
        if not hasattr(self, 'visibility_tool') or self.visibility_tool is None:
            self.visibility_tool = PartManagerWindow(self)
        
        self.visibility_tool.show()
        if idx in self.visibility_tool.id_to_item:
            item = self.visibility_tool.id_to_item[idx]
            self.visibility_tool.tree.setCurrentItem(item)
            self.visibility_tool.tree.scrollToItem(item)

    def _set_floor_origin(self):
        v, ok = QtWidgets.QInputDialog.getText(self, "Floor Origin", "Origin (x,y,z):", text=",".join(map(str, self.floor_origin)))
        if ok:
            try: self.floor_origin = [float(x) for x in v.split(",")]; self._update_floor()
            except ValueError: self._show_warning("Input Error", "Invalid input.")

    def _set_floor_normal(self):
        v, ok = QtWidgets.QInputDialog.getText(self, "Floor Normal", "Normal (nx,ny,nz):", text=",".join(map(str, self.floor_normal)))
        if ok:
            try: self.floor_normal = [float(x) for x in v.split(",")]; self._update_floor()
            except ValueError: self._show_warning("Input Error", "Invalid input.")

    def _set_floor_size(self):
        v, ok = QtWidgets.QInputDialog.getText(self, "Floor Size", "Size (W, H):", text=f"{self.floor_w},{self.floor_h}")
        if ok:
            try: self.floor_w, self.floor_h = [float(x) for x in v.split(",")]; self._update_floor()
            except ValueError: self._show_warning("Input Error", "Invalid input.")

    def _update_floor(self):
        self.v_int.remove_actor(self.ground)
        gp = pv.Plane(center=self.floor_origin, direction=self.floor_normal, i_size=self.floor_w, j_size=self.floor_h)
        self.ground = self.v_int.add_mesh(gp, color="blue", opacity=0.1); self.v_int.render()

    def _update_selection_ui(self):
        for i, ax in enumerate(self.axes):
            cl, w = ("#1A73E8", 2.0) if i == self.active_slot else ("#DADCE0", 0.5)
            for s in ax.spines.values(): s.set_color(cl); s.set_linewidth(w)
        self.can.draw_idle()

    def _on_axis_clicked(self, event):
        if event.inaxes is None: return
        try:
            idx = self.axes.index(event.inaxes); self.active_slot = idx; self._update_selection_ui()
        except ValueError: pass

    def _on_category_changed(self, cat=None):
        """카테고리 변경 시 컴포넌트 목록(cmb_comp)을 동적 갱신합니다."""
        if cat is None:
            cat = self.cmb_cat.currentText()
            
        if self._categories is not None and cat in self._categories:
            self.cmb_comp.blockSignals(True)
            self.cmb_comp.clear()
            
            items = self._categories[cat]
            if items is not None and len(items) > 0:
                for comp, fk in sorted(items): 
                    self.cmb_comp.addItem(comp, fk)
                    
            self.cmb_comp.blockSignals(False)
            self._on_component_changed()

    def _show_add_plot_dialog(self):
        if not self.mgr: self._show_warning("No Data", "Please load a result file first."); return
        parts_list = [p.name for p in self.mgr.analyzers]
        layout_map = {"2x1": (2,1), "1x1": (1,1), "1x2": (1,2), "2x2": (2,2), "3x2": (3,2)}
        rows, cols = layout_map.get(self.cmb_lay.currentText(), (2, 2))
        dialog = AddPlotDialog(self.active_slot, parts_list, self.field_keys, self.stat_keys, parent=self, rows=rows, cols=cols, plot_slots=self.plot_slots)
        if dialog.exec():
            final_slot = dialog.slot_idx; config = dialog.get_config()
            self.plot_slots[final_slot] = config; self.active_slot = final_slot
            if final_slot < len(self.ims): self.ims[final_slot] = self.vls[final_slot] = self.cbs[final_slot] = None
            self._update_selection_ui(); 
            # [WHTOOLS] 그래프 추가 시 1회 정렬
            self.fig.tight_layout(pad=3.0)
            self.update_frame(self.current_frame); self.can.draw_idle()

    def _on_toggle_play(self):
        if not hasattr(self, 'timer'): return
        if self.timer.isActive(): self.timer.stop(); self.is_playing = False; self.bp.setText("▶️")
        else: self.timer.start(); self.is_playing = True; self.bp.setText("⏸️")

    def _on_reset_animation(self):
        self.timer.stop()
        if hasattr(self, 'btn_play'): self.btn_play.setText("▶️ Play")
        self.sld.setValue(0); self.update_frame(0)

    def _ctrl_slot(self, c):
        n_frames = len(self.mgr.times) if (self.mgr is not None and self.mgr.times is not None) else 1
        if not self.mgr: return
        if c == -2: self._on_toggle_play()
        elif c == 0: self._on_reset_animation()
        elif c == 9999: self.update_frame(n_frames - 1)
        else:
            step = getattr(self, 'anim_step', 1)
            self.update_frame(max(0, min(n_frames-1, self.current_frame + c * step)))

    def _show_critical_error(self, title, msg):
        print(f"\n[CRITICAL ERROR] {title}: {msg}"); QtWidgets.QMessageBox.critical(self, title, msg)

    # --- WHT Inspector Field Management Logic ---
    def _populate_category_combos(self):
        """[WHTOOLS] 필드 키를 카테고리와 컴포넌트로 분리하여 콤보박스에 채웁니다."""
        if not self.field_keys: return
        
        categories = {}
        for fk in self.field_keys:
            if "[" in fk:
                cat = fk.split("[")[0].strip()
                comp = "[" + fk.split("[")[1]
            else:
                cat, comp = fk, "Value"
            
            if cat not in categories: categories[cat] = []
            categories[cat].append((comp, fk)) # (표시 이름, 실제 키) 저장
            
        self.cmb_cat.blockSignals(True); self.cmb_cat.clear()
        self.cmb_cat.addItem("Body Color")
        for cat in sorted(categories.keys()): self.cmb_cat.addItem(cat)
        self.cmb_cat.blockSignals(False)
        
        self._categories = categories
        self._on_category_changed(self.cmb_cat.currentText())


    def _on_component_changed(self):
        """컴포넌트 변경 시 범위 초기화 및 프레임 업데이트"""
        fk = self.cmb_comp.currentData()
        if fk is None or str(fk) == "Body Color" or str(fk) == "":
            self.update_frame(self.current_frame)
            return
            
        mode = self.rng_3d.cmb_mode.currentText()
        if mode == "Static":
            self._on_fit_range() 
        elif mode == "Robust":
            pct = self.rng_3d.get_robust_pct()
            r_min, r_max = self._calculate_robust_range(fk, p_low=(100-pct)/2, p_high=100-(100-pct)/2)
            self.rng_3d.set_range(r_min, r_max)
            
        self.update_frame(self.current_frame)

    def _show_range_dialog(self):
        fk = self.cmb_comp.currentData()
        if fk is None or str(fk) == "Body Color" or str(fk) == "": return
        def get_limits():
            all_vals = []
            for i, a in enumerate(self.mgr.analyzers):
                if i in self.part_actors and self.part_actors[i]['visible'] and fk in a.results:
                    all_vals.append(a.results[fk])
            if all_vals is None or len(all_vals) == 0: return 0.0, 1.0
            merged = np.concatenate([v.ravel() for v in all_vals])
            return float(np.nanmin(merged)), float(np.nanmax(merged))
        
        def get_robust(pct):
            return self._calculate_robust_range(fk, p_low=(100-pct)/2, p_high=100-(100-pct)/2)
            
        dlg = WHTRangeDialog(self, fk, self.rng_3d, get_limits, get_robust)
        dlg.exec()

    def _show_range_dialog_2d(self):
        cfg = self.plot_slots[self.active_slot]
        if not cfg: return
        fk = cfg.data_key
        
        def get_limits():
            all_vals = []
            for slot_cfg in self.plot_slots:
                if slot_cfg and slot_cfg.data_key == fk:
                    for p_idx in slot_cfg.part_indices:
                        ana = self.mgr.analyzers[p_idx]
                        if fk in ana.results: all_vals.append(ana.results[fk])
            if not all_vals: return 0.0, 1.0
            merged = np.concatenate([v.ravel() for v in all_vals])
            return float(np.nanmin(merged)), float(np.nanmax(merged))
            
        def get_robust(pct):
            all_vals = []
            for slot_cfg in self.plot_slots:
                if slot_cfg and slot_cfg.data_key == fk:
                    for p_idx in slot_cfg.part_indices:
                        ana = self.mgr.analyzers[p_idx]
                        if fk in ana.results: all_vals.append(ana.results[fk])
            if not all_vals: return 0.0, 1.0
            merged = np.concatenate([v.ravel() for v in all_vals])
            v_min = np.nanpercentile(merged, (100-pct)/2)
            v_max = np.nanpercentile(merged, 100-(100-pct)/2)
            return float(v_min), float(v_max)

        dlg = WHTRangeDialog(self, fk, self.rng_2d, get_limits, get_robust)
        dlg.exec()

    def _calculate_robust_range(self, field_key, p_low=2.0, p_high=98.0):
        """[WHTOOLS] 가시적인 파트의 전체 시간 데이터에서 유효 범위를 계산합니다."""
        if self.mgr is None: return 0.0, 1.0
        all_vals = []
        for i, ana in enumerate(self.mgr.analyzers):
            # [WHTOOLS] Only include data from VISIBLE parts
            if i in self.part_actors and self.part_actors[i]['visible']:
                if field_key in ana.results:
                    all_vals.append(ana.results[field_key].ravel())
        if not all_vals: return 0.0, 1.0
        
        merged = np.concatenate(all_vals)
        merged = merged[~np.isnan(merged)]
        if len(merged) == 0: return 0.0, 1.0

        # [DEBUG LOG] Statistical Distribution
        v_max = float(merged.max()); v_min = float(merged.min())
        p99 = float(np.percentile(merged, 99.0))
        p95 = float(np.percentile(merged, 95.0))
        p90 = float(np.percentile(merged, 90.0))
        
        print(f"\n[WHT-STATS] Field: {field_key}")
        print(f"  > Absolute MAX: {v_max:.4e}")
        print(f"  > 99th Pct    : {p99:.4e}")
        print(f"  > 95th Pct    : {p95:.4e}")
        print(f"  > 90th Pct    : {p90:.4e}")
        print(f"  > Absolute MIN: {v_min:.4e}")
        print(f"  > Total Nodes : {len(merged):,}")
        
        v_min_robust = np.nanpercentile(merged, p_low)
        v_max_robust = np.nanpercentile(merged, p_high)
        
        # [WHT] 데이터가 모두 동일한 경우를 대비한 수치 해석적 안정성 확보
        if v_min_robust >= v_max_robust:
            v_max_robust = v_min_robust + 1e-6
            
        return float(v_min_robust), float(v_max_robust)

    def _highlight_outliers(self, field_key, threshold):
        """Visualizes nodes that exceed the specified threshold as bright points."""
        f_i = self.current_frame
        found_any = False
        
        # Clear existing highlight if any
        if hasattr(self, "_outlier_actor"):
            self.v_int.remove_actor(self._outlier_actor)
            delattr(self, "_outlier_actor")

        all_outlier_pts = []
        for i, ana in enumerate(self.mgr.analyzers):
            if i not in self.part_actors or not self.part_actors[i]['visible']: continue
            if field_key not in ana.results: continue
            
            val = ana.results[field_key][f_i].ravel()
            mask = val > threshold
            if np.any(mask):
                inf = self.part_actors[i]
                pts = np.array(inf['poly'].points)[mask]
                all_outlier_pts.append(pts)
        
        if all_outlier_pts:
            merged_pts = np.concatenate(all_outlier_pts)
            import pyvista as pv
            cloud = pv.PolyData(merged_pts)
            self._outlier_actor = self.v_int.add_mesh(
                cloud, color="magenta", point_size=12, 
                render_points_as_spheres=True, label="Outliers",
                name="_wht_outliers"
            )
            found_any = True
            print(f" -> [WHT-DEBUG] Found {len(merged_pts)} nodes exceeding {threshold:.4e}")
        else:
            print(f" -> [WHT-DEBUG] No nodes found exceeding {threshold:.4e}")
        
        self.v_int.render()

    def clear_outliers(self):
        """Removes the magenta outlier highlight actor."""
        if hasattr(self, "_outlier_actor"):
            try:
                self.v_int.remove_actor(self._outlier_actor)
                delattr(self, "_outlier_actor")
                self.v_int.render()
            except: pass

    def _on_cmap_changed(self):
        """[WHTOOLS] 컬러맵 변경 시 LUT 재구축을 통해 색상 반전 버그를 방지합니다."""
        cmap_base = self.cmb_cmap.currentText()
        is_rev = self.ch_cmap_r.isChecked()
        cmap_name = cmap_base + "_r" if is_rev else cmap_base
        
        # [WHT] LUT 객체를 새로 생성하여 상태를 초기화 (반전 버그 해결책 및 범위 외 색상 제거)
        self.lut = pv.LookupTable(cmap=cmap_name)
        self.lut.below_range_color = None
        self.lut.above_range_color = None
        
        # 기존 액터들에게 새 LUT 적용
        for act in self.part_actors.values():
            if act['mesh']: act['mesh'].mapper.lookup_table = self.lut
            
        if hasattr(self, 'sb') and self.sb:
            self.sb.SetLookupTable(self.lut)
            
        self.update_frame(self.current_frame)

    def _update_scalar_bar(self, fk, v_min, v_max):
        """[WHTOOLS] 컬러 범례(Scalar Bar) 업데이트 (Professional Style)"""
        if not hasattr(self, 'sb') or self.sb is None: return
        
        fsize = self.v_font_size
        mode = self.cmb_cb_type.currentText()
        levels = int(self.cmb_cb_lv.currentText())
        decimals = self.sp_cb_dec.value()
        
        r_min, r_max = v_min, v_max
        if r_min >= r_max: r_max = r_min + 1e-6
        
        self.sb.SetVisibility(True)
        # [WHT] 타이틀과 바 사이의 간격을 확보하기 위해 끝에 줄바꿈(\n)과 공백 추가
        title_str = fk.replace(" ", "\n").replace("_", "\n") if len(fk) > 12 else fk
        self.sb.title = title_str + "\n "
        
        self.sb.format = f"%.{decimals}e"
        self.sb.n_labels = (levels + 1) if mode == "Discrete" else 11
        
        self.sb.unconstrained_font_size = True
        self.sb.title_font_size = fsize
        self.sb.label_font_size = max(6, fsize)
        
        # [WHT] VTK SetColor 근본 해결: GetVTKObject().SetColor(r, g, b) 패턴 사용
        bg_r, bg_g, bg_b = self.v_int.renderer.GetBackground()
        bg_brightness = 0.299 * bg_r + 0.587 * bg_g + 0.114 * bg_b
        fg_color = "black" if bg_brightness > 0.5 else "white"
        fr, fg_v, fb = pv.Color(fg_color).float_rgb
        
        try:
            self.sb.title_text_property.GetVTKObject().SetColor(fr, fg_v, fb)
            self.sb.label_text_property.GetVTKObject().SetColor(fr, fg_v, fb)
        except AttributeError:
            try:
                self.sb.GetTitleTextProperty().SetColor(fr, fg_v, fb)
                self.sb.GetLabelTextProperty().SetColor(fr, fg_v, fb)
            except Exception:
                pass
        
        # ParaView Style Position (Right side, Vertical)
        self.sb.position_x = 0.88; self.sb.position_y = 0.1
        self.sb.height = 0.7; self.sb.width = 0.08
        
        # LUT Update
        self.lut.n_values = levels if mode == "Discrete" else 256
        self.lut.scalar_range = (r_min, r_max)
        
        # [WHT] Disable explicit out-of-range colors to use colormap ends
        self.lut.above_range_color = None
        self.lut.below_range_color = None
            
        self.lut.Build()
        
        if hasattr(self.v_int, 'update_scalar_bar_range'):
            self.v_int.update_scalar_bar_range((r_min, r_max))
        
        # [WHT] Ensure scalar bar actor itself is updated
        if hasattr(self, 'sb') and self.sb is not None:
            self.sb.Modified()
        for act in self.part_actors.values():
            if act.get('mesh') is not None and hasattr(act['mesh'], 'mapper'):
                try: act['mesh'].mapper.SetScalarRange(r_min, r_max)
                except: pass
            
        # Status Text (Scientific notation, dynamic decimals synced with UI)
        unit = " [mm]" if "Disp" in fk else ""
        stat_msg = f"[{fk}]{unit}\nMin: {v_min:.{decimals}e}\nMax: {v_max:.{decimals}e}"
        self.v_int.add_text(stat_msg, position='upper_left', font_size=fsize, color=fg_color, name='st_ov', shadow=False)
        
        if hasattr(self, 'visibility_tool'): self.visibility_tool.update_info()

    def _on_bg_changed(self, color_name):
        """[WHT] 배경색 변경 및 폰트/그라운드 색상 자동 최적화 (Premium Grad. 지원)"""
        # 1. 배경색 및 폰트색(Foreground) 결정
        if color_name == "Black":
            self.v_int.set_background("black")
            fg = "white"
        elif color_name == "White":
            self.v_int.set_background("white")
            fg = "black"
        elif color_name == "Dark Grey":
            self.v_int.set_background("#222222")
            fg = "white"
        elif color_name == "Light Grey":
            self.v_int.set_background("#D3D3D3")
            fg = "black"
        elif color_name == "Grey Grad.":
            # ParaView Style: Dark Grey to Black
            self.v_int.set_background("#666666", top="black")
            fg = "white"
        elif color_name == "Light Grey Grad.":
            # [WHT-NEW] Light Grey to White (Default)
            self.v_int.set_background("white", top="#D3D3D3")
            fg = "black"
        elif color_name == "Light Sky Grad.":
            # [WHT-NEW] Light Sky Blue to White
            self.v_int.set_background("white", top="#E0F7FA")
            fg = "black"
        else: return

        # [WHT] VTK SetColor 에러 근본 해결:
        # pv.Color().float_rgb는 np.ndarray를 반환하며, VTK의 title_text_property.color에
        # ndarray를 직접 할당하면 SetColor 시그니처와 불일치로 TypeError가 발생합니다.
        # 반드시 .GetVTKObject().SetColor(r, g, b) 패턴을 사용해야 합니다.
        r, g, b = pv.Color(fg).float_rgb
        pv.global_theme.font.color = fg
        
        if hasattr(self, 'sb') and self.sb is not None:
            try:
                self.sb.title_text_property.GetVTKObject().SetColor(r, g, b)
                self.sb.label_text_property.GetVTKObject().SetColor(r, g, b)
            except AttributeError:
                try:
                    # PyVista 구버전 fallback: VTK 객체에 직접 접근
                    self.sb.GetTitleTextProperty().SetColor(r, g, b)
                    self.sb.GetLabelTextProperty().SetColor(r, g, b)
                except Exception:
                    pass
        
        if hasattr(self, 'ground') and self.ground is not None:
            # 밝은 배경에서는 더 어두운 바닥 그리드 사용
            g_r, g_g, g_b = pv.Color("#111111" if fg == "black" else "#333333").float_rgb
            self.ground.prop.color = (g_r, g_g, g_b)
            
        self.v_int.render()

    def _toggle_bg(self):
        """[WHT] 현재 배경색 반전 (White <-> Black)"""
        bg_r, bg_g, bg_b = self.v_int.renderer.GetBackground()
        bg_brightness = 0.299 * bg_r + 0.587 * bg_g + 0.114 * bg_b
        if bg_brightness > 0.5:
            self._on_bg_changed("Black")
        else:
            self._on_bg_changed("White")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = QtVisualizerV2()
    window.show()
    sys.exit(app.exec())
