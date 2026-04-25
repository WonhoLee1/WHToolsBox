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
            
            btn_show = QtWidgets.QPushButton("💡 Show All")
            btn_hide = QtWidgets.QPushButton("🌑 Hide All")
            
            btn_show.clicked.connect(partial(self._bulk_set, col_idx, True))
            btn_hide.clicked.connect(partial(self._bulk_set, col_idx, False))
            
            h_layout.addWidget(btn_show)
            h_layout.addWidget(btn_hide)
            global_layout.addLayout(h_layout)
        layout.addWidget(global_group)
        
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Part", "Mesh", "Markers", "View Mode", "Info (Min / Max)"])
        self.tree.setColumnWidth(0, 150)
        self.tree.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.tree)
        
        # 3. Focus Mode Button
        self.btn_focus = QtWidgets.QPushButton("🎯 Focus Selection (Contour Only)")
        self.btn_focus.setStyleSheet("background-color: #E8F0FE; font-weight: bold; padding: 5px;")
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
            
            item.setCheckState(1, m_v)
            item.setCheckState(2, mk_v)
            item.setText(2, f"Markers ({n_markers})")
            
            # View Mode ComboBox
            cmb = QtWidgets.QComboBox()
            cmb.addItems(["Surface w/ Edge", "Surface only", "Wireframe", "Outline"])
            cmb.setCurrentText(actor_data.get('view_mode', "Surface w/ Edge"))
            cmb.currentTextChanged.connect(partial(self._on_view_mode_changed, i))
            self.tree.setItemWidget(item, 3, cmb)
            
            item.setText(4, "-")
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
        if part_idx in self.parent.part_actors:
            self.parent.part_actors[part_idx]['view_mode'] = mode
            self.parent.update_frame(self.parent.current_frame)

    def _on_focus_view(self):
        """[WHTOOLS] 선택한 파트만 Contour, 나머지는 Wireframe으로 전환하는 상용 S/W 기법"""
        selected_items = self.tree.selectedItems()
        if not selected_items: return
        
        self.tree.blockSignals(True)
        target_id = selected_items[0].data(0, QtCore.Qt.UserRole)
        
        for i, act in self.parent.part_actors.items():
            if i == target_id:
                act['visible'] = True
                act['view_mode'] = "Surface only"
                self.id_to_item[i].setCheckState(1, QtCore.Qt.Checked)
            else:
                act['view_mode'] = "Wireframe"
            
            # UI ComboBox 동기화
            widget = self.tree.itemWidget(self.id_to_item[i], 3)
            if isinstance(widget, QtWidgets.QComboBox):
                widget.setCurrentText(act['view_mode'])
                
        self.tree.blockSignals(False)
        self.parent.update_frame(self.parent.current_frame)

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
                item.setText(4, f"{val.min():.2e} / {val.max():.2e}")
            else:
                item.setText(4, "-")
        self.tree.blockSignals(False)

    def _apply(self):
        """변경된 가시성 설정을 메인 렌더러에 즉각 반영"""
        for i, item in self.id_to_item.items():
            if i in self.parent.part_actors:
                self.parent.part_actors[i]['visible'] = (item.checkState(1) == QtCore.Qt.Checked)
                self.parent.part_actors[i]['visible_markers'] = (item.checkState(2) == QtCore.Qt.Checked)
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
        self.rb_v5 = QtWidgets.QRadioButton("마커 재구성 모드 (경계 근사형) - Marker Reconstruction (Approx. Boundary)")
        self.rb_v6 = QtWidgets.QRadioButton("정밀 형상 모드 (경계 확정형) - Verified Geometry (Exact Boundary)")
        
        # [WHTOOLS] 데이터 구조 기반 자동 추천
        if hasattr(self.result_data, 'time_history'):
            self.rb_v5.setChecked(True)
        else:
            self.rb_v6.setChecked(True)
            
        mode_layout.addWidget(self.rb_v5)
        mode_layout.addWidget(QtWidgets.QLabel("  <i style='color:gray; font-size:9pt;'>- 마커 위치를 추적하여 형상을 유추합니다. 정확한 외곽 치수(박스 크기) 보장이 어려울 수 있습니다.</i>"))
        mode_layout.addWidget(self.rb_v6)
        mode_layout.addWidget(QtWidgets.QLabel("  <i style='color:gray; font-size:9pt;'>- 코너 및 정밀 메쉬 정보를 직접 사용하여 실제 설계 치수와 완벽히 일치하는 결과를 보장합니다.</i>"))
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
            # metadata 위주로 빠르게 스캔 (실제 함수 재사용)
            markers, _ = get_assembly_data_from_sim(self.result_data, target_prefixes, mode=mode)
            
            self.table.setRowCount(0)
            self.part_configs = []
            
            for p_name, faces in markers.items():
                for f_name, m_data in faces.items():
                    if not m_data: continue
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
        self.visibility_tool = VisibilityToolWindow(self)
        self._init_ui()
        self._init_3d_view()
        self._init_2d_plots()
        
        if self.mgr:
            self._apply_initial_preset()
            # [WHTOOLS] Ensure initial font settings are applied
            self._update_vtk_font(self.v_font_size)
            self.update_frame(0)

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
        
        # Part Manager 동기화
        was_visible = False
        if hasattr(self, 'visibility_tool') and self.visibility_tool is not None:
            was_visible = self.visibility_tool.isVisible()
            self.visibility_tool.close()
            self.visibility_tool.deleteLater()
            
        self.visibility_tool = VisibilityToolWindow(self)
        if was_visible:
            self.visibility_tool.show()
            
        self.current_frame = 0
        self.is_playing = False
        if hasattr(self, 'timer'):
            self.timer.stop()
            try: self.timer.deleteLater()
            except: pass
        if hasattr(self, 'bp'): self.bp.setText("▶️")
        
        valid_analyzers = [a for a in manager.analyzers if a.results]
        if valid_analyzers:
            p0 = valid_analyzers[0]
            self.field_keys = [k for k in p0.results if p0.results[k].ndim == 3]
            self.stat_keys = [k for k in p0.results if k not in self.field_keys]
        else:
            self.field_keys = []; self.stat_keys = []
            
        if hasattr(self, 'cmb_3d'):
            self.cmb_3d.blockSignals(True)
            self.cmb_3d.clear()
            self.cmb_3d.addItems(["Body Color", "Face Color"] + self.field_keys)
            self.cmb_3d.blockSignals(False)
            
        self.v_int.clear()
        self.part_actors = {}
        self._init_3d_view()
        
        if hasattr(self, 'sld'):
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
        
        # [WHTOOLS] Colormap Style (3D 탭에서 이동됨)
        f_style = QtWidgets.QFrame(); l_style = QtWidgets.QHBoxLayout(f_style); l_style.setContentsMargins(0, 0, 0, 0)
        l_style.addWidget(QtWidgets.QLabel("Cmap:")); self.cmb_cmap = QtWidgets.QComboBox()
        self.cmb_cmap.addItems(["jet", "turbo", "rainbow", "viridis", "coolwarm", "plasma", "magma"])
        self.cmb_cmap.setCurrentText("jet"); self.cmb_cmap.setFixedWidth(100); l_style.addWidget(self.cmb_cmap)
        
        self.ch_cmap_r = QtWidgets.QCheckBox("Rev."); self.ch_cmap_r.setChecked(False); l_style.addWidget(self.ch_cmap_r)
        
        self.cmb_cmap.currentTextChanged.connect(self._on_cmap_changed)
        self.ch_cmap_r.toggled.connect(self._on_cmap_changed)
        layout.addWidget(f_style); layout.addWidget(self._create_v_line())
        
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
        pv.global_theme.font.label_size = max(6, v - 1)
        # 2. Matplotlib Global Params
        plt.rcParams['font.size'] = v
        # 3. UI Sync (2D Font ComboBox)
        if hasattr(self, 'cmb_font_2d'):
            self.cmb_font_2d.blockSignals(True)
            self.cmb_font_2d.setCurrentText(str(v))
            self.cmb_font_2d.blockSignals(False)
        # 4. Immediate Refresh
        if hasattr(self, 'sb'):
            self.sb.label_font_size = v
            self.sb.title_font_size = v
        self.update_frame(self.current_frame)

    def _on_2d_font_changed(self, text):
        """[WHTOOLS] 2D 그래프(Matplotlib)의 전역 폰트 크기 변경"""
        try:
            sz = int(text)
            plt.rcParams['font.size'] = sz
            self.update_frame(self.current_frame)
        except: pass

    def _apply_initial_preset(self):
        target_name = "opencell_front"; target_idx = -1
        for i, ana in enumerate(self.mgr.analyzers):
            if target_name in ana.name.lower(): target_idx = i; break
        if target_idx == -1: return
        self.cmb_3d.setCurrentText("Curvature Mean [1/mm]")
        self.cmb_lay.setCurrentText("2x1")
        self.plot_slots[0] = PlotSlotConfig(part_indices=[target_idx], plot_type="contour", data_key="Curvature Mean [1/mm]")
        self.plot_slots[1] = PlotSlotConfig(part_indices=[-2], plot_type="curve", data_key="Max-Curvature Mean [1/mm]")

    def _update_step(self, v):
        self.anim_step = v
        if hasattr(self, 'sp_step_ui'):
            self.sp_step_ui.blockSignals(True); self.sp_step_ui.setValue(v); self.sp_step_ui.blockSignals(False)

    def _init_3d_controls(self, p):
        """
        [WHTOOLS] 3D 뷰어 제어 패널을 구성합니다. (Ribbon Style)
        사용자 요청에 따라 2행 레이아웃으로 구성하였습니다.
        Row 1: Part Manager, View Mode, Scale, Environment, Field Selection
        Row 2: Range, Min, Max, Fit 버튼
        """
        main_layout = QtWidgets.QVBoxLayout(p)
        main_layout.setContentsMargins(15, 2, 15, 2)
        main_layout.setSpacing(5)
        
        row1 = QtWidgets.QHBoxLayout()
        row2 = QtWidgets.QHBoxLayout()
        main_layout.addLayout(row1)
        main_layout.addLayout(row2)
        
        # --- Row 1: 기본 관리 및 필드 선택 ---
        f_pm = QtWidgets.QFrame(); l_pm = QtWidgets.QHBoxLayout(f_pm); l_pm.setContentsMargins(0, 0, 0, 0)
        self.btn_pm = QtWidgets.QPushButton("🧱 Part Manager 👁️")
        self.btn_pm.setMinimumHeight(32)
        self.btn_pm.clicked.connect(lambda: self.visibility_tool.show() if self.visibility_tool else None)
        l_pm.addWidget(self.btn_pm)
        row1.addWidget(f_pm); row1.addWidget(self._create_v_line())
        
        f_vm = QtWidgets.QFrame(); l_vm = QtWidgets.QHBoxLayout(f_vm); l_vm.setContentsMargins(0, 0, 0, 0)
        l_vm.addWidget(QtWidgets.QLabel("View:")); self.cmb_v = QtWidgets.QComboBox(); self.cmb_v.addItems(["Global", "Local"])
        self.cmb_v.currentTextChanged.connect(lambda: self.update_frame(self.current_frame)); l_vm.addWidget(self.cmb_v)
        l_vm.addWidget(QtWidgets.QLabel("Scale:")); self.sp_sc = QtWidgets.QDoubleSpinBox(); self.sp_sc.setRange(1.0, 1000.0); self.sp_sc.setValue(1.0)
        self.sp_sc.valueChanged.connect(lambda: self.update_frame(self.current_frame)); l_vm.addWidget(self.sp_sc)
        row1.addWidget(f_vm); row1.addWidget(self._create_v_line())
        
        f_env = QtWidgets.QFrame(); l_env = QtWidgets.QHBoxLayout(f_env); l_env.setContentsMargins(0, 0, 0, 0)
        v_col = QtWidgets.QVBoxLayout(); v_col.setSpacing(2)
        self.ch_per = QtWidgets.QCheckBox("Persp."); self.ch_per.setChecked(True); self.ch_per.toggled.connect(self._on_persp_toggled)
        v_col.addWidget(self.ch_per)
        self.btn_fv = QtWidgets.QPushButton("📐"); self.btn_fv.setToolTip("Fit View")
        self.btn_fv.setMinimumHeight(24); self.btn_fv.clicked.connect(lambda: self.v_int.reset_camera())
        v_col.addWidget(self.btn_fv)
        l_env.addLayout(v_col)
        l_env.addWidget(QtWidgets.QLabel("BG:")); self.cmb_bg = QtWidgets.QComboBox(); self.cmb_bg.addItems(["Grey Grad.", "Sky Grad.", "White", "Black"])
        self.cmb_bg.setCurrentText("Grey Grad."); self.cmb_bg.currentTextChanged.connect(self._on_bg_changed); l_env.addWidget(self.cmb_bg)
        row1.addWidget(f_env); row1.addWidget(self._create_v_line())
        
        f_fld = QtWidgets.QFrame(); l_fld = QtWidgets.QHBoxLayout(f_fld); l_fld.setContentsMargins(0, 0, 0, 0)
        l_fld.addWidget(QtWidgets.QLabel("Field:")); self.cmb_3d = QtWidgets.QComboBox(); self.cmb_3d.addItems(["Body Color", "Face Color"])
        self.cmb_3d.setFixedWidth(300); l_fld.addWidget(self.cmb_3d)
        row1.addWidget(f_fld)
        row1.addStretch(1)
        
        # --- Row 2: 범례 및 수치 제어 ---
        f_leg = QtWidgets.QFrame(); l_leg = QtWidgets.QHBoxLayout(f_leg); l_leg.setContentsMargins(0, 0, 0, 0)
        l_leg.addWidget(QtWidgets.QLabel("Range:")); self.cmb_l = QtWidgets.QComboBox(); self.cmb_l.addItems(["Dynamic", "Static"]); l_leg.addWidget(self.cmb_l)
        l_leg.addWidget(QtWidgets.QLabel("Min:")); self.sp_min = QtWidgets.QDoubleSpinBox(); self.sp_min.setRange(-1e9, 1e9); self.sp_min.setDecimals(4)
        self.sp_min.setFixedWidth(90); self.sp_min.setEnabled(False); l_leg.addWidget(self.sp_min)
        l_leg.addWidget(QtWidgets.QLabel("Max:")); self.sp_max = QtWidgets.QDoubleSpinBox(); self.sp_max.setRange(-1e9, 1e9); self.sp_max.setDecimals(4)
        self.sp_max.setFixedWidth(90); self.sp_max.setEnabled(False); l_leg.addWidget(self.sp_max)
        self.btn_fit = QtWidgets.QPushButton("🔄 Fit")
        self.btn_fit.setFixedWidth(50); self.btn_fit.setEnabled(False)
        self.btn_fit.clicked.connect(self._on_fit_range)
        l_leg.addWidget(self.btn_fit)
        row2.addWidget(f_leg)
        row2.addStretch(1)
        
        # Signals 연결
        self.cmb_3d.currentTextChanged.connect(self._on_field_changed)
        self.cmb_l.currentTextChanged.connect(self._on_legend_mode_changed)
        self.sp_min.valueChanged.connect(lambda: self.update_frame(self.current_frame))
        self.sp_max.valueChanged.connect(lambda: self.update_frame(self.current_frame))

    def _init_2d_controls(self, p):
        """
        [WHTOOLS] 2D 그래프 세션 제어 패널을 구성합니다. (Ribbon Style)
        화면을 N x M 그리드로 분할하여 여러 그래프를 동시에 비교 분석할 수 있는 UI를 제공하며,
        각 그래프의 범위(Range), 테마, 폰트 크기 등을 조절하는 옵션을 포함합니다.
        """
        layout = QtWidgets.QHBoxLayout(p)
        layout.setContentsMargins(15, 2, 15, 2); layout.setSpacing(10)
        
        # 1. Grid Layout & Plot Manager (그리드 분할 및 플롯 관리)
        f_lay = QtWidgets.QFrame(); l_lay = QtWidgets.QHBoxLayout(f_lay); l_lay.setContentsMargins(0, 0, 0, 0)
        l_lay.addWidget(QtWidgets.QLabel("Grid:")); self.cmb_lay = QtWidgets.QComboBox(); self.cmb_lay.addItems(["2x1", "1x1", "1x2", "2x2", "3x2"])
        
        # [WHTOOLS] 활성 슬롯(Active Slot) 명시적 선택 위젯 추가
        l_lay.addWidget(QtWidgets.QLabel("Slot:")); self.cmb_slot = QtWidgets.QComboBox()
        self.cmb_slot.setFixedWidth(70)
        self.cmb_slot.currentIndexChanged.connect(self._on_slot_changed)
        l_lay.addWidget(self.cmb_slot)
        
        # 시그널 연결 후 초기값 설정 (초기 슬롯 목록 생성 트리거)
        self.cmb_lay.currentTextChanged.connect(self._on_grid_layout_changed)
        self.cmb_lay.setCurrentText("2x1")
        l_lay.addWidget(self.cmb_lay)
        
        # 활성화된 슬롯에 새로운 데이터 플롯을 추가하는 버튼
        bt_add = QtWidgets.QPushButton("➕ Add Plot"); bt_add.clicked.connect(self._show_add_plot_dialog); l_lay.addWidget(bt_add)
        
        # 현재 화면에 생성된 모든 2D 그래프를 일괄 삭제하는 버튼
        self.btn_clear_2d = QtWidgets.QPushButton("🗑️ Clear Plots")
        self.btn_clear_2d.clicked.connect(self._on_clear_2d_plots)
        l_lay.addWidget(self.btn_clear_2d)
        
        layout.addWidget(f_lay); layout.addWidget(self._create_v_line())
        
        # 2. 2D Data Range Control (2D 전용 범위 제어 세트)
        f_rng = QtWidgets.QFrame(); l_rng = QtWidgets.QHBoxLayout(f_rng); l_rng.setContentsMargins(0, 0, 0, 0)
        l_rng.addWidget(QtWidgets.QLabel("Range:")); self.cmb_range_2d = QtWidgets.QComboBox()
        self.cmb_range_2d.addItems(["Dynamic", "Static"]); l_rng.addWidget(self.cmb_range_2d)
        
        l_rng.addWidget(QtWidgets.QLabel("Min:")); self.sp_min_2d = QtWidgets.QDoubleSpinBox(); self.sp_min_2d.setRange(-1e9, 1e9); self.sp_min_2d.setDecimals(4)
        self.sp_min_2d.setFixedWidth(80); self.sp_min_2d.setEnabled(False); l_rng.addWidget(self.sp_min_2d)
        
        l_rng.addWidget(QtWidgets.QLabel("Max:")); self.sp_max_2d = QtWidgets.QDoubleSpinBox(); self.sp_max_2d.setRange(-1e9, 1e9); self.sp_max_2d.setDecimals(4)
        self.sp_max_2d.setFixedWidth(80); self.sp_max_2d.setEnabled(False); l_rng.addWidget(self.sp_max_2d)
        
        # 모든 활성화된 2D 슬롯의 전체 데이터 Min/Max를 스캔하여 Static 범위로 자동 기입
        self.btn_fit_2d = QtWidgets.QPushButton("🔄 Fit")
        self.btn_fit_2d.setFixedWidth(50); self.btn_fit_2d.setEnabled(False)
        self.btn_fit_2d.clicked.connect(self._on_fit_range_2d)
        l_rng.addWidget(self.btn_fit_2d)
        
        self.cmb_range_2d.currentTextChanged.connect(self._on_range_mode_changed_2d)
        self.sp_min_2d.valueChanged.connect(lambda: self.update_frame(self.current_frame))
        self.sp_max_2d.valueChanged.connect(lambda: self.update_frame(self.current_frame))
        
        layout.addWidget(f_rng); layout.addWidget(self._create_v_line())
        f_opt = QtWidgets.QFrame(); l_opt = QtWidgets.QHBoxLayout(f_opt); l_opt.setContentsMargins(0, 0, 0, 0)
        self.checks = {}
        for t, s in [("Sync", True), ("Interp.", True)]:
            c = QtWidgets.QCheckBox(t); c.setChecked(s); c.toggled.connect(lambda: self.update_frame(self.current_frame))
            l_opt.addWidget(c); self.checks[t.replace('.','')] = c
        layout.addWidget(f_opt); layout.addWidget(self._create_v_line())
        f_tls = QtWidgets.QFrame(); l_tls = QtWidgets.QHBoxLayout(f_tls); l_tls.setContentsMargins(0, 0, 0, 0)
        bt_pop = QtWidgets.QPushButton("📺 Pop-out View"); bt_pop.clicked.connect(self._pop_out_2d); l_tls.addWidget(bt_pop)
        l_tls.addWidget(QtWidgets.QLabel("Theme:")); self.cmb_theme = QtWidgets.QComboBox(); self.cmb_theme.addItems(['default', 'ggplot', 'bmh', 'dark_background'])
        self.cmb_theme.currentTextChanged.connect(lambda: self.update_frame(self.current_frame)); l_tls.addWidget(self.cmb_theme)
        
        l_tls.addWidget(QtWidgets.QLabel("Font:")); self.cmb_font_2d = QtWidgets.QComboBox()
        self.cmb_font_2d.addItems([str(i) for i in range(6, 31)]) # Range extended to match 3D
        self.cmb_font_2d.setCurrentText("9") # Default to 9
        self.cmb_font_2d.currentTextChanged.connect(lambda v: self.sp_vtk_font.setValue(int(v)))
        l_tls.addWidget(self.cmb_font_2d)
        
        layout.addWidget(f_tls); layout.addStretch(1)

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
        for i, ana in enumerate(analyzers):
            if ana.m_raw is None or ana.sol is None:
                self.part_actors[i] = {'mesh': None, 'visible': False}
                continue
                
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
            
            mp = pv.PolyData(np.array(ana.m_raw[0]))
            n_m = ana.m_raw.shape[1]
            mp.point_data["names"] = [f"{ana.name}_M{j:02d}" for j in range(n_m)]
            mka = self.v_int.add_mesh(
                mp, 
                render_points_as_spheres=True, 
                point_size=10, 
                color='skyblue'
            )
            
            la = self.v_int.add_point_labels(
                mp, "names", 
                font_size=self.v_font_size, 
                text_color='black', 
                always_visible=True, 
                point_size=0, 
                shadow=False
            )
            
            if not hasattr(ana.sol, 'X_mesh') or ana.sol.X_mesh is None:
                ma.SetVisibility(False)
                self.part_actors[i] = {'mesh': ma, 'visible': False}
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
                "Field Analysis [mm]", 
                position_x=0.15, position_y=0.05, 
                width=0.7, 
                mapper=self.part_actors[f_i]['mesh'].mapper,
                title_font_size=fsize,
                label_font_size=fsize - 1,
                n_labels=5,
                fmt="%.3e"
            )
        else:
            self.sb = self.v_int.add_scalar_bar("No Data", position_x=0.15)
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

    def _on_cmap_changed(self):
        """[WHTOOLS] 3D 컬러맵(LUT)을 즉시 업데이트하고 범례 및 2D 플롯과 동기화합니다."""
        cmap_base = self.cmb_cmap.currentText()
        is_rev = self.ch_cmap_r.isChecked()
        
        try:
            cmap_obj = plt.get_cmap(cmap_base)
            colors = cmap_obj(np.linspace(0, 1, 256))
            if is_rev: colors = colors[::-1]
            
            self.lut.SetNumberOfTableValues(256)
            for i in range(256):
                c = colors[i]
                self.lut.SetTableValue(i, c[0], c[1], c[2], c[3])
            
            self.lut.below_range_color = 'lightgrey'
            self.lut.above_range_color = 'magenta'
            self.lut.Modified() 
        except Exception as e:
            print(f"[WHTOOLS] Low-level LUT update failed: {e}")
        
        for ai in self.part_actors.values():
            if ai['mesh'] is not None:
                ai['mesh'].mapper.lookup_table = self.lut
                ai['mesh'].mapper.Modified()
        
        if hasattr(self, 'sb') and self.sb is not None:
            self.sb.SetLookupTable(self.lut)
            self.sb.Modified()
            self.v_int.render()
            
        self.update_frame(self.current_frame)
        self.v_int.render()

    def _update_scalar_bar(self, fk, v_min, v_max, dy):
        """[WHTOOLS] 컬러 범례(Scalar Bar)의 범위 및 폰트 정보를 업데이트합니다."""
        if not hasattr(self, 'sb') or self.sb is None:
            return
            
        current_state = (fk, v_min, v_max, dy, self.v_font_size)
        if hasattr(self, '_prev_sb_state') and self._prev_sb_state == current_state:
            return
        self._prev_sb_state = current_state

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
        
        r_min, r_max = clim[0], clim[1]
        
        self.lut.scalar_range = (r_min, r_max)
        self.sb.SetVisibility(True)
        self.sb.title = f"[{fk}] Analysis [mm]"
        
        self.v_int.update_scalar_bar_range((r_min, r_max))
        
        for ai in self.part_actors.values():
            if ai['mesh'] is not None:
                ai['mesh'].mapper.SetScalarRange(r_min, r_max)
        
        fsize = self.v_font_size
        self.sb.title_font_size = fsize
        self.sb.label_font_size = max(6, fsize)
        
        unit_str = "" if ("(" in fk or "[" in fk) else " mm"
        status_text = f"[{fk}]\nMin: {v_min:.3f}{unit_str}\nMax: {v_max:.3f}{unit_str}"
        self.v_int.add_text(status_text, position='upper_left', font_size=fsize, color='white', name='st_ov', shadow=True)

    def update_frame(self, f_i: int):
        """
        주어진 시점(f_i)으로 모든 파트의 변형 정보를 실시간 업데이트.
        """
        if self.mgr is None or self.mgr.times is None:
            return
            
        if not hasattr(self, 'sp_min') or not hasattr(self, 'sp_max') or not hasattr(self, 'sb') or not hasattr(self, 'sld'):
            return
            
        self.current_frame = f_i
        
        self.sld.blockSignals(True)
        self.sld.setValue(f_i)
        self.sld.blockSignals(False)
        
        n_frames_tot = len(self.mgr.times) if self.mgr.times is not None else 1
        self.lf.setText(f"Frame: {f_i} / {n_frames_tot - 1}")
        
        vm = self.cmb_v.currentText()
        fk = self.cmb_3d.currentText()
        sc = self.sp_sc.value()
        dy = self.cmb_l.currentText() == "Dynamic"
        
        active_values = []
        
        for i, ana in enumerate(self.mgr.analyzers):
            if i not in self.part_actors: continue
            inf = self.part_actors[i]
            if inf['mesh'] is None: continue
            
            mv = inf['visible']; mkv = inf['visible_markers']
            inf['mesh'].SetVisibility(mv)
            inf['markers'].SetVisibility(mkv)
            inf['labels'].SetVisibility(mkv)
            
            if not mv and not mkv: continue
            
            displacement_w = ana.results.get('Displacement [mm]', np.zeros((n_frames_tot, ana.sol.res, ana.sol.res)))[f_i]
            
            points_local = inf['p_base'].copy()
            points_local[:, 2] = displacement_w.ravel() * sc
            
            R_matrix = ana.results.get('R_matrix')[f_i]
            cur_centroid = ana.results.get('cur_centroid')[f_i]
            ref_centroid = ana.results.get('ref_centroid')[f_i]
            
            if ana.kin:
                local_basis = np.array(ana.kin.local_basis_axes)
                local_cent_0 = np.array(ana.kin.local_centroid_0)
            else:
                local_basis = np.array(ana.results.get('local_basis_axes', np.eye(3)))
                local_cent_0 = np.array(ana.results.get('local_centroid_0', np.zeros(3)))
            
            if vm == "Global":
                inf['poly'].points = (
                    points_local @ local_basis.T + 
                    local_cent_0 - ref_centroid
                ) @ R_matrix + cur_centroid
                inf['m_poly'].points = np.array(ana.m_raw[f_i])
            else:
                inf['poly'].points = points_local
                inf['m_poly'].points = np.array(ana.results.get('local_markers')[f_i])
                
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
                        
            vm_mode = inf.get('view_mode', "Surface w/ Edge")
            prop = inf['mesh'].GetProperty()
            if vm_mode == "Surface w/ Edge":
                prop.SetRepresentationToSurface(); prop.SetEdgeVisibility(True)
            elif vm_mode == "Surface only":
                prop.SetRepresentationToSurface(); prop.SetEdgeVisibility(False)
            elif vm_mode == "Wireframe":
                prop.SetRepresentationToWireframe(); prop.SetEdgeVisibility(True)
            elif vm_mode == "Outline":
                prop.SetRepresentationToWireframe(); prop.SetEdgeVisibility(False); prop.SetOpacity(0.3)
            else:
                prop.SetOpacity(1.0)
            
            inf['poly'].Modified()
            inf['m_poly'].Modified()
            
        if active_values and fk not in ["Body Color", "Face Color"]:
            v_min = float(min(v.min() for v in active_values))
            v_max = float(max(v.max() for v in active_values))
            self._update_scalar_bar(fk, v_min, v_max, dy)
        else:
            self.sb.SetVisibility(False)
            self.v_int.add_text("", position='upper_left', name='st_ov')
            
        self._update_2d_plots(f_i)
        self.v_int.render()

    # --------------------------------------------------------------------------
    # --- 2D Plotting Engine (Matplotlib) ---
    # --------------------------------------------------------------------------

    def _on_grid_layout_changed(self, text):
        """그리드 레이아웃 변경 시 슬롯 선택 위젯 항목을 동기화합니다."""
        self._init_2d_plots()
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
            if item.widget(): item.widget().setParent(None)
                
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
        if self.mgr is None or self.mgr.times is None or not self.axes: return
            
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
                
            p_idx_main = cfg.part_indices[0] if cfg.part_indices else 0
            if p_idx_main < 0: p_idx_main = 0
            ana = self.mgr.analyzers[p_idx_main]
            key = cfg.data_key
            
            if cfg.plot_type == "contour":
                if ana.sol is None: continue
                data_2d = ana.results.get(key, np.zeros((len(self.mgr.times), ana.sol.res, ana.sol.res)))[f_i]
                dy_2d = self.cmb_range_2d.currentText() == "Dynamic"
                vmin, vmax = float(data_2d.min()), float(data_2d.max())
                
                if dy_2d:
                    self.sp_min_2d.blockSignals(True); self.sp_min_2d.setValue(vmin); self.sp_min_2d.blockSignals(False)
                    self.sp_max_2d.blockSignals(True); self.sp_max_2d.setValue(vmax); self.sp_max_2d.blockSignals(False)
                    clim = [vmin, vmax]
                else: clim = [self.sp_min_2d.value(), self.sp_max_2d.value()]
                
                if clim[0] >= clim[1]: clim[1] = clim[0] + 1e-6
                
                if self.ims[i] is None:
                    ax.clear()
                    if self.cbs[i] is not None:
                        try: self.cbs[i].ax.remove(); self.cbs[i] = None
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
        
        if getattr(self, '_is_first_2d_update', False):
            self.fig.tight_layout(); self._is_first_2d_update = False
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

    def _on_bg_changed(self, text):
        if text == "Grey Grad.": self.v_int.set_background("white", top="lightgray")
        elif text == "Sky Grad.": self.v_int.set_background("white", top="#E0F7FA")
        elif text == "White": self.v_int.set_background("white")
        elif text == "Black": self.v_int.set_background("black")
        self.v_int.render()

    def _on_legend_mode_changed(self, mode):
        if self.mgr is None: return
        is_static = (mode == "Static")
        if hasattr(self, 'sp_min'): self.sp_min.setEnabled(is_static)
        if hasattr(self, 'sp_max'): self.sp_max.setEnabled(is_static)
        if hasattr(self, 'btn_fit'): self.btn_fit.setEnabled(is_static)
        if is_static: self._on_fit_range()
        self.update_frame(self.current_frame)

    def _on_fit_range(self):
        if self.mgr is None: return
        field_key = self.cmb_3d.currentText()
        if field_key in ["Body Color", "Face Color"]: return
        all_values = []
        for a in self.mgr.analyzers:
            if field_key in a.results: all_values.append(a.results[field_key])
        if all_values:
            v_min = float(min(v.min() for v in all_values)); v_max = float(max(v.max() for v in all_values))
            self.sp_min.blockSignals(True); self.sp_min.setValue(v_min); self.sp_min.blockSignals(False)
            self.sp_max.blockSignals(True); self.sp_max.setValue(v_max); self.sp_max.blockSignals(False)
            self.update_frame(self.current_frame)

    def _on_field_changed(self, field_key):
        if self.mgr is None: return
        if self.cmb_l.currentText() == "Static": self._on_fit_range()
        self.update_frame(self.current_frame)

    def _on_range_mode_changed_2d(self, mode):
        is_static = (mode == "Static")
        self.sp_min_2d.setEnabled(is_static); self.sp_max_2d.setEnabled(is_static); self.btn_fit_2d.setEnabled(is_static)
        if is_static: self._on_fit_range_2d()
        self.update_frame(self.current_frame)

    def _on_fit_range_2d(self):
        if self.mgr is None: return
        all_vals = []
        for cfg in self.plot_slots:
            if cfg and cfg.plot_type == "contour":
                p_idx = cfg.part_indices[0] if cfg.part_indices else 0
                ana = self.mgr.analyzers[p_idx]
                if cfg.data_key in ana.results: all_vals.append(ana.results[cfg.data_key])
        if all_vals:
            v_min = float(min(v.min() for v in all_vals)); v_max = float(max(v.max() for v in all_vals))
            self.sp_min_2d.blockSignals(True); self.sp_min_2d.setValue(v_min); self.sp_min_2d.blockSignals(False)
            self.sp_max_2d.blockSignals(True); self.sp_max_2d.setValue(v_max); self.sp_max_2d.blockSignals(False)
            self.update_frame(self.current_frame)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Space: self._ctrl_slot(-2)
        else: super().keyPressEvent(event)

    def _show_part_menu(self, pos=None):
        if pos is None: pos = self.v_int.mapFromGlobal(QtGui.QCursor.pos())
        menu = QtWidgets.QMenu(self); menu.addAction("Visibility Manager", self.visibility_tool.show); menu.addSeparator()
        view_actions = [("XY Plane", self.v_int.view_xy), ("YZ Plane", self.v_int.view_yz), ("ZX Plane", self.v_int.view_zx), ("Isometric", self.v_int.view_isometric)]
        for name, func in view_actions: menu.addAction(name, func)
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
            self._update_selection_ui(); self._is_first_2d_update = True; self.update_frame(self.current_frame); self.can.draw_idle()

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

    def _show_warning(self, title, msg):
        print(f"\n[WARNING] {title}: {msg}"); QtWidgets.QMessageBox.warning(self, title, msg)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = QtVisualizerV2()
    window.show()
    sys.exit(app.exec())
