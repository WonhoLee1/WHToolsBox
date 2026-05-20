# -*- coding: utf-8 -*-
"""
[WHTOOLS] Simulation Control Center v1.0
PySide6 기반의 현대적인 MuJoCo 시뮬레이션 제어 패널입니다.
"""

import os
import sys
import time
import ast
import json
import tempfile
from functools import partial
import numpy as np
from pathlib import Path
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QSlider, QLabel, QFrame, QGroupBox, QDoubleSpinBox, QAbstractSpinBox,
    QPlainTextEdit, QDialog, QMessageBox, QSplitter, QTreeWidget, QTreeWidgetItem
)
from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QFont, QIcon, QColor, QPalette, QPixmap
import ctypes
from ctypes import wintypes
from .whts_theme import (
    apply_app_theme, GLOBAL_QSS, TREE_QSS,
    C_BG_TABLE, C_BG_BTN, C_BG_INPUT, C_BG_EDITOR, C_ACCENT, C_TEXT_DIM, C_BORDER, C_BORDER_IN, C_TEXT_TREE, C_TEXT_MUTED, C_SEL, C_BG,
    C_BTN_GREEN, C_BTN_GREEN2, C_BTN_BLUE, C_BTN_BLUE2, C_BTN_BLUE3,
    C_BTN_INDIGO, C_BTN_BROWN, C_BTN_RED, C_BTN_TEAL, C_BTN_NAVY, C_BTN_NAVY_BORDER, C_ACCENT_BLUE,
    C_STATE_SLOW_MOTION, C_STATE_RECORDING, C_STATE_REC_TEXT, C_STATE_ORANGE,
    C_STATUS_OK, C_STATUS_WARN, C_STATUS_INFO, C_STATUS_ERR,
    C_STATUS_TEXT_OK, C_STATUS_TEXT_WARN,
)
import mujoco

# [WHTOOLS] 설정 키 메타데이터 (설명 및 카테고리)
CONFIG_METADATA = {
    # [Geometry]
    "box_w": {"desc": "Outer Box Width (m)", "cat": "Geometry"},
    "box_h": {"desc": "Outer Box Height (m)", "cat": "Geometry"},
    "box_d": {"desc": "Outer Box Depth (m)", "cat": "Geometry"},
    "box_thick": {"desc": "Outer Box Board Thickness (m)", "cat": "Geometry"},
    "assy_w": {"desc": "Chassis Width (m)", "cat": "Geometry"},
    "assy_h": {"desc": "Chassis Height (m)", "cat": "Geometry"},
    "chassis_d": {"desc": "Chassis Thickness (m)", "cat": "Geometry"},
    "opencell_d": {"desc": "OpenCell Thickness (m)", "cat": "Geometry"},
    "cush_gap": {"desc": "Cushion Gap Size (m)", "cat": "Geometry"},
    "opencellcoh_d": {"desc": "Cohesive Tape Thickness (m)", "cat": "Geometry"},
    "occ_ithick": {"desc": "Cohesive Tape Width/Interval Thickness (m)", "cat": "Geometry"},

    # [Components & Balancing Options]
    "components": {"desc": "Component Specifications (Mass, Meshing, Weld, Color)", "cat": "Components"},
    "components_balance": {"desc": "Assembly Mass, CoG & MoI Target Specifications", "cat": "Components Balance"},

    # [Drop Env]
    "drop_mode": {"desc": "Drop Mode (LTL, PARCEL, etc.)", "cat": "Drop Env"},
    "drop_direction": {"desc": "Drop Target (e.g. Corner 2-3-5)", "cat": "Drop Env"},
    "drop_height": {"desc": "Drop Height (m)", "cat": "Drop Env"},
    "sim_duration": {"desc": "Simulation Duration (s)", "cat": "Drop Env"},
    "include_paperbox": {"desc": "Include Outer Paper Box in Simulation", "cat": "Drop Env"},
    "include_cushion": {"desc": "Include Cushion in Simulation", "cat": "Drop Env"},
    "use_postprocess_ui": {"desc": "Enable Post-process UI", "cat": "Drop Env"},
    "use_postprocess_v2": {"desc": "Use V2 Post-processor", "cat": "Drop Env"},
    "use_viewer": {"desc": "Enable MuJoCo Interactive Viewer", "cat": "Drop Env"},
    "initial_tilt_deg": {"desc": "Initial Tilt Latitude (deg)", "cat": "Drop Env"},
    "initial_tilt_azimuth_deg": {"desc": "Initial Tilt Azimuth (deg)", "cat": "Drop Env"},

    # [Meshing]
    "chassis_div": {"desc": "Chassis Element Divisions [nx, ny, nz]", "cat": "Meshing"},
    "chassis_use_weld": {"desc": "Weld Chassis Elements Together", "cat": "Meshing"},
    "opencell_div": {"desc": "OpenCell Element Divisions [nx, ny, nz]", "cat": "Meshing"},
    "opencell_use_weld": {"desc": "Weld OpenCell Elements Together", "cat": "Meshing"},
    "opencellcoh_div": {"desc": "Tape Element Divisions [nx, ny, nz]", "cat": "Meshing"},
    "opencellcoh_use_weld": {"desc": "Weld Tape Elements Together", "cat": "Meshing"},
    "cush_div": {"desc": "Cushion Element Divisions [nx, ny, nz]", "cat": "Meshing"},
    "cush_use_weld": {"desc": "Weld Cushion Elements Together", "cat": "Meshing"},
    "box_div": {"desc": "Paper Box Element Divisions [nx, ny, nz]", "cat": "Meshing"},
    "box_use_weld": {"desc": "Weld Paper Box Elements Together", "cat": "Meshing"},

    # [Solver]
    "sim_timestep": {"desc": "Solver Time Step (s)", "cat": "Solver"},
    "sim_integrator": {"desc": "Integrator (implicitfast, etc.)", "cat": "Solver"},
    "sim_iterations": {"desc": "Max Solver Iterations", "cat": "Solver"},
    "sim_noslip_iterations": {"desc": "Max Solver No-slip Iterations", "cat": "Solver"},
    "sim_tolerance": {"desc": "Solver Convergence Tolerance", "cat": "Solver"},
    "sim_impratio": {"desc": "Solver Impedance Ratio", "cat": "Solver"},
    "sim_gravity": {"desc": "Gravity Vector [gx, gy, gz]", "cat": "Solver"},

    # [Weld Physics]
    "welds": {"desc": "Weld Connector Specifications (solref, solimp, torquescale)", "cat": "Weld Physics"},

    # [Contact Specs]
    "cush_friction": {"desc": "Cushion Friction Coefficient", "cat": "Contact Specs"},
    "paper_friction": {"desc": "Paper Box Friction Coefficient", "cat": "Contact Specs"},
    "ground_friction": {"desc": "Ground Friction Coefficient", "cat": "Contact Specs"},
    "cush_contact_solref": {"desc": "Cushion Contact Solref Vector", "cat": "Contact Specs"},
    "cush_contact_solimp": {"desc": "Cushion Contact Solimp Vector", "cat": "Contact Specs"},
    "cush_corner_solref": {"desc": "Cushion Corner Contact Solref Vector", "cat": "Contact Specs"},
    "cush_corner_solimp": {"desc": "Cushion Corner Contact Solimp Vector", "cat": "Contact Specs"},
    "ground_solref": {"desc": "Ground Contact Solref Vector", "cat": "Contact Specs"},
    "ground_solimp": {"desc": "Ground Contact Solimp Vector", "cat": "Contact Specs"},

    # [Plasticity]
    "enable_plasticity": {"desc": "Enable Cushion Plasticity", "cat": "Plasticity"},
    "plasticity_ratio": {"desc": "Cushion Plasticity Ratio", "cat": "Plasticity"},
    "cush_yield_pressure": {"desc": "Cushion Yield Pressure (Pa)", "cat": "Plasticity"},
    "plastic_hardening_modulus": {"desc": "Plastic Hardening Modulus (Pa)", "cat": "Plasticity"},
    "plastic_color_limit": {"desc": "Plastic Color Visualization Limit", "cat": "Plasticity"},
    "plastic_max_strain": {"desc": "Maximum Allowed Plastic Strain", "cat": "Plasticity"},
    "debug_plasticity": {"desc": "Print Plasticity Debug Info", "cat": "Plasticity"},

    # [Light/Visuals]
    "light_main_diffuse": {"desc": "Main Light Diffuse Vector", "cat": "Light/Visuals"},
    "light_main_ambient": {"desc": "Main Light Ambient Vector", "cat": "Light/Visuals"},
    "light_sub_diffuse": {"desc": "Sub Light Diffuse Vector", "cat": "Light/Visuals"},
    "light_head_ambient": {"desc": "Head Light Ambient Vector", "cat": "Light/Visuals"},
    "light_head_diffuse": {"desc": "Head Light Diffuse Vector", "cat": "Light/Visuals"},

    # [Air Fluidics]
    "enable_air_drag": {"desc": "Enable Air Drag Force", "cat": "Air Fluidics"},
    "air_density": {"desc": "Air Density (kg/m^3)", "cat": "Air Fluidics"},
    "air_viscosity": {"desc": "Air Viscosity (Pa*s)", "cat": "Air Fluidics"},
    "air_cd_drag": {"desc": "Air Drag Coefficient", "cat": "Air Fluidics"},
    "air_cd_viscous": {"desc": "Air Viscous Drag Coefficient", "cat": "Air Fluidics"},
    "air_coef_squeeze": {"desc": "Air Squeeze Film Coefficient", "cat": "Air Fluidics"},
    "air_squeeze_hmax": {"desc": "Squeeze Film Maximum Gap (m)", "cat": "Air Fluidics"},
    "air_squeeze_hmin": {"desc": "Squeeze Film Minimum Gap (m)", "cat": "Air Fluidics"},
    "enable_air_squeeze": {"desc": "Enable Squeeze Film Damping", "cat": "Air Fluidics"},
}

class VisualSchematicWidget(QtWidgets.QWidget):
    """[WHTOOLS] 박스 및 부품 크기 비율을 가시화하는 2D 스키매틱 위젯"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(250, 110)  # [WHTOOLS] 박스와 SET 크기 preview 영역 높이를 200 -> 110으로 대폭 축소
        self.config = {}

    def update_config(self, config):
        self.config = config
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        # 배경
        painter.fillRect(self.rect(), QtGui.QColor("#1e1e1e"))
        
        if not self.config: return
        
        bw = self.config.get("box_w", 1.0)
        bh = self.config.get("box_h", 0.8)
        aw = self.config.get("assy_w", 0.8)
        ah = self.config.get("assy_h", 0.6)
        
        # 캔버스 매핑
        rect = self.rect().adjusted(20, 16, -20, -16)  # 높이 축소에 따른 세로 여백 마진 피팅
        max_dim = max(bw, bh)
        scale = min(rect.width() / bw, rect.height() / bh) if max_dim > 0 else 1.0
        
        # 중심점
        cx, cy = rect.center().x(), rect.center().y()
        
        # 박스 그리기
        box_w = bw * scale
        box_h = bh * scale
        box_rect = QtCore.QRectF(cx - box_w/2, cy - box_h/2, box_w, box_h)
        
        painter.setPen(QtGui.QPen(QtGui.QColor("#8d6e63"), 2)) # Brown
        painter.setBrush(QtGui.QColor(141, 110, 99, 40))
        painter.drawRect(box_rect)
        
        # 박스 중심선 그리기 (가로/세로 점선)
        centerline_pen = QtGui.QPen(QtGui.QColor(141, 110, 99, 150), 1.0, QtCore.Qt.DashLine)
        painter.setPen(centerline_pen)
        painter.drawLine(QtCore.QLineF(box_rect.left(), cy, box_rect.right(), cy)) # 가로선
        painter.drawLine(QtCore.QLineF(cx, box_rect.top(), cx, box_rect.bottom())) # 세로선
        
        # SET/Chassis 그리기
        tv_w = aw * scale
        tv_h = ah * scale
        tv_rect = QtCore.QRectF(cx - tv_w/2, cy - tv_h/2, tv_w, tv_h)
        
        painter.setPen(QtGui.QPen(QtGui.QColor("#42a5f5"), 1.5)) # Blue
        painter.setBrush(QtGui.QColor(66, 165, 245, 80))
        painter.drawRect(tv_rect)
        
        # CoG (무게중심) 위치 마킹
        cog_list = [0.0, 0.0, 0.0]
        if "components_balance" in self.config and "target_cog" in self.config["components_balance"]:
            cog_list = self.config["components_balance"]["target_cog"]
        elif "target_cog" in self.config:
            cog_list = self.config["target_cog"]
            
        if len(cog_list) >= 2:
            cog_x, cog_y = cog_list[0], cog_list[1]
            # 물리 좌표계 (0,0) 원점을 cx, cy에 매핑하고 상향 Y축을 하향 화면좌표계에 맞게 cy - cog_y*scale 적용
            screen_cog_x = cx + cog_x * scale
            screen_cog_y = cy - cog_y * scale
            
            # 빨간색 붉은 빛의 선명한 CoG 마커 그리기
            painter.setPen(QtGui.QPen(QtGui.QColor("#ff1744"), 1.5))
            painter.setBrush(QtGui.QColor(255, 23, 68, 200))
            painter.drawEllipse(QtCore.QPointF(screen_cog_x, screen_cog_y), 4.5, 4.5)
            
            # CoG 라벨 텍스트 드로잉
            painter.setPen(QtGui.QColor("#ff3d00"))
            font = painter.font()
            font.setPointSize(7)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(QtCore.QPointF(screen_cog_x + 8, screen_cog_y + 3), f"CoG ({cog_x*1000:.1f}, {cog_y*1000:.1f} mm)")
        
        # 텍스트 정보 (기본 회색)
        painter.setPen(QtGui.QColor("#aaaaaa"))
        font = painter.font()
        font.setPointSize(8)
        font.setBold(False)
        painter.setFont(font)
        
        # Box 텍스트는 사각형 외부(상단 왼쪽)에 배치
        box_text_rect = QtCore.QRectF(box_rect.left(), box_rect.top() - 16, box_rect.width(), 16)
        painter.drawText(box_text_rect, Qt.AlignBottom | Qt.AlignLeft, f"Box: {bw:.2f}x{bh:.2f}")
        
        # SET 텍스트는 사각형 내부(상단 왼쪽)에 배치
        painter.drawText(tv_rect.adjusted(5, 5, -5, -5), Qt.AlignTop | Qt.AlignLeft, f"SET: {aw:.2f}x{ah:.2f}")

class XMLEditorDialog(QtWidgets.QDialog):
    """
    [WHTOOLS] MuJoCo XML 라이브 에디터 다이얼로그
    사용자가 직접 XML 수식을 수정하고 시뮬레이션에 즉각 반영할 수 있도록 합니다.
    """
    def __init__(self, parent=None, initial_xml="", model_path=None):
        super().__init__(parent)
        self.setWindowTitle("[WHTOOLS] Live XML Editor")
        self.setMinimumSize(900, 700)
        self.model_path = model_path
        self._xml_modified = False

        # 레이아웃 구성
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(15, 10, 15, 10)
        layout.setSpacing(5)

        # 상단 안내문
        info_label = QtWidgets.QLabel(
            "<b>MuJoCo XML Editor:</b> 수식을 직접 수정하고 [Apply & Reload]를 누르면 즉시 반영됩니다.<br>"
            "<small>Tip: 외부 에디터 버튼을 누르면 VS Code나 메모장 등 평소 쓰시는 도구로 편집할 수 있습니다.</small>"
        )
        info_label.setStyleSheet("margin-bottom: 5px;")
        info_label.setFixedHeight(info_label.sizeHint().height() + 5)
        layout.addWidget(info_label)

        # 메인 영역 (트리 + 에디터) - QSplitter 사용
        self.splitter = QtWidgets.QSplitter(Qt.Horizontal)
        
        # 1. 트리 뷰 (좌측)
        self.tree_view = QtWidgets.QTreeWidget()
        self.tree_view.setColumnCount(2)
        self.tree_view.setHeaderLabels(["Element", "Attributes"])
        self.tree_view.setColumnWidth(0, 180)
        
        self.tree_view.setStyleSheet(TREE_QSS)
        self.splitter.addWidget(self.tree_view)
        
        # 2. 텍스트 에디터 (우측)
        self.editor = QtWidgets.QPlainTextEdit()
        self.editor.setPlainText(initial_xml)
        
        # 고정폭 폰트 적용
        font = QtGui.QFont("Consolas", 9)
        if not font.fixedPitch():
            font = QtGui.QFont("Courier New", 9)
        self.editor.setFont(font)
        self.editor.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.splitter.addWidget(self.editor)
        
        # 스플리터 비율 설정
        self.splitter.setStretchFactor(0, 1) # Tree
        self.splitter.setStretchFactor(1, 3) # Editor
        
        layout.addWidget(self.splitter)

        # 트리 업데이트용 타이머 (입력 중 잦은 파싱 방지)
        self.tree_timer = QtCore.QTimer(self)
        self.tree_timer.setSingleShot(True)
        self.tree_timer.timeout.connect(self._update_tree)
        self.editor.textChanged.connect(lambda: self.tree_timer.start(500))
        
        # 트리 항목 변경 시 에디터 이동 연결
        self.tree_view.currentItemChanged.connect(self._on_tree_current_item_changed)
        
        # 초기 트리 생성
        self._update_tree()

        # 버튼 영역
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.setContentsMargins(3, 3, 3, 3)
        btn_layout.setSpacing(3)
        
        self.btn_external = QtWidgets.QPushButton(" 📝 Open in External Editor")
        self.btn_external.setMinimumHeight(40)
        self.btn_external.clicked.connect(self._on_open_external)
        
        self.btn_apply = QtWidgets.QPushButton(" 🚀 Apply & Reload")
        self.btn_apply.setMinimumHeight(40)
        self.btn_apply.setStyleSheet(f"background-color: {C_BTN_GREEN}; color: white; font-weight: bold;")
        self.btn_apply.clicked.connect(self.accept)

        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.setMinimumHeight(40)
        self.btn_cancel.clicked.connect(self.reject)

        btn_layout.addWidget(self.btn_external)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_apply)
        
        layout.addLayout(btn_layout)

    def get_xml_content(self):
        return self.editor.toPlainText()

    def _update_tree(self):
        """현재 에디터의 텍스트를 파싱하여 트리 구조를 업데이트합니다."""
        try:
            from lxml import etree as ET
            is_lxml = True
        except ImportError:
            import xml.etree.ElementTree as ET
            is_lxml = False
        
        xml_text = self.editor.toPlainText().strip()
        if not xml_text:
            self.tree_view.clear()
            return
            
        success = False
        try:
            # 1. XML 파싱
            if is_lxml:
                parser = ET.XMLParser(remove_blank_text=True, recover=True)
                try:
                    root = ET.fromstring(xml_text.encode('utf-8'), parser=parser)
                except:
                    root = ET.fromstring(xml_text, parser=parser)
            else:
                root = ET.fromstring(xml_text)
            
            # 2. 트리 업데이트
            self.tree_view.setUpdatesEnabled(False)
            self.tree_view.clear()
            
            self._node_count = 0
            self._max_nodes = 3000 
            
            self._dir_icon = self.style().standardIcon(QtWidgets.QStyle.SP_DirIcon)
            self._file_icon = self.style().standardIcon(QtWidgets.QStyle.SP_FileIcon)
            
            self._populate_tree_item(root, self.tree_view.invisibleRootItem())
            
            for i in range(self.tree_view.topLevelItemCount()):
                self.tree_view.topLevelItem(i).setExpanded(True)
            
            success = True
        except Exception:
            success = False
        finally:
            self.tree_view.setUpdatesEnabled(True)
            self._set_tree_style(success)

    def _set_tree_style(self, success: bool):
        """성공/실패 여부에 따른 트리 뷰 스타일을 동적으로 설정합니다."""
        border_color = C_BG_BTN if success else "#822"
        text_color = C_TEXT_TREE if success else C_STATUS_ERR
        self.tree_view.setStyleSheet(f"""
            QTreeWidget {{
                background-color: {C_BG_TABLE};
                color: {text_color};
                border: 1px solid {border_color};
                font-size: 9pt;
            }}
            QTreeWidget::item:hover {{ background-color: {C_BG_BTN}; }}
            QTreeWidget::item:selected {{ background-color: {C_SEL}; color: white; }}
            QHeaderView::section {{
                background-color: {C_BG_BTN};
                color: {C_TEXT_MUTED};
                padding: 4px;
                border: none;
                border-bottom: 1px solid {C_BG};
                font-weight: bold;
                font-size: 9pt;
            }}
        """)

    def _populate_tree_item(self, element, parent_item):
        """재귀적으로 XML 요소를 트리에 추가하며 라인 정보를 저장합니다."""
        if self._node_count >= self._max_nodes:
            if self._node_count == self._max_nodes:
                limit_item = QtWidgets.QTreeWidgetItem(parent_item)
                limit_item.setText(0, "... (Too many nodes, truncated)")
                self._node_count += 1
            return

        self._node_count += 1
        
        # 속성 문자열 생성
        attr_str = ", ".join([f"{k}: {v}" for k, v in element.attrib.items()])
        
        item = QtWidgets.QTreeWidgetItem(parent_item)
        item.setText(0, element.tag)
        item.setText(1, attr_str)
        
        # 라인 정보 저장 (lxml sourceline)
        if hasattr(element, 'sourceline') and element.sourceline is not None:
            item.setData(0, Qt.UserRole, element.sourceline)
        
        if len(element) > 0:
            item.setIcon(0, self._dir_icon)
            for child in element:
                self._populate_tree_item(child, item)
        else:
            item.setIcon(0, self._file_icon)

    def _on_tree_current_item_changed(self, current, previous):
        """트리 항목 변경 시 해당 라인으로 에디터 커서 이동."""
        if current is None:
            return
            
        line_no = current.data(0, Qt.UserRole)
        if line_no is None:
            return
            
        # QPlainTextEdit에서 해당 라인으로 이동 (lxml은 1-based)
        doc = self.editor.document()
        block = doc.findBlockByLineNumber(line_no - 1)
        
        if not block.isValid():
            return

        cursor = self.editor.textCursor()
        cursor.setPosition(block.position())
        
        # 라인 전체 선택하여 강조 (비주얼 피드백)
        cursor.movePosition(QtGui.QTextCursor.EndOfBlock, QtGui.QTextCursor.KeepAnchor)
        
        # 에디터 업데이트
        self.editor.setTextCursor(cursor)
        self.editor.ensureCursorVisible()
        
        # 트리 네비게이션 중에는 포커스를 뺏지 않음 (키보드 연속 이동 지원)
        # 단, 마우스로 직접 클릭한 경우나 트리에 포커스가 없는 경우 등에 대비해 필요한 경우에만 조절

    def _on_open_external(self):
        """임시 파일을 생성하고 시스템 기본 에디터로 엽니다."""
        import os
        import tempfile

        try:
            content = self.editor.toPlainText()
            with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False, encoding='utf-8') as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            self._external_tmp_path = tmp_path  # 다이얼로그 종료 시 삭제용

            os.startfile(tmp_path)

            QtWidgets.QMessageBox.information(
                self, "External Editor",
                "외부 에디터에서 파일을 수정하고 저장한 후,\n"
                "본 창에서 [Apply & Reload]를 눌러주세요.\n\n"
                f"임시 파일 경로: {tmp_path}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to open external editor: {e}")

def parse_size_str(size_str):
    """
    '1625 x 935 x 170' 형식의 문자열을 파싱하여 세 개의 실수로 반환합니다.
    """
    try:
        parts = [float(p.strip()) for p in size_str.replace('mm', '').split('x')]
        if len(parts) == 3:
            return parts[0], parts[1], parts[2]
    except Exception as e:
        pass
    return 0.0, 0.0, 0.0

def parse_float_list(s, expected_count):
    """쉼표 또는 공백으로 구분된 실수 목록을 파싱합니다. 실패 시 None 반환."""
    if not s or not s.strip():
        return None
    try:
        import re
        parts = [float(x) for x in re.split(r'[,\s]+', s.strip()) if x]
        if len(parts) == expected_count:
            return parts
    except Exception:
        pass
    return None

class SelectTVModelDialog(QtWidgets.QDialog):
    """
    [WHTOOLS] 삼성 TV 참조 모델 선택 다이얼로그 (PySide6 QTableWidget 기반)
    tv_ref_model_info.csv 데이터를 로드하여 정렬 가능한 현대적 스타일의 테이블을 제공합니다.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select TV Reference Model")
        self.resize(1050, 600)
        self.setStyleSheet(GLOBAL_QSS)
        self.selected_model = None
        self._init_ui()
        self._load_data()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)
        
        lbl = QtWidgets.QLabel("📺 삼성 TV 참조 모델 데이터베이스 (정렬 가능, 더블클릭 시 자동 선택)")
        layout.addWidget(lbl)
        
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(14)
        self.table.setHorizontalHeaderLabels([
            "Model Name", "Inch", "Pkg Size (mm)", "Pkg Mass (kg)",
            "Set w/ Stand Size", "Set w/ Stand Mass",
            "Set wo/ Stand Size", "Set wo/ Stand Mass", "Stand Base",
            "Cushion (kg)", "Chassis (kg)", "Opencell (kg)",
            "CoG (m)", "MoI (kg·m²)"
        ])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setSortingEnabled(True)
        self.table.doubleClicked.connect(self._on_double_clicked)
        layout.addWidget(self.table)
        
        btn_box = QtWidgets.QHBoxLayout()
        btn_apply = QtWidgets.QPushButton("Apply Selected Model")
        btn_apply.setStyleSheet(f"background-color: {C_BTN_BLUE3}; color: white; padding: 8px 15px; font-size: 9.5pt;")
        btn_apply.clicked.connect(self._on_apply)
        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_cancel.setStyleSheet(f"background-color: {C_BTN_RED}; color: white;")
        btn_cancel.clicked.connect(self.reject)
        
        btn_box.addWidget(btn_cancel)
        btn_box.addStretch()
        btn_box.addWidget(btn_apply)
        layout.addLayout(btn_box)

    def _load_data(self):
        # [WHTOOLS] __file__ 기준 상대 경로로 resources/tv_ref_model_info.csv 참조
        # run_drop_simulator/ -> TVPackageMotionSim/ -> resources/
        from pathlib import Path
        import csv
        import os
        csv_path = Path(__file__).parent.parent / "resources" / "tv_ref_model_info.csv"
        
        models = []
        if os.path.exists(csv_path):
            try:
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        models.append(row)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Load Error", f"CSV 로드 실패: {e}")
        
        if not models:
            return
            
        self.table.setRowCount(len(models))
        for row_idx, m in enumerate(models):
            self.table.setItem(row_idx, 0, QtWidgets.QTableWidgetItem(m.get('name', '')))
            self.table.setItem(row_idx, 1, QtWidgets.QTableWidgetItem(m.get('inch', '')))
            self.table.setItem(row_idx, 2, QtWidgets.QTableWidgetItem(m.get('pkg_size', '')))
            self.table.setItem(row_idx, 3, QtWidgets.QTableWidgetItem(m.get('pkg_m', '')))
            self.table.setItem(row_idx, 4, QtWidgets.QTableWidgetItem(m.get('set_w_std_size', '')))
            self.table.setItem(row_idx, 5, QtWidgets.QTableWidgetItem(m.get('set_w_std_m', '')))
            self.table.setItem(row_idx, 6, QtWidgets.QTableWidgetItem(m.get('set_wo_std_size', '')))
            self.table.setItem(row_idx, 7, QtWidgets.QTableWidgetItem(m.get('set_wo_std_m', '')))
            self.table.setItem(row_idx, 8, QtWidgets.QTableWidgetItem(m.get('stand_base', '')))
            self.table.setItem(row_idx, 9, QtWidgets.QTableWidgetItem(m.get('cushion_m', '')))
            self.table.setItem(row_idx, 10, QtWidgets.QTableWidgetItem(m.get('chassis_m', '')))
            self.table.setItem(row_idx, 11, QtWidgets.QTableWidgetItem(m.get('opencell_m', '')))
            self.table.setItem(row_idx, 12, QtWidgets.QTableWidgetItem(m.get('cog', '')))
            self.table.setItem(row_idx, 13, QtWidgets.QTableWidgetItem(m.get('moi', '')))
            
        self.table.resizeColumnsToContents()

    def _on_double_clicked(self, index):
        self._on_apply()

    def _on_apply(self):
        row = self.table.currentRow()
        if row < 0:
            QtWidgets.QMessageBox.information(self, "Selection Required", "모델을 먼저 선택하십시오.")
            return
            
        self.selected_model = {
            'name': self.table.item(row, 0).text(),
            'inch': int(self.table.item(row, 1).text()) if self.table.item(row, 1).text() else 0,
            'pkg_size': self.table.item(row, 2).text(),
            'pkg_m': float(self.table.item(row, 3).text()) if self.table.item(row, 3).text() else 0.0,
            'set_w_std_size': self.table.item(row, 4).text(),
            'set_w_std_m': float(self.table.item(row, 5).text()) if self.table.item(row, 5).text() else 0.0,
            'set_wo_std_size': self.table.item(row, 6).text(),
            'set_wo_std_m': float(self.table.item(row, 7).text()) if self.table.item(row, 7).text() else 0.0,
            'stand_base': self.table.item(row, 8).text()
        }
        self.accept()

class IstaSetupHelperDialog(QtWidgets.QDialog):
    """
    [WHTOOLS] ISTA 6-Amazon 규격 자가 진단 및 물리 낙하 포스처 계산용 헬퍼 UI.
    """
    def __init__(self, current_config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Size and ISTA 6-Amazon Setup Helper")
        self.setFixedSize(540, 680)  # SET 치수 입력부 추가에 따른 높이 정밀 확장
        self.parent_dialog = parent
        self.config = current_config.copy()
        
        self.setStyleSheet(GLOBAL_QSS)
        
        self._init_ui()
        self._load_config_values()
        self._update_all()
        
        # 부모 창의 정중앙에 배치하되, 화면 상단을 이탈하여 제목 표시줄이 숨겨지는 현상 예방
        if parent:
            parent_geo = parent.geometry()
            cx = parent_geo.x() + parent_geo.width() // 2
            cy = parent_geo.y() + parent_geo.height() // 2
            
            new_x = cx - self.width() // 2
            new_y = cy - self.height() // 2
            
            # 상단 제목 표시줄(Title bar) 가려짐 예방을 위해 Y축 최소값 45픽셀 보장 및 음수 방지
            if new_y < 45:
                new_y = 45
            if new_x < 10:
                new_x = 10
                
            self.move(new_x, new_y)

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)
        
        # 1. 스펙 입력 영역
        input_group = QtWidgets.QGroupBox("Package & SET Spec & Shipment Method")
        input_layout = QtWidgets.QGridLayout(input_group)
        input_layout.setSpacing(6)
        input_layout.setContentsMargins(8, 8, 8, 8)
        
        # [Row 0] Actions & Drop Mode (Pkg W 보다 위로 위치 상향 조정)
        self.btn_ref_model = QtWidgets.QPushButton("💾 Select Ref. Model")
        self.btn_ref_model.setStyleSheet(f"background-color: {C_BTN_BLUE}; color: white;")
        self.btn_ref_model.clicked.connect(self._on_select_ref_model)
        input_layout.addWidget(self.btn_ref_model, 0, 0, 1, 2)
        
        input_layout.addWidget(QtWidgets.QLabel("Mode:"), 0, 2)
        self.btn_group_mode = QtWidgets.QButtonGroup(self)
        self.radio_parcel = QtWidgets.QRadioButton("Parcel")
        self.radio_ltl = QtWidgets.QRadioButton("LTL")
        self.radio_custom = QtWidgets.QRadioButton("Custom")
        self.btn_group_mode.addButton(self.radio_parcel)
        self.btn_group_mode.addButton(self.radio_ltl)
        self.btn_group_mode.addButton(self.radio_custom)
        
        mode_lay = QtWidgets.QHBoxLayout()
        mode_lay.setSpacing(8)
        mode_lay.addWidget(self.radio_parcel)
        mode_lay.addWidget(self.radio_ltl)
        mode_lay.addWidget(self.radio_custom)
        input_layout.addLayout(mode_lay, 0, 3, 1, 3)
        
        self.radio_parcel.toggled.connect(self._on_input_changed)
        self.radio_ltl.toggled.connect(self._on_input_changed)
        self.radio_custom.toggled.connect(self._on_input_changed)

        # [Row 1] Package Dimensions (기존 Row 0에서 Row 1로 이동)
        lbl_pkg_w = QtWidgets.QLabel("Pkg W (m):")
        input_layout.addWidget(lbl_pkg_w, 1, 0)
        self.spin_w = QtWidgets.QDoubleSpinBox()
        self.spin_w.setDecimals(3)
        self.spin_w.setRange(0.01, 5.0)
        self.spin_w.setSingleStep(0.05)
        self.spin_w.setValue(self.config.get("box_w", 1.4))
        self.spin_w.valueChanged.connect(self._on_input_changed)
        self.spin_w.setFixedWidth(115)  # 스타일시트 패딩 대응 너비 확보
        input_layout.addWidget(self.spin_w, 1, 1)
        
        lbl_pkg_h = QtWidgets.QLabel("Pkg H (m):")
        input_layout.addWidget(lbl_pkg_h, 1, 2)
        self.spin_h = QtWidgets.QDoubleSpinBox()
        self.spin_h.setDecimals(3)
        self.spin_h.setRange(0.01, 5.0)
        self.spin_h.setSingleStep(0.05)
        self.spin_h.setValue(self.config.get("box_h", 0.85))
        self.spin_h.valueChanged.connect(self._on_input_changed)
        self.spin_h.setFixedWidth(115)
        input_layout.addWidget(self.spin_h, 1, 3)
        
        lbl_pkg_d = QtWidgets.QLabel("Pkg D (m):")
        input_layout.addWidget(lbl_pkg_d, 1, 4)
        self.spin_d = QtWidgets.QDoubleSpinBox()
        self.spin_d.setDecimals(3)
        self.spin_d.setRange(0.01, 5.0)
        self.spin_d.setSingleStep(0.05)
        self.spin_d.setValue(self.config.get("box_d", 0.15))
        self.spin_d.valueChanged.connect(self._on_input_changed)
        self.spin_d.setFixedWidth(115)
        input_layout.addWidget(self.spin_d, 1, 5)
        
        # [Row 2] SET (Chassis/OpenCell) Dimensions (기존 Row 1에서 Row 2로 이동)
        lbl_set_w = QtWidgets.QLabel("SET W (m):")
        input_layout.addWidget(lbl_set_w, 2, 0)
        self.spin_set_w = QtWidgets.QDoubleSpinBox()
        self.spin_set_w.setDecimals(3)
        self.spin_set_w.setRange(0.01, 5.0)
        self.spin_set_w.setSingleStep(0.05)
        self.spin_set_w.setValue(self.config.get("assy_w", 1.23))
        self.spin_set_w.valueChanged.connect(self._on_input_changed)
        self.spin_set_w.setFixedWidth(115)
        input_layout.addWidget(self.spin_set_w, 2, 1)
        
        lbl_set_h = QtWidgets.QLabel("SET H (m):")
        input_layout.addWidget(lbl_set_h, 2, 2)
        self.spin_set_h = QtWidgets.QDoubleSpinBox()
        self.spin_set_h.setDecimals(3)
        self.spin_set_h.setRange(0.01, 5.0)
        self.spin_set_h.setSingleStep(0.05)
        self.spin_set_h.setValue(self.config.get("assy_h", 0.71))
        self.spin_set_h.valueChanged.connect(self._on_input_changed)
        self.spin_set_h.setFixedWidth(115)
        input_layout.addWidget(self.spin_set_h, 2, 3)
        
        # SET Depth 초기값 계산: chassis_d + opencell_d + opencellcoh_d + cush_gap
        init_set_d = (
            self.config.get("chassis_d", 0.05) + 
            self.config.get("opencell_d", 0.005) + 
            self.config.get("opencellcoh_d", 0.002) + 
            self.config.get("cush_gap", 0.003)
        )
        lbl_set_d = QtWidgets.QLabel("SET D (m):")
        input_layout.addWidget(lbl_set_d, 2, 4)
        self.spin_set_d = QtWidgets.QDoubleSpinBox()
        self.spin_set_d.setDecimals(3)
        self.spin_set_d.setRange(0.01, 5.0)
        self.spin_set_d.setSingleStep(0.005)
        self.spin_set_d.setValue(init_set_d)
        self.spin_set_d.valueChanged.connect(self._on_input_changed)
        self.spin_set_d.setFixedWidth(115)
        input_layout.addWidget(self.spin_set_d, 2, 5)
        
        layout.addWidget(input_group)
        
        # 2-1. Custom Mode Drop Setup (중간 영역 배치)
        self.custom_setup_group = QtWidgets.QGroupBox("Drop Direction Custom")
        custom_setup_lay = QtWidgets.QGridLayout(self.custom_setup_group)
        custom_setup_lay.setSpacing(6)
        custom_setup_lay.setContentsMargins(8, 8, 8, 8)
        
        custom_setup_lay.addWidget(QtWidgets.QLabel("Drop Type:"), 0, 0)
        self.combo_custom_type = QtWidgets.QComboBox()
        self.combo_custom_type.addItems(["Face", "Edge", "Corner"])
        self.combo_custom_type.currentTextChanged.connect(self._on_custom_type_changed)
        custom_setup_lay.addWidget(self.combo_custom_type, 0, 1)
        
        custom_setup_lay.addWidget(QtWidgets.QLabel("Drop Direction:"), 0, 2)
        self.combo_custom_direction = QtWidgets.QComboBox()
        self.combo_custom_direction.currentTextChanged.connect(self._on_custom_dropdown_changed)
        custom_setup_lay.addWidget(self.combo_custom_direction, 0, 3, 1, 3)
        self._update_direction_combo_by_type("Face")  # 초기 기본값 로드
        
        custom_setup_lay.addWidget(QtWidgets.QLabel("Custom Drop Height (m):"), 1, 0, 1, 2)
        self.spin_custom_height = QtWidgets.QDoubleSpinBox()
        self.spin_custom_height.setRange(0.01, 5.0)
        self.spin_custom_height.setSingleStep(0.05)
        self.spin_custom_height.setDecimals(3)
        self.spin_custom_height.setValue(self.config.get("drop_height", 0.5))
        self.spin_custom_height.valueChanged.connect(self._on_custom_dropdown_changed)
        custom_setup_lay.addWidget(self.spin_custom_height, 1, 2, 1, 4)
        
        layout.addWidget(self.custom_setup_group)
        
        # ─── QStackedWidget 도입 (LTL/Parcel 정보 뷰와 Custom 모드 정보 뷰 전환) ───
        self.stacked_widget = QtWidgets.QStackedWidget()
        
        # 1) Page 0: LTL / Parcel 전용 정보 뷰
        self.page_ista = QtWidgets.QWidget()
        ista_lay = QtWidgets.QVBoxLayout(self.page_ista)
        ista_lay.setContentsMargins(0, 0, 0, 0)
        ista_lay.setSpacing(8)
        # LTL / Parcel 전용 정보 뷰 메인 레이아웃 여백 설정
        ista_lay.setContentsMargins(6, 6, 6, 6)
        ista_lay.setSpacing(10)
        
        # 2. 자가 진단 결과 (QGroupBox 제거 -> QLabel 소제목 및 알짜 위젯 직접 배치)
        self.diag_group = QtWidgets.QLabel("📝 ISTA Type Diagnosis & Recommendations")
        self.diag_group.setStyleSheet("font-weight: bold; color: #eceff4; font-size: 9.5pt; margin-top: 4px;")
        self.lbl_diag_result = QtWidgets.QLabel("Diagnosing...")
        self.lbl_diag_result.setFont(QFont("Consolas", 10))
        self.lbl_diag_result.setStyleSheet(f"color: {C_STATUS_TEXT_WARN}; background-color: {C_BG_INPUT}; padding: 8px; border-radius: 4px; border: 1px solid {C_BORDER_IN};")
        self.lbl_diag_result.setWordWrap(True)
        ista_lay.addWidget(self.diag_group)
        ista_lay.addWidget(self.lbl_diag_result)
        
        # 3. 면 번호 가이드 (QGroupBox 제거 -> QLabel 소제목 및 알짜 위젯 직접 배치)
        self.face_desc_box = QtWidgets.QLabel("📐 ISTA Standard Face Numbering Reference")
        self.face_desc_box.setStyleSheet("font-weight: bold; color: #eceff4; font-size: 9.5pt; margin-top: 4px;")
        self.lbl_face_desc = QtWidgets.QLabel("Loading Face mapping reference...")
        self.lbl_face_desc.setWordWrap(True)
        self.lbl_face_desc.setStyleSheet(f"color: {C_STATUS_TEXT_OK}; font-size: 9.5pt; font-family: Consolas; background-color: {C_BG_INPUT}; padding: 8px; border-radius: 4px; border: 1px solid {C_BORDER_IN};")
        ista_lay.addWidget(self.face_desc_box)
        ista_lay.addWidget(self.lbl_face_desc)
        
        # 4. 시퀀스 테이블 (QGroupBox 제거 -> QLabel 소제목 및 알짜 위젯 직접 배치)
        self.seq_group = QtWidgets.QLabel("📋 ISTA 6-Amazon Test Sequence Table")
        self.seq_group.setStyleSheet("font-weight: bold; color: #eceff4; font-size: 9.5pt; margin-top: 4px;")
        
        self.table_seq = QtWidgets.QTableWidget()
        self.table_seq.setColumnCount(4)
        self.table_seq.setHorizontalHeaderLabels([
            "Step", "Drop Type", "ISTA Target Point", "Height (mm)"
        ])
        self.table_seq.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table_seq.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table_seq.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table_seq.doubleClicked.connect(self._on_apply_and_sync)
        
        ista_lay.addWidget(self.seq_group)
        ista_lay.addWidget(self.table_seq)
        
        self.stacked_widget.addWidget(self.page_ista)
        
        # 2) Page 1: Custom 전용 정보 뷰
        self.page_custom = QtWidgets.QWidget()
        custom_lay = QtWidgets.QVBoxLayout(self.page_custom)
        custom_lay.setContentsMargins(0, 0, 0, 0)
        custom_lay.setSpacing(8)
        
        # Custom Mode 전용 안내 그룹
        self.custom_guide_group = QtWidgets.QGroupBox("Custom Mode Guide")
        custom_guide_layout = QtWidgets.QVBoxLayout(self.custom_guide_group)
        custom_guide_layout.setContentsMargins(8, 8, 8, 8)
        self.lbl_custom_guide = QtWidgets.QLabel(
            "<b>[Custom Drop Mode Selected]</b><br/>"
            "사용자가 임의로 조합한 면/모서리 방향과 낙하 높이로 단일 낙하 해석을 준비합니다.<br/>"
            "<small style='color: #888888;'>• Drop Type과 Drop Direction의 드롭다운을 조작하여 단일 낙하 자세를 정밀하게 구성합니다.</small>"
        )
        self.lbl_custom_guide.setWordWrap(True)
        self.lbl_custom_guide.setStyleSheet(f"color: {C_TEXT_DIM}; background-color: {C_BG_INPUT}; padding: 10px; border-radius: 4px; border: 1px solid {C_BORDER_IN}; line-height: 1.4;")
        custom_guide_layout.addWidget(self.lbl_custom_guide)
        custom_lay.addWidget(self.custom_guide_group)
        custom_lay.addStretch()
        
        self.stacked_widget.addWidget(self.page_custom)
        
        # StackedWidget 추가
        layout.addWidget(self.stacked_widget)
        
        # 5. 하단 액션 버튼 영역
        btn_box = QtWidgets.QHBoxLayout()
        btn_box.setSpacing(10)
        self.btn_apply = QtWidgets.QPushButton("🎯 Apply Selected Drop Posture to Main Panel")        
        self.btn_apply.clicked.connect(self._on_apply_and_sync)
        
        btn_cancel = QtWidgets.QPushButton("Cancel")        
        btn_cancel.clicked.connect(self.reject)
        
        btn_box.addWidget(btn_cancel)
        btn_box.addStretch()
        btn_box.addWidget(self.btn_apply)
        layout.addLayout(btn_box)

    def _load_config_values(self):
        # 복원 작업 중 시그널에 의한 _update_all 연쇄 호출을 전면 차단하여
        # 원본 데이터 유실 및 오버라이트를 원천 예방합니다.
        self.radio_parcel.blockSignals(True)
        self.radio_ltl.blockSignals(True)
        self.radio_custom.blockSignals(True)
        
        mode = self.config.get("drop_mode", "PARCEL").upper()
        if mode == "LTL":
            self.radio_ltl.setChecked(True)
        elif mode == "GENERAL":
            self.radio_custom.setChecked(True)
            dir_str = self.config.get("drop_direction", "front-bottom-left")
            parts = dir_str.split('-')
            
            # 신호 차단 후 안전하게 콤보박스 복원
            self.combo_custom_type.blockSignals(True)
            self.combo_custom_direction.blockSignals(True)
            
            if len(parts) == 3:
                self.combo_custom_type.setCurrentText("Corner")
                self._update_direction_combo_by_type("Corner")
                self.combo_custom_direction.setCurrentText(dir_str)
            elif len(parts) == 2:
                self.combo_custom_type.setCurrentText("Edge")
                self._update_direction_combo_by_type("Edge")
                self.combo_custom_direction.setCurrentText(dir_str)
            else:
                self.combo_custom_type.setCurrentText("Face")
                self._update_direction_combo_by_type("Face")
                self.combo_custom_direction.setCurrentText(dir_str)
                
            self.combo_custom_type.blockSignals(False)
            self.combo_custom_direction.blockSignals(False)
            
            self.spin_custom_height.setValue(self.config.get("drop_height", 0.5))
        else:
            self.radio_parcel.setChecked(True)
            
        self.radio_parcel.blockSignals(False)
        self.radio_ltl.blockSignals(False)
        self.radio_custom.blockSignals(False)

    def _on_select_ref_model(self):
        dlg = SelectTVModelDialog(self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted and dlg.selected_model:
            m = dlg.selected_model
            # 1) Package 치수 파싱 및 반영
            w_mm, h_mm, d_mm = parse_size_str(m['pkg_size'])
            if w_mm > 0:
                self.spin_w.setValue(w_mm / 1000.0)
                self.spin_h.setValue(h_mm / 1000.0)
                self.spin_d.setValue(d_mm / 1000.0)
                
            # 2) SET 치수 파싱 및 반영 (Stand 제외 순수 SET 크기)
            set_w_mm, set_h_mm, set_d_mm = parse_size_str(m['set_wo_std_size'])
            if set_w_mm > 0:
                self.spin_set_w.setValue(set_w_mm / 1000.0)
                self.spin_set_h.setValue(set_h_mm / 1000.0)
                self.spin_set_d.setValue(set_d_mm / 1000.0)
            
            # [WHTOOLS] 가장 최근 선택된 Ref. Model의 Package 질량을 config에 저장
            # ComponentBalanceDialog에서 "from Ref. Model" 힌트로 표시되어 편의성 제공
            if m.get('pkg_m', 0) > 0:
                self.config["last_ref_pkg_mass"] = float(m['pkg_m'])

            # 3) 컴포넌트 질량 반영 (cushion / chassis / opencell)
            try:
                v = float(m.get('cushion_m', '') or '')
                if v > 0:
                    self.spin_cushion_mass.setValue(v)
            except (ValueError, TypeError):
                pass
            try:
                v = float(m.get('chassis_m', '') or '')
                if v > 0:
                    self.spin_chassis_mass.setValue(v)
            except (ValueError, TypeError):
                pass
            try:
                v = float(m.get('opencell_m', '') or '')
                if v > 0:
                    self.spin_opencell_mass.setValue(v)
            except (ValueError, TypeError):
                pass

            # 4) CoG (3값: x y z, m 단위) → config["chassis_cog"]
            cog_vals = parse_float_list(m.get('cog', ''), 3)
            if cog_vals is not None:
                self.config["chassis_cog"] = cog_vals

            # 5) MoI (6값: Ixx Iyy Izz Ixy Ixz Iyz, kg·m²) → config["chassis_moi"]
            moi_vals = parse_float_list(m.get('moi', ''), 6)
            if moi_vals is not None:
                self.config["chassis_moi"] = moi_vals

            self.temp_selected_ref_model = m
            self._update_all()

    def _on_input_changed(self):
        self._update_all()

    def _on_custom_dropdown_changed(self):
        self._update_reporting()

    def _on_custom_type_changed(self, text):
        """Drop Type(Face, Edge, Corner) 변경 시 세부 방향 목록을 동적으로 변경합니다."""
        self._update_direction_combo_by_type(text)
        self._on_custom_dropdown_changed()

    def _update_direction_combo_by_type(self, type_text):
        """Drop Type 문자열에 대응하는 세부 방향 목록을 콤보박스에 로드합니다."""
        self.combo_custom_direction.blockSignals(True)
        self.combo_custom_direction.clear()
        
        if type_text == "Face":
            items = ["front", "back", "top", "bottom", "left", "right"]
        elif type_text == "Edge":
            items = [
                "front-bottom", "front-top", "front-left", "front-right",
                "back-bottom", "back-top", "back-left", "back-right",
                "bottom-left", "bottom-right", "top-left", "top-right"
            ]
        elif type_text == "Corner":
            items = [
                "front-bottom-left", "front-bottom-right",
                "front-top-left", "front-top-right",
                "back-bottom-left", "back-bottom-right",
                "back-top-left", "back-top-right"
            ]
        else:
            items = []
            
        self.combo_custom_direction.addItems(items)
        self.combo_custom_direction.blockSignals(False)

    def _update_reporting(self) -> float:
        """create_model을 사용하여 예상 질량 정보를 리포팅하고 총 질량을 반환합니다."""
        try:
            from run_discrete_builder import create_model
            import tempfile
            import os
            
            # Custom 모드인 경우 실시간으로 drop_direction과 drop_height를 임시 반영
            if self.radio_custom.isChecked():
                self.config["drop_direction"] = self.combo_custom_direction.currentText()
                self.config["drop_height"] = self.spin_custom_height.value()
                self.config["drop_mode"] = "GENERAL"
                self.config["initial_tilt_deg"] = 0.0
                self.config["initial_tilt_azimuth_deg"] = 0.0
                
            self.config["box_w"] = self.spin_w.value()
            self.config["box_h"] = self.spin_h.value()
            self.config["box_d"] = self.spin_d.value()

            # [WHTOOLS] SET 치수 반영 및 Chassis Thickness(chassis_d) 계산 공식 연동
            self.config["assy_w"] = self.spin_set_w.value()
            self.config["assy_h"] = self.spin_set_h.value()
            
            opencell_d = self.config.get("opencell_d", 0.005)
            opencellcoh_d = self.config.get("opencellcoh_d", 0.002)
            cush_gap = self.config.get("cush_gap", 0.003)
            
            calculated_chassis_d = self.spin_set_d.value() - (opencell_d + opencellcoh_d + cush_gap)
            self.config["chassis_d"] = max(0.001, calculated_chassis_d)

            with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp:
                tmp_path = tmp.name
            
            _, mass, cog, moi, details = create_model(tmp_path, config=self.config, logger=lambda x: None)
            os.unlink(tmp_path)
            
            return mass
        except Exception as e:
            return self.config.get("total_mass", 12.0)

    def _update_all(self):
        from whts_ista_helper import ISTA6ASimulator, IstaFaceMapper
        
        mass = self._update_reporting()
        
        is_custom = self.radio_custom.isChecked()
        is_ltl = self.radio_ltl.isChecked()
        is_parcel = self.radio_parcel.isChecked()
        
        # Custom 설정 위젯은 Custom 모드일 때만 노출
        self.custom_setup_group.setVisible(is_custom)
        
        # 하단 정보 전환 스택 위젯 인덱스 스위칭
        if is_custom:
            self.stacked_widget.setCurrentIndex(1)
            return
        else:
            self.stacked_widget.setCurrentIndex(0)
            
        w = self.spin_w.value()
        h = self.spin_h.value()
        d = self.spin_d.value()
        
        shipment = "LTL" if is_ltl else "Parcel"
        handling = "Standard"
        product = "TV/Monitor" if is_ltl else "General"
        
        sim = ISTA6ASimulator()
        seq, type_code = sim.generate_test_sequence(
            mass_kg=mass, width_mm=w * 1000.0, depth_mm=d * 1000.0, height_mm=h * 1000.0,
            shipment_method=shipment, handling_method=handling, product_type=product
        )
        _, reason = sim.determine_ista_type(
            mass_kg=mass, width_mm=w * 1000.0, depth_mm=d * 1000.0, height_mm=h * 1000.0,
            shipment_method=shipment, handling_method=handling, product_type=product
        )
        
        self.lbl_diag_result.setText(f"<b>Diagnosed Code:</b> Type {type_code}<br/><b>Reason:</b> {reason}")
        
        mapper = IstaFaceMapper(is_ltl=is_ltl)
        if is_ltl:
            face_str = (
                "<b>[LTL Mode Face Numbering Mapping Guide]</b><br/>"
                "Face 1: Top (+Y) | Face 2: Back (-Z) | Face 3: Bottom (-Y) | "
                "Face 4: Front (+Z) | Face 5: Right (+X) | Face 6: Left (-X)"
            )
        else:
            face_str = (
                "<b>[Parcel Mode Face Numbering Mapping Guide]</b><br/>"
                "Face 1: Top (+Y) | Face 2: Bottom (-Y) | Face 3: Right (+X) | "
                "Face 4: Left (-X) | Face 5: Front (+Z) | Face 6: Back (-Z)"
            )
        self.lbl_face_desc.setText(face_str)
        
        self.table_seq.setRowCount(len(seq))
        for idx, step in enumerate(seq):
            self.table_seq.setItem(idx, 0, QtWidgets.QTableWidgetItem(str(step['num'])))
            self.table_seq.setItem(idx, 1, QtWidgets.QTableWidgetItem(step['type'].upper()))
            self.table_seq.setItem(idx, 2, QtWidgets.QTableWidgetItem(step['name']))
            
            h_val = step['height']
            if step['type'] in ['rot_edge', 'rot_corner']:
                h_disp = h_val * 1000.0
            else:
                h_disp = h_val
            
            self.table_seq.setItem(idx, 3, QtWidgets.QTableWidgetItem(f"{h_disp:.0f}"))
            
        self.table_seq.resizeColumnsToContents()
        self.current_generated_seq = seq

    def _convert_to_ista_direction_name(self, direction_str, is_ltl):
        """
        기존 방향 좌표계 문자열(예: 'front-bottom-left', 'front-bottom', 'bottom')을
        ISTA 면 번호 맵핑 시스템(예: 'Corner 3-4-6', 'Edge 2-3', 'Face 3') 포맷으로 정확하게 변환합니다.
        """
        if not direction_str:
            return direction_str
            
        # LTL 및 Parcel 면 번호 규격 가이드 맵핑
        if is_ltl:
            mapping = {
                "top": 1,
                "back": 2,
                "bottom": 3,
                "front": 4,
                "right": 5,
                "left": 6
            }
        else:
            mapping = {
                "top": 1,
                "bottom": 2,
                "right": 3,
                "left": 4,
                "front": 5,
                "back": 6
            }
            
        parts = direction_str.lower().split('-')
        nums = []
        for p in parts:
            if p in mapping:
                nums.append(mapping[p])
                
        nums.sort()
        
        if len(nums) == 1:
            return f"Face {nums[0]}"
        elif len(nums) == 2:
            return f"Edge {nums[0]}-{nums[1]}"
        elif len(nums) == 3:
            return f"Corner {nums[0]}-{nums[1]}-{nums[2]}"
            
        return direction_str  # 맵핑 매칭 불가 시 fallback

    def _on_apply_and_sync(self):
        is_ltl = self.radio_ltl.isChecked()
        if self.radio_custom.isChecked():
            direction_str = self.combo_custom_direction.currentText()
            # Custom일지라도 맵핑 가능한 좌표 조합명이면 LTL/Parcel 표준 넘버링 명칭으로 통일
            mapped_direction = self._convert_to_ista_direction_name(direction_str, is_ltl)
            
            self.config["drop_direction"] = mapped_direction
            self.config["drop_height"] = self.spin_custom_height.value()
            self.config["drop_mode"] = "GENERAL"
            self.config["initial_tilt_deg"] = 0.0
            self.config["initial_tilt_azimuth_deg"] = 0.0
        else:
            row = self.table_seq.currentRow()
            if row < 0:
                QtWidgets.QMessageBox.information(self, "Selection Required", "시험할 시퀀스 낙하 단계를 선택하십시오.")
                return
                
            step = self.current_generated_seq[row]
            self.config["drop_mode"] = "LTL" if is_ltl else "PARCEL"
            
            # [WHTOOLS] step['direction']을 LTL/Parcel 모드에 맞춰 Face 3, Edge 2-3, Corner 3-4-6 등으로 정확히 변환하여 전달
            mapped_direction = self._convert_to_ista_direction_name(step['direction'], is_ltl)
            self.config["drop_direction"] = mapped_direction
            
            h_val = step['height']
            if step['type'] in ['rot_edge', 'rot_corner']:
                self.config["drop_height"] = h_val
            else:
                self.config["drop_height"] = h_val / 1000.0
                
            self.config["initial_tilt_deg"] = float(step['tilt_lat'])
            self.config["initial_tilt_azimuth_deg"] = float(step['tilt_az'])

        self.config["box_w"] = self.spin_w.value()
        self.config["box_h"] = self.spin_h.value()
        self.config["box_d"] = self.spin_d.value()
        
        # [WHTOOLS] SET 치수 및 chassis_d 계산 공식 적용하여 config 갱신
        self.config["assy_w"] = self.spin_set_w.value()
        self.config["assy_h"] = self.spin_set_h.value()
        
        opencell_d = self.config.get("opencell_d", 0.005)
        opencellcoh_d = self.config.get("opencellcoh_d", 0.002)
        cush_gap = self.config.get("cush_gap", 0.003)
        
        calculated_chassis_d = self.spin_set_d.value() - (opencell_d + opencellcoh_d + cush_gap)
        self.config["chassis_d"] = max(0.001, calculated_chassis_d)
                
        if self.parent_dialog:
            self.parent_dialog.config.update(self.config)
            self.parent_dialog._populate_config_tree()
            self.parent_dialog.schematic.update_config(self.parent_dialog.config)
            
            # [WHTOOLS BUG-FIX] combo_ista.setCurrentText() 호출 시 _on_ista_changed 시그널이 발화되어
            # edit_direction을 기본값으로 덮어쓰는 것을 방지하기 위해 blockSignals 처리
            self.parent_dialog.combo_ista.blockSignals(True)
            self.parent_dialog.combo_ista.setCurrentText(self.config["drop_mode"])
            self.parent_dialog.combo_ista.blockSignals(False)
            
            # 시그널 차단 후 올바른 방향 값 설정 (이 순서가 보장되어야 함)
            self.parent_dialog.edit_direction.setText(self.config["drop_direction"])
            self.parent_dialog.config["drop_direction"] = self.config["drop_direction"]
            self.parent_dialog.spin_height.setValue(self.config["drop_height"])
            self.parent_dialog.spin_lat.setValue(int(self.config["initial_tilt_deg"]))
            self.parent_dialog.spin_azimuth.setValue(int(self.config["initial_tilt_azimuth_deg"]))
            self.parent_dialog.schematic.update_config(self.parent_dialog.config)
            
        self.accept()

class ComponentBalanceDialog(QtWidgets.QDialog):
    """
    [WHTOOLS] Assembly Inertia Correction Dialog
    목표 질량/CoG/MoI를 분석적으로 계산하여 MuJoCo <inertial fullinertia> 가상 바디로
    정확히 보정합니다. 물리적 보조 질량 배치 없이 어떤 목표 관성도 정확히 달성합니다.
    """
    def __init__(self, parent=None, config=None):
        super().__init__(parent)
        self.config = config.copy() if config else {}
        self.parent_dialog = parent
        self.setWindowTitle("⚖️ [WHTOOLS] Assembly Inertia Correction")
        self.setMinimumSize(820, 560)
        self.resize(860, 580)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowMinMaxButtonsHint)
        self.setSizeGripEnabled(True)

        if "components_balance" not in self.config:
            self.config["components_balance"] = {
                "target_mass": 42.2,
                "target_inertia": [3.0, 8.0, 14.0, 0.1, 0.1, 0.1],
                "target_cog": [0.001, 0.007, 0.010],
            }

        self.inertia_correction = self.config.get("inertia_correction", None)

        self.setStyleSheet(GLOBAL_QSS)
        self._init_ui()
        self._calculate_delta_inertia()

    def _init_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(12, 12, 12, 12)

        specs_group = QtWidgets.QGroupBox("🎯 Target Specifications")
        specs_grid = QtWidgets.QGridLayout(specs_group)
        specs_grid.setSpacing(6)

        bal_cfg = self.config["components_balance"]

        # Target Mass 행: SpinBox + "from Ref. Model" 힌트 레이블
        specs_grid.addWidget(QtWidgets.QLabel("⚖️ Target Mass (kg):"), 0, 0)
        mass_lay = QtWidgets.QHBoxLayout()
        self.spin_target_mass = QtWidgets.QDoubleSpinBox()
        self.spin_target_mass.setRange(1.0, 500.0)
        self.spin_target_mass.setSingleStep(0.1)
        self.spin_target_mass.setValue(bal_cfg.get("target_mass", 42.2))
        mass_lay.addWidget(self.spin_target_mass)
        
        # [WHTOOLS] 가장 최근 Ref. Model의 Package 질량을 힌트로 표시
        ref_pkg_mass = self.config.get("last_ref_pkg_mass", None)
        self.lbl_ref_mass = QtWidgets.QLabel()
        if ref_pkg_mass is not None:
            self.lbl_ref_mass.setText(f"📌 from Ref. Model: <b>{ref_pkg_mass:.1f} kg</b>")
            self.lbl_ref_mass.setTextFormat(QtCore.Qt.RichText)
            self.lbl_ref_mass.setStyleSheet("color: #78dce8; font-size: 9pt; margin-left: 6px;")
        else:
            self.lbl_ref_mass.setText("📌 from Ref. Model: —")
            self.lbl_ref_mass.setStyleSheet("color: #808080; font-size: 9pt; margin-left: 6px;")
        mass_lay.addWidget(self.lbl_ref_mass)
        mass_lay.addStretch()
        specs_grid.addLayout(mass_lay, 0, 1)

        specs_grid.addWidget(QtWidgets.QLabel("📍 Target CoG X/Y/Z (m):"), 1, 0)
        cog_lay = QtWidgets.QHBoxLayout()
        cog_vals = bal_cfg.get("target_cog", [0.001, 0.007, 0.010])
        self.spin_cog_x = QtWidgets.QDoubleSpinBox(); self.spin_cog_x.setRange(-2.0, 2.0); self.spin_cog_x.setSingleStep(0.001); self.spin_cog_x.setDecimals(4); self.spin_cog_x.setValue(cog_vals[0])
        self.spin_cog_y = QtWidgets.QDoubleSpinBox(); self.spin_cog_y.setRange(-2.0, 2.0); self.spin_cog_y.setSingleStep(0.001); self.spin_cog_y.setDecimals(4); self.spin_cog_y.setValue(cog_vals[1])
        self.spin_cog_z = QtWidgets.QDoubleSpinBox(); self.spin_cog_z.setRange(-1.0, 1.0); self.spin_cog_z.setSingleStep(0.001); self.spin_cog_z.setDecimals(4); self.spin_cog_z.setValue(cog_vals[2])
        cog_lay.addWidget(self.spin_cog_x); cog_lay.addWidget(self.spin_cog_y); cog_lay.addWidget(self.spin_cog_z)
        specs_grid.addLayout(cog_lay, 1, 1)

        specs_grid.addWidget(QtWidgets.QLabel("🌀 Target MoI Diagonal (Ixx, Iyy, Izz):"), 2, 0)
        moi_diag_lay = QtWidgets.QHBoxLayout()
        moi_vals = bal_cfg.get("target_inertia", [3.0, 8.0, 14.0, 0.1, 0.1, 0.1])
        self.spin_moi_xx = QtWidgets.QDoubleSpinBox(); self.spin_moi_xx.setRange(0.01, 500.0); self.spin_moi_xx.setDecimals(4); self.spin_moi_xx.setValue(moi_vals[0])
        self.spin_moi_yy = QtWidgets.QDoubleSpinBox(); self.spin_moi_yy.setRange(0.01, 500.0); self.spin_moi_yy.setDecimals(4); self.spin_moi_yy.setValue(moi_vals[1])
        self.spin_moi_zz = QtWidgets.QDoubleSpinBox(); self.spin_moi_zz.setRange(0.01, 500.0); self.spin_moi_zz.setDecimals(4); self.spin_moi_zz.setValue(moi_vals[2])
        moi_diag_lay.addWidget(self.spin_moi_xx); moi_diag_lay.addWidget(self.spin_moi_yy); moi_diag_lay.addWidget(self.spin_moi_zz)
        
        # [WHTOOLS] 균일 직육면체 근사치 계산 버튼
        self.btn_guess_moi = QtWidgets.QPushButton("🎲 Guess Uniform")
        self.btn_guess_moi.setToolTip(
            "Package 크기(W×H×D)와 SET 크기, Target Mass를 기반으로\n"
            "균일 밀도 직육면체로 근사한 Ixx, Iyy, Izz를 자동 계산하여 적용합니다.\n"
            "I_xx = m/12*(h²+d²), I_yy = m/12*(w²+d²), I_zz = m/12*(w²+h²)"
        )
        self.btn_guess_moi.setStyleSheet(f"background-color: {C_BTN_INDIGO}; color: white; padding: 3px 8px; font-size: 9pt;")
        self.btn_guess_moi.clicked.connect(self._on_guess_uniform_moi)
        moi_diag_lay.addWidget(self.btn_guess_moi)
        specs_grid.addLayout(moi_diag_lay, 2, 1)

        specs_grid.addWidget(QtWidgets.QLabel("🌀 Target MoI Product (Ixy, Ixz, Iyz):"), 3, 0)
        moi_prod_lay = QtWidgets.QHBoxLayout()
        self.spin_moi_xy = QtWidgets.QDoubleSpinBox(); self.spin_moi_xy.setRange(-100.0, 100.0); self.spin_moi_xy.setDecimals(4); self.spin_moi_xy.setValue(moi_vals[3])
        self.spin_moi_xz = QtWidgets.QDoubleSpinBox(); self.spin_moi_xz.setRange(-100.0, 100.0); self.spin_moi_xz.setDecimals(4); self.spin_moi_xz.setValue(moi_vals[4])
        self.spin_moi_yz = QtWidgets.QDoubleSpinBox(); self.spin_moi_yz.setRange(-100.0, 100.0); self.spin_moi_yz.setDecimals(4); self.spin_moi_yz.setValue(moi_vals[5])
        moi_prod_lay.addWidget(self.spin_moi_xy); moi_prod_lay.addWidget(self.spin_moi_xz); moi_prod_lay.addWidget(self.spin_moi_yz)
        specs_grid.addLayout(moi_prod_lay, 3, 1)

        # 4) Component Masses (Cushion, Opencell, Chassis) 추가
        specs_grid.addWidget(QtWidgets.QLabel("📦 Component Masses (kg):"), 4, 0)
        comp_mass_lay = QtWidgets.QHBoxLayout()
        comp_mass_lay.setSpacing(8)
        
        lbl_cush = QtWidgets.QLabel("Cushion:")
        lbl_cush.setFont(QFont("Consolas", 9))
        self.spin_cushion_mass = QtWidgets.QDoubleSpinBox()
        self.spin_cushion_mass.setRange(0.01, 100.0)
        self.spin_cushion_mass.setSingleStep(0.1)
        self.spin_cushion_mass.setDecimals(2)
        self.spin_cushion_mass.setFixedWidth(80)
        self.spin_cushion_mass.setAlignment(Qt.AlignRight)
        
        lbl_open = QtWidgets.QLabel("Opencell:")
        lbl_open.setFont(QFont("Consolas", 9))
        self.spin_opencell_mass = QtWidgets.QDoubleSpinBox()
        self.spin_opencell_mass.setRange(0.01, 100.0)
        self.spin_opencell_mass.setSingleStep(0.1)
        self.spin_opencell_mass.setDecimals(2)
        self.spin_opencell_mass.setFixedWidth(80)
        self.spin_opencell_mass.setAlignment(Qt.AlignRight)
        
        lbl_chas = QtWidgets.QLabel("Chassis:")
        lbl_chas.setFont(QFont("Consolas", 9))
        self.spin_chassis_mass = QtWidgets.QDoubleSpinBox()
        self.spin_chassis_mass.setRange(0.01, 200.0)
        self.spin_chassis_mass.setSingleStep(0.1)
        self.spin_chassis_mass.setDecimals(2)
        self.spin_chassis_mass.setFixedWidth(80)
        self.spin_chassis_mass.setAlignment(Qt.AlignRight)
        
        # 초기값 동기화
        comp_cfg = self.config.get("components", {})
        self.spin_cushion_mass.setValue(comp_cfg.get("cushion", {}).get("mass", 3.0))
        self.spin_opencell_mass.setValue(comp_cfg.get("opencell", {}).get("mass", 5.0))
        self.spin_chassis_mass.setValue(comp_cfg.get("chassis", {}).get("mass", 10.0))
        
        comp_mass_lay.addWidget(lbl_cush)
        comp_mass_lay.addWidget(self.spin_cushion_mass)
        comp_mass_lay.addSpacing(10)
        comp_mass_lay.addWidget(lbl_open)
        comp_mass_lay.addWidget(self.spin_opencell_mass)
        comp_mass_lay.addSpacing(10)
        comp_mass_lay.addWidget(lbl_chas)
        comp_mass_lay.addWidget(self.spin_chassis_mass)
        comp_mass_lay.addStretch()
        
        specs_grid.addLayout(comp_mass_lay, 4, 1)

        main_layout.addWidget(specs_group)

        results_group = QtWidgets.QGroupBox("📊 Inertia Correction Result  (Base → Delta → Target)")
        results_vlay = QtWidgets.QVBoxLayout(results_group)

        self.table_results = QtWidgets.QTableWidget()
        self.table_results.setColumnCount(5)
        self.table_results.setHorizontalHeaderLabels(["Metric", "Base", "Target", "Delta (Correction)", "Feasibility"])
        self.table_results.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.table_results.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        results_vlay.addWidget(self.table_results)

        # [WHTOOLS] 물리적 관성 성립 가능성 진단 및 실시간 보정 필요성 가이드 안내창 신설
        self.lbl_guide = QtWidgets.QLabel()
        self.lbl_guide.setTextFormat(QtCore.Qt.RichText)
        self.lbl_guide.setWordWrap(True)
        self.lbl_guide.setFont(QtGui.QFont("Segoe UI", 9))
        self.lbl_guide.setStyleSheet("background-color: #242424; border: 1px solid #444444; border-radius: 4px; padding: 10px;")
        self.lbl_guide.setText("💡 <b style='color: #ffd866;'>안내:</b> Calculate 버튼을 클릭하시면 보정 관성 텐서의 고유치(Eigenvalues) 분석을 거쳐 물리적 실현 가능성 및 보정 가이드가 여기에 친절하게 출력됩니다.")
        results_vlay.addWidget(self.lbl_guide)

        btn_calc_lay = QtWidgets.QHBoxLayout()
        self.btn_calc = QtWidgets.QPushButton("⚡ Calculate Inertia Correction")
        self.btn_calc.setStyleSheet(f"background-color: {C_BTN_GREEN2}; color: white; font-weight: bold;")
        self.btn_calc.clicked.connect(lambda: self._calculate_delta_inertia(show_popup=True))
        btn_calc_lay.addStretch()
        btn_calc_lay.addWidget(self.btn_calc)
        results_vlay.addLayout(btn_calc_lay)

        main_layout.addWidget(results_group)

        bottom_btn_lay = QtWidgets.QHBoxLayout()
        self.btn_apply = QtWidgets.QPushButton("💾 Apply to Configuration")
        self.btn_apply.setStyleSheet(f"background-color: {C_BTN_GREEN}; font-weight: bold; min-width: 150px;")
        self.btn_apply.clicked.connect(self.on_apply_clicked)
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        bottom_btn_lay.addWidget(self.btn_cancel)
        bottom_btn_lay.addStretch()
        bottom_btn_lay.addWidget(self.btn_apply)
        main_layout.addLayout(bottom_btn_lay)

    def _on_guess_uniform_moi(self):
        """
        [WHTOOLS] Package/SET 크기와 Target Mass를 기반으로 균일 밀도 직육면체 MoI 대각 성분을 추정합니다.

        추정 방법:
          - Package 크기(box_w, box_h, box_d)와 SET 크기(assy_w, assy_h, chassis_d)를 동시에 고려
          - 두 크기의 가중 평균 (Package 70%, SET 30%)으로 유효 치수 계산
          - 균일 직육면체 관성 공식 적용:
              I_xx = m/12 * (h² + d²)
              I_yy = m/12 * (w² + d²)
              I_zz = m/12 * (w² + h²)
        """
        m = self.spin_target_mass.value()

        # Package 크기 (m 단위)
        box_w = float(self.config.get("box_w", 1.8))
        box_h = float(self.config.get("box_h", 0.12))
        box_d = float(self.config.get("box_d", 1.05))

        # SET 크기 (m 단위) - chassis_d 대신 box_d와 근사
        assy_w = float(self.config.get("assy_w", box_w * 0.95))
        assy_h = float(self.config.get("assy_h", box_h * 0.85))
        assy_d = float(self.config.get("chassis_d", box_d * 0.90))

        # 가중 평균 유효 치수 (Package 70% + SET 30%)
        w = box_w * 0.70 + assy_w * 0.30
        h = box_h * 0.70 + assy_h * 0.30
        d = box_d * 0.70 + assy_d * 0.30

        # 균일 직육면체 관성 모멘트 공식 (단위: kg·m²)
        ixx = m / 12.0 * (h**2 + d**2)
        iyy = m / 12.0 * (w**2 + d**2)
        izz = m / 12.0 * (w**2 + h**2)

        # SpinBox에 적용
        self.spin_moi_xx.setValue(round(ixx, 4))
        self.spin_moi_yy.setValue(round(iyy, 4))
        self.spin_moi_zz.setValue(round(izz, 4))

        # 사용자 안내 메시지
        self.lbl_guide.setText(
            f"🎲 <b style='color: #ffd866;'>Guess Uniform MoI 적용 완료:</b><br/>"
            f"유효 치수 W={w*1000:.0f}mm, H={h*1000:.0f}mm, D={d*1000:.0f}mm (Pkg 70% + SET 30%)<br/>"
            f"• Ixx = {ixx:.4f} kg·m²,  Iyy = {iyy:.4f} kg·m²,  Izz = {izz:.4f} kg·m²<br/>"
            f"<span style='color: #aaaaaa;'>▶ 이 값은 균일 밀도 직육면체 근사입니다. "
            f"계산 후 Calculate로 정확한 보정치를 확인하세요.</span>"
        )
        self.lbl_guide.setStyleSheet(
            "background-color: #1f2230; border: 1px solid #6272a4; border-radius: 4px; padding: 10px;"
        )

    def _calculate_delta_inertia(self, show_popup=False):
        """Analytic delta-inertia computation. No optimization needed — result is exact."""
        try:
            t_mass = self.spin_target_mass.value()
            t_cog = np.array([self.spin_cog_x.value(), self.spin_cog_y.value(), self.spin_cog_z.value()])
            t_moi = np.array([
                self.spin_moi_xx.value(), self.spin_moi_yy.value(), self.spin_moi_zz.value(),
                self.spin_moi_xy.value(), self.spin_moi_xz.value(), self.spin_moi_yz.value()
            ])

            from run_discrete_builder.whtb_physics import _get_assembly_inertia_base
            m_base, c_base, i_base, _ = _get_assembly_inertia_base(self.config)

            m_delta = t_mass - m_base
            if abs(m_delta) < 1e-9:
                pos_delta = t_cog.copy()
            else:
                pos_delta = (t_cog * t_mass - c_base * m_base) / m_delta

            # Parallel axis: move i_base from c_base to t_cog
            d = t_cog - c_base
            i_base_at_tcog = np.zeros(6)
            i_base_at_tcog[:3] = i_base[:3] + m_base * np.array([d[1]**2+d[2]**2, d[0]**2+d[2]**2, d[0]**2+d[1]**2])
            i_base_at_tcog[3:6] = i_base[3:6] - m_base * np.array([d[0]*d[1], d[0]*d[2], d[1]*d[2]])

            i_delta_at_tcog = t_moi - i_base_at_tcog
            i_delta = i_delta_at_tcog.copy()
            if abs(m_delta) > 1e-9:
                dp = pos_delta - t_cog
                i_delta[0] -= m_delta * (dp[1]**2 + dp[2]**2)
                i_delta[1] -= m_delta * (dp[0]**2 + dp[2]**2)
                i_delta[2] -= m_delta * (dp[0]**2 + dp[1]**2)
                i_delta[3] += m_delta * dp[0] * dp[1]
                i_delta[4] += m_delta * dp[0] * dp[2]
                i_delta[5] += m_delta * dp[1] * dp[2]

            # [WHTOOLS] whtb_physics와 동일한 삼각부등식 및 고유치 보정 필터 실행
            from run_discrete_builder.whtb_physics import _clamp_inertia_triangle, _ensure_positive_eigenvalues
            i_delta_clamped = _clamp_inertia_triangle(i_delta, label="I_delta (UI Dialog)")
            i_delta_valid = _ensure_positive_eigenvalues(i_delta_clamped, label="I_delta (UI Dialog)")

            # 보정 전/후 고유치 정밀 진단
            I_mat_raw = np.array([
                [i_delta[0], -i_delta[3], -i_delta[4]],
                [-i_delta[3], i_delta[1], -i_delta[5]],
                [-i_delta[4], -i_delta[5], i_delta[2]]
            ])
            raw_eigs = np.linalg.eigvalsh(I_mat_raw)

            I_mat_valid = np.array([
                [i_delta_valid[0], -i_delta_valid[3], -i_delta_valid[4]],
                [-i_delta_valid[3], i_delta_valid[1], -i_delta_valid[5]],
                [-i_delta_valid[4], -i_delta_valid[5], i_delta_valid[2]]
            ])
            valid_eigs = np.linalg.eigvalsh(I_mat_valid)

            self.inertia_correction = {
                "m_delta": float(m_delta),
                "pos_delta": [float(pos_delta[0]), float(pos_delta[1]), float(pos_delta[2])],
                "I_delta": [float(v) for v in i_delta_valid],
            }

            # 테이블 결과 갱신 (고유치 분석 포함)
            self.update_table(m_base, c_base, i_base, t_mass, t_cog, t_moi, m_delta, pos_delta, i_delta_valid, raw_eigs, valid_eigs)

            # 실시간 물리 가이드 가시화 및 피드백 갱신
            has_raw_invalid = any(ev <= 1e-4 for ev in raw_eigs)
            if has_raw_invalid:
                guide_text = (
                    "⚠️ <b style='color: #ff5555;'>물리적 관성 한계 초과 감지 (보정 필요):</b><br/>"
                    "입력한 목표 관성(Target MoI)이 패키지 기본 형태 대비 너무 작거나 곱관성이 지나치게 큽니다.<br/>"
                    f"• <b>보정 전 고유치:</b> λ1={raw_eigs[0]:.5f}, λ2={raw_eigs[1]:.5f}, λ3={raw_eigs[2]:.5f} <span style='color: #ff5555;'>(0 이하 존재로 MuJoCo 충돌 유발)</span><br/>"
                    "⚡ <b>[시스템 자가 치유 조치 완료]:</b><br/>"
                    "시뮬레이터 크래시를 전면 방지하기 위해 <b>대각 관성 모멘트(Ixx, Iyy, Izz) 성분을 자동으로 안전선까지 Compensation(보정)</b> 하였습니다.<br/>"
                    f"• <b>보정 후 고유치:</b> λ1={valid_eigs[0]:.5f}, λ2={valid_eigs[1]:.5f}, λ3={valid_eigs[2]:.5f} <span style='color: #a9dc76;'>(안전하게 양의 고유치 충족)</span><br/>"
                    "💡 <b>가이드:</b> 자동 보정량을 최소화하고 싶으시다면 Target Specifications of MoI 대각성분(Ixx, Iyy, Izz)을 다소 늘리거나, "
                    "곱관성 모멘트(Ixy, Ixz, Iyz) 성분의 절대값을 0에 가깝게 조절해 주시기 바랍니다."
                )
                self.lbl_guide.setStyleSheet("background-color: #2b1f1f; border: 1px solid #ff5555; border-radius: 4px; padding: 10px;")
            else:
                guide_text = (
                    "✅ <b style='color: #a9dc76;'>물리적 관성 타당성 통과 (안정적인 상태):</b><br/>"
                    "계산된 보정 관성 텐서의 고유치가 모두 양수 조건을 충족합니다.<br/>"
                    f"• <b>현재 고유치:</b> λ1={valid_eigs[0]:.5f}, λ2={valid_eigs[1]:.5f}, λ3={valid_eigs[2]:.5f}<br/>"
                    "• 가상 바디 `<InertiaCorrection>`가 무조코 엔진에 정상 로딩되며, 보정 필요성 없이 목표 사양을 대변합니다."
                )
                self.lbl_guide.setStyleSheet("background-color: #1e261f; border: 1px solid #a9dc76; border-radius: 4px; padding: 10px;")

            self.lbl_guide.setText(guide_text)

            # [WHTOOLS] 사용자가 Calculate 버튼을 클릭하여 명시적으로 계산을 가동했을 때 결과 팝업을 제공 (오직 실패 시에만 경고 출력)
            if show_popup:
                if has_raw_invalid:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "⚠️ 물리적 관성 한계 초과 (보정 완료)",
                        "목표 사양으로 계산된 보정 관성 텐서가 물리적으로 성립할 수 없어 자동 보정이 수행되었습니다!\n\n"
                        f"• 보정 전 최소 고유치: {raw_eigs[0]:.5f}\n"
                        f"• 보정 후 최소 고유치: {valid_eigs[0]:.5f} (물리적 안전 확보)\n\n"
                        "시뮬레이션 크래시를 차단하기 위해 가상 바디의 대각 성분이 자동 보정되었습니다. 보정량을 줄이고자 하실 경우 MoI Diagonal 수치를 상승시키거나 MoI Product(곱관성) 수치를 0에 가깝게 조절하시는 것을 권장합니다."
                    )

            sep = "─" * 72
            print(f"\n{sep}")
            print(f"  [WHTOOLS] Inertia Correction (Delta-Inertia Approach)")
            print(sep)
            print(f"  m_delta   : {m_delta:+.6f} kg  at pos ({pos_delta[0]:.5f}, {pos_delta[1]:.5f}, {pos_delta[2]:.5f}) m")
            print(f"  I_delta   : Ixx={i_delta_valid[0]:+.6f}  Iyy={i_delta_valid[1]:+.6f}  Izz={i_delta_valid[2]:+.6f}")
            print(f"              Ixy={i_delta_valid[3]:+.6f}  Ixz={i_delta_valid[4]:+.6f}  Iyz={i_delta_valid[5]:+.6f}")
            print(f"  Eigenvals : λ1={valid_eigs[0]:.6f}  λ2={valid_eigs[1]:.6f}  λ3={valid_eigs[2]:.6f}")
            if has_raw_invalid:
                print(f"  ⚠ Auto-compensated by eigenvalues correction (Raw min eig: {raw_eigs[0]:.6f})")
            print(sep + "\n")
        except Exception as e:
            import traceback; traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Calculation Error", f"Failed to compute inertia correction:\n{e}")

    def _unused_optimization_engine_legacy(self):
        t_mass = self.spin_target_mass.value()
        t_cog = np.array([self.spin_cog_x.value(), self.spin_cog_y.value(), self.spin_cog_z.value()])
        t_moi = np.array([
            self.spin_moi_xx.value(), self.spin_moi_yy.value(), self.spin_moi_zz.value(),
            self.spin_moi_xy.value(), self.spin_moi_xz.value(), self.spin_moi_yz.value()
        ])
        count = self.spin_count.value()
        focus_slider_val = self.focus_slider.value()
        focus_moi = focus_slider_val / 100.0
        
        from run_discrete_builder.whtb_physics import _get_assembly_inertia_base
        temp_cfg = self.config.copy()
        temp_cfg["chassis_aux_masses"] = [m for m in temp_cfg.get("chassis_aux_masses", []) if not m.get("name", "").startswith("AutoBalance_")]
        temp_cfg["component_aux"] = {k: v for k, v in temp_cfg.get("component_aux", {}).items() if not k.startswith("AutoBalance_")}
        m_base, c_base, i_base, _ = _get_assembly_inertia_base(temp_cfg)
        
        bw = self.config.get("box_w", 2.0)
        bh = self.config.get("box_h", 1.4)
        bd = self.config.get("box_d", 0.25)
        limit_x, limit_y, limit_z = bw/2.0 * 2.0, bh/2.0 * 2.0, bd/2.0 * 2.0

        m_aux = t_mass - m_base
        if m_aux < 1e-4:
            m_aux = 1e-4

        from scipy.optimize import minimize

        pos_aux_init = (t_cog * t_mass - m_base * c_base) / m_aux

        dx_init, dy_init, dz_init = 0.1, 0.1, 0.05
        a_init, b_init, g_init = 0.0, 0.0, 0.0

        p0 = np.concatenate([pos_aux_init, [dx_init, dy_init, dz_init], [a_init, b_init, g_init]])

        bounds = [
            (-limit_x, limit_x), (-limit_y, limit_y), (-limit_z, limit_z),  # pos_aux: 박스 150%
            (0.001, limit_x),    (0.001, limit_y),    (0.001, limit_z),      # dx/dy/dz: 양수, 150% 이내
            (-0.95, 0.95), (-0.95, 0.95), (-0.95, 0.95)
        ]

        def get_masses_and_positions(p):
            pos_aux = p[0:3]
            dx, dy, dz = p[3:6]
            a, b, g = p[6:9]

            masses = []
            positions = []
            m_avg = m_aux / count

            if count <= 1:
                masses = [m_aux]
                positions = [pos_aux]
            elif count == 2:
                for sx in [-1, 1]:
                    masses.append(m_avg)
                    positions.append([pos_aux[0] + sx*dx, pos_aux[1], pos_aux[2]])
            elif count == 4:
                for sx in [-1, 1]:
                    for sy in [-1, 1]:
                        m_this = m_avg * (1.0 + a*sx*sy)
                        masses.append(max(m_this, 1e-6))
                        positions.append([pos_aux[0] + sx*dx, pos_aux[1] + sy*dy, pos_aux[2]])
            else:
                for sx in [-1, 1]:
                    for sy in [-1, 1]:
                        for sz in [-1, 1]:
                            m_this = m_avg * (1.0 + a*sx*sy + b*sx*sz + g*sy*sz)
                            masses.append(max(m_this, 1e-6))
                            positions.append([pos_aux[0] + sx*dx, pos_aux[1] + sy*dy, pos_aux[2] + sz*dz])

            m_sum = sum(masses)
            if m_sum > 0:
                masses = [m * m_aux / m_sum for m in masses]

            return masses, [np.array(pt, dtype=float) for pt in positions]
            
        # MoI 정규화 스케일: 대각 성분 평균 기준 (off-diagonal이 0에 가까울 때 수치 폭발 방지)
        moi_scale = max(np.abs(t_moi[:3]).mean(), 0.1)

        def objective(p):
            masses, positions = get_masses_and_positions(p)
            final_mass = m_base + sum(masses)
            final_cog = (m_base * c_base + sum(m * pt for m, pt in zip(masses, positions))) / final_mass

            d = final_cog - c_base
            i_base_moved = np.zeros(6)
            i_base_moved[:3] = i_base[:3] + m_base * np.array([d[1]**2 + d[2]**2, d[0]**2 + d[2]**2, d[0]**2 + d[1]**2])
            i_base_moved[3:6] = i_base[3:6] - m_base * np.array([d[0]*d[1], d[0]*d[2], d[1]*d[2]])

            i_aux = np.zeros(6)
            for m, pt in zip(masses, positions):
                dp = pt - final_cog
                i_aux[0] += m * (dp[1]**2 + dp[2]**2)
                i_aux[1] += m * (dp[0]**2 + dp[2]**2)
                i_aux[2] += m * (dp[0]**2 + dp[1]**2)
                i_aux[3] -= m * dp[0] * dp[1]
                i_aux[4] -= m * dp[0] * dp[2]
                i_aux[5] -= m * dp[1] * dp[2]

            final_moi = i_base_moved + i_aux

            # Mass 오차: 정규화된 상대 오차 (목표 질량 대비)
            mass_err = ((final_mass - t_mass) / (t_mass + 1e-3)) ** 2
            # CoG 오차: m² 단위, 박스 크기 기준 정규화
            cog_scale = max(bw, bh, bd) / 2.0
            cog_err = np.sum(((final_cog - t_cog) / cog_scale) ** 2)
            # MoI 오차: 대각 성분 평균 기준 일관된 정규화 (off-diagonal 포함)
            moi_err = np.sum(((final_moi - t_moi) / moi_scale) ** 2)

            # 3가지 목표를 동시에 달성: mass(고정 가중치) + CoG + MoI(focus 슬라이더 배분)
            w_cog = (1.0 - focus_moi)
            w_moi = focus_moi
            loss = mass_err * 1e6 + w_cog * cog_err * 1e4 + w_moi * moi_err * 1e4
            return loss
            
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Values in x were outside bounds.*")
            res = minimize(objective, p0, bounds=bounds, method='SLSQP', options={'maxiter': 200})
        
        final_masses, final_positions = get_masses_and_positions(res.x)
        
        mf = m_base + sum(final_masses)
        cf = (m_base * c_base + sum(m * pt for m, pt in zip(final_masses, final_positions))) / mf
        
        d = cf - c_base
        i_base_moved = np.zeros(6)
        i_base_moved[:3] = i_base[:3] + m_base * np.array([d[1]**2 + d[2]**2, d[0]**2 + d[2]**2, d[0]**2 + d[1]**2])
        i_base_moved[3:6] = i_base[3:6] - m_base * np.array([d[0]*d[1], d[0]*d[2], d[1]*d[2]])

        i_aux = np.zeros(6)
        for m, pt in zip(final_masses, final_positions):
            dp = pt - cf
            i_aux[0] += m * (dp[1]**2 + dp[2]**2)
            i_aux[1] += m * (dp[0]**2 + dp[2]**2)
            i_aux[2] += m * (dp[0]**2 + dp[1]**2)
            i_aux[3] -= m * dp[0] * dp[1]
            i_aux[4] -= m * dp[0] * dp[2]
            i_aux[5] -= m * dp[1] * dp[2]

        ifi = i_base_moved + i_aux
        
        self.optimized_masses = []
        for j, (m, pt) in enumerate(zip(final_masses, final_positions)):
            self.optimized_masses.append({
                "name": f"AutoBalance_{j+1}",
                "pos": [float(pt[0]), float(pt[1]), float(pt[2])],
                "mass": float(m),
                "size": [0.01, 0.01, 0.01]
            })

        # ── 터미널 출력 ──────────────────────────────────────────────────────
        sep = "─" * 72
        print(f"\n{sep}")
        print(f"  [WHTOOLS] Balancing Mass Optimization Result")
        print(sep)
        print(f"  {'#':<4} {'Name':<20} {'Mass (kg)':>10}  {'Pos X (m)':>10}  {'Pos Y (m)':>10}  {'Pos Z (m)':>10}")
        print(f"  {'─'*4} {'─'*20} {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")
        for am in self.optimized_masses:
            px, py, pz = am["pos"]
            print(f"  {am['name'].replace('AutoBalance_',''):<4} {am['name']:<20} {am['mass']:>10.4f}  {px:>10.4f}  {py:>10.4f}  {pz:>10.4f}")
        print(sep)
        mass_err_pct = abs(mf - t_mass) / (t_mass + 1e-9) * 100
        cog_err_mm   = np.linalg.norm(cf - t_cog) * 1000
        moi_scale_p  = max(np.abs(t_moi[:3]).mean(), 0.1)
        moi_err_pct  = np.linalg.norm((ifi - t_moi) / moi_scale_p) / np.sqrt(len(t_moi)) * 100
        print(f"  Total aux mass : {sum(am['mass'] for am in self.optimized_masses):.4f} kg  "
              f"(target {t_mass:.4f} kg, err {mass_err_pct:.2f}%)")
        print(f"  Final Mass     : {mf:.4f} kg")
        print(f"  Final CoG      : ({cf[0]:.4f}, {cf[1]:.4f}, {cf[2]:.4f}) m  "
              f"[err {cog_err_mm:.2f} mm]")
        print(f"  Final MoI diag : ({ifi[0]:.4f}, {ifi[1]:.4f}, {ifi[2]:.4f}) kg·m²  "
              f"[rel err {moi_err_pct:.2f}%]")
        print(f"  Target MoI diag: ({t_moi[0]:.4f}, {t_moi[1]:.4f}, {t_moi[2]:.4f}) kg·m²")
        print(sep + "\n")

        return m_base, c_base, i_base, t_mass, t_cog, t_moi, mf, cf, ifi

    def update_table(self, m0, c0, i0, tm, tc, ti, m_delta, pos_delta, i_delta, raw_eigs, valid_eigs):
        self.table_results.setRowCount(5)

        def make_item(text, color=None, mono=True):
            item = QtWidgets.QTableWidgetItem(text)
            item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            item.setFont(QtGui.QFont("Consolas", 9) if mono else QtGui.QFont("Segoe UI", 9))
            if color:
                item.setForeground(QtGui.QBrush(QtGui.QColor(color)))
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            return item

        # Row 0: Mass
        self.table_results.setItem(0, 0, make_item("Total Mass", mono=False))
        self.table_results.setItem(0, 1, make_item(f"{m0:.4f} kg"))
        self.table_results.setItem(0, 2, make_item(f"{tm:.4f} kg"))
        delta_sign = "+" if m_delta >= 0 else ""
        self.table_results.setItem(0, 3, make_item(f"{delta_sign}{m_delta:.4f} kg"))
        self.table_results.setItem(0, 4, make_item("✅ Exact", color="#a9dc76", mono=False))

        # Row 1: CoG
        self.table_results.setItem(1, 0, make_item("CoG (x,y,z)", mono=False))
        self.table_results.setItem(1, 1, make_item(f"({c0[0]:.4f}, {c0[1]:.4f}, {c0[2]:.4f})"))
        self.table_results.setItem(1, 2, make_item(f"({tc[0]:.4f}, {tc[1]:.4f}, {tc[2]:.4f})"))
        self.table_results.setItem(1, 3, make_item(f"({pos_delta[0]:.4f}, {pos_delta[1]:.4f}, {pos_delta[2]:.4f})"))
        self.table_results.setItem(1, 4, make_item("✅ Exact", color="#a9dc76", mono=False))

        # Row 2: MoI Diagonal
        self.table_results.setItem(2, 0, make_item("MoI Diagonal", mono=False))
        self.table_results.setItem(2, 1, make_item(f"({i0[0]:.4f}, {i0[1]:.4f}, {i0[2]:.4f})"))
        self.table_results.setItem(2, 2, make_item(f"({ti[0]:.4f}, {ti[1]:.4f}, {ti[2]:.4f})"))
        self.table_results.setItem(2, 3, make_item(f"({i_delta[0]:+.4f}, {i_delta[1]:+.4f}, {i_delta[2]:+.4f})"))
        neg_diag = any(i_delta[k] < 0 for k in range(3))
        feas_diag = "⚠️ Neg diag" if neg_diag else "✅ Exact"
        feas_color = "#ffd866" if neg_diag else "#a9dc76"
        self.table_results.setItem(2, 4, make_item(feas_diag, color=feas_color, mono=False))

        # Row 3: MoI Product
        self.table_results.setItem(3, 0, make_item("MoI Product", mono=False))
        i0p = i0[3:6] if len(i0) >= 6 else [0, 0, 0]
        self.table_results.setItem(3, 1, make_item(f"({i0p[0]:.4f}, {i0p[1]:.4f}, {i0p[2]:.4f})"))
        self.table_results.setItem(3, 2, make_item(f"({ti[3]:.4f}, {ti[4]:.4f}, {ti[5]:.4f})"))
        self.table_results.setItem(3, 3, make_item(f"({i_delta[3]:+.4f}, {i_delta[4]:+.4f}, {i_delta[5]:+.4f})"))
        self.table_results.setItem(3, 4, make_item("✅ Exact", color="#a9dc76", mono=False))

        # Row 4: MoI Eigenvalues (신설 - 고유치 물리 타당성 리포트)
        self.table_results.setItem(4, 0, make_item("Eigenvalues", mono=False))
        self.table_results.setItem(4, 1, make_item("N/A"))
        self.table_results.setItem(4, 2, make_item("N/A"))
        self.table_results.setItem(4, 3, make_item(f"λ=({valid_eigs[0]:.4f}, {valid_eigs[1]:.4f}, {valid_eigs[2]:.4f})"))
        
        has_raw_invalid = any(ev <= 1e-4 for ev in raw_eigs)
        if has_raw_invalid:
            self.table_results.setItem(4, 4, make_item("⚠️ Auto-Comp", color="#ffd866", mono=False))
        else:
            self.table_results.setItem(4, 4, make_item("✅ Valid", color="#a9dc76", mono=False))

        self.table_results.resizeRowsToContents()
        self.table_results.resizeColumnsToContents()

    def on_apply_clicked(self):
        try:
            self._calculate_delta_inertia()
            if self.inertia_correction is None:
                QMessageBox.warning(self, "Not Calculated", "Please calculate the inertia correction first.")
                return

            self.config["components_balance"] = {
                "target_mass": self.spin_target_mass.value(),
                "target_inertia": [
                    self.spin_moi_xx.value(), self.spin_moi_yy.value(), self.spin_moi_zz.value(),
                    self.spin_moi_xy.value(), self.spin_moi_xz.value(), self.spin_moi_yz.value()
                ],
                "target_cog": [self.spin_cog_x.value(), self.spin_cog_y.value(), self.spin_cog_z.value()],
            }
            self.config["inertia_correction"] = self.inertia_correction

            # [WHTOOLS] 입력받은 부품 무게(cushion, opencell, chassis)를 components 설정에 실시간 적용
            if "components" not in self.config:
                self.config["components"] = {}
            comp = self.config["components"]
            for key, val in [("cushion", self.spin_cushion_mass.value()),
                             ("opencell", self.spin_opencell_mass.value()),
                             ("chassis", self.spin_chassis_mass.value())]:
                if key not in comp:
                    comp[key] = {}
                comp[key]["mass"] = val

            if self.parent_dialog:
                self.parent_dialog.config = self.config
                self.parent_dialog._populate_config_tree()
                self.parent_dialog._update_reporting()

            self.accept()
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Apply Error", f"Failed to apply inertia correction:\n{e}")

class ModelSetupDialog(QtWidgets.QDialog):
    """
    [WHTOOLS] MuJoCo Model Configuration & Setup Dialog
    사용자가 직관적으로 모델을 구성하고 물리 파라미터를 수정할 수 있는 통합 인터페이스입니다.
    """
    def _update_reporting(self):
        """설정 갱신 시 트리 뷰와 스키매틱을 실시간 갱신합니다."""
        self._populate_config_tree()
        if hasattr(self, 'schematic') and self.schematic:
            self.schematic.update_config(self.config)

    def _on_balance_clicked(self):
        """Mass, CoG, MoI Balancing Optimizer 대화상자를 호출합니다."""
        dlg = ComponentBalanceDialog(parent=self, config=self.config)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.config = dlg.config.copy()
            self._populate_config_tree()
            self._update_reporting()

    def __init__(self, parent=None, simulator=None):
        super().__init__(parent)
        self.sim = simulator
        self.config = simulator.config.copy() if simulator else {}
        self.setWindowTitle("🛠️ [WHTOOLS] Model Configuration & Setup")
        self.setMinimumSize(750, 690)  # [WHTOOLS] 프리뷰 컴팩트 슬림화에 따른 세로 최소 높이 축소 (780 -> 690)
        self.resize(800, 700)          # 기본 열림 세로 크기도 800 -> 700 으로 슬림화
        
        self.setStyleSheet(GLOBAL_QSS)
        
        self._init_ui()
        # [WHTOOLS BUG-FIX] _on_ista_changed 호출 전에 config의 drop_direction을 edit_direction에 먼저 주입.
        # 이렇게 해야 _on_ista_changed의 방어 로직이 기존 유효한 Edge/Corner 방향을 인식하고 보존함.
        _saved_direction = self.config.get("drop_direction", "")
        if _saved_direction:
            self.edit_direction.setText(_saved_direction)
        self._on_ista_changed(self.combo_ista.currentText())

    def _init_ui(self):
        # [WHTOOLS] 레거시 호환용 shadow 위젯 생성 및 숨김 처리 (combo_ista AttributeError 원천 방지)
        self.combo_ista = QtWidgets.QComboBox()
        self.combo_ista.addItems(["PARCEL", "LTL", "GENERAL"])
        self.combo_ista.setCurrentText(self.config.get("drop_mode", "PARCEL"))
        self.combo_ista.hide()

        self.combo_preset = QtWidgets.QComboBox()
        self.combo_preset.hide()

        self.combo_gen_type = QtWidgets.QComboBox()
        self.combo_gen_type.hide()

        self.combo_gen_p1 = QtWidgets.QComboBox()
        self.combo_gen_p1.hide()

        self.combo_gen_p2 = QtWidgets.QComboBox()
        self.combo_gen_p2.hide()

        self.combo_gen_p3 = QtWidgets.QComboBox()
        self.combo_gen_p3.hide()

        self.edit_direction = QtWidgets.QLineEdit()
        self.edit_direction.hide()

        self.spin_height = QtWidgets.QDoubleSpinBox()
        self.spin_height.setValue(self.config.get("drop_height", 0.5))
        self.spin_height.hide()

        self.spin_azimuth = QtWidgets.QDoubleSpinBox()
        self.spin_azimuth.setValue(self.config.get("initial_tilt_azimuth_deg", 0.0))
        self.spin_azimuth.hide()

        self.spin_lat = QtWidgets.QDoubleSpinBox()
        self.spin_lat.setValue(self.config.get("initial_tilt_deg", 0.0))
        self.spin_lat.hide()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(6)
        
        # 1. 상단 섹션: Preset & Drop Setup + Visual Schematic
        top_splitter = QtWidgets.QSplitter(Qt.Horizontal)
        
        # 1-1. 왼쪽: Preset & Drop Controls
        setup_group = QtWidgets.QGroupBox("Setup")
        setup_vlay = QtWidgets.QVBoxLayout(setup_group)
        setup_vlay.setSpacing(6)
        setup_vlay.setContentsMargins(12, 16, 12, 12)
        
        # 1) Setup 버튼
        self.btn_select_ref_model_direct = QtWidgets.QPushButton("🔍 Size and ISTA")        
        self.btn_select_ref_model_direct.setFixedHeight(30)
        self.btn_select_ref_model_direct.clicked.connect(self._on_select_sequence)
        setup_vlay.addWidget(self.btn_select_ref_model_direct)
        
        # 2) Mass, CoG, MoI 버튼 (Balance Optimizer UI 호출)
        self.btn_mass_cog_moi = QtWidgets.QPushButton("⚖️ Mass, CoG, MoI")
        self.btn_mass_cog_moi.setFixedHeight(30)
        self.btn_mass_cog_moi.clicked.connect(self._on_balance_clicked)
        setup_vlay.addWidget(self.btn_mass_cog_moi)
        
        # 3) Blocks (Mesh Resolution Preset) 추가
        blocks_group = QtWidgets.QGroupBox("Blocks (Mesh Preset)")
        blocks_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        blocks_vlay = QtWidgets.QVBoxLayout(blocks_group)
        blocks_vlay.setSpacing(6)
        blocks_vlay.setContentsMargins(8, 10, 8, 8)
        
        # Normal
        lay_normal = QtWidgets.QHBoxLayout()
        self.btn_normal = QtWidgets.QPushButton("Normal")
        self.btn_normal.setCheckable(True)        
        lbl_normal_desc = QtWidgets.QLabel("5x5x3, 4x4x1 (Weld)")        
        lbl_normal_desc.setStyleSheet(f"color: {C_TEXT_DIM};")
        lay_normal.addWidget(self.btn_normal)
        lay_normal.addWidget(lbl_normal_desc)
        lay_normal.addStretch()
        blocks_vlay.addLayout(lay_normal)
        
        # Fast
        lay_fast = QtWidgets.QHBoxLayout()
        self.btn_fast = QtWidgets.QPushButton("Fast")
        self.btn_fast.setCheckable(True)        
        lbl_fast_desc = QtWidgets.QLabel("3x3x3, 3x3x1 (Full Rigid)")        
        lbl_fast_desc.setStyleSheet(f"color: {C_TEXT_DIM};")
        lay_fast.addWidget(self.btn_fast)
        lay_fast.addWidget(lbl_fast_desc)
        lay_fast.addStretch()
        blocks_vlay.addLayout(lay_fast)
        
        setup_vlay.addWidget(blocks_group)
        
        self.block_group = QtWidgets.QButtonGroup(self)
        self.block_group.addButton(self.btn_normal)
        self.block_group.addButton(self.btn_fast)
        
        # 초기 활성 Preset 판정 및 스타일 적용
        comp = self.config.get("components", {})
        paper_div = comp.get("paper", {}).get("div", [5, 5, 3])
        if paper_div == [3, 3, 3]:
            self.btn_fast.setChecked(True)
            self.btn_fast.setStyleSheet(f"background-color: {C_BTN_BLUE3}; color: white; font-weight: bold;")
        else:
            self.btn_normal.setChecked(True)
            self.btn_normal.setStyleSheet(f"background-color: {C_BTN_BLUE3}; color: white; font-weight: bold;")
            
        # 클릭 이벤트 연결
        self.btn_normal.clicked.connect(lambda: self._on_block_preset_changed("Normal"))
        self.btn_fast.clicked.connect(lambda: self._on_block_preset_changed("Fast"))
        
        setup_vlay.addStretch()
        
                
        top_splitter.addWidget(setup_group)
        
        # 1-2. 오른쪽: Visual Schematic
        self.schematic = VisualSchematicWidget()
        self.schematic.update_config(self.config)
        top_splitter.addWidget(self.schematic)
        
        layout.addWidget(top_splitter)
        
        # 3. 중앙 섹션: Config Tree + Python Editor
        mid_splitter = QtWidgets.QSplitter(Qt.Vertical)
        
        # 3-1. 위쪽: Config Tree & Category
        self.config_tree = QtWidgets.QTreeWidget()
        self.config_tree.setColumnCount(3)
        self.config_tree.setHeaderLabels(["Configuration Key", "Value", "Description"])
        self.config_tree.setColumnWidth(0, 200)
        self.config_tree.setColumnWidth(1, 250)  # Value 열 폭
        self.config_tree.setColumnWidth(2, 250)  # Description 열 폭
        self.config_tree.itemClicked.connect(self._on_config_item_clicked)
        self.config_tree.itemChanged.connect(self._on_tree_item_changed)
        self.config_tree.setStyleSheet(TREE_QSS)
        
        mid_splitter.addWidget(self.config_tree)
        
        # 3-2. 아래쪽: Editor Area
        editor_widget = QtWidgets.QWidget()
        editor_layout = QtWidgets.QVBoxLayout(editor_widget)
        
        self.lbl_editor_title = QtWidgets.QLabel("Select a key to edit...")
        self.lbl_editor_title.setStyleSheet(f"font-weight: bold; color: {C_ACCENT_BLUE};")
        editor_layout.addWidget(self.lbl_editor_title)
        
        self.py_editor = QtWidgets.QPlainTextEdit()
        self.py_editor.setStyleSheet(f"background-color: {C_BG_EDITOR}; color: {C_STATUS_TEXT_OK}; border: 1px solid {C_BORDER};")
        editor_layout.addWidget(self.py_editor)
        
        edit_btn_layout = QtWidgets.QHBoxLayout()
        self.btn_apply_val = QtWidgets.QPushButton("✅ Apply Value")
        self.btn_apply_val.clicked.connect(self._on_apply_value)
        self.lbl_status = QtWidgets.QLabel("Ready")
        edit_btn_layout.addWidget(self.btn_apply_val)
        edit_btn_layout.addWidget(self.lbl_status)
        edit_btn_layout.addStretch()
        editor_layout.addLayout(edit_btn_layout)
        
        mid_splitter.addWidget(editor_widget)
        mid_splitter.setStretchFactor(0, 2)
        mid_splitter.setStretchFactor(1, 1)
        
        layout.addWidget(mid_splitter)
        
        # 4. 버튼 영역
        btn_box = QtWidgets.QHBoxLayout()
        self.btn_create = QtWidgets.QPushButton("🚀 Create && Reload Model")
        self.btn_create.setMinimumHeight(30)                
        self.btn_create.clicked.connect(self._on_create_and_reload)
        
        self.btn_close = QtWidgets.QPushButton("Close")
        self.btn_close.setMinimumHeight(45)
        self.btn_close.clicked.connect(self.reject)
        
        btn_box.addWidget(self.btn_close)
        btn_box.addStretch()
        btn_box.addWidget(self.btn_create)
        layout.addLayout(btn_box)
        
        # [WHTOOLS] 레거시 호환용 속성 바인딩 매핑 (AttributeError 원천 방지)
        self.btn_select_sequence = self.btn_select_ref_model_direct
        self.general_dropdowns_container = QtWidgets.QWidget(self) # 다이얼로그 자식으로 소속시켜 독립 top-level 창 격발 오작동 차단
        self.general_dropdowns_container.setGeometry(0, 0, 0, 0)
        self.general_dropdowns_container.hide()

        # 초기 트리 채우기
        self._populate_config_tree()

    def _add_dict_items(self, parent_item, data, path_prefix):
        """dict 형태의 설정 데이터를 트리 구조로 하위에 배치합니다."""
        for k, v in sorted(data.items(), key=lambda x: str(x[0])):
            item_path = path_prefix + (k,)
            child_item = QtWidgets.QTreeWidgetItem(parent_item)
            child_item.setText(0, str(k)) # 0번째 열(Configuration Key)에 키명 설정! 화살표/들여쓰기 적용
            
            # 값이 dict인 경우 재귀적으로 하위 노드 생성
            if isinstance(v, dict):
                child_item.setText(1, "(dictionary)") # col=1이 Value 열
                child_item.setData(0, Qt.UserRole, item_path) # UserRole은 0번째 열에 귀속
                self._add_dict_items(child_item, v, item_path)
            else:
                child_item.setText(1, repr(v)) # col=1이 Value 열
                child_item.setData(0, Qt.UserRole, item_path) # UserRole은 0번째 열에 귀속
                # Leaf 노드인 경우 Value 열 편집 가능 지정
                child_item.setFlags(child_item.flags() | QtCore.Qt.ItemIsEditable)

    def _populate_config_tree(self):
        """CONFIG_METADATA를 기반으로 트리를 구성하고 메타데이터에 없는 키도 추가합니다.
        딕셔너리 구조의 설정 값은 하위에 트리 노드로 나누어 표시합니다."""
        self.config_tree.blockSignals(True) # In-place 갱신 무한 루프 방지 시그널 차단
        self.config_tree.clear()
        categories = {}
        
        # 1. 메타데이터 기반 분류
        for key, meta in CONFIG_METADATA.items():
            cat = meta["cat"]
            if cat not in categories:
                cat_item = QtWidgets.QTreeWidgetItem(self.config_tree)
                cat_item.setText(0, cat)
                cat_item.setExpanded(True)
                categories[cat] = cat_item
            
            val = self.config.get(key)
            
            # [PREMIUM UI/UX] 카테고리 껍데기 노드 간소화(Flattening) 기법 적용!
            # 카테고리가 'Components'이거나 'Weld Physics'인 경우, 껍데기 노드 없이 대분류 하위에 다이렉트 주입!
            if cat in ["Components", "Weld Physics"] and isinstance(val, dict):
                categories[cat].setData(0, Qt.UserRole, (key,))
                self._add_dict_items(categories[cat], val, (key,))
            else:
                key_item = QtWidgets.QTreeWidgetItem(categories[cat])
                key_item.setText(0, key) # 0번째 열(Configuration Key)에 주입하여 완벽한 트리 들여쓰기 렌더링 실현!
                key_item.setText(2, meta["desc"]) # col=2가 Description 열
                
                if isinstance(val, dict):
                    key_item.setText(1, "(dictionary)") # col=1이 Value 열
                    key_item.setData(0, Qt.UserRole, (key,))
                    self._add_dict_items(key_item, val, (key,))
                else:
                    key_item.setText(1, repr(val)) # col=1이 Value 열
                    key_item.setData(0, Qt.UserRole, (key,))
                    # Leaf 노드인 경우 Value 열 편집 가능 지정
                    key_item.setFlags(key_item.flags() | QtCore.Qt.ItemIsEditable)
            
        # 2. 메타데이터에 없는 기타 키들 추가
        misc_cat = None
        for key in sorted(self.config.keys()):
            if key not in CONFIG_METADATA:
                if misc_cat is None:
                    misc_cat = QtWidgets.QTreeWidgetItem(self.config_tree)
                    misc_cat.setText(0, "Miscellaneous")
                    misc_cat.setExpanded(True)
                key_item = QtWidgets.QTreeWidgetItem(misc_cat)
                key_item.setText(0, key) # 0번째 열(Configuration Key)에 키 주입!
                
                val = self.config.get(key)
                if isinstance(val, dict):
                    key_item.setText(1, "(dictionary)") # col=1이 Value 열
                    key_item.setData(0, Qt.UserRole, (key,))
                    self._add_dict_items(key_item, val, (key,))
                else:
                    key_item.setText(1, repr(val)) # col=1이 Value 열
                    key_item.setData(0, Qt.UserRole, (key,))
                    # Leaf 노드인 경우 Value 열 편집 가능 지정
                    key_item.setFlags(key_item.flags() | QtCore.Qt.ItemIsEditable)
        self.config_tree.blockSignals(False) # 갱신 후 시그널 복원

    def _on_config_item_clicked(self, item, col):
        key_path = item.data(0, Qt.UserRole) # 0번째 열(Configuration Key)에서 UserRole 경로 획득!
        if not key_path: return
        
        if isinstance(key_path, str):
            key_path = (key_path,)
            
        path_str = " ➔ ".join(str(k) for k in key_path)
        self.lbl_editor_title.setText(f"Editing Key: {path_str}")
        
        # 딕셔너리 경로 탐색
        val = self.config
        for k in key_path:
            val = val[k]
            
        self.py_editor.setPlainText(repr(val))
        self.current_editing_key = key_path

    def _on_tree_item_changed(self, item, column):
        """트리 테이블의 Value 열(column = 1)에서 직접 값을 수정한 경우 config에 즉시 동기화합니다."""
        if column != 1: return # col=1이 Value 열
        
        key_path = item.data(0, Qt.UserRole) # 0번째 열(Configuration Key)에서 UserRole 경로 획득!
        if not key_path: return
        
        if isinstance(key_path, str):
            key_path = (key_path,)
            
        code = item.text(1).strip() # col=1에서 수정된 텍스트 획득!
        try:
            # 안전하게 파싱 시도 (ast.literal_eval)
            try:
                new_val = ast.literal_eval(code)
            except:
                new_val = eval(code, {"__builtins__": None}, {})
                
            # 딕셔너리 경로를 따라가며 설정 값 갱신
            d = self.config
            for pk in key_path[:-1]:
                d = d[pk]
            d[key_path[-1]] = new_val
            
            self.lbl_status.setText(f"✅ In-place: '{key_path[-1]}' updated to {repr(new_val)}")
            self.lbl_status.setStyleSheet(f"color: {C_STATUS_TEXT_OK};")
            
            # 스키매틱 갱신 및 에디터 텍스트 동기화
            self.schematic.update_config(self.config)
            
            # 만약 현재 에디터가 이 키를 보고 있었다면 에디터 본문 내용도 함께 갱신
            if hasattr(self, 'current_editing_key') and self.current_editing_key == key_path:
                self.py_editor.blockSignals(True)
                self.py_editor.setPlainText(repr(new_val))
                self.py_editor.blockSignals(False)
                
        except Exception as e:
            self.lbl_status.setText(f"❌ Editing Error: {e}")
            self.lbl_status.setStyleSheet(f"color: {C_STATUS_ERR};")
            # 값 복원 (에러 발생 시 원래 유효한 값으로 트리 롤백)
            self._populate_config_tree()

    def _on_apply_value(self):
        if not hasattr(self, 'current_editing_key'): return
        
        code = self.py_editor.toPlainText().strip()
        try:
            # ast.literal_eval로 안전하게 파싱 시도, 안되면 eval (주의)
            try:
                new_val = ast.literal_eval(code)
            except:
                new_val = eval(code, {"__builtins__": None}, {})
                
            key_path = self.current_editing_key
            if isinstance(key_path, str):
                key_path = (key_path,)
                
            # 딕셔너리 경로를 따라가며 설정 값 갱신
            d = self.config
            for pk in key_path[:-1]:
                d = d[pk]
            d[key_path[-1]] = new_val
            
            self.lbl_status.setText("✅ Value applied.")
            self.lbl_status.setStyleSheet(f"color: {C_STATUS_TEXT_OK};")
            
            # 스키매틱 및 트리 갱신
            self.schematic.update_config(self.config)
            self._populate_config_tree()
        except Exception as e:
            self.lbl_status.setText(f"❌ Error: {e}")
            self.lbl_status.setStyleSheet(f"color: {C_STATUS_ERR};")

    def _on_preset_changed(self, text):
        if text == "Custom": return
        
        presets = {
            "55 inch": {"box_w": 1.40, "box_h": 0.85, "box_d": 0.15, "assy_w": 1.25, "assy_h": 0.72, "mass_chassis": 12.0},
            "65 inch": {"box_w": 1.60, "box_h": 0.95, "box_d": 0.17, "assy_w": 1.45, "assy_h": 0.83, "mass_chassis": 18.0},
            "75 inch": {"box_w": 1.84, "box_h": 1.10, "box_d": 0.18, "assy_w": 1.67, "assy_h": 0.96, "mass_chassis": 25.0},
            "85 inch": {"box_w": 2.05, "box_h": 1.25, "box_d": 0.22, "assy_w": 1.90, "assy_h": 1.08, "mass_chassis": 35.0},
            "98 inch": {"box_w": 2.35, "box_h": 1.45, "box_d": 0.25, "assy_w": 2.18, "assy_h": 1.25, "mass_chassis": 55.0},
        }
        
        if text in presets:
            self.config.update(presets[text])
            self.schematic.update_config(self.config)
            self._populate_config_tree() # 트리 값 갱신

    def _on_select_sequence(self):
        """LTL/PARCEL 모드일 때 ISTA 시퀀스 및 각도 헬퍼 다이얼로그를 모달로 엽니다."""
        dlg = IstaSetupHelperDialog(self.config, self)
        dlg.exec_()

    def _update_general_combos(self):
        """GENERAL 모드일 때 Drop Type(Face, Edge, Corner)에 따라 하위 콤보박스 아이템 및 가시성을 조절합니다."""
        dtype = self.combo_gen_type.currentText()
        
        self.combo_gen_p1.blockSignals(True)
        self.combo_gen_p2.blockSignals(True)
        self.combo_gen_p3.blockSignals(True)
        
        if dtype == "Face":
            self.combo_gen_p1.setVisible(True)
            self.combo_gen_p2.setVisible(False)
            self.combo_gen_p3.setVisible(False)
            
            self.combo_gen_p1.clear()
            self.combo_gen_p1.addItems(["top", "bottom", "left", "right", "front", "back"])
            
        elif dtype == "Edge":
            self.combo_gen_p1.setVisible(True)
            self.combo_gen_p2.setVisible(True)
            self.combo_gen_p3.setVisible(False)
            
            self.combo_gen_p1.clear()
            self.combo_gen_p1.addItems(["front", "back", "top", "bottom"])
            
            self.combo_gen_p2.clear()
            self.combo_gen_p2.addItems(["top", "bottom", "left", "right"])
            
        elif dtype == "Corner":
            self.combo_gen_p1.setVisible(True)
            self.combo_gen_p2.setVisible(True)
            self.combo_gen_p3.setVisible(True)
            
            self.combo_gen_p1.clear()
            self.combo_gen_p1.addItems(["front", "back"])
            
            self.combo_gen_p2.clear()
            self.combo_gen_p2.addItems(["top", "bottom"])
            
            self.combo_gen_p3.clear()
            self.combo_gen_p3.addItems(["left", "right"])
            
        self.combo_gen_p1.blockSignals(False)
        self.combo_gen_p2.blockSignals(False)
        self.combo_gen_p3.blockSignals(False)

    def _on_general_dropdowns_changed(self):
        """Dynamic General Dropdown 선택 변화 시 drop_direction을 계산하여 edit_direction 및 config에 반영합니다."""
        sender = self.sender()
        if sender == self.combo_gen_type:
            self._update_general_combos()
            
        dtype = self.combo_gen_type.currentText()
        p1 = self.combo_gen_p1.currentText()
        p2 = self.combo_gen_p2.currentText()
        p3 = self.combo_gen_p3.currentText()
        
        if dtype == "Face":
            val = p1
        elif dtype == "Edge":
            val = f"{p1}-{p2}"
        else:
            val = f"{p1}-{p2}-{p3}"
            
        self.edit_direction.setText(val)
        self.config["drop_direction"] = val
        self.schematic.update_config(self.config)

    def _on_numeric_ui_changed(self):
        """Spinbox 수치 조작 시 config 실시간 동기화 및 2D 스키매틱/질량 리포트 실시간 갱신"""
        self.config["drop_height"] = self.spin_height.value()
        self.config["initial_tilt_azimuth_deg"] = self.spin_azimuth.value()
        self.config["initial_tilt_deg"] = self.spin_lat.value()
        self.schematic.update_config(self.config)

    def _on_ista_changed(self, text):
        """
        [WHTOOLS] 낙하 모드(ISTA) 변경 시 UI 상태를 갱신합니다.
        
        핵심 방어 로직: 이미 유효한 Edge/Corner/Face 번호 방향이 설정되어 있고,
        해당 방향이 현재 선택된 모드와 일치하는 경우에는 기본값으로 덮어쓰지 않습니다.
        이는 Setup Helper에서 적용한 방향이 콤보박스 시그널에 의해 소거되는 버그를 방지합니다.
        
        Args:
            text (str): 선택된 모드 문자열 ("PARCEL", "LTL", "GENERAL")
        """
        self.config["drop_mode"] = text
        if text == "GENERAL":
            self.btn_select_sequence.setVisible(True) # GENERAL 모드에서도 항상 버튼 노출하여 치수/ISTA 설정 가이드 확보
            self.general_dropdowns_container.setVisible(True)
            self.edit_direction.setReadOnly(True)
            self._update_general_combos()
            self._on_general_dropdowns_changed()
        else:
            self.btn_select_sequence.setVisible(True)
            self.general_dropdowns_container.setVisible(False)
            self.edit_direction.setReadOnly(True)
            
            # [WHTOOLS BUG-FIX] 기존에 유효한 방향이 설정되어 있으면 기본값으로 덮어쓰지 않음
            # Setup Helper에서 Apply로 설정한 Edge/Corner/Face 방향 보존
            current_direction = self.edit_direction.text().strip()
            _default_parcel = "front-bottom-right"
            _default_ltl    = "bottom"
            
            # 현재 방향이 비어있거나, 반대편 모드의 기본값인 경우에만 기본값으로 초기화
            _is_empty = not current_direction
            _is_opposite_default = (
                (text == "LTL"    and current_direction == _default_parcel) or
                (text == "PARCEL" and current_direction == _default_ltl)
            )
            
            if _is_empty or _is_opposite_default:
                if text == "PARCEL":
                    self.edit_direction.setText(_default_parcel)  # Mapped to Corner 2-3-5
                elif text == "LTL":
                    self.edit_direction.setText(_default_ltl)     # Mapped to Face 3
            # 그 외(유효한 Face/Edge/Corner 번호 방향 등)는 기존 값 그대로 유지
            
            self.config["drop_direction"] = self.edit_direction.text()
            self.schematic.update_config(self.config)



    def _on_create_and_reload(self):
        """최종 설정을 적용하고 시뮬레이터를 리로드합니다."""
        # UI 입력값 최종 동기화
        self.config["drop_direction"] = self.edit_direction.text()
        self.config["drop_height"] = self.spin_height.value()
        self.config["initial_tilt_azimuth_deg"] = self.spin_azimuth.value()
        self.config["initial_tilt_deg"] = self.spin_lat.value()
        
        # 시뮬레이터에 설정 전달 및 리로드 요청
        if self.sim:
            self.sim.config.update(self.config)
            self.sim.ctrl_reload_request = True
            self.accept()
        else:
            self.accept()

    def _on_block_preset_changed(self, mode):
        """[WHTOOLS] Blocks 프리셋 변경에 따라 components의 div 및 weld 속성을 정교하게 리매핑합니다."""
        if "components" not in self.config:
            self.config["components"] = {}
            
        comp = self.config["components"]
        
        # UI 스타일 피드백 업데이트 (눌린 버튼 하이라이트)
        for mode_name, btn in [("Normal", self.btn_normal), ("Fast", self.btn_fast)]:
            if mode_name == mode:
                btn.setStyleSheet(f"background-color: {C_BTN_BLUE3}; color: white; font-weight: bold;")
            else:
                btn.setStyleSheet("")
                
        # 기본 rgba 폴백 매핑
        fallback_rgba = {
            "paper": "1.0 0.85 0.7 1.0",
            "cushion": "0.8 0.8 0.8 0.6",
            "opencell": "0.1 0.1 0.1 1.0",
            "opencellcoh": "1.0 0.0 0.0 0.4",
            "chassis": "0.0 0.2 0.4 1.0"
        }
        
        # 1. Normal Mode
        if mode == "Normal":
            # paper
            if "paper" not in comp: comp["paper"] = {}
            comp["paper"].update({
                "div": [5, 5, 3],
                "use_weld": True,
                "mass": comp["paper"].get("mass", 4.0),
                "rgba": comp["paper"].get("rgba", fallback_rgba["paper"])
            })
            # cushion
            if "cushion" not in comp: comp["cushion"] = {}
            comp["cushion"].update({
                "div": [5, 5, 3],
                "use_weld": True,
                "mass": comp["cushion"].get("mass", 3.0),
                "rgba": comp["cushion"].get("rgba", fallback_rgba["cushion"])
            })
            # opencell
            if "opencell" not in comp: comp["opencell"] = {}
            comp["opencell"].update({
                "div": [4, 4, 1],
                "use_weld": True,
                "mass": comp["opencell"].get("mass", 5.0),
                "rgba": comp["opencell"].get("rgba", fallback_rgba["opencell"])
            })
            # opencellcoh
            if "opencellcoh" not in comp: comp["opencellcoh"] = {}
            comp["opencellcoh"].update({
                "div": [4, 4, 1],
                "use_weld": True,
                "mass": comp["opencellcoh"].get("mass", 0.1),
                "rgba": comp["opencellcoh"].get("rgba", fallback_rgba["opencellcoh"]),
                "enable_btm_weld": False
            })
            # chassis
            if "chassis" not in comp: comp["chassis"] = {}
            comp["chassis"].update({
                "div": [4, 4, 1],
                "use_weld": True,
                "mass": comp["chassis"].get("mass", 10.0),
                "rgba": comp["chassis"].get("rgba", fallback_rgba["chassis"])
            })
            
        # 2. Fast Mode
        elif mode == "Fast":
            # paper
            if "paper" not in comp: comp["paper"] = {}
            comp["paper"].update({
                "div": [3, 3, 3],
                "use_weld": True,
                "mass": comp["paper"].get("mass", 4.0),
                "rgba": comp["paper"].get("rgba", fallback_rgba["paper"])
            })
            # cushion
            if "cushion" not in comp: comp["cushion"] = {}
            comp["cushion"].update({
                "div": [3, 3, 3],
                "use_weld": True,
                "mass": comp["cushion"].get("mass", 3.0),
                "rgba": comp["cushion"].get("rgba", fallback_rgba["cushion"])
            })
            # opencell
            if "opencell" not in comp: comp["opencell"] = {}
            comp["opencell"].update({
                "div": [3, 3, 1],
                "use_weld": False,
                "mass": comp["opencell"].get("mass", 5.0),
                "rgba": comp["opencell"].get("rgba", fallback_rgba["opencell"])
            })
            # opencellcoh
            if "opencellcoh" not in comp: comp["opencellcoh"] = {}
            comp["opencellcoh"].update({
                "div": [3, 3, 1],
                "use_weld": False,
                "mass": comp["opencellcoh"].get("mass", 0.1),
                "rgba": comp["opencellcoh"].get("rgba", fallback_rgba["opencellcoh"]),
                "enable_btm_weld": True
            })
            # chassis
            if "chassis" not in comp: comp["chassis"] = {}
            comp["chassis"].update({
                "div": [3, 3, 1],
                "use_weld": False,
                "mass": comp["chassis"].get("mass", 10.0),
                "rgba": comp["chassis"].get("rgba", fallback_rgba["chassis"])
            })
            
        # 변경 사항 실시간 UI 동기화
        self._populate_config_tree()
        self.schematic.update_config(self.config)

    def done(self, result):
        super().done(result)

class StructuralDynamicsDialog(QtWidgets.QDialog):
    """
    [WHTOOLS] Structural Dynamics Data Extraction Dialog
    근사화된 구조 해석용 하중점(Cushion Corners) 데이터를 추출하여 CSV로 저장합니다.
    """
    def __init__(self, parent=None, simulator=None):
        super().__init__(parent)
        self.sim = simulator
        self.setWindowTitle("🏗️ Structural Dynamics Extraction")
        self.setMinimumWidth(500)
        self._init_ui()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(15)

        # 1. 안내 메시지
        info_label = QtWidgets.QLabel(
            "<b>[안내]</b> 근사화된 구조 해석용 하중점 추출 프로토콜입니다.<br>"
            "박스 모서리에 위치한 Cushion 요소들의 초기 위치와 시간에 따른 글로벌 좌표를 추출합니다."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # 2. 파일 저장 경로
        path_layout = QtWidgets.QHBoxLayout()
        self.edit_path = QtWidgets.QLineEdit()
        default_path = str(self.sim.output_dir / "structural_dynamics.csv") if self.sim else "structural_dynamics.csv"
        self.edit_path.setText(default_path)
        self.btn_browse = QtWidgets.QPushButton("Browse...")
        self.btn_browse.clicked.connect(self._on_browse)
        path_layout.addWidget(QtWidgets.QLabel("Save Path:"))
        path_layout.addWidget(self.edit_path)
        path_layout.addWidget(self.btn_browse)
        layout.addLayout(path_layout)

        # 3. Chassis Geometry (mm 입력)
        geo_group = QtWidgets.QGroupBox("Chassis Geometry (mm)")
        geo_layout = QtWidgets.QGridLayout(geo_group)
        
        self.spin_width = QtWidgets.QDoubleSpinBox()
        self.spin_width.setRange(0, 5000); self.spin_width.setValue(1600.0)
        self.spin_height = QtWidgets.QDoubleSpinBox()
        self.spin_height.setRange(0, 5000); self.spin_height.setValue(1200.0)
        self.spin_thick = QtWidgets.QDoubleSpinBox()
        self.spin_thick.setRange(0, 1000); self.spin_thick.setValue(120.0)
        
        geo_layout.addWidget(QtWidgets.QLabel("Width (mm):"), 0, 0)
        geo_layout.addWidget(self.spin_width, 0, 1)
        geo_layout.addWidget(QtWidgets.QLabel("Height (mm):"), 1, 0)
        geo_layout.addWidget(self.spin_height, 1, 1)
        geo_layout.addWidget(QtWidgets.QLabel("Depth/Thick (mm):"), 2, 0)
        geo_layout.addWidget(self.spin_thick, 2, 1)
        layout.addWidget(geo_group)

        # 4. 분석 시작 시간
        time_layout = QtWidgets.QHBoxLayout()
        self.spin_start_time = QtWidgets.QDoubleSpinBox()
        self.spin_start_time.setRange(0, 10.0); self.spin_start_time.setValue(0.0)
        time_layout.addWidget(QtWidgets.QLabel("Start Time (s):"))
        time_layout.addWidget(self.spin_start_time)
        time_layout.addStretch()
        layout.addLayout(time_layout)

        # 5. 실행 버튼
        self.btn_do = QtWidgets.QPushButton("🚀 Do it")
        self.btn_do.clicked.connect(self._on_execute)
        layout.addWidget(self.btn_do)

    def _on_browse(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save CSV", self.edit_path.text(), "CSV Files (*.csv)")
        if path:
            self.edit_path.setText(path)

    def _on_execute(self):
        """데이터 추출 및 저장 로직 수행"""
        if not self.sim or not self.sim.snapshots:
            QtWidgets.QMessageBox.warning(self, "Error", "No simulation snapshots available.")
            return

        save_path = self.edit_path.text()
        start_time = self.spin_start_time.value()
        
        # 버튼 상태 업데이트 (시각적 피드백)
        self.btn_do.setText("⏳ Processing...")
        self.btn_do.setEnabled(False)
        QtWidgets.QApplication.processEvents()
        
        try:
            # [WHTOOLS] Chassis C1-C8 코너 포인트 식별 및 데이터 추출
            # 모니터(whts_monitor)와 동일한 C1-C8 정의 및 인덱스 사용
            w_input = self.spin_width.value() / 1000.0
            h_input = self.spin_height.value() / 1000.0
            d_input = self.spin_thick.value() / 1000.0
            
            # whts_monitor.py의 정의와 일치하는 코너 정보
            corner_configs = [
                {"id": "C1", "name": "Front-Top-Right",    "s": [ 1,  1,  1]},
                {"id": "C2", "name": "Front-Bottom-Right", "s": [ 1, -1,  1]},
                {"id": "C3", "name": "Front-Bottom-Left",  "s": [-1, -1,  1]},
                {"id": "C4", "name": "Front-Top-Left",     "s": [-1,  1,  1]},
                {"id": "C5", "name": "Rear-Top-Right",     "s": [ 1,  1, -1]},
                {"id": "C6", "name": "Rear-Bottom-Right",  "s": [ 1, -1, -1]},
                {"id": "C7", "name": "Rear-Bottom-Left",   "s": [-1, -1, -1]},
                {"id": "C8", "name": "Rear-Top-Left",      "s": [-1,  1, -1]},
            ]
            
            # 데이터 추출용 임시 mjData 생성 (스레드 안전성 확보)
            temp_data = mujoco.MjData(self.sim.model)
            
            with open(save_path, "w", encoding="utf-8") as f:
                # 헤더 정보: 사용자 요청 양식 준수
                f.write(f"# chassis, {self.spin_width.value():.0f}, {self.spin_height.value():.0f}, {self.spin_thick.value():.0f}\n")
                f.write(f"# start time, {start_time:.1f}\n")
                
                for c in corner_configs:
                    sx, sy, sz = c["s"]
                    # 각 값은 half값이므로 *2를 하고 mm 단위(*1000)로 변환
                    lx_full = sx * w_input * 1000.0
                    ly_full = sy * h_input * 1000.0
                    lz_full = sz * d_input * 1000.0
                    f.write(f"# {c['id']}, {lx_full:.0f}, {ly_full:.0f}, {lz_full:.0f}\n")
                
                # 데이터 헤더
                cols = ["Frame", "Time"]
                for c in corner_configs:
                    cols.extend([f"{c['id']}_X", f"{c['id']}_Y", f"{c['id']}_Z"])
                f.write(",".join(cols) + "\n")
                
                # 타임 시리즈 데이터 추출 루프 (모든 시점 기록)
                for i, snap in enumerate(self.sim.snapshots):
                    t = snap['time']
                    # [WHTOOLS] start_time 필터링 제거 (0부터 전체 기록)
                    
                    row = [str(i), f"{t:.6f}"]
                    # 임시 데이터 객체에 물리 상태 복구
                    mujoco.mj_setState(self.sim.model, temp_data, snap['state'], mujoco.mjtState.mjSTATE_PHYSICS)
                    mujoco.mj_forward(self.sim.model, temp_data)
                    
                    rid = self.sim.root_id
                    base_pos = temp_data.xpos[rid]
                    base_mat = temp_data.xmat[rid].reshape(3,3)
                    
                    for c in corner_configs:
                        sx, sy, sz = c["s"]
                        loc = np.array([sx * w_input/2.0, sy * h_input/2.0, sz * d_input/2.0])
                        glob_pos = base_pos + base_mat @ loc
                        row.extend([f"{glob_pos[0]:.6f}", f"{glob_pos[1]:.6f}", f"{glob_pos[2]:.6f}"])
                    
                    f.write(",".join(row) + "\n")

            QtWidgets.QMessageBox.information(self, "Success", f"Chassis C1-C8 data exported to:\n{save_path}")
            self.accept()
        except Exception as e:
            error_msg = f"Execution failed: {e}"
            if self.sim:
                self.sim.log(f"❌ [Structural Dynamics] {error_msg}", level="error")
            QtWidgets.QMessageBox.warning(self, "Error", error_msg)
        finally:
            self.btn_do.setText("🚀 Do it")
            self.btn_do.setEnabled(True)

class ControlPanel(QMainWindow):
    """
    MuJoCo 시뮬레이션을 실시간으로 제어하기 위한 PySide6 메인 윈도우입니다.
    """
    def __init__(self, simulator):
        super().__init__()
        self.sim = simulator
        self.setWindowTitle("[WHTOOLS] Simulation Control Center")
        self.setMinimumWidth(500)
        self.setWindowFlags(Qt.WindowStaysOnTopHint) # 항상 위에 표시
        
        self._init_ui()
        
        # 상태 업데이트용 타이머 (100ms 간격)
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_status)
        self.timer.start(100)
        
        # [WHTOOLS] 모니터링 창 리스트 (복수 모달리스 지원)
        self.monitor_windows = []
        self._mujoco_aligned = False
        self._reloading = False  # _do_reload 재진입 방지 플래그
        self._last_mujoco_hwnd = None
        self._last_mujoco_rect = None

    def showEvent(self, event):
        """창이 표시될 때 정렬을 위해 부모 이벤트를 호출합니다."""
        super().showEvent(event)
        # [WHTOOLS] 기본 화면 구석 이동 로직 제거 (MuJoCo 정렬에 집중)

    def _init_ui(self):
        """현대적인 Dark Mode 스타일의 UI를 구성합니다."""
        # [WHTOOLS] 메뉴바 추가
        menubar = self.menuBar()
        model_menu = menubar.addMenu("📁 Model")
        
        act_new = model_menu.addAction("🆕 New Model Setup")
        act_new.triggered.connect(self._on_model_new)
        
        model_menu.addSeparator()
        
        act_load = model_menu.addAction("📂 Load Config (JSON)")
        act_load.triggered.connect(self._on_model_load)
        
        act_save = model_menu.addAction("💾 Save Config (JSON)")
        act_save.triggered.connect(self._on_model_save)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(15, 2, 15, 15)

        # Top Header (Logo + Status)
        header_layout = QHBoxLayout()
        header_layout.setSpacing(20)

        # 0. 로고 표시 (TVPackageMotionSim/sidebar_logo.png)
        logo_label = QLabel()
        logo_path = Path(__file__).parent.parent / "sidebar_logo.png"
        
        found_logo = False
        if logo_path.exists():
            pixmap = QPixmap(str(logo_path))
            if not pixmap.isNull():
                logo_label.setPixmap(pixmap.scaledToHeight(120, Qt.SmoothTransformation)) # Height 120
                logo_label.setToolTip("WHTOOLS Engine - Powered by MuJoCo & JAX")
                header_layout.addWidget(logo_label, 0, Qt.AlignVCenter)
                found_logo = True
        
        if not found_logo:
            # 대체 경로 시도
            alt_path = Path("sidebar_logo.png")
            if alt_path.exists():
                pixmap = QPixmap(str(alt_path))
                logo_label.setPixmap(pixmap.scaledToHeight(120, Qt.SmoothTransformation))
                header_layout.addWidget(logo_label, 0, Qt.AlignVCenter)

        # 1. 상태 표시 그룹
        status_group = QGroupBox("Simulation Status")
        status_layout = QVBoxLayout(status_group)
        
        # [WHTOOLS] 목표 시간 스핀박스 및 현재 시간 표시 레이아웃 변경 (초슬림 내장 화살표 단일 QDoubleSpinBox 적용)
        time_layout = QHBoxLayout()        
        self.lbl_time = QLabel("Time: 0.000 s")        
        
        # [WHTOOLS] Target 라벨 폰트 통일
        self.lbl_target = QLabel(" / ")
                
        self.spin_duration = QDoubleSpinBox()
        self.spin_duration.setRange(0.1, 100.0)
        self.spin_duration.setSingleStep(0.5)
        self.spin_duration.setDecimals(3)
        self.spin_duration.setSuffix(" s")
        self.spin_duration.setFixedWidth(90) # 내장 버튼 공간 고려 90px로 최적 세팅
        self.spin_duration.setFixedHeight(20) # 20px로 초슬림 다이어트!
        self.spin_duration.setButtonSymbols(QAbstractSpinBox.UpDownArrows) # 내장 콤팩트 화살표 사용!!
        self.spin_duration.setAlignment(Qt.AlignCenter) # 텍스트 중앙 정렬
        
        # 초기값 설정
        init_dur = self.sim.config.get("sim_duration", 1.0) if self.sim else 1.0
        self.spin_duration.setValue(init_dur)
        self.spin_duration.valueChanged.connect(self._on_duration_changed)
        
        time_layout.addWidget(self.lbl_time)
        time_layout.addWidget(self.lbl_target)
        time_layout.addWidget(self.spin_duration)
        time_layout.addStretch()  # [WHTOOLS] 스핀박스가 우측으로 밀리지 않고 / 바로 옆에 붙도록 밀착 정렬 적용
        
        status_layout.addLayout(time_layout)
        
        self.lbl_status = QLabel("Status: Ready")
        
        # [WHTOOLS] Step:과 Snapshots를 수직이 아닌 수평(가로) 1행에 나란히 배치하도록 개선
        step_snap_layout = QHBoxLayout()
        step_snap_layout.setSpacing(15)  # 적정 가로 여백 부여
        self.lbl_step = QLabel("Step: 0")
        self.lbl_snapshots = QLabel("Snapshots: 0")
        step_snap_layout.addWidget(self.lbl_step)
        step_snap_layout.addWidget(self.lbl_snapshots)
        step_snap_layout.addStretch()  # 왼쪽 밀착 정렬
        
        status_layout.addWidget(self.lbl_status)
        status_layout.addLayout(step_snap_layout)
        
        header_layout.addWidget(status_group, 1) # Status group expands
        main_layout.addLayout(header_layout)

        # 1-1. 카메라 시점 제어 그룹 (NEW)
        from functools import partial
        cam_group = QGroupBox("Camera Orientation (MuJoCo View)")
        cam_layout = QHBoxLayout(cam_group)        
        
        views = ["+X", "-X", "+Y", "-Y", "+Z", "-Z", "+ISO", "-ISO"]
        for v in views:
            btn = QPushButton(v)
            btn.setMinimumHeight(22)
            btn.setMinimumWidth(42)            
            btn.clicked.connect(partial(self._on_cam_view, v))
            cam_layout.addWidget(btn)
            
        main_layout.addWidget(cam_group)

        # 2. 재생 제어 그룹
        playback_group = QGroupBox("Playback Controls")
        playback_layout = QHBoxLayout(playback_group)
        
        self.btn_reset = QPushButton("🔄 Reset")
        self.btn_back = QPushButton("⏪ Back")
        self.btn_play = QPushButton("▶️ Play")
        self.btn_forward = QPushButton("⏩ Forward")
        
        for btn in [self.btn_reset, self.btn_back, self.btn_play, self.btn_forward]:
            btn.setFixedHeight(30)            
            playback_layout.addWidget(btn)
            
        self.btn_reset.clicked.connect(self._on_reset)
        self.btn_back.clicked.connect(self._on_back)
        self.btn_play.clicked.connect(self._on_play_pause)
        self.btn_forward.clicked.connect(self._on_forward)
        
        main_layout.addWidget(playback_group)

        # 2-1. 인터랙티브 효과 그룹 (NEW)
        fx_group = QGroupBox("Interactive")
        fx_layout = QHBoxLayout(fx_group)
        
        self.btn_slow = QPushButton("🐌 Slow Motion")
        self.btn_slow.setCheckable(True)
        self.btn_slow.clicked.connect(self._on_slow_motion)
        
        self.btn_rec = QPushButton("⏺️ Rec.Hist.")
        self.btn_rec.setCheckable(True)
        self.btn_rec.clicked.connect(self._on_record)
        
        self.btn_monitor = QPushButton("📈 Monitor")
        self.btn_monitor.clicked.connect(self._on_monitor)
        
        self.btn_struct = QPushButton("🏗️ Dyn.Loads")
        self.btn_struct.clicked.connect(self._on_structural_dynamics)
        
        for btn in [self.btn_slow, self.btn_rec, self.btn_monitor, self.btn_struct]:
            btn.setFixedHeight(30)
            fx_layout.addWidget(btn)
        
        main_layout.addWidget(fx_group)

        # 3. 타임라인 슬라이더 및 재생 속도 조절
        slider_group = QGroupBox("Timeline Navigation")
        slider_layout = QVBoxLayout(slider_group)
        
        info_layout = QHBoxLayout()        
        self.lbl_frame_info = QLabel("Frame: 0 / 0")
        self.lbl_frame_info.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)        
        info_layout.addWidget(self.lbl_frame_info)
        
        # [WHTOOLS] Speed Multiplier와 spinbox를 lbl_frame_info 우측에 나란히 배치하여 공간 극강 압축
        info_layout.addStretch()
        lbl_speed = QLabel("Speed Multiplier:")
        info_layout.addWidget(lbl_speed)
        
        self.spin_speed = QDoubleSpinBox()
        self.spin_speed.setRange(0.1, 10.0)
        self.spin_speed.setSingleStep(0.1)
        self.spin_speed.setValue(1.0)
        self.spin_speed.setDecimals(1)
        self.spin_speed.setSuffix("x")
        self.spin_speed.setFixedWidth(85) # 글자 잘림 방지 및 우측 가용 공간 활용을 위해 너비를 85px로 확장
        self.spin_speed.setFixedHeight(20) # 20px 초슬림 피팅
        self.spin_speed.setButtonSymbols(QAbstractSpinBox.UpDownArrows) # 내장 화살표 상속
        self.spin_speed.setAlignment(Qt.AlignCenter)
        self.spin_speed.valueChanged.connect(self._on_speed_changed)
        info_layout.addWidget(self.spin_speed)
        
        slider_layout.addLayout(info_layout)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self._on_slider_moved)
        slider_layout.addWidget(self.slider)

        main_layout.addWidget(slider_group)

        # 5. 유틸리티 버튼
        util_layout = QVBoxLayout()
        
        row1 = QHBoxLayout()
        self.btn_config = QPushButton("⚙️ Edit Configuration")
        self.btn_config.setMinimumHeight(35)
        self.btn_config.clicked.connect(self._on_open_config)
        row1.addWidget(self.btn_config)
        
        self.btn_camera = QPushButton("📸 Capture Camera")
        self.btn_camera.setMinimumHeight(35)
        self.btn_camera.clicked.connect(self._on_camera_export)
        row1.addWidget(self.btn_camera)

        # [WHTOOLS] 모션 상태 로그 버튼 추가
        self.btn_log_motion = QPushButton("📋 Log Motion")
        self.btn_log_motion.setMinimumHeight(35)
        self.btn_log_motion.clicked.connect(self._on_log_motion)
        row1.addWidget(self.btn_log_motion)
        
        util_layout.addLayout(row1)

        self.btn_reload_xml = QPushButton("📂 Open & Reload XML File")
        self.btn_reload_xml.setMinimumHeight(40)
        self.btn_reload_xml.setStyleSheet(f"background-color: {C_BTN_NAVY}; font-weight: bold; border: 1px solid {C_BTN_NAVY_BORDER};")
        self.btn_reload_xml.setToolTip("Select and load an external MuJoCo XML file.")
        self.btn_reload_xml.clicked.connect(self._on_reload_xml)
        util_layout.addWidget(self.btn_reload_xml)
        
        main_layout.addLayout(util_layout)

        # 스타일 시트 적용 (Premium Dark Theme)
        self.setStyleSheet(GLOBAL_QSS)

    def _update_status(self):
        """시뮬레이터의 현재 상태를 UI에 반영합니다."""
        try:
            self._align_with_mujoco_window()

            # reload 요청은 메인 스레드에서 전담 처리
            if self.sim.ctrl_reload_request:
                self._do_reload()
                return

            if self.sim is None or self.sim.data is None:
                return
                
            # 시간 및 스텝 정보 업데이트
            curr_time = self.sim.data.time
            target_time = self.sim.config.get("sim_duration", 1.0)
            
            self.lbl_time.setText(f"Time: {curr_time:.3f} s")
            
            # [WHTOOLS] 스핀박스 활성 포커스 없을 때 외부 최신 값 동기화
            if not self.spin_duration.hasFocus():
                self.spin_duration.blockSignals(True)
                self.spin_duration.setValue(target_time)
                self.spin_duration.blockSignals(False)
            self.lbl_step.setText(f"Step: {self.sim.step_idx}")
            snap_count = len(self.sim.snapshots)
            self.lbl_snapshots.setText(f"Snapshots: {snap_count}")

            # 데이터 수집 상태 판정
            if curr_time >= target_time:
                self.lbl_status.setText("Status: Collection Complete ✅")
                self.lbl_status.setStyleSheet(f"color: {C_STATUS_OK}; font-weight: bold;") # Green
            elif self.sim.ctrl_paused:
                self.lbl_status.setText("Status: Paused ⏸️")
                self.lbl_status.setStyleSheet(f"color: {C_STATUS_WARN};") # Yellow
            else:
                self.lbl_status.setText("Status: Data Collecting... ⏳")
                self.lbl_status.setStyleSheet(f"color: {C_STATUS_INFO};") # Blue
            
            # 슬라이더 범위 업데이트 및 재생 중 최신 위치 추적
            if snap_count > 0:
                self.slider.setRange(0, snap_count - 1)
                
                # 재생 중에는 슬라이더 핸들을 자동으로 맨 뒤로 이동
                if not self.sim.ctrl_paused:
                    self.slider.blockSignals(True)
                    try:
                        self.slider.setValue(snap_count - 1)
                    finally:
                        self.slider.blockSignals(False)

            # 재생 버튼 텍스트 업데이트
            self.btn_play.setText("▶️ Play" if self.sim.ctrl_paused else "⏸️ Pause")
            
            # 타임라인 정보 업데이트
            current_snap = self.slider.value()
            self.lbl_frame_info.setText(f"Frame: {current_snap} / {max(0, snap_count - 1)}")
            
            # 효과 버튼 상태 동기화 (시뮬레이터 내부 상태 -> UI)
            self.btn_slow.setChecked(self.sim.ctrl_slow_motion)
            self.btn_slow.setStyleSheet(f"background-color: {C_STATE_SLOW_MOTION};" if self.sim.ctrl_slow_motion else "")
            
            self.btn_rec.setChecked(self.sim.is_recording)
            self.btn_rec.setStyleSheet(f"background-color: {C_STATE_RECORDING}; color: {C_STATE_REC_TEXT}; font-weight: bold;" if self.sim.is_recording else "")
        except (AttributeError, RuntimeError, KeyboardInterrupt):
            pass

    def _align_with_mujoco_window(self):
        """
        MuJoCo 뷰어 창을 찾아 Control Center를 우측 상단에 정렬합니다.
        사용자가 수동으로 옮기기 전까지는 MuJoCo 창을 따라다닙니다.
        """
        if not sys.platform.startswith('win'):
            return

        try:
            # RECT 구조체 정의
            class RECT(ctypes.Structure):
                _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                            ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

            found_hwnd = [None]
            def callback(hwnd, lParam):
                if ctypes.windll.user32.IsWindowVisible(hwnd):
                    length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
                    if length > 0:
                        buff = ctypes.create_unicode_buffer(length + 1)
                        ctypes.windll.user32.GetWindowTextW(hwnd, buff, length + 1)
                        title = buff.value
                        if "MuJoCo" in title: 
                            found_hwnd[0] = hwnd
                            return False
                return True

            cb_func = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)(callback)
            ctypes.windll.user32.EnumWindows(cb_func, 0)

            if found_hwnd[0]:
                hwnd = found_hwnd[0]
                
                # [WHTOOLS] 새로운 MuJoCo 창 감지 시, 직전 창의 위치와 크기 복원 수행
                if self._last_mujoco_hwnd is not None and self._last_mujoco_hwnd != hwnd:
                    if self._last_mujoco_rect is not None:
                        old_rect = self._last_mujoco_rect
                        w = old_rect.right - old_rect.left
                        h = old_rect.bottom - old_rect.top
                        # 이전 위치와 크기로 새 창을 알아서 이동 및 조정
                        ctypes.windll.user32.MoveWindow(hwnd, old_rect.left, old_rect.top, w, h, True)
                        # 새 창이 정렬되었으므로 컨트롤 패널도 다시 새 위치에 맞춰 따라가도록 격발
                        self._mujoco_aligned = False
                
                self._last_mujoco_hwnd = hwnd
                
                # 실시간으로 현재 무조코 창의 실제 윈도우 RECT 캐싱 (사용자 최신 배치 상시 반영)
                current_rect = RECT()
                if ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(current_rect)):
                    self._last_mujoco_rect = current_rect

                rect = RECT()
                
                # [WHTOOLS] DWMWA_EXTENDED_FRAME_BOUNDS (9)를 사용하여 그림자 제외 실제 가시 영역 획득
                ctypes.windll.dwmapi.DwmGetWindowAttribute(
                    hwnd, 9, ctypes.byref(rect), ctypes.sizeof(rect)
                )
                
                # 현재 스크린의 배율(DPI Ratio) 획득
                dpi_ratio = self.screen().devicePixelRatio()
                
                # 물리적 좌표를 논리적 좌표로 변환 (Qt move용)
                m_right_logical = rect.right / dpi_ratio
                m_top_logical = rect.top / dpi_ratio
                
                # Control Center의 프레임 포함 크기 (이미 논리적 단위임)
                win_geo = self.frameGeometry()
                
                # 목표 위치: Control Center의 우측 상단이 MuJoCo의 우측 상단에 일치
                target_x = int(m_right_logical - win_geo.width())
                target_y = int(m_top_logical)
                
                # 처음 한 번만 정렬 (오차 허용 범위 2px)
                if not self._mujoco_aligned:
                    self.move(target_x, target_y)
                    self._mujoco_aligned = True

        except Exception:
            pass

    def _do_reload(self):
        """XML reload 요청을 메인 스레드에서 처리합니다.
        시뮬 스레드 종료 → viewer 닫기 → 모델 재로드 → viewer 재시작 → 새 시뮬 스레드 시작.
        """
        if self._reloading:
            return
        self._reloading = True

        self.sim.log("♻️ Reloading simulation (main thread)...")
        self.lbl_status.setText("Status: Reloading... ♻️")
        self.lbl_status.setStyleSheet(f"color: {C_STATE_ORANGE};")

        try:
            # 1. 물리 스레드가 reload 플래그를 감지해 while을 탈출할 때까지 대기
            if hasattr(self.sim, 'sim_thread') and self.sim.sim_thread.isRunning():
                self.sim.sim_thread.wait(3000)

            # 2. 기존 viewer 닫기
            self.sim.stop_viewer()

            # 3. 새 모델로 setup (ctrl_reload_xml_path / ctrl_reload_only_xml 플래그 이용)
            try:
                self.sim.setup()
            except Exception as e:
                self.sim.log(f"Reload failed during setup: {e}", level="error")
                self.sim.ctrl_reload_request = False
                return

            # 4. Passive viewer 재시작
            self.sim.start_viewer()
            self._mujoco_aligned = False

            # 5. 플래그 초기화 후 새 물리 스레드 시작
            self.sim.ctrl_paused = True
            self.sim.ctrl_reload_request = False
            self.sim._restart_sim_thread()
            self.sim.log("✅ Reload complete. Press Play to resume.")
        finally:
            self._reloading = False

    def closeEvent(self, event):
        """창이 닫힐 때 하위 모니터 창들도 모두 닫습니다."""
        if hasattr(self, 'monitor_windows'):
            for win in self.monitor_windows:
                try:
                    win.close()
                except:
                    pass
        self.timer.stop()
        event.accept()

    def _on_play_pause(self):
        self.sim.ctrl_paused = not self.sim.ctrl_paused

    def _on_reset(self):
        """Reset 버튼 클릭 시 시뮬레이션을 처음 상태(Frame 0)로 완전히 초기화합니다."""
        self.sim.ctrl_reset_request = True

    def _on_back(self):
        self.sim.ctrl_step_backward_request = True

    def _on_forward(self):
        self.sim.ctrl_step_forward_request = True

    def _on_slider_moved(self, value):
        self.sim.ctrl_jump_snapshot_idx = value
        # [WHTOOLS] 슬라이더 이동 시 즉시 모니터 창들의 마커를 업데이트하여 반응성 향상
        if hasattr(self, 'monitor_windows'):
            for win in self.monitor_windows:
                if win.isVisible():
                    win._update_plot()

    def _on_speed_changed(self, value):
        self.sim.ctrl_speed_multiplier = value

    def _on_duration_changed(self, value):
        """
        사용자가 Target Duration 스핀 박스를 수정했을 때 호출되어
        시뮬레이션 설정의 sim_duration을 실시간으로 업데이트합니다.

        Parameters
        ----------
        value : float
            스핀 박스를 통해 수정된 새로운 시뮬레이션 목표 시간 (초 단위)
        """
        if self.sim and self.sim.config:
            self.sim.config["sim_duration"] = value

    def _on_slow_motion(self, checked):
        self.sim.ctrl_slow_motion = checked

    def _on_record(self, checked):
        self.sim.is_recording = checked

    def _on_camera_export(self):
        self.sim.ctrl_export_camera = True

    def _on_log_motion(self):
        """현재 시점의 강체 거동 정보를 로그로 출력합니다."""
        if not self.sim.rot_axis_hist:
            self.sim.log("⚠️ No motion data available yet.", level="warning")
            return
            
        axis = self.sim.rot_axis_hist[-1]
        speed = self.sim.rot_speed_hist[-1]
        tvel = self.sim.trans_vel_hist[-1]
        tvel_res = self.sim.trans_vel_res_hist[-1]
        
        import numpy as np
        azi = np.degrees(np.arctan2(axis[1], axis[0]))
        ele = np.degrees(np.arcsin(np.clip(axis[2], -1, 1)))
        
        msg = (
            f"\n[📊 Current Motion State at Time {self.sim.data.time:.4f}s]\n"
            f"- Rotation Axis: [{axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f}] (Azi: {azi:.1f}°, Ele: {ele:.1f}°)\n"
            f"- Rotation Speed: {speed:.4f} rad/s\n"
            f"- Trans. Velocity: [{tvel[0]:.4f}, {tvel[1]:.4f}, {tvel[2]:.4f}] (Resultant: {tvel_res:.4f} m/s)"
        )
        self.sim.log(msg, level="info")

    def _on_cam_view(self, view_name):
        """MuJoCo 뷰어의 시점 전환 요청을 시뮬레이터로 전달합니다."""
        if hasattr(self.sim, 'ctrl_cam_view'):
            self.sim.ctrl_cam_view = view_name
        else:
            # 동적 속성으로라도 추가하여 엔진에서 감지할 수 있게 함
            self.sim.ctrl_cam_view = view_name

    def _on_monitor(self):
        """실시간 모니터링 설정 다이얼로그를 띄우고 그래프 윈도우를 생성합니다."""
        from .whts_monitor import MonitorConfigDialog, RealTimeMonitorWindow
        dialog = MonitorConfigDialog(self)
        if dialog.exec():
            config = dialog.get_config()
            
            # 새 모니터 창 생성 및 리스트 관리
            win = RealTimeMonitorWindow(self.sim, config)
            self.monitor_windows.append(win)
            
            # 창이 닫히면 리스트에서 제거하여 메모리 관리
            win.setAttribute(Qt.WA_DeleteOnClose)
            win.destroyed.connect(lambda: self.monitor_windows.remove(win) if win in self.monitor_windows else None)
            win.show()

    def _on_structural_dynamics(self):
        """구조 해석용 데이터 추출 다이얼로그를 실행합니다."""
        dialog = StructuralDynamicsDialog(self, self.sim)
        dialog.exec()

    def _on_open_config(self):
        """XML 라이브 에디터를 엽니다."""
        model_path = self.sim.config.get("model_path")
        if not model_path or not os.path.exists(model_path):
            QtWidgets.QMessageBox.warning(self, "Error", "현재 로드된 모델 XML 파일을 찾을 수 없습니다.")
            return

        try:
            with open(model_path, "r", encoding="utf-8") as f:
                xml_content = f.read()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"파일을 읽는 중 오류가 발생했습니다: {e}")
            return

        dialog = XMLEditorDialog(self, xml_content, model_path)
        if dialog.exec() == QtWidgets.QDialog.Accepted:
            new_xml = dialog.get_xml_content()
            self.sim.log("📝 User modified XML. Triggering Reload...")
            self.sim.reload_xml(xml_string=new_xml)

    def _on_model_new(self):
        """새로운 모델 설정을 위한 다이얼로그를 엽니다."""
        dlg = ModelSetupDialog(self, self.sim)
        dlg.exec()

    def _on_model_load(self):
        """저장된 JSON 설정 파일을 불러옵니다."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Config", "", "JSON Files (*.json)")
        if path:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    new_cfg = json.load(f)
                self.sim.config.update(new_cfg)
                self.sim.ctrl_reload_request = True
                QtWidgets.QMessageBox.information(self, "Success", "Configuration loaded and model reload requested.")
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Error", f"Failed to load config: {e}")

    def _on_model_save(self):
        """현재 시뮬레이션 설정을 JSON 파일로 저장합니다."""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Config", "config_backup.json", "JSON Files (*.json)")
        if path:
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(self.sim.config, f, indent=4, ensure_ascii=False)
                QtWidgets.QMessageBox.information(self, "Success", f"Config saved to:\n{path}")
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Error", f"Failed to save config: {e}")

    def _on_reload_xml(self):
        from PySide6.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select MuJoCo Simulation XML",
            str(self.sim.output_dir), "MuJoCo XML (*.xml);;All Files (*)"
        )
        if file_path:
            self.sim.reload_xml(file_path)

def launch_control_panel(simulator):
    """외부에서 컨트롤 패널을 실행하기 위한 진입점입니다."""
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    apply_app_theme(app)
    panel = ControlPanel(simulator)
    panel.show()
    return app, panel

if __name__ == "__main__":
    class MockSim:
        def __init__(self):
            self.data = type('obj', (object,), {'time': 0.0})
            self.step_idx = 0
            self.snapshots = []
            self.ctrl_paused = False
            self.ctrl_step_forward_request = False
            self.ctrl_step_backward_request = False
            self.ctrl_jump_snapshot_idx = -1
            self.ctrl_speed_multiplier = 1.0
            def log(self, t, level="info"): print(f"[{level}] {t}")
            self.log = log
            
    app, panel = launch_control_panel(MockSim())
    sys.exit(app.exec())
