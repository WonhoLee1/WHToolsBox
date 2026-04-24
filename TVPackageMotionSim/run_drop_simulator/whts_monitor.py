# -*- coding: utf-8 -*-
"""
[WHTOOLS] Real-Time Simulation Monitor
시뮬레이션 진행 중 주요 포인트(Corner)의 궤적 및 속도를 실시간으로 모니터링하는 모듈입니다.
"""

import sys
import numpy as np
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QCheckBox, 
    QRadioButton, QButtonGroup, QPushButton, QLabel, 
    QFrame, QWidget, QScrollArea
)
from PySide6.QtCore import Qt, QTimer
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import koreanize_matplotlib

class MonitorConfigDialog(QDialog):
    """모니터링 대상 및 항목을 설정하는 다이얼로그입니다."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📈 Monitor Configuration")
        self.setMinimumWidth(400)
        self._init_ui()
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # 1. Corners Selection (8 points)
        layout.addWidget(QLabel("<b>1. Select Corners to Monitor:</b>"))
        corner_grid = QVBoxLayout()
        self.corner_checks = []
        corner_names = [
            "Front-Bottom-Left", "Front-Bottom-Right", "Front-Top-Left", "Front-Top-Right",
            "Rear-Bottom-Left", "Rear-Bottom-Right", "Rear-Top-Left", "Rear-Top-Right"
        ]
        for i, name in enumerate(corner_names):
            cb = QCheckBox(f"Point {i}: {name}")
            if i == 0: cb.setChecked(True) # 기본값
            self.corner_checks.append(cb)
            corner_grid.addWidget(cb)
        layout.addLayout(corner_grid)
        
        layout.addWidget(self._create_separator())
        
        # 2. Data Type Selection (Position / Velocity)
        layout.addWidget(QLabel("<b>2. Select Data Type:</b>"))
        type_layout = QHBoxLayout()
        self.radio_pos = QRadioButton("Position (m)")
        self.radio_vel = QRadioButton("Velocity (m/s)")
        self.radio_pos.setChecked(True)
        self.type_group = QButtonGroup(self)
        self.type_group.addButton(self.radio_pos)
        self.type_group.addButton(self.radio_vel)
        type_layout.addWidget(self.radio_pos)
        type_layout.addWidget(self.radio_vel)
        layout.addLayout(type_layout)
        
        layout.addWidget(self._create_separator())
        
        # 3. Axes Selection (X, Y, Z)
        layout.addWidget(QLabel("<b>3. Select Axes (Each creates a subplot):</b>"))
        axis_layout = QHBoxLayout()
        self.check_x = QCheckBox("X-Axis")
        self.check_y = QCheckBox("Y-Axis")
        self.check_z = QCheckBox("Z-Axis")
        self.check_z.setChecked(True) # 기본값 Z
        axis_layout.addWidget(self.check_x)
        axis_layout.addWidget(self.check_y)
        axis_layout.addWidget(self.check_z)
        layout.addLayout(axis_layout)
        
        layout.addWidget(self._create_separator())
        
        # Action Buttons
        btn_layout = QHBoxLayout()
        self.btn_ok = QPushButton("Launch Monitor")
        self.btn_ok.setStyleSheet("background-color: #0078d7; color: white; font-weight: bold; padding: 8px;")
        self.btn_ok.clicked.connect(self.accept)
        btn_layout.addWidget(self.btn_ok)
        layout.addLayout(btn_layout)

    def _create_separator(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: #444;")
        return line

    def get_config(self):
        """설정된 값을 딕셔너리로 반환합니다."""
        return {
            "selected_corners": [i for i, cb in enumerate(self.corner_checks) if cb.isChecked()],
            "data_type": "pos" if self.radio_pos.isChecked() else "vel",
            "axes": [ax for ax, cb in [('X', self.check_x), ('Y', self.check_y), ('Z', self.check_z)] if cb.isChecked()]
        }

class RealTimeMonitorWindow(QWidget):
    """실시간 그래프를 표시하는 모달리스 윈도우입니다."""
    def __init__(self, sim, config):
        super().__init__()
        self.sim = sim
        self.config = config
        self.setWindowTitle(f"WHTS Real-Time Monitor - {config['data_type'].upper()}")
        self.resize(400, 300)
        self.setWindowFlags(Qt.Window) # 모달리스
        
        self._init_ui()
        
        # 타이머 설정 (UI 업데이트)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_plot)
        self.timer.start(50) # 20 FPS
        
    def _init_ui(self):
        self.setStyleSheet("background-color: #121212; color: #e0e0e0;")
        layout = QVBoxLayout(self)
        
        # Matplotlib Figure 생성
        self.fig = Figure(figsize=(4, 3), facecolor='#121212')
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        
        num_subplots = len(self.config['axes'])
        if num_subplots == 0:
            num_subplots = 1 # Fallback
            
        self.axes_plots = []
        for i, axis_label in enumerate(self.config['axes']):
            ax = self.fig.add_subplot(num_subplots, 1, i + 1)
            ax.set_facecolor('#1e1e1e')
            ax.set_title(f"{self.config['data_type'].capitalize()} - {axis_label} Axis", color='white', fontsize=10)
            ax.tick_params(axis='both', colors='gray', labelsize=8)
            ax.grid(True, color='#333', linestyle='--')
            
            # X축 범위 설정 (시뮬레이션 종료 시간까지)
            duration = self.sim.config.get("sim_duration", 1.0)
            ax.set_xlim(0, duration)
            
            # 그래프 라인 객체 생성 (각 선택된 코너별로)
            lines = {}
            for c_idx in self.config['selected_corners']:
                line, = ax.plot([], [], label=f"P{c_idx}", linewidth=1.5)
                lines[c_idx] = line
            
            if i == 0:
                ax.legend(loc='upper right', fontsize=7, facecolor='#121212', edgecolor='#444', labelcolor='white')
                
            self.axes_plots.append({"ax": ax, "lines": lines, "axis_idx": "XYZ".index(axis_label)})
            
        self.fig.tight_layout()

    def _update_plot(self):
        """시뮬레이터의 데이터를 가져와 그래프를 갱신합니다."""
        if not self.sim or not hasattr(self.sim, 'time_history'):
            return
            
        times = self.sim.time_history
        if not times:
            return
            
        # 데이터 소스 선택
        if self.config['data_type'] == "pos":
            data_source = self.sim.corner_pos_hist
        else:
            data_source = self.sim.corner_vel_hist
            
        if not data_source:
            return
            
        # 비동기 업데이트 중 길이 불일치 방지를 위해 최소 길이에 맞춰 슬라이싱 및 복사
        min_len = min(len(times), len(data_source))
        if min_len <= 1: # 데이터가 충분하지 않으면 스킵
            return
            
        try:
            # 명시적으로 복사본(numpy array) 생성하여 원본 리스트와의 참조 끊기
            times_np = np.array(times[:min_len])
            data_arr = np.array(data_source[:min_len])
            
            for plot_data in self.axes_plots:
                ax_idx = plot_data['axis_idx']
                lines = plot_data['lines']
                
                for c_idx, line in lines.items():
                    y_data = data_arr[:, c_idx, ax_idx]
                    # x, y 데이터의 최종 길이 확인 후 설정
                    if len(times_np) == len(y_data):
                        line.set_data(times_np, y_data)
                
                # Y축 범위 자동 조절
                plot_data['ax'].relim()
                plot_data['ax'].autoscale_view(scalex=False, scaley=True)
        except (ValueError, IndexError, RuntimeError):
            # 비동기 데이터 접근 시 발생할 수 있는 일시적 오류 무시
            return
            
        self.canvas.draw()

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)
