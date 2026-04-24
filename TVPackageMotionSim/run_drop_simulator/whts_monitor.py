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
    QFrame, QWidget, QScrollArea, QMenuBar, QMenu,
    QFileDialog, QApplication, QTabWidget, QSpinBox, QDoubleSpinBox
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QIcon
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import koreanize_matplotlib

class MonitorConfigDialog(QDialog):
    """모니터링 대상 및 항목을 설정하는 다이얼로그입니다."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📈 Monitor Configuration")
        self.setMinimumWidth(400)
        # Remove Window Icon using Flags (Safe Version)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setWindowIcon(QIcon())
        self._init_ui()
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # --- Tab 1: Corner Kinematics ---
        tab_kin = QWidget()
        kin_layout = QVBoxLayout(tab_kin)
        
        # 1. Corners Selection
        kin_layout.addWidget(QLabel("<b>1. Select Corners:</b>"))
        sel_btn_layout = QHBoxLayout()
        btn_sel_all = QPushButton("Select All")
        btn_desel_all = QPushButton("Deselect All")
        btn_sel_all.clicked.connect(lambda: self._set_all_corners(True))
        btn_desel_all.clicked.connect(lambda: self._set_all_corners(False))
        sel_btn_layout.addWidget(btn_sel_all)
        sel_btn_layout.addWidget(btn_desel_all)
        kin_layout.addLayout(sel_btn_layout)

        corner_grid = QVBoxLayout()
        self.corner_checks = []
        corner_names = [
            "Front-Bottom-Left", "Front-Bottom-Right", "Front-Top-Left", "Front-Top-Right",
            "Rear-Bottom-Left", "Rear-Bottom-Right", "Rear-Top-Left", "Rear-Top-Right"
        ]
        for i, name in enumerate(corner_names):
            cb = QCheckBox(f"P{i}: {name}")
            if i == 0: cb.setChecked(True)
            self.corner_checks.append(cb)
            corner_grid.addWidget(cb)
        kin_layout.addLayout(corner_grid)
        
        kin_layout.addWidget(self._create_separator())
        
        # 2. Data Type
        kin_layout.addWidget(QLabel("<b>2. Data Type:</b>"))
        type_layout = QHBoxLayout()
        self.radio_pos = QRadioButton("Position")
        self.radio_vel = QRadioButton("Velocity")
        self.radio_acc = QRadioButton("Acceleration")
        self.radio_pos.setChecked(True)
        self.type_group = QButtonGroup(self)
        for r in [self.radio_pos, self.radio_vel, self.radio_acc]: self.type_group.addButton(r)
        type_layout.addWidget(self.radio_pos); type_layout.addWidget(self.radio_vel); type_layout.addWidget(self.radio_acc)
        kin_layout.addLayout(type_layout)
        
        kin_layout.addWidget(self._create_separator())
        
        # 3. Axes
        kin_layout.addWidget(QLabel("<b>3. Select Axes:</b>"))
        axis_layout = QHBoxLayout()
        self.check_x = QCheckBox("X"); self.check_y = QCheckBox("Y"); self.check_z = QCheckBox("Z")
        self.check_res = QCheckBox("Resultant") # [WHTOOLS] Resultant 추가
        self.check_z.setChecked(True)
        for c in [self.check_x, self.check_y, self.check_z, self.check_res]: axis_layout.addWidget(c)
        kin_layout.addLayout(axis_layout)
        
        self.tabs.addTab(tab_kin, "📍 Corner Kinematics")
        
        # --- Tab 2: Dynamics & Physics ---
        tab_dyn = QWidget()
        dyn_layout = QVBoxLayout(tab_dyn)
        
        dyn_layout.addWidget(QLabel("<b>Scalar & Physics Metrics:</b>"))
        self.check_impact = QCheckBox("Corner Ground Impact Force (N)")
        self.check_drag = QCheckBox("Total Air Drag Force (N)")
        self.check_squeeze = QCheckBox("Air Squeeze Film Force (N)")
        
        # [WHTOOLS] 강체 거동 추가
        self.check_rot_axis = QCheckBox("Rotation Axis (Azimuth/Elevation)")
        self.check_rot_speed = QCheckBox("Rotation Speed (rad/s)")
        self.check_trans_vel_xyz = QCheckBox("Translational Velocity (X/Y/Z)")
        self.check_trans_vel_res = QCheckBox("Translational Velocity (Resultant)")
        
        for c in [self.check_impact, self.check_drag, self.check_squeeze, 
                  self.check_rot_axis, self.check_rot_speed, 
                  self.check_trans_vel_xyz, self.check_trans_vel_res]:
            dyn_layout.addWidget(c)
            
        dyn_layout.addStretch()
        self.tabs.addTab(tab_dyn, "⚡ Dynamics & Physics")

        # --- Tab 3: Window Management [NEW] ---
        tab_win = QWidget()
        win_layout = QVBoxLayout(tab_win)
        
        win_layout.addWidget(QLabel("<b>Active Monitor Windows:</b>"))
        self.win_list = QLabel("None")
        self.win_list.setStyleSheet("color: #888; font-style: italic; background: #222; padding: 5px;")
        win_layout.addWidget(self.win_list)
        
        # Update window list if parent has them
        if hasattr(self.parent(), 'monitor_windows'):
            wins = self.parent().monitor_windows
            if wins:
                titles = [w.windowTitle() for w in wins]
                self.win_list.setText("\n".join(titles))
                self.win_list.setStyleSheet("color: #ccc; background: #222; padding: 5px;")

        win_layout.addWidget(self._create_separator())
        win_layout.addWidget(QLabel("<b>Grid Layout (Top-Right Aligned):</b>"))
        
        grid_form = QHBoxLayout()
        self.spin_cols = QSpinBox()
        self.spin_cols.setRange(1, 10); self.spin_cols.setValue(2)
        grid_form.addWidget(QLabel("Cols:"))
        grid_form.addWidget(self.spin_cols)
        
        self.spin_w = QSpinBox(); self.spin_w.setRange(200, 2000); self.spin_w.setValue(400)
        self.spin_h = QSpinBox(); self.spin_h.setRange(150, 2000); self.spin_h.setValue(300)
        grid_form.addWidget(QLabel("W:"))
        grid_form.addWidget(self.spin_w)
        grid_form.addWidget(QLabel("H:"))
        grid_form.addWidget(self.spin_h)
        win_layout.addLayout(grid_form)
        
        btn_layout_apply = QPushButton("📐 Apply Grid Layout")
        btn_layout_apply.clicked.connect(self._apply_grid_layout)
        win_layout.addWidget(btn_layout_apply)
        
        win_layout.addStretch()
        self.tabs.addTab(tab_win, "🖥️ Window Mgmt")
        
        # Action Buttons
        btn_layout = QHBoxLayout()
        self.btn_ok = QPushButton("Launch Monitor")
        self.btn_ok.setStyleSheet("background-color: #0078d7; color: white; font-weight: bold; padding: 10px;")
        self.btn_ok.clicked.connect(self.accept)
        btn_layout.addWidget(self.btn_ok)
        layout.addLayout(btn_layout)

    def _set_all_corners(self, state: bool):
        for cb in self.corner_checks:
            cb.setChecked(state)

    def _apply_grid_layout(self):
        """현재 열린 모든 모니터 창을 화면 우측 상단 기준으로 그리드 정렬합니다."""
        if not hasattr(self.parent(), 'monitor_windows'):
            return
            
        windows = self.parent().monitor_windows
        if not windows:
            return
            
        cols = self.spin_cols.value()
        w = self.spin_w.value()
        h = self.spin_h.value()
        
        screen = QApplication.primaryScreen().geometry()
        start_x = screen.right() - w
        start_y = screen.top() + 50 # 작업표시줄/타이틀바 여유
        
        for i, win in enumerate(windows):
            r = i // cols
            c = i % cols
            
            target_x = start_x - (c * w)
            target_y = start_y + (r * h)
            
            win.resize(w, h)
            win.move(target_x, target_y)
            # Tight layout 및 다시 그리기 강제
            if hasattr(win, 'fig'):
                win.fig.tight_layout()
                win.canvas.draw()
                
    def _create_separator(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: #444;")
        return line

    def get_config(self):
        """설정된 값을 딕셔너리로 반환합니다."""
        dtype = "pos"
        if self.radio_vel.isChecked(): dtype = "vel"
        elif self.radio_acc.isChecked(): dtype = "acc"
        
        return {
            "selected_corners": [i for i, cb in enumerate(self.corner_checks) if cb.isChecked()],
            "data_type": dtype,
            "axes": [ax for ax, cb in [('X', self.check_x), ('Y', self.check_y), ('Z', self.check_z), ('Res', self.check_res)] if cb.isChecked()],
            "physics": {
                "impact": self.check_impact.isChecked(),
                "drag": self.check_drag.isChecked(),
                "squeeze": self.check_squeeze.isChecked(),
                "rot_axis": self.check_rot_axis.isChecked(),
                "rot_speed": self.check_rot_speed.isChecked(),
                "trans_vel_xyz": self.check_trans_vel_xyz.isChecked(),
                "trans_vel_res": self.check_trans_vel_res.isChecked()
            }
        }

class RealTimeMonitorWindow(QWidget):
    """실시간 그래프를 표시하는 모달리스 윈도우입니다."""
    def __init__(self, sim, config):
        super().__init__()
        self.sim = sim
        self.config = config
        self.setWindowTitle(f"📈 WHTS Real-Time Monitor - {config['data_type'].upper()}")
        self.resize(600, 450)
        # Remove Window Icon using Flags (Safe Version)
        self.setWindowFlags(Qt.Window | Qt.CustomizeWindowHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        self.setWindowIcon(QIcon())
        
        self._init_ui()
        
        # 타이머 설정 (UI 업데이트)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_plot)
        self.timer.start(50) # 20 FPS
        
    def _init_ui(self):
        self.setStyleSheet("background-color: #121212; color: #e0e0e0;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Matplotlib Figure & Canvas
        self.fig = Figure(figsize=(5, 4), facecolor='#121212')
        self.canvas = FigureCanvas(self.fig)
        
        # Navigation Toolbar (Hidden, used for menu actions)
        self.nav_toolbar = NavigationToolbar(self.canvas, self)
        self.nav_toolbar.hide()
        
        # Menu Bar
        self.menu_bar = QMenuBar(self)
        
        # 1. Tool Menu (Export)
        self.tool_menu = self.menu_bar.addMenu("🛠️ Tool")
        
        self.act_save_csv = QAction("Save (CSV)", self)
        self.act_save_tab = QAction("Save (Tab)", self)
        self.act_save_md  = QAction("Save (MD Table)", self)
        
        self.act_to_clipboard = QAction("To clipboard", self)
        self.act_to_file = QAction("To a file", self)
        self.act_to_clipboard.setCheckable(True)
        self.act_to_file.setCheckable(True)
        self.act_to_clipboard.setChecked(True)
        
        # Group targets to make them exclusive
        self.target_group = QButtonGroup(self) 
        # Note: QAction is not a QAbstractButton, so we handle exclusivity manually
        self.act_to_clipboard.triggered.connect(lambda: self._set_target_exclusive(True))
        self.act_to_file.triggered.connect(lambda: self._set_target_exclusive(False))
        
        self.tool_menu.addAction(self.act_save_csv)
        self.tool_menu.addAction(self.act_save_tab)
        self.tool_menu.addAction(self.act_save_md)
        self.tool_menu.addSeparator()
        self.tool_menu.addAction(self.act_to_clipboard)
        self.tool_menu.addAction(self.act_to_file)
        
        self.act_save_csv.triggered.connect(lambda: self._export_data("csv"))
        self.act_save_tab.triggered.connect(lambda: self._export_data("tab"))
        self.act_save_md.triggered.connect(lambda: self._export_data("md"))
        
        # 2. View Menu (Matplotlib Navigation)
        self.view_menu = self.menu_bar.addMenu("🔍 View")
        self.act_home = QAction("🏠 Home", self)
        self.act_back = QAction("⬅️ Back", self)
        self.act_forward = QAction("➡️ Forward", self)
        self.act_pan = QAction("🖐️ Pan", self)
        self.act_pan.setCheckable(True)
        self.act_zoom = QAction("🔍 Zoom", self)
        self.act_zoom.setCheckable(True)
        self.act_config = QAction("⚙️ Configure Subplots", self)
        self.act_tight = QAction("📐 Tight Layout", self)
        
        self.act_home.triggered.connect(self.nav_toolbar.home)
        self.act_back.triggered.connect(self.nav_toolbar.back)
        self.act_forward.triggered.connect(self.nav_toolbar.forward)
        self.act_pan.triggered.connect(self.nav_toolbar.pan)
        self.act_zoom.triggered.connect(self.nav_toolbar.zoom)
        self.act_config.triggered.connect(self.nav_toolbar.configure_subplots)
        self.act_tight.triggered.connect(lambda: (self.fig.tight_layout(), self.canvas.draw()))
        
        self.view_menu.addAction(self.act_home)
        self.view_menu.addAction(self.act_back)
        self.view_menu.addAction(self.act_forward)
        self.view_menu.addSeparator()
        self.view_menu.addAction(self.act_pan)
        self.view_menu.addAction(self.act_zoom)
        self.view_menu.addSeparator()
        self.view_menu.addAction(self.act_tight)
        self.view_menu.addAction(self.act_config)
        
        layout.setMenuBar(self.menu_bar)
        
        # Add Canvas to layout
        layout.addWidget(self.canvas)
        
        self.axes_plots = []
        
        # 1. Corner Kinematics Plots
        for i, axis_label in enumerate(self.config['axes']):
            ax = self.fig.add_subplot(1, 1, 1) # Placeholder
            self.axes_plots.append({"type": "kin", "ax": ax, "label": axis_label, "axis_idx": "XYZRes".find(axis_label)})

        # 2. Physics Plots
        if self.config['physics']['impact']:
            self.axes_plots.append({"type": "impact", "ax": None, "label": "Impact Force (N)"})
        if self.config['physics']['drag']:
            self.axes_plots.append({"type": "drag", "ax": None, "label": "Air Drag (N)"})
        if self.config['physics']['squeeze']:
            self.axes_plots.append({"type": "squeeze", "ax": None, "label": "Air Squeeze (N)"})
        if self.config['physics']['rot_axis']:
            self.axes_plots.append({"type": "rot_axis", "ax": None, "label": "Rot Axis (Azimuth/Elevation)"})
        if self.config['physics']['rot_speed']:
            self.axes_plots.append({"type": "rot_speed", "ax": None, "label": "Rot Speed (rad/s)"})
        if self.config['physics']['trans_vel_xyz']:
            self.axes_plots.append({"type": "trans_vel_xyz", "ax": None, "label": "Trans Vel (X/Y/Z)"})
        if self.config['physics']['trans_vel_res']:
            self.axes_plots.append({"type": "trans_vel_res", "ax": None, "label": "Trans Vel (Res)"})

        num_subplots = len(self.axes_plots)
        self.fig.clear()
        
        for i, plot_info in enumerate(self.axes_plots):
            ax = self.fig.add_subplot(num_subplots, 1, i + 1)
            ax.set_facecolor('#1e1e1e')
            ax.set_title(plot_info['label'], color='white', fontsize=9)
            ax.tick_params(axis='both', colors='gray', labelsize=8)
            ax.grid(True, color='#333', linestyle='--')
            
            # X축 범위
            duration = self.sim.config.get("sim_duration", 1.0)
            ax.set_xlim(0, duration)
            
            lines = {}
            if plot_info['type'] in ["kin", "impact"]:
                for c_idx in self.config['selected_corners']:
                    line, = ax.plot([], [], label=f"P{c_idx}", linewidth=1.2)
                    lines[c_idx] = line
                if i == 0:
                    ax.legend(loc='upper right', fontsize=6, facecolor='#121212', labelcolor='white')
            elif plot_info['type'] == "rot_axis":
                line_azi, = ax.plot([], [], label="Azimuth", color="orange", linewidth=1.2)
                line_ele, = ax.plot([], [], label="Elevation", color="magenta", linewidth=1.2)
                lines["azi"] = line_azi
                lines["ele"] = line_ele
                ax.legend(loc='upper right', fontsize=6, facecolor='#121212', labelcolor='white')
                ax.set_ylabel("Degree (°)", color='gray', fontsize=8)
            elif plot_info['type'] == "trans_vel_xyz":
                line_x, = ax.plot([], [], label="VX", color="#ff4444", linewidth=1.2)
                line_y, = ax.plot([], [], label="VY", color="#44ff44", linewidth=1.2)
                line_z, = ax.plot([], [], label="VZ", color="#4444ff", linewidth=1.2)
                lines["x"] = line_x; lines["y"] = line_y; lines["z"] = line_z
                ax.legend(loc='upper right', fontsize=6, facecolor='#121212', labelcolor='white')
            else: # Global scalars (Drag, Squeeze, RotSpeed, TransVelRes)
                line, = ax.plot([], [], color='cyan', linewidth=1.5)
                lines["total"] = line
            
            # [WHTOOLS] 현재 시점을 표시하는 수직선 추가
            marker = ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
            
            plot_info['ax'] = ax
            plot_info['lines'] = lines
            plot_info['marker'] = marker
            
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
        elif self.config['data_type'] == "vel":
            data_source = self.sim.corner_vel_hist
        else: # acc
            data_source = self.sim.corner_acc_hist
            
        if not data_source:
            return
            
        try:
            times_np = np.array(times[:min_len])
            
            for plot_info in self.axes_plots:
                lines = plot_info['lines']
                
                if plot_info['type'] == "kin":
                    # Kinematics data selection
                    if self.config['data_type'] == "pos":
                        ds = self.sim.corner_pos_hist if plot_info['axis_idx'] < 3 else self.sim.corner_pos_res_hist
                    elif self.config['data_type'] == "vel":
                        ds = self.sim.corner_vel_hist if plot_info['axis_idx'] < 3 else self.sim.corner_vel_res_hist
                    else: # acc
                        ds = self.sim.corner_acc_hist if plot_info['axis_idx'] < 3 else self.sim.corner_acc_res_hist
                    
                    data_arr = np.array(ds[:min_len])
                    for c_idx, line in lines.items():
                        y_data = data_arr[:, c_idx, plot_info['axis_idx']] if plot_info['axis_idx'] < 3 else data_arr[:, c_idx]
                        line.set_data(times_np, y_data)
                        
                elif plot_info['type'] == "impact":
                    data_arr = np.array(self.sim.corner_impact_hist[:min_len])
                    for c_idx, line in lines.items():
                        line.set_data(times_np, data_arr[:, c_idx])
                        
                elif plot_info['type'] == "drag":
                    line = lines["total"]
                    line.set_data(times_np, np.array(self.sim.air_drag_hist[:min_len]))
                    
                elif plot_info['type'] == "squeeze":
                    line = lines["total"]
                    line.set_data(times_np, np.array(self.sim.air_squeeze_hist[:min_len]))
                    
                elif plot_info['type'] == "rot_axis":
                    data_arr = np.array(self.sim.rot_axis_hist[:min_len])
                    # Azi = atan2(y, x), Ele = asin(z)
                    azi = np.degrees(np.arctan2(data_arr[:, 1], data_arr[:, 0]))
                    ele = np.degrees(np.arcsin(np.clip(data_arr[:, 2], -1, 1)))
                    lines["azi"].set_data(times_np, azi)
                    lines["ele"].set_data(times_np, ele)
                    
                elif plot_info['type'] == "rot_speed":
                    line = lines["total"]
                    line.set_data(times_np, np.array(self.sim.rot_speed_hist[:min_len]))
                    
                elif plot_info['type'] == "trans_vel_xyz":
                    data_arr = np.array(self.sim.trans_vel_hist[:min_len])
                    lines["x"].set_data(times_np, data_arr[:, 0])
                    lines["y"].set_data(times_np, data_arr[:, 1])
                    lines["z"].set_data(times_np, data_arr[:, 2])
                    
                elif plot_info['type'] == "trans_vel_res":
                    line = lines["total"]
                    line.set_data(times_np, np.array(self.sim.trans_vel_res_hist[:min_len]))

                # 현재 시점 마커 업데이트 (슬라이더 이동 시 동기화)
                plot_info['marker'].set_xdata([self.sim.data.time])

                # Y축 범위 자동 조절
                plot_info['ax'].relim()
                plot_info['ax'].autoscale_view(scalex=False, scaley=True)
        except Exception:
            return
            
        self.canvas.draw()

    def _set_target_exclusive(self, to_clipboard: bool):
        self.act_to_clipboard.setChecked(to_clipboard)
        self.act_to_file.setChecked(not to_clipboard)

    def _export_data(self, mode: str):
        """현재까지 수집된 데이터를 익스포트합니다."""
        if not self.sim or not self.sim.time_history:
            return

        # 데이터 헤더 구성
        header = ["Frame", "Time"]
        corners = self.config['selected_corners']
        
        for plot_info in self.axes_plots:
            if plot_info['type'] == "kin":
                for c_idx in corners:
                    header.append(f"P{c_idx}_{self.config['data_type']}_{plot_info['label']}")
            elif plot_info['type'] == "impact":
                for c_idx in corners:
                    header.append(f"P{c_idx}_ImpactForce")
            elif plot_info['type'] == "rot_axis":
                header.append("RotAxis_Azimuth")
                header.append("RotAxis_Elevation")
            elif plot_info['type'] == "trans_vel_xyz":
                header.extend(["TransVel_X", "TransVel_Y", "TransVel_Z"])
            else:
                header.append(plot_info['label'].replace(" ", "_"))

        min_len = len(self.sim.time_history)
        times = self.sim.time_history

        rows = []
        for i in range(min_len):
            row = [str(i), f"{times[i]:.6f}"]
            
            for plot_info in self.axes_plots:
                if plot_info['type'] == "kin":
                    if self.config['data_type'] == "pos":
                        ds = self.sim.corner_pos_hist if plot_info['axis_idx'] < 3 else self.sim.corner_pos_res_hist
                    elif self.config['data_type'] == "vel":
                        ds = self.sim.corner_vel_hist if plot_info['axis_idx'] < 3 else self.sim.corner_vel_res_hist
                    else:
                        ds = self.sim.corner_acc_hist if plot_info['axis_idx'] < 3 else self.sim.corner_acc_res_hist
                    
                    data_pt = ds[i]
                    for c_idx in corners:
                        val = data_pt[c_idx, plot_info['axis_idx']] if plot_info['axis_idx'] < 3 else data_pt[c_idx]
                        row.append(f"{val:.6f}")
                
                elif plot_info['type'] == "impact":
                    data_pt = self.sim.corner_impact_hist[i]
                    for c_idx in corners:
                        row.append(f"{data_pt[c_idx]:.6f}")
                
                elif plot_info['type'] == "drag":
                    row.append(f"{self.sim.air_drag_hist[i]:.6f}")
                
                elif plot_info['type'] == "squeeze":
                    row.append(f"{self.sim.air_squeeze_hist[i]:.6f}")
                
                elif plot_info['type'] == "rot_axis":
                    vec = self.sim.rot_axis_hist[i]
                    azi = np.degrees(np.arctan2(vec[1], vec[0]))
                    ele = np.degrees(np.arcsin(np.clip(vec[2], -1, 1)))
                    row.append(f"{azi:.4f}")
                    row.append(f"{ele:.4f}")
                
                elif plot_info['type'] == "rot_speed":
                    row.append(f"{self.sim.rot_speed_hist[i]:.6f}")
                
                elif plot_info['type'] == "trans_vel_xyz":
                    vec = self.sim.trans_vel_hist[i]
                    row.extend([f"{vec[0]:.6f}", f"{vec[1]:.6f}", f"{vec[2]:.6f}"])
                
                elif plot_info['type'] == "trans_vel_res":
                    row.append(f"{self.sim.trans_vel_res_hist[i]:.6f}")
            rows.append(row)

        # 포맷팅
        content = ""
        if mode == "csv":
            content = ",".join(header) + "\n"
            content += "\n".join([",".join(r) for r in rows])
        elif mode == "tab":
            content = "\t".join(header) + "\n"
            content += "\n".join(["\t".join(r) for r in rows])
        elif mode == "md":
            content = "| " + " | ".join(header) + " |\n"
            content += "| " + " | ".join(["---"] * len(header)) + " |\n"
            content += "\n".join(["| " + " | ".join(r) + " |" for r in rows])

        # 출력 대상 처리
        if self.act_to_clipboard.isChecked():
            QApplication.clipboard().setText(content)
            # 윈도우 타이틀에 잠시 표시
            orig_title = self.windowTitle()
            self.setWindowTitle("✅ Copied to Clipboard!")
            QTimer.singleShot(2000, lambda: self.setWindowTitle(orig_title))
        else:
            ext = ".csv" if mode == "csv" else ".txt" if mode == "tab" else ".md"
            path, _ = QFileDialog.getSaveFileName(self, "Save Data", "", f"Data Files (*{ext});;All Files (*)")
            if path:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)
