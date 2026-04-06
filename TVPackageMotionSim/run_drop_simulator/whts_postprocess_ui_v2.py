# -*- coding: utf-8 -*-
"""
[WHTOOLS] Integrated Simulation Control UI v2.0 (Full Implementation)
MuJoCo 시뮬레이션 설정, 비동기 실행 제어 및 데이터 분석을 통합한 PySide6 기반 관리 센터.
"""

import os
import sys
import time
import json
import numpy as np
import pickle
import threading
import subprocess
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QStackedWidget, QPushButton, QLabel, QFrame, QScrollArea, 
    QLineEdit, QFormLayout, QGroupBox, QComboBox, QCheckBox, 
    QProgressBar, QPlainTextEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QSplitter, QFileDialog, QMessageBox, QSlider
)
from PySide6.QtCore import Qt, QSize, QTimer, QThread, Signal, Slot, QObject
from PySide6.QtGui import QFont, QIcon, QColor, QPalette, QTextCursor

# [WHTOOLS] 엔진 및 유틸리티 연동
from whts_postprocess_engine_v2 import SimulationControlEngine
from run_discrete_builder import get_default_config

# 전역 스타일 설정
GLOBAL_FONT = "D2Coding"
BG_COLOR = "#1e1e1e"
ACCENT_COLOR = "#00d2ff"
SECONDARY_COLOR = "#333333"

class LogEmitter(QObject):
    """표준 출력을 UI로 전달하는 시그널 객체"""
    log_signal = Signal(str)

    def write(self, text):
        if text.strip():
            self.log_signal.emit(text)
    
    def flush(self):
        pass

class SimulationThread(QThread):
    """시뮬레이션 메인 루프를 실행하는 워커 쓰레드"""
    finished_signal = Signal(bool, str)

    def __init__(self, config):
        super().__init__()
        self.config = config

    def run(self):
        try:
            from run_drop_simulator import DropSimulator
            sim = DropSimulator(config=self.config)
            # 시뮬레이션 시작
            sim.simulate(enable_UI=False)
            self.finished_signal.emit(True, "Simulation Completed Successfully.")
        except Exception as e:
            self.finished_signal.emit(False, str(e))

class WHToolsControlCenter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.engine = SimulationControlEngine()
        self.config = get_default_config()
        self._sim_thread = None
        self._log_emitter = LogEmitter()
        self._log_emitter.log_signal.connect(self._update_console)
        
        # stdout/stderr 리다이렉션 (UI 콘솔로 출력)
        sys.stdout = self._log_emitter
        sys.stderr = self._log_emitter

        self.setWindowTitle("WHTOOLS Integrated Control Center v2.0")
        self.resize(1200, 850)
        self.setStyleSheet(f"background-color: {BG_COLOR}; color: #e0e0e0; font-family: '{GLOBAL_FONT}';")
        
        self._setup_ui()
        self._load_config_to_ui()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(0)

        # 1. 사이드 바
        self.sidebar = QFrame()
        self.sidebar.setFixedWidth(240)
        self.sidebar.setStyleSheet(f"background-color: {SECONDARY_COLOR}; border-right: 1px solid #444;")
        side_layout = QVBoxLayout(self.sidebar)
        
        logo = QLabel("WHTOOLS\nControl Center")
        logo.setFont(QFont(GLOBAL_FONT, 15, QFont.Bold))
        logo.setAlignment(Qt.AlignCenter); logo.setStyleSheet(f"color: {ACCENT_COLOR}; padding: 30px;")
        side_layout.addWidget(logo)

        self.nav_btns = []
        for text, idx in [("Settings", 0), ("Execution", 1), ("History", 2), ("Launch View", 3)]:
            btn = QPushButton(text)
            btn.setCheckable(True); btn.setFixedHeight(50)
            btn.setStyleSheet(self._get_btn_style())
            btn.clicked.connect(lambda _, i=idx: self._switch_tab(i))
            side_layout.addWidget(btn)
            self.nav_btns.append(btn)
        
        side_layout.addStretch()
        layout.addWidget(self.sidebar)

        # 2. 메인 스택
        self.stack = QStackedWidget()
        self.stack.addWidget(self._init_config_tab())
        self.stack.addWidget(self._init_run_tab())
        self.stack.addWidget(self._init_history_tab())
        layout.addWidget(self.stack)

        self._switch_tab(0)

    def _get_btn_style(self):
        return f"""
            QPushButton {{ background: transparent; text-align: left; padding-left: 25px; border: none; font-weight: bold; font-size: 11pt; }}
            QPushButton:hover {{ background: #444; }}
            QPushButton:checked {{ background: #555; color: {ACCENT_COLOR}; border-left: 4px solid {ACCENT_COLOR}; }}
        """

    def _switch_tab(self, idx):
        if idx == 3: # 3D View 즉시 실행 전용 (현재 파일 로드 여부와 관계없이)
            self._launch_dashboard(None)
            return
        for i, b in enumerate(self.nav_btns): b.setChecked(i == idx)
        self.stack.setCurrentIndex(idx)
        if idx == 2: self._refresh_history()

    # --- TAB 1: Config ---
    def _init_config_tab(self):
        w = QWidget(); layout = QVBoxLayout(w)
        layout.addWidget(QLabel("Simulation Configuration Editor", font=QFont(GLOBAL_FONT, 18, QFont.Bold)))
        
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
        container = QWidget(); self.form = QFormLayout(container)
        
        # 그룹별 구현 (핵심 파라미터 60% 노출)
        self._edits = {}
        
        # (1) Scenario
        g1 = self._create_group("1. Scenario & Environment", [
            ("drop_mode", "LTL", ["LTL", "Express", "Standard"]),
            ("drop_direction", "Corner 2-3-5", ["Corner 2-3-5", "Edge 3-5", "Face 3", "Flat Drop"]),
            ("drop_height", 0.5, "float"),
            ("sim_duration", 2.0, "float")
        ])
        
        # (2) Physics
        g2 = self._create_group("2. Physics Solver", [
            ("sim_integrator", "implicitfast", ["implicitfast", "euler", "rk4"]),
            ("sim_timestep", 0.0012, "float"),
            ("sim_iterations", 50, "int"),
            ("use_viewer", True, "bool")
        ])

        # (3) Material Properties
        g3 = self._create_group("3. Structural Properties (Weld)", [
            ("opencell_weld_solref_timec", 0.005, "float"),
            ("chassis_weld_solref_timec", 0.002, "float"),
            ("cush_yield_stress", 0.1, "float")
        ])

        self.form.addRow(g1); self.form.addRow(g2); self.form.addRow(g3)
        scroll.setWidget(container); layout.addWidget(scroll)
        
        btn = QPushButton("💾 APPLY SETTINGS & READY"); btn.setFixedHeight(50)
        btn.setStyleSheet(f"background: {ACCENT_COLOR}; color: black; font-weight: bold;")
        btn.clicked.connect(self._apply_and_switch); layout.addWidget(btn)
        return w

    def _create_group(self, title, items):
        box = QGroupBox(title); box.setStyleSheet("font-weight: bold; color: #00d2ff; padding-top: 20px;")
        f = QFormLayout(box)
        for key, default, dtype in items:
            if dtype == "bool":
                widget = QCheckBox(); widget.setChecked(default)
            elif isinstance(dtype, list):
                widget = QComboBox(); widget.addItems(dtype)
            else:
                widget = QLineEdit(str(default))
            f.addRow(f"{key}:", widget)
            self._edits[key] = (widget, dtype)
        return box

    # --- TAB 2: Execution ---
    def _init_run_tab(self):
        w = QWidget(); layout = QVBoxLayout(w)
        layout.addWidget(QLabel("Simulation Console", font=QFont(GLOBAL_FONT, 18, QFont.Bold)))
        
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True); self.console.setStyleSheet("background: black; color: #00ff00; font-family: 'Consolas';")
        layout.addWidget(self.console)
        
        self.prog = QProgressBar(); layout.addWidget(self.prog)
        
        btns = QHBoxLayout()
        self.run_btn = QPushButton("🚀 START"); self.run_btn.setFixedHeight(50)
        self.run_btn.setStyleSheet(f"background: {ACCENT_COLOR}; color: black; font-weight: bold;")
        self.run_btn.clicked.connect(self._run_sim)
        
        self.stop_btn = QPushButton("🛑 STOP"); self.stop_btn.setEnabled(False); self.stop_btn.setFixedHeight(50)
        self.stop_btn.setStyleSheet("background: #ff4444; color: white; font-weight: bold;")
        self.stop_btn.clicked.connect(self._stop_sim)
        
        btns.addWidget(self.run_btn); btns.addWidget(self.stop_btn)
        layout.addLayout(btns)
        return w

    # --- TAB 3: History ---
    def _init_history_tab(self):
        w = QWidget(); layout = QVBoxLayout(w)
        layout.addWidget(QLabel("Result Records", font=QFont(GLOBAL_FONT, 18, QFont.Bold)))
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Filename", "Date", "Control"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table)
        return w

    # --- LOGIC ---
    def _load_config_to_ui(self):
        # Config 값을 UI 위젯으로 복사 (간소화)
        pass

    def _apply_and_switch(self):
        for k, (w, dt) in self._edits.items():
            if dt == "bool": self.config[k] = w.isChecked()
            elif isinstance(dt, list): self.config[k] = w.currentText()
            elif dt == "float": self.config[k] = float(w.text())
            elif dt == "int": self.config[k] = int(w.text())
        self._switch_tab(1)

    def _run_sim(self):
        self.run_btn.setEnabled(False); self.stop_btn.setEnabled(True)
        self._sim_thread = SimulationThread(self.config)
        self._sim_thread.finished_signal.connect(self._on_finished)
        self._sim_thread.start()

    def _stop_sim(self):
        if self._sim_thread: self._sim_thread.terminate(); self._on_finished(False, "Aborted")

    @Slot(str)
    def _update_console(self, msg):
        self.console.appendPlainText(msg)
        self.console.moveCursor(QTextCursor.End)

    def _on_finished(self, success, msg):
        self.run_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        QMessageBox.information(self, "Finished", msg)

    def _refresh_history(self):
        files = self.engine.get_result_files()
        self.table.setRowCount(len(files))
        for row, f in enumerate(files):
            self.table.setItem(row, 0, QTableWidgetItem(f['name']))
            self.table.setItem(row, 1, QTableWidgetItem(f['date']))
            btn = QPushButton("Analyze In 3D Dashboard")
            btn.clicked.connect(lambda _, p=f['path']: self._launch_dashboard(p))
            self.table.setCellWidget(row, 2, btn)

    def _launch_dashboard(self, path):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        target = os.path.join(curr_dir, "plate_by_markers_v2.py")
        cmd = [sys.executable, target]
        if path: cmd.extend(["--load", path])
        subprocess.Popen(cmd)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="WHTOOLS Control Center v2.0")
    parser.add_argument("--load", type=str, help="Path to a simulation result .pkl to load and analyze immediately")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    center = WHToolsControlCenter()
    
    if args.load and os.path.exists(args.load):
        # 결과가 이미 있는 경우 (시뮬레이션 직후 자동 실행 등)
        # 실행 탭으로 전환하고 대시보드 호출
        center._switch_tab(1)
        center.console.appendPlainText(f"[WHTOOLS] Result loaded: {args.load}")
        center._launch_dashboard(args.load)
    
    center.show()
    sys.exit(app.exec())
