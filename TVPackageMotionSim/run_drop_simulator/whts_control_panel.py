# -*- coding: utf-8 -*-
"""
[WHTOOLS] Simulation Control Center v1.0
PySide6 기반의 현대적인 MuJoCo 시뮬레이션 제어 패널입니다.
"""

import os
import sys
import time
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QSlider, QLabel, QFrame, QGroupBox, QDoubleSpinBox
)
from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QFont, QIcon, QColor, QPalette, QPixmap

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

    def _init_ui(self):
        """현대적인 Dark Mode 스타일의 UI를 구성합니다."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

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
        self.lbl_time = QLabel("Time: 0.000 / 0.000 s")
        self.lbl_status = QLabel("Status: Ready")
        self.lbl_step = QLabel("Step: 0")
        self.lbl_snapshots = QLabel("Snapshots: 0")
        
        for lbl in [self.lbl_time, self.lbl_status, self.lbl_step, self.lbl_snapshots]:
            lbl.setFont(QFont("Consolas", 10))
            status_layout.addWidget(lbl)
        
        header_layout.addWidget(status_group, 1) # Status group expands
        main_layout.addLayout(header_layout)

        # 2. 재생 제어 그룹
        playback_group = QGroupBox("Playback Controls")
        playback_layout = QHBoxLayout(playback_group)
        
        self.btn_reset = QPushButton("🔄 Reset")
        self.btn_back = QPushButton("⏪ Step")
        self.btn_play = QPushButton("▶️ Play")
        self.btn_forward = QPushButton("⏩ Step")
        
        for btn in [self.btn_reset, self.btn_back, self.btn_play, self.btn_forward]:
            btn.setMinimumHeight(40)
            btn.setFont(QFont("Segoe UI Emoji", 10)) # 이모지 폰트 명시
            playback_layout.addWidget(btn)
            
        self.btn_reset.clicked.connect(self._on_reset)
        self.btn_back.clicked.connect(self._on_back)
        self.btn_play.clicked.connect(self._on_play_pause)
        self.btn_forward.clicked.connect(self._on_forward)
        
        main_layout.addWidget(playback_group)

        # 2-1. 인터랙티브 효과 그룹 (NEW)
        fx_group = QGroupBox("Interactive Effects & Recording")
        fx_layout = QHBoxLayout(fx_group)
        
        self.btn_slow = QPushButton("🐌 Slow Motion")
        self.btn_slow.setCheckable(True)
        self.btn_slow.clicked.connect(self._on_slow_motion)
        
        self.btn_rec = QPushButton("⏺️ Record History")
        self.btn_rec.setCheckable(True)
        self.btn_rec.clicked.connect(self._on_record)
        
        self.btn_monitor = QPushButton("📈 Monitor")
        self.btn_monitor.clicked.connect(self._on_monitor)
        
        for btn in [self.btn_slow, self.btn_rec, self.btn_monitor]:
            btn.setMinimumHeight(35)
            fx_layout.addWidget(btn)
        
        main_layout.addWidget(fx_group)

        # 3. 타임라인 슬라이더
        slider_group = QGroupBox("Timeline Navigation")
        slider_layout = QVBoxLayout(slider_group)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self._on_slider_moved)
        slider_layout.addWidget(self.slider)
        main_layout.addWidget(slider_group)

        # 4. 속도 제어 그룹
        speed_group = QGroupBox("Simulation Speed")
        speed_layout = QHBoxLayout(speed_group)
        speed_layout.addWidget(QLabel("Speed Multiplier:"))
        self.spin_speed = QDoubleSpinBox()
        self.spin_speed.setRange(0.1, 10.0)
        self.spin_speed.setSingleStep(0.1)
        self.spin_speed.setValue(1.0)
        self.spin_speed.valueChanged.connect(self._on_speed_changed)
        speed_layout.addWidget(self.spin_speed)
        main_layout.addWidget(speed_group)

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
        util_layout.addLayout(row1)

        self.btn_reload_xml = QPushButton("📂 Open & Reload XML File")
        self.btn_reload_xml.setMinimumHeight(40)
        self.btn_reload_xml.setStyleSheet("background-color: #2c3e50; font-weight: bold; border: 1px solid #34495e;")
        self.btn_reload_xml.setToolTip("Select and load an external MuJoCo XML file.")
        self.btn_reload_xml.clicked.connect(self._on_reload_xml)
        util_layout.addWidget(self.btn_reload_xml)
        
        main_layout.addLayout(util_layout)

        # 스타일 시트 적용 (Premium Dark Theme)
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QGroupBox { 
                color: #ffffff; 
                font-weight: bold; 
                border: 1px solid #333333; 
                margin-top: 10px;
                padding-top: 15px;
            }
            QLabel { color: #cccccc; }
            QPushButton { 
                background-color: #333333; 
                color: white; 
                border-radius: 5px; 
                border: 1px solid #444444;
            }
            QPushButton:hover { background-color: #444444; }
            QPushButton:pressed { background-color: #555555; }
            QSlider::handle:horizontal {
                background: #0078d7;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
        """)

    def _update_status(self):
        """시뮬레이터의 현재 상태를 UI에 반영합니다."""
        if self.sim.data is None:
            return
            
        # 시간 및 스텝 정보 업데이트
        curr_time = self.sim.data.time
        target_time = self.sim.config.get("sim_duration", 1.0)
        
        self.lbl_time.setText(f"Time: {curr_time:.3f} / {target_time:.3f} s")
        self.lbl_step.setText(f"Step: {self.sim.step_idx}")
        snap_count = len(self.sim.snapshots)
        self.lbl_snapshots.setText(f"Snapshots: {snap_count}")

        # 데이터 수집 상태 판정
        if curr_time >= target_time:
            self.lbl_status.setText("Status: Collection Complete ✅")
            self.lbl_status.setStyleSheet("color: #2ecc71; font-weight: bold;") # Green
        elif self.sim.ctrl_paused:
            self.lbl_status.setText("Status: Paused ⏸️")
            self.lbl_status.setStyleSheet("color: #f1c40f;") # Yellow
        else:
            self.lbl_status.setText("Status: Data Collecting... ⏳")
            self.lbl_status.setStyleSheet("color: #3498db;") # Blue
        
        # 슬라이더 범위 업데이트 및 재생 중 최신 위치 추적
        if snap_count > 0:
            self.slider.setRange(0, snap_count - 1)
            
            # [WHTOOLS] 재생 중에는 슬라이더 핸들을 자동으로 맨 뒤로 이동
            if not self.sim.ctrl_paused:
                self.slider.blockSignals(True)
                self.slider.setValue(snap_count - 1)
                self.slider.blockSignals(False)
            # 현재 시점이 스냅샷 중 어디인지 대략적으로 표시 (선택 사항)
            # self.slider.setValue(snap_count - 1) 

        # 재생 버튼 텍스트 업데이트
        self.btn_play.setText("▶️ Resume" if self.sim.ctrl_paused else "⏸️ Pause")
        
        # 효과 버튼 상태 동기화 (시뮬레이터 내부 상태 -> UI)
        self.btn_slow.setChecked(self.sim.ctrl_slow_motion)
        self.btn_slow.setStyleSheet("background-color: #554400;" if self.sim.ctrl_slow_motion else "")
        
        self.btn_rec.setChecked(self.sim.is_recording)
        self.btn_rec.setStyleSheet("background-color: #550000; color: #ff0000; font-weight: bold;" if self.sim.is_recording else "")

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

    def _on_speed_changed(self, value):
        self.sim.ctrl_speed_multiplier = value

    def _on_slow_motion(self, checked):
        self.sim.ctrl_slow_motion = checked

    def _on_record(self, checked):
        self.sim.is_recording = checked

    def _on_camera_export(self):
        self.sim.ctrl_export_camera = True

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

    def _on_open_config(self):
        """설정 편집기(ConfigEditor)를 메인 스레드에서 직접 실행합니다."""
        from .whts_gui import ConfigEditor
        if not hasattr(self, 'config_editor') or self.config_editor is None:
            self.config_editor = ConfigEditor(self.sim)
        self.config_editor.show()
        self.config_editor.raise_()
        self.config_editor.activateWindow()

    def _on_reload_xml(self):
        from PySide6.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select MuJoCo Simulation XML",
            str(self.sim.output_dir), "MuJoCo XML (*.xml);;All Files (*)"
        )
        if file_path:
            self.sim.reload_xml(file_path)

def launch_control_panel(simulator):
    """
    외부에서 컨트롤 패널을 실행하기 위한 진입점입니다.
    """
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    
    panel = ControlPanel(simulator)
    panel.show()
    return app, panel

if __name__ == "__main__":
    # 테스트용 가상 시뮬레이터 객체 (실제 실행 시에는 DropSimulator 인스턴스가 전달됨)
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
            
    app, panel = launch_control_panel(MockSim())
    sys.exit(app.exec())
