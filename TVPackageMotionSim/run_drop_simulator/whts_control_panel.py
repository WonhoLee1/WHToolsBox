# -*- coding: utf-8 -*-
"""
[WHTOOLS] Simulation Control Center v1.0
PySide6 기반의 현대적인 MuJoCo 시뮬레이션 제어 패널입니다.
"""

import os
import sys
import time
from pathlib import Path
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QSlider, QLabel, QFrame, QGroupBox, QDoubleSpinBox,
    QPlainTextEdit, QDialog, QMessageBox
)
from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QFont, QIcon, QColor, QPalette, QPixmap
import ctypes
from ctypes import wintypes

# [WHTOOLS] UTF-8 인코딩 강제 설정 (이모지 및 한글 깨짐 방지)
if sys.stdout.encoding != 'utf-8':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except (AttributeError, io.UnsupportedOperation):
        pass

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

        # 상단 안내문
        info_label = QtWidgets.QLabel(
            "<b>MuJoCo XML Editor:</b> 수식을 직접 수정하고 [Apply & Reload]를 누르면 즉시 반영됩니다.<br>"
            "<small>Tip: 외부 에디터 버튼을 누르면 VS Code나 메모장 등 평소 쓰시는 도구로 편집할 수 있습니다.</small>"
        )
        layout.addWidget(info_label)

        # 텍스트 에디터
        self.editor = QtWidgets.QPlainTextEdit()
        self.editor.setPlainText(initial_xml)
        
        # 고정폭 폰트 적용
        font = QtGui.QFont("Consolas", 11)
        if not font.fixedPitch():
            font = QtGui.QFont("Courier New", 11)
        self.editor.setFont(font)
        self.editor.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        layout.addWidget(self.editor)

        # 버튼 영역
        btn_layout = QtWidgets.QHBoxLayout()
        
        self.btn_external = QtWidgets.QPushButton(" 📝 Open in External Editor")
        self.btn_external.setMinimumHeight(40)
        self.btn_external.clicked.connect(self._on_open_external)
        
        self.btn_apply = QtWidgets.QPushButton(" 🚀 Apply & Reload")
        self.btn_apply.setMinimumHeight(40)
        self.btn_apply.setStyleSheet("background-color: #2E7D32; color: white; font-weight: bold;")
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

    def done(self, result):
        """다이얼로그 종료 시 외부 에디터용 임시파일을 정리합니다."""
        tmp = getattr(self, '_external_tmp_path', None)
        if tmp:
            try:
                import os
                os.unlink(tmp)
            except OSError:
                pass
            self._external_tmp_path = None
        super().done(result)

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

    def showEvent(self, event):
        """창이 표시될 때 정렬을 위해 부모 이벤트를 호출합니다."""
        super().showEvent(event)
        # [WHTOOLS] 기본 화면 구석 이동 로직 제거 (MuJoCo 정렬에 집중)

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

        # 1-1. 카메라 시점 제어 그룹 (NEW)
        from functools import partial
        cam_group = QGroupBox("Camera Orientation (MuJoCo View)")
        cam_layout = QHBoxLayout(cam_group)
        cam_layout.setSpacing(5)
        
        views = ["+X", "-X", "+Y", "-Y", "+Z", "-Z", "+ISO", "-ISO"]
        for v in views:
            btn = QPushButton(v)
            btn.setMinimumHeight(30)
            btn.setMinimumWidth(42)
            btn.setFont(QFont("Consolas", 8, QFont.Bold)) # User requested 8pt
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
        
        info_layout = QHBoxLayout()
        info_layout.addWidget(QLabel("Snapshot History:"))
        self.lbl_frame_info = QLabel("Frame: 0 / 0")
        self.lbl_frame_info.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lbl_frame_info.setFont(QFont("Consolas", 10))
        info_layout.addWidget(self.lbl_frame_info)
        slider_layout.addLayout(info_layout)

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

        # [WHTOOLS] 모션 상태 로그 버튼 추가
        self.btn_log_motion = QPushButton("📋 Log Motion")
        self.btn_log_motion.setMinimumHeight(35)
        self.btn_log_motion.clicked.connect(self._on_log_motion)
        row1.addWidget(self.btn_log_motion)
        
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
            self.btn_slow.setStyleSheet("background-color: #554400;" if self.sim.ctrl_slow_motion else "")
            
            self.btn_rec.setChecked(self.sim.is_recording)
            self.btn_rec.setStyleSheet("background-color: #550000; color: #ff0000; font-weight: bold;" if self.sim.is_recording else "")
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
        self.lbl_status.setStyleSheet("color: #e67e22;")

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
