# -*- coding: utf-8 -*-
"""
[WHTOOLS] Optimization & DOE Dashboard (v1.0)
Gooey를 활용한 탭 기반의 범용 설계변수 입력기 UI 및 
PySide6 기반의 해석 결과 비교(Overlay Plotter) 및 최적안 탐색을 위한 대시보드(DOE Monitor UI)입니다.
"""

import os
import sys
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# GUI 라이브러리 및 Matplotlib 연동
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox,
    QPushButton, QTableWidget, QTableWidgetItem, QTabWidget, QSplitter,
    QFileDialog, QMessageBox, QCheckBox, QPlainTextEdit, QAbstractSpinBox
)

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import koreanize_matplotlib

# Gooey 라이브러리 임포트 (Gooey 환경변수 무시 대비)
from gooey import Gooey, GooeyParser

# [WHTOOLS] 경로 설정 보완 (run_parameter_study용)
curr_dir = os.path.dirname(os.path.abspath(__file__))
if curr_dir not in sys.path:
    sys.path.insert(0, curr_dir)
parent_dir = os.path.dirname(curr_dir)  # TVPackageMotionSim
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
grandparent_dir = os.path.dirname(parent_dir)  # WHToolsBox
if grandparent_dir not in sys.path:
    sys.path.append(grandparent_dir)

# 내부 로컬 모듈 임포트
from whts_optimization_engine import DOEEngine, DOEBatchRunner, OptimizationEvaluator
from run_drop_simulator.whts_theme import GLOBAL_QSS, apply_app_theme


# ─── PySide6: Matplotlib Overlay Canvas ─────────────────────────────────────

class MplOverlayCanvas(FigureCanvas):
    """
    [WHTOOLS] 다중 DOE 케이스의 해석 결과 데이터를 겹쳐서 가시화하는 Matplotlib 캔버스 위젯입니다.
    """
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig, self.axes = plt.subplots(3, 1, figsize=(width, height), sharex=True)
        plt.rc('font', size=9) # global 9pt 폰트 준수
        super().__init__(self.fig)
        self.setParent(parent)
        
        # 다크 모드에 최적화된 테마 설정
        self.fig.patch.set_facecolor('#121212')
        for ax in self.axes:
            ax.set_facecolor('#1e1e1e')
            ax.spines['bottom'].set_color('#888888')
            ax.spines['top'].set_color('#333333')
            ax.spines['left'].set_color('#888888')
            ax.spines['right'].set_color('#333333')
            ax.tick_params(colors='white', labelsize=8)
            ax.yaxis.label.set_color('white')
            ax.xaxis.label.set_color('white')
            ax.title.set_color('#88c0d0')
            ax.grid(True, linestyle='--', color='#444444', alpha=0.5)

        self.fig.tight_layout()

    def plot_cases(self, selected_cases_data: List[Dict[str, Any]]):
        """
        선택된 다수 케이스 데이터를 중첩(Overlay)하여 플로팅합니다.
        """
        for ax in self.axes:
            ax.clear()
            ax.grid(True, linestyle='--', color='#444444', alpha=0.5)

        if not selected_cases_data:
            self.draw()
            return

        colors = plt.cm.rainbow(np.linspace(0, 1, len(selected_cases_data)))

        for idx, case in enumerate(selected_cases_data):
            case_id = case["case_id"]
            times = case.get("time_history", [])
            force = case.get("force_history", [])
            disp = case.get("z_displacement_history", [])
            
            # 단위 변환: m -> mm, s -> s, N -> N (기본값)
            disp_mm = np.array(disp) * 1000.0 if np.max(np.abs(disp)) < 5.0 else np.array(disp)
            
            label_str = f"Case {case_id:03d}"
            c = colors[idx]

            # 1. Z-displacement
            self.axes[0].plot(times, disp_mm, color=c, label=label_str, alpha=0.85, linewidth=1.5)
            # 2. Ground Contact Force
            self.axes[1].plot(times, force, color=c, label=label_str, alpha=0.85, linewidth=1.5)
            
        # 3. Max Von-Mises Stress Bar/Point 비교 (시간 이력이 없으므로 bar chart 혹은 VM peak 값 비교)
        case_labels = [f"C{c['case_id']}" for c in selected_cases_data]
        peak_stresses = [c.get("max_stress_mpa", 0.0) for c in selected_cases_data]
        bar_colors = [colors[i] for i in range(len(selected_cases_data))]
        
        # 3번째 축은 Bar chart로 최대 Stress 표시
        y_pos = np.arange(len(case_labels))
        self.axes[2].barh(y_pos, peak_stresses, color=bar_colors, alpha=0.8, edgecolor='#888888')
        self.axes[2].set_yticks(y_pos)
        self.axes[2].set_yticklabels(case_labels, color='white')
        self.axes[2].set_xlabel("Max VM Stress [MPa]", color='white')

        # 제목 및 라벨
        self.axes[0].set_title("Z-Displacement History", color='#88c0d0')
        self.axes[0].set_ylabel("Displacement [mm]", color='white')
        self.axes[1].set_title("Ground Contact Force History", color='#88c0d0')
        self.axes[1].set_ylabel("Force [N]", color='white')
        self.axes[2].set_title("Peak Von-Mises Stress", color='#88c0d0')
        
        # Legend: axes[0]에 외곽으로 배치
        self.axes[0].legend(loc='upper right', bbox_to_anchor=(1.15, 1.05), fontsize=8, ncol=1)

        self.fig.tight_layout()
        self.draw()


# ─── PySide6: DOE Monitor UI (Optimization Dashboard) ───────────────────────

class DOEMonitorDashboard(QMainWindow):
    """
    [WHTOOLS] DOE 결과를 시계열로 비교 분석하고 제약 조건에 따른 최적안을 제안하는 
    PySide6 기반의 최적화 대시보드(DOE Monitor UI)입니다.
    """
    def __init__(self, summary_json_path: str):
        super().__init__()
        self.summary_json_path = Path(summary_json_path)
        self.output_base_dir = self.summary_json_path.parent
        self.summary_data: List[Dict[str, Any]] = []
        
        self.setWindowTitle("📈 WHTOOLS DOE Monitor & Optimization Dashboard (v1.0)")
        self.resize(1200, 800)
        
        # 기본 UI 구성 및 스타일셋
        self.setStyleSheet(GLOBAL_QSS)
        
        self._init_ui()
        self.load_data()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        
        # 좌우 스플리터 구분
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # ── Left: DOE Cases & List Panel ──
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        cases_group = QGroupBox("📋 DOE Run Cases")
        cases_layout = QVBoxLayout(cases_group)
        
        # Table 구성
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels([
            "Case ID", "Status", "Max Force [N]", "Max Stress [MPa]", "Max Disp [mm]"
        ])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.itemChanged.connect(self._on_table_item_changed)
        cases_layout.addWidget(self.table)
        
        # 하단 제어 버튼
        btn_layout = QHBoxLayout()
        self.btn_select_all = QPushButton("☑️ Select All")
        self.btn_deselect_all = QPushButton("⬜ Deselect All")
        self.btn_reload = QPushButton("🔄 Reload Data")
        
        self.btn_select_all.clicked.connect(self.select_all_cases)
        self.btn_deselect_all.clicked.connect(self.deselect_all_cases)
        self.btn_reload.clicked.connect(self.load_data)
        
        btn_layout.addWidget(self.btn_select_all)
        btn_layout.addWidget(self.btn_deselect_all)
        btn_layout.addWidget(self.btn_reload)
        cases_layout.addLayout(btn_layout)
        
        left_layout.addWidget(cases_group)
        splitter.addWidget(left_widget)
        
        # ── Right: Plotting & Optimization Panel ──
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Tab Widget 분할 (1: Graphs, 2: Optimization)
        self.tabs = QTabWidget()
        right_layout.addWidget(self.tabs)
        
        # 1) Graph Tab
        graph_tab = QWidget()
        graph_tab_layout = QVBoxLayout(graph_tab)
        
        self.canvas = MplOverlayCanvas(self, width=6, height=5)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        graph_tab_layout.addWidget(self.toolbar)
        graph_tab_layout.addWidget(self.canvas)
        self.tabs.addTab(graph_tab, "📊 Overlay Plots Comparison")
        
        # 2) Optimization Tab
        opt_tab = QWidget()
        opt_layout = QVBoxLayout(opt_tab)
        
        # 제약 조건 설정 그룹
        constraints_group = QGroupBox("🚧 Optimization Constraints")
        cons_grid = QtWidgets.QGridLayout(constraints_group)
        
        # Max Displacement 제약
        cons_grid.addWidget(QLabel("Max Displacement Limit (mm):"), 0, 0)
        self.spin_lim_disp = QDoubleSpinBox()
        self.spin_lim_disp.setRange(0.1, 1000.0)
        self.spin_lim_disp.setValue(15.0)
        self.spin_lim_disp.setSuffix(" mm")
        cons_grid.addWidget(self.spin_lim_disp, 0, 1)
        
        # Max Stress 제약
        cons_grid.addWidget(QLabel("Max Von-Mises Stress Limit (MPa):"), 1, 0)
        self.spin_lim_stress = QDoubleSpinBox()
        self.spin_lim_stress.setRange(0.1, 100000.0)
        self.spin_lim_stress.setValue(200.0)
        self.spin_lim_stress.setSuffix(" MPa")
        cons_grid.addWidget(self.spin_lim_stress, 1, 1)
        
        # Max Ground Force 제약
        cons_grid.addWidget(QLabel("Max Ground Force Limit (N):"), 2, 0)
        self.spin_lim_force = QDoubleSpinBox()
        self.spin_lim_force.setRange(1.0, 1000000.0)
        self.spin_lim_force.setValue(30000.0)
        self.spin_lim_force.setSuffix(" N")
        cons_grid.addWidget(self.spin_lim_force, 2, 1)
        
        opt_layout.addWidget(constraints_group)
        
        # 목적 함수 & 고급 규칙 그룹
        rules_group = QGroupBox("🎯 Objective & Python Rule Filter")
        rules_layout = QVBoxLayout(rules_group)
        
        # 목적 함수 선택
        obj_row = QHBoxLayout()
        obj_row.addWidget(QLabel("Objective Target:"))
        self.combo_obj = QComboBox()
        self.combo_obj.addItems([
            "Minimize Ground Force (min_force)",
            "Minimize Max Stress (min_stress)",
            "Minimize Max Displacement (min_disp)"
        ])
        obj_row.addWidget(self.combo_obj)
        rules_layout.addLayout(obj_row)
        
        # 고급 필터 식
        rules_layout.addWidget(QLabel("Advanced Python Rule Expression (e.g. max_stress_mpa < 180 and max_ground_force < 8000):"))
        self.txt_expression = QLineEdit()
        self.txt_expression.setPlaceholderText("max_stress_mpa < 180.0 and max_displacement_mm < 12.0")
        rules_layout.addWidget(self.txt_expression)
        
        self.btn_opt_run = QPushButton("⚡ Evaluate Best Case")
        self.btn_opt_run.setFixedHeight(35)
        self.btn_opt_run.setStyleSheet("background-color: #2e7d32; color: white;")
        self.btn_opt_run.clicked.connect(self.run_optimization)
        rules_layout.addWidget(self.btn_opt_run)
        
        opt_layout.addWidget(rules_group)
        
        # 결과 보고창
        best_group = QGroupBox("🏆 Optimization Result (Best Case)")
        best_layout = QVBoxLayout(best_group)
        self.txt_report = QPlainTextEdit()
        self.txt_report.setReadOnly(True)
        self.txt_report.setStyleSheet("background-color: #1a1a1a; color: #a9dc76; font-family: Consolas;")
        best_layout.addWidget(self.txt_report)
        opt_layout.addWidget(best_group)
        
        self.tabs.addTab(opt_tab, "🎯 Optimal Target Selection")
        
        splitter.addWidget(right_widget)
        splitter.setSizes([500, 700])

    def load_data(self):
        """요약 JSON 데이터를 읽어와 테이블에 매핑합니다."""
        if not self.summary_json_path.exists():
            QMessageBox.critical(self, "Error", f"Summary file not found at:\n{self.summary_json_path}")
            return

        try:
            with open(self.summary_json_path, "r", encoding="utf-8") as f:
                self.summary_data = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to parse JSON:\n{e}")
            return

        # 파라미터(설계변수) 고유 키 수집
        param_keys = set()
        for case in self.summary_data:
            param_keys.update(case.get("parameters", {}).keys())
        self.param_keys = sorted(list(param_keys))

        # 테이블 헤더 동적 재구성
        base_headers = ["Case ID", "Status", "Max Force [N]", "Max Stress [MPa]", "Max Disp [mm]"]
        all_headers = base_headers + self.param_keys
        
        self.table.blockSignals(True)
        self.table.setColumnCount(len(all_headers))
        self.table.setHorizontalHeaderLabels(all_headers)
        self.table.setRowCount(len(self.summary_data))
        
        for idx, case in enumerate(self.summary_data):
            case_id = case["case_id"]
            status = case.get("status", "Unknown")
            max_force = case.get("max_ground_force", 0.0)
            max_stress = case.get("max_stress_mpa", 0.0)
            max_disp = case.get("max_displacement_mm", 0.0)
            params_dict = case.get("parameters", {})

            # 체크박스가 포함된 첫 번째 아이템 생성
            item_id = QTableWidgetItem(f"Case {case_id:04d}")
            item_id.setFlags(item_id.flags() | Qt.ItemIsUserCheckable)
            item_id.setCheckState(Qt.Unchecked)
            
            self.table.setItem(idx, 0, item_id)
            self.table.setItem(idx, 1, QTableWidgetItem(status))
            
            # 수치 정보들 소수점 가시화
            self.table.setItem(idx, 2, QTableWidgetItem(f"{max_force:.2f}"))
            self.table.setItem(idx, 3, QTableWidgetItem(f"{max_stress:.2f}"))
            self.table.setItem(idx, 4, QTableWidgetItem(f"{max_disp:.2f}"))
            
            # 파라미터 값들을 동적으로 개별 컬럼에 배치
            for col_idx, key in enumerate(self.param_keys, start=5):
                val = params_dict.get(key, "")
                if isinstance(val, float):
                    self.table.setItem(idx, col_idx, QTableWidgetItem(f"{val:.6f}"))
                else:
                    self.table.setItem(idx, col_idx, QTableWidgetItem(str(val)))
            
        self.table.blockSignals(False)
        self.update_plots()

    def select_all_cases(self):
        self.table.blockSignals(True)
        for idx in range(self.table.rowCount()):
            item = self.table.item(idx, 0)
            if item:
                item.setCheckState(Qt.Checked)
        self.table.blockSignals(False)
        self.update_plots()

    def deselect_all_cases(self):
        self.table.blockSignals(True)
        for idx in range(self.table.rowCount()):
            item = self.table.item(idx, 0)
            if item:
                item.setCheckState(Qt.Unchecked)
        self.table.blockSignals(False)
        self.update_plots()

    def _on_table_item_changed(self, item):
        if item.column() == 0:
            self.update_plots()

    def update_plots(self):
        """체크박스가 선택된 케이스 데이터를 필터링하여 Overlay 플롯을 갱신합니다."""
        selected_cases = []
        for idx in range(self.table.rowCount()):
            item = self.table.item(idx, 0)
            if item and item.checkState() == Qt.Checked:
                selected_cases.append(self.summary_data[idx])
        
        self.canvas.plot_cases(selected_cases)

    def run_optimization(self):
        """제약 조건 및 고급 필터에 부합하는 최고의 설계안을 탐색하고 리포팅합니다."""
        if not self.summary_data:
            QMessageBox.warning(self, "No Data", "적재된 해석 결과 데이터가 존재하지 않습니다.")
            return

        # 1단계: 기본 스핀박스 제약
        constraints = {
            "max_displacement_mm": self.spin_lim_disp.value(),
            "max_stress_mpa": self.spin_lim_stress.value(),
            "max_ground_force": self.spin_lim_force.value()
        }

        # 2단계: 목적지 추출
        obj_text = self.combo_obj.currentText()
        if "min_force" in obj_text:
            obj_mode = "min_force"
        elif "min_stress" in obj_text:
            obj_mode = "min_stress"
        else:
            obj_mode = "min_disp"

        # 3단계: 고급 필터 식 적용
        adv_expr = self.txt_expression.text().strip()
        
        filtered_cases = []
        for case in self.summary_data:
            if case.get("status") != "Success":
                continue

            # 기본 제약 체크
            val_disp = case.get("max_displacement_mm", 0.0)
            val_stress = case.get("max_stress_mpa", 0.0)
            val_force = case.get("max_ground_force", 0.0)

            if val_disp > constraints["max_displacement_mm"]:
                continue
            if val_stress > constraints["max_stress_mpa"]:
                continue
            if val_force > constraints["max_ground_force"]:
                continue

            # 고급 Python Expression 검증
            if adv_expr:
                try:
                    # 안전을 위해 로컬 변수 네임스페이스 격리
                    eval_namespace = {
                        "max_stress_mpa": val_stress,
                        "max_displacement_mm": val_disp,
                        "max_ground_force": val_force,
                        **case.get("parameters", {})
                    }
                    if not eval(adv_expr, {"__builtins__": None}, eval_namespace):
                        continue
                except Exception as e:
                    self.txt_report.setPlainText(f"❌ Python Rule Expression Eval Error:\n{e}")
                    return

            filtered_cases.append(case)

        if not filtered_cases:
            self.txt_report.setPlainText("⚠️ 지정된 제약 조건 및 고급 필터를 충족하는 케이스가 존재하지 않습니다.")
            return

        # 목적 변수에 따른 정렬
        if obj_mode == "min_force":
            filtered_cases.sort(key=lambda x: x.get("max_ground_force", float('inf')))
        elif obj_mode == "min_stress":
            filtered_cases.sort(key=lambda x: x.get("max_stress_mpa", float('inf')))
        elif obj_mode == "min_disp":
            filtered_cases.sort(key=lambda x: x.get("max_displacement_mm", float('inf')))

        best = filtered_cases[0]

        best_id = best["case_id"]
        # 리포트 문자열 구성
        report = (
            f"==========================================================\n"
            f"🏆 [WHTOOLS] Best Optimized Parameter Set Found\n"
            f"==========================================================\n"
            f"- Best Case ID : Case {best_id:04d}\n"
            f"- Output Path  : {self.output_base_dir / f'case_{best_id:04d}'}\n"
            f"\n"
            f"[📐 Physical Responses]\n"
            f"- Peak Ground Force  : {best['max_ground_force']:.3f} N (Constraint: < {constraints['max_ground_force']:.0f} N)\n"
            f"- Max VM Stress      : {best['max_stress_mpa']:.3f} MPa (Constraint: < {constraints['max_stress_mpa']:.0f} MPa)\n"
            f"- Max Displacement   : {best['max_displacement_mm']:.3f} mm (Constraint: < {constraints['max_displacement_mm']:.0f} mm)\n"
            f"\n"
            f"[⚙️ Optimized Design Variables]\n"
        )
        for k, v in best["parameters"].items():
            report += f"  - {k:<20} : {v:.6f}\n"
        report += f"==========================================================\n"
        
        self.txt_report.setPlainText(report)
        
        # 최적 케이스 테이블 하이라이트 및 체크
        self.table.blockSignals(True)
        for idx in range(self.table.rowCount()):
            item = self.table.item(idx, 0)
            if item:
                if idx == best["case_id"] - 1:
                    item.setCheckState(Qt.Checked)
                    self.table.selectRow(idx)
                else:
                    item.setCheckState(Qt.Unchecked)
        self.table.blockSignals(False)
        self.update_plots()


# ─── Gooey: CLI Parser & GUI Setup ──────────────────────────────────────────

def setup_gooey_parser() -> GooeyParser:
    """
    [WHTOOLS] Gooey GUI 화면을 구성하기 위한 동적 탭/그룹 argparse 파서를 빌드합니다.
    Target CoG 3축 성분만을 설계변수로 다루도록 정제되었습니다.
    """
    parser = GooeyParser(description="[WHTOOLS] Drop Simulation Optimization & DOE Framework")
    
    # ── [Tab 1] Base Configuration ──
    base_group = parser.add_argument_group("📁 Base Configuration Settings", "기본 입출력 경로 및 샘플링 방식을 설정합니다.")
    
    base_group.add_argument(
        "--base_config",
        type=str,
        default=os.path.join(parent_dir, "config.json"),
        widget="FileChooser",
        help="기준이 되는 Drop Simulator JSON 설정 파일을 선택하세요."
    )
    
    base_group.add_argument(
        "--output_dir",
        type=str,
        default="doe_results",
        widget="DirChooser",
        help="DOE 결과 파일들이 저장될 폴더를 지정하세요."
    )
    
    base_group.add_argument(
        "--sampling_method",
        type=str,
        default="LHS",
        choices=["LHS", "Random", "FullFact"],
        help="실험계획법(DOE) 샘플링 방식을 선택하세요."
    )
    
    base_group.add_argument(
        "--sample_count",
        type=int,
        default=10,
        help="LHS 및 Random 샘플링 시 생성할 실험 횟수(샘플 수)를 입력하세요."
    )

    # ── [Tab 2] Target CoG Bounds ──
    cog_group = parser.add_argument_group("🎯 Target CoG Bounds Config", "내용물의 타겟 무게중심(target_cog) [m] 범위를 지정합니다.")
    
    # target_cog_x
    cog_group.add_argument("--tune_target_cog_x", action="store_true", help="CoG X축 타겟을 튜닝 변수로 지정합니다.")
    cog_group.add_argument("--target_cog_x_min", type=float, default=-0.020, help="target_cog_x 최소값 [m]")
    cog_group.add_argument("--target_cog_x_max", type=float, default=0.020, help="target_cog_x 최대값 [m]")
    cog_group.add_argument("--target_cog_x_step", type=float, default=0.005, help="Discrete 타입일 때 간격 (연속형일 시 0.0 입력)")

    # tune_target_cog_y
    cog_group.add_argument("--tune_target_cog_y", action="store_true", help="CoG Y축 타겟을 튜닝 변수로 지정합니다.")
    cog_group.add_argument("--target_cog_y_min", type=float, default=-0.020, help="target_cog_y 최소값 [m]")
    cog_group.add_argument("--target_cog_y_max", type=float, default=0.020, help="target_cog_y 최대값 [m]")
    cog_group.add_argument("--target_cog_y_step", type=float, default=0.005, help="Discrete 타입일 때 간격 (연속형일 시 0.0 입력)")

    # tune_target_cog_z
    cog_group.add_argument("--tune_target_cog_z", action="store_true", help="CoG Z축 타겟을 튜닝 변수로 지정합니다.")
    cog_group.add_argument("--target_cog_z_min", type=float, default=-0.020, help="target_cog_z 최소값 [m]")
    cog_group.add_argument("--target_cog_z_max", type=float, default=0.020, help="target_cog_z 최대값 [m]")
    cog_group.add_argument("--target_cog_z_step", type=float, default=0.005, help="Discrete 타입일 때 간격 (연속형일 시 0.0 입력)")

    # ── [Tab 3] Mode & Post Dashboard ──
    mode_group = parser.add_argument_group("🖥️ Dashboard Options", "해석 완료 후 또는 단독 시각화 모니터 기동 방식을 제안합니다.")
    mode_group.add_argument(
        "--only_monitor",
        action="store_true",
        help="시뮬레이션 해석을 수행하지 않고, 기존에 생성된 summary 데이터를 기반으로 결과 비교 대시보드(Monitor UI)만 실행합니다."
    )

    return parser


@Gooey(
    program_name="WHTOOLS Optimization Framework",
    navigation="Tabbed", # 입력창 탭으로 구분
    default_size=(800, 650),
    progress_regex=r"^progress: (?P<current>\d+(\.\d+)?)\%",
    progress_expr="current",
    hide_sidebar=True,
    encoding="utf-8"
)
def main():
    parser = setup_gooey_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    summary_path = output_dir / "doe_summary.json"

    # 시각화 대시보드만 단독 실행할 경우
    if args.only_monitor:
        print(f"\n[WHTOOLS] Launching DOE Monitor Dashboard with cached results...", flush=True)
        app = QApplication.instance()
        if not app:
            app = QApplication(sys.argv)
        apply_app_theme(app)
        dashboard = DOEMonitorDashboard(str(summary_path))
        dashboard.show()
        sys.exit(app.exec())

    # 튜닝 대상 변수 리스트화 (오직 Target CoG 3축 성분만 추출)
    tuned_variables = []
    
    if args.tune_target_cog_x:
        tuned_variables.append({
            "name": "target_cog_x",
            "min": args.target_cog_x_min,
            "max": args.target_cog_x_max,
            "type": "Discrete" if args.target_cog_x_step > 0 else "Continuous",
            "step": args.target_cog_x_step
        })
        
    if args.tune_target_cog_y:
        tuned_variables.append({
            "name": "target_cog_y",
            "min": args.target_cog_y_min,
            "max": args.target_cog_y_max,
            "type": "Discrete" if args.target_cog_y_step > 0 else "Continuous",
            "step": args.target_cog_y_step
        })
        
    if args.tune_target_cog_z:
        tuned_variables.append({
            "name": "target_cog_z",
            "min": args.target_cog_z_min,
            "max": args.target_cog_z_max,
            "type": "Discrete" if args.target_cog_z_step > 0 else "Continuous",
            "step": args.target_cog_z_step
        })

    if not tuned_variables:
        print("❌ [Warning] 선택된 튜닝 설계 변수가 없습니다. 최소 하나 이상의 튜닝 체크박스를 켜주십시오.", flush=True)
        sys.exit(1)

    print(f"🔧 Tuned Variables Config: {tuned_variables}", flush=True)

    # 1. DOE 테이블 생성
    engine = DOEEngine(
        variables=tuned_variables,
        sampling_method=args.sampling_method,
        sample_count=args.sample_count
    )
    doe_table = engine.generate_doe_table()

    # 2. 배치 시뮬레이션 해석 시작
    runner = DOEBatchRunner(
        base_config_path=args.base_config,
        output_base_dir=args.output_dir
    )
    runner.run_doe_batch(doe_table)

    # 3. 배치 완료 후 결과 대시보드 팝업 실행
    print(f"\n[WHTOOLS] Opening Optimization Dashboard UI...", flush=True)
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    apply_app_theme(app)
    dashboard = DOEMonitorDashboard(str(summary_path))
    dashboard.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
