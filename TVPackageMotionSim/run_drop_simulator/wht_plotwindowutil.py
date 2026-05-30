# -*- coding: utf-8 -*-
"""
[WHTOOLS] Plot Window Utility
PySide6 기반 창에 Matplotlib 캔버스를 임베딩하고, Tab 위젯을 통해 여러 플롯을 선택적으로 확인할 수 있는 유틸리티입니다.
"""

import sys
from pathlib import Path

# [WHTOOLS] 단독 실행(main) 시 부모 디렉토리를 sys.path에 등록하여 절대 경로 임포트 보장
curr_file = Path(__file__).resolve()
parent_dir = curr_file.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import pandas as pd
from typing import List, Union

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTabWidget, QMessageBox, QComboBox, QLabel
)
import numpy as np

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

class PlotTab(QWidget):
    def __init__(self, title: str, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.df = df
        
        # Matplotlib Figure 생성 (폰트 9pt)
        self.fig = Figure(figsize=(8, 6))
        self.fig.subplots_adjust(right=0.85) # 범례 공간 확보
        self.ax = self.fig.add_subplot(111)
        
        # 툴바와 캔버스 임베딩
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        
        # 상단 컨트롤 패널(콤보박스)
        self.control_layout = QHBoxLayout()
        self.control_layout.addWidget(QLabel("Data Type:"))
        self.combo = QComboBox()
        self.combo.addItems(["Position", "Velocity", "Acceleration"])
        self.combo.currentTextChanged.connect(self.on_combo_changed)
        self.control_layout.addWidget(self.combo)
        self.control_layout.addStretch()
        
        self.layout.addWidget(self.toolbar)
        self.layout.addLayout(self.control_layout)
        self.layout.addWidget(self.canvas)
        
        self.tab_title = title
        self.plot_data(self.tab_title, "Position")
        
    def on_combo_changed(self, text):
        self.plot_data(self.tab_title, text)
        
    def plot_data(self, title: str, data_type: str):
        self.ax.clear()
        
        # 대소문자 및 공백 무시하여 time 컬럼 찾기
        time_col = next((c for c in self.df.columns if c.strip().lower() == 'time'), None)
        if time_col is None:
            return
            
        time_data = self.df[time_col]
        
        # 대소문자 및 공백 무시하여 제외할 컬럼 목록(frame, time) 구성
        exclude_cols = [c for c in self.df.columns if c.strip().lower() in ['frame', 'time']]
        
        for col in self.df.columns:
            if col not in exclude_cols:
                y_data = self.df[col].values
                
                # 미분을 통한 속도/가속도 계산
                if data_type == "Velocity":
                    y_data = np.gradient(y_data, time_data.values)
                elif data_type == "Acceleration":
                    v_data = np.gradient(y_data, time_data.values)
                    y_data = np.gradient(v_data, time_data.values)
                    
                self.ax.plot(time_data, y_data, label=col.strip())
                
        self.ax.set_title(f"{title} ({data_type})", fontsize=9)
        self.ax.set_xlabel('Time (s)', fontsize=9)
        
        ylabel = 'Value'
        if data_type == "Position": ylabel = "Position (m/deg)"
        elif data_type == "Velocity": ylabel = "Velocity"
        elif data_type == "Acceleration": ylabel = "Acceleration"
        self.ax.set_ylabel(ylabel, fontsize=9)
        
        self.ax.grid(True, linestyle="--", alpha=0.7)
        self.ax.tick_params(axis='both', which='major', labelsize=9)
        
        # 레전드 우측 배치
        self.ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), prop={'size': 9})
        self.fig.tight_layout()
        self.canvas.draw()


class PlotWindowUtil(QMainWindow):
    def __init__(self, data_folder: Union[str, Path], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Simulation Data Monitor")
        self.resize(1000, 700)
        
        self.data_folder = Path(data_folder)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)
        
        self.load_csv_data()
        
    def load_csv_data(self):
        """지정된 폴더의 모든 CSV를 읽어 탭으로 추가합니다."""
        if not self.data_folder.exists() or not self.data_folder.is_dir():
            QMessageBox.warning(self, "Error", f"Data folder not found: {self.data_folder}")
            return
            
        csv_files = list(self.data_folder.glob("*.csv"))
        if not csv_files:
            QMessageBox.information(self, "Info", "No CSV files found in the directory.")
            return
            
        # CSV 파일 정렬 (engineering 먼저, 그 다음 알파벳 순)
        csv_files.sort(key=lambda x: (x.name != 'engineering.csv', x.name))
        
        for csv_file in csv_files:
            try:
                # pandas로 읽을 때 UTF-8 명시 (인코딩 표준 준수), 메타데이터(#) 주석 무시
                df = pd.read_csv(csv_file, encoding='utf-8', comment='#')
                tab_title = csv_file.stem
                tab = PlotTab(tab_title, df)
                self.tabs.addTab(tab, tab_title)
            except Exception as e:
                print(f"Failed to load {csv_file.name}: {e}")

def show_plot_window(data_folder: Union[str, Path]):
    """외부에서 쉽게 호출할 수 있는 헬퍼 함수입니다."""
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
        
    window = PlotWindowUtil(data_folder)
    window.show()
    
    if not QApplication.instance().activeWindow():
        app.exec()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_folder = sys.argv[1]
        show_plot_window(target_folder)
    else:
        print("Usage: python wht_plotwindowutil.py [path_to_data_folder]")
