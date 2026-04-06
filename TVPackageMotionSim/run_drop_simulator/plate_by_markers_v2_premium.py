import sys
import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6 import QtWidgets, QtCore, QtGui
from typing import List, Dict, Optional, Tuple

# [WHTOOLS] V2.1 Premium (Tab UI & Bootstrap Loading Mode)

class PlateConfig:
    def __init__(self):
        self.lx = 300.0
        self.ly = 200.0
        self.thickness = 1.0
        self.E = 70e3
        self.nu = 0.3
        self.rho = 2700

class ShellDeformationAnalyzer:
    """[WHTOOLS] Kirchhoff Plate Theory based Deformation & Stress Analyzer"""
    def __init__(self, lx=0, ly=0, thickness=1.0, E=70e9, nu=0.3, name="Part"):
        self.name = name
        self.lx = lx # mm
        self.ly = ly # mm
        self.thickness = thickness # mm
        self.E = E
        self.nu = nu
        self.D = (E * (thickness**3)) / (12 * (1 - nu**2))
        
        # Reference State (LCS)
        self.ref_markers = None # Initial marker positions in world
        self.ref_basis = None   # [u_vec, v_vec, w_vec] in world
        self.ref_center = None  # Center of markers in world
        self.o_data = None      # 2D Projected local coordinates [N, 2]
        
        self.results = {}

    def fit_reference_plane(self, m_data_init):
        """
        [WHTOOLS] SVD를 이용한 자율 평면 피팅 및 로컬 좌표계 생성
        m_data_init: [N_markers, 3] (Initial positions)
        """
        self.ref_markers = m_data_init.copy()
        self.ref_center = np.mean(m_data_init, axis=0)
        centered = m_data_init - self.ref_center
        
        # SVD로 주성분 추출
        _, _, Vh = np.linalg.svd(centered)
        
        u_vec = Vh[0] # 가로축 (가장 긴 축)
        v_vec = Vh[1] # 세로축
        w_vec = Vh[2] # 법선축
        
        self.ref_basis = np.array([u_vec, v_vec, w_vec]) # [3, 3]
        
        # 2D 투영 좌표 (o_data) 생성
        self.o_data = np.zeros((len(m_data_init), 2))
        self.o_data[:, 0] = centered @ u_vec
        self.o_data[:, 1] = centered @ v_vec
        
        # 만약 lx, ly가 0이면 데이터로부터 치수 도출
        if self.lx == 0:
            self.lx = float(np.max(self.o_data[:, 0]) - np.min(self.o_data[:, 0]))
        if self.ly == 0:
            self.ly = float(np.max(self.o_data[:, 1]) - np.min(self.o_data[:, 1]))
            
        print(f"[Analyzer] Autonomous Fit: {self.name} | Size: {self.lx:.1f} x {self.ly:.1f} mm")
        return self.o_data

    def remove_rigid_motion(self, m_data_frame):
        """
        [WHTOOLS] Orthogonal Procrustes를 이용한 강체 운동(Rigid Motion) 제거
        """
        if self.ref_markers is None: return None
        
        current_center = np.mean(m_data_frame, axis=0)
        P = (self.ref_markers - self.ref_center).T # [3, N]
        Q = (m_data_frame - current_center).T      # [3, N]
        
        H = P @ Q.T
        U, S, Vh = np.linalg.svd(H)
        R = Vh.T @ U.T
        
        m_rigid = (R.T @ Q).T 
        diff = m_rigid - (self.ref_markers - self.ref_center)
        w_displacement = diff @ self.ref_basis[2]
        
        return w_displacement

    def analyze(self, m_data_hist, o_data_hint=None):
        """
        [WHTOOLS] 전체 시간 이력에 대한 변형 분석 수행
        m_data_hist: [N_markers, N_frames, 3]
        """
        n_markers, n_frames, _ = m_data_hist.shape
        
        if self.ref_markers is None:
            self.fit_reference_plane(m_data_hist[:, 0, :])
            if o_data_hint is not None:
                self.o_data = o_data_hint 
        
        all_w = np.zeros((n_frames, n_markers))
        for f in range(n_frames):
            all_w[f] = self.remove_rigid_motion(m_data_hist[:, f, :])
            
        self.results['w_raw'] = all_w
        self.results['o_data'] = self.o_data
        self.results['displacement'] = np.zeros((n_frames, 20, 20)) # Placeholder
        
        return True


class PlateAssemblyManager:
    def __init__(self):
        self.analyzers = {}
        self.time_hist = None
        self.metadata = {}

    def add_part(self, name, analyzer):
        self.analyzers[name] = analyzer

    def run_all(self, result_data):
        self.time_hist = result_data.time_history
        for name, analyzer in self.analyzers.items():
            analyzer.analyze(self.time_hist, result_data.pos_hist, result_data.quat_hist, None)

def scale_result_to_mm(result):
    if result.pos_hist is not None:
        result.pos_hist *= 1000.0
    return result

class QtVisualizerV2(QtWidgets.QMainWindow):
    """[WHTOOLS] Premium Post-Processing Dashboard (Tab-Widget Based)"""
    def __init__(self, manager: Optional[PlateAssemblyManager] = None, config: PlateConfig = None, ground_size=(3000, 3000)):
        super().__init__()
        self.manager = manager
        self.setWindowTitle(f"WHTOOLS Premium Digital Twin Dashboard (Tab UI)")
        self.resize(1600, 1000)
        
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QtWidgets.QVBoxLayout(self.central_widget)
        
        self.tabs = QtWidgets.QTabWidget()
        self.main_layout.addWidget(self.tabs)
        
        self.tab_3d = QtWidgets.QWidget()
        self.tab_kinematics = QtWidgets.QWidget()
        self.tab_structural = QtWidgets.QWidget()
        
        self.tabs.addTab(self.tab_3d, "3D View")
        self.tabs.addTab(self.tab_kinematics, "Kinematics Analysis")
        self.tabs.addTab(self.tab_structural, "Structural Analysis")
        
        # ... (Subprocess Loading Logic) ...
        self.scrubber = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.main_layout.addWidget(self.scrubber)

    def load_data(self, pkl_path: str):
        """Bootstrap manager from .pkl file."""
        print(f"Loading results from {pkl_path}...")
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        # Re-hydration logic (Implementation Details)
        self.manager = PlateAssemblyManager()
        self.manager.run_all(data)
        print("Data Loading Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str, help="Path to pkl file")
    args = parser.parse_args()
    
    app = QtWidgets.QApplication(sys.argv)
    window = QtVisualizerV2()
    if args.load:
        window.load_data(args.load)
    window.show()
    sys.exit(app.exec())
