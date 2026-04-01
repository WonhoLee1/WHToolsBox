import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6 import QtWidgets, QtCore, QtGui
from scipy.interpolate import RBFInterpolator
import warnings
import sys

# ==========================================
# 1. Kinematics Module (기구학 및 좌표계 관리)
# ==========================================
class KinematicsManager:
    def __init__(self, marker_history):
        self.raw_data = marker_history
        self.n_frames, self.n_markers, _ = marker_history.shape
        self._setup_global_to_local()

    def _setup_global_to_local(self):
        """Frame 0을 기준으로 3D 공간상의 로컬 평면 좌표계(Basis) 추출"""
        valid_P0 = self.raw_data[0, ~np.isnan(self.raw_data[0, :, 0])]
        if len(valid_P0) < 4:
            raise ValueError("초기 프레임에 유효한 마커가 최소 4개 이상 필요합니다.")
            
        self.centroid_0 = np.mean(valid_P0, axis=0)
        P0_centered = valid_P0 - self.centroid_0
        
        # PCA 축 추출
        cov = np.cov(P0_centered.T)
        evals, evecs = np.linalg.eigh(cov)
        idx = evals.argsort()[::-1]
        self.local_axes = evecs[:, idx] # [L_x, L_y, Normal]
        
        # 로컬 투영을 통한 해석 영역(Bounding Box) 유추
        p_local = P0_centered @ self.local_axes
        margin = 0.15
        self.x_bounds = [p_local[:, 0].min()*(1-margin), p_local[:, 0].max()*(1+margin)]
        self.y_bounds = [p_local[:, 1].min()*(1-margin), p_local[:, 1].max()*(1+margin)]

    def extract_kinematics(self, frame_idx):
        """특정 프레임의 강체 운동(R, c_Q) 분리 및 로컬 변위 좌표 반환"""
        valid_mask_0 = ~np.isnan(self.raw_data[0, :, 0])
        valid_mask_f = ~np.isnan(self.raw_data[frame_idx, :, 0])
        common_mask = valid_mask_0 & valid_mask_f
        
        P_common = self.raw_data[0, common_mask]
        Q_common = self.raw_data[frame_idx, common_mask]
        
        if len(P_common) < 3:
            warnings.warn(f"Frame {frame_idx}: 공통 마커 부족.")
            return None, np.eye(3), np.zeros(3), np.zeros(3), np.empty((0,3)), None
            
        c_P = np.mean(P_common, axis=0)
        c_Q = np.mean(Q_common, axis=0)
        
        # Kabsch Algorithm (Q를 P에 정렬)
        H = (Q_common - c_Q).T @ (P_common - c_P)
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0: 
            Vt[-1, :] *= -1; R = Vt.T @ U.T
            
        # 1. 초기 마커의 로컬 좌표 (Reference)
        P_valid = self.raw_data[0, valid_mask_0]
        P_local = (P_valid - self.centroid_0) @ self.local_axes
        
        # 2. 현재 마커의 로컬 좌표 (Aligned to P0)
        Q_valid = self.raw_data[frame_idx, valid_mask_0]
        if np.any(np.isnan(Q_valid)):
            Q_valid = P_valid
            
        Q_aligned = (Q_valid - c_Q) @ R.T + c_P
        Q_local = (Q_aligned - self.centroid_0) @ self.local_axes
        
        # 원본 글로벌 유효 마커
        Q_global_valid = self.raw_data[frame_idx, valid_mask_f]
        
        return Q_local, R, c_Q, c_P, Q_global_valid, P_local

# ==========================================
# 2. Mechanics Solver Module (평판 역학 해석)
# ==========================================
class PlateMechanicsSolver:
    def __init__(self, thickness, E, nu, grid_res=25):
        self.t = thickness
        self.E = E
        self.nu = nu
        self.D = (E * thickness**3) / (12 * (1 - nu**2))
        self.grid_res = grid_res

    def setup_mesh(self, x_bounds, y_bounds):
        """해석을 위한 정형 그리드 생성"""
        self.x_lin = np.linspace(x_bounds[0], x_bounds[1], self.grid_res)
        self.y_lin = np.linspace(y_bounds[0], y_bounds[1], self.grid_res)
        self.X_mesh, self.Y_mesh = np.meshgrid(self.x_lin, self.y_lin)
        self.grid_points = np.column_stack((self.X_mesh.ravel(), self.Y_mesh.ravel()))
        self.ds_x = self.x_lin[1] - self.x_lin[0]
        self.ds_y = self.y_lin[1] - self.y_lin[0]

    def solve(self, Q_local, P_local):
        """Relative Marker Displacement 기반 변형장, 응률장 연산"""
        if Q_local is None or len(Q_local) < 4:
            empty = np.zeros_like(self.X_mesh)
            return empty, empty, 0.0, 0.0, empty

        # 1. 변위 추출 (Z축 방향 상대 변위)
        dz = Q_local[:, 2] - P_local[:, 2]
        
        # RBF 표면 보간
        rbf = RBFInterpolator(P_local[:, :2], dz, kernel='thin_plate_spline')
        W = rbf(self.grid_points).reshape(self.X_mesh.shape)
        
        # 곡률 계산
        dw_dx, dw_dy = np.gradient(W, self.ds_x, self.ds_y)
        d2w_dx2, d2w_dxdy = np.gradient(dw_dx, self.ds_x, self.ds_y)
        _, d2w_dy2 = np.gradient(dw_dy, self.ds_x, self.ds_y)
        k_xx, k_yy, k_xy = -d2w_dx2, -d2w_dy2, -d2w_dxdy
        
        # 응력 및 변형률
        sig_x = (self.E * self.t / (2*(1-self.nu**2))) * (k_xx + self.nu*k_yy)
        sig_y = (self.E * self.t / (2*(1-self.nu**2))) * (k_yy + self.nu*k_xx)
        tau_xy = (self.E * self.t / (2*(1+self.nu))) * k_xy
        vm_stress = np.sqrt(sig_x**2 - sig_x*sig_y + sig_y**2 + 3*tau_xy**2)
        eps_eq = vm_stress / self.E 
        
        # 변형 에너지
        U_dens = 0.5 * self.D * ((k_xx + k_yy)**2 - 2*(1-self.nu)*(k_xx*k_yy - k_xy**2))
        total_energy = np.sum(U_dens) * (self.ds_x * self.ds_y)
        
        return W, vm_stress, total_energy, np.max(vm_stress), eps_eq

# ==========================================
# 3. Qt Visualization Module
# ==========================================
class QtVisualizer(QtWidgets.QMainWindow):
    def __init__(self, kinematics, solver, results):
        super().__init__()
        self.kin = kinematics
        self.solver = solver
        self.res = results
        self.current_frame = 0
        
        self.setWindowTitle("WHTOOLS: Multi-View Plate Analyzer (Spatial Mode + Multi-Charts)")
        self.resize(1700, 900)
        
        self._init_ui()
        self.update_frame(0)

    def _init_ui(self):
        font_name = "Noto Sans KR"
        if font_name not in QtGui.QFontDatabase.families():
            font_name = "Segoe UI"
        self.app_font = QtGui.QFont(font_name, 8)
        self.setFont(self.app_font)

        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        vbox = QtWidgets.QVBoxLayout(main_widget)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        
        # 1. Global View (Spatial) - PyVista
        self.pv_global = QtInteractor(self)
        splitter.addWidget(self.pv_global)
        self.pv_global.set_background("white")
        label_global = QtWidgets.QLabel("Spatial View (Global Motion)", self.pv_global)
        label_global.move(10, 10)
        label_global.setFont(self.app_font)

        # 2. Analysis Panel (Matplotlib 2x2)
        self.figure = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        splitter.addWidget(self.canvas)
        
        vbox.addWidget(splitter)

        control_panel = QtWidgets.QWidget()
        hbox = QtWidgets.QHBoxLayout(control_panel)
        self.frame_label = QtWidgets.QLabel(f"Frame: 0 / {self.kin.n_frames - 1}")
        self.frame_label.setFont(self.app_font)
        hbox.addWidget(self.frame_label)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, self.kin.n_frames - 1)
        self.slider.valueChanged.connect(self.update_frame)
        hbox.addWidget(self.slider)
        vbox.addWidget(control_panel)

        self._setup_pyvista()

    def _setup_pyvista(self):
        x, y = self.solver.X_mesh, self.solver.Y_mesh
        self.mesh_global = pv.StructuredGrid(x, y, np.zeros_like(x))
        self.mesh_global["Strain"] = np.zeros(x.size).astype(np.float32)
        
        all_raw = self.kin.raw_data[~np.isnan(self.kin.raw_data)].reshape(-1, 3)
        self.pv_global.reset_camera(bounds=(all_raw[:,0].min()-0.2, all_raw[:,0].max()+0.2, 
                                            all_raw[:,1].min()-0.2, all_raw[:,1].max()+0.2, 
                                            all_raw[:,2].min()-0.2, all_raw[:,2].max()+0.2))

        self.pv_global.add_mesh(self.mesh_global, scalars="Strain", cmap="jet", show_edges=True, name="plate")
        self.pv_global.add_key_event("Right", lambda: self.slider.setValue(min(self.slider.value() + 1, self.kin.n_frames - 1)))
        self.pv_global.add_key_event("Left", lambda: self.slider.setValue(max(self.slider.value() - 1, 0)))

    def update_frame(self, f_idx):
        self.current_frame = f_idx
        self.frame_label.setText(f"Frame: {f_idx} / {self.kin.n_frames - 1}")
        
        W = self.res['W'][f_idx]
        Strain = self.res['Strain'][f_idx]
        strain_flat = Strain.ravel().astype(np.float32)
        
        # 1. Global Update (Reverse Kinematics)
        R = self.res['R'][f_idx]
        c_Q = self.res['c_Q'][f_idx]
        c_P = self.res['c_P'][f_idx]
        
        pts_local = np.vstack([self.solver.X_mesh.ravel(), self.solver.Y_mesh.ravel(), W.ravel()]).T
        pts_intermediate = (pts_local @ self.kin.local_axes.T) + self.kin.centroid_0 - c_P
        pts_global = (pts_intermediate @ R) + c_Q
        
        self.mesh_global.points = pts_global
        self.mesh_global.point_data["Strain"] = strain_flat
        self.mesh_global.set_active_scalars("Strain")
        
        s_min, s_max = strain_flat.min(), strain_flat.max()
        self.pv_global.add_mesh(self.mesh_global, scalars="Strain", cmap="jet", name="plate", 
                                clim=[s_min, max(s_max, s_min+1e-12)], show_scalar_bar=True, reset_camera=False)
        
        m_global = self.res['M_Global'][f_idx]
        if m_global is not None:
            self.pv_global.add_mesh(pv.PolyData(m_global).glyph(scale=False, orient=False, geom=pv.Sphere(radius=0.008)), color="red", name="markers", lighting=False)

        self.pv_global.render()
        self._update_matplotlib(f_idx)

    def _update_matplotlib(self, f_idx):
        self.figure.clear()
        ax1 = self.figure.add_subplot(221)
        ax2 = self.figure.add_subplot(222)
        ax3 = self.figure.add_subplot(212)

        # Displacement Contour
        W_data = self.res['W'][f_idx]
        im1 = ax1.contourf(self.solver.X_mesh, self.solver.Y_mesh, W_data, cmap="RdBu_r")
        self.figure.colorbar(im1, ax=ax1, format="%.1e")
        ax1.set_title("Displacement [W]", fontsize=9)
        ax1.set_aspect('equal')

        # Strain Contour
        Strain_data = self.res['Strain'][f_idx]
        im2 = ax2.contourf(self.solver.X_mesh, self.solver.Y_mesh, Strain_data, cmap="jet")
        self.figure.colorbar(im2, ax=ax2, format="%.1e")
        ax2.set_title("Eq. Strain", fontsize=9)
        ax2.set_aspect('equal')

        # History Plot
        ax3.plot(self.res['Energy'], 'b-', label='Strain Energy (J)', alpha=0.7)
        ax3_twin = ax3.twinx()
        ax3_twin.plot(self.res['Max_S'], 'r--', label='Max Stress (Pa)', alpha=0.7)
        ax3.axvline(x=f_idx, color='k', ls=':', label='Current')
        ax3.legend(loc='upper left', fontsize=7)
        ax3.set_xlabel("Frame", fontsize=8)
        
        self.figure.tight_layout()
        self.canvas.draw()

# ==========================================
# 4. Main Controller
# ==========================================
class ShellDeformationAnalyzer:
    def __init__(self, marker_history, thickness=0.002, E=210e9, nu=0.3):
        self.kinematics = KinematicsManager(marker_history)
        self.solver = PlateMechanicsSolver(thickness, E, nu)
        self.solver.setup_mesh(self.kinematics.x_bounds, self.kinematics.y_bounds)
        self.results = {
            'W': [], 'Stress': [], 'Strain': [], 'Energy': [], 'Max_S': [],
            'R': [], 'c_Q': [], 'c_P': [], 'M_Global': [], 'M_Local': []
        }
        
    def run_analysis(self):
        print("[System] Solving mechanics over all frames...")
        for f in range(self.kinematics.n_frames):
            q_local, R, c_Q, c_P, m_global, p_local = self.kinematics.extract_kinematics(f)
            W, stress, energy, max_s, strain = self.solver.solve(q_local, p_local)
            
            self.results['W'].append(W)
            self.results['Stress'].append(stress)
            self.results['Strain'].append(strain)
            self.results['Energy'].append(energy)
            self.results['Max_S'].append(max_s)
            self.results['R'].append(R)
            self.results['c_Q'].append(c_Q)
            self.results['c_P'].append(c_P)
            self.results['M_Global'].append(m_global)
            self.results['M_Local'].append(q_local)
        print("[System] Analysis complete.")
        
    def show_results(self):
        app = QtWidgets.QApplication(sys.argv)
        viewer = QtVisualizer(self.kinematics, self.solver, self.results)
        viewer.show()
        sys.exit(app.exec())

# --- 실행부 ---
if __name__ == "__main__":
    def create_sophisticated_test_data(frames=60):
        grid = np.meshgrid([-0.15, 0, 0.15], [-0.15, 0, 0.15])
        P0 = np.column_stack((grid[0].ravel(), grid[1].ravel(), np.zeros(9)))
        data = np.zeros((frames, 9, 3))
        for i in range(frames):
            t = i / frames * 2 * np.pi
            rx, ry, rz = 0.15*np.sin(t*1.2), 0.10*np.cos(t*0.8), 0.20*np.sin(t*1.5)
            Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
            Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
            Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
            R = Rz @ Ry @ Rx
            T = np.array([0.1*np.cos(t), 0.1*np.sin(t), 0.05*np.cos(t*1.3)])
            
            P_def = P0.copy()
            P_def[4, 2] = -0.04 * np.sin(t*2.0) # Center
            for idx in [0, 1, 2, 3, 5, 6, 7, 8]: # Boundary
                P_def[idx, 2] += 0.005 * np.sin(t*5.0 + idx)
            data[i] = (P_def @ R.T) + T
        return data

    analyzer = ShellDeformationAnalyzer(create_sophisticated_test_data(80))
    analyzer.run_analysis()
    analyzer.show_results()