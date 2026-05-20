import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RBFInterpolator
import warnings
import re

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
        margin = 0.1
        self.x_bounds = [p_local[:, 0].min()*(1-margin), p_local[:, 0].max()*(1+margin)]
        self.y_bounds = [p_local[:, 1].min()*(1-margin), p_local[:, 1].max()*(1+margin)]

    def extract_kinematics(self, frame_idx):
        """특정 프레임의 강체 운동(R, c_Q) 분리 및 로컬 변형 좌표 반환"""
        valid_mask_0 = ~np.isnan(self.raw_data[0, :, 0])
        valid_mask_f = ~np.isnan(self.raw_data[frame_idx, :, 0])
        common_mask = valid_mask_0 & valid_mask_f
        
        P_common = self.raw_data[0, common_mask]
        Q_common = self.raw_data[frame_idx, common_mask]
        
        if len(P_common) < 3:
            warnings.warn(f"Frame {frame_idx}: 공통 마커 부족.")
            return None, np.eye(3), np.zeros(3), np.zeros(3), np.empty((0,3))
            
        c_P = np.mean(P_common, axis=0)
        c_Q = np.mean(Q_common, axis=0)
        
        # Kabsch Algorithm
        H = (P_common - c_P).T @ (Q_common - c_Q)
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0: 
            Vt[-1, :] *= -1; R = Vt.T @ U.T
            
        # 순수 변형 로컬 좌표 추출
        Q_valid = self.raw_data[frame_idx, valid_mask_f]
        Q_aligned = (Q_valid - c_Q) @ R
        Q_local = (Q_aligned + c_P - self.centroid_0) @ self.local_axes
        
        # 원본 글로벌 유효 마커
        Q_global_valid = self.raw_data[frame_idx, valid_mask_f]
        
        return Q_local, R, c_Q, c_P, Q_global_valid

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

    def solve(self, local_markers):
        """로컬 마커를 기반으로 변형장, 응력장, 에너지 연산"""
        if local_markers is None or len(local_markers) < 4:
            return np.zeros_like(self.X_mesh), np.zeros_like(self.X_mesh), 0.0, 0.0

        # RBF 표면 보간
        rbf = RBFInterpolator(local_markers[:, :2], local_markers[:, 2], kernel='thin_plate_spline')
        W = rbf(self.grid_points).reshape(self.X_mesh.shape)
        
        # 곡률 계산 (2차 차분)
        dw_dx, dw_dy = np.gradient(W, self.ds_x, self.ds_y)
        d2w_dx2, d2w_dxdy = np.gradient(dw_dx, self.ds_x, self.ds_y)
        _, d2w_dy2 = np.gradient(dw_dy, self.ds_x, self.ds_y)
        k_xx, k_yy, k_xy = -d2w_dx2, -d2w_dy2, -d2w_dxdy
        
        # von-Mises 응력
        sig_x = (self.E * self.t / (2*(1-self.nu**2))) * (k_xx + self.nu*k_yy)
        sig_y = (self.E * self.t / (2*(1-self.nu**2))) * (k_yy + self.nu*k_xx)
        tau_xy = (self.E * self.t / (2*(1+self.nu))) * k_xy
        vm_stress = np.sqrt(sig_x**2 - sig_x*sig_y + sig_y**2 + 3*tau_xy**2)
        
        # 변형 에너지
        U_dens = 0.5 * self.D * ((k_xx + k_yy)**2 - 2*(1-self.nu)*(k_xx*k_yy - k_xy**2))
        total_energy = np.sum(U_dens) * (self.ds_x * self.ds_y)
        max_stress = np.max(vm_stress)
        
        return W, vm_stress, total_energy, max_stress

# ==========================================
# 3. Post-Processor Module (시각화 및 UI)
# ==========================================
class PostProcessor:
    def __init__(self, kinematics, solver, results):
        self.kin = kinematics
        self.solver = solver
        self.res = results
        
        # 절대 좌표계 범위 계산 (Plot 한계 고정용)
        valid_all = self.kin.raw_data[~np.isnan(self.kin.raw_data)]
        self.v_min, self.v_max = valid_all.min(), valid_all.max()
        self.global_max_stress = np.max(self.res['Max_S']) + 1e-9

    def _inverse_kinematics_transform(self, W, f_idx):
        """로컬 메쉬를 절대 좌표계 공간으로 역변환 (Spatial 모드용)"""
        R = self.res['R'][f_idx]
        c_Q = self.res['c_Q'][f_idx]
        c_P = self.res['c_P'][f_idx]
        
        pts_local = np.vstack([self.solver.X_mesh.ravel(), self.solver.Y_mesh.ravel(), W.ravel()]).T
        pts_global = (((pts_local @ self.kin.local_axes.T) - c_P + self.kin.centroid_0) @ R.T) + c_Q
        
        X = pts_global[:, 0].reshape(self.solver.X_mesh.shape)
        Y = pts_global[:, 1].reshape(self.solver.Y_mesh.shape)
        Z = pts_global[:, 2].reshape(self.solver.Y_mesh.shape)
        return X, Y, Z

    def display(self):
        print("\n" + "="*60)
        print(" 1. Spatial Frame (절대 좌표계 모션 중심)")
        print(" 2. Material Frame (평판 중심 변형 관찰)")
        print("="*60)
        mode = input("모드 선택 (1 또는 2) >> ").strip()
        viz_mode = 'spatial' if mode == '1' else 'material'
        
        fig = plt.figure(figsize=(16, 8))
        
        def update_plot(f_idx):
            plt.clf()
            ax1 = fig.add_subplot(121, projection='3d')
            W, Stress = self.res['W'][f_idx], self.res['Stress'][f_idx]
            
            if np.all(W == 0) and f_idx > 0:
                ax1.text2D(0.5, 0.5, "Data Loss", transform=ax1.transAxes, color='red', fontsize=20, ha='center')
                return

            color_field = plt.cm.jet(Stress / self.global_max_stress)

            if viz_mode == 'material':
                X_plot, Y_plot, Z_plot = self.solver.X_mesh, self.solver.Y_mesh, W
                z_limit = np.max(np.abs(self.res['W']))
                ax1.set_zlim(-z_limit - 0.01, z_limit + 0.01)
                ax1.set_xlabel('Local X'); ax1.set_ylabel('Local Y'); ax1.set_zlabel('Deformation W')
            else:
                X_plot, Y_plot, Z_plot = self._inverse_kinematics_transform(W, f_idx)
                ax1.set_xlim(self.v_min, self.v_max); ax1.set_ylim(self.v_min, self.v_max); ax1.set_zlim(self.v_min, self.v_max)
                ax1.set_xlabel('Global X'); ax1.set_ylabel('Global Y'); ax1.set_zlabel('Global Z')
                
                m_global = self.res['M_Global'][f_idx]
                if len(m_global) > 0:
                    ax1.scatter(m_global[:,0], m_global[:,1], m_global[:,2], c='k', s=50)

            ax1.plot_surface(X_plot, Y_plot, Z_plot, facecolors=color_field, shade=False, alpha=0.8)
            ax1.set_title(f"Frame {f_idx} ({viz_mode.capitalize()} View)\nStress Field")
            
            ax2 = fig.add_subplot(222); ax2.plot(self.res['Energy'], 'b-'); ax2.axvline(x=f_idx, color='r', ls='--')
            ax2.set_title("Strain Energy (J)")
            
            ax3 = fig.add_subplot(224); ax3.plot(self.res['Max_S'], 'g-'); ax3.axvline(x=f_idx, color='r', ls='--')
            ax3.set_title("Max Stress (Pa)")
            plt.tight_layout(); plt.draw()

        self._run_ui_loop(update_plot)

    def _run_ui_loop(self, update_plot_func):
        print(f"\n[Player] 0 ~ {self.kin.n_frames-1} 프레임 ('숫자', 'a', 'a시작-종료', 'q')")
        while True:
            val = input("\nCmd >> ").strip().lower()
            if val in ['q', 'exit']: break
            if val.startswith('a'):
                s, e = 0, self.kin.n_frames - 1
                match = re.match(r'a(\d+)-(\d+)', val)
                if match: 
                    s, e = max(0, int(match.group(1))), min(self.kin.n_frames-1, int(match.group(2)))
                for f in range(s, e + 1):
                    update_plot_func(f); plt.show(block=False); plt.pause(0.03)
                continue
            try:
                idx = int(val)
                if 0 <= idx < self.kin.n_frames:
                    update_plot_func(idx); plt.show(block=False); plt.pause(0.1)
            except: pass

# ==========================================
# 4. Main Controller (해석 파이프라인 관장)
# ==========================================
class ShellDeformationAnalyzer:
    def __init__(self, marker_history, thickness=0.002, E=210e9, nu=0.3):
        print("[System] Initializing Modular Shell Analysis...")
        
        # 1. 모듈 초기화
        self.kinematics = KinematicsManager(marker_history)
        self.solver = PlateMechanicsSolver(thickness, E, nu)
        self.solver.setup_mesh(self.kinematics.x_bounds, self.kinematics.y_bounds)
        
        # 2. 결과 저장용 컨테이너
        self.results = {
            'W': [], 'Stress': [], 'Energy': [], 'Max_S': [],
            'R': [], 'c_Q': [], 'c_P': [], 'M_Global': []
        }
        
    def run_analysis(self):
        print("[System] Solving mechanics over all frames...")
        for f in range(self.kinematics.n_frames):
            # 기구학 해석 (강체 분리)
            local_m, R, c_Q, c_P, m_global = self.kinematics.extract_kinematics(f)
            
            # 역학 해석 (응력, 변형률)
            W, stress, energy, max_s = self.solver.solve(local_m)
            
            # 데이터 적재
            self.results['W'].append(W)
            self.results['Stress'].append(stress)
            self.results['Energy'].append(energy)
            self.results['Max_S'].append(max_s)
            self.results['R'].append(R)
            self.results['c_Q'].append(c_Q)
            self.results['c_P'].append(c_P)
            self.results['M_Global'].append(m_global)
            
        print("[System] Analysis complete.")
        
    def show_results(self):
        # 3. 후처리 및 시각화 모듈 호출
        post_processor = PostProcessor(self.kinematics, self.solver, self.results)
        post_processor.display()


# --- 실행부 ---
if __name__ == "__main__":
    # 임의의 테스트 데이터 (이전 코드의 create_complex_test_data() 사용 가능)
    # marker_data = create_complex_test_data() 
    
    # 더미 데이터 생성 (간단화)
    frames = 30
    x, y = np.meshgrid([-0.2, 0, 0.2], [-0.2, 0, 0.2])
    data = np.zeros((frames, 9, 3))
    for i in range(frames):
        data[i, :, 0] = x.ravel() + i*0.01 # x 병진
        data[i, :, 1] = y.ravel()
        data[i, 4, 2] = -0.05 * np.sin(i*0.2) # 중심부 굽힘 변형
        
    # OOP 기반 메인 컨트롤러 실행
    analyzer = ShellDeformationAnalyzer(data, thickness=0.003, E=70e9, nu=0.33)
    analyzer.run_analysis()
    analyzer.show_results()
