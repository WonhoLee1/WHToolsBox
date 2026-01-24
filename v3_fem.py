import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation as R
from numba import njit

# ==========================================
# 0. 전역 상수 설정
# ==========================================
BOX_WIDTH, BOX_DEPTH, BOX_HEIGHT = 2.0, 1.6, 0.2
BOX_MASS = 30.0
NX, NY, NZ = 5, 5, 3  # 요소 분할
GRAVITY = 9.81
GROUND_K = 2.0e9      # 지면 강성
DAMPING_RATIO = 0.9   # 감쇠비
INITIAL_COM_Z = 0.7
CORNER_RELATIVE_Z = np.array([0.05, 0.02, 0.0, 0.03]) # 바닥 4점 초기 틀어짐

# ==========================================
# 1. 노드 생성 및 외곽면 인덱스 추출
# ==========================================
def get_fem_structure():
    x = np.linspace(-BOX_WIDTH/2, BOX_WIDTH/2, NX+1)
    y = np.linspace(-BOX_DEPTH/2, BOX_DEPTH/2, NY+1)
    z = np.linspace(-BOX_HEIGHT/2, BOX_HEIGHT/2, NZ+1)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    nodes_local = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    
    # 8개 꼭짓점 인덱스 추출
    corners_idx = []
    for cz in [z[0], z[-1]]:
        for cy in [y[0], y[-1]]:
            for cx in [x[0], x[-1]]:
                dist = np.sum((nodes_local - [cx, cy, cz])**2, axis=1)
                corners_idx.append(np.argmin(dist))
                
    # 외곽면(Free Surface) 노드 추출 (Shading용)
    # 각 면의 노드 인덱스를 평면으로 구성
    surfaces = []
    ni, nj, nk = NX+1, NY+1, NZ+1
    idx_map = np.arange(nodes_local.shape[0]).reshape((ni, nj, nk))
    
    surfaces.append(idx_map[0, :, :])   # Left
    surfaces.append(idx_map[-1, :, :])  # Right
    surfaces.append(idx_map[:, 0, :])   # Front
    surfaces.append(idx_map[:, -1, :])  # Back
    surfaces.append(idx_map[:, :, 0])   # Bottom
    surfaces.append(idx_map[:, :, -1])  # Top
    
    return nodes_local, np.array(corners_idx), surfaces

NODES_LOCAL, CORNERS_IDX, SURFACE_MAPS = get_fem_structure()

# ==========================================
# 2. 물리 엔진 (Numba 가속)
# ==========================================
@njit
def compute_physics(h, v, rot, omegas, nodes_local, mass):
    # 1. 노드당 분배된 기본 물리량
    num_nodes = nodes_local.shape[0]
    # 침투 최소화를 위해 지면 강성을 기존보다 높게 설정 (상단 GROUND_K 참조)
    k_node = GROUND_K / num_nodes
    # 임계 감쇠 계수 계산
    c_node = 2.0 * np.sqrt(k_node * (mass / num_nodes)) * DAMPING_RATIO
    
    total_f_z = 0.0
    total_tau = np.zeros(3)
    node_pos_z = np.zeros(num_nodes)
    node_vel_z = np.zeros(num_nodes)
    
    # 2. 모든 노드 순회 (5x5x3 분할 노드 전체 대상)
    for i in range(num_nodes):
        # 강체 회전 변환 및 세계 좌표 계산
        r_l = nodes_local[i]
        r_w = rot @ r_l
        
        # 노드의 현재 높이(pos_z)와 수직 속도(vel_z)
        pos_z = h + r_w[2]
        vel_z = v + (omegas[0]*r_w[1] - omegas[1]*r_w[0])
        
        fz = 0.0
        # 3. 침투 발생 시 접촉력 계산 (Penetration Handling)
        if pos_z < 0:
            penetration = abs(pos_z)
            
            # [개선 1] 비선형 탄성력: 침투가 깊어질수록 강성이 제곱으로 증가 (Hertzian 모사)
            # 선형 스프링보다 침투 억제력이 훨씬 강력함
            f_elastic = k_node * (penetration**1.5) 
            
            # [개선 2] 가변 감쇠: 침투량에 비례하여 감쇠력을 조절하여 불연속적인 튐 방지
            # 감쇠력 = c * v * (침투 깊이 비율)
            f_damping = -c_node * vel_z * (1.0 + 10.0 * penetration)
            
            # 최종 수직 항력 (바닥 아래로만 작용)
            fz = max(0.0, f_elastic + f_damping)
            
        # 4. 전체 힘과 토크 누적
        total_f_z += fz
        total_tau[0] += r_w[1] * fz  # Roll 토크 (Y축 방향 힘에 의한 X축 회전)
        total_tau[1] -= r_w[0] * fz  # Pitch 토크 (X축 방향 힘에 의한 Y축 회전)
        
        # 데이터 저장
        node_pos_z[i] = pos_z
        node_vel_z[i] = vel_z
        
    return total_f_z, total_tau, node_pos_z, node_vel_z

# ==========================================
# 3. 시뮬레이션 및 데이터 처리
# ==========================================
class FEMBoxSimulator:
    def __init__(self):
        self.I_inv = np.linalg.inv(np.diag([
            (1/12)*BOX_MASS*(BOX_DEPTH**2 + BOX_HEIGHT**2),
            (1/12)*BOX_MASS*(BOX_WIDTH**2 + BOX_HEIGHT**2),
            (1/12)*BOX_MASS*(BOX_WIDTH**2 + BOX_DEPTH**2)
        ]))

    def ode(self, t, y):
        h, v, phi, theta, psi, wx, wy, wz = y
        rot = R.from_euler('xyz', [phi, theta, psi]).as_matrix()
        fz, tau, _, _ = compute_physics(h, v, rot, np.array([wx, wy, wz]), NODES_LOCAL, BOX_MASS)
        dv = -GRAVITY + fz / BOX_MASS
        dw = self.I_inv @ tau
        return [v, dv, wx, wy, wz, dw[0], dw[1], dw[2]]

    def run(self):
        p = np.arctan2(CORNER_RELATIVE_Z[3] - CORNER_RELATIVE_Z[0], BOX_DEPTH)
        r = np.arctan2(CORNER_RELATIVE_Z[1] - CORNER_RELATIVE_Z[0], BOX_WIDTH)
        sol = solve_ivp(self.ode, (0, 0.8), [INITIAL_COM_Z, 0, r, p, 0, 0, 0, 0], 
                        method='RK45', max_step=0.001)
        return sol

class Analysis:
    def __init__(self, sol):
        self.t, self.y = sol.t, sol.y
        self.corner_data = {idx: {'pos':[], 'vel':[], 'acc':[]} for idx in CORNERS_IDX}
        
    def process(self):
        for i in range(len(self.t)):
            yi = self.y[:, i]
            rot = R.from_euler('xyz', yi[2:5]).as_matrix()
            fz, tau, pz, vz = compute_physics(yi[0], yi[1], rot, yi[5:8], NODES_LOCAL, BOX_MASS)
            
            # 가속도 계산용
            dwdt = self.I_inv_val @ tau
            a_com = -GRAVITY + fz / BOX_MASS
            
            for idx in CORNERS_IDX:
                rw = rot @ NODES_LOCAL[idx]
                self.corner_data[idx]['pos'].append(yi[0] + rw[2])
                self.corner_data[idx]['vel'].append(yi[1] + (yi[5]*rw[1] - yi[6]*rw[0]))
                # 가속도 근사: a_com + alpha x r
                self.corner_data[idx]['acc'].append(a_com + (dwdt[0]*rw[1] - dwdt[1]*rw[0]))

    def plot_corners(self):
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        fig.suptitle("8개 꼭짓점 상세 물리 데이터 분석")
        
        for idx in CORNERS_IDX:
            axes[0].plot(self.t, self.corner_data[idx]['pos'], alpha=0.7)
            axes[1].plot(self.t, self.corner_data[idx]['vel'], alpha=0.7)
            axes[2].plot(self.t, self.corner_data[idx]['acc'], alpha=0.7)
            
        axes[0].set_ylabel("Disp (m)"); axes[0].grid(True)
        axes[1].set_ylabel("Vel (m/s)"); axes[1].grid(True)
        axes[2].set_ylabel("Accel (m/s²)"); axes[2].grid(True); axes[2].set_xlabel("Time (s)")
        plt.tight_layout()

    def animate_shading(self):
        fig = plt.figure("3D Surface Shading Animation", figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        def update(frame):
            ax.cla()
            yi = self.y[:, frame]
            rot = R.from_euler('xyz', yi[2:5]).as_matrix()
            nodes_world = (rot @ NODES_LOCAL.T).T + [0, 0, yi[0]]
            
            # 6개 면 Shading 처리
            for s_idx in SURFACE_MAPS:
                sx = nodes_world[s_idx, 0]
                sy = nodes_world[s_idx, 1]
                sz = nodes_world[s_idx, 2]
                ax.plot_surface(sx, sy, sz, color='cyan', alpha=0.6, edgecolor='blue', lw=0.5)
            
            # 바닥 평면
            gx, gy = np.meshgrid([-1.5, 1.5], [-1.5, 1.5])
            ax.plot_surface(gx, gy, np.zeros_like(gx), color='gray', alpha=0.2)
            
            ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_zlim(-0.1, 1.2)
            ax.set_title(f"Time: {self.t[frame]:.3f}s")

        ani = FuncAnimation(fig, update, frames=range(0, len(self.t), 5), interval=30)
        plt.show()

# ==========================================
# 4. 실행
# ==========================================
if __name__ == "__main__":
    sim = FEMBoxSimulator()
    sol = sim.run()
    
    # 분석 데이터 처리를 위한 관성행렬 전달
    ana = Analysis(sol)
    ana.I_inv_val = sim.I_inv
    ana.process()
    
    ana.plot_corners()    # 그래프 추가 (속도-변위-가속도)
    ana.animate_shading() # 3D Plot 수정 (Surface Mesh Shading)