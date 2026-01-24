import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation as R
from numba import njit

# ==========================================
# 0. 전역 상수 설정 (User Parameters)
# ==========================================
BOX_WIDTH = 2.0       
BOX_DEPTH = 1.6       
BOX_MASS = 30.0       
WALL_THICKNESS = 0.05 
GRAVITY = 9.81        

INITIAL_HEIGHT = 0.3  
CORNER_RELATIVE_Z = np.array([0.25, 0.22, 0.0, 0.03]) 

YOUNGS_MODULUS = 1.0E6    
GROUND_K = 1.5e9          # 지면 강성 조정
DAMPING_RATIO = 0.8       # 감쇠 상향 (안정성 확보)
CONTACT_SMOOTH_DIST = 0.01 

AIR_RHO = 1.225       
DRAG_COEFF = 1.1      
GRID_N = 10           

T_END = 0.8           
MAX_STEP = 0.0005     

# ==========================================
# 1. Numba 가속: 물리 엔진 (가속도 계산 포함)
# ==========================================
@njit
def compute_physics_jit(h, v, rot, omegas, W, D, M, rho, E, Ib, corners_local):
    # --- 공기역학 ---
    da = (W * D) / (GRID_N**2)
    f_aero_z = 0.0
    tau_aero = np.zeros(3)
    for i in range(GRID_N):
        for j in range(GRID_N):
            lx, ly = -W/2 + (i+0.5)*(W/GRID_N), -D/2 + (j+0.5)*(D/GRID_N)
            r_w = rot @ np.array([lx, ly, 0.0])
            hg = max(h + r_w[2], 0.0005)
            vg = v + (omegas[0]*r_w[1] - omegas[1]*r_w[0])
            f_grid = -0.5 * rho * DRAG_COEFF * da * vg * abs(vg)
            if vg < 0:
                f_grid += 0.5 * rho * (abs(vg) * 3.2 / (7.2 * hg))**2 * da
            f_aero_z += f_grid
            tau_aero[0] += r_w[1] * f_grid
            tau_aero[1] -= r_w[0] * f_grid

    # --- 구조적 접촉 모델 ---
    f_contact_z = 0.0
    tau_contact = np.zeros(3)
    f_corners = np.zeros(4)
    deltas = np.zeros(4)
    
    k_eq = (3 * E * Ib) / ( (W/2)**3 )
    c_eq = 2.0 * np.sqrt(k_eq * (M/4.0)) * DAMPING_RATIO

    for idx in range(4):
        cw = rot @ corners_local[idx]
        hc = h + cw[2]
        vc = v + (omegas[0]*cw[1] - omegas[1]*cw[0])
        
        if hc < CONTACT_SMOOTH_DIST:
            activation = 0.5 * (1.0 - np.tanh(hc / (CONTACT_SMOOTH_DIST * 0.5)))
            f_elastic = GROUND_K * max(0.0, -hc)
            f_damping = -c_eq * vc * activation
            fz = max(0.0, (f_elastic + f_damping) * activation)
            
            f_contact_z += fz
            f_corners[idx] = fz
            tau_contact[0] += cw[1] * fz
            tau_contact[1] -= cw[0] * fz
            deltas[idx] = fz / k_eq
    
        # 변형량 계산 시 '강성 경화(Strain Hardening)' 효과 추가
        d_raw = fz / k_linear
        # d_raw가 커질수록 분모가 작아져서 변형 저항이 급증함
        deltas[idx] = d_raw / (1.0 + (d_raw / (L_eff * MAX_BENDING_RATIO))**2)
        
    return f_aero_z + f_contact_z, tau_aero + tau_contact, f_corners, deltas

# ==========================================
# 2. 결과 처리 및 시각화 (물리 데이터 확장)
# ==========================================
class BoxDropResult:
    def __init__(self, t, states, sim):
        self.t, self.states, self.sim = t, states, sim
        # 분석용 데이터 저장소
        self.com_accel = []
        self.corner_pos = [[] for _ in range(4)]
        self.corner_vel = [[] for _ in range(4)]
        self.corner_acc = [[] for _ in range(4)]
        self.energy_history = []

    def process(self):
        for i in range(len(self.t)):
            y = self.states[:, i]
            rot = R.from_euler('xyz', y[2:5]).as_matrix()
            w = y[5:8]
            
            # 물리 엔진 재호출로 가속도 및 변위 추출
            f_net, tau_net, f_corners, deltas = compute_physics_jit(
                y[0], y[1], rot, w, self.sim.width, self.sim.depth, self.sim.mass,
                AIR_RHO, self.sim.E, self.sim.Ib, self.sim.corners_local
            )
            
            # 1. CoM 가속도 (f/m)
            a_com = -GRAVITY + f_net / self.sim.mass
            self.com_accel.append(a_com)
            
            # 2. 각 꼭짓점 물리량 (변형 반영)
            dwdt = self.sim.I_inv @ tau_net
            for j in range(4):
                cl = self.sim.corners_local[j]
                cw = rot @ (cl + np.array([0, 0, deltas[j]]))
                
                # 변위
                self.corner_pos[j].append(y[0] + cw[2])
                # 속도 (v + w x r)
                v_corner = y[1] + (w[0]*cw[1] - w[1]*cw[0])
                self.corner_vel[j].append(v_corner)
                # 가속도 (a + alpha x r + w x (w x r)) -> 수직 성분 단순화
                a_corner = a_com + (dwdt[0]*cw[1] - dwdt[1]*cw[0])
                self.corner_acc[j].append(a_corner)
            
            self.energy_history.append(np.sum(0.5 * f_corners * deltas))

    def plot_physics(self):
        # 창 1: 변위 및 에너지
        plt.figure("분석 1: 변위 및 에너지", figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(self.t, self.states[0], 'k', lw=2, label='CoM Height')
        for j in range(4): plt.plot(self.t, self.corner_pos[j], '--', label=f'Corner {j+1}')
        plt.ylabel("Displacement (m)"); plt.grid(True); plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(self.t, self.energy_history, 'g', label='Strain Energy')
        plt.ylabel("Energy (J)"); plt.xlabel("Time (s)"); plt.grid(True); plt.legend()

        # 창 2: 속도 및 가속도
        plt.figure("분석 2: 속도 및 가속도", figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(self.t, self.states[1], 'k', lw=2, label='CoM Velocity')
        for j in range(4): plt.plot(self.t, self.corner_vel[j], '--', label=f'Corner {j+1} Vel')
        plt.ylabel("Velocity (m/s)"); plt.grid(True); plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(self.t, self.com_accel, 'k', lw=2, label='CoM Accel')
        for j in range(4): plt.plot(self.t, self.corner_acc[j], '--', label=f'Corner {j+1} Acc')
        plt.ylabel("Acceleration (m/s²)"); plt.xlabel("Time (s)"); plt.grid(True); plt.legend()

    def animate(self):
        fig = plt.figure("분석 3: 3D 동역학 애니메이션", figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        def update(frame):
            ax.cla()
            y = self.states[:, frame]
            rot = R.from_euler('xyz', y[2:5]).as_matrix()
            _, _, _, deltas = compute_physics_jit(y[0], y[1], rot, y[5:8], 
                                                 self.sim.width, self.sim.depth, self.sim.mass, 
                                                 AIR_RHO, self.sim.E, self.sim.Ib, self.sim.corners_local)
            w_8 = (rot @ self.sim.box_8_local.T).T + [0, 0, y[0]]
            for j in range(4): w_8[j] += rot @ np.array([0, 0, deltas[j]])
            edges = [[0,1,2,3,0], [4,5,6,7,4], [0,4], [1,5], [2,6], [3,7]]
            for e in edges: ax.plot(w_8[e,0], w_8[e,1], w_8[e,2], 'b-')
            ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_zlim(-0.1, 0.5)
            ax.set_title(f"T: {self.t[frame]:.3f}s | CoM Accel: {self.com_accel[frame]:.1f}")
        ani = FuncAnimation(fig, update, frames=range(0, len(self.t), 5), interval=10)
        plt.show()

# ==========================================
# 3. 시뮬레이터 클래스 (구조 유지)
# ==========================================
class BoxDropSimulator:
    def __init__(self):
        self.width, self.depth, self.mass = BOX_WIDTH, BOX_DEPTH, BOX_MASS
        self.E, self.Ib = YOUNGS_MODULUS, (BOX_DEPTH * WALL_THICKNESS**3) / 12
        self.corners_local = np.array([[-self.width/2, -self.depth/2, 0.], [self.width/2, -self.depth/2, 0.],
                                      [self.width/2, self.depth/2, 0.], [-self.width/2, self.depth/2, 0.]])
        self.box_8_local = np.vstack((self.corners_local, self.corners_local + np.array([0,0,0.1])))
        self.I_inv = np.linalg.inv(np.diag([(1/12)*self.mass*self.depth**2, 
                                            (1/12)*self.mass*self.width**2, 
                                            (1/12)*self.mass*(self.width**2+self.depth**2)]))

    def ode(self, t, y):
        h, v, phi, theta, psi, wx, wy, wz = y
        rot = R.from_euler('xyz', [phi, theta, psi]).as_matrix()
        f_net, tau_net, _, _ = compute_physics_jit(h, v, rot, np.array([wx, wy, wz]), 
                                                self.width, self.depth, self.mass, AIR_RHO, 
                                                self.E, self.Ib, self.corners_local)
        dv = -GRAVITY + f_net / self.mass
        dw = self.I_inv @ tau_net
        return [v, dv, wx, wy, wz, dw[0], dw[1], dw[2]]

    def run(self):
        p = np.arctan2(CORNER_RELATIVE_Z[3] - CORNER_RELATIVE_Z[0], self.depth)
        r = np.arctan2(CORNER_RELATIVE_Z[1] - CORNER_RELATIVE_Z[0], self.width)
        sol = solve_ivp(self.ode, (0, T_END), [INITIAL_HEIGHT, 0, r, p, 0, 0, 0, 0], 
                        method='RK45', max_step=MAX_STEP)
        res = BoxDropResult(sol.t, sol.y, self)
        res.process()
        return res

if __name__ == "__main__":
    sim = BoxDropSimulator()
    result = sim.run()
    result.plot_physics()
    result.animate()