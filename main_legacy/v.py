import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation as R

# ==========================================
# 1. BoxDropResult: 데이터 저장 및 시각화 클래스
# ==========================================
class BoxDropResult:
    def __init__(self, t, states, params):
        self.t = t
        self.states = states
        self.params = params
        self.com = {}
        self.corners = {i: {} for i in range(4)}
        self.aero_data = {'drag': [], 'squeeze': []}

    def process(self, dynamics_func):
        n = len(self.t)
        self.com['h'] = self.states[0]
        self.com['v'] = self.states[1]
        self.com['accel'] = np.zeros(n)
        
        for j in range(4):
            for m in ['h', 'v', 'a', 'f']:
                self.corners[j][m] = np.zeros(n)

        for i in range(n):
            y = self.states[:, i]
            h, v, angles, omegas = y[0], y[1], y[2:5], y[5:8]
            
            f_aero_info, tau, f_total_contact, f_corner_list = dynamics_func(h, v, angles, omegas, detailed=True)
            
            self.aero_data['drag'].append(f_aero_info['drag'])
            self.aero_data['squeeze'].append(f_aero_info['squeeze'])
            
            a_com_z = -9.81 + (f_aero_info['total'] + f_total_contact) / self.params['mass']
            self.com['accel'][i] = a_com_z
            
            alpha = np.linalg.inv(self.params['I_matrix']) @ tau
            rot = R.from_euler('xyz', angles).as_matrix()

            for j in range(4):
                r_w = rot @ self.params['corners_local'][j]
                self.corners[j]['h'][i] = h + r_w[2]
                self.corners[j]['v'][i] = v + np.cross(omegas, r_w)[2]
                a_corner_z = a_com_z + np.cross(alpha, r_w)[2] + np.cross(omegas, np.cross(omegas, r_w))[2]
                self.corners[j]['a'][i] = a_corner_z
                self.corners[j]['f'][i] = f_corner_list[j]

    def plot_2d_physics(self):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
        colors = ['red', 'green', 'blue', 'magenta']
        metrics = [('h', 'Height (m)'), ('v', 'Velocity (m/s)'), 
                   ('a', 'Acceleration (m/s^2)'), ('f', 'Impact Force (N)')]
        
        for idx, (key, ylabel) in enumerate(metrics):
            ax = axes.flatten()[idx]
            if key in self.com:
                ax.plot(self.t, self.com[key], 'k--', label='CoM', alpha=0.6)
            for j in range(4):
                ax.plot(self.t, self.corners[j][key], color=colors[j], label=f'Corner {j+1}')
            ax.set_ylabel(ylabel); ax.grid(True); ax.legend(fontsize='small', ncol=2)
        plt.tight_layout(); plt.show()

    def plot_aero_analysis(self):
        plt.figure(figsize=(10, 4))
        plt.plot(self.t, self.aero_data['drag'], 'b-', label='Drag Force')
        plt.plot(self.t, self.aero_data['squeeze'], 'r-', label='Squeeze Film Force')
        plt.title("Aerodynamic Component Analysis (Z-axis)"); plt.ylabel("Force (N)")
        plt.grid(True); plt.legend(); plt.show()

    def plot_3d_animation(self):
        """개선된 3D 박스 모션 애니메이션"""
        fig = plt.figure(figsize=(10, 10)) # 정사각형 캔버스 권장
        ax = fig.add_subplot(111, projection='3d')
        
        # 시각화용 박스 꼭짓점 (8개) - 파라미터 기반 동적 생성
        W, D = self.params['width'], self.params['depth']
        H = 0.1 # 시각화용 박스 두께
        local_8 = np.array([
            [-W/2, -D/2, 0], [W/2, -D/2, 0], [W/2, D/2, 0], [-W/2, D/2, 0], # 밑면
            [-W/2, -D/2, H], [W/2, -D/2, H], [W/2, D/2, H], [-W/2, D/2, H]  # 윗면
        ])

        # 축 범위 설정
        xlim = [-1.5, 1.5]
        ylim = [-1.5, 1.5]
        zlim = [-0.1, 0.6]

        def update(frame):
            ax.cla()
            # 지면 표시
            gx, gy = np.meshgrid([-2, 2], [-2, 2])
            ax.plot_surface(gx, gy, np.zeros_like(gx), alpha=0.1, color='gray')
            
            # 현재 프레임 상태
            y = self.states[:, frame]
            h, angles = y[0], y[2:5]
            rot = R.from_euler('xyz', angles).as_matrix()
            
            # 로컬 좌표 회전 및 이동 적용
            w_8 = (rot @ local_8.T).T + [0, 0, h]
            
            # 박스 와이어프레임 그리기
            edges = [[0,1,2,3,0], [4,5,6,7,4], [0,4], [1,5], [2,6], [3,7]]
            for e in edges:
                ax.plot(w_8[e, 0], w_8[e, 1], w_8[e, 2], 'b-', linewidth=1.5)
            
            # [중요] 축 비율 고정 및 범위 설정
            ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
            ax.set_box_aspect((np.ptp(xlim), np.ptp(ylim), np.ptp(zlim)))
            
            # 시각적 보조를 위한 시점 고정 (선택 사항)
            ax.view_init(elev=20, azim=45) 
            ax.set_title(f"Time: {self.t[frame]:.3f}s / Roll: {np.degrees(angles[0]):.1f} deg")

        ani = FuncAnimation(fig, update, frames=range(0, len(self.t), 5), interval=50)
        plt.show()

# ==========================================
# 2. BoxDropSimulator: 물리 엔진 클래스
# ==========================================
class BoxDropSimulator:
    def __init__(self, E_pa=100e6):
        self.width, self.depth, self.mass = 2.0, 1.6, 30.0
        self.rho = 1.225
        # 영률 기반 강성 및 감쇠
        self.k_ground = (E_pa * (self.width * self.depth / 4)) / 0.1 
        self.c_ground = 2 * np.sqrt(self.k_ground * (self.mass/4)) * 0.3
        
        self.params = {
            'width': self.width, 'depth': self.depth, 'mass': self.mass,
            'I_matrix': np.diag([(1/12)*30*1.6**2, (1/12)*30*2.0**2, (1/12)*30*(2.0**2+1.6**2)]),
            'corners_local': np.array([[-self.width/2, -self.depth/2, 0], [self.width/2, -self.depth/2, 0], 
                                      [self.width/2, self.depth/2, 0], [-self.width/2, self.depth/2, 0]])
        }

    def get_dynamics(self, h, v, angles, omegas, detailed=False):
        rot = R.from_euler('xyz', angles).as_matrix()
        
        # 1. 공기역학 격자 적분
        NX, NY = 10, 10
        x_range = np.linspace(-self.width/2, self.width/2, NX)
        y_range = np.linspace(-self.depth/2, self.depth/2, NY)
        X, Y = np.meshgrid(x_range, y_range)
        da = (self.width * self.depth) / (NX * NY)
        r_w = (rot @ np.stack([X.flatten(), Y.flatten(), np.zeros(NX*NY)], axis=1).T).T
        h_g = np.maximum(h + r_w[:, 2], 0.0005)
        v_g = v + np.cross(omegas, r_w)[:, 2]
        
        # Squeeze & Drag
        f_squeeze_grid = np.zeros_like(v_g)
        comp = v_g < 0
        if np.any(comp):
            v_esc = np.abs(v_g[comp]) * (3.2 / (7.2 * h_g[comp]))
            f_squeeze_grid[comp] = 0.5 * self.rho * v_esc**2 * da
        f_drag_grid = -0.5 * self.rho * 1.1 * da * v_g * np.abs(v_g)
        
        f_aero_total = np.sum(f_squeeze_grid + f_drag_grid)
        tau_aero = np.sum(np.cross(r_w, np.stack([np.zeros(NX*NY), np.zeros(NX*NY), 
                                                 f_squeeze_grid+f_drag_grid], axis=1)), axis=0)

        # 2. 지면 반력
        f_contact_z, f_corner_list, tau_contact = 0, [], np.zeros(3)
        c_w = (rot @ self.params['corners_local'].T).T
        for cw in c_w:
            hc, vc = h + cw[2], v + np.cross(omegas, cw)[2]
            if hc < 0:
                fz = max(0, -self.k_ground * hc - self.c_ground * vc)
                f_contact_z += fz
                f_corner_list.append(fz)
                tau_contact += np.cross(cw, [0, 0, fz])
            else:
                f_corner_list.append(0)
        
        if detailed:
            aero_info = {'total': f_aero_total, 'drag': np.sum(f_drag_grid), 'squeeze': np.sum(f_squeeze_grid)}
            return aero_info, tau_aero + tau_contact, f_contact_z, f_corner_list
        return f_aero_total, tau_aero + tau_contact, f_contact_z

    def ode(self, t, y):
        h, v, phi, theta, psi, wx, wy, wz = y
        fa, tau, fc = self.get_dynamics(h, v, [phi, theta, psi], [wx, wy, wz])
        dvdt = -9.81 + (fa + fc) / self.mass
        dwdt = np.linalg.inv(self.params['I_matrix']) @ tau
        return [v, dvdt, wx, wy, wz, dwdt[0], dwdt[1], dwdt[2]]

    def run(self, h0, angles0, t_end=0.6):
        y0 = [h0, 0, *angles0, 0, 0, 0]
        sol = solve_ivp(self.ode, (0, t_end), y0, method='RK45', max_step=0.0005)
        res = BoxDropResult(sol.t, sol.y, self.params)
        res.process(self.get_dynamics)
        return res

# ==========================================
# 3. 실행
# ==========================================
# 영률 100MPa 적용 시뮬레이터
sim = BoxDropSimulator(E_pa=100e6)
# 초기 높이 0.3m, Roll 5도, Pitch 3도
result = sim.run(h0=0.3, angles0=[np.radians(5), np.radians(3), 0])

# 결과 그래프 및 애니메이션 출력
result.plot_2d_physics()     # 2D 물리 지표
result.plot_aero_analysis()  # 공기역학 분석
result.plot_3d_animation()   # 개선된 3D 애니메이션
