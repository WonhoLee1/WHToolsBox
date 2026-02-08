import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation as R
from numba import njit
import matplotlib.font_manager as fm

# ==========================================
# 0. 물리 및 기하학 상수
# ==========================================
BOX_WIDTH, BOX_DEPTH, BOX_HEIGHT = 2.0, 1.6, 0.2
BOX_MASS = 30.0
BOX_E = 1e7          # 영률 (Pa)
BOX_NU = 0.3         # 포아송비
AIR_RHO = 1.225
AIR_MU = 1.81e-5
CD_DRAG = 1.1
H_SQ_LIMIT = 0.05
GROUND_K = 2e7
NX, NY, NZ = 5, 5, 3

# 폰트 설정 (D2Coding 우선, 없을 경우 기본 폰트)
try:
    FONT_PROP = fm.FontProperties(family='D2Coding', size=9)
except:
    FONT_PROP = fm.FontProperties(size=9)
    
# ==========================================
# 1. 격자 및 인덱스 정의
# ==========================================
def get_mesh_info():
    x = np.linspace(-BOX_WIDTH/2, BOX_WIDTH/2, NX+1)
    y = np.linspace(-BOX_DEPTH/2, BOX_DEPTH/2, NY+1)
    z = np.linspace(-BOX_HEIGHT/2, BOX_HEIGHT/2, NZ+1)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    nodes = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    idx_map = np.arange(len(nodes)).reshape((NX+1, NY+1, NZ+1))
    
    # 윗면(Top) 및 아랫면(Bottom) 꼭짓점 4개씩 추출
    top_corners = [idx_map[0,0,-1], idx_map[-1,0,-1], idx_map[-1,-1,-1], idx_map[0,-1,-1]]
    bot_corners = [idx_map[0,0,0], idx_map[-1,0,0], idx_map[-1,-1,0], idx_map[0,-1,0]]
    bottom_surface_nodes = idx_map[:, :, 0].flatten()
    
    return nodes, top_corners, bot_corners, bottom_surface_nodes, idx_map

NODES_L, TOP_C, BOT_C, BOT_SURF, IDX_MAP = get_mesh_info()

# ==========================================
# 2. 물리 계산 코어 (Numba)
# ==========================================
@njit
def compute_all_forces(h, v, rot, omegas, nodes_l, mass):
    num_nodes = nodes_l.shape[0]
    total_fz = 0.0
    total_tau = np.zeros(3)
    area_per_node = (BOX_WIDTH * BOX_DEPTH) / ((NX+1)*(NY+1))
    
    # 결과 저장을 위한 배열
    node_fz_contact = np.zeros(num_nodes)
    node_fz_squeeze = np.zeros(num_nodes)
    
    # 전체 공기 저항 (Drag)
    f_drag_total = -0.5 * AIR_RHO * CD_DRAG * (BOX_WIDTH * BOX_DEPTH) * v * abs(v)
    
    for i in range(num_nodes):
        r_w = rot @ nodes_l[i]
        pz = h + r_w[2]
        vz = v + (omegas[0]*r_w[1] - omegas[1]*r_w[0])
        
        # 1. 지면 충격력
        if pz < 0:
            pen = abs(pz)
            kn = GROUND_K / num_nodes
            cn = 2.0 * np.sqrt(kn * (mass/num_nodes))
            node_fz_contact[i] = kn * (pen**1.5) - cn * vz
        
        # 2. 스퀴즈 필름 (아랫면 근처)
        if pz > 0 and pz < H_SQ_LIMIT:
            h_eff = max(pz, 0.0008)
            node_fz_squeeze[i] = -(1.5 * AIR_MU * (area_per_node**2) * vz) / (np.pi * (h_eff**3))
        
        fz_sum = max(0.0, node_fz_contact[i]) + node_fz_squeeze[i]
        total_fz += fz_sum
        total_tau[0] += r_w[1] * fz_sum
        total_tau[1] -= r_w[0] * fz_sum
        
    return total_fz + f_drag_total, total_tau, node_fz_contact, node_fz_squeeze, f_drag_total

# ==========================================
# 3. 데이터 로깅 및 시뮬레이션
# ==========================================
class FullAnalysisSimulator:
    def __init__(self):
        self.I = np.diag([(1/12)*BOX_MASS*(BOX_DEPTH**2+BOX_HEIGHT**2), (1/12)*BOX_MASS*(BOX_WIDTH**2+BOX_HEIGHT**2), (1/12)*BOX_MASS*(BOX_WIDTH**2+BOX_DEPTH**2)])
        self.I_inv = np.linalg.inv(self.I)

    def run(self):
        y0 = [0.6, 0.0, 0.05, 0.03, 0.0, 0.0, 0.0, 0.0]
        sol = solve_ivp(self.ode, (0, 1.2), y0, method='Radau', max_step=0.001)
        return self.post_process(sol)

    def ode(self, t, y):
        h, v, phi, theta, psi, wx, wy, wz = y
        rot = R.from_euler('xyz', [phi, theta, psi]).as_matrix()
        fz, tau, _, _, _ = compute_all_forces(h, v, rot, np.array([wx, wy, wz]), NODES_L, BOX_MASS)
        return [v, -9.81 + fz/BOX_MASS, wx, wy, wz, (self.I_inv @ tau)[0], (self.I_inv @ tau)[1], (self.I_inv @ tau)[2]]

    def post_process(self, sol):
        # 모든 요청 데이터를 저장할 딕셔너리
        data = {'t': sol.t, 'h_com': sol.y[0], 'v_com': sol.y[1]}
        num_steps = len(sol.t)
        
        # 에너지 및 저항력 배열 초기화
        data['KE'], data['PE_strain'] = np.zeros(num_steps), np.zeros(num_steps)
        data['f_drag'], data['f_squeeze_total'], data['f_contact_total'] = np.zeros(num_steps), np.zeros(num_steps), np.zeros(num_steps)
        
        # 스트레스/변형률 (아랫면 평균)
        data['avg_stress'], data['avg_strain'] = np.zeros(num_steps), np.zeros(num_steps)

        for i in range(num_steps):
            y = sol.y[:, i]
            rot = R.from_euler('xyz', y[2:5]).as_matrix()
            fz, tau, f_con_n, f_sq_n, f_drag = compute_all_forces(y[0], y[1], rot, y[5:8], NODES_L, BOX_MASS)
            
            # 1~6번 항목: 힘 데이터
            data['f_drag'][i] = f_drag
            data['f_squeeze_total'][i] = np.sum(f_sq_n)
            data['f_contact_total'][i] = np.sum(f_con_n)
            
            # 7번 항목: 등가 응력/변형률 근사 (지면 반력 기반)
            avg_p = np.sum(f_con_n) / (BOX_WIDTH * BOX_DEPTH)
            data['avg_strain'][i] = avg_p / BOX_E
            data['avg_stress'][i] = avg_p  # 수직 응력 지배적 가정
            
            # 8번 항목: 에너지
            data['KE'][i] = 0.5 * BOX_MASS * y[1]**2 + 0.5 * np.dot(y[5:8], self.I @ y[5:8])
            # 변형 에너지는 지면 반력에 의한 가상 일로 근사
            data['PE_strain'][i] = 0.5 * np.sum(f_con_n * np.abs(np.minimum(0, y[0]))) 

        return data

# ==========================================
# 4. 결과 출력 및 그래프 (종합 리포트)
# ==========================================
def plot_full_report(data):
    t = data['t']
    
    # 윈도우 정렬 함수
    def apply_cascade(idx):
        mgr = plt.get_current_fig_manager()
        try: 
            # TkAgg backend assumed or similar
            mgr.window.wm_geometry(f"+{50+idx*30}+{50+idx*30}")
        except: 
            pass

    # [그래프 1] 충격력 및 저항력
    plt.figure("Forces Analysis", figsize=(8, 6))
    plt.plot(t, data['f_contact_total'], label='Total Contact Force (Floor)')
    plt.plot(t, data['f_squeeze_total'], label='Squeeze Film Force')
    plt.plot(t, data['f_drag'], label='Aero Drag')
    plt.title("Forces over Time", fontproperties=FONT_PROP)
    plt.legend(prop=FONT_PROP)
    plt.grid(True)
    apply_cascade(0)

    # [그래프 2] 에너지 변화
    plt.figure("Energy Analysis", figsize=(8, 6))
    plt.plot(t, data['KE'], label='Kinetic Energy')
    plt.plot(t, data['PE_strain'], label='Strain Energy (Approx)')
    plt.title("Energy Balance", fontproperties=FONT_PROP)
    plt.legend(prop=FONT_PROP)
    plt.grid(True)
    apply_cascade(1)

    # [그래프 3] 아랫면 등가 응력 및 변형률
    fig, ax3 = plt.subplots(figsize=(8, 6))
    fig.canvas.manager.set_window_title("Stress & Strain")
    ax3.plot(t, data['avg_stress'], 'r-', label='Avg. Stress (Pa)')
    ax3.set_ylabel("Stress (Pa)")
    ax3_2 = ax3.twinx()
    ax3_2.plot(t, data['avg_strain'], 'b--', label='Avg. Strain')
    ax3_2.set_ylabel("Strain")
    ax3.set_title("Bottom Surface Stress & Strain", fontproperties=FONT_PROP)
    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_2.get_legend_handles_labels()
    ax3.legend(lines + lines2, labels + labels2, loc='upper left', prop=FONT_PROP)
    ax3.grid(True)
    apply_cascade(2)

    # [그래프 4] CoM 높이 및 속도
    plt.figure("CoM Motion", figsize=(8, 6))
    plt.plot(t, data['h_com'], label='CoM Height')
    plt.plot(t, data['v_com'], label='CoM Velocity')
    plt.title("CoM Motion", fontproperties=FONT_PROP)
    plt.legend(prop=FONT_PROP)
    plt.grid(True)
    apply_cascade(3)

    plt.show()

if __name__ == "__main__":
    analyzer = FullAnalysisSimulator()
    print("Running Simulation (Radau)...")
    results = analyzer.run()
    print("Simulation Complete. Plotting Results...")
    plot_full_report(results)