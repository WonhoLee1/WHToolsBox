import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from numba import njit
import matplotlib.font_manager as fm
import time

# ==============================================================================
# [Section 1] 시스템 제어 상수 (Control Constants)
# ==============================================================================
# 해석 모드 설정: 'EXPLICIT' 또는 'IMPLICIT'
INTEGRATION_MODE = 'EXPLICIT' 
TOTAL_SIMULATION_TIME = 1.0
TIME_STEP_SIZE = 5.0e-6             # Explicit 해석의 안정성을 위한 미세 증분
LOG_DISPLAY_INTERVAL = 0.01         # 진행 상황 표시 간격 (s)
DATA_SAMPLING_INTERVAL = 0.001      # 데이터 기록 간격 (s)

# 박스 기하 구조 및 물성
BOX_WIDTH = 2.0
BOX_DEPTH = 1.6
BOX_HEIGHT = 0.2
INITIAL_COM_HEIGHT = 0.85           # 낙하 효과 관찰을 위해 0.85m로 설정
BOX_MASS = 30.0
YOUNGS_MODULUS = 2.0e7              # Pa
POISSON_RATIO = 0.3

# FEM 격자 설정
ELEMENTS_COUNT_X = 8
ELEMENTS_COUNT_Y = 8
ELEMENTS_COUNT_Z = 4

# 환경 물리 상수
GRAVITY_ACCEL = 9.80665
AIR_DENSITY = 1.225
DRAG_COEFFICIENT = 1.1
GROUND_STIFFNESS = 1.0e9
GROUND_DAMPING_RATIO = 0.5

# 폰트 설정 (D2Coding 우선, 없을 경우 기본 폰트)
try:
    FONT_PROP = fm.FontProperties(family='D2Coding', size=9)
except:
    FONT_PROP = fm.FontProperties(size=9)

# ==============================================================================
# [Section 2] FEM 구조 및 초기 자세 계산 (Initialization)
# ==============================================================================
class FEMMeshStructure:
    def __init__(self):
        # 노드 생성
        x = np.linspace(-BOX_WIDTH/2, BOX_WIDTH/2, ELEMENTS_COUNT_X + 1)
        y = np.linspace(-BOX_DEPTH/2, BOX_DEPTH/2, ELEMENTS_COUNT_Y + 1)
        z = np.linspace(-BOX_HEIGHT/2, BOX_HEIGHT/2, ELEMENTS_COUNT_Z + 1)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        self.local_nodes = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        self.node_map = np.arange(len(self.local_nodes)).reshape((ELEMENTS_COUNT_X+1, ELEMENTS_COUNT_Y+1, ELEMENTS_COUNT_Z+1))
        
        # 요소 및 적분점(Gauss Point = Centroid) 정의
        self.elements = []
        self.element_centroids = []
        for i in range(ELEMENTS_COUNT_X):
            for j in range(ELEMENTS_COUNT_Y):
                for k in range(ELEMENTS_COUNT_Z):
                    nodes = [
                        self.node_map[i,j,k], self.node_map[i+1,j,k], self.node_map[i+1,j+1,k], self.node_map[i,j+1,k],
                        self.node_map[i,j,k+1], self.node_map[i+1,j,k+1], self.node_map[i+1,j+1,k+1], self.node_map[i,j+1,k+1]
                    ]
                    self.elements.append(nodes)
                    self.element_centroids.append(np.mean(self.local_nodes[nodes], axis=0))
        
        self.elements = np.array(self.elements)
        self.element_centroids = np.array(self.element_centroids)
        
        # 바닥면 노드 및 요소 필터링 (스퀴즈 필름 및 접촉용)
        self.bottom_node_indices = self.node_map[:, :, 0].flatten()
        self.bottom_element_indices = np.arange(ELEMENTS_COUNT_X * ELEMENTS_COUNT_Y) # 하단 1레이어
        
        # 꼭짓점 인덱스 (T1~T4, B1~B4)
        self.top_corners = [self.node_map[0,0,-1], self.node_map[-1,0,-1], self.node_map[-1,-1,-1], self.node_map[0,-1,-1]]
        self.bot_corners = [self.node_map[0,0,0], self.node_map[-1,0,0], self.node_map[-1,-1,0], self.node_map[0,-1,0]]

# ==============================================================================
# [Section 3] Numba 가속 물리 연산 엔진 (Numba Kernels)
# ==============================================================================
@njit
def compute_physics_kernel(h, v, rot, omega, nodes, elements, bot_nodes):
    """지면 반력, 유체력, 응력 계산을 통합 수행"""
    num_nodes = nodes.shape[0]
    fz_ground = 0.0
    torque_ground = np.zeros(3)
    node_forces = np.zeros(num_nodes)
    
    # 1. 지면 충격력 (Penalty Method)
    for i in range(num_nodes):
        rw = rot @ nodes[i]
        zn = h + rw[2]
        vn = v + (omega[0]*rw[1] - omega[1]*rw[0])
        if zn < 0:
            kn = GROUND_STIFFNESS / num_nodes
            cn = GROUND_DAMPING_RATIO * 2.0 * np.sqrt(kn * (BOX_MASS/num_nodes))
            fn = max(0.0, kn * (abs(zn)**1.2) - cn * vn)
            node_forces[i] = fn
            fz_ground += fn
            torque_ground[0] += rw[1] * fn
            torque_ground[1] -= rw[0] * fn

    # 2. 유체 역학 (전체 면적 기준 적분)
    area_total = BOX_WIDTH * BOX_DEPTH
    h_gap = max(h - BOX_HEIGHT/2, 0.0005)
    f_drag = -0.5 * AIR_DENSITY * DRAG_COEFFICIENT * area_total * v * abs(v)
    f_squeeze = 0.0
    if v < 0:
        f_squeeze = 0.5 * AIR_DENSITY * (abs(v) * 3.2 / (7.2 * h_gap))**2 * area_total
    
    p_avg = f_squeeze / area_total if h_gap < 0.05 else 0.0
    v_air = (abs(v) * (BOX_WIDTH/2)) / h_gap if v < 0 else 0.0

    return fz_ground, torque_ground, node_forces, f_drag, f_squeeze, p_avg, v_air

# ==============================================================================
# [Section 4] 시뮬레이션 메인 클래스 (Simulator)
# ==============================================================================
class BoxFEMDynamicsEngine:
    def __init__(self, mesh):
        self.mesh = mesh
        self.I = np.diag([(1/12)*BOX_MASS*(BOX_DEPTH**2+BOX_HEIGHT**2), 
                          (1/12)*BOX_MASS*(BOX_WIDTH**2+BOX_HEIGHT**2), 
                          (1/12)*BOX_MASS*(BOX_WIDTH**2+BOX_DEPTH**2)])
        self.I_inv = np.linalg.inv(self.I)
        self.history = {k: [] for k in ['t','h','v','top_z','top_v','bot_z','bot_v',
                                       'f_imp','f_com','f_drag','f_sq','p_avg','v_air',
                                       'stress_e','strain_e','ke','se','de','state']}
        self.max_strain_map = np.zeros(len(mesh.elements))

    def run(self):
        # 초기 상태: [z, vz, roll, pitch, yaw, wx, wy, wz]
        # 비대칭 낙하를 위해 초기 각도 부여 (Roll 5도, Pitch 10도)
        s = np.array([INITIAL_COM_HEIGHT, 0.0, np.radians(5), np.radians(10), 0.0, 0.0, 0.0, 0.0])
        t_cur, last_log = 0.0, -LOG_DISPLAY_INTERVAL
        
        while t_cur <= TOTAL_SIMULATION_TIME:
            if t_cur >= last_log + LOG_DISPLAY_INTERVAL:
                print(f"[{INTEGRATION_MODE}] Time: {t_cur:.3f}s / {TOTAL_SIMULATION_TIME}s")
                last_log = t_cur
            
            if int(t_cur/TIME_STEP_SIZE) % int(DATA_SAMPLING_INTERVAL/TIME_STEP_SIZE) == 0:
                self._record_step(t_cur, s)

            # Solver Update
            rot = R.from_euler('xyz', s[2:5]).as_matrix()
            fz_g, tq_g, fn, fd, fsq, p, va = compute_physics_kernel(s[0], s[1], rot, s[5:], self.mesh.local_nodes, self.mesh.elements, self.mesh.bottom_node_indices)
            
            # Acceleration
            dv = -GRAVITY_ACCEL + (fz_g + fd + fsq)/BOX_MASS
            dw = self.I_inv @ tq_g
            
            # Step
            s[0] += s[1] * TIME_STEP_SIZE
            s[1] += dv * TIME_STEP_SIZE
            s[2:5] += s[5:8] * TIME_STEP_SIZE
            s[5:8] += dw * TIME_STEP_SIZE
            t_cur += TIME_STEP_SIZE
            
        return self.history

    def _record_step(self, t, s):
        rot = R.from_euler('xyz', s[2:5]).as_matrix()
        fz_g, tq_g, fn, fd, fsq, p, va = compute_physics_kernel(s[0], s[1], rot, s[5:], self.mesh.local_nodes, self.mesh.elements, self.mesh.bottom_node_indices)
        
        # 요소별 응력/변형률
        area_elem = (BOX_WIDTH * BOX_DEPTH) / (ELEMENTS_COUNT_X * ELEMENTS_COUNT_Y)
        stresses = []
        for i in self.mesh.bottom_element_indices:
            f_elem = np.sum(fn[self.mesh.elements[i]])
            sigma = f_elem / area_elem
            stresses.append(sigma)
            # 애니메이션용 Max 변형률 기록
            eps = sigma / YOUNGS_MODULUS
            if eps > self.max_strain_map[i]: self.max_strain_map[i] = eps
            
        # 에너지 분석
        ke = 0.5 * BOX_MASS * s[1]**2 + 0.5 * np.dot(s[5:8], self.I @ s[5:8])
        vol_elem = (BOX_WIDTH*BOX_DEPTH*BOX_HEIGHT) / (len(self.mesh.elements))
        se = np.sum(0.5 * (np.array(stresses)**2) / YOUNGS_MODULUS) * vol_elem * ELEMENTS_COUNT_Z
        
        # 기록
        self.history['t'].append(t); self.history['h'].append(s[0]); self.history['v'].append(s[1])
        self.history['f_imp'].append(np.sum(fn)); self.history['f_com'].append(fz_g + fd + fsq)
        self.history['f_drag'].append(fd); self.history['f_sq'].append(fsq)
        self.history['p_avg'].append(p); self.history['v_air'].append(va)
        self.history['stress_e'].append(stresses); self.history['strain_e'].append(np.array(stresses)/YOUNGS_MODULUS)
        self.history['ke'].append(ke); self.history['se'].append(se); self.history['de'].append(abs(fd*s[1]*DATA_SAMPLING_INTERVAL))
        
        # 꼭짓점 궤적
        tz, tv, bz, bv = [], [], [], []
        for idx in self.mesh.top_corners:
            rw = rot @ self.mesh.local_nodes[idx]
            tz.append(s[0]+rw[2]); tv.append(s[1]+(s[5]*rw[1]-s[6]*rw[0]))
        for idx in self.mesh.bot_corners:
            rw = rot @ self.mesh.local_nodes[idx]
            bz.append(s[0]+rw[2]); bv.append(s[1]+(s[5]*rw[1]-s[6]*rw[0]))
        self.history['top_z'].append(tz); self.history['top_v'].append(tv)
        self.history['bot_z'].append(bz); self.history['bot_v'].append(bv)
        self.history['state'].append(s.copy())

# ==============================================================================
# [Section 5] Cascade 시각화 및 결과 분석 (Visualization)
# ==============================================================================
def visualize_cascade(data, mesh_obj):
    t = np.array(data['t'])
    win_offset = 30
    
    def apply_cascade(idx):
        mgr = plt.get_current_fig_manager()
        try: mgr.window.wm_geometry(f"+{50+idx*win_offset}+{50+idx*win_offset}")
        except: pass

    # 1. Top Corners
    plt.figure("1. Top Surface Dynamics", figsize=(10, 5))
    plt.subplot(1,2,1); plt.plot(t, data['top_z']); plt.title("Top Z-Pos", fontproperties=FONT_PROP); plt.grid(True)
    plt.subplot(1,2,2); plt.plot(t, data['top_v']); plt.title("Top Z-Vel", fontproperties=FONT_PROP); plt.grid(True)
    plt.tight_layout(); apply_cascade(0)

    # 2. Bottom Corners
    plt.figure("2. Bottom Surface Dynamics", figsize=(10, 5))
    plt.subplot(1,2,1); plt.plot(t, data['bot_z']); plt.title("Bottom Z-Pos", fontproperties=FONT_PROP); plt.grid(True)
    plt.subplot(1,2,2); plt.plot(t, data['bot_v']); plt.title("Bottom Z-Vel", fontproperties=FONT_PROP); plt.grid(True)
    plt.tight_layout(); apply_cascade(1)

    # 3 & 4. Forces
    plt.figure("3. Ground Impact Force"); plt.plot(t, data['f_imp'], 'r'); plt.title("Total Ground Force (N)", fontproperties=FONT_PROP); plt.grid(True); apply_cascade(2)
    plt.figure("4. Net CoM Force"); plt.plot(t, data['f_com'], 'b'); plt.title("Net Force at CoM (N)", fontproperties=FONT_PROP); plt.grid(True); apply_cascade(3)

    # 5. Fluid (Analysis)
    
    plt.figure("5. Fluid Analysis (Summed)"); plt.plot(t, data['f_drag'], label='Drag'); plt.plot(t, data['f_sq'], label='Squeeze Sum'); 
    plt.legend(prop=FONT_PROP); plt.grid(True); plt.title("Global Fluid Forces", fontproperties=FONT_PROP); apply_cascade(4)

    # 6. Element Fluid
    plt.figure("6. Element Fluid Forces"); plt.plot(t, np.array(data['f_sq'])/64, alpha=0.5); plt.title("Squeeze Share per Element", fontproperties=FONT_PROP); plt.grid(True); apply_cascade(5)

    # 7. Stress & Strain
    
    plt.figure("7. Element Stress & Strain", figsize=(10, 6))
    plt.subplot(2,1,1); plt.plot(t, data['stress_e']); plt.title("Equivalent Stress (Pa)", fontproperties=FONT_PROP); plt.grid(True)
    plt.subplot(2,1,2); plt.plot(t, data['strain_e']); plt.title("Equivalent Strain", fontproperties=FONT_PROP); plt.grid(True)
    plt.tight_layout(); apply_cascade(6)

    # 8. Energy Transformation
    
    plt.figure("8. Energy Balance Analysis")
    plt.plot(t, data['ke'], label='Kinetic'); plt.plot(t, data['se'], label='Strain'); plt.plot(t, np.cumsum(data['de']), 'k--', label='Air Loss')
    plt.legend(prop=FONT_PROP); plt.grid(True); plt.title("System Energy Conversion", fontproperties=FONT_PROP); apply_cascade(7)

    # 3D Animation
    fig3d = plt.figure("3D FEM Motion", figsize=(8, 8))
    ax3d = fig3d.add_subplot(111, projection='3d')
    states = np.array(data['state'])
    def update_3d(i):
        ax3d.cla()
        s = states[i]
        rot = R.from_euler('xyz', s[2:5]).as_matrix()
        pos = (rot @ mesh_obj.local_nodes.T).T + [0, 0, s[0]]
        # 바닥면 변형률 컬러 맵핑 시뮬레이션
        ax3d.scatter(pos[:,0], pos[:,1], pos[:,2], c=pos[:,2], cmap='viridis', s=2)
        ax3d.set_zlim(-0.1, 1.0); ax3d.set_title(f"Time: {t[i]:.3f}s")
    
    ani = FuncAnimation(fig3d, update_3d, frames=range(0, len(t), 10), interval=30)
    plt.show()

# ==============================================================================
# [Section 6] 실행
# ==============================================================================
if __name__ == "__main__":
    mesh = FEMMeshStructure()
    engine = BoxFEMDynamicsEngine(mesh)
    results = engine.run()
    visualize_cascade(results, mesh)

    