import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation as R

class BoxDropResult:
    """
    시뮬레이션 결과를 저장하고, 후처리하며, 다양한 시각화(2D 그래프, 3D 애니메이션)를
    수행하는 클래스입니다.
    """
    def __init__(self, t, states, params):
        """
        초기화 메서드. 시뮬레이션 결과(시간, 상태, 파라미터)를 저장합니다.
        
        Args:
            t (np.array): 시뮬레이션 시간 배열 (단위: 초).
            states (np.array): 시간에 따른 상태 변수 배열. 
                               각 열은 [h, v, phi, theta, psi, wx, wy, wz]를 나타냅니다.
            params (dict): 시뮬레이션에 사용된 물리적 파라미터 딕셔너리.
        """
        self.t = t  # 시간 배열
        # 상태 변수: [0:z(높이), 1:vz(속도), 2:phi(롤), 3:theta(피치), 4:psi(요), 5:wx, 6:wy, 7:wz(각속도)]
        self.states = states 
        self.params = params
        
        # 상세 물리량 저장을 위한 딕셔너리 초기화
        self.com = {}  # 무게중심(Center of Mass) 데이터
        self.corners = {i: {} for i in range(4)}  # 4개 바닥면 모서리 데이터
        self.aero_data = {}  # 공기역학 데이터 (Squeeze Film, Drag)
        self.contact_data = {} # 접촉력 데이터 (Contact Force)

    def calculate_all_metrics(self, dynamics_func):
        print(f"\nDEBUG: calculate_all_metrics using mass = {self.params['mass']} kg")
        print("\n--- Starting Post-processing: Calculating Detailed Metrics ---")
        n = len(self.t)  # 시간 스텝의 총 개수
        total_time = self.t[-1] if n > 0 else 0
        last_printed_percent = -1

        # 무게중심(CoM) 데이터 배열 초기화
        self.com['h'] = self.states[0]  # 높이 (z)
        self.com['v'] = self.states[1]  # 수직 속도 (vz)
        self.com['a'] = np.zeros(n)     # 수직 가속도 (az)
        self.com['f'] = np.zeros(n)     # 총 지면 반력 (Ground Reaction Force)
        
        # 공기역학 데이터 배열 초기화
        self.aero_data['squeeze'] = np.zeros(n)  # Squeeze Film 효과로 인한 힘
        self.aero_data['drag'] = np.zeros(n)     # 공기 저항으로 인한 힘
        self.contact_data['f_contact'] = np.zeros(n) # 순수 물리적 접촉력
        
        # 모서리 데이터 배열 초기화 (바닥면 4개)
        for j in range(4):
            for m in ['h', 'v', 'a', 'f']:  # 높이, 속도, 가속도, 힘
                self.corners[j][m] = np.zeros(n)

        # 각 시간 스텝(i)에 대해 상세 물리량 계산
        for i in range(n):
            y = self.states[:, i]
            h, v, phi, theta, psi = y[0:5]  # 위치 및 오일러 각
            omegas = y[5:8]                 # 각속도 벡터 [wx, wy, wz]
            
            # 물리 엔진 함수를 호출하여 현재 상태에서의 힘과 토크를 다시 계산
            f_sq, f_dr, tau, f_total_contact = dynamics_func(h, v, [phi, theta, psi], omegas)
            f_aero = f_sq + f_dr  # 총 공기역학적 힘

            # 진행률 및 물리 정보 출력 (1% 단위)
            if total_time > 0:
                current_percent = int((self.t[i] / total_time) * 100)
                if current_percent > last_printed_percent:
                    last_printed_percent = current_percent
                    
                    # --- 출력할 물리 정보 계산 ---
                    # 1. 운동에너지 (KE = Translational + Rotational)
                    ke_trans = 0.5 * self.params['mass'] * v**2
                    ke_rot = 0.5 * np.dot(omegas, self.params['I_matrix'] @ omegas)
                    total_ke = ke_trans + ke_rot
                    
                    # --- 포맷에 맞춰 콘솔에 출력 ---
                    print(
                        f"  [Progress: {current_percent:3d}%] "
                        f"Frame: {i:5d} | "
                        f"Time: {self.t[i]:.4f}s | "
                        f"H: {h:7.4f}m | "
                        f"V: {v:8.5f}m/s | "
                        f"GRF_Net: {f_total_contact + f_sq:8.2f} N (Sq:{f_sq:6.1f}) | "
                        f"KE: {total_ke:8.5f} J"
                    )

            # 1. 무게중심(CoM)의 가속도 및 충격력 계산
            # 뉴턴의 제2법칙: F_net = m * a  => a = F_net / m
            # F_net = -mg + F_aero + F_contact
            a_com_z = -9.81 + (f_aero + f_total_contact) / self.params['mass']
            # 오일러 방정식 (Full term)
            I = self.params['I_matrix']
            gyro_term = np.cross(omegas, I @ omegas)
            alpha = np.linalg.inv(I) @ (tau - gyro_term)  # 각가속도 벡터
            
            self.com['a'][i] = a_com_z
            # 지면 반력(GRF_Net) = 순수 접촉력 + 스퀴즈 필름력
            self.com['f'][i] = f_total_contact + f_sq
            self.contact_data['f_contact'][i] = f_total_contact
            self.aero_data['squeeze'][i] = f_sq
            self.aero_data['drag'][i] = f_dr

            # 2. 각 모서리별 물리량 계산
            # 현재 시간의 회전 행렬 계산
            rot = R.from_euler('xyz', [phi, theta, psi]).as_matrix()
            
            # 바닥면 4개 모서리(인덱스 0~3)에 대해 반복
            for j in range(4): 
                # 모서리의 로컬 좌표 벡터 (CoM 기준)
                r_local = self.params['corners_local'][j]
                # 월드 좌표계에서의 모서리 위치 벡터 (CoM 기준)
                r_w = rot @ r_local
                
                # 모서리의 높이 (h_c): CoM 높이 + 회전에 의한 z-offset
                h_c = h + r_w[2]
                
                # 모서리의 속도 (v_c): v_c = v_com + ω x r
                # 여기서 v_c와 v_com은 z축 성분만 고려
                v_c = v + np.cross(omegas, r_w)[2]
                
                # 모서리의 가속도 (a_c): a_c = a_com + α x r + ω x (ω x r)
                # α x r: 접선 가속도, ω x (ω x r): 구심 가속도
                a_c = a_com_z + np.cross(alpha, r_w)[2] + np.cross(omegas, np.cross(omegas, r_w))[2]
                
                # 모서리에 가해지는 개별 충격력
                # 참고: 이 시뮬레이션에서는 힘을 바닥면 전체에 분포된 노드에서 계산하므로,
                # 개별 모서리의 힘을 직접 계산하지는 않음. 그래프용으로 0으로 설정.
                f_c = 0
                
                self.corners[j]['h'][i] = h_c
                self.corners[j]['v'][i] = v_c
                self.corners[j]['a'][i] = a_c
                self.corners[j]['f'][i] = f_c
        
        print(f"  [Progress: 100%] Calculation finished.")
        print("--- Metrics Calculation Finished ---\n")

    def plot_physics_metrics(self):
        """
        계산된 물리량(변위, 속도, 가속도, 충격력)을 2D 그래프로 시각화합니다.
        CoM과 4개 모서리의 데이터를 함께 보여줍니다.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
        colors = ['red', 'green', 'blue', 'magenta']
        titles = ['Displacement (Height)', 'Velocity', 'Acceleration', 'Total Ground Load (Contact + Squeeze)']
        ylabels = ['Height (m)', 'Velocity (m/s)', 'Accel (m/s^2)', 'Force (N)']
        keys = ['h', 'v', 'a', 'f']

        for idx, ax in enumerate(axes.flatten()):
            key = keys[idx]
            if key == 'f':
                # 충격력 그래프의 경우 성분별로 나누어 3개의 곡선 출력
                ax.plot(self.t, self.com['f'], 'r-', label='Total GRF', lw=1.5)
                ax.plot(self.t, self.contact_data['f_contact'], 'k--', label='Contact (Physical)', alpha=0.6)
                ax.plot(self.t, self.aero_data['squeeze'], 'b:', label='Squeeze (Air Cushion)', alpha=0.8)
            else:
                # 무게중심(CoM) 데이터 플롯 (검은색 점선)
                ax.plot(self.t, self.com[key], 'k--', label='CoM', alpha=0.7)
                # 4개 모서리 데이터 플롯
                for j in range(4):
                    ax.plot(self.t, self.corners[j][key], color=colors[j], label=f'Corner {j+1}')
            
            ax.set_title(titles[idx])
            ax.set_ylabel(ylabels[idx])
            ax.grid(True)
            if idx == 0 or key == 'f': ax.legend(loc='upper right', fontsize='x-small', ncol=1)

        plt.xlabel('Time (s)')
        plt.tight_layout()
        plt.show()

    def plot_3d_motion(self, interval=30, total_frames=100):
        """
        Matplotlib 3D를 사용하여 상자의 낙하 및 충돌 과정을 애니메이션으로 보여줍니다.
        (사용자 요구사항을 반영하여 재작성된 버전)
        
        Args:
            interval (int): 애니메이션 프레임 간의 시간 간격 (밀리초).
        """
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # 상자의 8개 꼭짓점에 대한 로컬 좌표 (무게중심 기준)
        local_corners = self.params['corners_local']
        
        # 와이어프레임을 그리기 위한 모서리 연결 정보
        edges = [
            [0,1], [1,2], [2,3], [3,0], # 바닥면
            [4,5], [5,6], [6,7], [7,4], # 윗면
            [0,4], [1,5], [2,6], [3,7]  # 기둥
        ]
        
        total_time = self.t[-1]
        num_frames = len(self.t)
        # 애니메이션이 너무 느려지지 않도록 약 total_frames 내외로 조절
        frame_step = max(1, num_frames // total_frames)

        # 애니메이션 업데이트 함수
        def update(frame_index):
            ax.cla()
            
            # --- 1. 현재 상태 정보 가져오기 ---
            current_time = self.t[frame_index]
            y = self.states[:, frame_index]
            h, v, phi, theta, psi = y[0:5]
            
            # --- 2. 강체 변환(Rigid Body Transformation) 계산 ---
            # 개념: 최종 월드 좌표 = 무게중심 월드 좌표 + 회전된 로컬 좌표
            
            # 2-1. 무게중심의 월드 좌표 벡터 (이 시뮬레이션에서는 Z축으로만 이동)
            com_world_pos = np.array([0.0, 0.0, h])
            
            # 2-2. 현재 오일러 각도로부터 회전 행렬 생성
            rotation_matrix = R.from_euler('xyz', [phi, theta, psi]).as_matrix()
            
            # 2-3. 로컬 좌표의 모든 꼭짓점을 회전
            # (3x3) 행렬과 (3x8) 행렬의 곱 -> (3x8) 결과 -> 전치 -> (8x3) 배열
            rotated_corners = (rotation_matrix @ local_corners.T).T
            
            # 2-4. 회전된 모든 꼭짓점을 무게중심 위치만큼 평행 이동
            world_corners = rotated_corners + com_world_pos
            
            # --- 3. 와이어프레임 그리기 ---
            for edge in edges:
                # edge = [꼭짓점1_idx, 꼭짓점2_idx]
                # pts는 두 꼭짓점의 [x,y,z] 좌표를 담은 (2, 3) 크기의 배열
                pts = world_corners[edge]
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'b-')
                
            # --- 4. 추가 정보 표시 및 플롯 환경 설정 ---
            
            # 지면(z=0) 표시
            plot_lim = 1.5
            xx, yy = np.meshgrid(np.linspace(-plot_lim, plot_lim, 5), 
                                 np.linspace(-plot_lim, plot_lim, 5))
            ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.2, color='gray')
            
            # 진행률 계산 및 제목 설정
            progress_percent = (current_time / total_time) * 100
            title_info = (f"Frame: {frame_index}/{num_frames} | "
                          f"Time: {current_time:.3f}s / {total_time:.3f}s | "
                          f"Progress: {progress_percent:.0f}%")
            ax.set_title(title_info, fontsize=12)
            
            # 축 레이블 및 범위 설정
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (Height, m)')
            ax.set_xlim(-plot_lim, plot_lim)
            ax.set_ylim(-plot_lim, plot_lim)
            ax.set_zlim(-0.1, max(1.0, self.states[0,0] * 1.2)) # 초기 높이에 맞춰 Z축 범위 조절

            # *** 중요: 축의 비율을 동일하게 설정하여 시각적 왜곡 방지 ***
            ax.set_aspect('equal', adjustable='box')
            
            # 보기 좋은 각도로 시점 고정
            ax.view_init(elev=20., azim=-60)

        # FuncAnimation을 사용하여 애니메이션 생성 및 표시
        ani = FuncAnimation(fig, update, frames=range(0, num_frames, frame_step), interval=interval)
        plt.show()


class BoxDropSimulator:
    """
    상자 낙하 시뮬레이션을 위한 물리 엔진 클래스.
    6자유도(6DOF) 강체 동역학 모델을 기반으로 하며, 공기역학(Squeeze Film, Drag)과
    분포된 노드 접촉 모델(Distributed Node Contact)을 포함합니다.
    """
    def __init__(self, sim_params):
        """
        시뮬레이터 초기화 및 물리적 파라미터 설정.
        
        Args:
            sim_params (dict): 시뮬레이션에 필요한 모든 물리 상수를 담은 딕셔너리.
        """
        self.params = sim_params.copy() # 원본 파라미터가 수정되지 않도록 복사
        print(f"DEBUG: BoxDropSimulator initialized with mass = {self.params['mass']} kg")
        
        # 전달된 파라미터를 기반으로 추가 파라미터 계산
        m, w, d, h = self.params['mass'], self.params['width'], self.params['depth'], self.params['height']
        
        # 직육면체의 관성 모멘트 텐서 (주축 기준이므로 대각 행렬)
        self.params['I_matrix'] = np.diag([
            (1/12)*m*(d**2 + h**2), 
            (1/12)*m*(w**2 + h**2), 
            (1/12)*m*(w**2 + d**2)
        ])
        print(f"DEBUG: Calculated Inertia Tensor (Diagonal): {np.diag(self.params['I_matrix'])}")
        
        # 8개 꼭짓점의 로컬 좌표 (무게중심 기준)
        self.params['corners_local'] = np.array([
            [-w/2, -d/2, -h/2], [ w/2, -d/2, -h/2], 
            [ w/2,  d/2, -h/2], [-w/2,  d/2, -h/2],
            [-w/2, -d/2,  h/2], [ w/2, -d/2,  h/2], 
            [ w/2,  d/2,  h/2], [-w/2,  d/2,  h/2]
        ])
        
        self._setup_contact_nodes()
        self._setup_faces()

    def _setup_contact_nodes(self):
        """
        바닥면의 접촉 및 공기력 계산을 위한 노드(node) 격자를 설정합니다.
        """
        p = self.params
        NX, NY = p['contact_nodes_x'], p['contact_nodes_y']
        x = np.linspace(-p['width']/2, p['width']/2, NX)
        y = np.linspace(-p['depth']/2, p['depth']/2, NY)
        xx, yy = np.meshgrid(x, y)
        
        self.nodes_local = np.column_stack([xx.flatten(), yy.flatten(), np.full(NX*NY, -p['height']/2)])
        
        area_total = p['width'] * p['depth']
        da = area_total / (NX * NY)
        self.params['da'] = da
        
        L_char = p['height'] / 2.0
        k_box = p['E_box'] * da / L_char
        k_gnd = p['E_ground'] * da / L_char
        
        k_eq = (k_box * k_gnd) / (k_box + k_gnd)
        self.params['node_k'] = k_eq
        
        m_node = p['mass'] / (NX * NY)
        self.params['node_c'] = 2.0 * np.sqrt(k_eq * m_node) * p['damping_ratio']

    def _setup_faces(self):
        """
        상자의 6개 면(Face)에 대한 정보를 설정합니다. (공기 저항 계산용)
        """
        p = self.params
        w, d, h = p['width'], p['depth'], p['height']
        
        # 6개 면 정의: (로컬 중심 좌표, 로컬 법선 벡터, 면적, 이름)
        # 법선은 상자 바깥쪽 방향이 + 입니다.
        self.faces = [
            {'center': [0, 0, -h/2], 'normal': [0, 0, -1], 'area': w*d, 'name': 'bottom'},
            {'center': [0, 0,  h/2], 'normal': [0, 0,  1], 'area': w*d, 'name': 'top'},
            {'center': [-w/2, 0, 0], 'normal': [-1, 0, 0], 'area': d*h, 'name': 'left'},
            {'center': [ w/2, 0, 0], 'normal': [ 1, 0, 0], 'area': d*h, 'name': 'right'},
            {'center': [0,  d/2, 0], 'normal': [ 0,  1, 0], 'area': w*h, 'name': 'front'},
            {'center': [0, -d/2, 0], 'normal': [ 0, -1, 0], 'area': w*h, 'name': 'back'},
        ]

    def calculate_initial_state_from_corners(self, h_corners):
        """
        사용자가 입력한 네 바닥 모서리의 초기 높이로부터 무게중심(CoM)의
        높이(h)와 회전 각도(phi, theta)를 역산합니다.
        
        Args:
            h_corners (list or array): 4개 모서리 높이 [m].
                순서: 뒤왼(RL), 뒤오(RR), 앞오(FR), 앞왼(FL).
        
        Returns:
            (float, list): 계산된 CoM 높이, [phi, theta, psi] 각도 리스트.
        """
        h1, h2, h3, h4 = h_corners
        w, d = self.params['width'], self.params['depth']
        
        dz_dx = ((h2 + h3) - (h1 + h4)) / (2 * w)
        theta = np.arctan(dz_dx)
        
        dz_dy = ((h3 + h4) - (h1 + h2)) / (2 * d)
        phi = -np.arctan(dz_dy)
        
        h_base_center = np.mean(h_corners)
        h_com = h_base_center + (self.params['height'] / 2.0) * np.cos(phi) * np.cos(theta)
        
        print(f"--- Initial State Calculated ---")
        print(f"Input Corners: {h_corners}")
        print(f"Calculated Angles -> Roll(phi): {np.degrees(phi):.2f}°, Pitch(theta): {np.degrees(theta):.2f}°")
        print(f"Calculated CoM Height: {h_com:.4f} m")
        
        return h_com, [phi, theta, 0.0]

    def get_dynamics(self, h, v, angles, omegas):
        """
        특정 시점의 상태(위치, 속도, 각도, 각속도)가 주어졌을 때,
        객체에 작용하는 총 힘(Z축)과 총 토크(벡터)를 계산합니다.
        """
        p = self.params
        rot = R.from_euler('xyz', angles).as_matrix()
        
        r_world = (rot @ self.nodes_local.T).T
        
        node_h_world = h + r_world[:, 2]
        node_v_world = v + np.cross(omegas, r_world)[:, 2]
        
        h_eff = np.maximum(node_h_world, p['squeeze_min_h'])

        # --- 1. 스퀴즈 압력 계산 (노드별 개선 모델) ---
        # 1. 각 노드별로 국부적인 스퀴즈 압력을 계산하여 합산합니다.
        # 이는 박스가 기울어져 있을 때 더욱 정확한 힘과 토크를 발생시킵니다.
        
        area_total = p['width'] * p['depth']
        perim = 2 * (p['width'] + p['depth'])
        L_char = np.sqrt(area_total)
        
        # 1-1. 관성 압력 (Inertial): v^2 기반
        v_esc_nodes = np.abs(node_v_world) * (area_total / (perim * h_eff))
        p_sq_in = 0.5 * p['rho'] * v_esc_nodes**2
        
        # 1-2. 점성 압력 (Viscous): Stefan's Law 기반 (v/h^3)
        # 좁은 틈에서 속도에 비례하는 저항력을 생성합니다.
        p_sq_visc = (3.0 * p['mu'] * (L_char**2) * np.abs(node_v_world)) / (h_eff**3)
        
        # 국부 스퀴즈 압력 합산
        p_sq_nodes = p_sq_in + p_sq_visc
        
        # --- [추가] squeeze_min_h 이하에서 압력 0 처리 ---
        # 사용자의 요청에 따라 특정 높이 이하에서는 공기 쿠션 효과를 소멸시킵니다.
        p_sq_nodes[node_h_world <= p['squeeze_min_h']] = 0.0
        
        # --- [추가] 지면 근접 시 공기역학적 패널티 (Cubic Decay) ---
        # 사용자의 제안에 따라 h < 0.001m 구간에서 (h/0.001)^3 형태의 패널티를 적용합니다.
        # 이는 스퀴즈 압력이 1/h^3에 비례하여 무한히 커지는 현상을 상쇄시켜,
        # 박스가 지면에 완전히 안착할 수 있도록 도와줍니다.
        h_penalty_limit = p.get('h_aero_penalty', 0.001)
        
        # 기본 가중치는 1.0 (패널티 없음)
        gamma = np.ones_like(h_eff)
        # 패널티 구간에 있는 노드들에 대해 (h/h_limit)^3 적용
        penalty_mask = h_eff < h_penalty_limit
        if np.any(penalty_mask):
            gamma[penalty_mask] = (h_eff[penalty_mask] / h_penalty_limit)**3
        
        # 노드별 스퀴즈 힘 계산 (방향은 속도의 반대, 패널티 적용)
        v_eps = 1e-4  # 속도 전이 임계값
        f_squeeze_nodes = p_sq_nodes * p['da'] * -np.tanh(node_v_world / v_eps) * gamma
        
        # --- 스케일 계수 적용 (scale_squeeze) ---
        f_squeeze_nodes *= p.get('scale_squeeze', 1.0)
        
        f_squeeze_total = np.sum(f_squeeze_nodes)

        # --- 2. 6면 전체에 대한 공기 저항(Drag) 계산 ---
        f_drag_3d_vec = np.zeros(3)
        tau_drag_3d_vec = np.zeros(3)
        
        for face in self.faces:
            n_local = np.array(face['normal'])
            c_local = np.array(face['center'])
            
            # 법선 및 중심 좌표를 월드 좌표계로 변환
            n_world = rot @ n_local
            r_world_face = rot @ c_local
            
            # 해당 면 중심에서의 월드 속도 계산
            v_face_world = np.array([0, 0, v]) + np.cross(omegas, r_world_face)
            v_n = np.dot(v_face_world, n_world) # 법선 방향 속도 성분
            
            # 면이 공기를 가르고 나가는 방향(v_n > 1e-6)일 때 저항 발생
            if v_n > 1e-6:
                # 바닥면의 경우 지면 근처 패널티 적용
                face_gamma = 1.0
                '''
                if face['name'] == 'bottom':
                    # 바닥면의 평균적 높이 확인 (단순화)
                    h_face_mean = np.mean(h + r_world_face[2])
                    face_gamma = np.tanh(np.maximum(h_face_mean, 0) / p.get('h_aero_penalty', 0.001))
                '''
                # 저항력 f_vec = -0.5 * rho * Cd * Area * (v_n^2) * normal
                f_mag = 0.5 * p['rho'] * p['Cd'] * face['area'] * (v_n**2) * face_gamma
                f_vec = -(f_mag * p.get('scale_drag', 1.0)) * n_world
                
                f_drag_3d_vec += f_vec
                tau_drag_3d_vec += np.cross(r_world_face, f_vec)

        f_drag_z = f_drag_3d_vec[2]

        # --- 3. 지면 반력 계산 (스프링-감쇠 및 장벽 함수) ---
        f_contact_nodes = np.zeros_like(node_h_world)
        
        # [장벽 함수] h < 2mm 구간에서 무한 발산 (에너지 소산/감쇠 포함)
        barrier_h = p.get('contact_barrier_h', 0.002)
        barrier_k = p.get('contact_barrier_k', 5.0)
        barrier_mask = node_h_world < barrier_h
        if np.any(barrier_mask):
            h_val = np.maximum(node_h_world[barrier_mask], 1e-6)
            v_pen = -node_v_world[barrier_mask]
            
            # 강성 성분 (Inverse-Square)
            f_barrier_spring = barrier_k * p['node_k'] * ((barrier_h / h_val) - 1.0)**2
            
            # 감쇠 성분 (높이에 반비례하여 충격 흡수 강화)
            # 지면에 가까워질수록 감쇠가 커져서 튕겨나가는 에너지를 억제함
            f_barrier_damp = p['node_c'] * v_pen * (barrier_h / h_val)
            
            f_contact_nodes[barrier_mask] += np.maximum(0.0, f_barrier_spring + f_barrier_damp)

        # [물리적 접촉] h < 0 구간
        contact_mask = node_h_world < 0
        if np.any(contact_mask):
            delta = -node_h_world[contact_mask]
            v_pen = -node_v_world[contact_mask]
            f_spring = p['node_k'] * delta
            f_damp = p['node_c'] * v_pen
            f_contact_nodes[contact_mask] += np.maximum(0.0, f_spring + f_damp)

        f_contact_total = np.sum(f_contact_nodes)
        
        # --- 4. 수평 마찰력(Friction) 계산 ---
        # 수평 속도는 CoM의 이동이 없으므로 오직 각속도(omegas)에 의한 회전 성분만 존재합니다.
        v_all_nodes = np.cross(omegas, r_world) # 각 노드의 월드 좌표계 속도 벡터
        v_xy_nodes = v_all_nodes[:, :2]         # 수평(X, Y) 속도 성분
        
        tau_friction_vec = np.zeros(3)
        mu_f = p.get('mu_friction', 0.3)
        
        # 접촉력이 발생하는 노드들에 대해 마찰력 적용
        active_mask = (f_contact_nodes > 0)
        if np.any(active_mask):
            f_n = f_contact_nodes[active_mask]
            v_xy = v_xy_nodes[active_mask]
            
            v_xy_norm = np.linalg.norm(v_xy, axis=1, keepdims=True)
            v_eps_friction = 1e-3
            
            # 마찰력 방향: 수평 속도의 반대 방향 (매끄러운 전이를 위해 tanh 사용)
            f_friction_mag = mu_f * f_n
            f_friction_vec_xy = - (v_xy / (v_xy_norm + 1e-9)) * f_friction_mag[:, np.newaxis] * np.tanh(v_xy_norm / v_eps_friction)
            
            # 마찰력에 의한 토크 계산
            r_active = r_world[active_mask]
            f_friction_vec_3d = np.zeros((len(f_n), 3))
            f_friction_vec_3d[:, :2] = f_friction_vec_xy
            
            tau_friction_vec = np.sum(np.cross(r_active, f_friction_vec_3d), axis=0)

        # --- 5. 총 토크 합산 (Squeeze + Contact + Drag + Friction) ---
        # Squeeze와 Contact는 수직(Z) 방향이므로 별도 계산
        f_z_nodes = f_squeeze_nodes + f_contact_nodes
        force_vectors = np.zeros((len(r_world), 3))
        force_vectors[:, 2] = f_z_nodes
        tau_z_forces = np.sum(np.cross(r_world, force_vectors), axis=0)
        
        total_tau = tau_z_forces + tau_drag_3d_vec + tau_friction_vec
        
        return f_squeeze_total, f_drag_z, total_tau, f_contact_total
                
    def ode_func(self, t, y):
        """
        상미분방정식(ODE) 시스템을 정의합니다.
        """
        h, v, phi, theta, psi, wx, wy, wz = y
        omegas = np.array([wx, wy, wz])
        
        f_sq, f_dr, tau, f_contact = self.get_dynamics(h, v, [phi, theta, psi], omegas)
        f_aero = f_sq + f_dr
        
        dvdt = -self.params['gravity'] + (f_aero + f_contact) / self.params['mass']
        
        # --- [개선] 오일러 회전 방정식 (Full Euler's Rotation Equations) ---
        # tau = I * alpha + omega x (I * omega) => alpha = I^-1 * (tau - omega x (I * omega))
        I = self.params['I_matrix']
        gyro_term = np.cross(omegas, I @ omegas)
        dwdt = np.linalg.inv(I) @ (tau - gyro_term)
        
        s_phi, c_phi = np.sin(phi), np.cos(phi)
        s_theta, c_theta = np.sin(theta), np.cos(theta)

        if abs(c_theta) < 1e-9:
            t_theta = 1e9 * np.sign(s_theta)
            sec_theta = 1e9
        else:
            t_theta = s_theta / c_theta
            sec_theta = 1.0 / c_theta

        dphidt = wx + s_phi * t_theta * wy + c_phi * t_theta * wz
        dthetadt = c_phi * wy - s_phi * wz
        dpsidt = s_phi * sec_theta * wy + c_phi * sec_theta * wz
        
        return [v, dvdt, dphidt, dthetadt, dpsidt, dwdt[0], dwdt[1], dwdt[2]]

    def run(self, h_corners, t_sim, max_step):
        """
        전체 시뮬레이션을 실행합니다.
        """
        h0, angles0 = self.calculate_initial_state_from_corners(h_corners)
        
        sol = solve_ivp(
            fun=self.ode_func, 
            t_span=(0, t_sim), 
            y0=[h0, 0, *angles0, 0, 0, 0], 
            method='RK45',
            max_step=max_step
        )
        
        res = BoxDropResult(sol.t, sol.y, self.params)
        res.calculate_all_metrics(self.get_dynamics)
        return res

# =========================================================================
# --- 시뮬레이션 파라미터 설정 (여기서 값을 변경하여 시뮬레이션 수행) ---
# =========================================================================
PHYSICS_PARAMETERS = {
    # 1. 상자(Box)의 물리적 특성
    "width": 2.0,         # 상자 너비 (m)
    "depth": 1.6,         # 상자 깊이 (m)
    "height": 0.2,        # 상자 높이 (m)
    "mass": 30.0,         # 상자 질량 (kg)

    # 2. 환경 및 재료 물성 (Environment & Material)
    "gravity": 9.81,          # 중력 가속도 (m/s^2)
    "rho": 1.225,             # 공기 밀도 (kg/m^3)
    "mu": 1.81e-5,            # 공기 점성 계수 (Pa*s)
    "E_box": 1e8,           # 상자 재료의 영률 (Pa), e.g., 100 MPa
    "E_ground": 1e8,      # 지면 재료의 영률 (Pa), e.g., 10 GPa

    # 3. 공기 저항 및 접촉 모델 파라미터 (Aero & Contact Model)
    "Cd": 1.1,                      # 공기 저항 계수 (Drag Coefficient)
    "squeeze_min_h": 0.0001,        # 스퀴즈 필름 최소 높이 (m), 발산 방지
    "h_aero_penalty": 0.001,        # 공기 역학 패널티 시작 높이 (m) - h가 이보다 작아지면 공기력 감쇠
    "scale_drag": 9.0,              # 드래그 힘 스케일 계수 (0.0이면 비활성화)
    "scale_squeeze": 0.0,           # 스퀴즈 힘 스케일 계수 (0.0이면 비활성화)
    "damping_ratio": 0.5,           # 접촉 모델의 감쇠비 (zeta)
    "contact_exp_alpha": 0.1,       # 접촉 지수 경화 계수 (값이 클수록 침투 억제 강함)
    "contact_nodes_x": 20,          # 접촉 노드 해상도 (X축 방향 개수)
    "contact_nodes_y": 20,          # 접촉 노드 해상도 (Y축 방향 개수)
    "mu_friction": 0.4,             # 지면 수평 마찰 계수 (Coulomb Friction)
}

# =========================================================================
# --- 메인 실행 블록 ---
# =========================================================================
if __name__ == "__main__":
    
    # --- 1. 시뮬레이션 초기 조건 설정 ---
    # 네 모서리의 초기 높이 (m). 순서: 뒤왼(RL), 뒤오(RR), 앞오(FR), 앞왼(FL)
    #initial_corners = [0.50, 0.35, 0.30, 0.30] 
    initial_corners = [0.30, 0.30, 0.30, 0.30] 
    #initial_corners = [0.30, 1.70, 1.70, 0.30]
    
    # --- 2. 시뮬레이션 제어 파라미터 ---
    simulation_time = 1.2          # 총 시뮬레이션 시간 (s)
    solver_max_step = 0.0005        # ODE 솔버 최대 시간 스텝 (값이 작을수록 정밀도 증가)

    # --- 3. 시뮬레이터 객체 생성 및 실행 ---
    sim = BoxDropSimulator(sim_params=PHYSICS_PARAMETERS)
    result = sim.run(
        h_corners=initial_corners, 
        t_sim=simulation_time, 
        max_step=solver_max_step
    )

    # --- 4. 결과 시각화 ---
    # 4-1. 주요 물리 지표 2D 그래프 출력
    result.plot_physics_metrics()

    # 4-2. 3D 모션 애니메이션 재생
    #result.plot_3d_motion(interval=5, total_frames=100)

    # 4-3. 공기역학 성분 분석 그래프 추가
    plt.figure(figsize=(10, 4))
    plt.plot(result.t, result.aero_data['drag'], label='Drag Force')
    plt.plot(result.t, result.aero_data['squeeze'], label='Squeeze Film Force')
    plt.title("Aerodynamic Force Components (Z-axis)")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.grid(True)
    plt.legend()
    plt.show()