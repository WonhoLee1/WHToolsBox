import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import time

# ==========================================
# 1. 시뮬레이션 상수 및 설정
# ==========================================
L, W, H = 1.2, 0.8, 0.1  # m
MASS = 30.0  # kg
G_ACC = 9.806
DT = 0.001
TOTAL_STEPS = 2500  # 2.5초

# 공기 물성
AIR_DENSITY = 1.225
AIR_VISCOSITY = 1.8e-5

# 충돌체 설정
CUBE_SIZE = 0.02

# ==========================================
# 2. Helper 함수 (물성 변환)
# ==========================================
def calc_solref_from_youngs(E_mpa, damping_ratio, size_m, effective_mass=MASS):
    """MPa -> Solref 변환"""
    E_pa = E_mpa * 1e6
    k = E_pa * size_m
    if k <= 0: return "0.02 1.0"
    
    omega_n = np.sqrt(k / effective_mass)
    time_const = 1.0 / omega_n
    return f"{time_const:.5f} {damping_ratio}"

# ==========================================
# 3. Parametric Model Generator
# ==========================================
def create_model_xml(E_impact, E_mid, E_opposite):
    """
    3개 지점의 영률(MPa)을 받아 MuJoCo XML 문자열 생성
    - E_impact: 낙하 코너 (0번)
    - E_mid: 중간 지점
    - E_opposite: 반대 코너 (1번)
    * 나머지 코너들은 기본값(100MPa) 적용
    """
    # 1. 기본 설정 (일반 플라스틱)
    default_E = 100.0
    default_damping = 1.0
    default_solref = calc_solref_from_youngs(default_E, default_damping, CUBE_SIZE)
    
    # 2. 타겟 물성 설정
    # Damping은 일단 고정하거나 E에 따라 조절할 수 있으나, 여기선 E 변수화에 집중하기 위해 고정
    # (일반적으로 단단할수록 댐핑은 낮아짐. 여기선 0.5로 통일하여 강성 효과만 비교)
    target_damping = 0.5
    
    solref_impact = calc_solref_from_youngs(E_impact, target_damping, CUBE_SIZE)
    solref_mid = calc_solref_from_youngs(E_mid, target_damping, CUBE_SIZE)
    solref_opposite = calc_solref_from_youngs(E_opposite, target_damping, CUBE_SIZE)
    
    # 3. 코너 및 중간체 문자열 생성
    geoms_str = ""
    
    # (1) 8개 코너
    for i in range(8):
        c = np.array([(-L/2 if i in [0,2,4,6] else L/2), 
                      (-W/2 if i in [0,1,4,5] else W/2), 
                      (-H/2 if i in [0,1,2,3] else H/2)])
        
        # 물성 선택
        if i == 0:  # 낙하 코너
            solref = solref_impact
            rgba = "1 0 0 0.8" # Red
        elif i == 1: # 반대 코너
            solref = solref_opposite
            rgba = "0 0 1 0.8" # Blue
        else:
            solref = default_solref
            rgba = "0 1 0 0.3" # Green (Default)
            
        # Inset
        inset_pos = c - np.sign(c) * CUBE_SIZE
        
        geoms_str += f"""
        <geom name="g_corner_{i}" type="box" size="{CUBE_SIZE} {CUBE_SIZE} {CUBE_SIZE}" 
              pos="{inset_pos[0]} {inset_pos[1]} {inset_pos[2]}" 
              rgba="{rgba}" solref="{solref}" friction="0.4 0.005 0.0001" />
        """
        
    # (2) 4개 기둥 중간체
    depth_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    for (idx1, idx2) in depth_pairs:
        # 좌표 계산
        c1 = np.array([(-L/2 if idx1 in [0,2,4,6] else L/2), (-W/2 if idx1 in [0,1,4,5] else W/2), (-H/2 if idx1 in [0,1,2,3] else H/2)])
        c2 = np.array([(-L/2 if idx2 in [0,2,4,6] else L/2), (-W/2 if idx2 in [0,1,4,5] else W/2), (-H/2 if idx2 in [0,1,2,3] else H/2)])
        c_mid = (c1 + c2) / 2
        mid_inset_pos = c_mid - np.sign(c_mid) * CUBE_SIZE
        
        # 물성: 0-1번 사이만 E_mid 적용
        if (idx1, idx2) == (0, 1):
            solref = solref_mid
            rgba = "1 1 0 0.8" # Yellow
        else:
            solref = default_solref
            rgba = "0 1 0 0.3"
            
        geoms_str += f"""
        <geom name="g_mid_{idx1}_{idx2}" type="box" size="{CUBE_SIZE} {CUBE_SIZE} {CUBE_SIZE}" 
              pos="{mid_inset_pos[0]} {mid_inset_pos[1]} {mid_inset_pos[2]}" 
              rgba="{rgba}" solref="{solref}" friction="0.4 0.005 0.0001" />
        """

    # 4. XML 조립
    # 초기 회전 계산 (대각선 낙하)
    diagonal = np.array([L, W, H])
    diagonal_norm = diagonal / np.linalg.norm(diagonal)
    target_axis = np.array([0.01, 0, 1])
    rot_axis = np.cross(diagonal_norm, target_axis)
    rot_axis /= np.linalg.norm(rot_axis)
    angle = np.arccos(np.dot(diagonal_norm, target_axis))
    
    # 쿼터니언 변환
    r = R.from_rotvec(angle * rot_axis)
    quat = r.as_quat() # x,y,z,w
    quat_mj = [quat[3], quat[0], quat[1], quat[2]] # w,x,y,z
    
    # 회전 후 최저점 계산 -> 높이 설정
    # 코너 좌표 생성
    corners = np.array([[x,y,z] for x in [-L/2,L/2] for y in [-W/2,W/2] for z in [-H/2,H/2]])
    rot_mat = r.as_matrix()
    rot_corners = corners @ rot_mat.T
    min_z = np.min(rot_corners[:, 2])
    initial_z = 0.5 - min_z # 최저점 500mm
    
    # 관성 텐서
    Ixx = (1/12) * MASS * (W**2 + H**2)
    Iyy = (1/12) * MASS * (L**2 + H**2)
    Izz = (1/12) * MASS * (L**2 + W**2)

    xml = f"""
    <mujoco>
      <option timestep="{DT}" gravity="0 0 -{G_ACC}" density="{AIR_DENSITY}" viscosity="{AIR_VISCOSITY}">
        <flag contact="enable"/>
      </option>
      <worldbody>
        <geom name="floor" type="plane" size="3 3 1" rgba="0.8 0.9 0.8 1"/>
        <body name="box" pos="0 0 {initial_z}" quat="{quat_mj[0]} {quat_mj[1]} {quat_mj[2]} {quat_mj[3]}">
            <freejoint/>
            <inertial mass="{MASS}" diaginertia="{Ixx} {Iyy} {Izz}" pos="0 0 0"/>
            <geom type="box" size="{L/2} {W/2} {H/2}" rgba="0.5 0.5 0.5 0.2" contype="0" conaffinity="0"/>
            {geoms_str}
        </body>
      </worldbody>
    </mujoco>
    """
    return xml

# ==========================================
# 4. Simulation Runner (Advanced Analysis)
# ==========================================
def run_simulation(xml, label):
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    
    # ID Lookup
    geom_floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    geom_c0_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "g_corner_0")
    
    # Corner Local Positions for Kinematics
    # (XML 생성 로직과 동일하게 계산)
    corners_local = []
    for i in range(8):
        corners_local.append(np.array([
            (-L/2 if i in [0,2,4,6] else L/2), 
            (-W/2 if i in [0,1,4,5] else W/2), 
            (-H/2 if i in [0,1,2,3] else H/2)
        ]))
    corners_local = np.array(corners_local)

    # Metrics
    max_force_primary = 0.0   # 0번 코너 충격력
    max_force_secondary = 0.0 # 나머지 코너 충격력 (넘어질 때)
    
    max_slam_vel = 0.0        # 반대편(1번) 코너의 최대 하강 속도
    max_vel_diff = 0.0        # 코너간 최대 속도 차이
    
    t_first_impact = None
    t_second_impact = None
    
    # Run Simulation
    for _ in range(TOTAL_STEPS):
        mujoco.mj_step(model, data)
        sim_time = data.time
        
        # -----------------------------------
        # 1. Kinematics (Corner Velocity)
        # -----------------------------------
        # v_corner = v_com + w x r
        v_com = data.qvel[0:3]
        w_body = data.qvel[3:6]
        
        # 현재 회전 행렬
        quat = data.qpos[3:7] # w,x,y,z
        r_mat = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        
        # World frame corner velocities
        corner_vels = []
        for i in range(8):
            r_world = r_mat @ corners_local[i] # CoM에서 코너까지 벡터 (World)
            v_rot = np.cross(w_body, r_world)
            v_total = v_com + v_rot
            corner_vels.append(v_total)
            
        v_c0 = corner_vels[0] # Impact Corner
        v_c1 = corner_vels[1] # Opposite Corner (Slamming Target)
        
        # Slamming Speed Check (하강 속도 중 최대값)
        if v_c1[2] < 0: # 떨어지는 중
            # 하강 속도의 크기(|vz|)를 기록
            if abs(v_c1[2]) > max_slam_vel:
                max_slam_vel = abs(v_c1[2])
                
        # Velocity Difference (Distortion/Whip severity)
        v_diff = np.linalg.norm(v_c1 - v_c0)
        if v_diff > max_vel_diff:
            max_vel_diff = v_diff

        # -----------------------------------
        # 2. Contact Force Analysis
        # -----------------------------------
        f_primary_step = 0.0
        f_secondary_step = 0.0
        is_secondary_contact = False
        
        for i in range(data.ncon):
            contact = data.contact[i]
            
            # 충돌력 가져오기
            c_force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, c_force)
            force_mag = np.linalg.norm(c_force[0:3]) # Normal + Tangential
            # force_mag = c_force[0] # Normal only (if needed)
            
            # 충돌체 판별
            g1, g2 = contact.geom1, contact.geom2
            
            # 0번 코너와의 충돌인가?
            if geom_c0_id in (g1, g2):
                f_primary_step += force_mag
                if t_first_impact is None and force_mag > 1.0:
                    t_first_impact = sim_time
            # 바닥과의 충돌인데 0번이 아니다? -> Secondary
            elif geom_floor_id in (g1, g2): 
                # (바닥과 충돌한 모든 것들 중 0번 아닌 것)
                f_secondary_step += force_mag
                is_secondary_contact = True
        
        # Update Peak Forces
        max_force_primary = max(max_force_primary, f_primary_step)
        max_force_secondary = max(max_force_secondary, f_secondary_step)
        
        if is_secondary_contact and t_second_impact is None and f_secondary_step > 10.0:
             # 첫 충돌 이후 약간의 시간 차(Bounce)가 있을 수 있으므로
             # Primary랑 동시에 닿는 경우는 제외하거나 로직 정교화 필요하지만
             # 일단 2nd 힘이 발생하는 시점을 기록
             if t_first_impact is not None and (sim_time - t_first_impact > 0.05):
                 t_second_impact = sim_time

    # Calculate Lag
    time_lag = 0.0
    if t_first_impact and t_second_impact:
        time_lag = t_second_impact - t_first_impact
        
    return {
        'label': label,
        'F_prim_max': max_force_primary,
        'F_sec_max': max_force_secondary, # 중요: 넘어질 때 충격력
        'V_slam_max': max_slam_vel,       # 중요: 내려 꽂히는 속도
        'V_diff_max': max_vel_diff,       # 중요: 비틀림/채찍 강도
        'Time_Lag': time_lag
    }

# ==========================================
# 5. DOE Runner
# ==========================================
def run_doe():
    # 실험 계획 (Cases) [Impact(0), Mid, Opposite(1)] (MPa)
    cases = [
        ("Soft Homogeneous (10)",    10, 10, 10),
        ("Hard Homogeneous (1k)",  1000, 1000, 1000),
        ("Gradient Soft->Hard",      10, 100, 1000), 
        ("Gradient Hard->Soft",    1000, 100, 10), 
        ("Sandwich (S-H-S)",         10, 1000, 10),
        ("Sandwich (H-S-H)",       1000, 10, 1000),
    ]
    
    results = []
    print(f"🚀 Starting DOE Study ({len(cases)} cases) - Secondary Impact Analysis")
    print(f"{'Case':<22} | {'F_2nd (N)':<10} | {'V_slam (m/s)':<12} | {'V_diff':<10} | {'Lag (s)':<8}")
    print("-" * 75)
    
    for label, e1, e2, e3 in cases:
        xml = create_model_xml(e1, e2, e3)
        res = run_simulation(xml, label)
        results.append(res)
        
        print(f"{res['label']:<22} | {res['F_sec_max']:>10.1f} | {res['V_slam_max']:>12.2f} | {res['V_diff_max']:>10.2f} | {res['Time_Lag']:>8.3f}")
        
    print("-" * 75)
    
    # Plotting
    labels = [r['label'] for r in results]
    f2_vals = [r['F_sec_max'] for r in results]
    v_slam_vals = [r['V_slam_max'] for r in results]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Bar Chart for Force
    x = np.arange(len(labels))
    ax1.bar(x - 0.2, f2_vals, 0.4, label='2nd Impact Force (N)', color='tomato')
    ax1.set_ylabel('Force (N)', color='tomato')
    ax1.tick_params(axis='y', labelcolor='tomato')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha='right')
    
    # Line Chart for Slam Velocity
    ax2 = ax1.twinx()
    ax2.plot(x, v_slam_vals, 'b-o', label='Slam Velocity (m/s)', linewidth=2)
    ax2.set_ylabel('Velocity (m/s)', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    plt.title('Secondary Impact Analysis: Force & Slam Velocity')
    plt.tight_layout()
    plt.savefig('doe_secondary_impact.png')
    print("\n📊 Advanced Plot saved: doe_secondary_impact.png")

if __name__ == "__main__":
    run_doe()
