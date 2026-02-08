import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import time

# ==========================================
# 1. 물리 파라미터 및 모델 설정
# ==========================================
# MuJoCo는 SI 단위계 사용: 미터(m), 킬로그램(kg), 초(s)
L, W, H = 1.8, 1.2, 0.22  # m (1200mm, 800mm, 100mm)
MASS = 30.0  # kg
G_ACC = 9.806  # m/s^2
DT = 0.001  # 1ms (샘플링 간격)
TOTAL_STEPS = 2500  # 2.5초 시뮬레이션

# 상자 코너 8개 (로컬 좌표계, 미터 단위)
corners_local = np.array([
    [x, y, z]
    for x in [-L/2, L/2]
    for y in [-W/2, W/2]
    for z in [-H/2, H/2]
])

# 무게 중심 오프셋 (CoM Offset from Geometric Center)
# 예: [0, 0, -0.05] -> 무게 중심을 아래로 5cm 이동 (오뚝이 효과)
# 예: [0.3, 0.2, 0] -> 무게 중심을 X, Y 방향으로 편심
CoM_offset = np.array([0.0, 0.0, 0.00]) 
#CoM_offset = np.array([0.2, 0.1, 0.02]) 

# ==========================================
# Corner Drop 회전 계산
# ==========================================
# 대각선 벡터 (한 코너에서 반대편 코너로)
# 예: [-L/2, -W/2, -H/2] -> [+L/2, +W/2, +H/2]
diagonal = np.array([L, W, H])
diagonal_normalized = diagonal / np.linalg.norm(diagonal)

# 목표: 이 대각선이 Z축(0, 0, 1)과 평행하도록 회전
# Viewer Reset 시 초기 속도가 0이 되는 문제를 방지하기 위해 
# 아주 미세하게 기울여서(약 0.5도) 중력만으로도 바로 넘어지게 함
target_axis = np.array([0.01, 0, 1])

# 회전축: diagonal과 Z축의 외적
rotation_axis = np.cross(diagonal_normalized, target_axis)
rotation_axis_norm = np.linalg.norm(rotation_axis)

if rotation_axis_norm > 1e-6:
    rotation_axis = rotation_axis / rotation_axis_norm
    # 회전각: 두 벡터 사이의 각도
    cos_angle = np.dot(diagonal_normalized, target_axis)
    rotation_angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    
    # 축-각도 표현에서 회전 행렬 생성
    rot = R.from_rotvec(rotation_angle * rotation_axis)
else:
    # 이미 정렬됨
    rot = R.from_quat([0, 0, 0, 1])

quat = rot.as_quat()  # [x, y, z, w]
quat_mj = [quat[3], quat[0], quat[1], quat[2]]  # MuJoCo 순서: [w, x, y, z]

# 회전된 코너 계산
rotated_corners = corners_local @ rot.as_matrix().T
min_z = np.min(rotated_corners[:, 2])
max_z = np.max(rotated_corners[:, 2])

# 정밀 초기 고도 계산
initial_center_z = 0.5 - min_z  # 최저점이 Z=0.5m (500mm)가 되도록

# 각도 계산 (디버그용)
euler_angles = rot.as_euler('xyz', degrees=True)

# 관성 텐서 계산 (직육면체)
Ixx = (1/12) * MASS * (W**2 + H**2)
Iyy = (1/12) * MASS * (L**2 + H**2)
Izz = (1/12) * MASS * (L**2 + W**2)

# XML 모델 (초기 회전 + 관성 포함!)
# 초기 각속도를 미리 계산 (keyframe에 포함)
# 마찰이 낮을수록 초기 흔들림이 커야 균형이 깨짐
np.random.seed(42)
initial_angvel = np.random.uniform(-0.003, 0.003, 3)  # rad/s (±0.3 = 약 ±17도/초)

# ==========================================
# 1-2. 유체 역학 파라미터 (Fluid Dynamics)
# ==========================================
# 공기 물성
AIR_DENSITY = 1.225         # kg/m^3 (20°C, 1atm)
AIR_VISCOSITY = 1.8e-5      # Pa.s (Dynamic Viscosity)

# 유체 계수 (Fluid Coefficients for Box)
# [1] Blunt Drag: 정면 저항 (박스 형태는 0.8~1.2 내외)
# [2] Slender Drag: 측면 마찰 저항
# [3] Angular Drag: 회전 저항
# [4] Lift: 양력 계수 (기본값 0, 판자 형태는 0.1~0.5 가능)
# [5] Magnus: 회전 양력 (마그누스 효과)
COEF_BLUNT_DRAG = 0.5       
COEF_SLENDER_DRAG = 0.25
COEF_ANGULAR_DRAG = 1.5
COEF_LIFT = 1.0             # 양력 효과 추가!
COEF_MAGNUS = 1.0

# [New] Ground Effect (Air Cushion)
# 바닥 근처에서 공기가 빠져나가며 압력이 차오르는 현상 구현
# 값이 클수록 바닥 직전에 '푹신'하게 감속됨 (0.0이면 효과 없음)
COEF_GROUND_EFFECT = 1.0

# [New] Plastic Deformation Scale
# 코너 패드 변형(이동/축소)의 정도를 조절하는 계수
# 값이 클수록 충돌 시 더 많이 찌그러지고 안쪽으로 이동함. (기본값 0.5 -> 0.2로 완화)
# 0.0으로 설정 시 변형 없음 (형상 유지)
PLASTIC_DEFORMATION_RATIO = 0.1

# ==========================================
# [Helper] 재료 물성 변환 함수 (Young's Modulus -> Solref)
# ==========================================
def calc_solref_from_youngs(E_mpa, damping_ratio, size_m, effective_mass=MASS):
    """
    영률(Young's Modulus, MPa)을 MuJoCo solref로 변환
    - E_mpa (MPa): 재료의 영률 (예: 고무=10~100, 폼=5~10, 플라스틱=2000~)
    - size_m (m): 충돌체 한 변의 길이 (구조적 강성 계산용)
    
    공식: k(N/m) = E(Pa) * size(m)  (단순 큐브 압축 모델 가정)
    """
    # 1. MPa -> Pa 변환
    E_pa = E_mpa * 1e6
    
    # 2. 강성 k 계산 (k = E * s)
    k = E_pa * size_m
    
    
    # 3. Solref 계산 (기존 로직 재사용)
    if k <= 0: return "0.02 1.0"
    omega_n = np.sqrt(k / effective_mass)
    time_const = 1.0 / omega_n
    
    return f"{time_const:.5f} {damping_ratio}"

# ==========================================
# [사용자 튜닝 섹션] 재료 물성(MPa) 기반 코너 설정
# ==========================================
# 패드 크기 설정 (반폭 Half-Size 기준)
PAD_XY = 0.1        # 가로/세로 20cm -> 반폭 0.1m
PAD_Z  = H / 6.0    # 높이의 1/3 크기 -> 반폭 H/6

# 코너별 설정 리스트 초기화 (기본 재료: 일반 플라스틱, E=1000 MPa)
DEFAULT_SOLREF = calc_solref_from_youngs(1000, 1.0, 0.02) # size는 강성 계산용 참조값(작게 유지)
CORNER_PROPERTIES = []

for _ in range(8):
    CORNER_PROPERTIES.append({
        'solref': DEFAULT_SOLREF, 
        'rgba': "0 1 0 0.5"
    })

'''
🧪 조합 예시 (Simulation Recipe)
느낌	solref 값	설명
딱딱한 플라스틱	"0.004 1.0"	단단하고 튀지 않음 (기본값)
탱탱볼 (슈퍼볼)	"0.010 0.1"	약간만 부드럽고, 엄청나게 튕김
일반 고무	"0.020 0.5"	적당히 들어가고 적당히 튕김
메모리 폼 (스펀지)	"0.100 2.0"	쑥 들어가고(Soft), 천천히 나옴(Over-damped)
찰흙 / 찌그러짐	"0.150 8.0"	깊게 들어가고(Very Soft), 거의 안 나옴(Extreme Damping)
'''
# [New] Contact Parameters (Bouncing Effect)
# solref = (time_const, damping_ratio)
SOLREF_TIME_CONST = 0.05
SOLREF_DAMPING_RATIO = 0.5

# [New] Friction Parameters
# friction = "sliding torsional rolling"
# sliding: 미끄럼 마찰 (보통 0.5~1.0)
# torsional: 비틀림 마찰 (회전 저항, 0.005)
# rolling: 구름 마찰 (0.0001)
BOX_FRICTION_PARAMS = "0.3 0.005 0.0001"
# -------------------------------------------------------------
# Case 1: 낙하 지점(0번, 빨강) - "Plastic Distortion" (소성 변형 모사)
# -------------------------------------------------------------
foam_E = 0.01         # 0.01 MPa
foam_damping = 8.0    # [Key] 과감쇠(Over-damped) -> 찌그러진 후 복원 매우 느림
# TimeConst=0.05 (부드러움), Damping=8.0 (복원 억제)
CORNER_PROPERTIES[0]['solref'] = "0.05 0.5" 
CORNER_PROPERTIES[0]['rgba'] = "1 0 0 0.8" 

# [Safety Fix] 변수 재정의
PAD_XY = 0.1
PAD_Z = H / 6.0

# -------------------------------------------------------------
# Case 2: 인접 지점(1번, 파랑) - "Hard Rubber"
# -------------------------------------------------------------
hard_E = 0.01        
hard_damping = 0.9    
# [Fix] CUBE_SIZE -> 0.02 등 변수명 에러 수정
CORNER_PROPERTIES[1]['solref'] = calc_solref_from_youngs(hard_E, hard_damping, 0.02, effective_mass=0.01)
CORNER_PROPERTIES[1]['solref'] = "0.05 0.5"
CORNER_PROPERTIES[1]['rgba'] = "0 0 1 0.8"

# -------------------------------------------------------------
# Case 3: 중간 지점 (Midpoint) - MID_PROPERTIES 도입
# -------------------------------------------------------------
# 4개의 기둥 쌍: (0,1), (2,3), (4,5), (6,7)
MID_PROPERTIES = []
DEFAULT_MID_SOLREF = "0.05 0.5"
DEFAULT_MID_RGBA = "1 1 0 0.8" # Yellow

for _ in range(4): # 4 pairs
    MID_PROPERTIES.append({
        'solref': DEFAULT_MID_SOLREF,
        'rgba': DEFAULT_MID_RGBA
    })

# 예: 0번 쌍(0-1 사이)에 대해 특별한 물성 적용 (소성 변형 테스트)
# MID_PROPERTIES[0]['solref'] = "0.05 8.0" 
 
# ==========================================
# 1-3. XML 모델 생성
# ==========================================

# XML 모델용 코너 Site, Sensor, 그리고 [New] Collision Geom 문자열 생성
corner_sites_str = ""
corner_sensors_str = ""
corner_geoms_str = "" # [New] 충돌용 구 생성

# 패드 크기 설정 (반폭 Half-Size 기준) - 사용자 요청
PAD_XY = 0.1        # 20cm -> 반폭 0.1m
PAD_Z  = H / 6.0    # 높이의 1/3 크기 -> 반폭 H/6

# 1. 8개 코너 (상/하단 1/3 지점 배치)
for i in range(8):
    c = corners_local[i]
    props = CORNER_PROPERTIES[i] # solref, rgba만 참조
    
    # [Inset & Position Logic] 
    # XY: 코너 끝에서 PAD_XY(0.1)만큼 안으로
    # Z:  코너 끝에서 PAD_Z(H/6)만큼 안으로 (즉, 상/하단 1/3 영역)
    c_sign = np.sign(c)
    pos_x = c[0] - c_sign[0] * PAD_XY
    pos_y = c[1] - c_sign[1] * PAD_XY
    pos_z = c[2] - c_sign[2] * PAD_Z
    
    # 1. Site
    corner_sites_str += f'      <site name="s_corner_{i}" pos="{c[0]} {c[1]} {c[2]}" size="0.01" rgba="0 0 0 0"/>\n'
    # 2. Sensor
    corner_sensors_str += f'    <velocimeter name="vel_corner_{i}" site="s_corner_{i}" cutoff="50"/>\n'
    
    # 3. Collision Geom
    corner_geoms_str += f"""
      <geom name="g_corner_{i}" type="box" size="{PAD_XY} {PAD_XY} {PAD_Z}" 
            pos="{pos_x} {pos_y} {pos_z}" 
            rgba="{props['rgba']}" solref="{props['solref']}" 
            friction="{BOX_FRICTION_PARAMS}" />
    """

# 2. 4개 중간 기둥 (중앙 1/3 지점 배치)
depth_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]

for idx, (idx1, idx2) in enumerate(depth_pairs):
    props = MID_PROPERTIES[idx] # [New] 속성 참조
    
    c1 = corners_local[idx1]
    # 중간 지점은 Z=0이고, XY는 코너와 동일하게 Inset
    c_sign = np.sign(c1)
    mid_pos_x = c1[0] - c_sign[0] * PAD_XY
    mid_pos_y = c1[1] - c_sign[1] * PAD_XY
    
    corner_geoms_str += f"""
      <!-- Midpoint Collision ({idx1}-{idx2}) -->
      <geom name="g_mid_{idx1}_{idx2}" type="box" size="{PAD_XY} {PAD_XY} {PAD_Z}" 
            pos="{mid_pos_x} {mid_pos_y} 0.0"
            rgba="{props['rgba']}" solref="{props['solref']}" 
            friction="{BOX_FRICTION_PARAMS}" />
    """

# 3. [New] 4개의 외부 보호 블록 (Surface Protection Blocks) - [Refined] Non-overlapping
# 사용자의 요청에 따라 코너 패드와 겹치지 않게 크기를 정밀 조정하여 4면 커버
# -> 코너 블록의 안쪽 끝(Inner Edge)에 딱 맞게 배치

# 공통 설정
blk_thick = PAD_XY  # 블록 두께는 코너와 동일
blk_z = H / 2.0     # 높이는 전체 H 커버 (코너 위아래도 커버)

# 3-1. Front/Back Blocks (장변 커버)
# 코너의 중심: L/2 - PAD_XY, 코너의 반폭: PAD_XY
# -> 코너의 안쪽 끝(Inner Edge): L/2 - 2*PAD_XY
fb_sx = L/2.0 - 2.0 * PAD_XY 
fb_sy = blk_thick 
fb_pos_y = W/2.0 - fb_sy 

# 방어 코드
fb_sx = max(fb_sx, 0.001)

corner_geoms_str += f"""
      <!-- Front/Back Blocks (Long Edge Protection) - Mass negligible -->
      <geom name="g_front" type="box" size="{fb_sx} {fb_sy} {blk_z}" 
            pos="0 -{fb_pos_y} 0"
            rgba="0.3 0.3 0.3 1.0" solref="0.005 1.0" friction="{BOX_FRICTION_PARAMS}"
            mass="0.001" />
      <geom name="g_back" type="box" size="{fb_sx} {fb_sy} {blk_z}" 
            pos="0 {fb_pos_y} 0"
            rgba="0.3 0.3 0.3 1.0" solref="0.005 1.0" friction="{BOX_FRICTION_PARAMS}" 
            mass="0.001" />
"""

# 3-2. Left/Right Blocks (단변 커버)
lr_sx = blk_thick 
lr_sy = W/2.0 - 2.0 * PAD_XY # 양쪽 코너 제외
lr_pos_x = L/2.0 - lr_sx

# 방어 코드
lr_sy = max(lr_sy, 0.001)

corner_geoms_str += f"""
      <!-- Left/Right Blocks (Short Edge Protection) - Mass negligible -->
      <geom name="g_left" type="box" size="{lr_sx} {lr_sy} {blk_z}" 
            pos="-{lr_pos_x} 0 0"
            rgba="0.3 0.3 0.3 1.0" solref="0.005 1.0" friction="{BOX_FRICTION_PARAMS}" 
            mass="0.001" />
      <geom name="g_right" type="box" size="{lr_sx} {lr_sy} {blk_z}" 
            pos="{lr_pos_x} 0 0"
            rgba="0.3 0.3 0.3 1.0" solref="0.005 1.0" friction="{BOX_FRICTION_PARAMS}" 
            mass="0.001" />
    """

xml = f"""
<mujoco>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".8 .8 .8" rgb2=".9 .9 .9" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="8 8" texuniform="true"/>
  </asset>
  
  <!-- 유체 역학 설정 (Density, Viscosity) -->
  <option timestep="{DT}" gravity="0 0 -{G_ACC}" density="{AIR_DENSITY}" viscosity="{AIR_VISCOSITY}">
    <flag contact="enable"/>
  </option>
  
  <worldbody>
    <light pos="0 0 5" dir="0 0 -1" diffuse="0.7 0.7 0.7"/>
    <light pos="3 3 3" dir="-1 -1 -1" diffuse="0.5 0.5 0.5"/>

    <!-- 바닥: XY 평면 (Z=0) -->
    <geom name="floor" type="plane" pos="0 0 0" zaxis="0 0 1" size="3 3 1" material="grid" 
          friction="{BOX_FRICTION_PARAMS}" solref="0.01 1"/>
    
    <!-- 좌표축 시각화 -->
    <site name="origin" pos="0 0 0" size="0.03" rgba="1 1 1 0.8" type="sphere"/>
    <geom name="axis_x" type="capsule" fromto="0 0 0 0.5 0 0" size="0.008" rgba="1 0 0 1" contype="0" conaffinity="0"/> 
    <geom name="axis_y" type="capsule" fromto="0 0 0 0 0.5 0" size="0.008" rgba="0 1 0 1" contype="0" conaffinity="0"/> 
    <geom name="axis_z" type="capsule" fromto="0 0 0 0 0 0.5" size="0.008" rgba="0 0 1 1" contype="0" conaffinity="0"/> 
    
    <body name="box" pos="0 0 {initial_center_z}" quat="{quat_mj[0]} {quat_mj[1]} {quat_mj[2]} {quat_mj[3]}">
      <freejoint/>
      <!-- 무게 중심 (CoM) 설정 -->
      <inertial pos="{CoM_offset[0]} {CoM_offset[1]} {CoM_offset[2]}" mass="{MASS}" diaginertia="{Ixx} {Iyy} {Izz}"/>
      
      <!-- [Main Body] 시각 효과 및 공기역학 담당 (충돌 끔: contype=0 conaffinity=0) -->
      <!-- 오직 모양만 보여주고, 실제 벽 충돌은 코너 Sphere들이 담당함 -->
      <geom name="box_visual" type="box" size="{L/2} {W/2} {H/2}" rgba="0.1 0.5 0.8 0.3" 
            contype="0" conaffinity="0"
            fluidshape="ellipsoid"
            fluidcoef="{COEF_BLUNT_DRAG} {COEF_SLENDER_DRAG} {COEF_ANGULAR_DRAG} {COEF_LIFT} {COEF_MAGNUS}" />
      
      <!-- [Collision Bodies] 8개 코너별 개별 충돌체 -->
      {corner_geoms_str}
      
      <!-- 속도 측정을 위한 Site -->
      <site name="s_center" pos="0 0 0" size="0.01" rgba="1 1 0 1"/>
      
      <!-- 8개 코너 Site (자동 생성 - 위의 루프에서 포함됨) -->
      {corner_sites_str}
      
      <!-- 무게 중심 시각화 (빨간 점) -->
      <site name="s_com" pos="{CoM_offset[0]} {CoM_offset[1]} {CoM_offset[2]}" size="0.02" rgba="1 0 0 1"/>
    </body>
  </worldbody>
  
  <!-- 센서 정의 -->
  <sensor>
    <velocimeter name="vel_center" site="s_center" cutoff="50"/>
    <gyro name="angvel_box" site="s_center" cutoff="50"/>
    
    <!-- 8개 코너 Sensor (자동 생성) -->
{corner_sensors_str}
  </sensor>

  <!-- Keyframe: 초기 각속도 포함 (Viewer reset 시 복원) -->
  <keyframe>
    <key name="initial" 
         qpos="0 0 {initial_center_z} {quat_mj[0]} {quat_mj[1]} {quat_mj[2]} {quat_mj[3]}" 
         qvel="0 0 0 {initial_angvel[0]} {initial_angvel[1]} {initial_angvel[2]}"/>
  </keyframe>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
box_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box") # 박스 ID 캐싱

# ==========================================
# 2. 초기 상태 확정 + 균형 깨기
# ==========================================
# Keyframe에서 초기 상태 로드 (회전 + 초기 각속도 포함)
mujoco.mj_resetDataKeyframe(model, data, 0)  # keyframe 0번 "initial" 로드

# ==========================================
# 2-1. 커스텀 물리 효과 함수 (Air Cushion)
# ==========================================


# ==========================================
# 2-1. 커스텀 물리 효과 함수 (Air Cushion) - Global Callback
# ==========================================
def apply_air_cushion(model, data):
    """
    MuJoCo Physics Callback (Passive Forces)
    Squeeze Film Effect: [Advanced] Surface Integration Method
    바닥면을 격자(Grid)로 분할하여 각 지점의 높이와 속도를 기반으로 공기 저항력을 적분.
    -> 불균형한 힘 분포와 그로 인한 회전 모멘트(Torque)까지 정확히 계산.
    """
    body_id = box_body_id
    
    # Body State
    pos = data.xpos[body_id]
    rmat = data.xmat[body_id].reshape(3, 3) # (3,3) Rotation Matrix
    vel = data.cvel[body_id] # (6,) vector: [rot_vel(3), lin_vel(3)] 주의: cvel은 com 기준
    # 편의상 qvel 사용 (Free Joint인 경우)
    lin_vel = data.qvel[0:3]
    ang_vel = data.qvel[3:6] # Global frame angular velocity
    
    # 1. 바닥면 찾기 (Find Downward Face)
    # Body Frame의 Basis Vectors (X, Y, Z축)
    xaxis = rmat[:, 0]
    yaxis = rmat[:, 1]
    zaxis = rmat[:, 2]
    
    # Global Z (0,0,1)과 내적하여 가장 아래(-Z)를 향하는 축 찾기
    dots = [xaxis[2], yaxis[2], zaxis[2]] # Z component only (dot with [0,0,1])
    abs_dots = [abs(d) for d in dots]
    axis_idx = np.argmax(abs_dots) # 0=X, 1=Y, 2=Z
    sign = np.sign(dots[axis_idx]) # +1 or -1
    
    # 선택된 면의 정의 (Local Frame)
    # Normal Vector, Dimensions (u_size, v_size)
    if axis_idx == 0:   # X-face (YZ plane)
        normal = xaxis * sign
        u_vec, v_vec = yaxis, zaxis
        u_len, v_len = W, H
        local_normal_dist = L / 2.0 * sign * np.array([1,0,0])
    elif axis_idx == 1: # Y-face (XZ plane)
        normal = yaxis * sign
        u_vec, v_vec = xaxis, zaxis
        u_len, v_len = L, H
        local_normal_dist = W / 2.0 * sign * np.array([0,1,0])
    else:               # Z-face (XY plane, Default Bottom)
        normal = zaxis * sign
        u_vec, v_vec = xaxis, yaxis
        u_len, v_len = L, W
        local_normal_dist = H / 2.0 * sign * np.array([0,0,1])
        
    # 만약 가장 아랫면이 위를 보고 있다면(뒤집힘), sign을 고려해야 함.
    # 하지만 Squeeze는 '바닥에 가까운 면'이므로, 무조건 Global Z가 낮은 쪽을 택해야 함.
    # 위 로직에서 abs_dots로 축을 찾고, sign으로 방향(위/아래)을 체크하는데,
    # 우리가 원하는 건 'Normal이 -Z(아래)'인 면.
    if dots[axis_idx] > 0: # Normal이 위를 향함 -> 반대편 면이 바닥면
        normal = -normal
        local_normal_dist = -local_normal_dist
        
    # 2. 격자 적분 (Grid Integration)
    # 10x10 Grid (High Resolution for Corner Detection)
    N = 10
    dA = (u_len * v_len) / (N * N) # 격자 하나 면적
    
    total_force_z = 0.0
    total_torque = np.zeros(3)
    
    # Grid Loop
    # 면 중심(Center)에서 u, v 방향으로 순회
    grid_steps = np.linspace(-0.5 + 0.5/N, 0.5 - 0.5/N, N)
    
    # Body Center Position (World)
    body_pos = data.qpos[0:3]
    
    # Face Center Position (Relative to Body)
    face_center_local = local_normal_dist 
    face_center_world_vec = rmat @ face_center_local 

    # [물리 모델 업데이트]
    # 사용자 제안: Escape Velocity 기반 Bernoulli Pressure
    # v_escape = v_z * (Area / Perimeter_Gap)
    #          = v_z * (L*W) / (2*(L+W)*h)
    # P = 0.5 * rho * v_escape^2
    
    # 기하학적 특성 길이 (Hydraulic Diameter 유사)
    # flat_char_len = (L * W) / (2 * (L + W)) 
    # 코너 드랍 시에는 바닥에 닿는 면적이 변하지만, 
    # 최대 저항력을 결정하는 "유효 길이" 척도로서 전체 치수 사용
    geometric_factor = ((L * W) / (2 * (L + W))) ** 2
    
    # 최종 물리 계수
    # dF = (0.5 * rho * geo_factor) * (v/h)^2 * dA
    # COEF_GROUND_EFFECT는 사용자가 강도를 조절하는 '배율'로 사용 (기본 1.0)
    PHYSICS_COEF = 0.5 * AIR_DENSITY * geometric_factor * COEF_GROUND_EFFECT

    for u in grid_steps:
        for v in grid_steps:
            # R * (u*U + v*V)
            # u_vec, v_vec are World Frame Unit Vectors
            rel_pos = face_center_world_vec + (u * u_len) * u_vec + (v * v_len) * v_vec
            
            # Point World Position
            point_pos = body_pos + rel_pos
            h = point_pos[2] # Height form ground
            
            # 유효 높이 체크 (20cm 이내)
            if h < 0.001 or h > 0.2: continue
            
            # Point Velocity
            point_vel = lin_vel + np.cross(ang_vel, rel_pos)
            v_z = point_vel[2]
            
            if v_z < 0: # 내려갈 때만
                safe_h = max(h, 0.001) # 1mm 안전장치
                
                # [Physics]
                dF = PHYSICS_COEF * dA * (v_z / safe_h)**2
                
                # Force Limit (물리적 한계)
                dF = min(dF, 1000.0) 

                total_force_z += dF
                
                # Torque: r x F
                total_torque[0] += rel_pos[1] * dF
                total_torque[1] -= rel_pos[0] * dF
    
    # 3. Apply Forces
    data.xfrc_applied[body_id][2] = total_force_z
    data.xfrc_applied[body_id][3:6] = total_torque
    
    # [디버깅] 힘이 발생하면 출력 (1N 이상)
    # if total_force_z > 1.0:
    #     print(f"💨 Cushion ACTIVE: Fz={total_force_z:.1f} N, Tz={total_torque[2]:.1f}")


# 콜백 등록: mj_step 호출 시마다 자동 실행됨!
mujoco.set_mjcb_control(apply_air_cushion)

mujoco.mj_forward(model, data)  # 파생 물리량 계산

# ==========================================
# [New] Plastic Deformation Logic (소성 변형)
# ==========================================

def apply_plastic_deformation(model, data, plastic_ratio=0.5):
    """
    충돌 시 침투 깊이만큼 Geom을 영구적으로 변형(축소+이동)시킵니다.
    [개선] 
    1. Geom별 Max Penetration만 적용 (중복 변형 방지)
    2. Size 축소보다 안쪽 이동(Shift) 가중치 부여 (시각적 리얼리티)
    """
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    
    # Geom별 최대 침투 깊이 저장소
    geom_deformations = {}
    
    for i in range(data.ncon):
        con = data.contact[i]
        
        # 바닥과의 충돌인지 확인
        g1, g2 = con.geom1, con.geom2
        target_geom = None
        
        if g1 == floor_id: target_geom = g2
        elif g2 == floor_id: target_geom = g1
        else: continue
            
        # 침투 깊이 확인
        penetration = -con.dist
        if penetration > 1e-4: # 0.1mm 이상
            current_max = geom_deformations.get(target_geom, 0.0)
            if penetration > current_max:
                geom_deformations[target_geom] = penetration

    # 집계된 최대 침투 깊이로 변형 적용
    for geom_id, penetration in geom_deformations.items():
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if name is None: continue
        
        if "g_corner" in name or "g_mid" in name or name in ["g_front", "g_back", "g_left", "g_right"]:
            
            # Global Scale 적용            
            deformation = penetration * plastic_ratio
            
            # 내측 방향 벡터 (로컬 좌표계 기준, 중심을 향해)
            current_pos = model.geom_pos[geom_id]
            # 중심(0,0,0) 방향이 안쪽임.
            # 코너는 (+,+), (+,-) 등이므로 sign 반대가 안쪽.
            inward_dir = -np.sign(current_pos[:3])
            
            current_size = model.geom_size[geom_id]
            
            if current_size[0] > 0.005:
                # [전략] Shift 위주. Size 감소는 20%, Shift는 80%
                shrink_amount = deformation * 0.2
                shift_amount = deformation * 0.8
                
                # XY 평면 수축
                model.geom_size[geom_id][0] -= shrink_amount
                model.geom_size[geom_id][1] -= shrink_amount
                
                # 중심 이동 (안쪽으로)
                model.geom_pos[geom_id][0] += inward_dir[0] * shift_amount
                model.geom_pos[geom_id][1] += inward_dir[1] * shift_amount
                
                # 최소 크기 방어
                model.geom_size[geom_id][0] = max(model.geom_size[geom_id][0], 0.001)
                model.geom_size[geom_id][1] = max(model.geom_size[geom_id][1], 0.001)

print("="*70)
print("🎯 Box Drop Simulation - Corner Drop (Diagonal Vertical)")
print("="*70)
print(f"📦 Box: {L*1000:.0f} × {W*1000:.0f} × {H*1000:.0f} mm, {MASS} kg")
print(f"📏 Drop height: {initial_center_z*1000:.1f} mm (lowest corner at 500 mm)")
print(f"� Diagonal length: {np.linalg.norm(diagonal)*1000:.1f} mm")
print(f"�🔄 Rotation (calculated): Roll={euler_angles[0]:.1f}°, Pitch={euler_angles[1]:.1f}°, Yaw={euler_angles[2]:.1f}°")
print(f" Diagonal length: {np.linalg.norm(diagonal)*1000:.1f} mm")
print(f"🔄 Rotation (calculated): Roll={euler_angles[0]:.1f}°, Pitch={euler_angles[1]:.1f}°, Yaw={euler_angles[2]:.1f}°")
print(f"   Quaternion (WXYZ): [{data.qpos[3]:.3f}, {data.qpos[4]:.3f}, {data.qpos[5]:.3f}, {data.qpos[6]:.3f}]")
print(f"   Vertical span: {(max_z - min_z)*1000:.1f} mm (min={min_z*1000:.1f}, max={max_z*1000:.1f})")
print(f"💨 Air resistance: ρ=1.225 kg/m³, ν=1.48×10⁻⁵ m²/s")
print("="*70 + "\n")

# ==========================================
# 3. Phase 1: 인터랙티브 미리보기 (모드 선택)
# ==========================================
import msvcrt

# 초기 상태 복원용 백업 (Phase 2에서 사용)
initial_qpos = data.qpos.copy()
initial_qvel = data.qvel.copy()

def run_standard_viewer():
    """기본 MuJoCo 뷰어 실행 (마우스 제어 중심)"""
    print("\n🎮 Mode 1: Standard Viewer")
    print("   Controls:")
    print("   - Space: Pause/Resume")
    print("   - Right Arrow: Advance 1 step (when paused)")
    print("   - Backspace: Reset (Speed may be zeroed out!)")
    print("   - Close window: Start data collection\n")
    
    mujoco.viewer.launch(model, data)

def run_passive_viewer(xml_string):
    """커스텀 제어 루프 (키보드 제어 중심, 물리적 리셋 지원)"""
    print("\n🎮 Mode 2: Passive Viewer (Custom Control)")
    print("   (Press SPACE to start, BACKSPACE to Reset, ESC to Finish)")

    # 초기 형상 백업 (물리적 리셋용)
    initial_geom_size = model.geom_size.copy()
    initial_geom_pos = model.geom_pos.copy()

    # 상태 변수
    paused = True
    reset_trigger = False
    step_trigger = False
    slow_motion = 5.0
    run_start_time = None
    MAX_RUN_TIME = 10.0
    
    should_quit = False

    def key_callback(keycode):
        nonlocal paused, reset_trigger, step_trigger, slow_motion, run_start_time, should_quit
        
        # Spacebar (32): Toggle Pause
        if keycode == 32:
            paused = not paused
            if not paused:
                run_start_time = time.time()
                print(f"   [RUNNING] Speed: 1/{slow_motion:.1f}x")
            else:
                print("   [PAUSED]")
        
        # Right Arrow (262): Step forward
        elif keycode == 262 and paused:
            step_trigger = True

        # Backspace (259 in GLFW) or R (82): Reset
        elif keycode == 259 or keycode == 82:
            reset_trigger = True
            
        # ESC (256 in GLFW): Quit to Phase 2
        elif keycode == 256:
            should_quit = True
            
        # Minus (-): Slower
        elif keycode == 45: 
            slow_motion = min(slow_motion + 1.0, 20.0)
            print(f"   [SPEED] Slower -> 1/{slow_motion:.1f}x")
            
        # Equal (=): Faster
        elif keycode == 61:
            slow_motion = max(slow_motion - 1.0, 0.1)
            if slow_motion < 1.0: slow_motion = 1.0
            print(f"   [SPEED] Faster -> 1/{slow_motion:.1f}x")

    try:
        with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
            # 초기 카메라
            viewer.cam.distance = 3.0
            viewer.cam.lookat = [0, 0, 0.5]
            viewer.sync()
            
            while viewer.is_running():
                # Quit Check
                if should_quit:
                    print("   [QUIT] Proceeding to Data Collection...")
                    viewer.close()
                    return

                # Reset Check
                if reset_trigger:
                    paused = True
                    reset_trigger = False
                    print("   [RESET] Physics State & Geometry Reset (Visual update may lag)")
                    
                    # 1. 물리 상태 초기화
                    mujoco.mj_resetData(model, data)            
                    mujoco.mj_resetDataKeyframe(model, data, 0) 
                    mujoco.mj_forward(model, data)
                    
                    # 2. 형상(Geom) 원상 복구 (물리적)
                    model.geom_size[:] = initial_geom_size[:]
                    model.geom_pos[:] = initial_geom_pos[:]
                    
                    # 3. 씬 갱신 시도 (MuJoCo 한계로 시각적 반영 안 될 수 있음)
                    # viewer.update_hfield(0) 
                    viewer.sync()
                    continue

                # Step (1 frame)
                if step_trigger:
                    step_trigger = False
                    print("   [STEP] +1 frame")
                    mujoco.mj_step(model, data)
                    apply_plastic_deformation(model, data, plastic_ratio=PLASTIC_DEFORMATION_RATIO)
                    viewer.sync()

                # Running
                if not paused:
                    step_start = time.time()
                    mujoco.mj_step(model, data)
                    apply_plastic_deformation(model, data, plastic_ratio=PLASTIC_DEFORMATION_RATIO)
                    viewer.sync()
                    
                    # Auto Stop
                    if run_start_time and time.time() - run_start_time > MAX_RUN_TIME * slow_motion:
                        print(f"   [AUTO-STOP] Timeout. Pausing.")
                        paused = True

                    # Slow Motion Sync
                    elapsed = time.time() - step_start
                    target_delay = model.opt.timestep * slow_motion
                    time_until_next_step = target_delay - elapsed
                    
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
                else:
                    viewer.sync()
                    time.sleep(0.01)

    except Exception as e:
        print(f"⚠️  Error in passive viewer: {e}")


# 사용자 입력 받기
print("Select Preview Mode:")
print("1. Standard Viewer (Simpler, Reset glitch exists)")
print("2. Passive Viewer (Custom Control, Perfect Reset)")
mode = input("Enter mode (1 or 2, default=2): ").strip()

if mode == "1":
    run_standard_viewer()
else:
    run_passive_viewer(xml)  # Pass xml string explicitly

print("\n✅ Preview 완료\n")

# ==========================================
# 4. Phase 2: 데이터 수집 시뮬레이션
# ==========================================
# [Important] 모델 재로딩 (Phase 1에서 찌그러진 것 초기화)
print("🔄 Resetting model for data collection...")

# 1. [Safety] 기존 콜백 해제 (충돌 방지)
mujoco.set_mjcb_control(None)

# 2. 모델 새로 생성
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# 3. [Restore] 콜백 다시 등록
mujoco.set_mjcb_control(apply_air_cushion)

# Keyframe에서 초기 상태 로드 (다시)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

# Phase 2 루프 내에는 apply_plastic_deformation을 직접 호출하므로
# 중복 방지를 위해 콜백을 다시 해제하거나 유지할 수 있음.
# 여기서는 apply_air_cushion은 유지해야 하므로 해제하지 않음!
# (단, 소성 변형 함수는 수동 호출)

print("📊 Phase 2: Data Collection")
print(f"   Duration: {TOTAL_STEPS * DT:.1f}s, Sampling: {DT*1000:.1f}ms\n")

# 초기 상태 복원
data.qpos[:] = initial_qpos
data.qvel[:] = initial_qvel
mujoco.mj_forward(model, data)  # 관성 텐서 등 파생 물리량 재계산

# 데이터 기록 구조
history = {
    'time': [],
    'center': {'pos': [], 'vel': [], 'acc': []},
    'corners': [{'pos': [], 'vel': [], 'acc': []} for _ in range(8)],
    'impact_force': [],  # 총 충격력 (수직항력 합계)
    'cushion_force': []  # [New] 에어 쿠션 힘 기록
}

prev_center_vel = np.zeros(3)
prev_corner_vels = np.zeros((8, 3))

# 데이터 수집 시뮬레이션 (백그라운드, 뷰어 없음)
print("   Running simulation steps...", end="", flush=True)

for step_count in range(TOTAL_STEPS):
    # --------------------------------------
    # 1. Physics Step (Air Cushion Callback 자동 적용됨)
    # --------------------------------------
    mujoco.mj_step(model, data)
    
    # --------------------------------------
    # 2. Plastic Deformation (Post-Step)
    # --------------------------------------
    apply_plastic_deformation(model, data, plastic_ratio=PLASTIC_DEFORMATION_RATIO)

    # --------------------------------------
    # 3. 데이터 기록 (Logging)
    # --------------------------------------
    t = data.time
    
    # 3-1. Air Cushion Force (적용된 외력 읽기)
    f_cushion_val = data.xfrc_applied[box_body_id][2]
    
    history['time'].append(t)
    history['cushion_force'].append(f_cushion_val)
    
    # 3-2. Center State
    history['center']['pos'].append(data.qpos[0:3].copy())
    history['center']['vel'].append(data.qvel[0:3].copy())
    history['center']['acc'].append(data.qacc[0:3].copy()) 
    
    # 3-3. Corner States
    for i in range(8):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"s_corner_{i}")
        if site_id != -1:
            # Position
            history['corners'][i]['pos'].append(data.site_xpos[site_id].copy())
            
            # Velocity (Linear)
            res = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, site_id, res, 0)
            curr_vel = res[3:6].copy()
            history['corners'][i]['vel'].append(curr_vel)
            
            # Acceleration (Numerical Diff)
            # prev_corner_vels는 루프 밖에서 초기화됨 (Step 808쯤)
            acc = (curr_vel - prev_corner_vels[i]) / DT
            history['corners'][i]['acc'].append(acc)
            
            # Update prev velocity for next step
            prev_corner_vels[i] = curr_vel.copy() 
    
    # 3-4. Impact Force (Contact Normal Force Sum)
    total_impact = 0.0
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    
    for i in range(data.ncon):
        con = data.contact[i]
        # 바닥 충돌만 추출
        if con.geom1 == floor_id or con.geom2 == floor_id:
            c_force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, c_force)
            # Contact Frame Z-axis is Normal
            total_impact += c_force[0]
            
    history['impact_force'].append(total_impact)
    
    # Progress Bar
    if step_count % (TOTAL_STEPS // 10) == 0:
        print(".", end="", flush=True)

print(" Done!")
print("\n✅ Data collection 완료\n")
# ==========================================
# 5. 데이터 변환 (list -> numpy array)
# ==========================================
history['time'] = np.array(history['time'])
history['impact_force'] = np.array(history['impact_force'])
history['cushion_force'] = np.array(history['cushion_force']) # [New]
for key in ['pos', 'vel', 'acc']:
    history['center'][key] = np.array(history['center'][key])
    for idx in range(8):
        history['corners'][idx][key] = np.array(history['corners'][idx][key])

# ==========================================
# 6. 그래프 생성
# ==========================================
print("📈 Generating plots...\n")

# Figure 1: 위치, 속도, 가속도
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
fig.suptitle('Box Drop Simulation: Position, Velocity, Acceleration (Center + 8 Corners)', fontsize=14, fontweight='bold')

labels = ['X', 'Y', 'Z']
row_titles = ['Position (mm)', 'Velocity (mm/s)', 'Acceleration (mm/s²)']
data_keys = ['pos', 'vel', 'acc']
scale_factors = [1000, 1000, 1000]

colors = plt.cm.tab10(np.linspace(0, 1, 9))

for row, (data_key, row_title, scale) in enumerate(zip(data_keys, row_titles, scale_factors)):
    for col, axis_label in enumerate(labels):
        ax = axes[row, col]
        # 중심 데이터
        ax.plot(history['time'], history['center'][data_key][:, col] * scale, 
                label='Center', color=colors[0], linewidth=2, alpha=0.8)
        # 8개 코너 데이터
        for idx in range(8):
            ax.plot(history['time'], history['corners'][idx][data_key][:, col] * scale, 
                    label=f'Corner {idx+1}', color=colors[idx+1], linewidth=1, alpha=0.6)
        
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel(f'{row_title}', fontsize=10)
        ax.set_title(f'{row_title} - {axis_label} axis', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if row == 0 and col == 2:
            ax.legend(loc='upper right', fontsize=8, ncol=1)

plt.tight_layout()
plt.savefig('box_drop_analysis.png', dpi=150, bbox_inches='tight')
print("📊 Graph saved: box_drop_analysis.png")

# Figure 2: 충격력 (Impact Force) + 에어 쿠션 (Cushion Force)
plt.figure(figsize=(10, 6))
# 충격력 (빨강)
plt.plot(history['time'], history['impact_force'], color='red', linewidth=1.5, label='Contact Impact Force')
# 에어 쿠션 힘 (파랑)
plt.plot(history['time'], history['cushion_force'], color='blue', linewidth=1.5, linestyle='--', label='Air Cushion Force (Global Z)', alpha=0.7)

max_force = np.max(history['impact_force'])
max_force_idx = np.argmax(history['impact_force'])
max_force_time = history['time'][max_force_idx]

plt.scatter(max_force_time, max_force, color='black', zorder=5)
plt.annotate(f'Peak Impact: {max_force:.1f} N\n@ {max_force_time:.3f} s', 
             xy=(max_force_time, max_force), 
             xytext=(max_force_time + 0.2, max_force),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.title('Forces on Box: Impact & Air Cushion', fontsize=14, fontweight='bold')
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Force (N)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('box_drop_impact_force.png', dpi=150, bbox_inches='tight')
print("📊 Graph saved: box_drop_impact_force.png")
plt.show()

print("\n✅ All tasks completed!")
