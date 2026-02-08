import os
import jax
from jax import numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
from scipy.spatial.transform import Rotation as R
import mediapy as media
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import mujoco.viewer
import time

# ==========================================
# 1. 물리 규격 및 환경 설정
# ==========================================
L, W, H = 1600.0, 800.0, 200.0  # mm
MASS = 0.030 * 1000  # tonne to kg (30kg)
G_ACC = 9806.0       # mm/s^2
DT = 0.002
TOTAL_STEPS = 2000 # 4초 분량 (0.002 * 2000)

# 상자 코너 8개 (로컬 좌표계)
corners_local = np.array([
    [x, y, z]
    for x in [-L/2, L/2]
    for y in [-W/2, W/2]
    for z in [-H/2, H/2]
])

xml = f"""
<mujoco>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".8 .8 .8" rgb2=".9 .9 .9" width="300" height="300" mark="edge" markrgb=".8 .8 .8"/>
    <material name="grid" texture="grid" texrepeat="5 5" texuniform="true"/>
  </asset>
  <option timestep="{DT}" gravity="0 -{G_ACC} 0">
    <flag contact="enable"/>
  </option>
  <visual>
    <global offwidth="1920" offheight="1080"/>
  </visual>
  <worldbody>
    <!-- Lighting -->
    <light pos="0 5000 5000" dir="0 -1 -1" diffuse="0.7 0.7 0.7"/>
    <light pos="0 5000 -5000" dir="0 -1 1" diffuse="0.4 0.4 0.4"/>

    <!-- 바닥 설정: zaxis="0 1 0"으로 법선을 위쪽(+Y)으로 고정 -->
    <geom name="floor" type="plane" pos="0 0 0" zaxis="0 1 0" size="3000 3000 1" material="grid" friction="0.8" solref="0.02 1"/>
    
    <!-- 좌표축 시각화 (물리 계산 제외: contype/conaffinity 0) -->
    <site name="origin" pos="0 0 0" size="30" rgba="1 1 1 0.8" type="sphere"/>
    <geom name="axis_x" type="cylinder" fromto="0 0 0 500 0 0" size="10" rgba="1 0 0 1" contype="0" conaffinity="0"/> 
    <geom name="axis_y" type="cylinder" fromto="0 0 0 0 500 0" size="10" rgba="0 1 0 1" contype="0" conaffinity="0"/> 
    <geom name="axis_z" type="cylinder" fromto="0 0 0 0 0 500" size="10" rgba="0 0 1 1" contype="0" conaffinity="0"/> 
    
    <body name="box" pos="0 500 0">
      <freejoint/>
      <geom name="box_geom" type="box" size="{L/2} {W/2} {H/2}" mass="{MASS}" rgba="0.1 0.5 0.8 1" solref="0.01 1" solimp="0.95 0.99 0.001"/>
    </body>
  </worldbody>
</mujoco>
"""

# JAX 호환 코너 좌표 (8x3)
jax_corners_local = jnp.array(corners_local)

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
mx_model = mjx.put_model(model)

# ==========================================
# 2. 물리 엔진 (Squeeze Film Effect)
# ==========================================
@jax.jit
def step_fn(m, d):
    # 상자의 현재 위치 및 쿼터니언 회전
    pos = d.qpos[0:3]
    quat = d.qpos[3:7] # [w, x, y, z]
    
    # 쿼터니언을 회전 행렬로 변환
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    mat = jnp.array([
        [1-2*y**2-2*z**2, 2*x*y-2*w*z, 2*x*z+2*w*y],
        [2*x*y+2*w*z, 1-2*x**2-2*z**2, 2*y*z-2*w*x],
        [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x**2-2*y**2]
    ])
    
    # 8개 코너의 월드 좌표 Y값 계산
    corners_world_y = pos[1] + jnp.dot(jax_corners_local, mat[1, :])
    min_corner_y = jnp.min(corners_world_y)
    
    # 지면과의 틈새
    h_gap = jnp.maximum(min_corner_y, 0.1) 
    
    # 베르누이 효과에 의한 공기 탈출 속도
    vel_y = d.qvel[1]
    v_escape = jnp.abs(vel_y) * ((L*W) / (2*(L+W) * h_gap))
    
    # Bernoulli 압력 기반 반력
    f_squeeze = jnp.where((vel_y < 0) & (h_gap < 150), 0.5 * 1.225e-9 * (v_escape**2) * (L*W), 0.0)
    
    # 외력 적용
    d = d.replace(qfrc_applied=d.qfrc_applied.at[1].set(f_squeeze))
    
    return mjx.step(m, d)

# ==========================================
# 3. 초기 상태 설정 (ISTA 6A 코너 낙하)
# ==========================================
rot = R.from_euler('xyz', [34, 22, 15], degrees=True)
quat = rot.as_quat() # [x, y, z, w] -> Mujoco는 [w, x, y, z]

# --- 정밀 초기 고도 계산 ---
# 로컬 코너 좌표를 회전시킨 후, 가장 낮은 지점의 y좌표를 찾음
rotated_corners = corners_local @ rot.as_matrix().T
min_y = np.min(rotated_corners[:, 1])

# 가장 낮은 점이 지면으로부터 300mm가 되도록 중심 위치(pos_y) 설정
initial_center_y = 300.0 - min_y

data.qpos[0:3] = [0, initial_center_y, 0]
data.qpos[3:7] = [quat[3], quat[0], quat[1], quat[2]]
mx_data = mjx.put_data(model, data)

# ==========================================
# 4. 시뮬레이션 및 데이터 추출
# ==========================================
history = {
    'time': [],
    'ke': [], 'pe': [], 'ie': [], 'total_e': [],
    'grf': [],
    'corners': [[] for _ in range(8)]
}

print(f"🚀 초기 중심 고도: {initial_center_y:.1f} mm (최저점 300 mm 기준)")
print("� 시뮬레이션 및 영상 렌더링 시작...")

# 렌더러 초기화
renderer = mujoco.Renderer(model, height=720, width=1280)

# 카메라 설정
camera = mujoco.MjvCamera()
camera.azimuth = 135
camera.elevation = -25
camera.distance = 4500
camera.lookat = [0, 200, 0]

frames = []
prev_corner_vels = np.zeros((8, 3))

# 시뮬레이션 루프
for i in range(TOTAL_STEPS):
    # MJX Step
    mx_data = step_fn(mx_model, mx_data)
    d_host = mjx.get_data(model, mx_data)
    
    # 렌더링 (매 10스텝마다 한 번씩 저장하여 렌더링 부하 감소)
    if i % 10 == 0:
        mujoco.mj_forward(model, d_host) # 렌더링을 위해 CPU 모델 업데이트
        renderer.update_scene(d_host, camera=camera)
        frames.append(renderer.render())
        print(f"⏳ 진행률: {i/TOTAL_STEPS*100:.1f}%", end='\r')

    # --- 데이터 기록 ---
    t = i * DT
    history['time'].append(t)
    
    lin_vel = d_host.qvel[0:3]
    ke = 0.5 * MASS * np.sum(lin_vel**2)
    pe = MASS * G_ACC * d_host.qpos[1]
    
    history['ke'].append(ke)
    history['pe'].append(pe)
    history['ie'].append(0.0)
    history['total_e'].append(ke + pe)
    
    # 지면 반력
    grf = 0.0
    for j in range(d_host.ncon):
        c_force = np.zeros(6)
        mujoco.mj_contactForce(model, d_host, j, c_force)
        grf += c_force[0]
    history['grf'].append(grf)
    
    # 8개 꼭지점 거동 추적
    pos = d_host.qpos[0:3]
    rot_mat = R.from_quat([d_host.qpos[4], d_host.qpos[5], d_host.qpos[6], d_host.qpos[3]]).as_matrix()
    for idx in range(8):
        local_p = corners_local[idx]
        world_p = pos + rot_mat @ local_p
        vel_center = d_host.qvel[0:3]
        omega = d_host.qvel[3:6]
        r_vec = world_p - pos
        vel_corner = vel_center + np.cross(omega, r_vec)
        acc_corner = (vel_corner - prev_corner_vels[idx]) / DT
        prev_corner_vels[idx] = vel_corner
        history['corners'][idx].append({'y': world_p[1], 'vy': vel_corner[1], 'ay': acc_corner[1]})

print("\n✅ 시뮬레이션 완료. 영상 저장 중...")

# 영상 저장 (FPS는 DT와 프레임 스킵 고려: 30FPS 정도)
# 1 step = 0.002s, 10 step skip = 0.02s -> 50FPS
media.write_video('simulation_video.mp4', frames, fps=50)
print("🎬 'simulation_video.mp4' 파일이 생성되었습니다.")

# 리소스 해제
renderer.close()

print("✅ 시뮬레이션 종료. 그래프 생성 중...")

# ==========================================
# 5. 시각화 및 리포트
# ==========================================
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# 1. 꼭지점 거동 (Height/Vel/Acc for Corner 0)
c0_y = [s['y'] for s in history['corners'][0]]
c0_vy = [s['vy'] for s in history['corners'][0]]
c0_ay = [s['ay'] for s in history['corners'][0]]

axs[0].plot(history['time'], c0_y, label='Height (mm)')
axs[0].set_title("Corner 0 Kinematics")
axs[0].legend(loc='upper right')
axs[0].grid(True)

ax0_2 = axs[0].twinx()
ax0_2.plot(history['time'], c0_vy, 'g-', label='Velocity (mm/s)', alpha=0.5)
ax0_2.plot(history['time'], np.array(c0_ay)/1000, 'r-', label='Accel (m/s^2)', alpha=0.3)
ax0_2.legend(loc='lower right')

# 2. 에너지 변화
axs[1].plot(history['time'], history['ke'], label='Kinetic')
axs[1].plot(history['time'], history['pe'], label='Potential')
axs[1].plot(history['time'], history['ie'], label='Internal (Elastic)')
axs[1].plot(history['time'], history['total_e'], 'k--', label='Total')
axs[1].set_title("Energy Balance")
axs[1].legend()
axs[1].grid(True)

# 3. 지면 반력
axs[2].plot(history['time'], history['grf'], color='orange')
axs[2].set_title("Ground Reaction Force (N-scaled)")
axs[2].set_xlabel("Time (s)")
axs[2].grid(True)

plt.tight_layout()
plt.savefig("analysis_plots.png")
plt.close()

print("🏁 모든 작업 완료. 'analysis_plots.png'를 확인하세요.")
