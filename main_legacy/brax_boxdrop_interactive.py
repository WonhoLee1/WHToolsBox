import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R
import time

# ==========================================
# 1. 물리 파라미터 및 모델 설정
# ==========================================
L, W, H = 1200.0, 800.0, 100.0  # mm (길이 x 폭 x 높이)
MASS = 30.0  # kg
G_ACC = 9806.0  # mm/s^2
DT = 0.001  # 1ms

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
    <texture name="grid" type="2d" builtin="checker" rgb1=".8 .8 .8" rgb2=".9 .9 .9" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="8 8" texuniform="true"/>
  </asset>
  <option timestep="{DT}" gravity="0 0 -{G_ACC}">
    <flag contact="enable"/>
  </option>
  <worldbody>
    <light pos="0 0 5000" dir="0 0 -1" diffuse="0.7 0.7 0.7"/>
    <light pos="3000 3000 3000" dir="-1 -1 -1" diffuse="0.5 0.5 0.5"/>

    <!-- 바닥: XY 평면 (Z=0) -->
    <geom name="floor" type="plane" pos="0 0 0" zaxis="0 0 1" size="3000 3000 1" material="grid" friction="0.8" solref="0.02 1"/>
    
    <!-- 좌표축 시각화: X(Red) Y(Green) Z(Blue) -->
    <site name="origin" pos="0 0 0" size="30" rgba="1 1 1 0.8" type="sphere"/>
    <geom name="axis_x" type="capsule" fromto="0 0 0 500 0 0" size="8" rgba="1 0 0 1" contype="0" conaffinity="0"/> 
    <geom name="axis_y" type="capsule" fromto="0 0 0 0 500 0" size="8" rgba="0 1 0 1" contype="0" conaffinity="0"/> 
    <geom name="axis_z" type="capsule" fromto="0 0 0 0 0 500" size="8" rgba="0 0 1 1" contype="0" conaffinity="0"/> 
    
    <body name="box" pos="0 0 500">
      <freejoint/>
      <geom name="box_geom" type="box" size="{L/2} {W/2} {H/2}" mass="{MASS}" rgba="0.1 0.5 0.8 1" solref="0.01 1" solimp="0.95 0.99 0.001"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# ==========================================
# 2. 초기 상태 설정 (ISTA 6A Corner Drop)
# ==========================================
# ISTA 6A Parcel Corner drop: 34° roll, 22° pitch, 15° yaw
rot = R.from_euler('xyz', [34, 22, 15], degrees=True)
quat = rot.as_quat()  # [x, y, z, w]

# 정밀 초기 고도 계산 (Z축 기준)
rotated_corners = corners_local @ rot.as_matrix().T
min_z = np.min(rotated_corners[:, 2])  # Z 좌표의 최소값
initial_center_z = 300.0 - min_z  # 최저점이 Z=300mm가 되도록

# MuJoCo는 quat [w, x, y, z] 순서
data.qpos[0:3] = [0, 0, initial_center_z]  # X, Y, Z
data.qpos[3:7] = [quat[3], quat[0], quat[1], quat[2]]

print(f"🚀 초기 중심 고도: {initial_center_z:.1f} mm (최저점 Z=300 mm 기준)")
print(f"   회전: Roll={34}°, Pitch={22}°, Yaw={15}° (ISTA 6A Corner Drop)")
print("\n" + "="*70)
print("🎮 [인터랙티브 뷰어 조작법]")
print("  - 마우스 왼쪽 클릭 & 드래그: 회전")
print("  - 마우스 오른쪽 클릭 & 드래그: 이동 (Pan)")
print("  - 마우스 휠: 확대/축소")
print("  - Space: 일시정지/재생")
print("  - Backspace: 초기 상태로 리셋")
print("  - ESC 또는 창 닫기: 종료")
print("="*70 + "\n")
print("💡 좌표계: X(좌우-Red), Y(앞뒤-Green), Z(위아래-Blue)")
print("🎬 3초 후 시뮬레이션이 자동으로 시작됩니다...")

# ==========================================
# 3. Squeeze Film Effect 콜백 함수
# ==========================================
def squeeze_film_force(model, data):
    """매 스텝마다 호출되어 공기 저항력을 계산하고 적용"""
    # 상자의 현재 위치 및 쿼터니언
    pos = data.qpos[0:3]
    quat = data.qpos[3:7]  # [w, x, y, z]
    
    # 쿼터니언을 회전 행렬로 변환
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    mat = np.array([
        [1-2*y**2-2*z**2, 2*x*y-2*w*z, 2*x*z+2*w*y],
        [2*x*y+2*w*z, 1-2*x**2-2*z**2, 2*y*z-2*w*x],
        [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x**2-2*y**2]
    ])
    
    # 8개 코너의 Z 좌표 계산 (높이)
    corners_world_z = pos[2] + corners_local @ mat[2, :]
    min_corner_z = np.min(corners_world_z)
    
    # 지면과의 틈새
    h_gap = max(min_corner_z, 0.1)
    
    # 베르누이 효과 (Z축 속도 기준)
    vel_z = data.qvel[2]
    if vel_z < 0 and h_gap < 150:
        v_escape = abs(vel_z) * ((L*W) / (2*(L+W) * h_gap))
        f_squeeze = 0.5 * 1.225e-9 * (v_escape**2) * (L*W)
        data.qfrc_applied[2] = f_squeeze  # Z축 방향 힘
    else:
        data.qfrc_applied[2] = 0.0

# ==========================================
# 4. 인터랙티브 시뮬레이션
# ==========================================
# MuJoCo 표준 뷰어 실행
with mujoco.viewer.launch_passive(model, data) as viewer:
    # 카메라 초기 설정 (Z-up 좌표계)
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -25
    viewer.cam.distance = 4500
    viewer.cam.lookat[:] = [0, 0, 200]  # XYZ
    
    # 시뮬레이션 루프
    step_count = 0
    freeze_steps = 5000  # 5초간 정지 (5s / 0.001s = 5000 steps)
    
    print("⏸️  5초간 정지 상태입니다. 시점을 조정하세요...")
    print("   (이후 자동으로 시작되며, 상단 UI의 'Pause' 버튼으로 제어 가능)")
    
    while viewer.is_running():
        # 처음 5초는 물리 계산 안 함 (freeze)
        if step_count < freeze_steps:
            # 뷰어만 업데이트 (시점 조정 가능)
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
                viewer.sync()
            time.sleep(0.001)
            step_count += 1
            
            # 카운트다운 표시
            if step_count % 1000 == 0:
                remaining = (freeze_steps - step_count) // 1000
                print(f"   {remaining}초 남음...")
            
            if step_count == freeze_steps:
                print("\n▶️  시작!\n")
        else:
            # 외력 적용
            squeeze_film_force(model, data)
            
            # 물리 스텝
            mujoco.mj_step(model, data)
            
            # 뷰어 동기화 (매 10스텝마다)
            if step_count % 10 == 0:
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
                    viewer.sync()
            
            step_count += 1
            
            # 진행률 출력 (매 1000 스텝마다)
            if step_count % 1000 == 0:
                elapsed_time = (step_count - freeze_steps) * DT / 1000.0
                box_height = data.qpos[2]  # Z 좌표
                print(f"⏱️  시간: {elapsed_time:.2f}s | 상자 높이(Z): {box_height:.1f} mm")

print("\n✅ 시뮬레이션 종료.")
