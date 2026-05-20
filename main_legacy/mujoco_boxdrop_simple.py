import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R

# ==========================================
# 1. 물리 파라미터 및 모델 설정
# ==========================================
L, W, H = 1200.0, 800.0, 100.0  # mm (길이 x 폭 x 높이)
MASS = 30.0  # kg
G_ACC = 9806.0  # mm/s^2
DT = 0.002  # 2ms (안정성을 위해 증가)

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
  <option timestep="{DT}" gravity="0 0 -{G_ACC}" density="1.225" viscosity="0.00001815">
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
      <geom name="box_geom" type="box" size="{L/2} {W/2} {H/2}" mass="{MASS}" rgba="0.1 0.5 0.8 1" 
            solref="0.01 1" solimp="0.95 0.99 0.001"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# ==========================================
# 2. 초기 상태 설정 (ISTA 6A Corner Drop)
# ==========================================
rot = R.from_euler('xyz', [34, 22, 15], degrees=True)
quat = rot.as_quat()  # [x, y, z, w]

# 정밀 초기 고도 계산 (Z축 기준)
rotated_corners = corners_local @ rot.as_matrix().T
min_z = np.min(rotated_corners[:, 2])
initial_center_z = 500.0 - min_z  # 최저점이 Z=500mm가 되도록

# MuJoCo는 quat [w, x, y, z] 순서
data.qpos[0:3] = [0, 0, initial_center_z]  # X, Y, Z
data.qpos[3:7] = [quat[3], quat[0], quat[1], quat[2]]

print(f"🚀 초기 중심 고도: {initial_center_z:.1f} mm (최저점 Z=500 mm 기준)")
print(f"   회전: Roll={34}°, Pitch={22}°, Yaw={15}° (ISTA 6A Corner Drop)")
print(f"   공기 저항: Enabled (ρ=1.225 kg/m³, ν=1.48×10⁻⁵ m²/s)")
print("\n" + "="*70)
print("🎮 [MuJoCo Viewer 조작법]")
print("  - 마우스 좌클릭 & 드래그: 회전")
print("  - 마우스 우클릭 & 드래그: 이동 (Pan)")
print("  - 마우스 휠: 확대/축소")
print("  - 창 우측 상단 UI 버튼으로 Pause/Run 제어")
print("  - 더블클릭: 자동 시점 조정")
print("  - ESC: 종료")
print("="*70 + "\n")
print("💡 좌표계: X(좌우-Red), Y(앞뒤-Green), Z(위아래-Blue)")
print("🎬 뷰어가 열립니다. 마우스로 자유롭게 회전하며 관찰하세요!\n")

# ==========================================
# 3. 인터랙티브 시뮬레이션
# ==========================================
# launch() 사용 - 가장 간단하고 안정적!
mujoco.viewer.launch(model, data)

print("\n✅ 시뮬레이션 종료.")
