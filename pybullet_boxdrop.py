import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation as R
import time

# ==========================================
# 1. 물리 파라미터
# ==========================================
L, W, H = 1.2, 0.8, 0.1  # m (PyBullet은 미터 단위)
MASS = 30.0  # kg
G_ACC = 9.806  # m/s^2
DT = 0.001  # 1ms

# 상자 코너 8개 (로컬 좌표계, 미터 단위)
corners_local = np.array([
    [x, y, z]
    for x in [-L/2, L/2]
    for y in [-W/2, W/2]
    for z in [-H/2, H/2]
])

# ==========================================
# 2. PyBullet 초기화
# ==========================================
print("🚀 PyBullet 시뮬레이션 시작\n")

# GUI 모드로 연결
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -G_ACC)
p.setTimeStep(DT)

# 바닥 생성 (XY 평면)
planeId = p.loadURDF("plane.urdf")

# 상자 생성
boxCollisionShape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[L/2, W/2, H/2])
boxVisualShape = p.createVisualShape(p.GEOM_BOX, halfExtents=[L/2, W/2, H/2], 
                                      rgbaColor=[0.1, 0.5, 0.8, 1])

# ISTA 6A Corner Drop 회전
rot = R.from_euler('xyz', [34, 22, 15], degrees=True)
quat = rot.as_quat()  # [x, y, z, w]

# 정밀 초기 고도 계산 (Z축 기준)
rotated_corners = corners_local @ rot.as_matrix().T
min_z = np.min(rotated_corners[:, 2])
initial_center_z = 0.3 - min_z  # 최저점이 Z=0.3m가 되도록

# 상자 생성
boxId = p.createMultiBody(
    baseMass=MASS,
    baseCollisionShapeIndex=boxCollisionShape,
    baseVisualShapeIndex=boxVisualShape,
    basePosition=[0, 0, initial_center_z],
    baseOrientation=quat  # PyBullet도 [x, y, z, w]
)

# 물성 설정
p.changeDynamics(boxId, -1, 
                 restitution=0.3,  # 반발 계수
                 lateralFriction=0.8,
                 spinningFriction=0.05,
                 rollingFriction=0.01)

# 좌표축 시각화
p.addUserDebugLine([0, 0, 0], [0.5, 0, 0], [1, 0, 0], lineWidth=3)  # X: Red
p.addUserDebugLine([0, 0, 0], [0, 0.5, 0], [0, 1, 0], lineWidth=3)  # Y: Green
p.addUserDebugLine([0, 0, 0], [0, 0, 0.5], [0, 0, 1], lineWidth=3)  # Z: Blue

# 카메라 설정
p.resetDebugVisualizerCamera(
    cameraDistance=4.5,
    cameraYaw=135,
    cameraPitch=-25,
    cameraTargetPosition=[0, 0, 0.2]
)

# GUI 설정
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

print(f"🚀 초기 중심 고도: {initial_center_z*1000:.1f} mm (최저점 Z=300 mm 기준)")
print(f"   회전: Roll={34}°, Pitch={22}°, Yaw={15}° (ISTA 6A Corner Drop)")
print("\n" + "="*70)
print("🎮 [PyBullet 뷰어 조작법]")
print("  - 마우스 왼쪽 클릭 & 드래그: 회전")
print("  - 마우스 휠 클릭 & 드래그: 이동 (Pan)")
print("  - 마우스 휠: 확대/축소")
print("  - G: 그리드 온/오프")
print("  - W: 와이어프레임 모드")
print("  - ESC 또는 창 닫기: 종료")
print("="*70 + "\n")
print("💡 좌표계: X(좌우-Red), Y(앞뒤-Green), Z(위아래-Blue)")
print("🎬 시뮬레이션 시작!\n")

# ==========================================
# 3. Squeeze Film Effect
# ==========================================
def apply_squeeze_film(boxId):
    """공기 저항력 계산 및 적용"""
    pos, ori = p.getBasePositionAndOrientation(boxId)
    vel, ang_vel = p.getBaseVelocity(boxId)
    
    # 회전 행렬
    rot_mat = R.from_quat(ori).as_matrix()
    
    # 8개 코너의 Z 좌표
    corners_world = pos + corners_local @ rot_mat.T
    min_corner_z = np.min(corners_world[:, 2])
    
    # 지면과의 틈새
    h_gap = max(min_corner_z, 0.0001)  # m
    
    # 베르누이 효과
    vel_z = vel[2]
    if vel_z < 0 and h_gap < 0.15:  # 150mm
        v_escape = abs(vel_z) * ((L*W) / (2*(L+W) * h_gap))
        f_squeeze = 0.5 * 1.225 * (v_escape**2) * (L*W)
        p.applyExternalForce(boxId, -1, [0, 0, f_squeeze], pos, p.WORLD_FRAME)

# ==========================================
# 4. 시뮬레이션 루프
# ==========================================
step_count = 0
start_time = time.time()

try:
    while True:
        # Squeeze film 적용
        apply_squeeze_film(boxId)
        
        # 물리 스텝
        p.stepSimulation()
        
        step_count += 1
        
        # 진행률 출력 (매 1000 스텝마다)
        if step_count % 1000 == 0:
            elapsed_time = step_count * DT
            pos, _ = p.getBasePositionAndOrientation(boxId)
            box_height = pos[2] * 1000  # mm
            print(f"⏱️  시간: {elapsed_time:.2f}s | 상자 높이(Z): {box_height:.1f} mm")
        
        # 실시간 속도 제어 (선택적)
        # time.sleep(DT)  # 주석 해제하면 실시간 속도로 실행

except KeyboardInterrupt:
    print("\n\n⚠️  사용자 중단 (Ctrl+C)")

finally:
    p.disconnect()
    print("\n✅ 시뮬레이션 종료.")
