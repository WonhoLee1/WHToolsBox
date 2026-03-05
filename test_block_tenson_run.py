import os
import mujoco
import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# [1] 시뮬레이션 환경 설정 및 모델 로드
# =====================================================================
base_path = r'D:\PythonCodeStudy\mujoco-3.5.0-windows-x86_64\model'
xml_path = os.path.join(base_path, 'main.xml')

# XML 모델을 파싱하여 MuJoCo 구조체로 변환
try:
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
except Exception as e:
    print(f"모델 로드 실패! 경로를 확인하세요: {xml_path}\n에러: {e}")
    exit()

# N, M 분할 개수 (앞선 생성 코드와 동일해야 함)
N, M = 5, 4
INITIAL_GAP = 0.02 # 초기 설계 간격 20mm

# 센서 ID 추출 (엔진 내부에서 센서 데이터를 빠르게 찾기 위함)
try:
    sensor_gap_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "gap_center")
    # 4개 코너의 터치 센서 ID
    sensor_f_0_0 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "force_corner_0_0")
    sensor_f_N_0 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"force_corner_{N-1}_0")
    sensor_f_0_M = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"force_corner_0_{M-1}")
    sensor_f_N_M = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"force_corner_{N-1}_{M-1}")
except KeyError:
    print("센서 ID를 찾을 수 없습니다. main.xml의 센서 이름을 확인하세요.")
    exit()

# =====================================================================
# [2] 시뮬레이션 루프 실행 및 데이터 수집
# =====================================================================
# =====================================================================
# [2] 시뮬레이션 루프 실행 및 데이터 수집 (수정됨)
# =====================================================================
time_history = []
gap_history = []
force_0_0_history = []
force_N_0_history = []
force_0_M_history = []
force_N_M_history = []

SIMULATION_TIME = 0.8  

print(">>> 코너 낙하 시뮬레이션을 연산 중입니다...")

while data.time < SIMULATION_TIME:
    mujoco.mj_step(model, data)
    time_history.append(data.time)
    
    # 1. 중앙 간격(Gap) 데이터 추출: framepos는 3차원 벡터를 반환하므로 Norm을 계산
    current_gap_vec = data.sensor("gap_center").data.copy()
    gap_dist = np.linalg.norm(current_gap_vec)
    gap_history.append(gap_dist)
    
    # 2. 모서리 타격 하중(Impact Force) 데이터 추출: 터치 센서는 스칼라 값(N) 반환
    force_0_0_history.append(data.sensor("force_corner_0_0").data[0])
    force_N_0_history.append(data.sensor(f"force_corner_{N-1}_0").data[0])
    force_0_M_history.append(data.sensor(f"force_corner_0_{M-1}").data[0])
    force_N_M_history.append(data.sensor(f"force_corner_{N-1}_{M-1}").data[0])

print(">>> 연산 완료! 해석 그래프를 생성합니다.")

# =====================================================================
# [3] Matplotlib을 이용한 결과 시각화
# =====================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# --- 그래프 1: 디스플레이와 섀시 간의 중심 간격(Gap) 변화 ---
ax1.plot(time_history, gap_history, label="Center Gap Distance", color="#1f77b4", linewidth=2)
ax1.axhline(y=INITIAL_GAP, color='green', linestyle='--', label=f"Initial Gap ({INITIAL_GAP*1000}mm)")
ax1.axhline(y=0.0, color='red', linestyle='-', linewidth=2, label="Collision Danger (0mm)")

ax1.set_title("Display-Chassis Center Gap Variance (Flex/Bottom-out Check)", fontsize=14, fontweight='bold')
ax1.set_ylabel("Distance (m)", fontsize=12)
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.legend(loc="upper right")

# --- 그래프 2: 섀시 코너부의 쿠션 타격 하중(Impact Force) ---
ax2.plot(time_history, force_0_0_history, label="Corner (0, 0) - Drop Corner", color="red", linewidth=2)
ax2.plot(time_history, force_N_0_history, label=f"Corner ({N-1}, 0)", color="orange", alpha=0.7)
ax2.plot(time_history, force_0_M_history, label=f"Corner (0, {M-1})", color="blue", alpha=0.7)
ax2.plot(time_history, force_N_M_history, label=f"Corner ({N-1}, {M-1})", color="purple", alpha=0.7)

ax2.set_title("Chassis Corner Impact Force (Touch Sensor)", fontsize=14, fontweight='bold')
ax2.set_xlabel("Time (s)", fontsize=12)
ax2.set_ylabel("Force (N)", fontsize=12)
ax2.grid(True, linestyle=':', alpha=0.7)
ax2.legend(loc="upper right")

plt.tight_layout()
plt.show()