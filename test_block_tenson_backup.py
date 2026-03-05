import os
import numpy as np


# =====================================================================
# [0] 파일 저장 경로 및 파일명 설정 (File Path Configuration)
# =====================================================================
# 생성된 XML 모델 파일들이 저장될 절대 또는 상대 경로
base_path = r'D:\PythonCodeStudy\mujoco-3.5.0-windows-x86_64\model'

MAIN_XML_FILE = 'main.xml'
BLOCKS_XML_FILE = 'blocks.xml'
TENDONS_XML_FILE = 'tendons.xml'

# os.path.join을 사용하여 운영체제에 맞는 안전한 경로 생성
main_xml_path = os.path.join(base_path, MAIN_XML_FILE)
blocks_xml_path = os.path.join(base_path, BLOCKS_XML_FILE)
tendons_xml_path = os.path.join(base_path, TENDONS_XML_FILE)

# 지정한 폴더가 존재하지 않으면 자동으로 폴더 트리 생성
if not os.path.exists(base_path):
    os.makedirs(base_path)
    print(f">>> 지정된 폴더가 없어 새로 생성했습니다: {base_path}")

# =====================================================================
# [1] 형상, 배치 및 질량 파라미터 (Geometry & Mass Configuration)
# =====================================================================

# 1-1. 그리드 분할 및 쿠션 외곽 크기
N, M, L = 5, 4, 3                 # X(가로), Y(세로), Z(두께) 축의 블록 분할 개수.
CUSHION_OUTER = [2.0, 1.4, 0.25]  # 패키지 외곽 전체 크기 (m). [가로 2m, 세로 1.4m, 두께 25cm]
DROP_HEIGHT = 0.5                 # ISTA 6A 코너 낙하 규격: 지면에서 하단 코너 끝까지의 거리 (0.5m)

# 1-2. TV 어셈블리 형상 설정 (쿠션 내부에 들어갈 내용물)
TV_RATIO = 0.7                    # 쿠션 단면적 대비 TV가 차지하는 비율 (70%). 남은 30%가 테두리 완충 영역.
THICK_DISP = 0.01                 # 디스플레이 패널 두께 (0.01m = 10mm).
THICK_CHAS = 0.08                 # 뒷면 섀시 두께 (0.08m = 80mm).
GAP = 0.02                        # ★ 디스플레이와 섀시 사이의 초기 이격 거리 (0.02m = 20mm).
                                  # 낙하 충격 시 디스플레이가 섀시 쪽으로 휘어지며 이 공간을 침범함.
                                  
TV_DIM = [CUSHION_OUTER[0] * TV_RATIO, CUSHION_OUTER[1] * TV_RATIO] # TV의 실제 X, Y 크기
TV_TOTAL_THICK = THICK_DISP + GAP + THICK_CHAS                      # 쿠션에서 파내야 할(Carved-out) 총 Z축 두께

# 1-3. 질량 및 점하중(Lumped Mass) 설정
MASS_CUSHION_TOTAL = 1.2          # ★ 쿠션 패키지 전체의 목표 질량 (kg). (EPS 스티로폼 가정)
MASS_DISP_TOTAL = 3.0             # 디스플레이 패널의 전체 질량 (kg).
MASS_CHAS_BASE = 3.0              # 섀시 뼈대 및 껍데기의 기본 질량 (kg).

# 섀시와 디스플레이는 N x M 개수로 쪼개지므로 1개 블록당 가질 균등 질량 산출
mass_per_disp = MASS_DISP_TOTAL / (N * M)       
mass_per_chas_base = MASS_CHAS_BASE / (N * M)   

# [점하중 부품 목록] 무게중심(CoG) 편향 및 국부 응력 집중을 모사하기 위한 추가 질량체
# 섀시 블록의 (i, j) 인덱스 위치에 부착되어 실제 제품처럼 무거운 곳이 더 강한 관성을 받도록 함.
COMPONENTS = [
    {"name": "power_board", "mass": 1.5, "i": 2, "j": 1, "color": "0.8 0.2 0.2 1", "size": [0.1, 0.08, 0.02]}, # 파워보드 (무거움)
    {"name": "main_board",  "mass": 0.8, "i": 2, "j": 2, "color": "0.2 0.8 0.2 1", "size": [0.15, 0.1, 0.015]},# 메인보드
    {"name": "speaker_L",   "mass": 0.5, "i": 1, "j": 0, "color": "0.2 0.2 0.8 1", "size": [0.05, 0.05, 0.04]}, # 좌측 스피커
    {"name": "speaker_R",   "mass": 0.5, "i": 3, "j": 0, "color": "0.2 0.2 0.8 1", "size": [0.05, 0.05, 0.04]}  # 우측 스피커
]

# =====================================================================
# [2] 물리 파라미터 상세 (Physics & Solver Settings)
# =====================================================================

# 2-1. 텐던(Tendon) 물성 : 블록 간의 탄성 연결부 (스프링-댐퍼 모델)
TENDON_CUSHION = {
    "stiffness": 2000.0,  # 강성(k): 쿠션의 단단함 정도. 충격을 받았을 때 형태를 유지하려는 저항력.
    "damping": 50.0,      # 감쇠(c): 진동 에너지를 흡수하여 출렁임을 멈추게 하는 속도 저항.
    "frictionloss": 5.0   # 마찰 손실: 폼 내부의 미세 파괴로 인한 비가역적 에너지 소산 (먹먹한 재질감).
}
TENDON_TV = {
    "stiffness": 8000.0,  # 강성: 유리나 금속 프레임이므로 쿠션보다 변형 저항(강성)이 훨씬 높아야 함.
    "damping": 150.0,     # 감쇠: 고체 구조물의 빠른 진동 억제.
    "frictionloss": 1.0   # 마찰 손실: 탄성 충돌에 가까우므로 에너지 소실이 적음.
}

# 2-2. 솔버 파라미터 : 변형 시의 비선형적 반발력과 복원 속도 제어
SOLREF_DICT = {
    "timeconst": 0.02,    # 제약 조건 복원 시간 상수. 작을수록 단단하고 빠른 복원 반응을 보임.
    "dampratio": 1.0      # 감쇠비(1.0 = 임계감쇠). 진동(Overshoot) 없이 목표 위치로 깔끔하게 수렴.
}
SOLIMP_DICT = {
    "dmin": 0.9,          # 압축 초기 단계에서 버티는 부드러운 힘의 비율 (0~1).
    "dmax": 0.95,         # 극한 압축(Bottom-out) 시 버티는 강력한 힘의 비율.
    "width": 0.001,       # dmin에서 dmax로 힘이 급변하는 물리적 여유 거리.
    "midpoint": 0.5,      # 저항이 급증하기 시작하는 중간 타이밍.
    "power": 2            # 곡선의 가파름. 2 이상이면 한계점에서 벽에 부딪히듯 급격히 딱딱해짐.
}

# 2-3. 테두리 접착용 Soft Weld : 디스플레이와 섀시 테두리를 묶는 양면 테이프 모사
WELD_SOLREF = {
    "timeconst": 0.04,    # 쿠션(0.02)보다 높게 설정하여 테이프가 고무줄처럼 끈적하게 늘어나는 것을 허용.
    "dampratio": 1.2      # 과감쇠(Over-damped) 상태. 늘어난 테이프가 튕기지 않고 서서히 원래대로 돌아감.
}
WELD_SOLIMP = {
    "dmin": 0.9, "dmax": 0.95, "width": 0.001, "midpoint": 0.5, "power": 2
}


# =====================================================================
# [2] 물리 파라미터 (시각적 변형 확인을 위한 약한 강성 설정)
# =====================================================================

# 2-1. 텐던(Tendon) 물성 : 기존보다 1/10 수준으로 약화
TENDON_CUSHION = {
    "stiffness": 200.0,   # (기존 2000 -> 200) 매우 말랑말랑한 스펀지 수준
    "damping": 10.0,      # 감쇠도 함께 낮춰서 출렁거림이 눈에 보이게 함
    "frictionloss": 0.5   # 내부 마찰을 줄여 에너지가 활발하게 전달되도록 함
}
TENDON_TV = {
    "stiffness": 500.0,   # (기존 8000 -> 500) 패널이 눈에 띄게 휘어지도록 대폭 낮춤
    "damping": 20.0,      
    "frictionloss": 0.1   
}

# 2-2. 솔버 파라미터 : 복원 속도를 늦춰 '천천히' 찌그러지게 함
SOLREF_DICT = {
    "timeconst": 0.15,    # (기존 0.02 -> 0.15) 복원 시간을 7배 늘려 변형 과정을 느릿하게 관찰 가능
    "dampratio": 0.8      # (기존 1.0 -> 0.8) 약간의 진동(Overshoot)을 허용하여 출렁임을 강조
}
SOLIMP_DICT = {
    "dmin": 0.5,          # (기존 0.9 -> 0.5) 초기 저항을 낮춰 쉽게 눌리게 함
    "dmax": 0.8,          # 최대 저항도 낮춤
    "width": 0.01,        # 전이 구간을 넓혀 부드러운 변형 유도
    "midpoint": 0.5,      
    "power": 2            
}

# 2-3. 테두리 접착용 Soft Weld : 테두리 접착제가 껌처럼 늘어나도록 함
WELD_SOLREF = {
    "timeconst": 0.2,     # (기존 0.04 -> 0.2) 접착부가 아주 천천히 복원됨 (많이 늘어남)
    "dampratio": 0.5      # 테두리가 덜렁거리는 효과를 줄 수 있음
}
WELD_SOLIMP = {
    "dmin": 0.5, "dmax": 0.8, "width": 0.01, "midpoint": 0.5, "power": 2
}

def get_xml(d): return " ".join(map(str, d.values()))

# =====================================================================
# ★ [사전 연산] TV 공간을 제외한 실제 쿠션 블록 개수 산출 및 질량 분배
# =====================================================================
cs_dx, cs_dy, cs_dz = CUSHION_OUTER[0]/N/2, CUSHION_OUTER[1]/M/2, CUSHION_OUTER[2]/L/2

num_cushion_blocks = 0
for k in range(L):
    for i in range(N):
        for j in range(M):
            px, py, pz = -CUSHION_OUTER[0]/2 + cs_dx + i*(CUSHION_OUTER[0]/N), -CUSHION_OUTER[1]/2 + cs_dy + j*(CUSHION_OUTER[1]/M), -CUSHION_OUTER[2]/2 + cs_dz + k*(CUSHION_OUTER[2]/L)
            
            # 중앙 TV 체적 영역과 겹치는지 판단 (Z축은 총 두께, X/Y축은 평면 크기 반영)
            is_overlap_z = abs(pz) < (TV_TOTAL_THICK/2 + cs_dz)
            is_overlap_xy = abs(px) < (TV_DIM[0]/2) and abs(py) < (TV_DIM[1]/2)
            
            # 겹치지 않는 부분만 실제 생성될 쿠션 블록으로 카운트
            if not (is_overlap_xy and is_overlap_z):
                num_cushion_blocks += 1

# 쿠션 전체 질량을 생성되는 블록 개수로 나누어 단위 블록당 정확한 질량 산출
mass_per_cush = MASS_CUSHION_TOTAL / num_cushion_blocks if num_cushion_blocks > 0 else 0.001
print(f">>> [계산 완료] 생성 예정 쿠션 블록 수: {num_cushion_blocks}개 (블록 1개당 질량: {mass_per_cush:.4f}kg)")


# =====================================================================
# [3] 블록, 점하중, 센서 사이트 생성 (blocks.xml & tendons.xml)
# =====================================================================
def is_border(i, j, n, m): return i == 0 or i == n-1 or j == 0 or j == m-1

tv_dx, tv_dy = TV_DIM[0]/N/2, TV_DIM[1]/M/2

with open(blocks_xml_path, "w", encoding="utf-8") as fb, open(tendons_xml_path, "w", encoding="utf-8") as ft:
    fb.write("<mujoco>\n")
    ft.write("<mujoco>\n  <tendon>\n")
    
    # Z축 기준: TV 전체 두께를 기준으로 상단은 디스플레이, 하단은 섀시 위치 배정
    z_chas = -TV_TOTAL_THICK/2 + THICK_CHAS/2
    z_disp = TV_TOTAL_THICK/2 - THICK_DISP/2
    
    JOINT_XML = """    <joint type="slide" axis="1 0 0"/>
    <joint type="slide" axis="0 1 0"/>
    <joint type="slide" axis="0 0 1"/>
    <joint type="ball"/>\n"""

    # --- A. 디스플레이 생성 ---
    for i in range(N):
        for j in range(M):
            px, py = -TV_DIM[0]/2 + tv_dx + i*(TV_DIM[0]/N), -TV_DIM[1]/2 + tv_dy + j*(TV_DIM[1]/M)
            fb.write(f'  <body name="b_disp_{i}_{j}" pos="{px:.4f} {py:.4f} {z_disp:.4f}">\n')
            fb.write(f'    <geom type="box" size="{tv_dx:.4f} {tv_dy:.4f} {THICK_DISP/2:.4f}" rgba="0.1 0.5 0.8 0.9" mass="{mass_per_disp:.4f}" friction="0.3"/>\n')
            fb.write(f'    <site name="s_disp_{i}_{j}" pos="0 0 0" size="0.005"/>\n')
            fb.write(JOINT_XML)  # <--- 이 줄을 디스플레이, 섀시, 쿠션 body 바로 아래에 모두 추가!
            fb.write(f'  </body>\n')
            
            for ni, nj, dist in [(i+1, j, TV_DIM[0]/N), (i, j+1, TV_DIM[1]/M)]:
                if ni < N and nj < M:
                    ft.write(f'    <spatial name="t_disp_{i}{j}_{ni}{nj}" stiffness="{TENDON_TV["stiffness"]}" damping="{TENDON_TV["damping"]}" frictionloss="{TENDON_TV["frictionloss"]}" solreflimit="{get_xml(SOLREF_DICT)}" solimplimit="{get_xml(SOLIMP_DICT)}" limited="true" range="{dist*0.8:.3f} {dist*1.2:.3f}">\n')
                    ft.write(f'      <site site="s_disp_{i}_{j}"/> <site site="s_disp_{ni}_{nj}"/>\n')
                    ft.write(f'    </spatial>\n')

    # --- B. 섀시 및 점하중(Lumped Mass) + 타격 감지 센서 생성 ---
    for i in range(N):
        for j in range(M):
            px, py = -TV_DIM[0]/2 + tv_dx + i*(TV_DIM[0]/N), -TV_DIM[1]/2 + tv_dy + j*(TV_DIM[1]/M)
            fb.write(f'  <body name="b_chas_{i}_{j}" pos="{px:.4f} {py:.4f} {z_chas:.4f}">\n')
            fb.write(f'    <geom name="g_chas_{i}_{j}" type="box" size="{tv_dx:.4f} {tv_dy:.4f} {THICK_CHAS/2:.4f}" rgba="0.3 0.3 0.3 0.9" mass="{mass_per_chas_base:.4f}" friction="0.3"/>\n')
            fb.write(JOINT_XML)  # <--- 이 줄을 디스플레이, 섀시, 쿠션 body 바로 아래에 모두 추가!
            # 1. 점하중(부품) 부착: 해당 그리드(i, j)에 부품이 있으면 추가 Geom 생성
            for comp in COMPONENTS:
                if comp["i"] == i and comp["j"] == j:
                    comp_z_offset = -THICK_CHAS/2 - comp["size"][2] # 섀시 뒷면에 튀어나오게 배치
                    fb.write(f'    <geom name="g_comp_{comp["name"]}" type="box" size="{comp["size"][0]} {comp["size"][1]} {comp["size"][2]}" pos="0 0 {comp_z_offset:.4f}" rgba="{comp["color"]}" mass="{comp["mass"]:.4f}"/>\n')
            
            # 2. 터치 센서 볼륨(노란색) 부착: 모서리 4군데 블록에만 충격 감지 영역을 섀시보다 2% 크게 생성
            if (i == 0 and j == 0) or (i == N-1 and j == 0) or (i == 0 and j == M-1) or (i == N-1 and j == M-1):
                fb.write(f'    <site name="touch_site_chas_{i}_{j}" type="box" size="{tv_dx*1.02:.4f} {tv_dy*1.02:.4f} {THICK_CHAS/2*1.02:.4f}" rgba="1 1 0 0.3"/>\n')
                
            fb.write(f'    <site name="s_chas_{i}_{j}" pos="0 0 0" size="0.005"/>\n')
            fb.write(f'  </body>\n')
            
            for ni, nj, dist in [(i+1, j, TV_DIM[0]/N), (i, j+1, TV_DIM[1]/M)]:
                if ni < N and nj < M:
                    ft.write(f'    <spatial name="t_chas_{i}{j}_{ni}{nj}" stiffness="{TENDON_TV["stiffness"]}" damping="{TENDON_TV["damping"]}" frictionloss="{TENDON_TV["frictionloss"]}" solreflimit="{get_xml(SOLREF_DICT)}" solimplimit="{get_xml(SOLIMP_DICT)}" limited="true" range="{dist*0.8:.3f} {dist*1.2:.3f}">\n')
                    ft.write(f'      <site site="s_chas_{i}_{j}"/> <site site="s_chas_{ni}_{nj}"/>\n')
                    ft.write(f'    </spatial>\n')

    # --- C. 쿠션 블록 생성 루프 수정 ---
    for k in range(L):
        for i in range(N):
            for j in range(M):
                px, py, pz = -CUSHION_OUTER[0]/2 + cs_dx + i*(CUSHION_OUTER[0]/N), -CUSHION_OUTER[1]/2 + cs_dy + j*(CUSHION_OUTER[1]/M), -CUSHION_OUTER[2]/2 + cs_dz + k*(CUSHION_OUTER[2]/L)
                if not (abs(pz) < (TV_TOTAL_THICK/2 + cs_dz) and abs(px) < (TV_DIM[0]/2) and abs(py) < (TV_DIM[1]/2)):
                    wx, wy, wz = get_world_pose(px, py, pz)
                    fb.write(f'  <body name="b_cush_{i}_{j}_{k}" pos="{wx:.4f} {wy:.4f} {wz:.4f}" axisangle="{axisangle_str}">\n')
                    fb.write(f'    <freejoint/>\n')
                    
                    # [핵심 수정] contype="2" conaffinity="1" 설정
                    # - contype="2": 이 물체는 2번 그룹에 속함
                    # - conaffinity="1": 이 물체는 1번 그룹(바닥/TV)과만 부딪힘
                    # 결과적으로 자기들끼리(2번 그룹)는 부딪히지 않고 통과하여 연산량 급감!
                    fb.write(f'    <geom type="box" size="{cs_dx*0.98:.4f} {cs_dy*0.98:.4f} {cs_dz*0.98:.4f}" ')
                    fb.write(f'rgba="0.8 0.8 0.8 0.4" mass="{mass_per_cush:.4f}" ')
                    fb.write(f'contype="2" conaffinity="1"/>\n') # ★ 이 부분이 속도 향상의 핵심
                    
                    fb.write(f'    <site name="s_cush_{i}_{j}_{k}" pos="0 0 0" size="0.01"/>\n')
                    fb.write(f'  </body>\n')
                        
                    # 텐던 역시 TV 공간을 가로지르지 않도록 인접 블록의 위치를 검사한 후 연결
                    for ni, nj, nk, dist in [(i+1, j, k, CUSHION_OUTER[0]/N), (i, j+1, k, CUSHION_OUTER[1]/M), (i, j, k+1, CUSHION_OUTER[2]/L)]:
                        if ni < N and nj < M and nk < L:
                            n_px, n_py, n_pz = -CUSHION_OUTER[0]/2 + cs_dx + ni*(CUSHION_OUTER[0]/N), -CUSHION_OUTER[1]/2 + cs_dy + nj*(CUSHION_OUTER[1]/M), -CUSHION_OUTER[2]/2 + cs_dz + nk*(CUSHION_OUTER[2]/L)
                            if not (abs(n_pz) < (TV_TOTAL_THICK/2 + cs_dz) and abs(n_px) < (TV_DIM[0]/2) and abs(n_py) < (TV_DIM[1]/2)):
                                ft.write(f'    <spatial name="t_cush_{i}{j}{k}_{ni}{nj}{nk}" stiffness="{TENDON_CUSHION["stiffness"]}" damping="{TENDON_CUSHION["damping"]}" frictionloss="{TENDON_CUSHION["frictionloss"]}" solreflimit="{get_xml(SOLREF_DICT)}" solimplimit="{get_xml(SOLIMP_DICT)}" limited="true" range="{dist*0.7:.3f} {dist*1.3:.3f}">\n')
                                ft.write(f'      <site site="s_cush_{i}_{j}_{k}"/> <site site="s_cush_{ni}_{nj}_{nk}"/>\n')
                                ft.write(f'    </spatial>\n')

    fb.write("</mujoco>\n")
    ft.write("  </tendon>\n</mujoco>\n")

# =====================================================================
# [4] ISTA 6A 엄밀 코너 각도 계산 및 메인 환경 생성 (main.xml)
# =====================================================================
# 박스의 무게중심(CG)에서 하단 낙하 코너로 향하는 대각선 벡터를 지면에 완벽히 수직이 되도록 정렬
corner_vec = np.array([CUSHION_OUTER[0]/2, CUSHION_OUTER[1]/2, -CUSHION_OUTER[2]/2])
corner_dist = np.linalg.norm(corner_vec)
target_vec = np.array([0, 0, -corner_dist])
rot_axis = np.cross(corner_vec, target_vec)
rot_axis /= np.linalg.norm(rot_axis) # 회전축을 단위 벡터로 정규화
angle_deg = np.degrees(np.arccos(np.dot(corner_vec, target_vec) / (corner_dist**2))) # 회전 각도 계산
z_start = DROP_HEIGHT + corner_dist  # 바닥(Z=0)과 코너 끝단 사이가 정확히 0.5m가 되도록 중심점 고도 상승 설정

main_content = f"""<mujoco model="tv_package_full_flex">
    <compiler autolimits="true" angle="degree"/>
    <option timestep="0.001" gravity="0 0 -9.81">
        <flag contact="enable" energy="enable"/>
    </option>
    
    <visual>
        <map znear="0.01"/>
    </visual>
    
    <worldbody>
        <light pos="0 0 5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
        <geom name="floor" type="plane" size="5 5 0.1" rgba="0.9 0.9 0.9 1" friction="0.8" contype="1" conaffinity="1"/>
        
        <body name="package_assembly" pos="0 0 {z_start:.4f}" axisangle="{rot_axis[0]:.4f} {rot_axis[1]:.4f} {rot_axis[2]:.4f} {angle_deg:.4f}">
            <include file="{BLOCKS_XML_FILE}"/>
        </body>
    </worldbody>
    
    <include file="{TENDONS_XML_FILE}"/>
    
    <equality>
"""
# 중앙부는 출렁일 수 있도록 놔두고, 테두리 블록(is_border)들끼리만 Soft Weld로 결합
for i in range(N):
    for j in range(M):
        if is_border(i, j, N, M):
            main_content += f'        <weld body1="b_disp_{i}_{j}" body2="b_chas_{i}_{j}" solref="{get_xml(WELD_SOLREF)}" solimp="{get_xml(WELD_SOLIMP)}"/>\n'

main_content += f"""    </equality>
    
    <sensor>
        <framepos name="gap_center" objtype="site" objname="s_disp_{N//2}_{M//2}" reftype="site" refname="s_chas_{N//2}_{M//2}"/>
        
        <touch name="force_corner_0_0" site="touch_site_chas_0_0"/>
        <touch name="force_corner_{N-1}_0" site="touch_site_chas_{N-1}_0"/>
        <touch name="force_corner_0_{M-1}" site="touch_site_chas_0_{M-1}"/>
        <touch name="force_corner_{N-1}_{M-1}" site="touch_site_chas_{N-1}_{M-1}"/>
    </sensor>
</mujoco>
"""

# 최종 main.xml 파일 저장
with open(main_xml_path, "w", encoding="utf-8") as f:
    f.write(main_content)

print(f">>> [저장 완료] 다음 위치에 XML 파일들이 성공적으로 생성되었습니다:\n    - {main_xml_path}\n    - {blocks_xml_path}\n    - {tendons_xml_path}")
