import os
import json
import numpy as np
import io
import math

# =====================================================================
# [1] 시스템 설정 및 전역 유틸리티
# =====================================================================
def get_local_pose(vec, drop_height, rot_axis, angle_rad, corner_dist):
    """최상위 루트(BPackagingBox) 구동을 위한 글로벌 포즈 계산용"""
    v = np.array(vec)
    v_rot = v * np.cos(angle_rad) + np.cross(rot_axis, v) * np.sin(angle_rad) + rot_axis * np.dot(rot_axis, v) * (1 - np.cos(angle_rad))
    return v_rot + np.array([0, 0, drop_height + corner_dist])

def calculate_solref(K, C):
    """
    물리적인 강성(Stiffness, K)과 점성 감쇠(Damping, C)를 입력받아
    MuJoCo의 기본 solref 양수 입력방식인 (timeconst, dampratio)로 유추/변환하는 헬퍼 함수입니다.
    이 함수는 시뮬레이션 dt(0.001초 기준)의 2배인 0.002 미만으로 timeconst가 떨어지는 것을
    방지하여, 지나친 강성 부여로 인한 시뮬레이션 폭발(분해)을 방지합니다.
    """
    if K <= 0:
        raise ValueError("강성(Stiffness) K는 0보다 커야 합니다.")
        
    # 부드러울수록(K가 작을수록) timeconst는 커짐 (최소 하한 0.002)
    timeconst = 1.0 / math.sqrt(K)
    
    if timeconst < 0.002:
        timeconst = 0.002
        K_safe = 1.0 / (timeconst**2)
    else:
        K_safe = K
        
    # 임계 감쇠(Critical Damping)를 1.0으로 하는 감쇠비(dampratio) 산출
    dampratio = C / (2.0 * math.sqrt(K_safe))
    
    # 소수점 5자리까지만 정리하여 반환
    return round(timeconst, 5), round(dampratio, 5)

def get_default_config(user_config=None):
    """기본 모델 파라미터 설정을 반환하는 헬퍼 함수. 
    사용자가 전달한 config에 누락된 값이 있으면 이 기본값들이 채워지며,
    내부 연산(예: 박스 크기에 비례한 내부 구조물 크기 산출)도 여기서 처리됩니다."""
    
    if user_config is None:
        user_config = {}
        
    # [시뮬레이션/솔버 파라미터 (sim_*)]
    # MuJoCo의 <option> 태그에 직접 반영될 파라미터들입니다.
    sim_integrator = user_config.get("sim_integrator", "implicitfast")
    sim_timestep   = user_config.get("sim_timestep", 0.001)
    sim_iterations = user_config.get("sim_iterations", 80)
    sim_noslip_iterations = user_config.get("sim_noslip_iterations", 5)
    sim_tolerance  = user_config.get("sim_tolerance", 1e-6)
    sim_impratio   = user_config.get("sim_impratio", 1.0) # 접촉 임피던스 비율
    sim_gravity    = user_config.get("sim_gravity", [0, 0, -9.81])
    sim_nthread    = user_config.get("sim_nthread", 4)  # [NEW] 멀티코어 스레드 수 (CPU 물리 코어 수 권장)
        
    # [솔버 파라미터 분리] 문자열 대신 개별 변수로 관리 (최적화 친화적)    
    cush_solref_stiff = user_config.get("cush_solref_stiff", 0.02) # 쿠션 timeconst
    cush_solref_damp = user_config.get("cush_solref_damp", 1.0) # 쿠션 dampratio
    
    # solimp = [dmin, dmax, width, midpoint, power] 
    cush_solimp_dmin = user_config.get("cush_solimp_dmin", 0.95)   # 쿠션 최소 임피던스
    cush_solimp_dmax = user_config.get("cush_solimp_dmax", 0.999)  # 쿠션 최대 임피던스
    cush_solimp_width = user_config.get("cush_solimp_width", 0.001)  # 쿠션 임피던스 폭
    cush_solimp_mid = user_config.get("cush_solimp_mid", 0.5)        # 쿠션 임피던스 중간점
    cush_solimp_power = user_config.get("cush_solimp_power", 2.0)    # 쿠션 임피던스 거듭제곱
    
    # tape (Cohesive / Frame)
    # 극단적인 충격(285G)에서도 버티도록 강도를 대폭 상향 (timeconst 0.02 -> 0.005)
    tape_solref_stiff = user_config.get("tape_solref_stiff", 0.05) 
    tape_solref_damp = user_config.get("tape_solref_damp", 1.0)

    tape_solimp_dmin = user_config.get("tape_solimp_dmin", 0.9)   # 테이프 최소 임피던스 
    tape_solimp_dmax = user_config.get("tape_solimp_dmax", 0.999) # 테이프 최대 임피던스 (관통 및 분리 방지)
    tape_solimp_width = user_config.get("tape_solimp_width", 0.001)  # 테이프 임피던스 폭
    tape_solimp_mid = user_config.get("tape_solimp_mid", 0.5)        # 테이프 임피던스 중간점
    tape_solimp_power = user_config.get("tape_solimp_power", 2.0)    # 테이프 임피던스 거듭제곱
    
    # cell (TV Panel)
    cell_solref_stiff = user_config.get("cell_solref_stiff", 0.01) # 패널 timeconst
    cell_solref_damp = user_config.get("cell_solref_damp", 1.0) # 패널 dampratio
    
    cell_solimp_dmin = user_config.get("cell_solimp_dmin", 0.9)   # 패널 최소 임피던스
    cell_solimp_dmax = user_config.get("cell_solimp_dmax", 0.999)  # 패널 최대 임피던스 (쿠션 관통 방지)
    cell_solimp_width = user_config.get("cell_solimp_width", 0.001)  # 패널 임피던스 폭
    cell_solimp_mid = user_config.get("cell_solimp_mid", 0.5)        # 패널 임피던스 중간점
    cell_solimp_power = user_config.get("cell_solimp_power", 2.0)    # 패널 임피던스 거듭제곱
    
    # tv (Chassis)
    tv_solref_stiff = user_config.get("tv_solref_stiff", 0.01) # 구조물 timeconst
    tv_solref_damp = user_config.get("tv_solref_damp", 1.0) # 구조물 dampratio
    
    tv_solimp_dmin = user_config.get("tv_solimp_dmin", 0.9)   # 구조물 최소 임피던스
    tv_solimp_dmax = user_config.get("tv_solimp_dmax", 0.95)  # 구조물 최대 임피던스
    tv_solimp_width = user_config.get("tv_solimp_width", 0.001)  # 구조물 임피던스 폭
    tv_solimp_mid = user_config.get("tv_solimp_mid", 0.5)        # 구조물 임피던스 중간점
    tv_solimp_power = user_config.get("tv_solimp_power", 2.0)    # 구조물 임피던스 거듭제곱
    
    # gorund contact    
    # [BUG FIX] 사용자가 cfg["ground_solref_stiff"] 등을 직접 수정했을 때 문자열(ground_solref)에 즉각 반영되도록 유도
    ground_solref_stiff = user_config.get("ground_solref_stiff", 0.01)
    ground_solref_damp = user_config.get("ground_solref_damp", 1.0)
    
    # solref: 오리지널 (timeconst, dampratio) 방식
    cush_solref = f"{cush_solref_stiff} {cush_solref_damp}"
    tape_solref = f"{tape_solref_stiff} {tape_solref_damp}"
    cell_solref = f"{cell_solref_stiff} {cell_solref_damp}"
    tv_solref = f"{tv_solref_stiff} {tv_solref_damp}"
    
    # 만약 사용자가 이미 완성된 문자열 "ground_solref"를 직접 주었고, 
    # 개별 파라미터(stiff, damp)는 고치지 않았다면 기존 문자열 유지
    if "ground_solref" in user_config and "ground_solref_stiff" not in user_config and "ground_solref_damp" not in user_config:
        ground_solref = user_config["ground_solref"]
    else:
        # 개별 파라미터가 하나라도 있으면 새로 조립
        ground_solref = f"{ground_solref_stiff} {ground_solref_damp}"
        print (f"rebuild ground_solref {ground_solref}")
    
    cush_solimp = f"{cush_solimp_dmin} {cush_solimp_dmax} {cush_solimp_width} {cush_solimp_mid} {cush_solimp_power}"
    tape_solimp = f"{tape_solimp_dmin} {tape_solimp_dmax} {tape_solimp_width} {tape_solimp_mid} {tape_solimp_power}"
    cell_solimp = f"{cell_solimp_dmin} {cell_solimp_dmax} {cell_solimp_width} {cell_solimp_mid} {cell_solimp_power}"
    tv_solimp = f"{tv_solimp_dmin} {tv_solimp_dmax} {tv_solimp_width} {tv_solimp_mid} {tv_solimp_power}"
    
    # 기본 치수 설정 및 종속 변수 연산 처리
    box_w = user_config.get("box_w", 2.0)
    box_h = user_config.get("box_h", 1.4)
    box_d = user_config.get("box_d", 0.25)
    box_thick = user_config.get("box_thick", 0.01)
    
    cush_w = box_w - 2 * box_thick
    cush_h = box_h - 2 * box_thick
    cush_d = box_d - 2 * box_thick
    
    assy_w = user_config.get("assy_w", cush_w - 0.3)
    assy_h = user_config.get("assy_h", cush_h - 0.3)
    
    # Calculate approx cushion volume to translate density to mass
    cush_density = user_config.get("cush_density", None)
    mass_cushion = user_config.get("mass_cushion", 1.0)
    if cush_density is not None:
        if user_config.get("unit_size") is not None: # Unit test mode
            us = user_config.get("unit_size")
            mass_cushion = cush_density * (us[0] * us[1] * us[2])
        else: # Full mode
            assy_d = 0.020 + 0.005 + 0.050 # approximate depth inside
            ext_vol = cush_w * cush_h * cush_d
            int_vol = assy_w * assy_h * assy_d
            cush_vol = max(0.01, ext_vol - int_vol)
            mass_cushion = cush_density * cush_vol
    
    config = {
        "drop_mode": "PARCEL",
        "drop_height": 0.5,
        "include_paperbox": True,
        "include_cushion": True,
        "sim_duration": 1.0,  # 헤드리스 시뮬레이션 기본 구동 시간 (초)
        
        # 외부 껍데기 박스 치수
        "box_w": box_w, 
        "box_h": box_h, 
        "box_d": box_d,
        "box_thick": box_thick,
        "cush_gap": 0.001,
        
        # 내부 구조물(Chassis, OpenCell, Tape 등을 포함하는 AssySet) 기본 크기
        # 만약 사용자가 넘기지 않으면, 기본값으로 외부 박스 크기 대비 적정 비율(cush_w - 0.3)로 자동 계산됩니다.
        "assy_w": user_config.get("assy_w", cush_w - 0.3),
        "assy_h": user_config.get("assy_h", cush_h - 0.3),
        
        # 내부 구조물(Assy) 부품별 각 두께
        "oc_d"      : 0.020,        # oc: OpenCell (TV 디스플레이 패널부) 두께
        "occ_d"     : 0.005,      # occ: OpenCell Cohesive (패널 부착용 테이프/액자 프레임) 두께
        "chas_d"    : 0.050,      # chas: Chassis (TV 후면 기구물) 두께
        "occ_ithick": 0.050,  # Cohesive(테이프)가 차지하는 실제 테두리 폭(액자 형태 두께)

        # [NEW] 상세 분할 정보 (Per-body div)
        "box_div": user_config.get("box_div", [5, 4, 3]),
        "cush_div": user_config.get("cush_div", [5, 4, 3]),
        "oc_div": user_config.get("oc_div", [5, 4, 1]),
        "occ_div": user_config.get("occ_div", [5, 4, 1]),
        "chassis_div": user_config.get("chassis_div", [5, 4, 1]),
        "assy_div": user_config.get("assy_div", [5, 4, 1]), # Legacy/Fallback

        # [NEW] 내부 Weld 구속조건 사용 여부 (Zero Weld 모드 선택권)
        # False 설정 시, 해당 부품의 모든 블록이 하나의 바디(Single-Body)에 Geom으로 직접 추가되어 
        # 구속조건(Equality)이 0개가 되며 시뮬레이션 속도가 비약적으로 향상됩니다.
        "box_use_weld": user_config.get("box_use_weld", True),
        "cush_use_weld": user_config.get("cush_use_weld", True),
        "oc_use_weld": user_config.get("oc_use_weld", True),
        "occ_use_weld": user_config.get("occ_use_weld", True),
        "chassis_use_weld": user_config.get("chassis_use_weld", True),

        # 질량 (Mass, kg)
        "mass_paper": 4.0,
        "mass_cushion": mass_cushion,
        "mass_oc": 5.0,       # OpenCell 질량
        "mass_occ": 0.1,     # Cohesive (테이프) 질량
        "mass_chassis": 10.0,  # Chassis 질량

        # 재료 물성치 (솔버 제어 및 물리 특성)
        "cush_solref": cush_solref,
        "tape_solref": tape_solref,
        "cush_solimp": cush_solimp,
        "tape_solimp": tape_solimp,
        "mat_paper": {"rgba": "0.7 0.6 0.4 0.9", "solref": "0.01 1.0", "solimp": "0.8 0.95 0.001 0.5 2", "contype": "1", "conaffinity": "1", "friction": "0.8"}, # Default init
        "mat_cush" : {"rgba": "0.9 0.9 0.9 0.5", "solref": cush_solref, "solimp": cush_solimp, "contype": "1", "conaffinity": "1", "friction": user_config.get("cush_friction", 0.8)},
        "mat_tape" : {"rgba": "1.0 0.1 0.1 0.8", "solref": tape_solref, "solimp": tape_solimp, "contype": "1", "conaffinity": "1", "friction": "0.8"},
        "mat_cell" : {"rgba": "0.1 0.1 0.1 1.0", "solref": cell_solref, "solimp": cell_solimp, "contype": "1", "conaffinity": "1", "friction": "0.8"},
        "mat_tv"   : {"rgba": "0.1 0.5 0.8 1.0", "solref": tv_solref, "solimp": tv_solimp, "contype": "1", "conaffinity": "1", "friction": "0.8"},
        
        # 바닥(Ground) 접촉 물질 정보 (Cushion과 유사하게 처리하기 위해 추가)
        "ground_solref": ground_solref, 
        "ground_solimp": user_config.get("ground_solimp", "0.90 0.99 0.01 0.5 2"),
        "ground_friction": user_config.get("ground_friction", 0.2), # 바닥 마찰계수 기본값 0.2

        # 조명(Lighting) 설정 - 조절이 쉽도록 옵션으로 추출
        "light_head_ambient" : "0.28 0.28 0.28", # 헤드라이트 환경광
        "light_head_diffuse" : "0.56 0.56 0.56", # 헤드라이트 확산광
        "light_main_ambient" : "0.21 0.21 0.21", # 주 조명 환경광 
        "light_main_diffuse" : "0.49 0.49 0.49", # 주 조명 확산광
        "light_sub_diffuse"  : "0.21 0.21 0.21", # 보조 조명 확산광

        # 공기 저항 (Air Resistance) 설정
        "air_density"      : 1.225,    # 공기 밀도
        "air_viscosity"    : 1.81e-5,  # 공기 동점성계수
        "air_cd_drag"      : 1.05,     # Blunt drag 계수
        "air_cd_viscous"   : 0.0,      # Slender(점성) drag 계수
        "air_squeeze_hmin" : 0.001,    # Squeeze Film 최소 높이
        "enable_air_drag"    : True,   # MuJoCo 빌트인 Drag/Viscous 활성화
        "enable_air_squeeze" : True,   # 수동 Squeeze Film 공기 쿠션 활성화
        
        # -----------------------------------------------------------------
        # [NEW] 임의 추가 질량 (CoG/MoI 조정용)
        # -----------------------------------------------------------------
        "chassis_aux_masses": user_config.get("chassis_aux_masses", []),

        # 시뮬레이션 파라미터 재등록
        "sim_integrator": sim_integrator,
        "sim_timestep": sim_timestep,
        "sim_iterations": sim_iterations,
        "sim_noslip_iterations": sim_noslip_iterations,
        "sim_impratio": sim_impratio,
        "sim_tolerance": sim_tolerance,
        "sim_gravity": sim_gravity,
        "sim_nthread": sim_nthread,
    }
    
    # 사용자가 직접 전달한 나머지 설정값들을 최종 병합(덮어쓰기)합니다.
    for k, v in user_config.items():
        config[k] = v
        
    # [FINAL SYNC] 사용자가 직접 config["ground_solref_stiff"] 등을 수정했을 경우를 대비해
    # 모든 문자열 파라미터를 최종 병합된 값 기준으로 한 번 더 재조립합니다.
    g_s = config.get("ground_solref_stiff", ground_solref_stiff)
    g_d = config.get("ground_solref_damp", ground_solref_damp)
    config["ground_solref"] = f"{g_s} {g_d}"
    
    # 쿠션 및 기타 부품들도 동일하게 최종 동기화
    c_s = config.get("cush_solref_stiff", cush_solref_stiff)
    c_d = config.get("cush_solref_damp", cush_solref_damp)
    config["cush_solref"] = f"{c_s} {c_d}"
    if "mat_cush" in config:
        config["mat_cush"]["solref"] = config["cush_solref"]

    return config

# =====================================================================
# [2] 기본 데이터 구조: 이산형 블록(Geom) 정보 클래스
# =====================================================================
class DiscreteBlock:
    def __init__(self, idx, cx, cy, cz, dx, dy, dz, mass, material):
        self.idx = idx         # (i, j, k) 튜플 형태의 지역 인덱스
        self.cx = cx           # 로컬 기준점(원점) 대비 X 좌표
        self.cy = cy
        self.cz = cz
        self.dx = dx           # Half-size X (블록 실제 너비의 절반)
        self.dy = dy
        self.dz = dz
        self.mass = mass       # 체적비례 할당된 질량
        self.material = material # 종이, 완충재, 테이프 등 재질 정보 (dict)
        self.volume = (2*dx) * (2*dy) * (2*dz)

# =====================================================================
# [3] 최상위 계층 기저 클래스 (Base Discrete Body)
# =====================================================================
class BaseDiscreteBody:
    def __init__(self, name, width, height, depth, mass, div, material_props, use_internal_weld=True):
        self.name = name
        self.width = width
        self.height = height
        self.depth = depth
        self.total_mass = mass
        self.div = div  # [Nx, Ny, Nz]
        self.material_props = material_props # 예: {"rgba": "...", "solref": "..."}
        self.use_internal_weld = use_internal_weld # [NEW] 내부 Weld 구속조건 사용 여부 플래그
        
        self.blocks = {} # 딕셔너리로 개별 블록 저장 {(i,j,k): DiscreteBlock}
        self.children = [] # 하위 컴포넌트 목록
        self.parent = None # 상위 컨테이너 참조

    def add_child(self, child_body):
        child_body.parent = self
        self.children.append(child_body)

    def _generate_strict_grid_axis(self, length, num_div, required_cuts=[]):
        """
        필수 절단선(required_cuts)을 반드시 포함하여 길이를 분할하는 배열 반환
        length: 전체 길이 (원점 0을 중심으로 -len/2 ~ +len/2)
        num_div: 기본 등분 개수
        required_cuts: 반드시 지나가야 하는 로컬 좌표 리스트 (예: gap 경계면)
        """
        # 1. 박스 전체 경계 및 필수 절단선 포함
        edges = set([-length/2, length/2])
        for cut in required_cuts:
            if -length/2 <= cut <= length/2:
                edges.add(cut)
        edges = sorted(list(edges))
        
        # 2. 필수 절단선 사이의 구간을 div로 추가 분할 시도
        # 사용자의 요청: "나머지 길이는 기본 분할에 맞춰 진행. 육면체로 분할 할 수 있는 것은 다 분할"
        # 구현: 원래 등분 간격인 (length / num_div) 보다 구간이 길면, 해당 구간을 쪼갠다.
        target_step = length / num_div
        final_nodes = []
        
        for i in range(len(edges)-1):
            start = edges[i]
            end = edges[i+1]
            segment_len = end - start
            
            # 구간 병합 처리 방지: 부동소수점 오류 해결을 위한 미세 라운딩
            start = round(start, 5)
            end = round(end, 5)
            
            if len(final_nodes) == 0:
                final_nodes.append(start)
            elif abs(final_nodes[-1] - start) > 1e-6:
                final_nodes.append(start)
                
            # 구간이 타겟 스텝보다 눈에 띄게 큰 경우 그 안을 다시 등분
            if segment_len > target_step * 1.01:
                # 구간 내 몇 개로 나눌 것인가?
                sub_divs = max(1, int(round(segment_len / target_step)))
                sub_nodes = np.linspace(start, end, sub_divs + 1)
                for n in sub_nodes[1:]:
                    final_nodes.append(round(n, 5))
            else:
                final_nodes.append(round(end, 5))
                
        return sorted(list(set(final_nodes)))

    def is_cavity(self, block_cx, block_cy, block_cz, box_dx, box_dy, box_dz):
        """특정 위치의 블록이 파여야 하는 공간인지 확인. 하위 클래스에서 오버라이딩"""
        return False

    def build_geometry(self, local_offset=[0,0,0], required_cuts_x=[], required_cuts_y=[], required_cuts_z=[]):
        """지정된 크기와 필수 절단 좌표를 기반으로 이산 블록(Geoms) 생성"""
        
        nodes_x = self._generate_strict_grid_axis(self.width, self.div[0], required_cuts_x)
        nodes_y = self._generate_strict_grid_axis(self.height, self.div[1], required_cuts_y)
        nodes_z = self._generate_strict_grid_axis(self.depth, self.div[2], required_cuts_z)
        
        temp_blocks = []
        total_vol = 0.0
        
        for i in range(len(nodes_x)-1):
            for j in range(len(nodes_y)-1):
                for k in range(len(nodes_z)-1):
                    cx = (nodes_x[i] + nodes_x[i+1]) / 2.0
                    cy = (nodes_y[j] + nodes_y[j+1]) / 2.0
                    cz = (nodes_z[k] + nodes_z[k+1]) / 2.0
                    
                    dx = (nodes_x[i+1] - nodes_x[i]) / 2.0
                    dy = (nodes_y[j+1] - nodes_y[j]) / 2.0
                    dz = (nodes_z[k+1] - nodes_z[k]) / 2.0

                    if self.is_cavity(cx, cy, cz, dx, dy, dz):
                        continue # 이 블록은 건너뜀 (내부 빈 공간)
                        
                    blk = DiscreteBlock((i,j,k), cx + local_offset[0], cy + local_offset[1], cz + local_offset[2],
                                        dx, dy, dz, 0, self.material_props)
                    temp_blocks.append(blk)
                    total_vol += blk.volume

        # 질량 분배 (Volumetric Mass Distribution)
        for blk in temp_blocks:
            blk.mass = self.total_mass * (blk.volume / total_vol)
            self.blocks[blk.idx] = blk

    def get_weld_xml_strings(self):
        """부품 내부 인접 블록 간의 Weld 연결 XML 태그 생성 반환"""
        weld_xml = []
        
        # [최적화] use_internal_weld 가 False 면 내부 Weld 를 생성하지 않음 (구속조건 0 모드)
        if not self.use_internal_weld:
            # 자식 컴포넌트들이 개별적으로 Weld 가 필요할 수 있으므로 자식은 재귀 호출
            for child in self.children:
                weld_xml.extend(child.get_weld_xml_strings())
            return weld_xml

        # 내부 블록 간 연결 (i기준 인접, j기준 인접, k기준 인접)
        # 키 목록을 가져와서 인접한 것이 있으면 연결
        solref = self.material_props.get("solref", "0.02 1.0")
        solimp = self.material_props.get("solimp", "0.9 0.95 0.001 0.5 2")
        
        # 3. 인접 블록 수색 및 1방향(단방향) Weld 생성
        # 각 블록(blk1)에 대해 +X, +Y, +Z 방향에 인접한 블록이 있는지 검사하고 연결 (중복 방지)
        block_keys = set(self.blocks.keys())
        for (i, j, k), blk1 in self.blocks.items():
            
            dx, dy, dz = blk1.dx, blk1.dy, blk1.dz # 블록의 반폭들(Half-size)
            
            # +X 방향 인접
            if (i+1, j, k) in block_keys:
                blk2 = self.blocks[(i+1, j, k)]
                if abs((blk1.cx + blk1.dx) - (blk2.cx - blk2.dx)) < 1e-4:
                    site1_name = f"s_{self.name}_{i}_{j}_{k}_PX"
                    site2_name = f"s_{self.name}_{i+1}_{j}_{k}_NX"
                    weld_xml.append(f'        <weld site1="{site1_name}" site2="{site2_name}" solref="{solref}" solimp="{solimp}"/>')
            
            # +Y 방향 인접
            if (i, j+1, k) in block_keys:
                blk2 = self.blocks[(i, j+1, k)]
                if abs((blk1.cy + blk1.dy) - (blk2.cy - blk2.dy)) < 1e-4:
                    site1_name = f"s_{self.name}_{i}_{j}_{k}_PY"
                    site2_name = f"s_{self.name}_{i}_{j+1}_{k}_NY"
                    weld_xml.append(f'        <weld site1="{site1_name}" site2="{site2_name}" solref="{solref}" solimp="{solimp}"/>')
                    
            # +Z 방향 인접
            if (i, j, k+1) in block_keys:
                blk2 = self.blocks[(i, j, k+1)]
                if abs((blk1.cz + blk1.dz) - (blk2.cz - blk2.dz)) < 1e-4:
                    site1_name = f"s_{self.name}_{i}_{j}_{k}_PZ"
                    site2_name = f"s_{self.name}_{i}_{j}_{k+1}_NZ"
                    weld_xml.append(f'        <weld site1="{site1_name}" site2="{site2_name}" solref="{solref}" solimp="{solimp}"/>')
        
        # 자식 컴포넌트들의 weld 태그도 수집
        for child in self.children:
            weld_xml.extend(child.get_weld_xml_strings())
            
        return weld_xml

    def calculate_inertia(self):
        """
        자신과 모든 자식 블록들을 취합하여 전체 질량, 무게중심(CoG), 관성 모멘트(MoI)를 계산합니다.
        평행축 정리(Parallel Axis Theorem)를 적용하여 기준 좌표계에서의 올바른 관성 텐서를 도출합니다.
        
        추가로 리턴 값에 각 개별 부품(Body)들의 독립적인 관성 데이터 목록을 포함하여, 
        리포트 출력 시 하부 부품별 질량 분포를 확인할 수 있게 합니다.

        Returns:
            total_mass (float): 전체 질량 합계 (kg)
            total_cog (np.array): 전체 무게중심 [x, y, z]
            final_total_moi (np.array): 전체 관성 모멘트 합계 [Ixx, Iyy, Izz] (kg*m^2)
            individual_details (list): 각 개별 부품의 정보 딕셔너리 리스트
        """
        # 1. 모든 말단 이산 블록(DiscreteBlock) 정보를 재귀적으로 수집 및 개별 바디 데이터 산출
        all_primitive_blocks = []
        individual_details = []
        
        def _collect_all_blocks_and_calculate_sub_inertias(body):
            """트리를 순회하며 모든 블록을 수집하고, 각 노드(Body)별 관성 특성을 계산합니다."""
            nonlocal all_primitive_blocks, individual_details
            
            this_body_mass = 0.0
            this_body_weighted_cog_sum = np.zeros(3)
            this_body_local_moi_sum = np.zeros(3)
            this_body_blocks_list = []
            
            # (A) 해당 Body가 직접 가지고 있는 블록들을 순회
            for blk in body.blocks.values():
                this_body_mass += blk.mass
                this_body_weighted_cog_sum += blk.mass * np.array([blk.cx, blk.cy, blk.cz])
                
                # 블록 자체의 로컬 MoI (중심 기준)
                # MuJoCo Geoms (Box) 공식: I = (1/12) * mass * (width^2 + height^2)
                # 여기서 dx, dy, dz는 Half-size이므로 실제 변의 길이는 2배입니다.
                w_full, h_full, d_full = 2.0 * blk.dx, 2.0 * blk.dy, 2.0 * blk.dz
                ixx = (1.0/12.0) * blk.mass * (h_full**2 + d_full**2)
                iyy = (1.0/12.0) * blk.mass * (w_full**2 + d_full**2)
                izz = (1.0/12.0) * blk.mass * (w_full**2 + h_full**2)
                
                this_body_local_moi_sum += np.array([ixx, iyy, izz])
                this_body_blocks_list.append(blk)
                all_primitive_blocks.append(blk)
            
            # (B) 해당 Body에 실제 질량이 존재한다면(Geom이 있다면) 리스트에 기록
            # AssySet 같이 자식만 있는 컨테이너는 개별 리스트에서 제외하거나 
            # 필요에 따라 합산 결과를 넣을 수 있습니다. 여기서는 블록 보유 바디만 기록합니다.
            if this_body_mass > 0:
                # 이 바디만의 독자적 무게중심
                this_body_cog = this_body_weighted_cog_sum / this_body_mass
                
                # 평행축 정리 적용 (이 바디의 자체 CoG 기준의 관성 텐서로 보정)
                this_body_parallel_correction = np.zeros(3)
                for blk in this_body_blocks_list:
                    b_pos = np.array([blk.cx, blk.cy, blk.cz])
                    dist_sq_from_body_cog = (b_pos - this_body_cog)**2
                    this_body_parallel_correction[0] += blk.mass * (dist_sq_from_body_cog[1] + dist_sq_from_body_cog[2])
                    this_body_parallel_correction[1] += blk.mass * (dist_sq_from_body_cog[0] + dist_sq_from_body_cog[2])
                    this_body_parallel_correction[2] += blk.mass * (dist_sq_from_body_cog[0] + dist_sq_from_body_cog[1])
                
                this_body_final_moi = this_body_local_moi_sum + this_body_parallel_correction
                
                individual_details.append({
                    "name"  : body.name,
                    "mass"  : this_body_mass,
                    "cog"   : this_body_cog,
                    "moi"   : this_body_final_moi
                })
            
            # (C) 자식 컴포넌트들에 대해 재귀적으로 탐색 수행
            for child in body.children:
                _collect_all_blocks_and_calculate_sub_inertias(child)
                
        # 재귀 호출 수행 (self가 루트가 됨)
        _collect_all_blocks_and_calculate_sub_inertias(self)
        
        # 2. 전역(Assembly 전체) 합계 데이터 계산
        total_mass = 0.0
        total_weighted_cog_sum = np.zeros(3)
        total_pure_local_moi_sum = np.zeros(3)
        
        # 수집된 모든 블록을 한 번에 처리 (가장 정확한 합산 방식)
        for blk in all_primitive_blocks:
            total_mass += blk.mass
            total_weighted_cog_sum += blk.mass * np.array([blk.cx, blk.cy, blk.cz])
            
            w_full, h_full, d_full = 2.0 * blk.dx, 2.0 * blk.dy, 2.0 * blk.dz
            total_pure_local_moi_sum[0] += (1.0/12.0) * blk.mass * (h_full**2 + d_full**2)
            total_pure_local_moi_sum[1] += (1.0/12.0) * blk.mass * (w_full**2 + d_full**2)
            total_pure_local_moi_sum[2] += (1.0/12.0) * blk.mass * (w_full**2 + h_full**2)
            
        if total_mass > 0:
            # 전체 조립체의 전역 CoG
            total_cog = total_weighted_cog_sum / total_mass
            
            # 전체 조립체 CoG 기준의 평행축 정리 보정량 계산
            total_parallel_moI_correction = np.zeros(3)
            for blk in all_primitive_blocks:
                b_pos = np.array([blk.cx, blk.cy, blk.cz])
                dist_sq_from_total_cog = (b_pos - total_cog)**2
                total_parallel_moI_correction[0] += blk.mass * (dist_sq_from_total_cog[1] + dist_sq_from_total_cog[2])
                total_parallel_moI_correction[1] += blk.mass * (dist_sq_from_total_cog[0] + dist_sq_from_total_cog[2])
                total_parallel_moI_correction[2] += blk.mass * (dist_sq_from_total_cog[0] + dist_sq_from_total_cog[1])
                
            final_total_moi = total_pure_local_moi_sum + total_parallel_moI_correction
        else:
            total_cog = np.zeros(3)
            final_total_moi = np.zeros(3)
            
        return total_mass, total_cog, final_total_moi, individual_details


    def get_worldbody_xml_strings(self, indent_level=2):
        """자신과 자식의 Body 구문을 재귀적으로 생성. 
        계층적 구조(Tree)를 XML에 그대로 반영합니다."""
        xml_outs = []
        ind = "  " * indent_level
        
        # 자신이 순수 컨테이너(가상 루트 등)라면 하위 노드만 출력
        if not self.blocks and self.children:
            for child in self.children:
                xml_outs.extend(child.get_worldbody_xml_strings(indent_level))
            return xml_outs
            
        # [NEW] Determine geom group based on naming convention
        lower_name = self.name.lower()
        if "cushion" in lower_name:
            geom_group = 1
        elif "chassis" in lower_name or "opencell" in lower_name or "adhesive" in lower_name or "tape" in lower_name or "cohesive" in lower_name:
            geom_group = 2
        elif "aux" in lower_name or "speaker" in lower_name:
            geom_group = 3
        else:
            geom_group = 0

        # [NEW] Single-Body 모드 처리 (use_internal_weld=False 일 때)
        # 이 모드에서는 최하위 격자마다 body 를 만들지 않고, 
        # 하나의 부모 body 안에 모든 geom 을 직접 쏟아부어 구속조건을 0으로 만듭니다.
        if not self.use_internal_weld:
            # 부모 바디 시작
            xml_outs.append(f'{ind}<body name="{self.name}">')
            ind_c = ind + "  "
            
            # [FIX] 단일 바디 모드에서도 부모(박스/어셈블리) 대비 움직일 수 있도록 조인트 추가
            #PackagingBox나 AssySet같은 가상/그룹 루트는 제외
            if self.name not in ["PackagingBox", "AssySet"]:
                xml_outs.append(f'{ind_c}<joint type="slide" axis="1 0 0"/>')
                xml_outs.append(f'{ind_c}<joint type="slide" axis="0 1 0"/>')
                xml_outs.append(f'{ind_c}<joint type="slide" axis="0 0 1"/>')
                xml_outs.append(f'{ind_c}<joint type="ball"/>')
            
            # 모든 격자 블록을 Geom으로 추가 (Body 없이)
            for (i, j, k), blk in self.blocks.items():
                rgba = blk.material.get("rgba", "0.8 0.8 0.8 1.0")
                contype = blk.material.get("contype", "1")
                conaffinity = blk.material.get("conaffinity", "1")
                solref = blk.material.get("solref", "0.02 1.0")
                solimp = blk.material.get("solimp", "0.9 0.95 0.001 0.5 2")
                
                friction = blk.material.get("friction", "0.8")
                
                # 단일 바디이므로 pos 는 블록의 절대(로컬) 위치 적용
                xml_outs.append(f'{ind_c}<geom name="g_{self.name.lower()}_{i}_{j}_{k}" type="box" '
                                 f'pos="{blk.cx:.5f} {blk.cy:.5f} {blk.cz:.5f}" '
                                 f'size="{blk.dx:.5f} {blk.dy:.5f} {blk.dz:.5f}" mass="{blk.mass:.6f}" '
                                 f'rgba="{rgba}" contype="{contype}" conaffinity="{conaffinity}" group="{geom_group}" '
                                 f'friction="{friction}" solref="{solref}" solimp="{solimp}"/>')
                
                # 부품 간 결합(Inter-weld)을 위해 사이트는 유지
                xml_outs.append(f'{ind_c}<site name="s_{self.name}_{i}_{j}_{k}_PX" pos="{blk.cx+blk.dx:.5f} {blk.cy:.5f} {blk.cz:.5f}"/>')
                xml_outs.append(f'{ind_c}<site name="s_{self.name}_{i}_{j}_{k}_NX" pos="{blk.cx-blk.dx:.5f} {blk.cy:.5f} {blk.cz:.5f}"/>')
                xml_outs.append(f'{ind_c}<site name="s_{self.name}_{i}_{j}_{k}_PY" pos="{blk.cx:.5f} {blk.cy+blk.dy:.5f} {blk.cz:.5f}"/>')
                xml_outs.append(f'{ind_c}<site name="s_{self.name}_{i}_{j}_{k}_NY" pos="{blk.cx:.5f} {blk.cy-blk.dy:.5f} {blk.cz:.5f}"/>')
                xml_outs.append(f'{ind_c}<site name="s_{self.name}_{i}_{j}_{k}_PZ" pos="{blk.cx:.5f} {blk.cy:.5f} {blk.cz+blk.dz:.5f}"/>')
                xml_outs.append(f'{ind_c}<site name="s_{self.name}_{i}_{j}_{k}_NZ" pos="{blk.cx:.5f} {blk.cy:.5f} {blk.cz-blk.dz:.5f}"/>')

            # 자식 컴포넌트 처리
            for child in self.children:
                xml_outs.extend(child.get_worldbody_xml_strings(indent_level + 1))
            
            xml_outs.append(f'{ind}</body>')
            return xml_outs

        # [기존] 이산 바디 모드 (Weld 가 필요한 유연한 시뮬레이션 시 사용)
        xml_outs.append(f'{ind}<body name="{self.name}">')
        ind_c = ind + "  "
        for (i, j, k), blk in self.blocks.items():
            rgba = blk.material.get("rgba", "0.8 0.8 0.8 1.0")
            contype = blk.material.get("contype", "1")
            conaffinity = blk.material.get("conaffinity", "1")
            solref = blk.material.get("solref", "0.02 1.0")
            solimp = blk.material.get("solimp", "0.9 0.95 0.001 0.5 2")
            
            block_body_name = f"b_{self.name.lower()}_{i}_{j}_{k}"
            xml_outs.append(f'{ind_c}<body name="{block_body_name}" pos="{blk.cx:.5f} {blk.cy:.5f} {blk.cz:.5f}">')
            xml_outs.append(f'{ind_c}  <joint type="slide" axis="1 0 0"/>')
            xml_outs.append(f'{ind_c}  <joint type="slide" axis="0 1 0"/>')
            xml_outs.append(f'{ind_c}  <joint type="slide" axis="0 0 1"/>')
            xml_outs.append(f'{ind_c}  <joint type="ball"/>')
            
            friction = blk.material.get("friction", "0.8")
            
            xml_outs.append(f'{ind_c}  <geom name="g_{self.name.lower()}_{i}_{j}_{k}" type="box" '
                             f'size="{blk.dx:.5f} {blk.dy:.5f} {blk.dz:.5f}" mass="{blk.mass:.6f}" '
                             f'rgba="{rgba}" contype="{contype}" conaffinity="{conaffinity}" group="{geom_group}" '
                             f'friction="{friction}" solref="{solref}" solimp="{solimp}"/>')
            
            xml_outs.append(f'{ind_c}  <site name="s_{self.name}_{i}_{j}_{k}_PX" pos="{blk.dx:.5f} 0 0"/>')
            xml_outs.append(f'{ind_c}  <site name="s_{self.name}_{i}_{j}_{k}_NX" pos="{-blk.dx:.5f} 0 0"/>')
            xml_outs.append(f'{ind_c}  <site name="s_{self.name}_{i}_{j}_{k}_PY" pos="0 {blk.dy:.5f} 0"/>')
            xml_outs.append(f'{ind_c}  <site name="s_{self.name}_{i}_{j}_{k}_NY" pos="0 {-blk.dy:.5f} 0"/>')
            xml_outs.append(f'{ind_c}  <site name="s_{self.name}_{i}_{j}_{k}_PZ" pos="0 0 {blk.dz:.5f}"/>')
            xml_outs.append(f'{ind_c}  <site name="s_{self.name}_{i}_{j}_{k}_NZ" pos="0 0 {-blk.dz:.5f}"/>')
            xml_outs.append(f'{ind_c}</body>')

        for child in self.children:
            xml_outs.extend(child.get_worldbody_xml_strings(indent_level + 1))
            
        xml_outs.append(f'{ind}</body>')
        return xml_outs

# =====================================================================
# [4] 개별 부품 파생 클래스
# =====================================================================

class BPaperBox(BaseDiscreteBody):
    def __init__(self, name, width, height, depth, mass, div, thick, material_props, use_internal_weld=True):
        super().__init__(name, width, height, depth, mass, div, material_props, use_internal_weld)
        self.thick = thick
        
    def is_cavity(self, cx, cy, cz, dx, dy, dz):
        # 종이 박스는 두께(thick) 속을 제외한 내부는 전부 공동(cavity)이다.
        # 중심점 위치가 전체 치수에서 양쪽 껍질 두께를 뺀 내부에 속하는지 검사
        in_x = abs(cx) < (self.width/2 - self.thick - 1e-4) # -1e-4 는 소수점 오차 마진
        in_y = abs(cy) < (self.height/2 - self.thick - 1e-4)
        in_z = abs(cz) < (self.depth/2 - self.thick - 1e-4)
        if in_x and in_y and in_z:
            return True
        return False

class BCushion(BaseDiscreteBody):
    def __init__(self, name, width, height, depth, mass, div, material_props, assy_bbox, gap, cushion_cutter, use_internal_weld=True):
        super().__init__(name, width, height, depth, mass, div, material_props, use_internal_weld)
        self.assy_bbox = assy_bbox # AssySet이 차지하는 범위 [min_x, max_x, min_y, max_y, min_z, max_z]
        self.gap = gap
        self.cushion_cutter = cushion_cutter # dict of {key: [cx, cy, cz, w, h, d]}

    def is_cavity(self, cx, cy, cz, dx, dy, dz):
        # 1. AssySet + gap 영역에 포함되는가?
        ax_min, ax_max, ay_min, ay_max, az_min, az_max = self.assy_bbox
        if (ax_min - self.gap <= cx <= ax_max + self.gap and
            ay_min - self.gap <= cy <= ay_max + self.gap and
            az_min - self.gap <= cz <= az_max + self.gap):
            return True
            
        # 2. 추가 Cutter 영역에 포함되는가? (사용자 지정 형상 제거기)
        for cut_vals in self.cushion_cutter.values():
            ctx, cty, ctz, cw, ch, cd = cut_vals
            if (ctx - cw/2 <= cx <= ctx + cw/2 and
                cty - ch/2 <= cy <= cty + ch/2 and
                ctz - cd/2 <= cz <= ctz + cd/2):
                return True
                
        return False

class BOpenCellCohesive(BaseDiscreteBody):
    def __init__(self, name, width, height, depth, mass, div, ithick, material_props, use_internal_weld=True):
        super().__init__(name, width, height, depth, mass, div, material_props, use_internal_weld)
        self.ithick = ithick

    def is_cavity(self, cx, cy, cz, dx, dy, dz):
        # 테이프(액자) 형상: 내부 (width - 2*ithick) x (height - 2*ithick) 만큼 뚫려있음
        # 단 두께 방향(Z)으로는 전부 채워져 있음 (테이프 두께만큼 얇은 블록)
        in_x = abs(cx) < (self.width/2 - self.ithick - 1e-4)
        in_y = abs(cy) < (self.height/2 - self.ithick - 1e-4)
        if in_x and in_y:
            return True
        return False

class BOpenCell(BaseDiscreteBody): # 단순 솔리드 블록
    pass

class BChassis(BaseDiscreteBody): # 단순 솔리드 블록
    pass

class BAuxBoxMass(BaseDiscreteBody):
    """
    CoG/MoI 조절을 위해 임의로 추가하는 박스 형상의 질량 블록 클래스입니다.
    이 블록은 MuJoCo 시뮬레이션 내에서 다른 지오메트리들과 물리적인 접촉(Contact)을 
    일으키지 않도록 설정되며(contype=0, conaffinity=0), 
    주로 섀시(Chassis) 등 특정 부품에 용접(Weld)되어 전체 시스템의 관성 특성을 
    변경하는 용도로 사용됩니다.
    """
    def __init__(self, name, width, height, depth, mass, material_props=None):
        # 사용자가 별도의 재질 정보를 주지 않을 경우, 시각적 확인이 쉽도록 반투명한 빨간색을 기본값으로 사용합니다.
        if material_props is None:
            material_props = {
                "rgba": "1.0 0.0 0.0 0.4",      # 반투명 빨간색
                "solref": "0.02 1.0",           # 기본 접촉 파라미터 (접촉은 안 하지만 구조상 포함)
                "solimp": "0.9 0.95 0.001"
            }
            
        # 질량 블록은 단일 박스로 구성하므로 내부 분할(div)은 [1, 1, 1]로 고정합니다.
        super().__init__(name, width, height, depth, mass, [1, 1, 1], material_props)
        
        # 접촉(Collision) 기능을 완전히 비활성화합니다.
        self.material_props["contype"] = "0"
        self.material_props["conaffinity"] = "0"

    def build_geometry(self, local_offset=[0, 0, 0]):
        """
        주어진 오프셋 위치에 단일 이산 블록(Discrete Block)을 생성합니다.
        """
        # (0, 0, 0) 인덱스 하나만 가지는 단일 블록으로 처리합니다.
        half_width  = self.width / 2.0
        half_height = self.height / 2.0
        half_depth  = self.depth / 2.0
        
        # DiscreteBlock 인스턴스를 하나 생성하여 blocks 딕셔너리에 저장합니다.
        # local_offset은 부모 좌표계(예: Chassis) 기준의 설치 위치입니다.
        aux_block = DiscreteBlock(
            idx=(0, 0, 0), 
            cx=local_offset[0], 
            cy=local_offset[1], 
            cz=local_offset[2],
            dx=half_width, 
            dy=half_height, 
            dz=half_depth, 
            mass=self.total_mass, 
            material=self.material_props
        )
        self.blocks[(0, 0, 0)] = aux_block

class BUnitBlock(BaseDiscreteBody): # 순수 테스트용 직육면체 단위 블록
    pass

# =====================================================================
# [5] 메인 어셈블리 및 파일 생성
# =====================================================================

def parse_drop_target(mode_str, box_w, box_h, box_d):
    """
    모드 문자열(F, B, L, R, T 등 조합)을 파싱하여,
    바닥에 닿아야 할 Local Box 기준(가장 Z축 아래를 향해야 할) 타겟 지점(벡터)을 반환합니다.
    """
    mode = str(mode_str).upper()
    
    # 1. Legacy compatibility
    if mode == "PARCEL":
        return np.array([0, 0, box_d/2]) # Front face (+Z) 가 바닥
    if mode == "LTL":
        return np.array([0, -box_h/2, 0]) # Bottom face (-Y) 가 바닥
        
    # 2. Parse ISTA string combinations
    tokens = mode.split('-')
    vec = [0.0, 0.0, 0.0]
    
    has_L = 'L' in tokens or 'LEFT' in tokens
    has_F = 'F' in tokens or 'FRONT' in tokens
    
    for tk in tokens:
        tk = tk.strip()
        if tk in ['F', 'FRONT']:
            vec[2] = 1.0  # Front
        elif tk in ['T', 'TOP']:
            vec[1] = 1.0  # Top
        elif tk in ['B', 'BOT', 'BOTTOM']:
            vec[1] = -1.0 # Bottom
        elif tk in ['L', 'LEFT']:
            vec[0] = -1.0 # Left
        elif tk in ['R', 'RIGHT', 'REAR']:
            # Ambiguity resolution for 'R' (Right vs Rear)
            if tk == 'RIGHT':
                vec[0] = 1.0
            elif tk == 'REAR':
                vec[2] = -1.0
            else:
                # L이 있으면 R은 Rear, F가 있으면 R은 Right 로 추정
                if has_L:
                    vec[2] = -1.0 # Rear
                elif has_F:
                    vec[0] = 1.0  # Right
                else: 
                    vec[0] = 1.0  # Default to Right
                    
    # Target point in local coordinates
    target_pt = np.array([vec[0] * box_w/2, vec[1] * box_h/2, vec[2] * box_d/2])
    
    # 유효하지 않은 입력이면 기본 PARCEL 복귀
    if np.linalg.norm(target_pt) < 1e-6:
        target_pt = np.array([0, 0, box_d/2])
        
    return target_pt

def get_single_body_instance(body_name, config=None):
    """
    단일 부품(개단품) 강성 평가용 지오메트리 추출기.
    메인 어셈블리(create_model)와 100% 동일한 필수 절단선(required_cuts), 
    공동(Cavity) 로직, 그리드 분할 방식을 강제로 통일하기 위한 팩토리 함수입니다.
    """
    config = get_default_config(config)
    box_w = config["box_w"]; box_h = config["box_h"]; box_d = config["box_d"]
    box_thick = config["box_thick"]; box_div = config["box_div"]
    
    cush_gap = config["cush_gap"]
    cush_w, cush_h, cush_d = box_w - 2 * box_thick, box_h - 2 * box_thick, box_d - 2 * box_thick
    
    assy_w = config.get("assy_w", cush_w - 0.3)
    assy_h = config.get("assy_h", cush_h - 0.3)
    oc_d = config["oc_d"]; occ_d = config["occ_d"]; chas_d = config["chas_d"]
    assy_d = oc_d + occ_d + chas_d
    
    occ_ithick = config["occ_ithick"]
    assy_div = config["assy_div"]; cush_div = config["cush_div"]
    
    # Z-axis placement logic from create_model
    oc_z   = assy_d/2 - oc_d/2
    occ_z  = oc_z - oc_d/2 - occ_d/2
    chas_z = occ_z - occ_d/2 - chas_d/2
    
    # OCC (테이프) 중심 비우기 컷 정보: 중앙 여백의 경계면
    occ_cut_x = [-assy_w/2 + occ_ithick, assy_w/2 - occ_ithick]
    occ_cut_y = [-assy_h/2 + occ_ithick, assy_h/2 - occ_ithick]
    
    if body_name == "BPaperBox":
        b = BPaperBox("BPaperBox", box_w, box_h, box_d, config["mass_paper"], config["box_div"], box_thick, config["mat_paper"], config["box_use_weld"])
        b.build_geometry(local_offset=[0,0,0], 
                         required_cuts_x=[-box_w/2+box_thick, box_w/2-box_thick],
                         required_cuts_y=[-box_h/2+box_thick, box_h/2-box_thick],
                         required_cuts_z=[-box_d/2+box_thick, box_d/2-box_thick])
        return b
        
    elif body_name == "BCushion":
        assy_bbox = [-assy_w/2, assy_w/2, -assy_h/2, assy_h/2, -assy_d/2, assy_d/2]
        BCushion_cutter = { "center": [0, 0, 0, cush_w*0.5, cush_h*0.5, cush_d*2] }
        b = BCushion("BCushion", cush_w, cush_h, cush_d, config["mass_cushion"], config["cush_div"], config["mat_cush"], assy_bbox, cush_gap, BCushion_cutter, config["cush_use_weld"])
        
        req_cuts_cush_x = [-assy_w/2 - cush_gap, assy_w/2 + cush_gap]
        req_cuts_cush_y = [-assy_h/2 - cush_gap, assy_h/2 + cush_gap]
        req_cuts_cush_z = [-assy_d/2 - cush_gap, assy_d/2 + cush_gap]
        for cut_vals in BCushion_cutter.values():
            ctx, cty, ctz, cw, ch, cd = cut_vals
            req_cuts_cush_x.extend([ctx - cw/2, ctx + cw/2])
            req_cuts_cush_y.extend([cty - ch/2, cty + ch/2])
            req_cuts_cush_z.extend([ctz - cd/2, ctz + cd/2])
            
        b.build_geometry(local_offset=[0,0,0], required_cuts_x=req_cuts_cush_x, required_cuts_y=req_cuts_cush_y, required_cuts_z=req_cuts_cush_z)
        return b
        
    elif body_name == "BOpenCell":
        b = BOpenCell("BOpenCell", assy_w, assy_h, oc_d, config["mass_oc"], config["oc_div"], config["mat_cell"], config["oc_use_weld"])
        # stiffness test 에서는 중심점을 원점(0,0,0)에 맞추어 평가를 용이하게 함
        b.build_geometry(local_offset=[0,0,0], required_cuts_x=occ_cut_x, required_cuts_y=occ_cut_y)
        return b
        
    elif body_name == "BOpenCellCohesive":
        b = BOpenCellCohesive("BOpenCellCohesive", assy_w, assy_h, occ_d, config["mass_occ"], config["occ_div"], occ_ithick, config["mat_tape"], config["occ_use_weld"])
        b.build_geometry(local_offset=[0,0,0], required_cuts_x=occ_cut_x, required_cuts_y=occ_cut_y)
        return b
        
    elif body_name == "BChassis":
        b = BChassis("BChassis", assy_w, assy_h, chas_d, config["mass_chassis"], config["chassis_div"], config["mat_tv"], config["chassis_use_weld"])
        b.build_geometry(local_offset=[0,0,0], required_cuts_x=occ_cut_x, required_cuts_y=occ_cut_y)
        return b
        
    elif body_name == "BUnitBlock":
        size = config.get("unit_size", [1.0, 1.0, 1.0])
        div = config.get("unit_div", [5, 5, 5])
        mass = config.get("mass_cushion", 1.0)
        b = BUnitBlock("BUnitBlock", size[0], size[1], size[2], mass, div, config["mat_cush"])
        b.build_geometry(local_offset=[0,0,0])
        return b
        
    else:
        raise ValueError(f"Unknown discrete body type: {body_name}")

def create_model(export_path, config=None, logger=print):
    """
    구성(config)에 따라 MuJoCo 용 XML 모델 파일을 생성하고 저장합니다.
    - export_path: 저장할 XML 파일 경로
    - config: 설정 딕셔너리 (None일 경우 기본값 사용)
    - logger: 출력용 함수 (기본값 print, 외부 로깅용으로 교체 가능)
    """
    # 사용자가 제공한 설정과 디폴트 설정을 내부적으로 계산하여 완벽하게 통합
    config = get_default_config(config)
        
    drop_mode = config["drop_mode"]
    drop_height = config["drop_height"]
    include_paperbox = config["include_paperbox"]
    include_cushion = config["include_cushion"]

    # --- 1. 형상 파라미터 초기화 ---
    box_w = config["box_w"]
    box_h = config["box_h"]
    box_d = config["box_d"]
    box_thick = config["box_thick"]
    box_div = config["box_div"]
    
    cush_gap = config["cush_gap"]
    cush_w, cush_h, cush_d = box_w - 2*box_thick, box_h - 2*box_thick, box_d - 2*box_thick
    
    assy_w = config.get("assy_w", cush_w - 0.3)
    assy_h = config.get("assy_h", cush_h - 0.3)
    oc_d = config["oc_d"]
    occ_d = config["occ_d"]
    chas_d = config["chas_d"]
    
    assy_d = oc_d + occ_d + chas_d
    occ_ithick = config["occ_ithick"]
    
    mat_paper = config["mat_paper"]
    mat_cush  = config["mat_cush"]
    mat_tape  = config["mat_tape"]
    mat_cell  = config["mat_cell"]
    mat_tv    = config["mat_tv"]

    # --- 2. 컴포넌트 인스턴스화 (조립 계층 정의) ---
    root_container = BaseDiscreteBody("PackagingBox", 0,0,0, 0, [1,1,1], {})
    
    # [NEW] 자기 충돌(Self-Collision) 방지를 위한 비트 마스크 할당
    # 각 부품별로 고유 비트를 할당하고(contype), 자기 비트를 제외한 비트들과만 충돌하게(conaffinity) 설정
    bit_paper   = 1    # 00001
    bit_cushion = 2    # 00010
    bit_oc      = 4    # 00100
    bit_occ     = 8    # 01000
    bit_chassis = 16   # 10000
    
    # 모든 부품 비트의 합 (바닥과의 접촉 등을 위해 필요)
    all_bits = bit_paper | bit_cushion | bit_oc | bit_occ | bit_chassis
    
    # 각 재질 설정에 비트 적용 (자기 자신과는 충돌하지 않도록 conaffinity 에서 자기 비트 제외)
    # [주의] mat_cush["contype"] = bit_cushion, mat_cush["conaffinity"] = all_bits ^ bit_cushion
    mat_paper["contype"] = f"{bit_paper}";   mat_paper["conaffinity"] = f"{all_bits ^ bit_paper}"
    mat_cush["contype"]  = f"{bit_cushion}"; mat_cush["conaffinity"]  = f"{all_bits ^ bit_cushion}"
    mat_cell["contype"]  = f"{bit_oc}";      mat_cell["conaffinity"]  = f"{all_bits ^ bit_oc}"
    mat_tape["contype"]  = f"{bit_occ}";     mat_tape["conaffinity"]  = f"{all_bits ^ bit_occ}"
    mat_tv["contype"]    = f"{bit_chassis}"; mat_tv["conaffinity"]    = f"{all_bits ^ bit_chassis}"

    if include_paperbox:
        b_paper = BPaperBox("BPaperBox", box_w, box_h, box_d, config["mass_paper"], config["box_div"], box_thick, mat_paper, config["box_use_weld"])
    else:
        b_paper = None
    
    assy_group = BaseDiscreteBody("AssySet", 0,0,0, 0, [1,1,1], {})
    
    oc_z   = assy_d/2 - oc_d/2
    occ_z  = oc_z - oc_d/2 - occ_d/2
    chas_z = occ_z - occ_d/2 - chas_d/2
    
    b_opencell = BOpenCell("BOpenCell", assy_w, assy_h, oc_d, config["mass_oc"], config["oc_div"], mat_cell, config["oc_use_weld"])
    b_occ      = BOpenCellCohesive("BOpenCellCohesive", assy_w, assy_h, occ_d, config["mass_occ"], config["occ_div"], occ_ithick, mat_tape, config["occ_use_weld"])
    b_chassis  = BChassis("BChassis", assy_w, assy_h, chas_d, config["mass_chassis"], config["chassis_div"], mat_tv, config["chassis_use_weld"])

    b_cushion = None
    if include_cushion:
        assy_bbox = [-assy_w/2, assy_w/2, -assy_h/2, assy_h/2, -assy_d/2, assy_d/2]
        BCushion_cutter = { "center": [0, 0, 0, cush_w*0.5, cush_h*0.5, cush_d*2] }
        b_cushion = BCushion("BCushion", cush_w, cush_h, cush_d, config["mass_cushion"], config["cush_div"], mat_cush, assy_bbox, cush_gap, BCushion_cutter, config["cush_use_weld"])
    
    # 트리 구조 편입 (계층 조립)
    assy_group.add_child(b_opencell)
    assy_group.add_child(b_occ)
    assy_group.add_child(b_chassis)
    
    if include_paperbox:
        root_container.add_child(b_paper)
    if include_cushion:
        root_container.add_child(b_cushion)
    root_container.add_child(assy_group)

    # --- 3. 엄격한 조건(필수 절단선) 도출 및 지오메트리 빌드 ---
    if include_paperbox:
        b_paper.build_geometry(required_cuts_x=[-box_w/2+box_thick, box_w/2-box_thick],
                               required_cuts_y=[-box_h/2+box_thick, box_h/2-box_thick],
                               required_cuts_z=[-box_d/2+box_thick, box_d/2-box_thick])
                               
    if include_cushion:
        # 기본 위치 기반 절단선 (AssySet이 들어가는 내부 여백)
        req_cuts_cush_x = [-assy_w/2 - cush_gap, assy_w/2 + cush_gap]
        req_cuts_cush_y = [-assy_h/2 - cush_gap, assy_h/2 + cush_gap]
        req_cuts_cush_z = [-assy_d/2 - cush_gap, assy_d/2 + cush_gap]
        
        # 커터(딕셔너리) 영역 기반 필수 절단선 동적 추가
        # 면적이 파이는 테두리에 맞추어 블록이 깔끔하게 분할되도록 설정
        for cut_vals in BCushion_cutter.values():
            ctx, cty, ctz, cw, ch, cd = cut_vals
            req_cuts_cush_x.extend([ctx - cw/2, ctx + cw/2])
            req_cuts_cush_y.extend([cty - ch/2, cty + ch/2])
            req_cuts_cush_z.extend([ctz - cd/2, ctz + cd/2])
            
        b_cushion.build_geometry(required_cuts_x=req_cuts_cush_x, 
                                 required_cuts_y=req_cuts_cush_y, 
                                 required_cuts_z=req_cuts_cush_z)
                                 
    # OCC (테이프) 중심 비우기 컷 정보: 중앙 여백의 경계면
    occ_cut_x = [-assy_w/2 + occ_ithick, assy_w/2 - occ_ithick]
    occ_cut_y = [-assy_h/2 + occ_ithick, assy_h/2 - occ_ithick]
    
    b_opencell.build_geometry(local_offset=[0, 0, oc_z], required_cuts_x=occ_cut_x, required_cuts_y=occ_cut_y)
    b_occ.build_geometry(local_offset=[0, 0, occ_z], required_cuts_x=occ_cut_x, required_cuts_y=occ_cut_y)
    b_chassis.build_geometry(local_offset=[0, 0, chas_z], required_cuts_x=occ_cut_x, required_cuts_y=occ_cut_y)

    # -----------------------------------------------------------------
    # [NEW] 3.5 추가 질량(Auxiliary Masses) 처리 및 Chassis 에 Weld 부착
    # -----------------------------------------------------------------
    aux_mass_objects = []
    chassis_aux_configs = config.get("chassis_aux_masses", [])
    
    for i, aux_item_config in enumerate(chassis_aux_configs):
        aux_name   = aux_item_config.get("name", f"AuxMass_{i}")
        aux_pos    = aux_item_config.get("pos", [0.0, 0.0, 0.0])
        aux_size   = aux_item_config.get("size", [0.1, 0.1, 0.1])
        aux_mass_kg = aux_item_config.get("mass", 1.0)
        
        b_aux_mass = BAuxBoxMass(
            name=aux_name, 
            width=aux_size[0], 
            height=aux_size[1], 
            depth=aux_size[2], 
            mass=aux_mass_kg
        )
        
        b_aux_mass.build_geometry(local_offset=[aux_pos[0], aux_pos[1], aux_pos[2] + chas_z])
        b_chassis.add_child(b_aux_mass)
        aux_mass_objects.append(b_aux_mass)

    # --- 4. 이기종 부품 간 직접 Weld 연결 로직 (Inter-Component Welds) ---
    inter_weld_xml = []
    tape_solref_val = mat_tape.get("solref", "0.005 1.0")
    tape_solimp_val = mat_tape.get("solimp", "0.9 0.999 0.001 0.5 2")
    
    for (i,j,k_occ), blk_occ in b_occ.blocks.items():
        if (i,j,0) in b_opencell.blocks:
            site_occ_pz = f"s_BOpenCellCohesive_{i}_{j}_{0}_PZ"
            site_oc_nz  = f"s_BOpenCell_{i}_{j}_{0}_NZ"
            inter_weld_xml.append(f'        <weld site1="{site_occ_pz}" site2="{site_oc_nz}" solref="{tape_solref_val}" solimp="{tape_solimp_val}"/>')
            
        if (i,j,0) in b_chassis.blocks:
            site_occ_nz = f"s_BOpenCellCohesive_{i}_{j}_{0}_NZ"
            site_chas_pz = f"s_BChassis_{i}_{j}_{0}_PZ"
            inter_weld_xml.append(f'        <weld site1="{site_occ_nz}" site2="{site_chas_pz}" solref="{tape_solref_val}" solimp="{tape_solimp_val}"/>')

    for b_aux_mass in aux_mass_objects:
        blk_aux = b_aux_mass.blocks[(0, 0, 0)]
        target_ax, target_ay, target_az = blk_aux.cx, blk_aux.cy, blk_aux.cz
        
        min_distance_sq = float('inf')
        nearest_chassis_key = None
        
        for block_key, blk_chassis in b_chassis.blocks.items():
            dist_sq = (blk_chassis.cx - target_ax)**2 + \
                      (blk_chassis.cy - target_ay)**2 + \
                      (blk_chassis.cz - target_az)**2
            if dist_sq < min_distance_sq:
                min_distance_sq = dist_sq
                nearest_chassis_key = block_key
        
        if nearest_chassis_key is not None:
            ci, cj, ck = nearest_chassis_key
            body_aux = b_aux_mass.name if not b_aux_mass.use_internal_weld else f"b_{b_aux_mass.name.lower()}_0_0_0"
            body_chas = b_chassis.name if not b_chassis.use_internal_weld else f"b_{b_chassis.name.lower()}_{ci}_{cj}_{ck}"
            
            inter_weld_xml.append(
                f'        <weld body1="{body_aux}" body2="{body_chas}" '
                f'solref="0.002 1.0" solimp="0.9 0.999 0.001"/>'
            )

    # --- 5. 낙하 조건 (ISTA 6A 등) 로테이션 연산 ---
    target_pt = parse_drop_target(drop_mode, box_w, box_h, box_d)
    target_dist = np.linalg.norm(target_pt)
    rot_axis = np.cross(target_pt, [0, 0, -target_dist])
    
    if np.linalg.norm(rot_axis) < 1e-6:
        if target_pt[2] < 0:
            rot_axis = np.array([1.0, 0.0, 0.0])
            angle_rad = 0.0
        else:
            rot_axis = np.array([1.0, 0.0, 0.0])
            angle_rad = np.pi
    else:
        rot_axis /= np.linalg.norm(rot_axis)
        dot_val = np.dot(target_pt, [0, 0, -target_dist]) / (target_dist**2)
        dot_val = np.clip(dot_val, -1.0, 1.0)
        angle_rad = np.arccos(dot_val)

    wx, wy, wz = get_local_pose([0,0,0], drop_height, rot_axis, angle_rad, target_dist)
    rot_str = f"{rot_axis[0]:.4f} {rot_axis[1]:.4f} {rot_axis[2]:.4f} {np.degrees(angle_rad):.4f}"
    
    # --- 6. XML 파일 작성 ---
    xml_str_io = io.StringIO()
    xml_str_io.write('<mujoco model="discrete_custom_box">\n')
    xml_str_io.write('  <size memory="512M"/>\n')
    s_itgr = config["sim_integrator"]
    s_dt   = config["sim_timestep"]
    s_iter = config["sim_iterations"]
    s_ns   = config["sim_noslip_iterations"]
    s_tol  = config["sim_tolerance"]
    s_imp  = config["sim_impratio"]
    s_grav = config["sim_gravity"]
    s_nth  = config["sim_nthread"]
    
    if config.get("enable_air_drag", True):
        air_rho = config.get("air_density", 1.225)
        air_mu  = config.get("air_viscosity", 1.81e-5)
    else:
        air_rho = 0.0
        air_mu  = 0.0
    
    xml_str_io.write(f'  <option integrator="{s_itgr}" timestep="{s_dt}" iterations="{s_iter}" '
                     f'noslip_iterations="{s_ns}" tolerance="{s_tol}" impratio="{s_imp}" '
                     f'gravity="{s_grav[0]} {s_grav[1]} {s_grav[2]}" '
                     f'density="{air_rho}" viscosity="{air_mu}">\n')
    xml_str_io.write('    <flag contact="enable"/>\n')
    xml_str_io.write('  </option>\n')
    
    xml_str_io.write('  <visual>\n')
    h_amb = config.get("light_head_ambient", "0.28 0.28 0.28")
    h_dif = config.get("light_head_diffuse", "0.56 0.56 0.56")
    xml_str_io.write(f'    <headlight ambient="{h_amb}" diffuse="{h_dif}" specular="0.07 0.07 0.07"/>\n')
    xml_str_io.write('    <map znear="0.01"/>\n')
    xml_str_io.write('  </visual>\n')

    xml_str_io.write('  <default>\n')
    xml_str_io.write('    <joint armature="0.05" damping="1.0"/>\n')
    xml_str_io.write('    <geom friction="0.8" solref="0.02 1.0" solimp="0.9 0.95 0.001"/>\n')
    xml_str_io.write('  </default>\n')
    
    xml_str_io.write('  <worldbody>\n')
    m_dif = config.get("light_main_diffuse", "0.49 0.49 0.49")
    m_amb = config.get("light_main_ambient", "0.21 0.21 0.21")
    s_dif = config.get("light_sub_diffuse", "0.21 0.21 0.21")
    
    xml_str_io.write(f'    <light pos="0 0 6" dir="0 0 -1" directional="false" diffuse="{m_dif}" ambient="{m_amb}" castshadow="true"/>\n')
    xml_str_io.write(f'    <light pos="3 3 5" dir="-1 -1 -1" directional="false" diffuse="{s_dif}" castshadow="false"/>\n')
    
    g_solref = config.get("ground_solref", "0.01 1.0")
    g_solimp = config.get("ground_solimp", "0.9 0.95 0.001 0.5 2")
    g_fric   = config.get("ground_friction", 0.3)
    xml_str_io.write(f'    <geom name="ground" type="plane" size="5 5 0.1" friction="{g_fric}" contype="1" conaffinity="1" group="0" solref="{g_solref}" solimp="{g_solimp}"/>\n')
    
    xml_str_io.write(f'    <body name="BPackagingBox" pos="{wx:.5f} {wy:.5f} {wz:.5f}" axisangle="{rot_str}">\n')
    xml_str_io.write('      <freejoint/>\n')
    xml_str_io.write('      <geom type="box" size="0.001 0.001 0.001" mass="0.000021" rgba="0 0 0 0" contype="0" conaffinity="0" friction="0.8"/>\n')
    
    bodies_xml = root_container.get_worldbody_xml_strings(indent_level=3)
    for line in bodies_xml:
        xml_str_io.write(line + "\n")
        
    xml_str_io.write('    </body>\n')
    xml_str_io.write('  </worldbody>\n')
    
    xml_str_io.write('  <equality>\n')
    internal_weld_xml = root_container.get_weld_xml_strings()
    for line in internal_weld_xml:
        xml_str_io.write(line + "\n")
        
    for line in inter_weld_xml:
        xml_str_io.write(line + "\n")
    xml_str_io.write('  </equality>\n')
    
    xml_str_io.write('</mujoco>\n')
    
    final_xml_str = xml_str_io.getvalue()
    xml_str_io.close()
    
    with open(export_path, "w", encoding="utf-8") as f:
        f.write(final_xml_str)

    # --- 7. 관성 텐서(CoG, MoI) 시스템 출력 ---
    total_mass, cog, moi, individual_details = root_container.calculate_inertia()
    
    name_width = 30
    mass_width = 12
    cog_width  = 32
    moi_width  = 38
    
    header = (f"{'Body Name'.ljust(name_width)} | "
              f"{'Mass (kg)'.rjust(mass_width)} | "
              f"{'CoG (x, y, z)'.center(cog_width)} | "
              f"{'MoI (Ixx, Iyy, Izz)'.center(moi_width)}")
    separator_width = len(header)

    logger("\n" + "=" * separator_width)
    logger(f"[Assembly Inertia Report] Mode: {drop_mode}")
    logger(f"{'-' * separator_width}")
    logger(f" - 전체 질량 (Total Mass)        : {total_mass:12.4f} kg")
    logger(f" - 전체 무게 중심 (Global CoG)    : ({cog[0]:8.4f}, {cog[1]:8.4f}, {cog[2]:8.4f})")
    logger(f" - 전체 관성 모멘트 (Global MoI)  : ({moi[0]:10.6f}, {moi[1]:10.6f}, {moi[2]:10.6f})")
    logger(f"{'-' * separator_width}")
    
    logger(header)
    logger("-" * separator_width)
    
    total_row = (f"{'Total (Assembly)'.ljust(name_width)} | "
                 f"{total_mass:12.4f} | "
                 f"{f'({cog[0]:.3f}, {cog[1]:.3f}, {cog[2]:.3f})'.center(cog_width)} | "
                 f"{f'({moi[0]:.6f}, {moi[1]:.6f}, {moi[2]:.6f})'.center(moi_width)}")
    logger(total_row)
    logger("-" * separator_width)
    
    for detail in individual_details:
        b_name_str = str(detail['name']).ljust(name_width)
        b_mass_val = detail['mass']
        b_cog_vals = detail['cog']
        b_moi_vals = detail['moi']
        
        cog_str = f"({b_cog_vals[0]:.3f}, {b_cog_vals[1]:.3f}, {b_cog_vals[2]:.3f})".center(cog_width)
        moi_str = f"({b_moi_vals[0]:.6f}, {b_moi_vals[1]:.6f}, {b_moi_vals[2]:.6f})".center(moi_width)
        
        logger(f"{b_name_str} | {b_mass_val:12.4f} | {cog_str} | {moi_str}")
        
    logger("=" * separator_width + "\n")

    return final_xml_str, total_mass, cog, moi, individual_details

if __name__ == "__main__":
    out_dir = r"c:\Users\GOODMAN\WHToolsBox\test_box_mujoco"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "test_shapes_check.xml")
    
    print(f"Generating Discrete Model XML to {out_file}...")
    cfg = get_default_config()
    
    # [VERIFICATION] 추가 질량 테스트 설정
    cfg["chassis_aux_masses"] = [
        {"name": "UpperWeight", "pos": [0, 0.5, 0], "size": [0.2, 0.1, 0.1], "mass": 10.0},
        {"name": "SideWeight", "pos": [0.5, 0, 0], "size": [0.1, 0.2, 0.1], "mass": 5.0}
    ]
    cfg["ground_friction"] = 0.5 # 마찰계수 변경 테스트
    
    # 예: 전면 하단 꼭짓점 낙하 자세로 설정
    cfg["drop_mode"] = "L-F-B" 
    cfg["include_paperbox"] = False # 로컬 테스트용 오버라이드
    create_model(out_file, config=cfg)
    print("Generation Complete.")



'''
낙하 자세(Orientation)는 

run_discrete_builder.py의 config 딕셔너리에 있는 drop_mode 옵션을 통해 매우 간편하게 설정할 수 있습니다.
단순히 상하좌우를 넘어, ISTA 낙하 시험 기준에 맞춘 면(Face), 모서리(Edge), 꼭짓점(Corner) 낙하 자세를 문자열 조합으로 생성해줍니다.
🛠️ 설정 방법 및 예시 create_model 을 호출할 때 config["drop_mode"] 값을 다음 중 하나로 변경하면 됩니다.

1. 주요 면(Face) 낙하
가장 넓은 면이나 바닥면을 아래로 향하게 합니다.

PARCEL: 전면(Front, +Z)이 바닥을 향함 (기본값)
F (Front): +Z 방향이 바닥
B (Bottom): -Y 방향이 하단 (TV가 똑바로 서서 떨어지는 자세)
T (Top): +Y 방향이 하단 (거꾸로 뒤집힌 자세)
L (Left): -X 방향이 하단
R (Right): +X 방향이 하단
2. 모서리(Edge) 낙하
두 면의 문자를 하이픈(-)으로 연결하면, 그 사이 모서리가 바닥을 향하도록 자동 회전합니다. (단, 무거운 쪽이 아래로 쏠리는 물리적 무게중심을 고려하여 정렬됩니다.)

F-B (Front-Bottom): 전면 하단 모서리 낙하
L-B (Left-Bottom): 좌측 하단 모서리 낙하
3. 꼭짓점(Corner) 낙하
세 면의 문자를 하이픈으로 연결합니다.

L-F-B (Left-Front-Bottom): 좌측 전면 하단 꼭짓점 낙하 (가장 가혹한 조건 중 하나)
🔍 코드 적용 예시
수정 대신 실행 시점이나 다른 스크립트에서 다음과 같이 활용하실 수 있습니다.

python
cfg = get_default_config()
# 예: 전면 하단 꼭짓점 낙하 자세로 설정
cfg["drop_mode"] = "L-F-B" 
# 모델 생성 (자동으로 바닥 기준 높이 조정 및 회전 적용됨)
create_model("test_shapes_check.xml", config=cfg)
이 drop_mode 문자열은 내부의 

parse_drop_target
 함수가 읽어서 어떤 벡터가 지구 중심(중력 방향)을 향해야 할지 계산한 뒤, 

create_model
 프로세스 마지막 단계에서 시뮬레이션 초기 자세(

pos
, axisangle)를 완벽하게 셋팅해 줍니다.

혹시 특정 각도로 직접 돌리고 싶으시다면 axisangle 속성을 수동으로 조정하는 방법도 있으나, 위 문자열 옵션을 활용하시는 것이 가장 빠르고 직관적입니다!


### light 여기서 다시 50%를 더 줄이고 싶다면?
cfg["light_main_diffuse"] = "0.25 0.25 0.25"
cfg["light_head_diffuse"] = "0.3 0.3 0.3"
'''