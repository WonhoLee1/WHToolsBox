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
        
    # [솔버 파라미터 분리] 문자열 대신 개별 변수로 관리 (최적화 친화적)    
    cush_solref_stiff = user_config.get("cush_solref_stiff", 0.02) # 쿠션 timeconst
    cush_solref_damp = user_config.get("cush_solref_damp", 1.0) # 쿠션 dampratio
    
    # solimp = [dmin, dmax, width, midpoint, power] 
    cush_solimp_dmin = user_config.get("cush_solimp_dmin", 0.95)   # 쿠션 최소 임피던스
    cush_solimp_dmax = user_config.get("cush_solimp_dmax", 0.99)  # 쿠션 최대 임피던스
    cush_solimp_width = user_config.get("cush_solimp_width", 0.001)  # 쿠션 임피던스 폭
    cush_solimp_mid = user_config.get("cush_solimp_mid", 0.5)        # 쿠션 임피던스 중간점
    cush_solimp_power = user_config.get("cush_solimp_power", 2.0)    # 쿠션 임피던스 거듭제곱
    
    # tape
    tape_solref_stiff = user_config.get("tape_solref_stiff", 0.02) # 테이프 timeconst
    tape_solref_damp = user_config.get("tape_solref_damp", 1.0) # 테이프 dampratio

    tape_solimp_dmin = user_config.get("tape_solimp_dmin", 0.9)   # 테이프 최소 임피던스 
    tape_solimp_dmax = user_config.get("tape_solimp_dmax", 0.95)  # 테이프 최대 임피던스
    tape_solimp_width = user_config.get("tape_solimp_width", 0.001)  # 테이프 임피던스 폭
    tape_solimp_mid = user_config.get("tape_solimp_mid", 0.5)        # 테이프 임피던스 중간점
    tape_solimp_power = user_config.get("tape_solimp_power", 2.0)    # 테이프 임피던스 거듭제곱
    
    # cell (TV Panel)
    cell_solref_stiff = user_config.get("cell_solref_stiff", 0.01) # 패널 timeconst
    cell_solref_damp = user_config.get("cell_solref_damp", 1.0) # 패널 dampratio
    
    cell_solimp_dmin = user_config.get("cell_solimp_dmin", 0.9)   # 패널 최소 임피던스
    cell_solimp_dmax = user_config.get("cell_solimp_dmax", 0.95)  # 패널 최대 임피던스
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
    ground_solref_stiff =  user_config.get("ground_solref_stiff", 0.01) # 바닥 timeconst (0.002 미만시 폭발 위험)
    ground_solref_damp = user_config.get("ground_solref_damp", 1.0) # 바닥 dampratio

    # solref: 오리지널 (timeconst, dampratio) 방식
    cush_solref = f"{cush_solref_stiff} {cush_solref_damp}"
    tape_solref = f"{tape_solref_stiff} {tape_solref_damp}"
    cell_solref = f"{cell_solref_stiff} {cell_solref_damp}"
    tv_solref = f"{tv_solref_stiff} {tv_solref_damp}"
    ground_solref = f"{ground_solref_stiff} {ground_solref_damp}"

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

        # 분할 정보 (div)
        "box_div": [5, 4, 3],
        "cush_div": [5, 4, 3],
        "assy_div": [5, 4, 1], # OpenCell, Cohesive, Chassis 세 부품 공통으로 적용되는 분할 수

        # 질량 (Mass, kg)
        "mass_paper": 4.0,
        "mass_cushion": mass_cushion,
        "mass_oc": 5.0,       # OpenCell 질량
        "mass_occ": 0.1,     # Cohesive (테이프) 질량
        "mass_chassis": 5.0,  # Chassis 질량

        # 재료 물성치 (솔버 제어 및 물리 특성)
        "cush_solref": cush_solref,
        "tape_solref": tape_solref,
        "cush_solimp": cush_solimp,
        "tape_solimp": tape_solimp,
        "mat_paper": {"rgba": "0.7 0.6 0.4 0.9", "solref": "0.01 1.0", "solimp": "0.8 0.95 0.001 0.5 2", "contype": "1", "conaffinity": "2"},
        "mat_cush" : {"rgba": "0.9 0.9 0.9 0.5", "solref": cush_solref, "solimp": cush_solimp, "contype": "2", "conaffinity": "5"},
        "mat_tape" : {"rgba": "1.0 0.1 0.1 0.8", "solref": tape_solref, "solimp": tape_solimp, "contype": "4", "conaffinity": "2"},
        "mat_cell" : {"rgba": "0.1 0.1 0.1 1.0", "solref": cell_solref, "solimp": cell_solimp, "contype": "4", "conaffinity": "2"},
        "mat_tv"   : {"rgba": "0.1 0.5 0.8 1.0", "solref": tv_solref, "solimp": tv_solimp, "contype": "4", "conaffinity": "2"},
        
        # 바닥(Floor) 접촉 물질 정보 (Cushion과 유사하게 처리하기 위해 추가)
        "ground_solref": ground_solref, 
        "ground_solimp": "0.9 0.95 0.001 0.5 2",
        "ground_friction": 0.3,      # 바닥 마찰계수 기본값 0.3

        # 조명(Lighting) 설정 - 조절이 쉽도록 옵션으로 추출
        "light_head_ambient" : "0.28 0.28 0.28", # 헤드라이트 환경광
        "light_head_diffuse" : "0.56 0.56 0.56", # 헤드라이트 확산광
        "light_main_ambient" : "0.21 0.21 0.21", # 주 조명 환경광 
        "light_main_diffuse" : "0.49 0.49 0.49", # 주 조명 확산광
        "light_sub_diffuse"  : "0.21 0.21 0.21", # 보조 조명 확산광

        # 공기 저항 (Air Resistance) 설정
        # MuJoCo fluidshape 기반 drag/viscous 및 커스텀 squeeze film 효과
        "air_density"      : 1.225,    # 공기 밀도 (kg/m^3, 20도 1atm)
        "air_viscosity"    : 1.81e-5,  # 공기 동점성계수 (Pa.s)
        "air_cd_drag"      : 1.05,     # Blunt drag 계수 (박스 형태 기준 1.0~1.2)
        "air_cd_viscous"   : 0.0,      # Slender(점성) drag 계수 (박스는 보통 0)
        "air_coef_squeeze" : 1.0,      # Squeeze Film 효과 강도 배율 (0=비활성화)
        "air_squeeze_hmax" : 0.20,     # Squeeze Film 활성화 최대 높이 (m)
        "air_squeeze_hmin" : 0.001,    # Squeeze Film 최소 높이 (분모 안전값, m)
        "enable_air_resistance": True, # 전체 공기 저항 활성화 여부
    }
    
    # 사용자가 직접 전달한 나머지 설정값들을 최종 병합(덮어쓰기)합니다.
    for k, v in user_config.items():
        config[k] = v
        
    # 만약 상위 수준의 변수(cush_solref_damp 등)가 user_config로 전달된 경우,
    # mat_cush 등의 하위 파생 딕셔너리가 과거의 값으로 덮어써지는 문제를 방지하기 위해 
    # 병합 완료 후 다시 한 번 갱신합니다.
    if "mat_cush" in config:
        config["mat_cush"]["solref"] = f"{config.get('cush_solref_stiff', 0.02)} {config.get('cush_solref_damp', 1.0)}"
        config["mat_cush"]["solimp"] = f"{config.get('cush_solimp_dmin', 0.95)} {config.get('cush_solimp_dmax', 0.99)} {config.get('cush_solimp_width', 0.001)} {config.get('cush_solimp_mid', 0.5)} {config.get('cush_solimp_power', 2.0)}"
    if "mat_tape" in config:
        config["mat_tape"]["solref"] = f"{config.get('tape_solref_stiff', 0.02)} {config.get('tape_solref_damp', 1.0)}"
        config["mat_tape"]["solimp"] = f"{config.get('tape_solimp_dmin', 0.9)} {config.get('tape_solimp_dmax', 0.95)} {config.get('tape_solimp_width', 0.001)} {config.get('tape_solimp_mid', 0.5)} {config.get('tape_solimp_power', 2.0)}"
    if "mat_cell" in config:
        config["mat_cell"]["solref"] = f"{config.get('cell_solref_stiff', 0.01)} {config.get('cell_solref_damp', 1.0)}"
        config["mat_cell"]["solimp"] = f"{config.get('cell_solimp_dmin', 0.9)} {config.get('cell_solimp_dmax', 0.95)} {config.get('cell_solimp_width', 0.001)} {config.get('cell_solimp_mid', 0.5)} {config.get('cell_solimp_power', 2.0)}"
    if "mat_tv" in config:
        config["mat_tv"]["solref"] = f"{config.get('tv_solref_stiff', 0.01)} {config.get('tv_solref_damp', 1.0)}"
        config["mat_tv"]["solimp"] = f"{config.get('tv_solimp_dmin', 0.9)} {config.get('tv_solimp_dmax', 0.95)} {config.get('tv_solimp_width', 0.001)} {config.get('tv_solimp_mid', 0.5)} {config.get('tv_solimp_power', 2.0)}"

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
    def __init__(self, name, width, height, depth, mass, div, material_props):
        self.name = name
        self.width = width
        self.height = height
        self.depth = depth
        self.total_mass = mass
        self.div = div  # [Nx, Ny, Nz]
        self.material_props = material_props # 예: {"rgba": "...", "solref": "..."}
        
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
        """자신과 모든 자식 블록들을 취합하여 전체 질량, 무게중심(CoG), 관성 모멘트(MoI)를 계산합니다."""
        total_mass = 0.0
        cog = np.zeros(3)
        moi = np.zeros(3) # 로컬 Ixx, Iyy, Izz (평행축 정리 적용 전)
        
        # 1. 속한 개별 블록들의 질량과 중심 누적
        for blk in self.blocks.values():
            total_mass += blk.mass
            cog += blk.mass * np.array([blk.cx, blk.cy, blk.cz])
            
            # 직육면체의 로컬 관성 모멘트 I = 1/12 * m * (w^2 + h^2)
            # 여기서는 dx, dy, dz 가 half-size 이므로 실제 너비는 2*dx
            w, h, d = 2*blk.dx, 2*blk.dy, 2*blk.dz
            ixx = (1/12.0) * blk.mass * (h**2 + d**2)
            iyy = (1/12.0) * blk.mass * (w**2 + d**2)
            izz = (1/12.0) * blk.mass * (w**2 + h**2)
            
            moi += np.array([ixx, iyy, izz])
            
        # 2. 자식 컴포넌트들의 결과 누적 (재귀)
        for child in self.children:
            child_mass, child_cog, child_moi, child_blocks_data = child.calculate_inertia()
            if child_mass > 0:
                total_mass += child_mass
                cog += child_mass * child_cog
                moi += child_moi
                
        # 3. 전체 평균 무게 중심(CoG) 계산
        if total_mass > 0:
            cog = cog / total_mass
            
        # 4. 평행축 정리(Parallel Axis Theorem) 적용하여 CoG 기준 글로벌 MoI 계산
        # 개별 블록들의 리스트를 다시 평면 순회하여 거리를 구해야 정확한 텐서가 나옴
        global_moi = np.zeros(3)
        
        def _accumulate_global_moi(body):
            nonlocal global_moi
            for blk in body.blocks.values():
                b_cog = np.array([blk.cx, blk.cy, blk.cz])
                dist_sq = (b_cog - cog)**2
                # Ixx는 y, z 거리 제곱의 합, Iyy는 x, z 등... 
                global_moi[0] += blk.mass * (dist_sq[1] + dist_sq[2])
                global_moi[1] += blk.mass * (dist_sq[0] + dist_sq[2])
                global_moi[2] += blk.mass * (dist_sq[0] + dist_sq[1])
            for ch in body.children:
                _accumulate_global_moi(ch)
                
        if total_mass > 0:
            _accumulate_global_moi(self)
        
        # 최종 총합: (개별 로컬 MoI의 합) + (CoG 중심 기준 거리 제곱 질량의 합)
        final_moi = moi + global_moi
        
        return total_mass, cog, final_moi, None

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

        # 내부에 속한 기하 형상(블록)들을 자식 body로 배치
        if self.blocks.items():
            xml_outs.append(f'{ind}<body name="{self.name}">')
        else:
            # 껍데기 전용 (컨테이너 그룹)
            xml_outs.append(f'{ind}<body name="{self.name}">')
            
        ind_c = ind + "  "
        for (i, j, k), blk in self.blocks.items():
            rgba = blk.material.get("rgba", "0.8 0.8 0.8 1.0")
            contype = blk.material.get("contype", "1")
            conaffinity = blk.material.get("conaffinity", "1")
            solref = blk.material.get("solref", "0.02 1.0")
            solimp = blk.material.get("solimp", "0.9 0.95 0.001 0.5 2")
            
            # 소문자로 블록 구동 접두어 적용
            block_body_name = f"b_{self.name.lower()}_{i}_{j}_{k}"
            xml_outs.append(f'{ind_c}<body name="{block_body_name}" pos="{blk.cx:.5f} {blk.cy:.5f} {blk.cz:.5f}">')
            # MuJoCo 에러 (freejoint can only be used on top level) 회피를 위해 
            # 6자유도를 수동으로 부여 (3개 슬라이드 + 1개 볼 조인트)
            xml_outs.append(f'{ind_c}  <joint type="slide" axis="1 0 0"/>')
            xml_outs.append(f'{ind_c}  <joint type="slide" axis="0 1 0"/>')
            xml_outs.append(f'{ind_c}  <joint type="slide" axis="0 0 1"/>')
            xml_outs.append(f'{ind_c}  <joint type="ball"/>')
            
            xml_outs.append(f'{ind_c}  <geom type="box" size="{blk.dx:.5f} {blk.dy:.5f} {blk.dz:.5f}" mass="{blk.mass:.6f}" rgba="{rgba}" contype="{contype}" conaffinity="{conaffinity}" friction="0.8" solref="{solref}" solimp="{solimp}"/>')
            
            # Weld 연결을 위한 접점 사이트 생성
            # 부품의 맨 가장자리 표면 중심들에 기준점(site) 배치
            xml_outs.append(f'{ind_c}  <site name="s_{self.name}_{i}_{j}_{k}_PX" pos="{blk.dx:.5f} 0 0"/>')
            xml_outs.append(f'{ind_c}  <site name="s_{self.name}_{i}_{j}_{k}_NX" pos="{-blk.dx:.5f} 0 0"/>')
            xml_outs.append(f'{ind_c}  <site name="s_{self.name}_{i}_{j}_{k}_PY" pos="0 {blk.dy:.5f} 0"/>')
            xml_outs.append(f'{ind_c}  <site name="s_{self.name}_{i}_{j}_{k}_NY" pos="0 {-blk.dy:.5f} 0"/>')
            xml_outs.append(f'{ind_c}  <site name="s_{self.name}_{i}_{j}_{k}_PZ" pos="0 0 {blk.dz:.5f}"/>')
            xml_outs.append(f'{ind_c}  <site name="s_{self.name}_{i}_{j}_{k}_NZ" pos="0 0 {-blk.dz:.5f}"/>')
            xml_outs.append(f'{ind_c}</body>')

        # 하위 논리적 그룹(AssySet의 자식들 등)이 있다면 추가
        for child in self.children:
            xml_outs.extend(child.get_worldbody_xml_strings(indent_level + 1))
            
        xml_outs.append(f'{ind}</body>')
        return xml_outs

# =====================================================================
# [4] 개별 부품 파생 클래스
# =====================================================================

class BPaperBox(BaseDiscreteBody):
    def __init__(self, name, width, height, depth, mass, div, thick, material_props):
        super().__init__(name, width, height, depth, mass, div, material_props)
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
    def __init__(self, name, width, height, depth, mass, div, material_props, assy_bbox, gap, cushion_cutter):
        super().__init__(name, width, height, depth, mass, div, material_props)
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
    def __init__(self, name, width, height, depth, mass, div, ithick, material_props):
        super().__init__(name, width, height, depth, mass, div, material_props)
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
        b = BPaperBox("BPaperBox", box_w, box_h, box_d, config["mass_paper"], box_div, box_thick, config["mat_paper"])
        b.build_geometry(local_offset=[0,0,0], 
                         required_cuts_x=[-box_w/2+box_thick, box_w/2-box_thick],
                         required_cuts_y=[-box_h/2+box_thick, box_h/2-box_thick],
                         required_cuts_z=[-box_d/2+box_thick, box_d/2-box_thick])
        return b
        
    elif body_name == "BCushion":
        assy_bbox = [-assy_w/2, assy_w/2, -assy_h/2, assy_h/2, -assy_d/2, assy_d/2]
        BCushion_cutter = { "center": [0, 0, 0, cush_w*0.5, cush_h*0.5, cush_d*2] }
        b = BCushion("BCushion", cush_w, cush_h, cush_d, config["mass_cushion"], cush_div, config["mat_cush"], assy_bbox, cush_gap, BCushion_cutter)
        
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
        b = BOpenCell("BOpenCell", assy_w, assy_h, oc_d, config["mass_oc"], assy_div, config["mat_tv"])
        # stiffness test 에서는 중심점을 원점(0,0,0)에 맞추어 평가를 용이하게 함
        b.build_geometry(local_offset=[0,0,0], required_cuts_x=occ_cut_x, required_cuts_y=occ_cut_y)
        return b
        
    elif body_name == "BOpenCellCohesive":
        b = BOpenCellCohesive("BOpenCellCohesive", assy_w, assy_h, occ_d, config["mass_occ"], assy_div, occ_ithick, config["mat_tape"])
        b.build_geometry(local_offset=[0,0,0], required_cuts_x=occ_cut_x, required_cuts_y=occ_cut_y)
        return b
        
    elif body_name == "BChassis":
        b = BChassis("BChassis", assy_w, assy_h, chas_d, config["mass_chassis"], assy_div, config["mat_tv"])
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

def create_model(export_path, config=None):
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
    
    if include_paperbox:
        b_paper = BPaperBox("BPaperBox", box_w, box_h, box_d, config["mass_paper"], box_div, box_thick, mat_paper)
    else:
        b_paper = None
    
    assy_group = BaseDiscreteBody("AssySet", 0,0,0, 0, [1,1,1], {})
    
    oc_z   = assy_d/2 - oc_d/2
    occ_z  = oc_z - oc_d/2 - occ_d/2
    chas_z = occ_z - occ_d/2 - chas_d/2
    
    assy_div = config["assy_div"]
    b_opencell = BOpenCell("BOpenCell", assy_w, assy_h, oc_d, config["mass_oc"], assy_div, mat_cell)
    b_occ      = BOpenCellCohesive("BOpenCellCohesive", assy_w, assy_h, occ_d, config["mass_occ"], assy_div, occ_ithick, mat_tape)
    b_chassis  = BChassis("BChassis", assy_w, assy_h, chas_d, config["mass_chassis"], assy_div, mat_tv)

    b_cushion = None
    if include_cushion:
        assy_bbox = [-assy_w/2, assy_w/2, -assy_h/2, assy_h/2, -assy_d/2, assy_d/2]
        BCushion_cutter = { "center": [0, 0, 0, cush_w*0.5, cush_h*0.5, cush_d*2] }
        b_cushion = BCushion("BCushion", cush_w, cush_h, cush_d, config["mass_cushion"], config["cush_div"], mat_cush, assy_bbox, cush_gap, BCushion_cutter)
    
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

    # --- 4. 이기종 부품 간 직접 Weld 연결 로직 (Inter-Component Welds) ---
    # 이기종 부품 간 직접 Weld 연결 로직 (Inter-Component Welds)
    inter_weld_xml = []
    # AssySet 조립: OpenCell(앞) <---> OCC(테이프) <---> Chassis(뒤)
    # Z축 방향으로만 맞물리므로, (x, y) 인덱스가 같고 마주보는 파트끼리 테이프 강성으로 묶음.
    tape_solref = mat_tape.get("solref", "0.005 1.5")
    tape_solimp = mat_tape.get("solimp", "0.9 0.95 0.001 0.5 2")
    
    for (i,j,k_occ), blk_occ in b_occ.blocks.items():
        # OCC 블록(테이프) 각각에 대해 앞으로는 OpenCell 블록을, 뒤로는 Chassis 블록을 찾는다.
        # 격자 분할이 동일하게 적용되었으므로 동일한 i,j 키를 가진다. k는 0(유일)
        
        # 1. OCC 윗면(+Z) -> OpenCell 아랫면(-Z) 부착
        if (i,j,0) in b_opencell.blocks:
            site_occ_pz = f"s_BOpenCellCohesive_{i}_{j}_{0}_PZ"
            site_oc_nz  = f"s_BOpenCell_{i}_{j}_{0}_NZ"
            inter_weld_xml.append(f'        <weld site1="{site_occ_pz}" site2="{site_oc_nz}" solref="{tape_solref}" solimp="{tape_solimp}"/>')
            
        # 2. OCC 아랫면(-Z) -> Chassis 윗면(+Z) 부착
        if (i,j,0) in b_chassis.blocks:
            site_occ_nz = f"s_BOpenCellCohesive_{i}_{j}_{0}_NZ"
            site_chas_pz = f"s_BChassis_{i}_{j}_{0}_PZ"
            inter_weld_xml.append(f'        <weld site1="{site_occ_nz}" site2="{site_chas_pz}" solref="{tape_solref}" solimp="{tape_solimp}"/>')

    # --- 5. 낙하 조건 (ISTA 6A 등) 로테이션 연산 ---
    # 파싱된 드랍 모드에 따라 목표 임팩트 벡터(target_pt) 추출
    target_pt = parse_drop_target(drop_mode, box_w, box_h, box_d)
    target_dist = np.linalg.norm(target_pt)
    
    # 목표 지점 벡터가 정확히 전역 중력방향([0, 0, -target_dist])을 향하도록 회전축/각 파출
    rot_axis = np.cross(target_pt, [0, 0, -target_dist])
    
    if np.linalg.norm(rot_axis) < 1e-6:
        # 이미 중력 방향이거나 정반대(180도)인 경우의 예외처리
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

    # 전역 좌표 변환 (초기 낙하 자세 생성)
    wx, wy, wz = get_local_pose([0,0,0], drop_height, rot_axis, angle_rad, target_dist)
    rot_str = f"{rot_axis[0]:.4f} {rot_axis[1]:.4f} {rot_axis[2]:.4f} {np.degrees(angle_rad):.4f}"
    
    # --- 6. XML 파일 작성 ---
    xml_str_io = io.StringIO()
    xml_str_io.write('<mujoco model="discrete_custom_box">\n')
    xml_str_io.write('  <option integrator="implicitfast" timestep="0.001"><flag contact="enable"/></option>\n')
    
    # [시각적 개선] config에 정의된 조명 파라미터를 사용하여 씬 구성
    xml_str_io.write('  <visual>\n')
    h_amb = config.get("light_head_ambient", "0.28 0.28 0.28")
    h_dif = config.get("light_head_diffuse", "0.56 0.56 0.56")
    xml_str_io.write(f'    <headlight ambient="{h_amb}" diffuse="{h_dif}" specular="0.07 0.07 0.07"/>\n')
    # map: 그림자 해상도 및 가시 거리 조정
    xml_str_io.write('    <map znear="0.01"/>\n')
    xml_str_io.write('  </visual>\n')

    xml_str_io.write('  <default>\n')
    # 요소가 많으므로 contact 제외 및 계산 고속화 필수 (Weld에 의한 구속이 뼈대 역할)
    xml_str_io.write('    <joint armature="0.05" damping="1.0"/>\n')
    xml_str_io.write('    <geom friction="0.8" solref="0.02 1.0" solimp="0.9 0.95 0.001"/>\n')
    xml_str_io.write('  </default>\n')
    
    xml_str_io.write('  <worldbody>\n')
    # [조명 설정] 천장에서 아래로 비추는 주 조명 및 보조 조명
    m_dif = config.get("light_main_diffuse", "0.49 0.49 0.49")
    m_amb = config.get("light_main_ambient", "0.21 0.21 0.21")
    s_dif = config.get("light_sub_diffuse", "0.21 0.21 0.21")
    
    xml_str_io.write(f'    <light pos="0 0 6" dir="0 0 -1" directional="false" diffuse="{m_dif}" ambient="{m_amb}" castshadow="true"/>\n')
    xml_str_io.write(f'    <light pos="3 3 5" dir="-1 -1 -1" directional="false" diffuse="{s_dif}" castshadow="false"/>\n')
    
    # 바닥(Floor) 설정: config에서 가져온 solref/solimp/friction 적용
    g_solref = config.get("ground_solref", "0.01 1.0")
    g_solimp = config.get("ground_solimp", "0.9 0.95 0.001 0.5 2")
    g_fric   = config.get("ground_friction", 0.3)
    xml_str_io.write(f'    <geom name="floor" type="plane" size="5 5 0.1" friction="{g_fric}" contype="1" conaffinity="1" solref="{g_solref}" solimp="{g_solimp}"/>\n')
    
    # 최상단 글로벌 패키지 박스 그룹. 여기서만 오프셋/회전이 적용됨.
    xml_str_io.write(f'    <body name="BPackagingBox" pos="{wx:.5f} {wy:.5f} {wz:.5f}" axisangle="{rot_str}">\n')
    xml_str_io.write('      <freejoint/>\n')
    xml_str_io.write('      <geom type="box" size="0.001 0.001 0.001" mass="0.000021" rgba="0.9 0.9 0.9 0.5" contype="2" conaffinity="5" friction="0.8"/>\n')
    
    # 전체 바디 트리 출력
    bodies_xml = root_container.get_worldbody_xml_strings(indent_level=3)
    for line in bodies_xml:
        xml_str_io.write(line + "\n")
        
    xml_str_io.write('    </body>\n')
    xml_str_io.write('  </worldbody>\n')
    
    xml_str_io.write('  <equality>\n')
    # 컴포넌트 내부의 격자 블록들을 이어주는 Weld (PaperBox 내부, Cushion 내부, TV 내부 조립 등)
    internal_weld_xml = root_container.get_weld_xml_strings()
    for line in internal_weld_xml:
        xml_str_io.write(line + "\n")
        
    # 이기종 부품 간의 어셈블리 Weld (테이프)
    for line in inter_weld_xml:
        xml_str_io.write(line + "\n")
    xml_str_io.write('  </equality>\n')
    
    xml_str_io.write('</mujoco>\n')
    
    final_xml_str = xml_str_io.getvalue()
    xml_str_io.close()
    
    with open(export_path, "w", encoding="utf-8") as f:
        f.write(final_xml_str)

    # --- 7. 관성 텐서(CoG, MoI) 시스템 출력 ---
    total_mass, cog, moi, _ = root_container.calculate_inertia()
    print("\n" + "="*60)
    print(f"[Assembly Inertia Report] Mode: {drop_mode}")
    print(f" - Total Mass : {total_mass:.4f} kg")
    print(f" - CoG (x, y, z): {cog[0]:.4f},  {cog[1]:.4f},  {cog[2]:.4f}")
    print(f" - MoI (Ixx, Iyy, Izz): {moi[0]:.6f},  {moi[1]:.6f},  {moi[2]:.6f}")
    print("="*60 + "\n")
    
    return final_xml_str

if __name__ == "__main__":
    out_dir = r"c:\Users\GOODMAN\WHToolsBox\test_box_mujoco"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "test_shapes_check.xml")
    
    print(f"Generating Discrete Model XML to {out_file}...")
    cfg = get_default_config()
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