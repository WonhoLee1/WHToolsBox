# -*- coding: utf-8 -*-
"""
[WHTOOLS] Drop Simulation v4.0 - Multi-Case Execution Pipeline
ISTA 6-Amazon LTL 등의 규격 낙하 시나리오를 고도화된 v4 패키지로 실행합니다.
본 파일은 기존 'run_drop_simulation_cases.py'의 정밀 파라미터를 100% 계승합니다.
"""

import os
import sys
import numpy as np

# [WHTOOLS] 시뮬레이션 패키지 임포트
# 패키지 실행 환경에 따라 경로 조정 (WHToolsBox 혹은 TVPackageMotionSim 내부 실행 모두 대응)
curr_dir = os.path.dirname(os.path.abspath(__file__))
if curr_dir not in sys.path: sys.path.append(curr_dir)
parent_dir = os.path.dirname(curr_dir)
if parent_dir not in sys.path: sys.path.append(parent_dir)

from run_drop_simulator import DropSimulator
from run_discrete_builder import get_default_config

# =====================================================================
# [WHTOOLS: Case Execution Pipeline for ISTA 6-Amazon LTL]
# =====================================================================

def test_run_case_1(enable_UI: bool = False):
    """
    [Case 1] 표준 낙하 규격 테스트 (ISTA 6-Amazon LTL Corner 2-3-5)
    - 기존의 상세 물리 파라미터(solref, solimp 등)를 모두 유지하며 v4 엔진으로 실행
    """
    print("\n" + "="*85)
    print("🚀 Running Case 1: Standard Corner 2-3-5 (0.5m) - Full Spec Sync (V4)")
    print("="*85)
    
    cfg = get_default_config()

    # [1. GEOMETRY OPTIONS] : 외관 및 어셈블리 형상 정의
    cfg["box_w"] = 1.841          # 박스 외곽 가로 치수 [m]
    cfg["box_h"] = 1.103          # 박스 외곽 세로 치수 [m]
    cfg["box_d"] = 0.170          # 박스 외곽 깊이 치수 [m]
    cfg["box_thick"] = 0.008      # 박스 골판지(더블 월 등) 두께 [m]
    cfg["assy_w"] = 1.670         # 제품(TV) 어셈블리 가로 [m]
    cfg["assy_h"] = 0.960         # 제품(TV) 어셈블리 세로 [m]
    cfg["cush_gap"] = 0.005       # 쿠션과 제품 사이의 조립 공극(Tolerance) [m]
    
    # [2. DROP ENV] : 낙하 시나리오 및 환경 설정
    cfg["drop_mode"] = "LTL"      # 낙하 테스트 모드 (LTL: Less than Truckload)
    cfg["drop_direction"] = "Corner 2-3-5" # 낙하시 지향 방향 (코너 낙하)
    cfg["drop_height"] = 0.5      # 자유 낙하 높이 [m]
    cfg["sim_duration"] = 2.0     # 시뮬레이션 총 시간 [s]
    cfg["include_paperbox"] = False # 종이 박스 메쉬 모델의 시각화/물리 포함 여부
    cfg["use_postprocess_ui"] = True # 시뮬레이션 종료 후 결과 분석 UI 실행 여부
    cfg["use_viewer"] = True      # MuJoCo Viewer(GUI) 실행 여부

    # [3. COMPONENTS OPTIONS] : 각 컴포넌트의 이산화(Meshing) 및 구속 설정
    cfg["chassis_div"]      = [5, 5, 1]    # Chassis(섀시) 부품의 X, Y, Z축 분할 수
    cfg["chassis_use_weld"] = True         # Chassis 내부 블록 간의 Weld 구속 사용 여부
    
    cfg["oc_div"]           = [5, 5, 1]    # Open Cell(패널) 부품의 분할 수
    cfg["oc_use_weld"]      = True         # 패널 내부 이산화 블록 간 Weld 구속 활성화
    
    cfg["occ_div"]          = [5, 5, 1]    # Open Cell Cover(커버) 부품의 분할 수
    cfg["occ_use_weld"]     = True         # 커버 내부 이산화 블록 간 Weld 구속 활성화
    
    cfg["cush_div"]         = [5, 5, 3]    # Cushion(EPS/EPP 쿠션) 부품의 분할 수
    cfg["cush_use_weld"]    = True         # 쿠션 내부 블록 간 Weld 구속 활성화
    
    cfg["box_div"]          = [5, 5, 2]    # Box(외부 박스) 부품의 분할 수
    cfg["box_use_weld"]     = False        # 박스 내부 블록 간 Weld 구속 비활성화 (보통 접촉으로 처리)
    
    # [4. PHYSICS PARAMETERS] : Solver 및 접촉 물성 설정
    cfg["cush_weld_solref_timec"]   = 0.008  # 쿠션 내부 Weld의 시간 상수(Time Constant) Solver Reference
    cfg["cush_weld_solref_damprr"]  = 0.8    # 쿠션 내부 Weld의 감쇠비(Damping Ratio) Solver Reference
    cfg["cush_weld_corner_solref_timec"] = 0.008 # 쿠션 코너 Weld의 시간 상수 (반응 속도)
    cfg["cush_weld_corner_solref_damprr"] = 0.8  # 쿠션 코너 Weld의 감쇠비
    
    cfg["opencell_weld_solref_timec"]   = 0.005   # OpenCell 내부 Weld의 시간 상수
    cfg["opencell_weld_solref_damprr"]  = 0.5    # OpenCell 내부 Weld의 감쇠비
    cfg["chassis_weld_solref_timec"]    = 0.002  # Chassis 내부 Weld의 시간 상수
    cfg["chassis_weld_solref_damprr"]   = 0.3    # Chassis 내부 Weld의 감쇠비

    cfg["cush_contact_solref"]    = "0.01 0.8"  # 쿠션 일반 접촉 Solver Reference
    cfg["cush_contact_solimp"]    = "0.1 0.95 0.005 0.5 2" # 쿠션 일반 접촉 Solver Impedance 곡선 정의
    cfg["cush_corner_solref"]     = "0.01 0.8"  # 쿠션 코너 접촉 전용 Solver Reference
    cfg["cush_corner_solimp"]     = "0.1 0.95 0.005 0.5 2" # 쿠션 코너 접촉 전용 Solver Impedance
    
    # [5. PLASTICITY & HARDENING] : 쿠션의 소성 변형 및 하드닝 설정
    cfg["enable_plasticity"]    = True     # 쿠션 소성 변형(Plasticity) 모델 활성화
    cfg["plasticity_ratio"]     = 0.5      # 소성 변형 진행 속도 비율 (클수록 즉각 변형)
    cfg["cush_yield_pressure"]  = 1000.0    # 초기 항복 압력 (어느 정도 압력 이상에서 영구 변형 시작할지 설정)
    cfg["plastic_hardening_modulus"] = 2000.0 # 가공 경화 계수 (압축될수록 저항력이 수속적으로 상승)
    cfg["plastic_color_limit"]  = 0.08     # 뷰어 시각화 시, 어느 정도의 변형률(5%)에서 파란색(완전 소성)으로 표현할지 결정
    cfg["plastic_max_strain"]   = 0.5      # 최대 허용 수축 한계 (50% 이상 압축 불가 등)
    cfg["debug_plasticity"]     = False    # 개별 소성 변형 로그 활성화 여부
    
    # [6. MASS TOTALS] : 각 부위별 질량 설정 [kg] (전체 합계: 25.0kg)
    cfg["mass_paper"]   = 4.0      # 외부 골판지 박스(Paper Box) 질량
    cfg["mass_cushion"] = 2.0      # 완충재(EPS/EPP Cushion) 전체 질량
    cfg["mass_oc"]      = 5.0      # 내부 LCD Open Cell 패널 질량
    cfg["mass_occ"]     = 0.1      # Open Cell 보호용 커버/필름 질량
    cfg["mass_chassis"] = 10.0     # 내부 샤시(Chassis) 및 주요 기구부 질량
    
    # 보조 질량(Auxiliary Mass)을 추가하여 전체 시스템 중량을 25kg으로 맞춤
    cfg["chassis_aux_masses"] = [
        {"name": "InertiaAux_Single", "size": [0.1, 0.1, 0.1], "mass": 3.9, "pos": [0, 0, 0]}
    ]
        
    # [7. GROUND PROPERTIES] : 지면(충돌 바닥면)의 물리적 반발 특성
    cfg["ground_solref_timec"] = 0.001     # 지면 접촉 시간 상수(Time Constant) 파라미터 (낮을수록 견고함)
    cfg["ground_solref_damprr"]  = 0.0001    # 지면 접촉 감쇠비(Damping Ratio) 파라미터
    cfg["ground_friction"]     = 0.2       # 지면 마찰 계수 (낙하 후 미끄러짐 방지용)
    cfg["ground_solimp"] = "0.1 0.95 0.002 0.5 2" # 지면 접촉 Impedance

    # [8. SOLVER & REPORTING OPTIONS] : 해석 엔진 및 결과 저장 설정
    cfg["sim_integrator"] = "implicitfast" # 적분 방식 (Implicit Fast: 안정성 중심)
    cfg["sim_timestep"]   = 0.0012         # 수치 해석 시간 간격 [s]
    cfg["sim_iterations"] = 50             # Solver 반복 횟수 (정밀도와 속도의 균형)
    cfg["sim_noslip_iterations"] = 0       # 마찰 등의 No-slip 구속에 대한 추가 반복 수
    cfg["sim_tolerance"]  = 1e-5           # 수렴 오차 허용치
    cfg["sim_gravity"]    = [0, 0, -9.81]  # 중력 가속도 [m/s^2]
    cfg["sim_nthread"]    = 4              # 멀티코어 해석 시 사용할 CPU 스레드 수
    cfg["sim_impratio"]   = 1.0            # Solver Impedance 비율 조정
    cfg["reporting_interval"] = 0.0024     # 데이터 저장 및 리포트 간격 [s] (0.0013 * 5 = 5스텝마다)
    cfg["sim_duration"] = 2.2              # 최종 시뮬레이션 연장 시간 설정

    # [9. AIR FLUIDICS] : 내부 공기압 및 공기 저항 로직
    cfg["air_density"]      = 1.225        # 공기 밀도 [kg/m^3]
    cfg["air_viscosity"]    = 1.81e-5      # 공기 점도 [Pa*s]
    cfg["air_cd_drag"]      = 1.05         # 공기 항력 계수 (박스 시뮬레이션용)
    cfg["air_cd_viscous"]   = 0.01         # 공기 점성 마찰 계수
    cfg["air_coef_squeeze"] = 0.2          # Squeeze Film(공기층 압착) 효과 계수
    cfg["air_squeeze_hmax"] = 0.20         # Squeeze 효과 최대 거리 [m]
    cfg["air_squeeze_hmin"] = 0.005        # Squeeze 효과 최소 활성 거리 [m]
    cfg["enable_air_drag"]    = True       # 공기 저항 활성화 여부
    cfg["enable_air_squeeze"] = False      # 공기 압착 효과 활성화 여부

    # [10. AUTO BALANCING] : 목표 실중량 달성을 위한 질량 자동 배분 기능
    cfg["enable_target_balancing"] = True  # 전체 질량 목표치 자동 맞춤 활성화
    cfg["target_mass"] = 25.0              # 전체 세트의 목표 실중량 [kg]
    cfg["num_balancing_masses"] = 8        # 밸런싱을 위해 추가할 지점별 질량 블록 개수

    sim = DropSimulator(config=cfg)
    try:
        sim.log(">> [Case Manager] Case 1 설정 완료. 시뮬레이션 시작...")
        # v4에서는 setup()이 simulate() 내부에 포함되어 있거나, 필요 시 명시적 호출 가능
        sim.simulate(enable_UI=enable_UI)
        print("\n✅ Case 1 Finished Successfully.")
    except Exception as e:
        print(f"\n❌ Case 1 Failed: {e}")

def test_run_case_2(enable_UI: bool = False):
    """[Case 2] 모서리 낙하 (Edge 3-5) - LTL 모드"""
    print("\n" + "="*85)
    print("🚀 Running Case 2: Edge 3-5 (0.5m)")
    print("="*85)
    
    cfg = get_default_config()
    cfg.update({
        "drop_mode": "LTL",
        "drop_direction": "Edge 3-5",
        "drop_height": 0.5,
        "include_paperbox": False,
        "use_viewer": False,
        "enable_target_balancing": True,
        "target_mass": 25.0,
        "num_balancing_masses": 4 
    })
    
    sim = DropSimulator(config=cfg)
    sim.simulate(enable_UI=enable_UI)

def test_run_case_3(enable_UI: bool = False):
    """[Case 3] 평평한 바닥면 낙하 (Face 3)"""
    print("\n" + "="*85)
    print("🚀 Running Case 3: Flat Face 3 (0.5m)")
    print("="*85)
    
    cfg = get_default_config()
    cfg.update({
        "drop_mode": "LTL",
        "drop_direction": "Face 3",
        "drop_height": 0.5,
        "include_paperbox": False,
        "use_viewer": False,
        "enable_target_balancing": True,
        "target_mass": 25.0
    })
    
    sim = DropSimulator(config=cfg)
    sim.simulate(enable_UI=enable_UI)

if __name__ == "__main__":
    # 1. UI 없이 즉시 실행 (기존 방식)
    test_run_case_1(enable_UI=False)
    
    # 2. UI 설정 화면을 먼저 띄우고 실행 (새로운 방식)
    #test_run_case_1(enable_UI=True)
    
    # test_run_case_2()
    # test_run_case_3()
