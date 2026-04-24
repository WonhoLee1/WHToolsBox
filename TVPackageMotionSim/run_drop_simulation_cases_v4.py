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

def test_run_case_1():
    """
    [Case 1] 표준 낙하 테스트 (Golden Case)
    가장 안정적인 물리 계수와 형상 정보를 포함하는 기준 케이스입니다.
    """
    print("\n" + "="*85)
    print("🚀 Running Case 1: Standard Corner 2-3-5 (0.5m)")
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
    cfg["use_postprocess_ui"] = True # 엔진 내부의 구버전 UI 실행 여부
    cfg["use_viewer"] = True      # MuJoCo Viewer(GUI) 실행 여부

    # [3. COMPONENTS OPTIONS] : 각 컴포넌트의 이산화(Meshing) 및 구속 설정
    cfg["chassis_div"]      = [5, 5, 1]    # Chassis(섀시) 부품의 X, Y, Z축 분할 수
    cfg["chassis_use_weld"] = True         # Chassis 내부 블록 간의 Weld 구속 사용 여부
    cfg["opencell_div"]     = [5, 5, 1]    # Open Cell(패널) 부품의 분할 수
    cfg["opencell_use_weld"] = True        # 패널 내부 이산화 블록 간 Weld 구속 활성화
    cfg["opencellcoh_div"]  = [5, 5, 1]    # Open Cell Cover(커버) 부품의 분할 수
    cfg["opencellcoh_use_weld"] = True     # 커버 내부 이산화 블록 간 Weld 구속 활성화
    cfg["cush_div"]         = [5, 5, 3]    # Cushion(EPS/EPP 쿠션) 부품의 분할 수
    cfg["cush_use_weld"]    = True         # 쿠션 내부 블록 간 Weld 구속 활성화
    cfg["include_paperbox"] = False        # 종이 박스 메쉬 모델 활성화
    cfg["box_div"]          = [5, 5, 1]    # Box(외부 박스) 부품의 분할 수
    cfg["box_use_weld"]     = True         # 박스 내부 블록 간 Weld 구속 활성화
    
    # [4. PHYSICS PARAMETERS] : Solver 및 접촉 물성 설정
    cfg["cush_weld_solref_timec"]   = 0.008
    cfg["cush_weld_solref_damprr"]  = 0.8
    cfg["opencell_weld_solref_timec"] = 0.005
    cfg["opencell_weld_solref_damprr"] = 0.5
    cfg["chassis_weld_solref_timec"]  = 0.002
    cfg["chassis_weld_solref_damprr"] = 0.5

    cfg["cush_contact_solref"]    = "0.01 0.8"
    cfg["cush_contact_solimp"]    = "0.1 0.95 0.005 0.5 2"
    cfg["cush_corner_solref"]     = "0.01 0.8"
    cfg["cush_corner_solimp"]     = "0.1 0.95 0.005 0.5 2"
    
    # [5. PLASTICITY & HARDENING]
    cfg["enable_plasticity"]    = True
    cfg["plasticity_ratio"]     = 0.5
    cfg["cush_yield_pressure"]  = 1000.0
    cfg["plastic_hardening_modulus"] = 2000.0
    
    # [6. MASS TOTALS] : (전체 합계: 25.0kg)
    cfg["mass_paper"]   = 4.0
    cfg["mass_cushion"] = 2.0
    cfg["mass_oc"]      = 5.0
    cfg["mass_occ"]     = 0.1
    cfg["mass_chassis"] = 10.0
    cfg["chassis_aux_masses"] = [
        {"name": "InertiaAux_Single", "size": [0.1, 0.1, 0.1], "mass": 3.9, "pos": [0, 0, 0]}
    ]
        
    # [7. GROUND PROPERTIES]
    cfg["ground_solref_timec"] = 0.002
    cfg["ground_solref_damprr"] = 0.001
    cfg["ground_friction"]     = 1.0
    cfg["ground_solimp"] = "0.9 0.99 0.001"

    # [8. SOLVER & REPORTING OPTIONS]
    cfg["sim_integrator"] = "implicitfast"
    cfg["sim_timestep"]   = 0.0012
    cfg["sim_iterations"] = 50
    cfg["sim_noslip_iterations"] = 0
    cfg["sim_tolerance"]  = 1e-5
    cfg["sim_gravity"]    = [0, 0, -9.81]
    cfg["sim_nthread"]    = 4
    cfg["reporting_interval"] = 0.0024
    cfg["sim_duration"] = 1.5

    # [9. AIR FLUIDICS]
    cfg["enable_air_drag"]    = True
    cfg["enable_air_squeeze"] = False

    # [10. AUTO BALANCING]
    cfg["enable_target_balancing"] = True
    cfg["target_mass"] = 25.0
    cfg["num_balancing_masses"] = 8

    # 4. 시뮬레이션 실행
    sim = DropSimulator(config=cfg)
    try:
        sim.log(">> [Case Manager] Golden Case 설정 복원 완료. 시뮬레이션 시작...")
        sim.simulate()
        print("\n✅ Case 1 Finished Successfully.")
    except Exception as e:
        print(f"\n❌ Case 1 Failed: {e}")

def test_run_case_2():
    """
    [Case 2] v2 통합 컨트롤러 연동 테스트 (Corner 2-3-5)
    - 이제 get_default_config()에 최적화된 값이 기본으로 포함되어 있어 코드가 간소화됨
    - 시뮬레이션 종료 후 신형 PySide6 UI 실행
    """
    print("\n" + "="*85)
    print("🚀 Running Case 2: Standard Corner with V2 UI Engine (V4.5)")
    print("="*85)
    
    cfg = get_default_config()

    # V2 UI 연동을 위한 전용 설정만 유지
    cfg.update({
        "use_postprocess_ui": False, # 구버전 UI 끔
        "use_postprocess_v2": True,  # 신형 V2 UI 켬
        "use_viewer": True
    })

    sim = DropSimulator(config=cfg)
    try:
        sim.log(">> [Case Manager] Case 2 (V2 UI Mode) 시작...")
        sim.simulate()
        print("\n✅ Case 2 Finished Successfully.")
    except Exception as e:
        print(f"\n❌ Case 2 Failed: {e}")

def test_run_case_3():
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
    sim.simulate()

if __name__ == "__main__":
    # 1. 기존 UI (Tkinter) 실행 모드
    test_run_case_1()
    
    # 2. 신형 V2 UI (PySide6) 실행 모드
    #test_run_case_2()
    
    # test_run_case_3()
