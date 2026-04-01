import os
import sys
import numpy as np
from run_drop_simulation_v3 import DropSimulator
from run_discrete_builder import get_default_config

# =====================================================================
# [WHTOOLS: Case Execution Pipeline for ISTA 6-Amazon LTL]
# =====================================================================

def test_run_case_1():
    """
    [Case 1] 표준 낙하 규격 테스트 (ISTA 6-Amazon LTL Corner 2-3-5)
    - 가장 취약한 점(Vertex) 충격 시나리오
    - 질량 보정: 25.0kg 타겟 (Auto Balancer 활성화)
    """
    print("\n" + "="*85)
    print("🚀 Running Case 1: Standard Corner 2-3-5 (0.5m)")
    print("="*85)
    
    cfg = get_default_config()

    # [GEOMETRY] 기본 치수 보전
    cfg["box_w"] = 1.841          # 포장 박스 가로 (m)
    cfg["box_h"] = 1.103          # 포장 박스 세로 (m)
    cfg["box_d"] = 0.170          # 포장 박스 깊이 (m)
    cfg["box_thick"] = 0.008      # 박스 골판지 두께 (m)
    cfg["assy_w"] = 1.670         # 내부 제품 가로 (m)
    cfg["assy_h"] = 0.960         # 내부 제품 세로 (m)
    cfg["cush_gap"] = 0.005       # 부품 간 간격
    
    # [DROP ENV]
    cfg["drop_mode"] = "LTL"
    cfg["drop_direction"] = "Corner 2-3-5" 
    cfg["drop_height"] = 0.5    
    cfg["sim_duration"] = 2.1
    
    # [ARCHITECTURE]
    cfg["include_paperbox"] = False 
    cfg["plot_results"] = True    
    cfg["use_viewer"] = True # 첫 케이스는 눈으로 확인
    
    # [RESOLUTION]
    cfg["chassis_div"]      = [3, 3, 1]    
    cfg["cush_div"]         = [5, 4, 3]    
    cfg["box_div"]          = [5, 4, 3]    
    cfg["chassis_use_weld"] = True
    cfg["cush_use_weld"]    = True
    
    # [PHYSICS] 고유 물성 보전 (중요 정보)
    cfg["cush_weld_solref_timec"] = 0.004      # 구조적 벤딩 보호
    cfg["cush_weld_corner_solref_timec"] = 0.02 # 코너 특화
    cfg["cush_contact_solref"]    = "0.01 0.8" 
    cfg["cush_contact_solimp"]    = "0.1 0.95 0.005 0.5 2" 
    
    # [PLASTICITY] v3 소성 변형 알고리즘 설정
    cfg["enable_plasticity"] = True
    cfg["plasticity_ratio"] = 1.0
    cfg["cush_yield_strain"] = 0.01   
    cfg["cush_yield_pressure"] = 80.0  
    
    # [BALANCING] 신규 기능 통합
    cfg["enable_target_balancing"] = True
    cfg["target_mass"] = 25.0
    cfg["num_balancing_masses"] = 8 # 정밀 보정
    
    sim = DropSimulator(config=cfg)
    try:
        sim.setup()
        sim.simulate()
        sim.plot_results()
        print("\n✅ Case 1 Finished.")
    except Exception as e:
        print(f"\n❌ Case 1 Failed: {e}")

# ... (기타 케이스 생략 가능하나 여기서는 전체 백업본으로 저장)
