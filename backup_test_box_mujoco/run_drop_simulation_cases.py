import os
import sys
import numpy as np
from run_drop_simulation_v3 import DropSimulator
from run_discrete_builder import get_default_config

# =====================================================================
# [WHTOOLS: Case Execution Pipeline for ISTA 6-Amazon LTL]
# =====================================================================

# 본 파일을 수정할 때는 기존 test_run_case_~ 의 내용을 무단 삭제하지 않는다.
# 큰 변화가 필요하면 신규로 test 함수를 만들도록 한다.

def test_run_case_1():
    """
    [Case 1] 표준 낙하 규격 테스트 (ISTA 6-Amazon LTL Corner 2-3-5)
    - _v2.py의 상세 설정을 기반으로 정밀 물리 파라미터 동기화
    """
    print("\n" + "="*85)
    print("🚀 Running Case 1: Standard Corner 2-3-5 (0.5m) - Full Spec Sync")
    print("="*85)
    
    cfg = get_default_config()

    # [1. GEOMETRY OPTIONS]
    cfg["box_w"] = 1.841          
    cfg["box_h"] = 1.103          
    cfg["box_d"] = 0.170          
    cfg["box_thick"] = 0.008      
    cfg["assy_w"] = 1.670         
    cfg["assy_h"] = 0.960         
    cfg["cush_gap"] = 0.005       
    
    # [2. DROP ENV]
    cfg["drop_mode"] = "LTL"
    cfg["drop_direction"] = "Corner 2-3-5" 
    cfg["drop_height"] = 0.5    
    cfg["sim_duration"] = 2.1
    cfg["include_paperbox"] = False 
    cfg["plot_results"] = True    
    cfg["use_viewer"] = True 

    # [3. COMPONENTS OPTIONS] - v2 Spec Sync
    cfg["chassis_div"]      = [3, 3, 1]    
    cfg["chassis_use_weld"] = True        
    
    cfg["oc_div"]           = [3, 3, 1]    
    cfg["oc_use_weld"]      = True         
    
    cfg["occ_div"]          = [3, 3, 1]    
    cfg["occ_use_weld"]     = True         
    
    cfg["cush_div"]         = [5, 5, 3]    
    cfg["cush_use_weld"]    = True         
    
    cfg["box_div"]          = [5, 5, 2]    
    cfg["box_use_weld"]     = False        
    
    # [4. PHYSICS PARAMETERS] - v2 Spec Sync
    cfg["cush_weld_solref_stiff"] = 0.004      
    cfg["cush_weld_solref_damp"]  = 1.0      
    cfg["cush_weld_corner_solref_timec"] = 0.02 
    cfg["cush_weld_corner_solref_dampr"] = 1.0  
    cfg["cush_contact_solref"]    = "0.01 0.8" 
    cfg["cush_contact_solimp"]    = "0.1 0.95 0.005 0.5 2" 
    cfg["cush_corner_solref"]     = "0.01 0.8" 
    cfg["cush_corner_solimp"]     = "0.1 0.95 0.005 0.5 2"

    # [5. PLASTICITY]
    cfg["enable_plasticity"] = True
    cfg["plasticity_ratio"] = 1.0
    cfg["cush_yield_strain"] = 0.10   
    cfg["cush_yield_pressure"] = 40.0  
    cfg["plasticity_algorithm"] = 2  
    
    # [6. MASS TOTALS]
    cfg["mass_paper"] = 4.0
    cfg["mass_cushion"] = 2.0
    cfg["mass_oc"] = 5.0
    cfg["mass_occ"] = 0.1
    cfg["mass_chassis"] = 10.0
        
    # [7. GROUND PROPERTIES]
    cfg["ground_solref_stiff"] = 0.001  
    cfg["ground_solref_damp"]  = 0.0001    
    cfg["ground_friction"]     = 0.1   
    cfg["ground_solimp"] = "0.1 0.95 0.001 0.5 2"
    
    # [8. SOLVER OPTIONS]
    cfg["sim_integrator"] = "implicitfast" 
    cfg["sim_timestep"]   = 0.0013         
    cfg["sim_iterations"] = 50             
    cfg["sim_noslip_iterations"] = 0       
    cfg["sim_tolerance"]  = 1e-5           
    cfg["sim_gravity"]    = [0, 0, -9.81]  
    cfg["sim_nthread"]    = 4              
    cfg["sim_impratio"]   = 5.0             

    # [9. AIR FLUIDICS]
    cfg["air_density"]      = 1.225     
    cfg["air_viscosity"]    = 1.81e-5   
    cfg["air_cd_drag"]      = 1.05      
    cfg["air_cd_viscous"]   = 0.01      
    cfg["air_coef_squeeze"] = 0.2       
    cfg["air_squeeze_hmax"] = 0.20      
    cfg["air_squeeze_hmin"] = 0.005      
    cfg["enable_air_drag"]    = True   
    cfg["enable_air_squeeze"] = False   

    # [10. AUTO BALANCING]
    cfg["enable_target_balancing"] = True
    cfg["target_mass"] = 25.0
    cfg["num_balancing_masses"] = 8 

    sim = DropSimulator(config=cfg)
    try:
        sim.setup()
        sim.simulate()
        sim.plot_results()
        print("\n✅ Case 1 Finished Successfully.")
    except Exception as e:
        print(f"\n❌ Case 1 Failed: {e}")

def test_run_case_2():
    """
    [Case 2] 모서리 낙하 (Edge 3-5) - LTL 모드
    """
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
    sim.setup()
    sim.simulate()

def test_run_case_3():
    """
    [Case 3] 평평한 바닥면 낙하 (Face 3)
    """
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
    sim.setup()
    sim.simulate()

if __name__ == "__main__":
    test_run_case_1()
    # test_run_case_2()
    # test_run_case_3()
