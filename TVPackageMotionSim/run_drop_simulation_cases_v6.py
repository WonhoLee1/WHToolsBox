# -*- coding: utf-8 -*-
"""
[WHTOOLS] Drop Simulation v6.0 - Autonomous Structural Analysis & Multi-Format Export
MuJoCo 시뮬레이션 결과를 최소한의 정보(마커 궤적)만으로 자율 분석하고,
VTKHDF 및 GLB 포맷으로 결과를 내보내는 차세대 파이프라인입니다.
"""

import os
import sys
import numpy as np
import pickle
import jax.numpy as jnp
from datetime import datetime
from typing import Any

# [WHTOOLS] UTF-8 인코딩 강제 설정 (표준 출력/에러)
if sys.stdout.encoding != 'utf-8':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except (AttributeError, io.UnsupportedOperation):
        pass

# [WHTOOLS] 경로 설정
curr_dir = os.path.dirname(os.path.abspath(__file__))
if curr_dir not in sys.path: sys.path.append(curr_dir)

from run_drop_simulator import DropSimulator
from run_discrete_builder import get_default_config, get_rgba_by_name, calculate_plate_twist_weld_params
from run_drop_simulator.whts_mapping import get_assembly_data_from_sim
from run_drop_simulator.whts_multipostprocessor_engine import (
    ShellDeformationAnalyzer, 
    PlateAssemblyManager, 
    PlateConfig,
    scale_result_to_mm
)
from run_drop_simulator.whts_exporter import WHToolsExporter
from run_drop_simulator.whts_multipostprocessor_ui import QtVisualizerV2
from PySide6 import QtWidgets
import run_drop_simulator.whts_multipostprocessor_engine as eng
print(f"🔍 [DEBUG] Engine Path: {eng.__file__}", flush=True)

def run_analysis_and_dashboard_minimal(result: Any):
    """
    [WHTOOLS] Minimalist Analysis Entry Point
    설계 치수나 오프셋 힌트 없이, 오직 마커의 3D 궤적(Trajectories) 데이터만으로 자율 분석을 수행합니다.
    """
    # 1. 데이터 단위 변환 (m -> mm)
    result = scale_result_to_mm(result)
    times = np.array(result.time_history)
    
    # 2. 마커 데이터만 추출 (v5와 달리 offsets 정보는 의도적으로 사용하지 않음)
    target_parts = ['bpaperbox', 'bcushion', 'bchassis', 'bopencell']
    # [v6] mode='statistical' 사용: 시뮬레이션의 회전 행렬 도움 없이 자율 정렬 시도
    assembly_markers, _ = get_assembly_data_from_sim(result, target_parts, mode='statistical')
    
    # 3. Plate Assembly Manager 구성
    manager = PlateAssemblyManager(times)
    
    print("\n📦 [v6] Organizing Analyzers with MINIMAL information...")
    for part_name, faces in assembly_markers.items():
        for face_name, markers in faces.items():
            if not markers: continue
            
            full_name = f"{part_name.replace('b','').capitalize()}_{face_name}"
            
            # 마커 이름 순서에 맞춰 데이터 정렬 및 스택
            m_names = sorted(list(markers.keys()))
            m_data = np.stack([markers[name] for name in m_names], axis=0).transpose(1, 0, 2)
            
            # [WHTOOLS] [CRITICAL] 파트별 고유 물성치를 라이브러리에서 가져와 적용합니다.
            p_cfg = PlateConfig.from_simulation_data(result, full_name)
            analyzer = ShellDeformationAnalyzer(
                W=0, H=0, 
                thickness=p_cfg.thickness, 
                E=p_cfg.youngs_modulus, 
                nu=p_cfg.poisson_ratio, 
                name=full_name
            )
            analyzer.m_data_hist = m_data
            
            manager.add_analyzer(analyzer)
    
    print(f"✅ Setup Complete. Total Analyzers: {len(manager.analyzers)}")
    
    # 4. 자율 통합 해석 실행
    print("⏳ Running Autonomous Structural Analysis (JAX Accelerated)...")
    manager.run_all()
    
    # [WHTOOLS] [NEW] JAX 정밀 리포트 출력 (Markers & Stress 확인용)
    manager.show_report()
    
    # [WHTOOLS] 결과 영구 저장 (v6.7 Persistence) - 우선순위 최상위로 조정
    try:
        import pickle
        res_path = os.path.join(curr_dir, "results", "latest_results.pkl")
        if not os.path.exists(os.path.join(curr_dir, "results")): 
            os.makedirs(os.path.join(curr_dir, "results"))
        
        # [WHTOOLS] 가벼운 저장을 위해 float32 변환 및 핵심 데이터 갈무리
        def lightweight_results(res_dict):
            diet_res = {}
            for k, v in res_dict.items():
                if isinstance(v, (np.ndarray, jnp.ndarray)):
                    arr = np.array(v)
                    # 정수·불리언 배열은 타입 유지, 부동소수점만 float32로 다운샘플
                    if np.issubdtype(arr.dtype, np.floating):
                        diet_res[k] = arr.astype(np.float32)
                    else:
                        diet_res[k] = arr
                else:
                    diet_res[k] = v
            return diet_res

        dump_data = {
            'times': manager.times.astype(np.float32),
            'analyzers': {a.name: lightweight_results(a.results) for a in manager.analyzers if a.results}
        }
        with open(res_path, 'wb') as f:
            pickle.dump(dump_data, f)
        print(f"✅ [WHTOOLS] Results persisted to: {res_path}", flush=True)
    except Exception as save_err:
        print(f"⚠️ Failed to persist results: {save_err}", flush=True)

    # 5. [NEW] 전문 데이터 내보내기 (VTKHDF, GLB)
    try:
        export_path = os.path.join(curr_dir, "results", f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        exporter = WHToolsExporter(manager)
        exporter.register_paraview_macro()
        vtkhdf_path = exporter.export_to_vtkhdf(os.path.join(export_path, "vtk"))
        exporter.export_to_glb(os.path.join(export_path, "glb"))
        exporter.export_summary()
        
        # ParaView Dashboard 자동 실행
        exporter.launch_paraview(vtkhdf_path)
    except Exception as e:
        print(f"⚠️ Export skipped/failed: {e}")

    # 6. 대시보드 시각화 실행 (GUI 환경에서만 실행)
    print("\n🎨 Launching Post-Processing Dashboard (Visual Verification)...", flush=True)
    try:
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        gui = QtVisualizerV2(manager)
        gui.show()
        print(">> Dashboard active. Close window to exit.")
        app.exec()
    except Exception as e:
        print(f"\n⚠️ Dashboard Launch Skipped: {e}")

    # [WHTOOLS] 표준 정상 종료 처리
    sys.stdout.flush()
    sys.exit(0)

def run_digital_twin_pipeline_v6(case_func):
    """
    MuJoCo 시뮬레이션 결과를 기반으로 v6 최소 정보 분석 파이프라인을 실행합니다.
    """
    print("\n" + "="*85)
    print(f"🚀 Digital Twin Pipeline v6.0 (Autonomous): {case_func.__name__}")
    print("="*85)
    
    sim = case_func()
    if sim is None or sim.result is None:
        print("❌ Simulation failed.")
        return

    # [v6] 최소 정보 기반 분석 호출
    #run_analysis_and_dashboard_minimal(sim.result)

def test_case_1_setup():
    """
    [V5.2.8.4] v4의 test_run_case_1 설정을 100% 계승하고 v5용 옵션을 추가함
    [Case 1] 표준 낙하 테스트 (Golden Case)
    가장 안정적인 물리 계수와 형상 정보를 포함하는 기준 케이스입니다.

    심플모드로 openchell, chassis 변형 미고려하고, cushion은 이산화 최소화 한다.
    """
    print("\n" + "="*85)
    print("🚀 Running Case 1: Standard Corner 2-3-5 (0.5m)")
    print("="*85)
    
    cfg = get_default_config()
    cfg["use_viewer"] = True  # 인터랙티브 모드 활성화 (컨트롤 패널 표시)
    
    # [1. GEOMETRY OPTIONS] : 외관 및 어셈블리 형상 정의
    cfg["box_w"] = 2.056          # 박스 외곽 가로 치수 [m]
    cfg["box_h"] = 1.200          # 박스 외곽 세로 치수 [m]
    cfg["box_d"] = 0.178          # 박스 외곽 깊이 치수 [m]
    cfg["box_thick"] = 0.008      # 박스 골판지(더블 월 등) 두께 [m]
    cfg["assy_w"] = 1.892         # 제품(TV) 어셈블리 가로 [m]
    cfg["assy_h"] = 1.082         # 제품(TV) 어셈블리 세로 [m]
    cfg["cush_gap"] = 0.0001       # 쿠션과 제품 사이의 조립 공극(Tolerance) [m]
    # [Geometry]
    cfg["opencell_d"] = 0.012        # Open Cell의 두께 (기본값: 12mm)
    cfg["opencellcoh_d"] = 0.002     # Open Cell Cohesion(Tape 등)의 두께
    cfg["chassis_d"] = 0.035         # Chassis의 두께
    cfg["occ_ithick"] = 0.030        # 인터페이스 두께 관련 변수로 추정
    # [2. DROP ENV] : 낙하 시나리오 및 환경 설정
    cfg["drop_mode"] = "LTL"      # 낙하 테스트 모드 (LTL: Less than Truckload)
    cfg["drop_direction"] = "Corner 2-3-5" # 낙하시 지향 방향 (코너 낙하)
    cfg["drop_height"] = 0.3      # 자유 낙하 높이 [m]
    cfg["use_postprocess_ui"] = False  # 엔진 내부의 구버전 UI 실행 여부
    cfg["use_viewer"] = True          # MuJoCo Viewer(GUI) 실행 여부
    
    # [v6.1] Posture Tilt 옵션 테스트 (LTL 코너 낙하 시 유효)
    cfg["initial_tilt_deg"] = 0.0           # 수직에서 5도 기울임
    cfg["initial_tilt_azimuth_deg"] = 0.0 # 45도 방향으로 기울임

    # [8. SOLVER & REPORTING OPTIONS]
    cfg["sim_integrator"] = "implicitfast"
    cfg["sim_timestep"]   = 0.0012
    cfg["sim_iterations"] = 50
    cfg["sim_noslip_iterations"] = 0
    cfg["sim_tolerance"]  = 1e-5
    cfg["sim_gravity"]    = [0, 0, -9.81]
    cfg["sim_nthread"]    = 4
    cfg["reporting_interval"] = 0.0024
    cfg["sim_duration"] = 2.0

    # [9. AIR FLUIDICS]
    cfg["enable_air_drag"]    = True
    cfg["enable_air_squeeze"] = True

    # [WHTOOLS] PREMIUM VISUALS: Fog & Infinite Ground Effect
    cfg["visual"] = {
        "fogstart": 3.0,              # 3m 지점부터 안개 시작
        "fogend": 10.0,               # 10m 지점에서 완전 안개 (지면 경계 삭제)
        "skybox_rgba": "0.6 0.6 0.6", # 밝은 그레이 배경 (기존보다 밝게)
    }

    # [3. COMPONENTS OPTIONS] : 각 컴포넌트의 설정 (Meshing, Weld, Mass) 통합 관리
    cfg["components"] = {
        "paper"         : {"div": [5, 5, 3], "use_weld": True, "mass": 4.0,  "rgba": get_rgba_by_name("paper", 1.0)},
        "cushion"       : {"div": [5, 5, 3], "use_weld": True, "mass": 3.0,  "rgba": "0.8 0.8 0.8 0.6"},
        "opencell"      : {"div": [4, 4, 1], "use_weld": True, "mass": 5.0,  "rgba": get_rgba_by_name("black", 1.0)},
        "opencellcoh"   : {"div": [4, 4, 1], "use_weld": True, "mass": 0.1,  "rgba": get_rgba_by_name("red", 0.4), "enable_btm_weld": False},
        "chassis"       : {"div": [4, 4, 1], "use_weld": True, "mass": 10.0, "rgba": "0.0 0.2 0.4 1.0"},
    }
    cfg["include_paperbox"] = False        # 종이 박스 메쉬 모델 활성화

    # fast mode
    '''
    cfg["components"] = {
        "paper"         : {"div": [3, 3, 3], "use_weld": True, "mass": 4.0,  "rgba": get_rgba_by_name("paper", 1.0)},
        "cushion"       : {"div": [3, 3, 3], "use_weld": True, "mass": 3.0,  "rgba": "0.8 0.8 0.8 0.6"},
        "opencell"      : {"div": [3, 3, 1], "use_weld": False, "mass": 5.0,  "rgba": get_rgba_by_name("black", 1.0)},
        "opencellcoh"   : {"div": [3, 3, 1], "use_weld": False, "mass": 0.1,  "rgba": get_rgba_by_name("red", 0.4), "enable_btm_weld": True},
        "chassis"       : {"div": [3, 3, 1], "use_weld": False, "mass": 10.0, "rgba": "0.0 0.2 0.4 1.0"},
    }
    cfg["sim_integrator"] = "euler"
    cfg["sim_iterations"] = 30
    '''
    cfg["include_paperbox"] = False        # 종이 박스 메쉬 모델 활성화
    # [4. CONTACT & PAIR PARAMETERS] : 명시적 접촉 쌍 설정 (A1/A2 통합 점검)
    common_friction = [0.9, 0.9]
    p_solref = [-25000.0,-1200.0]
    p_solimp = [0.90, 0.95, 0.001, 0.5, 2]
    cfg["contacts"] = {
        ("ground", "cushion")       : {"friction": common_friction, "solref": [0.001, 1.0], "solimp": p_solimp},
        ("ground", "cushion_edge")  : {"friction": common_friction, "solref": [0.001, 1.0], "solimp": p_solimp},
        ("ground", "paper")         : {"friction": common_friction, "solref": [0.001, 1.0], "solimp": p_solimp},
        ("cushion", "opencell")     : {"friction": common_friction, "solref": p_solref, "solimp": p_solimp},
        ("cushion", "chassis")      : {"friction": common_friction, "solref": p_solref, "solimp": p_solimp},        
        ("cushion", "paper")        : {"friction": common_friction, "solref": [0.001, 1.0], "solimp": p_solimp},
    }
    
    # [WHTOOLS] 실제 물성(E, h_real) 기반 인장 강성 산출 + 실험치(f_target) 기반 벤딩 튜닝
    # Chassis: 0.6t, E=170GPa / OpenCell: 1.0t, E=70GPa
    k_oc, d_oc, ts_oc = calculate_plate_twist_weld_params(
        mass=cfg["components"]["opencell"]["mass"],
        width=cfg["assy_w"],
        height=cfg["assy_h"],
        thickness=cfg["opencell_d"],
        div=cfg["components"]["opencell"]["div"],
        E_real=70e9,          # 실제 유리 탄성계수
        real_thickness=0.001, # 실제 유리 두께 (1mm)
        target_freq_hz=1.0,   # 판 전체 목표 벤딩 진동수 (1Hz)
        zeta=0.05
    )
    k_chas, d_chas, ts_chas = calculate_plate_twist_weld_params(
        mass=cfg["components"]["chassis"]["mass"],
        width=cfg["assy_w"],
        height=cfg["assy_h"],
        thickness=cfg["chassis_d"],
        div=cfg["components"]["chassis"]["div"],
        E_real=170e9,         # 실제 Chassis 탄성계수 (Steel계열)
        real_thickness=0.0006, # 실제 Chassis 두께 (0.6mm)
        target_freq_hz=4.0,   # 판 전체 목표 벤딩 진동수 (4Hz)
        zeta=0.05
    )

    cfg["welds"] = {
        "paper"          : {"solref": [0.010, 1.00], "solimp": [0.10, 0.95, 0.01, 0.5, 2]},
        "cushion"        : {"solref": p_solref, "solimp": p_solimp},
        "cushion_corner" : {"solref": p_solref, "solimp": p_solimp},
        "opencell"       : {"solref": [k_oc, d_oc], "solimp": [0.10, 0.95, 0.1, 0.5, 2], "torquescale": ts_oc},
        "opencellcoh"    : {"solref": [-50000.0, -500.0], "solimp": [0.10, 0.95, 0.01, 0.5, 2]},
        "chassis"        : {"solref": [k_chas, d_chas], "solimp": [0.10, 0.99, 0.1, 0.5, 2], "torquescale": ts_chas},
    }
    
    # [5. PLASTICITY & HARDENING]
    cfg["enable_plasticity"]    = True
    cfg["plasticity_ratio"]     = 0.3
    cfg["cush_yield_pressure"]  = 25000.0
    cfg["plastic_hardening_modulus"] = 300000.0
    
    # [6. MASS TOTALS] : (전체 합계: 25.0kg)
    # [6. MASS TOTALS & AUTO BALANCING]
    cfg["components_balance"] = {
        "target_mass": 42.2,
        "target_inertia": [3.0, 8.0, 14.0],
        "target_cog": [0.001, 0.007, 0.010],  # 10cm 편심 배치 시도
        "count": 8
    }
    # analyze_and_balance_components가 실행되면, 위 설정을 바탕으로 aux 질량이 생성되어 component_aux에 추가됩니다.

    # [7. GROUND PROPERTIES]
    # (Unused legacy keys removed)


    # [WHTOOLS] 0. 내부 컴포넌트 관성 측정 및 Auto-Balancing 확인
    from run_discrete_builder.whtb_physics import analyze_and_balance_components
    cfg = analyze_and_balance_components(cfg, verbose=True)

    # 4. 시뮬레이션 실행
    sim = DropSimulator(config=cfg)
    sim.simulate()
    return sim

if __name__ == "__main__":
    # Case 1 기반으로 v6 파이프라인 실행
    run_digital_twin_pipeline_v6(test_case_1_setup)
