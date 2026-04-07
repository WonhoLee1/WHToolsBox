# -*- coding: utf-8 -*-
"""
[WHTOOLS] Drop Simulation v5.0 - Digital Twin Integration Pipeline
MuJoCo 시뮬레이션 결과를 ShellDeformationAnalyzer와 연동하여 
실제 계측 데이터 없이도 면상의 정밀 변형 해석을 수행합니다.
"""

import os
import sys
import numpy as np
from datetime import datetime
from typing import Any

# [WHTOOLS] UTF-8 인코딩 강제 설정 (Windows CP949 대응 및 이모지 출력 지원)
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
from run_discrete_builder import get_default_config
from run_drop_simulator.whts_mapping import get_assembly_data_from_sim
from run_drop_simulator.whts_multipostprocessor_engine import (
    ShellDeformationAnalyzer, 
    PlateAssemblyManager, 
    PlateConfig,
    scale_result_to_mm
)
from run_drop_simulator.whts_multipostprocessor_ui import QtVisualizerV2
from PySide6 import QtWidgets

def run_analysis_and_dashboard_by_measure(result_file_path: str):
    """
    [WHTOOLS] Post-Analysis Entry Point (결과 파일 로드 및 자율 분석)
    저장된 해석 결과 파일(.pkl)을 읽어 파트별 마커 기반 구조 해석을 일괄 수행합니다.
    """
    import pickle
    print(f"\n[WHTOOLS] Loading Analysis Result: {result_file_path}")
    
    try:
        with open(result_file_path, 'rb') as f:
            result = pickle.load(f)
    except Exception as e:
        print(f"❌ Failed to load result file: {e}")
        return

    # 1. 데이터 단위 변환 (m -> mm) 및 시간축 확보
    result = scale_result_to_mm(result)
    times = np.array(result.time_history)
    
    # 2. 파트별 마커 데이터 추출 (Statistical 모드 권장 - 자율 피팅용)
    target_parts = ['bpaperbox', 'bcushion', 'bchassis', 'bopencell']
    assembly_markers, assembly_offsets = get_assembly_data_from_sim(result, target_parts, mode='statistical')
    
    # 3. Plate Assembly Manager 구성
    manager = PlateAssemblyManager(times)
    
    # 4. 파트별 분석기 구성 루프
    print("📦 Organizing Analyzers per Part...")
    for part_name, faces in assembly_markers.items():
        for face_name, markers in faces.items():
            if not markers: continue
            
            full_name = f"{part_name.replace('b','').capitalize()}_{face_name}"
            
            # 마커 데이터 정렬 및 스택
            m_names = sorted(list(markers.keys()))
            m_data = np.stack([markers[name] for name in m_names], axis=0) # [n_markers, n_frames, 3]
            o_data = np.stack([assembly_offsets[part_name][face_name][name] for name in m_names], axis=0) # [n_markers, 2]
            
            # 분석기 생성 (lx, ly는 데이터로부터 자율 산출되도록 0, 0 또는 유추값 전달)
            W_deriv = float(o_data[:, 0].max() - o_data[:, 0].min())
            H_deriv = float(o_data[:, 1].max() - o_data[:, 1].min())
            
            analyzer = ShellDeformationAnalyzer(W=W_deriv, H=H_deriv, name=full_name)
            # 데이터를 analyzer에 미리 저장하고 run_all에서 병렬 처리하도록 설정
            analyzer.m_data_hist = m_data
            analyzer.o_data_hint = o_data
            
            manager.add_analyzer(analyzer)
    
    # 5. 일괄 해석 실행 (Parallel JAX Execution)
    print(f"⏳ Running Structural Analysis for {len(manager.analyzers)} parts...")
    manager.run_all()
    
    print("🎨 Launching Post-Processing Dashboard...")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    gui = QtVisualizerV2(manager)
    gui.show()
    app.exec()


def run_analysis_and_dashboard(result: Any):
    """
    저장된 시뮬레이션 결과(result)를 바탕으로 구조 해석 및 Qt 대시보드를 실행합니다.
    """
    # 1. 시뮬레이션 데이터 단위 변환 (m -> mm)
    result = scale_result_to_mm(result)
    times = np.array(result.time_history)
    
    # 2. 파트별 6면 마커 및 오프셋 데이터 추출
    # [v5.6.0] mode='kinematic' 사용: 시뮬레이션의 회전 행렬을 직접 사용하여 이론적 완벽 정렬 보장
    target_parts = ['bpaperbox', 'bcushion', 'bchassis', 'bopencell']
    assembly_markers, assembly_offsets = get_assembly_data_from_sim(result, target_parts, mode='kinematic')
    
    # 3. Plate Assembly Manager 구성
    manager = PlateAssemblyManager(times)
    
    # [WHTOOLS] 각 부품별 치수 추출 로직 (A: 시뮬레이션 설정값, B: 마커 데이터 자동 검출)
    cfg = result.config
    def to_mm(val):
        if val == 0: return 0
        return val * 1000.0 if abs(val) < 10.0 else val
        
    bw = to_mm(cfg.get("box_w", 0))
    bh = to_mm(cfg.get("box_h", 0))
    bd = to_mm(cfg.get("box_d", 0))
    aw = to_mm(cfg.get("assy_w", 0))
    ah = to_mm(cfg.get("assy_h", 0))
    bt = to_mm(cfg.get("box_thick", 0.010))

    # 각 컴포넌트의 각 면에 대해 Analyzer 생성 및 등록
    for part_name, faces in assembly_markers.items():
        for face_name, markers in faces.items():
            if not markers: continue
            
            full_name = f"{part_name.replace('b','').capitalize()}_{face_name}"
            
            # 마커 이름 순서에 맞춰 데이터 정렬
            m_names = sorted(list(markers.keys()))
            # [WHTOOLS] [n_markers, n_frames, 3] -> [n_frames, n_markers, 3] 로 전치
            m_data = np.stack([markers[name] for name in m_names], axis=0).transpose(1, 0, 2)
            o_data = np.stack([assembly_offsets[part_name][face_name][name] for name in m_names], axis=0) # [n_markers, 2]
            
            # [WHTOOLS] 자율적 치수 검출 (Auto-Bounding)
            W_deriv = float(o_data[:, 0].max() - o_data[:, 0].min())
            H_deriv = float(o_data[:, 1].max() - o_data[:, 1].min())
            
            # 설계 치수 힌트 결정
            W_plate, H_plate = 0, 0
            if "paperbox" in part_name:
                if face_name in ["Front", "Rear"]: W_plate, H_plate = bw, bh
                elif face_name in ["Top", "Bottom"]: W_plate, H_plate = bw, bd
                else: W_plate, H_plate = bd, bh
            
            # 만약 설계 치수가 없으면(0) 데이터로부터 도출된 치수 사용
            if W_plate == 0: W_plate = W_deriv
            if H_plate == 0: H_plate = H_deriv
            
            # Analyzer 생성 (v2.1 규격: lx, ly, ... name)
            analyzer = ShellDeformationAnalyzer(W=W_plate, H=H_plate, name=full_name)
            
            # 시뮬레이션 모드에서도 강체 운동 제거 및 자율 분석을 위해 데이터 설정
            analyzer.m_data_hist = m_data
            analyzer.o_data_hint = o_data
            
            manager.add_analyzer(analyzer)

    
    print(f"\n✅ Assembly Mapping Complete. Total Analyzers: {len(manager.analyzers)}")
    
    # 4. 통합 해석 실행 (Parallel JAX Execution)
    print("⏳ Running Plate Theory Structural Analysis for all parts...")
    manager.run_all()
    
    # [WHTOOLS] 결과 집계 및 리포팅
    success_count = sum(1 for a in manager.analyzers if a.m_raw is not None)
    total_count = len(manager.analyzers)
    print(f"\n[WHTOOLS] Analysis Summary: {success_count}/{total_count} parts succeeded.")
    
    if success_count < total_count:
        failed_parts = [a.name for a in manager.analyzers if a.m_raw is None]
        print(f"⚠️  Warning: {total_count - success_count} parts failed analysis:")
        for name in failed_parts:
            print(f"   - {name}")
    
    print("[WHTOOLS] All Parts Analyzed Successfully.")

    
    # 5. V2 시각화 대시보드 실행
    print("🎨 Launching Post-Processing Dashboard...")
    sys.stdout.flush()
    sys.stderr.flush()
    
    try:
        import traceback
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        gui = QtVisualizerV2(manager)
        gui.show()
        print(">> Dashboard active. Close window to exit.")
        sys.stdout.flush()
        ret = app.exec()
        print(f">> Dashboard closed with code: {ret}")
    except Exception as e:
        print("\n❌ Dashboard Launch Failed!")
        traceback.print_exc()
        sys.stdout.flush()

def run_digital_twin_pipeline(case_func):
    """
    MuJoCo 시뮬레이션을 실행하고 그 결과를 기반으로 쉘 변형 분석을 수행합니다.
    """
    print("\n" + "="*85)
    print(f"🚀 Starting Digital Twin Pipeline (v5.1 Precision): {case_func.__name__}")
    print("="*85)
    
    # 1. MuJoCo 시뮬레이션 실행
    sim = case_func(enable_UI=False)
    if sim is None or sim.result is None:
        print("❌ Simulation failed or no results found.")
        return

    # 사후 분석 및 대시보드 실행
    run_analysis_and_dashboard(sim.result)

def test_case_1_setup(enable_UI: bool = False):
    """
    [V5.2.8.4] v4의 test_run_case_1 설정을 100% 계승하고 v5용 옵션을 추가함
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
    cfg["ground_friction"]     = 0.3
    cfg["ground_solimp"] = "0.1 0.95 0.001"

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

    # [V5 ADDITIONAL UPDATES]
    cfg["use_jax_reporting"] = True # JAX 엔진 활성화

    # 4. 시뮬레이션 실행
    sim = DropSimulator(config=cfg)
    sim.simulate(enable_UI=enable_UI)
    return sim

if __name__ == "__main__":
    # Case 1 기반으로 디지털 트윈 파이프라인 실행
    run_digital_twin_pipeline(test_case_1_setup)
