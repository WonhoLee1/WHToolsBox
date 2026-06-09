# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pickle
import jax.numpy as jnp
from datetime import datetime
from typing import Any

from PySide6 import QtWidgets

# Assuming DropSimResult is available if needed, but 'result' is just passed.
from run_drop_simulator.whts_mapping import get_assembly_data_from_sim
from run_drop_simulator.whts_multipostprocessor_engine import (
    ShellDeformationAnalyzer, 
    PlateAssemblyManager, 
    PlateConfig,
    scale_result_to_mm
)
from run_drop_simulator.whts_exporter import WHToolsExporter
from run_drop_simulator.whts_multipostprocessor_ui import QtVisualizerV2

def run_analysis_pipeline(result: Any, curr_dir: str, standalone: bool = False,
                          mode: str = 'visualizer', visualizer_settings: dict = None):
    """
    [WHTOOLS] Minimalist Analysis Entry Point
    설계 치수나 오프셋 힌트 없이, 오직 마커의 3D 궤적(Trajectories) 데이터만으로 자율 분석을 수행합니다.

    Args:
        result: DropSimResult 시뮬레이션 결과 객체
        curr_dir: TVPackageMotionSim 루트 디렉토리 경로
        standalone: True 시 app.exec() 호출 후 sys.exit(0)
        mode: 출력 모드
            - 'paraview': VTKHDF 내보내기 + ParaView 자동 실행
            - 'visualizer': WHT Visualizer (Qt 대시보드) 실행
        visualizer_settings: OpenSettingsDialog.get_settings() 반환값 (선택)
            - 'parts_data': 파트별 마커 설정 리스트
            - 'mode': 마커 추출 모드 ('statistical' | 'direct')
            - 'res': 해석 해상도 (sol.res)
            - 'interpretation_mode': 'v5' | 'v6'
    """
    # 1. 데이터 단위 변환 (m -> mm)
    result = scale_result_to_mm(result)
    times = np.array(result.time_history)
    
    # 2. 마커 데이터만 추출 (v5와 달리 offsets 정보는 의도적으로 사용하지 않음)
    # components 키를 우선 사용, 없으면 b-prefix 레거시 이름으로 fallback
    if hasattr(result, 'components') and result.components:
        target_parts = list(result.components.keys())
        print(f"  [Pipeline] components: {target_parts}")
        for pn, bmap in result.components.items():
            print(f"    {pn}: {len(bmap)} bodies")
    else:
        target_parts = ['bcushion', 'bchassis', 'bopencell']
        print(f"  [Pipeline] no components — using fallback names: {target_parts}")

    pos = getattr(result, 'pos_hist', None)
    quat = getattr(result, 'quat_hist', None)
    print(f"  [Pipeline] pos_hist: {None if pos is None else np.array(pos).shape}")
    print(f"  [Pipeline] quat_hist: {None if quat is None else np.array(quat).shape}")

    # [v6] 마커 추출 모드: visualizer_settings 있으면 그 설정 우선, 없으면 statistical 기본
    marker_mode = 'statistical'
    if visualizer_settings and 'mode' in visualizer_settings:
        marker_mode = visualizer_settings['mode']
    assembly_markers, _ = get_assembly_data_from_sim(result, target_parts, mode=marker_mode)
    
    # 3. Plate Assembly Manager 구성
    manager = PlateAssemblyManager(times)
    
    print(f"\n[v6] Organizing Analyzers with MINIMAL information...")
    for part_name, faces in assembly_markers.items():
        non_empty = {f: len(m) for f, m in faces.items() if m}
        if non_empty:
            print(f"  {part_name}: {non_empty}")
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
    
    print(f"Setup Complete. Total Analyzers: {len(manager.analyzers)}")
    
    if not manager.analyzers:
        print("[WARNING] No valid parts with markers found for analysis. Exiting pipeline.")
        return None

    # 4. 자율 통합 해석 실행
    print("Running Autonomous Structural Analysis (JAX Accelerated)...")
    manager.run_all()
    
    # [WHTOOLS] [NEW] JAX 정밀 리포트 출력 (Markers & Stress 확인용)
    manager.show_report()
    
    # [WHTOOLS] 결과 영구 저장 (v6.7 Persistence) - 우선순위 최상위로 조정
    try:
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
        print(f"[WHTOOLS] Results persisted to: {res_path}", flush=True)
    except Exception as save_err:
        print(f"Failed to persist results: {save_err}", flush=True)

    # ── 모드별 분기 ──
    if mode == 'paraview':
        return _launch_paraview_mode(manager, curr_dir)
    else:
        return _launch_visualizer_mode(manager, standalone, visualizer_settings)


def _launch_paraview_mode(manager: PlateAssemblyManager, curr_dir: str):
    """
    [WHTOOLS] ParaView 모드: VTKHDF v2.2 내보내기 + ParaView 자동 실행
    """
    try:
        export_path = os.path.join(curr_dir, "results", f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        exporter = WHToolsExporter(manager)
        exporter.register_paraview_macro()
        vtkhdf_path = exporter.export_to_vtkhdf(os.path.join(export_path, "vtk"))
        exporter.export_summary()
        
        # ParaView 자동 실행
        if vtkhdf_path:
            exporter.launch_paraview(vtkhdf_path)
        return vtkhdf_path
    except Exception as e:
        print(f"Export failed: {e}")
        return None


def _launch_visualizer_mode(manager: PlateAssemblyManager, standalone: bool = False,
                             visualizer_settings: dict = None):
    """
    [WHTOOLS] WHT Visualizer 모드: Qt 대시보드 실행

    Args:
        manager: 분석 완료된 PlateAssemblyManager
        standalone: True 시 app.exec() + sys.exit(0) 호출
        visualizer_settings: OpenSettingsDialog.get_settings() 결과 (선택)
            interpretation_mode 'v6' 여부에 따라 res 설정 적용
    """
    print("\nLaunching WHT Visualizer...", flush=True)

    # visualizer_settings의 res 값이 있으면 각 analyzer의 sol.res에 적용
    if visualizer_settings and 'res' in visualizer_settings:
        res_override = visualizer_settings['res']
        for az in manager.analyzers:
            if az.sol is not None:
                az.sol.res = res_override

    gui = None
    try:
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        gui = QtVisualizerV2(manager)
        gui.show()
        print(">> Dashboard active.")

        if standalone:
            print(">> Close window to exit.")
            app.exec()
            sys.stdout.flush()
            sys.exit(0)
    except Exception as e:
        print(f"\nDashboard Launch Skipped: {e}")

    return gui
