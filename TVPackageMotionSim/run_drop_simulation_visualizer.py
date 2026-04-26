# -*- coding: utf-8 -*-
"""
[WHTOOLS] Integrated Drop Simulation Visualizer (v1.0)
저장된 .pkl 파일을 로드하여 3D 시각화 및 구조 분석을 수행하는 독립형 뷰어입니다.
"""

import os
import sys
import pickle
import glob
from PySide6 import QtWidgets, QtCore

# [WHTOOLS] UTF-8 인코딩 강제 설정 (Windows CP949 대응)
if sys.stdout.encoding != 'utf-8':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except (AttributeError, io.UnsupportedOperation):
        pass

# [WHTOOLS] 프로젝트 루트 경로 추가
curr_dir = os.path.dirname(os.path.abspath(__file__))
if curr_dir not in sys.path:
    sys.path.append(curr_dir)

from run_drop_simulator.whts_multipostprocessor_ui import QtVisualizerV2

def get_latest_result():
    """가장 최근에 생성된 시뮬레이션 결과 파일(.pkl)을 프로젝트 루트 기준 절대 경로로 찾습니다."""
    # [WHT] 실행 경로에 상관없이 결과를 찾을 수 있도록 curr_dir 기준 절대 경로 사용
    results_base = os.path.join(curr_dir, "results")
    if not os.path.exists(results_base):
        return None
        
    result_files = glob.glob(os.path.join(results_base, "rds-*", "simulation_result.pkl"))
    if not result_files:
        return None
    # 파일 수정 시간 순으로 정렬
    result_files.sort(key=os.path.getmtime, reverse=True)
    return result_files[0]

def main():
    app = QtWidgets.QApplication(sys.argv)
    
    # 1. 뷰어 초기화 (빈 상태로 시작)
    viewer = QtVisualizerV2()
    viewer.show()
    
    # 2. 로딩할 타겟 파일 결정 (CLI 인자 우선 -> 최신 결과 자동 탐색 순)
    target_pkl = None
    if len(sys.argv) > 1:
        target_pkl = sys.argv[1]
    
    if not target_pkl or not os.path.exists(target_pkl):
        latest = get_latest_result()
        if latest:
            print(f"[WHTOOLS] No valid input provided. Auto-detecting latest: {latest}")
            target_pkl = latest

    # 3. 파일 자동 로딩 실행
    if target_pkl and os.path.exists(target_pkl):
        print(f"[WHTOOLS] Target File: {target_pkl}")
        # UI 루프 진입 후 안전하게 로딩을 트리거합니다.
        QtCore.QTimer.singleShot(200, lambda: viewer._on_open_file(target_pkl))
    else:
        print("[WHTOOLS] No simulation results found. Ready for manual load.")

    sys.exit(app.exec())

if __name__ == "__main__":
    print("\n" + "="*85)
    print(" 🛠️  WHTOOLS Drop Simulation Visualizer (Standalone Mode)")
    print("="*85)
    main()
