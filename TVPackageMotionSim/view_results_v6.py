# -*- coding: utf-8 -*-
"""
[WHTOOLS] Standalone Result Viewer (v6.7)
시뮬레이터 없이 저장된 pkl 파일만 로드하여 고성능 QtVisualizerV2 대시보드를 즉시 실행합니다.
"""

import os
import sys
import pickle
import numpy as np

# [WHTOOLS] 모듈 경로 추가
sys.path.append(os.path.join(os.getcwd(), "run_drop_simulator"))

from whtb_visualizer_v2 import QtVisualizerV2
from PySide6.QtWidgets import QApplication

def launch_standalone_viewer(pkl_path: str = "results/latest_results.pkl"):
    """[WHTOOLS] 저장된 해석 결과를 로드하여 GUI를 실행합니다."""
    print("="*80)
    print(f"📦 [WHTOOLS] Standalone Viewer: {pkl_path}")
    print("="*80)
    
    if not os.path.exists(pkl_path):
        print(f"❌ Error: Result file not found at {pkl_path}")
        return

    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            
        times = data['times']
        analyzers_data = data['analyzers']
        
        # [WHTOOLS] GUI 호환용 가상 Manager(MOCK) 생성
        class MockAnalyzer:
            def __init__(self, name, results):
                self.name = name
                self.results = results
        
        class MockManager:
            def __init__(self, times, az_data):
                self.times = times
                self.analyzers = [MockAnalyzer(name, res) for name, res in az_data.items()]
        
        manager = MockManager(times, analyzers_data)
        
        # Qt 애플리케이션 실행
        app = QApplication(sys.argv)
        gui = QtVisualizerV2(manager)
        gui.setWindowTitle(f"[WHTOOLS] Standalone Dashboard - {os.path.basename(pkl_path)}")
        gui.show()
        
        print(">> Standalone Dashboard active. Close window to exit.")
        sys.exit(app.exec())
        
    except Exception as e:
        print(f"❌ Critical Error during playback: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    target = os.path.join("results", "latest_results.pkl")
    launch_standalone_viewer(target)
