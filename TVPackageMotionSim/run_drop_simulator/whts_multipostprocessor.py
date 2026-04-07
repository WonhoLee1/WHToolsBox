# -*- coding: utf-8 -*-
"""
[WHTOOLS] Integrated Multi-PostProcessor
통합 포스트 프로세서 실행기입니다.
Engine 모듈의 수치 해석과 UI 모듈의 시각화를 결합하여 최종 대시보드를 구동합니다.
"""

import os
import sys
import numpy as np
from PySide6 import QtWidgets, QtGui

# [WHTOOLS] 현재 디렉토리를 경로에 추가 (부모 디렉토리에서 임포트 대비)
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.append(_current_dir)

# 내부 모듈 임포트
from whts_multipostprocessor_engine import (
    PlateAssemblyManager, 
    ShellDeformationAnalyzer, 
    scale_result_to_mm
)
from whts_multipostprocessor_ui import QtVisualizerV2

def run_demo():
    """리팩토링 검증을 위한 데모 데이터 생성 및 실행"""
    print("[WHTOOLS] Generating Demo Data...")
    n_samples = 100
    times = np.linspace(0, 1.0, n_samples)
    
    # 큐브 형태의 6개 면 데이터 생성 (더미)
    manager = PlateAssemblyManager(times)
    face_names = ["Bottom", "Top", "Front", "Back", "Left", "Right"]
    
    for name in face_names:
        # 3x3 마커 그리드 생성
        mx, my = np.meshgrid(np.linspace(-500, 500, 3), np.linspace(-400, 400, 3))
        off = np.column_stack([mx.ravel(), my.ravel()])
        m_hist = np.zeros((n_samples, 9, 3))
        for f_idx, t in enumerate(times):
            deform = 5.0 * np.sin(2 * np.pi * t) * np.exp(-t*2)
            m_hist[f_idx] = np.column_stack([off, np.full(9, deform)]) + [0, 0, 500]
            
        analyzer = ShellDeformationAnalyzer(W=1200, H=1000, name=name)
        analyzer.m_raw = m_hist
        analyzer.o_data_hint = off
        manager.add_analyzer(analyzer)
        
    manager.run_all()
    return manager

def main():
    """메인 엔트리 포인트"""
    app = QtWidgets.QApplication(sys.argv)
    
    # 아규먼트 처리 (추후 데이터 로딩 로직 확장 가능)
    # 현재는 기본 데모 실행
    mgr = run_demo()
    
    gui = QtVisualizerV2(mgr)
    gui.show()
    
    print("[WHTOOLS] Dashboard Launched Successfully.")
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
