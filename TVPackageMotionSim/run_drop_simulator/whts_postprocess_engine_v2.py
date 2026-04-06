# -*- coding: utf-8 -*-
"""
[WHTOOLS] Integrated Simulation Control Engine v2.0
MuJoCo 시뮬레이션 제어, 파라미터 관리 및 결과 요약 추출을 담당하는 핵심 로직 모듈.
SSR 관련 레거시 의존성을 제거하고, PySide6 UI와의 연동에 최적화되었습니다.
"""

import os
import sys
import json
import numpy as np
import pickle
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# [WHTOOLS] 기초 설정 및 빌더 연동
try:
    from .whts_mapping import TV_COMPONENTS # 필요 시 참조
except ImportError:
    pass

class SimulationControlEngine:
    """
    MuJoCo 시뮬레이션의 실행, 중지 및 결과 관리를 담당하는 엔진 클래스.
    """
    def __init__(self, workspace_dir: str = None):
        self.workspace_dir = workspace_dir or os.getcwd()
        self.current_sim = None
        self.is_running = False
        self.config_history = []
        
    def load_config(self, path: str) -> Dict[str, Any]:
        """지정한 경로의 JSON 설정 파일을 로드합니다."""
        if not os.path.exists(path):
            return {}
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_config(self, path: str, config: Dict[str, Any]):
        """설정 데이터를 JSON 파일로 저장합니다."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)

    def get_summary_from_pkl(self, pkl_path: str) -> List[Dict[str, Any]]:
        """
        저장된 .pkl 결과 파일에서 핵심 물리 지표를 추출하여 반환합니다.
        (SSR 연산 없이 기존 시계열 데이터에서 Peak 값만 추출)
        """
        if not os.path.exists(pkl_path):
            return []
            
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            # DropSimulator 혹은 결과 Dict 구조인지 확인
            metrics = getattr(data, 'metrics', {}) if hasattr(data, 'metrics') else data.get('metrics', {})
            time_hist = getattr(data, 'time_history', []) if hasattr(data, 'time_history') else data.get('time_history', [])
            global_metrics = getattr(data, 'structural_time_series', {}).get('comp_global_metrics', {})
            
            summary_list = []
            for comp_name, m in metrics.items():
                # 1. RRG Peak
                rrg_max = 0.0
                for grid_idx, rrg_hist in m.get('all_blocks_rrg', {}).items():
                    if rrg_hist:
                        local_max = max([abs(v) for v in rrg_hist])
                        if local_max > rrg_max: rrg_max = local_max
                
                # 2. Stress Peak
                stress_max = 0.0
                for grid_idx, s_hist in m.get('all_blocks_s_bend', {}).items():
                    if s_hist:
                        local_max = max(s_hist)
                        if local_max > stress_max: stress_max = local_max

                # 3. PBA Peak
                pba_hist = m.get('max_pba_hist', [])
                pba_max = max(pba_hist) if pba_hist else 0.0
                
                # 4. Global Indices (GTI/GBI)
                g_metrics = global_metrics.get(comp_name, {'gti': [0], 'gbi': [0]})
                gti_max = max(g_metrics.get('gti', [0]) or [0])
                gbi_max = max(g_metrics.get('gbi', [0]) or [0])

                # Status 판단 로직 (Legacy 기준 계승)
                status = "PASS"
                if rrg_max > 3.0: status = "⚠️ 局部 집중"
                if stress_max > 80.0: status = "❗ 항복 위험"
                if gti_max > 8.0: status = "🚨 비틀림 심각"

                summary_list.append({
                    "Component": comp_name,
                    "Max RRG": f"{rrg_max:.2f}",
                    "Max Stress": f"{stress_max:.1f} MPa",
                    "PBA Peak": f"{pba_max:.1f}°",
                    "GTI": f"{gti_max:.1f}",
                    "Status": status
                })
            return summary_list
        except Exception as e:
            print(f">> [EngineV2] PKL Summary Error: {e}")
            return []

    def get_result_files(self, results_dir: str = "results") -> List[Dict[str, str]]:
        """results 디렉토리 내의 .pkl 파일 목록을 반환합니다."""
        if not os.path.exists(results_dir):
            return []
        
        files = []
        for f in os.listdir(results_dir):
            if f.endswith(".pkl"):
                path = os.path.join(results_dir, f)
                mtime = os.path.getmtime(path)
                files.append({
                    "name": f,
                    "path": path,
                    "date": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                })
        return sorted(files, key=lambda x: x['date'], reverse=True)

# [v2.0] PySide6 연동을 위한 비동기 시뮬레이션 Runner
# (UI와 별도 쓰레드에서 실행되어야 UI 스터터링이 없음)

def run_simulation_async(config: Dict[str, Any], callback_log=None, callback_finished=None):
    """
    새로운 쓰레드에서 MuJoCo 시뮬레이션을 실행합니다.
    """
    from run_drop_simulator import DropSimulator
    
    def worker():
        try:
            # 1. 시뮬레이터 인스턴스 생성
            sim = DropSimulator(config=config)
            
            # 2. 로그 리다이렉션 (옵션)
            if callback_log:
                # 몽키 패치 혹은 직접 주입 방식으로 sim.log 호출 가로채기 가능
                orig_log = sim.log
                def new_log(msg, level="INFO"):
                    orig_log(msg, level)
                    callback_log(f"[{level}] {msg}")
                sim.log = new_log

            # 3. 실행
            sim.simulate(enable_UI=False) # Qt UI를 사용할 것이므로 sim 내부 Tkinter UI는 끔
            
            if callback_finished:
                callback_finished(True, "Simulation Completed Successfully")
        except Exception as e:
            if callback_finished:
                callback_finished(False, str(e))

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return t
