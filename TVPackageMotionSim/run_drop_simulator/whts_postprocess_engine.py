# -*- coding: utf-8 -*-
"""
[WHTOOLS] Post-Processing Analysis Engine v4.6
UI에서 분리된 데이터 가공 및 수치 해석 핵심 로직 모듈.
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import Any, Dict, List, Optional, Tuple

def get_contour_grid_data(sim: Any, step: int, comp_name: str, metric: str, mode: str = "temporal") -> Optional[Tuple]:
    """
    지정한 시점(step)에서 컴포넌트의 실제 물리 좌표 기반 2D 그리드 데이터를 추출합니다.
    """
    if comp_name not in sim.metrics: return None

    comp_m = sim.metrics[comp_name]
    metric_key_map = {
        "bend": "all_blocks_bend", "twist": "all_blocks_twist",
        "rrg": "all_blocks_rrg", "angle": "all_blocks_angle",
        "bend_x": "all_blocks_bend_x", "bend_y": "all_blocks_bend_y",
        "s_bend": "all_blocks_s_bend", "s_twist": "all_blocks_s_twist",
        "moment": "all_blocks_moment", "energy": "all_blocks_energy"
    }
    data_key = metric_key_map.get(metric, "all_blocks_bend")
    block_data = comp_m.get(data_key, {})
    if not block_data: return None

    # 1. 그리드 인덱스 범위 및 물리 좌표 추출
    all_idxs = list(block_data.keys())
    max_i = max(idx[0] for idx in all_idxs)
    max_j = max(idx[1] for idx in all_idxs)
    
    grid = np.zeros((max_i + 1, max_j + 1))
    pos_3d = np.zeros((max_i + 1, max_j + 1, 3))
    pos_filled = np.zeros((max_i + 1, max_j + 1), dtype=bool)
    
    # [v4.6 FIX] 시뮬레이션에서는 매 스텝 구조 데이터를 수집하므로 
    # 별도의 decimation( // 5) 없이 1:1 매핑 적용
    dec_step = step 
    comp_tree = sim.components.get(comp_name, {})

    for idx, series in block_data.items():
        if not series: continue
        val = abs(series[min(dec_step, len(series) - 1)]) if mode == "temporal" else max(abs(v) for v in series)
        
        # [v4.6] 3D 컴포넌트 대응: 동일 (i, j) 좌표에서 최대값(Peak Perspective) 투영
        grid[idx[0], idx[1]] = max(grid[idx[0], idx[1]], val)
        
        # 물리 좌표 매핑 (최초 1회만 기록하여 그리드 일관성 유지)
        if not pos_filled[idx[0], idx[1]]:
            body_id = comp_tree.get(idx)
            if body_id is not None:
                pos_3d[idx[0], idx[1]] = sim.nominal_local_pos.get(body_id, [0,0,0])
                pos_filled[idx[0], idx[1]] = True

    # 2. 주 평면(Major Plane) 결정 (분산이 큰 두 축 선택)
    variances = np.var(pos_3d.reshape(-1, 3), axis=0)
    dims = np.argsort(variances)[-2:] 
    dims = sorted(dims)
    
    X_phys = pos_3d[:, :, dims[0]]
    Y_phys = pos_3d[:, :, dims[1]]
    
    dim_labels = ["X", "Y", "Z"]
    used_labels = [dim_labels[d] for d in dims]
    
    return X_phys, Y_phys, grid, used_labels

def apply_psr_interpolation(X_orig: np.ndarray, Y_orig: np.ndarray, grid: np.ndarray, 
                           vmin: float, vmax: float, res: int = 40) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    국부 다항식 회귀(PSR)를 이용하여 고해상도 그리드를 생성합니다. (Sharp-Edge SSR)
    """
    flat_x, flat_y, flat_v = X_orig.flatten(), Y_orig.flatten(), grid.flatten()
    tree = cKDTree(np.column_stack([flat_x, flat_y]))
    
    new_x = np.linspace(X_orig.min(), X_orig.max(), res)
    new_y = np.linspace(Y_orig.min(), Y_orig.max(), res)
    X_high, Y_high = np.meshgrid(new_x, new_y)
    grid_high = np.zeros_like(X_high)
    
    for r in range(res):
        for c in range(res):
            px, py = X_high[r, c], Y_high[r, c]
            dist, idxs = tree.query([px, py], k=min(9, len(flat_x)))
            lx, ly = flat_x[idxs] - px, flat_y[idxs] - py
            A = np.column_stack([np.ones(len(lx)), lx, ly, lx**2, lx*ly, ly**2])
            weights = 1.0 / (dist + 1e-6)
            coeffs, _, _, _ = np.linalg.lstsq(np.diag(weights) @ A, np.diag(weights) @ flat_v[idxs], rcond=None)
            grid_high[r, c] = coeffs[0]
            
    grid_high = np.clip(grid_high, vmin, vmax * 1.5)
    return X_high, Y_high, grid_high

def extract_global_summary_data(sim: Any) -> List[Dict[str, Any]]:
    """
    전체 컴포넌트의 피크 지표(PBA, RRG, Stress 등)를 추출하여 요약 리스트를 반환합니다.
    """
    summary_list = []
    comp_metrics = sim.structural_time_series.get('comp_global_metrics', {})
    time_hist = sim.time_history
    
    for comp_name in sorted(list(sim.metrics.keys())):
        m = sim.metrics[comp_name]
        g_metrics = comp_metrics.get(comp_name, {'gti': [0], 'gbi': [0], 'energy': [0]})
        
        gti_max = max(g_metrics.get('gti', [0]) or [0])
        gbi_max = max(g_metrics.get('gbi', [0]) or [0])
        energy_max = max(g_metrics.get('energy', [0]) or [0])
        
        stress_max = 0.0
        for g_idx, stress_hist in m.get('all_blocks_s_bend', {}).items():
            if stress_hist:
                local_s_max = max(stress_hist)
                if local_s_max > stress_max: stress_max = local_s_max
        
        # PBA Peak
        pba_hist = m.get('max_pba_hist', [])
        pba_max = max(pba_hist) if pba_hist else 0.0
        pba_time = 0.0
        pba_dir_info = ""
        if pba_hist:
            pba_idx = int(np.argmax(pba_hist))
            t_idx = min(pba_idx, len(time_hist)-1)
            pba_time = time_hist[t_idx] if t_idx >= 0 else 0.0
            azi = m.get('pba_azi_hist', [0])[pba_idx] if 'pba_azi_hist' in m else 0.0
            ele = m.get('pba_ele_hist', [0])[pba_idx] if 'pba_ele_hist' in m else 0.0
            pba_dir_info = f" [Az:{azi:.0f}, El:{ele:.0f}]"
            
        # RRG Peak
        rrg_max = 0.0
        rrg_time = 0.0
        rrg_block = "-"
        for grid_idx, rrg_hist in m.get('all_blocks_rrg', {}).items():
            if rrg_hist:
                local_max = max(rrg_hist)
                if local_max > rrg_max:
                    rrg_max = local_max
                    idx_max = int(np.argmax(rrg_hist))
                    t_idx = min(idx_max * 5, len(time_hist)-1)
                    rrg_time = time_hist[t_idx] if t_idx >= 0 else 0.0
                    rrg_block = f"{grid_idx}"

        # Status logic from UI
        status = "정상"
        if gti_max > 5.0 or pba_max > 8.0: status = "⚠️ 비틀림 위험"
        if gbi_max > 10.0: status = "⚠️ 과도 굽힘 발생"
        if stress_max > 100.0: status = "❗ 항복 응력 도달 위험"
        if rrg_max > 3.0: status = "❗ 국부 응력 집중"
        if gti_max > 10.0 or gbi_max > 20.0: status = "🚨 구조적 변형 심각"

        summary_list.append({
            "comp": comp_name,
            "pba_peak": f"{pba_max:.2f} ({pba_time:.3f}s){pba_dir_info}",
            "rrg_peak": f"{rrg_max:.2f} ({rrg_time:.3f}s) @ {rrg_block}",
            "max_stress": f"{stress_max:.1f}",
            "total_energy": f"{energy_max:.2f}",
            "gti": f"{gti_max:.2f}",
            "gbi": f"{gbi_max:.2f}",
            "status": status
        })
        
    return summary_list
