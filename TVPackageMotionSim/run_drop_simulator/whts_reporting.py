import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Tuple, Union

def compute_structural_step_metrics(sim: Any) -> None:
    """
    각 시뮬레이션 타임스텝에서 부품별/블록별 구조적 변형 지표를 연산합니다.
    RRG(Relative Rotation Gradient) 및 PBA(Principal Bending Axis)를 포함합니다.
    
    Args:
        sim (Any): DropSimulator 인스턴스
    """
    d = sim.data
    m = sim.model
    inv_root_mat = d.xmat[sim.root_id].reshape(3, 3).T
    
    global_rot_vectors = []
    step_rrg_max = 0.0
    step_gti_max = 0.0
    step_gbi_max = 0.0

    for comp_name, comp_metric in sim.metrics.items():
        list_of_angles = []
        step_deviation_cache = {}
        
        for grid_idx, body_uid in sim.components[comp_name].items():
            block_mat = d.xmat[body_uid].reshape(3, 3)
            # 중심 바디 대비 상대 회전
            relative_rot = inv_root_mat @ block_mat
            
            if comp_metric['block_nominal_mats'][grid_idx] is None:
                comp_metric['block_nominal_mats'][grid_idx] = relative_rot.copy()
            
            # 초기 상태 대비 편차 행렬
            deviation_mat = comp_metric['block_nominal_mats'][grid_idx].T @ relative_rot
            step_deviation_cache[grid_idx] = deviation_mat
            
            # Bending(Tilt) & Twist(Torsion) 분해
            bend_deg = np.degrees(np.arccos(np.clip(deviation_mat[2, 2], -1.0, 1.0)))
            twist_deg = np.degrees(np.arctan2(deviation_mat[1, 0], deviation_mat[0, 0]))
            
            comp_metric['all_blocks_bend'][grid_idx].append(bend_deg)
            comp_metric['all_blocks_twist'][grid_idx].append(twist_deg)
            
            # 통합 회전각 (Rotation Angle)
            trace_val = np.trace(deviation_mat)
            rotation_angle = np.degrees(np.arccos(np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0)))
            comp_metric['all_blocks_angle'][grid_idx].append(rotation_angle)
            list_of_angles.append(rotation_angle)
            
            # Rotation Vector (Axis-Angle) 추출
            theta = np.radians(rotation_angle)
            if abs(theta) > 1e-6:
                sin_theta = np.sin(theta)
                if abs(sin_theta) > 1e-8:
                    ux = (deviation_mat[2, 1] - deviation_mat[1, 2]) / (2.0 * sin_theta)
                    uy = (deviation_mat[0, 2] - deviation_mat[2, 0]) / (2.0 * sin_theta)
                    uz = (deviation_mat[1, 0] - deviation_mat[0, 1]) / (2.0 * sin_theta)
                    rot_vec = theta * np.array([ux, uy, uz])
                else: rot_vec = np.zeros(3)
            else: rot_vec = np.zeros(3)
            
            comp_metric['all_blocks_rotvec'][grid_idx].append(rot_vec)
            global_rot_vectors.append(rot_vec)
            
            # --- Mechanics & Energy 역산 (Solref 기반) ---
            geom_id = -1
            if m.body_geomnum[body_uid] > 0:
                geom_id = m.body_geomadr[body_uid]
                
            timeconst = 0.002
            if geom_id >= 0:
                timeconst = float(m.geom_solref[geom_id][0])
                if timeconst < 0.002: timeconst = 0.002
                
            k_lin = 1.0 / (timeconst**2)
            
            size = m.geom_size[geom_id] if geom_id >= 0 else np.array([0.1, 0.1, 0.1])
            dx, dy, dz = size[0]*2, size[1]*2, size[2]*2
            A = dx * dy
            L = dz if dz > 1e-4 else 0.01
            
            E_eff = (k_lin * L) / A if A > 1e-4 else 0.0
            
            I_x = (dx * (dy**3)) / 12.0
            I_y = (dy * (dx**3)) / 12.0
            I_avg = (I_x + I_y) / 2.0
            K_rot = (E_eff * I_avg) / L
            
            J_polar = I_x + I_y
            K_tor = (E_eff * J_polar) / (2.6 * L) # Poisson ratio ~0.3 (2(1+v)=2.6)
            
            theta_bend_rad = np.radians(bend_deg)
            theta_twist_rad = np.radians(twist_deg)
            
            M_bend = K_rot * theta_bend_rad
            c_dist = (dx + dy) / 4.0
            sigma_bend_mpa = (M_bend * c_dist) / I_avg / 1e6 if I_avg > 1e-12 else 0.0
            
            T_twist = K_tor * theta_twist_rad
            tau_twist_mpa = (T_twist * c_dist) / J_polar / 1e6 if J_polar > 1e-12 else 0.0
            
            U_rot = 0.5 * K_rot * (theta_bend_rad**2)
            U_tor = 0.5 * K_tor * (theta_twist_rad**2)
            U_total = U_rot + U_tor
            
            # Tilt X/Y 성분
            bend_x_deg = np.degrees(np.arctan2(deviation_mat[2, 1], deviation_mat[2, 2]))
            bend_y_deg = np.degrees(np.arctan2(-deviation_mat[2, 0], np.sqrt(deviation_mat[2, 1]**2 + deviation_mat[2, 2]**2)))
            
            comp_metric['all_blocks_bend_x'][grid_idx].append(bend_x_deg)
            comp_metric['all_blocks_bend_y'][grid_idx].append(bend_y_deg)
            comp_metric['all_blocks_s_bend'][grid_idx].append(sigma_bend_mpa)
            comp_metric['all_blocks_s_twist'][grid_idx].append(tau_twist_mpa)
            comp_metric['all_blocks_moment'][grid_idx].append(M_bend)
            comp_metric['all_blocks_energy'][grid_idx].append(U_total)
        # RRG (Relative Rotation Gradient) - 이웃 블록 간 상대 회전
        if comp_name in sim.neighbor_map:
            for grid_idx in sim.components[comp_name]:
                neighbors = sim.neighbor_map[comp_name].get(grid_idx, [])
                max_rel_angle = 0.0
                if grid_idx in step_deviation_cache:
                    dev_i = step_deviation_cache[grid_idx]
                    for n_idx in neighbors:
                        if n_idx in step_deviation_cache:
                            dev_j = step_deviation_cache[n_idx]
                            r_rel = dev_i.T @ dev_j
                            rel_angle = np.degrees(np.arccos(np.clip((np.trace(r_rel) - 1.0) / 2.0, -1.0, 1.0)))
                            if rel_angle > max_rel_angle: max_rel_angle = rel_angle
                
                comp_metric['all_blocks_rrg'][grid_idx].append(max_rel_angle)
                step_rrg_max = max(step_rrg_max, max_rel_angle)
            
            # [v4.8.3] 컴포넌트 레벨 RRG 히스토리 저장 (UI 그래프 출력용)
            c_rrg_max = 0.0
            for g_idx in sim.components[comp_name]:
                if comp_metric['all_blocks_rrg'][g_idx]:
                    c_rrg_max = max(c_rrg_max, comp_metric['all_blocks_rrg'][g_idx][-1])
            comp_metric.setdefault('max_rrg_hist', []).append(c_rrg_max)
        else:
            # neighbor_map이 없는 경우에도 빈 데이터 구조는 유지
            comp_metric.setdefault('max_rrg_hist', []).append(0.0)

        # GTI/GBI/Energy (Global Indices)
        if comp_name not in sim.structural_time_series['comp_global_metrics']:
            sim.structural_time_series['comp_global_metrics'][comp_name] = {'gti': [], 'gbi': [], 'energy': []}
            
        step_total_energy = 0.0
        for g_idx in sim.components[comp_name]:
            if len(comp_metric['all_blocks_energy'][g_idx]) > 0:
                step_total_energy += comp_metric['all_blocks_energy'][g_idx][-1]
        sim.structural_time_series['comp_global_metrics'][comp_name]['energy'].append(step_total_energy)

        # [v4.5] 컴포넌트별 PBA (Principal Bending Axis) 연산
        if list_of_angles:
            comp_rv_array = np.array([comp_metric['all_blocks_rotvec'][idx][-1] for idx in sim.components[comp_name]])
            if len(comp_rv_array) > 2:
                c_cov = np.cov(comp_rv_array.T)
                c_evals, c_evecs = np.linalg.eigh(c_cov)
                c_p_idx = np.argmax(c_evals)
                c_pba_mag = np.degrees(np.sqrt(max(0, c_evals[c_p_idx])))
                c_pba_vec = c_evecs[:, c_p_idx]
                
                comp_metric.setdefault('pba_mag_hist', []).append(c_pba_mag) # Backward compatibility
                comp_metric['max_pba_hist'].append(c_pba_mag)
                comp_metric.setdefault('pba_vec_hist', []).append(c_pba_vec)
                
                # Azimuth/Elevation 산출
                c_azi = np.degrees(np.arctan2(c_pba_vec[1], c_pba_vec[0])) % 180
                c_ele = np.degrees(np.arcsin(np.clip(c_pba_vec[2], -1.0, 1.0)))
                comp_metric.setdefault('pba_azi_hist', []).append(c_azi)
                comp_metric.setdefault('pba_ele_hist', []).append(c_ele)
            else:
                comp_metric['max_pba_hist'].append(0.0)
                comp_metric.setdefault('pba_vec_hist', []).append(np.zeros(3))
                comp_metric.setdefault('pba_azi_hist', []).append(0.0)
                comp_metric.setdefault('pba_ele_hist', []).append(0.0)


            rms_distortion = np.sqrt(np.mean(np.array(list_of_angles)**2))
            comp_metric['total_distortion'].append(rms_distortion)
            sim.structural_time_series['comp_global_metrics'][comp_name]['gti'].append(rms_distortion)
            
            gbi_val = sum((a**2 for a in list_of_angles)) / len(list_of_angles)
            sim.structural_time_series['comp_global_metrics'][comp_name]['gbi'].append(gbi_val)
            
    # [v4.5] 전체 시스템 PBA (Global) - 3D PCA
    if global_rot_vectors:
        rv_array = np.array(global_rot_vectors)
        if len(rv_array) > 2:
            cov_mat = np.cov(rv_array.T)
            evals, evecs = np.linalg.eigh(cov_mat)
            p_idx = np.argmax(evals)
            pba_mag_deg = np.degrees(np.sqrt(max(0, evals[p_idx])))
            pba_vec = evecs[:, p_idx]
            
            azimuth = np.degrees(np.arctan2(pba_vec[1], pba_vec[0])) % 180
            elevation = np.degrees(np.arcsin(np.clip(pba_vec[2], -1.0, 1.0)))
            
            sim.structural_time_series['pba_magnitude'].append(pba_mag_deg)
            sim.structural_time_series['pba_angle'].append(azimuth)
            sim.structural_time_series['pba_vector'].append(pba_vec)
            
            if 'pba_elevation' not in sim.structural_time_series:
                sim.structural_time_series['pba_elevation'] = []
            sim.structural_time_series['pba_elevation'].append(elevation)

    sim.structural_time_series['rrg_max'].append(step_rrg_max)
    sim.structural_time_series['mean_distortion'].append(np.mean([m['total_distortion'][-1] for m in sim.metrics.values() if m['total_distortion']]))


def compute_critical_timestamps(sim: Any) -> Dict[str, Any]:
    """[v4.5] 전체 시뮬레이션 데이터에서 주요 임계 시점을 자동 검출합니다."""
    ts = sim.time_history
    metrics = sim.structural_time_series
    results = {}

    # 1. RRG Global Peak 탐지
    if metrics.get('rrg_max'):
        idx = int(np.argmax(metrics['rrg_max']))
        results['local_peak_time'] = ts[idx]
        results['local_peak_rrg'] = float(metrics['rrg_max'][idx])

    # 2. Avg Distortion (GTI) Peak 탐지
    if metrics.get('mean_distortion'):
        idx = int(np.argmax(metrics['mean_distortion']))
        results['global_avg_peak_time'] = ts[idx]
        results['global_avg_peak_val'] = float(metrics['mean_distortion'][idx])

    # 3. PBA Global Peak 탐지 (v4.5 3D 지원)
    if metrics.get('pba_magnitude'):
        idx = int(np.argmax(metrics['pba_magnitude']))
        results['pba_peak_time'] = ts[idx]
        results['pba_peak_mag'] = float(metrics['pba_magnitude'][idx])
        results['pba_peak_angle'] = float(metrics['pba_angle'][idx])
        if 'pba_elevation' in metrics:
            results['pba_peak_ele'] = float(metrics['pba_elevation'][idx])

    return results


def finalize_simulation_results(sim: Any) -> None:
    """시뮬레이션 종료 후 데이터를 정리하고 정밀 리포트를 출력합니다."""
    # [v4.8.6] 칸 맞춤 및 정렬 개선
    col_width = 22
    total_w = 20 + (col_width + 3) * 4 + 2
    
    print("\n" + "=" * total_w)
    print(f" [ WHTOOLS Simulation Final Report ] - {sim.timestamp}")
    print("-" * total_w)
    
    # 헤더 구성
    header = (f" {'Component':<20} | {'Max Bend @Blk':^{col_width}} | {'Max Twist @Blk':^{col_width}} | "
              f"{'Max BS(MPa) @Blk':^{col_width}} | {'Max RRG @Blk':^{col_width}}")
    print(header)
    print("-" * total_w)

    for comp_name, comp_metric in sim.metrics.items():
        # (1) Bending Max
        b_max, b_idx = 0.0, "-"
        for idx, h in comp_metric.get('all_blocks_bend', {}).items():
            if h:
                m = max([abs(val) for val in h])
                if m > b_max: b_max, b_idx = m, idx

        # (2) Twist Max
        t_max, t_idx = 0.0, "-"
        for idx, h in comp_metric.get('all_blocks_twist', {}).items():
            if h:
                m = max([abs(val) for val in h])
                if m > t_max: t_max, t_idx = m, idx

        # (3) Bending Stress (BS) Max
        s_max, s_idx = 0.0, "-"
        for idx, h in comp_metric.get('all_blocks_s_bend', {}).items():
            if h:
                m = max([abs(val) for val in h])
                if m > s_max: s_max, s_idx = m, idx
        
        # (4) RRG Max
        r_max, r_idx = 0.0, "-"
        for idx, h in comp_metric.get('all_blocks_rrg', {}).items():
            if h:
                m = max([abs(val) for val in h])
                if m > r_max: r_max, r_idx = m, idx

        # 데이터 행 출력 - 칸 맞춤 적용
        def _fmt(val, idx):
            if idx == "-": return f"{val:6.2f} @{' - ':<13}"
            return f"{val:6.2f} @{str(idx):<13}"

        print(f" {comp_name:<20} | {_fmt(b_max, b_idx)} | {_fmt(t_max, t_idx)} | "
              f"{_fmt(s_max, s_idx)} | {_fmt(r_max, r_idx)}")

    print("=" * total_w + "\n")

def apply_rank_heatmap(sim: Any) -> None:
    """변형 정도에 따른 랭크 히트맵을 시각적으로 적용합니다."""
    import matplotlib.cm as cm
    cmap = cm.get_cmap('RdYlBu_r')
    
    for comp_name, comp_metric in sim.metrics.items():
        scores = {idx: (max(comp_metric['all_blocks_bend'][idx]) + max(comp_metric['all_blocks_twist'][idx]))/2.0 
                  for idx in comp_metric['all_blocks_bend'] if comp_metric['all_blocks_bend'][idx]}
        if not scores: continue
        
        sorted_idxs = sorted(scores.items(), key=lambda x: x[1])
        n = len(sorted_idxs)
        for rank, (idx, _) in enumerate(sorted_idxs):
            f = rank / (n - 1) if n > 1 else 1.0
            color = cmap(f)
            b_uid = sim.components[comp_name][idx]
            for g_id in range(sim.model.ngeom):
                if sim.model.geom_bodyid[g_id] == b_uid:
                    sim.model.geom_rgba[g_id] = color
    if sim.viewer: sim.viewer.sync()

def compute_ssr_shell_metrics(comp_name: str, positions: np.ndarray, values: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    [WHTOOLS] Sharp-Edge SSR(Structural Surface Reconstruction) 쉘 응력 해석 엔진.
    RBF 보간을 제거하고, 각 블록에서 국부 다항식 회귀(PSR)를 통해 뭉게짐 없이 정밀한 응력을 산출합니다.
    
    Args:
        comp_name: 컴포넌트 이름
        positions: 블록들의 로컬 좌표 (N, 2) - [x, y]
        values: 블록들의 변위 또는 회전각 (N,)
        config: 엔진 설정
        
    Returns:
        Dict: PSR 분석 결과
    """
    if len(positions) < 6: # 2차 곡면 피팅을 위해 최소 6개 이상의 점 필요
        return {"error": "Insufficient points (min 6 required for Quadratic PSR)"}

    x, y = positions[:, 0], positions[:, 1]
    v = values
    
    # --- 1. 전역 연산을 위한 그리드 준비 ---
    res = config.get("ssr_resolution", 40)
    xi = np.linspace(x.min(), x.max(), res)
    yi = np.linspace(y.min(), y.max(), res)
    XI, YI = np.meshgrid(xi, yi)
    
    # 결과 저장소 초기화 (응력 필드 및 보간 필드)
    stress_field = np.zeros_like(XI)
    interp_field = np.zeros_like(XI)
    
    # --- 2. Local Polynomial Regression (PSR) 실행 ---
    from scipy.spatial import cKDTree
    tree = cKDTree(positions)
    
    # 쉘 파라미터
    t = config.get("ssr_thickness", 0.002)
    E = config.get("ssr_youngs_modulus", 70e9)
    nu = config.get("ssr_poisson_ratio", 0.22)
    D = (E * (t**3)) / (12 * (1 - nu**2))
    
    # 각 그리드 포인트에 대해 국부 피팅 수행
    # (블록 데이터가 조밀하므로 그리드 포인트별로 최근접 블록들을 찾아 피팅)
    neighbor_k = config.get("psr_neighbor_count", 9) # 3x3 근방 (기본값)
    
    for r in range(res):
        for c in range(res):
            px, py = XI[r, c], YI[r, c]
            dist, idxs = tree.query([px, py], k=min(neighbor_k, len(positions)))
            
            # 국부 좌표계로 변환 (중심 이동)
            lx = x[idxs] - px
            ly = y[idxs] - py
            lv = v[idxs]
            
            # 디자인 행렬 A: [1, x, y, x^2, xy, y^2]
            A = np.column_stack([
                np.ones(len(lx)), lx, ly, lx**2, lx*ly, ly**2
            ])
            
            try:
                # 가중 최소자승법 (거리에 따른 가중치 부여로 뭉게짐 방지 및 부드러움 확보)
                # w = 1 / (d + eps)
                weights = 1.0 / (dist + 1e-6)
                W_mat = np.diag(weights)
                
                # c = (A_T * W * A)^-1 * A_T * W * v
                coeffs, _, _, _ = np.linalg.lstsq(W_mat @ A, W_mat @ lv, rcond=None)
                
                # 계수 추출: a0=const, a1=x, a2=y, a3=x^2, a4=xy, a5=y^2
                w_val = coeffs[0] # px, py 지점의 보간된 값
                w_xx = 2 * coeffs[3]
                w_yy = 2 * coeffs[5]
                w_xy = coeffs[4]
                
                interp_field[r, c] = w_val
                
                # Shell 이론 기반 모멘트 및 응력
                Mx = -D * (w_xx + nu * w_yy)
                My = -D * (w_yy + nu * w_xx)
                Mxy = D * (1 - nu) * w_xy
                
                # 주모멘트 및 최대 표면 응력
                Mavg = (Mx + My) / 2.0
                Mdiff = np.sqrt(max(0.0, ((Mx - My)/2.0)**2 + Mxy**2))
                M1 = Mavg + Mdiff
                M2 = Mavg - Mdiff
                
                s1 = (6.0 * abs(M1)) / (t**2)
                s2 = (6.0 * abs(M2)) / (t**2)
                stress_field[r, c] = max(s1, s2) / 1e6 # MPa
                
            except:
                continue

    return {
        "max_stress": float(np.max(stress_field)),
        "avg_stress": float(np.mean(stress_field)),
        "max_loc": [float(XI[np.unravel_index(np.argmax(stress_field), stress_field.shape)]),
                    float(YI[np.unravel_index(np.argmax(stress_field), stress_field.shape)])],
        "grid_x": XI.tolist(),
        "grid_y": YI.tolist(),
        "stress_field": stress_field.tolist(),
        "displacement_field": interp_field.tolist()
    }

