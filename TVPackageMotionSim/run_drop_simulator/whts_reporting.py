import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Tuple, Union
import time
import jax
import jax.numpy as jnp
from jax import vmap, jit
from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel

def compute_structural_step_metrics(sim: Any) -> None:
    """
    각 시뮬레이션 타임스텝에서 부품별/블록별 구조적 변형 지표를 연산합니다.
    RRG(Relative Rotation Gradient) 및 PBA(Principal Bending Axis)를 포함합니다.
    """
    d = sim.data
    m = sim.model
    inv_root_mat = d.xmat[sim.root_id].reshape(3, 3).T
    
    global_rot_vectors = []
    step_rrg_max = 0.0

    for comp_name, comp_metric in sim.metrics.items():
        list_of_angles = []
        step_deviation_cache = {}
        
        for grid_idx, body_uid in sim.components[comp_name].items():
            block_mat = d.xmat[body_uid].reshape(3, 3)
            relative_rot = inv_root_mat @ block_mat
            
            if comp_metric['block_nominal_mats'][grid_idx] is None:
                comp_metric['block_nominal_mats'][grid_idx] = relative_rot.copy()
            
            deviation_mat = comp_metric['block_nominal_mats'][grid_idx].T @ relative_rot
            step_deviation_cache[grid_idx] = deviation_mat
            
            bend_deg = np.degrees(np.arccos(np.clip(deviation_mat[2, 2], -1.0, 1.0)))
            twist_deg = np.degrees(np.arctan2(deviation_mat[1, 0], deviation_mat[0, 0]))
            
            comp_metric['all_blocks_bend'][grid_idx].append(bend_deg)
            comp_metric['all_blocks_twist'][grid_idx].append(twist_deg)
            
            trace_val = np.trace(deviation_mat)
            rotation_angle = np.degrees(np.arccos(np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0)))
            comp_metric['all_blocks_angle'][grid_idx].append(rotation_angle)
            list_of_angles.append(rotation_angle)
            
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
            
            # Mechanical Stress Estimation
            geom_id = m.body_geomadr[body_uid] if m.body_geomnum[body_uid] > 0 else -1
            timeconst = 0.002
            if geom_id >= 0:
                timeconst = max(0.002, float(m.geom_solref[geom_id][0]))
            
            k_lin = 1.0 / (timeconst**2)
            size = m.geom_size[geom_id] if geom_id >= 0 else np.array([0.1, 0.1, 0.1])
            dx, dy, dz = size[0]*2, size[1]*2, size[2]*2
            A = dx * dy; L = dz if dz > 1e-4 else 0.01
            E_eff = (k_lin * L) / A if A > 1e-4 else 0.0
            I_avg = (dx*dy**3 + dy*dx**3) / 24.0
            sigma_bend_mpa = (E_eff * np.radians(bend_deg) * (dx+dy)/4.0) / L / 1e6 if L > 0 else 0.0
            
            comp_metric['all_blocks_s_bend'][grid_idx].append(sigma_bend_mpa)
            comp_metric['all_blocks_energy'][grid_idx].append(0.5 * k_lin * (np.radians(rotation_angle)**2))

        # RRG calculation
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
                            max_rel_angle = max(max_rel_angle, rel_angle)
                comp_metric['all_blocks_rrg'][grid_idx].append(max_rel_angle)
                step_rrg_max = max(step_rrg_max, max_rel_angle)

    sim.structural_time_series['rrg_max'].append(step_rrg_max)

def _compute_batch_metrics_jax(sim: Any) -> None:
    """JAX vmap/jit을 활용한 초고속 구조 해석 엔진"""
    if not sim.quat_hist: return
    q_hist = jnp.array(sim.quat_hist)
    n_frames, n_bodies, _ = q_hist.shape
    root_id = sim.root_id
    
    @jit
    @vmap
    @vmap
    def quat_to_mat(q):
        w, x, y, z = q
        return jnp.array([
            [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
        ])
    
    mats = quat_to_mat(q_hist)
    inv_root_mats = jnp.transpose(mats[:, root_id], (0, 2, 1))
    
    for comp_name, comp_metric in sim.metrics.items():
        sorted_keys = sorted(sim.components[comp_name].keys())
        body_ids = jnp.array([sim.components[comp_name][idx] for idx in sorted_keys])
        target_mats = mats[:, body_ids]
        rel_mats = jnp.matmul(inv_root_mats[:, jnp.newaxis, :, :], target_mats)
        
        nom_mats_list = []
        for idx in sorted_keys:
            if comp_metric['block_nominal_mats'][idx] is None:
                comp_metric['block_nominal_mats'][idx] = np.array(rel_mats[0, sorted_keys.index(idx)])
            nom_mats_list.append(comp_metric['block_nominal_mats'][idx])
        
        nom_mats = jnp.array(nom_mats_list)
        dev_mats = jnp.matmul(jnp.transpose(nom_mats, (0, 2, 1))[jnp.newaxis, :, :, :], rel_mats)
        
        bend = jnp.degrees(jnp.arccos(jnp.clip(dev_mats[:, :, 2, 2], -1.0, 1.0)))
        twist = jnp.degrees(jnp.arctan2(dev_mats[:, :, 1, 0], dev_mats[:, :, 0, 0]))
        
        # [V5.8.5] BS (Bending Stress) Estimation - JAX Vectorized
        e_eff_list, l_list, dxy_list = [], [], []
        m = sim.model
        for b_id in body_ids:
            g_id = m.body_geomadr[b_id] if m.body_geomnum[b_id] > 0 else -1
            tc = max(0.002, float(m.geom_solref[g_id][0])) if g_id >= 0 else 0.002
            sz = m.geom_size[g_id] if g_id >= 0 else np.array([0.1, 0.1, 0.1])
            dx, dy, dz = sz[0]*2, sz[1]*2, sz[2]*2
            a_area = dx * dy; length = dz if dz > 1e-4 else 0.01
            k_lin = 1.0 / (tc**2)
            e_eff_list.append((k_lin * length) / a_area if a_area > 1e-4 else 0.0)
            l_list.append(length)
            dxy_list.append((dx+dy)/4.0)

        e_eff_j = jnp.array(e_eff_list); l_j = jnp.array(l_list); dxy_j = jnp.array(dxy_list)
        # Sigma = (E * Theta_rad * c) / L
        bs_mpa = (e_eff_j[jnp.newaxis, :] * jnp.radians(bend) * dxy_j[jnp.newaxis, :]) / (l_j[jnp.newaxis, :] + 1e-9) / 1e6
        
        bend_np, twist_np, bs_np = np.array(bend), np.array(twist), np.array(bs_mpa)
        for i, idx in enumerate(sorted_keys):
            comp_metric['all_blocks_bend'][idx] = bend_np[:, i].tolist()
            comp_metric['all_blocks_twist'][idx] = twist_np[:, i].tolist()
            comp_metric['all_blocks_s_bend'][idx] = bs_np[:, i].tolist()

        # [V5.8.6] RRG (Rotational Rigidity Gradient) - JAX Vectorized
        # Neighbor relative rotation logic
        if comp_name in sim.neighbor_map:
            for idx in sorted_keys:
                neighbors = sim.neighbor_map[comp_name].get(idx, [])
                if not neighbors: 
                    comp_metric['all_blocks_rrg'][idx] = [0.0] * n_frames
                    continue
                
                i_pos = sorted_keys.index(idx)
                n_positions = [sorted_keys.index(n) for n in neighbors if n in sorted_keys]
                if not n_positions:
                    comp_metric['all_blocks_rrg'][idx] = [0.0] * n_frames
                    continue
                
                # Fetch relative mats for neighbors
                # rel_mats shape: [frames, bodies, 3, 3]
                dev_i = dev_mats[:, i_pos]
                rel_rot_max = jnp.zeros(n_frames)
                for n_pos in n_positions:
                    dev_j = dev_mats[:, n_pos]
                    # dev_i^T @ dev_j = relative rotation between neighbors
                    r_rel = jnp.matmul(jnp.transpose(dev_i, (0, 2, 1)), dev_j)
                    tr = jnp.trace(r_rel, axis1=1, axis2=2)
                    angle = jnp.degrees(jnp.arccos(jnp.clip((tr - 1.0) / 2.0, -1.0, 1.0)))
                    rel_rot_max = jnp.maximum(rel_rot_max, angle)
                comp_metric['all_blocks_rrg'][idx] = np.array(rel_rot_max).tolist()

def compute_batch_structural_metrics(sim: Any) -> None:
    """[V5.3.2] JAX 가속 엔진 전용 배치 해석 모델"""
    print(f"\n" + "="*70)
    print(f" [ WHTOOLS BATCH ANALYSIS CORE ] - Powered by JAX")
    print("-" * 70)
    t1 = time.perf_counter()
    _compute_batch_metrics_jax(sim)
    dur = time.perf_counter() - t1
    print(f" >> Structural Metrics Processed (JAX): {dur:8.4f} sec")
    print("=" * 70 + "\n")

def finalize_simulation_results(sim: Any) -> None:
    """시뮬레이션 종료 후 데이터를 정리하고 정밀 리포트를 출력합니다."""
    console = Console()
    table = Table(
        title=f"📝 [WHTOOLS] Simulation Final Report - {sim.timestamp}",
        box=box.HEAVY_HEAD,
        header_style="bold magenta"
    )
    
    table.add_column("📦 Component", style="cyan", width=20)
    table.add_column("📐 Max Bend @Blk", justify="center")
    table.add_column("🌪️ Max Twist @Blk", justify="center")
    table.add_column("💥 Max BS(MPa) @Blk", justify="center")
    table.add_column("📈 Max RRG @Blk", justify="center")

    for comp_name, comp_metric in list(sim.metrics.items()):
        b_max, b_idx = 0.0, "-"
        for idx, h in comp_metric.get('all_blocks_bend', {}).items():
            if h:
                m = max([abs(val) for val in h])
                if m > b_max: b_max, b_idx = m, idx

        t_max, t_idx = 0.0, "-"
        for idx, h in comp_metric.get('all_blocks_twist', {}).items():
            if h:
                m = max([abs(val) for val in h])
                if m > t_max: t_max, t_idx = m, idx

        s_max, s_idx = 0.0, "-"
        for idx, h in comp_metric.get('all_blocks_s_bend', {}).items():
            if h:
                m = max([abs(val) for val in h])
                if m > s_max: s_max, s_idx = m, idx
        
        r_max, r_idx = 0.0, "-"
        for idx, h in comp_metric.get('all_blocks_rrg', {}).items():
            if h:
                m = max([abs(val) for val in h])
                if m > r_max: r_max, r_idx = m, idx

        def _fmt(val, idx):
            if idx == "-": return f"{val:5.2f} @ -"
            return f"{val:5.2f} @ {str(idx)}"

        table.add_row(comp_name, _fmt(b_max, b_idx), _fmt(t_max, t_idx), _fmt(s_max, s_idx), _fmt(r_max, r_idx))

    console.print(table)
    legend_text = (
        "• [bold]Bend[/bold]  : Principal Bending (Tilt) Angle [deg]\n"
        "• [bold]Twist[/bold] : Torsional (Twist) Angle [deg]\n"
        "• [bold]BS[/bold]    : Max Bending Stress calculated from internal moments [MPa]\n"
        "• [bold]RRG[/bold]   : Rotational Rigidity Gradient (Relative rotation between adjacent blocks) [deg]"
    )
    console.print(Panel(legend_text, title="📖 Metrics Legend", border_style="blue", expand=False))

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

def apply_rank_heatmap(sim: Any) -> None:
    """변형 정도에 따른 랭크 히트맵을 시각적으로 적용합니다."""
    import matplotlib.cm as cm
    # [v4.8.7] Colormap 호환성 개선 (Matplotlib 3.8+ 대응)
    try:
        cmap = cm.get_cmap('RdYlBu_r')
    except:
        cmap = matplotlib.colormaps['RdYlBu_r']
    
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
    
    # 결과 저장소 초기화
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
    
    neighbor_k = config.get("psr_neighbor_count", 9)
    
    for r in range(res):
        for c in range(res):
            px, py = XI[r, c], YI[r, c]
            dist, idxs = tree.query([px, py], k=min(neighbor_k, len(positions)))
            lx = x[idxs] - px
            ly = y[idxs] - py
            lv = v[idxs]
            A = np.column_stack([np.ones(len(lx)), lx, ly, lx**2, lx*ly, ly**2])
            try:
                weights = 1.0 / (dist + 1e-6)
                W_mat = np.diag(weights)
                coeffs, _, _, _ = np.linalg.lstsq(W_mat @ A, W_mat @ lv, rcond=None)
                w_val = coeffs[0]
                w_xx, w_yy, w_xy = 2*coeffs[3], 2*coeffs[5], coeffs[4]
                interp_field[r, c] = w_val
                Mx = -D * (w_xx + nu * w_yy); My = -D * (w_yy + nu * w_xx); Mxy = D * (1 - nu) * w_xy
                Mavg = (Mx + My) / 2.0
                Mdiff = np.sqrt(max(0.0, ((Mx - My)/2.0)**2 + Mxy**2))
                s1 = (6.0 * abs(Mavg + Mdiff)) / (t**2)
                s2 = (6.0 * abs(Mavg - Mdiff)) / (t**2)
                stress_field[r, c] = max(s1, s2) / 1e6 # MPa
            except: continue

    return {
        "max_stress": float(np.max(stress_field)),
        "avg_stress": float(np.mean(stress_field)),
        "max_loc": [float(XI[np.unravel_index(np.argmax(stress_field), stress_field.shape)]),
                    float(YI[np.unravel_index(np.argmax(stress_field), stress_field.shape)])],
        "grid_x": XI.tolist(), "grid_y": YI.tolist(),
        "stress_field": stress_field.tolist(), "displacement_field": interp_field.tolist()
    }
