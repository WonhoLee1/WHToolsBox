# -*- coding: utf-8 -*-
"""
[WHTOOLS] Integrated Multi-PostProcessor Engine (High-Stability v7.5)
기계공학적 Kirchhoff-Love 판 이론과 JAX 고속 연산을 결합한 정밀 구조 변형 해석 엔진입니다.
plate_by_markers.py의 검증된 고안정성 알고리즘을 어셈블리 환경에 맞게 최적화했습니다.
"""

import os
import sys
import time
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from functools import partial
import logging
from rich.console import Console

import jax
import jax.numpy as jnp
from jax import vmap, jit
from jax.numpy.linalg import solve

# [WHTOOLS] UTF-8 인코딩 강제 설정 (Rich/Console 호환성)
if sys.stdout.encoding != 'utf-8':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except (AttributeError, io.UnsupportedOperation):
        pass

logger = logging.getLogger("WHTS_MPP_Engine")
console = Console()

# JAX 64비트 정밀도 활성화
jax.config.update("jax_enable_x64", True)

def scale_result_to_mm(result: Any):
    """
    [WHTOOLS] 시뮬레이션 결과 데이터(m)를 해석 규격(mm)으로 일괄 변환합니다.
    객체 속성뿐만 아니라 딕셔너리(v6 형태) 데이터도 지원합니다.
    """
    fields_to_scale = [
        'pos_hist', 'cog_pos_hist', 'geo_center_pos_hist', 
        'corner_pos_hist', 'z_hist', 'vel_hist', 
        'cog_vel_hist', 'geo_center_vel_hist',
        'marker_pos_history', 'marker_vel_history', 'pos_ref'
    ]
    
    # Case 1: Dictionary (v6 Lightweight or nested results)
    if isinstance(result, dict):
        for f in fields_to_scale:
            if f in result and result[f] is not None:
                val = np.array(result[f])
                if val.size > 0:
                    # 이미 mm 단위인지 대략적 확인 (v_max < 5.0m 기준)
                    v_max = np.abs(val).max()
                    if v_max < 5.0:
                        result[f] = val * 1000.0
        
        # 물리적 치수 보정
        for dim in ['W', 'H']:
            if dim in result and result[dim] is not None:
                if result[dim] < 5.0:
                    result[dim] *= 1000.0
                
        # 중첩된 analyzers 처리 (재귀)
        if 'analyzers' in result:
            for res_dict in result['analyzers'].values():
                scale_result_to_mm(res_dict)
                
        # v5 스타일의 analyzer_results 처리
        if 'analyzer_results' in result:
            for res_dict in result['analyzer_results'].values():
                scale_result_to_mm(res_dict)
                
        # 마커 데이터 딕셔너리 처리
        if 'marker_data' in result:
            for m_name, m_val in result['marker_data'].items():
                val = np.array(m_val)
                if val.size > 0 and np.abs(val).max() < 5.0:
                    result['marker_data'][m_name] = val * 1000.0
                    
        return result

    # Case 2: Object (v5 Raw Simulation result)
    # [WHTOOLS] 객체 내의 analyzer_results도 탐색
    if hasattr(result, 'analyzer_results'):
        for res_dict in result.analyzer_results.values():
            scale_result_to_mm(res_dict)
            
    for field_name in fields_to_scale:
        if hasattr(result, field_name) and getattr(result, field_name) is not None:
            val = np.array(getattr(result, field_name))
            if val.size > 0:
                v_max = np.abs(val).max()
                if v_max < 5.0:
                    scaled_data = val * 1000.0
                    setattr(result, field_name, scaled_data)
            
    if hasattr(result, 'block_half_extents'):
        for bid in result.block_half_extents:
            vals = np.array(result.block_half_extents[bid])
            if vals.size > 0 and vals.max() < 5.0:
                result.block_half_extents[bid] = (vals * 1000.0).tolist()
            
    if hasattr(result, 'nominal_local_pos'):
        for bid in result.nominal_local_pos:
            vals = np.array(result.nominal_local_pos[bid])
            if vals.size > 0 and np.abs(vals).max() < 5.0:
                result.nominal_local_pos[bid] = (vals * 1000.0).tolist()
            
    return result

# [WHTOOLS] 표준 공학 물성 라이브러리 (Default Engineering Library)
# 단위: mm, MPa
WHTOOLS_MATERIAL_LIB = {
    'opencell': {'t': 0.7,  'E': 72000.0,  'nu': 0.23, 'desc': 'Glass (Liquid Crystal Display)'},
    'chassis':  {'t': 1.2,  'E': 210000.0, 'nu': 0.30, 'desc': 'Steel (SECC/SGCC)'},
    'paper':    {'t': 3.0,  'E': 3000.0,   'nu': 0.35, 'desc': 'Corrugated Paperboard'},
    'cushion':  {'t': 25.0, 'E': 15.0,     'nu': 0.05, 'desc': 'EPS Foam'},
    'default':  {'t': 2.0,  'E': 210000.0, 'nu': 0.30, 'desc': 'Standard Steel'}
}

@dataclass
class PlateConfig:
    """
    [WHTOOLS] 평판의 물리적 특성 및 수치 해석 설정을 관리합니다.
    """
    thickness: float = 2.0
    youngs_modulus: float = 210000.0
    poisson_ratio: float = 0.3
    polynomial_degree: int = 4
    regularization_lambda: float = 1e-4
    mesh_resolution: int = 10
    batch_size: int = 256

    @staticmethod
    def from_simulation_data(simulation_data: Any, part_name: str) -> 'PlateConfig':
        """
        [WHTOOLS-A1] 시뮬레이션 결과에서 파트별 물성을 자동 매칭하고 단위를 보정(m -> mm)합니다.
        """
        config = PlateConfig()
        p_name_lower = part_name.lower()
        
        # 1. 라이브러리 기반 기본값 할당 (Heuristic Matching)
        print(f"DEBUG: Mapping material for {part_name}...", flush=True)
        found_lib = False
        matched_key = 'default'
        for key, props in WHTOOLS_MATERIAL_LIB.items():
            if key != 'default' and key in p_name_lower:
                config.thickness = props['t']
                config.youngs_modulus = props['E']
                config.poisson_ratio = props['nu']
                matched_key = key
                found_lib = True
                break
        
        if not found_lib:
            config.thickness = WHTOOLS_MATERIAL_LIB['default']['t']
            config.youngs_modulus = WHTOOLS_MATERIAL_LIB['default']['E']
            config.poisson_ratio = WHTOOLS_MATERIAL_LIB['default']['nu']
            matched_key = 'default'

        # 2. Simulation Config 매핑 (m -> mm)
        sim_cfg = getattr(simulation_data, 'config', None)
        if sim_cfg is None and isinstance(simulation_data, dict):
            sim_cfg = simulation_data.get('config')
            
        if sim_cfg is not None:
            part_key = p_name_lower.split('_')[0].replace('b', '')
            
            # [두께 결정] 파트 전용 키가 최우선
            thick_val = sim_cfg.get(f"{part_key}_thickness") or sim_cfg.get(f"{part_key}_d") or sim_cfg.get(f"{part_key}_t")
            if thick_val is None and matched_key == 'default':
                thick_val = sim_cfg.get("box_thick")
                
            if thick_val is not None:
                if thick_val < 0.5: thick_val *= 1000.0
                config.thickness = thick_val
                    
            # [영률 결정] 파트 전용 키가 최우선
            E_val = sim_cfg.get(f"{part_key}_E")
            if E_val is None and matched_key == 'default':
                E_val = sim_cfg.get("youngs_modulus")
                
            if E_val is not None:
                if E_val < 1000.0: E_val *= 1000.0 
                config.youngs_modulus = E_val

        # 최종 확인 로그 (사용자 확인용)
        # print(f"  [DEBUG] Part: {part_name:<15} | Key: {matched_key:<10} | E: {config.youngs_modulus:>8.0f} MPa", flush=True)
            
        return config

class RigidBodyKinematicsManager:
    """
    [WHTOOLS] 마커 데이터로부터 강체의 운동을 추출하고 로컬 좌표계 변환을 관리합니다.
    plate_by_markers.py의 고안정성 Kabsch 알고리즘을 계승합니다.
    """
    def __init__(self, marker_history: np.ndarray, W: float = 0, H: float = 0):
        self.raw_marker_data = jnp.array(marker_history)
        self.n_frames, self.n_markers, _ = self.raw_marker_data.shape
        self.W, self.H = W, H
        self._initialize_local_frame()

    def _initialize_local_frame(self):
        """초기 프레임(Frame 0)을 기준으로 로컬 좌표계(PCA 기반)를 수립합니다."""
        initial_markers = self.raw_marker_data[0]
        initial_centroid = jnp.mean(initial_markers, axis=0)
        centered_markers = initial_markers - initial_centroid
        
        # PCA를 통한 주축(Principal Axes) 추출
        covariance_matrix = np.cov(np.array(centered_markers).T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix + np.eye(3)*1e-9)
        sort_indices = eigenvalues.argsort()[::-1]
        self.local_basis_axes = jnp.array(eigenvectors[:, sort_indices])
        
        # 로컬 좌표계로 투영하여 범위 확인
        markers_local = centered_markers @ self.local_basis_axes
        p_min, p_max = markers_local.min(axis=0), markers_local.max(axis=0)
        
        # 수치적 안정성을 위한 여유(Margin) 설정 (5% -> 1%로 축소하여 정밀도 향상)
        applied_margin = jnp.max(p_max - p_min) * 0.01
        
        # 로컬 원점(Centroid) 재설정
        self.local_centroid_0 = initial_centroid + ((p_min + p_max) / 2.0) @ self.local_basis_axes.T
        markers_local_corrected = (initial_markers - self.local_centroid_0) @ self.local_basis_axes
        
        # 수동 입력된 W, H 가 없을 경우 데이터로부터 추정
        if self.W == 0: self.W = float(markers_local_corrected[:, 0].max() - markers_local_corrected[:, 0].min())
        if self.H == 0: self.H = float(markers_local_corrected[:, 1].max() - markers_local_corrected[:, 1].min())
        
        self.x_bounds = [
            float(markers_local_corrected[:, 0].min() - applied_margin), 
            float(markers_local_corrected[:, 0].max() + applied_margin)
        ]
        self.y_bounds = [
            float(markers_local_corrected[:, 1].min() - applied_margin), 
            float(markers_local_corrected[:, 1].max() + applied_margin)
        ]
        # Raw dimensions for UI
        self.actual_w = float(markers_local_corrected[:, 0].max() - markers_local_corrected[:, 0].min())
        self.actual_h = float(markers_local_corrected[:, 1].max() - markers_local_corrected[:, 1].min())
        self.y_bounds = [
            float(markers_local_corrected[:, 1].min() - applied_margin), 
            float(markers_local_corrected[:, 1].max() + applied_margin)
        ]

    @partial(jit, static_argnums=(0,))
    def extract_kinematics_vmap(self, frame_markers_batch: jnp.ndarray):
        """배치 단위 Kabsch 알고리즘으로 회전 및 이동량 추출"""
        reference_markers = self.raw_marker_data[0]
        def kabsch_single(current_markers):
            ref_c = jnp.mean(reference_markers, axis=0)
            cur_c = jnp.mean(current_markers, axis=0)
            H = (current_markers - cur_c).T @ (reference_markers - ref_c)
            U, S, Vh = jnp.linalg.svd(H)
            R = Vh.T @ U.T
            R_corr = jnp.where(jnp.linalg.det(R) < 0, (Vh.T @ jnp.diag(jnp.array([1.0, 1.0, -1.0]))) @ U.T, R)
            local_disp = ((current_markers - cur_c) @ R_corr.T + ref_c - self.local_centroid_0) @ self.local_basis_axes
            return local_disp, R_corr, cur_c, ref_c
        return vmap(kabsch_single)(frame_markers_batch)

class KirchhoffPlateOptimizer:
    """[WHTOOLS] 다항식 기저 기반 평판 해석 최적화 엔진"""
    def __init__(self, degree: int = 4):
        self.degree = degree
        self.active_degree_x = degree
        self.active_degree_y = degree
        self.n_markers = 25

    def update_active_degree(self, n_pts: int, aspect_ratio: float = 1.0):
        """마커 개수 및 종횡비에 기반하여 수치적으로 안전한 차수를 결정합니다."""
        self.n_markers = n_pts
        
        # 1. 마커 수 기준 전체 차수 하향 (Underdetermined 방지)
        common_deg = self.degree
        if n_pts < 16: common_deg = min(common_deg, 2)
        elif n_pts < 25: common_deg = min(common_deg, 3)
        
        self.active_degree_x = common_deg
        self.active_degree_y = common_deg
        
        # 2. [WHTOOLS v7.5.5] 종횡비 기반 차수 제한 (Thin edge 발산 방지)
        # 가로가 훨씬 긴 경우, 세로(y) 방향 차수 제한
        if aspect_ratio > 20.0:
            self.active_degree_y = 0
            # print(f"      [Robust] High Aspect Ratio ({aspect_ratio:.1f}) detected. Limiting 2D basis to 1D-dominant.")
        elif aspect_ratio < 0.05:
            self.active_degree_x = 0

    @property
    def n_basis(self):
        return (self.active_degree_x + 1) * (self.active_degree_y + 1)

    def get_basis_matrix(self, x_norm, y_norm):
        n_pts = x_norm.shape[0]
        dx, dy = self.active_degree_x, self.active_degree_y
        A = jnp.zeros((n_pts, (dx + 1) * (dy + 1)))
        for i in range(dx + 1):
            for j in range(dy + 1):
                idx = i * (dy + 1) + j
                A = A.at[:, idx].set((x_norm**i) * (y_norm**j))
        return A

    def get_hessian_basis(self, x_norm, y_norm):
        n_pts = x_norm.shape[0]
        dx, dy = self.active_degree_x, self.active_degree_y
        H = jnp.zeros((n_pts, (dx + 1) * (dy + 1), 3))
        for i in range(dx + 1):
            for j in range(dy + 1):
                idx = i * (dy + 1) + j
                if i >= 2: H = H.at[:, idx, 0].set(i * (i - 1) * (x_norm**(i - 2)) * (y_norm**j))
                if j >= 2: H = H.at[:, idx, 1].set(j * (j - 1) * (x_norm**i) * (y_norm**(j - 2)))
                if i >= 1 and j >= 1: H = H.at[:, idx, 2].set(i * j * (x_norm**(i - 1)) * (y_norm**(j - 1)))
        return H

    @partial(jit, static_argnums=(0,))
    def solve_coefficients(self, local_displacements, ref_markers_local, reg_lambda):
        x_raw, y_raw = ref_markers_local[:, 0], ref_markers_local[:, 1]
        x_scale, y_scale = jnp.max(jnp.abs(x_raw)) + 1e-9, jnp.max(jnp.abs(y_raw)) + 1e-9
        x_norm, y_norm = x_raw / x_scale, y_raw / y_scale
        
        # [WHTOOLS] 동적 차수 적용을 위한 행렬 생성
        Phi = self.get_basis_matrix(x_norm, y_norm)
        H_basis = self.get_hessian_basis(x_norm, y_norm)
        
        # 실제 사용된 기저 차수 확인
        n_basis_active = Phi.shape[1]
        
        Bxx, Byy, Bxy = H_basis[:,:,0]/(x_scale**2), H_basis[:,:,1]/(y_scale**2), H_basis[:,:,2]/(x_scale*y_scale)
        n_pts = Phi.shape[0]
        
        K = (Phi.T @ Phi) / n_pts
        R = reg_lambda * (Bxx.T @ Bxx + Byy.T @ Byy + 2.0 * Bxy.T @ Bxy) / n_pts
        
        # [WHTOOLS] v7.5.2: 릿지 가중치 강화 (Singular/Ill-conditioned 대응)
        ridge = jnp.eye(self.n_basis) * (1e-6 if n_pts < self.n_basis * 2 else 1e-9)
        M = K + R + ridge
        
        Z = local_displacements[:, :, 2]
        @vmap
        def solve_one(zi): return solve(M, (Phi.T @ zi) / n_pts)
        return solve_one(Z), (x_scale, y_scale)

class PlateMechanicsSolver:
    """[WHTOOLS] 응력 및 변형률 필드 산출기"""
    def __init__(self, config: PlateConfig):
        self.cfg = config
        self.D = (config.youngs_modulus * config.thickness**3) / (12.0 * (1.0 - config.poisson_ratio**2))
        self.res = config.mesh_resolution
        self.opt = KirchhoffPlateOptimizer(degree=config.polynomial_degree)

    def setup_mesh(self, xb, yb):
        self.x_lin = jnp.linspace(xb[0], xb[1], self.res)
        self.y_lin = jnp.linspace(yb[0], yb[1], self.res)
        self.X_mesh, self.Y_mesh = jnp.meshgrid(self.x_lin, self.y_lin)

    @partial(jit, static_argnums=(0,))
    def compute_fields_batch(self, coeffs_batch, scales):
        xs, ys = scales
        Xn, Yn = self.X_mesh.ravel() / xs, self.Y_mesh.ravel() / ys
        Phi_eval = self.opt.get_basis_matrix(Xn, Yn)
        H_eval = self.opt.get_hessian_basis(Xn, Yn)
        
        @vmap
        def eval_one(p):
            w = (Phi_eval @ p).reshape(self.res, self.res)
            kxx = -jnp.dot(H_eval[:,:,0], p).reshape(self.res, self.res) / (xs**2)
            kyy = -jnp.dot(H_eval[:,:,1], p).reshape(self.res, self.res) / (ys**2)
            kxy = -jnp.dot(H_eval[:,:,2], p).reshape(self.res, self.res) / (xs*ys)
            
            s_coeff = 6.0 * self.D / (self.cfg.thickness**2)
            sx, sy, txy = s_coeff*(kxx + self.cfg.poisson_ratio*kyy), s_coeff*(kyy + self.cfg.poisson_ratio*kxx), s_coeff*(1.-self.cfg.poisson_ratio)*kxy
            vm = jnp.sqrt(jnp.maximum(sx**2 - sx*sy + sy**2 + 3*txy**2, 1e-12))
            
            # 주응력
            s_avg, s_rad = (sx + sy)/2.0, jnp.sqrt(jnp.maximum(((sx-sy)/2.0)**2 + txy**2, 1e-12))
            
            # 변형률 (Strain) - [A2] Kirchhoff 집중
            eps_x, eps_y, gam_xy = (self.cfg.thickness/2.0)*kxx, (self.cfg.thickness/2.0)*kyy, (self.cfg.thickness)*kxy
            e_avg, e_rad = (eps_x + eps_y)/2.0, jnp.sqrt(jnp.maximum(((eps_x-eps_y)/2.0)**2 + (gam_xy/2.0)**2, 1e-20))
            eq_eps = (2.0/3.0) * jnp.sqrt(jnp.maximum(1.5*(eps_x**2 + eps_y**2) + 0.75*gam_xy**2, 1e-20))
            
            # 곡률 (Curvature) - [WHTOOLS] 기하학적 표면 특성 분석
            h_mean = -(kxx + kyy) / 2.0
            k_gauss = kxx * kyy - kxy**2
            
            fields = {
                'Displacement [mm]': w, 'Stress XX [MPa]': sx, 'Stress YY [MPa]': sy, 'Stress XY [MPa]': txy,
                'Von-Mises [MPa]': vm, 'Principal Max [MPa]': s_avg + s_rad, 'Principal Min [MPa]': s_avg - s_rad,
                'Strain XX [mm/mm]': eps_x, 'Strain YY [mm/mm]': eps_y, 'Strain XY [mm/mm]': gam_xy,
                'Strain Max Principal [mm/mm]': e_avg + e_rad, 'Strain Min Principal [mm/mm]': e_avg - e_rad,
                'Eq. Strain [mm/mm]': eq_eps,
                'Curvature Mean [1/mm]': h_mean, 'Curvature Gauss [1/mm^2]': k_gauss
            }
            # 리프터 호환성용 별칭
            fields['Bending Stress [MPa]'] = fields['Von-Mises [MPa]']
            
            stats = {f'Mean-{k}': jnp.mean(v) for k, v in fields.items()}
            stats.update({f'Max-{k}': jnp.max(v) for k, v in fields.items()})
            return {**fields, **stats}
        return eval_one(coeffs_batch)

class ShellDeformationAnalyzer:
    """[WHTOOLS] 멀티 파트 통합 분석기 (Stabilized v7.5)"""
    def __init__(self, W=0, H=0, thickness=2.0, E=210000.0, nu=0.3, name="Part"):
        self.name = name
        self.W, self.H = W, H
        self.cfg = PlateConfig(thickness=thickness, youngs_modulus=E, poisson_ratio=nu)
        self.m_raw = self.m_data_hist = self.times = None
        self.kin = self.sol = None
        self.results = {}

    @property
    def ref_basis(self): return self.kin.local_basis_axes if self.kin else None
    @property
    def ref_center(self): return self.kin.local_centroid_0 if self.kin else None

    def run_analysis(self, sim_data=None) -> bool:
        """분석 파이프라인 실행"""
        # [v7.5.1] m_raw 또는 m_data_hist(v6 호환성) 중 하나가 있으면 사용
        markers = self.m_raw if self.m_raw is not None else self.m_data_hist
        if markers is None: return False
        self.m_raw = markers # 내부적으로 m_raw로 통일
        
        # [A1] 시뮬레이션 데이터에서 설정 자동 추출 시도
        if sim_data is not None:
            self.cfg = PlateConfig.from_simulation_data(sim_data, self.name)
        
        # 매칭된 재료 정보 로그 출력
        p_name_lower = self.name.lower()
        mat_desc = "Standard Steel"
        for key, props in WHTOOLS_MATERIAL_LIB.items():
            if key in p_name_lower:
                mat_desc = props['desc']
                break
                
        print(f"  > [Analyzing] {self.name:<24} | Material: {mat_desc}", flush=True)
        print(f"    - Properties: E={self.cfg.youngs_modulus:,.0f} MPa, t={self.cfg.thickness:.2f} mm, v={self.cfg.poisson_ratio:.2f}", flush=True)

        # [WHTOOLS] 데이터 유효성 검사 (IndexError 방지)
        if self.m_raw is None or len(self.m_raw) == 0:
            print(f"  ⚠️ [Warning] {self.name:<24} has no marker data. Skipping.", flush=True)
            return

        print(f"    - Dimensions: {len(self.m_raw[0])} markers, {len(self.m_raw)} frames", flush=True)
        
        # 기구학 매니저 초기화
        self.kin = RigidBodyKinematicsManager(self.m_raw)
        
        # UI 시각화용 치수는 마진을 제외한 순수 마커 범위 사용 (사용자 피드백 반영: Tight Bounding Box)
        self.W, self.H = self.kin.actual_w, self.kin.actual_h
        aspect = self.W / (self.H + 1e-9)
        # print(f"    - Local Area: {lx[1]-lx[0]:.1f} x {ly[1]-ly[0]:.1f} mm (Aspect: {aspect:.2f})", flush=True)
        
        # 솔버 초기화 및 메쉬 설정 (마커 개수 및 종횡비 비례하여 차수 자율 결정)
        self.sol = PlateMechanicsSolver(self.cfg)
        self.sol.opt.update_active_degree(len(self.m_raw[0]), aspect_ratio=aspect)
        self.sol.setup_mesh(self.kin.x_bounds, self.kin.y_bounds)
        
        # 1. 기구학 분리 (JAX vmap)
        local_disp, R_mat, cur_c, ref_c = self.kin.extract_kinematics_vmap(self.m_raw)
        
        # 2. 다항식 피팅
        coeffs, scales = self.sol.opt.solve_coefficients(local_disp, local_disp[0], self.cfg.regularization_lambda)
        
        # [WHTOOLS] Reference Configuration Offset (Relative Deformation) 적용
        # 첫 번째 프레임(Frame 0)의 피팅 계수를 모든 프레임에서 빼주어, 
        # 초기 상태의 미세한 곡률이나 조립 오차를 'Zero' 상태로 강제합니다.
        # 이를 통해 Displacement, Stress, Strain 등의 모든 필드는 Frame 0 대비 '순수 변화량'을 나타내게 됩니다.
        coeffs = coeffs - coeffs[0]
        
        # 3. 물리 필드 산출 (배치 처리)
        n_frames = len(self.m_raw)
        batch_size = self.cfg.batch_size
        all_frames_results = []
        
        for i in range(0, n_frames, batch_size):
            batch_coeffs = coeffs[i : i + batch_size]
            batch_res = self.sol.compute_fields_batch(batch_coeffs, scales)
            all_frames_results.append({k: np.array(v) for k, v in batch_res.items()})
            
        # 결과 병합 및 기구학 데이터 저장 (시각화/내보내기 연동용)
        self.results = {k: np.concatenate([res[k] for res in all_frames_results]) for k in all_frames_results[0].keys()}
        self.results.update({
            'R': np.array(R_mat), 'c_Q': np.array(cur_c), 'c_P': np.array(ref_c), # Legacy compatibility
            'R_matrix': np.array(R_mat), 'cur_centroid': np.array(cur_c), 'ref_centroid': np.array(ref_c),
            'local_markers': np.array(local_disp), 'Marker Global Disp. [mm]': np.array(self.m_raw - self.m_raw[0])
        })
        
        # 정렬 오차(R-RMSE) 계산 및 물리적 붕괴 감지
        ref_local = np.array(local_disp[0])
        r_rmses = np.sqrt(np.mean(np.square(np.array(local_disp) - ref_local[None, :, :])[:, :, :2], axis=(1, 2)))
        self.results['r_rmse'] = r_rmses
        
        if np.mean(r_rmses) > 15.0:
            print(f"  ❌ [CRASH] {self.name} failed alignment stability test.")
            return False
            
        print(f"  ✅ [SUCCESS] {self.name} analysis completed.", flush=True)
        return True

class PlateAssemblyManager:
    """[WHTOOLS] 다중 파트 어셈블리 관리자"""
    def __init__(self, times: np.ndarray, sim_data=None):
        self.analyzers = []
        self.times = times
        self.sim_data = sim_data

    def add_analyzer(self, analyzer: ShellDeformationAnalyzer):
        analyzer.times = self.times
        self.analyzers.append(analyzer)
        return analyzer

    def run_all(self):
        print(f"[WHTOOLS] Starting Assembly Analysis (Parts: {len(self.analyzers)})...")
        for ana in self.analyzers:
            ana.run_analysis(sim_data=self.sim_data)
        print("[WHTOOLS] Assembly Analysis Finished.")

    def show_report(self):
        from rich.console import Console
        from rich.table import Table
        console = Console()
        table = Table(title="[WHTOOLS] Assembly Structural Report")
        table.add_column("Part Name"); table.add_column("Markers", style="cyan"); table.add_column("Max Disp [mm]", style="green"); table.add_column("Max Stress [MPa]", style="magenta"); table.add_column("Max Gauss [1/mm^2]", style="yellow")
        for a in self.analyzers:
            if not a.results: continue
            n_m = len(a.m_raw[0]) if a.m_raw is not None else 0
            max_gauss = np.max(np.abs(a.results.get('Curvature Gauss [1/mm^2]', [0.0])))
            table.add_row(
                a.name, 
                str(n_m), 
                f"{np.max(np.abs(a.results['Displacement [mm]'])):.2f}", 
                f"{np.max(a.results['Von-Mises [MPa]']):.2f}",
                f"{max_gauss:.2e}"
            )
        console.print(table)
