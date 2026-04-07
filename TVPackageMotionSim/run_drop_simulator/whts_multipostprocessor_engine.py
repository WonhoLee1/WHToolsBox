# -*- coding: utf-8 -*-
"""
[WHTOOLS] Integrated Multi-PostProcessor Engine
기계공학적 판 이론(Plate Theory)과 JAX 고속 연산을 결합한 정밀 구조 변형 해석 엔진 모듈입니다.
UI와 독립적으로 수치 해석 및 데이터 처리를 전담합니다.
"""

import os
import sys
import time
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
from jax import vmap, jit
from jax.numpy.linalg import solve

# JAX 64비트 정밀도 활성화
jax.config.update("jax_enable_x64", True)

def scale_result_to_mm(result: Any):
    """
    [WHTOOLS] 시뮬레이션 결과 데이터(m)를 대시보드 규격(mm)으로 일괄 변환합니다.
    """
    fields_to_scale = [
        'pos_hist', 'cog_pos_hist', 'geo_center_pos_hist', 
        'corner_pos_hist', 'z_hist', 'vel_hist', 
        'cog_vel_hist', 'geo_center_vel_hist'
    ]
    
    for field_name in fields_to_scale:
        if hasattr(result, field_name) and getattr(result, field_name) is not None:
            scaled_data = np.array(getattr(result, field_name)) * 1000.0
            setattr(result, field_name, scaled_data)
            
    if hasattr(result, 'block_half_extents'):
        for bid in result.block_half_extents:
            result.block_half_extents[bid] = [v * 1000.0 for v in result.block_half_extents[bid]]
            
    if hasattr(result, 'nominal_local_pos'):
        for bid in result.nominal_local_pos:
            result.nominal_local_pos[bid] = [v * 1000.0 for v in result.nominal_local_pos[bid]]
            
    return result

@dataclass
class PlateConfig:
    """
    [WHTOOLS] 판(Plate) 물리적 특성 및 수치 해석 설정
    """
    thickness: float = 2.0
    youngs_modulus: float = 2.1e5
    poisson_ratio: float = 0.3
    poly_degree: int = 4
    reg_lambda: float = 1e-4
    grad_lambda: float = 1e-6
    mesh_resolution: int = 25
    batch_size: int = 256
    theory_type: str = "KIRCHHOFF"
    shear_correction: float = 5.0/6.0
    margin_ratio: float = 0.05

class AlignmentManager:
    """
    [WHTOOLS] 마커 데이터의 좌표계 정렬 및 강체 운동(Rigid Body Motion) 관리자
    """
    def __init__(self, raw_markers, W, H, offsets):
        self.raw_data = jnp.array(raw_markers)
        self.W, self.H = W, H
        self.offsets = jnp.array(offsets)
        self.n_frames, self.n_markers, _ = self.raw_data.shape
        self._calibrate()

    def _calibrate(self):
        P0 = np.array(self.raw_data[0])
        P_target = np.column_stack([self.offsets, np.zeros(self.n_markers)])
        c_P = np.mean(P0, axis=0)
        c_T = np.mean(P_target, axis=0)
        H_mat = (P_target - c_T).T @ (P0 - c_P)
        U, S, Vt = np.linalg.svd(H_mat)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            R = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
        self.local_axes = R.T
        self.centroid_0 = c_P
        self.x_bounds = [-self.W/2 - 5, self.W/2 + 5]
        self.y_bounds = [-self.H/2 - 5, self.H/2 + 5]

    @partial(jit, static_argnums=(0,))
    def extract_kinematics_vmap(self, frame_markers):
        P_ref = jnp.column_stack([self.offsets, jnp.zeros(self.n_markers)])
        c_P_ref = jnp.mean(P_ref, axis=0)
        def kabsch_single(Q):
            c_Q = jnp.mean(Q, axis=0)
            H = (P_ref - c_P_ref).T @ (Q - c_Q)
            U, S, Vt = jnp.linalg.svd(H)
            R = U @ Vt
            R_corr = jnp.where(
                jnp.linalg.det(R) < 0, 
                U @ jnp.diag(jnp.array([1.0, 1.0, -1.0])) @ Vt, 
                R
            )
            q_local = (Q - c_Q) @ R_corr + c_P_ref
            return q_local, R_corr, c_Q, c_P_ref
        return vmap(kabsch_single)(frame_markers)

class AdvancedPlateOptimizer:
    """
    [WHTOOLS] 다항식 표면 근사(Polynomial Surface Fitting) 최적화 엔진
    """
    def __init__(self, degree_x: int = 4, degree_y: int = 4):
        self.basis_indices = [(i, j) for i in range(degree_x + 1) for j in range(degree_y + 1)]
        self.num_basis = len(self.basis_indices)
        self.degree_x = degree_x
        self.degree_y = degree_y

    def get_basis_matrix(self, x_coords: jnp.ndarray, y_coords: jnp.ndarray) -> jnp.ndarray:
        return jnp.stack([x_coords**i * y_coords**j for i, j in self.basis_indices], axis=1)

    def get_hessian_basis(self, x_coords: jnp.ndarray, y_coords: jnp.ndarray) -> jnp.ndarray:
        def compute_single_hessian(xi, yi):
            hessian_elements = []
            for i, j in self.basis_indices:
                w_xx = i*(i-1) * xi**(jnp.maximum(0, i-2)) * yi**j if i >= 2 else 0.0
                w_yy = j*(j-1) * xi**i * yi**(jnp.maximum(0, j-2)) if j >= 2 else 0.0
                w_xy = i*j * xi**(jnp.maximum(0, i-1)) * yi**(jnp.maximum(0, j-1)) if i >= 1 and j >= 1 else 0.0
                hessian_elements.append(jnp.array([w_xx, w_yy, w_xy]))
            return jnp.stack(hessian_elements, axis=0)
        return vmap(compute_single_hessian)(x_coords, y_coords)

    def get_gradient_basis(self, x_coords: jnp.ndarray, y_coords: jnp.ndarray) -> jnp.ndarray:
        def compute_single_gradient(xi, yi):
            grad_elements = []
            for i, j in self.basis_indices:
                w_x = i * xi**(jnp.maximum(0, i-1)) * yi**j if i >= 1 else 0.0
                w_y = j * xi**i * yi**(jnp.maximum(0, j-1)) if j >= 1 else 0.0
                grad_elements.append(jnp.array([w_x, w_y]))
            return jnp.stack(grad_elements, axis=0)
        return vmap(compute_single_gradient)(x_coords, y_coords)

    @partial(jit, static_argnums=(0,))
    def solve_analytical(self, q_local: jnp.ndarray, p_ref: jnp.ndarray, 
                         reg_lambda: float, grad_lambda: float = 0.0,
                         fixed_stats: Optional[jnp.ndarray] = None):
        Z_history = q_local[:, :, 2]
        x_raw, y_raw = p_ref[:, 0], p_ref[:, 1]
        
        if fixed_stats is not None:
            x_ctr, x_rng, y_ctr, y_rng = fixed_stats
        else:
            x_min, x_max = jnp.min(x_raw), jnp.max(x_raw)
            y_min, y_max = jnp.min(y_raw), jnp.max(y_raw)
            x_ctr, y_ctr = (x_max + x_min) / 2.0, (y_max + y_min) / 2.0
            x_rng, y_rng = (x_max - x_min) / 2.0 + 1e-9, (y_max - y_min) / 2.0 + 1e-9
        
        x_norm, y_norm = (x_raw - x_ctr) / x_rng, (y_raw - y_ctr) / y_rng
        Phi = self.get_basis_matrix(x_norm, y_norm)
        G_basis, H_basis = self.get_gradient_basis(x_norm, y_norm), self.get_hessian_basis(x_norm, y_norm)
        num_pts = Phi.shape[0]
        
        Gx_phys, Gy_phys = G_basis[:, :, 0] / jnp.maximum(1.0, x_rng), G_basis[:, :, 1] / jnp.maximum(1.0, y_rng)
        K_grad = (Gx_phys.T @ Gx_phys + Gy_phys.T @ Gy_phys) / num_pts
        
        Bxx_phys = H_basis[:, :, 0] / (jnp.maximum(1.0, x_rng)**2)
        Byy_phys = H_basis[:, :, 1] / (jnp.maximum(1.0, y_rng)**2)
        Bxy_phys = H_basis[:, :, 2] / (jnp.maximum(1.0, x_rng) * jnp.maximum(1.0, y_rng))
        K_bending = (Bxx_phys.T @ Bxx_phys + Byy_phys.T @ Byy_phys + 2.0 * Bxy_phys.T @ Bxy_phys) / num_pts
        
        System_Matrix = (Phi.T @ Phi) / num_pts + reg_lambda * K_bending + grad_lambda * K_grad
        System_Matrix += jnp.eye(self.num_basis) * 1e-10
        
        @vmap
        def solve_frame(z_frame):
            p = solve(System_Matrix, (Phi.T @ z_frame) / num_pts)
            z_fit = Phi @ p
            rmse = jnp.sqrt(jnp.mean((z_frame - z_fit)**2))
            return p, rmse
            
        params, rmses = solve_frame(Z_history)
        stats = jnp.array([x_ctr, x_rng, y_ctr, y_rng])
        return params, rmses, stats

class PlateMechanicsSolver:
    """
    [WHTOOLS] 기계공학적 판 이론(Plate Theory) 물리 해석기
    """
    def __init__(self, config: PlateConfig):
        self.cfg = config
        self.D = (config.youngs_modulus * config.thickness**3) / (12.0 * (1.0 - config.poisson_ratio**2))
        self.res = config.mesh_resolution
        self.optimizer = AdvancedPlateOptimizer(degree_x=config.poly_degree, degree_y=config.poly_degree)

    def setup_mesh(self, x_bounds: Tuple[float, float], y_bounds: Tuple[float, float]):
        self.x_lin = jnp.linspace(x_bounds[0], x_bounds[1], self.res)
        self.y_lin = jnp.linspace(y_bounds[0], y_bounds[1], self.res)
        self.X_mesh, self.Y_mesh = jnp.meshgrid(self.x_lin, self.y_lin)

    @partial(jit, static_argnums=(0,))
    def evaluate_batch(self, p_coeffs_batch: jnp.ndarray, norm_stats: jnp.ndarray):
        x_ctr, x_rng, y_ctr, y_rng = norm_stats
        X_flat, Y_flat = self.X_mesh.ravel(), self.Y_mesh.ravel()
        X_norm, Y_norm = (X_flat - x_ctr) / x_rng, (Y_flat - y_ctr) / y_rng
        Phi_mesh = self.optimizer.get_basis_matrix(X_norm, Y_norm)
        H_mesh = self.optimizer.get_hessian_basis(X_norm, Y_norm)

        def compute_triple_derivatives(xi, yi):
            triple_el = []
            for i, j in self.optimizer.basis_indices:
                w_xxx = i*(i-1)*(i-2) * xi**(jnp.maximum(0,i-3)) * yi**j if i>=3 else 0.0
                w_yyy = j*(j-1)*(j-2) * xi**i * yi**(jnp.maximum(0,j-3)) if j>=3 else 0.0
                w_xxy = i*(i-1)*j * xi**(jnp.maximum(0,i-2)) * yi**(jnp.maximum(0,j-1)) if i>=2 and j>=1 else 0.0
                w_xyy = i*j*(j-1) * xi**(jnp.maximum(0,i-1)) * yi**(jnp.maximum(0,j-2)) if i>=1 and j>=2 else 0.0
                triple_el.append(jnp.array([w_xxx, w_yyy, w_xxy, w_xyy]))
            return jnp.stack(triple_el, axis=0)

        def compute_gradient_basis(xi, yi):
            grad_el = []
            for i, j in self.optimizer.basis_indices:
                dw_dx = i * xi**(jnp.maximum(0, i-1)) * yi**j if i>=1 else 0.0
                dw_dy = j * xi**i * yi**(jnp.maximum(0, j-1)) if j>=1 else 0.0
                grad_el.append(jnp.array([dw_dx, dw_dy]))
            return jnp.stack(grad_el, axis=0)

        T_mesh = vmap(compute_triple_derivatives)(X_flat, Y_flat)
        G_mesh = vmap(compute_gradient_basis)(X_flat, Y_flat)

        @vmap
        def evaluate_single_frame(p):
            w_field = (Phi_mesh @ p).reshape(self.res, self.res)
            k_raw = -jnp.einsum('nkd,k->nd', H_mesh, p).reshape(self.res, self.res, 3)
            kxx, kyy, kxy = k_raw[..., 0], k_raw[..., 1], k_raw[..., 2]
            stress_coeff = 6.0 * self.D / (self.cfg.thickness**2)
            sigma_x_bending = stress_coeff * (kxx + self.cfg.poisson_ratio * kyy)
            sigma_y_bending = stress_coeff * (kyy + self.cfg.poisson_ratio * kxx)
            tau_xy_bending = stress_coeff * (1.0 - self.cfg.poisson_ratio) * kxy
            total_sx, total_sy = sigma_x_bending, sigma_y_bending

            if self.cfg.theory_type == "MINDLIN":
                t_vals = jnp.einsum('nkd,k->nd', T_mesh, p).reshape(self.res, self.res, 4)
                Vx = -self.D * (t_vals[..., 0] + t_vals[..., 3])
                Vy = -self.D * (t_vals[..., 1] + t_vals[..., 2])
            elif self.cfg.theory_type == "VON_KARMAN":
                g_vals = jnp.einsum('nkd,k->nd', G_mesh, p).reshape(self.res, self.res, 2)
                dw_dx, dw_dy = g_vals[..., 0], g_vals[..., 1]
                E_eff = self.cfg.youngs_modulus / (1.0 - self.cfg.poisson_ratio**2)
                total_sx += E_eff * (0.5 * dw_dx**2 + self.cfg.poisson_ratio * 0.5 * dw_dy**2)
                total_sy += E_eff * (0.5 * dw_dy**2 + self.cfg.poisson_ratio * 0.5 * dw_dx**2)

            von_mises = jnp.sqrt(total_sx**2 - total_sx*total_sy + total_sy**2 + 3*tau_xy_bending**2)
            mean_curvature = 0.5 * (kxx + kyy) 
            gauss_curvature = kxx * kyy - kxy**2
            s_mean = 0.5 * (total_sx + total_sy)
            s_diff = 0.5 * (total_sx - total_sy)
            s_radius = jnp.sqrt(jnp.maximum(s_diff**2 + tau_xy_bending**2, 1e-12))
            sigma_1, sigma_2 = s_mean + s_radius, s_mean - s_radius

            fields = {
                'Displacement [mm]': w_field, 'Stress XX [MPa]': total_sx, 'Stress YY [MPa]': total_sy,
                'Stress XY [MPa]': tau_xy_bending, 'Shear Stress XY [MPa]': tau_xy_bending,
                'Von-Mises [MPa]': von_mises, 'Max. Principal Stress [MPa]': sigma_1,
                'Min. Principal Stress [MPa]': sigma_2, 'Mean Curvature [1/mm]': mean_curvature,
                'Gauss Curvature [1/mm^2]': gauss_curvature
            }
            stats = {f'Mean-{k}': jnp.mean(v) for k, v in fields.items()}
            stats.update({f'Max-{k}': jnp.max(v) for k, v in fields.items()})
            return {**fields, **stats}
        return evaluate_single_frame(p_coeffs_batch)

class ShellDeformationAnalyzer:
    """
    [WHTOOLS] 통합 판 변형 분석기 (JAX 가속 버전)
    """
    def __init__(self, W: float = 0, H: float = 0, thickness: float = 1.0, E: float = 70e9, nu: float = 0.3, name: str = "Part"):
        self.name, self.W, self.H, self.thickness, self.E, self.nu = name, W, H, thickness, E, nu
        self.cfg = PlateConfig(thickness=thickness, youngs_modulus=E, poisson_ratio=nu)
        self.m_raw = self.m_data_hist = self.o_data = self.o_data_hint = None
        self.ref_markers = self.ref_basis = self.ref_center = self.weights = None
        self.sol = PlateMechanicsSolver(self.cfg)
        self.results = {}

    def run_analysis(self) -> bool:
        try:
            if self.m_raw is not None: return self.analyze(self.m_raw, o_data_hint=self.o_data_hint)
            elif self.m_data_hist is not None: return self.analyze(self.m_data_hist, o_data_hint=self.o_data_hint)
        except Exception as e:
            print(f"❌ Critical Error in {self.name}: {str(e)}")
            import traceback; traceback.print_exc()
        return False

    def fit_reference_plane(self, m_data_init: np.ndarray, o_data_hint: np.ndarray = None) -> np.ndarray:
        self.ref_markers = np.array(m_data_init)
        self.ref_center = np.mean(m_data_init, axis=0)
        if o_data_hint is not None:
            self.o_data = np.array(o_data_hint)
            P_world, P_local = self.ref_markers - self.ref_center, np.column_stack([self.o_data, np.zeros(len(self.o_data))])
            c_L = np.mean(P_local, axis=0)
            H = (P_local - c_L).T @ P_world
            U, S, Vt = np.linalg.svd(H)
            R = U @ Vt
            if np.linalg.det(R) < 0: R = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
            self.ref_basis = R
        else:
            U, S, Vh = np.linalg.svd(self.ref_markers - self.ref_center)
            self.ref_basis = Vh
            self.o_data = (self.ref_markers - self.ref_center) @ self.ref_basis[:2].T
            actual_w, actual_h = np.max(self.o_data[:, 0]) - np.min(self.o_data[:, 0]), np.max(self.o_data[:, 1]) - np.min(self.o_data[:, 1])
            if self.W > 0 and self.H > 0:
                if (self.W > self.H and actual_w < actual_h) or (self.W < self.H and actual_w > actual_h):
                    self.ref_basis[[0, 1]] = self.ref_basis[[1, 0]]
                    self.o_data = (self.ref_markers - self.ref_center) @ self.ref_basis[:2].T
        if self.W == 0: self.W = float(np.max(self.o_data[:, 0]) - np.min(self.o_data[:, 0]))
        if self.H == 0: self.H = float(np.max(self.o_data[:, 1]) - np.min(self.o_data[:, 1]))
        dist_sq = np.sum(np.square(self.o_data), axis=1)
        sigma = min(self.W, self.H) * 0.4
        self.weights = np.exp(-dist_sq / (2 * sigma**2 + 1e-9))
        self.weights /= np.sum(self.weights)
        r = self.cfg.margin_ratio
        margin_x, margin_y = np.clip(self.W * r, 3.0, 10.0), np.clip(self.H * r, 3.0, 10.0)
        self.sol.setup_mesh((np.min(self.o_data[:, 0]) - margin_x, np.max(self.o_data[:, 0]) + margin_x), (np.min(self.o_data[:, 1]) - margin_y, np.max(self.o_data[:, 1]) + margin_y))
        return self.o_data

    def remove_rigid_motion(self, m_data_frame: np.ndarray, prev_normal: np.ndarray = None):
        if self.ref_markers is None: return (None, None, None, None, None, None)
        W = self.weights[:, np.newaxis]
        curr_center, ref_center_w = np.sum(m_data_frame * W, axis=0), np.sum(self.ref_markers * W, axis=0)
        P_c, Q_c = self.ref_markers - ref_center_w, m_data_frame - curr_center
        H = (P_c * W).T @ Q_c
        U, S, Vt = np.linalg.svd(H); R = U @ Vt
        if np.linalg.det(R) < 0: R = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
        if prev_normal is not None:
            if np.dot(R[:, 2], prev_normal) < 0: R[:, 2] *= -1.0
        m_rigid_aligned = Q_c @ R.T
        r_rmse = np.sqrt(np.sum(np.square(m_rigid_aligned - P_c)[:, :2] * self.weights[:, np.newaxis]))
        normal_displacement = (m_rigid_aligned - P_c) @ self.ref_basis[2]
        return (R, curr_center, ref_center_w, m_rigid_aligned, normal_displacement, r_rmse)

    def analyze(self, m_data_hist: np.ndarray, o_data_hint: np.ndarray = None) -> bool:
        if np.max(np.abs(m_data_hist)) < 2.0:
            print(f"  > [Unit-Check] Meters detected in {self.name}. Scaling to mm...")
            m_data_hist *= 1000.0
            if o_data_hint is not None: o_data_hint *= 1000.0
        n_frames, n_markers, _ = m_data_hist.shape
        self.m_raw = m_data_hist
        if self.ref_markers is None: self.fit_reference_plane(m_data_hist[0], o_data_hint=o_data_hint)
        
        # [WHTOOLS] 메쉬 설정 보장 (o_data_hint가 없어도 fit_reference_plane에서 검출된 o_data 사용)
        if o_data_hint is not None:
            self.o_data = o_data_hint
        
        self.W, self.H = float(np.max(self.o_data[:, 0]) - np.min(self.o_data[:, 0])), float(np.max(self.o_data[:, 1]) - np.min(self.o_data[:, 1]))
        r = self.cfg.margin_ratio; margin_x, margin_y = np.clip(self.W*r, 3, 10), np.clip(self.H*r, 3, 10)
        self.sol.setup_mesh(
            (np.min(self.o_data[:, 0]) - margin_x, np.max(self.o_data[:, 0]) + margin_x), 
            (np.min(self.o_data[:, 1]) - margin_y, np.max(self.o_data[:, 1]) + margin_y)
        )
        all_w, all_R, all_cq, all_rmses, prev_normal = [], [], [], [], None
        for i in range(n_frames):
            R, cq, cp, m_aligned, w_disp, r_rmse = self.remove_rigid_motion(m_data_hist[i], prev_normal=prev_normal)
            all_w.append(w_disp); all_R.append(R); all_cq.append(cq); all_rmses.append(r_rmse); prev_normal = R[:, 2]
        all_displacement_w = np.stack(all_w); all_displacement_w_rel = all_displacement_w - all_displacement_w[0]
        def count_unique(coords, tol=1e-3):
            sorted_coords = np.sort(coords); diffs = np.diff(sorted_coords)
            return np.sum(diffs > tol) + 1
        n_ux, n_uy = count_unique(self.o_data[:, 0]), count_unique(self.o_data[:, 1])
        deg_x, deg_y = min(self.cfg.poly_degree, n_ux - 1), min(self.cfg.poly_degree, n_uy - 1)
        if max(self.W, self.H) / (min(self.W, self.H) + 1e-9) > 8.0:
            if self.W < self.H: deg_x = min(deg_x, 2)
            else: deg_y = min(deg_y, 2)
        self.sol.optimizer = AdvancedPlateOptimizer(degree_x=deg_x, degree_y=deg_y)
        q_loc_jax = jnp.zeros((n_frames, n_markers, 3)).at[:, :, 0].set(self.o_data[:, 0]).at[:, :, 1].set(self.o_data[:, 1]).at[:, :, 2].set(jnp.array(all_displacement_w_rel))
        p_ref_jax = jnp.column_stack([self.o_data, jnp.zeros(n_markers)])
        _, _, init_stats = self.sol.optimizer.solve_analytical(q_loc_jax[:1], p_ref_jax, self.cfg.reg_lambda)
        params, rmses, norm_stats = self.sol.optimizer.solve_analytical(q_loc_jax, p_ref_jax, self.cfg.reg_lambda, grad_lambda=self.cfg.grad_lambda, fixed_stats=init_stats)
        batch_results = self.sol.evaluate_batch(params, norm_stats)
        m_global_disp = m_data_hist - m_data_hist[0]
        dt = np.median(np.diff(self.times)) if hasattr(self, 'times') and len(self.times) > 1 else 0.001
        m_vel = np.zeros_like(m_global_disp); m_vel[1:] = np.diff(m_global_disp, axis=0) / dt
        m_acc = np.zeros_like(m_vel); m_acc[1:] = np.diff(m_vel, axis=0) / dt
        self.results = {k: np.array(v) for k, v in batch_results.items()}
        self.results.update({'R': np.stack(all_R), 'c_Q': np.stack(all_cq), 'c_P': np.repeat(self.ref_center[None, :], n_frames, axis=0), 'Q_local': np.array(q_loc_jax), 'rmse': np.array(rmses), 'r_rmse': np.array(all_rmses), 'Marker Local Disp. [mm]': all_displacement_w, 'Marker Global Disp. [mm]': m_global_disp, 'Marker Velocity [mm/s]': m_vel, 'Marker Acceleration [mm/s^2]': m_acc, 'Marker Performance [mm]': np.linalg.norm(m_global_disp, axis=2)})
        max_marker_disp, max_fit_disp = np.max(np.abs(all_displacement_w_rel)), np.max(np.abs(self.results['Displacement [mm]']))
        if max_fit_disp > max_marker_disp * 1.5 and max_marker_disp > 0.1: print(f" ⚠️ [WARN] {self.name}: Fit ({max_fit_disp:.2f}mm) > Markers ({max_marker_disp:.2f}mm)!")
        print(f"  > [PART-OK] {self.name:<24} analyzed. (Avg F-RMSE: {np.mean(rmses):.2e} mm, Avg R-RMSE: {np.mean(all_rmses):.2e} mm) [{deg_x}x{deg_y}]")
        return True

class PlateAssemblyManager:
    """
    [WHTOOLS] 다중 파트(Assembly) 구조 해석 관리자
    """
    def __init__(self, times: np.ndarray):
        self.analyzers = []
        self.times = times

    def add_analyzer(self, analyzer: ShellDeformationAnalyzer) -> ShellDeformationAnalyzer:
        analyzer.times = self.times
        self.analyzers.append(analyzer)
        return analyzer

    def run_all(self):
        num_parts = len(self.analyzers)
        print(f"\n[WHTOOLS] Multi-Part Assembly Analysis Started ({num_parts} parts)...")
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(lambda analyzer: analyzer.run_analysis(), self.analyzers))
        print(f"[WHTOOLS] Assembly Analysis Completed.")
