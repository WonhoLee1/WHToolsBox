# -*- coding: utf-8 -*-
"""
[WHTOOLS] JAX-SSR Core Physics Module
JAX 가속 기반의 Kirchhoff 박판 이론 및 Kabsch 기구학 엔진.
plate_by_markers.py에서 추출된 고성능 수치 해석 코어.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap, jit
from jax.numpy.linalg import solve
from functools import partial
from dataclasses import dataclass

# --- JAX Configuration (High Precision) ---
jax.config.update("jax_enable_x64", True)

@dataclass
class PlateConfig:
    """박판 해석을 위한 물리 및 수치 파라미터 [mm-tonne-sec]"""
    thickness: float = 2.0         # [mm]
    youngs_modulus: float = 2.1e5  # [MPa] (210 GPa)
    poisson_ratio: float = 0.3
    poly_degree: int = 5           # 다항식 차수
    reg_lambda: float = 1e-4       # Tikhonov 정규화 매개변수
    mesh_resolution: int = 40      # SSR 그리드 해상도 (기본 40x40)
    batch_size: int = 256          # 배치 처리 크기

class KinematicsManager:
    """강체 운동(Rotation/Translation)을 분리하고 로컬 좌표계를 관리합니다."""
    
    def __init__(self, marker_history: np.ndarray):
        """
        Args:
            marker_history: (n_frames, n_markers, 3) [mm]
        """
        self.raw_data = jnp.array(marker_history)
        self.n_frames, self.n_markers, _ = self.raw_data.shape
        self._setup_global_to_local()

    def _setup_global_to_local(self):
        """t=0 시점의 데이터를 기준으로 주평면(PCA) 및 로컬 좌표계를 수립합니다."""
        valid_P0 = self.raw_data[0]
        temp_centroid = jnp.mean(valid_P0, axis=0)
        P0_centered = valid_P0 - temp_centroid
        
        # PCA 기반 주평면 추출
        cov = np.cov(np.array(P0_centered).T)
        evals, evecs = np.linalg.eigh(cov)
        idx = evals.argsort()[::-1]
        self.local_axes = jnp.array(evecs[:, idx])
        
        # 경계 상자 및 중심점 보정
        p_local = P0_centered @ self.local_axes
        p_min, p_max = p_local.min(axis=0), p_local.max(axis=0)
        self.centroid_0 = temp_centroid + ((p_min + p_max) / 2.0) @ self.local_axes.T
        
        # 로컬 좌표 범위 (X, Y)
        p_local_corr = (valid_P0 - self.centroid_0) @ self.local_axes
        margin = 0.05
        self.x_bounds = [float(p_local_corr[:, 0].min() - margin), float(p_local_corr[:, 0].max() + margin)]
        self.y_bounds = [float(p_local_corr[:, 1].min() - margin), float(p_local_corr[:, 1].max() + margin)]

    @partial(jit, static_argnums=(0,))
    def extract_kinematics_vmap(self, frame_markers):
        """Kabsch 알고리즘을 배치 처리하여 강체 운동 정보를 추출합니다."""
        valid_P0 = self.raw_data[0]
        def kabsch_single(Q):
            c_P, c_Q = jnp.mean(valid_P0, axis=0), jnp.mean(Q, axis=0)
            H = (Q - c_Q).T @ (valid_P0 - c_P)
            U, S, Vt = jnp.linalg.svd(H)
            R = Vt.T @ U.T
            # Reflection correction
            R_corr = jnp.where(jnp.linalg.det(R) < 0, (Vt.T @ jnp.diag(jnp.array([1.0, 1.0, -1.0]))) @ U.T, R)
            ql = ((Q - c_Q) @ R_corr.T + c_P - self.centroid_0) @ self.local_axes
            return ql, R_corr, c_Q, c_P
        return vmap(kabsch_single)(frame_markers)

class KirchhoffPlateOptimizer:
    """다항식 피팅을 통해 Kirchhoff 박판 계수를 산출합니다."""
    
    def __init__(self, degree=5):
        self.basis_indices = [(i, j) for i in range(degree + 1) for j in range(degree + 1 - i)]
        self.n = len(self.basis_indices)

    def get_basis_matrix(self, x, y):
        """X, Y 좌표에 대한 다항식 기저 행렬 생성"""
        return jnp.stack([x**i * y**j for i, j in self.basis_indices], axis=1)

    def get_hessian_basis(self, x, y):
        """2계 도함수(곡률) 계산을 위한 Hessian 기저 생성"""
        def h_val(xi, yi):
            return jnp.stack([
                jnp.array([
                    i*(i-1)*xi**(jnp.maximum(0,i-2))*yi**j if i>=2 else 0., 
                    j*(j-1)*xi**i*yi**(jnp.maximum(0,j-2)) if j>=2 else 0., 
                    i*j*xi**(jnp.maximum(0,i-1))*yi**(jnp.maximum(0,j-1)) if i>=1 and j>=1 else 0.
                ]) for i, j in self.basis_indices
            ], axis=0)
        return vmap(h_val)(x, y)

    @partial(jit, static_argnums=(0,))
    def solve_analytical(self, q_loc, p_ref, reg):
        """Tikhonov 정규화된 최소자승법으로 다항식 계수(Coefficients)를 산출합니다."""
        # 로컬 변위 Z - p_ref(기준 높이)
        Z = q_loc[:, :, 2] - p_ref[:, 2]
        X = self.get_basis_matrix(p_ref[:, 0], p_ref[:, 1])
        H = self.get_hessian_basis(p_ref[:, 0], p_ref[:, 1])
        
        Bxx, Byy, Bxy = H[:, :, 0], H[:, :, 1], H[:, :, 2]
        # Gram Matrix + Smoothing Regularization
        M = (X.T @ X) / X.shape[0] + \
            reg * (Bxx.T @ Bxx + Byy.T @ Byy + 2.0*Bxy.T @ Bxy) / X.shape[0] + \
            jnp.eye(self.n)*1e-12
            
        return vmap(lambda z: solve(M, (X.T @ z) / X.shape[0]))(Z)

class PlateMechanicsSolver:
    """박판 물리 특성을 기반으로 응력 및 변형 필드를 계산합니다."""
    
    def __init__(self, config: PlateConfig):
        self.cfg = config
        self.D = (config.youngs_modulus * config.thickness**3) / (12.0 * (1.0 - config.poisson_ratio**2))
        self.res = config.mesh_resolution
        self.opt = KirchhoffPlateOptimizer(degree=config.poly_degree)

    def setup_mesh(self, x_bounds, y_bounds):
        """해석 그리드 초기화"""
        self.x_lin = jnp.linspace(x_bounds[0], x_bounds[1], self.res)
        self.y_lin = jnp.linspace(y_bounds[0], y_bounds[1], self.res)
        self.X_mesh, self.Y_mesh = jnp.meshgrid(self.x_lin, self.y_lin)

    @partial(jit, static_argnums=(0,))
    def evaluate_batch(self, p_coeffs_batch):
        """배치 단위로 변형 및 응력 필드(XX, Von-Mises, Strain)를 계산합니다."""
        Xf, Yf = self.X_mesh.ravel(), self.Y_mesh.ravel()
        Xm = self.opt.get_basis_matrix(Xf, Yf)
        Hm = self.opt.get_hessian_basis(Xf, Yf)
        
        @vmap
        def eval_fn(p):
            # 1. 수직 변위 w [mm]
            w = (Xm @ p).reshape(self.res, self.res)
            
            # 2. 곡률 k = -grad(grad(w)) [1/mm]
            k_raw = -jnp.einsum('nkd,k->nd', Hm, p).reshape(self.res, self.res, 3) 
            kxx, kyy, kxy = k_raw[..., 0], k_raw[..., 1], k_raw[..., 2]
            
            # 3. 응력 계산 [MPa]
            s_c = 6.0 * self.D / (self.cfg.thickness**2)
            sx = s_c * (kxx + self.cfg.poisson_ratio * kyy)
            sy = s_c * (kyy + self.cfg.poisson_ratio * kxx)
            txy = s_c * (1.0 - self.cfg.poisson_ratio) * kxy
            
            # Von-Mises 응력 (Shell Surface)
            vm = jnp.sqrt(jnp.maximum(sx**2 + sy**2 - sx*sy + 3.0*txy**2, 1e-12))
            svm = vm * jnp.sign(sx + sy) # 시각화를 위한 부호 있는 Von-Mises
            
            # 4. 등가 변형률 (Equivalent Strain)
            ex, ey, gxy = (self.cfg.thickness/2.0)*kxx, (self.cfg.thickness/2.0)*kyy, (self.cfg.thickness)*kxy
            eq_e = (2.0/3.0) * jnp.sqrt(jnp.maximum(1.5*(ex**2+ey**2)+0.75*gxy**2, 1e-20))
            
            return {
                'Displacement [mm]': w,
                'Curvature XX [1/mm]': kxx,
                'Stress XX [MPa]': sx,
                'Signed Von-Mises [MPa]': svm,
                'Eq. Strain [mm/mm]': eq_e,
                'Curvature Mean [1/mm]': -(kxx + kyy) / 2.0,
                'Curvature Gauss [1/mm^2]': kxx * kyy - kxy**2,
                'Mean-VM': jnp.mean(vm),
                'Max-VM': jnp.max(vm)
            }
            
        return eval_fn(p_coeffs_batch)

# --- SSR Helper Function ---
def compute_jax_ssr_field(markers_history: np.ndarray, config: PlateConfig):
    """
    고성능 JAX-SSR 연산을 수행하여 결과를 반환합니다.
    (whts_postprocess_engine에서 호출용)
    """
    kin = KinematicsManager(markers_history)
    sol = PlateMechanicsSolver(config)
    sol.setup_mesh(kin.x_bounds, kin.y_bounds)
    
    # t=0 기준 피팅 기저점
    p_ref = (kin.raw_data[0] - kin.centroid_0) @ kin.local_axes
    
    # 배치 분석 실행
    n_frames = kin.n_frames
    bs = config.batch_size
    results_list = []
    
    for i in range((n_frames + bs - 1) // bs):
        idx_s = i * bs
        idx_e = min((i + 1) * bs, n_frames)
        q_loc, _, _, _ = kin.extract_kinematics_vmap(kin.raw_data[idx_s:idx_e])
        params = sol.opt.solve_analytical(q_loc, p_ref, config.reg_lambda)
        batch_res = sol.evaluate_batch(params)
        results_list.append(batch_res)
        
    # 결과 병합
    full_results = {k: np.concatenate([np.array(b[k]) for b in results_list], axis=0) for k in results_list[0].keys()}
    full_results['X_mesh'] = np.array(sol.X_mesh)
    full_results['Y_mesh'] = np.array(sol.Y_mesh)
    
    return full_results
