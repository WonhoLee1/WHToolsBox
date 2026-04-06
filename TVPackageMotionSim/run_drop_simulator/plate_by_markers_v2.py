# -*- coding: utf-8 -*-
"""
[WHTOOLS] Integrated Shell Deformation Analysis Engine v2.5
기계공학적 판 이론(Plate Theory)과 JAX 고속 연산을 결합한 정밀 구조 변형 해석 모듈입니다.
본 모듈은 MuJoCo 시뮬레이션 및 실제 MoCap 계측 데이터를 모두 지원합니다.
"""

import os
import sys
import time
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6 import QtWidgets, QtCore, QtGui

import jax
import jax.numpy as jnp
from jax import vmap, jit
from jax.numpy.linalg import solve

# JAX 64비트 정밀도 활성화 (기계공학 해석의 수치 안정성을 위해 필수)
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
            # 리스트 또는 어레이 데이터를 mm 단위로 변환
            scaled_data = np.array(getattr(result, field_name)) * 1000.0
            setattr(result, field_name, scaled_data)
            
    # 기하학적 치수 변환
    if hasattr(result, 'block_half_extents'):
        for bid in result.block_half_extents:
            result.block_half_extents[bid] = [v * 1000.0 for v in result.block_half_extents[bid]]
            
    if hasattr(result, 'nominal_local_pos'):
        for bid in result.nominal_local_pos:
            result.nominal_local_pos[bid] = [v * 1000.0 for v in result.nominal_local_pos[bid]]
            
    return result

@dataclass
class PlotSlotConfig:
    """[WHTOOLS] 2D 그래프 슬롯별 시각화 구성 설정"""
    part_idx: int = 0
    plot_type: str = 'contour'         # 'contour' 또는 'curve'
    data_key: str = 'Displacement [mm]' # 해석 결과 딕셔너리의 키값

@dataclass
class DashboardConfig:
    """[WHTOOLS] 통합 대시보드 레이아웃 및 제어 설정"""
    layout_2d: str = '2x2'
    plots_2d: List[PlotSlotConfig] = field(default_factory=list)
    v_font_size: int = 9
    animation_step: int = 1
    animation_speed_ms: int = 30

@dataclass
class PlateConfig:
    """
    [WHTOOLS] 판(Plate) 물리적 특성 및 수치 해석 설정
    기계공학적 강성 및 다항식 근사 차수를 제어합니다.
    """
    thickness: float = 2.0             # 판 두께 [mm]
    youngs_modulus: float = 2.1e5      # 영률 (Young's Modulus) [MPa]
    poisson_ratio: float = 0.3         # 포아송 비 (Poisson's Ratio)
    poly_degree: int = 4               # 다항식 근사 차수 (Polynomial Degree)
    reg_lambda: float = 1e-4           # Tikhonov 정규화 계수
    mesh_resolution: int = 25          # 시각화 격자 해상도
    batch_size: int = 256              # JAX 배치 처리 크기
    theory_type: str = "KIRCHHOFF"     # 'KIRCHHOFF', 'MINDLIN', 'VON_KARMAN'
    shear_correction: float = 5.0/6.0  # 전단 보정 계수 (Mindlin 이론용)
    margin_ratio: float = 0.05         # [WHTOOLS] 해석 영역 외구 마진 비율 (초기값 5%)
                                       # [Importance] 좁은 면에서의 발산 방지와 시각적 가독성 사이의 균형을 조절합니다.


class AlignmentManager:
    """
    [WHTOOLS] 마커 데이터의 좌표계 정렬 및 강체 운동(Rigid Body Motion) 관리자
    Kabsch 알고리즘을 사용하여 전역 좌표계의 마커를 설계 로컬 좌표계(u, v, w)로 정렬합니다.
    """
    def __init__(self, raw_markers, W, H, offsets):
        # 입력 데이터의 JAX 어레이 변환 및 기본 정보 저장
        self.raw_data = jnp.array(raw_markers)
        self.W, self.H = W, H
        self.offsets = jnp.array(offsets)
        self.n_frames, self.n_markers, _ = self.raw_data.shape
        
        # 초기 정렬(Calibration) 수행
        self._calibrate()

    def _calibrate(self):
        """
        [WHTOOLS] 0번 프레임을 기준으로 설계 좌표계와 정렬하기 위한 초기 회전/병진 행렬 산출
        """
        # 초기 프레임 마커 및 목표 설계 좌표(offsets) 준비
        P0 = np.array(self.raw_data[0])
        # 설계 좌표에 z=0을 추가하여 3D 점으로 구성
        P_target = np.column_stack([self.offsets, np.zeros(self.n_markers)])
        
        # 각 점집합의 중심점(Centroid) 계산
        c_P = np.mean(P0, axis=0)
        c_T = np.mean(P_target, axis=0)
        
        # 공분산 행렬 H = P_target^T @ P0 (Standard Kabsch for P -> Q)
        H_mat = (P_target - c_T).T @ (P0 - c_P)
        
        # SVD(Singular Value Decomposition)를 통한 최적 회전 행렬 산출
        U, S, Vt = np.linalg.svd(H_mat)
        R = U @ Vt
        
        # 반사(Reflection) 방지 처리
        if np.linalg.det(R) < 0:
            R = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
            
        # 결과 저장: R은 P(Target) -> Q(P0) 정방향 매핑
        self.local_axes = R.T
        self.centroid_0 = c_P
        
        # 해석 경계 영역 설정 (설계 치수 기반)
        self.x_bounds = [-self.W/2 - 5, self.W/2 + 5]
        self.y_bounds = [-self.H/2 - 5, self.H/2 + 5]

    @partial(jit, static_argnums=(0,))
    def extract_kinematics_vmap(self, frame_markers):
        """
        [WHTOOLS] JAX vmap을 이용한 고속 강체 운동 추출
        P(Ref) -> Q(Current) 정방향 회전 산출 및 로컬 투영
        """
        P_ref = jnp.column_stack([self.offsets, jnp.zeros(self.n_markers)])
        c_P_ref = jnp.mean(P_ref, axis=0)
        
        def kabsch_single(Q):
            c_Q = jnp.mean(Q, axis=0)
            
            # H = P^T @ Q (Forward mapping: P -> Q)
            H = (P_ref - c_P_ref).T @ (Q - c_Q)
            U, S, Vt = jnp.linalg.svd(H)
            
            R = U @ Vt
            R_corr = jnp.where(
                jnp.linalg.det(R) < 0, 
                U @ jnp.diag(jnp.array([1.0, 1.0, -1.0])) @ Vt, 
                R
            )
            
            # 투영된 로컬 마커 좌표 산출 (Q = P @ R 이므로 P = Q @ R.T)
            q_local = (Q - c_Q) @ R_corr + c_P_ref
            
            return q_local, R_corr, c_Q, c_P_ref
            
        return vmap(kabsch_single)(frame_markers)


class AdvancedPlateOptimizer:
    """
    [WHTOOLS] 다항식 표면 근사(Polynomial Surface Fitting) 최적화 엔진
    마커 데이터를 받아 변위 함수 w(x, y)를 r차 다항식으로 최적화하여 근사합니다.
    """
    def __init__(self, degree_x: int = 4, degree_y: int = 4):
        # [WHTOOLS] 축별 차수를 독립적으로 설정할 수 있도록 텐서 곱(Tensor Product) 기반 기저 구성
        # 이는 좁은 면(Side Face)에서 한쪽 축의 차수를 낮추어 수치적 안정성을 확보하는 데 유리합니다.
        self.basis_indices = [
            (i, j) for i in range(degree_x + 1) for j in range(degree_y + 1)
        ]
        self.num_basis = len(self.basis_indices)
        self.degree_x = degree_x
        self.degree_y = degree_y

    def get_basis_matrix(self, x_coords: jnp.ndarray, y_coords: jnp.ndarray) -> jnp.ndarray:
        """[WHTOOLS] 설계 좌표 (x, y)에 대한 다항식 기저 행렬(Phi) 생성: Phi_ij = x^i * y^j"""
        return jnp.stack(
            [x_coords**i * y_coords**j for i, j in self.basis_indices], 
            axis=1
        )

    def get_hessian_basis(self, x_coords: jnp.ndarray, y_coords: jnp.ndarray) -> jnp.ndarray:
        """
        [WHTOOLS] 곡률 계산을 위한 2계 도함수(Hessian) 기저 행렬 생성
        반환값: [N, Degree, 3] (w,xx, w,yy, w,xy)
        """
        def compute_single_hessian(xi, yi):
            # 각 기저 함수(x^i * y^j)의 2계 미분을 미리 계산
            hessian_elements = []
            for i, j in self.basis_indices:
                w_xx = i*(i-1) * xi**(jnp.maximum(0, i-2)) * yi**j if i >= 2 else 0.0
                w_yy = j*(j-1) * xi**i * yi**(jnp.maximum(0, j-2)) if j >= 2 else 0.0
                w_xy = i*j * xi**(jnp.maximum(0, i-1)) * yi**(jnp.maximum(0, j-1)) if i >= 1 and j >= 1 else 0.0
                hessian_elements.append(jnp.array([w_xx, w_yy, w_xy]))
            return jnp.stack(hessian_elements, axis=0)
            
        return vmap(compute_single_hessian)(x_coords, y_coords)

    @partial(jit, static_argnums=(0,))
    def solve_analytical(self, q_local: jnp.ndarray, p_ref: jnp.ndarray, reg_lambda: float):
        """
        [WHTOOLS] 해석적 기법을 이용한 전 시계열 다항식 계수 배치 산출
        L2 정규화(Tikhonov)와 굽힘 에너지 최소화(Bending Energy Penalty)를 병행합니다.
        [Stability] 데이터 범위 정규화를 통해 수치적 안정성을 확보합니다.
        """
        # 측정 축(Z) 준비
        Z_history = q_local[:, :, 2]
        
        # [WHTOOLS] Min-Max 기반 정규화 (데이터 범위를 [-1, 1]로 강제 매핑)
        # Z-score(std) 방식은 좁은 면에서 분산이 극도로 작아질 때 수치적 발산을 초래할 수 있어 이를 보완합니다.
        x_raw, y_raw = p_ref[:, 0], p_ref[:, 1]
        
        x_min, x_max = jnp.min(x_raw), jnp.max(x_raw)
        y_min, y_max = jnp.min(y_raw), jnp.max(y_raw)
        
        x_ctr = (x_max + x_min) / 2.0
        y_ctr = (y_max + y_min) / 2.0
        x_rng = (x_max - x_min) / 2.0 + 1e-9 # Zero-division 방지
        y_rng = (y_max - y_min) / 2.0 + 1e-9
        
        x_norm = (x_raw - x_ctr) / x_rng
        y_norm = (y_raw - y_ctr) / y_rng
        
        # 정규화된 좌표로 기저 행렬 준비
        Phi = self.get_basis_matrix(x_norm, y_norm)
        
        # 2계 도함수 기반의 굽힘 패널티 행렬(Hessian) 구성
        H_basis = self.get_hessian_basis(x_norm, y_norm)
        Bxx, Byy, Bxy = H_basis[:, :, 0], H_basis[:, :, 1], H_basis[:, :, 2]
        
        # [WHTOOLS] Physical Curvature Scaling (물리적 곡률 스케일링)
        # 정규화된 좌표([-1, 1])에서의 미분값은 물리적 실제 곡률과 x_rng**2 배 차이가 납니다.
        # 이를 상쇄해야 좁은 면에서도 등방성(Isotropic) 굽힘 에너지가 유지됩니다.
        num_pts = Phi.shape[0]
        Bxx_phys = Bxx / (jnp.maximum(1.0, x_rng)**2)
        Byy_phys = Byy / (jnp.maximum(1.0, y_rng)**2)
        Bxy_phys = Bxy / (jnp.maximum(1.0, x_rng) * jnp.maximum(1.0, y_rng))
        
        # 물리적 곡률 기반 페널티 행렬 (Tikhonov Regularization)
        K_penalty = (Bxx_phys.T @ Bxx_phys + Byy_phys.T @ Byy_phys + 2.0 * Bxy_phys.T @ Bxy_phys) / num_pts
        
        # 최종 시스템 행렬 M: 최소자승법 + 정규화
        System_Matrix = (Phi.T @ Phi) / num_pts + reg_lambda * K_penalty
        System_Matrix += jnp.eye(self.num_basis) * 1e-10  # 수치 안정성을 위한 작은 값
        
        @vmap
        def solve_frame(z_frame):
            # 개별 프레임에 대한 계수(p) 및 RMSE 계산
            p = solve(System_Matrix, (Phi.T @ z_frame) / num_pts)
            z_fit = Phi @ p
            rmse = jnp.sqrt(jnp.mean((z_frame - z_fit)**2))
            return p, rmse
            
        params, rmses = solve_frame(Z_history)
        
        # [WHTOOLS] 정규화 파라미터 반환 (Evaluation 시 동일하게 적용 필요)
        stats = jnp.array([x_ctr, x_rng, y_ctr, y_rng])
        return params, rmses, stats



class PlateMechanicsSolver:
    """
    [WHTOOLS] 기계공학적 판 이론(Plate Theory) 물리 해석기
    구해진 다항식 계수를 바탕으로 전 영역의 응력 및 변형 필드를 생성합니다.
    """
    def __init__(self, config: PlateConfig):
        self.cfg = config
        # 굽힘 강성 D = (E * h^3) / (12 * (1 - nu^2))
        self.D = (config.youngs_modulus * config.thickness**3) / \
                 (12.0 * (1.0 - config.poisson_ratio**2))
        self.res = config.mesh_resolution
        # Optimizer 초기 생성 시 기본 차수 사용 (analyze에서 재지정 가능)
        self.optimizer = AdvancedPlateOptimizer(degree_x=config.poly_degree, degree_y=config.poly_degree)

    def setup_mesh(self, x_bounds: Tuple[float, float], y_bounds: Tuple[float, float]):
        """[WHTOOLS] 해석 및 가시화를 위한 정규 격자(Mesh) 생성"""
        self.x_lin = jnp.linspace(x_bounds[0], x_bounds[1], self.res)
        self.y_lin = jnp.linspace(y_bounds[0], y_bounds[1], self.res)
        self.X_mesh, self.Y_mesh = jnp.meshgrid(self.x_lin, self.y_lin)

    @partial(jit, static_argnums=(0,))
    def evaluate_batch(self, p_coeffs_batch: jnp.ndarray, norm_stats: jnp.ndarray):
        """
        [WHTOOLS] JAX 가속을 이용한 시계열/해석 필드 일괄 산출
        norm_stats: [x_ctr, x_rng, y_ctr, y_rng]
        """
        x_ctr, x_rng, y_ctr, y_rng = norm_stats
        
        # 격자 좌표 플래닝 및 정규화 적용
        X_flat, Y_flat = self.X_mesh.ravel(), self.Y_mesh.ravel()
        X_norm = (X_flat - x_ctr) / x_rng
        Y_norm = (Y_flat - y_ctr) / y_rng
        
        # 정규화된 좌표로 기저 행렬 사전 계산
        Phi_mesh = self.optimizer.get_basis_matrix(X_norm, Y_norm)
        H_mesh = self.optimizer.get_hessian_basis(X_norm, Y_norm) # [N, Deg, 3]


        def compute_triple_derivatives(xi, yi):
            """전단력(Shear force) 계산을 위한 3계 도함수 산출"""
            triple_el = []
            for i, j in self.optimizer.basis_indices:
                w_xxx = i*(i-1)*(i-2) * xi**(jnp.maximum(0,i-3)) * yi**j if i>=3 else 0.0
                w_yyy = j*(j-1)*(j-2) * xi**i * yi**(jnp.maximum(0,j-3)) if j>=3 else 0.0
                w_xxy = i*(i-1)*j * xi**(jnp.maximum(0,i-2)) * yi**(jnp.maximum(0,j-1)) if i>=2 and j>=1 else 0.0
                w_xyy = i*j*(j-1) * xi**(jnp.maximum(0,i-1)) * yi**(jnp.maximum(0,j-2)) if i>=1 and j>=2 else 0.0
                triple_el.append(jnp.array([w_xxx, w_yyy, w_xxy, w_xyy]))
            return jnp.stack(triple_el, axis=0)

        def compute_gradient_basis(xi, yi):
            """경사도(Slope) 계산을 위한 1계 도함수 산출"""
            grad_el = []
            for i, j in self.optimizer.basis_indices:
                dw_dx = i * xi**(jnp.maximum(0, i-1)) * yi**j if i>=1 else 0.0
                dw_dy = j * xi**i * yi**(jnp.maximum(0, j-1)) if j>=1 else 0.0
                grad_el.append(jnp.array([dw_dx, dw_dy]))
            return jnp.stack(grad_el, axis=0)

        # 3계 및 1계 도함수 기저 생성 (병렬)
        T_mesh = vmap(compute_triple_derivatives)(X_flat, Y_flat)
        G_mesh = vmap(compute_gradient_basis)(X_flat, Y_flat)

        @vmap
        def evaluate_single_frame(p):
            """개별 시점(Frame)의 계수(p)로부터 물리량 매핑"""
            # 1. 변위 필드 필링
            w_field = (Phi_mesh @ p).reshape(self.res, self.res)
            
            # 2. 곡률(Curvatures) 추출: k = -d^2w/dx^2
            k_raw = -jnp.einsum('nkd,k->nd', H_mesh, p).reshape(self.res, self.res, 3)
            kxx, kyy, kxy = k_raw[..., 0], k_raw[..., 1], k_raw[..., 2]
            
            # 3. 굽힘 응력(Bending Stress) 상수 및 산출
            stress_coeff = 6.0 * self.D / (self.cfg.thickness**2)
            sigma_x_bending = stress_coeff * (kxx + self.cfg.poisson_ratio * kyy)
            sigma_y_bending = stress_coeff * (kyy + self.cfg.poisson_ratio * kxx)
            tau_xy_bending = stress_coeff * (1.0 - self.cfg.poisson_ratio) * kxy
            
            # 초기화 (추가 이론 모드용)
            sigma_x_membrane, sigma_y_membrane, tau_xz, tau_yz = 0.0, 0.0, 0.0, 0.0

            # 4. 고차 이론 모델링 (선택 사양)
            if self.cfg.theory_type == "MINDLIN":
                # 가로 전단 변형 효과 (Mindlin-Reissner)
                t_vals = jnp.einsum('nkd,k->nd', T_mesh, p).reshape(self.res, self.res, 4)
                # 전단력 Qx = -D * (w,xxx + w,xyy)
                Vx = -self.D * (t_vals[..., 0] + t_vals[..., 3])
                Vy = -self.D * (t_vals[..., 1] + t_vals[..., 2])
                tau_xz = Vx / (self.cfg.thickness * self.cfg.shear_correction)
                tau_yz = Vy / (self.cfg.thickness * self.cfg.shear_correction)
                
            elif self.cfg.theory_type == "VON_KARMAN":
                # 대변형 막 응력 효과 (Von-Karman Nonlinearity)
                g_vals = jnp.einsum('nkd,k->nd', G_mesh, p).reshape(self.res, self.res, 2)
                dw_dx, dw_dy = g_vals[..., 0], g_vals[..., 1]
                E_eff = self.cfg.youngs_modulus / (1.0 - self.cfg.poisson_ratio**2)
                sigma_x_membrane = E_eff * (0.5 * dw_dx**2 + self.cfg.poisson_ratio * 0.5 * dw_dy**2)
                sigma_y_membrane = E_eff * (0.5 * dw_dy**2 + self.cfg.poisson_ratio * 0.5 * dw_dx**2)

            # 5. 최종 필드 통합 및 Von-Mises 응력 계산
            total_sx = sigma_x_bending + sigma_x_membrane
            total_sy = sigma_y_bending + sigma_y_membrane
            
            # Von-Mises [MPa]: sqrt(sx^2 + sy^2 - sx*sy + 3*(txy^2 + txz^2 + tyz^2))
            von_mises = jnp.sqrt(jnp.maximum(
                total_sx**2 + total_sy**2 - total_sx*total_sy + \
                3.0 * (tau_xy_bending**2 + tau_xz**2 + tau_yz**2), 
                1e-12
            ))
            
            fields = {
                'Displacement [mm]': w_field,
                'Stress XX [MPa]': total_sx,
                'Stress YY [MPa]': total_sy,
                'Shear Stress XY [MPa]': tau_xy_bending,
                'Von-Mises [MPa]': von_mises
            }
            
            # 통계 데이터 추가 (Mean, Max)
            stats = {f'Mean-{k}': jnp.mean(v) for k, v in fields.items()}
            stats.update({f'Max-{k}': jnp.max(v) for k, v in fields.items()})
            
            return {**fields, **stats}

        return evaluate_single_frame(p_coeffs_batch)


class ShellDeformationAnalyzer:
    """
    [WHTOOLS] 통합 판 변형 분석기 (JAX 가속 버전)
    개별 파트의 마커 데이터를 입력받아 자율적으로 좌표계를 설정하고, 
    강체 운동을 제거한 뒤 판 이론에 따른 구조 변형 해석을 수행합니다.
    """
    def __init__(self, W: float = 0, H: float = 0, 
                 thickness: float = 1.0, E: float = 70e9, 
                 nu: float = 0.3, name: str = "Part"):
        self.name = name
        self.W, self.H = W, H
        self.thickness, self.E, self.nu = thickness, E, nu
        
        # 물리 해석 설정
        self.cfg = PlateConfig(thickness=thickness, youngs_modulus=E, poisson_ratio=nu)
        
        # 데이터 관리
        self.m_raw = None          # [Frames, Markers, 3] 원본 마커 히스토리
        self.m_data_hist = None    # [Markers, Frames, 3] 분석용 (analyzer.m_data_hist 셰이프 호환용)
        self.o_data = None         # [Markers, 2] 로컬 2D 좌표 (u, v)
        self.o_data_hint = None    # 외부 좌표 힌트
        
        self.ref_markers = None    # [N, 3] 기준 마커
        self.ref_basis = None      # [u, v, w] 로컬 기저
        self.ref_center = None     # 초기 중심점
        
        self.sol = PlateMechanicsSolver(self.cfg)
        self.results = {}

    def run_analysis(self) -> bool:
        """[WHTOOLS] 통합 해석 실행 및 예외 처리"""
        try:
            if self.m_raw is not None:
                return self.analyze(self.m_raw, o_data_hint=self.o_data_hint)
            elif self.m_data_hist is not None:
                 return self.analyze(self.m_data_hist, o_data_hint=self.o_data_hint)
        except Exception as e:
            print(f"❌ Critical Error in {self.name}: {str(e)}")
            import traceback
            traceback.print_exc()
        return False

    def fit_reference_plane(self, m_data_init: np.ndarray, o_data_hint: np.ndarray = None) -> np.ndarray:
        """[WHTOOLS] 초기 프레임 데이터를 이용한 로컬 평면 및 설계 좌표계 정의"""
        self.ref_markers = np.array(m_data_init)
        self.ref_center = np.mean(m_data_init, axis=0)
        
        if o_data_hint is not None:
            # [WHTOOLS] 힌트 기반 캘리브레이션 (Legacy v2 호환)
            # 힌트 좌표(o_data_hint)를 3D 평면상의 타겟으로 설정하고, 실제 마커와 Kabsch 정렬을 통해 기저를 산출합니다.
            self.o_data = np.array(o_data_hint)
            P_world = self.ref_markers - self.ref_center
            P_local = np.column_stack([self.o_data, np.zeros(len(self.o_data))])
            c_L = np.mean(P_local, axis=0)
            
            # P_local @ R = P_world  =>  H = P_local^T @ P_world (Forward: Local -> World)
            H = (P_local - c_L).T @ P_world
            U, S, Vt = np.linalg.svd(H)
            R = U @ Vt
            if np.linalg.det(R) < 0:
                R = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
            
            # R의 행 벡터(row vectors)가 곧 로컬 u, v, n 축이 됩니다.
            # 이 Basis는 p_local @ basis = p_world 관계를 완벽히 충족합니다.
            self.ref_basis = R
        else:
            # [WHTOOLS] 자율적 PCA 기반 기저 산출
            U, S, Vh = np.linalg.svd(self.ref_markers - self.ref_center)
            self.ref_basis = Vh # [3, 3] matrix of row vectors (u, v, n)
            
            # 데이터 방향성 보정 (설계 W < H 인데 데이터 상의 u방향 분산이 더 큰 경우 축 스왑)
            self.o_data = (self.ref_markers - self.ref_center) @ self.ref_basis[:2].T
            actual_w = np.max(self.o_data[:, 0]) - np.min(self.o_data[:, 0])
            actual_h = np.max(self.o_data[:, 1]) - np.min(self.o_data[:, 1])
            if self.W > 0 and self.H > 0:
                if (self.W > self.H and actual_w < actual_h) or (self.W < self.H and actual_w > actual_h):
                    self.ref_basis[[0, 1]] = self.ref_basis[[1, 0]]
                    self.o_data = (self.ref_markers - self.ref_center) @ self.ref_basis[:2].T

        if self.W == 0: self.W = float(np.max(self.o_data[:, 0]) - np.min(self.o_data[:, 0]))
        if self.H == 0: self.H = float(np.max(self.o_data[:, 1]) - np.min(self.o_data[:, 1]))
        
        # 해석용 메쉬 그리드 설정 (Adaptive Margin)
        r = self.cfg.margin_ratio
        margin_x = np.clip(self.W * r, 3.0, 10.0)
        margin_y = np.clip(self.H * r, 3.0, 10.0)
        
        min_x, max_x = np.min(self.o_data[:, 0]) - margin_x, np.max(self.o_data[:, 0]) + margin_x
        min_y, max_y = np.min(self.o_data[:, 1]) - margin_y, np.max(self.o_data[:, 1]) + margin_y
        self.sol.setup_mesh((min_x, max_x), (min_y, max_y))
        return self.o_data

    def remove_rigid_motion(self, m_data_frame: np.ndarray):
        """
        [WHTOOLS] Orthogonal Procrustes 기반 강체 운동 제거 (Forward Mapping 버전)
        현재 프레임의 마커 셋을 기준 셋(Reference)과 최적으로 중첩시켜 순수 변형량을 추출합니다.
        [Concept] Q_curr = (P_ref @ R + cq) 관계에서 R과 cq를 추출합니다.
        """
        if self.ref_markers is None:
            return (None, None, None, None, None, None)
            
        curr_center = np.mean(m_data_frame, axis=0)
        
        # 1. P_c(Reference) -> Q_c(Current) 정방향 회전 산출
        P_c = self.ref_markers - self.ref_center
        Q_c = m_data_frame - curr_center
        
        # 2. 공분산 행렬 H = P^T @ Q (Forward: P -> Q)
        H = P_c.T @ Q_c
        U, S, Vt = np.linalg.svd(H)
        
        # 3. 회전 행렬 R = U @ Vt (P -> Q 매핑)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            R = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
            
        # 4. 강체 운동 제거: 현재 마커 Q를 기준 좌표계 P로 정렬
        # Q = P @ R 이 성립하므로 P = Q @ R.T 가 됨
        m_rigid_aligned = Q_c @ R.T
        
        # 5. 성능 지표 산출 (Rigid Registration RMSE)
        diff_vector = m_rigid_aligned - P_c
        r_rmse = np.sqrt(np.mean(np.square(diff_vector[:, :2])))
        
        # 6. 로컬 면외 변위(Normal) 산출 (Reference Normal Basis 활용)
        normal_displacement = diff_vector @ self.ref_basis[2]
        
        return (R, curr_center, self.ref_center, m_rigid_aligned, normal_displacement, r_rmse)

    def analyze(self, m_data_hist: np.ndarray, o_data_hint: np.ndarray = None) -> bool:
        """
        [WHTOOLS] JAX 가속 엔진 배치 해석 (Full Cleanup & Stabilization)
        """
        m_data_hist = np.array(m_data_hist)
        
        # Shape 정규화
        if o_data_hint is not None and m_data_hist.shape[1] != o_data_hint.shape[0]:
            if m_data_hist.shape[0] == o_data_hint.shape[0]:
                m_data_hist = np.swapaxes(m_data_hist, 0, 1)

        n_frames, n_markers, _ = m_data_hist.shape
        self.m_raw = m_data_hist 
        
        if self.ref_markers is None:
            self.fit_reference_plane(m_data_hist[0], o_data_hint=o_data_hint)
            
        if o_data_hint is not None:
            self.o_data = o_data_hint
            self.W = float(np.max(self.o_data[:, 0]) - np.min(self.o_data[:, 0]))
            self.H = float(np.max(self.o_data[:, 1]) - np.min(self.o_data[:, 1]))
            r = self.cfg.margin_ratio
            margin_x, margin_y = np.clip(self.W*r, 3, 10), np.clip(self.H*r, 3, 10)
            self.sol.setup_mesh((np.min(self.o_data[:, 0])-margin_x, np.max(self.o_data[:, 0])+margin_x),
                                (np.min(self.o_data[:, 1])-margin_y, np.max(self.o_data[:, 1])+margin_y))
            
        # 강체 운동 제거 및 로컬 변위 추출
        all_w, all_R, all_cq, all_rmses = [], [], [], []
        for i in range(n_frames):
            R, cq, cp, m_aligned, w_disp, r_rmse = self.remove_rigid_motion(m_data_hist[i])
            all_w.append(w_disp)
            all_R.append(R)
            all_cq.append(cq)
            all_rmses.append(r_rmse)
            
        all_displacement_w = np.stack(all_w) 
        
        # 차수 적응 장치 (Markers-Density Aware)
        def count_unique(coords, tol=1e-3):
            sorted_coords = np.sort(coords)
            diffs = np.diff(sorted_coords)
            return np.sum(diffs > tol) + 1

        n_unique_x = count_unique(self.o_data[:, 0])
        n_unique_y = count_unique(self.o_data[:, 1])
        
        deg_x = min(self.cfg.poly_degree, n_unique_x - 1)
        deg_y = min(self.cfg.poly_degree, n_unique_y - 1)
        
        aspect_ratio = max(self.W, self.H) / (min(self.W, self.H) + 1e-9)
        if aspect_ratio > 8.0:
            if self.W < self.H: deg_x = min(deg_x, 2)
            else:               deg_y = min(deg_y, 2)

        self.sol.optimizer = AdvancedPlateOptimizer(degree_x=deg_x, degree_y=deg_y)
        
        # JAX 솔버 구동
        all_displacement_w_rel = all_displacement_w - all_displacement_w[0]
        q_loc_jax = jnp.zeros((n_frames, n_markers, 3))
        q_loc_jax = q_loc_jax.at[:, :, 0].set(self.o_data[:, 0])
        q_loc_jax = q_loc_jax.at[:, :, 1].set(self.o_data[:, 1])
        q_loc_jax = q_loc_jax.at[:, :, 2].set(jnp.array(all_displacement_w_rel))

        p_ref_jax = jnp.column_stack([self.o_data, jnp.zeros(n_markers)])
        params, rmses, norm_stats = self.sol.optimizer.solve_analytical(q_loc_jax, p_ref_jax, self.cfg.reg_lambda)
        batch_results = self.sol.evaluate_batch(params, norm_stats)

        # 결과 패키징
        self.results = {k: np.array(v) for k, v in batch_results.items()}
        self.results.update({
            'R': np.stack(all_R),
            'c_Q': np.stack(all_cq),
            'c_P': np.repeat(self.ref_center[None, :], n_frames, axis=0),
            'Q_local': np.array(q_loc_jax),
            'rmse': np.array(rmses),
            'r_rmse': np.array(all_rmses),
            'Marker Local Disp. [mm]': all_displacement_w
        })
        
        max_marker_disp = np.max(np.abs(all_displacement_w_rel))
        max_fit_disp = np.max(np.abs(self.results['Displacement [mm]']))
        
        warn_str = ""
        if max_fit_disp > max_marker_disp * 1.5 and max_marker_disp > 0.1:
            warn_str = f" ⚠️ [WARN] Fit displacement ({max_fit_disp:.2f}mm) exceeds markers ({max_marker_disp:.2f}mm)!"
            
        print(f"  > [PART-OK] {self.name:<24} analyzed. (Avg F-RMSE: {np.mean(rmses):.2e} mm, Avg R-RMSE: {np.mean(all_rmses):.2e} mm) [{deg_x}x{deg_y}]{warn_str}")
        return True

class PlateAssemblyManager:
    """
    [WHTOOLS] 다중 파트(Assembly) 구조 해석 관리자
    병렬 프로세싱을 통해 대규모 어셈블리의 해석 속도를 극대화합니다.
    """
    def __init__(self, times: np.ndarray):
        self.analyzers = []
        self.times = times
        self.n_frames = len(times)

    def add_analyzer(self, analyzer: ShellDeformationAnalyzer) -> ShellDeformationAnalyzer:
        """분석기 추가"""
        self.analyzers.append(analyzer)
        return analyzer

    def run_all(self):
        """[WHTOOLS] 모든 파트에 대해 병렬 해석 수행 (Threaded Execution)"""
        num_parts = len(self.analyzers)
        print(f"\n[WHTOOLS] Multi-Part Assembly Analysis Started ({num_parts} parts)...")
        
        # I/O 및 데이터 처리 위주이므로 CPU 바운드 작업에 대해 ThreadPool 활용
        with ThreadPoolExecutor(max_workers=8) as executor:
            # 각 파트별 run_analysis() 실행
            results = list(executor.map(lambda analyzer: analyzer.run_analysis(), self.analyzers))
        
        print(f"[WHTOOLS] Assembly Analysis Completed. Processes: {len(results)}")


class VisibilityToolWindow(QtWidgets.QWidget):
    """
    [WHTOOLS] 가시성 관리자 (Visibility Manager)
    각 파트의 메쉬 및 마커 가시성을 트리 구조로 제어합니다.
    """
    def __init__(self, parent=None):
        super().__init__(parent, QtCore.Qt.Window | QtCore.Qt.WindowStaysOnTopHint)
        self.setWindowTitle("Visibility Manager")
        self.resize(350, 500)
        self.parent = parent
        
        # 메인 레이아웃 구성
        layout = QtWidgets.QVBoxLayout(self)
        
        # 1. Global Control 섹션
        global_group = QtWidgets.QGroupBox("Global Control")
        global_layout = QtWidgets.QVBoxLayout(global_group)
        
        for title, col_idx in [("Mesh", 1), ("Markers", 2)]:
            h_layout = QtWidgets.QHBoxLayout()
            h_layout.addWidget(QtWidgets.QLabel(f"{title}:"))
            
            btn_show = QtWidgets.QPushButton("Show All")
            btn_hide = QtWidgets.QPushButton("Hide All")
            
            btn_show.clicked.connect(partial(self._bulk_set, col_idx, True))
            btn_hide.clicked.connect(partial(self._bulk_set, col_idx, False))
            
            h_layout.addWidget(btn_show)
            h_layout.addWidget(btn_hide)
            global_layout.addLayout(h_layout)
            
        layout.addWidget(global_group)
        
        # 2. 파트별 트리 위젯 구성
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Part", "Mesh", "Markers"])
        self.tree.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.tree)
        
        self.groups = {}
        self.id_to_item = {}
        self._init_tree()

    def _init_tree(self):
        """[WHTOOLS] 어셈블리 구조를 분석하여 트리 아이템 초기화"""
        self.tree.blockSignals(True)
        for i, part in enumerate(self.parent.mgr.analyzers):
            # 접두어(Prefix)를 기준으로 그룹핑 (예: Plate_01, Plate_02 -> Plate 그룹)
            prefix = part.name.split('_')[0]
            if prefix not in self.groups:
                self.groups[prefix] = QtWidgets.QTreeWidgetItem(self.tree, [prefix])
                self.groups[prefix].setExpanded(True)
                
            item = QtWidgets.QTreeWidgetItem(self.groups[prefix], [part.name])
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setData(0, QtCore.Qt.UserRole, i)
            
            # 현재 가시성 상태 반영
            mesh_state = QtCore.Qt.Checked if self.parent.part_actors[i]['visible'] else QtCore.Qt.Unchecked
            marker_state = QtCore.Qt.Checked if self.parent.part_actors[i]['visible_markers'] else QtCore.Qt.Unchecked
            
            item.setCheckState(1, mesh_state)
            item.setCheckState(2, marker_state)
            self.id_to_item[i] = item
            
        self.tree.blockSignals(False)

    def _bulk_set(self, column: int, state: bool):
        """[WHTOOLS] 일괄 가시성 설정"""
        self.tree.blockSignals(True)
        check_state = QtCore.Qt.Checked if state else QtCore.Qt.Unchecked
        
        for i in range(self.tree.topLevelItemCount()):
            group = self.tree.topLevelItem(i)
            group.setCheckState(column, check_state)
            for j in range(group.childCount()):
                group.child(j).setCheckState(column, check_state)
                
        self.tree.blockSignals(False)
        self._apply()

    def _on_item_changed(self, item, column):
        """[WHTOOLS] 트리 아이템 체크 상태 변경 시 하위 아이템 동기화 및 적용"""
        if item.data(0, QtCore.Qt.UserRole) is None:  # 그룹 아이템인 경우
            self.tree.blockSignals(True)
            for j in range(item.childCount()):
                item.child(j).setCheckState(column, item.checkState(column))
            self.tree.blockSignals(False)
        self._apply()

    def _apply(self):
        """[WHTOOLS] 설정된 가시성 상태를 3D 뷰어에 반영"""
        for i, item in self.id_to_item.items():
            self.parent.part_actors[i]['visible'] = (item.checkState(1) == QtCore.Qt.Checked)
            self.parent.part_actors[i]['visible_markers'] = (item.checkState(2) == QtCore.Qt.Checked)
        
        self.parent.update_frame(self.parent.current_frame)

class AddPlotDialog(QtWidgets.QDialog):
    """
    [WHTOOLS] 2D 그래프 추가 다이얼로그
    특정 파트의 물리량(Contour) 또는 통계량(Time Series)을 선택하여 슬롯에 배치합니다.
    """
    def __init__(self, slot_idx, parts, field_keys, stat_keys, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Add Plot to Slot {slot_idx + 1}")
        
        layout = QtWidgets.QVBoxLayout(self)
        grid = QtWidgets.QGridLayout()
        layout.addLayout(grid)
        
        # 1. 파트 선택
        grid.addWidget(QtWidgets.QLabel("Part:"), 0, 0)
        self.cmb_part = QtWidgets.QComboBox()
        self.cmb_part.addItems(parts)
        grid.addWidget(self.cmb_part, 0, 1)
        
        # 2. 그래프 타입 선택 (Contour vs Curve)
        grid.addWidget(QtWidgets.QLabel("Type:"), 1, 0)
        h_box = QtWidgets.QHBoxLayout()
        self.rb_contour = QtWidgets.QRadioButton("Contour")
        self.rb_curve = QtWidgets.QRadioButton("Curve")
        self.rb_contour.setChecked(True)
        h_box.addWidget(self.rb_contour)
        h_box.addWidget(self.rb_curve)
        grid.addLayout(h_box, 1, 1)
        
        # 3. 데이터 키 선택
        grid.addWidget(QtWidgets.QLabel("Key:"), 2, 0)
        self.cmb_key = QtWidgets.QComboBox()
        grid.addWidget(self.cmb_key, 2, 1)
        
        self.f_keys = field_keys
        self.s_keys = stat_keys
        
        # 신호 연결
        self.rb_contour.toggled.connect(self._update_keys)
        self.rb_curve.toggled.connect(self._update_keys)
        
        self._update_keys()
        
        # 확인/취소 버튼
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _update_keys(self):
        """그래프 타입에 따라 선택 가능한 데이터 키 목록 갱신"""
        self.cmb_key.clear()
        if self.rb_contour.isChecked():
            self.cmb_key.addItems(self.f_keys)
        else:
            self.cmb_key.addItems(self.s_keys)

    def get_config(self) -> PlotSlotConfig:
        """선택된 설정을 Config 객체로 반환"""
        return PlotSlotConfig(
            part_idx=self.cmb_part.currentIndex(), 
            plot_type="contour" if self.rb_contour.isChecked() else "curve", 
            data_key=self.cmb_key.currentText()
        )

class AboutDialog(QtWidgets.QDialog):
    """
    [WHTOOLS] 프로그램 정보 다이얼로그
    WHTOOLS의 브랜드 아이덴티티와 적용된 핵심 기술들을 소개합니다.
    """
    def __init__(self, logo_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About WHTOOLS Dashboard")
        self.setFixedSize(550, 650)
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)
        
        # 1. 로고 이미지
        if os.path.exists(logo_path):
            img_label = QtWidgets.QLabel()
            pixmap = QtGui.QPixmap(logo_path).scaledToHeight(220, QtCore.Qt.SmoothTransformation)
            img_label.setPixmap(pixmap)
            img_label.setAlignment(QtCore.Qt.AlignCenter)
            layout.addWidget(img_label)
            
        # 2. 제목 및 비전
        title = QtWidgets.QLabel("WHTOOLS Structural Dashboard v5.9")
        title.setStyleSheet("font-size: 20pt; font-weight: bold; color: #1A73E8;")
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title)
        
        subtitle = QtWidgets.QLabel("Expert Structural Analysis & Digital Twin Solution")
        subtitle.setStyleSheet("font-size: 11pt; color: #5F6368; font-style: italic;")
        subtitle.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(subtitle)
        
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        layout.addWidget(line)
        
        # 3. 핵심 기술 소개
        features = QtWidgets.QLabel(
            "<b>Advanced Computational Core:</b><br>"
            "• <b>Multi-Theory Shell Solver:</b> Kirchhoff / Mindlin / Von Karman High-Fidelity formulations<br>"
            "• <b>JAX-SSR Engine:</b> Ultra-fast surface reconstruction via XLA-compiled JAX core<br>"
            "• <b>Autonomous Alignment:</b> SVD-based plane fitting & Procrustes rigid body removal<br>"
            "• <b>Real-time Integration:</b> Seamless Digital Twin synchronization with MuJoCo/Qt pipeline<br>"
            "• <b>Expert Visualization:</b> Multi-slot 3D/2D visual interaction with premium aesthetics"
        )
        features.setStyleSheet("font-size: 11pt; line-height: 170%; color: #3C4043;")
        features.setWordWrap(True)
        layout.addWidget(features)
        
        layout.addStretch()
        
        # 4. 푸터 및 닫기 버튼
        copyright_label = QtWidgets.QLabel("© 2026 WHTOOLS. All Rights Reserved.")
        copyright_label.setStyleSheet("font-size: 9pt; color: #9AA0A6;")
        copyright_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(copyright_label)
        
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.setFixedWidth(120)
        btn_close.setStyleSheet("padding: 10px; font-weight: bold; background-color: #F8F9FA; border: 1px solid #DADCE0;")
        btn_close.clicked.connect(self.accept)
        
        h_layout = QtWidgets.QHBoxLayout()
        h_layout.addStretch()
        h_layout.addWidget(btn_close)
        h_layout.addStretch()
        layout.addLayout(h_layout)


class QtVisualizerV2(QtWidgets.QMainWindow):
    """
    [WHTOOLS] 차세대 구조 변형 분석 대시보드 (V2)
    VTK 기반 3D 뷰어와 Matplotlib 기반 2D 그래프를 결합하여 실시간 데이터 분석을 지원합니다.
    """
    def __init__(
        self, 
        manager: PlateAssemblyManager, 
        config: DashboardConfig = None, 
        ground_size=(3000, 3000)
    ):
        super().__init__()
        # 1. 상태 및 설정 초기화
        self.mgr = manager
        self.cfg = config or DashboardConfig()
        self.ground_size = ground_size
        
        self.current_frame = 0
        self.is_playing = False
        self.active_slot = 0
        self.anim_step = self.cfg.animation_step
        self.plot_slots: List[Optional[PlotSlotConfig]] = [None] * 6
        self.part_actors = {}
        self.v_font_size = self.cfg.v_font_size
        
        # 2. 분석 키(Key) 추출
        # 첫 번째 분석기를 기준으로 필드 및 통계 키를 분류
        p0 = manager.analyzers[0]
        n_frames = len(self.mgr.times)
        res_sq = p0.sol.res**2
        
        # 3D 맵핑 가능한 필드 키 (그리드 크기와 일치하는 데이터)
        self.field_keys = [
            k for k in p0.results 
            if p0.results[k].ndim == 3 and p0.results[k].size // n_frames == res_sq
        ]
        
        # 2D 그래프용 통계 키
        self.stat_keys = [
            k for k in p0.results 
            if k not in self.field_keys and p0.results[k].ndim < 3
        ] + ['Marker Local Disp. [mm]', 'Marker Global Disp. [mm]']
        
        # 3. 리소스 경로 설정 (폰트, 로고 등)
        curr_dir = os.path.dirname(__file__)
        self.res_dir = os.path.join(curr_dir, "resources")
        self.logo_path = os.path.join(self.res_dir, "logo.png")
        self.font_path = os.path.join(self.res_dir, "D2Coding-Ver1.3.2-20180524-ligature.ttf")

        # 4. 앱 폰트 설정 (가독성 최우선)
        if os.path.exists(self.font_path):
            fid = QtGui.QFontDatabase.addApplicationFont(self.font_path)
            if fid != -1:
                font_name = QtGui.QFontDatabase.applicationFontFamilies(fid)[0]
                font = QtGui.QFont(font_name, 9)
                QtWidgets.QApplication.setFont(font)
        
        # 5. UI 및 뷰어 초기화
        self._init_ui()
        self._init_3d_view()
        self._init_2d_plots()
        self.update_frame(0)
        
        self.visibility_tool = VisibilityToolWindow(self)

    def _init_ui(self):
        """[WHTOOLS] 메인 윈도우 레이아웃 및 구성 요소 초기화"""
        self.setWindowTitle("WHTOOLS Structural Dashboard v5.9")
        self.resize(1650, 980)
        
        # 메뉴바 초기화
        self._init_menus()
        
        # 중앙 위젯 및 메인 수직 레이아웃
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        
        # 상단 애니메이션 컨트롤 툴바
        self._init_animation_toolbar()
        
        self.split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(self.split)
        
        # 좌측 3D 패널
        self.p3d = QtWidgets.QWidget()
        l3d = QtWidgets.QVBoxLayout(self.p3d)
        self._init_3d_panel(l3d)
        
        # 우측 2D 패널
        self.p2d = QtWidgets.QWidget()
        l2d = QtWidgets.QVBoxLayout(self.p2d)
        self._init_2d_panel(l2d)
        
        self.split.addWidget(self.p3d)
        self.split.addWidget(self.p2d)
        
        # 초기 비율 설정 (3D:2D = 6:4)
        self.split.setStretchFactor(0, 6)
        self.split.setStretchFactor(1, 4)
        
    def _init_3d_panel(self, layout: QtWidgets.QVBoxLayout):
        """[WHTOOLS] 3D 제어 패널 구성 - 누락 메서드 복구"""
        # 1. 3D Interactor (VTK Viewer)
        self.v_int = QtInteractor(self.p3d)
        self.v_int.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.v_int.customContextMenuRequested.connect(self._show_part_menu)
        layout.addWidget(self.v_int)
        
        # 2. 3D Control Group
        ctrl_group = QtWidgets.QGroupBox("3D Control")
        grid = QtWidgets.QGridLayout(ctrl_group)
        
        # View Mode (Global vs Local)
        grid.addWidget(QtWidgets.QLabel("View:"), 0, 0)
        self.cmb_view = QtWidgets.QComboBox()
        self.cmb_view.addItems(["Global", "Local"])
        self.cmb_view.currentTextChanged.connect(lambda: self.update_frame(self.current_frame))
        grid.addWidget(self.cmb_view, 0, 1)
        
        # Deformation Scale
        grid.addWidget(QtWidgets.QLabel("Scale:"), 0, 2)
        self.spin_scale = QtWidgets.QDoubleSpinBox()
        self.spin_scale.setRange(1.0, 500.0)
        self.spin_scale.setValue(1.0)
        self.spin_scale.setSingleStep(1.0)
        self.spin_scale.valueChanged.connect(lambda: self.update_frame(self.current_frame))
        grid.addWidget(self.spin_scale, 0, 3)
        
        # Field Selection
        grid.addWidget(QtWidgets.QLabel("Field:"), 1, 0)
        self.cmb_3d = QtWidgets.QComboBox()
        self.cmb_3d.addItems(["Body Color", "Face Color"] + self.field_keys)
        self.cmb_3d.currentTextChanged.connect(lambda: self.update_frame(self.current_frame))
        grid.addWidget(self.cmb_3d, 1, 1)
        
        # Legend Mode (Dynamic vs Static)
        grid.addWidget(QtWidgets.QLabel("Legend:"), 1, 2)
        self.cmb_leg = QtWidgets.QComboBox()
        self.cmb_leg.addItems(["Dynamic", "Static"])
        self.cmb_leg.currentTextChanged.connect(self._on_legend_mode_changed)
        grid.addWidget(self.cmb_leg, 1, 3)
        
        # Legend Range (Static)
        self.spin_min = QtWidgets.QDoubleSpinBox()
        self.spin_max = QtWidgets.QDoubleSpinBox()
        for s in [self.spin_min, self.spin_max]:
            s.setRange(-1e6, 1e6)
            s.setDecimals(4)
            s.valueChanged.connect(lambda: self.update_frame(self.current_frame))
            
        grid.addWidget(QtWidgets.QLabel("Min:"), 2, 0)
        grid.addWidget(self.spin_min, 2, 1)
        grid.addWidget(QtWidgets.QLabel("Max:"), 2, 2)
        grid.addWidget(self.spin_max, 2, 3)
        
        layout.addWidget(ctrl_group)

    def _init_animation_toolbar(self):
        """[WHTOOLS] 재생 컨트롤 툴바 구성"""
        toolbar = self.addToolBar("Animation Control")
        toolbar.setFixedHeight(45)
        toolbar.setMovable(False)
        
        # 내비게이션 버튼들 (<<, <, >, >>)
        controls = [("<<", 0), ("<", -1), (">", 1), (">>", 9999)]
        for text, step in controls:
            btn = QtWidgets.QPushButton(text)
            btn.setFixedSize(35, 30)
            btn.clicked.connect(partial(self._ctrl_slot, step))
            toolbar.addWidget(btn)
            
        toolbar.addSeparator()
        
        # 재생/일시정지 버튼
        self.btn_play = QtWidgets.QPushButton("▶")
        self.btn_play.setFixedSize(45, 30)
        self.btn_play.setStyleSheet("font-weight: bold; color: #1A73E8;")
        self.btn_play.clicked.connect(lambda: self._ctrl_slot(-2))
        toolbar.addWidget(self.btn_play)
        
        toolbar.addSeparator()
        
        # 프레임 슬라이더
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, self.mgr.n_frames - 1)
        self.slider.valueChanged.connect(self.update_frame)
        self.slider.setMinimumWidth(500)
        toolbar.addWidget(self.slider)
        
        toolbar.addSeparator()
        
        # 프레임 정보 라벨
        self.lbl_f = QtWidgets.QLabel("Frame: 0")
        self.lbl_f.setMinimumWidth(120)
        toolbar.addWidget(self.lbl_f)
        
        toolbar.addSeparator()
        
        # 재생 간격 (Step) 설정
        toolbar.addWidget(QtWidgets.QLabel(" Step: "))
        self.cmb_step = QtWidgets.QComboBox()
        self.cmb_step.addItems([str(i) for i in range(1, 11)])
        self.cmb_step.setCurrentText(str(self.anim_step))
        self.cmb_step.currentTextChanged.connect(self._on_step_changed)
        toolbar.addWidget(self.cmb_step)
        
        # 재생 속도 (Speed) 설정
        toolbar.addWidget(QtWidgets.QLabel(" Interval(ms): "))
        self.cmb_speed = QtWidgets.QComboBox()
        self.cmb_speed.addItems(["0", "15", "30", "50", "100", "200"])
        self.cmb_speed.setCurrentText(str(self.cfg.animation_speed_ms))
        self.cmb_speed.currentTextChanged.connect(self._on_speed_changed)
        toolbar.addWidget(self.cmb_speed)

    def _init_2d_panel(self, layout: QtWidgets.QVBoxLayout):
        """[WHTOOLS] 2D 그래프 패널 및 레이아웃 제어부 구성"""
        ctrl_group = QtWidgets.QGroupBox("2D Plot Control")
        h_layout = QtWidgets.QHBoxLayout(ctrl_group)
        
        # 1. 그리드 레이아웃 설정
        self.cmb_layout = QtWidgets.QComboBox()
        self.cmb_layout.addItems(["1x1", "1x2", "2x2", "3x2"])
        self.cmb_layout.setCurrentText(self.cfg.layout_2d)
        self.cmb_layout.currentTextChanged.connect(self._init_2d_plots)
        
        h_layout.addWidget(QtWidgets.QLabel("Layout:"))
        h_layout.addWidget(self.cmb_layout)
        
        # 2. 기능 버튼 (그래프 추가, 팝업 뷰)
        for text, func in [("+ Plot", self._show_add_plot_dialog), ("Pop-out", self._pop_out_2d)]:
            btn = QtWidgets.QPushButton(text)
            btn.clicked.connect(func)
            h_layout.addWidget(btn)
            
        # 3. 제어 옵션 (동기화, 보간)
        self.checks = {}
        for text, initial_state in [("Sync", True), ("Interp", True)]:
            check = QtWidgets.QCheckBox(text)
            check.setChecked(initial_state)
            check.toggled.connect(lambda: self.update_frame(self.current_frame))
            h_layout.addWidget(check)
            self.checks[text] = check
            
        layout.addWidget(ctrl_group)
        
        # 4. Canvas 컨테이너
        self._canvas_widget = QtWidgets.QWidget()
        self._canvas_layout = QtWidgets.QVBoxLayout(self._canvas_widget)
        self._canvas_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas_widget, stretch=1)

    def _init_3d_view(self):
        """[WHTOOLS] VTK 뷰어의 렌더링 파이프라인 및 초기 객체 생성"""
        self.v_int.set_background("white")
        self.v_int.add_axes()
        
        # 글로벌 테마 설정
        pv.global_theme.font.size = 12
        pv.global_theme.font.color = 'black'
        
        # 바닥면(Ground) 생성
        ground_plane = pv.Plane(i_size=self.ground_size[0], j_size=self.ground_size[1])
        self.ground = self.v_int.add_mesh(ground_plane, color="blue", opacity=0.1)
        
        # 컬러맵 설정
        self.lut = pv.LookupTable(cmap="turbo")
        self.lut.below_range_color = 'lightgrey'
        self.lut.above_range_color = 'magenta'
        
        # 각 파트별 메쉬 및 마커 액터 생성
        for i, analyzer in enumerate(self.mgr.analyzers):
            # [WHTOOLS] Robustness Guard: 해석 실패 부품 건너뛰기
            if analyzer.m_raw is None:
                continue
                
            # 구조체 메쉬

            poly = pv.Plane(
                i_size=analyzer.W, 
                j_size=analyzer.H, 
                i_resolution=analyzer.sol.res - 1, 
                j_resolution=analyzer.sol.res - 1
            )
            poly.point_data["S"] = np.zeros(analyzer.sol.res**2)
            
            mesh_actor = self.v_int.add_mesh(
                poly, 
                scalars="S", 
                cmap=self.lut, 
                show_edges=True, 
                edge_color="darkgray", 
                show_scalar_bar=False
            )
            
            # 마커용 PolyData
            markers_poly = pv.PolyData(np.array(analyzer.m_raw[0]))
            n_markers = analyzer.m_raw.shape[1]
            markers_poly.point_data["names"] = [
                f"{analyzer.name}_M{j:02d}" for j in range(n_markers)
            ]
            
            # 마커 구체 및 라벨
            marker_actor = self.v_int.add_mesh(
                markers_poly, 
                render_points_as_spheres=True, 
                point_size=10, 
                color='skyblue'
            )
            label_actor = self.v_int.add_point_labels(
                markers_poly, 
                "names", 
                font_size=self.v_font_size, 
                text_color='black', 
                always_visible=True, 
                point_size=0, 
                shadow=False
            )
            
            # 초기 상태: 마커/라벨 숨김
            marker_actor.SetVisibility(False)
            label_actor.SetVisibility(False)
            
            self.part_actors[i] = {
                'mesh': mesh_actor,
                'poly': poly,
                'm_poly': markers_poly,
                'markers': marker_actor,
                'labels': label_actor,
                'visible': True,
                'visible_markers': False,
                'p_base': np.column_stack([
                    analyzer.sol.X_mesh.ravel(), 
                    analyzer.sol.Y_mesh.ravel(), 
                    np.zeros(analyzer.sol.res**2)
                ])
            }
            
        # [WHTOOLS] 스칼라바 구성 (첫 번째 유효한 액터 기준)
        if self.part_actors:
            first_idx = min(self.part_actors.keys())
            self.scalar_bar = self.v_int.add_scalar_bar(
                "Field", 
                position_x=0.15, 
                position_y=0.05, 
                width=0.7, 
                mapper=self.part_actors[first_idx]['mesh'].mapper, 
                vertical=False, 
                n_labels=5, 
                fmt="%.2e"
            )
        else:
            # 기본 스칼라바 (데이터 없을 경우)
            self.scalar_bar = self.v_int.add_scalar_bar("No Data", position_x=0.15)
            self.scalar_bar.SetVisibility(False)

        
        # 폰트 적용 (D2Coding)
        if self.font_path:
            for prop in [self.scalar_bar.GetLabelTextProperty(), self.scalar_bar.GetTitleTextProperty()]:
                prop.SetFontFile(self.font_path)
                prop.SetFontSize(self.v_font_size + 1)
                prop.SetColor(0, 0, 0)
                prop.BoldOn()
                
        # 뷰어 초기화
        self.v_int.view_isometric()
        self.v_int.enable_parallel_projection()
        
        # 애니메이션 타이머
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(lambda: self._ctrl_slot(1))

    def _init_2d_plots(self):
        """[WHTOOLS] 2D Matplotlib 캔버스 및 서브플롯 초기화"""
        # 기존 위젯 제거
        for i in reversed(range(self._canvas_layout.count())):
            self._canvas_layout.itemAt(i).widget().setParent(None)
            
        # 캔버스 생성
        self.figure = Figure(figsize=(8, 8))
        self.canvas = FigureCanvas(self.figure)
        self._canvas_layout.addWidget(NavigationToolbar(self.canvas, self))
        self._canvas_layout.addWidget(self.canvas)
        
        self.canvas.mpl_connect('button_press_event', self._on_axis_clicked)
        
        # 레이아웃 결정
        layout_map = {"1x1": (1, 1), "1x2": (1, 2), "2x2": (2, 2), "3x2": (3, 2)}
        rows, cols = layout_map.get(self.cmb_layout.currentText(), (2, 2))
        
        self.axes = []
        self.ims = [None] * 6
        self.vlines = [None] * 6
        
        self.figure.clear()
        self.figure.subplots_adjust(hspace=0.4, wspace=0.3)
        
        for i in range(rows * cols):
            ax = self.figure.add_subplot(rows, cols, i + 1)
            self.axes.append(ax)
            if self.plot_slots[i]:
                ax.set_title(f"Slot {i+1}: {self.plot_slots[i].data_key}")
            else:
                ax.text(0.5, 0.5, f"Empty Slot {i+1}\nClick to add", ha='center', transform=ax.transAxes)
                
        self._update_selection_ui()
        self.canvas.draw_idle()

    def update_frame(self, frame_idx: int):
        """[WHTOOLS] 핵심 프레임 업데이트 루틴 (3D/2D 동시 동기화)"""
        self.current_frame = frame_idx
        
        # UI 동기화
        self.slider.blockSignals(True)
        self.slider.setValue(frame_idx)
        self.slider.blockSignals(False)
        self.lbl_f.setText(f"Frame: {frame_idx} | Time: {self.mgr.times[frame_idx]:.3f}s")
        
        # 3D 옵션 획득
        view_mode = self.cmb_view.currentText()
        field_key = self.cmb_3d.currentText()
        scale = self.spin_scale.value()
        is_dynamic = self.cmb_leg.currentText() == "Dynamic"
        
        all_visible_values = []
        
        for i, analyzer in enumerate(self.mgr.analyzers):
            if i not in self.part_actors:
                continue
                
            info = self.part_actors[i]
            is_visible = info['visible']
            markers_visible = is_visible and info['visible_markers']
            
            # 가시성 상태 갱신
            info['mesh'].SetVisibility(is_visible)
            info['markers'].SetVisibility(markers_visible)
            info['labels'].SetVisibility(markers_visible)
            
            if not is_visible:
                continue
                
            # 1. 기하학적 형태 변형 적용
            w_disp = analyzer.results['Displacement [mm]'][frame_idx]
            p_deformed = info['p_base'].copy()
            p_deformed[:, 2] = w_disp.ravel() * scale
            
            # 2. 강체 운동 및 좌표계 변환 (Step 4: Basis-Aware Recovery)
            R = analyzer.results['R'][frame_idx]
            cq = analyzer.results['c_Q'][frame_idx]
            basis = analyzer.ref_basis # [3, 3] -> (u, v, n) vectors
            
            if view_mode == "Global":
                # [WHTOOLS] P_local -> Q_world 변환 (Forward Mapping)
                p_world_oriented = p_deformed @ basis
                info['poly'].points = p_world_oriented @ R + cq
                info['m_poly'].points = np.array(analyzer.m_raw[frame_idx])


                info['m_poly'].points = np.array(analyzer.m_raw[frame_idx])
            else:
                info['poly'].points = p_deformed
                info['m_poly'].points = np.array(analyzer.results['Q_local'][frame_idx])


                
            # 3. 색상/필드 데이터 맵핑
            if field_key in ["Body Color", "Face Color"]:
                info['mesh'].mapper.scalar_visibility = False
                info['mesh'].GetProperty().SetColor(plt.cm.tab20(i % 20)[:3])
            else:
                info['mesh'].mapper.scalar_visibility = True
                key = field_key if field_key in analyzer.results else 'Displacement [mm]'
                f_val = analyzer.results[key][frame_idx]
                
                if f_val.size == analyzer.sol.res**2:
                    info['poly'].point_data["S"] = f_val.ravel()
                    info['poly'].set_active_scalars("S")
                    all_visible_values.append(f_val)
                    
            info['poly'].Modified()
            info['m_poly'].Modified()
            
        # 4. 컬러 레전드 및 통계 오버레이 업데이트
        if all_visible_values and field_key not in ["Body Color", "Face Color"]:
            combined = np.concatenate([v.ravel() for v in all_visible_values])
            v_min, v_max = float(combined.min()), float(combined.max())
            
            if is_dynamic:
                clim = [v_min, v_max]
                self.spin_min.blockSignals(True)
                self.spin_min.setValue(v_min)
                self.spin_min.blockSignals(False)
                self.spin_max.blockSignals(True)
                self.spin_max.setValue(v_max)
                self.spin_max.blockSignals(False)
            else:
                clim = [float(self.spin_min.value()), float(self.spin_max.value())]
                
            if clim[0] >= clim[1]:
                clim[1] = clim[0] + 1e-6
                
            self.lut.scalar_range = (clim[0], clim[1])
            self.scalar_bar.SetVisibility(True)
            self.scalar_bar.title = field_key
            for actor_info in self.part_actors.values():
                actor_info['mesh'].mapper.SetScalarRange(clim[0], clim[1])
                
            stats_text = f"[{field_key}]\nMin: {v_min:.3e}\nMax: {v_max:.3e}"
            self.v_int.add_text(
                stats_text, 
                position='upper_left', 
                font_size=self.v_font_size, 
                color='black', 
                name='stat_overlay'
            )
        else:
            self.scalar_bar.SetVisibility(False)
            self.v_int.add_text("", position='upper_left', name='stat_overlay')
            
        # 2D 플롯 업데이트
        self._update_2d_plots(frame_idx)
        self.v_int.render()

    def _update_2d_plots(self, frame_idx: int):
        """[WHTOOLS] 2D 차트의 시간축 및 데이터 데이터 맵핑 갱신"""
        current_time = self.mgr.times[frame_idx]
        use_interp = self.checks['Interp'].isChecked()
        
        for i, ax in enumerate(self.axes):
            config = self.plot_slots[i]
            if not config:
                continue
                
            analyzer = self.mgr.analyzers[config.part_idx]
            key = config.data_key
            
            if config.plot_type == "contour":
                data_2d = analyzer.results[key][frame_idx]
                if self.ims[i] is None:
                    ax.clear()
                    self.ims[i] = ax.imshow(data_2d, cmap='turbo', origin='lower')
                    self.figure.colorbar(self.ims[i], ax=ax, format="%.2e")
                self.ims[i].set_data(data_2d)
                self.ims[i].set_interpolation('bilinear' if use_interp else 'nearest')
                ax.set_title(f"[{analyzer.name}] {key}")
            else:
                if self.vlines[i] is None:
                    ax.clear()
                    ax.grid(True, alpha=0.3)
                    series_data = analyzer.results[key] if key in analyzer.results else analyzer.results['Marker Local Disp. [mm]']
                    if series_data.ndim == 1:
                        ax.plot(self.mgr.times, series_data, color='#1A73E8')
                    else:
                        for m in range(min(series_data.shape[1], 12)):
                            ax.plot(self.mgr.times, series_data[:, m], alpha=0.5, label=f"M{m}")
                    self.vlines[i] = ax.axvline(current_time, color='red', ls='--')
                    ax.set_ylabel(key)
                    ax.set_xlabel("Time [s]")
                    
                self.vlines[i].set_xdata([current_time])
                ax.set_title(f"[{analyzer.name}] {key}")
                
        self.canvas.draw_idle()

    def _pop_out_2d(self):
        """[WHTOOLS] 2D 그래프 창을 독립적인 창으로 분리하여 출력 (전문 분석용)"""
        self.pop_win = QtWidgets.QMainWindow(self)
        self.pop_win.setWindowTitle("Multi-Layer Analysis View")
        self.pop_win.resize(1100, 850)
        
        central = QtWidgets.QWidget()
        self.pop_win.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        
        fig = Figure(figsize=(10, 10))
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        
        layout_map = {"1x1": (1, 1), "1x2": (1, 2), "2x2": (2, 2), "3x2": (3, 2)}
        rows, cols = layout_map.get(self.cmb_layout.currentText(), (2, 2))
        
        for i in range(rows * cols):
            ax = fig.add_subplot(rows, cols, i + 1)
            config = self.plot_slots[i]
            if config:
                ana = self.mgr.analyzers[config.part_idx]
                key = config.data_key
                if config.plot_type == "contour":
                    im = ax.imshow(ana.results[key][self.current_frame], cmap='turbo', origin='lower')
                    fig.colorbar(im, ax=ax)
                else: 
                    vals = ana.results[key] if key in ana.results else ana.results['Marker Local Disp. [mm]']
                    if vals.ndim == 1:
                        ax.plot(self.mgr.times, vals)
                    else:
                        for m in range(min(vals.shape[1], 10)):
                            ax.plot(self.mgr.times, vals[:, m], alpha=0.5)
                    ax.axvline(self.mgr.times[self.current_frame], color='red')
            else:
                ax.text(0.5, 0.5, "Empty Slot", ha='center', transform=ax.transAxes)
                
        canvas.draw()
        self.pop_win.show()

    def _init_menus(self):
        """[WHTOOLS] 시스템 메뉴바 초기화"""
        mb = self.menuBar()
        # Settings 메뉴
        settings_menu = mb.addMenu("Settings")
        settings_menu.addAction("Visibility Manager", lambda: self.visibility_tool.show())
        settings_menu.addAction("Reset Camera (f)", lambda: self.v_int.reset_camera())
        
        # Help 메뉴
        help_menu = mb.addMenu("Help")
        help_menu.addAction("About WHTOOLS", self._show_about)

    def _show_about(self):
        """프로그램 정보 다이얼로그 출력"""
        dialog = AboutDialog(self.logo_path, self)
        dialog.exec()

    def _on_step_changed(self, value):
        """애니메이션 프레임 증감분 변경"""
        self.anim_step = int(value)

    def _on_speed_changed(self, value):
        """애니메이션 재생 간격(ms) 변경"""
        self.timer.setInterval(int(value))

    def _on_theory_changed(self, theory_name: str):
        """[WHTOOLS] 적용 해석 이론(Kirchhoff/Mindlin/Von Karman) 실시간 교체 및 재분석"""
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            print(f"\n[WHTOOLS] Switching Theory to {theory_name} & Re-analyzing...")
            for analyzer in self.mgr.analyzers:
                analyzer.cfg.theory_type = theory_name
                analyzer.run_analysis()
                
            # 콤보박스 아이템 동기화 (이론에 따라 결과 필드가 달라질 수 있음)
            p0 = self.mgr.analyzers[0]
            n_frames = len(self.mgr.times)
            res_sq = p0.sol.res**2
            field_keys = [
                k for k in p0.results 
                if p0.results[k].ndim == 3 and p0.results[k].size // n_frames == res_sq
            ]
            
            self.cmb_3d.blockSignals(True)
            self.cmb_3d.clear()
            self.cmb_3d.addItems(["Body Color", "Face Color"] + field_keys)
            self.cmb_3d.blockSignals(False)
            
            self.update_frame(self.current_frame)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _on_legend_mode_changed(self, mode: str):
        """컬러 레전드 범위 모드 변경"""
        if mode == "Static":
            self.spin_min.setValue(0.0)
            self.spin_max.setValue(0.01)
        self.update_frame(self.current_frame)

    def _show_part_menu(self, pos=None):
        """[WHTOOLS] 3D 뷰어 커스텀 컨텍스트 메뉴 (우클릭 제어)"""
        if pos is None:
            pos = self.v_int.mapFromGlobal(QtGui.QCursor.pos())
            
        menu = QtWidgets.QMenu(self)
        menu.addAction("Visibility Manager", self.visibility_tool.show)
        menu.addSeparator()
        
        # 뷰 전환 옵션
        view_actions = [
            ("XY Plane (+Z View)", self.v_int.view_xy),
            ("YZ Plane (+X View)", self.v_int.view_yz),
            ("ZX Plane (+Y View)", self.v_int.view_zx),
            ("Isometric View", self.v_int.view_isometric)
        ]
        for name, func in view_actions:
            menu.addAction(name, func)
            
        menu.addSeparator()
        
        # 가시성 토글 옵션
        act_floor = menu.addAction("Floor Visibility")
        act_floor.setCheckable(True)
        act_floor.setChecked(self.ground.GetVisibility())
        
        act_mesh_edge = menu.addAction("Show Mesh Edges")
        act_mesh_edge.setCheckable(True)
        edge_vis = self.part_actors[0]['mesh'].GetProperty().GetEdgeVisibility() if self.part_actors else True
        act_mesh_edge.setChecked(edge_vis)
        
        # 실행
        action = menu.exec_(self.v_int.mapToGlobal(pos))
        
        if action == act_floor:
            self.ground.SetVisibility(action.isChecked())
            self.v_int.render()
        elif action == act_mesh_edge:
            vis = action.isChecked()
            for actor_info in self.part_actors.values():
                actor_info['mesh'].GetProperty().SetEdgeVisibility(vis)
            self.v_int.render()

    def _create_combo(self, label: str, items: list, layout: QtWidgets.QLayout) -> QtWidgets.QComboBox:
        """콤보박스 생성 및 레이아웃 배치 유틸리티"""
        combo = QtWidgets.QComboBox()
        combo.addItems(items)
        combo.currentIndexChanged.connect(lambda: self.update_frame(self.current_frame))
        if layout:
            layout.addWidget(QtWidgets.QLabel(label))
            layout.addWidget(combo)
        return combo

    def _on_axis_clicked(self, event):
        """2D 그래프 슬롯 선택 시 활성화 상태 표시"""
        if event.inaxes:
            for i, ax in enumerate(self.axes):
                if event.inaxes == ax:
                    self.active_slot = i
                    break
            self._update_selection_ui()
            self.statusBar().showMessage(f"Active Plot Slot: {self.active_slot + 1}")

    def _update_selection_ui(self):
        """활성화된 그래프 슬롯의 시각적 강조 (파란색 테두리)"""
        for i, ax in enumerate(self.axes):
            color, width = ("#1A73E8", 2.0) if i == self.active_slot else ("#DADCE0", 0.5)
            for spine in ax.spines.values():
                spine.set_color(color)
                spine.set_linewidth(width)
        self.canvas.draw_idle()

    def _show_add_plot_dialog(self):
        """분석 대상 파트 및 데이터를 선택하여 그래프 슬롯에 배치"""
        p_names = [p.name for p in self.mgr.analyzers]
        dialog = AddPlotDialog(self.active_slot, p_names, self.field_keys, self.stat_keys, self)
        
        if dialog.exec():
            self.plot_slots[self.active_slot] = dialog.get_config()
            # 캐시 초기화하여 재렌더링 유도
            self.ims[self.active_slot] = None
            self.vlines[self.active_slot] = None
            self.update_frame(self.current_frame)

    def _ctrl_slot(self, command: int):
        """[WHTOOLS] 범용 애니메이션 컨트롤러"""
        if command == -2:  # Play / Pause
            if self.is_playing:
                self.timer.stop()
                self.btn_play.setText("▶")
            else:
                self.timer.start(int(self.cmb_speed.currentText()))
                self.btn_play.setText("⏸")
            self.is_playing = not self.is_playing
        elif command == 0:  # First
            self.update_frame(0)
        elif command == 9999:  # Last
            self.update_frame(self.mgr.n_frames - 1)
        else:  # Move by step
            new_f = max(0, min(self.mgr.n_frames - 1, self.current_frame + command * self.anim_step))
            self.update_frame(new_f)

# ==============================================================================
# 데모 데이터 및 실행 섹션
# ==============================================================================

def create_cube_markers(n_samples=100):
    """
    [WHTOOLS] 리팩토링 검증을 위한 데모용 정육면체 마커 클라우드 생성
    6개 면(Face)을 가진 정육면체 구조물을 시뮬레이션 데이터로 생성합니다.
    """
    times = np.linspace(0, 1.5, n_samples)
    L, W, H = 2000.0, 1200.0, 800.0
    
    face_configs = {
        "Bottom": [0, 0, -H/2],
        "Top": [0, 0, H/2],
        "Front": [0, -W/2, 0],
        "Back": [0, W/2, 0],
        "Left": [-L/2, 0, 0],
        "Right": [L/2, 0, 0]
    }
    
    data = {}
    for name, center_pos in face_configs.items():
        # 면의 크기 결정
        fw, fh = (L, W) if "Top" in name or "Bottom" in name else (
                 (L, H) if "Front" in name or "Back" in name else (W, H))
        
        # 마커 레이아웃 (3x3 그리드)
        mx, my = np.meshgrid(np.linspace(-fw * 0.4, fw * 0.4, 3), 
                             np.linspace(-fh * 0.4, fh * 0.4, 3))
        off = np.column_stack([mx.ravel(), my.ravel()])
        markers_hist = np.zeros((n_samples, 9, 3))
        
        for f_idx, t in enumerate(times):
            # 자유 낙하 및 정착 모션 시뮬레이션
            z_settle = 800.0 - 9810.0 * 0.5 * t**2 if t < 0.45 else 100.0 * np.exp(-3 * (t - 0.45))
            # 변형 프로파일 (Sine Wave)
            deform = 12.0 * np.sin(10.0 * t) * np.cos(off[:, 0] / 150.0)
            
            markers_hist[f_idx] = np.column_stack([off, deform]) + center_pos + [0, 0, max(z_settle, 0)]
            
        data[name] = (markers_hist, off, fw, fh)
        
    return data, times

if __name__ == "__main__":
    """[WHTOOLS] 대시보드 독립 실행 블록"""
    print("[WHTOOLS] Starting Structural Analyzer Demo...")
    
    # 데이터 생성
    raw_data, time_arr = create_cube_markers(100)
    manager = PlateAssemblyManager(time_arr)
    
    # 각 면(Face)을 개별 분석기로 등록
    for name, (m_hist, off, fw, fh) in raw_data.items():
        analyzer = ShellDeformationAnalyzer(
            W=fw, H=fh, name=name
        )
        analyzer.m_raw = m_hist 
        analyzer.o_data_hint = off
        manager.add_analyzer(analyzer)
        
    # 일괄 해석 구동 (JAX 가속)
    manager.run_all()
    
    # GUI 실행
    app = QtWidgets.QApplication(sys.argv)
    
    # 프리미엄 테마 적용 (선택 사항)
    app.setPalette(QtGui.QGuiApplication.palette())
    
    gui = QtVisualizerV2(manager)
    gui.show()
    
    print("[WHTOOLS] Launching Dashboard GUI...")
    sys.exit(app.exec())

