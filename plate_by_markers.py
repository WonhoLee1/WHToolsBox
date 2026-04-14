import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
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
import sys
import time
import os
import pickle
from functools import partial
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

# --- 전역 시각화 설정 (WHTOOLS Rules) ---
plt.rcParams['font.size'] = 9
FIGURE_WIDTH = 12
FIGURE_HEIGHT = 8

# --- JAX Configuration (고정밀 연산을 위한 x64 활성화) ---
jax.config.update("jax_enable_x64", True)

@dataclass
class PlateConfiguration:
    """
    유한요소(FEM) 및 기구학 해석을 위한 평판의 물리적 특성 및 수치 해석 설정을 관리하는 데이터 클래스입니다.
    
    Attributes:
        thickness (float): 평판의 두께 [mm]. 기본값 2.0.
        youngs_modulus (float): 영률 (Young's Modulus) [MPa]. 기본값 210,000 MPa (Steel 계열).
        poisson_ratio (float): 포아송 비 (Poisson's Ratio). 기본값 0.3.
        polynomial_degree (int): 표면 피팅을 위한 다항식 차수. 기본값 5.
        regularization_lambda (float): 수치적 안정성을 위한 정규화 계수. 기본값 1e-4.
        mesh_resolution (int): 시각화 및 해석을 위한 그리드 해상도 (N x N). 기본값 25.
        batch_size (int): JAX vmap 연산을 위한 프레임 배치 크기. 기본값 256.
    """
    thickness: float = 2.0
    youngs_modulus: float = 2.1e5
    poisson_ratio: float = 0.3
    polynomial_degree: int = 5
    regularization_lambda: float = 1e-4
    mesh_resolution: int = 25
    batch_size: int = 256

class RigidBodyKinematicsManager:
    """
    마커 데이터로부터 강체의 운동(Rigid Body Motion) 정보를 추출하고 
    로컬 좌표계와 글로벌 좌표계 간의 변환을 관리하는 클래스입니다.
    """
    def __init__(self, marker_history: np.ndarray):
        """
        인자 설명:
            marker_history (np.ndarray): 마커의 위치 이력 데이터. Shape: (프레임 수, 마커 수, 3)
        """
        self.raw_marker_data = jnp.array(marker_history)
        self.n_frames, self.n_markers, _ = self.raw_marker_data.shape
        self._initialize_global_to_local_axes()

    def _initialize_global_to_local_axes(self):
        """
        초기 프레임(Frame 0)을 기준으로 PCA(주성분 분석)를 수행하여 
        평판의 로컬 좌표계(Basis)를 수립합니다.
        """
        initial_markers = self.raw_marker_data[0]
        # 중심점 계산
        initial_centroid = jnp.mean(initial_markers, axis=0)
        centered_markers = initial_markers - initial_centroid
        
        # PCA를 위한 공분산 행렬 계산 및 고유값 분해
        covariance_matrix = np.cov(np.array(centered_markers).T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # 고유값이 큰 순서대로 정렬하여 로컬 축 수립 (Principal Components)
        sort_indices = eigenvalues.argsort()[::-1]
        self.local_basis_axes = jnp.array(eigenvectors[:, sort_indices])
        
        # 로컬 좌표계에서의 마커 위치 및 범위 계산
        markers_local = centered_markers @ self.local_basis_axes
        p_min, p_max = markers_local.min(axis=0), markers_local.max(axis=0)
        
        # [WHTOOLS] 스케일 대응형 여유(Margin) 설정 (전체 크기 대비 약 5%)
        dimension_span = p_max - p_min
        applied_margin = np.max(dimension_span) * 0.05
        
        # 로컬 원점(Centroid) 재설정 및 범위 확정
        self.local_centroid_0 = initial_centroid + ((p_min + p_max) / 2.0) @ self.local_basis_axes.T
        markers_local_corrected = (initial_markers - self.local_centroid_0) @ self.local_basis_axes
        
        self.x_bounds = [
            float(markers_local_corrected[:, 0].min() - applied_margin), 
            float(markers_local_corrected[:, 0].max() + applied_margin)
        ]
        self.y_bounds = [
            float(markers_local_corrected[:, 1].min() - applied_margin), 
            float(markers_local_corrected[:, 1].max() + applied_margin)
        ]

    @partial(jit, static_argnums=(0,))
    def extract_frame_kinematics_vmap(self, frame_markers_batch: jnp.ndarray):
        """
        JAX 기반의 Kabsch 알고리즘을 배치(Batch) 단위로 적용하여 각 프레임의 기구학 정보를 추출합니다.
        
        인자 설명:
            frame_markers_batch (jnp.ndarray): 해석 대상 프레임들의 마커 위치 데이터.
            
        반환값:
            (Tuple): 로컬 좌표, 회전 행렬, 해당 프레임 중심, 초기 준거 중심.
        """
        reference_markers = self.raw_marker_data[0]
        
        def kabsch_algorithm_single_frame(current_frame_markers: jnp.ndarray):
            """Kabsch 알고리즘: 두 점 집합 간의 최적 회전 및 정렬 수행"""
            ref_centroid = jnp.mean(reference_markers, axis=0)
            cur_centroid = jnp.mean(current_frame_markers, axis=0)
            
            # 상호 공분산 행렬 (Cross-covariance matrix)
            cross_covariance = (current_frame_markers - cur_centroid).T @ (reference_markers - ref_centroid)
            U_svd, S_svd, Vh_svd = jnp.linalg.svd(cross_covariance)
            
            # 최적 회전 행렬 계산
            rotation_matrix = Vh_svd.T @ U_svd.T
            # 반사(Reflection) 방지를 위한 보정
            rotation_corrected = jnp.where(
                jnp.linalg.det(rotation_matrix) < 0, 
                (Vh_svd.T @ jnp.diag(jnp.array([1.0, 1.0, -1.0]))) @ U_svd.T, 
                rotation_matrix
            )
            
            # 로컬 좌표계로 마커 투영
            local_displacement = (
                (current_frame_markers - cur_centroid) @ rotation_corrected.T + 
                ref_centroid - self.local_centroid_0
            ) @ self.local_basis_axes
            
            return local_displacement, rotation_corrected, cur_centroid, ref_centroid
            
        return vmap(kabsch_algorithm_single_frame)(frame_markers_batch)

class KirchhoffPlateOptimizer:
    """
    Kirchhoff-Love 평판 이론을 기반으로, 마커의 변형 데이터를 다항식 기저 함수에 피팅하고
    곡률(Curvature) 및 변위장을 최적화하는 클래스입니다.
    """
    def __init__(self, polynomial_degree: int = 5):
        """
        인자 설명:
            polynomial_degree (int): 피팅에 사용할 다항식의 최대 차수.
        """
        # 2차원 다항식 기저 함수의 지수 조합 생성 (i + j <= degree)
        self.basis_indices = [
            (i, j) for i in range(polynomial_degree + 1) for j in range(polynomial_degree + 1 - i)
        ]
        self.n_basis = len(self.basis_indices)

    def compute_basis_matrix(self, x_coords: jnp.ndarray, y_coords: jnp.ndarray) -> jnp.ndarray:
        """이차원 다항식 기저 행렬(Vandermonde-like matrix)을 생성합니다."""
        return jnp.stack([x_coords**i * y_coords**j for i, j in self.basis_indices], axis=1)

    def compute_hessian_basis_matrix(self, x_coords: jnp.ndarray, y_coords: jnp.ndarray) -> jnp.ndarray:
        """
        기저 함수의 2계 도함수(Hessian)를 계산하여 곡률 산정에 활용합니다.
        
        반환값:
            jnp.ndarray: [N_points, N_basis, 3] (d2/dx2, d2/dy2, d2/dxdy 순서)
        """
        def single_point_hessian(xi, yi):
            return jnp.stack([
                jnp.array([
                    i * (i - 1) * xi**(jnp.maximum(0, i - 2)) * yi**j if i >= 2 else 0., 
                    j * (j - 1) * xi**i * yi**(jnp.maximum(0, j - 2)) if j >= 2 else 0., 
                    i * j * xi**(jnp.maximum(0, i - 1)) * yi**(jnp.maximum(0, j - 1)) if i >= 1 and j >= 1 else 0.
                ]) for i, j in self.basis_indices
            ], axis=0)
        return vmap(single_point_hessian)(x_coords, y_coords)

    @partial(jit, static_argnums=(0,))
    def solve_fitting_coefficients(
        self, 
        local_displacements: jnp.ndarray, 
        reference_markers_local: jnp.ndarray, 
        reg_lambda: float
    ) -> jnp.ndarray:
        """
        마커의 Z방향 변위(Local W)와 곡률 에너지를 동시에 고려하여 최적의 다항식 계수를 산출합니다.
        
        인자 설명:
            local_displacements (jnp.ndarray): 프레임별 로컬 마커 좌표 (N_frames, N_markers, 3).
            reference_markers_local (jnp.ndarray): 기준 프레임의 로컬 마커 좌표 (N_markers, 3).
            reg_lambda (float): 곡률 정규화 계수 (Bending stiffness와 유사한 역할).
        """
        # 수치적 안정성을 위해 정규화된 좌표계 사용 (Scale-invariant fitting)
        x_scale = jnp.max(jnp.abs(reference_markers_local[:, 0])) + 1e-9
        y_scale = jnp.max(jnp.abs(reference_markers_local[:, 1])) + 1e-9
        x_norm, y_norm = reference_markers_local[:, 0] / x_scale, reference_markers_local[:, 1] / y_scale
        
        # 기저 행렬 및 Hessian Basis 행렬 생성
        basis_matrix = self.compute_basis_matrix(x_norm, y_norm)
        hessian_basis = self.compute_hessian_basis_matrix(x_norm, y_norm)
        
        # Hessian 보정 (스케일 팩터 재적용: Chain Rule)
        # d2w/dx2 = (1/sx^2) * d2w_norm/dx_norm^2
        basis_xx = hessian_basis[:, :, 0] / (x_scale**2)
        basis_yy = hessian_basis[:, :, 1] / (y_scale**2)
        basis_xy = hessian_basis[:, :, 2] / (x_scale * y_scale)
        
        # 시스템 행렬 M 구축: Least-square term + Bending Energy Regularization term
        # (X.T @ X) / N + Lambda * (Bxx.T @ Bxx + Byy.T @ Byy + 2*Bxy.T @ Bxy) / N
        n_points = basis_matrix.shape[0]
        stiffness_matrix = (basis_matrix.T @ basis_matrix) / n_points
        regularization_term = reg_lambda * (
            basis_xx.T @ basis_xx + basis_yy.T @ basis_yy + 2.0 * basis_xy.T @ basis_xy
        ) / n_points
        
        system_matrix = stiffness_matrix + regularization_term + jnp.eye(self.n_basis) * 1e-12
        
        # 프레임별 변위(Z) 데이터 추출
        z_displacements = local_displacements[:, :, 2]
        
        # 배치 솔버: 계수 p_coeffs 계산
        def solve_single(z):
            return solve(system_matrix, (basis_matrix.T @ z) / n_points)
            
        return vmap(solve_single)(z_displacements)


class PlateMechanicsSolver:
    """
    다항식 계수와 물리적 파라미터를 결합하여 평판의 응력, 변형률 등 역학적 필드 데이터를 생성합니다.
    """
    def __init__(self, configuration: PlateConfiguration):
        self.config = configuration
        # 굽힘 강성 (Bending Stiffness, D = E*h^3 / (12*(1-nu^2)))
        self.flexural_rigidity = (
            configuration.youngs_modulus * configuration.thickness**3
        ) / (12.0 * (1.0 - configuration.poisson_ratio**2))
        
        self.resolution = configuration.mesh_resolution
        self.optimizer = KirchhoffPlateOptimizer(polynomial_degree=configuration.polynomial_degree)

    def initialize_mesh_grid(self, x_bounds: List[float], y_bounds: List[float]):
        """해석 및 시각화를 위한 그리드 메쉬를 설정합니다."""
        self.x_lin = jnp.linspace(x_bounds[0], x_bounds[1], self.resolution)
        self.y_lin = jnp.linspace(y_bounds[0], y_bounds[1], self.resolution)
        self.X_mesh, self.Y_mesh = jnp.meshgrid(self.x_lin, self.y_lin)
        
        # 스케일 팩터 저장 (다항식 평가용)
        self.x_scale_factor = jnp.max(jnp.abs(jnp.array(x_bounds))) + 1e-9
        self.y_scale_factor = jnp.max(jnp.abs(jnp.array(y_bounds))) + 1e-9

    @partial(jit, static_argnums=(0,))
    def compute_mechanics_fields_batch(self, coefficients_batch: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        배치 단위의 다항식 계수로부터 모든 물리 필드(변위, 응력, 변형률)를 계산합니다.
        
        인자 설명:
            coefficients_batch (jnp.ndarray): 최적화된 다항식 계수들.
        """
        # 정규화 좌표 메쉬
        X_norm = self.X_mesh.ravel() / self.x_scale_factor
        Y_norm = self.Y_mesh.ravel() / self.y_scale_factor
        
        # 평가를 위한 기저 행렬 사전 계산
        basis_eval = self.optimizer.compute_basis_matrix(X_norm, Y_norm)
        hessian_basis_eval = self.optimizer.compute_hessian_basis_matrix(X_norm, Y_norm)
        
        @vmap
        def evaluate_single_frame(p):
            # 1. 변위 필드 산출
            displacement_w = (basis_eval @ p).reshape(self.resolution, self.resolution)
            
            # 2. 곡률(Curvature) 계산 (Chain Rule 적용)
            # kxx = -d2w/dx2, kyy = -d2w/dy2, kxy = -d2w/dxdy
            k_xx = -jnp.dot(hessian_basis_eval[:, :, 0], p).reshape(self.resolution, self.resolution) / (self.x_scale_factor**2)
            k_yy = -jnp.dot(hessian_basis_eval[:, :, 1], p).reshape(self.resolution, self.resolution) / (self.y_scale_factor**2)
            k_xy = -jnp.dot(hessian_basis_eval[:, :, 2], p).reshape(self.resolution, self.resolution) / (self.x_scale_factor * self.y_scale_factor)
            
            # 3. 응력(Stress) 계산 (Kirchhoff-Love Theory)
            # sigma_x = (E*z / (1-nu^2)) * (kxx + nu*kyy)
            # 표면(z = h/2)에서의 응력 계산 계수
            stress_coeff = 6.0 * self.flexural_rigidity / (self.config.thickness**2)
            
            sigma_x = stress_coeff * (k_xx + self.config.poisson_ratio * k_yy)
            sigma_y = stress_coeff * (k_yy + self.config.poisson_ratio * k_xx)
            tau_xy = stress_coeff * (1.0 - self.config.poisson_ratio) * k_xy
            
            # Von-Mises 응력 (평면 응력 상태 가정)
            von_mises = jnp.sqrt(jnp.maximum(sigma_x**2 + sigma_y**2 - sigma_x * sigma_y + 3.0 * tau_xy**2, 1e-12))
            signed_von_mises = von_mises * jnp.sign(sigma_x + sigma_y)
            
            # 주응력(Principal Stresses)
            stress_avg = (sigma_x + sigma_y) / 2.0
            stress_diff_radius = jnp.sqrt(jnp.maximum(((sigma_x - sigma_y) / 2.0)**2 + tau_xy**2, 1e-12))
            principal_max = stress_avg + stress_diff_radius
            principal_min = stress_avg - stress_diff_radius
            
            # 4. 변형률(Strain) 및 등가 변형률
            strain_x = (self.config.thickness / 2.0) * k_xx
            strain_y = (self.config.thickness / 2.0) * k_yy
            gamma_xy = self.config.thickness * k_xy
            
            # 주 변형률 산출을 위한 평균 및 반지름 계산
            strain_avg = (strain_x + strain_y) / 2.0
            strain_radius = jnp.sqrt(jnp.maximum(((strain_x - strain_y) / 2.0)**2 + (gamma_xy / 2.0)**2, 1e-20))
            
            equiv_strain = (2.0 / 3.0) * jnp.sqrt(jnp.maximum(1.5 * (strain_x**2 + strain_y**2) + 0.75 * gamma_xy**2, 1e-20))
            
            fields = {
                'Displacement [mm]': displacement_w, 
                'Curvature XX [1/mm]': k_xx, 
                'Mean Curvature [1/mm]': 0.5 * (k_xx + k_yy),
                'Stress XX [MPa]': sigma_x, 
                'Stress YY [MPa]': sigma_y,
                'Stress XY [MPa]': tau_xy,
                'Von-Mises [MPa]': von_mises,
                'Signed Von-Mises [MPa]': signed_von_mises, 
                'Principal Max [MPa]': principal_max,
                'Principal Min [MPa]': principal_min,
                'Strain XX [mm/mm]': strain_x,
                'Strain YY [mm/mm]': strain_y,
                'Strain XY [mm/mm]': gamma_xy,
                'Strain Max Principal [mm/mm]': strain_avg + strain_radius,
                'Strain Min Principal [mm/mm]': strain_avg - strain_radius,
                'Eq. Strain [mm/mm]': equiv_strain
            }
            # 통계 데이터 자동 생성
            statistics = {f'Mean-{key}': jnp.mean(val) for key, val in fields.items()}
            statistics.update({f'Max-{key}': jnp.max(val) for key, val in fields.items()})
            
            return {**fields, **statistics}
            
        return evaluate_single_frame(coefficients_batch)

class ShellDeformationAnalyzer:
    """
    마커 데이터 처리, 기구학 분리, 역학 해석 및 시각화를 총괄하는 메인 오케스트레이터 클래스입니다.
    """
    def __init__(self, markers: np.ndarray, configuration: PlateConfiguration, timestamps: Optional[np.ndarray] = None):
        """
        인자 설명:
            markers (np.ndarray): 마커 위치 이력 (N_frames, N_markers, 3).
            configuration (PlateConfiguration): 해석 설정 객체.
            timestamps (np.ndarray, optional): 프레임별 시간 정보.
        """
        self.raw_markers_mm = np.array(markers)
        self.config = configuration
        self.timestamps = timestamps if timestamps is not None else np.arange(len(markers))
        
        # 하위 모듈 인스턴스화
        self.kinematics_manager = RigidBodyKinematicsManager(markers)
        self.mechanics_solver = PlateMechanicsSolver(configuration)
        
        # 해석용 그리드 메쉬 생성
        self.mechanics_solver.initialize_mesh_grid(
            self.kinematics_manager.x_bounds, 
            self.kinematics_manager.y_bounds
        )
        self.analysis_results = {}

    def run_analysis(self):
        """전체 프레임에 대해 기구학 및 역학 해석 파이프라인을 실행합니다."""
        print(f"[WHTOOLS] 분석 시작 (총 {len(self.timestamps)} 프레임)...")
        start_time = time.time()
        n_frames = len(self.timestamps)
        
        # 1. 기구학(Kinematics) 추출: 로컬 변위, 회전 행렬, 중심 이동량 계산
        (
            local_displacement_jax, 
            rotation_matrices_jax, 
            current_centroids_jax, 
            reference_centroids_jax
        ) = self.kinematics_manager.extract_frame_kinematics_vmap(self.raw_markers_mm)
        
        # 2. 다항식 피팅 계수 산출 (Frame 0의 로컬 마커를 준거로 활용)
        reference_markers_local = local_displacement_jax[0]
        optimal_fitting_coefficients = self.mechanics_solver.optimizer.solve_fitting_coefficients(
            local_displacement_jax, 
            reference_markers_local, 
            self.config.regularization_lambda
        )
        
        # 3. 배치 단위로 물리 필드(응력, 변위 등) 계산
        batch_size = self.config.batch_size
        results_accumulator = None
        
        for i in range(0, n_frames, batch_size):
            current_batch_coeffs = optimal_fitting_coefficients[i : i + batch_size]
            batch_fields = self.mechanics_solver.compute_mechanics_fields_batch(current_batch_coeffs)
            
            # 결과 저장을 위한 버퍼 초기화
            if results_accumulator is None:
                field_keys = list(batch_fields.keys())
                essential_keys = ['R_matrix', 'cur_centroid', 'ref_centroid', 'local_markers']
                results_accumulator = {key: [] for key in field_keys + essential_keys}
            
            # 필드 데이터 누적
            for key, val in batch_fields.items():
                results_accumulator[key].append(np.array(val))
            
            # 기구학 데이터 누적
            results_accumulator['R_matrix'].append(np.array(rotation_matrices_jax[i : i + batch_size]))
            results_accumulator['cur_centroid'].append(np.array(current_centroids_jax[i : i + batch_size]))
            results_accumulator['ref_centroid'].append(np.array(reference_centroids_jax[i : i + batch_size]))
            results_accumulator['local_markers'].append(np.array(local_displacement_jax[i : i + batch_size]))

        # 리스트 형태의 조각들을 하나의 NumPy 배열로 병합
        self.analysis_results = {
            key: np.concatenate(val, axis=0) for key, val in results_accumulator.items()
        }
        
        # 추가 메트릭: 글로벌 마커 Z 및 로컬 변위 W 저장
        self.analysis_results['Marker Global Z [mm]'] = self.raw_markers_mm[:, :, 2]
        self.analysis_results['Marker Local W [mm]'] = self.analysis_results['local_markers'][:, :, 2]
        
        print(f"[WHTOOLS] 분석 완료: {time.time() - start_time:.4f}초 소요")

    def show_visualization(self, ground_plane_size: tuple = (2000.0, 2000.0), marker_label_names: list = None):
        """결과 데이터 시각화를 위한 GUI를 실행합니다."""
        qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        visualizer_gui = QtDeformationVisualizer(
            self, 
            ground_size=ground_plane_size, 
            marker_names=marker_label_names
        )
        visualizer_gui.show()
        sys.exit(qt_app.exec())

class QtDeformationVisualizer(QtWidgets.QMainWindow):
    """
    해석된 평판 변형 데이터를 3D(PyVista) 및 2D(Matplotlib)로 시각화하는 메인 GUI 클래스입니다.
    """
    def __init__(self, analyzer: ShellDeformationAnalyzer, ground_size: tuple = (2000.0, 2000.0), marker_names: list = None):
        super().__init__()
        self.analyzer = analyzer
        self.ground_size = ground_size
        self.kinematics = analyzer.kinematics_manager
        self.solver = analyzer.mechanics_solver
        self.results = analyzer.analysis_results
        
        # 마커 이름 설정
        self.marker_names = marker_names if marker_names else [
            f'M{i+1:02d}' for i in range(self.kinematics.n_markers)
        ]
        
        # 가용 필드 및 통계 키 추출
        self.field_keys = [k for k in self.results if self.results[k].ndim == 3]
        self.statistics_keys = [k for k in self.results if self.results[k].ndim == 1]
        self.statistics_keys += ['Marker Local Disp. [mm]', 'Marker Global Disp. [mm]']
        
        # 상태 변수
        self.current_frame = 0
        self.is_playing = False
        self.plot_objects = {}
        self.color_cycle = plt.cm.tab10.colors
        
        # UI 구성 호출
        self._initialize_main_ui_layout()
        self._initialize_3d_viewer()
        self._initialize_2d_trend_plots()
        self._apply_optimal_defaults()
        
        # 초기 프레임 업데이트
        self.update_visual_frame(0)

    def _initialize_main_ui_layout(self):
        """기본적인 창 구조와 컨트롤 위젯들을 배치합니다."""
        self.setWindowTitle("WHTOOLS: Advanced Shell Deformation Analyzer Pro++")
        self.resize(1800, 1000)
        
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        
        # --- 상부 제어판(Header) ---
        header_layout = QtWidgets.QHBoxLayout()
        
        # 로고 표시
        logo_label = QtWidgets.QLabel()
        logo_path = r"C:\Users\GOODMAN\WHToolsBox\TVPackageMotionSim\sidebar_logo.png"
        if os.path.exists(logo_path):
            logo_label.setPixmap(QtGui.QPixmap(logo_path).scaledToHeight(120))
        header_layout.addWidget(logo_label)
        
        # 3D 뷰 설정 그룹
        view_3d_group = QtWidgets.QGroupBox("3D 시각화 제어 (View Control)")
        view_3d_layout = QtWidgets.QVBoxLayout(view_3d_group)
        
        row1 = QtWidgets.QHBoxLayout()
        self.combo_view_mode = self._create_labeled_combobox("시점(View):", ["Global", "Local"], row1)
        self.combo_legend_mode = self._create_labeled_combobox("범례(Legend):", ["Dynamic", "Static"], row1)
        self.combo_3d_field = self._create_labeled_combobox("물리량(Field):", self.field_keys, row1)
        row1.addStretch()
        
        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(QtWidgets.QLabel("배율(Deform Scale):"))
        self.spin_deform_scale = QtWidgets.QDoubleSpinBox()
        self.spin_deform_scale.setRange(0.1, 1000.0); self.spin_deform_scale.setValue(1.0)
        row2.addWidget(self.spin_deform_scale)
        
        row2.addWidget(QtWidgets.QLabel("범위-최소:"))
        self.spin_clim_min = QtWidgets.QDoubleSpinBox()
        row2.addWidget(self.spin_clim_min)
        
        row2.addWidget(QtWidgets.QLabel("범위-최대:"))
        self.spin_clim_max = QtWidgets.QDoubleSpinBox()
        row2.addWidget(self.spin_clim_max)
        
        for widget in [self.spin_deform_scale, self.spin_clim_min, self.spin_clim_max]:
            widget.setRange(-1e12, 1e12); widget.setDecimals(4)
            widget.valueChanged.connect(lambda: self.update_visual_frame(self.current_frame))
        
        row2.addStretch()
        view_3d_layout.addLayout(row1); view_3d_layout.addLayout(row2)
        header_layout.addWidget(view_3d_group)
        
        # 2D 플롯 설정 그룹
        plot_2d_group = QtWidgets.QGroupBox("2D 트렌드 분석 제어 (Plot Control)")
        plot_2d_layout = QtWidgets.QVBoxLayout(plot_2d_group)
        
        row21 = QtWidgets.QHBoxLayout()
        self.combo_field_1 = self._create_labeled_combobox("이미지-1:", self.field_keys, row21)
        self.combo_field_2 = self._create_labeled_combobox("이미지-2:", self.field_keys, row21)
        row21.addStretch()
        
        row22 = QtWidgets.QHBoxLayout()
        self.combo_chart_1 = self._create_labeled_combobox("차트-1:", self.statistics_keys, row22)
        self.combo_chart_2 = self._create_labeled_combobox("차트-2:", self.statistics_keys, row22)
        row22.addStretch()
        
        plot_2d_layout.addLayout(row21); plot_2d_layout.addLayout(row22)
        header_layout.addWidget(plot_2d_group)
        
        main_layout.addLayout(header_layout)
        
        # --- 중앙 영역: 3D + 2D 분할창 ---
        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        
        # PyVista 3D Viewer
        self.v_interactor = QtInteractor(self)
        main_splitter.addWidget(self.v_interactor)
        
        # Matplotlib 2D Plots
        plot_container = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_container)
        self.canvas = FigureCanvas(Figure(figsize=(10, 10)))
        plot_layout.addWidget(NavigationToolbar(self.canvas, self))
        plot_layout.addWidget(self.canvas)
        main_splitter.addWidget(plot_container)
        
        main_splitter.setStretchFactor(0, 7)
        main_layout.addWidget(main_splitter, stretch=8)
        
        # --- 하부 재생 제어판 ---
        playback_layout = QtWidgets.QHBoxLayout()
        for text, step in [("▶", -2), ("◀", -1), ("▶", 1)]:
            btn = QtWidgets.QPushButton(text)
            btn.setFixedWidth(40)
            btn.clicked.connect(partial(self._handle_playback_action, step))
            playback_layout.addWidget(btn)
        
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setRange(0, self.analyzer.kinematics_manager.n_frames - 1)
        self.frame_slider.valueChanged.connect(self.update_visual_frame)
        playback_layout.addWidget(self.frame_slider)
        
        self.label_frame_info = QtWidgets.QLabel("Frame: 0")
        playback_layout.addWidget(self.label_frame_info)
        main_layout.addLayout(playback_layout)
        
        # 재생 타이머 설정
        self.playback_timer = QtCore.QTimer()
        self.playback_timer.timeout.connect(
            lambda: self.update_visual_frame(
                self.current_frame + 1 if self.current_frame < self.analyzer.kinematics_manager.n_frames - 1 else 0
            )
        )

    def _create_labeled_combobox(self, label_text: str, items: list, parent_layout: QtWidgets.QLayout) -> QtWidgets.QComboBox:
        """레이블과 콤보박스 세트를 생성하여 레이아웃에 추가합니다."""
        parent_layout.addWidget(QtWidgets.QLabel(label_text))
        combo = QtWidgets.QComboBox()
        combo.addItems(items)
        combo.currentIndexChanged.connect(lambda: self.update_visual_frame(self.current_frame))
        parent_layout.addWidget(combo)
        return combo

    def _apply_optimal_defaults(self):
        """초기 추천 물리량 필드를 설정합니다."""
        self.combo_3d_field.setCurrentText("Mean Curvature [1/mm]")
        self.combo_field_1.setCurrentText("Displacement [mm]")
        self.combo_field_2.setCurrentText("Stress XX [MPa]")
        self.combo_chart_1.setCurrentText("Max-Displacement [mm]")
        self.combo_chart_2.setCurrentText("Marker Local Disp. [mm]")

    def _initialize_3d_viewer(self):
        """PyVista 3D 씬을 설정합니다."""
        self.v_interactor.set_background("white")
        
        # 바닥 Plane 생성
        self.ground_mesh = self.v_interactor.add_mesh(
            pv.Plane(i_size=self.ground_size[0], j_size=self.ground_size[1]), 
            color="blue", opacity=0.3
        )
        
        # 평판 해석 메쉬 초기화
        nx, ny = self.solver.X_mesh.shape
        self.plate_base_points = np.column_stack([
            self.solver.X_mesh.ravel(), 
            self.solver.Y_mesh.ravel(), 
            np.zeros(nx * ny)
        ])
        
        # 시각화용 PyVista Plane 객체 수립
        self.plate_poly = pv.Plane(
            i_size=float(self.solver.x_lin.max() - self.solver.x_lin.min()), 
            j_size=float(self.solver.y_lin.max() - self.solver.y_lin.min()), 
            i_resolution=nx - 1, 
            j_resolution=ny - 1
        )
        self.plate_poly.point_data["Scalars"] = np.zeros(nx * ny)
        self.plate_actor = self.v_interactor.add_mesh(
            self.plate_poly, scalars="Scalars", cmap="turbo", 
            show_edges=True, edge_color="lightgrey"
        )
        
        # 마커 데이터 시각화
        self.marker_poly = pv.PolyData(np.array(self.analyzer.raw_markers_mm[0]))
        self.marker_poly.point_data["names"] = self.marker_names
        self.v_interactor.add_mesh(
            self.marker_poly, render_points_as_spheres=True, 
            point_size=12, color='darkblue'
        )
        self.v_interactor.add_point_labels(
            self.marker_poly, "names", font_size=9, 
            text_color='black', always_visible=True
        )
        self.v_interactor.view_isometric()

    def _initialize_2d_trend_plots(self):
        """Matplotlib 기반의 2D 이미징 및 차트 영역을 설정합니다."""
        fig = self.canvas.figure
        self.axes_list = [fig.add_subplot(2, 2, i + 1) for i in range(4)]
        
        # 이미지 플롯 핸들 및 트렌드 라인 핸들 저장용
        self.image_handles = [None, None]
        self.trend_lines = [[None] * 12 for _ in range(2)]  # 최대 12개 마커까지 지원
        self.trend_dots = [[None] * 12 for _ in range(2)]
        self.vertical_time_markers = [None, None]
        
        for i in range(2):
            self.image_handles[i] = self.axes_list[i].imshow(
                np.zeros_like(self.solver.X_mesh), 
                extent=[self.kinematics.x_bounds[0], self.kinematics.x_bounds[1], 
                        self.kinematics.y_bounds[0], self.kinematics.y_bounds[1]], 
                cmap='turbo'
            )
            self.axes_list[i + 2].grid(True, alpha=0.3)
            self.vertical_time_markers[i] = self.axes_list[i + 2].axvline(0, color='red', ls='--')

    def _handle_playback_action(self, step: int):
        """재생/일시정지 및 프레임 이동 로직을 처리합니다."""
        if step == -2:  # Play/Pause toggle
            if self.is_playing:
                self.playback_timer.stop()
            else:
                self.playback_timer.start(33)  # 약 30 FPS
            self.is_playing = not self.is_playing
        else:
            new_frame = max(0, min(self.analyzer.kinematics_manager.n_frames - 1, self.current_frame + step))
            self.update_visual_frame(new_frame)

    def update_visual_frame(self, frame_idx: int):
        """지정된 프레임으로 3D 및 2D 시각화 요소를 업데이트합니다."""
        self.current_frame = frame_idx
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(frame_idx)
        self.frame_slider.blockSignals(False)
        
        current_time = self.analyzer.timestamps[frame_idx]
        self.label_frame_info.setText(f"Frame: {frame_idx} | Time: {current_time:.3f}s")
        
        # UI 설정값 획득
        view_mode = self.combo_view_mode.currentText()
        field_key = self.combo_3d_field.currentText()
        deform_scale = self.spin_deform_scale.value()
        
        # 데이터 획득
        field_values = self.results[field_key][frame_idx]
        displacement_w = self.results['Displacement [mm]'][frame_idx]
        
        # 3D 위치 계산
        deformed_points = self.plate_base_points.copy()
        deformed_points[:, 2] = displacement_w.ravel() * deform_scale
        
        if view_mode == "Global":
            self.ground_mesh.SetVisibility(True)
            rotation = self.results['R_matrix'][frame_idx]
            cur_centroid = self.results['cur_centroid'][frame_idx]
            ref_centroid = self.results['ref_centroid'][frame_idx]
            # 로컬 -> 글로벌 변환 (Inverse Kinematics)
            final_points = (
                deformed_points @ np.array(self.kinematics.local_basis_axes).T + 
                np.array(self.kinematics.local_centroid_0) - ref_centroid
            ) @ rotation + cur_centroid
            current_marker_positions = self.analyzer.raw_markers_mm[frame_idx]
        else:
            self.ground_mesh.SetVisibility(False)
            final_points = deformed_points
            current_marker_positions = self.results['local_markers'][frame_idx]
            
        # PyVista 데이터 갱신
        self.plate_poly.points = np.array(final_points)
        self.plate_poly.point_data["Scalars"] = field_values.ravel()
        self.marker_poly.points = np.array(current_marker_positions)
        
        # Scalar Range (Color bar) 설정
        if self.combo_legend_mode.currentText() == "Dynamic":
            clim = [float(field_values.min()), float(field_values.max())]
            self.spin_clim_min.setValue(clim[0])
            self.spin_clim_max.setValue(clim[1])
        else:
            clim = [self.spin_clim_min.value(), self.spin_clim_max.value()]
            
        self.plate_actor.mapper.scalar_range = clim
        self.v_interactor.render()
        
        # 2D 플롯 업데이트 호출
        self.update_visual_plots(frame_idx, current_time)

    def update_visual_plots(self, frame_idx: int, current_time: float):
        """2D Matplotlib 차트들을 업데이트합니다."""
        # 1. 이미지 플롯 업데이트
        for i, combo in enumerate([self.combo_field_1, self.combo_field_2]):
            field_name = combo.currentText()
            data = self.results[field_name][frame_idx]
            self.image_handles[i].set_data(data)
            self.image_handles[i].set_clim(data.min(), data.max())
            self.axes_list[i].set_title(field_name)
            
        # 2. 트렌드 차트 업데이트
        for i, combo in enumerate([self.combo_chart_1, self.combo_chart_2]):
            stat_key = combo.currentText()
            ax = self.axes_list[i + 2]
            
            if stat_key in ['Marker Local Disp. [mm]', 'Marker Global Disp. [mm]']:
                data_key = 'Marker Global Z [mm]' if stat_key == 'Marker Global Disp. [mm]' else 'Marker Local W [mm]'
                trend_data = self.results[data_key]
                for j in range(min(12, trend_data.shape[1])):
                    if self.trend_lines[i][j] is None:
                        self.trend_lines[i][j], = ax.plot(
                            self.analyzer.timestamps, trend_data[:, j], 
                            alpha=0.7, lw=1, color=self.color_cycle[j % 10]
                        )
                        self.trend_dots[i][j], = ax.plot(
                            [current_time], [trend_data[frame_idx, j]], 
                            'o', color=self.color_cycle[j % 10], ms=5
                        )
                    else:
                        self.trend_lines[i][j].set_ydata(trend_data[:, j])
                        self.trend_dots[i][j].set_data([current_time], [trend_data[frame_idx, j]])
            else:
                y_data = self.results[stat_key]
                if self.trend_lines[i][0] is None:
                    self.trend_lines[i][0], = ax.plot(self.analyzer.timestamps, y_data, color='blue', lw=1.5)
                    self.trend_dots[i][0], = ax.plot([current_time], [y_data[frame_idx]], 'ro', ms=6)
                else: 
                    self.trend_lines[i][0].set_ydata(y_data)
                    self.trend_dots[i][0].set_data([current_time], [y_data[frame_idx]])
            
            ax.set_title(stat_key)
            ax.relim()
            ax.autoscale_view()
            self.vertical_time_markers[i].set_xdata([current_time, current_time])
            
        self.canvas.draw_idle()


def run_analysis_from_simulation_result(pickle_result_path: str, component_name: str = "opencell_front"):
    """
    MuJoCo 시뮬레이션 결과(Pickle)로부터 특정 파트의 마커를 추출하여 해석을 수행합니다.
    """
    if not os.path.exists(pickle_result_path):
        print(f"[ERROR] 결과 파일을 찾을 수 없습니다: {pickle_result_path}")
        return
        
    simulation_dir = os.path.dirname(os.path.abspath(pickle_result_path))
    parent_path = os.path.dirname(simulation_dir)
    if parent_path not in sys.path:
        sys.path.append(parent_path)
        
    try:
        from run_drop_simulator.whts_mapping import extract_face_markers
    except ImportError:
        print("[ERROR] 'run_drop_simulator' 모듈 부재로 마커 추출을 건너뜁니다.")
        return

    with open(pickle_result_path, "rb") as f:
        simulation_data = pickle.load(f)
        
    # 파트명 분석 (예: opencell_front -> base=opencell, face=Front)
    if "_" in component_name:
        comp_base, face_name = component_name.split("_", 1)
        face_name = face_name.capitalize()
    else:
        comp_base, face_name = component_name, "Front"
        
    all_face_markers_dict, _ = extract_face_markers(simulation_data, comp_base, use_virtual_markers=False)
    
    if face_name not in all_face_markers_dict or not all_face_markers_dict[face_name]:
        print(f"[ERROR] 파트 {comp_base}에서 {face_name} 페이스 마커를 찾을 수 없습니다.")
        return

    markers_dict = all_face_markers_dict[face_name]
    marker_label_names = sorted(list(markers_dict.keys()))
    
    # [WHTOOLS] 단위 변환: MuJoCo(m) -> Analyzer(mm)
    markers_raw_array = np.stack([markers_dict[m] for m in marker_label_names], axis=1)
    markers_mm = markers_raw_array * 1000.0
    
    time_history = np.array(simulation_data.time_history)
    
    # 물리적 스케일 검증
    bbox_min = np.min(markers_mm[0], axis=0)
    bbox_max = np.max(markers_mm[0], axis=0)
    dimensions = bbox_max - bbox_min
    print(f" > [CHECK] 물리적 크기 확인: {dimensions[0]:.1f} x {dimensions[1]:.1f} x {dimensions[2]:.1f} (mm)")
    
    # 해석 설정 수립
    thickness_val = simulation_data.config.get(f"{comp_base.lower()}_thickness", 5.0)
    analysis_config = PlateConfiguration(
        thickness=thickness_val, 
        youngs_modulus=2.1e5, 
        polynomial_degree=4
    )
    
    analyzer = ShellDeformationAnalyzer(markers_mm, analysis_config, timestamps=time_history)
    analyzer.run_analysis()
    analyzer.show_visualization(ground_plane_size=(2500, 2500), marker_label_names=marker_label_names)


def create_synthetic_example_markers(n_frames: int = 100):
    """테스트를 위한 가상의 변형 마커 데이터를 생성합니다."""
    time_array = np.linspace(0, 2.0, n_frames)
    mx, my = np.meshgrid(np.linspace(-900.0, 900.0, 3), np.linspace(-600.0, 600.0, 3))
    local_marker_positions = np.column_stack([mx.ravel(), my.ravel(), np.zeros(9)])
    
    synthetic_data = np.zeros((n_frames, 9, 3))
    z_initial, t_hit, gravity = 500.0, 0.32, 9810.0
    
    for f, t in enumerate(time_array):
        # 자유 낙하 및 리바운드 모사
        z_height = max(0.0, z_initial - 0.5 * gravity * t**2 if t < t_hit else 50.0 * np.exp(-3.0 * (t - t_hit)) * np.abs(np.sin(10.0 * (t - t_hit))))
        tilt_angle = np.deg2rad(15.0 if t < t_hit else 15.0 * np.exp(-10.0 * (t - t_hit)))
        
        # 회전 행렬 (Y축 기준)
        rot_y = np.array([
            [np.cos(tilt_angle), 0, np.sin(tilt_angle)], 
            [0, 1, 0], 
            [-np.sin(tilt_angle), 0, np.cos(tilt_angle)]
        ])
        
        # 중앙 중심의 물결 무늬 변형 모사
        ripple_deformation = (
            80.0 * np.exp(-4.0 * max(0, t - t_hit)) * 
            np.sin(25.0 * max(0, t - t_hit)) * 
            (mx.ravel() / 900.0 * my.ravel() / 600.0)
        )
        
        synthetic_data[f] = local_marker_positions @ rot_y.T + [0, 0, z_height]
        synthetic_data[f, :, 2] += ripple_deformation
        
    return synthetic_data, time_array


if __name__ == "__main__":
    # 시뮬레이션 결과 파일 로드 시도
    TARGET_RESULT_PKL = r"C:\Users\GOODMAN\WHToolsBox\TVPackageMotionSim\rds-20260414_044556\simulation_result.pkl"
    TARGET_COMPONENT = "Opencell_Rear"
    
    if os.path.exists(TARGET_RESULT_PKL):
        run_analysis_from_simulation_result(TARGET_RESULT_PKL, component_name=TARGET_COMPONENT)
    else:
        # 파일이 없을 경우 가상 데이터로 실행
        print("[INFO] 시뮬레이션 결과 파일을 찾을 수 없어 가상 데이터를 생성하여 실행합니다.")
        markers_data, timestamps_data = create_synthetic_example_markers(n_frames=100)
        
        test_config = PlateConfiguration(
            thickness=5.0, 
            youngs_modulus=100.0,  # 가독성을 위해 낮은 강성 설정
            mesh_resolution=25
        )
        
        test_analyzer = ShellDeformationAnalyzer(markers_data, test_config, timestamps=timestamps_data)
        test_analyzer.run_analysis()
        test_analyzer.show_visualization(ground_plane_size=(3000.0, 3000.0))