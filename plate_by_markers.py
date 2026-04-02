import numpy as np
import matplotlib.pyplot as plt
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
from functools import partial
from dataclasses import dataclass

# --- JAX Configuration (High Precision) ---
jax.config.update("jax_enable_x64", True)

@dataclass
class PlateConfig:
    thickness: float = 2.0         # [mm]
    youngs_modulus: float = 2.1e5  # [MPa] (210,000 MPa = 210 GPa)
    poisson_ratio: float = 0.3
    poly_degree: int = 5
    reg_lambda: float = 1e-4
    mesh_resolution: int = 25
    batch_size: int = 256

class KinematicsManager:
    def __init__(self, marker_history):
        self.raw_data = jnp.array(marker_history)
        self.n_frames, self.n_markers, _ = self.raw_data.shape
        self._setup_global_to_local()

    def _setup_global_to_local(self):
        valid_P0 = self.raw_data[0]
        temp_centroid = jnp.mean(valid_P0, axis=0)
        P0_centered = valid_P0 - temp_centroid
        cov = np.cov(np.array(P0_centered).T)
        evals, evecs = np.linalg.eigh(cov)
        idx = evals.argsort()[::-1]
        self.local_axes = jnp.array(evecs[:, idx])
        p_local = P0_centered @ self.local_axes
        p_min, p_max = p_local.min(axis=0), p_local.max(axis=0)
        margin = 0.05
        self.centroid_0 = temp_centroid + ((p_min + p_max) / 2.0) @ self.local_axes.T
        p_local_corr = (valid_P0 - self.centroid_0) @ self.local_axes
        self.x_bounds = [float(p_local_corr[:, 0].min() - margin), float(p_local_corr[:, 0].max() + margin)]
        self.y_bounds = [float(p_local_corr[:, 1].min() - margin), float(p_local_corr[:, 1].max() + margin)]

    @partial(jit, static_argnums=(0,))
    def extract_kinematics_vmap(self, frame_markers):
        valid_P0 = self.raw_data[0]
        def kabsch_single(Q):
            c_P, c_Q = jnp.mean(valid_P0, axis=0), jnp.mean(Q, axis=0)
            H = (Q - c_Q).T @ (valid_P0 - c_P)
            U, S, Vt = jnp.linalg.svd(H)
            R = Vt.T @ U.T
            R_corr = jnp.where(jnp.linalg.det(R) < 0, (Vt.T @ jnp.diag(jnp.array([1.0, 1.0, -1.0]))) @ U.T, R)
            ql = ((Q - c_Q) @ R_corr.T + c_P - self.centroid_0) @ self.local_axes
            return ql, R_corr, c_Q, c_P
        return vmap(kabsch_single)(frame_markers)

class PlateMechanicsSolver:
    def __init__(self, config: PlateConfig):
        self.cfg = config
        self.D = (config.youngs_modulus * config.thickness**3) / (12.0 * (1.0 - config.poisson_ratio**2))
        self.res = config.mesh_resolution
        self.opt = KirchhoffPlateOptimizer(degree=config.poly_degree)

    def setup_mesh(self, x_bounds, y_bounds):
        self.x_lin = jnp.linspace(x_bounds[0], x_bounds[1], self.res)
        self.y_lin = jnp.linspace(y_bounds[0], y_bounds[1], self.res)
        self.X_mesh, self.Y_mesh = jnp.meshgrid(self.x_lin, self.y_lin)

    @partial(jit, static_argnums=(0,))
    def evaluate_batch(self, p_coeffs_batch):
        Xf, Yf = self.X_mesh.ravel(), self.Y_mesh.ravel()
        Xm, Hm = self.opt.get_basis_matrix(Xf, Yf), self.opt.get_hessian_basis(Xf, Yf)
        @vmap
        def eval_fn(p):
            w = (Xm @ p).reshape(self.res, self.res)
            k_raw = -jnp.einsum('nkd,k->nd', Hm, p).reshape(self.res, self.res, 3) 
            kxx, kyy, kxy = k_raw[..., 0], k_raw[..., 1], k_raw[..., 2]
            s_c = 6.0 * self.D / (self.cfg.thickness**2)
            sx, sy, txy = s_c*(kxx + self.cfg.poisson_ratio*kyy), s_c*(kyy + self.cfg.poisson_ratio*kxx), s_c*(1.0-self.cfg.poisson_ratio)*kxy
            vm = jnp.sqrt(jnp.maximum(sx**2 + sy**2 - sx*sy + 3.0*txy**2, 1e-12))
            svm = vm * jnp.sign(sx + sy)
            ex, ey, gxy = (self.cfg.thickness/2.0)*kxx, (self.cfg.thickness/2.0)*kyy, (self.cfg.thickness)*kxy
            eq_e = (2.0/3.0) * jnp.sqrt(jnp.maximum(1.5*(ex**2+ey**2)+0.75*gxy**2, 1e-20))
            fields = {'Displacement [mm]': w, 'Curvature XX [1/mm]': kxx, 'Mean Curvature [1/mm]': 0.5*(kxx+kyy),
                      'Stress XX [MPa]': sx, 'Signed Von-Mises [MPa]': svm, 'Signed Eq. Strain [mm/mm]': eq_e*jnp.sign(ex+ey)}
            stats = {f'Mean-{k}': jnp.mean(v) for k, v in fields.items()}
            stats.update({f'Max-{k}': jnp.max(v) for k, v in fields.items()})
            return {**fields, **stats}
        return eval_fn(p_coeffs_batch)

class KirchhoffPlateOptimizer:
    def __init__(self, degree=5):
        self.basis_indices = [(i, j) for i in range(degree + 1) for j in range(degree + 1 - i)]
        self.n = len(self.basis_indices)
    def get_basis_matrix(self, x, y): return jnp.stack([x**i * y**j for i, j in self.basis_indices], axis=1)
    def get_hessian_basis(self, x, y):
        def h_val(xi, yi):
            return jnp.stack([jnp.array([i*(i-1)*xi**(jnp.maximum(0,i-2))*yi**j if i>=2 else 0., j*(j-1)*xi**i*yi**(jnp.maximum(0,j-2)) if j>=2 else 0., i*j*xi**(jnp.maximum(0,i-1))*yi**(jnp.maximum(0,j-1)) if i>=1 and j>=1 else 0.]) for i, j in self.basis_indices], axis=0)
        return vmap(h_val)(x, y)
    @partial(jit, static_argnums=(0,))
    def solve_analytical(self, q_loc, p_ref, reg):
        Z, X, H = q_loc[:, :, 2] - p_ref[:, 2], self.get_basis_matrix(p_ref[:, 0], p_ref[:, 1]), self.get_hessian_basis(p_ref[:, 0], p_ref[:, 1])
        Bxx, Byy, Bxy = H[:, :, 0], H[:, :, 1], H[:, :, 2]
        M = (X.T @ X) / X.shape[0] + reg * (Bxx.T @ Bxx + Byy.T @ Byy + 2.0*Bxy.T @ Bxy) / X.shape[0] + jnp.eye(self.n)*1e-12
        return vmap(lambda z: solve(M, (X.T @ z) / X.shape[0]))(Z)

# ==============================================================================
# [5] ShellDeformationAnalyzer — 전체 분석 파이프라인 오케스트레이터
# ==============================================================================
class ShellDeformationAnalyzer:
    """
    Kirchhoff 박판 구조 해석 파이프라인을 총괄하는 메인 클래스.

    역할:
        1. 마커 이력 데이터를 KinematicsManager 로 전달해 강체 운동 분리
        2. 배치 단위로 KirchhoffPlateOptimizer → PlateMechanicsSolver 연계 실행
        3. 결과(응력·변형·마커 변위 이력)를 self.results dict 에 집적
        4. 대시보드(QtVisualizer) 실행 진입점 제공

    단위계: mm-tonne-sec (모든 거리/변위: mm, 응력: MPa)
    """

    def __init__(self, markers: np.ndarray, config: PlateConfig, times: np.ndarray = None):
        """
        Args:
            markers : 마커 위치 이력 배열, shape=(n_frames, n_markers, 3) [mm]
            config  : PlateConfig 해석 파라미터 인스턴스
            times   : 각 프레임의 시각 배열 (n_frames,) [sec].
                      None 이면 0 ~ 1 s 균등 분할로 자동 생성.
        """
        # mm-tonne-sec 네이티브 단위계 — 추가 스케일링 없이 원본 그대로 사용
        self.cfg    = config
        self.m_raw  = jnp.array(markers)                       # (n_frames, n_markers, 3) [mm]
        self.kin    = KinematicsManager(markers)               # 강체 운동 분리기
        self.sol    = PlateMechanicsSolver(config)             # 응력 계산기
        self.times  = times if times is not None else np.linspace(0, 1.0, markers.shape[0])
        self.results = {}                                       # 분석 결과 저장소

    def run_analysis(self):
        """
        전체 프레임에 대해 배치 해석을 수행하고 self.results 를 채움.

        처리 순서:
            1. 메쉬 초기화 (로컬 좌표 범위 기준)
            2. 배치 루프: Kabsch 정렬 → 다항식 피팅 → 응력·변형 필드 계산
            3. 배치 결과 이어붙이기(concatenate)
            4. 마커 글로벌 Z 이력 및 로컬 W 이력 추가 저장
        """
        print(f"[WHTOOLS] Analysis started ({self.kin.n_frames} frames)...")
        start = time.time()

        # 1) 평가 메쉬 생성 (로컬 좌표 범위 기준)
        self.sol.setup_mesh(self.kin.x_bounds, self.kin.y_bounds)

        # 0번 프레임의 로컬 마커 위치 → 피팅 기준점(p_ref)
        p_ref = (self.kin.raw_data[0] - self.kin.centroid_0) @ self.kin.local_axes

        bs, n = self.cfg.batch_size, self.kin.n_frames
        buf = None   # 배치별 결과를 누적할 버퍼

        # 2) 배치 루프
        for i in range((n + bs - 1) // bs):
            idx_s = i * bs
            idx_e = min((i + 1) * bs, n)

            # Kabsch 정렬: 강체 운동 제거 → 로컬 변위 추출
            q_loc, rot, cq, cp = self.kin.extract_kinematics_vmap(self.m_raw[idx_s:idx_e])

            # 다항식 계수 피팅 (Tikhonov 정규화 적용)
            params = self.sol.opt.solve_analytical(q_loc, p_ref, self.cfg.reg_lambda)

            # 응력·변형·곡률 필드 일괄 계산
            batch = self.sol.evaluate_batch(params)

            # 3) 버퍼 초기화 (첫 배치 시)
            if buf is None:
                buf = {k: [] for k in list(batch.keys()) + ['R', 'c_Q', 'c_P', 'Q_local']}

            # 결과 누적
            for k, v in batch.items():
                buf[k].append(np.array(v))
            buf['R'].append(np.array(rot))       # 회전 행렬
            buf['c_Q'].append(np.array(cq))      # 현재 프레임 무게중심
            buf['c_P'].append(np.array(cp))      # 기준 프레임 무게중심
            buf['Q_local'].append(np.array(q_loc))  # 로컬 마커 좌표

        # 4) 배치 결과를 시간축으로 이어붙이기
        self.results = {k: np.concatenate(v, axis=0) for k, v in buf.items()}

        # 마커별 이력 데이터 추가 저장
        # - Marker Global Z [mm]: 전역 좌표계에서 마커의 절대 높이 이력
        self.results['Marker Global Z [mm]'] = np.array(self.m_raw[:, :, 2])
        # - Marker Local W [mm]: 로컬 좌표계에서 마커의 순수 굽힘 변위 이력
        self.results['Marker Local W [mm]']  = self.results['Q_local'][:, :, 2]

        print(f"[WHTOOLS] Done: {time.time()-start:.4f}s")

    def show(self, ground_size: tuple = (2000.0, 2000.0), marker_names: list = None):
        """
        PySide6 + PyVista + Matplotlib 통합 대시보드를 실행.

        Args:
            ground_size  : 시각화 바닥면 크기 (width_mm, height_mm). 기본 2000×2000 mm.
            marker_names : 3D 뷰포트에 표시할 마커 이름 리스트.
                           None 이면 'M01', 'M02', ... 자동 할당.
        """
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        gui = QtVisualizer(self, ground_size=ground_size, marker_names=marker_names)
        gui.show()
        sys.exit(app.exec())

# ==============================================================================
# [6] QtVisualizer — PySide6 + PyVista + Matplotlib 통합 대시보드
# ==============================================================================
class QtVisualizer(QtWidgets.QMainWindow):
    """
    WHTOOLS v8-Pro++ 프리미엄 구조 해석 대시보드.

    레이아웃 구성:
        ├─ 상단 헤더: 로고 + 3D View Control GroupBox + 2D Plot Control GroupBox
        ├─ 중단 스플리터: PyVista 3D 뷰포트(7) | Matplotlib 2D 그래프(3)
        └─ 하단 컴트롤: 타임슬라이더 + 재생/스텝 버튼

    단위: mm-tonne-sec (mm, MPa)
    """

    def __init__(self, analyzer, ground_size: tuple = (2000.0, 2000.0), marker_names: list = None):
        """
        Args:
            analyzer     : ShellDeformationAnalyzer 인스턴스 (run_analysis 완료 상태)
            ground_size  : 시각화 바닥면 크기 (width_mm, height_mm)
            marker_names : 3D 뷰포트 마커 라벨 리스트.
                           None 이면 'M01', 'M02', ... 자동 부여.
        """
        super().__init__()

        # 분석 객체 및 주요 데이터 참조 저장
        self.analyzer    = analyzer
        self.ground_size = ground_size
        self.kin         = analyzer.kin          # KinematicsManager
        self.sol         = analyzer.sol          # PlateMechanicsSolver
        self.results     = analyzer.results      # 필드·통계 결과 dict

        # 마커 이름: 입력이 없으면 M01 ~ M{n} 자동 할당
        self.marker_names = (
            marker_names if marker_names
            else [f'M{i+1:02d}' for i in range(self.kin.n_markers)]
        )

        # 3D 필드 키 목록: shape=(n_frames, res, res) 인 필드만 선별
        self.field_keys = [k for k in self.results if self.results[k].ndim == 3]
        # 2D 시계열 키 목록: shape=(n_frames,) 인 통계 필드
        self.stat_keys  = [k for k in self.results if self.results[k].ndim == 1]
        # 실시간 마커 변위 시계열 탤 추가
        self.stat_keys += ['Marker Local Disp. [mm]', 'Marker Global Disp. [mm]']

        # 플레이백 상태 변수
        self.current_frame = 0
        self.is_playing    = False
        self.plot_objs     = {}
        self.lbl_actor     = None   # PyVista 마커 라벨 액터 (Actor Refresh 용)

        # UI 초기화 순서 (3D 뷰 없이 UI 처리 불가, 2D 보다 3D 먼저)
        self._init_ui()
        self._init_3d_view()
        self._init_2d_plots()
        self._apply_defaults()
        self.update_frame(0)  # 첫 프레임 렌더링

    def _init_ui(self):
        self.setWindowTitle("TVPackageMotion Sim Pro++ Grouped Dashboard"); self.resize(1800, 1000)
        main_w = QtWidgets.QWidget(); self.setCentralWidget(main_w); main_l = QtWidgets.QVBoxLayout(main_w)
        header = QtWidgets.QHBoxLayout()
        # Logo
        logo_path = r"C:\Users\GOODMAN\WHToolsBox\TVPackageMotionSim\sidebar_logo.png"
        lbl_logo = QtWidgets.QLabel()
        if os.path.exists(logo_path): lbl_logo.setPixmap(QtGui.QPixmap(logo_path).scaledToHeight(150))
        header.addWidget(lbl_logo)
        
        # GB1: 3D View Control
        gb_3d = QtWidgets.QGroupBox("3D View Control")
        l_gb_3d = QtWidgets.QVBoxLayout(gb_3d)
        row1_3d = QtWidgets.QHBoxLayout(); row2_3d = QtWidgets.QHBoxLayout()
        self.cmb_view = self._create_combo("View:", ["Global", "Local"], row1_3d)
        self.cmb_leg = self._create_combo("Legend:", ["Dynamic", "Static"], row1_3d)
        self.cmb_3d = self._create_combo("Field:", self.field_keys, row1_3d); row1_3d.addStretch()
        
        row2_3d.addWidget(QtWidgets.QLabel("Scale:"))
        self.spin_scale = QtWidgets.QDoubleSpinBox(); self.spin_scale.setRange(0.1, 1000.); self.spin_scale.setValue(1.0)
        row2_3d.addWidget(self.spin_scale); row2_3d.addSpacing(10); row2_3d.addWidget(QtWidgets.QLabel("Range:"))
        self.spin_min = QtWidgets.QDoubleSpinBox(); self.spin_max = QtWidgets.QDoubleSpinBox()
        for s in [self.spin_scale, self.spin_min, self.spin_max]: 
            s.setRange(-1e12, 1e12); s.valueChanged.connect(lambda: self.update_frame(self.current_frame))
        row2_3d.addWidget(self.spin_min); row2_3d.addWidget(self.spin_max); row2_3d.addStretch()
        l_gb_3d.addLayout(row1_3d); l_gb_3d.addLayout(row2_3d); header.addWidget(gb_3d)
        
        # GB2: 2D Plot Control
        gb_2d = QtWidgets.QGroupBox("2D Plot Control")
        l_gb_2d = QtWidgets.QVBoxLayout(gb_2d); r1 = QtWidgets.QHBoxLayout(); r2 = QtWidgets.QHBoxLayout()
        self.cmb_f1 = self._create_combo("F-1:", self.field_keys, r1); self.cmb_f2 = self._create_combo("F-2:", self.field_keys, r1); r1.addStretch()
        self.cmb_c1 = self._create_combo("C-1:", self.stat_keys, r2); self.cmb_c2 = self._create_combo("C-2:", self.stat_keys, r2); r2.addStretch()
        l_gb_2d.addLayout(r1); l_gb_2d.addLayout(r2); header.addWidget(gb_2d); main_l.addLayout(header)
        
        # Splitter & Playback
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.v_int = QtInteractor(self); split.addWidget(self.v_int)
        c2d = QtWidgets.QWidget(); l2d = QtWidgets.QVBoxLayout(c2d); self.canv = FigureCanvas(Figure(figsize=(10,10)))
        l2d.addWidget(NavigationToolbar(self.canv, self)); l2d.addWidget(self.canv); split.addWidget(c2d); split.setStretchFactor(0, 7)
        main_l.addWidget(split, stretch=8); ctrl = QtWidgets.QHBoxLayout()
        for t, s in [("<<", 0), ("<", -1), ("▶", -2), (">", 1), (">>", 9999)]:
            btn = QtWidgets.QPushButton(t); btn.setFixedWidth(40); btn.clicked.connect(partial(self._ctrl_slot, s)); ctrl.addWidget(btn)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.slider.setRange(0, self.kin.n_frames - 1)
        self.slider.valueChanged.connect(self.update_frame); ctrl.addWidget(self.slider)
        self.lbl_f = QtWidgets.QLabel("Frame: 0"); ctrl.addWidget(self.lbl_f); main_l.addLayout(ctrl)
        self.timer = QtCore.QTimer(); self.timer.timeout.connect(lambda: self.update_frame(self.current_frame + 1 if self.current_frame < self.kin.n_frames-1 else 0))

    def _create_combo(self, label: str, items: list, layout) -> QtWidgets.QComboBox:
        """
        QLabel + QComboBox 쌍을 지정된 레이아웃에 추가하고 콤보박스를 반환.
        콤보박스 변경 시 자동으로 현재 프레임을 다시 렌더링.

        Args:
            label  : 콤보박스 않 라벨 텍스트
            items  : 콤보박스 항목 리스트
            layout : 위젯을 추가할 부모 QHBoxLayout

        Returns:
            초기화된 QComboBox 인스턴스
        """
        layout.addWidget(QtWidgets.QLabel(label))
        combo = QtWidgets.QComboBox()
        combo.addItems(items)
        combo.currentIndexChanged.connect(lambda: self.update_frame(self.current_frame))
        layout.addWidget(combo)
        return combo

    def _ctrl_slot(self, step: int):
        """
        타임라인 컴트롤 버튼 핸들러.

        Args:
            step : -2=재생토글, 0=첫프, 9999=마지, -1=이전프, +1=다음프
        """
        if step == -2:
            # 재생/일시정지 토글
            if self.is_playing:
                self.timer.stop()
            else:
                self.timer.start(30)  # 30ms ≈ 33fps
            self.is_playing = not self.is_playing
        elif step == 0:
            self.update_frame(0)                                        # 첫 프레임
        elif step == 9999:
            self.update_frame(self.kin.n_frames - 1)                   # 마지 프레임
        else:
            target = max(0, min(self.kin.n_frames - 1, self.current_frame + step))
            self.update_frame(target)                                   # 상대 이동

    def _apply_defaults(self):
        """
        콤보박스 기본값 설정 및 스핀박스 초기값 동기화.
        해석에서 실제 발생한 데이터 범위를 자동 접수하여 Min/Max 스핀박스에 반영.
        """
        # 콤보박스별 기본 선택 항목 맵
        defaults = {
            "Mean Curvature [1/mm]": self.cmb_3d,
            "Displacement [mm]":    self.cmb_f1,
            "Signed Von-Mises [MPa]": self.cmb_f2,
            "Max-Displacement [mm]": self.cmb_c1,
            "Marker Local Disp. [mm]": self.cmb_c2,
        }
        for key, combo in defaults.items():
            if combo.findText(key) >= 0:
                combo.setCurrentText(key)
        # 3D 필드의 실제 범위를 데칗어 수동 Min/Max 스핀박스 설정
        field_data = self.results[self.cmb_3d.currentText()]
        self.spin_min.setValue(float(field_data.min()))
        self.spin_max.setValue(float(field_data.max()))

    def _init_3d_view(self):
        """PyVista 3D 뷰포트 초기화 — 바닥면, 판 메쉬, 마커 구(Sphere), 라벨, 스커라바 생성."""
        self.v_int.set_background("white")

        # 바닥면 평면 생성 (Global 모드에서만 표시, 단위: mm)
        self.ground = self.v_int.add_mesh(
            pv.Plane(i_size=self.ground_size[0], j_size=self.ground_size[1]),
            color="blue", opacity=0.3
        )

        # 판 메쉬 지지점 기저 배열 생성 (Z=0 플렇, 동적으로 변형 적용)
        res = self.sol.res
        self.p_base = np.column_stack([
            self.analyzer.sol.X_mesh.ravel(),
            self.analyzer.sol.Y_mesh.ravel(),
            np.zeros(res**2)
        ])

        # PyVista Plane 메쉬 생성 (판 크기는 로컬 범위로 자동 결정)
        plate_w = float(self.analyzer.sol.x_lin.max() - self.analyzer.sol.x_lin.min())
        plate_h = float(self.analyzer.sol.y_lin.max() - self.analyzer.sol.y_lin.min())
        self.poly = pv.Plane(
            i_size=plate_w, j_size=plate_h,
            i_resolution=res-1, j_resolution=res-1
        )
        self.poly.point_data["S"] = np.zeros(res**2)  # 컴컴맴맴 데이터 슈드 (Turbo 색상맵)
        self.mesh_a = self.v_int.add_mesh(
            self.poly, scalars="S", cmap="turbo",
            show_edges=True, edge_color="lightgrey", show_scalar_bar=False
        )

        # 마커 구(Sphere) 초기 위치를 쳏 프레임 데이터로 설정
        # ⚠️ 여기서 None으로 설정하면 라벨러가 원점(0,0,0)에 고정되는 버그 발생
        m_init = np.array(self.analyzer.kin.raw_data[0])   # 쳏 프레임 마커 위치
        self.m_poly = pv.PolyData(m_init)
        self.m_a = self.v_int.add_mesh(
            self.m_poly, render_points_as_spheres=True, point_size=10, color='blue'
        )

        # 마커 이름 라벨 추가 (Actor Refresh 전략으로 실시간 추종, update_frame 에서 재생성)
        self.lbl_actor = self.v_int.add_point_labels(
            self.m_poly, self.marker_names,
            font_size=8, text_color='black', always_visible=True, point_size=0, shadow=False
        )

        # 하단 중앙 수평 스케일바 (Dynamic/Static 레전드 모드)
        self.sb = self.v_int.add_scalar_bar(
            "Field", position_x=0.15, position_y=0.05, width=0.70,
            vertical=False, title_font_size=14, label_font_size=14
        )
        self.v_int.view_isometric()

    def _init_2d_plots(self):
        """Matplotlib 2x2 초기화 — 콸투어 맵 2장(상단) + 시계열 커브 2장(하단)."""
        fig = self.canv.figure
        fig.subplots_adjust(hspace=0.4, wspace=0.3)
        self.axes = [fig.add_subplot(2, 2, i+1) for i in range(4)]

        # 컨투어 맵 에 대한 실시간 imshow 객체 (idx 0, 1)
        self.ims        = [None, None]
        # 시계열 커브 라인 객체 (idx [plot_idx][marker_idx])
        self.multi_lines = [[None]*9 for _ in range(2)]
        # 현재 시점 주표 점(Dot) 객체
        self.dots        = [[None]*9 for _ in range(2)]
        # 현재 프레임 시각을 나타내는 수직선
        self.vline       = [None, None]
        # 센서 채널별 고유 색상 팔레트 (Tab10, 10색 반복)
        self.colors      = plt.cm.tab10.colors

        # 상단 컨투어 서브플롯 초기화
        for i in range(2):
            extent = [*self.kin.x_bounds, *self.kin.y_bounds]
            self.ims[i] = self.axes[i].imshow(
                np.zeros((25, 25)), extent=extent, cmap='turbo'
            )
            fig.colorbar(self.ims[i], ax=self.axes[i])

        # 하단 시계열 서브플롯 초기화
        for i in range(2):
            self.axes[i+2].grid(True, alpha=0.3)
            self.vline[i] = self.axes[i+2].axvline(0, color='red', ls='--')

    def update_frame(self, f: int):
        """
        f 번 프레임이로 3D 및 2D 뷰를 일괄 업데이트.
        타임슬라이더, 자동재생 타이머, 콤보박스 변경 시 모두 호출.

        Args:
            f: 렌더링할 프레임 인덱스 (0 ≤ f < n_frames)
        """
        # 현재 프레임 및 슬라이더 동기화 (슬라이더 신호 루프 방지)
        self.current_frame = f
        self.slider.blockSignals(True)
        self.slider.setValue(f)
        self.slider.blockSignals(False)

        # 시각 정보 표시
        t = self.analyzer.times[f]
        self.lbl_f.setText(f"Frame: {f} | Time: {t:.3f}s")

        # 현재 UI 상태 가져오기
        view    = self.cmb_view.currentText()   # 'Global' 또는 'Local'
        field_k = self.cmb_3d.currentText()     # 3D 필드 키
        scale   = self.spin_scale.value()         # 변형 스케일

        # 필드 값 추출 (field_key 검증 후 백폴)
        valid_key  = field_k if field_k in self.results else 'Displacement [mm]'
        field_vals = self.results[valid_key][f]                     # 콜러맵용 필드
        field_w    = self.results['Displacement [mm]'][f]           # Z방향 변형용

        # 로컀 맴 좌표에 Z 변형럹 스케일 적용
        p_def = self.p_base.copy()
        p_def[:, 2] = field_w.ravel() * scale

        # ── 묨 모드별 평면 및 마커 좌표 결정 ──
        if view == "Global":
            # 전역 모드: 바닥면 표시, Kabsch 정렬으로 판을 전역 좌표로 변환
            self.ground.SetVisibility(True)
            R  = self.results['R'][f]      # 회전 행렬
            cq = self.results['c_Q'][f]   # 현재 무게중심
            cp = self.results['c_P'][f]   # 기준 무게중심
            p_f = (p_def @ np.array(self.kin.local_axes).T
                   + np.array(self.kin.centroid_0) - cp) @ R + cq
            m_f = np.array(self.kin.raw_data[f])   # 마커 전역 좌표
        else:
            # 로컀 모드: 바닥면 표시 끼고 로컀 체계 좌표 사용
            self.ground.SetVisibility(False)
            p_f = p_def
            m_f = np.array(self.results['Q_local'][f])  # 마커 로컀 좌표

        # ── 3D 공간 기하형 업데이트 ──
        self.poly.points        = np.array(p_f)         # 판 메쉬 좌표 갱신
        self.poly.point_data["S"] = field_vals.ravel()  # 컴컴맴맴 데이터 갱신
        self.m_poly.points      = np.array(m_f)         # 마커 구 좌표 갱신

        # 마커 라벨 액터 실시간 재생성 (Actor Refresh Strategy)
        # PyVista/Qt 환경에서 좌표 업데이트만으로는 라벨이 고정되는 문제를 방지
        if self.lbl_actor:
            self.v_int.remove_actor(self.lbl_actor)
        self.lbl_actor = self.v_int.add_point_labels(
            self.m_poly, self.marker_names,
            font_size=8, text_color='black', always_visible=True, point_size=0, shadow=False
        )

        # 레전드 스케일 업데이트
        clim = (
            [field_vals.min(), field_vals.max()]
            if self.cmb_leg.currentText() == "Dynamic"
            else [self.spin_min.value(), self.spin_max.value()]
        )
        if clim[0] == clim[1]:  # 단일 값 시 범위 경계 안전 처리
            clim = [clim[0] - 1e-6, clim[1] + 1e-6]
        self.sb.title = field_k
        self.v_int.update_scalar_bar_range(clim)
        self.mesh_a.mapper.scalar_range      = clim
        self.mesh_a.mapper.scalar_visibility = True

        # ── 2D 컨투어 맵 업데이트 (imshow set_data 성능 최적화) ──
        for i, cmb in enumerate([self.cmb_f1, self.cmb_f2]):
            d = self.results[cmb.currentText()][f]
            self.ims[i].set_data(d)
            self.ims[i].set_clim(d.min(), d.max())
            self.axes[i].set_title(cmb.currentText())

        # ── 2D 시계열 커브 업데이트 ──
        for i, cmb in enumerate([self.cmb_c1, self.cmb_c2]):
            key = cmb.currentText()
            ax  = self.axes[i + 2]
            ax.set_title(f"History: {key}")

            if key in ['Marker Local Disp. [mm]', 'Marker Global Disp. [mm]']:
                # 마커별 9개 독립 커브 그리기
                data_key = 'Marker Global Z [mm]' if key == 'Marker Global Disp. [mm]' else 'Marker Local W [mm]'
                data = self.results[data_key]   # shape: (n_frames, n_markers)
                for j in range(9):
                    y_cur = float(data[f, j])
                    if self.multi_lines[i][j] is None:
                        # 첫 렌더링: 라인 및 쫐 동시 생성
                        self.multi_lines[i][j], = ax.plot(
                            self.analyzer.times, data[:, j],
                            alpha=0.7, lw=1, color=self.colors[j % 10]
                        )
                        self.dots[i][j], = ax.plot(
                            [t], [y_cur], 'o', color=self.colors[j % 10], ms=5, zorder=5
                        )
                    else:
                        # 이후 프레임: set_data로 성능 유지
                        self.multi_lines[i][j].set_ydata(data[:, j])
                        self.dots[i][j].set_data([t], [y_cur])
                        self.dots[i][j].set_visible(True)
            else:
                # 단일 커브 모드: 마커 9개 쫐/라인 눈출 안어
                y_cur = float(self.results[key][f])
                for j in range(9):
                    if self.multi_lines[i][j]:
                        self.multi_lines[i][j].set_visible(False)
                    if self.dots[i][j]:
                        self.dots[i][j].set_visible(False)
                if self.multi_lines[i][0] is None:
                    # 첫 렌더링: 단일 라인 생성
                    self.multi_lines[i][0], = ax.plot(
                        self.analyzer.times, self.results[key],
                        color='#1A73E8', lw=1.5
                    )
                    self.dots[i][0], = ax.plot([t], [y_cur], 'ro', ms=6, zorder=5)
                else:
                    self.multi_lines[i][0].set_ydata(self.results[key])
                    self.multi_lines[i][0].set_visible(True)
                    self.dots[i][0].set_data([t], [y_cur])
                    self.dots[i][0].set_visible(True)

            ax.relim()
            ax.autoscale_view()
            self.vline[i].set_xdata([t, t])  # 현재 시각 수직선 이동

        self.canv.draw_idle()  # 비동기 렌더링 (Qt 이벤트 루프와 호환)

def create_example_markers(n=100):
    """
    [WHTOOLS] v8-Pro++ 고성능 테스트 데이터 생성기
    현실적인 낙하 및 충돌 거동(자유낙하 -> 충돌 -> 바운싱 -> 국부 변형)을 시뮬레이션합니다.
    """
    # 1. 시간 축 및 마커 배치 설정 (3x3 그리드)
    t_arr = np.linspace(0, 2.0, n)
    # 가로 1800mm, 세로 1200mm 범위에 9개의 마커를 균등 배치 (mm 단위)
    mx, my = np.meshgrid(np.linspace(-900.0, 900.0, 3), np.linspace(-600.0, 600.0, 3))
    mloc = np.column_stack([mx.ravel(), my.ravel(), np.zeros(9)])
    
    data = np.zeros((n, 9, 3))
    z0, t_hit = 500.0, 0.32  # 초기 높이 500mm, 충돌 시점 0.32초
    g = 9810.0  # 중력 가속도 9.81 m/s^2 -> 9810 mm/s^2
    
    for f, t in enumerate(t_arr):
        if t < t_hit:
            # [자유 낙하 단계] mm/s^2 기반 낙하 궤적
            z = z0 - 0.5 * g * t**2
            ang = np.deg2rad(15.0)
        else:
            # [충돌 및 바운싱 단계] 지면 비관통 강제 (max 0 처리)
            th = t - t_hit
            # 감쇠 바운싱: 낙하 높이의 약 10% 수준에서 진동
            z = 50.0 * np.exp(-3.0 * th) * np.abs(np.sin(10.0 * th)) 
            ang = np.deg2rad(15.0 * np.exp(-10.0 * th))             
        
        # 지면 비관통 보장 (Z >= 0)
        z = max(0.0, z)
        
        # 2. 강체 회전 행렬 생성 (Y축 기준 분석)
        ry = np.array([
            [np.cos(ang),  0, np.sin(ang)],
            [0,            1, 0],
            [-np.sin(ang), 0, np.cos(ang)]
        ])
        
        # 3. 국부 변형(Ripple) 시뮬레이션 (Scale: mm)
        # 80mm 진폭 수준에서 감쇠되는 파동
        ripple_amp = 80.0 * np.exp(-4.0 * max(0, t - t_hit)) * np.sin(25.0 * max(0, t - t_hit))
        # 정규화된 위치(-1~1) 기반으로 리플 파동 강도 분포
        ripple = ripple_amp * (mx.ravel() / 900.0 * my.ravel() / 600.0) 
        
        # 4. 최종 글로벌 좌표 합성 (mm 단위)
        data[f] = mloc @ ry.T + [0, 0, z]
        data[f, :, 2] += ripple
        
        # [Ground Collision Guard] 모든 마커가 지면(Z=0) 위에 있도록 프레임 전체를 상향 보정
        min_z = float(np.min(data[f, :, 2]))
        if min_z < 0:
            data[f, :, 2] -= min_z
        
    return data, t_arr

if __name__ == "__main__":
    # [WHTOOLS] v8-Pro++ 통합 구조 해석 대시보드 실행 예제
    # 1. 플레이트 및 해석 설정 (mm-tonne-sec 네이티브 단위계)
    # thickness: 플레이트 두께 [mm] / youngs_modulus: 탄성 계수 [MPa]
    cfg = PlateConfig(
        thickness=5.0,
        youngs_modulus=100.0,
        mesh_resolution=20,
        reg_lambda=1e-4
    )
    
    # 2. 예제 마커 데이터 생성 (100프레임 시뮬레이션)
    markers, times = create_example_markers(n=100)
    
    # 3. 마커 명칭 명시적 설정 (미설정 시 M01~ 자동 할당)
    custom_names = [f"M_{i+1:02d}" for i in range(9)]
    
    # 4. 분석기 초기화 및 최적화 루틴 실행 (JAX 가속)
    analyzer = ShellDeformationAnalyzer(markers, cfg, times=times)
    analyzer.run_analysis()
    
    # 5. 프리미엄 대시보드 가시화
    # ground_size: 시각화 바닥 크기 [mm x mm]
    analyzer.show(
        ground_size=(3000.0, 3000.0), 
        marker_names=custom_names
    )
