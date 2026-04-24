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
import pickle
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
        
        # [WHTOOLS] 스케일 대응형 Margin (전체 크기의 5%)
        span = p_max - p_min
        margin = np.max(span) * 0.05
        
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
        
        # [WHTOOLS] 정규화용 스케일 팩터 저장
        self.x_scale = jnp.max(jnp.abs(jnp.array(x_bounds))) + 1e-9
        self.y_scale = jnp.max(jnp.abs(jnp.array(y_bounds))) + 1e-9

    @partial(jit, static_argnums=(0,))
    def evaluate_batch(self, p_coeffs_batch):
        # [WHTOOLS] 정규화된 좌표에서 평가
        Xn, Yn = self.X_mesh.ravel() / self.x_scale, self.Y_mesh.ravel() / self.y_scale
        Xm = self.opt.get_basis_matrix(Xn, Yn)
        Hm = self.opt.get_hessian_basis(Xn, Yn)
        
        @vmap
        def eval_fn(p):
            w = (Xm @ p).reshape(self.res, self.res)
            # 곡률 계산 시 스케일 팩터 재적용 (Chain rule: d2w/dx2 = (1/sx^2) * d2w_norm/dx_norm^2)
            # k_raw[..., 0]: Bxx, k_raw[..., 1]: Byy, k_raw[..., 2]: Bxy
            kxx = -jnp.dot(Hm[:, :, 0], p).reshape(self.res, self.res) / (self.x_scale**2)
            kyy = -jnp.dot(Hm[:, :, 1], p).reshape(self.res, self.res) / (self.y_scale**2)
            kxy = -jnp.dot(Hm[:, :, 2], p).reshape(self.res, self.res) / (self.x_scale * self.y_scale)
            
            s_c = 6.0 * self.D / (self.cfg.thickness**2)
            sx, sy, txy = s_c*(kxx + self.cfg.poisson_ratio*kyy), s_c*(kyy + self.cfg.poisson_ratio*kxx), s_c*(1.0-self.cfg.poisson_ratio)*kxy
            # --- Stress Metrics Expansion ---
            vm = jnp.sqrt(jnp.maximum(sx**2 + sy**2 - sx*sy + 3.0*txy**2, 1e-12))
            svm = vm * jnp.sign(sx + sy)
            # Principal Stresses
            s_avg = (sx + sy) / 2.0
            s_diff = (sx - sy) / 2.0
            radius = jnp.sqrt(jnp.maximum(s_diff**2 + txy**2, 1e-12))
            s1, s2 = s_avg + radius, s_avg - radius
            
            ex, ey, gxy = (self.cfg.thickness/2.0)*kxx, (self.cfg.thickness/2.0)*kyy, (self.cfg.thickness)*kxy
            eq_e = (2.0/3.0) * jnp.sqrt(jnp.maximum(1.5*(ex**2+ey**2)+0.75*gxy**2, 1e-20))
            
            fields = {
                'Displacement [mm]': w, 
                'Curvature XX [1/mm]': kxx, 
                'Mean Curvature [1/mm]': 0.5*(kxx+kyy),
                'Stress XX [MPa]': sx, 
                'Stress YY [MPa]': sy,
                'Stress XY [MPa]': txy,
                'Von-Mises [MPa]': vm,
                'Signed Von-Mises [MPa]': svm, 
                'Principal Max [MPa]': s1,
                'Principal Min [MPa]': s2,
                'Signed Eq. Strain [mm/mm]': eq_e*jnp.sign(ex+ey)
            }
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
            return jnp.stack([jnp.array([
                i*(i-1)*xi**(jnp.maximum(0,i-2))*yi**j if i>=2 else 0., 
                j*(j-1)*xi**i*yi**(jnp.maximum(0,j-2)) if j>=2 else 0., 
                i*j*xi**(jnp.maximum(0,i-1))*yi**(jnp.maximum(0,j-1)) if i>=1 and j>=1 else 0.
            ]) for i, j in self.basis_indices], axis=0)
        return vmap(h_val)(x, y)

    @partial(jit, static_argnums=(0,))
    def solve_analytical(self, q_loc, p_ref, reg):
        # [WHTOOLS] 수치적 안정성을 위해 정규화 좌표 사용
        x_scale = jnp.max(jnp.abs(p_ref[:, 0])) + 1e-9
        y_scale = jnp.max(jnp.abs(p_ref[:, 1])) + 1e-9
        xn, yn = p_ref[:, 0] / x_scale, p_ref[:, 1] / y_scale
        
        Z, X = q_loc[:, :, 2], self.get_basis_matrix(xn, yn)
        H = self.get_hessian_basis(xn, yn)
        
        # Hessian 보정 (스케일 팩터 재적용)
        Bxx, Byy, Bxy = H[:, :, 0] / (x_scale**2), H[:, :, 1] / (y_scale**2), H[:, :, 2] / (x_scale * y_scale)
        
        M = (X.T @ X) / X.shape[0] + reg * (Bxx.T @ Bxx + Byy.T @ Byy + 2.0*Bxy.T @ Bxy) / X.shape[0] + jnp.eye(self.n)*1e-12
        return vmap(lambda z: solve(M, (X.T @ z) / X.shape[0]))(Z)

class ShellDeformationAnalyzer:
    def __init__(self, markers, config, times=None):
        self.m_raw = np.array(markers)
        self.cfg = config
        self.times = times if times is not None else np.arange(len(markers))
        self.kin = KinematicsManager(markers)
        self.sol = PlateMechanicsSolver(config)
        self.sol.setup_mesh(self.kin.x_bounds, self.kin.y_bounds)

    def run_analysis(self):
        print(f"[WHTOOLS] Analysis started ({len(self.times)} frames)...")
        start = time.time()
        n_frames = len(self.times)
        
        # 1. Kinematics 추출 (Q_local, 회전 행렬, 중심점 등)
        q_loc_jax, rot, cq, cp = self.kin.extract_kinematics_vmap(self.m_raw)
        
        # 2. 최적 계수(p_coeffs) 계산
        # [Fix] 기저 행렬은 메쉬 포인트가 아니라 '마커 지점'에서 생성되어야 함
        ref_markers_local = q_loc_jax[0] # Frame 0의 로컬 마커 좌표 (n_markers, 3)
        p_coeffs = self.sol.opt.solve_analytical(q_loc_jax, ref_markers_local, self.cfg.reg_lambda)
        
        batch_size = self.cfg.batch_size
        buf = None
        for i in range(0, n_frames, batch_size):
            params = p_coeffs[i:i+batch_size]
            batch = self.sol.evaluate_batch(params)
            if buf is None:
                buf = {k: [] for k in list(batch.keys()) + ['R', 'c_Q', 'c_P', 'Q_local']}
            for k, v in batch.items(): buf[k].append(np.array(v))
            buf['R'].append(np.array(rot[i:i+batch_size]))
            buf['c_Q'].append(np.array(cq[i:i+batch_size]))
            buf['c_P'].append(np.array(cp[i:i+batch_size]))
            buf['Q_local'].append(np.array(q_loc_jax[i:i+batch_size]))

        self.results = {k: np.concatenate(v, axis=0) for k, v in buf.items()}
        self.results['Marker Global Z [mm]'] = np.array(self.m_raw[:, :, 2])
        self.results['Marker Local W [mm]']  = self.results['Q_local'][:, :, 2]
        print(f"[WHTOOLS] Done: {time.time()-start:.4f}s")

    def show(self, ground_size: tuple = (2000.0, 2000.0), marker_names: list = None):
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        gui = QtVisualizer(self, ground_size=ground_size, marker_names=marker_names)
        gui.show()
        sys.exit(app.exec())

class QtVisualizer(QtWidgets.QMainWindow):
    def __init__(self, analyzer, ground_size: tuple = (2000.0, 2000.0), marker_names: list = None):
        super().__init__()
        self.analyzer = analyzer
        self.ground_size = ground_size
        self.kin = analyzer.kin
        self.sol = analyzer.sol
        self.results = analyzer.results
        self.marker_names = marker_names if marker_names else [f'M{i+1:02d}' for i in range(self.kin.n_markers)]
        self.field_keys = [k for k in self.results if self.results[k].ndim == 3]
        self.stat_keys = [k for k in self.results if self.results[k].ndim == 1]
        self.stat_keys += ['Marker Local Disp. [mm]', 'Marker Global Disp. [mm]']
        self.current_frame = 0
        self.is_playing = False
        self.plot_objs = {}
        self.colors = plt.cm.tab10.colors
        self._init_ui()
        self._init_3d_view()
        self._init_2d_plots()
        self._apply_defaults()
        self.update_frame(0)

    def _init_ui(self):
        self.setWindowTitle("TVPackageMotion Sim Pro++ Analyzer"); self.resize(1800, 1000)
        main_w = QtWidgets.QWidget(); self.setCentralWidget(main_w); main_l = QtWidgets.QVBoxLayout(main_w)
        header = QtWidgets.QHBoxLayout()
        logo_path = r"C:\Users\GOODMAN\WHToolsBox\TVPackageMotionSim\sidebar_logo.png"
        lbl_logo = QtWidgets.QLabel()
        if os.path.exists(logo_path): lbl_logo.setPixmap(QtGui.QPixmap(logo_path).scaledToHeight(150))
        header.addWidget(lbl_logo)
        gb_3d = QtWidgets.QGroupBox("3D View Control"); l_gb_3d = QtWidgets.QVBoxLayout(gb_3d)
        r1 = QtWidgets.QHBoxLayout(); r2 = QtWidgets.QHBoxLayout()
        self.cmb_view = self._create_combo("View:", ["Global", "Local"], r1)
        self.cmb_leg = self._create_combo("Legend:", ["Dynamic", "Static"], r1)
        self.cmb_3d = self._create_combo("Field:", self.field_keys, r1); r1.addStretch()
        r2.addWidget(QtWidgets.QLabel("Scale:")); self.spin_scale = QtWidgets.QDoubleSpinBox(); self.spin_scale.setRange(0.1, 1000.); self.spin_scale.setValue(1.0)
        r2.addWidget(self.spin_scale); r2.addWidget(QtWidgets.QLabel("Min:")); self.spin_min = QtWidgets.QDoubleSpinBox(); self.spin_max = QtWidgets.QDoubleSpinBox()
        for s in [self.spin_scale, self.spin_min, self.spin_max]: s.setRange(-1e12, 1e12); s.setDecimals(4); s.valueChanged.connect(lambda: self.update_frame(self.current_frame))
        r2.addWidget(self.spin_min); r2.addWidget(self.spin_max); r2.addStretch()
        l_gb_3d.addLayout(r1); l_gb_3d.addLayout(r2); header.addWidget(gb_3d)
        gb_2d = QtWidgets.QGroupBox("2D Plot Control"); l_gb_2d = QtWidgets.QVBoxLayout(gb_2d)
        r21 = QtWidgets.QHBoxLayout(); r22 = QtWidgets.QHBoxLayout()
        self.cmb_f1 = self._create_combo("F-1:", self.field_keys, r21); self.cmb_f2 = self._create_combo("F-2:", self.field_keys, r21); r21.addStretch()
        self.cmb_c1 = self._create_combo("C-1:", self.stat_keys, r22); self.cmb_c2 = self._create_combo("C-2:", self.stat_keys, r22); r22.addStretch()
        l_gb_2d.addLayout(r21); l_gb_2d.addLayout(r22); header.addWidget(gb_2d); main_l.addLayout(header)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal); self.v_int = QtInteractor(self); split.addWidget(self.v_int)
        c2d = QtWidgets.QWidget(); l2d = QtWidgets.QVBoxLayout(c2d); self.canv = FigureCanvas(Figure(figsize=(10,10)))
        l2d.addWidget(NavigationToolbar(self.canv, self)); l2d.addWidget(self.canv); split.addWidget(c2d); split.setStretchFactor(0, 7)
        main_l.addWidget(split, stretch=8); ctrl = QtWidgets.QHBoxLayout()
        for t, s in [("▶", -2), ("<", -1), (">", 1)]:
            btn = QtWidgets.QPushButton(t); btn.setFixedWidth(40); btn.clicked.connect(partial(self._ctrl_slot, s)); ctrl.addWidget(btn)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.slider.setRange(0, self.kin.n_frames - 1); self.slider.valueChanged.connect(self.update_frame); ctrl.addWidget(self.slider)
        self.lbl_f = QtWidgets.QLabel("Frame: 0"); ctrl.addWidget(self.lbl_f); main_l.addLayout(ctrl)
        self.timer = QtCore.QTimer(); self.timer.timeout.connect(lambda: self.update_frame(self.current_frame + 1 if self.current_frame < self.kin.n_frames-1 else 0))

    def _create_combo(self, label, items, layout):
        layout.addWidget(QtWidgets.QLabel(label)); combo = QtWidgets.QComboBox(); combo.addItems(items)
        combo.currentIndexChanged.connect(lambda: self.update_frame(self.current_frame)); layout.addWidget(combo)
        return combo

    def _ctrl_slot(self, step):
        if step == -2:
            if self.is_playing: self.timer.stop()
            else: self.timer.start(30)
            self.is_playing = not self.is_playing
        else: self.update_frame(max(0, min(self.kin.n_frames - 1, self.current_frame + step)))

    def _apply_defaults(self):
        self.cmb_3d.setCurrentText("Mean Curvature [1/mm]")
        self.cmb_f1.setCurrentText("Displacement [mm]")
        self.cmb_f2.setCurrentText("Stress XX [MPa]")
        self.cmb_c1.setCurrentText("Max-Displacement [mm]")
        self.cmb_c2.setCurrentText("Marker Local Disp. [mm]")

    def _init_3d_view(self):
        self.v_int.set_background("white")
        self.ground = self.v_int.add_mesh(pv.Plane(i_size=self.ground_size[0], j_size=self.ground_size[1]), color="blue", opacity=0.3)
        res = self.sol.res
        self.p_base = np.column_stack([self.sol.X_mesh.ravel(), self.sol.Y_mesh.ravel(), np.zeros(res**2)])
        self.poly = pv.Plane(i_size=float(self.sol.x_lin.max()-self.sol.x_lin.min()), j_size=float(self.sol.y_lin.max()-self.sol.y_lin.min()), i_resolution=res-1, j_resolution=res-1)
        self.poly.point_data["S"] = np.zeros(res**2)
        self.mesh_a = self.v_int.add_mesh(self.poly, scalars="S", cmap="turbo", show_edges=True, edge_color="lightgrey")
        self.m_poly = pv.PolyData(np.array(self.kin.raw_data[0]))
        self.m_poly.point_data["names"] = self.marker_names
        self.v_int.add_mesh(self.m_poly, render_points_as_spheres=True, point_size=10, color='blue')
        self.v_int.add_point_labels(self.m_poly, "names", font_size=8, text_color='black', always_visible=True)
        self.v_int.view_isometric()

    def _init_2d_plots(self):
        fig = self.canv.figure; self.axes = [fig.add_subplot(2, 2, i+1) for i in range(4)]
        self.ims = [None, None]; self.multi_lines = [[None]*9 for _ in range(2)]; self.dots = [[None]*9 for _ in range(2)]; self.vline = [None, None]
        for i in range(2):
            self.ims[i] = self.axes[i].imshow(np.zeros((self.sol.res, self.sol.res)), extent=[*self.kin.x_bounds, *self.kin.y_bounds], cmap='turbo')
            self.axes[i+2].grid(True, alpha=0.3); self.vline[i] = self.axes[i+2].axvline(0, color='red', ls='--')

    def update_frame(self, f):
        self.current_frame = f; self.slider.blockSignals(True); self.slider.setValue(f); self.slider.blockSignals(False)
        t = self.analyzer.times[f]; self.lbl_f.setText(f"Frame: {f} | Time: {t:.3f}s")
        view, field_k, scale = self.cmb_view.currentText(), self.cmb_3d.currentText(), self.spin_scale.value()
        field_vals = self.results[field_k][f]; field_w = self.results['Displacement [mm]'][f]
        p_def = self.p_base.copy(); p_def[:, 2] = field_w.ravel() * scale
        if view == "Global":
            self.ground.SetVisibility(True)
            R, cq, cp = self.results['R'][f], self.results['c_Q'][f], self.results['c_P'][f]
            p_f = (p_def @ np.array(self.kin.local_axes).T + np.array(self.kin.centroid_0) - cp) @ R + cq
            m_f = np.array(self.analyzer.m_raw[f])
        else:
            self.ground.SetVisibility(False); p_f = p_def; m_f = np.array(self.results['Q_local'][f])
        self.poly.points = np.array(p_f); self.poly.point_data["S"] = field_vals.ravel(); self.m_poly.points = np.array(m_f)
        clim = [float(field_vals.min()), float(field_vals.max())] if self.cmb_leg.currentText() == "Dynamic" else [self.spin_min.value(), self.spin_max.value()]
        self.mesh_a.mapper.scalar_range = clim; self.v_int.render()
        self.update_plots(f, t, clim)

    def update_plots(self, f, t, clim):
        for i, cmb in enumerate([self.cmb_f1, self.cmb_f2]):
            d = self.results[cmb.currentText()][f]; self.ims[i].set_data(d); self.ims[i].set_clim(d.min(), d.max()); self.axes[i].set_title(cmb.currentText())
        for i, cmb in enumerate([self.cmb_c1, self.cmb_c2]):
            key, ax = cmb.currentText(), self.axes[i+2]
            if key in ['Marker Local Disp. [mm]', 'Marker Global Disp. [mm]']:
                data_key = 'Marker Global Z [mm]' if key == 'Marker Global Disp. [mm]' else 'Marker Local W [mm]'
                data = self.results[data_key]
                for j in range(min(9, data.shape[1])):
                    if self.multi_lines[i][j] is None:
                        self.multi_lines[i][j], = ax.plot(self.analyzer.times, data[:, j], alpha=0.7, lw=1, color=self.colors[j % 10])
                        self.dots[i][j], = ax.plot([t], [data[f, j]], 'o', color=self.colors[j % 10], ms=5)
                    else:
                        self.multi_lines[i][j].set_ydata(data[:, j]); self.dots[i][j].set_data([t], [data[f, j]])
            else:
                y = self.results[key]
                if self.multi_lines[i][0] is None:
                    self.multi_lines[i][0], = ax.plot(self.analyzer.times, y, color='blue', lw=1.5)
                    self.dots[i][0], = ax.plot([t], [y[f]], 'ro', ms=6)
                else: 
                    self.multi_lines[i][0].set_ydata(y); self.dots[i][0].set_data([t], [y[f]])
            ax.relim(); ax.autoscale_view(); self.vline[i].set_xdata([t, t])
        self.canv.draw_idle()

def test_face_from_simulation(pkl_path: str, part_name: str = "opencell_front"):
    if not os.path.exists(pkl_path):
        print(f"[ERROR] Result file not found: {pkl_path}"); return
    sim_dir = os.path.dirname(os.path.abspath(pkl_path))
    parent_dir = os.path.dirname(sim_dir)
    if parent_dir not in sys.path: sys.path.append(parent_dir)
    try: from run_drop_simulator.whts_mapping import extract_face_markers
    except ImportError: print("[ERROR] Import failed"); return
    with open(pkl_path, "rb") as f: res = pickle.load(f)
    if "_" in part_name: comp_base, face_name = part_name.split("_", 1); face_name = face_name.capitalize()
    else: comp_base, face_name = part_name, "Front"
    all_face_markers, _ = extract_face_markers(res, comp_base, use_virtual_markers=False)
    if face_name not in all_face_markers or not all_face_markers[face_name]:
        print(f"[ERROR] Face {face_name} not found"); return

    markers_dict = all_face_markers[face_name]
    m_names = sorted(list(markers_dict.keys()))
    
    # [WHTOOLS] 단위 변환: MuJoCo(m) -> Analyzer(mm)
    # 시뮬레이션 결과가 m 단위인 경우 1000배 스케일링
    markers_raw = np.stack([markers_dict[m] for m in m_names], axis=1)
    markers_mm = markers_raw * 1000.0
    
    times = np.array(res.time_history)
    
    # [WHTOOLS] 데이터 무결성 체크 (단위 정합성 확인용)
    bbox_min = np.min(markers_mm[0], axis=0)
    bbox_max = np.max(markers_mm[0], axis=0)
    dims = bbox_max - bbox_min
    print(f" > Physical Scale Check: {dims[0]:.1f} x {dims[1]:.1f} x {dims[2]:.1f} (mm)")
    cfg = PlateConfig(thickness=res.config.get("opencell_thickness", 5.0), youngs_modulus=2.1e5, poly_degree=4)
    analyzer = ShellDeformationAnalyzer(markers_mm, cfg, times=times)
    analyzer.run_analysis(); analyzer.show(ground_size=(2500, 2500))

def create_example_markers(n=100):
    t_arr = np.linspace(0, 2.0, n)
    mx, my = np.meshgrid(np.linspace(-900.0, 900.0, 3), np.linspace(-600.0, 600.0, 3))
    mloc = np.column_stack([mx.ravel(), my.ravel(), np.zeros(9)])
    data = np.zeros((n, 9, 3)); z0, t_hit, g = 500.0, 0.32, 9810.0
    for f, t in enumerate(t_arr):
        z = max(0.0, z0 - 0.5 * g * t**2 if t < t_hit else 50.0 * np.exp(-3.0 * (t-t_hit)) * np.abs(np.sin(10.0 * (t-t_hit))))
        ang = np.deg2rad(15.0 if t < t_hit else 15.0 * np.exp(-10.0 * (t-t_hit)))
        ry = np.array([[np.cos(ang), 0, np.sin(ang)], [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]])
        ripple = 80.0 * np.exp(-4.0 * max(0, t - t_hit)) * np.sin(25.0 * max(0, t - t_hit)) * (mx.ravel() / 900.0 * my.ravel() / 600.0)
        data[f] = mloc @ ry.T + [0, 0, z]; data[f, :, 2] += ripple
    return data, t_arr

if __name__ == "__main__":
    target_pkl = r"C:\Users\GOODMAN\WHToolsBox\TVPackageMotionSim\rds-20260414_044556\simulation_result.pkl"
    target_part = "Opencell_Rear"
    if os.path.exists(target_pkl):
        test_face_from_simulation(target_pkl, part_name=target_part)
    else:
        markers, times = create_example_markers(n=100)
        analyzer = ShellDeformationAnalyzer(markers, PlateConfig(thickness=5.0, youngs_modulus=100.0, mesh_resolution=20), times=times)
        analyzer.run_analysis(); analyzer.show(ground_size=(3000.0, 3000.0))