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
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple, Optional
import matplotlib.font_manager as fm

# --- JAX Configuration ---
jax.config.update("jax_enable_x64", True)

def scale_result_to_mm(result: Any):
    fields_to_scale = ['pos_hist', 'cog_pos_hist', 'geo_center_pos_hist', 'corner_pos_hist', 'z_hist', 'vel_hist', 'cog_vel_hist', 'geo_center_vel_hist']
    for field in fields_to_scale:
        if hasattr(result, field) and getattr(result, field) is not None:
            setattr(result, field, np.array(getattr(result, field)) * 1000.0)
    if hasattr(result, 'block_half_extents'):
        for bid in result.block_half_extents:
            result.block_half_extents[bid] = [v * 1000.0 for v in result.block_half_extents[bid]]
    if hasattr(result, 'nominal_local_pos'):
        for bid in result.nominal_local_pos:
            result.nominal_local_pos[bid] = [v * 1000.0 for v in result.nominal_local_pos[bid]]
    return result

@dataclass
class PlotSlotConfig:
    part_idx: int = 0
    plot_type: str = 'contour'
    data_key: str = 'Displacement [mm]'

@dataclass
class DashboardConfig:
    layout_2d: str = '2x2'
    plots_2d: List[PlotSlotConfig] = None
    v_font_size: int = 9
    animation_step: int = 1
    animation_speed_ms: int = 30

def create_default_config():
    return DashboardConfig(plots_2d=[])

@dataclass
class PlateConfig:
    thickness: float = 2.0
    youngs_modulus: float = 2.1e5
    poisson_ratio: float = 0.3
    poly_degree: int = 4
    reg_lambda: float = 1e-4
    mesh_resolution: int = 25
    batch_size: int = 256
    theory_type: str = "KIRCHHOFF"
    shear_correction: float = 5./6.

# --- Physics Engines (JAX-SSR) ---

class AlignmentManager:
    def __init__(self, raw_markers, W, H, offsets):
        self.raw_data = jnp.array(raw_markers); self.W, self.H = W, H; self.offsets = jnp.array(offsets); self.n_frames, self.n_markers, _ = self.raw_data.shape; self._calibrate()
    def _calibrate(self):
        P0 = self.raw_data[0]; P_target = jnp.column_stack([self.offsets, jnp.zeros(self.n_markers)]); c_P, c_T = jnp.mean(P0, axis=0), jnp.mean(P_target, axis=0); H_mat = (P_target - c_T).T @ (P0 - c_P); U, S, Vt = jnp.linalg.svd(H_mat); R = Vt.T @ U.T
        if jnp.linalg.det(R) < 0: R = (Vt.T @ jnp.diag(jnp.array([1.0, 1.0, -1.0]))) @ U.T
        self.local_axes = R.T; self.centroid_0 = c_P; self.x_bounds = [-self.W/2 - 5, self.W/2 + 5]; self.y_bounds = [-self.H/2 - 5, self.H/2 + 5]
    @partial(jit, static_argnums=(0,))
    def extract_kinematics_vmap(self, frame_markers):
        P_ref = jnp.column_stack([self.offsets, jnp.zeros(self.n_markers)]); c_P = jnp.mean(P_ref, axis=0)
        def kabsch_single(Q):
            c_Q = jnp.mean(Q, axis=0); H = (Q - c_Q).T @ (P_ref - c_P); U, S, Vt = jnp.linalg.svd(H); R = Vt.T @ U.T
            R_corr = jnp.where(jnp.linalg.det(R) < 0, (Vt.T @ jnp.diag(jnp.array([1.0, 1.0, -1.0]))) @ U.T, R)
            ql = (Q - c_Q) @ R_corr.T + c_P; return ql, R_corr, c_Q, c_P
        return vmap(kabsch_single)(frame_markers)

class AdvancedPlateOptimizer:
    def __init__(self, degree=5): self.basis_indices = [(i, j) for i in range(degree + 1) for j in range(degree + 1 - i)]; self.n = len(self.basis_indices)
    def get_basis_matrix(self, x, y): return jnp.stack([x**i * y**j for i, j in self.basis_indices], axis=1)
    def get_hessian_basis(self, x, y):
        def h_val(xi, yi): return jnp.stack([jnp.array([i*(i-1)*xi**(jnp.maximum(0,i-2))*yi**j if i>=2 else 0., j*(j-1)*xi**i*yi**(jnp.maximum(0,j-2)) if j>=2 else 0., i*j*xi**(jnp.maximum(0,i-1))*yi**(jnp.maximum(0,j-1)) if i>=1 and j>=1 else 0.]) for i, j in self.basis_indices], axis=0)
        return vmap(h_val)(x, y)
    @partial(jit, static_argnums=(0,))
    def solve_analytical(self, q_loc, p_ref, reg_lambda: float):
        Z, X = q_loc[:, :, 2], self.get_basis_matrix(p_ref[:, 0], p_ref[:, 1]); H = self.get_hessian_basis(p_ref[:, 0], p_ref[:, 1]); Bxx, Byy, Bxy = H[:, :, 0], H[:, :, 1], H[:, :, 2]
        M_bend = (Bxx.T @ Bxx + Byy.T @ Byy + 2.0*Bxy.T @ Bxy) / X.shape[0]; M = (X.T @ X) / X.shape[0] + reg_lambda * M_bend; M += jnp.eye(self.n) * 1e-12
        @vmap
        def solve_with_error(z):
            p = solve(M, (X.T @ z) / X.shape[0])
            z_fit = X @ p
            rmse = jnp.sqrt(jnp.mean((z - z_fit)**2))
            return p, rmse
        return solve_with_error(Z)

class PlateMechanicsSolver:
    def __init__(self, config: PlateConfig): self.cfg = config; self.D = (config.youngs_modulus * config.thickness**3) / (12.0 * (1.0 - config.poisson_ratio**2)); self.res = config.mesh_resolution; self.opt = AdvancedPlateOptimizer(degree=config.poly_degree)
    def setup_mesh(self, x_bounds, y_bounds): self.x_lin, self.y_lin = jnp.linspace(x_bounds[0], x_bounds[1], self.res), jnp.linspace(y_bounds[0], y_bounds[1], self.res); self.X_mesh, self.Y_mesh = jnp.meshgrid(self.x_lin, self.y_lin)
    @partial(jit, static_argnums=(0,))
    def evaluate_batch(self, p_coeffs_batch):
        Xf, Yf = self.X_mesh.ravel(), self.Y_mesh.ravel(); Xm, Hm = self.opt.get_basis_matrix(Xf, Yf), self.opt.get_hessian_basis(Xf, Yf)
        def triple_h_val(xi, yi): return jnp.stack([jnp.array([i*(i-1)*(i-2)*xi**(jnp.maximum(0,i-3))*yi**j if i>=3 else 0., j*(j-1)*(j-2)*xi**i*yi**(jnp.maximum(0,j-3)) if j>=3 else 0., i*(i-1)*j*xi**(jnp.maximum(0,i-2))*yi**(jnp.maximum(0,j-1)) if i>=2 and j>=1 else 0., i*j*(j-1)*xi**(jnp.maximum(0,i-1))*yi**(jnp.maximum(0,j-2)) if i>=1 and j>=2 else 0.]) for i, j in self.opt.basis_indices], axis=0)
        Tm = vmap(triple_h_val)(Xf, Yf)
        def gradient_basis_val(xi, yi): return jnp.stack([jnp.array([i*xi**(jnp.maximum(0,i-1))*yi**j if i>=1 else 0., j*xi**i*yi**(jnp.maximum(0,j-1)) if j>=1 else 0.]) for i, j in self.opt.basis_indices], axis=0)
        Gm = vmap(gradient_basis_val)(Xf, Yf)
        @vmap
        def eval_fn(p):
            w = (Xm @ p).reshape(self.res, self.res); k_raw = -jnp.einsum('nkd,k->nd', Hm, p).reshape(self.res, self.res, 3); kxx, kyy, kxy = k_raw[..., 0], k_raw[..., 1], k_raw[..., 2]; s_c = 6.0 * self.D / (self.cfg.thickness**2)
            sx_b, sy_b, txy_b = s_c*(kxx + self.cfg.poisson_ratio*kyy), s_c*(kyy + self.cfg.poisson_ratio*kxx), s_c*(1.0-self.cfg.poisson_ratio)*kxy; sx_m, sy_m, txz, tyz = 0., 0., 0., 0.
            if self.cfg.theory_type == "MINDLIN": t_vals = jnp.einsum('nkd,k->nd', Tm, p).reshape(self.res, self.res, 4); Vx, Vy = -self.D * (t_vals[..., 0] + t_vals[..., 3]), -self.D * (t_vals[..., 1] + t_vals[..., 2]); txz, tyz = Vx / (self.cfg.thickness * self.cfg.shear_correction), Vy / (self.cfg.thickness * self.cfg.shear_correction)
            elif self.cfg.theory_type == "VON_KARMAN": g_vals = jnp.einsum('nkd,k->nd', Gm, p).reshape(self.res, self.res, 2); dw_dx, dw_dy = g_vals[..., 0], g_vals[..., 1]; E_mod = self.cfg.youngs_modulus / (1.0 - self.cfg.poisson_ratio**2); sx_m, sy_m = E_mod * (0.5*dw_dx**2 + self.cfg.poisson_ratio*0.5*dw_dy**2), E_mod * (0.5*dw_dy**2 + self.cfg.poisson_ratio*0.5*dw_dx**2)
            fields = {'Displacement [mm]': w, 'Stress XX [MPa]': sx_b+sx_m, 'Stress YY [MPa]': sy_b+sy_m, 'Shear Stress XY [MPa]': txy_b, 'Von-Mises [MPa]': jnp.sqrt(jnp.maximum((sx_b+sx_m)**2 + (sy_b+sy_m)**2 - (sx_b+sx_m)*(sy_b+sy_m) + 3.0*(txy_b**2 + txz**2 + tyz**2), 1e-12))}
            return {**fields, **{f'Mean-{k}': jnp.mean(v) for k,v in fields.items()}, **{f'Max-{k}': jnp.max(v) for k,v in fields.items()}}
        return eval_fn(p_coeffs_batch)

class ShellDeformationAnalyzer:
    def __init__(self, name, markers, W, H, offsets, config: PlateConfig, times):
        self.name, self.m_raw, self.W, self.H, self.offsets, self.cfg, self.times = name, jnp.array(markers), W, H, offsets, config, times; m_count = self.m_raw.shape[1]; max_safe_d = 2
        while (max_safe_d + 2) * (max_safe_d + 3) / 2 <= m_count * 0.85: max_safe_d += 1
        self.cfg.poly_degree = min(config.poly_degree, max_safe_d); self.align = AlignmentManager(markers, W, H, offsets); self.sol = PlateMechanicsSolver(self.cfg); self.results = {}
    def run_analysis(self):
        self.sol.setup_mesh(self.align.x_bounds, self.align.y_bounds); p_ref = jnp.column_stack([self.align.offsets, jnp.zeros(self.align.n_markers)]); n_frames, bs = self.align.n_frames, max(self.cfg.batch_size, 2048); buf = None
        for i in range((n_frames + bs - 1) // bs):
            idx_s, idx_e = i * bs, min((i+1)*bs, n_frames); q_loc, rot, cq, cp = self.align.extract_kinematics_vmap(self.m_raw[idx_s:idx_e]); params, rmses = self.sol.opt.solve_analytical(q_loc, p_ref, self.cfg.reg_lambda); batch = self.sol.evaluate_batch(params)
            if buf is None: buf = {k: [] for k in list(batch.keys()) + ['R', 'c_Q', 'c_P', 'Q_local', 'rmse']}
            for k,v in batch.items(): buf[k].append(np.array(v))
            buf['R'].append(np.array(rot)); buf['c_Q'].append(np.array(cq)); buf['c_P'].append(np.array(cp)); buf['Q_local'].append(np.array(q_loc)); buf['rmse'].append(np.array(rmses))
        self.results = {k: np.concatenate(v, axis=0) for k, v in buf.items()}; self.results['Marker Global Disp. [mm]'] = np.linalg.norm(np.array(self.m_raw) - np.array(self.m_raw[0]), axis=2); self.results['Marker Local Disp. [mm]'] = np.array(self.results['Q_local'][:, :, 2])
        self.avg_rmse = np.mean(self.results['rmse']); print(f"  > [PART] {self.name:<25} analyzed. (Avg RMSE: {self.avg_rmse:.4e} mm)")

class PlateAssemblyManager:
    def __init__(self, times): self.analyzers, self.times, self.n_frames = [], times, len(times)
    def add_analyzer(self, analyzer): self.analyzers.append(analyzer); return analyzer
    def run_all(self):
        print(f"[WHTOOLS] Assembly Analysis Started ({len(self.analyzers)} parts)...")
        with ThreadPoolExecutor(max_workers=8) as ex: list(ex.map(lambda p: p.run_analysis(), self.analyzers))
        print(f"[WHTOOLS] All {len(self.analyzers)} Parts Analyzed Successfully.")

# --- UI Components (Premium Polish) ---

class VisibilityToolWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent, QtCore.Qt.Window | QtCore.Qt.WindowStaysOnTopHint); self.setWindowTitle("Visibility Manager"); self.resize(350, 500); self.parent = parent
        l = QtWidgets.QVBoxLayout(self); g1 = QtWidgets.QGroupBox("Global Control"); l1 = QtWidgets.QVBoxLayout(g1)
        for t, c in [("Mesh", 1), ("Markers", 2)]:
            h = QtWidgets.QHBoxLayout(); h.addWidget(QtWidgets.QLabel(f"{t}:")); b1, b2 = QtWidgets.QPushButton("Show All"), QtWidgets.QPushButton("Hide All"); b1.clicked.connect(partial(self._bulk_set, c, True)); b2.clicked.connect(partial(self._bulk_set, c, False)); h.addWidget(b1); h.addWidget(b2); l1.addLayout(h)
        l.addWidget(g1); self.tree = QtWidgets.QTreeWidget(); self.tree.setHeaderLabels(["Part", "Mesh", "Markers"]); self.tree.itemChanged.connect(self._on_item_changed); l.addWidget(self.tree); self.groups, self.id_to_item = {}, {}; self._init_tree()
    def _init_tree(self):
        self.tree.blockSignals(True)
        for i, p in enumerate(self.parent.mgr.analyzers):
            pre = p.name.split('_')[0]
            if pre not in self.groups: self.groups[pre] = QtWidgets.QTreeWidgetItem(self.tree, [pre]); self.groups[pre].setExpanded(True)
            it = QtWidgets.QTreeWidgetItem(self.groups[pre], [p.name]); it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable); it.setData(0, QtCore.Qt.UserRole, i); it.setCheckState(1, QtCore.Qt.Checked if self.parent.part_actors[i]['visible'] else QtCore.Qt.Unchecked); it.setCheckState(2, QtCore.Qt.Checked if self.parent.part_actors[i]['visible_markers'] else QtCore.Qt.Unchecked); self.id_to_item[i] = it
        self.tree.blockSignals(False)
    def _bulk_set(self, col, state):
        self.tree.blockSignals(True); st = QtCore.Qt.Checked if state else QtCore.Qt.Unchecked
        for i in range(self.tree.topLevelItemCount()):
            g = self.tree.topLevelItem(i); g.setCheckState(col, st)
            for j in range(g.childCount()): g.child(j).setCheckState(col, st)
        self.tree.blockSignals(False); self._apply()
    def _on_item_changed(self, item, col):
        if item.data(0, QtCore.Qt.UserRole) is None:
            self.tree.blockSignals(True); [item.child(j).setCheckState(col, item.checkState(col)) for j in range(item.childCount())]; self.tree.blockSignals(False)
        self._apply()
    def _apply(self):
        for i, it in self.id_to_item.items(): self.parent.part_actors[i]['visible'] = (it.checkState(1) == QtCore.Qt.Checked); self.parent.part_actors[i]['visible_markers'] = (it.checkState(2) == QtCore.Qt.Checked)
        self.parent.update_frame(self.parent.current_frame)

class AddPlotDialog(QtWidgets.QDialog):
    def __init__(self, slot_idx, parts, field_keys, stat_keys, parent=None):
        super().__init__(parent); self.setWindowTitle(f"Add Plot to Slot {slot_idx + 1}"); l = QtWidgets.QVBoxLayout(self); gl = QtWidgets.QGridLayout(); l.addLayout(gl)
        gl.addWidget(QtWidgets.QLabel("Part:"), 0, 0); self.cmb_part = QtWidgets.QComboBox(); self.cmb_part.addItems(parts); gl.addWidget(self.cmb_part, 0, 1)
        gl.addWidget(QtWidgets.QLabel("Type:"), 1, 0); hb = QtWidgets.QHBoxLayout(); self.rb_contour = QtWidgets.QRadioButton("Contour"); self.rb_curve = QtWidgets.QRadioButton("Curve"); self.rb_contour.setChecked(True); hb.addWidget(self.rb_contour); hb.addWidget(self.rb_curve); gl.addLayout(hb, 1, 1)
        gl.addWidget(QtWidgets.QLabel("Key:"), 2, 0); self.cmb_key = QtWidgets.QComboBox(); gl.addWidget(self.cmb_key, 2, 1); self.f_keys, self.s_keys = field_keys, stat_keys; self.rb_contour.toggled.connect(self._update_keys); self.rb_curve.toggled.connect(self._update_keys); self._update_keys(); bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel); bb.accepted.connect(self.accept); bb.rejected.connect(self.reject); l.addWidget(bb)
    def _update_keys(self): self.cmb_key.clear(); self.cmb_key.addItems(self.f_keys if self.rb_contour.isChecked() else self.s_keys)
    def get_config(self): return PlotSlotConfig(part_idx=self.cmb_part.currentIndex(), plot_type="contour" if self.rb_contour.isChecked() else "curve", data_key=self.cmb_key.currentText())

class AboutDialog(QtWidgets.QDialog):
    def __init__(self, logo_path, parent=None):
        super().__init__(parent); self.setWindowTitle("About WHTOOLS Dashboard"); self.setFixedSize(500, 600); l = QtWidgets.QVBoxLayout(self); l.setContentsMargins(30, 30, 30, 30); l.setSpacing(15)
        if os.path.exists(logo_path):
            img = QtWidgets.QLabel(); pix = QtGui.QPixmap(logo_path).scaledToHeight(240, QtCore.Qt.SmoothTransformation); img.setPixmap(pix); img.setAlignment(QtCore.Qt.AlignCenter); l.addWidget(img)
        t = QtWidgets.QLabel("WHTOOLS Dashboard v5.9.3"); t.setStyleSheet("font-size: 18pt; font-weight: bold; color: #1A73E8;"); t.setAlignment(QtCore.Qt.AlignCenter); l.addWidget(t)
        v = QtWidgets.QLabel("Expert Structural Analysis & Digital Twin Solution"); v.setStyleSheet("font-size: 10pt; color: #666; font-style: italic;"); v.setAlignment(QtCore.Qt.AlignCenter); l.addWidget(v); line = QtWidgets.QFrame(); line.setFrameShape(QtWidgets.QFrame.HLine); line.setFrameShadow(QtWidgets.QFrame.Sunken); l.addWidget(line)
        f = QtWidgets.QLabel(
            "<b>Core Technologies:</b><br>"
            "• <b>Precision Shell Theory:</b> Kirchhoff / Mindlin / Von Karman<br>"
            "• <b>JAX SSR Engine:</b> High-performance JAX-powered reconstruction<br>"
            "• <b>Real-time Digital Twin:</b> 3D/2D sync pipeline for MuJoCo<br>"
            "• <b>Expert Metrics:</b> Stress, PBA, RRG, and Displacement Analysis<br>"
            "• <b>Premium Visualization:</b> High-fidelity VTK & Matplotlib grid"
        ); f.setStyleSheet("font-size: 11pt; line-height: 160%;"); f.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter); l.addWidget(f); l.addStretch()
        c = QtWidgets.QLabel("© 2026 WHTOOLS. All Rights Reserved."); c.setStyleSheet("font-size: 9pt; color: #999;"); c.setAlignment(QtCore.Qt.AlignCenter); l.addWidget(c)
        b = QtWidgets.QPushButton("Close"); b.setFixedWidth(100); b.setStyleSheet("padding: 8px; font-weight: bold;"); b.clicked.connect(self.accept); h = QtWidgets.QHBoxLayout(); h.addStretch(); h.addWidget(b); h.addStretch(); l.addLayout(h)

class QtVisualizerV2(QtWidgets.QMainWindow):
    def __init__(self, manager: PlateAssemblyManager, config: DashboardConfig = None, ground_size=(3000, 3000)):
        super().__init__(); self.mgr, self.cfg, self.ground_size = manager, config or create_default_config(), ground_size; self.current_frame, self.is_playing, self.active_slot = 0, False, 0; self.anim_step = self.cfg.animation_step; self.plot_slots: List[Optional[PlotSlotConfig]] = [None]*6; self.part_actors, self.v_font_size = {}, self.cfg.v_font_size
        p0 = manager.analyzers[0]; n_f = len(self.mgr.times); r_sq = p0.sol.res**2; self.field_keys = [k for k in p0.results if p0.results[k].ndim==3 and p0.results[k].size // n_f == r_sq]; self.stat_keys = [k for k in p0.results if k not in self.field_keys and p0.results[k].ndim < 3] + ['Marker Local Disp. [mm]', 'Marker Global Disp. [mm]']
        curr_dir = os.path.dirname(__file__)
        self.res_dir = os.path.join(curr_dir, "resources")
        self.logo_path = os.path.join(self.res_dir, "logo.png")
        self.font_path = os.path.join(self.res_dir, "D2Coding-Ver1.3.2-20180524-ligature.ttf")
        
        if os.path.exists(self.font_path):
            fid = QtGui.QFontDatabase.addApplicationFont(self.font_path)
            if fid != -1: fname = QtGui.QFontDatabase.applicationFontFamilies(fid)[0]; QtWidgets.QApplication.setFont(QtGui.QFont(fname, 9))
        self._init_ui(); self._init_3d_view(); self._init_2d_plots(); self.update_frame(0); self.visibility_tool = VisibilityToolWindow(self)

    def _init_ui(self):
        self.setWindowTitle("WHTOOLS Dashboard v5.9.3"); self.resize(1600, 950); self._init_menus(); cw = QtWidgets.QWidget(); self.setCentralWidget(cw); ml = QtWidgets.QVBoxLayout(cw); self._init_animation_toolbar()
        self.split = QtWidgets.QSplitter(QtCore.Qt.Horizontal); ml.addWidget(self.split); self.p3d, self.p2d = QtWidgets.QWidget(), QtWidgets.QWidget(); l3, l2 = QtWidgets.QVBoxLayout(self.p3d), QtWidgets.QVBoxLayout(self.p2d); self._init_3d_panel(l3); self._init_2d_panel(l2); self.split.addWidget(self.p3d); self.split.addWidget(self.p2d)

    def _init_animation_toolbar(self):
        tb = self.addToolBar("Animation Control"); tb.setFixedHeight(30)
        # Ordering: << < > >>, then Play button
        for t, s in [("<<", 0), ("<", -1), (">", 1), (">>", 9999)]:
            btn = QtWidgets.QPushButton(t); btn.setFixedSize(30, 26); btn.clicked.connect(partial(self._ctrl_slot, s)); tb.addWidget(btn)
        tb.addSeparator()
        self.btn_play = QtWidgets.QPushButton("▶"); self.btn_play.setFixedSize(40, 26); self.btn_play.clicked.connect(lambda: self._ctrl_slot(-2)); tb.addWidget(self.btn_play)
        tb.addSeparator()
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.slider.setRange(0, self.mgr.n_frames-1); self.slider.valueChanged.connect(self.update_frame); self.slider.setMinimumWidth(400); tb.addWidget(self.slider)
        tb.addSeparator(); self.lbl_f = QtWidgets.QLabel("Frame: 0"); self.lbl_f.setFixedHeight(26); tb.addWidget(self.lbl_f); tb.addSeparator()
        l_step = QtWidgets.QLabel(" Step: "); l_step.setFixedHeight(26); tb.addWidget(l_step)
        self.cmb_step = QtWidgets.QComboBox(); self.cmb_step.setFixedHeight(26); self.cmb_step.addItems([str(i) for i in range(1, 11)]); self.cmb_step.setCurrentText(str(self.anim_step)); self.cmb_step.currentTextChanged.connect(self._on_step_changed); tb.addWidget(self.cmb_step)
        l_speed = QtWidgets.QLabel(" Speed: "); l_speed.setFixedHeight(26); tb.addWidget(l_speed)
        self.cmb_speed = QtWidgets.QComboBox(); self.cmb_speed.setFixedHeight(26); self.cmb_speed.addItems(["15", "30", "50", "100", "200"]); self.cmb_speed.setCurrentText(str(self.cfg.animation_speed_ms)); self.cmb_speed.currentTextChanged.connect(self._on_speed_changed); tb.addWidget(self.cmb_speed)

    def _init_3d_panel(self, layout):
        # Header Layout: [Logo (100px)] | [3D View Control Box]
        header = QtWidgets.QHBoxLayout()
        if os.path.exists(self.logo_path):
            l_logo = QtWidgets.QLabel()
            pix = QtGui.QPixmap(self.logo_path).scaledToHeight(100, QtCore.Qt.SmoothTransformation)
            l_logo.setPixmap(pix); l_logo.setFixedHeight(100); header.addWidget(l_logo); header.addSpacing(15)
        
        gb = QtWidgets.QGroupBox("3D View Control"); gl = QtWidgets.QGridLayout(gb)
        self.cmb_view = self._create_combo("View:", ["Global", "Local"], None); gl.addWidget(QtWidgets.QLabel("View:"), 0, 0); gl.addWidget(self.cmb_view, 0, 1)
        self.cmb_3d = self._create_combo("Field:", ["Body Color", "Face Color"]+self.field_keys, None); gl.addWidget(QtWidgets.QLabel("Field:"), 0, 2); gl.addWidget(self.cmb_3d, 0, 3)
        self.cmb_theory = self._create_combo("Theory:", ["KIRCHHOFF", "MINDLIN", "VON_KARMAN"], None); gl.addWidget(QtWidgets.QLabel("Theory:"), 0, 4); gl.addWidget(self.cmb_theory, 0, 5); self.cmb_theory.currentTextChanged.connect(self._on_theory_changed)
        
        self.spin_scale = QtWidgets.QDoubleSpinBox(); self.spin_scale.setRange(0.1, 1000.); self.spin_scale.setValue(1.0); self.spin_scale.valueChanged.connect(lambda: self.update_frame(self.current_frame)); gl.addWidget(QtWidgets.QLabel("Scale:"), 1, 0); gl.addWidget(self.spin_scale, 1, 1)
        self.cmb_leg = self._create_combo("Legend:", ["Dynamic", "Static"], None); gl.addWidget(QtWidgets.QLabel("Legend:"), 1, 2); gl.addWidget(self.cmb_leg, 1, 3); self.cmb_leg.currentTextChanged.connect(self._on_legend_mode_changed)
        
        self.spin_min, self.spin_max = QtWidgets.QDoubleSpinBox(), QtWidgets.QDoubleSpinBox()
        for s in [self.spin_min, self.spin_max]: s.setRange(-1e12, 1e12); s.setDecimals(4); s.valueChanged.connect(lambda: self.update_frame(self.current_frame))
        gl.addWidget(QtWidgets.QLabel("Range:"), 1, 4); rb = QtWidgets.QHBoxLayout(); rb.addWidget(self.spin_min); rb.addWidget(self.spin_max); gl.addLayout(rb, 1, 5)
        
        header.addWidget(gb, stretch=1); layout.addLayout(header)
        self.v_int = QtInteractor(self); self.v_int.setContextMenuPolicy(QtCore.Qt.CustomContextMenu); self.v_int.customContextMenuRequested.connect(self._show_part_menu); layout.addWidget(self.v_int, stretch=1)

    def _init_2d_panel(self, layout):
        gb = QtWidgets.QGroupBox("2D Plot Control"); r = QtWidgets.QHBoxLayout(gb)
        self.cmb_layout = QtWidgets.QComboBox(); self.cmb_layout.addItems(["1x1", "1x2", "2x2", "3x2"]); self.cmb_layout.setCurrentText(self.cfg.layout_2d); self.cmb_layout.currentTextChanged.connect(self._init_2d_plots); r.addWidget(QtWidgets.QLabel("Layout:")); r.addWidget(self.cmb_layout)
        for t, f in [("+ Plot", self._show_add_plot_dialog), ("Pop-out", self._pop_out_2d)]:
            btn = QtWidgets.QPushButton(t); btn.clicked.connect(f); r.addWidget(btn)
        self.checks = {}
        for t, s in [("Sync", True), ("Interp", True)]:
            ck = QtWidgets.QCheckBox(t); ck.setChecked(s); ck.toggled.connect(lambda: self.update_frame(self.current_frame)); r.addWidget(ck); self.checks[t] = ck
        layout.addWidget(gb); self._canv_w = QtWidgets.QWidget(); self._canv_l = QtWidgets.QVBoxLayout(self._canv_w); self._canv_l.setContentsMargins(0,0,0,0); layout.addWidget(self._canv_w, stretch=1)

    def _init_3d_view(self):
        self.v_int.set_background("white"); self.v_int.add_axes(); pv.global_theme.font.size, pv.global_theme.font.color = 12, 'black'
        fp = pv.Plane(i_size=self.ground_size[0], j_size=self.ground_size[1]); self.ground = self.v_int.add_mesh(fp, color="blue", opacity=0.1)
        self.lut = pv.LookupTable(cmap="turbo"); self.lut.below_range_color, self.lut.above_range_color = 'lightgrey', 'magenta'
        for i, p in enumerate(self.mgr.analyzers):
            poly = pv.Plane(i_size=p.W, j_size=p.H, i_resolution=p.sol.res-1, j_resolution=p.sol.res-1); ma = self.v_int.add_mesh(poly, scalars=np.zeros(p.sol.res**2), cmap=self.lut, show_edges=True, edge_color="dimgray", show_scalar_bar=False)
            mp = pv.PolyData(np.array(p.m_raw[0])); mp.point_data["names"] = [f"{p.name}_M{j:02d}" for j in range(p.align.n_markers)]; mk = self.v_int.add_mesh(mp, render_points_as_spheres=True, point_size=10, color='blue'); lb = self.v_int.add_point_labels(mp, "names", font_size=self.v_font_size, text_color='black', always_visible=True, point_size=0, shadow=False)
            self.part_actors[i] = {'mesh': ma, 'poly': poly, 'm_poly': mp, 'markers': mk, 'labels': lb, 'visible': True, 'visible_markers': False, 'p_base': np.column_stack([p.sol.X_mesh.ravel(), p.sol.Y_mesh.ravel(), np.zeros(p.sol.res**2)])}
        self.sb = self.v_int.add_scalar_bar("Field", position_x=0.15, position_y=0.05, width=0.7, mapper=self.part_actors[0]['mesh'].mapper, vertical=False, n_labels=5, fmt="%.2e")
        if self.font_path:
            for pr in [self.sb.GetLabelTextProperty(), self.sb.GetTitleTextProperty()]: pr.SetFontFile(self.font_path); pr.SetFontSize(self.v_font_size + 1); pr.SetColor(0,0,0); pr.BoldOn()
            try:
                ann = self.sb.GetAnnotationTextProperty()
                ann.SetFontFile(self.font_path); ann.SetFontSize(self.v_font_size + 1); ann.SetColor(0,0,0)
            except: pass
        self.v_int.view_isometric(); self.v_int.enable_parallel_projection(); self.timer = QtCore.QTimer(); self.timer.timeout.connect(lambda: self._ctrl_slot(1))

    def _init_2d_plots(self):
        for i in reversed(range(self._canv_l.count())): self._canv_l.itemAt(i).widget().setParent(None)
        self.canv = FigureCanvas(Figure(figsize=(8,8))); self._canv_l.addWidget(NavigationToolbar(self.canv, self)); self._canv_l.addWidget(self.canv)
        self.canv.mpl_connect('button_press_event', self._on_axis_clicked); m = {"1x1": (1,1), "1x2": (1,2), "2x2": (2,2), "3x2": (3,2)}; r, c = m.get(self.cmb_layout.currentText(), (2,2))
        self.axes, self.ims, self.vline = [], [None]*6, [None]*6; self.canv.figure.clear(); self.canv.figure.subplots_adjust(hspace=0.4, wspace=0.3)
        for i in range(r*c):
            ax = self.canv.figure.add_subplot(r, c, i+1); self.axes.append(ax)
            if self.plot_slots[i]: ax.set_title(f"Slot {i+1}: {self.plot_slots[i].data_key}")
            else: ax.text(0.5, 0.5, f"Empty Slot {i+1}\nClick to add", ha='center', transform=ax.transAxes)
        self._update_selection_ui(); self.canv.draw_idle()

    def update_frame(self, f):
        self.current_frame = f; self.slider.blockSignals(True); self.slider.setValue(f); self.slider.blockSignals(False); self.lbl_f.setText(f"Frame: {f} | Time: {self.mgr.times[f]:.3f}s")
        view, fk, scale = self.cmb_view.currentText(), self.cmb_3d.currentText(), self.spin_scale.value(); is_dyn, all_vals = self.cmb_leg.currentText()=="Dynamic", []
        for i, p in enumerate(self.mgr.analyzers):
            info = self.part_actors[i]; vis = info['visible']; m_vis = vis and info['visible_markers']; info['mesh'].SetVisibility(vis); info['markers'].SetVisibility(m_vis); info['labels'].SetVisibility(m_vis)
            if not vis: continue
            fw = p.results['Displacement [mm]'][f]; p_def = info['p_base'].copy(); p_def[:, 2] = fw.ravel() * scale; R, cq, cp = p.results['R'][f], p.results['c_Q'][f], p.results['c_P'][f]
            if view == "Global": info['poly'].points = (p_def - cp) @ R + cq; info['m_poly'].points = np.array(p.m_raw[f])
            else: info['poly'].points = p_def; info['m_poly'].points = np.array(p.results['Q_local'][f])
            if fk in ["Body Color", "Face Color"]: info['mesh'].mapper.scalar_visibility = False; info['mesh'].GetProperty().SetColor(plt.cm.tab20(i%20)[:3])
            else:
                info['mesh'].mapper.scalar_visibility = True; key = fk if fk in p.results else 'Displacement [mm]'; fv = p.results[key][f]
                if fv.size == p.sol.res**2: info['poly'].point_data["S"] = fv.ravel(); all_vals.append(fv)
            info['poly'].Modified(); info['m_poly'].Modified()
        if all_vals and fk not in ["Body Color", "Face Color"]:
            comb = np.concatenate([v.ravel() for v in all_vals]); v_min, v_max = float(comb.min()), float(comb.max()); clim = [v_min, v_max] if is_dyn else [float(self.spin_min.value()), float(self.spin_max.value())]
            if clim[0]>=clim[1]: clim[1]=clim[0]+1e-6
            self.lut.scalar_range=(clim[0], clim[1]); self.sb.SetVisibility(True); self.sb.title = fk; [a['mesh'].mapper.set_scalar_range(clim[0], clim[1]) for a in self.part_actors.values()]
            self.v_int.add_text(f"[{fk}]\nMin: {v_min:.3e}\nMax: {v_max:.3e}", position='upper_left', font_size=self.v_font_size, color='black', name='stat_overlay')
        else: self.sb.SetVisibility(False); self.v_int.add_text("", position='upper_left', name='stat_overlay')
        self._update_2d_plots(f); self.v_int.render()

    def _update_2d_plots(self, f):
        t = self.mgr.times[f]; interp = self.checks['Interp'].isChecked()
        for i, ax in enumerate(self.axes):
            cfg = self.plot_slots[i]
            if not cfg: continue
            ana = self.mgr.analyzers[cfg.part_idx]; key = cfg.data_key
            if cfg.plot_type == "contour":
                d2 = ana.results[key][f]
                if self.ims[i] is None: ax.clear(); self.ims[i] = ax.imshow(d2, cmap='turbo', origin='lower'); self.canv.figure.colorbar(self.ims[i], ax=ax, format="%.2e")
                self.ims[i].set_data(d2); self.ims[i].set_interpolation('bilinear' if interp else 'nearest'); ax.set_title(f"[{ana.name}] {key}")
            else:
                if self.vline[i] is None:
                    ax.clear(); ax.grid(True, alpha=0.3); vals = ana.results[key] if key in ana.results else ana.results['Marker Local Disp. [mm]']
                    if vals.ndim == 1: ax.plot(self.mgr.times, vals, color='#1A73E8')
                    else: [ax.plot(self.mgr.times, vals[:, m], alpha=0.5) for m in range(min(vals.shape[1], 10))]
                    self.vline[i] = ax.axvline(t, color='red', ls='--'); ax.set_ylabel(key); ax.set_xlabel("Time [s]")
                self.vline[i].set_xdata([t]); ax.set_title(f"[{ana.name}] {key}")
        self.canv.draw_idle()

    def _pop_out_2d(self):
        self.pop_win = QtWidgets.QMainWindow(self); self.pop_win.resize(1000, 800); c = QtWidgets.QWidget(); self.pop_win.setCentralWidget(c); l = QtWidgets.QVBoxLayout(c); self.pop_fig = Figure(figsize=(10,10)); self.pop_canv = FigureCanvas(self.pop_fig); l.addWidget(self.pop_canv); m = {"1x1": (1,1), "1x2": (1,2), "2x2": (2,2), "3x2": (3,2)}; r, cols = m.get(self.cmb_layout.currentText(), (2,2)); self.pop_axes, self.pop_ims, self.pop_vlines = [], [None]*6, [None]*6
        for i in range(r*cols):
            ax = self.pop_fig.add_subplot(r, cols, i+1); self.pop_axes.append(ax); cfg = self.plot_slots[i]
            if cfg:
                ana = self.mgr.analyzers[cfg.part_idx]; key = cfg.data_key
                if cfg.plot_type == "contour": im = ax.imshow(ana.results[key][self.current_frame], cmap='turbo', origin='lower'); self.pop_ims[i] = im; self.pop_fig.colorbar(im, ax=ax)
                else: 
                    vals = ana.results[key] if key in ana.results else ana.results['Marker Local Disp. [mm]']
                    if vals.ndim==1: ax.plot(self.mgr.times, vals)
                    else: [ax.plot(self.mgr.times, vals[:, m], alpha=0.5) for m in range(min(vals.shape[1], 10))]
                    self.pop_vlines[i] = ax.axvline(self.mgr.times[self.current_frame], color='red')
            else: ax.text(0.5,0.5,"Empty", ha='center', transform=ax.transAxes)
        self.pop_canv.draw(); self.pop_win.show()

    def _init_menus(self): mb = self.menuBar(); sm = mb.addMenu("Settings"); sm.addAction("Visibility Manager", lambda: self.visibility_tool.show()); sm.addAction("Reset Camera (f)", lambda: self.v_int.reset_camera()); hm = mb.addMenu("Help"); hm.addAction("About", self._show_about)
    def _show_about(self):
        dlg = AboutDialog(self.logo_path, self); dlg.exec()
    def _on_step_changed(self, v): self.anim_step = int(v)
    def _on_speed_changed(self, v): self.timer.setInterval(int(v))
    def _on_theory_changed(self, t):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            for p in self.mgr.analyzers: p.cfg.theory_type = t; p.run_analysis()
            p0 = self.mgr.analyzers[0]; n_frames = len(self.mgr.times); res_sq = p0.sol.res**2; fkeys = [k for k in p0.results if p0.results[k].ndim==3 and p0.results[k].size // n_frames == res_sq]; self.cmb_3d.blockSignals(True); self.cmb_3d.clear(); self.cmb_3d.addItems(["Body Color", "Face Color"] + fkeys); self.cmb_3d.blockSignals(False); self.update_frame(self.current_frame)
        finally: QtWidgets.QApplication.restoreOverrideCursor()
    def _on_legend_mode_changed(self, m):
        if m == "Static": self.spin_min.setValue(0.0); self.spin_max.setValue(0.01)
        self.update_frame(self.current_frame)
    def _show_part_menu(self, pos=None):
        if pos is None: pos = self.v_int.mapFromGlobal(QtGui.QCursor.pos())
        menu = QtWidgets.QMenu(self); menu.addAction("Visibility Manager", self.visibility_tool.show); menu.addSeparator(); [menu.addAction(t, f) for t, f in [("XY Plane (+Z)", self.v_int.view_xy), ("YZ Plane (+X)", self.v_int.view_yz), ("ZX Plane (+Y)", self.v_int.view_zx), ("Isometric View", self.v_int.view_isometric)]]; menu.addSeparator()
        af = menu.addAction("Floor Visibility"); af.setCheckable(True); af.setChecked(self.ground.GetVisibility())
        am = menu.addAction("Show Mesh Edges"); am.setCheckable(True); am.setChecked(self.part_actors[0]['mesh'].GetProperty().GetEdgeVisibility() if self.part_actors else True)
        act = menu.exec_(self.v_int.mapToGlobal(pos))
        if act == af: self.ground.SetVisibility(act.isChecked()); self.v_int.render()
        elif act == am:
            v = am.isChecked()
            for a in self.part_actors.values(): a['mesh'].GetProperty().SetEdgeVisibility(v)
            self.v_int.render()
    def _create_combo(self, label, items, layout): cb = QtWidgets.QComboBox(); cb.addItems(items); cb.currentIndexChanged.connect(lambda: self.update_frame(self.current_frame)); [layout.addWidget(QtWidgets.QLabel(label)), layout.addWidget(cb)] if layout else None; return cb
    def _on_axis_clicked(self, event):
        if event.inaxes:
            for i, ax in enumerate(self.axes):
                if event.inaxes == ax: self.active_slot = i; break
            self._update_selection_ui(); self.statusBar().showMessage(f"Active Slot: {self.active_slot+1}")
    def _update_selection_ui(self):
        for i, ax in enumerate(self.axes): color, lw = ("#1A73E8", 2) if i == self.active_slot else ("lightgrey", 0.5); [s.set_color(color) for s in ax.spines.values()]; [s.set_linewidth(lw) for s in ax.spines.values()]; self.canv.draw_idle()
    def _show_add_plot_dialog(self):
        pnames = [p.name for p in self.mgr.analyzers]; dlg = AddPlotDialog(self.active_slot, pnames, self.field_keys, self.stat_keys, self)
        if dlg.exec(): self.plot_slots[self.active_slot] = dlg.get_config(); self.ims[self.active_slot] = self.vline[self.active_slot] = None; self.update_frame(self.current_frame)
    def _ctrl_slot(self, s):
        if s == -2:
            if self.is_playing: self.timer.stop(); self.btn_play.setText("▶")
            else: self.timer.start(int(self.cmb_speed.currentText())); self.btn_play.setText("⏸")
            self.is_playing = not self.is_playing
        elif s == 0: self.update_frame(0)
        elif s == 9999: self.update_frame(self.mgr.n_frames-1)
        else: self.update_frame(max(0, min(self.mgr.n_frames-1, self.current_frame+s*self.anim_step)))

def create_cube_markers(n=100):
    t_arr = np.linspace(0, 2.5, n); L, W, H = 2000., 1200., 800.; f_configs = {"Bottom": [0,0,-H/2], "Top": [0,0,H/2], "Front": [0,-W/2,0], "Back": [0,W/2,0], "Left": [-L/2,0,0], "Right": [L/2,0,0]}; data = {}
    for name, pos in f_configs.items():
        fw, fh = (L,W) if "Top" in name or "Bottom" in name else ((L,H) if "Front" in name or "Back" in name else (W,H)); mx, my = np.meshgrid(np.linspace(-fw*0.4, fw*0.4, 3), np.linspace(-fh*0.4, fw*0.4, 3)); off = np.column_stack([mx.ravel(), my.ravel()]); markers = np.zeros((n, 9, 3))
        for f, t in enumerate(t_arr):
            z_c = 1000 - 9810*0.5*t**2 if t<0.45 else 100*np.exp(-3*(t-0.45))
            markers[f] = np.column_stack([off, 10*np.sin(10*t)*np.cos(off[:,0]/100)]) + pos + [0,0,max(z_c, 0)]
        data[name] = (markers, off, fw, fh)
    return data, t_arr

if __name__ == "__main__":
    raw_data, times = create_cube_markers(100); manager = PlateAssemblyManager(times)
    for name, (m, off, fw, fh) in raw_data.items(): manager.add_analyzer(ShellDeformationAnalyzer(name, m, fw, fh, off, PlateConfig(), times))
    manager.run_all(); app = QtWidgets.QApplication(sys.argv); gui = QtVisualizerV2(manager); gui.show(); sys.exit(app.exec())
