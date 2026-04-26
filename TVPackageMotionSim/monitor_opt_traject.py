# -*- coding: utf-8 -*-
"""
[WHTOOLS] CMA-ES Optimization Monitor — PySide6 Real-time Dashboard

Usage:
    python monitor_opt_traject.py [result_dir]

The result_dir should contain opt_meta.json and evallog.csv.
"""

import sys
import os
import csv
import json
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTabWidget, QSplitter,
    QScrollArea, QSizePolicy, QGroupBox, QGridLayout,
)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QPixmap, QFont, QPalette, QColor

import numpy as np

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Apply Dark Theme to Matplotlib
plt.style.use('dark_background')
matplotlib.rcParams['font.size'] = 8
matplotlib.rcParams['axes.titlesize'] = 9
matplotlib.rcParams['axes.labelsize'] = 8
matplotlib.rcParams['xtick.labelsize'] = 7
matplotlib.rcParams['ytick.labelsize'] = 7
matplotlib.rcParams['legend.fontsize'] = 7

# ── Utilities ──────────────────────────────────────────────────────────────────

def _load_meta(result_dir: Path) -> dict:
    p = result_dir / "opt_meta.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_csv(result_dir: Path):
    """Read evallog.csv and return header list and rows list[list]."""
    p = result_dir / "evallog.csv"
    if not p.exists():
        return [], []
    try:
        with open(p, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        if len(rows) < 2:
            return (rows[0] if rows else []), []
        header = rows[0]
        data = rows[1:]
        return header, data
    except Exception:
        return [], []


def _col(header, name):
    try:
        return header.index(name)
    except ValueError:
        return None


def _safe_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return float("nan")


# ── UI Components ────────────────────────────────────────────────────────────

class ClickableImageLabel(QLabel):
    def __init__(self, path: Path, parent=None):
        super().__init__(parent)
        self.path = path
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Click to open in default viewer")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            import os
            try:
                os.startfile(str(self.path))
            except Exception as e:
                print(f"Failed to open image: {e}")


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=96):
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.fig.patch.set_facecolor('#1e1e1e')


# ── Tab 1: Convergence ───────────────────────────────────────────────────────

class ConvergenceTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.canvas = MplCanvas(self, width=8, height=5)
        layout.addWidget(self.canvas)

    def refresh(self, header, rows, meta):
        fig = self.canvas.fig
        fig.clear()

        if not rows:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No Data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11)
            self.canvas.draw()
            return

        ci_cost  = _col(header, "cost")
        ci_sigma = _col(header, "sigma")
        ci_gen   = _col(header, "gen")

        gens   = [_safe_float(r[ci_gen])   for r in rows] if ci_gen   is not None else list(range(len(rows)))
        costs  = [_safe_float(r[ci_cost])  for r in rows] if ci_cost  is not None else []
        sigmas = [_safe_float(r[ci_sigma]) for r in rows] if ci_sigma is not None else []

        # Best cost per generation
        gen_best = {}
        for g, c in zip(gens, costs):
            if np.isfinite(c):
                if g not in gen_best or c < gen_best[g]:
                    gen_best[g] = c
        sorted_gens = sorted(gen_best)
        running_best = []
        cur = float("inf")
        for g in sorted_gens:
            cur = min(cur, gen_best[g])
            running_best.append(cur)

        ax1 = fig.add_subplot(211)
        ax1.scatter(gens, costs, s=8, alpha=0.4, color="steelblue", label="Individual Cost")
        if sorted_gens:
            ax1.plot(sorted_gens, running_best, color="tomato", lw=1.8, label="Running Best")
        ax1.set_ylabel("Cost")
        ax1.set_title("Convergence Trend")
        ax1.legend()
        ax1.grid(True, ls="--", alpha=0.3)

        ax2 = fig.add_subplot(212)
        if sigmas:
            ax2.plot(gens, sigmas, color="darkorange", lw=1.2, label="σ")
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Sigma")
        ax2.set_title("CMA-ES Sigma Evolution")
        ax2.legend()
        ax2.grid(True, ls="--", alpha=0.3)

        self.canvas.draw()


# ── Tab 2: Objective Components ────────────────────────────────────────────────

class ObjectiveEvolutionTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.canvas = MplCanvas(self, width=8, height=5)
        layout.addWidget(self.canvas)

    def refresh(self, header, rows, meta):
        fig = self.canvas.fig
        fig.clear()

        if not rows:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No Data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11)
            self.canvas.draw()
            return

        ci_gen   = _col(header, "gen")
        ci_disp  = _col(header, "f_disp")
        ci_vel   = _col(header, "f_vel")

        if ci_disp is None or ci_vel is None:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "f_disp or f_vel not found in CSV", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11)
            self.canvas.draw()
            return

        gens  = [_safe_float(r[ci_gen])  for r in rows] if ci_gen is not None else list(range(len(rows)))
        disps = [_safe_float(r[ci_disp]) for r in rows]
        vels  = [_safe_float(r[ci_vel])  for r in rows]

        ax = fig.add_subplot(111)
        ax.scatter(gens, disps, s=6, alpha=0.4, color="skyblue", label="f_disp (Displacement)")
        ax.scatter(gens, vels,  s=6, alpha=0.4, color="orange",  label="f_vel (Velocity)")
        
        # Best trend for components
        gen_best_d = {}
        gen_best_v = {}
        for g, d, v in zip(gens, disps, vels):
            if np.isfinite(d):
                if g not in gen_best_d or d < gen_best_d[g]: gen_best_d[g] = d
            if np.isfinite(v):
                if g not in gen_best_v or v < gen_best_v[g]: gen_best_v[g] = v
        
        sg = sorted(gen_best_d)
        if sg:
            ax.plot(sg, [gen_best_d[g] for g in sg], color="deepskyblue", lw=1.5, label="Best f_disp")
        sgv = sorted(gen_best_v)
        if sgv:
            ax.plot(sgv, [gen_best_v[g] for g in sgv], color="darkorange", lw=1.5, label="Best f_vel")

        ax.set_xlabel("Generation")
        ax.set_ylabel("Metric Value")
        ax.set_title("Objective Function Components Evolution")
        ax.legend()
        ax.grid(True, ls="--", alpha=0.3)

        self.canvas.draw()


# ── Tab 3: Parameter Status ──────────────────────────────────────────────────

class ParamStatusTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.canvas = MplCanvas(self, width=8, height=6)
        layout.addWidget(self.canvas)

    def refresh(self, header, rows, meta):
        fig = self.canvas.fig
        fig.clear()

        param_defs = meta.get("param_defs", [])
        if not param_defs or not rows:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No Parameter Metadata\n(Check opt_meta.json)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11)
            self.canvas.draw()
            return

        # best row (min cost)
        ci_cost = _col(header, "cost")
        ci_div  = _col(header, "diverged")
        best_row = None
        best_c = float("inf")
        for r in rows:
            div = r[ci_div].strip().lower() if ci_div is not None else "false"
            if div in ("true", "1"):
                continue
            c = _safe_float(r[ci_cost]) if ci_cost is not None else float("inf")
            if c < best_c:
                best_c = c
                best_row = r

        n = len(param_defs)
        ax = fig.add_subplot(111)

        labels, inits, lows, highs, bests = [], [], [], [], []
        for i, pdef in enumerate(param_defs):
            # Format: [name, init, lo, hi, is_log]
            name, init, lo, hi = pdef[0], pdef[1], pdef[2], pdef[3]
            ci_p = _col(header, name)
            bval = _safe_float(best_row[ci_p]) if (best_row and ci_p is not None) else init
            labels.append(name)
            inits.append(init)
            lows.append(lo)
            highs.append(hi)
            bests.append(bval)

        x = np.arange(n)
        # Normalize for visualization
        ranges = [max(h - l, 1e-12) for l, h in zip(lows, highs)]
        norm_init = [(v - l) / r for v, l, r in zip(inits, lows, ranges)]
        norm_best = [(v - l) / r for v, l, r in zip(bests, lows, ranges)]

        ax.barh(x, norm_init, height=0.35, left=0, color="#444444",
                label="Initial (Normalized)", align="center")
        ax.barh(x + 0.35, norm_best, height=0.35, left=0, color="steelblue",
                label=f"Best (Cost={best_c:.5f})", align="center")
        ax.set_yticks(x + 0.175)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Normalized Value (0=Min, 1=Max)")
        ax.set_xlim(0, 1.05)
        ax.axvline(0.5, ls="--", lw=0.8, color="gray", alpha=0.5)
        ax.legend()
        ax.set_title("Current Parameter Status")
        ax.grid(True, axis="x", ls="--", alpha=0.2)

        self.canvas.draw()


# ── Tab 4: Parameter Evolution ───────────────────────────────────────────────

class ParamEvolutionTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.canvas = MplCanvas(self, width=8, height=10)
        layout.addWidget(self.canvas)

    def refresh(self, header, rows, meta):
        fig = self.canvas.fig
        fig.clear()

        param_defs = meta.get("param_defs", [])
        if not param_defs or not rows:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No Data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11)
            self.canvas.draw()
            return

        ci_gen  = _col(header, "gen")
        gens = [_safe_float(r[ci_gen]) for r in rows] if ci_gen is not None else list(range(len(rows)))

        n = len(param_defs)
        ncols = 3 if n > 6 else 2
        nrows = (n + ncols - 1) // ncols

        for i, pdef in enumerate(param_defs):
            name, init, lo, hi = pdef[0], pdef[1], pdef[2], pdef[3]
            ci_p = _col(header, name)
            if ci_p is None: continue
            vals = [_safe_float(r[ci_p]) for r in rows]
            ax = fig.add_subplot(nrows, ncols, i + 1)
            ax.scatter(gens, vals, s=4, alpha=0.4, color="steelblue")
            ax.axhline(init, ls="--", lw=1.0, color="tomato", label=f"Init: {init:.3g}")
            ax.axhline(lo,   ls=":",  lw=0.7, color="gray")
            ax.axhline(hi,   ls=":",  lw=0.7, color="gray")
            ax.set_title(name)
            ax.tick_params(labelsize=6)
            ax.grid(True, ls="--", alpha=0.2)

        self.canvas.draw()


# ── Tab 5: Response Distribution ──────────────────────────────────────────────

class ResponseTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.canvas = MplCanvas(self, width=9, height=5)
        layout.addWidget(self.canvas)

    def refresh(self, header, rows, meta):
        fig = self.canvas.fig
        fig.clear()

        corners  = meta.get("selected_corners", [])
        rw       = meta.get("response_weights", {})

        if not corners or not rows:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No Corner Data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11)
            self.canvas.draw()
            return

        # best row
        ci_cost = _col(header, "cost")
        ci_div  = _col(header, "diverged")
        best_row = None
        best_c = float("inf")
        for r in rows:
            div = r[ci_div].strip().lower() if ci_div is not None else "false"
            if div in ("true", "1"):
                continue
            c = _safe_float(r[ci_cost]) if ci_cost is not None else float("inf")
            if c < best_c:
                best_c = c
                best_row = r

        ax = fig.add_subplot(111)
        x = np.arange(len(corners))
        w = 0.35

        disp_vals, vel_vals = [], []
        disp_active, vel_active = [], []
        for c in corners:
            ci_d = _col(header, f"{c}_disp")
            ci_v = _col(header, f"{c}_vel")
            dv = _safe_float(best_row[ci_d]) if (best_row and ci_d is not None) else float("nan")
            vv = _safe_float(best_row[ci_v]) if (best_row and ci_v is not None) else float("nan")
            disp_vals.append(dv)
            vel_vals.append(vv)
            cw = rw.get(c, {})
            disp_active.append(float(cw.get("disp", 1.0)) > 0)
            vel_active.append(float(cw.get("vel",  1.0)) > 0)

        for i, (dv, da) in enumerate(zip(disp_vals, disp_active)):
            color = "steelblue" if da else "#333333"
            ax.bar(x[i] - w / 2, dv if np.isfinite(dv) else 0,
                   width=w, color=color, alpha=0.85,
                   label="Disp DTW" if i == 0 else "_")

        for i, (vv, va) in enumerate(zip(vel_vals, vel_active)):
            color = "darkorange" if va else "#333333"
            ax.bar(x[i] + w / 2, vv if np.isfinite(vv) else 0,
                   width=w, color=color, alpha=0.85,
                   label="Vel DTW" if i == 0 else "_")

        ax.set_xticks(x)
        ax.set_xticklabels(corners)
        ax.set_ylabel("DTW Cost")
        ax.set_title(f"Response Distribution per Corner - Best (Cost={best_c:.5f})  *Gray=Inactive")
        ax.legend()
        ax.grid(True, axis="y", ls="--", alpha=0.2)

        self.canvas.draw()


# ── Tab 6: Comparison Images ─────────────────────────────────────────────────

class ImageTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: #1e1e1e; }")
        layout.addWidget(scroll)

        container = QWidget()
        container.setStyleSheet("background-color: #1e1e1e;")
        self._grid = QGridLayout(container)
        scroll.setWidget(container)
        self._labels = []

    def refresh(self, result_dir: Path):
        for lbl in self._labels:
            lbl.setParent(None)
        self._labels.clear()

        if result_dir is None or not result_dir.exists():
            return

        pngs = sorted(result_dir.glob("*.png"))
        if not pngs:
            lbl = QLabel("No PNG files found")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("color: gray;")
            self._grid.addWidget(lbl, 0, 0)
            self._labels.append(lbl)
            return

        ncols = 2
        for idx, p in enumerate(pngs):
            row, col = divmod(idx, ncols)
            frame = QGroupBox(p.name)
            frame.setStyleSheet("QGroupBox { font-weight: bold; color: #aaaaaa; }")
            vbox = QVBoxLayout(frame)
            img_lbl = ClickableImageLabel(p)
            img_lbl.setAlignment(Qt.AlignCenter)
            px = QPixmap(str(p))
            if not px.isNull():
                px = px.scaled(550, 380, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                img_lbl.setPixmap(px)
            else:
                img_lbl.setText("Failed to load image")
            vbox.addWidget(img_lbl)
            self._grid.addWidget(frame, row, col)
            self._labels.append(frame)


# ── Main Window ─────────────────────────────────────────────────────────────

class MonitorWindow(QMainWindow):
    def __init__(self, result_dir: str = None):
        super().__init__()
        self.setWindowTitle("CMA-ES Optimization Monitor")
        self.resize(1200, 850)
        self.setStyleSheet("""
            QMainWindow { background-color: #121212; }
            QWidget { background-color: #121212; color: #d4d4d4; }
            QTabWidget::pane { border: 1px solid #333; background: #1e1e1e; }
            QTabBar::tab { background: #2d2d2d; border: 1px solid #444; padding: 6px 12px; margin-right: 2px; }
            QTabBar::tab:selected { background: #3e3e42; border-bottom-color: #007acc; }
            QPushButton { background-color: #333; border: 1px solid #555; padding: 4px 10px; }
            QPushButton:hover { background-color: #444; }
            QLabel { background-color: transparent; }
        """)

        self._result_dir: Path | None = None
        self._meta: dict = {}

        # ── Header / Toolbar ──────────────────────────────────────────────
        header = QWidget()
        header.setFixedHeight(60)
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(10, 5, 10, 5)

        # Logo
        self._logo_label = QLabel()
        logo_path = Path(__file__).parent / "resources" / "sidebar_logo.png"
        if not logo_path.exists(): # Fallback search
             logo_path = Path("C:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/resources/sidebar_logo.png")
        
        if logo_path.exists():
            pix = QPixmap(str(logo_path))
            if not pix.isNull():
                self._logo_label.setPixmap(pix.scaled(120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        h_layout.addWidget(self._logo_label)
        h_layout.addSpacing(20)

        self._dir_label = QLabel("Result Folder: (Not Selected)")
        self._dir_label.setFont(QFont("Consolas", 8))
        self._dir_label.setStyleSheet("color: #888;")
        h_layout.addWidget(self._dir_label, stretch=1)

        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self._browse)
        h_layout.addWidget(btn_browse)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #007acc; font-weight: bold; font-size: 8pt;")
        h_layout.addWidget(self._status_label)

        # ── Tabs ─────────────────────────────────────────────────────────────
        self._tabs = QTabWidget()
        self._tab_conv   = ConvergenceTab()
        self._tab_obj    = ObjectiveEvolutionTab()
        self._tab_pstat  = ParamStatusTab()
        self._tab_pevol  = ParamEvolutionTab()
        self._tab_resp   = ResponseTab()
        self._tab_img    = ImageTab()

        self._tabs.addTab(self._tab_conv,  "Convergence")
        self._tabs.addTab(self._tab_obj,   "Obj Evolution")
        self._tabs.addTab(self._tab_pstat, "Param Status")
        self._tabs.addTab(self._tab_pevol, "Param Evolution")
        self._tabs.addTab(self._tab_resp,  "Response Dist.")
        self._tabs.addTab(self._tab_img,   "Images")
        self._tabs.currentChanged.connect(self._on_tab_changed)

        # ── Layout ───────────────────────────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        main_layout.addWidget(header)
        main_layout.addWidget(self._tabs, stretch=1)

        # ── Auto-refresh Timer (5s) ─────────────────────────────────────────
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(5000)

        # Initial load if path provided
        if result_dir:
            self._set_dir(Path(result_dir))

    # ── Event Handlers ───────────────────────────────────────────────────────

    def _browse(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select Result Folder",
            str(self._result_dir) if self._result_dir else os.getcwd(),
        )
        if path:
            self._set_dir(Path(path))

    def _set_dir(self, p: Path):
        self._result_dir = p
        self._dir_label.setText(f"Path: {p}")
        self._meta = _load_meta(p)
        self._refresh()

    def _on_tab_changed(self, idx):
        self._refresh_tab(idx)

    def _refresh(self):
        if self._result_dir is None:
            return
        # Reload meta (it might be updated after start)
        self._meta = _load_meta(self._result_dir)
        self._refresh_tab(self._tabs.currentIndex())
        self._update_status()

    def _update_status(self):
        if self._result_dir is None:
            return
        _, rows = _load_csv(self._result_dir)
        n = len(rows)
        max_e = self._meta.get("max_evals", "?")
        self._status_label.setText(f"Eval: {n} / {max_e}  (Auto-refreshing...)")

    def _refresh_tab(self, idx: int):
        if self._result_dir is None:
            return
        header, rows = _load_csv(self._result_dir)
        meta = self._meta

        if idx == 0:
            self._tab_conv.refresh(header, rows, meta)
        elif idx == 1:
            self._tab_obj.refresh(header, rows, meta)
        elif idx == 2:
            self._tab_pstat.refresh(header, rows, meta)
        elif idx == 3:
            self._tab_pevol.refresh(header, rows, meta)
        elif idx == 4:
            self._tab_resp.refresh(header, rows, meta)
        elif idx == 5:
            self._tab_img.refresh(self._result_dir)


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Font Settings
    font = QFont("Noto Sans", 8)
    if not QFont(font).exactMatch():
        font = QFont("Segoe UI", 8) # Fallback for Windows
    app.setFont(font)

    # Dark Palette (Global)
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(18, 18, 18))
    palette.setColor(QPalette.WindowText, QColor(212, 212, 212))
    palette.setColor(QPalette.Base, QColor(30, 30, 30))
    palette.setColor(QPalette.AlternateBase, QColor(18, 18, 18))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ToolTipText, QColor(212, 212, 212))
    palette.setColor(QPalette.Text, QColor(212, 212, 212))
    palette.setColor(QPalette.Button, QColor(50, 50, 50))
    palette.setColor(QPalette.ButtonText, QColor(212, 212, 212))
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(0, 122, 204))
    palette.setColor(QPalette.Highlight, QColor(0, 122, 204))
    palette.setColor(QPalette.HighlightedText, Qt.white)
    app.setPalette(palette)

    init_dir = sys.argv[1] if len(sys.argv) > 1 else None
    win = MonitorWindow(result_dir=init_dir)
    win.show()
    sys.exit(app.exec())
