# -*- coding: utf-8 -*-
"""
[WHTOOLS] Shared UI Components Library
Standardized UI widgets for WHT Visualizer, PostProcessors, and Inspectors.
Design Philosophy: Decouple Layout/Design from Data Logic.
"""

from PySide6 import QtWidgets, QtCore, QtGui
import numpy as np

class WHTRangeControlGroup(QtWidgets.QFrame):
    """
    [WHTOOLS] Unified Range Control Widget
    Handles UI layout for scalar ranges. Emits signals for logic integration.
    """
    rangeChanged = QtCore.Signal(float, float) # (min, max)
    modeChanged = QtCore.Signal(str)           # "Dynamic", "Robust", "Static"
    fitRequested = QtCore.Signal()             # Triggered when 'Fit' is clicked

    def __init__(self, parent=None, show_robust=True):
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.show_robust = show_robust
        self.robust_pct = 98.0 # [WHT] Store robust percentage as state variable since widget is removed from main bar
        self._init_ui()

    def _init_ui(self):
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        self.cmb_mode = QtWidgets.QComboBox()
        modes = ["Dynamic", "Robust", "Static"] if self.show_robust else ["Dynamic", "Static"]
        self.cmb_mode.addItems(modes)
        self.cmb_mode.setFixedWidth(85)

        self.sp_min = QtWidgets.QDoubleSpinBox(); self.sp_max = QtWidgets.QDoubleSpinBox()
        for sp in [self.sp_min, self.sp_max]:
            sp.setRange(-1e15, 1e15); sp.setDecimals(4); sp.setFixedWidth(85)
            sp.setReadOnly(True) # [WHT] Read-only as per user request
            sp.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons) # Hide arrows to look like text box

        self.btn_adj = QtWidgets.QPushButton("🛠️ Adjust")
        self.btn_adj.setFixedWidth(85); self.btn_adj.setStyleSheet("font-weight: bold;")

        layout.addWidget(self.cmb_mode)
        layout.addWidget(QtWidgets.QLabel("Min:")); layout.addWidget(self.sp_min)
        layout.addWidget(QtWidgets.QLabel("Max:")); layout.addWidget(self.sp_max)
        layout.addWidget(self.btn_adj)

        self.cmb_mode.currentTextChanged.connect(self._on_mode_changed)
        self.sp_min.valueChanged.connect(lambda: self.rangeChanged.emit(self.sp_min.value(), self.sp_max.value()))
        self.sp_max.valueChanged.connect(lambda: self.rangeChanged.emit(self.sp_min.value(), self.sp_max.value()))

    def _on_mode_changed(self, mode):
        self.modeChanged.emit(mode)

    def set_range(self, v_min, v_max):
        self.sp_min.blockSignals(True); self.sp_min.setValue(v_min); self.sp_min.blockSignals(False)
        self.sp_max.blockSignals(True); self.sp_max.setValue(v_max); self.sp_max.blockSignals(False)

    def get_range(self):
        return self.sp_min.value(), self.sp_max.value()

    def get_robust_pct(self):
        return self.robust_pct

    def set_robust_pct(self, v):
        self.robust_pct = v


class WHTRangeDialog(QtWidgets.QDialog):
    """
    [WHT Premium UI] Unified interactive dialog for real-time scalar range adjustment.
    Commonly used for both 3D Fields and 2D Plotting.
    Logic is injected via get_limits_fn and get_robust_fn.
    """
    def __init__(self, parent, field_name: str, rng_group: WHTRangeControlGroup, get_limits_fn, get_robust_fn):
        super().__init__(parent)
        self.p = parent
        self.field = field_name
        self.rng_group = rng_group
        self.get_limits_fn = get_limits_fn
        self.get_robust_fn = get_robust_fn
        
        self.setWindowTitle(f"Adjust Range: {field_name}")
        self.setMinimumWidth(550)
        
        # Physical limits of the current context
        self.g_min, self.g_max = self.get_limits_fn()
        
        # Mapping range (50% wider for flexibility)
        span = self.g_max - self.g_min
        if span <= 0: span = 1e-6
        self.s_min = self.g_min - 0.5 * span
        self.s_max = self.g_max + 0.5 * span
        
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(15)
        
        def create_entry(label, current_val, limit_val):
            group = QtWidgets.QGroupBox(label)
            vbox = QtWidgets.QVBoxLayout(group)
            lbl_limit = QtWidgets.QLabel(f"Context {label.split()[0]} (Actual): {limit_val:.4e}")
            lbl_limit.setStyleSheet("color: #888888; font-size: 8pt; font-family: 'Consolas', monospace;")
            vbox.addWidget(lbl_limit)
            hbox = QtWidgets.QHBoxLayout()
            edit = QtWidgets.QLineEdit(f"{current_val:.4e}")
            val_validator = QtGui.QDoubleValidator()
            val_validator.setNotation(QtGui.QDoubleValidator.ScientificNotation)
            edit.setValidator(val_validator)
            edit.setMinimumWidth(120)
            btn_snap = QtWidgets.QPushButton("|<" if "Minimum" in label else ">|")
            btn_snap.setFixedWidth(40)
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setRange(0, 1000)
            
            def update_slider_from_edit():
                try:
                    v = float(edit.text())
                    pct = (v - self.s_min) / (self.s_max - self.s_min)
                    slider.blockSignals(True)
                    slider.setValue(int(np.clip(pct, 0, 1) * 1000))
                    slider.blockSignals(False)
                    self._trigger_update()
                except ValueError: pass

            def update_edit_from_slider(v):
                val = self.s_min + (v / 1000.0) * (self.s_max - self.s_min)
                edit.blockSignals(True); edit.setText(f"{val:.4e}"); edit.blockSignals(False)
                self._trigger_update()
                
            def snap_to_limit():
                edit.setText(f"{limit_val:.4e}"); update_slider_from_edit()

            edit.textChanged.connect(update_slider_from_edit)
            edit.returnPressed.connect(self._trigger_update) # Apply on Enter
            slider.valueChanged.connect(update_edit_from_slider)
            btn_snap.clicked.connect(snap_to_limit)
            
            pct_init = (current_val - self.s_min) / (self.s_max - self.s_min)
            slider.blockSignals(True); slider.setValue(int(np.clip(pct_init, 0, 1) * 1000)); slider.blockSignals(False)
            
            hbox.addWidget(slider, 3); hbox.addWidget(edit, 1); hbox.addWidget(btn_snap)
            vbox.addLayout(hbox)
            return group, edit, slider

        c_min, c_max = self.rng_group.get_range()
        self.grp_min, self.edit_min, self.slider_min = create_entry("Minimum Threshold", c_min, self.g_min)
        self.grp_max, self.edit_max, self.slider_max = create_entry("Maximum Threshold", c_max, self.g_max)
        layout.addWidget(self.grp_min); layout.addWidget(self.grp_max)
        
        group_robust = QtWidgets.QGroupBox("Statistical Robustness")
        h_robust = QtWidgets.QHBoxLayout(group_robust)
        self.spin_robust = QtWidgets.QDoubleSpinBox()
        self.spin_robust.setRange(50.0, 100.0)
        self.spin_robust.setValue(self.rng_group.get_robust_pct())
        self.spin_robust.setSuffix(" %")
        self.spin_robust.setToolTip("데이터 분포의 중심 백분율을 설정합니다.\n(예: 98% 설정 시 상/하위 1%씩을 특이점으로 간주하여 제외)")
        
        btn_robust = QtWidgets.QPushButton("Apply Robust Auto"); btn_robust.clicked.connect(self._apply_robust)
        btn_robust.setToolTip("설정된 백분율을 기반으로 특이점을 제외한 유효 범위를 자동 계산하여 적용합니다.")
        
        btn_global = QtWidgets.QPushButton("Full Global Auto"); btn_global.clicked.connect(self._apply_global)
        btn_global.setToolTip("전체 데이터의 절대 최소/최대값으로 범위를 확장합니다.")
        
        btn_find = QtWidgets.QPushButton("🔍 Find Outliers"); btn_find.clicked.connect(self._find_outliers)
        btn_find.setToolTip("현재 입력된 최대값(Max Threshold)을 초과하는 노드들을 3D 뷰에서 강조 표시합니다.")
        
        h_robust.addWidget(QtWidgets.QLabel("Threshold:")); h_robust.addWidget(self.spin_robust)
        h_robust.addWidget(btn_robust); h_robust.addWidget(btn_global); h_robust.addWidget(btn_find)
        layout.addWidget(group_robust)
        
        btn_close = QtWidgets.QPushButton("Done"); btn_close.clicked.connect(self.accept)
        btn_close.setAutoDefault(False); btn_close.setDefault(False) # Prevent Enter from closing
        layout.addWidget(btn_close)
        self.finished.connect(self._cleanup)

    def _cleanup(self):
        """Removes outlier highlights from parent visualizer when dialog is closed."""
        if hasattr(self.p, "clear_outliers"):
            self.p.clear_outliers()

    def _trigger_update(self):
        try:
            v_min = float(self.edit_min.text())
            v_max = float(self.edit_max.text())
            
            # [WHT] Critical Sync Order: Update values FIRST, then change mode.
            # This prevents any mode-change listeners from pulling stale values.
            self.rng_group.set_range(v_min, v_max)
            self.rng_group.cmb_mode.blockSignals(True)
            self.rng_group.cmb_mode.setCurrentText("Static")
            self.rng_group.cmb_mode.blockSignals(False)
            
            # Manually trigger updates that would have been triggered by signals
            self.rng_group.rangeChanged.emit(v_min, v_max)
        except ValueError: pass

    def _apply_robust(self):
        pct = self.spin_robust.value()
        self.rng_group.set_robust_pct(pct) # [WHT] Sync state back to group
        rng = self.get_robust_fn(pct)
        self.edit_min.setText(f"{rng[0]:.4e}"); self.edit_max.setText(f"{rng[1]:.4e}")

    def _apply_global(self):
        self.edit_min.setText(f"{self.g_min:.4e}"); self.edit_max.setText(f"{self.g_max:.4e}")

    def _find_outliers(self):
        """Highlights nodes exceeding the current max threshold in the parent visualizer."""
        try:
            threshold = float(self.edit_max.text())
            if hasattr(self.p, "_highlight_outliers"):
                self.p._highlight_outliers(self.field, threshold)
        except ValueError: pass
