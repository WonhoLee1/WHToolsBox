"""DOE Setup Dialog — PySide6 QDialog for interactive DOE parameter definition."""
from __future__ import annotations
import sys
import copy
from typing import Any, Dict, List, Optional, Tuple

try:
    from PySide6.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QComboBox, QSpinBox, QTableWidget, QTableWidgetItem,
        QTreeWidget, QTreeWidgetItem, QMessageBox, QDialogButtonBox,
        QGroupBox, QFormLayout, QLineEdit, QSplitter, QHeaderView,
        QAbstractItemView, QWidget
    )
    from PySide6.QtCore import Qt
    PYSIDE6_AVAILABLE = True
except ImportError:
    PYSIDE6_AVAILABLE = False

import pandas as pd
from whtb_doe.definition import DOEDefinition


def _build_tree(config, parent_item, prefix='cfg'):
    """Recursively build QTreeWidget from config dict or list."""
    if isinstance(config, dict):
        items = config.items()
    elif isinstance(config, list):
        items = enumerate(config)
    else:
        return

    for key, val in items:
        if isinstance(key, int):
            full_key = f"{prefix}[{key}]"
            display_key = str(key)
        else:
            full_key = f"{prefix}['{key}']"
            display_key = key
            
        if isinstance(val, dict):
            item = QTreeWidgetItem(parent_item, [display_key, '(dict)', ''])
            _build_tree(val, item, full_key)
        elif isinstance(val, list):
            item = QTreeWidgetItem(parent_item, [display_key, '(list)', str(val)])
            item.setData(0, Qt.UserRole, full_key)
            _build_tree(val, item, full_key)
        else:
            item = QTreeWidgetItem(parent_item, [display_key, type(val).__name__, str(val)])
            item.setData(0, Qt.UserRole, full_key)


class DOESetupDialog(QDialog):
    def __init__(self, base_config: dict, parent=None):
        super().__init__(parent)
        self.base_config = base_config
        self.doe_table: Optional[pd.DataFrame] = None
        self.config_list: Optional[List[dict]] = None
        self._var_rows: List[Tuple[str, QLineEdit]] = []  # (key_path_expr, line_edit)
        self._setup_ui()
        self.setWindowTitle("DOE Setup")
        self.resize(900, 650)

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        top = QSplitter(Qt.Horizontal)

        # Left: variable list
        left = QGroupBox("Variables")
        left_layout = QVBoxLayout(left)
        btn_add = QPushButton("Add Variable")
        btn_add.clicked.connect(self._on_add_variable)
        left_layout.addWidget(btn_add)
        self._var_table = QTableWidget(0, 2)
        self._var_table.setHorizontalHeaderLabels(["Key Path", "Definition"])
        self._var_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        left_layout.addWidget(self._var_table)
        top.addWidget(left)

        # Right: settings
        right = QGroupBox("DOE Settings")
        right_layout = QFormLayout(right)
        self._method_combo = QComboBox()
        self._method_combo.addItems(["lhs", "fullfact", "montecarlo"])
        right_layout.addRow("Method:", self._method_combo)
        self._n_spin = QSpinBox()
        self._n_spin.setRange(1, 10000)
        self._n_spin.setValue(100)
        right_layout.addRow("N samples:", self._n_spin)
        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 99999)
        self._seed_spin.setValue(42)
        right_layout.addRow("Seed:", self._seed_spin)
        top.addWidget(right)

        main_layout.addWidget(top)

        # Action buttons
        action_row = QHBoxLayout()
        btn_validate = QPushButton("Validate")
        btn_validate.clicked.connect(self._on_validate)
        btn_make = QPushButton("Make DOE List")
        btn_make.clicked.connect(self._on_make_doe)
        self._status_label = QLabel("")
        action_row.addWidget(btn_validate)
        action_row.addWidget(btn_make)
        action_row.addWidget(self._status_label)
        action_row.addStretch()
        main_layout.addLayout(action_row)

        # DOE table
        self._doe_table_widget = QTableWidget(0, 0)
        self._doe_table_widget.setEditTriggers(QAbstractItemView.AllEditTriggers)
        main_layout.addWidget(self._doe_table_widget)

        # OK/Cancel
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)
        main_layout.addWidget(buttons)

    def _build_dsl(self) -> str:
        lines = []
        for row in range(self._var_table.rowCount()):
            key_item = self._var_table.item(row, 0)
            val_item = self._var_table.item(row, 1)
            if key_item and val_item:
                lines.append(f"{key_item.text()} = {val_item.text()}")
        return '\n'.join(lines)

    def _on_add_variable(self):
        dlg = _ConfigTreeDialog(self.base_config, self)
        if dlg.exec() == QDialog.Accepted and dlg.selected_path:
            row = self._var_table.rowCount()
            self._var_table.insertRow(row)
            self._var_table.setItem(row, 0, QTableWidgetItem(dlg.selected_path))
            self._var_table.setItem(row, 1, QTableWidgetItem("[10, 20, 30]"))

    def _on_validate(self):
        dsl = self._build_dsl()
        if not dsl.strip():
            self._status_label.setText("No variables defined")
            return
        try:
            ok, msg = DOEDefinition(dsl, self.base_config).validate()
            self._status_label.setText(("OK: " if ok else "Error: ") + msg)
        except Exception as e:
            self._status_label.setText(f"Error: {e}")

    def _on_make_doe(self):
        dsl = self._build_dsl()
        if not dsl.strip():
            QMessageBox.warning(self, "DOE Setup", "No variables defined.")
            return
        try:
            doe_def = DOEDefinition(dsl, self.base_config)
            method = self._method_combo.currentText()
            n = self._n_spin.value()
            seed = self._seed_spin.value()
            self.doe_table, self.config_list = doe_def.generate(method=method, n_samples=n, seed=seed)
            self._populate_doe_table(self.doe_table)
            self._status_label.setText(f"Generated {len(self.doe_table)} cases")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _populate_doe_table(self, df: pd.DataFrame):
        self._doe_table_widget.clear()
        self._doe_table_widget.setRowCount(len(df))
        self._doe_table_widget.setColumnCount(len(df.columns))
        self._doe_table_widget.setHorizontalHeaderLabels(list(df.columns))
        for i, row in df.iterrows():
            for j, val in enumerate(row):
                self._doe_table_widget.setItem(i, j, QTableWidgetItem(str(val)))

    def _read_doe_table(self) -> Optional[pd.DataFrame]:
        if self.doe_table is None:
            return None
        cols = [self._doe_table_widget.horizontalHeaderItem(j).text()
                for j in range(self._doe_table_widget.columnCount())]
        rows = []
        for i in range(self._doe_table_widget.rowCount()):
            row = {}
            for j, col in enumerate(cols):
                item = self._doe_table_widget.item(i, j)
                try:
                    row[col] = float(item.text()) if item else 0.0
                except ValueError:
                    row[col] = item.text() if item else ''
            rows.append(row)
        return pd.DataFrame(rows)

    def _on_ok(self):
        if self.doe_table is None:
            QMessageBox.information(self, "DOE Setup", "Please click 'Make DOE List' first.")
            return
        current_table = self._read_doe_table()
        if current_table is not None and not current_table.equals(
                self.doe_table.reset_index(drop=True).astype(str).applymap(str)):
            reply = QMessageBox.question(
                self, "Apply Changes?",
                "DOE table was edited. Apply changes to config list?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                dsl = self._build_dsl()
                doe_def = DOEDefinition(dsl, self.base_config)
                self.doe_table, self.config_list = doe_def.regenerate(current_table)
        self.accept()

    @classmethod
    def run_dialog(cls, base_config: dict, parent=None) -> Optional[Tuple[pd.DataFrame, List[dict]]]:
        if not PYSIDE6_AVAILABLE:
            raise RuntimeError("PySide6 not available")
        dlg = cls(base_config, parent)
        if dlg.exec() == QDialog.Accepted:
            return dlg.doe_table, dlg.config_list
        return None


class _ConfigTreeDialog(QDialog):
    """Config tree browser for selecting a variable."""
    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.selected_path: Optional[str] = None
        self.setWindowTitle("Select Variable")
        self.resize(400, 400)
        layout = QVBoxLayout(self)
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Key", "Type", "Value"])
        self._tree.itemDoubleClicked.connect(self._on_double_click)
        root = QTreeWidgetItem(self._tree, ["config", "dict", ""])
        _build_tree(config, root)
        root.setExpanded(True)
        layout.addWidget(self._tree)
        btn = QPushButton("Select")
        btn.clicked.connect(self._on_select)
        layout.addWidget(btn)

    def _on_double_click(self, item, col):
        path = item.data(0, Qt.UserRole)
        if path:
            self.selected_path = path
            self.accept()

    def _on_select(self):
        item = self._tree.currentItem()
        if item:
            path = item.data(0, Qt.UserRole)
            if path:
                self.selected_path = path
                self.accept()
