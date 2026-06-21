"""Gemini CLI-based DOE analysis chat dialog."""
import json
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional

try:
    from PySide6.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit,
        QPushButton, QLabel, QMessageBox, QSplitter, QGroupBox
    )
    from PySide6.QtCore import Qt, QThread, Signal
    PYSIDE6_AVAILABLE = True
except ImportError:
    PYSIDE6_AVAILABLE = False


def _find_gemini_cli() -> Optional[str]:
    """Find gemini CLI executable in PATH or config."""
    # Try PATH first
    gemini = shutil.which('gemini')
    if gemini:
        return gemini
    # Try config file
    config_path = os.path.join(os.path.dirname(__file__), '..', 'whtb_doe_config.json')
    try:
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        path = cfg.get('gemini_path')
        if path and os.path.exists(path):
            return path
    except Exception:
        pass
    return None


def _build_context(results: List[Dict]) -> str:
    """Build a concise context summary for Gemini."""
    from run_doe_analysis import group_by_contact_seq
    groups = group_by_contact_seq(results)
    n_total = len(results)
    group_summary = {k: len(v) for k, v in groups.items()}

    # Sample pkl field names from first result
    fields = []
    if results:
        result_obj = results[0].get('result')
        if result_obj:
            fields = [a for a in dir(result_obj)
                      if not a.startswith('_') and not callable(getattr(result_obj, a, None))][:15]

    ctx = f"""# DOE Results Summary
- total_cases: {n_total}
- groups (ContactSeqStr -> count): {json.dumps(group_summary, ensure_ascii=False)}
- result object fields (sample): {fields}

# How to access data in Python
```python
import pickle
r = pickle.load(open('<case_dir>/result.pkl', 'rb'))
import numpy as np
cih = np.asarray(r.corner_impact_hist)  # shape (N, 8) -- per-corner force history
times = np.asarray(r.time_history)       # shape (N,)
ground = np.asarray(r.ground_impact_hist)  # shape (N,) -- total ground force
seq = r.contact_seq  # dict with ContactSeqStr, TimeList, ForceList, ImpactList
```
"""
    return ctx


if PYSIDE6_AVAILABLE:
    class GeminiWorker(QThread):
        result_ready = Signal(str)
        error_occurred = Signal(str)

        def __init__(self, gemini_path: str, context: str, user_input: str):
            super().__init__()
            self.gemini_path = gemini_path
            self.context = context
            self.user_input = user_input

        def run(self):
            try:
                prompt = f"{self.context}\n\nUser: {self.user_input}"
                proc = subprocess.run(
                    [self.gemini_path, '-p', prompt],
                    capture_output=True, text=True, encoding='utf-8', timeout=120
                )
                output = proc.stdout or proc.stderr or "(no output)"
                self.result_ready.emit(output)
            except subprocess.TimeoutExpired:
                self.error_occurred.emit("Gemini CLI timed out after 120s")
            except Exception as e:
                self.error_occurred.emit(str(e))

    class GeminiDOEChatDialog(QDialog):
        def __init__(self, results: List[Dict], parent=None):
            super().__init__(parent)
            self.results = results
            self._gemini_path = _find_gemini_cli()
            self._context = _build_context(results)
            self._worker: Optional[GeminiWorker] = None
            self._pending_code: Optional[str] = None
            self._setup_ui()
            self.setWindowTitle("DOE Analysis - Gemini Chat")
            self.resize(900, 700)

        def _setup_ui(self):
            layout = QVBoxLayout(self)

            if not self._gemini_path:
                layout.addWidget(QLabel(
                    "Gemini CLI not found. Install with: npm install -g @google/gemini-cli\n"
                    "Or set 'gemini_path' in whtb_doe_config.json"
                ))
                return

            # Chat history
            self._chat = QTextEdit()
            self._chat.setReadOnly(True)
            self._chat.append(f"<b>[System]</b> Gemini CLI found: {self._gemini_path}")
            self._chat.append(f"<b>[System]</b> {len(self.results)} cases loaded. Ask anything about the DOE results.")
            layout.addWidget(self._chat)

            # Code execution area
            self._code_box = QGroupBox("Generated Code")
            code_layout = QVBoxLayout(self._code_box)
            self._code_preview = QTextEdit()
            self._code_preview.setReadOnly(True)
            self._code_preview.setMaximumHeight(120)
            code_layout.addWidget(self._code_preview)
            run_row = QHBoxLayout()
            self._run_btn = QPushButton("Run this code (full user privileges)")
            self._run_btn.clicked.connect(self._on_run_code)
            self._run_btn.setEnabled(False)
            run_row.addWidget(self._run_btn)
            run_row.addWidget(QLabel("Runs with full user privileges - review before running"))
            code_layout.addLayout(run_row)
            self._code_box.setVisible(False)
            layout.addWidget(self._code_box)

            # Input
            input_row = QHBoxLayout()
            self._input = QLineEdit()
            self._input.setPlaceholderText("Ask a question about the DOE results...")
            self._input.returnPressed.connect(self._on_send)
            self._send_btn = QPushButton("Send")
            self._send_btn.clicked.connect(self._on_send)
            input_row.addWidget(self._input)
            input_row.addWidget(self._send_btn)
            layout.addLayout(input_row)

        def _on_send(self):
            user_text = self._input.text().strip()
            if not user_text:
                return
            self._chat.append(f"<b>[You]</b> {user_text}")
            self._input.clear()
            self._send_btn.setEnabled(False)
            self._code_box.setVisible(False)
            self._pending_code = None

            self._worker = GeminiWorker(self._gemini_path, self._context, user_text)
            self._worker.result_ready.connect(self._on_response)
            self._worker.error_occurred.connect(self._on_error)
            self._worker.start()

        def _on_response(self, text: str):
            self._send_btn.setEnabled(True)
            self._chat.append(f"<b>[Gemini]</b><pre>{text}</pre>")

            # Detect Python code blocks
            import re
            code_match = re.search(r'```python\s*(.*?)\s*```', text, re.DOTALL)
            if code_match:
                self._pending_code = code_match.group(1)
                self._code_preview.setPlainText(self._pending_code)
                self._run_btn.setEnabled(True)
                self._code_box.setVisible(True)

        def _on_error(self, msg: str):
            self._send_btn.setEnabled(True)
            self._chat.append(f"<b>[Error]</b> {msg}")

        def _on_run_code(self):
            if not self._pending_code:
                return
            import matplotlib.pyplot as plt
            import numpy as np
            ns = {
                'results': self.results,
                'plt': plt,
                'np': np,
                'print': print,
            }
            try:
                exec(self._pending_code, ns)
                self._chat.append("<b>[System]</b> Code executed successfully.")
            except Exception as e:
                self._chat.append(f"<b>[System]</b> Code execution error: {e}")


def open_doe_analysis(doe_output_dir: str, parent=None):
    """Convenience function: load results and open chat dialog."""
    if not PYSIDE6_AVAILABLE:
        raise RuntimeError("PySide6 not available")
    from run_doe_analysis import load_doe_results
    results = load_doe_results(doe_output_dir)
    dlg = GeminiDOEChatDialog(results, parent)
    dlg.exec()
