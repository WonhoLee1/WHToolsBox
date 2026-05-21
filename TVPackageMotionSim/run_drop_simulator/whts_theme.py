# -*- coding: utf-8 -*-
"""
[WHTOOLS] Centralized UI Theme
모든 WHTOOLS UI 모듈이 공유하는 색상 토큰과 QSS 스타일시트를 정의합니다.
적용 방법: launch_control_panel() 내에서 apply_app_theme(app) 한 번 호출.
"""

# ─── Color Tokens ────────────────────────────────────────────────────────────
C_BG          = "#1e1e1e"   # Window / Dialog background
C_BG_DARK     = "#121212"   # Darker background (monitor window, matplotlib)
C_BG_INPUT    = "#2b2b2b"   # Input field background
C_BG_TABLE    = "#252525"   # Table / Tree background
C_BG_EDITOR   = "#1a1a1a"   # Code / script editor background
C_BG_CONSOLE  = "#000000"   # Terminal console background
C_BG_BTN      = "#333333"   # Default button background
C_BG_BTN_HOV  = "#444444"   # Button hover
C_BG_BTN_PRS  = "#555555"   # Button pressed
C_BORDER      = "#333333"   # Generic border
C_BORDER_IN   = "#444444"   # Input / Table border
C_TEXT        = "#ffffff"   # Primary text
C_TEXT_DIM    = "#e0e0e0"   # Secondary label text
C_TEXT_MUTED  = "#888888"   # Muted / placeholder text
C_TEXT_TREE   = "#dcdcdc"   # Tree item text
C_ACCENT      = "#88c0d0"   # GroupBox title / accent cyan
C_ACCENT_POST = "#00d2ff"   # Accent for post-processor UI
C_SEL         = "#3d4b5c"   # Tree / table selection background
C_SLIDER      = "#0078d7"   # Slider handle (Windows blue)

# Action button colors
C_BTN_GREEN   = "#2e7d32"
C_BTN_GREEN2  = "#00796b"
C_BTN_BLUE    = "#0288d1"
C_BTN_BLUE2   = "#1976d2"
C_BTN_BLUE3   = "#0d47a1"
C_BTN_INDIGO  = "#3f51b5"
C_BTN_BROWN   = "#795548"
C_BTN_RED     = "#c62828"
C_BTN_TEAL    = "#0b5345"
C_BTN_NAVY         = "#2c3e50"   # Reload XML button (dark navy)
C_BTN_NAVY_BORDER  = "#34495e"   # Reload XML button border
C_BTN_RED2         = "#ff4444"   # Stop button (bright red)
C_ACCENT_BLUE      = "#42a5f5"   # Light blue (editor title)
C_TEXT_CONSOLE     = "#00ff00"   # Terminal green (console output)

# Dynamic state colors (toggled by simulation state, not action buttons)
C_STATE_SLOW_MOTION = "#554400"   # Slow-motion active (dark yellow)
C_STATE_RECORDING   = "#550000"   # Recording active (dark red bg)
C_STATE_REC_TEXT    = "#ff0000"   # Recording active (red text)
C_STATE_ORANGE      = "#e67e22"   # Warning/initializing status

# Status colors
C_STATUS_OK   = "#2ecc71"
C_STATUS_WARN = "#f1c40f"
C_STATUS_INFO = "#3498db"
C_STATUS_ERR  = "#ff6b68"
C_STATUS_TEXT_OK   = "#a9dc76"
C_STATUS_TEXT_WARN = "#e2c08d"

# ─── Shared SVG Arrow Sprites (for SpinBox) ──────────────────────────────────
_SVG_UP   = "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 16 16'><path d='M3 11l5-5 5 5' stroke='%23ffffff' stroke-width='2.5' fill='none' stroke-linecap='round' stroke-linejoin='round'/></svg>\")"
_SVG_UP_H = "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 16 16'><path d='M3 11l5-5 5 5' stroke='%2342a5f5' stroke-width='3.0' fill='none' stroke-linecap='round' stroke-linejoin='round'/></svg>\")"
_SVG_DN   = "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 16 16'><path d='M3 5l5 5 5-5' stroke='%23ffffff' stroke-width='2.5' fill='none' stroke-linecap='round' stroke-linejoin='round'/></svg>\")"
_SVG_DN_H = "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 16 16'><path d='M3 5l5 5 5-5' stroke='%2342a5f5' stroke-width='3.0' fill='none' stroke-linecap='round' stroke-linejoin='round'/></svg>\")"

# ─── QSS Blocks ──────────────────────────────────────────────────────────────

GLOBAL_QSS = f"""
* {{
    font-family: 'Segoe UI', 'Malgun Gothic', sans-serif;
}}
QMainWindow, QDialog, QWidget {{
    background-color: {C_BG};
    color: {C_TEXT};
}}
QLabel {{
    font-size: 10pt;
    color: {C_TEXT_DIM};
}}
QGroupBox {{
    font-weight: bold;
    border: 1px solid {C_BORDER};
    margin-top: 10px;
    padding: 10px;
    color: {C_TEXT};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    color: {C_ACCENT};
}}
QLineEdit, QComboBox {{
    background-color: {C_BG_INPUT};
    color: {C_TEXT};
    border: 1px solid {C_BORDER_IN};
    padding: 4px;
    border-radius: 4px;
}}
QComboBox QAbstractItemView {{
    background-color: {C_BG_INPUT};
    color: {C_TEXT};
    selection-background-color: {C_BTN_BLUE3};
}}
QDoubleSpinBox, QSpinBox {{
    background-color: {C_BG_INPUT};
    color: {C_TEXT};
    border: 1px solid {C_BORDER_IN};
    padding: 0px 4px 0px 4px; /* 오른쪽 가드 패딩을 22px로 확장하여 Qt 파서 패딩 버그 및 글자 잘림 완치 */
    border-radius: 4px;
    min-height: 18px;
    height: 18px;
}}
QDoubleSpinBox::up-button, QSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 16px; /* 16px로 콤팩트 다이어트 */
    border-left: 1px solid {C_BORDER_IN};
    border-top-right-radius: 4px;
    background-color: {C_BG_BTN};
}}
QDoubleSpinBox::up-button:hover, QSpinBox::up-button:hover {{
    background-color: {C_BG_BTN_HOV};
}}
QDoubleSpinBox::up-arrow, QSpinBox::up-arrow {{
    image: none; /* 기하학 삼각형 그리기 */
    width: 0;
    height: 0;
    border-left: 3px solid transparent;
    border-right: 3px solid transparent;
    border-bottom: 4px solid #b0b0b0; /* 극도로 샤프한 위 삼각형 */
}}
QDoubleSpinBox::up-arrow:hover, QSpinBox::up-arrow:hover {{
    border-bottom-color: {C_ACCENT};
}}
QDoubleSpinBox::down-button, QSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 16px; /* 16px로 콤팩트 다이어트 */
    border-left: 1px solid {C_BORDER_IN};
    border-bottom-right-radius: 4px;
    background-color: {C_BG_BTN};
}}
QDoubleSpinBox::down-button:hover, QSpinBox::down-button:hover {{
    background-color: {C_BG_BTN_HOV};
}}
QDoubleSpinBox::down-arrow, QSpinBox::down-arrow {{
    image: none; /* 기하학 삼각형 그리기 */
    width: 0;
    height: 0;
    border-left: 3px solid transparent;
    border-right: 3px solid transparent;
    border-top: 4px solid #b0b0b0; /* 극도로 샤프한 아래 삼각형 */
}}
QDoubleSpinBox::down-arrow:hover, QSpinBox::down-arrow:hover {{
    border-top-color: {C_ACCENT};
}}
QPushButton {{
    background-color: {C_BG_BTN};
    color: {C_TEXT};
    padding: 6px 12px;
    border-radius: 4px;
    border: 1px solid {C_BORDER_IN};
    font-weight: bold;
}}
QPushButton:hover  {{ background-color: {C_BG_BTN_HOV}; }}
QPushButton:pressed {{ background-color: {C_BG_BTN_PRS}; }}
QTableWidget, QTreeWidget {{
    background-color: {C_BG_TABLE};
    color: {C_TEXT};
    gridline-color: {C_BORDER_IN};
    border: 1px solid {C_BORDER};
    border-radius: 4px;
}}
QHeaderView::section {{
    background-color: {C_BG_BTN};
    color: {C_TEXT};
    padding: 5px;
    border: 1px solid {C_BORDER_IN};
    font-weight: bold;
}}
QTreeWidget::item:hover {{ background-color: {C_BG_BTN}; }}
QTreeWidget::item:selected {{ background-color: {C_SEL}; color: {C_TEXT}; }}
QRadioButton, QCheckBox {{ color: {C_TEXT}; font-size: 10pt; }}
QSlider::handle:horizontal {{
    background: {C_SLIDER};
    width: 18px;
    margin: -5px 0;
    border-radius: 9px;
}}
QFrame[frameShape="4"] {{ background-color: {C_BORDER_IN}; }}
QTabWidget::pane {{ border: 1px solid {C_BORDER}; }}
QTabBar::tab {{
    background: {C_BG_INPUT};
    color: {C_TEXT_DIM};
}}
QTabBar::tab:selected {{ 
    background: {C_BG_BTN_HOV}; 
    color: {C_TEXT}; 
    border-color: {C_BORDER_IN};
}}
QScrollBar:vertical {{
    background: #1a1a1a;
    width: 12px;
    margin: 0px;
}}
QScrollBar::handle:vertical {{
    background: #606060;
    border-radius: 6px;
    min-height: 24px;
    margin: 2px;
}}
QScrollBar::handle:vertical:hover {{
    background: #808080;
}}
QScrollBar::handle:vertical:pressed {{
    background: #a0a0a0;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
QScrollBar:horizontal {{
    background: #1a1a1a;
    height: 12px;
    margin: 0px;
}}
QScrollBar::handle:horizontal {{
    background: #606060;
    border-radius: 6px;
    min-width: 24px;
    margin: 2px;
}}
QScrollBar::handle:horizontal:hover {{
    background: #808080;
}}
QScrollBar::handle:horizontal:pressed {{
    background: #a0a0a0;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
}}
"""

# 헤더-트리 전용 QSS (XML 에디터 등 독립 위젯에 추가 적용 시 사용)
TREE_QSS = f"""
QTreeWidget {{
    background-color: {C_BG_TABLE};
    color: {C_TEXT_TREE};
    border: 1px solid {C_BORDER};
    font-size: 9pt;
}}
QTreeWidget::item {{
    min-height: 24px;
    padding: 2px;
}}
QTreeWidget::item:hover {{ background-color: {C_BG_BTN_HOV}; }}
QTreeWidget::item:selected {{ background-color: {C_SEL}; color: {C_TEXT}; }}
QTreeWidget QLineEdit {{
    background-color: {C_BG_BTN_HOV};
    color: {C_TEXT};
    border: 1px solid {C_SEL};
    border-radius: 2px;
    padding: 1px 4px;
    margin: 0px;
    font-size: 9pt;
}}
QHeaderView::section {{
    background-color: {C_BG_BTN};
    color: {C_TEXT_MUTED};
    padding: 4px;
    border: none;
    border-bottom: 1px solid {C_BG};
    font-weight: bold;
    font-size: 9pt;
}}
"""

# 실시간 모니터 창 — matplotlib 배경과 맞춤
MONITOR_WINDOW_QSS = f"background-color: {C_BG_DARK}; color: {C_TEXT_DIM};"

# post-processor 전용 constants (whts_postprocess_ui_v2 에서 import)
POSTPROC_BG      = C_BG
POSTPROC_SIDEBAR = "#333333"
POSTPROC_ACCENT  = C_ACCENT_POST
POSTPROC_FONT    = "D2Coding"


def apply_app_theme(app) -> None:
    """QApplication 인스턴스에 전역 다크 테마를 적용합니다.
    launch_control_panel() 내에서 QApplication 생성 직후 한 번만 호출하세요.
    """
    app.setStyleSheet(GLOBAL_QSS)
