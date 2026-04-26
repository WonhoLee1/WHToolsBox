# -*- coding: utf-8 -*-
"""
[WHTOOLS] Post-Processing UI v4
시뮬레이션 완료 후 결과를 탐색하는 고도화 포스트 프로세싱 UI 모듈.

주요 기능:
    - [탭 1] 기구학 분석: 매트릭스 레이아웃(행:변위/속도/가속도, 열:X/Y/Z),
                         위치 멀티 선택(8 코너, 기하 중심, 질량 중심)
    - [탭 2] 구조 해석: PBA, RRG, Bending/Twist 시계열 그래프 + 임계 시점 수직선
    - [탭 3] 2D 컨투어: 시간 슬라이더 + 애니메이션 (Temporal / Overall-Max 전환)
    - 시간 컨트롤: Play/Pause/Stop, 슬라이더 스크러빙, 속도 조절
    - PNG 일괄 저장: 전체 프레임 컨투어 이미지 자동 저장
"""

import sys
import os
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading # [v4.4] 백그라운드 SSR 연산용

from .whts_reporting import compute_ssr_shell_metrics # [v4.4] SSR 코어 엔진 연동
from .whts_postprocess_engine import (
    get_contour_grid_data,
    apply_psr_interpolation,
    apply_jax_kirchhoff_ssr,
    _JAX_AVAILABLE,
    extract_global_summary_data
)

# Matplotlib 백엔드 (기본: QtAgg(PySide), 실패 시 TkAgg 폴백)
import matplotlib
try:
    # [v4.3] QtAgg 시도 전, 화면 감지 오류 여부를 확인하기 위해 dummy figure 생성 시도
    matplotlib.use("QtAgg")
    import matplotlib.pyplot as plt
    # 실제로 창을 띄우지 않고 백엔드가 작동하는지 간단히 확인
    _ = plt.figure(figsize=(1,1))
    plt.close(_)
except Exception as e:
    # 화면 감지 오류(qpa) 또는 라이브러리 부재 시 TkAgg로 강제 전환
    print(f">> [System] QtAgg 백엔드 초기화 실패 (오류: {e}). TkAgg로 전환합니다.")
    matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.figure as mfig
from matplotlib.figure import Figure # Primary
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# [v4.3] 독립 확장 모듈 임포트
try:
    try:
        from .mpl_extension import WHToolsPlotManager, PlotExporter, ExportDialog
    except (ImportError, ValueError):
        from mpl_extension import WHToolsPlotManager, PlotExporter, ExportDialog
    _EXT_AVAILABLE = True
except ImportError:
    _EXT_AVAILABLE = False

# [v4.1] 폰트 설정 (D2Coding 우선, 없으면 맑은 고딕)
_DETECTOR_CACHED_FONT = None

def get_ui_font(size=9, bold=False):
    global _DETECTOR_CACHED_FONT
    import tkinter.font as tkf
    if _DETECTOR_CACHED_FONT is None:
        try:
            families = tkf.families()
            if 'D2Coding' in families:
                _DETECTOR_CACHED_FONT = "D2Coding"
            elif 'D2Coding' in " ".join(families): # [D2Coding v.1.3] 등의 버리에이션 대응
                for f in families:
                    if "D2Coding" in f:
                        _DETECTOR_CACHED_FONT = f
                        break
            else:
                _DETECTOR_CACHED_FONT = "Malgun Gothic"
        except:
            _DETECTOR_CACHED_FONT = "Malgun Gothic"
            
    return (_DETECTOR_CACHED_FONT, size, "bold" if bold else "normal")

plt.rcParams['font.family'] = 'Malgun Gothic' # Matplotlib은 여전히 맑은 고딕 (D2Coding은 그래프 가독성 문제 가능성)
plt.rcParams['axes.unicode_minus'] = False 

try:
    from PIL import Image, ImageTk
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

try:
    from ttkthemes import ThemedStyle
    _THEMES_AVAILABLE = True
except ImportError:
    _THEMES_AVAILABLE = False


class PostProcessingUI(tk.Toplevel):
    """
    [WHTOOLS v4] 시뮬레이션 완료 후 결과를 탐색하는 고도화 포스트 프로세싱 UI.

    Args:
        parent_sim: DropSimulator 인스턴스 (데이터 소스)
    """

    # 위치 레이블 상수 (Legacy Std: Width-Height-Depth)
    # X(-/+) -> L/R (Left/Right), Y(-/+) -> B/T (Bottom/Top), Z(-/+) -> R/F (Rear/Front)
    LOCATION_LABELS = [
        "L-B-R", "L-B-F",  # (-x, -y, -z), (-x, -y, +z)
        "L-T-R", "L-T-F",  # (-x, +y, -z), (-x, +y, +z)
        "R-B-R", "R-B-F",  # (+x, -y, -z), (+x, -y, +z)
        "R-T-R", "R-T-F",  # (+x, +y, -z), (+x, +y, +z)
        "Geo Center", "Mass Center (CoM)"
    ]

    def __init__(self, parent_sim, master=None):
        super().__init__(master=master)
        self.sim = parent_sim
        self.title("WHTOOLS Post-Processing Explorer v4")
        self.geometry("1160x820")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # ---- 애니메이션 상태 변수 ----
        self._anim_running = False
        self._anim_job = None
        self._anim_speed_ms = 80
        self._total_frames = 0
        self._current_frame = 0
        self._plot_all_blocks_var = tk.BooleanVar(value=False) # [v4.8.4] 상세 플롯 모드

        # ---- [v4.3] 확장 모듈 매니저 ----
        if _EXT_AVAILABLE:
            self.plot_manager = WHToolsPlotManager(self)
        else:
            self.plot_manager = None

        # ---- 가시화 모드 및 SSR 추가 ----
        self._ssr_mode_var = tk.BooleanVar(value=False)
        self._show_pba_var = tk.BooleanVar(value=True) # [v4.1]

        # ---- 선택 변수 ----
        self._loc_vars = [tk.BooleanVar(value=(i < 2)) for i in range(10)]
        self._kin_dtype_vars = {
            "displacement": tk.BooleanVar(value=True),
            "velocity":     tk.BooleanVar(value=False),
            "acceleration": tk.BooleanVar(value=False)
        }
        self._coord_system_var = tk.StringVar(value="Global") # [v4.2] Global/Local 구분
        self._axis_vars = {
            "X": tk.BooleanVar(value=False),
            "Y": tk.BooleanVar(value=False),
            "Z": tk.BooleanVar(value=True),
        }
        self._contour_mode_var = tk.StringVar(value="temporal")
        self._contour_metric_vars = {
            "bend":    tk.BooleanVar(value=True),
            "twist":   tk.BooleanVar(value=False),
            "rrg":     tk.BooleanVar(value=False),
            "angle":   tk.BooleanVar(value=False),
            "bend_x":  tk.BooleanVar(value=False),
            "bend_y":  tk.BooleanVar(value=False),
            "s_bend":  tk.BooleanVar(value=False),
            "s_twist": tk.BooleanVar(value=False),
            "moment":  tk.BooleanVar(value=False),
            "energy":  tk.BooleanVar(value=False)
        }

        comp_list = sorted(list(self.sim.metrics.keys()))
        self._comp_var = tk.StringVar(value=comp_list[0] if comp_list else "")

        # [v4.1] 실시간 연동 플래그 및 임베디드 캔버스
        self._live_sync_var = tk.BooleanVar(value=False)
        self._current_tab_name = "Kinematics"
        self._plot_window_mode = tk.StringVar(value="location") 
        self._fig_embed = None
        self._canvas_embed = None

        # [v4.3] 매트릭스 팝업창 전용 변수 (Scale 조절용)
        self._matrix_vmin_var = tk.StringVar(value="auto")
        self._matrix_vmax_var = tk.StringVar(value="auto")
        
        # [v4.4] SSR 재계산 결과 저장소 (휘발성)
        self._ssr_data_store = {} # comp_name -> {step_idx: ssr_dict}

        self._set_ui_theme("breeze")
        
        self._build_ui()
        self._update_frame_count()

    def _set_ui_theme(self, theme_name):
        """[v4.2] UI 테마 변경 및 스타일 재적용"""
        if _THEMES_AVAILABLE:
            try:
                self._style = ThemedStyle(self)
                self._style.set_theme(theme_name)
                # 다크 테마 감지 시 차트 배경 자동 조절
                if "dark" in theme_name.lower() or "equilux" in theme_name.lower():
                    plt.style.use('dark_background')
                else:
                    plt.style.use('default')
                
                # [v4.2] 한글 깨짐 방지 설정 상시 적용 (Matplotlib 전역 설정)
                plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우 표준 한글 폰트
                plt.rcParams['axes.unicode_minus'] = False     # 마이너스 기호 깨짐 방지
            except:
                self._style = ttk.Style()
                self._style.theme_use("clam")  # 안전한 기본 테마로 복구
        else:
            self._style = ttk.Style()
            self._style.theme_use("clam")
            
        self._apply_custom_styles()

    def _apply_custom_styles(self):
        """[v4.2] 현재 테마 고유의 스타일을 최대한 존중하며 폰트 등 필수 요소만 조정"""
        f_name, f_size, _ = get_ui_font(9)
        self._style.configure(".", font=(f_name, f_size))
        
        # 테마 기본 배경색 추출
        current_bg = self._style.lookup("TFrame", "background")
        current_fg = self._style.lookup("TFrame", "foreground")
        
        # LabelFrame: 테마별 배경 적용
        self._style.configure("TLabelframe", background=current_bg)
        self._style.configure("TLabelframe.Label", font=(f_name, f_size, "bold"), background=current_bg)
        
        # Checkbutton / Radiobutton
        self._style.configure("TCheckbutton", background=current_bg)
        self._style.configure("TRadiobutton", background=current_bg)
        
        # 사이드바 및 인포 패널 배경 업데이트 (실시간 변경 대응)
        if hasattr(self, "_side_menu"):
            self._side_menu.config(bg=current_bg)
        if hasattr(self, "_side_logo_label"):
            self._side_logo_label.config(bg=current_bg)
        if hasattr(self, "_info_f"):
            self._info_f.config(bg=current_bg)
            for child in self._info_f.winfo_children():
                if isinstance(child, tk.Label):
                    child.config(bg=current_bg)
        
        # [NEW v4.2] 모든 위젯에 대해 재귀적으로 폰트 적용
        self.option_add("*Font", (f_name, f_size))
        self._apply_font_recursive(self, (f_name, f_size))

    def _apply_font_recursive(self, widget, font_tuple):
        """지정한 위젯과 모든 자식 위젯에 대해 폰트를 강제로 적용합니다."""
        try:
            # ttk 위젯은 보통 스타일을 따르지만, 일부 테마에서는 개별 설정이 필요한 경우가 있음
            widget.configure(font=font_tuple)
        except (tk.TclError, AttributeError):
            # 폰트 속성이 없는 위젯(Frame 등)은 무시
            pass
            
        for child in widget.winfo_children():
            self._apply_font_recursive(child, font_tuple)

    # ==============================================================
    # UI 빌드
    # ==============================================================

    def _build_ui(self):
        """전체 UI 레이아웃을 구성합니다. (Side-Menu 구조)"""
        # 상단 메뉴바 추가
        self._build_menu()

        # 메인 컨테이너 (Side Menu + Content)
        self._main_container = tk.Frame(self)
        self._main_container.pack(fill="both", expand=True)

        # 1. 사이드 메뉴 프레임 (배경색은 테마와 연동됨)
        self._side_menu = tk.Frame(self._main_container, width=180)
        self._side_menu.pack(side="left", fill="y")
        self._side_menu.pack_propagate(False)

        # 2. 내용 프레임 (Content Area)
        self._content_area = tk.Frame(self._main_container)
        self._content_area.pack(side="right", fill="both", expand=True)

        # 탭 프레임 생성 (Content Area 내부에 중첩)
        self._tabs = {}
        for name in ["Kinematics", "Structural", "Contour"]:
            self._tabs[name] = ttk.Frame(self._content_area)

        # 메뉴 버튼 생성
        self._build_side_menu_buttons()

        # 공통 시간 컨트롤 패널 (하단 고정)
        self._build_time_control_panel()

        # 초기 탭 설정
        self._switch_tab("Kinematics")

        # 각 탭의 실제 내용 빌드
        self._build_kinematics_tab(self._tabs["Kinematics"])
        self._build_structural_tab(self._tabs["Structural"])
        self._build_contour_tab(self._tabs["Contour"])

    def _build_menu(self):
        """상단 메인 메뉴바를 구성합니다."""
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        # [Menu 1] File
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="파일 (File)", menu=file_menu)
        file_menu.add_command(label="데이터 다시 로드", command=lambda: messagebox.showinfo("Info", "데이터를 다시 읽어옵니다."))
        file_menu.add_separator()
        file_menu.add_command(label="종료", command=self.on_close)

        # [Menu 2] UI 테마 (Theme) - 카테고리별 분류 적용
        theme_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="UI 테마 (Theme)", menu=theme_menu)
        
        if _THEMES_AVAILABLE:
            categories = {
                "Modern (추천)": ["breeze", "arc", "adapta", "clearlooks", "plastik", "yaru", "ubuntu"],
                "Dark Mode": ["equilux", "black"],
                "Classic / OS": ["vista", "xpnative", "winxpblue", "clam", "alt", "default", "classic"],
                "Artistic": ["aquativo", "radiance", "elegance", "kroc", "smog"],
                "Color Sets": ["scidblue", "scidgreen", "scidgrey", "scidmint", "scidpink", "scidpurple", "scidsand"]
            }
            
            for cat_name, t_list in categories.items():
                sub = tk.Menu(theme_menu, tearoff=0)
                theme_menu.add_cascade(label=cat_name, menu=sub)
                for t_name in t_list:
                    sub.add_command(label=t_name.capitalize(), 
                                    command=lambda tn=t_name: self._set_ui_theme(tn))
        else:
            theme_menu.add_command(label="standard (Built-in)", command=lambda: self._set_ui_theme("clam"))
            theme_menu.add_separator()
            theme_menu.add_command(label="[Notice] ttkthemes가 설치되지 않음", state="disabled")

        # [Menu 3] Plot Options
        plot_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="그래프 옵션 (Plot Options)", menu=plot_menu)
        
        # Font Submenu
        font_menu = tk.Menu(plot_menu, tearoff=0)
        plot_menu.add_cascade(label="폰트 및 크기", menu=font_menu)
        font_menu.add_command(label="Standard (Malgun Gothic)", command=lambda: self._set_plot_font("Malgun Gothic"))
        font_menu.add_command(label="Consolas", command=lambda: self._set_plot_font("Consolas"))
        font_menu.add_separator()
        font_menu.add_command(label="Font Size: 8pt", command=lambda: self._set_plot_fontsize(8))
        font_menu.add_command(label="Font Size: 10pt", command=lambda: self._set_plot_fontsize(10))

        # Theme Submenu
        g_theme_menu = tk.Menu(plot_menu, tearoff=0)
        plot_menu.add_cascade(label="그래프 테마 (Matplotlib)", menu=g_theme_menu)
        for theme in sorted(plt.style.available):
            g_theme_menu.add_command(label=theme, command=lambda t=theme: self._set_theme(t))

        plot_menu.add_separator()
        plot_menu.add_command(label="모든 그래프 창 닫기", command=lambda: plt.close('all'))
        plot_menu.add_command(label="그래프 앞으로 가져오기", command=self._bring_plots_to_front)

        # [Menu 4] Help / Tools
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="도구 / 도움말", menu=tools_menu)
        tools_menu.add_command(label="📊 창 정렬 (Bring to Front)", command=self._bring_plots_to_front)
        tools_menu.add_command(label="🧹 모든 어노테이션 제거", command=self._on_clear_all_annotations)
        tools_menu.add_separator()
        tools_menu.add_command(label="버전 정보", command=lambda: messagebox.showinfo("About", "WHTOOLS Post-Explorer v4.3.1\nCoordinate Synced Version"))

    def _build_side_menu_buttons(self):
        """사이드 메뉴바에 버튼을 배치합니다."""
        # [v4.2] 사이드바 로고 이미지 로드 및 표시
        base_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(base_dir, "resources", "sidebar_logo.png"),
            os.path.join(base_dir, "sidebar_logo.png"),
            os.path.join(os.path.dirname(base_dir), "run_drop_simulator", "resources", "sidebar_logo.png")
        ]
        
        logo_path = None
        for p in possible_paths:
            if os.path.exists(p):
                logo_path = p
                break

        if logo_path and _PIL_AVAILABLE:
            try:
                img = Image.open(logo_path)
                w, h = img.size
                # 사이드바 너비(180) 보다 약간 작은 160px로 리사이징
                target_w = 160
                scale = target_w / w
                target_h = int(h * scale)
                img = img.resize((target_w, target_h), Image.LANCZOS)
                self._sidebar_logo_img = ImageTk.PhotoImage(img)
                self._side_logo_label = tk.Label(self._side_menu, image=self._sidebar_logo_img)
                self._side_logo_label.pack(pady=20)
            except Exception:
                self._side_logo_label = tk.Label(self._side_menu, text="WHTOOLS v4", 
                                                 font=("Arial", 12, "bold"), pady=20)
                self._side_logo_label.pack()
        else:
            self._side_logo_label = tk.Label(self._side_menu, text="WHTOOLS v4", 
                                             font=("Arial", 12, "bold"), pady=20)
            self._side_logo_label.pack()
        
        btns = [
            ("📊 모션 분석", "Kinematics"),
            ("📉 구조 해석",   "Structural"),
            ("🌈 필드 컨투어",  "Contour")
        ]
        self._menu_btns = {}
        for txt, target in btns:
            btn = tk.Button(self._side_menu, text=txt, font=("Malgun Gothic", 10), 
                            fg="white", bg="#34495e", relief="flat", height=2,
                            command=lambda t=target: self._switch_tab(t))
            btn.pack(fill="x", padx=10, pady=5)
            self._menu_btns[target] = btn

        # [v4.2] 시뮬레이션 정보 표시 (사이드바 하단)
        self._info_f = tk.Frame(self._side_menu)
        self._info_f.pack(side="bottom", fill="x", pady=20)
        
        tk.Label(self._info_f, text="[ Simulation Info ]", font=("Arial", 8)).pack()
        
        mode = self.sim.config.get('drop_mode', 'N/A')
        direction = self.sim.config.get('drop_direction', 'N/A')
        
        tk.Label(self._info_f, text=f"Mode: {mode}", font=("Consolas", 10, "bold")).pack(pady=2)
        tk.Label(self._info_f, text=f"Dir: {direction}", font=("Consolas", 10, "bold")).pack(pady=2)
        
        # 초기 테마 로드 시 배경색 동기화를 위해 한 번 더 호출
        self._apply_custom_styles()

    def _switch_tab(self, target_name):
        """선택한 탭으로 화면을 전환합니다."""
        self._current_tab_name = target_name
        for name, frame in self._tabs.items():
            if name == target_name:
                frame.pack(fill="both", expand=True)
                self._menu_btns[name].config(bg="#1abc9c") # Active color
            else:
                frame.pack_forget()
                self._menu_btns[name].config(bg="#34495e") # Default color

    def _set_plot_font(self, font_name):
        plt.rcParams['font.family'] = font_name
        messagebox.showinfo("Font Change", f"Matplotlib 폰트가 {font_name}으로 설정되었습니다.")

    def _set_plot_fontsize(self, sz):
        plt.rcParams['font.size'] = sz
        messagebox.showinfo("Size Change", f"Matplotlib 폰트 크기가 {sz}pt로 설정되었습니다.")

    def _set_theme(self, theme_name):
        """Matplotlib의 스타일 테마를 변경합니다."""
        plt.style.use(theme_name)
        messagebox.showinfo("Theme Change", f"그래프 테마가 '{theme_name}'으로 설정되었습니다.\n새창부터 적용됩니다.")

    def _set_backend(self, backend_name):
        pass # 상단 사용자 요청으로 기능 제거

    def _bring_plots_to_front(self):
        # Matplotlib 창들을 최상단으로 올리기 (백엔드 종속적)
        for i in plt.get_fignums():
            plt.figure(i).canvas.manager.window.attributes('-topmost', 1)
            plt.figure(i).canvas.manager.window.attributes('-topmost', 0)
    
    # ==============================================================
    # [공통] 시간 컨트롤 패널
    # ==============================================================

    def _build_time_control_panel(self):
        """
        시뮬레이션 시간 축을 제어하는 공통 패널입니다.

        구성:
            - 시간 슬라이더 (Time Scrubber): 드래그로 특정 시점 선택
            - Play / Pause / Stop 버튼
            - 속도 조절 콤보박스 (0.25x ~ 4x)
            - 현재 시간 레이블 (t=xxx.xxxx s)
            - 키보드 단축키: ← → (1프레임), Space (Play/Pause)
        """
        ctrl_frame = ttk.LabelFrame(self, text="  시간 컨트롤 (모든 탭 공통 적용)  ")
        ctrl_frame.pack(fill="x", padx=8, pady=(4, 0))

        # --- Row 1: 슬라이더 ---
        row1 = ttk.Frame(ctrl_frame)
        row1.pack(fill="x", padx=5, pady=2)

        ttk.Label(row1, text="Time:", font=("Consolas", 9)).pack(side="left")

        self._time_var = tk.DoubleVar(value=0.0)
        self._time_slider = ttk.Scale(
            row1, from_=0, to=100, orient="horizontal",
            variable=self._time_var, command=self._on_time_slider_change
        )
        self._time_slider.pack(side="left", fill="x", expand=True, padx=5)

        self._time_lbl = ttk.Label(
            row1, text="t = 0.0000 s  [    0 /    0]",
            font=("Consolas", 9), width=28
        )
        self._time_lbl.pack(side="left", padx=5)

        # --- Row 2: 재생 컨트롤 ---
        row2 = ttk.Frame(ctrl_frame)
        row2.pack(fill="x", padx=5, pady=(0, 4))

        self._play_btn = ttk.Button(row2, text="▶ Play",   width=9, command=self._on_play)
        self._play_btn.pack(side="left", padx=3)
        ttk.Button(row2, text="⏸ Pause", width=9, command=self._on_pause).pack(side="left", padx=3)
        ttk.Button(row2, text="⏹ Stop",  width=9, command=self._on_stop ).pack(side="left", padx=3)

        ttk.Separator(row2, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Label(row2, text="속도:").pack(side="left")
        self._speed_var = tk.StringVar(value="1x")
        speed_cb = ttk.Combobox(row2, textvariable=self._speed_var,
                                values=["0.25x", "0.5x", "1x", "2x", "4x"],
                                state="readonly", width=6)
        speed_cb.pack(side="left", padx=3)
        speed_cb.bind("<<ComboboxSelected>>", self._on_speed_change)

        ttk.Separator(row2, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Label(row2, text="← / → : 1프레임 이동   |   Space : Play/Pause",
                  font=("NanumGothic", 8), foreground="gray").pack(side="left", padx=5)

        # 키보드 단축키
        self.bind("<Left>",  lambda e: self._step_frame(-1))
        self.bind("<Right>", lambda e: self._step_frame(+1))
        self.bind("<space>", lambda e: self._on_play() if not self._anim_running else self._on_pause())

    # ==============================================================
    # [탭 1] 기구학 분석
    # ==============================================================

    def _build_kinematics_tab(self, parent):
        """[v4.2] 모션 분석 탭: 레이아웃 개편 (스크롤 지원)."""
        main_holder = ttk.Frame(parent)
        main_holder.pack(fill="both", expand=True)

        # [v4.8.8] 전체 스크롤을 위한 캔버스 생성 및 가로 너비 동기화
        k_canvas = tk.Canvas(main_holder, highlightthickness=0)
        k_vscroll = ttk.Scrollbar(main_holder, orient="vertical", command=k_canvas.yview)
        container = ttk.Frame(k_canvas)
        
        k_canvas.create_window((0, 0), window=container, anchor="nw")
        k_canvas.configure(yscrollcommand=k_vscroll.set)
        
        k_vscroll.pack(side="right", fill="y")
        k_canvas.pack(side="left", fill="both", expand=True)

        # 가로 너비 동기화 로직 추가
        def _on_k_canvas_configure(event, canvas=k_canvas, frame=container):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(canvas.find_withtag("all")[0], width=event.width)

        k_canvas.bind("<Configure>", _on_k_canvas_configure)
        
        def _on_mousewheel_k(event):
            k_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        k_canvas.bind_all("<MouseWheel>", _on_mousewheel_k)

        # -----------------------------------------------------------------
        # [1단계] 최상단: 분석 좌표계 선택 (가로 배치)
        # -----------------------------------------------------------------
        top_f = ttk.LabelFrame(container, text="  0. 분석 좌표계 선택 (Coordinate System)  ")
        top_f.pack(fill="x", pady=5)
        
        coord_inner = ttk.Frame(top_f)
        coord_inner.pack(pady=8)
        ttk.Radiobutton(coord_inner, text="Global (절대 좌표계: World 기준 위치/속도)", 
                        variable=self._coord_system_var, value="Global").pack(side="left", padx=40)
        ttk.Radiobutton(coord_inner, text="Local (상대 변위계: t=0 대비 변위/상대속도)",  
                        variable=self._coord_system_var, value="Local").pack(side="left", padx=40)

        # (0번 그룹은 상단 좌측에 고정)
        top_f.grid(row=0, column=0, columnspan=2, sticky="nw", pady=5)
        
        # --- (1) 데이터 종류 (dtype_f) ---
        dtype_f = ttk.LabelFrame(container, text="  1. 데이터 종류 (Data Types)  ")
        dtype_f.grid(row=1, column=0, sticky="nw", padx=5, pady=5)
        
        for key, txt in [("displacement", "위치 / 변위 (Disp)"),
                          ("velocity",     "속도 (Velocity)"),
                          ("acceleration", "가속도 (Acceleration)")]:
            ttk.Checkbutton(dtype_f, text=txt, variable=self._kin_dtype_vars[key]).pack(anchor="w", padx=20, pady=3)
        
        btn_dtype_f = ttk.Frame(dtype_f)
        btn_dtype_f.pack(fill="x", padx=15, pady=5)
        ttk.Button(btn_dtype_f, text="전부 선택", width=12, 
                   command=lambda: [v.set(True) for v in self._kin_dtype_vars.values()]).pack(side="left", padx=2)
        ttk.Button(btn_dtype_f, text="선택 해제", width=12, 
                   command=lambda: [v.set(False) for v in self._kin_dtype_vars.values()]).pack(side="left", padx=2)

        # --- (2) 분석 축 (axis_f) ---
        axis_f = ttk.LabelFrame(container, text="  2. 분석 축 (Axis)  ")
        axis_f.grid(row=1, column=1, sticky="nw", padx=5, pady=5)
        
        axis_inner = ttk.Frame(axis_f)
        axis_inner.pack(expand=True, pady=10)
        for ax in ["X", "Y", "Z"]:
            ttk.Checkbutton(axis_inner, text=ax + " 축", variable=self._axis_vars[ax]).pack(side="left", padx=20)
        
        btn_ax_f = ttk.Frame(axis_f)
        btn_ax_f.pack(fill="x", padx=15, pady=5)
        ttk.Button(btn_ax_f, text="전부 선택", width=12, 
                   command=lambda: [v.set(True) for v in self._axis_vars.values()]).pack(side="left", padx=2)
        ttk.Button(btn_ax_f, text="선택 해제", width=12, 
                   command=lambda: [v.set(False) for v in self._axis_vars.values()]).pack(side="left", padx=2)

        # --- (3) 추출 위치 (loc_f) ---
        loc_f = ttk.LabelFrame(container, text="  3. 데이터 추출 위치 (Location)  ")
        loc_f.grid(row=2, column=0, sticky="nw", padx=5, pady=5)
        
        grid_f = ttk.Frame(loc_f)
        grid_f.pack(fill="x", padx=15, pady=8)
        for idx in range(8):
            r, c = divmod(idx, 4) 
            ttk.Checkbutton(grid_f, text=self.LOCATION_LABELS[idx], variable=self._loc_vars[idx]).grid(row=r, column=c, sticky="w", padx=10, pady=4)
        
        ttk.Checkbutton(grid_f, text="Geometry Center (박스중심)", variable=self._loc_vars[8]).grid(row=2, column=0, columnspan=2, sticky="w", padx=10, pady=4)
        ttk.Checkbutton(grid_f, text="Mass Center (질량중심: CoM)", variable=self._loc_vars[9]).grid(row=2, column=2, columnspan=2, sticky="w", padx=10, pady=4)
        
        btn_loc_f = ttk.Frame(loc_f)
        btn_loc_f.pack(fill="x", padx=15, pady=5)
        
        ttk.Label(btn_loc_f, text="그룹 선택 (Group):").pack(side="left", padx=2)
        self._loc_group_var = tk.StringVar(value="All")
        group_cb = ttk.Combobox(btn_loc_f, textvariable=self._loc_group_var,
                                values=["All", "Left", "Right", "Top", "Bottom", "Front", "Rear"],
                                state="readonly", width=8)
        group_cb.pack(side="left", padx=2)
        
        ttk.Button(btn_loc_f, text="선택", width=8, 
                   command=lambda: self._apply_loc_group(True)).pack(side="left", padx=2)
        ttk.Button(btn_loc_f, text="해제", width=8, 
                   command=lambda: self._apply_loc_group(False)).pack(side="left", padx=2)

        # --- (4) 결과 창 분류 및 실행 (exec_f) ---
        exec_f = ttk.LabelFrame(container, text="  4. 결과 창 분류 및 실행  ")
        exec_f.grid(row=2, column=1, sticky="nw", padx=10, pady=5)
        
        win_f = ttk.Frame(exec_f)
        win_f.pack(fill="x", pady=10)
        for val, txt in [("location", "기본 (Default: 추출 위치별 매트릭스 분류)"), 
                         ("data_type", "데이터 종류별 창 분류 (P/V/A)"), 
                         ("axis", "분축 축별 창 분류 (X/Y/Z)    ")]:
            ttk.Radiobutton(win_f, text=txt, variable=self._plot_window_mode, value=val).pack(anchor="w", padx=20, pady=4)
            
        ttk.Button(exec_f, text="📊 매트릭스 그래프 생성", command=self._on_plot_kinematics, 
                   width=30).pack(pady=20, padx=15, ipady=15)


    # ==============================================================
    # [탭 2] 구조 해석
    # ==============================================================

    def _build_structural_tab(self, parent):
        """[v4] 구조 해석 탭: 지표 선택 + 부품 선택 + 상세 도움말 (전체 스크롤 지원)."""
        main_holder = ttk.Frame(parent)
        main_holder.pack(fill="both", expand=True)

        # 1. 전체 스크롤을 위한 캔버스 생성
        canvas = tk.Canvas(main_holder, highlightthickness=0)
        v_scroll = ttk.Scrollbar(main_holder, orient="vertical", command=canvas.yview)
        main_f = ttk.Frame(canvas)
        
        # 가로 방향도 창 크기에 맞게 자동 조절
        canvas.create_window((0, 0), window=main_f, anchor="nw")
        canvas.configure(yscrollcommand=v_scroll.set)
        
        v_scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # 윈도우 크기 변경 시 스크롤 영역 갱신 및 내부 프레임 너비 조절
        def _on_canvas_configure(event, canvas=canvas, frame=main_f):
            canvas.configure(scrollregion=canvas.bbox("all"))
            # 내부 프레임 너비를 캔버스 너비에 맞춤 (가로 확장 대응)
            canvas.itemconfig(canvas.find_withtag("all")[0], width=event.width)

        canvas.bind("<Configure>", _on_canvas_configure)
        
        # 마우스 휠 지원
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # --- 실제 UI 컨텐츠 구성 (이후 main_f 에 배치) ---
        
        # 1. 상단 컨트롤 영역
        ctrl = ttk.Frame(main_f)
        ctrl.pack(fill="x", pady=5)

        # (A) 지표 선택 그룹
        metric_f = ttk.LabelFrame(ctrl, text=" 1. 분석 지표 선택 ")
        metric_f.pack(side="left", fill="y", padx=5)

        opts = [
            ("PBA Magnitude (Principal Bending Axis)", "pba_magnitude", "pba_concept.png"),
            ("RRG (High Fidelity Twist / Local Stress)", "rrg_max", "rrg_concept.png"),
            ("Bending X (Spanwise: X축 방향 휨)", "bend_x_overall", "distortion_concept.png"),
            ("Bending Y (Ribwise: Y축 방향 휨)",  "bend_y_overall", "distortion_concept.png"),
            ("Bending Z (Overall Tilt: 전반적 기울기)", "bend_overall",    "distortion_concept.png"),
            ("Twist Angle (Overall Torsion)", "twist_overall", "distortion_concept.png"),
            ("Bending Stress (BS, MPa 단위 굽힘 응력)", "s_bend_overall", "rrg_concept.png"),
            ("Torsional Stress (TS, MPa 단위 비틀림 응력)", "s_twist_overall", "rrg_concept.png"),
            ("Bending Moment (BM, N·m 단위 굽힘 모멘트)", "moment_overall", "distortion_concept.png"),
            ("Global Torsion Index (GTI)", "gti_overall", "rrg_concept.png"),
            ("Global Bending Index (GBI)", "gbi_overall", "distortion_concept.png"),
            ("Total Strain Energy (TSE, J 단위 흡수 에너지)", "tse_overall", "distortion_concept.png"),
        ]
        self._struct_metric_vars = {}
        for txt, key, help_img in opts:
            row = ttk.Frame(metric_f)
            row.pack(fill="x", padx=2, pady=1)
            
            var = tk.BooleanVar(value=key in ("rrg_max", "pba_magnitude"))
            self._struct_metric_vars[key] = (var, txt)
            ttk.Checkbutton(row, text=txt, variable=var).pack(side="left", padx=2)
            
            btn = ttk.Button(row, text="?", width=2)
            btn.pack(side="right", padx=2)
            btn.bind("<Button-1>", lambda e, t=txt, i=help_img: self._show_metric_help(e, t, i))

        btn_m_f = ttk.Frame(metric_f)
        btn_m_f.pack(fill="x", side="bottom", padx=5, pady=5)
        ttk.Button(btn_m_f, text="전체 선택", width=12,
                   command=lambda: [v[0].set(True) for v in self._struct_metric_vars.values()]).pack(side="left", padx=2)
        ttk.Button(btn_m_f, text="전체 해제", width=12,
                   command=lambda: [v[0].set(False) for v in self._struct_metric_vars.values()]).pack(side="left", padx=2)

        # (B) 대상 부품 선택 그룹
        comp_f = ttk.LabelFrame(ctrl, text=" 2. 대상 부품 선택 ")
        comp_f.pack(side="left", fill="y", padx=5)
        
        # Header: Analyzer Button & SSR Checkbox
        c_header = ttk.Frame(comp_f)
        c_header.pack(fill="x", side="top", padx=5, pady=2)
        
        ttk.Button(c_header, text="🔍 SSR Analyzer", 
                   command=self._open_ssr_analyzer, width=18, style="Accent.TButton").pack(side="left", padx=2)
        
        ttk.Checkbutton(c_header, text="SSR 모드", variable=self._ssr_mode_var).pack(side="left", padx=5)
        ttk.Button(c_header, text="?", width=2, 
                   command=lambda: self._show_metric_help(None, "High Fidelity SSR", "ssr_concept")).pack(side="left")

        # Body: Scrollable Component List
        c_scroll_f = ttk.Frame(comp_f)
        c_scroll_f.pack(fill="both", expand=True, padx=2, pady=2)
        
        c_canvas = tk.Canvas(c_scroll_f, width=280, height=150, highlightthickness=0)
        c_scrollbar = ttk.Scrollbar(c_scroll_f, orient="vertical", command=c_canvas.yview)
        c_inner = ttk.Frame(c_canvas)
        
        c_inner.bind("<Configure>", lambda e: c_canvas.configure(scrollregion=c_canvas.bbox("all")))
        c_canvas.create_window((0, 0), window=c_inner, anchor="nw")
        c_canvas.configure(yscrollcommand=c_scrollbar.set)
        
        c_canvas.pack(side="left", fill="both", expand=True)
        c_scrollbar.pack(side="right", fill="y")

        self._comp_selection_vars = {}
        comp_list = sorted(list(self.sim.metrics.keys()))
        for c in comp_list:
            var = tk.BooleanVar(value=True)
            self._comp_selection_vars[c] = var
            ttk.Checkbutton(c_inner, text=c, variable=var).pack(anchor="w", padx=5)

        # [v4.8.4] 플롯 상세 옵션 (Max vs All) - 위치 이동: 부품 리스트 하단
        detail_f = ttk.LabelFrame(comp_f, text=" [v4.8.4] 플롯 상세 옵션 ")
        detail_f.pack(fill="x", padx=5, pady=5)
        
        ttk.Radiobutton(detail_f, text="Max. of Blocks (기본: 상한 출력)", 
                        variable=self._plot_all_blocks_var, value=False).pack(side="top", anchor="w", padx=10, pady=2)
        ttk.Radiobutton(detail_f, text="All Blocks (상세: 모든 블록 출력)", 
                        variable=self._plot_all_blocks_var, value=True).pack(side="top", anchor="w", padx=10, pady=2)
        
        ttk.Label(detail_f, text="* All Blocks 선택 시 부품 1개 권장", 
                  foreground="gray", font=("Arial", 8)).pack(side="top", anchor="w", padx=10)

        # Footer: Selection Buttons & Action Button
        c_footer = ttk.Frame(comp_f)
        c_footer.pack(fill="x", side="bottom", padx=5, pady=5)
        
        btn_sel_f = ttk.Frame(c_footer)
        btn_sel_f.pack(side="left")
        ttk.Button(btn_sel_f, text="전체 선택", width=9,
                   command=lambda: [v.set(True) for v in self._comp_selection_vars.values()]).pack(side="left", padx=1)
        ttk.Button(btn_sel_f, text="전체 해제", width=9,
                   command=lambda: [v.set(False) for v in self._comp_selection_vars.values()]).pack(side="left", padx=1)
        
        ttk.Button(c_footer, text="📈 그래프 생성", 
                   command=self._on_plot_structural, width=15).pack(side="right", padx=2)

        # 3. 임계 시점 정보 표시
        info_f = ttk.LabelFrame(main_f, text=" 3. 자동 검출 임계 시점 (Critical Events) ")
        info_f.pack(fill="both", expand=True, padx=5, pady=5)
        
        self._critical_text = tk.Text(info_f, height=5, font=("Consolas", 9), 
                                     state="disabled", bg="#f0f0f0")
        self._critical_text.pack(fill="both", expand=True, padx=5, pady=5)
        self._refresh_critical_info()

        # 4. 글로벌 지표 요약
        summary_f = ttk.LabelFrame(main_f, text=" 4. 전체 컴포넌트 지표 요약 (Global Integrity) ")
        summary_f.pack(fill="x", padx=5, pady=5)
        
        tree_container = ttk.Frame(summary_f)
        tree_container.pack(fill="both", expand=True, padx=5, pady=5)

        cols = ("comp", "pba_peak", "rrg_peak", "max_stress", "total_energy", "gti", "gbi", "status")
        self._global_summary_tree = ttk.Treeview(tree_container, columns=cols, 
                                                 show="headings", height=4)
        
        vsb = ttk.Scrollbar(tree_container, orient="vertical", command=self._global_summary_tree.yview)
        hsb = ttk.Scrollbar(tree_container, orient="horizontal", command=self._global_summary_tree.xview)
        self._global_summary_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self._global_summary_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        
        tree_container.columnconfigure(0, weight=1)
        tree_container.rowconfigure(0, weight=1)
        
        self._global_summary_tree.heading("comp",         text="컴포넌트 (Body)")
        self._global_summary_tree.heading("pba_peak",     text="PBA Peak (Time)")
        self._global_summary_tree.heading("rrg_peak",     text="RRG Peak (Time) @ Block")
        self._global_summary_tree.heading("max_stress",   text="Max Stress (MPa)")
        self._global_summary_tree.heading("total_energy", text="Total Energy (J)")
        self._global_summary_tree.heading("gti",          text="GTI (Max)")
        self._global_summary_tree.heading("gbi",          text="GBI (Max)")
        self._global_summary_tree.heading("status",       text="상태 분석")
        
        self._global_summary_tree.column("comp",         width=120, anchor="center")
        self._global_summary_tree.column("pba_peak",     width=140, anchor="center")
        self._global_summary_tree.column("rrg_peak",     width=200, anchor="center")
        self._global_summary_tree.column("max_stress",   width=110, anchor="center")
        self._global_summary_tree.column("total_energy", width=110, anchor="center")
        self._global_summary_tree.column("gti",          width=70,  anchor="center")
        self._global_summary_tree.column("gbi",          width=70,  anchor="center")
        self._global_summary_tree.column("status",       width=160, anchor="w")
        
        self._on_refresh_summary_data()

    def _on_refresh_summary_data(self):
        """[v4.1] 전체 요약 데이터를 갱신합니다."""
        self._refresh_critical_info()
        self._refresh_global_summary()

    def _refresh_global_summary(self):
        """[v4.6] Engine 모듈을 사용하여 요약 테이블 데이터를 갱신합니다."""
        if not hasattr(self, '_global_summary_tree'): return
        self._global_summary_tree.delete(*self._global_summary_tree.get_children())
        
        summary_data = extract_global_summary_data(self.sim)
        
        for row in summary_data:
            self._global_summary_tree.insert("", "end", values=(
                row["comp"], row["pba_peak"], row["rrg_peak"], 
                row["max_stress"], row["total_energy"],
                row["gti"], row["gbi"], row["status"]
            ))

    def _show_metric_help(self, event, title, img_name):
        """[v4.2] TV 포장 상자 기반의 수식과 가이드를 포함한 상세 도움말 창을 표시합니다."""
        if hasattr(self, '_help_win') and self._help_win.winfo_exists():
            self._help_win.destroy()
            
        self._help_win = tk.Toplevel(self)
        self._help_win.title("WHTOOLS: Structural Metrics Definition (TV Package)")
        self._help_win.geometry("520x750")
        self._help_win.config(bg="#ffffff")
        
        # 버튼 근처 배치
        if event:
            x = event.x_root + 20
            y = event.y_root - 100
            self._help_win.geometry(f"+{x}+{y}")

        main_canv = tk.Canvas(self._help_win, bg="#ffffff")
        vsb = ttk.Scrollbar(self._help_win, orient="vertical", command=main_canv.yview)
        scroll_f = tk.Frame(main_canv, bg="#ffffff", padx=15, pady=15)
        main_canv.create_window((0,0), window=scroll_f, anchor="nw")
        main_canv.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        main_canv.pack(fill="both", expand=True)

        tk.Label(scroll_f, text="▣ TV 포장 구조 분석 지표 가이드", 
                 font=get_ui_font(12, True), bg="#ffffff", fg="#2c3e50").pack(anchor="w", pady=(0,10))
        
        # 4대 지표 설명 및 이미지 루프 (TV 박스 테마)
        metric_info = [
            ("1. Bending Angle (굽힘)", "상자 표면의 장기적인 곡률 변화를 측정합니다.\n수식: κ = 1/R, M = EIκ", "bending_tv_box"),
            ("2. Twist Angle (비틀림)", "TV 상자 전체의 비대칭적 회전 변형을 측정합니다.\n수식: θ = TL / GJ", "twist_tv_box"),
            ("3. RRG (Rotation Gradient)", "상자 표면 법선 벡터의 국부적 회전 변화율입니다.\n수식: ∇R = ∂R/∂x", "rrg_tv_box"),
            ("4. PBA (Principal Bending Axis)", "주요 굽힘이 발생하는 축의 방향을 벡터로 표시합니다.\n수식: tan(2θ) = 2τ / (σx - σy)", "pba_tv_box"),
            ("5. SSR (Surface Reconstruction)", "이산된 격자 데이터를 고해상도 곡면으로 보간하여 시각화합니다.\n알고리즘: RBF (Radial Basis Function) Interpolation", "ssr_concept"),
            ("6. Live Sync (실시간 연동)", "MuJoCo 물리 엔진의 실시간 데이터를 UI 대시보드와 즉각 동기화합니다.\n메커니즘: Passive Viewer Loop Syncing\n• 2D Field Contour: 패널의 면상 변형을 시각화\n• [실시간 연동]: 시뮬레이션 진행과 동차로 데이터 갱신\n• [필드 결과 생성]: 모든 부품의 컨투어를 새 창으로 띄움\n• [PBA 벡터]: 주 변형 축의 방향과 강도를 표시", "live_sync_concept")
        ]

        brain_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._help_img_refs = []
        
        for m_title, m_desc, img_key in metric_info:
            sep = ttk.Separator(scroll_f, orient="horizontal")
            sep.pack(fill="x", pady=13)
            
            tk.Label(scroll_f, text=m_title, font=get_ui_font(10, True), bg="#ffffff", fg="#d35400").pack(anchor="w")
            tk.Label(scroll_f, text=m_desc, font=get_ui_font(9), bg="#ffffff", justify="left", wraplength=460).pack(anchor="w", pady=5)
            
            # TV 박스 기반 생성된 이미지 검색 및 로드
            img_path = ""
            for f in os.listdir(brain_dir):
                if img_key in f and f.endswith(".png"):
                    img_path = os.path.join(brain_dir, f)
                    break
            
            if img_path and os.path.exists(img_path) and _PIL_AVAILABLE:
                try:
                    img = Image.open(img_path).resize((450, 280), Image.LANCZOS)
                    p_img = ImageTk.PhotoImage(img)
                    self._help_img_refs.append(p_img)
                    tk.Label(scroll_f, image=p_img, bg="#ffffff").pack(pady=5)
                except: pass

        tk.Label(scroll_f, text="\n(닫으려면 클릭하세요)", font=get_ui_font(8), bg="#ffffff", fg="gray").pack()
        self._help_win.bind("<Button-1>", lambda e: self._help_win.destroy())
        
        scroll_f.update_idletasks()
        main_canv.config(scrollregion=main_canv.bbox("all"))

    # ==============================================================
    # [탭 3] 2D 컨투어
    # ==============================================================

    def _build_contour_tab(self, parent):
        """[v4] 2D 컨투어 탭: 레이아웃 최적화 및 상세 리스트박스 (스크롤 지원)."""
        main_holder = ttk.Frame(parent)
        main_holder.pack(fill="both", expand=True)
        
        # [v4.8.8] 마스터 스크롤 영역 (캔버스) 생성 및 가로 너비 동기화
        c_canvas = tk.Canvas(main_holder, highlightthickness=0)
        c_vscroll = ttk.Scrollbar(main_holder, orient="vertical", command=c_canvas.yview)
        # 전체를 담을 프레임 (left_f 등 모든 위젯 포함)
        master_f = ttk.Frame(c_canvas)

        c_canvas.create_window((0, 0), window=master_f, anchor="nw")
        c_canvas.configure(yscrollcommand=c_vscroll.set)
        
        c_vscroll.pack(side="right", fill="y")
        c_canvas.pack(side="left", fill="both", expand=True)

        def _on_c_canvas_configure(event, canvas=c_canvas, frame=master_f):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(canvas.find_withtag("all")[0], width=event.width)

        c_canvas.bind("<Configure>", _on_c_canvas_configure)
        
        # 마우스 휠 지원
        def _on_mousewheel_c(event):
            c_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        c_canvas.bind_all("<MouseWheel>", _on_mousewheel_c)

        # 내부를 좌/우로 나눔 (설정 영역 / Spatial Viewer)
        main_f = ttk.Frame(master_f)
        main_f.pack(fill="both", expand=True, padx=5, pady=5)
        
        # [v4.8.4] 제어/설정 하단 바 영역 (기존 왼쪽에서 하단으로 이동)
        left_f = ttk.Frame(main_f)
        left_f.pack(side="bottom", fill="x", padx=5, pady=5)
        
        # [v4.1] 설정 영역 가로 배치 (1.대상, 2.옵션, 가이드)
        config_f = ttk.Frame(left_f)
        config_f.pack(side="top", fill="x", padx=5, pady=2)
 
        # 1. 분석 영역 (Analysis Metric)
        opt_f = ttk.LabelFrame(config_f, text=" 1. 분석 지표 선택 ")
        opt_f.pack(side="left", fill="y", padx=5)

        metrics_info_v4 = [
            ("PBA Magnitude (Principal Bending Axis)", "angle", "pba_concept.png"),
            ("RRG (High Fidelity Twist / Local Stress)", "rrg", "rrg_concept.png"),
            ("Bending X (Spanwise: X축 방향 휨)", "bend_x", "distortion_concept.png"),
            ("Bending Y (Ribwise: Y축 방향 휨)", "bend_y", "distortion_concept.png"),
            ("Bending Z (Overall Tilt: 전반적 기울기)", "bend", "distortion_concept.png"),
            ("Twist Angle (Overall Torsion)", "twist", "distortion_concept.png"),
            ("Bending Stress (BS, MPa 단위 굽힘 응력)", "s_bend", "rrg_concept.png"),
            ("Torsional Stress (TS, MPa 단위 비틀림 응력)", "s_twist", "rrg_concept.png"),
            ("Bending Moment (BM, N·m 단위 굽힘 모멘트)", "moment", "distortion_concept.png"),
            ("Total Strain Energy (TSE, J 단위 흡수 에너지)", "energy", "distortion_concept.png")
        ]
        
        # [v4.7] 수직 리스트 레이아웃으로 변경 (Tab 2와 동기화)
        for name, key, help_img in metrics_info_v4:
            row = ttk.Frame(opt_f)
            row.pack(fill="x", padx=5, pady=1)
            
            # __init__에서 생성된 BooleanVar 가져오기 (없으면 생성)
            if key not in self._contour_metric_vars:
                self._contour_metric_vars[key] = tk.BooleanVar(value=(key=="bend"))
                
            ttk.Checkbutton(row, text=name, variable=self._contour_metric_vars[key]).pack(side="left")
            btn = ttk.Button(row, text="?", width=2)
            btn.pack(side="right", padx=2)
            btn.bind("<Button-1>", lambda e, t=name, i=help_img: self._show_metric_help(e, t, i))

        btn_m_f = ttk.Frame(opt_f)
        btn_m_f.pack(fill="x", side="bottom", padx=5, pady=5)
        ttk.Button(btn_m_f, text="전부 선택", width=12, 
                   command=lambda: [v.set(True) for v in self._contour_metric_vars.values()]).pack(side="left", padx=2)
        ttk.Button(btn_m_f, text="전부 해제", width=12, 
                   command=lambda: [v.set(False) for v in self._contour_metric_vars.values()]).pack(side="left", padx=2)

        ttk.Separator(opt_f, orient="horizontal").pack(fill="x", pady=5)

        # 가시화 모드
        ttk.Label(opt_f, text="가시화 모드 (Visualization Mode):").pack(anchor="w", padx=5)
        mode_f = ttk.Frame(opt_f)
        mode_f.pack(fill="x", padx=5, pady=2)
        # [REMOVED] 💡 도움말 및 가이드 (Control 영역으로 통합)
        ttk.Radiobutton(mode_f, text="실시간 변형 (Temporal)", variable=self._contour_mode_var, value="temporal").pack(side="left", padx=5)
        ttk.Radiobutton(mode_f, text="최대 누적 (Overall Max)", variable=self._contour_mode_var, value="overall").pack(side="left", padx=5)
        
        ttk.Separator(opt_f, orient="horizontal").pack(fill="x", pady=5)
        
        # SSR 및 PBA 옵션 (도움말 ? 추가)
        ssr_pba_f = ttk.Frame(opt_f)
        ssr_pba_f.pack(fill="x", padx=5, pady=2)
        
        pba_row = ttk.Frame(ssr_pba_f)
        pba_row.pack(fill="x", pady=2)
        ttk.Checkbutton(pba_row, text="PBA 벡터 표시 (Principal Axis)", variable=self._show_pba_var).pack(side="left")
        ttk.Button(pba_row, text="?", width=2, command=lambda: self._show_metric_help(None, "PBA Vector Visualization", "pba_concept")).pack(side="right")

        # 2. 가시화 대상 (질량 보정용 부폭 제외 필터링)
        comp_f = ttk.LabelFrame(config_f, text=" 2. 대상 부품 선택 ")
        comp_f.pack(side="left", fill="both", expand=True, padx=5)
        
        # [v4.8.2] Header: Analyzer Button & SSR Checkbox (Tab 2와 동일 구성)
        c_header = ttk.Frame(comp_f)
        c_header.pack(fill="x", side="top", padx=5, pady=2)
        
        ttk.Button(c_header, text="🔍 SSR Analyzer", 
                   command=self._open_ssr_analyzer, width=18, style="Accent.TButton").pack(side="left", padx=2)
        
        ttk.Checkbutton(c_header, text="SSR 모드", variable=self._ssr_mode_var).pack(side="left", padx=5)
        ttk.Button(c_header, text="?", width=2, 
                   command=lambda: self._show_metric_help(None, "High Fidelity SSR", "ssr_concept")).pack(side="left")

        # [v4.8.2] Body: Scrollable Component List
        c_scroll_f = ttk.Frame(comp_f)
        c_scroll_f.pack(fill="both", expand=True, padx=2, pady=2)
        
        canvas_c = tk.Canvas(c_scroll_f, highlightthickness=0, width=220)
        scroll_c = ttk.Scrollbar(c_scroll_f, orient="vertical", command=canvas_c.yview)
        inner_c = ttk.Frame(canvas_c)
        canvas_c.create_window((0, 0), window=inner_c, anchor="nw")
        
        self._contour_comp_vars = {}
        # 필터링: 'aux' 또는 'inertiaux_single' 포함 항목 제외
        full_comp_list = sorted(list(self.sim.metrics.keys()))
        filtered_list = [c for c in full_comp_list if not any(x in c.lower() for x in ['aux', 'inertiaux_single'])]
        
        for c in filtered_list:
            is_def_active = c.lower() in ('bcushion', 'bbox', 'tv', 'panel')
            var = tk.BooleanVar(value=is_def_active)
            self._contour_comp_vars[c] = var
            ttk.Checkbutton(inner_c, text=c, variable=var).pack(anchor="w", padx=5)
            
        inner_c.update_idletasks()
        canvas_c.config(scrollregion=canvas_c.bbox("all"), yscrollcommand=scroll_c.set)
        canvas_c.pack(side="left", fill="both", expand=True)
        scroll_c.pack(side="right", fill="y")

        # [v4.8.2] Footer: Selection Buttons & Action Button
        c_footer = ttk.Frame(comp_f)
        c_footer.pack(fill="x", side="bottom", padx=5, pady=5)
        
        btn_sel_f = ttk.Frame(c_footer)
        btn_sel_f.pack(side="left")
        ttk.Button(btn_sel_f, text="전체 선택", width=9,
                   command=lambda: [v.set(True) for v in self._contour_comp_vars.values()]).pack(side="left", padx=1)
        ttk.Button(btn_sel_f, text="전체 해제", width=9,
                   command=lambda: [v.set(False) for v in self._contour_comp_vars.values()]).pack(side="left", padx=1)
        
        ttk.Button(c_footer, text="🖼️ 필드 생성", 
                   command=self._on_show_contour_frame, width=15).pack(side="right", padx=2)

        # 3. [ Control ] 그룹 패널 (하단 바 하단 줄에 배치하여 줄바꿈 효과)
        control_group = ttk.LabelFrame(left_f, text=" [ Control ] ")
        control_group.pack(side="top", fill="x", padx=5, pady=5)
        
        # 통합 도움말 버튼 (?)
        def _show_combined_guide():
            self._show_metric_help(None, "Control & Live Sync Guide", "live_sync_concept")
            
        # [v4.8.9] 줄바꿈 레이아웃으로 변경 (가로 폭 확보)
        help_f = ttk.Frame(control_group)
        help_f.pack(fill="x", padx=5, pady=2)
        
        help_btn = ttk.Button(help_f, text="?", width=3, command=_show_combined_guide)
        help_btn.pack(side="left", padx=2)
        ttk.Label(help_f, text="시각화 및 실시간 연동 제어", font=get_ui_font(9, True)).pack(side="left", padx=5)
        
        # 실시간 연동 체크박스 (첫 번째 줄)
        self._live_sync_chk = ttk.Checkbutton(control_group, text="🔄 실시간 연동 (Live Sync: 시뮬레이션 진행 중 자동 갱신)", 
                                              variable=self._live_sync_var)
        self._live_sync_chk.pack(anchor="w", padx=10, pady=5)
        
        # [v4.3] Scale 조절 추가 (두 번째 줄)
        scale_f = ttk.Frame(control_group)
        scale_f.pack(anchor="w", padx=10, pady=5)
        ttk.Label(scale_f, text="🎨 컨투어 스케일 범위 (Scale Range):").pack(side="left")
        ttk.Entry(scale_f, textvariable=self._matrix_vmin_var, width=6).pack(side="left", padx=2)
        ttk.Label(scale_f, text="~").pack(side="left")
        ttk.Entry(scale_f, textvariable=self._matrix_vmax_var, width=6).pack(side="left", padx=2)
        ttk.Label(scale_f, text="(N·m, MPa, J 등 지표 단위)", foreground="gray").pack(side="left", padx=5)

        # 4. 임베디드 그래프 영역 (상단 메인 영역)
        self.plot_container = ttk.Frame(main_f)
        self.plot_container.pack(side="top", fill="both", expand=True, padx=10, pady=5)
        
        # 안내 텍스트 (그래프 전)
        self.plot_placeholder = ttk.Label(self.plot_container, 
                                           text="[실시간 연동] 또는 [매트릭스 컨투어 생성] 시 여기에 그래프가 표시됩니다.\n\n"
                                                "※ 시뮬레이션 시작 후 MuJoCo 뷰어에서 [Space]를 눌러야 물리 연산이 진행됩니다.",
                                           font=("Malgun Gothic", 10), foreground="#95a5a6", justify="center")
        self.plot_placeholder.place(relx=0.5, rely=0.5, anchor="center")

        # [REMOVED] 💡 도움말 및 가이드 (Control 영역의 ? 버튼으로 통합됨)


    # ==============================================================
    # 내부 유틸리티
    # ==============================================================

    def _get_total_timesteps(self):
        """시뮬레이션 시계열 데이터의 총 스텝 수를 반환합니다."""
        return len(self.sim.time_history) if self.sim.time_history else 0

    def _update_frame_count(self):
        """슬라이더 최대값과 총 프레임 수를 갱신합니다."""
        total = self._get_total_timesteps()
        self._total_frames = max(1, total)
        self._time_slider.config(to=max(1, self._total_frames - 1))

    def _current_time_val(self):
        """현재 선택된 프레임의 시뮬레이션 시간을 반환합니다."""
        step = int(self._time_var.get())
        if self.sim.time_history and step < len(self.sim.time_history):
            return self.sim.time_history[step]
        # 데이터가 아직 없는 경우 추정치
        dt = self.sim.model.opt.timestep if self.sim.model else 0.001
        return step * dt

    def update_live_data(self):
        """[v4] 시뮬레이션 진행 중에 호출되어 실시간으로 슬라이더와 레이블을 동기화합니다."""
        if not self.winfo_exists(): return
        
        total = self._get_total_timesteps()
        if total <= 0: return

        # 프레임 수 및 슬라이더 범위 갱신
        self._update_frame_count()
        
        # 실시간 모드: 현재 진행 중인 최신 프레임을 추적
        latest_frame = total - 1
        self._current_frame = latest_frame
        self._time_var.set(latest_frame)
        self._update_time_label(latest_frame)
        
        # [v4] 실시간 컨투어 연동
        if self._live_sync_var.get():
            self._update_embedded_contour()

        # [v4.1] 구조 해석 요약 실시간 갱신 (구조 탭 활성 시)
        if hasattr(self, '_current_tab_name') and self._current_tab_name == "Structural":
            self._on_refresh_summary_data()

        self.update_idletasks()

    def on_simulation_complete(self):
        """[v4.2] 시뮬레이션 종료 시 호출되어 최종 데이터를 바탕으로 슬라이더와 분석 요약을 갱신합니다."""
        if not self.winfo_exists(): return
        
        # 1. 프레임 수/슬라이더 최종 갱신
        self._update_frame_count()
        
        # 2. 마지막 프레임으로 이동 (최종 결과 상태 확인)
        total = self._get_total_timesteps()
        if total > 0:
            last_idx = total - 1
            self._time_var.set(last_idx)
            self._update_time_label(last_idx)
            self._current_frame = last_idx
        # 3. GUI 요약 데이터 강제 갱신
        try:
            self._on_refresh_summary_data()
            self._refresh_critical_info()
        except:
            pass

        self.update_idletasks()
        # [WHTOOLS] 분석 완료 팝업 제거 (사용자 요청)
        # try:
        #     messagebox.showinfo("분석 완료", "시뮬레이션이 종료되었습니다.\n최종 데이터(구조적 지표 등)가 연동되었습니다.", parent=self)
        # except:
        #     pass
        print(" >> Simulation Analysis Complete. Data synchronized.")

    def _update_time_label(self, step: int):
        """시간 레이블을 현재 스텝에 맞게 갱신합니다."""
        t = self._current_time_val()
        total = self._total_frames
        self._time_lbl.config(text=f"t = {t:.4f} s  [{step:5d}/{total-1:5d}]")

    def _refresh_critical_info(self):
        """임계 시점 정보를 텍스트 위젯에 갱신합니다."""
        ct = {}
        if self.sim.result is not None:
            ct = getattr(self.sim.result, 'critical_timestamps', {})

        lines = ["[자동 검출 임계 시점]\n"]
        if ct.get('local_peak_time') is not None:
            lines.append(f"  RRG Local Peak")
            lines.append(f"    t   = {ct['local_peak_time']:.4f} s")
            lines.append(f"    RRG = {ct.get('local_peak_rrg', 0):.3f} deg\n")
        if ct.get('global_avg_peak_time') is not None:
            lines.append(f"  Avg Distortion Peak")
            lines.append(f"    t   = {ct['global_avg_peak_time']:.4f} s")
            lines.append(f"    val = {ct.get('global_avg_peak_val', 0):.3f} deg\n")
        if ct.get('pba_peak_time') is not None:
            lines.append(f"  PBA Peak (Principal Bending Axis)")
            lines.append(f"    t   = {ct['pba_peak_time']:.4f} s")
            lines.append(f"    mag = {ct.get('pba_peak_mag', 0):.2f} deg")
            lines.append(f"    dir = Az:{ct.get('pba_peak_angle', 0):.1f}°, El:{ct.get('pba_peak_ele', 0):.1f}°\n")

        if not ct:
            lines.append("  (시뮬레이션 데이터 없음)")

        self._critical_text.config(state="normal")
        self._critical_text.delete("1.0", tk.END)
        self._critical_text.insert("1.0", "\n".join(lines))
        self._critical_text.config(state="disabled")

    # ==============================================================
    # 이벤트 핸들러 - 시간 컨트롤
    # ==============================================================

    def _on_time_slider_change(self, val):
        """
        슬라이더 값 변경 시 레이블 갱신 및 컨투어 탭 연동 업데이트.

        Args:
            val (str): ttk.Scale이 전달하는 현재 값 문자열
        """
        step = int(float(val))
        self._current_frame = step
        self._update_time_label(step)
        
        # [v4.2] 슬라이더 변경 시 MuJoCo 뷰어 및 열려 있는 컨투어 팝업 동기화
        if hasattr(self, 'sim') and hasattr(self.sim, 'set_state_at'):
            try:
                self.sim.set_state_at(step)
            except: pass
            
        if hasattr(self, '_contour_popup') and self._contour_popup.winfo_exists():
            self._update_popup_contours(step)
        
        if self._live_sync_var.get():
            self._update_live_contour_view()


    def _on_play(self):
        """애니메이션 재생을 시작합니다 (루프 방식)."""
        if self._anim_running:
            return
        self._anim_running = True
        self._play_btn.config(text="▶ 재생 중")
        self._animate_step()

    def _on_pause(self):
        """애니메이션을 일시정지합니다."""
        self._anim_running = False
        self._play_btn.config(text="▶ Play")
        if self._anim_job:
            self.after_cancel(self._anim_job)
            self._anim_job = None

    def _on_stop(self):
        """애니메이션을 정지하고 처음으로 돌아갑니다."""
        self._on_pause()
        self._current_frame = 0
        self._time_var.set(0)
        self._update_time_label(0)

    def _animate_step(self):
        """한 프레임씩 진행하는 재귀적 애니메이션 루프입니다."""
        if not self._anim_running:
            return
        self._current_frame += 1
        if self._current_frame >= self._total_frames:
            self._current_frame = 0  # 루프 재시작

        self._time_var.set(self._current_frame)
        self._update_time_label(self._current_frame)

        self._anim_job = self.after(self._anim_speed_ms, self._animate_step)

    def _step_frame(self, delta: int):
        """
        키보드 화살표로 1프레임씩 이동합니다.

        Args:
            delta (int): +1 (앞으로) 또는 -1 (뒤로)
        """
        self._on_pause()
        new_frame = max(0, min(self._total_frames - 1, self._current_frame + delta))
        self._current_frame = new_frame
        self._time_var.set(new_frame)
        self._update_time_label(new_frame)

    def _on_speed_change(self, event=None):
        """
        재생 속도 변경 콤보박스 이벤트 핸들러.
        선택된 배율에 따라 after() 간격(ms)을 변경합니다.
        """
        speed_map = {"0.25x": 320, "0.5x": 160, "1x": 80, "2x": 40, "4x": 20}
        self._anim_speed_ms = speed_map.get(self._speed_var.get(), 80)

    # ==============================================================
    # 이벤트 핸들러 - 기구학 그래프
    # ==============================================================

    def _on_plot_kinematics(self):
        """[v4.1] 선택된 데이터 종류, 축, 위치를 기반으로 윈도우별 그래프를 생성합니다."""
        dtypes_sel = [k for k, v in self._kin_dtype_vars.items() if v.get()]
        axes_sel = [ax for ax in ["X", "Y", "Z"] if self._axis_vars[ax].get()]
        locs_sel = [i for i in range(10) if self._loc_vars[i].get()]
        win_mode = self._plot_window_mode.get()

        if not dtypes_sel:
            messagebox.showwarning("선택 오류", "데이터 종류를 1개 이상 선택하세요.")
            return
        if not axes_sel:
            messagebox.showwarning("선택 오류", "축을 1개 이상 선택하세요.")
            return
        if not locs_sel:
            messagebox.showwarning("선택 오류", "위치를 1개 이상 선택하세요.")
            return

        t_arr = np.array(self.sim.time_history) if self.sim.time_history else np.array([0.0])
        
        if win_mode == "data_type":
            for dtype in dtypes_sel:
                fig, axes = plt.subplots(1, len(axes_sel), figsize=(4 * len(axes_sel), 4), squeeze=False)
                fig.canvas.manager.set_window_title(f"WHTOOLS: Motion - {dtype.capitalize()}")
                fig.suptitle(f"Motion Analysis: {dtype.upper()} Comparison ({self._coord_system_var.get()})", fontsize=11, fontweight='bold')
                y_label = {"displacement": "Position (m)", "velocity": "Velocity (m/s)", "acceleration": "Acceleration (m/s^2)"}.get(dtype, "Value")
                for col, ax_name in enumerate(axes_sel):
                    ax = axes[0][col]
                    ax.set_title(f"Axis: {ax_name}", fontsize=10)
                    ax.set_xlabel("Time (s)", fontsize=9)
                    ax.set_ylabel(y_label, fontsize=9)
                    for l_idx in locs_sel:
                        y_data = self._get_kinematic_series(dtype, ax_name, l_idx)
                        if len(y_data) > 0:
                            ax.plot(t_arr[:len(y_data)], y_data, label=self.LOCATION_LABELS[l_idx], alpha=0.8)
                
                # [v4.3] 인터랙티브 기능 연결
                if self.plot_manager:
                    self.plot_manager.attach_interactivity(fig)
                    ax.legend(fontsize=7, loc='upper right')
                    ax.grid(True, alpha=0.3)
                fig.tight_layout()
        elif win_mode == "axis":
            for ax_name in axes_sel:
                fig, axes = plt.subplots(len(dtypes_sel), 1, figsize=(6, 3 * len(dtypes_sel)), squeeze=False)
                fig.canvas.manager.set_window_title(f"WHTOOLS: Motion - {ax_name} Axis")
                fig.suptitle(f"Motion Analysis: {ax_name} Axis Comparison ({self._coord_system_var.get()})", fontsize=11, fontweight='bold')
                for row, dtype in enumerate(dtypes_sel):
                    ax = axes[row][0]
                    ax.set_title(f"Data: {dtype}", fontsize=10)
                    ax.set_xlabel("Time (s)", fontsize=9)
                    y_label = {"displacement": "Position (m)", "velocity": "Velocity (m/s)", "acceleration": "Acceleration (m/s^2)"}.get(dtype, "Value")
                    ax.set_ylabel(y_label, fontsize=9)
                    for l_idx in locs_sel:
                        y_data = self._get_kinematic_series(dtype, ax_name, l_idx)
                        if len(y_data) > 0:
                            ax.plot(t_arr[:len(y_data)], y_data, label=self.LOCATION_LABELS[l_idx], alpha=0.8)
                    ax.legend(fontsize=7, loc='upper right')
                    ax.grid(True, alpha=0.3)
                fig.tight_layout()
        else: # location -> Matrix Layout (One Window, Multiple Subplots) [v4.2 FIX]
            # [v4.2] 유저 요청: 행(DataType) x 열(Axis) 매트릭스 레이아웃 구현 (한 창에 모든 위치 비교)
            fig, axes = plt.subplots(len(dtypes_sel), len(axes_sel), 
                                     figsize=(4 * len(axes_sel), 3 * len(dtypes_sel)), squeeze=False)
            fig.canvas.manager.set_window_title(f"WHTOOLS: Motion Analysis Matrix")
            fig.suptitle(f"Motion Analysis Matrix Layout ({self._coord_system_var.get()})", fontsize=11, fontweight='bold')
            
            for r, dtype in enumerate(dtypes_sel):
                y_label = {"displacement": "Pos (m)", "velocity": "Vel (m/s)", "acceleration": "Acc (m/s^2)"}.get(dtype, "Val")
                for c, ax_name in enumerate(axes_sel):
                    ax = axes[r][c]
                    for l_idx in locs_sel:
                        y_data = self._get_kinematic_series(dtype, ax_name, l_idx)
                        if len(y_data) > 0:
                            ax.plot(t_arr[:len(y_data)], y_data, label=self.LOCATION_LABELS[l_idx], linewidth=1.2, alpha=0.8)
                    
                    ax.set_title(f"{dtype.capitalize()} / {ax_name}", fontsize=9)
                    ax.set_xlabel("Time (s)", fontsize=8)
                    ax.set_ylabel(y_label, fontsize=8)
                    ax.grid(True, alpha=0.3)
                    if r == 0 and c == len(axes_sel) - 1: # 우측 상단 셀에만 범례 표시하여 가독성 확보
                        ax.legend(fontsize=7, loc='upper right', framealpha=0.5)
            fig.tight_layout()
                
        plt.show(block=False)
        
        # [v4.7] 모든 열려 있는 Figure에 확장 기능 일괄 적용 (안전성 확보)
        if self.plot_manager:
            for f_idx in plt.get_fignums():
                self.plot_manager.attach_interactivity(plt.figure(f_idx))

    def _get_kinematic_series(self, dtype, axis_name, loc_idx):
        """기구학 시계열 데이터를 추출합니다."""
        try:
            sim = self.sim
            axis_idx = {"X": 0, "Y": 1, "Z": 2}[axis_name]
            if loc_idx == 9: # Mass Center (CoM)
                if dtype == "displacement": arr = [v[axis_idx] for v in sim.cog_pos_hist]
                elif dtype == "velocity": arr = [v[axis_idx] for v in sim.cog_vel_hist]
                else: arr = [v[axis_idx] for v in sim.cog_acc_hist]
            elif loc_idx == 8: # Geometry Center
                if dtype == "displacement": arr = [v[axis_idx] for v in sim.geo_center_pos_hist]
                elif dtype == "velocity": arr = [v[axis_idx] for v in sim.geo_center_vel_hist]
                else: arr = [v[axis_idx] for v in sim.geo_center_acc_hist]
            else: # Corners
                if dtype == "displacement":
                    hist = sim.corner_pos_hist
                    arr = [(frame[loc_idx][axis_idx] if (frame and loc_idx < len(frame)) else 0.0) for frame in hist]
                elif dtype == "velocity":
                    hist = sim.corner_vel_hist
                    arr = [(frame[loc_idx][axis_idx] if (frame and loc_idx < len(frame)) else 0.0) for frame in hist]
                else:
                    hist = sim.corner_acc_hist
                    arr = [(frame[loc_idx][axis_idx] if (frame and loc_idx < len(frame)) else 0.0) for frame in hist]
            if self._coord_system_var.get() == "Local" and len(arr) > 0:
                if dtype in ("displacement", "velocity"):
                    arr = [v - arr[0] for v in arr]
            return np.array(arr, dtype=float)
        except: return np.array([], dtype=float)

    def _apply_loc_group(self, is_select: bool):
        """[v4.7] 선택된 그룹(All, L, R, T, B, F, RE)에 따라 데이터 추출 위치 선택을 제어합니다."""
        group = self._loc_group_var.get()
        for i, label in enumerate(self.LOCATION_LABELS):
            if i >= len(self._loc_vars): break
            match = False
            
            if group == "All":
                match = True
            else:
                # [v4.7 FIX] L-B-R 형식 등 하이픈(-)으로 연결된 좌표 식별
                parts = label.split('-')
                if len(parts) == 3:
                     if group == "Left" and parts[0] == "L": match = True
                     elif group == "Right" and parts[0] == "R": match = True
                     elif group == "Bottom" and parts[1] == "B": match = True
                     elif group == "Top" and parts[1] == "T": match = True
                     elif group == "Front" and parts[2] == "F": match = True
                     elif group == "Rear" and parts[2] == "R": match = True
            
            if match:
                self._loc_vars[i].set(is_select)

    # ==============================================================
    # 이벤트 핸들러 - 구조 해석 그래프
    # ==============================================================

    def _on_plot_structural(self):
        """선택한 구조 해석 지표들의 시계열 그래프를 생성합니다."""
        selected_metrics = [
            (key, txt)
            for key, (var, txt) in self._struct_metric_vars.items()
            if var.get()
        ]
        if not selected_metrics:
            messagebox.showwarning("선택 오류", "지표를 1개 이상 선택하세요.")
            return

        # [v4] 선택된 부품 리스트 추출
        selected_comps = [name for name, var in self._comp_selection_vars.items() if var.get()]

        sts = self.sim.structural_time_series
        t_arr = np.array(self.sim.time_history) if self.sim.time_history else np.array([0.0])

        # [v4.6.2] Engine에서 이미 Decimation 되어 저장되므로 (time_history와 1:1)
        t_ts = np.array(self.sim.time_history) if self.sim.time_history else np.array([0.0])
        n = len(selected_metrics)

        # [v4.3.2] 그리드 레이아웃 및 크기 조정 (가로 2/3, 세로 2배)
        ncols = 2 if n >= 3 else 1
        nrows = (n + ncols - 1) // ncols
        
        # 기본 단위 크기: 가로 6.7 (기존 10의 2/3), 세로 2.5 (5.0의 1/2)
        fig_w = 6.7 * ncols
        fig_h = 2.5 * nrows
        
        fig, axes_array = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
        axes_list = axes_array.flatten()
        
        fig.canvas.manager.set_window_title("WHTOOLS: Structural Analysis Time-Series")
        fig.suptitle("구조 해석 상세 지표 분석 (Structural Time-Series)", fontsize=12, fontweight="bold")

        # 임계 시점 정보
        ct = {}
        if self.sim.result is not None:
            ct = getattr(self.sim.result, 'critical_timestamps', {})

        ct_markers = [
            ('local_peak_time',     'RRG Peak',  'red'),
            ('global_avg_peak_time','AvgPeak',   'orange'),
            ('pba_peak_time',       'PBA Peak',  'purple'),
        ]

        for idx, (key, label) in enumerate(selected_metrics):
            ax = axes_list[idx]
            
            # [v4.3.2] 긴 제목 줄바꿈 처리
            display_label = label.replace(" (", "\n(")
            ax.set_title(f"Metric: {display_label}", fontsize=10, fontweight='bold', loc='left')
            ax.set_xlabel("Time (s)", fontsize=9)
            ax.grid(True, alpha=0.3, linestyle='--')

            if key in ("pba_magnitude", "rrg_max", "bend_overall", "bend_x_overall", "bend_y_overall", 
                       "twist_overall", "gti_overall", "gbi_overall", "s_bend_overall", "s_twist_overall", 
                       "moment_overall", "tse_overall"):
                # 컴포넌트별 집계 데이터 (v4.1 Global Metrics 포함)
                data_map = {
                    "pba_magnitude":  "max_pba_hist",
                    "rrg_max":        "max_rrg_hist",
                    "bend_overall":   "all_blocks_bend",
                    "bend_x_overall": "all_blocks_bend_x",
                    "bend_y_overall": "all_blocks_bend_y",
                    "twist_overall":  "all_blocks_twist",
                    "gti_overall":    "gti", 
                    "gbi_overall":    "gbi",
                    "s_bend_overall": "all_blocks_s_bend",
                    "s_twist_overall": "all_blocks_s_twist",
                    "moment_overall": "all_blocks_moment",
                    "tse_overall":    "energy"
                }
                internal_key = data_map[key]
                plot_count = 0
                for comp_name in selected_comps:
                    # Global Metrics(GTI, GBI, TSE)는 comp_global_metrics에 위치
                    if key in ("gti_overall", "gbi_overall", "tse_overall"):
                        cg = self.sim.structural_time_series.get('comp_global_metrics', {}).get(comp_name, {})
                        series = cg.get(internal_key, [])
                        if len(series) > 0:
                            t_comp = t_arr[:len(series)]
                            if len(series) < len(t_arr) * 0.3: t_comp = t_ts[:len(series)]
                            ax.plot(t_comp, series, label=comp_name, linewidth=1.2)
                            plot_count += 1
                    else:
                        comp_m = self.sim.metrics.get(comp_name, {})
                        block_data = comp_m.get(internal_key, {})
                        
                        if not block_data: continue

                        # [v4.8.4] 상세 플롯 모드 분기
                        if self._plot_all_blocks_var.get() and isinstance(block_data, dict):
                            # 모든 개별 블록 출력
                            for g_idx, b_series in block_data.items():
                                if not b_series: continue
                                t_comp = t_arr[:len(b_series)]
                                if len(b_series) < len(t_arr) * 0.3: t_comp = t_ts[:len(b_series)]
                                ax.plot(t_comp, b_series, label=f"{comp_name} - B#{g_idx}", linewidth=0.8, alpha=0.7)
                                plot_count += 1
                        else:
                            # 기존 Max. of Blocks (Aggregated)
                            series = []
                            if key in ("pba_magnitude", "rrg_max"):
                                series = block_data # 이미 집계된 리스트
                            else:
                                min_len = min((len(s) for s in block_data.values() if s), default=0)
                                if min_len > 0:
                                    series = [max((abs(s[si]) for s in block_data.values() if si < len(s)), default=0.0) 
                                              for si in range(min_len)]
                            
                            if len(series) > 0:
                                t_comp = t_arr[:len(series)]
                                if len(series) < len(t_arr) * 0.3: t_comp = t_ts[:len(series)]
                                ax.plot(t_comp, series, label=comp_name, linewidth=1.5)
                                plot_count += 1
                
                ax.set_ylabel(f"{display_label}", fontsize=9)
                if plot_count > 0:
                    ax.legend(fontsize=8, loc='upper right', frameon=True, framealpha=0.8)

            elif key in sts:
                arr = np.array(sts[key])
                if arr.ndim == 2:
                    arr = np.linalg.norm(arr, axis=1)   # 벡터인 경우 크기
                
                plot_len = min(len(t_ts), len(arr))
                ax.plot(t_ts[:plot_len], arr[:plot_len], color="firebrick", linewidth=1.5)
                ax.set_ylabel("Value", fontsize=9)

            # 임계 시점 수직선
            for ct_key, ct_label, ct_color in ct_markers:
                if ct.get(ct_key) is not None:
                    ax.axvline(ct[ct_key], color=ct_color, linestyle="--",
                               alpha=0.6, linewidth=1.2, label=ct_label)
            
            # [v4.3.2] Y축 범위 최적화 (0부터 시작 유도, 상단 1.0 고정 제거)
            # BS, TS, RRG 등 미세 지표 가시화 확보를 위해 하한만 설정
            ax.set_ylim(bottom=0)
            if ax.get_ylim()[1] < 1e-9: # 데이터가 없거나 0인 경우 최소 범위 확보
                ax.set_ylim(0, 1.0)

        # 사용하지 않는 서브플롯 숨기기 (Grid 레이아웃 시 마지막 빈칸 대응)
        for j in range(len(selected_metrics), len(axes_list)):
            axes_list[j].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show(block=False)
        
        # [v4.3] 확장 매니저 연결 (인터랙티브 분석 및 어노테이션 활성화)
        # [v4.7 FIX] plt.show() 이후에 호출하여 윈도우 핸들(Qt/Tk) 존재 보장
        if self.plot_manager:
            self.plot_manager.attach_interactivity(fig)

    # ==============================================================
    # 이벤트 핸들러 - 2D 컨투어
    # ==============================================================

    def _get_contour_grid_at(self, step: int, comp: str = None, metric: str = None):
        """[v4.6] Engine 모듈을 사용하여 2D 그리드 데이터를 추출합니다."""
        if comp is None: comp = self._comp_var.get()
        if metric is None:
            selected = [k for k, v in self._contour_metric_vars.items() if v.get()]
            metric = selected[0] if selected else "bend"
        
        # [v4.9] JAX-SSR 데이터가 미리 분석되어 있는지 확인
        if self._ssr_mode_var.get() and comp in self._ssr_data_store:
            comp_store = self._ssr_data_store[comp]
            
            # (A) JAX 배치 데이터가 있는 경우
            if 'JAX_BATCH' in comp_store:
                jax_data = comp_store['JAX_BATCH']
                # Metric 매핑 (기존 UI 명칭 -> JAX 필드 명칭)
                jax_metric_map = {
                    "bend": "Displacement [mm]",
                    "s_bend": "Stress XX [MPa]",
                    "rrg": "Signed Von-Mises [MPa]",  # Von-Mises를 RRG 대체용으로 시각화 가능
                    "energy": "Eq. Strain [mm/mm]"
                }
                jax_key = jax_metric_map.get(metric, "Displacement [mm]")
                
                if jax_key in jax_data:
                    # jax_data[key] shape: (n_frames, res, res)
                    # XI, YI 로부터 물리 좌표계를 복원하거나 메쉬 그대로 사용
                    XI, YI = jax_data['X_mesh'], jax_data['Y_mesh']
                    grid = jax_data[jax_key][step]
                    return XI, YI, grid, ["X", "Y"]

            # (B) 개별 프레임 분석 데이터가 있는 경우 (Legacy PSR)
            if step in comp_store:
                ssr = comp_store[step]
                XI, YI = np.array(ssr['grid_x']), np.array(ssr['grid_y'])
                grid = np.array(ssr['stress_field']) if metric in ("s_bend", "rrg") else np.array(ssr['displacement_field'])
                return XI, YI, grid, ["X", "Y"]

        # 기본 시뮬레이션 데이터 추출 (Low-Fidelity)
        mode = self._contour_mode_var.get()
        return get_contour_grid_data(self.sim, step, comp, metric, mode)

    def _on_show_contour_frame(self):
        """[v4.2] 선택된 모든 부품의 컨투어를 비보정(Non-modal) 팝업 창으로 출력합니다."""
        selected_comps = [c for c, v in self._contour_comp_vars.items() if v.get()]
        selected_metrics = [k for k, v in self._contour_metric_vars.items() if v.get()]
        
        if not selected_comps or not selected_metrics:
            messagebox.showwarning("선택 오류", "이미 지표나 부품을 선택했는지 확인하세요.")
            return

        # 기존 창이 있으면 파괴하고 새로 열기
        if hasattr(self, '_contour_popup') and self._contour_popup.winfo_exists():
            self._contour_popup.destroy()

        self._contour_popup = tk.Toplevel(self)
        self._contour_popup.title("WHTOOLS: 2D Structural Contour Matrix (Live Sync)")
        self._contour_popup.geometry("1150x850")
        
        # [v4.3] 상단 툴바 (Tight Layout, Save 버튼)
        toolbar_f = ttk.Frame(self._contour_popup)
        toolbar_f.pack(fill="x", padx=5, pady=2)
        
        def _exec_tight():
            if hasattr(self, '_contour_fig_list'):
                for fig, canvas, _, _ in self._contour_fig_list:
                    fig.tight_layout()
                    canvas.draw_idle()

        def _exec_save():
            if hasattr(self, '_contour_fig_list') and self._contour_fig_list:
                path = filedialog.asksaveasfilename(defaultextension=".png", title="Save Matrix Screenshot")
                if path:
                    # 첫 번째 피겨 기반으로 저장하거나 팝업 전체 캡처 유도 (여기서는 첫 피겨 저장)
                    self._contour_fig_list[0][0].savefig(path)
                    messagebox.showinfo("Save", f"저장되었습니다: {path}")

        ttk.Button(toolbar_f, text="📐 Tight Layout", command=_exec_tight).pack(side="left", padx=2)
        ttk.Button(toolbar_f, text="💾 Save Current View", command=_exec_save).pack(side="left", padx=2)
        ttk.Label(toolbar_f, text=" |  (Hover for data, Click to Pin Marker)", font=get_ui_font(8), foreground="gray").pack(side="left", padx=10)

        main_canv = tk.Canvas(self._contour_popup, bg="#f5f5f5")
        vsb = ttk.Scrollbar(self._contour_popup, orient="vertical", command=main_canv.yview)
        scroll_f = ttk.Frame(main_canv)
        main_canv.create_window((0, 0), window=scroll_f, anchor="nw")
        main_canv.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        main_canv.pack(side="left", fill="both", expand=True)

        self._contour_fig_list = []
        step = self._current_frame
        
        # 매트릭스 레이아웃 (행: 부품, 열: 지표)
        for r, comp in enumerate(selected_comps):
            row_f = ttk.Frame(scroll_f)
            row_f.pack(fill="x", pady=12)
            ttk.Label(row_f, text=f"■ Component: {comp.upper()}", font=get_ui_font(10, True)).pack(anchor="w", padx=20)
            
            inner_row = ttk.Frame(row_f)
            inner_row.pack(fill="x")
            
            for c, metric in enumerate(selected_metrics):
                fig = Figure(figsize=(4.2, 3.8), dpi=90)
                ax = fig.add_subplot(111)
                
                f_box = ttk.Frame(inner_row, relief="ridge", borderwidth=1)
                f_box.pack(side="left", padx=8, pady=5)
                
                canvas = FigureCanvasTkAgg(fig, master=f_box)
                canvas.get_tk_widget().pack()
                
                self._draw_single_contour(ax, comp, metric, step)
                fig.tight_layout()
                canvas.draw()
                
                # [v4.3] 인터랙티브 기능 연결 (팝업 Figure에도 적용)
                if self.plot_manager:
                    self.plot_manager.attach_interactivity(fig)
                
                self._contour_fig_list.append((fig, canvas, comp, metric))

        scroll_f.update_idletasks()
        main_canv.config(scrollregion=main_canv.bbox("all"))

    def _update_popup_contours(self, step=None):
        """[v4.2] 열려 있는 팝업 창의 모든 컨투어를 현재 시점으로 갱신합니다."""
        if not hasattr(self, '_contour_popup') or not self._contour_popup.winfo_exists():
            return
        
        if step is None:
            step = self._current_frame
            
        for fig, canvas, comp, metric in self._contour_fig_list:
            ax = fig.axes[0]
            ax.clear()
            self._draw_single_contour(ax, comp, metric, step)
            fig.tight_layout()
            canvas.draw()

    # ==============================================================
    # [v4.3] 데이터 분석 및 익스포트 메뉴 핸들러
    # ==============================================================

    def _on_open_export_dialog(self):
        """다중 Figure 선택 및 데이터 내보내기 팝업을 엽니다."""
        if not _EXT_AVAILABLE:
            messagebox.showwarning("Error", "확장 모듈(mpl_extension.py)을 로드할 수 없습니다.")
            return
        ExportDialog(self, self.plot_manager)

    def _on_copy_active_fig_to_clipboard(self):
        """현재 활성화된 Matplotlib Figure의 데이터를 클립보드에 복사합니다."""
        active_fig = plt.gcf()
        if active_fig:
            PlotExporter.to_clipboard(active_fig, self)
        else:
            messagebox.showwarning("Warning", "활성화된 그래프 창이 없습니다.")

    def _on_clear_all_annotations(self):
        """모든 열린 그래프 창의 어노테이션 마킹을 제거합니다."""
        if not self.plot_manager: return
        for fig in self.plot_manager.active_figs:
            self.plot_manager.clear_annotations(fig)
        messagebox.showinfo("Clean", "모든 그래프의 마킹이 제거되었습니다.")

    def _update_embedded_contour(self):
        """[v4.3] 임베디드 영역에 매트릭스 컨투어를 실시간으로 그립니다. (리사이즈 및 컬러바 중첩 해결)"""
        if not self.winfo_exists(): return
        
        selected_comps = [c for c, v in self._contour_comp_vars.items() if v.get()]
        selected_metrics = [k for k, v in self._contour_metric_vars.items() if v.get()]
        
        if not selected_comps or not selected_metrics: return
            
        step = self._current_frame
        
        # [v4.3] plt.figure() 대신 Figure() 객체 직접 사용 (Tkinter 연동 최적화)
        if self._canvas_embed is None:
            self.plot_placeholder.place_forget()
            self._fig_embed = Figure(figsize=(8, 5), dpi=100)
            self._canvas_embed = FigureCanvasTkAgg(self._fig_embed, master=self.plot_container)
            cw = self._canvas_embed.get_tk_widget()
            cw.pack(fill="both", expand=True)
            
            # 창 크기 변경 시 레이아웃 자동 조정 바인딩
            cw.bind("<Configure>", lambda e: self._on_resize_embedded_contour(e))

        self._fig_embed.clear()
        
        n_comps = len(selected_comps)
        n_metrics = len(selected_metrics)
        
        # 격자 생성 (행: 컴포넌트, 열: 지표)
        fig_axes = self._fig_embed.subplots(n_comps, n_metrics, squeeze=False)
        
        time_hist = self.sim.time_history
        curr_t = time_hist[step] if step < len(time_hist) else 0.0
        header = f"Real-time Contour Matrix (t = {curr_t:.4f}s)"
        self._fig_embed.suptitle(header, fontsize=11, fontweight='bold', y=0.98)

        for r, comp in enumerate(selected_comps):
            for c, metric in enumerate(selected_metrics):
                ax = fig_axes[r, c]
                self._draw_single_contour(ax, comp, metric, step)
                if r == 0: ax.set_title(f"Metric: {metric.upper()}", fontsize=9)
                if c == 0: ax.set_ylabel(comp, fontsize=9, fontweight='bold')
                
        self._fig_embed.tight_layout(rect=[0, 0, 1, 0.95])
        self._canvas_embed.draw_idle()

    def _on_resize_embedded_contour(self, event):
        """창 크기 조절 시 컨투어 그래프의 tight_layout을 재조정합니다."""
        if hasattr(self, '_fig_embed') and self._fig_embed:
            # 너무 빈번한 호출 방지를 위해 idle 상태에서 드로우
            self._fig_embed.tight_layout(rect=[0, 0, 1, 0.95])
            self._canvas_embed.draw_idle()

    def _draw_single_contour(self, ax, comp, metric, step):
        """[v4.2] 실시간 데이터 스케일링 및 최소/최대 마킹이 적용된 정밀 컨투어 엔진"""
        result = self._get_contour_grid_at(step, comp, metric)
        if result is None:
            ax.text(0.5, 0.5, f"(No Data: {comp})", ha='center', fontsize=8)
            ax.axis('off')
            return
            
        X_orig, Y_orig, grid, dims = result
        
        # [v4.2] 동적 스케일링: 현재 프레임 데이터의 Min/Max를 Legend에 반영
        # [v4.3] 수동 스케일링 지원 (auto가 아닐 경우 사용자 입력값 사용)
        try:
            v_min_user = self._matrix_vmin_var.get().lower()
            v_max_user = self._matrix_vmax_var.get().lower()
            
            vmin = np.min(grid) if v_min_user == "auto" else float(v_min_user)
            vmax = np.max(grid) if v_max_user == "auto" else float(v_max_user)
        except:
            vmin = np.min(grid)
            vmax = np.max(grid)

        if vmin == vmax: vmax = vmin + 0.01

        # [v4.9] Sharp SSR (Structural Surface Reconstruction) 
        # JAX-SSR 결과물(이미 고해상도인 경우)은 추가 PSR 보간을 건너뜁니다.
        is_already_high_res = (grid.shape[0] >= 30 and grid.shape[1] >= 30)
        
        if self._ssr_mode_var.get() and not is_already_high_res and grid.size >= 6:
            try:
                X, Y, Z = apply_psr_interpolation(X_orig, Y_orig, grid, vmin, vmax)
            except Exception as e:
                print(f"[WHTOOLS] SSR Analysis (Legacy) failed: {e}")
                X, Y, Z = X_orig, Y_orig, grid
        else:
            X, Y, Z = X_orig, Y_orig, grid

        # [v4.6] Sharp-Edge 시각화: bicubic 대신 bilinear 또는 nearest 사용
        extent = [X.min(), X.max(), Y.min(), Y.max()]
        cnt = ax.imshow(Z, extent=extent, origin='lower', cmap="turbo", vmin=vmin, vmax=vmax, 
                        interpolation='bilinear', aspect='equal')

        
        # 폰트 8pt 고정 (축 레이블 및 타이틀에 지표 명칭 추가)
        ax.set_xlabel(f"{dims[0]} (m)", fontsize=8)
        ax.set_ylabel(f"{dims[1]} (m)", fontsize=8)
        ax.set_title(f"[{metric.upper()}] {comp.upper()}", fontsize=9, fontweight='bold')
        ax.tick_params(labelsize=8)

        # [v4.2] Min/Max 지점 마킹 (화살표 + 8pt 텍스트)
        try:
            max_idx = np.unravel_index(np.argmax(grid), grid.shape)
            min_idx = np.unravel_index(np.argmin(grid), grid.shape)
            
            # 최대값 마킹 (Red)
            ax.annotate(f"MAX:{vmax:.2f}", xy=(X_orig[max_idx], Y_orig[max_idx]), 
                        xytext=(5, 10), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color='red'), color='red', fontsize=8, fontweight='bold')
            # 최소값 마킹 (Blue)
            ax.annotate(f"MIN:{vmin:.2f}", xy=(X_orig[min_idx], Y_orig[min_idx]), 
                        xytext=(5, -15), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color='blue'), color='blue', fontsize=8, fontweight='bold')
        except:
            pass
        
        # 컬러바 (우측 정렬 - 중복 생성 방지 로직 적용)
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        
        # [v4.3] 기존에 생성된 컬러바 축(cax)이 있는지 확인하여 재사용하거나 초기화
        if hasattr(ax, '_cax') and ax._cax in ax.figure.axes:
            cax = ax._cax
            cax.clear()
        else:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            ax._cax = cax # 나중에 재사용하기 위해 저장

        cb = ax.figure.colorbar(cnt, cax=cax)
        cb.ax.tick_params(labelsize=8)
        
        if self._show_pba_var.get():
            self._overlay_pba_vectors(ax, comp, step)
        
        # PBA 오버레이
        if self._show_pba_var.get():
            self._overlay_pba_vectors(ax, comp, step)

    def _overlay_pba_vectors(self, ax, comp, step):
        """[v4.1] 현재 시점의 PBA (Principal Bending Axis) 벡터를 중앙에 표시합니다."""
        sts = self.sim.structural_time_series
        # [v4.6.2] 이미 Decimation된 1:1 매핑 데이터 사용
        dec_step = step
        
        # [v4.1] 컴포넌트 전체 혹은 전역 pba_angle 사용
        # 지표가 'angle'인 경우 혹은 전역 GTI 산출 시 사용된 주축을 표시
        if 'pba_angle' in sts and dec_step < len(sts['pba_angle']):
            angle_deg = sts['pba_angle'][dec_step]
            
            # 축 범위 기반 화살표 크기 결정
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            mag = (xlim[1] - xlim[0]) * 0.3
            
            cx, cy = (xlim[0] + xlim[1])/2, (ylim[0] + ylim[1])/2
            angle_rad = np.radians(angle_deg)
            dx = np.cos(angle_rad) * mag
            dy = np.sin(angle_rad) * mag
            
            # 그림자 효과 (Black outline)
            ax.quiver(cx, cy, dx, dy, color='white', scale_units='xy', scale=1, width=0.012, 
                      edgecolor='black', linewidth=0.5, headwidth=4, label='PBA Vector')
            
            ax.text(cx, cy - (ylim[1]-ylim[0])*0.12, f"PBA: {angle_deg:.1f}°", color='white', 
                    fontsize=7, fontweight='bold', ha='center',
                    bbox=dict(facecolor='black', alpha=0.4, edgecolor='none', pad=1))

    def _on_animate_contour(self):
        """
        애니메이션을 시작합니다.
        Play 버튼과 연동되어 슬라이더가 자동으로 진행됩니다.
        """
        if self._anim_running:
            messagebox.showinfo("알림", "이미 애니메이션이 실행 중입니다.\n⏸ Pause 버튼으로 정지하세요.")
            return
        self._notebook.select(2)  # 컨투어 탭으로 포커스
        self._on_play()

    def _on_save_contour_frames(self):
        """
        전체 시간 구간의 컨투어 프레임을 PNG 파일로 일괄 저장합니다.
        5스텝마다 1개씩 저장 (Decimation 동기화).
        """
        comp = self._comp_var.get()
        metric = self._contour_metric_var.get()
        save_dir = os.path.join(self.sim.output_dir, f"contour_{comp}_{metric}")
        os.makedirs(save_dir, exist_ok=True)

        _prev_backend = matplotlib.get_backend()
        matplotlib.use("Agg")

        try:
            total = self._total_frames
            saved = 0
            for step in range(0, total, 5):
                result = self._get_contour_grid_at(step, comp, metric)
                if result is None: continue
                i_arr, j_arr, grid = result
                t_val = self.sim.time_history[step] if step < len(self.sim.time_history) else 0.0

                fig, ax = plt.subplots(1, 1, figsize=(7, 5))
                JJ, II = np.meshgrid(j_arr, i_arr)
                vmin, vmax = 0.0, max(float(grid.max()), 0.01)
                cnt = ax.contourf(JJ, II, grid, levels=20, cmap="turbo", vmin=vmin, vmax=vmax)
                fig.colorbar(cnt, ax=ax, shrink=0.85)
                ax.set_title(f"[{comp}] {metric.upper()}  t = {t_val:.4f} s", fontsize=10)
                ax.set_aspect("equal")
                plt.tight_layout()
                fname = os.path.join(save_dir, f"frame_{step:05d}.png")
                plt.savefig(fname, dpi=120)
                plt.close(fig)
                saved += 1
        finally:
            matplotlib.use(_prev_backend)

        messagebox.showinfo("저장 완료", f"총 {saved}개 프레임을 저장했습니다.\n경로: {save_dir}")

    def _on_apply_heatmap(self):
        """[v4] 현재 시점의 컨투어 데이터를 MuJoCo 뷰어 상의 geom 색상으로 직접 전사(Mapping)합니다."""
        step = self._current_frame
        metric = self._contour_metric_var.get()
        
        self.sim.log(f"\n>> [UI] MuJoCo 뷰어 전역 히트맵 업데이트 (Metric: {metric})...")
        
        applied_count = 0
        for comp_name, comp_m in list(self.sim.metrics.items()):
            result = self._get_contour_grid_at(step, comp_name, metric)
            if result is None: continue
            _, _, Z = result
            vmax = Z.max() if Z.max() > 0 else 1.0
            
            if comp_name not in self.sim.components: continue
            
            for (i, j, k), body_uid in self.sim.components[comp_name].items():
                if i >= Z.shape[0] or j >= Z.shape[1]: continue
                val = Z[i, j]
                norm_val = np.clip(val / vmax, 0, 1)
                rgba = list(matplotlib.cm.turbo(norm_val))
                
                gid = self._find_geom_by_index(comp_name, (i, j, k))
                if gid != -1:
                    self.sim.model.geom_rgba[gid] = rgba
                    applied_count += 1
        
        if self.sim.viewer:
            self.sim.viewer.sync()
            messagebox.showinfo("완료", f"총 {applied_count}개 블록에 히트맵 색상을 적용했습니다.")

    def _open_ssr_analyzer(self):
        """[v4.4] SSR 분석기 대화상자를 엽니다."""
        SSRAnalyzerDialog(self, self.sim)

    def on_close(self):
        """UI 종료 시 애니메이션을 정지하고 창을 닫습니다."""
        self._on_stop()
        self.destroy()
        if self.master:
            try:
                self.master.destroy()
                self.master.quit()
            except:
                pass

class SSRAnalyzerDialog(tk.Toplevel):
    """
    [WHTOOLS v4.4.1] Precision Stress Field Analyzer (SSR) 대화상자.
    고해상도 쉘 이론 기반 구조 해석을 수행합니다. (속성 자동 예측 지원)
    """
    def __init__(self, parent, sim):
        super().__init__(parent)
        self.parent = parent
        self.sim = sim
        self.title("🔍 Precision Stress Field Analyzer (SSR) - [WHTOOLS]")
        self.geometry("620x650") # 가로 확장
        self.transient(parent)
        self.grab_set()

        self._comp_configs = {}
        self._build_ui()

    def _build_ui(self):
        main_f = ttk.Frame(self, padding=20)
        main_f.pack(fill="both", expand=True)

        header = ttk.Label(main_f, text="Precision Surface Metrics Audit (SSR)", 
                           font=get_ui_font(12, bold=True))
        header.pack(pady=(0, 10))

        # 1. 대상 부품 설정 (리스트형)
        comp_f = ttk.LabelFrame(main_f, text=" 1. 분석 대상 및 물리 속성 설정 (Auto-Predicted) ")
        comp_f.pack(fill="both", expand=True, pady=5)
        
        # 스크롤 영역 설정
        canvas = tk.Canvas(comp_f, highlightthickness=0)
        vsb = ttk.Scrollbar(comp_f, orient="vertical", command=canvas.yview)
        scroll_f = ttk.Frame(canvas)
        
        scroll_f.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_f, anchor="nw", width=560)
        canvas.configure(yscrollcommand=vsb.set)
        
        canvas.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        vsb.pack(side="right", fill="y")

        # 헤더 라벨
        h_row = ttk.Frame(scroll_f)
        h_row.pack(fill="x", pady=2)
        ttk.Label(h_row, text="부품명", font=get_ui_font(9, True), width=23).pack(side="left", padx=5)
        ttk.Label(h_row, text="두께 t(mm)", font=get_ui_font(9, True), width=12).pack(side="left", padx=5)
        ttk.Label(h_row, text="영률 E(GPa)", font=get_ui_font(9, True), width=12).pack(side="left", padx=5)

        for c_name in sorted(self.sim.metrics.keys()):
            # 초기값 예측 및 적합성 판별
            t_pred, e_pred = self._predict_properties(c_name)
            
            # [v4.4.2] 강체(Rigid)이거나 분석에 부적합한 경우 리스트에서 제외
            if t_pred is None:
                continue
            
            row = ttk.Frame(scroll_f)
            row.pack(fill="x", pady=2)
            
            sel_var = tk.BooleanVar(value=True)
            t_var = tk.DoubleVar(value=t_pred)
            e_var = tk.DoubleVar(value=e_pred)
            
            self._comp_configs[c_name] = {"selected": sel_var, "t": t_var, "e": e_var}
            
            ttk.Checkbutton(row, text=c_name, variable=sel_var, width=22).pack(side="left", padx=5)
            ttk.Entry(row, textvariable=t_var, width=10).pack(side="left", padx=5)
            # t_var = thick (mm) -> mm
            ttk.Entry(row, textvariable=e_var, width=10).pack(side="left", padx=5)
            # e_var = Young's modulus (GPa) -> GPa

        # 2. 분석 해상도 및 범위
        range_f = ttk.LabelFrame(main_f, text=" 2. 분석 범위 및 해상도 ")
        range_f.pack(fill="x", pady=5)

        ttk.Label(range_f, text="프레임 샘플링 간격 (Stride):").pack(anchor="w", padx=10)
        self._stride_var = tk.IntVar(value=10)
        ttk.Scale(range_f, from_=1, to=100, variable=self._stride_var, orient="horizontal").pack(fill="x", padx=10)
        lbl_stride = ttk.Label(range_f, text="매 10 프레임마다 분석")
        lbl_stride.pack(padx=10)
        self._stride_var.trace_add("write", lambda *a: lbl_stride.config(text=f"매 {self._stride_var.get()} 프레임마다 분석"))

        # 분석 시작 버튼
        self._btn_run = ttk.Button(main_f, text="🚀 Run High-Fidelity SSR Analysis", 
                                   command=self._start_calc, style="Accent.TButton")
        self._btn_run.pack(fill="x", pady=20)

        # 진행 바
        self._prog_var = tk.DoubleVar(value=0)
        self._prog_bar = ttk.Progressbar(main_f, variable=self._prog_var, maximum=100)
        self._prog_bar.pack(fill="x")
        self._status_lbl = ttk.Label(main_f, text="대기 중...")
        self._status_lbl.pack(pady=5)

    def _predict_properties(self, comp_name):
        """
        [v4.4.2] Weld Solref와 Depth 정보를 기반으로 Shell 속성을 자동 예측합니다.
        강체(Rigid)이거나 부품 수가 부족한 경우 (None, None)을 반환하여 리스트에서 제외합니다.
        """
        m = self.sim.model
        
        # 0. 컴포넌트 유효성 및 블록 수 확인
        if comp_name not in self.sim.components:
            return None, None
            
        # [v4.4.2] SSR 분석에는 최소 2개 이상의 블록이 필요 (보통 4개 이상 권장)
        if len(self.sim.components[comp_name]) < 2:
            return None, None

        thick = 2.0  # mm (Default fallback)
        young = 70.0 # GPa (Default fallback)
        
        # 1. 첫 번째 블록에서 두께(depth) 추출
        grid_keys = list(self.sim.components[comp_name].keys())
        if grid_keys:
            # MuJoCo body index
            b_id = self.sim.components[comp_name][grid_keys[0]]
            # 바디의 첫 geom 찾기
            if m.body_geomnum[b_id] > 0:
                gid = m.body_geomadr[b_id]
                # depth는 보통 size[2] (MuJoCo half-size -> full size = *2)
                thick = m.geom_size[gid][2] * 2.0 * 1000.0 # m -> mm
                
                # 2. Weld Solref 기반 탄성계수 역산
                timeconst = 0.02 # Default MuJoCo
                found_sol = False
                for i in range(m.neq):
                    if m.eq_type[i] == 2: # mjEQ_WELD
                        if m.eq_obj1id[i] == b_id or m.eq_obj2id[i] == b_id:
                            timeconst = m.eq_solref[i, 0]
                            found_sol = True
                            break
                if not found_sol:
                    timeconst = m.geom_solref[gid, 0]

                # [v4.4.2] 강체 판별: timeconst가 0이하거나 너무 작으면(무한 강성) 제외
                if timeconst < 0.0005:
                    return None, None
                
                k_lin = 1.0 / (timeconst**2)
                
                # k = EA/L -> E = kL/A
                dx, dy = m.geom_size[gid][0]*2, m.geom_size[gid][1]*2
                area = dx * dy
                if area > 1e-6:
                    young = (k_lin * (thick/1000.0)) / area / 1e9 # GPa
        
        return round(max(0.1, thick), 2), round(max(0.1, young), 2)

    def _start_calc(self):
        self._btn_run.config(state="disabled")
        threading.Thread(target=self._calc_worker, daemon=True).start()

    def _calc_worker(self):
        try:
            stride = max(1, self._stride_var.get())
            selected_comps = [c for c, cfg in self._comp_configs.items() if cfg["selected"].get()]
            
            if not selected_comps:
                self.after(0, lambda: messagebox.showwarning("경고", "분석할 부품을 하나 이상 선택하세요."))
                return

            total_steps = len(self.sim.time_history)
            target_steps = range(0, total_steps, stride)
            total_work = len(selected_comps)
            current_work = 0

            for comp in selected_comps:
                self.after(0, lambda c=comp: self._status_lbl.config(text=f"Analyzing {c} (JAX-Engine)..."))
                
                # [v4.9.7] JAX-SSR 우선 시도 (전체 프레임 배치 처리)
                jax_result = None
                if _JAX_AVAILABLE:
                    # 사용자 입력값 추출
                    comp_cfg = self._comp_configs.get(comp, {})
                    u_thick = comp_cfg.get("t", tk.DoubleVar(value=2.0)).get()
                    u_young_gpa = comp_cfg.get("e", tk.DoubleVar(value=210.0)).get()
                    
                    # whts_postprocess_engine의 JAX 인터페이스 호출 (E는 MPa 단위로 변환)
                    jax_result = apply_jax_kirchhoff_ssr(
                        self.sim, comp, 
                        thickness=u_thick, 
                        youngs_modulus=u_young_gpa * 1000.0, 
                        res=40
                    )
                
                if jax_result is not None:
                    # JAX 결과는 전체 프레임 데이터를 포함함
                    if comp not in self.parent._ssr_data_store:
                        self.parent._ssr_data_store[comp] = {}
                    self.parent._ssr_data_store[comp]['JAX_BATCH'] = jax_result
                else:
                    # JAX 실패 시 기존 PSR 루프로 폴백
                    self.after(0, lambda c=comp: self._status_lbl.config(text=f"Fallback to PSR Loop: {c}..."))
                    if comp not in self.parent._ssr_data_store:
                        self.parent._ssr_data_store[comp] = {}
                    
                    for step in target_steps:
                        ssr_result = self._calculate_one_ssr(comp, step, {})
                        if "error" not in ssr_result:
                            self.parent._ssr_data_store[comp][step] = ssr_result
                
                current_work += 1
                prog = (current_work / total_work) * 100
                self.after(0, lambda p=prog: self._prog_var.set(p))

            self.after(0, self._on_complete)
        except Exception as e:
            self.after(0, lambda err=e: messagebox.showerror("Error", f"SSR Analysis failed: {err}"))
        finally:
            self.after(0, lambda: self._btn_run.config(state="normal"))

    def _calculate_one_ssr(self, comp, step, config):
        """특정 컴포넌트와 스텝에 대해 SSR 연산을 수행합니다."""
        try:
            positions = []
            values = []
            
            # components [(i, j, k)] -> body_uid
            for idx, b_id in self.sim.components[comp].items():
                nom_pos = self.sim.nominal_local_pos.get(b_id, [0, 0, 0])
                # bend 데이터 활용
                bend_val = 0.0
                try:
                    bend_val = self.sim.metrics[comp]['all_blocks_bend'][idx][step]
                except (IndexError, KeyError):
                    pass
                
                positions.append([nom_pos[0], nom_pos[1]]) # X-Y Plane 가정
                values.append(bend_val)
            
            if not positions:
                return {"error": "No blocks found"}
                
            return compute_ssr_shell_metrics(comp, np.array(positions), np.array(values), config)
        except Exception as e:
            return {"error": str(e)}

    def _on_complete(self):
        # [WHTOOLS] SSR 분석 완료 팝업 제거 (사용자 요청)
        # messagebox.showinfo("완료", "SSR 고정밀 구조 해석이 완료되었습니다.\n이제 컨투어 탭에서 '고정밀 모드' 시 시뮬레이션 데이터를 확인할 수 있습니다.")
        print(" >> SSR High-Precision Analysis Complete.")
        self.destroy()

