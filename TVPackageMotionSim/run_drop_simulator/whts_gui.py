import os
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from typing import Any, Dict, List, Optional, Tuple, Union

class ConfigEditor(tk.Toplevel):
    """
    [WHTOOLS] 시뮬레이션 설정 및 제어를 위한 Tkinter GUI입니다.
    사용자는 이 UI를 통해 실시간으로 물리 파라미터를 수정하고 시뮬레이션 거동을 제어할 수 있습니다.
    
    Attributes:
        sim (Any): 제어 대상인 DropSimulator 인스턴스
        widget_map (Dict): 설정 키와 UI 위젯 변수 간의 매핑
    """
    def __init__(self, parent_sim: Any):
        super().__init__()
        self.sim = parent_sim
        self.title("WHTOOLS Config Control UI - v4")
        self.geometry("750x900")
        self.protocol("WM_DELETE_WINDOW", self.on_cancel)
        
        # [데이터 정의] 설정 그룹 분류 및 설명
        self.groups = {
            "Environment": ["drop_height", "drop_mode", "drop_direction", "env_gravity", "env_wind"],
            "Physics": ["sim_timestep", "reporting_interval", "sim_duration", "sim_integrator", "sim_iterations", "sol_recession"],
            "Structure": ["target_mass", "num_balancing_masses", "box_w", "box_h", "box_d", "mass_cushion", "mass_chassis"],
            "Cushion": ["enable_plasticity", "cush_yield_strain", "cush_yield_pressure", "cush_weld_solref_timec", "cush_weld_solref_damprr", "plasticity_ratio"]
        }
        
        self.desc_map = {
            "drop_height": "낙하 높이 (Drop height, m)",
            "drop_mode": "매핑 모드 (LTL, PARCEL, CUSTOM)",
            "drop_direction": "충격 방향 (Direction, e.g. Corner 2-3-5)",
            "env_gravity": "중력 가속도 (m/s^2)",
            "target_mass": "목표 총 질량 (Target Mass, kg)",
            "cush_yield_strain": "항복 변형률 (Strain threshold, 0.0~1.0)",
            "cush_yield_pressure": "항복 압력 (Pressure threshold, Pa)",
            "enable_plasticity": "소성 변형 활성화 (Plasticity On/Off)",
            "sim_timestep": "타임스텝 (Timestep, s)",
            "reporting_interval": "데이터 저장 주기 (Save Intv, s)",
            "box_w": "박스 가로(W)", "box_h": "박스 세로(H)", "box_d": "박스 깊이(D)"
        }
        
        self.widget_map = {}
        self.build_ui()

    def build_ui(self) -> None:
        """UI 컴포넌트들을 배치하고 초기화합니다."""
        # 1. 상단 배너 로드 (v4 패키지 내부 리소스 경로 사용)
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        banner_path = os.path.join(pkg_dir, "resources", "ui_banner.png")
        
        if os.path.exists(banner_path):
            try:
                img = Image.open(banner_path)
                w, h = img.size
                target_w = 720
                target_h = int(h * (target_w / w))
                img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                self.banner_img = ImageTk.PhotoImage(img)
                lbl = tk.Label(self, image=self.banner_img, bg="#1a1a1a")
                lbl.pack(fill="x", padx=10, pady=5)
            except Exception as e:
                print(f"[GUI] Banner loading failed: {e}")

        # 2. 제어판 (Simulation Controls)
        ctrl_frame = ttk.LabelFrame(self, text=" [ Simulation Advanced Controls ] ")
        ctrl_frame.pack(fill="x", padx=10, pady=5)
        
        # 버튼 박스 1: Play/Reset/Print/Status
        btn_box1 = ttk.Frame(ctrl_frame)
        btn_box1.pack(fill="x", pady=5)
        
        self.play_btn = ttk.Button(btn_box1, text="Play / Pause", command=self.on_toggle_play)
        self.play_btn.pack(side="left", padx=5)
        
        ttk.Button(btn_box1, text="Rewind (Reset)", command=self.on_reset).pack(side="left", padx=5)
        ttk.Button(btn_box1, text="View Cam Info", command=self.on_print_view).pack(side="left", padx=5)
        
        self.status_var = tk.StringVar(value="Step: 0 | Time: 0.000s")
        self.status_lbl = ttk.Label(btn_box1, textvariable=self.status_var, font=("Consolas", 10, "bold"), foreground="#007acc")
        self.status_lbl.pack(side="left", padx=20)
        
        # 버튼 박스 2: Step Size Slider
        btn_box2 = ttk.Frame(ctrl_frame)
        btn_box2.pack(fill="x", pady=5)
        
        ttk.Label(btn_box2, text="Step Magnitude:").pack(side="left", padx=(5, 5))
        self.step_val_var = tk.IntVar(value=10)
        self.step_label = ttk.Label(btn_box2, text="10", width=4)
        self.step_label.pack(side="left")
        
        self.step_scale = ttk.Scale(btn_box2, from_=1, to=200, orient="horizontal", 
                                   variable=self.step_val_var, command=self.on_scale_change)
        self.step_scale.pack(side="left", fill="x", expand=True, padx=5)
        
        # 버튼 박스 3: Step Navigation (애니메이션 이동)
        btn_box3 = ttk.Frame(ctrl_frame)
        btn_box3.pack(fill="x", pady=5)
        
        ttk.Button(btn_box3, text="<< Step Back (Anim)", command=self.on_step_back).pack(side="left", padx=5, expand=True, fill="x")
        ttk.Button(btn_box3, text="Step Forward (Anim) >>", command=self.on_step_forward).pack(side="left", padx=5, expand=True, fill="x")

        # 3. 탭 인터페이스 (Notebook) - 설정 및 가이드
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=5)
        
        # [Tab 1] Settings (Scrollable)
        self.settings_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_tab, text=" Configuration Editor ")
        
        self.canvas = tk.Canvas(self.settings_tab)
        self.v_scrollbar = ttk.Scrollbar(self.settings_tab, orient="vertical", command=self.canvas.yview)
        self.scroll_frame = ttk.Frame(self.canvas)
        self.scroll_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set)
        
        self.v_scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        # 분류별 행 추가
        all_keys = set(self.sim.config.keys())
        for group_name, keys in self.groups.items():
            g_frame = ttk.LabelFrame(self.scroll_frame, text=f" {group_name} ")
            g_frame.pack(fill="x", padx=5, pady=5)
            for k in keys:
                if k in self.sim.config:
                    self.add_config_row(g_frame, k, self.sim.config[k])
                    all_keys.discard(k)
        
        if all_keys:
            etc_frame = ttk.LabelFrame(self.scroll_frame, text=" [Miscellaneous] ")
            etc_frame.pack(fill="x", padx=5, pady=5)
            for k in sorted(list(all_keys)):
                self.add_config_row(etc_frame, k, self.sim.config[k])

        # 4. 하단 액션 버튼
        action_frame = ttk.Frame(self)
        action_frame.pack(fill="x", pady=10)
        ttk.Button(action_frame, text="Apply & Reload", command=self.on_apply).pack(side="right", padx=10)
        ttk.Button(action_frame, text="Cancel", command=self.on_cancel).pack(side="right", padx=5)

    def add_config_row(self, parent: ttk.Frame, key: str, val: Any) -> None:
        """설정 항목을 위한 한 행(Row)을 생성합니다."""
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=5, pady=2)
        
        lbl = ttk.Label(row, text=f"{key:30}", width=25, font=("Consolas", 9))
        lbl.pack(side="left")
        
        var = tk.StringVar(value=str(val))
        if isinstance(val, bool):
            ent = ttk.Combobox(row, textvariable=var, values=["True", "False"], width=13, state="readonly")
        else:
            width = 25 if isinstance(val, (list, tuple)) else 15
            ent = ttk.Entry(row, textvariable=var, width=width)
        
        ent.pack(side="left", padx=5)
        desc = self.desc_map.get(key, "-")
        ttk.Label(row, text=f"({desc})", foreground="gray", font=("NanumGothic", 8)).pack(side="left", padx=5)
        self.widget_map[key] = (var, type(val))

    def update_status(self, step: int, t: float) -> None:
        """엔진으로부터 현재 진행 상태를 받아 UI를 갱신합니다."""
        if self.winfo_exists():
            self.status_var.set(f"Step: {step:5d} | Time: {t:6.3f}s")

    def on_apply(self) -> None:
        """변경된 값을 유효성 검사 후 엔진의 config에 반영합니다."""
        new_values = {}
        for key, (var, original_type) in self.widget_map.items():
            raw_val = var.get().strip()
            try:
                if original_type == bool:
                    new_val = raw_val == "True"
                elif original_type in (list, tuple):
                    new_val = eval(raw_val) # 보안 주의: 사용자 정의 입력 시 권장
                else:
                    new_val = original_type(raw_val)
                new_values[key] = new_val
            except:
                messagebox.showerror("Error", f"Invalid format for '{key}': {raw_val}")
                return

        self.sim.config.update(new_values)
        self.sim.ctrl_reload_request = True # 엔진에 재로드 신호 전달
        self.destroy()

    def on_toggle_play(self) -> None: self.sim.ctrl_paused = not self.sim.ctrl_paused
    def on_reset(self) -> None: self.sim.ctrl_reset_request = True
    def on_print_view(self) -> None: self.sim.ctrl_print_view = True
    def on_step_forward(self) -> None: self.sim.ctrl_step_forward_count = self.step_val_var.get()
    def on_step_back(self) -> None: self.sim.ctrl_step_backward_count = self.step_val_var.get()
    def on_scale_change(self, val: str) -> None: self.step_label.config(text=str(int(float(val))))
    def _on_mousewheel(self, event: Any) -> None: self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    def on_cancel(self) -> None: self.destroy()
