import sys
import io

# Windows 콘솔 및 리다이렉션 환경에서의 인코딩 안정성 확보
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import json
import time
import shutil
import pickle
import math
import pprint
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

# =====================================================================
# [WHTOOLS: Advanced Engineering Simulation Suite]
# =====================================================================
# 외부 모듈 참조 (동일 디렉토리 내 존재 가정)
# - create_model: XML 모델 생성 및 초기 관성 특성 반환
# - get_default_config: 시뮬레이션의 기본 물리/설계 파라미터 제공
from run_discrete_builder import create_model, get_default_config

# =====================================================================
# [1] 데이터 구조: 시뮬레이션 결과 컨테이너 (DropSimResult)
# =====================================================================
@dataclass
class DropSimResult:
    """
    시뮬레이션 전체 결과 데이터를 담는 통합 데이터 구조체입니다.
    이 객체는 바이너리(Pickle) 파일로 저장되어 추후 대량의 DOE 데이터 분석이나 
    실험 데이터와의 매칭(Correlation) 작업에서 핵심 자산으로 활용됩니다.
    
    Attributes:
        config (Dict): 시뮬레이션 수행 시 사용된 모든 설정값 (Physics/Geometry)
        metrics (Dict): 부품별 굽힘(Bending), 비틂(Twist), 뒤틀림(Distortion) 분석 지표
        max_g_force (float): 전체 조립체(Packaging Box)의 무게중심에서 측정된 최대 충격 가속도 (G)
        time_history (List[float]): 시뮬레이션 각 스텝의 타임스탬프 리스트
        z_hist (List[float]): 패키징 박스 중심점의 수직(Z축) 높이 변화 이력
        root_acc_history (List[float]): 시간별 패키징 박스의 절대 가속도 이력 (G-unit)
        pos_hist (List[np.ndarray]): 패키징 박스의 3차원 위치 좌표 [x, y, z] 이력
        cog_pos_hist (List[np.ndarray]): 전체 시스템의 통합 질량 중심(CoG) 이동 이력
        ground_impact_hist (List[float]): 지면과의 총 수직 충격력 합산 이력 (Newton)
        air_drag_hist (List[float]): 공기 저항력(Drag) 인가 이력
        air_squeeze_hist (List[float]): 충격 직전 발생하는 공기 스퀴즈 필름(Squeeze Film) 저항력 이력
    """
    config: Dict[str, Any]
    metrics: Dict[str, Any]
    max_g_force: float
    time_history: List[float]
    z_hist: List[float]
    root_acc_history: List[float]
    corner_acc_hist: List[Any]
    
    pos_hist: List[Any] = field(default_factory=list)
    vel_hist: List[Any] = field(default_factory=list)
    acc_hist: List[Any] = field(default_factory=list)
    
    cog_pos_hist: List[Any] = field(default_factory=list)
    cog_vel_hist: List[Any] = field(default_factory=list)
    cog_acc_hist: List[Any] = field(default_factory=list)
    
    corner_pos_hist: List[Any] = field(default_factory=list)
    corner_vel_hist: List[Any] = field(default_factory=list)
    
    ground_impact_hist: List[float] = field(default_factory=list)
    air_drag_hist: List[float] = field(default_factory=list)
    air_viscous_hist: List[float] = field(default_factory=list)
    air_squeeze_hist: List[float] = field(default_factory=list)
    
    def save(self, filepath: str):
        """시뮬레이션 결과(self)를 지정된 경로에 Pickle 형식으로 저장합니다."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
            
    @classmethod
    def load(cls, filepath: str):
        """저장된 바이너리 결과 파일을 읽어 DropSimResult 객체를 복원합니다."""
        with open(filepath, "rb") as f:
            return pickle.load(f)


# =====================================================================
# [1.5] GUI: Config Control & Editor (Tkinter Based)
# =====================================================================
class ConfigEditor(tk.Toplevel):
    """
    [WHTOOLS] 시뮬레이션 설정 및 제어를 위한 Tkinter GUI입니다.
    사용자는 이 UI를 통해 실시간으로 물리 파라미터를 수정하고 시제 거동을 제어할 수 있습니다.
    """
    def __init__(self, parent_sim):
        super().__init__()
        self.sim = parent_sim
        self.title("WHTOOLS Config Control UI")
        self.geometry("750x900")
        self.protocol("WM_DELETE_WINDOW", self.on_cancel)
        
        # [데이터 정의] 각 설정값에 대한 분류 및 설명 맵
        self.groups = {
            "Environment": ["drop_height", "drop_mode", "drop_direction", "env_gravity", "env_wind"],
            "Physics": ["sim_timestep", "sim_duration", "sim_integrator", "sim_iterations", "sol_recession"],
            "Structure": ["target_mass", "num_balancing_masses", "box_w", "box_h", "box_d", "mass_cushion", "mass_chassis"],
            "Cushion": ["enable_plasticity", "cush_yield_strain", "cush_yield_pressure", "cush_weld_solref_stiff", "cush_weld_solref_damp", "plasticity_ratio"]
        }
        
        self.desc_map = {
            "drop_height": "낙하 높이 (Drop height in meters, m)",
            "drop_mode": "매핑 모드 (LTL, PARCEL, or CUSTOM)",
            "drop_direction": "충격 방향 (Target impact orientation, e.g. Corner 2-3-5)",
            "env_gravity": "중력 가속도 지수 (Gravity acceleration index, m/s^2)",
            "env_wind": "풍속 벡터 (Wind speed vector [vx, vy, vz])",
            "target_mass": "밸런싱 목표 총 질량 (Target total mass, kg)",
            "num_balancing_masses": "보정 질량 개수 (1, 2, 4, 8)",
            "cush_weld_solref_stiff": "쿠션 골격 굽힘 강성 (Cushion skeletal bending stiffness, solref[0])",
            "cush_weld_solref_damp": "쿠션 감쇠비 (Cushion damping ratio, solref[1])",
            "cush_yield_strain": "소성 변형 항복 변형률 (Strain threshold, 0.0~1.0)",
            "cush_yield_pressure": "소성 변형 항복 압력 (Pressure threshold, Pa)",
            "plasticity_ratio": "변형 적용 비율 (Deformation application ratio, 0.0~1.0)",
            "enable_plasticity": "소성 변형 로직 활성화 여부 (Enable/Disable permanent deformation)",
            "sim_timestep": "시뮬레이션 타임스텝 (Integration timestep, s)",
            "sim_duration": "총 시뮬레이션 시간 (Total simulation time limit, s)",
            "sim_integrator": "수치 적분 방법 (MuJoCo integration: euler, implicit, etc.)",
            "sim_iterations": "솔버 최대 반복 횟수 (Max solver iterations per step)",
            "sol_recession": "솔버 접촉 거리 (Solver recession/contact distance)",
            "box_w": "패키징 박스 가로 (Box width, m)",
            "box_h": "패키징 박스 세로 (Box height, m)",
            "box_d": "패키징 박스 깊이 (Box depth, m)",
            "mass_cushion": "쿠션재 총 질량 (Total mass of cushioning material, kg)",
            "mass_chassis": "내부 새시 질량 (Mass of the internal chassis, kg)",
            "air_density": "공기 밀도 (Density of air, kg/m^3)",
            "air_cd_drag": "박스 항력 계수 (Drag coefficient, Cd)",
            "enable_air_squeeze": "에어 스퀴즈 저항 활성화 (Enable/Disable air squeeze film)",
            "air_coef_squeeze": "에어 스퀴즈 강도 배율 (Intensity multiplier for squeeze)",
            "box_div": "박스 격자 분할 [nx, ny, nz] (Box resolution)",
            "cush_div": "쿠션 격자 분할 [nx, ny, nz] (Cushion resolution)",
            "chassis_div": "새시 격자 분할 [nx, ny, nz] (Chassis resolution)",
            "sim_impratio": "접촉 솔버 임피던스 비율 (Impendance ratio for solver)",
            "cush_yield_stress": "쿠션 항복 응력 (Yield stress for cushion, Pa)",
            "include_cushion": "모델 내 쿠션 포함 여부 (Include cushion in model)",
            "include_paperbox": "외곽 박스 포함 여부 (Include outer packaging box)",
            "cush_use_weld": "쿠션 블록 간 웰드 구속 사용 (Use weld between cushion blocks)"
        }
        
        self.widget_map = {}
        self.build_ui()

    def build_ui(self):
        # 1. 상단 배너 이미지 (로컬 또는 Fallback 로딩)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        banner_path = os.path.join(script_dir, "ui_banner.png")
        if not os.path.exists(banner_path):
            banner_path = os.path.join(os.path.dirname(script_dir), "logo.png")

        if os.path.exists(banner_path):
            try:
                img = Image.open(banner_path)
                w, h = img.size
                target_w = 720
                target_h = int(h * (target_w / w))
                img = img.resize((target_w, target_h), Image.LANCZOS)
                self.banner_img = ImageTk.PhotoImage(img)
                lbl = tk.Label(self, image=self.banner_img, bg="#1a1a1a")
                lbl.pack(fill="x", padx=10, pady=5)
            except Exception as e:
                print(f"Banner loading failed: {e}")

        # 2. 제어판 (Simulation Controls)
        ctrl_frame = ttk.LabelFrame(self, text="Simulation Advanced Controls")
        ctrl_frame.pack(fill="x", padx=10, pady=5)
        
        # Row 1: Play/Stop & Reset & View Info & STATUS
        btn_box1 = ttk.Frame(ctrl_frame)
        btn_box1.pack(fill="x", pady=5)
        
        self.play_btn = ttk.Button(btn_box1, text="Play / Pause", command=self.on_toggle_play)
        self.play_btn.pack(side="left", padx=5)
        
        ttk.Button(btn_box1, text="Rewind (Reset)", command=self.on_reset).pack(side="left", padx=5)
        ttk.Button(btn_box1, text="Print View", command=self.on_print_view).pack(side="left", padx=5)
        
        # STATUS Label (Right side of buttons)
        self.status_var = tk.StringVar(value="Step: 0 | Time: 0.000s")
        self.status_lbl = ttk.Label(btn_box1, textvariable=self.status_var, font=("Consolas", 10, "bold"), foreground="#007acc")
        self.status_lbl.pack(side="left", padx=20)
        
        # Row 2: Step Size Slider
        btn_box2 = ttk.Frame(ctrl_frame)
        btn_box2.pack(fill="x", pady=5)
        
        ttk.Label(btn_box2, text="Step Size:").pack(side="left", padx=(5, 5))
        self.step_val_var = tk.IntVar(value=10)
        self.step_label = ttk.Label(btn_box2, text="10", width=4)
        self.step_label.pack(side="left")
        
        self.step_scale = ttk.Scale(btn_box2, from_=1, to=200, orient="horizontal", 
                                   variable=self.step_val_var, command=self.on_scale_change)
        self.step_scale.pack(side="left", fill="x", expand=True, padx=5)
        
        # Row 3: Step Navigation (Animated)
        btn_box3 = ttk.Frame(ctrl_frame)
        btn_box3.pack(fill="x", pady=5)
        
        ttk.Button(btn_box3, text="<< Step Backward (Anim)", command=self.on_step_back).pack(side="left", padx=5, expand=True, fill="x")
        ttk.Button(btn_box3, text="Step Forward (Anim) >>", command=self.on_step_forward).pack(side="left", padx=5, expand=True, fill="x")

        # 3. 탭 인터페이스 (Notebook)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=5)
        
        # [Tab 1] Settings
        self.settings_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_tab, text=" Configuration Settings ")
        
        self.canvas = tk.Canvas(self.settings_tab)
        self.v_scrollbar = ttk.Scrollbar(self.settings_tab, orient="vertical", command=self.canvas.yview)
        self.scroll_frame = ttk.Frame(self.canvas)
        self.scroll_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set)
        
        self.v_scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        # 그룹별 항목 생성
        all_keys = set(self.sim.config.keys())
        for group_name, keys in self.groups.items():
            g_frame = ttk.LabelFrame(self.scroll_frame, text=f" [{group_name}] ")
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

        # [Tab 2] Guide
        self.guide_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.guide_tab, text=" Help & Reference ")
        self._build_guide_tab()

    def _build_guide_tab(self):
        """도움말 및 기술 레퍼런스 가이드 탭을 구성합니다."""
        txt = tk.Text(self.guide_tab, wrap="word", font=("Consolas", 10), bg="#f8f9fa", padx=10, pady=10)
        txt.pack(fill="both", expand=True)
        
        guide_content = """[ WHTOOLS Technical Guide: MuJoCo Physics Parameters ]
=====================================================================

1. solref (Time Constant & Damping Ratio)
-----------------------------------------
MuJoCo의 'solref'는 접촉/구속의 강성과 감쇠를 정의하는 하방 파라미터입니다.
- solref[0] (Timeconst): 작을수록 딱딱해지며, 0.005 이하면 수치적 불안정이 
                         발생할 수 있습니다. (추천: 0.01~0.02)
- solref[1] (Dampratio): 1.0은 임계 감쇠(Critical damping)를 의미합니다.

2. solimp (Impedance Envelope)
------------------------------
- solimp[0, 1] (dmin, dmax): 접촉 시 솔버가 작용하는 영역을 %로 정의합니다.
- solimp[2] (width): 임피던스 전이가 발생하는 거리입니다.

[ 추천 파라미터 조합 (Material Recommendations) ]
-------------------------------------------------
| 재질 (Material) | solref (TC, DR)  | solimp (dmin, dmax, width) |
| :-------------- | :--------------- | :------------------------- |
| 쿠션 (Soft)     | [0.020, 1.0]     | [0.900, 0.950, 0.001]      |
| 테이프 (Hard)   | [0.005, 1.0]     | [0.950, 0.990, 0.001]      |
| 박스 / 새시      | [0.010, 1.0]     | [0.900, 0.950, 0.001]      |
| 디폴트 지면     | [0.020, 1.0]     | [0.900, 0.950, 0.001]      |

* 주의: solref[0] 값이 너무 작으면(예: 0.001) 충격 시 스프링 힘이 폭발하여 
  'NaN' 오류가 발생하며 시뮬레이션이 튕길 수 있습니다.
=====================================================================
"""
        txt.insert("1.0", guide_content)
        txt.config(state="disabled")

    def update_status(self, step, t):
        """현재 시뮬레이션 진행 상태를 UI 라벨에 업데이트합니다."""
        self.status_var.set(f"Step: {step:5d} | Time: {t:6.3f}s")

        # 4. 하단 액션 버튼
        action_frame = ttk.Frame(self)
        action_frame.pack(fill="x", pady=10)
        
        ttk.Button(action_frame, text="Apply Changes", command=self.on_apply).pack(side="right", padx=10)
        ttk.Button(action_frame, text="Close UI", command=self.on_cancel).pack(side="right", padx=5)

    def add_config_row(self, parent, key, val):
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=5, pady=2)
        
        lbl = ttk.Label(row, text=f"{key:30}", width=30, font=("Consolas", 10))
        lbl.pack(side="left")
        
        if isinstance(val, bool):
            # Boolean: Combobox 사용
            var = tk.StringVar(value=str(val))
            ent = ttk.Combobox(row, textvariable=var, values=["True", "False"], width=13, state="readonly")
            ent.pack(side="left", padx=5)
        elif isinstance(val, (list, tuple)):
            # List/Tuple: Entry + Eval hint
            var = tk.StringVar(value=str(val))
            ent = ttk.Entry(row, textvariable=var, width=25)
            ent.pack(side="left", padx=5)
            ttk.Label(row, text="[List]", foreground="blue", font=("Consolas", 8)).pack(side="left")
        else:
            # 기타 (float, int, str): Normal Entry
            var = tk.StringVar(value=str(val))
            ent = ttk.Entry(row, textvariable=var, width=15)
            ent.pack(side="left", padx=5)
        
        desc = self.desc_map.get(key, "-")
        ttk.Label(row, text=f"({desc})", foreground="gray", font=("NanumGothic", 8)).pack(side="left", padx=5)
        
        self.widget_map[key] = (var, type(val))

    def on_apply(self):
        """변경된 설정을 검증하고 Config에 반영합니다."""
        errors = []
        new_values = {}

        for key, (var, original_type) in self.widget_map.items():
            raw_val = var.get().strip()
            try:
                if original_type == bool:
                    new_val = raw_val == "True"
                elif original_type in (list, tuple):
                    # 리스트의 경우 eval을 통해 안전하게 변환 시도
                    new_val = eval(raw_val)
                    if not isinstance(new_val, (list, tuple)):
                        raise ValueError("Not a valid list/tuple format")
                elif original_type == int:
                    new_val = int(raw_val)
                elif original_type == float:
                    new_val = float(raw_val)
                else:
                    new_val = raw_val
                
                new_values[key] = new_val
            except Exception as e:
                errors.append(f"[{key}] : {raw_val} (Error: {str(e)})")

        if errors:
            err_msg = "다음 항목들의 입력 형식이 올바르지 않습니다:\n\n" + "\n".join(errors)
            messagebox.showerror("Validation Error", err_msg)
            return

        # 에러가 없는 경우에만 일괄 반영
        self.sim.config.update(new_values)
        self.sim.ctrl_reload_request = True
        self.destroy()

    def on_reset(self):
        """시뮬레이션 초기화 요청"""
        self.sim.ctrl_reset_request = True
        
    def on_step(self):
        """단일 스텝 진행 요청"""
        self.sim.ctrl_step_request = True
        
    def on_jump(self):
        """지정된 스텝만큼 고속 진행 요청"""
        steps = int(self.step_val_var.get())
        self.sim.ctrl_jump_steps = steps

    def on_toggle_play(self):
        """시뮬레이션 재생/정지 전환"""
        self.sim.ctrl_paused = not self.sim.ctrl_paused
        
    def on_step_forward(self):
        """N-Step 앞으로 감기"""
        steps = int(self.step_val_var.get())
        self.sim.ctrl_step_forward_count = steps
        
    def on_step_back(self):
        """N-Step 뒤로 감기"""
        steps = int(self.step_val_var.get())
        self.sim.ctrl_step_backward_count = steps

    def on_print_view(self):
        """현재 뷰어 카메라 정보 출력 요청"""
        self.sim.ctrl_print_view = True

    def on_scale_change(self, val):
        """슬라이더 값 변경 시 라벨 업데이트"""
        self.step_label.config(text=str(int(float(val))))

    def _on_mousewheel(self, event):
        """마우스 휠을 통한 캔버스 스크롤 지원"""
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def on_cancel(self):
        """변경 없이 닫기"""
        self.destroy()


class PostProcessingUI(tk.Toplevel):
    """
    [WHTOOLS] 시뮬레이션 완료 후 결과 분석을 위한 전용 포스트 프로세싱 UI입니다.
    """
    def __init__(self, parent_sim):
        super().__init__()
        self.sim = parent_sim
        self.title("WHTOOLS Post-Processing Explorer")
        self.geometry("600x450")
        self.attributes("-topmost", True)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self.build_ui()

    def build_ui(self):
        # 1. 상단 배너 (ConfigEditor와 동일 스타일)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        banner_path = os.path.join(script_dir, "ui_banner.png")
        if os.path.exists(banner_path):
            try:
                img = Image.open(banner_path)
                w, h = img.size
                target_w = 580
                target_h = int(h * (target_w / w))
                img = img.resize((target_w, target_h), Image.LANCZOS)
                self.banner_img = ImageTk.PhotoImage(img)
                lbl = tk.Label(self, image=self.banner_img, bg="#1a1a1a")
                lbl.pack(fill="x", padx=10, pady=5)
            except: pass

        # 2. 메인 안내문 (Grid-based layout for better spacing)
        main_frame = ttk.Frame(self)
        main_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        ttk.Label(main_frame, text="Simulation Analysis Complete", font=("Consolas", 14, "bold"), foreground="#d9534f").pack()
        ttk.Label(main_frame, text="Select analysis tools to visualize structural integrity.", font=("NanumGothic", 10)).pack(pady=5)

        # (NEW) 2.5 Body Selection Area
        select_frame = ttk.LabelFrame(main_frame, text=" 1. Target Body Selection ")
        select_frame.pack(fill="x", pady=10)
        
        ttk.Label(select_frame, text="Select Component:").pack(side="left", padx=10, pady=5)
        
        # 필드가 존재하는 바디 목록 추출
        comp_list = sorted(list(self.sim.metrics.keys()))
        self.comp_var = tk.StringVar(value=comp_list[0] if comp_list else "")
        self.comp_combo = ttk.Combobox(select_frame, textvariable=self.comp_var, values=comp_list, state="readonly", width=25)
        self.comp_combo.pack(side="left", padx=10, pady=5)

        # 3. 분석 도구 버튼 영역
        tools_frame = ttk.LabelFrame(main_frame, text=" 2. Analysis & Visualization ")
        tools_frame.pack(fill="both", expand=True, pady=10)

        # (1) MuJoCo Heatmap (Rank-based + RdYlBu_r)
        btn1 = ttk.Button(tools_frame, text="Apply Distortion Heatmap (MuJoCo)", 
                          command=self.on_apply_heatmap)
        btn1.pack(fill="x", padx=10, pady=5)
        ttk.Label(tools_frame, text=">> Rank-based gradient (RdYlBu_r) for maximum contrast.", 
                  foreground="gray", font=("NanumGothic", 8)).pack()

        # (2) Matplotlib 2D Map (Selected Body)
        btn2 = ttk.Button(tools_frame, text="Show 2D Distortion Map (Matplotlib)", 
                          command=self.on_show_plot)
        btn2.pack(fill="x", padx=10, pady=5)
        ttk.Label(tools_frame, text=">> 2D interpolated map for selected body (Equal Aspect).", 
                  foreground="gray", font=("NanumGothic", 8)).pack()

        # (3) [NEW] Show Impact Analysis
        btn3 = ttk.Button(tools_frame, text="Show Impact Analysis (G-Force/Motion)", 
                          command=self.on_show_impact)
        btn3.pack(fill="x", padx=10, pady=5)
        ttk.Label(tools_frame, text=">> Displays simulation trace plots (G-Force, Kinematics).", 
                  foreground="gray", font=("NanumGothic", 8)).pack()

        # 4. 하단 닫기
        footer = ttk.Frame(self)
        footer.pack(fill="x", pady=10)
        ttk.Button(footer, text="Exit Analysis", command=self.on_close).pack(side="right", padx=20)

    def on_apply_heatmap(self):
        self.sim.apply_rank_distortion_heatmap()
        messagebox.showinfo("Success", "Distortion Heatmap (RdYlBu_r Rank-based) applied.")

    def on_show_plot(self):
        # 선택된 바디 전달
        target_comp = self.comp_var.get()
        if target_comp:
            self.sim.plot_2d_distortion_map(target_comp)
        else:
            messagebox.showwarning("Warning", "Please select a component first.")

    def on_show_impact(self):
        # 기존 plot_results의 팝업 버전 실행
        self.sim.show_impact_plots()

    def on_close(self):
        self.destroy()


# =====================================================================
# [2] 핵심 엔진 클래스: DropSimulator
# =====================================================================
class DropSimulator:
    """
    MuJoCo 물리 엔진 기반의 낙하 시뮬레이션을 제어하는 통합 고도화 클래스입니다.
    
    기계공학적 관점에서의 해석 정밀도를 유지하기 위해 '이산화 모델링', 
    '비선형 소성 변형(Strain-based Plasticity)', '공기 유체 저항(Squeeze Film)' 등의 
    물리 로직이 객체 지향적으로 구현되어 있습니다.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        DropSimulator를 초기화하고 초기 상태 변수를 설정합니다.

        Args:
            config (Dict, optional): 사용자 정의 시뮬레이션 파라미터. 
                전달되지 않으면 'run_discrete_builder.py'의 기본값을 사용합니다.
        """
        # [0] Configuration 관리 (기본값 설정 및 사용자 정의 병합)
        self.config = get_default_config(config) if config else get_default_config()
        
        # [1] MuJoCo 핵심 객체 (setup 시 메모리에 로드)
        self.model: Optional[mujoco.MjModel] = None
        self.data: Optional[mujoco.MjData] = None
        self.viewer = None
        
        # [2] 작업 이력 관리 (파일명 자동 생성을 위한 타임스탬프)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"rds-{self.timestamp}"
        self.report_file_path = None
        
        # [3] 물리 상태 트래킹을 위한 변동 데이터 구조
        self.geom_state_tracker = {}     # 각 블록(Geom)의 최대 압축 및 소성 변형 레벨 추적
        self.corner_neighbor_pairs = []  # Neighbor pairs for strain calc
        self.corner_body_ids = []        # Root box corners
        self.state_buffer = []           # Step backward buffer
        self.max_buffer_size = 500
        self._step_corner_values = {}
        self._analysis_done = False      # 리포트 중복 생성 방지용 플래그
        self.components = {}             # 컴포넌트 이름 -> {(i,j,k): body_id} 구성 트리
        self.comp_max_idxs = {}          # 컴포넌트별 분할 격자수 (nx, ny, nz)
        self.nominal_local_pos = {}      # 모델 생성 시점의 초기 로컬 좌표 (변위 계산용)
        self.corner_neighbor_pairs = []  # 코너 블록 사이의 Strain 연산을 위한 거리 쌍 데이터
        
        # [4] 시뮬레이션 타임 스텝 데이터 히스토리 저장소
        self._init_histories()
        
        # [5] 뷰어 및 사용자 인터랙션 제어 플래그
        self.ctrl_paused = False
        self.ctrl_step_request = False
        self.ctrl_reset_request = False
        self.ctrl_quit_request = False
        self.ctrl_reload_request = False
        self.ctrl_jump_steps = 0
        self.ctrl_open_ui = False
        self.config_editor = None
        
        # 신규 제어 플래그
        self.ctrl_step_forward_count = 0
        self.ctrl_step_backward_count = 0
        self.ctrl_print_view = False
        self.state_buffer = []  # Step Backward를 위한 상태 저장소 (최대 500개)
        self.max_buffer_size = 500
        
        self.corner_body_ids = []
        
        # [6] 물리 연산 중간값 (에어로다이나믹 로깅용 출력 변수)
        self.result: Optional[DropSimResult] = None
        self._last_f_drag = 0.0
        self._last_f_sq = f_sq_accumulator = 0.0
        
        # [8] 상세 로깅 및 임시 상태 저장소
        self.plasticity_log_buffer = [] # 상세 소성 변형 메시지 저장소
        self._step_max_plasticity = {'press': 0.0, 'strain': 0.0, 'disp': 0.0} # 현 스텝 내 최대 지표
        self._step_corner_values = {}   # 매 스텝 각 Geom의 실시간 물리량 캐시

        # [8] 자동 질량 보정 (Auto Inertia Balancer)
        if self.config.get("enable_target_balancing", False):
            self.apply_balancing(
                target_mass = self.config.get("target_mass"),
                target_cog  = self.config.get("target_cog"),
                target_moi  = self.config.get("target_moi"),
                num_masses  = self.config.get("num_balancing_masses", 8)
            )

    def _init_histories(self):
        """시뮬레이션 진행 중 수집되는 모든 시계열 데이터 저장소를 초기화합니다."""
        self.time_history = []
        self.z_hist = []
        self.pos_hist = []
        self.vel_hist = []
        self.acc_hist = []
        self.cog_pos_hist = []
        self.cog_vel_hist = []
        self.cog_acc_hist = []
        self.corner_pos_hist = []
        self.corner_vel_hist = []
        self.corner_acc_hist = []
        self.ground_impact_hist = []
        self.air_drag_hist = []
        self.air_viscous_hist = []
        self.air_squeeze_hist = []
        self.metrics = {}  # 분석 메트릭스 (Bending/Twist 등)

    # -----------------------------------------------------------------
    # (A) 로깅 및 설정 출력 유틸리티
    # -----------------------------------------------------------------
    def log(self, msg: Any):
        """
        메시지를 터미널에 출력하고 동시에 요약 보고서 파일(summary_report.txt)에 기록합니다.

        Args:
            msg (Any): 기록할 메시지 내용
        """
        content = str(msg)
        print(content)
        
        if self.report_file_path:
            try:
                # 인코딩 안정성을 위해 utf-8 사용
                with open(self.report_file_path, "a", encoding="utf-8") as rf:
                    rf.write(content + "\n")
            except Exception as e:
                print(f"!!! [WHT_LOG_ERROR] 파일 로깅 중 오류 발생: {e}")

    def format_config_report(self) -> str:
        """
        현재 시뮬레이션에 적용된 물리/설계 파라미터를 체계적으로 정리하여 시각화된 문자열로 반환합니다.
        가시성을 위해 카테고리별로 공정하게 분류하여 출력합니다.
        """
        cfg = self.config
        lines = [
            "\n" + "=" * 95,
            f"  [ WHTOOLS MuJoCo DROP SIMULATION - CONFIGURATION REPORT ]  -  {self.timestamp}",
            "=" * 95
        ]
        
        # 출력 대상 카테고리 정의
        categories = {
            "1. 시뮬레이션 환경 (Run Env)": [
                "drop_mode", "drop_direction", "drop_height", "sim_duration", "sim_timestep", "use_viewer"
            ],
            "2. 모델 아키텍처 (Structure)": [
                "include_paperbox", "include_cushion", "box_use_weld", "cush_use_weld", "chassis_use_weld"
            ],
            "3. 격자 분할 해상도 (Resolution)": [
                "box_div", "cush_div", "chassis_div", "assy_div"
            ],
            "4. 소성 변형 파라미터 (Plasticity)": [
                "enable_plasticity", "plasticity_ratio", "cush_yield_stress", "cush_yield_strain", "cush_yield_pressure"
            ],
            "5. 솔버 및 수치 해석성 (Solver)": [
                "sim_integrator", "sim_iterations", "sim_impratio", "sim_tolerance", "sim_noslip_iterations"
            ]
        }
        
        for category_name, keys in categories.items():
            lines.append(f"\n[{category_name}]")
            for key in keys:
                if key in cfg:
                    value = str(cfg[key])
                    lines.append(f"  - {key:<25}: {value:<20}")
        
        lines.append("\n" + "=" * 95 + "\n")
        return "\n".join(lines)

    # -----------------------------------------------------------------
    # (B) 기구학적 분석 및 질량 평형 (Inertia Balancer)
    # -----------------------------------------------------------------
    def print_inertia_report(self, title: str = "[Inertia Report]"):
        """
        현재 모델 구성의 질량, CoG, MoI 수치를 측정하여 리포팅합니다.
        실제 제품 제원과의 오차를 확인하는 용도로 사용됩니다.

        Args:
            title (str): 리포트의 헤더 제목
        Returns:
            total_mass, cog, moi: 측정된 물리량 (MuJoCo 연산 결과)
        """
        # 측정용 임시 모델 생성
        temp_xml = "temp_inertia_check.xml"
        _, total_mass, cog, moi, details = create_model(temp_xml, config=self.config, logger=lambda x: None)
        
        self.log(f"\n{title}")
        self.log("-" * 65)
        self.log(f"  - 총 질량 (Total Mass)        : {total_mass:12.4f} kg")
        self.log(f"  - 질량 중심 (Center of Gravity) : ({cog[0]:.4f}, {cog[1]:.4f}, {cog[2]:.4f}) [m]")
        self.log(f"  - 관성 모멘트 (Moment of Inertia): ({moi[0]:.6f}, {moi[1]:.6f}, {moi[2]:.6f}) [kg*m^2]")
        self.log("-" * 65)
        
        return total_mass, cog, moi

    def calculate_required_aux_masses(self, target_mass: float, target_cog: List[float] = None, target_moi: List[float] = None, num_masses: int = 8):
        """
        설계 목표치(Target)를 달성하기 위해 필요한 추가 보정 질량들의 위치와 크기를 역산합니다.
        지원 개수: 1, 2, 3, 4, 8개 (그 외 숫자는 8개로 폴백)
        모든 질량체는 박스의 바운딩 박스 내부에 위치하도록 Clipping 됩니다.
        """
        # [0] Helper: 평행축 정리 기반의 관성 텐서 이동 (I = I_cg + m * d^2)
        def shift_moi_func(m, i_cg, cg, target_p):
            d = target_p - cg
            # I = I_cg + m * (dy^2 + dz^2) 형태
            return i_cg + m * np.array([d[1]**2 + d[2]**2, d[0]**2 + d[2]**2, d[0]**2 + d[1]**2])

        # [1] 기저 모델(보정 전)의 관성 데이터 확보
        temp_cfg = self.config.copy()
        temp_cfg["chassis_aux_masses"] = []
        _, m_base, c_base, i_base, _ = create_model("temp_inertia_balancer.xml", config=temp_cfg, logger=lambda x: None)
        
        m_base = float(m_base)
        c_base = np.array(c_base)
        i_base = np.array(i_base)
        
        # [2] 목표 파라미터 유지 (None일 경우 현재 상태 유지)
        t_mass = target_mass if target_mass is not None else m_base
        t_cog  = np.array(target_cog) if (target_cog is not None and len(target_cog) == 3) else c_base
        t_moi  = np.array(target_moi) if (target_moi is not None and len(target_moi) == 3) else None
        
        # [3] 추가 필요 질량 계산
        m_aux = t_mass - m_base
        
        if m_aux < 1e-6:
            if np.allclose(t_cog, c_base, atol=1e-4): return []
            self.log(f"   >> [Warning] 목표 질량이 현재보다 작습니다. CoG 보정이 완벽하지 않을 수 있습니다.")
            m_aux = 1e-4 # 최소값 방어
            
        # [4] 보정 질량 군집의 CoG 역산
        # M_total * CoG_total = M_base * CoG_base + M_aux * CoG_aux
        pos_aux = (t_cog * t_mass - m_base * c_base) / (m_aux if m_aux > 0 else 1)
        
        # [5] 바운딩 박스 가이드 (Clipping용)
        # 박스 외곽 치수 기반 (여유율 90% 적용하여 쿠션/박스 내부 안착 유도)
        bw, bh, bd = self.config.get('box_w', 2.0), self.config.get('box_h', 1.4), self.config.get('box_d', 0.25)
        limit_x, limit_y, limit_z = bw/2.0 * 0.9, bh/2.0 * 0.9, bd/2.0 * 0.9
        
        def clip_pos(p):
            return [
                np.clip(p[0], -limit_x, limit_x),
                np.clip(p[1], -limit_y, limit_y),
                np.clip(p[2], -limit_z, limit_z)
            ]

        aux_masses = []
        
        # [6] 분담 개수에 따른 배치 로직
        if num_masses <= 1 or t_moi is None:
            # 단일 질량 혹은 MoI 미지정 시: CoG 지점에 집중 배치
            aux_masses.append({
                "name" : "InertiaAux_Single",
                "pos"  : clip_pos(pos_aux),
                "mass" : float(m_aux),
                "size" : [0.01, 0.01, 0.01]
            })
        elif num_masses == 2:
            # 2개 질량: X축 방향으로 대칭 배치하여 Iyy, Izz 보충 시도
            m_each = m_aux / 2.0
            i_needed = t_moi[1] - i_base[1] if t_moi is not None else 0
            dx = math.sqrt(max(0.005, i_needed / (2.0 * m_each)))
            
            for sx in [-1, 1]:
                p = [pos_aux[0] + sx * dx, pos_aux[1], pos_aux[2]]
                aux_masses.append({"name": f"InertiaAux_{len(aux_masses)+1}", "pos": clip_pos(p), "mass": m_each, "size": [0.01]*3})
        
        elif num_masses == 3:
            # 3개 질량: 정삼각형 혹은 L자 배치
            m_each = m_aux / 3.0
            offsets = [[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]]
            for off in offsets:
                p = [pos_aux[0] + off[0], pos_aux[1] + off[1], pos_aux[2] + off[2]]
                aux_masses.append({"name": f"InertiaAux_{len(aux_masses)+1}", "pos": clip_pos(p), "mass": m_each, "size": [0.01]*3})

        elif num_masses == 4:
            # 4개 질량: XY 평면 상에 사각형 배치
            m_each = m_aux / 4.0
            dx = math.sqrt(max(0.005, (t_moi[1] - i_base[1]) / (4.0 * m_each))) if t_moi is not None else 0.05
            dy = math.sqrt(max(0.005, (t_moi[0] - i_base[0]) / (4.0 * m_each))) if t_moi is not None else 0.05
            
            for sx in [-1, 1]:
                for sy in [-1, 1]:
                    p = [pos_aux[0] + sx * dx, pos_aux[1] + sy * dy, pos_aux[2]]
                    aux_masses.append({"name": f"InertiaAux_{len(aux_masses)+1}", "pos": clip_pos(p), "mass": m_each, "size": [0.01]*3})

        else:
            # 8개 질량 (Symmetric 8-Point Mass)
            m_each = m_aux / 8.0
            i_base_at_target = shift_moi_func(m_base, i_base, c_base, t_cog)
            i_aux_total_at_target = (t_moi if t_moi is not None else i_base) - i_base_at_target
            i_aux_at_own_cog = i_aux_total_at_target - shift_moi_func(m_aux, np.zeros(3), pos_aux, t_cog)
            i_res = np.maximum(i_aux_at_own_cog, 1e-9)
            
            dx = math.sqrt(max(0, (i_res[1] + i_res[2] - i_res[0]) / (2.0 * m_aux)))
            dy = math.sqrt(max(0, (i_res[0] + i_res[2] - i_res[1]) / (2.0 * m_aux)))
            dz = math.sqrt(max(0, (i_res[0] + i_res[1] - i_res[2]) / (2.0 * m_aux)))

            for sx in [-1, 1]:
                for sy in [-1, 1]:
                    for sz in [-1, 1]:
                        p = [pos_aux[0] + sx * dx, pos_aux[1] + sy * dy, pos_aux[2] + sz * dz]
                        aux_masses.append({"name": f"InertiaAux_{len(aux_masses)+1}", "pos": clip_pos(p), "mass": m_each, "size": [0.01]*3})
                        
        return aux_masses

    def apply_balancing(self, target_mass: float = None, target_cog: List[float] = None, target_moi: List[float] = None, num_masses: int = 8):
        """
        정의된 타겟 데이터에 맞춰 모델의 질량 평형(Balancing)을 자동 수행하고 전/후 비교 리포트를 생성합니다.
        """
        self.log("\n[Inertia Balancer] 입력 보정 데이터에 따른 최적화 계산을 수행합니다...")
        
        # [1] Baseline 측정
        temp_cfg = self.config.copy()
        temp_cfg["chassis_aux_masses"] = []
        _, m_base, c_base, i_base, _ = create_model("temp_ib_base.xml", config=temp_cfg, logger=lambda x: None)
        
        # [2] 최적 보정 질량체 계산 및 Config 업데이트
        aux_items = self.calculate_required_aux_masses(target_mass, target_cog, target_moi, num_masses)
        self.config["chassis_aux_masses"] = aux_items
        
        # [3] 보정 후(Final) 관성 데이터 측정
        _, m_final, c_final, i_final, _ = create_model("temp_ib_final.xml", config=self.config, logger=lambda x: None)
        
        # [4] 요약 테이블 출력
        header = "-" * 95
        self.log("\n" + header)
        self.log(f"| {'Inertia Component':<25} | {'Baseline':^18} | {'Target':^18} | {'Final (Matched)':^18} |")
        self.log(header)
        
        # Mass
        self.log(f"| {'Total Mass (kg)':<25} | {m_base:18.4f} | {target_mass if target_mass else m_base:18.4f} | {m_final:18.4f} |")
        
        # CoG (X, Y, Z)
        for i, char in enumerate(['X', 'Y', 'Z']):
            t_val = target_cog[i] if (target_cog and len(target_cog)>i) else c_base[i]
            label = f"CoG {char} (m)"
            self.log(f"| {label:<25} | {c_base[i]:18.4f} | {t_val:18.4f} | {c_final[i]:18.4f} |")
            
        # MoI (Ixx, Iyy, Izz)
        for i, char in enumerate(['XX', 'YY', 'ZZ']):
            t_val = target_moi[i] if (target_moi and len(target_moi)>i) else i_base[i]
            label = f"MoI I{char} (kg*m^2)"
            self.log(f"| {label:<25} | {i_base[i]:18.6f} | {t_val:18.6f} | {i_final[i]:18.6f} |")
            
        self.log(header)
        self.log(f" >> 보정 질량체 생성 결과: {len(aux_items)}개 점질량 배치 완료 (Config 자동 업데이트)")
        self.log(header + "\n")

    # -----------------------------------------------------------------
    # (C) 시뮬레이션 환경 구축 및 모델 로드
    # -----------------------------------------------------------------
    def setup(self):
        """
        시뮬레이션 구동에 필요한 파일 시스템 환경을 준비하고, MuJoCo XML 모델을 생성/메모리에 로드합니다.
        이 메서드 호출 이후 'simulate()'를 호출할 수 있습니다.
        """
        # [1] 디렉토리 및 로깅 파일 초기화
        os.makedirs(self.output_dir, exist_ok=True)
        self.report_file_path = os.path.abspath(os.path.join(self.output_dir, "summary_report.txt"))
        
        with open(self.report_file_path, "w", encoding="utf-8") as f:
            f.write(f"WHTOOLS: Simulation Summary - {datetime.now()}\n")
            
        self.log(self.format_config_report())
        
        # [2] XML 시뮬레이션 모델 생성
        xml_file_path = "temp_drop_sim_v3.xml"
        self.log("\n[Step 1] Discrete XML Model 생성 및 컴파일 중...")
        
        # create_model을 통해 XML 문자열 획득
        xml_content, *_ = create_model(xml_file_path, config=self.config, logger=self.log)
        
        # MuJoCo 모델 객체화
        self.model = mujoco.MjModel.from_xml_string(xml_content)
        self.data = mujoco.MjData(self.model)
        
        # [NEW] 분석용 핵심 바디 ID 캐싱 (BPackagingBox)
        self.root_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "BPackagingBox")
        
        # [3] 초기 기하 형상 상태 백업 (영구 변형 계산을 위한 Reference)
        self.original_geom_pos = self.model.geom_pos.copy()
        self.original_geom_size = self.model.geom_size.copy()
        self.original_geom_rgba = self.model.geom_rgba.copy()
        
        # [4] 부품 분석 및 데이터 구조화
        self._discover_components()    # 블록 계층 구조 분석
        self._discover_neighbors()     # Strain 측정용 인접 쌍 탐색
        self._initialize_tracking_containers() # 분석 지표 저장소 준비
        
        # [5] 물리 거동 제어 및 콜백 설정
        # 에어로다이나믹 로직 등록 (mjcb_control)
        # Note: GC 방지를 위해 인스턴스 변수에 저장합니다.
        self._mjcb_control = lambda m, d: self._apply_aerodynamics(m, d)
        mujoco.set_mjcb_control(self._mjcb_control)
        
        # [6] 솔버 물리 특성 정적 분석 리포팅
        self._print_solver_analysis()

    def _discover_components(self):
        """
        이산화된 블록들의 이름을 해석하여 어떤 부품(Cushion, OC 등)에 속하는지, 
        그리고 격자 상의 위치(i,j,k)가 어디인지 분류하여 인덱싱합니다.
        """
        self.components = {}
        self.nominal_local_pos = {}
        self.comp_max_idxs = {}
        
        # MuJoCo 시스템 내 모든 바디를 순회
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and name.startswith("b_"):
                parts = name.split("_")
                # 형식: b_[Component]_[i]_[j]_[k]
                if len(parts) >= 5:
                    try:
                        i_idx = int(parts[-3])
                        j_idx = int(parts[-2])
                        k_idx = int(parts[-1])
                        grid_key = (i_idx, j_idx, k_idx)
                        
                        comp_name = "_".join(parts[1:-3]).lower()
                        
                        # 컴포넌트 트리 구축
                        if comp_name not in self.components:
                            self.components[comp_name] = {}
                        
                        self.components[comp_name][grid_key] = i
                        self.nominal_local_pos[i] = self.model.body_pos[i].copy()
                    except (ValueError, IndexError):
                        continue
        
        # 각 컴포넌트의 최대 분할 지수(nx, ny, nz) 조사
        for i in range(self.model.ngeom):
            g_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if g_name and g_name.startswith("g_"):
                parts = g_name.split("_")
                if len(parts) >= 5:
                    try:
                        c_name = "_".join(parts[1:-3]).lower()
                        idx_val = [int(parts[-3]), int(parts[-2]), int(parts[-1])]
                        
                        if c_name not in self.comp_max_idxs:
                            self.comp_max_idxs[c_name] = [0, 0, 0]
                        
                        for axis_dim in range(3):
                            if idx_val[axis_dim] > self.comp_max_idxs[c_name][axis_dim]:
                                self.comp_max_idxs[c_name][axis_dim] = idx_val[axis_dim]
                    except:
                        pass
        
        # 불변 데이터로 튜플화
        for key in self.comp_max_idxs:
            self.comp_max_idxs[key] = tuple(self.comp_max_idxs[key])

    def _discover_neighbors(self):
        """
        쿠션 재질의 코너 블록들 사이의 'Strain-based Plasticity'를 연산하기 위해 
        가까운 이웃 블록들과의 초기 거리를 측정하고 쌍(Pair)으로 저장합니다.
        """
        self.corner_neighbor_pairs = []
        
        for name, (nx, ny, nz) in self.comp_max_idxs.items():
            if "cushion" not in name:
                continue
            
            # 해당 쿠션을 구성하는 Geom ID들을 사전 매핑
            geoms = {}
            for gid in range(self.model.ngeom):
                g_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid)
                if g_name and g_name.lower().startswith("g_") and name in g_name.lower():
                    pts = g_name.split('_')
                    if len(pts) >= 5:
                        try:
                            geoms[(int(pts[-3]), int(pts[-2]), int(pts[-1]))] = gid
                        except:
                            pass
            
            # 수직 전면의 코너 블록에 대해 내부 방향 블록과의 쌍 구축
            for idx, gid in geoms.items():
                ci, cj, ck = idx
                # 바닥/천장 혹은 전후좌우 끝단 모서리 필터링
                if (ci == 0 or ci == nx) and (cj == 0 or cj == ny):
                    # 주요 팽창/수축 체크를 위해 인접 6방향 탐색
                    directions = [
                        (1, 0, 0, 0), (-1, 0, 0, 0),  # X-축 방향 인접
                        (0, 1, 0, 1), (0, -1, 0, 1),  # Y-축 방향 인접
                        (0, 0, 1, 2), (0, 0, -1, 2)   # Z-축 방향 인접
                    ]
                    
                    for di, dj, dk, axis in directions:
                        neighbor_idx = (ci + di, cj + dj, ck + dk)
                        if neighbor_idx in geoms:
                            nid = geoms[neighbor_idx]
                            b1_id = self.model.geom_bodyid[gid]
                            b2_id = self.model.geom_bodyid[nid]
                            # 바디 간의 설계 초기 거리 측정
                            dist0 = np.linalg.norm(self.model.body_pos[b1_id] - self.model.body_pos[b2_id])
                            
                            if dist0 > 0.001:
                                self.corner_neighbor_pairs.append((gid, nid, axis, dist0))

    def _initialize_tracking_containers(self):
        """
        부품별 구조 지표(Bending, Twist, Energy 등)와 고위험 블록의 상태를 기록할 분석 데이터 구조를 초기화합니다.
        """
        self.metrics = {}
        
        for comp in self.components:
            self.metrics[comp] = {
                'all_blocks_angle'   : {idx: [] for idx in self.components[comp]},
                'all_blocks_bend'    : {idx: [] for idx in self.components[comp]},
                'all_blocks_twist'   : {idx: [] for idx in self.components[comp]},
                'block_nominal_mats' : {idx: None for idx in self.components[comp]},
                'total_distortion'   : [],
                'corner_hists'       : {}
            }
            
            # 시뮬레이션 중 row(j)별 지표 계산을 위한 키 생성
            j_indices = sorted(list(set([k[1] for k in self.components[comp].keys()])))
            for j_val in j_indices:
                self.metrics[comp][j_val] = {
                    'bending': [], 'twist': [], 'energy': [], 
                    'loc_b': [], 'loc_t': [], 'loc_e': []
                }
            
            # [BCUSHION 전용] 코너 블록 시각적 강조 및 고정밀 트래킹 활성화
            if comp == 'bcushion':
                nx_max, ny_max, _ = self.comp_max_idxs.get(comp, (0, 0, 0))
                for coord in self.components[comp]:
                    if (coord[0] == 0 or coord[0] == nx_max) and (coord[1] == 0 or coord[1] == ny_max):
                        # 실제 Geom ID 탐색
                        gid = self._find_geom_by_index(comp, coord)
                        if gid != -1:
                            self.metrics[comp]['corner_hists'][gid] = {
                                'strain'   : [], 'press' : [], 
                                'disp'     : [], 'plastic'  : [], 
                                'name'     : f"CornerG_{comp}_{coord}"
                            }
                            # 시각적 식별을 위해 뷰어에서 노란색(Yellow)으로 표시
                            self.model.geom_rgba[gid] = [1.0, 1.0, 0.0, 1.0]

        # [Packaging Box] 8개 모서리(Corner) 트래킹 대상 바디 식별
        # BPackagingBox 내부의 b_Box_..._..._... 블록들 중 외곽 모서리 블록 탐색
        if 'bbox' in self.components:
            nx_max, ny_max, nz_max = self.comp_max_idxs.get('bbox', (0, 0, 0))
            for coord, body_id in self.components['bbox'].items():
                if (coord[0] in (0, nx_max)) and (coord[1] in (0, ny_max)) and (coord[2] in (0, nz_max)):
                    self.corner_body_ids.append(body_id)

    def _find_geom_by_index(self, component_name, grid_idx):
        """컴포넌트 이름과 격자 인덱스를 사용하여 MuJoCo Geom ID를 역추적합니다."""
        prefix = f"g_{component_name}_"
        suffix = f"_{grid_idx[0]}_{grid_idx[1]}_{grid_idx[2]}"
        
        # MuJoCo 시스템 내 모든 Geom 순회 (보통 수십~수백 수준)
        for i in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name and name.lower().startswith(prefix) and name.endswith(suffix):
                return i
        return -1

    def _print_solver_analysis(self):
        """
        MuJoCo solref 파라미터(timeconst, dampratio)로부터 실제 물리계의 스프링 강성(K)과 
        감쇠 상수(C)를 역산하여 기술적으로 리액티브하게 리포트합니다.
        """
        # 쿠션 웰드 강성 분석
        tc = self.config.get('cush_weld_solref_stiff', 0.02)
        dr = self.config.get('cush_weld_solref_damp', 1.0)
        
        # 블록 하나당 질량 추정
        nx, ny, nz = self.config.get('cush_div', [5, 5, 3])
        mass_total = self.config.get('mass_cushion', 1.0)
        m_block = mass_total / (nx * ny * nz)
        
        # MuJoCo 물리 공식: K = 1 / (d1 * d1 * d2 * d2 * m), C = 2 / (d1 * m)
        # solref = [timeconst(d1), dampratio(d2)]
        k_val = (1.0 / (tc**2 * dr**2)) * m_block
        c_val = (2.0 / tc) * m_block
        
        self.log(f"\n[물리적 강성 확인 (Physics Analysis)]")
        self.log(f"  - Cushion 내 부속 블록 질량 (Estimated): {m_block*1000:.2f} g")
        self.log(f"  - 계산된 스프링 강성 (K-calc): {k_val:10.1f} N/m")
        self.log(f"  - 계산된 감쇠 상수 (C-calc): {c_val:10.1f} Ns/m")
        self.log("-" * 65)

    # -----------------------------------------------------------------
    # (D) 물리 연산 코어 (Aerodynamics & Plasticity)
    # -----------------------------------------------------------------
    def _apply_aerodynamics(self, model, data):
        """
        공기 유동에 의한 힘(Drag force) 및 지면과 제품 사이의 
        '스퀴즈 필름(Squeeze Film)' 저항력을 수치적으로 적산하여 적용합니다.
        """
        try:
            # 설정값 캐싱 (루프 진입 전 1회만 조회하여 성능 최적화)
            cfg = self.config
            rho = cfg.get('air_density', 1.225)
            enable_squeeze = cfg.get('enable_air_squeeze', True)
            Cd = cfg.get('air_cd_drag', 1.05)
            
            # 시뮬레이션 박스 본체 ID
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "BPackagingBox")
            if body_id == -1: return

            # 거동 데이터 추출
            current_pos = data.xpos[body_id]
            orientation_mat = data.xmat[body_id].reshape(3, 3)
            lin_vel = data.cvel[body_id, 3:6]
            ang_vel = data.cvel[body_id, 0:3]
            
            f_sq_accumulator = 0.0
            applied_torque = np.zeros(3)

            # [1] 에어로다이나믹 기초 형상 정보 확보
            bw, bh, bd = cfg.get('box_w', 2.0), cfg.get('box_h', 1.4), cfg.get('box_d', 0.25)
            box_dims = [bw, bh, bd]
            total_area = 2 * (bw*bh + bh*bd + bd*bw)

            # [2] Squeeze Film Effect (지면 근접 시 압축 공기 저항)
            if enable_squeeze:
                intensity = cfg.get('air_coef_squeeze', 1.0)
                h_limit_max = cfg.get('air_squeeze_hmax', 0.20)
                h_limit_min = cfg.get('air_squeeze_hmin', 0.005)
                
                # 박스의 6개 면 중 지면을 향하고 있는 면들을 선별하여 적산
                for axis in range(3):
                    axis_vector = orientation_mat[:, axis]
                    
                    # 면의 법선 벡터가 지면(아래)을 향할 확률이 있는 축
                    if abs(axis_vector[2]) < 1e-3:
                        continue
                    
                    # 하단을 향하는 면의 로컬 방향 보정
                    inward_sign = -np.sign(axis_vector[2])
                    local_normal = np.zeros(3)
                    local_normal[axis] = inward_sign * (box_dims[axis] / 2.0)
                    
                    # 나머지 두 축 (u, v 축) 면적 계산
                    others = [i for i in range(3) if i != axis]
                    u_len, v_len = box_dims[others[0]], box_dims[others[1]]
                    u_vec, v_vec = orientation_mat[:, others[0]], orientation_mat[:, others[1]]
                    
                    # Squeeze Film 물리 상수 (P = 0.5 * rho * (V/h)^2 계통 근사)
                    viscosity_coeff = 0.5 * rho * ((u_len * v_len) / (2 * (u_len + v_len)))**2 * intensity
                    
                    # 면 중심 좌표 (글로벌 렐러티브)
                    face_center_p = orientation_mat @ local_normal
                    
                    # 가우스 적분을 위한 샘플링 (6x6 그리드)
                    grid_res = 6
                    patch_area = (u_len * v_len) / (grid_res * grid_res)
                    grid_points = np.linspace(-0.5 + 0.5/grid_res, 0.5 - 0.5/grid_res, grid_res)
                    
                    for u_off in grid_points:
                        for v_off in grid_points:
                            sample_rel_pos = face_center_p + (u_off * u_len) * u_vec + (v_off * v_len) * v_vec
                            h_ground = current_pos[2] + sample_rel_pos[2]
                            
                            if h_limit_min < h_ground < h_limit_max:
                                pt_vel = lin_vel + np.cross(ang_vel, sample_rel_pos)
                                v_down = pt_vel[2]
                                if v_down < 0:
                                    dF = viscosity_coeff * patch_area * (v_down / h_ground)**2
                                    f_sq_accumulator += dF
                                    applied_torque += np.cross(sample_rel_pos, np.array([0, 0, dF]))

                # 물리 엔진에 직접 외력 및 토크 인가
                data.xfrc_applied[body_id, 2] += f_sq_accumulator
                data.xfrc_applied[body_id, 3:6] += applied_torque

            v_total_mag = np.linalg.norm(lin_vel)
            self._last_f_drag = 0.5 * rho * (v_total_mag**2) * Cd * (total_area / 6.0)
            self._last_f_sq = f_sq_accumulator
        except Exception as e:
            # log to console once per few errors to avoid flooding
            if not hasattr(self, '_aero_err_cnt'): self._aero_err_cnt = 0
            self._aero_err_cnt += 1
            if self._aero_err_cnt % 100 == 1:
                print(f"\n!!! [WHT_ERROR] _apply_aerodynamics failure (cnt={self._aero_err_cnt}): {e}")

    def _apply_plasticity(self):
        """
        쿠션 재입(Cushioning) 효율 극대화를 위한 '소성 변형 알고리즘 v2'입니다.
        충격 시 발생하는 물리적 변형(Strain)이 항복점(Yield)을 넘어서면 형상을 영구적으로 수축시킵니다.
        """
        if not self.config.get("enable_plasticity", False):
            return
        
        # 성능 최적화: 충돌이 아예 없으면 연산 스킵
        if self.data.ncon == 0:
            return
        
        try:
            # 스텝 초기화
            self._step_max_plasticity = {'press': 0.0, 'strain': 0.0, 'disp': 0.0}
            
            # [1] 현재 시간 스텝의 모든 실시간 지압(Pressure) 데이터 수집
            realtime_pressures = {}
            for c_idx in range(self.data.ncon):
                contact = self.data.contact[c_idx]
                for gid in [contact.geom1, contact.geom2]:
                    g_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid)
                    if g_name and "cushion" in g_name.lower():
                        if gid not in realtime_pressures:
                            realtime_pressures[gid] = 0.0
                        
                        # 충격력 추출 및 단위 면적당 압력(Pa) 연산
                        f_vec = np.zeros(6)
                        mujoco.mj_contactForce(self.model, self.data, c_idx, f_vec)
                        
                        # 면적 근사 (Geom Size 기반)
                        sz = self.model.geom_size[gid]
                        a_min = 4.0 * min(sz[0]*sz[1], sz[0]*sz[2], sz[1]*sz[2])
                        
                        if a_min > 0:
                            p_val = abs(f_vec[0]) / a_min
                            realtime_pressures[gid] += p_val
                            if p_val > self._step_max_plasticity['press']:
                                self._step_max_plasticity['press'] = p_val

            # [2] 임계값(Threshold) 로드
            yield_st = self.config.get("cush_yield_strain", 0.05)
            yield_pr = self.config.get("cush_yield_pressure", 0.0) * 1000.0 # kPa to Pa
            ratio    = self.config.get("plasticity_ratio", 1.0)
            
            # [3] Strain 기반 활성화 검사 (Neighbor-based)
            activated_geoms = {}
            for cid, nid, axis, dist0 in self.corner_neighbor_pairs:
                b1 = self.model.geom_bodyid[cid]
                b2 = self.model.geom_bodyid[nid]
                dist_curr = np.linalg.norm(self.data.xpos[b1] - self.data.xpos[b2])
                
                # Strain 계산
                strain_val = (dist0 - dist_curr) / dist0
                press_val  = realtime_pressures.get(cid, 0.0)
                disp_val   = max(0, dist0 - dist_curr)
                
                # 스텝 최대값 추적
                if strain_val > self._step_max_plasticity['strain']: self._step_max_plasticity['strain'] = strain_val
                if disp_val > self._step_max_plasticity['disp']: self._step_max_plasticity['disp'] = disp_val

                # 실시간 트래킹 데이터 캐싱 (plot용)
                self._step_corner_values[cid] = {'strain': strain_val, 'press': press_val, 'disp': disp_val}

                # 듀얼 트리거 조건 (Strain & Pressure 모두 만족 시)
                if strain_val > yield_st and press_val >= yield_pr:
                    if cid not in activated_geoms or strain_val > activated_geoms[cid]['strain']:
                        activated_geoms[cid] = {
                            'strain': strain_val, 'axis': axis, 
                            'val'   : disp_val, 'press': press_val
                        }
            
            # [4] 상태 전이 및 트래킹 업데이트
            for gid, act in activated_geoms.items():
                if gid not in self.geom_state_tracker:
                    self.geom_state_tracker[gid] = {
                        'max_comp': 0.0, 'major_axis': act['axis'], 'log_cnt': 0
                    }
                    g_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid)
                    # [NEW] 소성 변위(Strain)에 따른 노란색 -> 파란색 그라데이션 시각적 피드백
                    # f = 0 (Yellow, 0% strain) -> f = 1 (Blue, yield_strain)
                    yield_st = self.config.get("cush_yield_strain", 0.05)
                    f = np.clip(act['strain'] / max(0.001, yield_st), 0.0, 1.0)
                    r = 1.0 - f
                    g = 1.0 - f
                    b = f
                    
                    # 수동으로 이미 강조된 블록(RED 등)이 아닐 때만 적용
                    curr_color = list(self.model.geom_rgba[gid])
                    if curr_color == list(self.original_geom_rgba[gid]) or (curr_color[0] == curr_color[1] and curr_color[2] > 0):
                        self.model.geom_rgba[gid] = [r, g, b, 1.0]
                
                # 사상 최대 압축량 갱신 (Peak memory)
                if act['val'] > self.geom_state_tracker[gid]['max_comp']:
                    self.geom_state_tracker[gid]['max_comp'] = act['val']
                    self.geom_state_tracker[gid]['last_press'] = act['press']

            # [5] 영구 변형(Plastic Deformation) 실시간 적용
            for gid, state in self.geom_state_tracker.items():
                axis_idx = state['major_axis']
                
                # 현재 스텝의 탄성적 압축량 재측정
                cur_compression = 0.0
                for _c, _n, _a, _d0 in self.corner_neighbor_pairs:
                    if _c == gid and _a == axis_idx:
                        b1, b2 = self.model.geom_bodyid[_c], self.model.geom_bodyid[_n]
                        cur_dist = np.linalg.norm(self.data.xpos[b1] - self.data.xpos[b2])
                        cur_compression = max(0, _d0 - cur_dist)
                        break
                
                # 현재 탄성 수축량이 사상 최대치(Peak)보다 작아지면(Recovery 시점), 
                # 그 차이만큼 기하학적 형상을 영구 축소시킵니다.
                if state['max_comp'] > 0.0005 and cur_compression < state['max_comp']:
                    deformation_delta = (state['max_comp'] - cur_compression) * ratio
                    
                    if deformation_delta > 1e-6:
                        body_ptr = self.model.geom_bodyid[gid]
                        # 중심부로 수축하기 위한 방향 지향성 (Inward shifting)
                        move_dir = -np.sign(self.model.body_pos[body_ptr][axis_idx])
                        if abs(move_dir) < 0.1: move_dir = -1.0
                        
                        # 1. 크기 축소 (Size Reduction)
                        self.model.geom_size[gid][axis_idx] = max(0.001, self.model.geom_size[gid][axis_idx] - deformation_delta/2.0)
                        # 2. 중심 이동 (Coordinate Shift)
                        self.model.geom_pos[gid][axis_idx] += move_dir * (deformation_delta/2.0)
                        
                        # 변형 피크 갱신 (현재 값으로 리셋하여 중복 변형 방지)
                        state['max_comp'] = cur_compression
                        
                        # [NEW] 영구 변형 로깅 버퍼링
                        current_total_shrink = (self.original_geom_size[gid][axis_idx] - self.model.geom_size[gid][axis_idx]) * 1000 # mm
                        if current_total_shrink > 0.01:
                            g_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid)
                            msg = f"  [Plasticity] {g_name} Deforming(v3): -{current_total_shrink:.1f}mm (Strain: {cur_compression/(cur_compression+1e-9):.2f})"
                            if not self.plasticity_log_buffer or self.plasticity_log_buffer[-1] != msg:
                                self.plasticity_log_buffer.append(msg)
                
                # 하중 인가 중에는 Peak 계속 갱신
                if cur_compression > state['max_comp']:
                    state['max_comp'] = cur_compression
        except Exception as e:
            print(f"\n!!! [Critical Error] _apply_plasticity 실패: {e}")
            import traceback
            traceback.print_exc()


    # -----------------------------------------------------------------
    # (E) 시뮬레이션 제어 및 루프
    # -----------------------------------------------------------------
    def simulate(self):
        """
        전체 시뮬레이션 타임 스텝 루프를 구동합니다.
        Config Reload 요청 시 모델을 재생성하고 루프를 리부트합니다.
        """
        # Tkinter Root (Hidden)
        self.tk_root = tk.Tk()
        self.tk_root.withdraw()
        
        while True:
            self._run_simulation_instance()
            if not self.ctrl_reload_request:
                break
            self.ctrl_reload_request = False
            self.log("\n♻️ [Reload] 🚩 Configuration changed. Re-generating model...")
            self.setup() # XML 재생성 및 새 모델 로드

    def _run_simulation_instance(self):
        """단일 모델 인스턴스에 대한 시뮬레이션 루프입니다."""
        if not self.model:
            self.setup()

        duration = self.config.get("sim_duration", 1.0)
        dt = self.model.opt.timestep
        total_steps = int(duration / dt)
        
        # [GUI 초기화] - Viewer가 켜지는 경우 안내 텍스트 출력
        if self.config.get("use_viewer", False):
            self.log("\n" + "*"*80)
            self.log("💡 [WHTOOLS] Config Control UI가 활성화되었습니다.")
            self.log("   >> 단축키 [ K ]를 누르면 파라미터 수정 UI가 팝업됩니다.")
            self.log("   >> 시뮬레이션이 '일시정지' 상태로 시작됩니다. [Space]를 눌러 시작하세요.")
            self.log("*"*80 + "\n")
            self.ctrl_paused = True # 처음엔 멈춘 상태로 시작

            with mujoco.viewer.launch_passive(self.model, self.data, key_callback=self._on_key) as viewer:
                self.viewer = viewer
                # [STATS] 시뮬레이션 정보 표시
                try:
                    if hasattr(mujoco.mjtVisFlag, 'mjVIS_STATS'):
                        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_STATS] = 1
                except:
                    pass
                
                self._main_loop(total_steps)
                
                # [REMOVED v10] Redundant terminal input prompt (PostProcessingUI now handles final state)
                if not (self.ctrl_quit_request or self.ctrl_reload_request):
                    self.log("\n[System] 시뮬레이션이 완료되었습니다. Post-Processing UI에서 결과를 분석하세요.")
        else:
            self.viewer = None
            self._main_loop(total_steps)

    def _print_terminal_header(self):
        """터미널 출력용 컬럼 헤더를 출력합니다."""
        header = "-" * 110
        self.log(header)
        self.log(f"| {'Step':^8} | {'Sim Time (s)':^12} | {'Real Time (s)':^13} | {'FPS':^8} | {'Corner (Max P/S/D)':^40} |")
        self.log(header)

    def _main_loop(self, total_steps):
        """핵심 시뮬레이션 물리 연산 루프 (Step/Jump/Update)"""
        # 헤더 출력
        self._print_terminal_header()
        
        start_wall_time = time.time()
        last_report_sim_time = -1.0
        current_step_cnt = 0
        
        try:
            while current_step_cnt < total_steps:
                if self.ctrl_quit_request or self.ctrl_reload_request:
                    break
                
                # Tkinter GUI 이벤트 처리
                self.tk_root.update()
                
                if self.ctrl_open_ui:
                    self.open_config_ui()
                    self.ctrl_open_ui = False
                
                # UI 데이터 업데이트 (열려 있는 경우)
                if self.config_editor and self.config_editor.winfo_exists():
                    self.config_editor.update_status(current_step_cnt, self.data.time)

                if self.ctrl_reset_request:
                    self._reset_simulation_state()
                    current_step_cnt = 0
                    self.state_buffer.clear()
                    continue
                
                # [NEW] View 정보 출력 요청 처리
                if self.ctrl_print_view and self.viewer:
                    self._print_view_info()
                    self.ctrl_print_view = False
                
                # [NEW] Step Backward 처리 (Animated)
                if self.ctrl_step_backward_count > 0:
                    # 버퍼에서 상태 복원 (1 스텝씩 반복 루프를 태움)
                    if len(self.state_buffer) > 0:
                        last_state = self.state_buffer.pop()
                        # 상태 복원
                        self.data.qpos[:] = last_state['qpos']
                        self.data.qvel[:] = last_state['qvel']
                        self.data.time = last_state['time']
                        current_step_cnt = max(0, current_step_cnt - 1)
                        
                        # 히스토리 데이터 동기화
                        self._pop_last_history()
                        
                        if self.viewer: self.viewer.sync()
                        time.sleep(0.001) # 가시적 애니메이션 효과
                        
                    self.ctrl_step_backward_count -= 1
                    if self.ctrl_step_backward_count == 0:
                        self.ctrl_paused = True # 이동 완료 후 정지
                    continue
                
                # [NEW] Step Forward (Animated) 처리
                if self.ctrl_step_forward_count > 0:
                    if current_step_cnt < total_steps:
                        self._step_once()
                        current_step_cnt += 1
                        if self.viewer: self.viewer.sync()
                        time.sleep(0.001) # 가시적 애니메이션 효과
                    
                    self.ctrl_step_forward_count -= 1
                    if self.ctrl_step_forward_count == 0:
                        self.ctrl_paused = True # 이동 완료 후 정지
                    continue

                # 기존 Jump steps 처리 (Play N Steps 호환 유지)
                if self.ctrl_jump_steps > 0:
                    jump_count = min(self.ctrl_jump_steps, total_steps - current_step_cnt)
                    for _ in range(jump_count):
                        self._step_once()
                        current_step_cnt += 1
                    self.ctrl_jump_steps = 0
                    if self.viewer: self.viewer.sync()
                
                # 일반 시뮬레이션 진행
                if not self.ctrl_paused or self.ctrl_step_request:
                    self._step_once()
                    self.ctrl_step_request = False
                    current_step_cnt += 1
                else:
                    if self.viewer: self.viewer.sync()
                    time.sleep(0.01)
                    continue
                
                if self.viewer: self.viewer.sync()
                
                # 주기적 터미널 보고
                sim_slice = int(self.data.time / 0.05)
                if sim_slice > last_report_sim_time:
                    elapsed = time.time() - start_wall_time
                    fps_val = current_step_cnt / elapsed if elapsed > 0 else 0
                    mp = self._step_max_plasticity
                    corner_msg = f"P:{mp['press']/1e3:5.1f}k, S:{mp['strain']:4.2f}, D:{mp['disp']*1000:5.1f}mm"
                    self.log(f"| {current_step_cnt:8d} | {self.data.time:12.3f} | {elapsed:13.2f} | {fps_val:8.1f} | {corner_msg:^40} |")
                    last_report_sim_time = sim_slice
            
            # [NEW] 시간 종료 시 즉시 결과 처리 및 시각화 반영
            if current_step_cnt >= total_steps:
                self.log("\n" + "="*95)
                self.log("⌛ [Simulator] 지정된 시물레이션 시간(sim_duration)에 도달했습니다.")
                self.log(" >> [System] 데이터 처리 및 구조적 변형 분석 중... 잠시만 기다려 주세요.")
                
                # 분석 및 강조 표시 즉시 수행
                self._finalize_simulation()
                
                # [NEW] 리포트 생성(Plotting)을 종료 안내 전 즉시 실시 (사용자 요청)
                if self.config.get("plot_results", False):
                    self.plot_results()
                    
                self.ctrl_paused = True
                
                # 사용자가 창을 닫거나 종료 요청 전까지 무조코 UI 루프 유지
                while self.viewer and self.viewer.is_running():
                    self.tk_root.update()
                    if self.ctrl_quit_request or self.ctrl_reload_request or self.ctrl_reset_request:
                        break
                    self.viewer.sync()
                    time.sleep(0.01)
                        
        except KeyboardInterrupt:
            self.log("\n>> [Simulator] 사용자에 의해 중단되었습니다.")
        finally:
            # 예외나 중단 시에도 데이터는 보존 (이미 호출되지 않은 경우)
            if not hasattr(self, 'result') or self.result is None:
                self._finalize_simulation()
            
            # Viewer 닫기 (Passive 모드이므로 close() 호출 권장)
            if self.viewer:
                self.viewer.close()

    def _step_once(self):
        """단일 물리 스텝을 수행하고 상태를 버퍼링합니다."""
        # 상태 저장 (Step Backward용)
        state = {
            'qpos': self.data.qpos.copy(),
            'qvel': self.data.qvel.copy(),
            'time': self.data.time
        }
        self.state_buffer.append(state)
        if len(self.state_buffer) > self.max_buffer_size:
            self.state_buffer.pop(0)

        # 물리 연산 수행
        mujoco.mj_step(self.model, self.data)
        self._apply_plasticity()
        self._collect_step_data()

    def _pop_last_history(self):
        """히스토리 데이터의 마지막 항목을 제거하여 Step Backward와 동기화합니다."""
        lists = [
            self.time_history, self.z_hist, self.pos_hist, self.vel_hist, self.acc_hist,
            self.cog_pos_hist, self.cog_vel_hist, self.cog_acc_hist,
            self.ground_impact_hist, self.air_drag_hist, self.air_viscous_hist, self.air_squeeze_hist
        ]
        for h_list in lists:
            if h_list: h_list.pop()
            
        # Metrics 내의 b/t/e 히스토리도 관리 가능하나 복잡성을 위해 주요 운동 정보 우선 처리
        # (필요 시 metrics[comp][j]['bending'].pop() 등 추가)

    def _print_view_info(self):
        """현재 무조코 뷰어의 카메라 상태를 터미널에 출력합니다."""
        if not self.viewer: return
        cam = self.viewer.cam
        self.log("\n" + "="*50)
        self.log("[WHTOOLS] Current Viewer Camera Info")
        self.log(f"  - LookAt:  [{cam.lookat[0]:.3f}, {cam.lookat[1]:.3f}, {cam.lookat[2]:.3f}]")
        self.log(f"  - Distance: {cam.distance:.3f}")
        self.log(f"  - Azimuth:  {cam.azimuth:.3f}")
        self.log(f"  - Elev:     {cam.elevation:.3f}")
        self.log("  >> 위 정보를 코드의 하드코딩된 카메라 위치에 사용할 수 있습니다.")
        self.log("="*50 + "\n")

    def _on_key(self, keycode):
        """뷰어 키보드 인터랙션 콜백 인터페이스"""
        if keycode == 32:       # Space (Pause / Resume)
            self.ctrl_paused = not self.ctrl_paused
        elif keycode == 262:    # Right Arrow (Single Step Forward)
            self.ctrl_step_request = True
        elif keycode == 263:    # Left Arrow (Single Step Backward)
            self.ctrl_step_backward_count = 1
        elif keycode == 256:    # ESC (Quit)
            self.ctrl_quit_request = True
        elif keycode == 259 or keycode == 82: # Backspace or R (Reset)
            self.ctrl_reset_request = True
        elif keycode == 75 or keycode == 107:     # K or k (Open Config Editor)
            self.ctrl_open_ui = True
        elif keycode == 73 or keycode == 105:     # I or i (Print View Info)
            self.ctrl_print_view = True

    def open_config_ui(self):
        """Tkinter 기반의 설정 UI를 새 창으로 띄웁니다."""
        if self.config_editor and self.config_editor.winfo_exists():
            self.config_editor.lift()
            return
            
        self.config_editor = ConfigEditor(self)
        self.config_editor.focus_force()

    def _reset_simulation_state(self):
        """시뮬레이션 데이터와 모델의 기하 상태를 맨 처음 상태로 복원합니다."""
        mujoco.mj_resetData(self.model, self.data)
        
        # 형상 상태(Size/Pos/RGBA) 복원
        self.model.geom_pos[:]  = self.original_geom_pos
        self.model.geom_size[:] = self.original_geom_size
        self.model.geom_rgba[:] = self.original_geom_rgba
        
        # 내부 트래커 초기화
        self.geom_state_tracker.clear()
        self._init_histories()
        self._initialize_tracking_containers()
        
        self.ctrl_paused = True
        self.ctrl_reset_request = False
        self.log("\n>> [Simulator] 시스템 상태가 초기화되었습니다.")

    def _collect_step_data(self):
        """매 시뮬레이션 타임 스텝에서 필요한 물리 지표를 추출하여 메모리에 저장합니다."""
        d = self.data
        m = self.model
        
        self.time_history.append(d.time)
        
        # [1] 패키징 박스(Body)의 글로벌 거동 수집
        root_id = self.root_id
        if root_id != -1:
            global_pos = d.xpos[root_id].copy()
            rot_mat    = d.xmat[root_id].reshape(3, 3).copy()
            
            self.z_hist.append(global_pos[2])
            self.pos_hist.append(global_pos)
            self.vel_hist.append(d.cvel[root_id].copy())
            self.acc_hist.append(d.cacc[root_id].copy())
            self.cog_pos_hist.append(d.subtree_com[root_id].copy())
            
            # [2] 지면 충격력(Ground Reaction Force) 적산
            net_impact_f = 0.0
            for c_idx in range(d.ncon):
                contact = d.contact[c_idx]
                if m.geom_bodyid[contact.geom1] == 0 or m.geom_bodyid[contact.geom2] == 0:
                    f_force = np.zeros(6)
                    mujoco.mj_contactForce(m, d, c_idx, f_force)
                    net_impact_f += abs(f_force[0])
                    
            self.ground_impact_hist.append(net_impact_f)
            self.air_drag_hist.append(self._last_f_drag)
            self.air_squeeze_hist.append(self._last_f_sq)

            # [4] 코너(8-Points) 기구학 추적
            c_pos_now = []
            for cb_id in self.corner_body_ids:
                c_pos_now.append(d.xpos[cb_id].copy())
            self.corner_pos_hist.append(c_pos_now)
            
            # (구조적 변형 분석을 위한 회전 행렬 확보)
            current_rot_mat = rot_mat
        else:
            # root_id를 찾지 못한 경우에도 리스트 길이를 맞추기 위해 기본값 삽입
            self.z_hist.append(0.0)
            self.pos_hist.append(np.zeros(3))
            self.vel_hist.append(np.zeros(6))
            self.acc_hist.append(np.zeros(6))
            self.cog_pos_hist.append(np.zeros(3))
            self.ground_impact_hist.append(0.0)
            self.air_drag_hist.append(0.0)
            self.air_squeeze_hist.append(0.0)
            self.corner_pos_hist.append([np.zeros(3)] * len(self.corner_body_ids))
            current_rot_mat = np.eye(3)

        # [3] 구조적 변형 분석 (Component-level Metrics) - 성능을 위해 5스텝마다 수행 (Decimation)
        if len(self.time_history) % 5 != 1:
            for comp_name, comp_metric in self.metrics.items():
                last_tdi = comp_metric['total_distortion'][-1] if comp_metric['total_distortion'] else 0.0
                comp_metric['total_distortion'].append(last_tdi)
                for grid_idx in self.components[comp_name]:
                    last_ang = comp_metric['all_blocks_angle'][grid_idx][-1] if comp_metric['all_blocks_angle'][grid_idx] else 0.0
                    comp_metric['all_blocks_angle'][grid_idx].append(last_ang)
                
                # [NEW] 코너 히스토리 데이터 복제 (Decimation)
                if 'corner_hists' in comp_metric:
                    for gid, h_data in comp_metric['corner_hists'].items():
                        for k in ['strain', 'press', 'disp', 'plastic']:
                            last_val = h_data[k][-1] if h_data[k] else 0.0
                            h_data[k].append(last_val)
            return

        inv_root_mat = current_rot_mat.T
        for comp_name, comp_metric in self.metrics.items():
            list_of_angles = []
            
            # 해당 컴포넌트 내 모든 블록들에 대한 상대 회전 분석
            for grid_idx, body_uid in self.components[comp_name].items():
                block_mat = d.xmat[body_uid].reshape(3, 3)
                
                # 박스 본체에 대한 상대 회전 행렬
                relative_rot = inv_root_mat @ block_mat
                
                # 최초 상태 기록
                if comp_metric['block_nominal_mats'][grid_idx] is None:
                    comp_metric['block_nominal_mats'][grid_idx] = relative_rot.copy()
                
                # 초기 상대 자세와의 편차(Angular Distortion) 계산
                deviation_mat = comp_metric['block_nominal_mats'][grid_idx].T @ relative_rot
                
                # [DECOMPOSITION] Bending (Tilt) and Twist (Torsion)
                # Bending: Rotation of the local Z axis away from nominal
                bend_deg = np.degrees(np.arccos(np.clip(deviation_mat[2, 2], -1.0, 1.0)))
                # Twist: Rotation around the local Z axis
                twist_deg = np.degrees(np.arctan2(deviation_mat[1, 0], deviation_mat[0, 0]))
                
                comp_metric['all_blocks_bend'][grid_idx].append(bend_deg)
                comp_metric['all_blocks_twist'][grid_idx].append(twist_deg)
                
                # 기존 통합 Angle (Root Mean Square or Absolute Vector Angle)
                trace_val = np.trace(deviation_mat)
                rotation_angle = np.degrees(np.arccos(np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0)))
                comp_metric['all_blocks_angle'][grid_idx].append(rotation_angle)
                list_of_angles.append(rotation_angle)
            
            # [NEW] 코너 히스토리 데이터 실시간 수집
            if 'corner_hists' in comp_metric:
                for gid, h_data in comp_metric['corner_hists'].items():
                    cv = self._step_corner_values.get(gid, {'strain': 0.0, 'press': 0.0, 'disp': 0.0})
                    # 소성 변형량 계산
                    body_idx = self.model.geom_bodyid[gid]
                    axis_idx = self.geom_state_tracker.get(gid, {}).get('major_axis', 2)
                    p_val = (self.original_geom_size[gid][axis_idx] - self.model.geom_size[gid][axis_idx])
                    
                    h_data['strain'].append(cv['strain'])
                    h_data['press'].append(cv['press'])
                    h_data['disp'].append(cv['disp'])
                    h_data['plastic'].append(p_val)
                
            # 전체 부품의 뒤틀림 강도 (RMS Angle)
            if list_of_angles:
                rms_distortion = np.sqrt(np.mean(np.array(list_of_angles)**2))
                comp_metric['total_distortion'].append(rms_distortion)
            else:
                comp_metric['total_distortion'].append(0.0)

    def _finalize_simulation(self):
        """시뮬레이션 루프가 끝난 뒤 결과 데이터의 최종 가공 및 객체 저장을 수행합니다."""
        
        # [NEW] 소성 변형(Plasticity) 및 구조적 변위 시각화
        highlight_plastic_count = 0
        highlight_distortion_count = 0
        
        # 테이블 너비 조정 (가독성 향상)
        print("\n" + "="*115)
        print(" [Component Distortion Summary Table]")
        print("-" * 115)
        print(f" {'Body Name':<25} | {'Max Bend(deg)':>15} | {'Max Twist(deg)':>15} | {'Max Plastic(mm)':>17} | {'Highlighted'}")
        print("-" * 115)

        for comp_name, comp_metric in self.metrics.items():
            max_bend = 0.0
            max_twist = 0.0
            max_total_score = 0.0
            max_block_uid = None
            max_block_idx = None
            
            # 1. Bending/Twist (Angular Distortion) 분석 및 점수화
            block_scores = {}
            for grid_idx in comp_metric['all_blocks_bend'].keys():
                bend_history = comp_metric['all_blocks_bend'][grid_idx]
                twist_history = comp_metric['all_blocks_twist'][grid_idx]
                
                if bend_history and twist_history:
                    m_bend = max([abs(b) for b in bend_history])
                    m_twist = max([abs(t) for t in twist_history])
                    
                    if m_bend > max_bend: max_bend = m_bend
                    if m_twist > max_twist: max_twist = m_twist
                    
                    # 변형 점수 (Bend와 Twist의 평균치)
                    score = (m_bend + m_twist) / 2.0
                    block_scores[grid_idx] = score
                    
                    if score > max_total_score:
                        max_total_score = score
                        max_block_uid = self.components[comp_name][grid_idx]
                        max_block_idx = grid_idx

            # [REMOVED v10] Automatic distortion coloring moved to PostProcessingUI
            
            # 2. 소성 변형(Plasticity) 시각화 및 최대치 산출
            max_plastic_mm = 0.0
            yield_st = self.config.get("cush_yield_strain", 0.05)
            
            for grid_idx, body_uid in self.components[comp_name].items():
                g_id = -1
                for gi in range(self.model.ngeom):
                    if self.model.geom_bodyid[gi] == body_uid:
                        g_id = gi
                        break
                
                if g_id >= 0:
                    state = self.geom_state_tracker.get(g_id, {})
                    axis_idx = state.get('major_axis', 2)
                    orig_sz = self.original_geom_size[g_id][axis_idx]
                    curr_sz = self.model.geom_size[g_id][axis_idx]
                    plastic_mm = (orig_sz - curr_sz) * 1000
                    
                    if plastic_mm > max_plastic_mm:
                        max_plastic_mm = plastic_mm
                    
                    # [BCUSHION/Plasticity] 소성 그라데이션 (Yellow -> Blue)
                    if plastic_mm > 0.1 and "cushion" in comp_name.lower():
                        plastic_strain = plastic_mm / (1000.0 * max(0.001, orig_sz))
                        f_p = np.clip(plastic_strain / max(0.0001, yield_st), 0.0, 1.0)
                        
                        r_p, g_p, b_p = 1.0 - f_p, 1.0 - f_p, 1.0
                        self.model.geom_rgba[g_id] = [r_p, g_p, b_p, 1.0]
                        highlight_plastic_count += 1

            # 테이블 출력 (Wider Columns)
            print(f" {comp_name:<25} | {max_bend:15.2f} | {max_twist:15.2f} | {max_plastic_mm:17.2f} | {str(max_block_idx)}")

        print("-" * 115)
        print(f" >> [Visual Result] Distortion Heatmap: Available in Post-UI, Plastic(BLUE Gradient): {highlight_plastic_count} blocks")
        print("="*115)
        
        # [NEW v9] 상세 블록별 변위 리포트 (Detailed Block-level Report)
        print("\n" + "="*80)
        print(" [Detailed Block-by-Block Distortion Breakdown]")
        print("="*80)
        
        for comp_name, comp_metric in self.metrics.items():
            print(f"\n >> Component: {comp_name}")
            print("-" * 60)
            print(f" {'Block (i,j,k)':<18} | {'Max Bend(deg)':>15} | {'Max Twist(deg)':>15}")
            print("-" * 60)
            
            # 그리드 인덱스 순으로 정렬하여 상세 출력
            sorted_indices = sorted(comp_metric['all_blocks_bend'].keys())
            for g_idx in sorted_indices:
                bend_hist = comp_metric['all_blocks_bend'][g_idx]
                twist_hist = comp_metric['all_blocks_twist'][g_idx]
                
                m_bend_val = max([abs(b) for b in bend_hist]) if bend_hist else 0.0
                m_twist_val = max([abs(t) for t in twist_hist]) if twist_hist else 0.0
                
                print(f" {str(g_idx):<18} | {m_bend_val:15.2f} | {m_twist_val:15.2f}")
            print("-" * 60)
            
        # [NEW v10] Post-Processing UI 실행
        self.tk_root.after(100, self.open_post_ui)
        
        if self.viewer: self.viewer.sync()

    def open_post_ui(self):
        """시뮬레이션 완료 후 포스트 프로세싱 UI를 엽니다."""
        if self.tk_root:
            self.post_ui = PostProcessingUI(self)
            self.tk_root.update()

    def apply_rank_distortion_heatmap(self):
        """
        부품 내에서 변형 점수(Bend+Twist)/2의 '순위(Rank)'를 매겨, 
        원래 색상에서 RED까지 선형적으로 배분합니다. (Max Contrast 확보)
        """
        highlight_count = 0
        for comp_name, comp_metric in self.metrics.items():
            block_scores = {}
            for grid_idx in comp_metric['all_blocks_bend'].keys():
                bend_h = comp_metric['all_blocks_bend'][grid_idx]
                twist_h = comp_metric['all_blocks_twist'][grid_idx]
                if bend_h and twist_h:
                    m_bend = max([abs(b) for b in bend_h])
                    m_twist = max([abs(t) for t in twist_h])
                    score = (m_bend + m_twist) / 2.0
                    # 노이즈 필터링 (0.5도 미만 무시)
                    if score > 0.5:
                        block_scores[grid_idx] = score
            
            if not block_scores:
                continue
                
            # 순위 매기기 (오름차순: 낮은 변형 -> 높은 변형)
            sorted_blocks = sorted(block_scores.items(), key=lambda x: x[1])
            num_blocks = len(sorted_blocks)
            
            for rank, (grid_idx, _) in enumerate(sorted_blocks):
                # 0순위(가장 낮음) -> f=0.0, 최종순위(가장 높음) -> f=1.0
                f = rank / (num_blocks - 1) if num_blocks > 1 else 1.0
                
                # [NEW v11] Matplotlib Colormap 동기화 (RdYlBu_r)
                import matplotlib.cm as cm
                cmap = cm.get_cmap('RdYlBu_r')
                rgba_color = cmap(f) # (R, G, B, A)
                
                b_uid = self.components[comp_name][grid_idx]
                for g_idx in range(self.model.ngeom):
                    if self.model.geom_bodyid[g_idx] == b_uid:
                        # MuJoCo geom_rgba 에 적용
                        self.model.geom_rgba[g_idx] = rgba_color
                        highlight_count += 1
        
        if self.viewer: self.viewer.sync()
        self.log(f">> [Visual Feedback] Rank-based Heatmap (RdYlBu_r) applied: {highlight_count} geoms.")

    def plot_2d_distortion_map(self, target_comp: str):
        """
        Matplotlib을 사용하여 지정된 바디의 2D 보간된 Bend/Twist 히트맵을 생성합니다.
        가로 10, 세로 5 인치, 폰트 9pt, RdYlBu_r 컬러맵 및 Equal Aspect 적용.
        """
        if target_comp not in self.metrics:
            self.log(f"!! [PlotError] '{target_comp}'에 대한 분석 데이터가 존재하지 않습니다.")
            return
            
        comp_metric = self.metrics[target_comp]
        all_idxs = list(comp_metric['all_blocks_bend'].keys())
        if not all_idxs:
            self.log(f"!! [PlotError] '{target_comp}' 블록 데이터가 비어 있습니다.")
            return

        # 1. 그리드 데이터 준비 (Z축 축합)
        max_i = max(idx[0] for idx in all_idxs)
        max_j = max(idx[1] for idx in all_idxs)
        
        grid_bend = np.zeros((max_j + 1, max_i + 1))
        grid_twist = np.zeros((max_j + 1, max_i + 1))
        
        for (i, j, k) in all_idxs:
            mb = max([abs(val) for val in comp_metric['all_blocks_bend'][(i,j,k)]])
            mt = max([abs(val) for val in comp_metric['all_blocks_twist'][(i,j,k)]])
            # Z축 여러 층이 있다면 최댓값 유지
            grid_bend[j, i] = max(grid_bend[j, i], mb)
            grid_twist[j, i] = max(grid_twist[j, i], mt)

        # 2. 시각화 설정 (10x5, Font 9pt)
        plt.rcParams.update({'font.size': 9, 'font.family': 'sans-serif'})
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        fig.patch.set_facecolor('#f5f5f5')
        
        fig.suptitle(f"[WHTOOLS Post-Analysis] 2D Distortion Map: {target_comp.upper()}", fontsize=12, fontweight='bold', color='#1a1a1a')
        cmap = 'RdYlBu_r'
        
        # Left: Bending
        im1 = ax1.imshow(grid_bend, interpolation='bicubic', cmap=cmap, origin='lower')
        ax1.set_title("Bending Intensity (Deg)", fontweight='bold', pad=10)
        ax1.set_xlabel("Grid X (i)"); ax1.set_ylabel("Grid Y (j)")
        ax1.set_aspect('equal', adjustable='box') # [v11] 비율 유지
        plt.colorbar(im1, ax=ax1, shrink=0.8, pad=0.05)

        # Right: Twist
        im2 = ax2.imshow(grid_twist, interpolation='bicubic', cmap=cmap, origin='lower')
        ax2.set_title("Twist Intensity (Deg)", fontweight='bold', pad=10)
        ax2.set_xlabel("Grid X (i)"); ax2.set_ylabel("Grid Y (j)")
        ax2.set_aspect('equal', adjustable='box') # [v11] 비율 유지
        plt.colorbar(im2, ax=ax2, shrink=0.8, pad=0.05)

        plt.tight_layout(rect=[0, 0.05, 1, 0.92])
        plt.show() # Windows 환경에서는 새 창으로 팝업됨

        z_array = np.array(self.z_hist)
        dt_val = self.model.opt.timestep
        
        # 가속도 수치 미분을 통한 정밀 G-Force 산출
        if len(z_array) > 3:
            vel_z = np.gradient(z_array, dt_val)
            acc_z = np.gradient(vel_z, dt_val)
            # 중력 가속도(9.81) 기반 G 환산
            root_acc_g = np.abs(acc_z) / 9.81
            max_peak_g = np.max(root_acc_g)
        else:
            root_acc_g = []
            max_peak_g = 0.0
            
        # 결과 데이터 컨테이너 생성
        self.result = DropSimResult(
            config             = self.config,
            metrics            = self.metrics,
            max_g_force        = max_peak_g,
            time_history       = self.time_history,
            z_hist             = self.z_hist,
            root_acc_history   = list(root_acc_g),
            corner_acc_hist    = [],
            pos_hist           = self.pos_hist,
            vel_hist           = self.vel_hist,
            acc_hist           = self.acc_hist,
            cog_pos_hist       = self.cog_pos_hist,
            ground_impact_hist = self.ground_impact_hist,
            air_drag_hist      = self.air_drag_hist,
            air_squeeze_hist   = self.air_squeeze_hist
        )
        
        self.log("\n" + "=" * 70)
        self.log(f"| 시뮬레이션 완료 | 최종 최대 가속도 : {max_peak_g:10.2f} G |")
        self.log("=" * 70)
        
        # [NEW] 소성 변형 상세 로그 요약 출력 (옵션)
        if self.config.get("print_corner_plasticity", False) and self.plasticity_log_buffer:
            self.log("\n[Cushion Plasticity Detailed Event Summary]")
            self.log("-" * 65)
            for msg in self.plasticity_log_buffer:
                self.log(msg)
            self.log("-" * 65)

        # 파일 저장
        output_filename = f"rds-{self.timestamp}_result.pkl"
        save_path = os.path.join(self.output_dir, output_filename)
        self.result.save(save_path)
        
        self.log(f">> [System] 전체 데이터가 바이너리 파일로 패키징되었습니다: {output_filename}")

    def plot_results(self):
        """
        수집된 핵심 데이터를 시각화하여 고해상도 리포트 파일로 저장합니다.
        v2의 강력한 분석 기능을 유지하면서 가독성과 필터링 기능을 강화했습니다.
        """
        if self._analysis_done:
            return # 이미 리포트가 생성됨
            
        if not self.result:
            self.log("!! [PlotError] 시뮬레이션 결과 데이터가 존재하지 않습니다.")
            return

        self.log("\n>> [Analysis] 고해상도 시뮬레이션 결과 리포트를 생성합니다...")
        p_dir = self.output_dir
        t_hist = self.result.time_history
        
        # [1] G-Force 충격 선도 (Z-axis Root Acceleration)
        plt.figure(figsize=(10, 5))
        plt.plot(t_hist, self.result.root_acc_history, color='navy', linewidth=1.5, label='Assembly G (Z-axis)')
        plt.fill_between(t_hist, self.result.root_acc_history, color='navy', alpha=0.1)
        plt.title("MuJoCo Drop Simulation: Impact G-Force", fontsize=14)
        plt.xlabel("Sim Time (sec)", fontsize=11); plt.ylabel("Acceleration (G)", fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.6); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(p_dir, "rds-impact_g_trace.png"), dpi=300); plt.close()
        self.log("   - 충격 선도 저장 완료: rds-impact_g_trace.png")

        # [2] 지면 충격력 및 공기 저항 성분 분석
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        
        g_hist = self.result.ground_impact_hist
        if len(g_hist) == len(t_hist):
            ax1.plot(t_hist, g_hist, label='Ground Normal Force (N)', color='crimson')
        ax1.set_ylabel('Force (N)'); ax1.set_title('Ground Impact Force'); ax1.grid(True); ax1.legend()
        
        d_hist = self.result.air_drag_hist
        s_hist = self.result.air_squeeze_hist
        if len(d_hist) == len(t_hist):
            ax2.plot(t_hist, d_hist, label='Estimated Air Drag', color='blue')
        if len(s_hist) == len(t_hist):
            ax2.plot(t_hist, s_hist, label='Squeeze Film Effect', color='orange')
            
        ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Force (N)'); ax2.set_title('Air Resistance Components'); ax2.grid(True); ax2.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(p_dir, "rds-ground_impact.png"), dpi=200); plt.close()

        # [3] 조립체 및 모서리 기구학 분석 (Motion All & Motion Z)
        pos_np = np.array(self.pos_hist) # (steps, 3)
        vel_np = np.array(self.vel_hist) # (steps, 6)
        acc_np = np.array(self.acc_hist) # (steps, 6)
        c_pos_np = np.array(self.corner_pos_hist) # (steps, 8, 3)
        
        # Z-Axis 전용 시각화 (데이터 부족 시 스킵)
        if pos_np.ndim < 2 or pos_np.shape[0] < 2:
            self.log("!! [PlotNotice] 데이터 부족으로 기구학 도표 생성을 생략합니다.")
            return

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Kinematics (Z-Axis Only): Position, Velocity, Acceleration', fontsize=15)
        titles_z = ['Z Position (m)', 'Z Velocity (m/s)', 'Z Acceleration (m/s^2)']
        
        # 6자유도의 경우 v_z=idx 5, a_z=idx 5
        v_z = vel_np[:, 5] if vel_np.shape[1] > 5 else vel_np[:, 2]
        a_z = acc_np[:, 5] if acc_np.shape[1] > 5 else acc_np[:, 2]
        metrics_z = [pos_np[:, 2], v_z, a_z]
        
        for i, (title, data) in enumerate(zip(titles_z, metrics_z)):
            axs[i].plot(t_hist, data, label='Root Center', color='black', linewidth=1.2)
            # 코너 8개 데이터 추가
            if c_pos_np.ndim == 3:
                for ci in range(min(8, c_pos_np.shape[1])):
                    axs[i].plot(t_hist, c_pos_np[:, ci, 2], alpha=0.3, linewidth=0.8)
            axs[i].set_title(title); axs[i].set_xlabel('Time (s)'); axs[i].grid(True)
        plt.tight_layout(); plt.savefig(os.path.join(p_dir, "rds-Motion_Z.png"), dpi=200); plt.close()

        # [4] 부품별 구조적 변형 분석 (Bending, Twist, Plasticity)
        for comp, metric in self.metrics.items():
            # 블록이 1개인 부품(Rigid/Aux)은 분석 제외
            if len(self.components.get(comp, {})) <= 1: continue

            # TDI (Total Distortion Index) 플롯
            if 'total_distortion' in metric and metric['total_distortion']:
                plt.figure(figsize=(10, 5))
                plt.plot(t_hist, metric['total_distortion'], color='purple', linewidth=1.8)
                plt.title(f'{comp} Structural Distortion Index (TDI)'); plt.grid(True)
                plt.xlabel('Time (s)'); plt.ylabel('RMS Angle (deg)')
                plt.savefig(os.path.join(p_dir, f"rds-{comp}_TDI.png"), dpi=200); plt.close()

            # 개별 블록 각도 변화 + i-j-k 가이드 맵
            all_angles = metric.get('all_blocks_angle', {})
            if all_angles:
                fig = plt.figure(figsize=(14, 7))
                ax_main = fig.add_axes([0.07, 0.1, 0.60, 0.8])
                for idx, a_hist in all_angles.items():
                    ax_main.plot(t_hist, a_hist, label=f"{idx}", alpha=0.7, linewidth=0.9)
                ax_main.set_title(f"{comp} Block Deformation Status"); ax_main.grid(True)
                ax_main.set_xlabel('Time (s)'); ax_main.set_ylabel('Defl. Angle (deg)')
                # 범례가 그래프를 가리지 않도록 외부에 배치 (유저 요청 해결)
                ax_main.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), ncol=3, fontsize=7)

                # i-j-k Reference Inset
                ax_inset = fig.add_axes([0.68, 0.05, 0.30, 0.30], projection='3d')
                noms = metric.get('block_nominals', {})
                if noms:
                    xs, ys, zs = zip(*[noms[k] for k in sorted(noms.keys())])
                    ax_inset.scatter(xs, ys, zs, s=15, color='blue', alpha=0.4)
                    ax_inset.set_title("i-j-k Reference Map", fontsize=9)
                    ax_inset.set_xticks([]); ax_inset.set_yticks([]); ax_inset.set_zticks([])
                plt.savefig(os.path.join(p_dir, f"rds-{comp}_deformation_all.png"), dpi=200); plt.close()

            # [REFINED] Corner Plasticity Analysis (B-Cushion에 대해서만 선택적 출력)
            if comp.lower() == "bcushion" and 'corner_hists' in metric:
                c_hists = metric['corner_hists']
                if c_hists:
                    # 데이터가 하나라도 있는지 확인 (ValueError 방지)
                    any_data = any(len(h['strain']) > 0 for h in c_hists.values())
                    if any_data:
                        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
                        fig.suptitle(f'Cushion Corner Impact Plasticity Analysis', fontsize=16)
                        keys = [('strain', 'Strain'), ('press', 'Pressure (Pa)'), ('disp', 'Compression (m)'), ('plastic', 'Perm. Def (m)')]
                        for i, (k_id, k_label) in enumerate(keys):
                            ax = axs[i//2, i%2]
                            for cid, hist in c_hists.items():
                                if len(hist[k_id]) == len(t_hist):
                                    ax.plot(t_hist, hist[k_id], label=hist.get('name', f'G{cid}'), alpha=0.6)
                            ax.set_title(k_label); ax.set_xlabel('Time (s)'); ax.grid(True)
                        axs[0, 1].legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), fontsize=7)
                        plt.tight_layout(rect=[0, 0, 0.88, 0.95])
                        plt.savefig(os.path.join(p_dir, f"rds-{comp}_corner_analysis.png"), dpi=250); plt.close()
                        self.log(f"   - [Cushion Analysis] 코너 변형 분석 도표 생성 완료.")

        self.log(f">> [System] 모든 분석 리포트가 {p_dir} 폴더에 저장되었습니다.")
        
        self._analysis_done = True
        
    def show_impact_plots(self):
        """
        [NEW v11] 저장된 시뮬레이션 결과(G-Force, Kinematics)를 
        Matplotlib 팝업 창으로 즉시 시가화합니다.
        """
        if not hasattr(self, 'result') or self.result is None:
            self.log("!! [PlotError] 시뮬레이션 결과 데이터가 존재하지 않습니다.")
            return

        t_hist = self.result.time_history
        
        # [1] G-Force 충격 선도 팝업
        plt.figure(figsize=(9, 5))
        plt.plot(t_hist, self.result.root_acc_history, color='navy', linewidth=1.5, label='Assembly G (Z-axis)')
        plt.fill_between(t_hist, self.result.root_acc_history, color='navy', alpha=0.1)
        plt.title("[WHTOOLS] Impact G-Force Trace", fontsize=12, fontweight='bold')
        plt.xlabel("Sim Time (sec)"); plt.ylabel("Acceleration (G)")
        plt.grid(True, linestyle='--', alpha=0.6); plt.legend()
        plt.tight_layout()
        plt.show(block=False) 

        # [2] Z-Axis Kinematics 팝업
        pos_np = np.array(self.pos_hist)
        vel_np = np.array(self.vel_hist)
        acc_np = np.array(self.acc_hist)
        
        if pos_np.ndim >= 2 and pos_np.shape[0] >= 2:
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle('[WHTOOLS] Kinematics Analysis (Z-Axis)', fontsize=14, fontweight='bold')
            
            v_z = vel_np[:, 5] if vel_np.shape[1] > 5 else vel_np[:, 2]
            a_z = acc_np[:, 5] if acc_np.shape[1] > 5 else acc_np[:, 2]
            
            titles = ['Z Position (m)', 'Z Velocity (m/s)', 'Z Acceleration (m/s^2)']
            datasets = [pos_np[:, 2], v_z, a_z]
            
            for i, (title, data) in enumerate(zip(titles, datasets)):
                axs[i].plot(t_hist, data, color='black')
                axs[i].set_title(title); axs[i].set_xlabel('Time (s)'); axs[i].grid(True)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()

# =====================================================================
# 단독 실행 모드
# =====================================================================
if __name__ == "__main__":
    # 인스턴스 생성 및 실행
    sim_engine = DropSimulator()
    sim_engine.simulate()
    sim_engine.plot_results()
