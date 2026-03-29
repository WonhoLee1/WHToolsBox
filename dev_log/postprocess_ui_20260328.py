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
from tkinter import ttk, messagebox

# Matplotlib 백엔드는 TkAgg 우선 (팝업 창 지원)
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

try:
    from PIL import Image, ImageTk
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


class PostProcessingUI(tk.Toplevel):
    """
    [WHTOOLS v4] 시뮬레이션 완료 후 결과를 탐색하는 고도화 포스트 프로세싱 UI.

    Args:
        parent_sim: DropSimulator 인스턴스 (데이터 소스)
    """

    # 위치 레이블 상수 (8코너 + 기하중심 + 질량중심)
    LOCATION_LABELS = [
        "Corner(-x,-y,-z)", "Corner(-x,-y,+z)",
        "Corner(-x,+y,-z)", "Corner(-x,+y,+z)",
        "Corner(+x,-y,-z)", "Corner(+x,-y,+z)",
        "Corner(+x,+y,-z)", "Corner(+x,+y,+z)",
        "Geo Center", "Mass Center (CoM)"
    ]

    def __init__(self, parent_sim):
        super().__init__()
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

        # ---- 선택 변수 ----
        self._loc_vars = [tk.BooleanVar(value=(i < 2)) for i in range(10)]
        self._data_type_var = tk.StringVar(value="displacement")
        self._axis_vars = {
            "X": tk.BooleanVar(value=False),
            "Y": tk.BooleanVar(value=False),
            "Z": tk.BooleanVar(value=True),
        }
        self._contour_mode_var = tk.StringVar(value="temporal")
        self._contour_metric_var = tk.StringVar(value="bend")

        comp_list = sorted(list(self.sim.metrics.keys()))
        self._comp_var = tk.StringVar(value=comp_list[0] if comp_list else "")

        self._build_ui()
        self._update_frame_count()

    # ==============================================================
    # UI 빌드
    # ==============================================================

    def _build_ui(self):
        """전체 UI 레이아웃을 구성합니다."""
        # 배너
        if _PIL_AVAILABLE:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            banner_path = os.path.join(script_dir, "ui_banner.png")
            if os.path.exists(banner_path):
                try:
                    img = Image.open(banner_path)
                    w, h = img.size
                    target_w = 1120
                    target_h = int(h * (target_w / w))
                    img = img.resize((target_w, target_h), Image.LANCZOS)
                    self._banner_img = ImageTk.PhotoImage(img)
                    tk.Label(self, image=self._banner_img, bg="#1a1a1a").pack(fill="x", padx=10, pady=3)
                except Exception:
                    pass

        # 공통 시간 컨트롤 패널
        self._build_time_control_panel()

        # 탭 노트북
        self._notebook = ttk.Notebook(self)
        self._notebook.pack(fill="both", expand=True, padx=8, pady=4)

        tab1 = ttk.Frame(self._notebook)
        tab2 = ttk.Frame(self._notebook)
        tab3 = ttk.Frame(self._notebook)

        self._notebook.add(tab1, text="  기구학 분석 (Kinematics)  ")
        self._notebook.add(tab2, text="  구조 해석 (Structural)  ")
        self._notebook.add(tab3, text="  2D 컨투어 (Contour Map)  ")

        self._build_kinematics_tab(tab1)
        self._build_structural_tab(tab2)
        self._build_contour_tab(tab3)

        # 하단 버튼
        footer = ttk.Frame(self)
        footer.pack(fill="x", pady=5)
        ttk.Button(footer, text="MuJoCo 히트맵 적용", command=self._on_apply_heatmap).pack(side="left", padx=10)
        ttk.Button(footer, text="닫기", command=self.on_close).pack(side="right", padx=10)

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
        """
        매트릭스 레이아웃 기구학 그래프 탭입니다.

        레이아웃:
            좌측: 데이터 종류 선택(변위/속도/가속도) + 축 멀티 선택 + 위치 멀티 선택
            우측: [매트릭스 그래프 생성] 버튼
        출력:
            행 = 선택한 데이터 종류, 열 = 선택한 축 종류,
            각 서브플롯 내 = 선택한 위치별 시계열 곡선
        """
        ctrl = ttk.Frame(parent)
        ctrl.pack(fill="x", padx=8, pady=6)

        # 데이터 종류 선택
        dtype_f = ttk.LabelFrame(ctrl, text="  1. 데이터 종류  ")
        dtype_f.pack(side="left", fill="y", padx=4)
        for val, txt in [("displacement", "변위 (Pos)"),
                          ("velocity",     "속도 (Vel)"),
                          ("acceleration", "가속도 (Acc)")]:
            ttk.Radiobutton(dtype_f, text=txt, variable=self._data_type_var,
                            value=val).pack(anchor="w", padx=6, pady=1)

        # 축 선택 (멀티)
        axis_f = ttk.LabelFrame(ctrl, text="  2. 축 (멀티 선택)  ")
        axis_f.pack(side="left", fill="y", padx=4)
        for ax in ["X", "Y", "Z"]:
            ttk.Checkbutton(axis_f, text=ax + " 축",
                            variable=self._axis_vars[ax]).pack(anchor="w", padx=6, pady=1)

        # 위치 선택 (멀티 체크박스)
        loc_f = ttk.LabelFrame(ctrl, text="  3. 추출 위치 (멀티 선택)  ")
        loc_f.pack(side="left", fill="y", padx=4)
        for i, lbl in enumerate(self.LOCATION_LABELS):
            ttk.Checkbutton(loc_f, text=lbl,
                            variable=self._loc_vars[i]).pack(anchor="w", padx=4, pady=1)

        # 전체 선택/해제 보조 버튼
        sel_f = ttk.Frame(loc_f)
        sel_f.pack(fill="x", padx=4)
        ttk.Button(sel_f, text="전체 선택", width=9,
                   command=lambda: [v.set(True) for v in self._loc_vars]).pack(side="left", pady=2)
        ttk.Button(sel_f, text="전체 해제", width=9,
                   command=lambda: [v.set(False) for v in self._loc_vars]).pack(side="left", pady=2)

        # 실행 버튼
        btn_f = ttk.Frame(ctrl)
        btn_f.pack(side="left", fill="y", padx=12)
        ttk.Button(btn_f, text="매트릭스 그래프 생성",
                   command=self._on_plot_kinematics, width=20).pack(pady=6, padx=5)
        ttk.Label(btn_f,
                  text="선택한 데이터 종류 x 축 종류\n매트릭스 레이아웃으로 출력합니다.\n각 그래프 내에 위치별 곡선 중첩.",
                  foreground="gray", font=("NanumGothic", 8), justify="left").pack(padx=5)

    # ==============================================================
    # [탭 2] 구조 해석
    # ==============================================================

    def _build_structural_tab(self, parent):
        """PBA, RRG, Bending/Twist 시계열 그래프 탭."""
        ctrl = ttk.Frame(parent)
        ctrl.pack(fill="x", padx=8, pady=6)

        # 지표 선택
        metric_f = ttk.LabelFrame(ctrl, text="  구조 해석 지표 선택  ")
        metric_f.pack(side="left", fill="y", padx=4)

        opts = [
            ("PBA Magnitude (°)",      "pba_magnitude"),
            ("PBA Direction (°)",      "pba_angle"),
            ("RRG Max (°)",            "rrg_max"),
            ("Mean Distortion (°)",    "mean_distortion"),
            ("Bending Max (전 블록)", "bend_overall"),
            ("Twist Max (전 블록)",   "twist_overall"),
        ]
        self._struct_metric_vars = {}
        for txt, key in opts:
            var = tk.BooleanVar(value=key in ("rrg_max", "pba_magnitude"))
            self._struct_metric_vars[key] = (var, txt)
            ttk.Checkbutton(metric_f, text=txt, variable=var).pack(anchor="w", padx=6, pady=1)

        # 그래프 생성 버튼
        btn_f = ttk.Frame(ctrl)
        btn_f.pack(side="left", fill="y", padx=12)
        ttk.Button(btn_f, text="구조 해석 그래프 생성",
                   command=self._on_plot_structural, width=20).pack(pady=6)

        # 임계 시점 정보 표시
        info_f = ttk.LabelFrame(ctrl, text="  자동 검출 임계 시점  ")
        info_f.pack(side="left", fill="y", padx=4)
        self._critical_text = tk.Text(info_f, width=38, height=10,
                                       font=("Consolas", 8), state="disabled", bg="#f8f8f8")
        self._critical_text.pack(padx=5, pady=5)
        ttk.Button(info_f, text="갱신", command=self._refresh_critical_info).pack(pady=3)
        self._refresh_critical_info()

    # ==============================================================
    # [탭 3] 2D 컨투어
    # ==============================================================

    def _build_contour_tab(self, parent):
        """
        2D 컨투어 지도 탭입니다.

        기능:
            - 컴포넌트 선택 / 지표 선택 / 모드 선택 (Temporal ↔ Overall-Max)
            - 현재 시점 컨투어 출력: 시간 슬라이더 위치 기준으로 즉시 출력
            - 컨투어 애니메이션: Play 버튼 연동, 프레임별 Matplotlib 팝업 시퀀스
            - PNG 일괄 저장: 5스텝마다 저장 (Decimation 동기화)
        """
        ctrl = ttk.Frame(parent)
        ctrl.pack(fill="x", padx=8, pady=6)

        # 컴포넌트 선택
        comp_f = ttk.LabelFrame(ctrl, text="  컴포넌트  ")
        comp_f.pack(side="left", fill="y", padx=4)
        comp_list = sorted(list(self.sim.metrics.keys()))
        ttk.Combobox(comp_f, textvariable=self._comp_var, values=comp_list,
                     state="readonly", width=18).pack(padx=5, pady=6)

        # 지표 선택
        metric_f = ttk.LabelFrame(ctrl, text="  지표  ")
        metric_f.pack(side="left", fill="y", padx=4)
        for txt, val in [("Bending",          "bend"),
                          ("Twist",            "twist"),
                          ("RRG",              "rrg"),
                          ("종합 Distortion",  "angle")]:
            ttk.Radiobutton(metric_f, text=txt,
                            variable=self._contour_metric_var, value=val).pack(anchor="w", padx=5, pady=1)

        # 모드 선택
        mode_f = ttk.LabelFrame(ctrl, text="  모드  ")
        mode_f.pack(side="left", fill="y", padx=4)
        ttk.Radiobutton(mode_f, text="Temporal (시간별)",
                        variable=self._contour_mode_var, value="temporal").pack(anchor="w", padx=5, pady=2)
        ttk.Radiobutton(mode_f, text="Overall Max (누적 최대값)",
                        variable=self._contour_mode_var, value="overall").pack(anchor="w", padx=5, pady=2)

        # 버튼
        btn_f = ttk.Frame(ctrl)
        btn_f.pack(side="left", fill="y", padx=10)
        ttk.Button(btn_f, text="현재 시점 컨투어 출력",
                   command=self._on_show_contour_frame, width=22).pack(fill="x", pady=3)
        ttk.Button(btn_f, text="컨투어 애니메이션 시작",
                   command=self._on_animate_contour, width=22).pack(fill="x", pady=3)
        ttk.Button(btn_f, text="모든 프레임 PNG 저장",
                   command=self._on_save_contour_frames, width=22).pack(fill="x", pady=3)

        # 안내문
        guide_f = ttk.Frame(ctrl)
        guide_f.pack(side="left", fill="y", padx=10)
        ttk.Label(guide_f,
                  text=(
                      "사용 방법:\n"
                      "  1. 컴포넌트 / 지표 / 모드 선택\n"
                      "  2. 시간 슬라이더로 시점 이동\n"
                      "     (← → 키 또는 마우스 드래그)\n"
                      "  3. '현재 시점 컨투어 출력' 클릭\n"
                      "     → Matplotlib 팝업 출력\n\n"
                      "  애니메이션:\n"
                      "  ▶ Play → 슬라이더 자동 진행\n"
                      "  원하는 시점에 ⏸ Pause 후\n"
                      "  '현재 시점 컨투어 출력' 클릭"
                  ),
                  foreground="gray", font=("NanumGothic", 8), justify="left").pack(padx=5)

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
        self._time_slider.config(to=max(1, total - 1))

    def _current_time_val(self):
        """현재 슬라이더 위치에 해당하는 시뮬레이션 시간(초)을 반환합니다."""
        step = int(self._time_var.get())
        if self.sim.time_history and step < len(self.sim.time_history):
            return self.sim.time_history[step]
        dt = self.sim.model.opt.timestep if self.sim.model else 0.001
        return step * dt

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
            lines.append(f"  PBA Peak")
            lines.append(f"    t   = {ct['pba_peak_time']:.4f} s")
            lines.append(f"    dir = {ct.get('pba_peak_angle', 0):.1f} deg\n")
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
        """
        선택된 데이터 종류, 축, 위치를 기반으로 매트릭스 레이아웃 그래프를 생성합니다.

        레이아웃:
            행 = 데이터 종류 (변위 / 속도 / 가속도 중 선택)
            열 = 축 종류 (X / Y / Z 중 선택)
            각 서브플롯 = 선택한 위치별 시계열 곡선 중첩
        """
        dtype = self._data_type_var.get()
        axes_sel = [ax for ax in ["X", "Y", "Z"] if self._axis_vars[ax].get()]
        locs_sel = [i for i in range(10) if self._loc_vars[i].get()]

        if not axes_sel:
            messagebox.showwarning("선택 오류", "축을 1개 이상 선택하세요.")
            return
        if not locs_sel:
            messagebox.showwarning("선택 오류", "위치를 1개 이상 선택하세요.")
            return

        n_rows = 1
        n_cols = len(axes_sel)
        ax_idx_map = {"X": 0, "Y": 1, "Z": 2}

        fig, axes_grid = plt.subplots(
            n_rows, n_cols,
            figsize=(5 * n_cols, 4 * n_rows),
            squeeze=False
        )
        unit_map = {"displacement": "m", "velocity": "m/s", "acceleration": "m/s²"}
        fig.suptitle(f"기구학 분석  [{dtype.upper()}]", fontsize=13, fontweight="bold")

        t_arr = np.array(self.sim.time_history) if self.sim.time_history else np.array([0.0])
        colors = plt.cm.tab10(np.linspace(0, 1, len(locs_sel)))

        for col_j, ax_name in enumerate(axes_sel):
            ax = axes_grid[0][col_j]
            ax.set_title(f"{dtype} / {ax_name}축", fontsize=9)
            ax.set_xlabel("Time (s)", fontsize=8)
            ax.set_ylabel(f"{ax_name} [{unit_map.get(dtype, '')}]", fontsize=8)
            ax.grid(True, alpha=0.3)
            ai = ax_idx_map[ax_name]

            for ci, loc_i in enumerate(locs_sel):
                label = self.LOCATION_LABELS[loc_i]
                color = colors[ci]
                data_arr = self._get_kinematic_series(dtype, loc_i, ai)
                if data_arr is not None and len(data_arr) > 0:
                    t_plot = t_arr[:len(data_arr)]
                    ax.plot(t_plot, data_arr, label=label, color=color, linewidth=1.0)

            ax.legend(fontsize=6, loc="upper right")

        plt.tight_layout()
        plt.show()

    def _get_kinematic_series(self, dtype: str, loc_idx: int, axis_idx: int):
        """
        위치 인덱스와 데이터 종류, 축 인덱스로 시계열 데이터를 반환합니다.

        Args:
            dtype (str): 'displacement', 'velocity', 'acceleration' 중 하나
            loc_idx (int): 0~7=코너, 8=기하중심, 9=질량중심(CoM)
            axis_idx (int): 0=X, 1=Y, 2=Z

        Returns:
            np.ndarray | None: 해당 시계열 데이터, 없으면 None
        """
        sim = self.sim
        try:
            if loc_idx == 9:
                # 질량 중심 (CoM)
                if dtype == "displacement":
                    arr = [v[axis_idx] for v in sim.cog_pos_hist]
                elif dtype == "velocity":
                    arr = [v[3 + axis_idx] for v in sim.vel_hist]
                else:
                    arr = [v[3 + axis_idx] for v in sim.acc_hist]

            elif loc_idx == 8:
                # 기하 중심 (Geo Center)
                if dtype == "displacement":
                    arr = [v[axis_idx] for v in sim.geo_center_pos_hist]
                elif dtype == "velocity":
                    arr = [v[axis_idx] for v in sim.geo_center_vel_hist]
                else:
                    arr = [v[axis_idx] for v in sim.geo_center_acc_hist]

            else:
                # 8개 코너 (0~7)
                if dtype == "displacement":
                    hist = sim.corner_pos_hist
                    arr = [
                        (frame[loc_idx][axis_idx] if loc_idx < len(frame) else 0.0)
                        for frame in hist
                    ]
                elif dtype == "velocity":
                    hist = sim.corner_vel_hist
                    arr = [
                        (frame[loc_idx][axis_idx] if loc_idx < len(frame) else 0.0)
                        for frame in hist
                    ]
                else:  # acceleration
                    hist = sim.corner_acc_hist
                    arr = [
                        (frame[loc_idx][axis_idx] if loc_idx < len(frame) else 0.0)
                        for frame in hist
                    ]

            return np.array(arr, dtype=float)

        except Exception:
            return None

    # ==============================================================
    # 이벤트 핸들러 - 구조 해석 그래프
    # ==============================================================

    def _on_plot_structural(self):
        """선택한 구조 해석 지표들의 시계열 그래프를 생성합니다."""
        selected = [
            (key, txt)
            for key, (var, txt) in self._struct_metric_vars.items()
            if var.get()
        ]
        if not selected:
            messagebox.showwarning("선택 오류", "지표를 1개 이상 선택하세요.")
            return

        sts = self.sim.structural_time_series
        t_arr = np.array(self.sim.time_history) if self.sim.time_history else np.array([0.0])

        # Temporal 데이터는 5 스텝 Decimation
        n_ts = len(sts.get('pba_magnitude', []))
        dt = self.sim.model.opt.timestep * 5 if self.sim.model else 0.005
        t_ts = np.arange(n_ts) * dt if n_ts > 0 else np.array([0.0])

        n = len(selected)
        fig, axes_list = plt.subplots(n, 1, figsize=(12, 3 * n), squeeze=False)
        fig.suptitle("구조 해석 지표 시계열", fontsize=13, fontweight="bold")

        # 임계 시점 정보
        ct = {}
        if self.sim.result is not None:
            ct = getattr(self.sim.result, 'critical_timestamps', {})

        ct_markers = [
            ('local_peak_time',     'RRG Peak',  'red'),
            ('global_avg_peak_time','AvgPeak',   'orange'),
            ('pba_peak_time',       'PBA Peak',  'purple'),
        ]

        for row_i, (key, label) in enumerate(selected):
            ax = axes_list[row_i][0]
            ax.set_title(label, fontsize=9)
            ax.set_xlabel("Time (s)", fontsize=8)
            ax.grid(True, alpha=0.3)

            if key in ("bend_overall", "twist_overall"):
                # 컴포넌트별 Bending/Twist 최대값 집계
                data_key = 'all_blocks_bend' if key == "bend_overall" else 'all_blocks_twist'
                for comp_name, comp_m in self.sim.metrics.items():
                    block_data = comp_m.get(data_key, {})
                    # 리스트 별 최소 길이로 통일
                    min_len = None
                    for series in block_data.values():
                        if series:
                            min_len = len(series) if min_len is None else min(min_len, len(series))
                    if min_len:
                        max_series = [
                            max((abs(series[si]) for series in block_data.values()
                                 if si < len(series)), default=0.0)
                            for si in range(min_len)
                        ]
                        t_comp = t_arr[:len(max_series)]
                        ax.plot(t_comp, max_series, label=comp_name, linewidth=1.0)
                ax.set_ylabel("Angle (°)", fontsize=8)

            elif key in sts:
                arr = np.array(sts[key])
                if arr.ndim == 2:
                    arr = np.linalg.norm(arr, axis=1)   # 벡터인 경우 크기
                ax.plot(t_ts[:len(arr)], arr, color="steelblue", linewidth=1.2)
                ax.set_ylabel(label, fontsize=8)

            # 임계 시점 수직선
            for ct_key, ct_label, ct_color in ct_markers:
                if ct.get(ct_key) is not None:
                    ax.axvline(ct[ct_key], color=ct_color, linestyle="--",
                               alpha=0.7, linewidth=1.2, label=ct_label)
            ax.legend(fontsize=7)

        plt.tight_layout()
        plt.show()

    # ==============================================================
    # 이벤트 핸들러 - 2D 컨투어
    # ==============================================================

    def _get_contour_grid_at(self, step: int):
        """
        지정한 time step에서 컴포넌트의 2D 그리드 데이터를 반환합니다.

        Args:
            step (int): 전체 time_history 인덱스 (내부에서 Decimation 5 고려 후 변환)

        Returns:
            tuple | None: (i_arr, j_arr, 2D grid array) 또는 None
        """
        comp = self._comp_var.get()
        metric = self._contour_metric_var.get()
        mode = self._contour_mode_var.get()

        if comp not in self.sim.metrics:
            return None

        comp_m = self.sim.metrics[comp]
        metric_key_map = {
            "bend":  "all_blocks_bend",
            "twist": "all_blocks_twist",
            "rrg":   "all_blocks_rrg",
            "angle": "all_blocks_angle",
        }
        data_key = metric_key_map.get(metric, "all_blocks_bend")
        block_data = comp_m.get(data_key, {})
        if not block_data:
            return None

        all_idxs = list(block_data.keys())
        if not all_idxs:
            return None

        max_i = max(idx[0] for idx in all_idxs)
        max_j = max(idx[1] for idx in all_idxs)

        # Decimation: 실제 데이터는 5스텝마다 1개
        dec_step = step // 5

        grid = np.zeros((max_i + 1, max_j + 1))
        for idx, series in block_data.items():
            if not series:
                continue
            if mode == "temporal":
                t_idx = min(dec_step, len(series) - 1)
                val = abs(series[t_idx])
            else:  # overall max
                val = max(abs(v) for v in series)
            grid[idx[0], idx[1]] = val

        return np.arange(max_i + 1), np.arange(max_j + 1), grid

    def _on_show_contour_frame(self):
        """
        현재 슬라이더 시점의 컨투어를 Matplotlib 팝업 창으로 출력합니다.
        시간 슬라이더 위치를 기준으로 해당 시점의 데이터를 추출합니다.
        """
        step = int(self._time_var.get())
        result = self._get_contour_grid_at(step)
        if result is None:
            messagebox.showwarning("데이터 없음",
                                   "선택한 컴포넌트/지표의 데이터가 없습니다.\n"
                                   "컴포넌트와 지표를 확인한 후 다시 시도하세요.")
            return

        i_arr, j_arr, grid = result
        t_val = self._current_time_val()
        mode = self._contour_mode_var.get()
        metric = self._contour_metric_var.get()
        comp = self._comp_var.get()
        mode_lbl = "Temporal" if mode == "temporal" else "Overall Max"

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.set_title(
            f"[{comp}] {metric.upper()} — {mode_lbl}  (t = {t_val:.4f} s)",
            fontsize=11, fontweight="bold"
        )
        JJ, II = np.meshgrid(j_arr, i_arr)
        vmin = 0.0
        vmax = max(float(grid.max()), 0.01)
        cnt = ax.contourf(JJ, II, grid, levels=20, cmap="RdYlBu_r", vmin=vmin, vmax=vmax)
        ax.contour(JJ, II, grid, levels=10, colors="k", linewidths=0.4, alpha=0.4)
        plt.colorbar(cnt, ax=ax, label="Angle (°)", shrink=0.85)
        ax.set_xlabel("j (Y grid)", fontsize=9)
        ax.set_ylabel("i (X grid)", fontsize=9)
        ax.set_aspect("equal")
        plt.tight_layout()
        plt.show()

    def _on_animate_contour(self):
        """
        애니메이션을 시작합니다.
        Play 버튼과 연동되어 슬라이더가 자동으로 진행됩니다.
        원하는 시점에 Pause 후 '현재 시점 컨투어 출력'으로 해당 시점을 확인하세요.
        """
        if self._anim_running:
            messagebox.showinfo("알림", "이미 애니메이션이 실행 중입니다.\n⏸ Pause 버튼으로 정지하세요.")
            return
        self._notebook.select(2)  # 컨투어 탭으로 포커스
        self._on_play()
        messagebox.showinfo(
            "애니메이션 시작됨",
            "▶ 시간 슬라이더가 자동 진행 중입니다.\n\n"
            "  ⏸ Pause 버튼으로 원하는 시점에 정지한 후\n"
            "  '현재 시점 컨투어 출력' 버튼을 클릭하면\n"
            "  해당 시점의 컨투어 그림이 팝업됩니다.\n\n"
            "  ← → 키로 1프레임씩 정밀 이동할 수 있습니다."
        )

    def _on_save_contour_frames(self):
        """
        전체 시간 구간의 컨투어 프레임을 PNG 파일로 일괄 저장합니다.
        5스텝마다 1개씩 저장 (Decimation 동기화).
        저장 경로: {output_dir}/contour_{comp}_{metric}/frame_XXXXX.png
        """
        comp = self._comp_var.get()
        metric = self._contour_metric_var.get()
        save_dir = os.path.join(self.sim.output_dir, f"contour_{comp}_{metric}")
        os.makedirs(save_dir, exist_ok=True)

        # 팝업 없이 파일 저장 (Agg 백엔드)
        _prev_backend = matplotlib.get_backend()
        matplotlib.use("Agg")

        try:
            total = self._total_frames
            saved = 0
            for step in range(0, total, 5):
                result = self._get_contour_grid_at(step)
                if result is None:
                    continue
                i_arr, j_arr, grid = result
                t_val = (self.sim.time_history[step]
                         if step < len(self.sim.time_history)
                         else step * 0.001)

                fig, ax = plt.subplots(1, 1, figsize=(7, 5))
                JJ, II = np.meshgrid(j_arr, i_arr)
                vmin = 0.0
                vmax = max(float(grid.max()), 0.01)
                cnt = ax.contourf(JJ, II, grid, levels=20, cmap="RdYlBu_r",
                                  vmin=vmin, vmax=vmax)
                plt.colorbar(cnt, ax=ax, shrink=0.85)
                ax.set_title(f"[{comp}] {metric.upper()}  t = {t_val:.4f} s", fontsize=10)
                ax.set_aspect("equal")
                plt.tight_layout()
                fname = os.path.join(save_dir, f"frame_{step:05d}.png")
                plt.savefig(fname, dpi=120)
                plt.close(fig)
                saved += 1
        finally:
            matplotlib.use(_prev_backend)

        messagebox.showinfo(
            "저장 완료",
            f"총 {saved}개 프레임을 저장했습니다.\n경로: {save_dir}"
        )

    # ==============================================================
    # 기타 핸들러
    # ==============================================================

    def _on_apply_heatmap(self):
        """MuJoCo 뷰어에 Rank 기반 왜곡 히트맵을 적용합니다."""
        self.sim.apply_rank_distortion_heatmap()
        messagebox.showinfo("완료", "MuJoCo 뷰어에 RdYlBu_r Rank-based 히트맵을 적용했습니다.")

    def on_close(self):
        """UI 종료 시 애니메이션을 정지하고 창을 닫습니다."""
        self._on_stop()
        self.destroy()
