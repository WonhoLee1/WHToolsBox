# -*- coding: utf-8 -*-
"""
[WHTOOLS] Drop Simulator Engine v6.0 - High-Fidelity Physical Integration
MuJoCo 시뮬레이션 메인 루프, 정밀 물리(소성/공기저항) 및 실시간 분석을 담당합니다.
이 모듈은 고성능 JAX 기반 최적화 파이프라인과의 연동을 염두에 두고 설계되었습니다.
"""

import os
import sys
import time
import json
import pickle
import logging
import numpy as np
import mujoco
import mujoco.viewer
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

# [WHTOOLS] 시각화 및 로깅 라이브러리
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich import box

# [WHTOOLS] 패키지 내부 모듈 임포트
from .whts_data import DropSimResult
from .whts_utils import compute_corner_kinematics, calculate_required_aux_masses
from .whts_reporting import (
    compute_structural_step_metrics, 
    finalize_simulation_results, 
    apply_rank_heatmap,
    compute_critical_timestamps,
    compute_batch_structural_metrics
)

# [WHTOOLS] 외부 패키지 임포트
from run_discrete_builder import create_model, get_default_config

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)
logger = logging.getLogger("WHTS_Engine")
console = Console()

class DropSimulator:
    """
    [WHTOOLS] 정밀 물리 로직과 프리미엄 인터페이스가 결합된 낙하 시뮬레이션 엔진입니다.
    
    Attributes:
        config (Dict[str, Any]): 시뮬레이션 설정 사전.
        timestamp (str): 생성 시점의 타임스탬프.
        output_dir (Path): 결과 파일 저장 경로.
        model (mujoco.MjModel): MuJoCo 모델 객체.
        data (mujoco.MjData): MuJoCo 데이터 객체.
        viewer (mujoco.viewer.Handle): 실시간 시각화 뷰어.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        DropSimulator 클래스의 인스턴스를 초기화합니다.
        
        Args:
            config (Optional[Dict[str, Any]]): 사용자 정의 설정. 없을 경우 기본 설정을 로드합니다.
        """
        self.config = get_default_config(config) if config else get_default_config()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 경로 관리 (Pathlib 사용)
        default_dir = f"rds-{self.timestamp}"
        base_dir = self.config.get("output_dir", Path("results") / default_dir)
        self.output_dir = Path(base_dir)
        
        self.model: Optional[mujoco.MjModel] = None
        self.data: Optional[mujoco.MjData] = None
        self.viewer = None
        
        # 상태 변수 및 히스토리 초기화
        self._init_state_variables()
        self._init_histories()
        
        # 제어 플래그 (UI 연동용 확장)
        self.ctrl_paused = True           # 시작 시 정지 상태로 대기 (사용자 요청)
        self.ctrl_reload_request = False
        self.ctrl_quit_request = False
        self.ctrl_open_ui = False
        
        self.ctrl_step_forward_request = False
        self.ctrl_step_backward_request = False
        self.ctrl_reset_request = False   # [WHTOOLS] 처음 상태로 리셋 요청
        self.ctrl_jump_snapshot_idx = -1
        self.ctrl_speed_multiplier = 1.0  # 1.0이 정상 속도
        self.ctrl_export_camera = False   # 카메라 정보 출력 요청
        self.ctrl_reload_only_xml = False # XML 생성을 건너뛰고 기존 파일만 로드할지 여부
        self.ctrl_reload_xml_path = None  # 리로드할 외부 XML 경로
        
        # UI 관련
        self.config_editor = None
        self.result = None
        self._tk_root = None 

        # 자동 밸런싱 적용
        if self.config.get("enable_target_balancing", False) or "components_balance" in self.config:
            self.apply_balancing()

    def _init_state_variables(self) -> None:
        """시뮬레이션 내부 상태 추적 변수들을 초기화합니다."""
        self.geom_state_tracker: Dict[int, Dict[str, Any]] = {}
        self.components: Dict[str, Dict[Tuple[int, int, int], int]] = {}
        self.metrics: Dict[str, Any] = {}
        self.neighbor_map: Dict[str, Dict[Tuple[int, int, int], List[Tuple[int, int, int]]]] = {}
        self.snapshots: List[Dict[str, Any]] = [] # 되감기용 스냅샷 저장소
        
        # 물리 캐시 및 임시 변수
        self._last_f_drag = 0.0
        self._last_f_sq = 0.0
        self._last_f_visc = 0.0
        self.nominal_local_pos: Dict[int, np.ndarray] = {}
        self.block_half_extents: Dict[int, np.ndarray] = {}
        self.body_index_map: Dict[int, Tuple[int, int, int]] = {}
        
        # 실시간 물리 지표 (최대치 추적)
        self.max_equiv_strain = 0.0
        self.max_applied_pressure_pa = 0.0
        self.max_deformation_mm = 0.0
        self.max_plastic_strain = 0.0
        self._last_reported_interval = -1
        self._report_count = 0
        self.start_real_time = 0.0
        
        # [WHTOOLS] 인터랙티브 레코딩 및 특수 효과
        self.is_recording = False    # 무한 모드에서의 데이터 누적 여부
        self.ctrl_slow_motion = False # 슬로우 모션 토글 상태
        self.step_idx = 0            # 현재 스텝 인덱스 (UI 동기화용)

    def _init_histories(self) -> None:
        """시뮬레이션 데이터 저장을 위한 히스토리 리스트를 초기화합니다."""
        self.time_history: List[float] = []
        self.z_hist: List[float] = []
        self.pos_hist: List[np.ndarray] = []
        self.vel_hist: List[np.ndarray] = []
        self.acc_hist: List[np.ndarray] = []
        self.quat_hist: List[np.ndarray] = []
        
        self.cog_pos_hist: List[np.ndarray] = []
        self.geo_center_pos_hist: List[np.ndarray] = []
        
        self.corner_pos_hist: List[List[np.ndarray]] = []
        self.corner_vel_hist: List[List[np.ndarray]] = []
        self.corner_acc_hist: List[List[np.ndarray]] = []
        
        self.ground_impact_hist: List[float] = []
        self.air_drag_hist: List[float] = []
        self.air_squeeze_hist: List[float] = []
        
        self.structural_time_series = {
            'rrg_max': [], 
            'mean_distortion': [], 
            'comp_global_metrics': {}
        }

    def log(self, text: str, level: str = "info") -> None:
        """
        전문적인 로그를 출력합니다.
        
        Args:
            text (str): 로그 메시지.
            level (str): 로그 레벨 ("info", "warning", "error", "debug").
        """
        if level == "info": logger.info(text)
        elif level == "warning": logger.warning(f"[bold yellow]{text}[/bold yellow]")
        elif level == "error": logger.error(f"[bold red]{text}[/bold red]")
        elif level == "debug": logger.debug(text)

    def setup(self) -> None:
        """
        시뮬레이션 환경을 설정합니다. 모델 XML 생성, MuJoCo 객체 초기화, 
        컴포넌트 식별 및 물리 콜백 등록을 포함합니다.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        xml_path = self.output_dir / "simulation_model.xml"
        
        try:
            # [WHTOOLS] 모델 생성 또는 외부 XML 로드
            current_xml_path = Path(self.ctrl_reload_xml_path) if self.ctrl_reload_xml_path else xml_path
            
            if not self.ctrl_reload_only_xml or not current_xml_path.exists():
                self.log("🛠️ Generating fresh simulation model from config...")
                xml_content, *_ = create_model(str(xml_path), config=self.config)
                self.model = mujoco.MjModel.from_xml_string(xml_content)
                current_xml_path = xml_path # 생성된 파일 경로로 고정
            else:
                self.log(f"📄 Loading specified XML: {current_xml_path}")
                self.model = mujoco.MjModel.from_xml_path(str(current_xml_path))
            
            self.data = mujoco.MjData(self.model)
            
            # 리로드 완료 후 플래그 리셋
            self.ctrl_reload_only_xml = False
            self.ctrl_reload_xml_path = None
            
            # Root Body (Chassis) 식별 로직
            self._identify_root_body()
            
            # 기하 원본 데이터 저장 (소성 변형 및 시각화 기준)
            self.original_geom_size = self.model.geom_size.copy()
            self.original_geom_rgba = self.model.geom_rgba.copy()
            
            # 구성 요소 탐색 및 추적 초기화
            self._discover_components()
            self._init_tracking_containers()
            self._init_plasticity_tracker()
            
            # [CRITICAL] 물리 제어 콜백 등록
            self._mjcb_control = lambda m, d: self._physics_control_callback(m, d)
            mujoco.set_mjcb_control(self._mjcb_control)
            
            self.start_real_time = time.time()
            self.log(f"📦 Assembly: {len(self.components)} components, {self.model.nbody} bodies identified.")
            self.log(f"🚀 Simulation Ready. Timestep: {self.model.opt.timestep:.6f}s")
            
        except Exception as e:
            self.log(f"Failed to setup simulation: {e}", level="error")
            raise

    def _identify_root_body(self) -> None:
        """모델 내에서 'chassis'를 포함하는 루트 바디를 식별합니다."""
        self.root_id = -1
        candidates = []
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and "chassis" in name.lower():
                candidates.append((len(name), i, name))
        
        if candidates:
            candidates.sort() # 이름이 가장 짧은 것을 우선 선택
            self.root_id = candidates[0][1]
            self.log(f"📍 Root Body: '{candidates[0][2]}' (ID: {self.root_id})")
        else:
            self.root_id = 0
            self.log("⚠️ Chassis body not found. Defaulting to WorldBody (ID: 0).", level="warning")

    def _discover_components(self) -> None:
        """모델의 바디 이름을 분석하여 컴포넌트(Paper, Cushion 등) 그룹을 생성합니다."""
        self.components = {}
        target_prefixes = ['paper', 'cushion', 'chassis', 'opencell', 'InertiaAux', 'AutoBalance']
        
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if not name: continue
            
            for prefix in target_prefixes:
                if prefix.lower() in name.lower():
                    comp_key = prefix.lower()
                    if comp_key not in self.components: 
                        self.components[comp_key] = {}
                    
                    # 인덱스 추출 (name_x_y_z 포맷 가정)
                    try:
                        parts = name.split('_')
                        idx = (int(parts[-3]), int(parts[-2]), int(parts[-1])) if len(parts) >= 4 else (0,0,0)
                    except (ValueError, IndexError):
                        idx = (0, 0, 0)
                        
                    self.components[comp_key][idx] = i
                    self.body_index_map[i] = idx
                    self.nominal_local_pos[i] = self.model.body_pos[i].copy()
                    
                    # 기하 정보 저장
                    if self.model.body_geomnum[i] > 0:
                        g_id = self.model.body_geomadr[i]
                        self.block_half_extents[i] = self.model.geom_size[g_id].copy()
                    break

    def _init_tracking_containers(self) -> None:
        """구조적 메트릭(왜곡, 굽힘 등) 추적을 위한 컨테이너를 생성합니다."""
        self.metrics = {}
        for c_name, blocks in self.components.items():
            self.metrics[c_name] = {
                'block_nominal_mats': {idx: None for idx in blocks},
                'all_blocks_bend': {idx: [] for idx in blocks},
                'all_blocks_twist': {idx: [] for idx in blocks},
                'all_blocks_angle': {idx: [] for idx in blocks},
                'all_blocks_rrg': {idx: [] for idx in blocks},
                'all_blocks_s_bend': {idx: [] for idx in blocks},
                'total_distortion': []
            }
        
        # 인접 블록 맵 (Neighbor Map) 생성 - 격자 구조 기반
        self.neighbor_map = {}
        for comp_name, blocks in self.components.items():
            self.neighbor_map[comp_name] = {}
            for idx in blocks:
                self.neighbor_map[comp_name][idx] = [
                    o for o in blocks if np.sum(np.abs(np.array(idx) - np.array(o))) == 1
                ]

    def _init_plasticity_tracker(self) -> None:
        """소성 변형이 허용된 Geoms(Cushion Edge 등)을 식별하고 초기 상태를 설정합니다."""
        for gi in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gi)
            if name and "cushion" in name.lower() and "_edge" in name.lower():
                self.geom_state_tracker[gi] = {
                    'is_plastic': True,
                    'yield_st': self.config.get('cush_yield_strain', 0.05),
                    'base_rgba': self.model.geom_rgba[gi].copy(),
                    'plastic_rgba': [1.0, 1.0, 0.0, 1.0] # 소성 강조색: Yellow
                }
                # 초기 시각적 강조 적용
                self.model.geom_rgba[gi] = [1.0, 1.0, 0.0, 1.0]

    def _physics_control_callback(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """MuJoCo 제어 루프에서 매 스텝 호출되는 물리 콜백 함수입니다."""
        self._apply_aerodynamics(model, data)

    def _apply_aerodynamics(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """공기 저항(Drag) 및 압착 효과(Squeeze Film)를 계산하여 외력으로 적용합니다."""
        if self.root_id == -1: return
        cfg = self.config
        if not cfg.get('enable_air_drag', True) and not cfg.get('enable_air_squeeze', False): 
            return

        rho = cfg.get('air_density', 1.225)
        vel = data.cvel[self.root_id]
        v_linear = vel[3:6]
        v_abs = np.linalg.norm(v_linear)
        
        # 투영 면적 근사 (Box surface area)
        bw, bh, bd = cfg.get('box_w', 2.0), cfg.get('box_h', 1.4), cfg.get('box_d', 0.25)
        total_area = 2 * (bw * bh + bh * bd + bd * bw)

        # 1. Quadratic Drag (고속 영역)
        cd_q = cfg.get('air_drag_coeff', 1.05)
        f_drag = -0.5 * rho * cd_q * total_area * (v_abs**2) * np.sign(v_linear[2]) if cfg.get('enable_air_drag', True) else 0.0
        self._last_f_drag = f_drag

        # 2. Viscous Drag (저속/점성 영역)
        mu = cfg.get('air_viscosity', 1.8e-5)
        cd_v = cfg.get('air_cd_viscous', 0.0)
        f_visc = -1.0 * mu * v_linear[2] * cd_v * total_area if cfg.get('enable_air_drag', True) else 0.0
        self._last_f_visc = f_visc

        # 3. Squeeze Film Effect (지면 근접 시 압축 공기 효과)
        f_sq = 0.0
        if cfg.get('enable_air_squeeze', False):
            z_gap = data.xpos[self.root_id][2] - (bd / 2.0)
            h_max = cfg.get('air_squeeze_hmax', 0.1)
            h_min = cfg.get('air_squeeze_hmin', 0.001)
            if h_min < z_gap < h_max:
                k_sq = cfg.get('air_coef_squeeze', 1.0)
                # Reynolds equation 근사: f ~ (mu * A^2 * v) / h^3
                f_sq = min((k_sq * mu * (total_area**2) * (-v_linear[2])) / (z_gap**3), 2000.0) if v_linear[2] < 0 else 0.0
        self._last_f_sq = f_sq
        
        # 합산된 공기역학적 힘 적용 (Z축)
        data.xfrc_applied[self.root_id][2] = f_drag + f_visc + f_sq

    def _apply_plasticity_v2(self) -> None:
        """
        접촉 압력을 기반으로 한 정밀 소성 변형 로직입니다.
        항복 응력 초과 시 영구적인 치수 감소를 적용합니다.
        """
        if not self.config.get("enable_plasticity", False): return
        d, m = self.data, self.model
        p_ratio = self.config.get("plasticity_ratio", 0.5)
        
        for c_idx in range(d.ncon):
            contact = d.contact[c_idx]
            for g_id in [contact.geom1, contact.geom2]:
                if g_id in self.geom_state_tracker:
                    # 현재 변형률 계산
                    strains = [
                        max(0.0, (self.original_geom_size[g_id][ax] - m.geom_size[g_id][ax]) / self.original_geom_size[g_id][ax]) 
                        for ax in range(3)
                    ]
                    equiv_strain = np.linalg.norm(strains)
                    
                    # 동적 접촉 면적 산출 (법선 벡터 기반)
                    lx = d.geom_xmat[g_id].reshape(3,3).T @ contact.frame[:3]
                    ax = int(np.argmax(np.abs(lx)))
                    other_axes = [i for i in range(3) if i != ax]
                    area = m.geom_size[g_id][other_axes[0]] * m.geom_size[g_id][other_axes[1]] * 4.0
                    
                    # 항복 압력 (경화 효과 포함)
                    yield_pr = self.config.get("cush_yield_pressure", 1000.0) + \
                               (self.config.get("plastic_hardening_modulus", 0.0) * equiv_strain)
                    
                    force = np.zeros(6)
                    mujoco.mj_contactForce(m, d, c_idx, force)
                    pressure = abs(force[0]) / (area + 1e-9)
                    
                    if pressure > yield_pr:
                        # 소성 유동 수식: 변형 속도 ~ 초과 압력
                        excess_mpa = (pressure - yield_pr) / 1e6
                        reduction = excess_mpa * p_ratio * m.opt.timestep * 2.0 
                        
                        m.geom_size[g_id][ax] = max(
                            self.original_geom_size[g_id][ax] * (1.0 - self.config.get("plastic_max_strain", 0.5)), 
                            m.geom_size[g_id][ax] - reduction
                        )
                    
                    # 통계 업데이트
                    self.max_applied_pressure_pa = max(self.max_applied_pressure_pa, pressure)
                    self.max_plastic_strain = max(self.max_plastic_strain, equiv_strain)
                    self.max_equiv_strain = max(self.max_equiv_strain, equiv_strain)
                    for i in range(3):
                        def_mm = (self.original_geom_size[g_id][i] - m.geom_size[g_id][i]) * 1000.0
                        self.max_deformation_mm = max(self.max_deformation_mm, def_mm)
                    
                    # 시각적 피드백: Strain-based Color Mapping (Blue -> Red)
                    sn = np.clip(equiv_strain / self.config.get("plastic_color_limit", 0.1), 0.0, 1.0)
                    m.geom_rgba[g_id] = [sn, 0.4, 1.0 - sn, 1.0]

    def _collect_history(self) -> None:
        """현재 타임스텝의 데이터를 히스토리에 기록합니다."""
        d = self.data; rid = self.root_id
        self.time_history.append(d.time)
        self.z_hist.append(d.xpos[rid, 2])
        self.pos_hist.append(d.xpos.copy())
        self.vel_hist.append(d.cvel[rid].copy())
        self.acc_hist.append(d.cacc[rid].copy())
        
        # 회전 데이터 (Quaternions)
        q_frame = np.zeros((self.model.nbody, 4))
        for i in range(self.model.nbody): 
            q_frame[i] = d.xquat[i].copy()
        self.quat_hist.append(q_frame)
        
        self.cog_pos_hist.append(d.subtree_com[rid].copy())
        
        # 충격력 및 공기저항 기록
        impact_force = np.sum([np.linalg.norm(d.contact[i].frame[:3]) for i in range(d.ncon)])
        self.ground_impact_hist.append(impact_force)
        self.air_drag_hist.append(self._last_f_drag)
        self.air_squeeze_hist.append(self._last_f_sq)
        
        # 코너 기구학 데이터 계산
        bw, bh, bd = self.config.get('box_w', 2.0), self.config.get('box_h', 1.4), self.config.get('box_d', 0.25)
        ck = compute_corner_kinematics(d.xpos[rid], d.xmat[rid].reshape(3,3), d.cvel[rid], d.cacc[rid], bw, bh, bd)
        
        self.corner_pos_hist.append([c['pos'] for c in ck])
        self.corner_vel_hist.append([c['vel'] for c in ck])
        self.corner_acc_hist.append([c['acc'] for c in ck])
        self.geo_center_pos_hist.append(np.mean([c['pos'] for c in ck], axis=0))

    @property
    def tk_root(self):
        """Tkinter 루트 윈도우를 Lazy-Loading 방식으로 생성합니다."""
        if self._tk_root is None:
            import tkinter as tk
            self._tk_root = tk.Tk()
            self._tk_root.withdraw()
        return self._tk_root

    def simulate(self) -> None:
        """
        전체 시뮬레이션 프로세스를 시작합니다.
        설정된 'use_viewer' 값에 따라 인터랙티브 UI 모드 또는 자율 시뮬레이션 모드로 작동합니다.
        """
        use_viewer = self.config.get("use_viewer", False)
        
        if use_viewer:
            self.log("🎨 Launching Interactive Mode with Premium Control Panel...", level="info")
            self.ctrl_paused = True # UI 모드면 정지 상태로 시작
            self._launch_with_control_panel()
            return
        else:
            self.log("🚀 Launching Autonomous Simulation Mode (Headless)...", level="info")
            self.ctrl_paused = False # 자동 모드면 즉시 시작

        try:
            while not self.ctrl_quit_request:
                self.setup()
                self._run_engine()
                if not self.ctrl_reload_request: 
                    break
                self.ctrl_reload_request = False
                self.log("♻️ Reloading Simulation Model...", level="info")
        finally:
            self._wrap_up()

    def _launch_with_control_panel(self) -> None:
        """PySide6 컨트롤 패널과 함께 시뮬레이션을 스레드로 실행합니다."""
        from .whts_control_panel import launch_control_panel
        from PySide6.QtCore import QThread

        class SimThread(QThread):
            def __init__(self, outer):
                super().__init__()
                self.outer = outer
            def run(self):
                try:
                    while not self.outer.ctrl_quit_request:
                        self.outer.setup()
                        self.outer._run_engine()
                        if not self.outer.ctrl_reload_request: break
                        self.outer.ctrl_reload_request = False
                finally:
                    self.outer._wrap_up()

        # UI 실행 (Main Thread)
        self.app, self.panel = launch_control_panel(self)
        
        # 시뮬레이션 실행 (Sub Thread)
        self.sim_thread = SimThread(self)
        self.sim_thread.start()
        
        # UI 루프 진입 (Blocking)
        self.app.exec()
        
        # UI 종료 시 시뮬레이션 종료 유도
        self.ctrl_quit_request = True
        self.sim_thread.wait()

    def _run_engine(self) -> None:
        """MuJoCo 뷰어 런칭 및 메인 루프 실행을 관리합니다."""
        if self.config.get("use_viewer", True):
            with mujoco.viewer.launch_passive(
                self.model, self.data, 
                key_callback=self._on_key,
                show_left_ui=False, 
                show_right_ui=False
            ) as viewer:
                self.viewer = viewer
                
                # [WHTOOLS] 초기 카메라 시점 설정 (사용자 요청 뷰)
                viewer.cam.lookat[:] = [-0.0295, -0.3909, 0.8668]
                viewer.cam.distance = 5.8341
                viewer.cam.elevation = -9.88
                viewer.cam.azimuth = 136.50
                
                self._main_loop()
        else:
            self._main_loop()

    def _main_loop(self) -> None:
        """실제 시뮬레이션 타임스텝을 진행하는 핵심 루프입니다."""
        self.step_idx = 0
        total_steps = int(self.config.get("sim_duration", 1.0) / self.model.opt.timestep)
        report_step = max(1, int(self.config.get("reporting_interval", 0.005) / self.model.opt.timestep))
        
        # [WHTOOLS] 시뮬레이션 시작 직전의 초기 상태(Frame 0)를 강제 저장
        self.snapshots = [] 
        mujoco.mj_forward(self.model, self.data) # 초기 상태 확정
        self._save_snapshot()
        
        self.log(f"🎬 Simulation Loop Started. Target Duration: {self.config.get('sim_duration', 1.0)}s")
        
        # 뷰어가 켜져 있는 동안 또는 뷰어가 없으면 타겟 스텝까지 무한 루프
        while not self.ctrl_quit_request and not self.ctrl_reload_request:
            # 뷰어 종료 감지
            if self.viewer and not self.viewer.is_running():
                break
                
            if self._tk_root: self._tk_root.update()
            
            # [WHTOOLS] UI 요청 처리 (Step, Reset, Jump 등)
            self._handle_ui_requests()

            if not self.ctrl_paused:
                # [WHTOOLS] 과거 시점에서 다시 시작하려 할 경우, 미래 데이터 절단
                self._check_and_truncate_future()
                
                # 1. Physics Step
                mujoco.mj_step(self.model, self.data)
                
                # 2. Advanced Physics Post-step
                self._apply_plasticity_v2()
                
                # 3. Data Collection & Snapshotting
                if self.step_idx % report_step == 0:
                    # [WHTOOLS] [FIX] 목표 시간(total_steps) 이내에서만 엄격하게 데이터 기록
                    if self.step_idx <= total_steps:
                        self._collect_history()
                    
                    # 되감기용 스냅샷 저장 (항시 가능하도록 타겟 시간 이후에도 저장)
                    self._save_snapshot()
                
                # 4. Progress Reporting
                self._report_progress(self.step_idx)
                
                # 타겟 도달 시 알림 (한 번만)
                if self.step_idx == total_steps:
                    self.log("✅ [DATA COLLECTION COMPLETE] Target simulation time reached.", level="info")
                    self.log(f"📊 Collected {len(self.pos_hist)} frames up to {self.data.time:.3f}s", level="info")
                    self.log("💡 [Tip] Simulation continues in interactive mode. Press 'L' to record more data.", level="debug")
                
                self.step_idx += 1
                if self.viewer: self.viewer.sync()
                
                # 속도 제어 (Speed Multiplier 및 Slow Motion 적용)
                effective_multiplier = self.ctrl_speed_multiplier
                if self.ctrl_slow_motion:
                    effective_multiplier *= 0.2 # 5배 느리게
                
                if effective_multiplier != 1.0:
                    base_sleep = self.model.opt.timestep / effective_multiplier
                    if base_sleep > 0.0001:
                        time.sleep(base_sleep)
            else:
                # 일시 정지 시에도 리포트 종료 처리
                if self._report_count > 0:
                    self._print_border()
                    self._report_count = -1
                if self.viewer: self.viewer.sync()
                time.sleep(0.01)

    def _handle_ui_requests(self) -> None:
        """UI로부터 전달된 제어 요청(Step, Jump 등)을 처리합니다."""
        # 1. 1프레임 전진 요청
        if self.ctrl_step_forward_request:
            self._check_and_truncate_future()
            mujoco.mj_step(self.model, self.data)
            self._apply_plasticity_v2()
            self.step_idx += 1
            if self.viewer: self.viewer.sync()
            self.ctrl_step_forward_request = False
            self.log(f"▶️ Stepped Forward (Step: {self.step_idx})")

        # 2. 1프레임 후진 요청 (스냅샷 이용)
        if self.ctrl_step_backward_request:
            self._rewind_snapshot()
            self.ctrl_step_backward_request = False

        # 2-1. 전체 리셋 요청 (Frame 0으로 이동)
        if self.ctrl_reset_request:
            if len(self.snapshots) > 0:
                self._jump_to_snapshot(0)
                self.ctrl_paused = True # 리셋 후 정지 상태 유지
                self.log("🔄 Simulation Reset Success. Press 'Resume' to start over.")
            else:
                self.log("⚠️ No initial snapshot found. Reset failed.", level="warning")
            self.ctrl_reset_request = False

        # 3. 특정 스냅샷으로 점프
        if self.ctrl_jump_snapshot_idx != -1:
            self._jump_to_snapshot(self.ctrl_jump_snapshot_idx)
            self.ctrl_jump_snapshot_idx = -1

        # 4. 카메라 XML 정보 출력
        if self.ctrl_export_camera:
            self._export_camera_xml()
            self.ctrl_export_camera = False

    def _export_camera_xml(self) -> None:
        """
        현재 MuJoCo 뷰어의 카메라 파라미터(lookat, distance, elevation, azimuth)를 
        XML 포맷으로 추출하여 콘솔에 출력하고 파일로 저장합니다.
        추후 모델 XML에 해당 카메라 설정을 복사하여 동일한 뷰를 재현할 수 있습니다.
        """
        if not self.viewer:
            self.log("⚠️ Viewer is not active. Cannot export camera.", level="warning")
            return
            
        cam = self.viewer.cam
        pos = cam.lookat
        dist = cam.distance
        elev = cam.elevation
        azim = cam.azimuth
        
        # [WHTOOLS] 이 값들은 XML <camera> 태그의 속성이 아니라, 
        # whts_engine.py의 viewer.cam 속성에 직접 할당해야 하는 값들입니다.
        msg = (f"\n📸 [Camera Export]\n"
               f"- LookAt: {pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}\n"
               f"- Distance: {dist:.4f}\n"
               f"- Elevation: {elev:.4f}\n"
               f"- Azimuth: {azim:.4f}\n"
               f"----------------------------------------\n"
               f"위 값을 whts_engine.py의 초기 카메라 설정 부분에 업데이트하십시오.")
        self.log(msg)
        
        # 보조 정보 저장용 포맷
        info_str = f"lookat='{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}' distance='{dist:.4f}' elevation='{elev:.4f}' azimuth='{azim:.4f}'"
        
        # 파일로도 저장
        cam_file = self.output_dir / "camera_config.txt"
        with open(cam_file, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {info_str}\n")
        self.log(f"📄 Camera config appended to: {cam_file}")

    def reload_xml(self, xml_path: Optional[str] = None) -> None:
        """
        특정 XML 파일을 로드하여 시뮬레이션을 재시작합니다.
        경로가 제공되지 않으면 파일 선택 창을 띄웁니다.
        """
        if xml_path is None:
            from tkinter import filedialog
            # Tkinter 루트를 통해 파일 선택 다이얼로그 실행
            selected = filedialog.askopenfilename(
                title="Select MuJoCo Simulation XML",
                filetypes=[("MuJoCo XML", "*.xml"), ("All files", "*.*")],
                initialdir=str(self.output_dir)
            )
            if not selected:
                self.log("🚫 Reload cancelled: No file selected.")
                return
            xml_path = selected

        self.log(f"♻️ Reloading from: {xml_path}")
        self.ctrl_reload_xml_path = xml_path
        self.ctrl_reload_only_xml = True
        self.ctrl_reload_request = True
        if self.viewer:
            self.viewer.close()

    def _jump_to_snapshot(self, idx: int) -> None:
        """
        저장된 스냅샷 리스트에서 특정 인덱스의 상태로 시뮬레이션을 되돌립니다.
        
        Args:
            idx (int): 이동할 스냅샷의 인덱스.
        
        Note:
            선택된 스냅샷 이후의 모든 히스토리 데이터는 Truncate 되어 삭제됩니다.
            이는 시뮬레이션의 인과관계를 유지하기 위함입니다.
        """
        if 0 <= idx < len(self.snapshots):
            snapshot = self.snapshots[idx]
            mujoco.mj_setState(self.model, self.data, snapshot['state'], mujoco.mjtState.mjSTATE_PHYSICS)
            self.step_idx = snapshot['step_idx']
            
            # 2. 모델 파라미터 (소성 변형) 복구
            if 'geom_size' in snapshot:
                self.model.geom_size[:] = snapshot['geom_size']
            if 'geom_rgba' in snapshot:
                self.model.geom_rgba[:] = snapshot['geom_rgba']
            
            # 3. 통계 데이터 초기화 (0번으로 돌아갈 때만 완전 초기화)
            if idx == 0:
                self.max_equiv_strain = 0.0
                self.max_applied_pressure_pa = 0.0
                self.max_plastic_strain = 0.0
                self.max_deformation_mm = 0.0
                self._last_reported_interval = -1

            # [WHTOOLS] 점프 시 즉시 Truncate 하지 않음 (비파괴적 탐색 지원)
            # self._truncate_histories(snapshot['hist_len'])
            # self.snapshots = self.snapshots[:idx+1]
            
            self.log(f"🚀 Jumped to Snapshot {idx} (Time: {snapshot['time']:.3f}s)")
            if self.viewer: self.viewer.sync()

    def _check_and_truncate_future(self) -> None:
        """현재 step_idx가 저장된 최신 스냅샷보다 과거라면 미래 데이터를 삭제합니다."""
        if len(self.snapshots) > 0:
            last_snap = self.snapshots[-1]
            if self.step_idx < last_snap['step_idx']:
                # 현재 step_idx에 가장 인접한 스냅샷 찾기 및 절단
                for i, snap in enumerate(self.snapshots):
                    if snap['step_idx'] == self.step_idx:
                        self._truncate_histories(snap['hist_len'])
                        self.snapshots = self.snapshots[:i+1]
                        self.log(f"✂️ Future truncated from step {self.step_idx} to maintain causality.")
                        break

    def _truncate_histories(self, h_idx: int) -> None:
        """모든 히스토리 데이터를 지정된 인덱스까지 잘라냅니다."""
        self.time_history = self.time_history[:h_idx]
        self.z_hist = self.z_hist[:h_idx]
        self.pos_hist = self.pos_hist[:h_idx]
        self.vel_hist = self.vel_hist[:h_idx]
        self.acc_hist = self.acc_hist[:h_idx]
        self.quat_hist = self.quat_hist[:h_idx]
        self.cog_pos_hist = self.cog_pos_hist[:h_idx]
        self.ground_impact_hist = self.ground_impact_hist[:h_idx]
        self.air_drag_hist = self.air_drag_hist[:h_idx]
        self.air_squeeze_hist = self.air_squeeze_hist[:h_idx]
        self.corner_pos_hist = self.corner_pos_hist[:h_idx]
        self.corner_vel_hist = self.corner_vel_hist[:h_idx]
        self.corner_acc_hist = self.corner_acc_hist[:h_idx]
        self.geo_center_pos_hist = self.geo_center_pos_hist[:h_idx]
        
        # 구조적 시계열 데이터 초기화 (필요 시 더 정밀하게 구현 가능)
        if hasattr(self, 'structural_time_series'):
            for k in ['rrg_max', 'mean_distortion']:
                if k in self.structural_time_series:
                    self.structural_time_series[k] = self.structural_time_series[k][:h_idx]

    def _save_snapshot(self) -> None:
        """
        현재의 MuJoCo 물리 상태(mjSTATE_PHYSICS)와 히스토리 포인터를 스냅샷으로 저장합니다.
        메모리 관리를 위해 최대 500개까지만 유지하며, 초과 시 가장 오래된 것부터 삭제합니다.
        """
        # 메모리 효율을 위해 최대 500개까지만 저장 (단, 0번 스냅샷은 리셋을 위해 절대 삭제하지 않음)
        if len(self.snapshots) > 500:
            self.snapshots.pop(1) 
            
        state = np.zeros(mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_PHYSICS))
        mujoco.mj_getState(self.model, self.data, state, mujoco.mjtState.mjSTATE_PHYSICS)
        
        self.snapshots.append({
            'time': self.data.time,
            'step_idx': self.step_idx,
            'state': state,
            'geom_size': self.model.geom_size.copy(),
            'geom_rgba': self.model.geom_rgba.copy(),
            'hist_len': len(self.time_history)
        })

    def _rewind_snapshot(self) -> None:
        """
        가장 최근의 스냅샷으로 시뮬레이션을 1단계 되돌립니다(Undo).
        이 기능은 'Backspace' 키와 연동되어 인터랙티브한 분석을 지원합니다.
        """
        if len(self.snapshots) <= 1:
            self.log("⚠️ No snapshots available to rewind.", level="warning")
            return
            
        # 1. 현재(가장 최신) 시점을 리스트에서 제거 (Moving backward)
        self.snapshots.pop()
        
        # 2. 새로운 '최신' 시점이 될 스냅샷 참조 (리스트에는 유지)
        snapshot = self.snapshots[-1]
        
        # 3. MuJoCo 물리 상태 복구
        mujoco.mj_setState(self.model, self.data, snapshot['state'], mujoco.mjtState.mjSTATE_PHYSICS)
        self.step_idx = snapshot['step_idx']
        
        # 4. 모델 파라미터 (소성 변형) 복구
        if 'geom_size' in snapshot:
            self.model.geom_size[:] = snapshot['geom_size']
        if 'geom_rgba' in snapshot:
            self.model.geom_rgba[:] = snapshot['geom_rgba']
        
        # 2. 히스토리 데이터 잘라내기 (Truncate)
        self._truncate_histories(snapshot['hist_len'])
        
        # 3. 리포트 인터벌 리셋
        self._last_reported_interval = int(self.data.time / 0.05) - 1
        
        self.log(f"⏪ Rewound to Time: {self.data.time:.3f}s (Step: {self.step_idx})")
        if self.viewer: self.viewer.sync()

    def _report_progress(self, step_idx: int) -> None:
        """터미널에 시뮬레이션 진행 상황과 주요 물리 지표를 출력합니다."""
        interval = int(self.data.time / 0.05)
        if interval > self._last_reported_interval:
            self._last_reported_interval = interval
            real_elapsed = time.time() - self.start_real_time
            fps = step_idx / real_elapsed if real_elapsed > 0 else 0
            
            if self._report_count <= 0:
                self._print_header()
                self._report_count = 0

            rec_status = "[bold red]● REC[/bold red]" if self.is_recording else "[dim]STANDBY[/dim]"
            slow_status = "[yellow]SLOW[/yellow]" if self.ctrl_slow_motion else "NORM"
            
            status = (f"SE:{self.max_equiv_strain:5.1%}, "
                      f"PRS:{self.max_applied_pressure_pa/1e6:6.3f}(MPa), "
                      f"PE:{self.max_plastic_strain:5.1%}, "
                      f"DF:{self.max_deformation_mm:4.1f}mm")
            
            row_str = f"   {step_idx:<9d}  {self.data.time:<11.3f}   {real_elapsed:<11.2f}   {fps:<10.1f}   {rec_status} | {slow_status} | {status}"
            console.print(row_str)
            self._report_count += 1

    def _print_header(self) -> None:
        """리포트 헤더를 출력합니다."""
        header_str = "   🔢 Step     ⏱️ Time       🚀 Real       ⚡ FPS      🔴 Rec | 🐌 Mode | 🗜️ Status (SE, PRS, PE, DF)"
        self._print_border()
        console.print(f"[bold green]{header_str}[/bold green]")
        self._print_border()

    def _print_border(self) -> None:
        """리포트 구분선을 출력합니다."""
        console.print(f"[bold white]{'━' * 100}[/bold white]")

    def _wrap_up(self) -> None:
        """시뮬레이션 종료 후 데이터를 정리하고 결과를 저장하며 UI를 호출합니다."""
        self.log("🏁 Simulation Finished. Wrapping up data...", level="info")
        
        # 데이터 수집 완결성 체크
        target_time = self.config.get("sim_duration", 1.0)
        curr_time = self.data.time
        is_complete = curr_time >= (target_time - 1e-5)
        
        status_msg = "COMPLETE ✅" if is_complete else f"INCOMPLETE ⚠️ ({curr_time:.3f}/{target_time:.3f}s)"
        self.log(f"📑 Data Status: {status_msg}")
        self.log(f"🎞️ Total Frames: {len(self.pos_hist)}")
        self._print_border()
        
        try:
            compute_batch_structural_metrics(self)
            finalize_simulation_results(self)
            apply_rank_heatmap(self)
            
            self.result = DropSimResult(
                config=self.config.copy(), 
                metrics=self.metrics.copy(),
                max_g_force=float(np.max(np.abs(self.acc_hist))/9.81) if self.acc_hist else 0.0,
                time_history=self.time_history, 
                z_hist=self.z_hist, 
                root_acc_history=[],
                corner_acc_hist=self.corner_acc_hist, 
                pos_hist=self.pos_hist,
                vel_hist=self.vel_hist, 
                acc_hist=self.acc_hist,
                cog_pos_hist=self.cog_pos_hist, 
                geo_center_pos_hist=self.geo_center_pos_hist,
                corner_pos_hist=self.corner_pos_hist, 
                ground_impact_hist=self.ground_impact_hist,
                air_drag_hist=self.air_drag_hist, 
                air_squeeze_hist=self.air_squeeze_hist,
                structural_metrics=self.structural_time_series, 
                critical_timestamps={},
                nominal_local_pos=self.nominal_local_pos, 
                quat_hist=self.quat_hist,
                components=self.components.copy(), 
                body_index_map=self.body_index_map, 
                block_half_extents=self.block_half_extents
            )
            
            result_path = self.output_dir / "simulation_result.pkl"
            self.result.save(str(result_path))
            self.log(f"💾 Results saved to: {result_path}")

            # 후처리 UI 자동 실행
            self._launch_postprocess()
            
        except Exception as e:
            self.log(f"Error during wrap-up: {e}", level="error")

    def _launch_postprocess(self) -> None:
        """설정에 따라 적절한 후처리 UI를 실행합니다."""
        if self.config.get("use_postprocess_v2", False):
            self.log(">> [Integrated UI] Launching V2 Control Center...")
            launch_v2_subprocess(self)
        elif self.ctrl_open_ui or self.config.get("use_postprocess_ui", True):
            self.log(">> [Legacy UI] Launching Tkinter Post-Processing...")
            from .whts_postprocess_ui import PostProcessingUI
            ui = PostProcessingUI(self, master=self.tk_root)
            ui.on_simulation_complete()
            self.tk_root.mainloop()

    def apply_balancing(self) -> None:
        """타겟 질량 및 관성을 맞추기 위한 보조 질량을 계산하여 설정에 적용합니다."""
        self.config["chassis_aux_masses"] = calculate_required_aux_masses(
            self.config, 
            self.config.get("target_mass"), 
            self.config.get("target_cog"), 
            self.config.get("target_moi")
        )

    def _on_key(self, keycode: int) -> None:
        """MuJoCo 뷰어에서의 키보드 입력을 처리합니다."""
        if keycode == 32: # Space: Pause
            self.ctrl_paused = not self.ctrl_paused
            state = "Paused" if self.ctrl_paused else "Resumed"
            self.log(f"⏸️ Simulation {state}")
        elif keycode == 8 or keycode == 259: # Backspace: Rewind
            self.ctrl_step_backward_request = True
        elif keycode == 82: # 'R': Reset to Start
            self.ctrl_jump_snapshot_idx = 0
        elif keycode == 83: # 'S': Toggle Slow Motion
            self.ctrl_slow_motion = not self.ctrl_slow_motion
            status = "ON" if self.ctrl_slow_motion else "OFF"
            self.log(f"🐌 Slow Motion: {status}")
        elif keycode == 76: # 'L': Toggle Recording
            self.is_recording = not self.is_recording
            status = "STARTED" if self.is_recording else "STOPPED"
            self.log(f"⏺️ History Recording: {status}")
        elif keycode == 67: # 'C': Export Camera XML
            self.ctrl_export_camera = True
            self.log("📷 Camera XML Export Queued.")
        elif keycode == 88: # 'X': Reload Modified XML
            self.reload_xml()
        elif keycode == 75: # 'K': Open Config UI
            self.ctrl_open_ui = True
        elif keycode == 256: # ESC: Quit
            self.ctrl_quit_request = True
            self.log("🛑 Quit Request Received.")

    def _reset_simulation(self) -> None:
        """시뮬레이션을 초기 상태(스냅샷 0)로 리셋하며 모든 히스토리를 초기화합니다."""
        if not self.snapshots:
            self.log("⚠️ No snapshots to reset.", level="warning")
            return
        
        # 첫 번째 스냅샷으로 점프
        self._jump_to_snapshot(0)
        
        # [WHTOOLS] [FIX] 히스토리 및 스냅샷 전체 초기화
        self._init_histories()
        self.snapshots = self.snapshots[:1] # 첫 스냅샷만 유지
        self.step_idx = 0
        
        self.log("♻️ Simulation Reset to Initial State. History cleared.")

def launch_v2_subprocess(sim: DropSimulator) -> None:
    """별도 프로세스로 V2 후처리 UI를 실행합니다."""
    try:
        import subprocess
        curr_dir = Path(__file__).parent.absolute()
        script_path = curr_dir / "whts_postprocess_ui_v2.py"
        result_path = sim.output_dir / "simulation_result.pkl"
        
        subprocess.Popen([
            sys.executable, 
            str(script_path), 
            "--load", 
            str(result_path)
        ])
    except Exception as e:
        logger.error(f"Failed to launch V2 UI subprocess: {e}")

if __name__ == "__main__":
    # 간단한 테스트 실행 예시
    simulator = DropSimulator()
    simulator.simulate()
