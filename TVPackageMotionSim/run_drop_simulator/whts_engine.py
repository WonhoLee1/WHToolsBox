# -*- coding: utf-8 -*-
"""
[WHTOOLS] Drop Simulator Engine v5.7 - Premium Physical Integration
MuJoCo 시뮬레이션 메인 루프, 정밀 물리(소성/공기저항) 및 실시간 분석을 담당합니다.
"""

import os
import sys
import time
import json
import pickle
import numpy as np
import mujoco
import mujoco.viewer
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from rich.console import Console
from rich.table import Table
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

class DropSimulator:
    """
    [WHTOOLS] 정밀 물리 로직과 프리미엄 인터페이스가 결합된 낙하 시뮬레이션 엔진입니다.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = get_default_config(config) if config else get_default_config()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.config.get("output_dir", f"rds-{self.timestamp}")
        
        self.model = None
        self.data = None
        self.viewer = None
        
        self._init_state_variables()
        self._init_histories()
        
        # 제어 플래그
        self.ctrl_paused = False
        self.ctrl_reload_request = False
        self.ctrl_quit_request = False
        self.ctrl_open_ui = False
        self.ctrl_step_backward_count = 0
        
        # UI 및 결과
        self.config_editor = None
        self.result = None
        self._tk_root = None 

        # 자동 밸런싱 적용
        if self.config.get("enable_target_balancing", False) or "components_balance" in self.config:
            self.apply_balancing()

    def _init_state_variables(self) -> None:
        self.geom_state_tracker = {}
        self.components = {}
        self.metrics = {}
        self.neighbor_map = {}
        self.qpos_hist = []
        self.qvel_hist = []
        self._last_f_drag = 0.0
        self._last_f_sq = 0.0
        self._last_f_visc = 0.0
        self.nominal_local_pos = {}
        self.quat_hist = []
        self.block_half_extents = {}
        self.body_index_map = {}
        
        # 물리 실시간 지표
        self.max_equiv_strain = 0.0
        self.max_applied_pressure_pa = 0.0
        self.max_deformation_mm = 0.0
        self.max_plastic_strain = 0.0
        self._last_reported_interval = -1
        self._report_count = 0

    def _init_histories(self) -> None:
        self.time_history = []
        self.z_hist = []
        self.pos_hist = []
        self.vel_hist = []
        self.acc_hist = []
        self.cog_pos_hist = []
        self.geo_center_pos_hist = []
        self.corner_pos_hist = []
        self.corner_vel_hist = []
        self.corner_acc_hist = []
        self.ground_impact_hist = []
        self.air_drag_hist = []
        self.air_squeeze_hist = []
        self.structural_time_series = {
            'rrg_max': [], 'mean_distortion': [], 'comp_global_metrics': {}
        }

    def log(self, text: str) -> None:
        entry = f"[{datetime.now().strftime('%H:%M:%S')}] {text}"
        print(entry)

    def setup(self) -> None:
        """환경 및 모델 구성 (물리 콜백 및 원본 데이터 캡처 포함)"""
        os.makedirs(self.output_dir, exist_ok=True)
        xml_path = os.path.join(self.output_dir, "simulation_model.xml")
        
        # [WHTOOLS] 리턴값: xml_content, mass, cog, moi, details
        xml_content, *_ = create_model(xml_path, config=self.config)
        
        self.model = mujoco.MjModel.from_xml_string(xml_content)
        self.data = mujoco.MjData(self.model)
        
        # 메타데이터 추출
        self.root_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        if self.root_id == -1: self.root_id = 0
        
        # 기하 원본 데이터 저장 (소성 변형 계산용)
        self.original_geom_size = self.model.geom_size.copy()
        self.original_geom_rgba = self.model.geom_rgba.copy()
        
        # 컴포넌트 및 추적기 초기화
        self._discover_components()
        self._init_tracking_containers()
        self._init_plasticity_tracker()
        
        # [CRITICAL] 공기 저항 콜백 등록
        self._mjcb_control = lambda m, d: self._aerodynamics_callback(m, d)
        mujoco.set_mjcb_control(self._mjcb_control)
        
        self.start_real_time = time.time()
        self.log(f"📦 Assembly Info: {len(self.components)} components identified: {list(self.components.keys())}. Physics callback registered.")

    def _discover_components(self) -> None:
        self.components = {}
        target_prefixes = ['paper', 'cushion', 'chassis', 'opencell', 'InertiaAux', 'AutoBalance']
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if not name: continue
            for prefix in target_prefixes:
                # [V5.7.5] b_ 접두사 처리 및 대소문자 무관 검색 (Builder 명명 규칙 대응)
                if prefix.lower() in name.lower():
                    comp_key = prefix.lower()
                    if comp_key not in self.components: self.components[comp_key] = {}
                    try:
                        parts = name.split('_')
                        idx = (int(parts[-3]), int(parts[-2]), int(parts[-1])) if len(parts) >= 4 else (0,0,0)
                    except: idx = (0, 0, 0)
                    self.components[comp_key][idx] = i
                    self.body_index_map[i] = idx
                    self.nominal_local_pos[i] = self.model.body_pos[i].copy()
                    if self.model.body_geomnum[i] > 0:
                        g_id = self.model.body_geomadr[i]
                        self.block_half_extents[i] = self.model.geom_size[g_id].copy()
                    break

    def _init_tracking_containers(self) -> None:
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
        
        self.neighbor_map = {}
        for comp_name, blocks in self.components.items():
            self.neighbor_map[comp_name] = {}
            for idx in blocks:
                self.neighbor_map[comp_name][idx] = [o for o in blocks if np.sum(np.abs(np.array(idx)-np.array(o))) == 1]

    def _init_plasticity_tracker(self) -> None:
        """코너 블록 소성 변형 추적기 초기화 및 초기 시각적 강조(Yellow)"""
        for gi in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gi)
            if name and "cushion" in name.lower() and "_edge" in name.lower():
                self.geom_state_tracker[gi] = {
                    'is_plastic': True,
                    'yield_st': self.config.get('cush_yield_strain', 0.05),
                    'base_rgba': self.model.geom_rgba[gi].copy(),
                    'plastic_rgba': [1.0, 1.0, 0.0, 1.0]
                }
                # 초기 시각적 강조 (Yellow) 적용
                self.model.geom_rgba[gi] = [1.0, 1.0, 0.0, 1.0]

    def _aerodynamics_callback(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """공기 저항(Aerodynamics) 계산 및 외력 적용"""
        if self.root_id == -1: return
        cfg = self.config
        if not cfg.get('enable_air_drag', True) and not cfg.get('enable_air_squeeze', False): return

        rho = cfg.get('air_density', 1.225)
        vel = data.cvel[self.root_id]
        v_abs = np.linalg.norm(vel[3:6])
        total_area = 2 * (cfg.get('box_w', 2.0)*cfg.get('box_h', 1.4) + cfg.get('box_h', 1.4)*cfg.get('box_d', 0.25) + cfg.get('box_d', 0.25)*cfg.get('box_w', 2.0))

        # Quadratic Drag
        cd_q = cfg.get('air_drag_coeff', 1.05)
        f_drag = -0.5 * rho * cd_q * total_area * (v_abs**2) * np.sign(vel[5]) if cfg.get('enable_air_drag', True) else 0.0
        self._last_f_drag = f_drag

        # Viscous Drag
        mu = cfg.get('air_viscosity', 1.8e-5)
        cd_v = cfg.get('air_cd_viscous', 0.0)
        f_visc = -1.0 * mu * vel[5] * cd_v * total_area if cfg.get('enable_air_drag', True) else 0.0
        self._last_f_visc = f_visc

        # Squeeze Film Effect
        f_sq = 0.0
        if cfg.get('enable_air_squeeze', False):
            z_gap = data.xpos[self.root_id][2] - (cfg.get('box_d', 0.25)/2.0)
            h_max = cfg.get('air_squeeze_hmax', 0.1)
            h_min = cfg.get('air_squeeze_hmin', 0.001)
            if h_min < z_gap < h_max:
                k_sq = cfg.get('air_coef_squeeze', 1.0)
                f_sq = min((k_sq * mu * total_area**2 * (-vel[5])) / (z_gap**3), 2000.0) if vel[5] < 0 else 0.0
        self._last_f_sq = f_sq
        data.xfrc_applied[self.root_id][2] = f_drag + f_visc + f_sq

    def _apply_plasticity_v2(self) -> None:
        """[V5.9.0] 정밀 소성 변형 로직 (동적 면적 보정 및 감도 개선)"""
        if not self.config.get("enable_plasticity", False): return
        d, m = self.data, self.model
        p_ratio = self.config.get("plasticity_ratio", 0.5)
        
        for c_idx in range(d.ncon):
            contact = d.contact[c_idx]
            for g_id in [contact.geom1, contact.geom2]:
                if g_id in self.geom_state_tracker:
                    # 현재 변형률 계산 (초기 크기 대비)
                    strains = [max(0.0, (self.original_geom_size[g_id][ax] - m.geom_size[g_id][ax])/self.original_geom_size[g_id][ax]) for ax in range(3)]
                    equiv_strain = np.sqrt(np.sum(np.square(strains)))
                    
                    # [WHTOOLS] 동적 접촉 면적 산출 로직
                    # 접촉 법선을 로컬 좌표계로 변환하여 주축(ax) 식별
                    lx = d.geom_xmat[g_id].reshape(3,3).T @ contact.frame[:3]
                    ax = int(np.argmax(np.abs(lx)))
                    
                    # 주축에 수직인 두 축의 곱으로 면적 산출 (Box geom 기준 4 * s1 * s2)
                    other_axes = [i for i in range(3) if i != ax]
                    area = m.geom_size[g_id][other_axes[0]] * m.geom_size[g_id][other_axes[1]] * 4.0
                    
                    yield_pr = self.config.get("cush_yield_pressure", 1000.0) + (self.config.get("plastic_hardening_modulus", 0.0) * equiv_strain)
                    force = np.zeros(6)
                    mujoco.mj_contactForce(m, d, c_idx, force)
                    
                    # 압력 계산 (Pa)
                    pressure = abs(force[0]) / (area + 1e-9)
                    
                    if pressure > yield_pr:
                        # [WHTOOLS] 최적화된 소성 유동 수식
                        # 압력 초과분에 비례하여 영구 변형(reduction) 적용
                        excess = (pressure - yield_pr) / 1e6 # MPa 단위 초과량
                        # 감도 향상을 위해 p_ratio를 감도 계수로 활용
                        reduction = excess * p_ratio * m.opt.timestep * 2.0 
                        
                        m.geom_size[g_id][ax] = max(
                            self.original_geom_size[g_id][ax] * (1.0 - self.config.get("plastic_max_strain", 0.5)), 
                            m.geom_size[g_id][ax] - reduction
                        )
                    
                    # 실시간 통계 업데이트
                    self.max_applied_pressure_pa = max(self.max_applied_pressure_pa, pressure)
                    self.max_plastic_strain = max(self.max_plastic_strain, equiv_strain)
                    self.max_equiv_strain = max(self.max_equiv_strain, equiv_strain)
                    for i in range(3):
                        self.max_deformation_mm = max(self.max_deformation_mm, (self.original_geom_size[g_id][i] - m.geom_size[g_id][i])*1000.0)
                    
                    # 시각적 피드백 시스템 (변향률에 비례하여 파란색에서 빨간색으로)
                    sn = np.clip(equiv_strain / self.config.get("plastic_color_limit", 0.1), 0.0, 1.0)
                    m.geom_rgba[g_id] = [sn, 0.4, 1.0 - sn, 1.0]

    def _collect_history(self) -> None:
        d = self.data; rid = self.root_id
        self.time_history.append(d.time)
        self.z_hist.append(d.xpos[rid, 2])
        self.pos_hist.append(d.xpos.copy())
        self.vel_hist.append(d.cvel[rid].copy())
        self.acc_hist.append(d.cacc[rid].copy())
        
        q_frame = np.zeros((self.model.nbody, 4))
        for i in range(self.model.nbody): q_frame[i] = d.xquat[i].copy()
        self.quat_hist.append(q_frame)
        self.cog_pos_hist.append(d.subtree_com[rid].copy())
        self.ground_impact_hist.append(np.sum([np.linalg.norm(d.contact[i].frame[:3]) for i in range(d.ncon)]))
        self.air_drag_hist.append(self._last_f_drag)
        self.air_squeeze_hist.append(self._last_f_sq)
        
        bw, bh, bd = self.config.get('box_w', 2.0), self.config.get('box_h', 1.4), self.config.get('box_d', 0.25)
        ck = compute_corner_kinematics(d.xpos[rid], d.xmat[rid].reshape(3,3), d.cvel[rid], d.cacc[rid], bw, bh, bd)
        self.corner_pos_hist.append([c['pos'] for c in ck])
        self.corner_vel_hist.append([c['vel'] for c in ck])
        self.corner_acc_hist.append([c['acc'] for c in ck])
        self.geo_center_pos_hist.append(np.mean([c['pos'] for c in ck], axis=0))

    @property
    def tk_root(self):
        if self._tk_root is None:
            import tkinter as tk
            self._tk_root = tk.Tk(); self._tk_root.withdraw()
        return self._tk_root

    def simulate(self, enable_UI: bool = False) -> None:
        if enable_UI: self.ctrl_open_ui = True
        while not self.ctrl_quit_request:
            self.setup()
            self._run_engine()
            if not self.ctrl_reload_request: break
            self.ctrl_reload_request = False
        self._wrap_up()

    def _run_engine(self) -> None:
        if self.config.get("use_viewer", True):
            with mujoco.viewer.launch_passive(self.model, self.data, key_callback=self._on_key) as viewer:
                self.viewer = viewer; self._main_loop()
        else: self._main_loop()

    def _main_loop(self) -> None:
        step_idx = 0; console = Console()
        total_steps = int(self.config.get("sim_duration", 1.0) / self.model.opt.timestep)
        while step_idx < total_steps and not self.ctrl_quit_request and not self.ctrl_reload_request:
            if self._tk_root: self._tk_root.update()
            if self.ctrl_open_ui:
                from .whts_gui import ConfigEditor
                self.config_editor = ConfigEditor(self); self.ctrl_open_ui = False
            
            if not self.ctrl_paused:
                mujoco.mj_step(self.model, self.data)
                self._apply_plasticity_v2()
                
                if step_idx % max(1, int(self.config.get("reporting_interval", 0.005)/self.model.opt.timestep)) == 0:
                    self._collect_history()
                
                if int(self.data.time / 0.05) > self._last_reported_interval:
                    self._last_reported_interval = int(self.data.time / 0.05)
                    real_elapsed = time.time() - self.start_real_time
                    fps = step_idx / real_elapsed if real_elapsed > 0 else 0
                    
                    # [V5.8.3] Pixel-Perfect Alignment & Robust Border logic
                    if self._report_count == 0:
                        # Emoji는 2칸을 차지함: 직접 정렬 튜닝
                        header_str = f"   🔢 Step     ⏱️ Time       🚀 Real       ⚡ FPS      🗜️ Status (SE, PRS, PE, DF)"
                        # 보더 길이는 Status 확장성을 고려하여 100자로 고정
                        border_line = "━" * 100 
                        console.print(f"[bold white]{border_line}[/bold white]")
                        console.print(f"[bold green]{header_str}[/bold green]")
                        console.print(f"[bold white]{border_line}[/bold white]")

                    status = f"SE:{self.max_equiv_strain:5.1%}, PRS:{self.max_applied_pressure_pa/1e6:6.3f}(MPa), PE:{self.max_plastic_strain:5.1%}, DF:{self.max_deformation_mm:4.1f}mm"
                    # 헤더 간격에 맞춘 정밀 데이터 포맷팅
                    row_str = f"   {step_idx:<9d}  {self.data.time:<11.3f}   {real_elapsed:<11.2f}   {fps:<10.1f}   {status}"
                    console.print(row_str)
                    self._report_count += 1
                step_idx += 1
                if self.viewer: self.viewer.sync()
            else:
                # 시뮬레이션 종료 시 마무리 선 출력
                if self._report_count > 0:
                    border_line = "━" * 100
                    console.print(f"[bold white]{border_line}[/bold white]")
                self._report_count = -1
                if self.viewer: self.viewer.sync()
                time.sleep(0.01)

    def _wrap_up(self) -> None:
        """데이터 정리 및 UI 자동 실행"""
        compute_batch_structural_metrics(self)
        finalize_simulation_results(self); apply_rank_heatmap(self)
        
        self.result = DropSimResult(
            config=self.config.copy(), metrics=self.metrics.copy(),
            max_g_force=float(np.max(np.abs(self.acc_hist))/9.81) if self.acc_hist else 0.0,
            time_history=self.time_history, z_hist=self.z_hist, root_acc_history=[],
            corner_acc_hist=self.corner_acc_hist, pos_hist=self.pos_hist,
            vel_hist=self.vel_hist, acc_hist=self.acc_hist,
            cog_pos_hist=self.cog_pos_hist, geo_center_pos_hist=self.geo_center_pos_hist,
            corner_pos_hist=self.corner_pos_hist, ground_impact_hist=self.ground_impact_hist,
            air_drag_hist=self.air_drag_hist, air_squeeze_hist=self.air_squeeze_hist,
            structural_metrics=self.structural_time_series, critical_timestamps={},
            nominal_local_pos=self.nominal_local_pos, quat_hist=self.quat_hist,
            components=self.components.copy(), body_index_map=self.body_index_map, block_half_extents=self.block_half_extents
        )
        self.result.save(os.path.join(self.output_dir, "simulation_result.pkl"))

        # UI 자동 실행 로직 복구
        if self.config.get("use_postprocess_v2", False):
            self.log(">> [Integrated UI] Launching V2 Control Center..."); subprocess_call(self)
        elif self.ctrl_open_ui or self.config.get("use_postprocess_ui", True):
            self.log(">> [Legacy UI] Launching Tkinter Post-Processing..."); from .whts_postprocess_ui import PostProcessingUI; PostProcessingUI(self).on_simulation_complete(); self.tk_root.mainloop()

    def apply_balancing(self) -> None:
        self.config["chassis_aux_masses"] = calculate_required_aux_masses(self.config, self.config.get("target_mass"), self.config.get("target_cog"), self.config.get("target_moi"))

    def _on_key(self, keycode: int) -> None:
        if keycode == 32: self.ctrl_paused = not self.ctrl_paused
        elif keycode == 75: self.ctrl_open_ui = True
        elif keycode == 256: self.ctrl_quit_request = True

def subprocess_call(sim):
    try:
        import subprocess; curr = os.path.dirname(os.path.abspath(__file__))
        subprocess.Popen([sys.executable, os.path.join(curr, "whts_postprocess_ui_v2.py"), "--load", os.path.join(sim.output_dir, "simulation_result.pkl")])
    except: pass
