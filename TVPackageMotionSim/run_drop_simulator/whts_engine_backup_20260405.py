import os
import time
import json
import pickle
import numpy as np
import mujoco
import mujoco.viewer
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

# [WHTOOLS] 패키지 내부 모듈 임포트
from .whts_data import DropSimResult
from .whts_utils import compute_corner_kinematics, calculate_required_aux_masses
from .whts_gui import ConfigEditor
from .whts_reporting import (
    compute_structural_step_metrics, 
    finalize_simulation_results, 
    apply_rank_heatmap,
    compute_critical_timestamps
)

# [WHTOOLS] 외부 패키지 임포트
from run_discrete_builder import create_model, get_default_config

class DropSimulator:
    """
    [WHTOOLS] MuJoCo 기반의 낙하 시뮬레이션 통합 엔진 (v4)입니다.
    
    이 클래스는 시뮬레이션의 전체 생명주기를 관리하며, 고도의 물리 로직(소성, 유체)과 
    실시간 분석 지표(PBA, RRG)를 결합하여 전문 엔지니어링 수준의 결과를 도출합니다.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        시뮬레이터 초기화 및 설정 로드.
        """
        self.config = get_default_config(config) if config else get_default_config()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"rds-{self.timestamp}"
        
        self.model = None
        self.data = None
        self.viewer = None
        
        self._init_state_variables()
        self._init_histories()
        
        # 제어 플래그
        self.ctrl_paused = False
        self.ctrl_reset_request = False
        self.ctrl_quit_request = False
        self.ctrl_reload_request = False
        self.ctrl_open_ui = False
        self.ctrl_step_forward_count = 0
        self.ctrl_step_backward_count = 0
        self.ctrl_print_view = False
        
        self.post_ui = None
        self.config_editor = None
        self.result = None
        self.tk_root = None

        if self.config.get("enable_target_balancing", False):
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
        self._step_corner_values = {}
        self.nominal_local_pos = {}
        
        # [v5.2] 평판 이론(Plate Theory) 분석 데이터 수집용
        self.quat_hist = []
        self.block_half_extents = {}
        self.body_index_map = {}
        
        # [v5.2.1] 소성 변형 및 압력 지표 초기화 (AttributeError 방지)
        self.max_equiv_strain = 0.0
        self.max_current_yield = 0.0
        self.max_applied_pressure_pa = 0.0
        self.max_deformation_mm = 0.0
        self.max_plastic_strain = 0.0

    def _init_histories(self) -> None:
        self.time_history = []
        self.z_hist = []
        self.pos_hist = []
        self.vel_hist = []
        self.acc_hist = []
        self.cog_pos_hist = []
        self.cog_vel_hist = []
        self.cog_acc_hist = []
        self.geo_center_pos_hist = []
        self.geo_center_vel_hist = []
        self.geo_center_acc_hist = []
        self.corner_pos_hist = []
        self.corner_vel_hist = []
        self.corner_acc_hist = []
        self.ground_impact_hist = []
        self.air_drag_hist = []
        self.air_squeeze_hist = []
        self.air_viscous_hist = []
        self.structural_time_series = {
            'pba_magnitude': [], 'pba_angle': [], 'pba_vector': [],
            'rrg_max': [], 'rrg_max_location': [], 'mean_distortion': [],
            'gti_max': [], 'gbi_max': [], 'comp_global_metrics': {}
        }

    def log(self, text: str) -> None:
        entry = f"[{datetime.now().strftime('%H:%M:%S')}] {text}"
        print(entry)
        if hasattr(self, 'report_file_path') and self.report_file_path:
            try:
                with open(self.report_file_path, "a", encoding="utf-8") as f:
                    f.write(entry + "\n")
            except: pass

    def setup(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        self.report_file_path = os.path.join(self.output_dir, "summary_report.txt")
        
        xml_path = os.path.join(self.output_dir, "simulation_model.xml")
        xml_content, *_ = create_model(xml_path, config=self.config)
        
        self.model = mujoco.MjModel.from_xml_string(xml_content)
        self.data = mujoco.MjData(self.model)
        
        self.root_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "BPackagingBox")
        self.original_geom_pos = self.model.geom_pos.copy()
        self.original_geom_size = self.model.geom_size.copy()
        self.original_geom_rgba = self.model.geom_rgba.copy()
        
        self._discover_components()
        self._discover_neighbors()
        self._init_tracking_containers()
        self._init_plasticity_tracker()
        
        self._mjcb_control = lambda m, d: self._aerodynamics_callback(m, d)
        mujoco.set_mjcb_control(self._mjcb_control)

    def _discover_components(self) -> None:
        self.components = {}
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and name.startswith("b_"):
                parts = name.split("_")
                if len(parts) >= 5:
                    try:
                        grid_key = (int(parts[-3]), int(parts[-2]), int(parts[-1]))
                        comp_name = "_".join(parts[1:-3]).lower()
                        if comp_name not in self.components: self.components[comp_name] = {}
                        self.components[comp_name][grid_key] = i
                        self.body_index_map[i] = name
                        # 초기 로컬 위치 및 크기 저장 (평판 이론 해석용)
                        self.nominal_local_pos[i] = self.model.body_pos[i].tolist()
                        
                        # [v5.2] 블록 절반 크기 저장 (평판 단면 계수 산출용)
                        if self.model.body_geomnum[i] > 0:
                            g_id = self.model.body_geomadr[i]
                            self.block_half_extents[i] = self.model.geom_size[g_id][:3].tolist()
                    except: continue

    def _discover_neighbors(self) -> None:
        self.neighbor_map = {}
        for c_name, blocks in self.components.items():
            self.neighbor_map[c_name] = {}
            for grid_idx in blocks:
                i, j, k = grid_idx
                neighbors = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0: continue
                        ni, nj = i + di, j + dj
                        if (ni, nj, k) in blocks: neighbors.append((ni, nj, k))
                self.neighbor_map[c_name][grid_idx] = neighbors

    def _init_tracking_containers(self) -> None:
        self.metrics = {}
        for c_name, blocks in self.components.items():
            self.metrics[c_name] = {
                'total_distortion': [],
                'all_blocks_angle': {idx: [] for idx in blocks},
                'all_blocks_bend': {idx: [] for idx in blocks},
                'all_blocks_twist': {idx: [] for idx in blocks},
                'all_blocks_bend_x': {idx: [] for idx in blocks},
                'all_blocks_bend_y': {idx: [] for idx in blocks},
                'all_blocks_rotvec': {idx: [] for idx in blocks},
                'all_blocks_rrg': {idx: [] for idx in blocks},
                'all_blocks_s_bend': {idx: [] for idx in blocks},
                'all_blocks_s_twist': {idx: [] for idx in blocks},
                'all_blocks_moment': {idx: [] for idx in blocks},
                'all_blocks_energy': {idx: [] for idx in blocks},
                'block_nominal_mats': {idx: None for idx in blocks},
                'max_rrg_hist': [], 'max_pba_hist': []
            }

    def _init_plasticity_tracker(self) -> None:
        """소성 변형 상태 추적을 위한 초기 설정을 수행합니다. 
           (8개 꼭짓점 및 Depth 방향 모서리 블록만 타겟팅)"""
        for gi in range(self.model.ngeom):
            g_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gi)
            if g_name and "bcushion" in g_name.lower() and "_edge" in g_name.lower():
                # 장축(Z축) 방향 탐지 및 초기 변수 할당
                size = self.model.geom_size[gi]
                major_axis = np.argmax(size)
                self.geom_state_tracker[gi] = {
                    'major_axis': major_axis,
                    'is_plastic': True,
                    'yield_st': self.config.get('cush_yield_strain', 0.05),
                    'base_rgba': self.model.geom_rgba[gi].copy(),
                    'plastic_rgba': [1.0, 1.0, 0.0, 1.0] # 초기 강조용 노란색
                }
                # 초기 시각적 강조 피드백 (노란색)
                self.model.geom_rgba[gi] = [1.0, 1.0, 0.0, 1.0]

    def _aerodynamics_callback(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self.root_id == -1: return
        cfg = self.config
        rho = cfg.get('air_density', 1.225)
        vel = data.cvel[self.root_id]
        v_abs = np.linalg.norm(vel[3:6])
        
        area = cfg.get('box_w', 2.0) * cfg.get('box_h', 1.4)
        cd = cfg.get('air_drag_coeff', 1.05)
        f_drag = 0.5 * rho * v_abs**2 * area * cd
        self._last_f_drag = f_drag
        
        z_gap = data.xpos[self.root_id][2] - (cfg.get('box_d', 0.25)/2.0)
        f_sq = 0.0
        if 0.001 < z_gap < 0.1:
            mu = 1.8e-5
            v_z = -vel[5]
            if v_z > 0:
                f_sq = min((mu * area**2 * v_z) / (z_gap**3), 2000.0)
        self._last_f_sq = f_sq
        data.xfrc_applied[self.root_id][2] = f_drag + f_sq

    def _apply_plasticity_v2(self) -> None:
        """[V4] 고도화된 소성 변형 로직: 접촉 방향(Normal) 기반 축 선택 및 실시간 파란색 전이"""
        if not self.config.get("enable_plasticity", False): return
        d, m = self.data, self.model
        p_ratio = self.config.get("plasticity_ratio", 0.8)
        
        # 전역 통계 초기화
        self.max_equiv_strain = 0.0
        self.max_current_yield = self.config.get("cush_yield_pressure", 1000.0)
        self.max_applied_pressure_pa = 0.0
        self.max_deformation_mm = 0.0
        self.max_plastic_strain = 0.0

        for c_idx in range(d.ncon):
            contact = d.contact[c_idx]
            g1, g2 = contact.geom1, contact.geom2
            world_normal = contact.frame[:3].copy()
            
            for g_id in [g1, g2]:
                if g_id in self.geom_state_tracker:
                    # 1. 먼저 현재의 등가 변형률(Equivalent Strain)을 산출
                    strains = []
                    for ax in range(3):
                        ax_orig = self.original_geom_size[g_id][ax]
                        ax_curr = m.geom_size[g_id][ax]
                        strains.append(max(0.0, (ax_orig - ax_curr) / ax_orig))
                    equiv_strain = np.sqrt(np.sum(np.square(strains)))
                    self.max_equiv_strain = max(self.max_equiv_strain, equiv_strain)

                    # 2. 하드닝 모델링
                    yield_pr_0 = self.config.get("cush_yield_pressure", 1000.0)
                    h_modulus = self.config.get("plastic_hardening_modulus", 0.0)
                    current_yield = yield_pr_0 + (h_modulus * equiv_strain)
                    self.max_current_yield = max(self.max_current_yield, current_yield)
                    
                    # 3. 압력 기반 수축 여부 판정 및 크기 반영
                    force = np.zeros(6)
                    mujoco.mj_contactForce(m, d, c_idx, force)
                    pressure = abs(force[0]) / (m.geom_size[g_id][0] * m.geom_size[g_id][1] * 4)
                    
                    if pressure > current_yield:
                        xmat = d.geom_xmat[g_id].reshape(3, 3)
                        local_normal = xmat.T @ world_normal
                        target_axis = int(np.argmax(np.abs(local_normal)))
                        
                        max_strain_limit = self.config.get("plastic_max_strain", 0.5)
                        orig_val = self.original_geom_size[g_id][target_axis]
                        reduction = (pressure / 1e6) * p_ratio * m.opt.timestep
                        
                        m.geom_size[g_id][target_axis] = max(orig_val * (1.0 - max_strain_limit), m.geom_size[g_id][target_axis] - reduction)
                    
                    # 전역 지표 업데이트
                    self.max_applied_pressure_pa = max(self.max_applied_pressure_pa, pressure)
                    self.max_plastic_strain = max(self.max_plastic_strain, equiv_strain) # 단순 모델: PE = Current Strain
                    
                    # DF(mm): 원본 대비 최대 수축량 산출
                    for ax in range(3):
                        def_mm = (self.original_geom_size[g_id][ax] - m.geom_size[g_id][ax]) * 1000.0
                        self.max_deformation_mm = max(self.max_deformation_mm, def_mm)
                    
                    # 4. 실시간 색상 업데이트
                    color_limit = self.config.get("plastic_color_limit", 0.1)
                    strain_norm = np.clip(equiv_strain / color_limit, 0.0, 1.0)
                    m.geom_rgba[g_id] = [1.0 - strain_norm, 1.0 - strain_norm, strain_norm, 1.0]

    def _collect_history(self) -> None:
        d, m = self.data, self.model
        self.qpos_hist.append(d.qpos.copy())
        self.qvel_hist.append(d.qvel.copy())
        self.time_history.append(d.time)
        
        rid = self.root_id
        if rid != -1:
            self.z_hist.append(d.xpos[rid][2])
            # [v5.1] 평판 변형 엔진 연동을 위해 전체 바디 좌표 및 회전 정보 기록
            self.pos_hist.append(d.xpos.copy())
            self.quat_hist.append(d.xquat.copy())
            self.vel_hist.append(d.cvel[rid].copy())
            self.acc_hist.append(d.cacc[rid].copy())
            self.cog_pos_hist.append(d.subtree_com[rid].copy())
            self.ground_impact_hist.append(np.sum([np.linalg.norm(d.contact[i].frame[:3]) for i in range(d.ncon)]))
            self.air_drag_hist.append(self._last_f_drag)
            self.air_squeeze_hist.append(self._last_f_sq)
            
            # Corner Kinematics (v4 Parity)
            bw, bh, bd = self.config.get('box_w', 2.0), self.config.get('box_h', 1.4), self.config.get('box_d', 0.25)
            ck = compute_corner_kinematics(d.xpos[rid], d.xmat[rid].reshape(3,3), d.cvel[rid], d.cacc[rid], bw, bh, bd)
            self.corner_pos_hist.append([c['pos'] for c in ck])
            self.corner_vel_hist.append([c['vel'] for c in ck])
            self.corner_acc_hist.append([c['acc'] for c in ck])
            self.geo_center_pos_hist.append(np.mean([c['pos'] for c in ck], axis=0))

    def simulate(self, enable_UI: bool = False) -> None:
        if enable_UI:
            self.ctrl_open_ui = True
            
        import tkinter as tk
        self.tk_root = tk.Tk()
        self.tk_root.withdraw()
        
        while not self.ctrl_quit_request:
            self.setup()
            self._run_engine()
            if not self.ctrl_reload_request: break
            self.ctrl_reload_request = False
            self.log(">> [Reload] 시뮬레이션을 재시작합니다.")
        
        # 종료 전 결과물 자동 생성 및 저장
        self._wrap_up()
        
        # [V5.3.3 FIX] Tkinter 리소스 안전 해제 (중복 파괴 방지)
        try:
            if hasattr(self, 'tk_root') and self.tk_root:
                self.tk_root.destroy()
        except:
            pass

    def _run_engine(self) -> None:
        if self.config.get("use_viewer", True):
            with mujoco.viewer.launch_passive(self.model, self.data, key_callback=self._on_key) as viewer:
                self.viewer = viewer
                self._main_loop()
        else:
            self._main_loop()

    def _main_loop(self) -> None:
        step_idx = 0
        total_steps = int(self.config.get("sim_duration", 1.0) / self.model.opt.timestep)
        
        # [WHTOOLS] 기존 테이블 스타일 초기화
        self._last_reported_interval = -1
        self._report_count = 0
        self.start_real_time = time.time()
        
        while step_idx < total_steps and not self.ctrl_quit_request and not self.ctrl_reload_request:
            self.tk_root.update()
            
            if self.ctrl_open_ui:
                self.config_editor = ConfigEditor(self)
                self.ctrl_open_ui = False
            
            if self.ctrl_step_backward_count > 0:
                step_idx = max(0, step_idx - self.ctrl_step_backward_count)
                self._restore_state(step_idx); self.ctrl_step_backward_count = 0; self.ctrl_paused = True

            if not self.ctrl_paused:
                mujoco.mj_step(self.model, self.data)
                self._apply_plasticity_v2()
                
                # [v4.6.2] 데이터 수집 및 구조 지표 연산 주기 최적화 (Memory & Storage)
                # config['reporting_interval'] (기본 0.005s) 기반으로 Decimation 수행
                report_dt = self.config.get("reporting_interval", 0.005)
                save_step_mod = max(1, int(report_dt / self.model.opt.timestep))
                
                if step_idx % save_step_mod == 0:
                    self._collect_history()
                    # compute_structural_step_metrics(self) # [V5.2.8.2] 배치 처리를 위해 제거 (FPS 복구)
                
                if self.config_editor: self.config_editor.update_status(step_idx, self.data.time)
                
                # [WHTOOLS] 기존 테이블 스타일 출력 (0.05s 간격)
                sim_interval = int(self.data.time / 0.05)
                if sim_interval > self._last_reported_interval:
                    real_elapsed = time.time() - self.start_real_time
                    fps = step_idx / real_elapsed if real_elapsed > 0 else 0
                    
                    if self._report_count % 30 == 0:
                        h_sep = "-" * 112
                        print(f"\n[Legend] SE: Equivalent Strain, PRS: Pressure (MPa), FPS: Frames Per Second, PE: Plastic Strain, DF: Deformation (mm)")
                        print(h_sep)
                        print(f"| {'Step':^8} | {'Time (s)':^10} | {'Real (s)':^10} | {'FPS':^8} | {'Cushion Status (SE, PRS, PE, DF)':^60} |")
                        print(h_sep)
                    
                    prs_mpa = self.max_applied_pressure_pa / 1e6
                    cush_status = f"SE:{self.max_equiv_strain:5.1%}, PRS:{prs_mpa:6.3f} (MPa), PE:{self.max_plastic_strain:5.1%}, DF:{self.max_deformation_mm:4.1f}mm"
                    print(f"| {step_idx:8d} | {self.data.time:10.3f} | {real_elapsed:10.2f} | {fps:8.1f} | {cush_status:<60} |")
                    
                    self._last_reported_interval = sim_interval
                    self._report_count += 1
                
                step_idx += 1
                if self.viewer: self.viewer.sync()
            else:
                if self.viewer: self.viewer.sync()
                time.sleep(0.01)

    def _wrap_up(self) -> None:
        """데이터 정리 및 Post-UI 연동"""
        # [V5.2.8.2] 배치 구조 해석 수행
        from .whts_reporting import compute_batch_structural_metrics
        compute_batch_structural_metrics(self)
        
        finalize_simulation_results(self)
        apply_rank_heatmap(self)
        
        # 결과 객체 직렬화 호출
        self.result = DropSimResult(
            config=self.config.copy(), metrics=self.metrics.copy(),
            max_g_force=float(np.max(np.abs(self.acc_hist))/9.81) if self.acc_hist else 0.0,
            time_history=self.time_history, z_hist=self.z_hist,
            root_acc_history=[], # (가속도 데이터 후처리 생략)
            corner_acc_hist=self.corner_acc_hist,
            pos_hist=self.pos_hist, vel_hist=self.vel_hist, acc_hist=self.acc_hist,
            cog_pos_hist=self.cog_pos_hist, geo_center_pos_hist=self.geo_center_pos_hist,
            corner_pos_hist=self.corner_pos_hist, ground_impact_hist=self.ground_impact_hist,
            air_drag_hist=self.air_drag_hist, air_squeeze_hist=self.air_squeeze_hist,
            structural_metrics=self.structural_time_series,
            critical_timestamps=compute_critical_timestamps(self),
            nominal_local_pos=self.nominal_local_pos,
            # [v5.2] SSR 전용 데이터 전달
            quat_hist=self.quat_hist,
            components=self.components.copy(),
            body_index_map=self.body_index_map,
            block_half_extents=self.block_half_extents
        )
        self.result.save(os.path.join(self.output_dir, "simulation_result.pkl"))
        
        # [V5.4] PostProcess UI 선택 (V1: Tkinter, V2: PySide6)
        if self.config.get("use_postprocess_v2", False):
            self.log(">> [Integrated UI] 신형 V2 Control Center를 호출합니다...")
            try:
                import subprocess
                curr_dir = os.path.dirname(os.path.abspath(__file__))
                ui_script = os.path.join(curr_dir, "whts_postprocess_ui_v2.py")
                res_path = os.path.join(self.output_dir, "simulation_result.pkl")
                # 신형 UI를 서브프로세스로 실행 (Tkinter와의 충돌 방지)
                subprocess.Popen([sys.executable, ui_script, "--load", res_path])
            except Exception as e:
                self.log(f"⚠️ V2 UI 실행 실패: {e}")
        elif self.config.get("use_postprocess_ui", True):
            try:
                from .whts_postprocess_ui import PostProcessingUI
                self.post_ui = PostProcessingUI(self)
                self.post_ui.on_simulation_complete() # [V4.2 FIX] 완료 이벤트 명시적 트리거
                self.tk_root.mainloop()
            except Exception as e:
                self.log(f"⚠️ Post-Processing UI 실행 실패: {e}")
        else:
            self.log("ℹ️ use_postprocess_ui 가 False이므로 구버전 분석 창을 띄우지 않습니다.")

    def _restore_state(self, idx: int) -> None:
        if idx < len(self.qpos_hist):
            self.data.qpos[:] = self.qpos_hist[idx]
            self.data.qvel[:] = self.qvel_hist[idx]
            mujoco.mj_forward(self.model, self.data)

    def apply_balancing(self) -> None:
        self.config["chassis_aux_masses"] = calculate_required_aux_masses(
            self.config, self.config.get("target_mass"),
            self.config.get("target_cog"), self.config.get("target_moi")
        )

    def _on_key(self, keycode: int) -> None:
        if keycode == 32: self.ctrl_paused = not self.ctrl_paused
        elif keycode == 75: self.ctrl_open_ui = True # 'K'
        elif keycode == 256: self.ctrl_quit_request = True
