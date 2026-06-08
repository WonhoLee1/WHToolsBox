# -*- coding: utf-8 -*-
"""
[WHTOOLS] Drop Simulator Engine v6.0 - High-Fidelity Physical Integration
MuJoCo 시뮬레이션 메인 루프, 정밀 물리(소성/공기저항) 및 실시간 분석을 담당합니다.
이 모듈은 고성능 JAX 기반 최적화 파이프라인과의 연동을 염두에 두고 설계되었습니다.
"""

import os
import sys
import time
import signal
import json
import pickle
import logging
import numpy as np
import mujoco
import mujoco.viewer
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

# [WHTOOLS] 최적화 모듈 (Numba JIT)
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func): return func
        return decorator

@njit(cache=True, fastmath=True)
def _numba_calc_aero(v_linear, z_gap, rho, cd_q, total_area, mu, cd_v, h_max, h_min, k_sq, enable_drag, enable_sq):
    v_abs = 0.0
    for i in range(3): v_abs += v_linear[i]**2
    v_abs = v_abs**0.5
    
    f_drag = 0.0
    f_visc = 0.0
    f_sq = 0.0
    
    if enable_drag:
        sign_z = 1.0 if v_linear[2] > 0 else (-1.0 if v_linear[2] < 0 else 0.0)
        f_drag = -0.5 * rho * cd_q * total_area * (v_abs**2) * sign_z
        f_visc = -1.0 * mu * v_linear[2] * cd_v * total_area
        
    if enable_sq and (h_min < z_gap < h_max) and v_linear[2] < 0:
        f_sq = (k_sq * mu * (total_area**2) * (-v_linear[2])) / (z_gap**3)
        if f_sq > 2000.0:
            f_sq = 2000.0
            
    return f_drag, f_visc, f_sq

# [WHTOOLS] 시각화 및 로깅 라이브러리
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich import box

# [WHTOOLS] 패키지 내부 모듈 임포트
from .whts_data import DropSimResult
from .whts_utils import compute_corner_kinematics, calculate_required_aux_masses, WHToolsSessionLogger
from .whts_reporting import (
    compute_structural_step_metrics, 
    finalize_simulation_results, 
    apply_rank_heatmap,
    compute_critical_timestamps,
    compute_batch_structural_metrics
)
from .whts_control_panel import launch_control_panel

# [WHTOOLS] 외부 패키지 임포트
from run_discrete_builder import create_model, get_default_config

# [WHTOOLS] Thread-safe MuJoCo callback dispatcher
import threading
_mujoco_thread_registry = {}

def _global_mujoco_control_callback(model, data):
    import threading
    tid = threading.get_ident()
    cb = _mujoco_thread_registry.get(tid)
    if cb is not None:
        try:
            cb(model, data)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise

# 로깅 설정
import platform
is_windows = platform.system() == "Windows"
_console = Console(color_system=None if is_windows else "auto")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=_console, rich_tracebacks=True, markup=True)]
)
# [WHTOOLS] UTF-8 인코딩 강제 설정 (Rich/Console 호환성)
if sys.stdout.encoding != 'utf-8':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except (AttributeError, io.UnsupportedOperation):
        pass

logger = logging.getLogger("WHTS_Engine")
logger.propagate = False
# 중복 콘솔 로깅 방지 및 WHTS_Engine 로거에만 RichHandler 단독 매핑
logger.handlers = []
logger.addHandler(RichHandler(console=_console, rich_tracebacks=True, markup=True))
console = _console

from PySide6.QtCore import QThread
class SimThread(QThread):
    """[WHTOOLS] 시뮬레이션 엔진을 메인 UI 스레드와 별개로 실행하는 스레드입니다."""
    def __init__(self, outer):
        super().__init__()
        self.outer = outer

    def run(self):
        try:
            self.outer._run_engine()
        finally:
            # reload 중에는 _wrap_up 생략 — UI 스레드가 새 세션을 시작함
            if not self.outer.ctrl_reload_request:
                self.outer._wrap_up()

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
        result_base = self.config.get("result_base_dir", "results")
        base_dir = self.config.get("output_dir", Path(result_base) / default_dir)
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
        self.ctrl_cam_view = None         # [WHTOOLS] 시점 전환 요청 (+X, -X, ISO 등)
        self.ctrl_reload_only_xml = False # XML 생성을 건너뛰고 기존 파일만 로드할지 여부
        self.ctrl_reload_xml_path = None  # 리로드할 외부 XML 경로
        
        # UI 관련
        self.config_editor = None
        self.result = None

        # 세션 로그 시작 (최초 인스턴스 생성 시 1회만 활성화)
        WHToolsSessionLogger.start()

        # 자동 밸런싱 적용
        if self.config.get("enable_target_balancing", False) or "components_balance" in self.config:
            self.apply_balancing()

    def __del__(self) -> None:
        WHToolsSessionLogger.release()

    # ── Config I/O ───────────────────────────────────────────────────────────

    def save_config(self, path: Union[str, Path]) -> Dict[str, Any]:
        """현재 self.config를 JSON 파일로 저장하고 config dict를 반환합니다.

        파일 상단에 패키지 규격·낙하 조건·물리 옵션 설명을 _comment_* 필드로 포함합니다.
        """
        from run_discrete_builder.whtb_config import save_config as _save
        cfg = self.config
        w  = cfg.get("box_w", 0) * 1000
        h  = cfg.get("box_h", 0) * 1000
        d  = cfg.get("box_d", 0) * 1000
        extra_meta = {
            "_comment_tool": (
                "WHTOOLS TV Package Motion Simulation Tool  |  "
                f"Saved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            ),
            "_comment_description": (
                "MuJoCo-based rigid-body drop simulation for TV package cushion analysis. "
                "Captures plasticity deformation, air fluidics, and multi-corner kinematics."
            ),
            "_comment_package": (
                f"Package Size: W={w:.0f}mm x H={h:.0f}mm x D={d:.0f}mm  |  "
                f"Target Mass: {cfg.get('target_mass', 0):.1f} kg"
            ),
            "_comment_drop": (
                f"Drop Mode: {cfg.get('drop_mode', 'N/A')}  |  "
                f"Direction: {cfg.get('drop_direction', 'N/A')}  |  "
                f"Height: {cfg.get('drop_height', 0) * 1000:.0f} mm"
            ),
            "_comment_physics": (
                f"Plasticity: {'ON' if cfg.get('enable_plasticity') else 'OFF'}  |  "
                f"Air Drag: {'ON' if cfg.get('enable_air_drag') else 'OFF'}"
            ),
        }
        _save(cfg, path, extra_meta=extra_meta)
        self.log(f"💾 Config saved → {Path(path).name}")
        return dict(cfg)

    def load_config(self, path: Union[str, Path]) -> Dict[str, Any]:
        """JSON 파일을 읽어 self.config를 갱신하고 새 config dict를 반환합니다.

        누락 키는 기본값으로 자동 채워지며, 물리 파라미터 동기화도 수행됩니다.
        리로드 요청 플래그를 설정하여 다음 루프에서 모델을 재생성합니다.
        """
        from run_discrete_builder.whtb_config import load_config as _load
        new_cfg = _load(path)
        self.config.update(new_cfg)
        self.ctrl_reload_request = True
        self.log(f"✅ Config loaded : {Path(path).name}")
        return dict(self.config)

    def load_config_from_dict(self, new_cfg) -> dict:
        self.config.update(new_cfg)
        self.ctrl_reload_request = True
        self.log("✅ Config loaded from dictionary/pickle")
        return dict(self.config)

    def _init_state_variables(self) -> None:
        """시뮬레이션 내부 상태 추적 변수들을 초기화합니다.
        setup() 이후에 호출될 때는 geom_state_tracker/components 등
        _init_plasticity_tracker()가 채운 데이터를 덮어쓰지 않도록 주의.
        _reset_loop_variables()가 루프 재시작 시 안전하게 리셋합니다."""
        self.geom_state_tracker: Dict[int, Dict[str, Any]] = {}
        self.components: Dict[str, Dict[Tuple[int, int, int], int]] = {}
        self.metrics: Dict[str, Any] = {}
        self.neighbor_map: Dict[str, Dict[Tuple[int, int, int], List[Tuple[int, int, int]]]] = {}
        self.snapshots: List[Dict[str, Any]] = []

        self._last_f_drag = 0.0
        self._last_f_sq = 0.0
        self._last_f_visc = 0.0
        self.nominal_local_pos: Dict[int, np.ndarray] = {}
        self.block_half_extents: Dict[int, np.ndarray] = {}
        self.body_index_map: Dict[int, Tuple[int, int, int]] = {}

        self._reset_loop_variables()

    def _reset_loop_variables(self) -> None:
        """루프 시작/재시작 시 리셋해야 하는 변수만 초기화합니다.
        setup()에서 채운 geom_state_tracker, components 등은 건드리지 않습니다."""
        # 실시간 물리 지표 (최대치 추적)
        self.max_equiv_strain = 0.0
        self.max_applied_pressure_pa = 0.0
        self.max_deformation_mm = 0.0
        self.max_plastic_strain = 0.0
        self._last_reported_interval = -1
        self._report_count = 0
        self.start_real_time = time.time()

        # [WHTOOLS] 인터랙티브 레코딩 및 특수 효과
        self.is_recording = False
        self.ctrl_slow_motion = False
        self.step_idx = 0

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
        
        # [WHTOOLS] 결과값(Resultant/Magnitude) 및 코너별 충격력
        self.corner_pos_res_hist: List[np.ndarray] = []
        self.corner_vel_res_hist: List[np.ndarray] = []
        self.corner_acc_res_hist: List[np.ndarray] = []
        self.corner_impact_hist: List[np.ndarray] = []
        
        # [WHTOOLS] 파트별(다중) 코너 기구학 데이터
        self.part_corner_hist: Dict[str, Dict[str, List[np.ndarray]]] = {
            "Cushion": {"pos": [], "vel": [], "acc": [], "pos_res": [], "vel_res": [], "acc_res": []},
            "Cushion-Rigid": {"pos": [], "vel": [], "acc": [], "pos_res": [], "vel_res": [], "acc_res": []},
            "Chassis": {"pos": [], "vel": [], "acc": [], "pos_res": [], "vel_res": [], "acc_res": []},
            "OpenCell": {"pos": [], "vel": [], "acc": [], "pos_res": [], "vel_res": [], "acc_res": []}
        }
        
        # [WHTOOLS] 강체 거동 대표 물리량 (회전축, 회전속도, 병진속도)
        self.rot_axis_hist: List[np.ndarray] = []
        self.rot_speed_hist: List[float] = []
        self.trans_vel_hist: List[np.ndarray] = []
        self.trans_vel_res_hist: List[float] = []
        
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
        # 이전 실행이 등록한 stale 콜백 해제 (프로세스 전역 싱글톤)
        # 해제하지 않으면 GC된 DropSimulator 인스턴스를 참조해 "Python exception raised" 발생
        import threading
        import mujoco
        mujoco.set_mjcb_control(None)
        _mujoco_thread_registry.pop(threading.get_ident(), None)

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
            
            # [WHTOOLS] 현재 로드된 모델 경로 저장 (에디터 연동용)
            self.config["model_path"] = str(current_xml_path.absolute())
            
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
            
            # [CRITICAL] 물리 제어 콜백 등록 (Thread-safe Dispatcher)
            self._mjcb_control = lambda m, d: self._physics_control_callback(m, d)
            import threading
            _mujoco_thread_registry[threading.get_ident()] = self._mjcb_control
            mujoco.set_mjcb_control(_global_mujoco_control_callback)
            
            self.start_real_time = time.time()
            self.log(f"📦 Assembly: {len(self.components)} components, {self.model.nbody} bodies identified.")
            self.log(f"🚀 Simulation Ready. Timestep: {self.model.opt.timestep:.6f}s")
            
        except Exception as e:
            import threading
            import traceback
            traceback.print_exc()
            _mujoco_thread_registry.pop(threading.get_ident(), None)  # dangling 콜백 방지
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
        # XML body names follow pattern: b_{comp}_{i}_{j}_{k}
        # comp names include the 'b' prefix: bcushion, bchassis, bopencell, etc.
        target_prefixes = ['bpaper', 'bcushion', 'bchassis', 'bopencell', 'inertiaaux', 'autobalance']
        
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
        """소성 변형 대상 Geoms(Cushion Edge 등)를 식별하고 초기 색상(Yellow)으로 강조합니다."""
        for gi in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gi)
            if name and "cushion" in name.lower() and "_edge" in name.lower():
                # [복원] 사용자의 요청으로 추적 대상 코너 블럭의 초기 색상을 눈에 띄게 노란색(Yellow)으로 즉시 변경
                self.model.geom_rgba[gi] = [1.0, 1.0, 0.0, 1.0]
                
                self.geom_state_tracker[gi] = {
                    'is_plastic': True,
                    'yield_st': self.config.get('cush_yield_strain', 0.05),
                    'base_rgba': self.model.geom_rgba[gi].copy(), # 노란색을 베이스 컬러로 저장
                    'plastic_rgba': [1.0, 0.0, 0.0, 1.0], # 찌그러질 때 빨간색(Red)으로 변형
                    'target_size': self.original_geom_size[gi].copy() # [WHTOOLS] 목표 크기
                }

    def _physics_control_callback(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """MuJoCo 제어 루프에서 매 스텝 호출되는 물리 콜백 함수입니다."""
        self._apply_aerodynamics(model, data)

    def _apply_aerodynamics_backup(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """[BACKUP] 기존 순수 파이썬 로직 기반 공기역학 (구버전)"""
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

    def _apply_aerodynamics(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """[OPTIMIZED] Numba JIT 기반 공기역학 가속 (파이썬 루프 최소화)"""
        if self.root_id == -1: return
        cfg = self.config
        en_drag = cfg.get('enable_air_drag', True)
        en_sq = cfg.get('enable_air_squeeze', False)
        if not en_drag and not en_sq: return
        
        v_linear = data.cvel[self.root_id][3:6]
        bw, bh, bd = cfg.get('box_w', 2.0), cfg.get('box_h', 1.4), cfg.get('box_d', 0.25)
        total_area = 2 * (bw * bh + bh * bd + bd * bw)
        z_gap = data.xpos[self.root_id][2] - (bd / 2.0)
        
        f_drag, f_visc, f_sq = _numba_calc_aero(
            v_linear, z_gap,
            cfg.get('air_density', 1.225), cfg.get('air_drag_coeff', 1.05), total_area,
            cfg.get('air_viscosity', 1.8e-5), cfg.get('air_cd_viscous', 0.0),
            cfg.get('air_squeeze_hmax', 0.1), cfg.get('air_squeeze_hmin', 0.001),
            cfg.get('air_coef_squeeze', 1.0),
            en_drag, en_sq
        )
        
        self._last_f_drag = f_drag
        self._last_f_visc = f_visc
        self._last_f_sq = f_sq
        data.xfrc_applied[self.root_id][2] = f_drag + f_visc + f_sq

    def _apply_plasticity_v2_backup(self) -> None:
        """[BACKUP] 접촉 압력을 기반으로 한 정밀 소성 변형 로직입니다 (기존 순수 파이썬 루프)."""
        if not self.config.get("enable_plasticity", False): return
        d, m = self.data, self.model
        p_ratio = self.config.get("plasticity_ratio", 0.5)
        
        active_geoms = set()
        for c_idx in range(d.ncon):
            contact = d.contact[c_idx]
            for g_id in [contact.geom1, contact.geom2]:
                if g_id in self.geom_state_tracker:
                    active_geoms.add(g_id)
                    state = self.geom_state_tracker[g_id]
                    
                    strains = [max(0.0, (self.original_geom_size[g_id][ax] - m.geom_size[g_id][ax]) / self.original_geom_size[g_id][ax]) for ax in range(3)]
                    equiv_strain = np.linalg.norm(strains)
                    
                    lx = d.geom_xmat[g_id].reshape(3,3).T @ contact.frame[:3]
                    ax = int(np.argmax(np.abs(lx)))
                    other_axes = [i for i in range(3) if i != ax]
                    area = m.geom_size[g_id][other_axes[0]] * m.geom_size[g_id][other_axes[1]] * 4.0
                    
                    yield_pr = self.config.get("cush_yield_pressure", 1000.0) + \
                                (self.config.get("plastic_hardening_modulus", 0.0) * equiv_strain)
                    
                    force = np.zeros(6)
                    mujoco.mj_contactForce(m, d, c_idx, force)
                    pressure = abs(force[0]) / (area + 1e-9)
                    
                    if pressure > yield_pr:
                        excess_mpa = (pressure - yield_pr) / 1e6
                        flow_rate = excess_mpa * 15.0 * m.opt.timestep * self.original_geom_size[g_id][ax]
                        min_allowed = self.original_geom_size[g_id][ax] * (1.0 - self.config.get("plastic_max_strain", 0.5))
                        state['target_size'][ax] = max(min_allowed, state['target_size'][ax] - flow_rate)
                    
                    self.max_applied_pressure_pa = max(self.max_applied_pressure_pa, pressure)
                    self.max_plastic_strain = max(self.max_plastic_strain, equiv_strain)
                    self.max_equiv_strain = max(self.max_equiv_strain, equiv_strain)

        for g_id, state in self.geom_state_tracker.items():
            for ax in range(3):
                if m.geom_size[g_id][ax] > state['target_size'][ax]:
                    diff = m.geom_size[g_id][ax] - state['target_size'][ax]
                    k = p_ratio * 50.0
                    reduction_step = diff * k * m.opt.timestep
                    m.geom_size[g_id][ax] -= min(diff, reduction_step)
            
            for i in range(3):
                def_mm = (self.original_geom_size[g_id][i] - m.geom_size[g_id][i]) * 1000.0
                self.max_deformation_mm = max(self.max_deformation_mm, def_mm)
            
            strains = [max(0.0, (self.original_geom_size[g_id][ax] - m.geom_size[g_id][ax]) / self.original_geom_size[g_id][ax]) for ax in range(3)]
            equiv_strain = np.linalg.norm(strains)
            sn = np.clip(equiv_strain / self.config.get("plastic_color_limit", 0.1), 0.0, 1.0)
            if self.config.get("enable_plasticity_color", True):
                base_color = np.array(state['base_rgba'])
                plastic_color = np.array([1.0, 0.0, 0.0, 1.0])  # Red
                m.geom_rgba[g_id] = base_color * (1.0 - sn) + plastic_color * sn

    def _apply_plasticity_v2(self) -> None:
        """[OPTIMIZED] Numpy 벡터화 기반 소성 변형 가속 (파이썬 루프 제거)"""
        if not self.config.get("enable_plasticity", False): return
        d, m = self.data, self.model
        p_ratio = self.config.get("plasticity_ratio", 0.5)
        
        tracked_gids = np.array(list(self.geom_state_tracker.keys()), dtype=np.int32)
        if len(tracked_gids) == 0: return
        
        if d.ncon > 0:
            contacts = d.contact.geom[:d.ncon] # (ncon, 2)
            # [WHTOOLS] 소성 변형 대상 지오메트리에 대한 접촉이 없는 경우 조기 반환(Early Exit)하여 파이썬 루프 방지
            intersect = np.intersect1d(contacts, tracked_gids)
            if len(intersect) == 0:
                return
                
            valid_c_idx = np.where(np.isin(contacts[:, 0], tracked_gids) | np.isin(contacts[:, 1], tracked_gids))[0]
            
            for c_idx in valid_c_idx:
                c_geom = d.contact.geom[c_idx]
                for g_id in c_geom:
                    if g_id in self.geom_state_tracker:
                        state = self.geom_state_tracker[g_id]
                        sizes = m.geom_size[g_id]
                        orig_sizes = self.original_geom_size[g_id]
                        
                        strains = np.maximum(0.0, (orig_sizes - sizes) / orig_sizes)
                        equiv_strain = np.linalg.norm(strains)
                        
                        lx = d.geom_xmat[g_id].reshape(3,3).T @ d.contact.frame[c_idx, :3]
                        ax = int(np.argmax(np.abs(lx)))
                        
                        area = sizes[(ax+1)%3] * sizes[(ax+2)%3] * 4.0
                        
                        yield_pr = self.config.get("cush_yield_pressure", 1000.0) + \
                                    (self.config.get("plastic_hardening_modulus", 0.0) * equiv_strain)
                        
                        force = np.zeros(6)
                        mujoco.mj_contactForce(m, d, c_idx, force)
                        pressure = abs(force[0]) / (area + 1e-9)
                        
                        if pressure > yield_pr:
                            excess_mpa = (pressure - yield_pr) / 1e6
                            flow_rate = excess_mpa * 15.0 * m.opt.timestep * orig_sizes[ax]
                            min_allowed = orig_sizes[ax] * (1.0 - self.config.get("plastic_max_strain", 0.5))
                            state['target_size'][ax] = max(min_allowed, state['target_size'][ax] - flow_rate)
                        
                        self.max_applied_pressure_pa = max(self.max_applied_pressure_pa, float(pressure))
                        self.max_plastic_strain = max(self.max_plastic_strain, float(equiv_strain))
                        self.max_equiv_strain = max(self.max_equiv_strain, float(equiv_strain))

        # 브로드캐스팅 수렴 및 통계
        for g_id, state in self.geom_state_tracker.items():
            for ax in range(3):
                if m.geom_size[g_id][ax] > state['target_size'][ax]:
                    diff = m.geom_size[g_id][ax] - state['target_size'][ax]
                    k = p_ratio * 50.0
                    reduction_step = diff * k * m.opt.timestep
                    m.geom_size[g_id][ax] -= min(diff, reduction_step)
                    # 원본 크기의 1% 미만으로 줄어들지 않도록 클램프
                    m.geom_size[g_id][ax] = max(self.original_geom_size[g_id][ax] * 0.01,
                                                m.geom_size[g_id][ax])
            
            for i in range(3):
                def_mm = (self.original_geom_size[g_id][i] - m.geom_size[g_id][i]) * 1000.0
                self.max_deformation_mm = max(self.max_deformation_mm, def_mm)
            
            strains = [max(0.0, (self.original_geom_size[g_id][ax] - m.geom_size[g_id][ax]) / self.original_geom_size[g_id][ax]) for ax in range(3)]
            equiv_strain = np.linalg.norm(strains)
            sn = np.clip(equiv_strain / self.config.get("plastic_color_limit", 0.1), 0.0, 1.0)
            if self.config.get("enable_plasticity_color", True):
                base_color = np.array(state['base_rgba'])
                plastic_color = np.array([1.0, 0.0, 0.0, 1.0])  # Red
                m.geom_rgba[g_id] = base_color * (1.0 - sn) + plastic_color * sn

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
        
        # 충격력 및 공기저항 기록 (contact.frame은 법선 방향벡터이므로 mj_contactForce로 실제 힘을 사용)
        total_impact = 0.0
        _cf = np.zeros(6)
        for ci in range(d.ncon):
            mujoco.mj_contactForce(self.model, d, ci, _cf)
            total_impact += np.linalg.norm(_cf[:3])
        self.ground_impact_hist.append(total_impact)
        self.air_drag_hist.append(self._last_f_drag)
        self.air_squeeze_hist.append(self._last_f_sq)
        
        # 코너 기구학 데이터 계산 (Cushion 기준 - 하위 호환성 유지)
        bw, bh, bd = self.config.get('box_w', 2.0), self.config.get('box_h', 1.4), self.config.get('box_d', 0.25)
        ck = self._get_discrete_corner_kinematics("cushion", bw, bh, bd)
        
        self.corner_pos_hist.append([c['pos'] for c in ck])
        self.corner_vel_hist.append([c['vel'] for c in ck])
        self.corner_acc_hist.append([c['acc'] for c in ck])
        self.geo_center_pos_hist.append(np.mean([c['pos'] for c in ck], axis=0))
        
        # 결과값(Resultant) 계산 및 저장
        self.corner_pos_res_hist.append(np.array([np.linalg.norm(c['pos']) for c in ck]))
        self.corner_vel_res_hist.append(np.array([np.linalg.norm(c['vel']) for c in ck]))
        self.corner_acc_res_hist.append(np.array([np.linalg.norm(c['acc']) for c in ck]))

        # [WHTOOLS] 파트별(다중) 코너 기구학 계산 추가 (개별 코너 블럭 기반)
        parts_info = {
            "Cushion": (bw, bh, bd),
            "Chassis": (self.config.get('assy_w', 1.892), self.config.get('assy_h', 1.082), self.config.get('chassis_d', 0.035)),
            "OpenCell": (self.config.get('assy_w', 1.892), self.config.get('assy_h', 1.082), self.config.get('opencell_d', 0.012))
        }
        
        for part_name, (w, h, d_val) in parts_info.items():
            p_ck = self._get_discrete_corner_kinematics(part_name, w, h, d_val)
            ph = self.part_corner_hist[part_name]
            ph["pos"].append([c['pos'] for c in p_ck])
            ph["vel"].append([c['vel'] for c in p_ck])
            ph["acc"].append([c['acc'] for c in p_ck])
            ph["pos_res"].append(np.array([np.linalg.norm(c['pos']) for c in p_ck]))
            ph["vel_res"].append(np.array([np.linalg.norm(c['vel']) for c in p_ck]))
            ph["acc_res"].append(np.array([np.linalg.norm(c['acc']) for c in p_ck]))

        # [WHTOOLS] Cushion-Rigid 계산 추가 (전체 통강체 중심 기준)
        from .whts_utils import compute_corner_kinematics
        cr_ck = compute_corner_kinematics(d.xpos[rid], d.xmat[rid].reshape(3,3), d.cvel[rid], d.cacc[rid], bw, bh, bd)
        ph_cr = self.part_corner_hist["Cushion-Rigid"]
        ph_cr["pos"].append([c['pos'] for c in cr_ck])
        ph_cr["vel"].append([c['vel'] for c in cr_ck])
        ph_cr["acc"].append([c['acc'] for c in cr_ck])
        ph_cr["pos_res"].append(np.array([np.linalg.norm(c['pos']) for c in cr_ck]))
        ph_cr["vel_res"].append(np.array([np.linalg.norm(c['vel']) for c in cr_ck]))
        ph_cr["acc_res"].append(np.array([np.linalg.norm(c['acc']) for c in cr_ck]))

        # [WHTOOLS] 코너별 바닥 충격력 추정 (가장 가까운 코너로 힘 할당)
        corner_impacts = np.zeros(8)
        corners = [c['pos'] for c in ck]
        for i in range(d.ncon):
            f_vec = np.zeros(6)
            mujoco.mj_contactForce(self.model, d, i, f_vec)
            f_mag = np.linalg.norm(f_vec[:3])
            # 가장 가까운 코너 탐색 (임계값 0.3m)
            dists = [np.linalg.norm(d.contact[i].pos - cp) for cp in corners]
            nearest = np.argmin(dists)
            if dists[nearest] < 0.3:
                corner_impacts[nearest] += f_mag
        self.corner_impact_hist.append(corner_impacts)

        # [WHTOOLS] 강체 거동 대표 물리량 계산 (Instantaneous Axis of Rotation & Translation)
        # MuJoCo cvel: [0:3] angular velocity, [3:6] linear velocity (Cartesian)
        cvel_6d = d.cvel[rid]
        omega = cvel_6d[:3]
        v_trans = cvel_6d[3:]
        
        rot_speed = np.linalg.norm(omega)
        rot_axis = omega / rot_speed if rot_speed > 1e-8 else np.zeros(3)
        
        self.rot_axis_hist.append(rot_axis)
        self.rot_speed_hist.append(rot_speed)
        self.trans_vel_hist.append(v_trans)
        self.trans_vel_res_hist.append(np.linalg.norm(v_trans))

    @property
    def app_instance(self):
        """PySide6 QApplication 인스턴스를 반환합니다."""
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if not app:
            app = QApplication([])
        return app

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


    def start_viewer(self) -> None:
        """MuJoCo 뷰어를 비차단(Non-blocking) 방식으로 실행합니다."""
        self.stop_viewer() # 기존 뷰어가 있다면 종료
        
        self.viewer = mujoco.viewer.launch_passive(
            self.model, self.data, 
            key_callback=self._on_key,
            show_left_ui=False, show_right_ui=False
        )
        
        # 초기 카메라 설정
        self.viewer.cam.lookat[:] = [-0.0295, -0.3909, 0.8668]
        self.viewer.cam.distance = 5.8341
        self.viewer.cam.elevation = -9.88
        self.viewer.cam.azimuth = 136.50
        self.viewer.sync()
        self.log("🌐 MuJoCo Viewer Started.")

    def stop_viewer(self) -> None:
        """MuJoCo 뷰어를 안전하게 종료합니다."""
        if hasattr(self, 'viewer') and self.viewer:
            try:
                self.viewer.close()
            except:
                pass
            self.viewer = None
            self.log("🌐 MuJoCo Viewer Closed.")

    def _restart_sim_thread(self) -> None:
        """시뮬레이션 스레드를 (재)시작합니다. 기존 스레드가 살아있으면 종료를 기다립니다."""
        if hasattr(self, 'sim_thread') and self.sim_thread.isRunning():
            self.sim_thread.wait(3000)
        self.sim_thread = SimThread(self)
        self.sim_thread.start()

    def _launch_with_control_panel(self) -> None:
        """제어 패널과 함께 시뮬레이션을 실행합니다."""
        import signal
        from PySide6.QtCore import QTimer

        # 1. 모델 준비
        self.setup()

        # 2. Control Center UI 먼저 생성 (메인 스레드)
        self.app, self.panel = launch_control_panel(self)

        # Ctrl+C (SIGINT) 처리: Qt는 기본적으로 Python SIGINT를 무시하므로
        # 타이머로 주기적으로 Python 이벤트를 처리할 기회를 부여하고,
        # SIGINT 핸들러에서 Qt 앱을 정상 종료시킵니다.
        def _on_sigint(*_):
            self.log("🛑 Ctrl+C received — shutting down.", level="info")
            self.ctrl_quit_request = True
            try: self.stop_viewer()
            except Exception: pass
            if hasattr(self, 'panel'):
                self.panel.close()
            self.app.quit()

        signal.signal(signal.SIGINT, _on_sigint)
        # Qt 이벤트 루프가 SIGINT를 받을 수 있도록 100ms마다 Python으로 제어 반환
        _sigint_timer = QTimer()
        _sigint_timer.setInterval(100)
        _sigint_timer.timeout.connect(lambda: None)
        _sigint_timer.start()

        # 3. Passive viewer 열기
        self.start_viewer()

        # 4. 물리 스레드 시작
        self._restart_sim_thread()

        # 5. UI 이벤트 루프 (창 닫힐 때까지 블록)
        self.app.exec()

        # 6. 종료 처리: sim_thread가 _wrap_up(build_and_save_result 포함)을 완료할 때까지 대기
        self.ctrl_quit_request = True
        self.stop_viewer()
        if hasattr(self, 'sim_thread') and self.sim_thread.isRunning():
            # _wrap_up + build_and_save_result 완료까지 충분히 대기 (최대 30초)
            self.sim_thread.wait(30000)
            if self.sim_thread.isRunning():
                self.log("⚠️ sim_thread did not finish in time, terminating.", level="warning")
                self.sim_thread.terminate()

    def _run_engine(self) -> None:
        """시뮬레이션 루프를 실행합니다. 뷰어는 이미 메인 스레드에서 실행 중입니다."""
        self._main_loop()

    def _main_loop(self) -> None:
        """실제 시뮬레이션 타임스텝을 진행하는 핵심 루프입니다. (단일 세션)"""
        # 세션 파라미터 초기화
        self.step_idx = 0
        total_steps = int(self.config.get("sim_duration", 1.0) / self.model.opt.timestep)
        report_step = max(1, int(self.config.get("reporting_interval", 0.005) / self.model.opt.timestep))

        # 초기 상태 저장 및 동기화
        self.snapshots = []
        self._reset_loop_variables()  # geom_state_tracker 등 setup()이 채운 데이터는 보존
        self._init_histories()
        mujoco.mj_forward(self.model, self.data)
        self._collect_history() # [WHTOOLS] time=0 시점 초기값 기록
        self._save_snapshot()

        self.log(f"🎬 Simulation Session Started. Target Duration: {self.config.get('sim_duration', 1.0)}s")

        # [WHTOOLS] 120 Hz 렌더링 스로틀링(Throttling)을 위한 실시간 시계 변수 초기화
        import time
        last_render_time = time.perf_counter()
        render_interval = 1.0 / 120.0  # 120 FPS (약 8.33ms)

        # 물리 연산 루프 (quit 또는 reload 요청 시 탈출)
        while not self.ctrl_quit_request and not self.ctrl_reload_request:
            # 뷰어 종료 감지
            if self.viewer and not self.viewer.is_running():
                self.ctrl_quit_request = True
                break

            # UI 요청 처리 (Step, Reset, Jump 등)
            self._handle_ui_requests()

            if not self.ctrl_paused:
                # 과거 시점에서 다시 시작하려 할 경우, 미래 데이터 절단
                self._check_and_truncate_future()

                # 1. Physics Step
                mujoco.mj_step(self.model, self.data)
                self.step_idx += 1 # [WHTOOLS] 물리 연산 완료 후 step_idx 즉시 증가

                # 2. Advanced Physics Post-step
                self._apply_plasticity_v2()

                # 3. Data Collection & Snapshotting
                if self.step_idx % report_step == 0:
                    dynamic_target_time = self.config.get("sim_duration", 1.0)
                    dynamic_total_steps = int(dynamic_target_time / self.model.opt.timestep)
                    if self.step_idx <= dynamic_total_steps or self.is_recording:
                        self._collect_history()
                    self._save_snapshot()

                # 4. Progress Reporting
                self._report_progress(self.step_idx)

                # [WHTOOLS] 동적으로 sim_duration을 재계산 (UI 실시간 변경 지원)
                dynamic_target_time = self.config.get("sim_duration", 1.0)
                dynamic_total_steps = int(dynamic_target_time / self.model.opt.timestep)

                # 타겟 도달 시 자동 일시 정지 (viewer 모드) 또는 종료 (headless)
                if self.step_idx == dynamic_total_steps:
                    self.log("✅ [DATA COLLECTION COMPLETE] Target simulation time reached.", level="info")
                    self.log(f"📊 Collected {len(self.pos_hist)} frames up to {self.data.time:.3f}s", level="info")
                    
                    # 목표 시간 도달 즉시 결과 저장 및 컴포넌트 정보 처리
                    self.build_and_save_result()
                    
                    if self.config.get("use_viewer", False):
                        self.ctrl_paused = True
                        self.log("✅ Paused for review.", level="info")
                        self.log("💡 [Tip] Press 'Play' to continue or 'L' to record more data.", level="debug")
                    else:
                        self.log("✅ Finishing simulation.", level="info")
                        self.ctrl_quit_request = True
                if self.viewer and self.viewer.is_running():
                    # [WHTOOLS] 매 물리 스텝마다의 GUI 동기화 오버헤드를 막기 위해 120 Hz 스로틀링 적용
                    curr_time = time.perf_counter()
                    if curr_time - last_render_time >= render_interval:
                        self.viewer.sync()
                        last_render_time = curr_time

                # 속도 제어 (Speed Multiplier 및 Slow Motion 적용)
                effective_multiplier = self.ctrl_speed_multiplier
                if self.ctrl_slow_motion:
                    effective_multiplier *= 0.2
                if effective_multiplier != 1.0:
                    base_sleep = self.model.opt.timestep / effective_multiplier
                    if base_sleep > 0.0001:
                        time.sleep(base_sleep)
            else:
                if self._report_count > 0:
                    self._print_border()
                    self._report_count = -1
                if self.viewer:
                    self.viewer.sync()
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

        # 2-1. 전체 리셋 요청 (Frame 0으로 이동 및 히스토리 초기화)
        if self.ctrl_reset_request:
            self._reset_simulation()
            self.ctrl_paused = True # 리셋 후 정지 상태 유지
            self.ctrl_reset_request = False

        # 3. 특정 스냅샷으로 점프
        if self.ctrl_jump_snapshot_idx != -1:
            self._jump_to_snapshot(self.ctrl_jump_snapshot_idx)
            self.ctrl_jump_snapshot_idx = -1

        # 4. 카메라 XML 정보 출력
        if self.ctrl_export_camera:
            self._export_camera_xml()
            self.ctrl_export_camera = False

        # 5. [WHTOOLS] MuJoCo 카메라 시점 전환 처리
        if self.ctrl_cam_view and self.viewer:
            cv = self.ctrl_cam_view
            cam = self.viewer.cam
            if cv == "+X":   cam.azimuth, cam.elevation = 0, 0
            elif cv == "-X": cam.azimuth, cam.elevation = 180, 0
            elif cv == "+Y": cam.azimuth, cam.elevation = 90, 0
            elif cv == "-Y": cam.azimuth, cam.elevation = 270, 0
            elif cv == "+Z": cam.azimuth, cam.elevation = 0, -90
            elif cv == "-Z": cam.azimuth, cam.elevation = 0, 90
            elif cv == "+ISO": cam.azimuth, cam.elevation = 45, -35
            elif cv == "-ISO": cam.azimuth, cam.elevation = 225, -35
            
            self.ctrl_cam_view = None
            self.viewer.sync()
            self.log(f"📸 Camera orientation switched to: {cv}")

    def _export_camera_xml(self) -> None:
        """
        현재 MuJoCo 뷰어의 카메라 파라미터(lookat, distance, elevation, azimuth)를 
        XML 포맷으로 추출하여 콘솔에 출력하고 파일로 저장합니다.
        """
        if not self.viewer:
            self.log("⚠️ Viewer is not active. Cannot export camera.", level="warning")
            return
            
        cam = self.viewer.cam
        pos = cam.lookat
        dist = cam.distance
        elev = cam.elevation
        azim = cam.azimuth
        
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

    def reload_xml(self, xml_path: Optional[str] = None, xml_string: Optional[str] = None) -> None:
        """
        특정 XML 파일 또는 XML 문자열을 로드하여 시뮬레이션을 재시작합니다.
        """
        if xml_string:
            # 에디터에서 수정한 XML을 output_dir의 고정 경로에 저장 (temp 파일 누적 방지)
            live_edit_path = self.output_dir / "simulation_model_live_edit.xml"
            live_edit_path.write_text(xml_string, encoding='utf-8')
            xml_path = str(live_edit_path)
            self.log(f"♻️ Reloading from live-edited XML: {xml_path}")
        elif xml_path is None:
            # 파일 선택 다이얼로그
            from PySide6.QtWidgets import QFileDialog
            selected, _ = QFileDialog.getOpenFileName(
                None, "Select MuJoCo Simulation XML",
                str(self.output_dir), "MuJoCo XML (*.xml);;All files (*.*)"
            )
            if not selected:
                self.log("🚫 Reload cancelled: No file selected.")
                return
            xml_path = selected
            self.log(f"♻️ Reloading from file: {xml_path}")
        else:
            self.log(f"♻️ Reloading from file: {xml_path}")

        self.ctrl_reload_xml_path = xml_path
        self.ctrl_reload_only_xml = True
        self.ctrl_reload_request = True

    def _jump_to_snapshot(self, idx: int) -> None:
        """저장된 스냅샷 리스트에서 특정 인덱스의 상태로 시뮬레이션을 되돌립니다."""
        if 0 <= idx < len(self.snapshots):
            snapshot = self.snapshots[idx]
            mujoco.mj_setState(self.model, self.data, snapshot['state'], mujoco.mjtState.mjSTATE_PHYSICS)
            self.data.time = snapshot['time']  # [WHTOOLS] 물리적 시간 필드 강제 복구
            mujoco.mj_forward(self.model, self.data)
            self.step_idx = snapshot['step_idx']
            
            # 2. 모델 파라미터 (소성 변형) 복구
            if 'geom_size' in snapshot:
                self.model.geom_size[:] = snapshot['geom_size']
            if 'geom_rgba' in snapshot:
                self.model.geom_rgba[:] = snapshot['geom_rgba']
            
            # [WHTOOLS] 소성 목표치(target_size) 복구 - 리셋 후 즉시 재변형 방지
            if 'plastic_targets' in snapshot:
                for g_id, t_size in snapshot['plastic_targets'].items():
                    if g_id in self.geom_state_tracker:
                        self.geom_state_tracker[g_id]['target_size'] = t_size.copy()

            # [WHTOOLS] Rank Heatmap
            if self.config.get('enable_rank_heatmap', False):
                apply_rank_heatmap(self)
            
            # 3. 통계 데이터 초기화 (0번으로 돌아갈 때만 완전 초기화)
            if idx == 0:
                self.max_equiv_strain = 0.0
                self.max_applied_pressure_pa = 0.0
                self.max_plastic_strain = 0.0
                self.max_deformation_mm = 0.0
                self._last_reported_interval = -1
            
            self.log(f"🚀 Jumped to Snapshot {idx} (Time: {snapshot['time']:.3f}s)")
            if self.viewer: self.viewer.sync()

    def _check_and_truncate_future(self) -> None:
        """현재 step_idx가 저장된 최신 스냅샷보다 과거라면 미래 데이터를 삭제합니다."""
        if len(self.snapshots) == 0:
            return
        last_snap = self.snapshots[-1]
        if self.step_idx >= last_snap['step_idx']:
            return

        # step_idx 이하인 스냅샷 중 가장 가까운 것(마지막)을 절단 기준으로 사용
        cut_i = None
        for i, snap in enumerate(self.snapshots):
            if snap['step_idx'] <= self.step_idx:
                cut_i = i
        if cut_i is not None:
            self._truncate_histories(self.snapshots[cut_i]['hist_len'])
            self.snapshots = self.snapshots[:cut_i + 1]
            self.log(f"✂️ Future truncated at snapshot {cut_i} (step {self.snapshots[-1]['step_idx']}).")
        else:
            self.log("⚠️ _check_and_truncate_future: no matching snapshot found, skipping truncation.",
                     level="warning")

    def _get_discrete_corner_kinematics(self, part_name: str, w: float, h: float, d_val: float) -> List[Dict[str, np.ndarray]]:
        """
        [WHTOOLS] 각 파트의 8개 코너 방향에 실제 위치한 개별 블럭(Corner Block)을 찾고,
        그 블럭의 가장 바깥쪽 끝점(Corner Point)의 글로벌 거동을 계산합니다.
        """
        d = self.data
        comp_key = part_name.lower()
        
        if comp_key not in self.components or not self.components[comp_key]:
            # 구성 요소가 없거나 단일 바디가 아닌 경우 fallback: 루트 바디(Chassis) 기준 계산
            rid = self.root_id
            from .whts_utils import compute_corner_kinematics
            return compute_corner_kinematics(d.xpos[rid], d.xmat[rid].reshape(3,3), d.cvel[rid], d.cacc[rid], w, h, d_val)
        
        comp_bodies = self.components[comp_key]
        body_ids = list(comp_bodies.values())
        body_xpos = d.xpos[body_ids]
        
        # 8개 코너 방향 정의 (whts_utils.compute_corner_kinematics와 순서 일치)
        # [- - -], [- - +], [- + -], [- + +], [+ - -], [+ - +], [+ + -], [+ + +]
        signs = []
        for sx in [-1, 1]:
            for sy in [-1, 1]:
                for sz in [-1, 1]:
                    signs.append([sx, sy, sz])
        
        results = []
        for sx, sy, sz in signs:
            # 1. 해당 코너 방향에 가장 치우친 블럭 탐색 (투영 점수 기반)
            # Score = sx*x + sy*y + sz*z
            scores = body_xpos[:, 0] * sx + body_xpos[:, 1] * sy + body_xpos[:, 2] * sz
            best_idx = np.argmax(scores)
            b_id = body_ids[best_idx]
            
            # 2. 해당 블럭의 실제 기하 사이즈(Half-extents) 획득
            hw, hh, hd = self.block_half_extents.get(b_id, [0.0, 0.0, 0.0])
            loc_corner = np.array([sx * hw, sy * hh, sz * hd])
            
            # 3. 강체 기구학 공식을 이용한 끝점 거동 역산
            r = d.xmat[b_id].reshape(3,3) @ loc_corner
            w_vel = d.cvel[b_id][0:3]
            v_vel = d.cvel[b_id][3:6]
            alpha = d.cacc[b_id][0:3]
            a_acc = d.cacc[b_id][3:6]
            
            v_corner = v_vel + np.cross(w_vel, r)
            a_corner = a_acc + np.cross(alpha, r) + np.cross(w_vel, np.cross(w_vel, r))
            
            results.append({
                'pos': d.xpos[b_id] + r,
                'vel': v_corner,
                'acc': a_corner
            })
        return results

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
        
        self.corner_pos_res_hist = self.corner_pos_res_hist[:h_idx]
        self.corner_vel_res_hist = self.corner_vel_res_hist[:h_idx]
        self.corner_acc_res_hist = self.corner_acc_res_hist[:h_idx]
        self.corner_impact_hist = self.corner_impact_hist[:h_idx]
        
        self.rot_axis_hist = self.rot_axis_hist[:h_idx]
        self.rot_speed_hist = self.rot_speed_hist[:h_idx]
        self.trans_vel_hist = self.trans_vel_hist[:h_idx]
        self.trans_vel_res_hist = self.trans_vel_res_hist[:h_idx]
        
        if hasattr(self, 'part_corner_hist'):
            for part in self.part_corner_hist.values():
                for k in part.keys():
                    part[k] = part[k][:h_idx]
        
        # 구조적 시계열 데이터 초기화
        if hasattr(self, 'structural_time_series'):
            for k in ['rrg_max', 'mean_distortion']:
                if k in self.structural_time_series:
                    self.structural_time_series[k] = self.structural_time_series[k][:h_idx]

    def _save_snapshot(self) -> None:
        """현재의 MuJoCo 물리 상태와 히스토리 포인터를 스냅샷으로 저장합니다."""
        # 타겟 시뮬레이션 시간(sim_duration) 내의 모든 step을 커버할 수 있도록 동적 한도 설정 (20% 여유, 최소 1000개)
        try:
            target_steps = int((self.config.get("sim_duration", 1.0) / self.model.opt.timestep) * 1.2)
        except Exception:
            target_steps = 2000
        snapshot_limit = max(1000, target_steps)
        
        if len(self.snapshots) >= snapshot_limit:
            self.snapshots.pop(1)  # index 0(초기 상태)는 Reset 기준점으로 보존
            
        state = np.zeros(mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_PHYSICS))
        mujoco.mj_getState(self.model, self.data, state, mujoco.mjtState.mjSTATE_PHYSICS)
        
        self.snapshots.append({
            'time': self.data.time,
            'step_idx': self.step_idx,
            'state': state,
            'geom_size': self.model.geom_size.copy(),
            'geom_rgba': self.model.geom_rgba.copy(),
            'plastic_targets': {g_id: state['target_size'].copy() for g_id, state in self.geom_state_tracker.items()},
            'hist_len': len(self.time_history)
        })

    def _rewind_snapshot(self) -> None:
        """가장 최근의 스냅샷으로 시뮬레이션을 1단계 되돌립니다(Undo)."""
        if len(self.snapshots) <= 1:
            self.log("⚠️ No snapshots available to rewind.", level="warning")
            return
            
        self.snapshots.pop()
        snapshot = self.snapshots[-1]
        
        mujoco.mj_setState(self.model, self.data, snapshot['state'], mujoco.mjtState.mjSTATE_PHYSICS)
        self.data.time = snapshot['time']  # [WHTOOLS] 물리적 시간 필드 강제 복구
        mujoco.mj_forward(self.model, self.data)
        self.step_idx = snapshot['step_idx']
        
        if 'geom_size' in snapshot:
            self.model.geom_size[:] = snapshot['geom_size']
        if 'geom_rgba' in snapshot:
            self.model.geom_rgba[:] = snapshot['geom_rgba']
        
        self._truncate_histories(snapshot['hist_len'])
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
        console.print(f"[bold white]{'━' * 128}[/bold white]")

    def _wrap_up(self) -> None:
        """시뮬레이션 종료 후 데이터를 정리하고 결과를 저장하며 UI를 호출합니다."""
        # 콜백 해제 — 이 인스턴스가 GC 된 후에도 전역 콜백이 남지 않도록 정리
        mujoco.set_mjcb_control(None)

        if self.data is None:
            self.log("⚠️ _wrap_up skipped: model/data not initialized (setup failed)", level="warning")
            return

        self.log("🏁 Simulation Finished. Wrapping up data...", level="info")

        target_time = self.config.get("sim_duration", 1.0)
        curr_time = self.data.time
        is_complete = curr_time >= (target_time - 1e-5)
        
        status_msg = "COMPLETE ✅" if is_complete else f"INCOMPLETE ⚠️ ({curr_time:.3f}/{target_time:.3f}s)"
        self.log(f"📑 Data Status: {status_msg}")
        self.log(f"🎞️ Total Frames: {len(self.pos_hist)}")
        self._print_border()
        
        n_bodies = sum(len(v) for v in self.components.values())
        self.log(f"📦 Components at wrap-up: {list(self.components.keys())} | total mapped bodies: {n_bodies}", level="info")
        try:
            self.build_and_save_result()
        except Exception as e:
            self.log(f"Error during wrap-up: {e}", level="error")

        # 사용자가 창을 닫아 종료한 경우 UI 후처리만 생략 (결과 저장은 완료)
        if self.ctrl_quit_request:
            self.log("🛑 Post-process UI skipped: quit requested by user.", level="info")
            return

    def build_and_save_result(self) -> None:
        if self.result is not None:
            return

        self.log("🏁 Building and saving simulation results...", level="info")
        try:
            compute_batch_structural_metrics(self)
            finalize_simulation_results(self)
            if self.config.get("enable_rank_heatmap", False):
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
                block_half_extents=self.block_half_extents,
                part_corner_hist=self.part_corner_hist
            )
            
            result_path = self.output_dir / "simulation_result.pkl"
            self.result.save(str(result_path))
            self.log(f"💾 Results saved to: {result_path}")

        except Exception as e:
            self.log(f"Error during result building: {e}", level="error")

    def _launch_postprocess(self) -> None:
        pass
        
    def get_output_dir(self) -> Path:
        """[WHTOOLS] pkl 파일이 저장된 폴더 경로를 반환합니다."""
        return self.output_dir
    
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
        elif keycode == 8 or keycode == 259: # Backspace: Reset to Start
            self.ctrl_reset_request = True
        elif keycode == 263: # Left Arrow: Step Backward
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

    def export_radioss(self, frame_idx: int = None, target_time: float = None) -> tuple:
        """현재 상태 또는 지정된 프레임/시간의 상태에서 OpenRadioss 모델을 생성합니다."""
        from pathlib import Path
        from .whts_radioss_builder import RadiossModelBuilder
        import numpy as np
        import mujoco

        rid = getattr(self, 'root_id', -1)
        
        # 만약 시뮬레이션 종료 후 결과가 있다면 result 객체에서 추출
        if hasattr(self, 'result') and self.result is not None and getattr(self.result, "quat_hist", None) is not None:
            if rid < 0 and getattr(self.result, "body_index_map", None):
                rid = self.result.body_index_map.get("BPackagingBox", -1)
            if rid < 0:
                self.log("❌ OpenRadioss: Could not find valid root_id.", level="error")
                return None, None
                
            # 프레임 결정
            if frame_idx is None and target_time is not None:
                times = np.array(self.result.time_history)
                frame_idx = int(np.argmin(np.abs(times - target_time)))
                self.log(f"  [Radioss] Target time: {target_time}s -> Extracted at frame {frame_idx} (time: {times[frame_idx]:.4f}s)")
            elif frame_idx is None:
                frame_idx = -1 # 가장 마지막 상태

            try:
                quat_mj = self.result.quat_hist[frame_idx][rid]
                R_flat = np.zeros(9)
                mujoco.mju_quat2Mat(R_flat, quat_mj)
                R_mat = R_flat.reshape(3, 3)
                t_vec = self.result.pos_hist[frame_idx][rid].copy()
                
                cvel = self.result.vel_hist[frame_idx]
                omega_vec = cvel[:3]
                v_vec = cvel[3:]
            except Exception as e:
                self.log(f"⚠️ Could not extract pose for Radioss at frame {frame_idx}: {e}", level="warning")
                return None, None
        else:
            # 실시간 데이터에서 직접 추출 (UI에서 현재 상태 생성 시)
            if rid < 0 or self.data is None:
                self.log("❌ OpenRadioss: 시뮬레이션 데이터가 없습니다.", level="error")
                return None, None
            R_mat = self.data.xmat[rid].reshape(3, 3).copy()
            t_vec = self.data.xpos[rid].copy()
            cvel = self.data.cvel[rid].copy()
            omega_vec = cvel[:3]
            v_vec = cvel[3:]

        h = self.config.get("drop_height", 0.5)
        out_dir = Path(self.output_dir)
        name = self.config.get("model_name", "TVDrop_Radioss")
        transform_mode = self.config.get("export_radioss_transform_mode", "parts")

        try:
            builder = RadiossModelBuilder(
                config=self.config,
                output_dir=out_dir,
                R_mat=R_mat,
                t_vec=t_vec,
                v_vec=v_vec,
                omega_vec=omega_vec,
                transform_mode=transform_mode,
                drop_height_m=h,
                model_name=name,
            )
            starter = builder.build()
            self._radioss_builder = builder
            self.log("✅ [WHTOOLS] Radioss Models generated successfully.")
            return starter, builder
        except Exception as e:
            self.log(f"❌ [WHTOOLS] Failed to generate Radioss Models: {e}", level="error")
            raise

    def _reset_simulation(self) -> None:
        """시뮬레이션을 초기 상태(스냅샷 0)로 리셋하며 모든 히스토리를 초기화합니다."""
        if not self.snapshots:
            self.log("⚠️ No snapshots to reset.", level="warning")
            return
        
        self._jump_to_snapshot(0)
        self._init_histories()
        self.snapshots = self.snapshots[:1]
        self.step_idx = 0
        self._collect_history()
        self._last_reported_interval = -1
        self._report_count = 0
        self.start_real_time = time.time()
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
    simulator = DropSimulator()
    simulator.simulate()
