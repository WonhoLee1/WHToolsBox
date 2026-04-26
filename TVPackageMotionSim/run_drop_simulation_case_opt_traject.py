# -*- coding: utf-8 -*-
"""
[WHTOOLS] Trajectory-Based Parameter Calibration via CMA-ES

실측 낙하 시험 궤적(profiles.txt)을 기준으로 시뮬레이션 파라미터를 자동 교정합니다.

최적화 파라미터:
  - 마찰계수          : ground-cushion, ground-paper
  - 쿠션 접촉 solref  : timeconstant, damping ratio
  - 쿠션 접촉 solimp  : min, max, transition_width
  - COG offset        : x, y, z (각 ±0.05 m 범위)

비교 지표:
  - 3D 변위 DTW  (Δpos = pos - pos₀, 정규화)
  - 속도 크기 DTW (Savitzky-Golay 미분, 변위와 동일 스케일)

모니터링:
  - Rich 터미널 Live 대시보드 (세대/비용/파라미터)
  - evallog.csv (평가별 상세 기록)
  - overlay_gen{N}.png / convergence.png (주기적 저장)
"""

import os
import sys
import copy
import time
import csv
import json
import logging
import pickle
import threading
import contextlib
import io
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich import box as rbox

# ── 경로 설정 ────────────────────────────────────────────────────────────────
curr_dir = os.path.dirname(os.path.abspath(__file__))
if curr_dir not in sys.path:
    sys.path.append(curr_dir)

from run_drop_simulator import DropSimulator
from run_discrete_builder import (
    get_default_config, get_rgba_by_name, calculate_plate_twist_weld_params
)

# ── Numba DTW (설치 안 된 경우 순수 numpy fallback) ──────────────────────────
try:
    from numba import njit as _njit

    @_njit(cache=True)
    def _dtw_1d_nb(s: np.ndarray, t: np.ndarray, radius: int) -> float:
        n, m = len(s), len(t)
        INF = 1e18
        dp = np.full((n + 1, m + 1), INF)
        dp[0, 0] = 0.0
        for i in range(1, n + 1):
            j_lo = max(1, i - radius)
            j_hi = min(m, i + radius)
            for j in range(j_lo, j_hi + 1):
                cost = abs(s[i - 1] - t[j - 1])
                dp[i, j] = cost + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
        return dp[n, m] / (n + m)

    @_njit(cache=True)
    def _dtw_3d_nb(a: np.ndarray, b: np.ndarray, radius: int) -> float:
        n, m = len(a), len(b)
        INF = 1e18
        dp = np.full((n + 1, m + 1), INF)
        dp[0, 0] = 0.0
        for i in range(1, n + 1):
            j_lo = max(1, i - radius)
            j_hi = min(m, i + radius)
            for j in range(j_lo, j_hi + 1):
                dx = a[i-1, 0] - b[j-1, 0]
                dy = a[i-1, 1] - b[j-1, 1]
                dz = a[i-1, 2] - b[j-1, 2]
                dp[i, j] = (dx*dx + dy*dy + dz*dz)**0.5 + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
        return dp[n, m] / (n + m)

    def dtw_1d(s: np.ndarray, t: np.ndarray, radius: int) -> float:
        return _dtw_1d_nb(
            np.ascontiguousarray(s, np.float64),
            np.ascontiguousarray(t, np.float64), radius)

    def dtw_3d(a: np.ndarray, b: np.ndarray, radius: int) -> float:
        return _dtw_3d_nb(
            np.ascontiguousarray(a, np.float64),
            np.ascontiguousarray(b, np.float64), radius)

    _NUMBA = True

except ImportError:
    _NUMBA = False

    def dtw_1d(s: np.ndarray, t: np.ndarray, radius: int) -> float:
        n, m = len(s), len(t)
        dp = np.full((n + 1, m + 1), np.inf)
        dp[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(max(1, i - radius), min(m, i + radius) + 1):
                dp[i, j] = abs(s[i-1] - t[j-1]) + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
        return dp[n, m] / (n + m)

    def dtw_3d(a: np.ndarray, b: np.ndarray, radius: int) -> float:
        n, m = len(a), len(b)
        dp = np.full((n + 1, m + 1), np.inf)
        dp[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(max(1, i - radius), min(m, i + radius) + 1):
                dx, dy, dz = a[i-1, 0]-b[j-1, 0], a[i-1, 1]-b[j-1, 1], a[i-1, 2]-b[j-1, 2]
                dp[i, j] = (dx*dx + dy*dy + dz*dz)**0.5 + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
        return dp[n, m] / (n + m)

# ── 전역 상수 ─────────────────────────────────────────────────────────────────
console = Console()

# C명칭 → corner_pos_hist 내부 인덱스
# compute_corner_kinematics 루프 순서 기준:
#   idx = x in [-w/2,+w/2] × y in [-h/2,+h/2] × z in [-d/2,+d/2]
#   전면(z=-d/2): C1=top-right=6, C2=bot-right=4, C3=bot-left=0, C4=top-left=2
#   후면(z=+d/2): C5=7, C6=5, C7=1, C8=3
C_TO_IDX: Dict[str, int] = {
    'C1': 6, 'C2': 4, 'C3': 0, 'C4': 2,
    'C5': 7, 'C6': 5, 'C7': 1, 'C8': 3,
}

_NOMINAL_COG = [0.001, 0.007, 0.010]   # doe_modeling_case_1 기준 nominal COG

# ── 파라미터 공간 정의 ────────────────────────────────────────────────────────
# (이름, 초기값, 최솟값, 최댓값, 로그스케일)
PARAM_DEFS: List[Tuple] = [
    ("friction_gnd_cush",          0.70,     0.10,      2.00,    False),  # 공통 마찰계수 (모든 접촉에 적용)
    ("cush_solref_0",          -10000.0, -100000.0,  -100.0,    True ),  # 지면-쿠션 강성 [N/m]
    ("cush_solref_1",            -100.0,  -10000.0,   -10.0,    True ),  # 지면-쿠션 감쇠 [N*s/m]
    ("p_solref_0",             -10000.0, -100000.0,  -100.0,    True ),  # 내부 접촉 강성 (opencell, chassis)
    ("p_solref_1",               -100.0,  -10000.0,   -10.0,    True ),  # 내부 접촉 감쇠
    ("cush_solimp_0",              0.10,     0.01,      0.50,    False),  # 쿠션 접촉 최소 임피던스
    ("cush_solimp_1",              0.95,     0.70,      0.999,   False),  # 쿠션 접촉 최대 임피던스
    ("cush_solimp_2",              0.02,     0.001,     2.00,    True ),  # 쿠션 접촉 전환 폭 (transition width)
    ("cog_x",                      0.00,    -0.01,      0.01,    False),  # COG x 방향 offset [m]
    ("cog_y",                      0.00,    -0.01,      0.01,    False),  # COG y 방향 offset [m]
    ("cog_z",                      0.00,    -0.01,      0.01,    False),  # COG z 방향 offset [m]
    ("plasticity_ratio",           0.30,     0.01,      0.80,    False),  # 소성 변형 비율 (탄성:소성 배분)
    ("cush_yield_pressure",        1500.0,   100.0,  15000.0,    True ),  # 쿠션 항복 압력 [Pa]
    ("plastic_hardening_modulus",  30000.0,  100.0, 300000.0,    True ),  # 소성 가공 경화 계수 [Pa]
    ("initial_tilt_deg",           0.00,    -1.00,      1.00,    False),  # 낙하 자세 틸트 각도 [deg]
    ("initial_tilt_azimuth_deg",   0.00,    -1.00,      1.00,    False),  # 낙하 자세 틸트 방위각 [deg]
]
N_PARAMS = len(PARAM_DEFS)
PARAM_NAMES = [p[0] for p in PARAM_DEFS]


# ── 신호 처리 ─────────────────────────────────────────────────────────────────

def sg_derivative(pos: np.ndarray, time: np.ndarray,
                  window: int = 15, poly: int = 4) -> np.ndarray:
    """
    Savitzky-Golay 다항식 미분으로 속도를 산출합니다.

    window=15, poly=4는 400Hz 급 데이터에서 노이즈를 충분히 억제하면서
    충격 피크의 형상 왜곡을 최소화합니다.
    """
    n = len(time)
    dt = float(np.mean(np.diff(time)))
    # 홀수, > poly, < n 조건 보장
    w = min(window, n - 1)
    if w % 2 == 0:
        w -= 1
    min_w = poly + 2 if (poly + 2) % 2 == 1 else poly + 3
    w = max(w, min_w)
    return savgol_filter(pos, window_length=w, polyorder=poly,
                         deriv=1, delta=dt, axis=0)


def preprocess(pos: np.ndarray, time: np.ndarray
               ) -> Tuple[np.ndarray, np.ndarray]:
    """
    단일 코너 위치 배열 → (정규화된 Δpos, 변위와 동일 스케일의 속도 크기)

    두 신호를 동일한 최대 크기로 정규화하여 DTW 비용의 단위를 일치시킵니다.
    """
    delta_pos = pos - pos[0]                                    # (T, 3)
    vel = sg_derivative(pos, time)                              # (T, 3)
    vel_mag = np.linalg.norm(vel, axis=1)                       # (T,)

    scale_d = np.max(np.abs(delta_pos)) + 1e-9
    scale_v = np.max(vel_mag) + 1e-9

    delta_pos_norm = delta_pos / scale_d                        # 정규화 변위
    vel_mag_norm = vel_mag * (scale_d / scale_v)                # 변위 스케일에 맞춘 속도 크기

    return delta_pos_norm, vel_mag_norm


def interpolate_to(t_ref: np.ndarray, t_src: np.ndarray,
                   data: np.ndarray) -> np.ndarray:
    """data(t_src)를 t_ref 시간 축으로 cubic spline 보간합니다."""
    f = interp1d(t_src, data, axis=0, kind='cubic',
                 bounds_error=False, fill_value=(data[0], data[-1]))
    return f(t_ref)


# ── 궤적 로더 ─────────────────────────────────────────────────────────────────

def load_trajectory(filepath: str, unit_scale: float = 1.0) -> Dict:
    """
    profiles.txt 포맷 로딩.

    헤더: Frame  Time  [C1~CN]_pos_X × N  [C1~CN]_pos_Y × N  [C1~CN]_pos_Z × N
    단위: 기본 m (unit_scale=0.001 지정 시 mm→m 변환)

    Returns:
        {'time': (T,), 'pos': {'C1': (T,3), ...}, 'corners': ['C1', ...]}
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split('\t')

    data = np.loadtxt(filepath, delimiter='\t', skiprows=1)

    time_idx = next(i for i, h in enumerate(header) if h.lower() == 'time')
    t = data[:, time_idx]

    corners = sorted(
        {h.split('_')[0] for h in header if h.startswith('C') and '_pos_' in h},
        key=lambda c: int(c[1:])
    )

    pos = {}
    for c in corners:
        xi = header.index(f'{c}_pos_X')
        yi = header.index(f'{c}_pos_Y')
        zi = header.index(f'{c}_pos_Z')
        pos[c] = data[:, [xi, yi, zi]] * unit_scale

    return {'time': t, 'pos': pos, 'corners': corners}


# ── 비용 함수 ─────────────────────────────────────────────────────────────────

def compute_fitness(
    t_meas: np.ndarray,
    meas_pos: Dict[str, np.ndarray],
    t_sim: np.ndarray,
    sim_corner_pos: np.ndarray,          # (T_sim, 8, 3)
    selected_corners: List[str],
    w_disp: float = 0.50,
    w_vel:  float = 0.50,
    dtw_r:  Optional[int] = None,
    response_weights: Optional[Dict] = None,  # {c: {'disp': float, 'vel': float}}
) -> Tuple[float, Dict]:
    """
    DTW 기반 변위 + 속도 복합 비용을 계산합니다.

    공통 시간 구간에서 실측 시간 그리드를 기준으로 시뮬레이션 궤적을 보간합니다.
    response_weights로 코너별·응답별 가중치를 지정합니다 (0 = 해당 응답 제외).
    """
    # 공통 시간 구간
    t0 = max(t_meas[0], t_sim[0])
    t1 = min(t_meas[-1], t_sim[-1])
    mask = (t_meas >= t0) & (t_meas <= t1)
    tc = t_meas[mask]

    if len(tc) < 20:
        return 1e6, {}

    r = dtw_r if dtw_r is not None else max(10, int(0.1 * len(tc)))

    rw = response_weights or {}
    total_d = total_v = w_d_sum = w_v_sum = 0.0
    detail: Dict = {}

    for c in selected_corners:
        sim_idx = C_TO_IDX[c]
        pm = meas_pos[c][mask]                                      # 실측 (N, 3)
        ps = interpolate_to(tc, t_sim, sim_corner_pos[:, sim_idx, :])  # 시뮬 보간

        dp_m, vm_m = preprocess(pm, tc)
        dp_s, vm_s = preprocess(ps, tc)

        f_d = dtw_3d(dp_m, dp_s, r)
        f_v = dtw_1d(vm_m, vm_s, r)

        wd = float(rw.get(c, {}).get('disp', 1.0))
        wv = float(rw.get(c, {}).get('vel',  1.0))

        detail[c] = {'f_disp': f_d, 'f_vel': f_v, 'w_disp': wd, 'w_vel': wv}
        total_d += f_d * wd
        total_v += f_v * wv
        w_d_sum += wd
        w_v_sum += wv

    cost = (w_disp * total_d / max(w_d_sum, 1e-9) +
            w_vel  * total_v / max(w_v_sum, 1e-9))
    return cost, detail


# ── 설정 빌더 ─────────────────────────────────────────────────────────────────

def build_base_cfg() -> dict:
    """Case 1 기반 헤드리스 최적화용 기본 cfg."""
    cfg = get_default_config()

    # ── UI / 출력 제어 ────────────────────────────────────────────────────
    cfg["use_viewer"]           = False   # MuJoCo 뷰어 비활성화 (헤드리스)
    cfg["use_jax_reporting"]    = False   # JAX 리포팅 비활성화 (속도 향상)
    cfg["batch_run_save_figures"] = False
    cfg["batch_run_show_figures"] = False

    # ── [1] 형상 치수 ─────────────────────────────────────────────────────
    cfg["box_w"]     = 2.056   # 박스 외곽 가로 [m]
    cfg["box_h"]     = 1.200   # 박스 외곽 세로 [m]
    cfg["box_d"]     = 0.178   # 박스 외곽 깊이 [m]
    cfg["box_thick"] = 0.008   # 골판지 두께 [m]
    cfg["assy_w"]    = 1.892   # TV 어셈블리 가로 [m]
    cfg["assy_h"]    = 1.082   # TV 어셈블리 세로 [m]
    cfg["cush_gap"]  = 0.001   # 쿠션-제품 조립 공극 [m]
    cfg["opencell_d"]    = 0.012   # Open Cell(유리패널 완충재) 두께 [m]
    cfg["opencellcoh_d"] = 0.002   # Open Cell 접착층(Tape) 두께 [m]
    cfg["chassis_d"]     = 0.035   # Chassis(후면 금속 프레임) 두께 [m]
    cfg["occ_ithick"]    = 0.030   # Open Cell 인터페이스 두께 [m]

    # ── [2] 낙하 시나리오 ─────────────────────────────────────────────────
    cfg["drop_mode"]      = "LTL"           # Less-than-Truckload 낙하 규격
    cfg["drop_direction"] = "Corner 2-3-5"  # 피벗 코너: C5 (면 2·3·5의 교점)
    cfg["drop_height"]    = 0.3             # 자유 낙하 높이 [m]
    cfg["visual"] = {
        "fogstart":    3.0,           # 안개 시작 거리 [m] (무한 지면 연출)
        "fogend":      10.0,          # 안개 완전 불투명 거리 [m]
        "skybox_rgba": "0.6 0.6 0.6", # 배경 색상
    }

    # ── [3] 컴포넌트 이산화 & 질량 ───────────────────────────────────────
    # div: [x, y, z] 방향 분할 수 → 클수록 정밀, 느림
    # use_weld: True면 내부 블록 간 weld 연결 (변형 시뮬레이션)
    cfg["components"] = {
        "paper":       {"div": [5, 5, 3], "use_weld": True,  "mass": 4.0,   # 종이박스 [kg]
                        "rgba": get_rgba_by_name("paper", 1.0)},
        "cushion":     {"div": [5, 5, 3], "use_weld": True,  "mass": 3.0,   # EPS 쿠션 [kg]
                        "rgba": "0.8 0.8 0.8 0.6"},
        "opencell":    {"div": [4, 4, 1], "use_weld": False, "mass": 5.0,   # 유리 패널 [kg]
                        "rgba": get_rgba_by_name("black", 1.0)},
        "opencellcoh": {"div": [4, 4, 1], "use_weld": False, "mass": 0.1,   # 접착 테이프 [kg]
                        "rgba": get_rgba_by_name("red", 0.4)},
        "chassis":     {"div": [4, 4, 1], "use_weld": False, "mass": 10.0,  # 후면 금속 프레임 [kg]
                        "rgba": "0.0 0.2 0.4 1.0"},
    }
    cfg["include_paperbox"] = False   # 종이박스 메쉬 모델 포함 여부

    # ── [4] 접촉 쌍 파라미터 ─────────────────────────────────────────────
    # solref: [timeconstant, damping_ratio] — 접촉 강성·감쇠
    #   timeconstant > 0: 스프링 timescale [s], 작을수록 딱딱함
    #   damping_ratio: 1.0 = 임계 감쇠
    # solimp: [d_min, d_max, width, mid, power] — 임피던스 곡선
    #   d_min/d_max: 침투 깊이에 따른 최소·최대 임피던스
    #   width: 전환 구간 폭 [m]
    fr    = [0.7, 0.7]               # 초기 마찰계수 [sliding, rolling]
    p_sr  = [-55000.0, -800.0]       # 쿠션 내부 weld solref (음수: 강성[N/m], 감쇠[N·s/m])
    p_si  = [0.10, 0.95, 0.02, 0.5, 2]
    cfg["contacts"] = {
        ("ground",  "cushion"):      {"friction": fr, "solref": [0.001, 1.0], "solimp": [0.1, 0.95, 0.02, 0.5, 2]},
        ("ground",  "cushion_edge"): {"friction": fr, "solref": [0.001, 1.0], "solimp": [0.1, 0.95, 0.02, 0.5, 2]},
        ("ground",  "paper"):        {"friction": fr, "solref": [0.001, 1.0], "solimp": [0.1, 0.95, 0.02, 0.5, 2]},
        ("cushion", "opencell"):     {"friction": fr, "solref": p_sr, "solimp": p_si},
        ("cushion", "chassis"):      {"friction": fr, "solref": p_sr, "solimp": p_si},
        ("cushion", "paper"):        {"friction": fr, "solref": [0.001, 1.0], "solimp": [0.1, 0.95, 0.02, 0.5, 2]},
    }

    # ── [4b] Weld 파라미터 (판 굽힘 강성 자동 산출) ─────────────────────
    # calculate_plate_twist_weld_params: 실제 E·두께 → 목표 진동수 기반 weld solref 역산
    k_oc, d_oc, ts_oc = calculate_plate_twist_weld_params(
        mass=cfg["components"]["opencell"]["mass"],
        width=cfg["assy_w"], height=cfg["assy_h"], thickness=cfg["opencell_d"],
        div=cfg["components"]["opencell"]["div"],
        E_real=70e9,        # 유리 탄성계수 [Pa]
        real_thickness=0.001,   # 실제 두께 1mm
        target_freq_hz=1.0,     # 판 전체 1차 벤딩 목표 주파수 [Hz]
        zeta=0.05)              # 감쇠비
    k_ch, d_ch, ts_ch = calculate_plate_twist_weld_params(
        mass=cfg["components"]["chassis"]["mass"],
        width=cfg["assy_w"], height=cfg["assy_h"], thickness=cfg["chassis_d"],
        div=cfg["components"]["chassis"]["div"],
        E_real=170e9,       # Chassis 강재 탄성계수 [Pa]
        real_thickness=0.0006,  # 실제 두께 0.6mm
        target_freq_hz=4.0,     # 판 전체 1차 벤딩 목표 주파수 [Hz]
        zeta=0.05)

    cfg["welds"] = {
        "paper":          {"solref": [0.010, 1.00], "solimp": [0.10, 0.95, 0.01, 0.5, 2]},
        "cushion":        {"solref": p_sr,           "solimp": p_si},
        "cushion_corner": {"solref": p_sr,           "solimp": p_si},
        "opencell":       {"solref": [k_oc, d_oc],  "solimp": [0.10, 0.95, 0.1, 0.5, 2], "torquescale": ts_oc},
        "opencellcoh":    {"solref": [-15000.0, -500.0], "solimp": [0.10, 0.95, 0.01, 0.5, 2]},
        "chassis":        {"solref": [k_ch, d_ch],  "solimp": [0.10, 0.99, 0.1, 0.5, 2], "torquescale": ts_ch},
    }

    # ── [5] 소성 모델 ─────────────────────────────────────────────────────
    # 충격 시 쿠션의 비가역적 변형(영구 압축)을 표현
    cfg["enable_plasticity"]         = True
    cfg["plasticity_ratio"]          = 0.3      # 소성 변형 비율 (0=순탄성, 1=완전소성)
    cfg["cush_yield_pressure"]       = 1500.0   # 항복 압력 [Pa] — 이 압력 이상에서 소성 시작
    cfg["plastic_hardening_modulus"] = 30000.0  # 가공 경화 계수 [Pa] — 클수록 항복 후 급격히 경화

    # ── [6] 질량·관성·무게중심 자동 밸런싱 ──────────────────────────────
    # analyze_and_balance_components가 target 값을 맞추도록 보조 질량을 자동 배치
    cfg["components_balance"] = {
        "target_mass":    42.2,                    # 목표 총 질량 [kg]
        "target_inertia": [3.0, 8.0, 14.0],        # 목표 주관성모멘트 Ixx, Iyy, Izz [kg·m²]
        "target_cog":     list(_NOMINAL_COG),       # 목표 무게중심 위치 [m] (최적화로 오프셋 적용)
        "count": 8,                                 # 배치할 보조 질량 개수
    }

    # ── [7] 솔버 & 시뮬레이션 설정 ───────────────────────────────────────
    cfg["sim_integrator"]        = "implicitfast"  # 암시적 적분기 (안정성↑, 속도↑)
    cfg["sim_timestep"]          = 0.0012          # 시뮬레이션 타임스텝 [s]
    cfg["sim_iterations"]        = 50              # 솔버 반복 횟수 (클수록 정확, 느림)
    cfg["sim_noslip_iterations"] = 0               # no-slip 반복 (0=비활성)
    cfg["sim_tolerance"]         = 1e-5            # 솔버 수렴 허용 오차
    cfg["sim_gravity"]           = [0, 0, -9.81]   # 중력 가속도 [m/s²]
    cfg["sim_nthread"]           = 4               # 병렬 스레드 수
    cfg["reporting_interval"]    = 0.0024          # 데이터 기록 주기 [s] (profiles.txt와 동일)
    cfg["sim_duration"]          = 2.0             # 시뮬레이션 총 시간 [s]
    cfg["enable_air_drag"]       = True            # 공기 저항력 활성화
    cfg["enable_air_squeeze"]    = True            # 공기 압착력 활성화 (낙하 시 하부 공기 압축)

    return cfg


# ── 파라미터 변환 ─────────────────────────────────────────────────────────────

def _theta_init() -> np.ndarray:
    return np.array([p[1] for p in PARAM_DEFS])

def theta_to_norm(theta: np.ndarray) -> np.ndarray:
    """실제값 → [0,1] 정규화."""
    norm = np.empty(N_PARAMS)
    for i, (*_, lo, hi, log_s) in enumerate(PARAM_DEFS):
        v = theta[i]
        norm[i] = (np.log(v/lo) / np.log(hi/lo)) if log_s else (v - lo) / (hi - lo)
    return np.clip(norm, 0.0, 1.0)

def norm_to_theta(norm: np.ndarray) -> np.ndarray:
    """[0,1] 정규화 → 실제값."""
    theta = np.empty(N_PARAMS)
    for i, (*_, lo, hi, log_s) in enumerate(PARAM_DEFS):
        t = float(np.clip(norm[i], 0.0, 1.0))
        theta[i] = lo * (hi/lo)**t if log_s else lo + t * (hi - lo)
    return theta

def apply_params(base_cfg: dict, theta: np.ndarray) -> dict:
    """파라미터 벡터를 cfg에 적용합니다.

    DropSimulator.__init__이 components_balance 키를 보고 apply_balancing()을 자동으로
    호출하므로, 여기서 analyze_and_balance_components를 별도 호출하지 않습니다.
    (중복 호출 시 balance mass가 이중으로 적재되어 XML이 깨지는 문제 방지)
    """
    c = copy.deepcopy(base_cfg)
    idx = {name: i for i, (name, *_) in enumerate(PARAM_DEFS)}

    # [1] 마찰계수 적용 (모든 접촉 쌍에 공통 적용)
    fr = theta[idx["friction_gnd_cush"]]
    for key in c["contacts"]:
        c["contacts"][key]["friction"] = [fr, fr]

    # [2] 지면 관련 접촉 solref / solimp (음수 solref: 강성/감쇠 직접 지정)
    sr0 = theta[idx["cush_solref_0"]]
    sr1 = theta[idx["cush_solref_1"]]
    si0 = theta[idx["cush_solimp_0"]]
    si1 = theta[idx["cush_solimp_1"]]
    si2 = theta[idx["cush_solimp_2"]]
    for key in [("ground", "cushion"), ("ground", "cushion_edge"), ("ground", "paper")]:
        if key in c["contacts"]:
            c["contacts"][key]["solref"] = [sr0, sr1]
            c["contacts"][key]["solimp"] = [si0, si1, si2, 0.5, 2]

    # [3] 내부 부품-쿠션 접촉 solref (p_solref)
    p_sr0 = theta[idx["p_solref_0"]]
    p_sr1 = theta[idx["p_solref_1"]]
    for key in [("cushion", "opencell"), ("cushion", "chassis")]:
        if key in c["contacts"]:
            c["contacts"][key]["solref"] = [p_sr0, p_sr1]

    c["components_balance"]["target_cog"] = [
        _NOMINAL_COG[0] + theta[idx["cog_x"]],
        _NOMINAL_COG[1] + theta[idx["cog_y"]],
        _NOMINAL_COG[2] + theta[idx["cog_z"]],
    ]

    # 소성 파라미터
    c["plasticity_ratio"]           = theta[idx["plasticity_ratio"]]
    c["cush_yield_pressure"]        = theta[idx["cush_yield_pressure"]]
    c["plastic_hardening_modulus"]  = theta[idx["plastic_hardening_modulus"]]

    # 낙하 자세 틸트 파라미터
    c["initial_tilt_deg"]          = theta[idx["initial_tilt_deg"]]
    c["initial_tilt_azimuth_deg"]  = theta[idx["initial_tilt_azimuth_deg"]]

    return c


# ── 시뮬레이션 러너 ───────────────────────────────────────────────────────────

def run_sim_headless(cfg: dict, sim_dir: str,
                     suppress_output: bool = True
                     ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    헤드리스 시뮬레이션 실행.

    Returns:
        (corner_pos (T,8,3), time (T,)) or None on failure
    """
    cfg_run = copy.deepcopy(cfg)
    cfg_run["output_dir"] = str(Path(sim_dir) / "latest")

    # DropSimulator 로그를 WARNING으로 억제
    engine_logger = logging.getLogger("WHTS_Engine")
    prev_level = engine_logger.level
    engine_logger.setLevel(logging.WARNING)

    try:
        if suppress_output:
            _sink = io.StringIO()
            _ctx = contextlib.redirect_stdout(_sink), contextlib.redirect_stderr(_sink)
        else:
            _sink = None
            _ctx = contextlib.nullcontext(), contextlib.nullcontext()
        with _ctx[0], _ctx[1]:
            sim = DropSimulator(config=cfg_run)
            sim.simulate()

        if sim.result is None or not sim.corner_pos_hist:
            return None

        t = np.array(sim.time_history, dtype=np.float64)
        pos = np.array(
            [[corner for corner in frame] for frame in sim.corner_pos_hist],
            dtype=np.float64)   # (T, 8, 3)
        return pos, t

    except Exception as e:
        tb = traceback.format_exc()
        # suppress_output 중 출력된 내용도 함께 기록
        captured = _sink.getvalue() if _sink else ""
        console.print(f"[red]  시뮬 오류: {e}[/red]")
        try:
            err_path = Path(sim_dir) / "error.log"
            err_path.parent.mkdir(parents=True, exist_ok=True)
            content = tb
            if captured.strip():
                content += "\n── captured stdout/stderr ──\n" + captured
            err_path.write_text(content, encoding="utf-8")
        except Exception:
            pass
        return None
    finally:
        engine_logger.setLevel(prev_level)


# ── 리포트 헬퍼 ──────────────────────────────────────────────────────────────

def save_convergence_png(cost_hist: list, sigma_hist: list, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        a1.semilogy(cost_hist, 'b-o', ms=3)
        a1.set_ylabel('Best Cost (log scale)')
        a1.set_title('CMA-ES Convergence')
        a1.grid(True)
        a2.semilogy(sigma_hist, 'r-o', ms=3)
        a2.set_ylabel('Sigma')
        a2.set_xlabel('Generation')
        a2.grid(True)
        fig.tight_layout()
        fig.savefig(str(path), dpi=150)
        plt.close(fig)
    except Exception:
        pass


def save_overlay_png(
    t_meas: np.ndarray, meas_pos: Dict,
    sim_pos: np.ndarray, sim_t: np.ndarray,
    selected: List[str], cost: float, path: Path
) -> None:
    try:
        import matplotlib.pyplot as plt
        nc = len(selected)
        fig, axes = plt.subplots(nc, 2, figsize=(14, 4 * nc))
        if nc == 1:
            axes = [axes]

        for row, c in enumerate(selected):
            si = C_TO_IDX[c]
            t0 = max(t_meas[0], sim_t[0])
            t1 = min(t_meas[-1], sim_t[-1])
            mask = (t_meas >= t0) & (t_meas <= t1)
            tc = t_meas[mask]

            pm = meas_pos[c][mask]
            ps = interpolate_to(tc, sim_t, sim_pos[:, si, :])

            dp_m = pm - pm[0]
            dp_s = ps - ps[0]
            vm_m = sg_derivative(pm, tc)
            vm_s = sg_derivative(ps, tc)

            ax_d, ax_v = axes[row]
            colors = ['#e74c3c', '#2ecc71', '#3498db']
            labels = ['X', 'Y', 'Z']
            for k in range(3):
                ax_d.plot(tc, dp_m[:, k], '--', color=colors[k], alpha=0.6,
                          label=f'Meas {labels[k]}', lw=1.5)
                ax_d.plot(tc, dp_s[:, k], '-',  color=colors[k],
                          label=f'Sim {labels[k]}',  lw=1.5)
            ax_d.set_title(f'{c}  Δpos  (cost={cost:.5f})')
            ax_d.set_ylabel('Displacement [m]')
            ax_d.legend(fontsize=7, ncol=3)
            ax_d.grid(True, alpha=0.4)

            ax_v.plot(tc, np.linalg.norm(vm_m, axis=1), '--k', alpha=0.6,
                      lw=1.5, label='Meas |v|')
            ax_v.plot(tc, np.linalg.norm(vm_s, axis=1), '-r', lw=1.5,
                      label='Sim |v|')
            ax_v.set_title(f'{c}  |velocity|')
            ax_v.set_ylabel('Speed [m/s]')
            ax_v.legend(fontsize=7)
            ax_v.grid(True, alpha=0.4)

        fig.tight_layout()
        fig.savefig(str(path), dpi=150)
        plt.close(fig)
    except Exception:
        pass


def save_drop_dir_png(
    t_meas: np.ndarray,
    meas_pos: Dict[str, np.ndarray],
    sim_pos: np.ndarray,
    sim_t: np.ndarray,
    selected: List[str],
    cost: float,
    path: Path,
    drop_axis: int = 2,   # 0=X, 1=Y, 2=Z (기본: Z = 중력 방향)
) -> None:
    """
    낙하 방향(drop_axis) 성분의 변위 & 속도를 2×2 subplot으로 저장합니다.

      [0,0] 실측 변위  |  [0,1] 시뮬 변위
      [1,0] 실측 속도  |  [1,1] 시뮬 속도

    각 코너가 하나의 curve로 표시되며, 왼쪽(target)과 오른쪽(sim)의
    색상·범례를 동일하게 맞춰 한눈에 비교할 수 있습니다.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        axis_label = ['X', 'Y', 'Z'][drop_axis]
        nc = len(selected)
        colors = [cm.tab10(i / max(nc, 1)) for i in range(nc)]

        # 공통 시간 구간
        t0_ = max(t_meas[0], sim_t[0])
        t1_ = min(t_meas[-1], sim_t[-1])
        mask = (t_meas >= t0_) & (t_meas <= t1_)
        tc = t_meas[mask]

        fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
        ax_md, ax_sd = axes[0]   # row 0: 변위
        ax_mv, ax_sv = axes[1]   # row 1: 속도

        for i, c in enumerate(selected):
            col = colors[i]
            si = C_TO_IDX[c]

            pm = meas_pos[c][mask]                              # (N, 3)
            ps = interpolate_to(tc, sim_t, sim_pos[:, si, :])  # (N, 3)

            disp_m = pm[:, drop_axis] - pm[0, drop_axis]
            disp_s = ps[:, drop_axis] - ps[0, drop_axis]
            vel_m  = sg_derivative(pm, tc)[:, drop_axis]
            vel_s  = sg_derivative(ps, tc)[:, drop_axis]

            kw = dict(lw=1.5, alpha=0.85, label=c)
            ax_md.plot(tc, disp_m, color=col, **kw)
            ax_sd.plot(tc, disp_s, color=col, **kw)
            ax_mv.plot(tc, vel_m,  color=col, **kw)
            ax_sv.plot(tc, vel_s,  color=col, **kw)

        ax_md.set_title(f'Target  Δpos-{axis_label}',   fontsize=11)
        ax_sd.set_title(f'Simulation  Δpos-{axis_label}  (cost={cost:.5f})', fontsize=11)
        ax_mv.set_title(f'Target  vel-{axis_label}',    fontsize=11)
        ax_sv.set_title(f'Simulation  vel-{axis_label}', fontsize=11)

        axes[0, 0].set_ylabel('Displacement [m]')
        axes[1, 0].set_ylabel('Velocity [m/s]')
        for ax in axes[1]:
            ax.set_xlabel('Time [s]')

        for ax in axes.flat:
            ax.grid(True, alpha=0.4)
            ax.legend(fontsize=8, ncol=min(nc, 4))

        fig.suptitle(
            f'Drop-direction ({axis_label}) — {nc} corners', fontsize=13)
        fig.tight_layout()
        fig.savefig(str(path), dpi=150)
        plt.close(fig)
    except Exception:
        pass


# ── 병렬 평가 워커 ───────────────────────────────────────────────────────────
# ProcessPoolExecutor가 사용하는 최상위 함수 (중첩 함수는 pickle 불가)
def _worker_eval(args: tuple) -> dict:
    """단일 시뮬레이션 평가 — ProcessPoolExecutor 워커."""
    (eid, norm_vec, base_cfg, sim_dir_str,
     t_meas, meas_pos_dict,
     selected_corners, w_disp, w_vel, dtw_radius, suppress_output,
     response_weights) = args

    import time as _t
    t0 = _t.time()
    theta = norm_to_theta(np.asarray(norm_vec, dtype=np.float64))
    cfg_eval = apply_params(base_cfg, theta)
    result = run_sim_headless(cfg_eval, sim_dir_str, suppress_output)
    elapsed = _t.time() - t0

    if result is None:
        return dict(eid=eid, cost=1e6, f_d=1e6, f_v=1e6, diverged=True,
                    theta=theta.tolist(), elapsed=elapsed,
                    sim_pos=None, sim_t=None, detail={})

    sim_pos, sim_t = result
    cost, detail = compute_fitness(
        t_meas, meas_pos_dict, sim_t, sim_pos,
        selected_corners, w_disp, w_vel, dtw_radius, response_weights)

    active = [c for c in selected_corners if c in detail]
    f_d = float(np.mean([detail[c]['f_disp'] for c in active])) if active else 1e6
    f_v = float(np.mean([detail[c]['f_vel']  for c in active])) if active else 1e6

    return dict(eid=eid, cost=cost, f_d=f_d, f_v=f_v, diverged=False,
                theta=theta.tolist(), elapsed=elapsed,
                sim_pos=sim_pos, sim_t=sim_t, detail=detail)


# ── CMA-ES 최적화 메인 ────────────────────────────────────────────────────────

def run_optimization(
    traj_file: str,
    result_dir: str          = "opt_traject_results",
    selected_corners: Optional[List[str]] = None,
    sigma0: float            = 0.25,
    popsize: int             = 10,
    max_evals: int           = 150,
    w_disp: float            = 0.50,
    w_vel:  float            = 0.50,
    dtw_radius: Optional[int] = None,
    n_workers: int           = 1,
    suppress_output: bool    = True,
    response_weights: Optional[Dict] = None,  # {c: {'disp': float, 'vel': float}}
) -> Optional[np.ndarray]:
    """
    궤적 기반 CMA-ES 파라미터 캘리브레이션을 실행합니다.

    Args:
        traj_file:        실측 궤적 파일 경로
        result_dir:       결과 저장 폴더
        selected_corners: 비교할 코너 (기본: C2, C5)
        sigma0:           초기 탐색 표준편차 (정규화 공간)
        popsize:          세대당 샘플 수
        max_evals:        최대 시뮬레이션 실행 횟수
        w_disp:           변위 DTW 가중치
        w_vel:            속도 DTW 가중치
        dtw_radius:       Sakoe-Chiba 반경 (None → 신호 길이의 10%)
        n_workers:        병렬 프로세스 수 (1 = 순차 실행)

    Returns:
        최적 파라미터 벡터 (theta)
    """
    try:
        import cma
    except ImportError:
        console.print("[bold red]cma 라이브러리 미설치: pip install cma[/bold red]")
        return None

    if selected_corners is None:
        selected_corners = ['C2', 'C5']

    out = Path(result_dir)
    out.mkdir(parents=True, exist_ok=True)
    sim_temp = str(out / "sim_temp")
    log_csv  = out / "evallog.csv"
    best_pkl = out / "best_params.pkl"

    # opt_meta.json — 모니터 UI가 파라미터 정의를 동적으로 읽기 위해 저장
    rw_save = response_weights or {c: {'disp': 1.0, 'vel': 1.0} for c in selected_corners}
    opt_meta = {
        'param_defs':        [[n, i, lo, hi, ls] for n, i, lo, hi, ls in PARAM_DEFS],
        'param_names':       PARAM_NAMES,
        'selected_corners':  selected_corners,
        'response_weights':  rw_save,
        'popsize':           popsize,
        'max_evals':         max_evals,
        'w_disp':            w_disp,
        'w_vel':             w_vel,
        'result_dir':        str(out),
        'started_at':        datetime.now().isoformat(),
    }
    (out / 'opt_meta.json').write_text(
        json.dumps(opt_meta, indent=2, ensure_ascii=False), encoding='utf-8'
    )

    # ── 실측 궤적 로드 ────────────────────────────────────────────────────
    console.print(f"[cyan]실측 궤적 로딩: {traj_file}[/cyan]")
    meas = load_trajectory(traj_file)
    t_meas   = meas['time']
    meas_pos = meas['pos']
    console.print(
        f"  코너: {meas['corners']}  |  T={len(t_meas)} 프레임  "
        f"|  dt={np.mean(np.diff(t_meas))*1000:.2f} ms")

    # ── 기본 cfg ──────────────────────────────────────────────────────────
    base_cfg = build_base_cfg()
    # 병렬 실행 시 프로세스당 MuJoCo 스레드 수 자동 조정 (CPU 과부하 방지)
    if n_workers > 1:
        base_cfg["sim_nthread"] = max(1, (os.cpu_count() or 4) // n_workers)

    # ── CMA-ES 초기화 ─────────────────────────────────────────────────────
    x0 = theta_to_norm(_theta_init())
    opts = cma.CMAOptions()
    opts['bounds']    = [[0.0] * N_PARAMS, [1.0] * N_PARAMS]
    opts['popsize']   = popsize
    opts['maxfevals'] = max_evals
    opts['verbose']          = -9
    opts['seed']             = 42
    opts['tolx']             = 1e-6          # 파라미터 변화 허용 오차 (완화)
    opts['tolfun']           = 1e-7          # 비용 변화 허용 오차 (완화)
    opts['tolstagnation']    = max_evals     # 정체 조기 종료 비활성화
    opts['tolflatfitness']   = 4             # 평탄 비용 조기 종료 비활성화
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    # ── CSV 로그 헤더 ─────────────────────────────────────────────────────
    corner_cols = [f'{c}_{k}' for c in selected_corners for k in ('disp', 'vel')]
    with open(log_csv, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(
            ['gen', 'eval', 'cost', 'f_disp', 'f_vel', 'sigma',
             'elapsed_s', 'diverged'] + corner_cols + PARAM_NAMES)

    # ── 상태 추적 ─────────────────────────────────────────────────────────
    eval_no    = [0]
    best_cost  = [float('inf')]
    best_theta = [_theta_init().copy()]
    best_sim   = [None]           # (sim_pos, sim_t) 최신 best 시뮬 결과 캐시
    cost_hist  = []
    sigma_hist = []
    t_start    = time.time()

    def _status_table() -> Table:
        t = Table(box=rbox.SIMPLE_HEAVY, header_style="bold cyan",
                  show_header=True, expand=True)
        t.add_column("항목",       style="dim", min_width=22)
        t.add_column("값",         min_width=40)
        elapsed = time.time() - t_start
        gen = len(cost_hist)
        t.add_row("세대 / 평가",  f"{gen} / {eval_no[0]}  (max {max_evals})")
        t.add_row("Best Cost",    f"[bold green]{best_cost[0]:.6f}[/bold green]")
        t.add_row("Sigma",        f"{es.sigma:.5f}")
        t.add_row("경과",         f"{elapsed/60:.1f} 분")
        if cost_hist:
            recent = cost_hist[-20:]
            lo, hi = min(recent), max(recent) + 1e-12
            blks = "▁▂▃▄▅▆▇█"
            bar = "".join(blks[int((c - lo) / (hi - lo) * 7)] for c in recent)
            t.add_row("비용 추이 (20gen)", bar)
        t.add_section()
        t.add_row("[bold]파라미터[/bold]", "[bold]Best → (초기값)[/bold]")
        for i, (name, init, lo, hi, _) in enumerate(PARAM_DEFS):
            v = best_theta[0][i]
            changed = abs(v - init) > 0.005 * abs(hi - lo)
            style = "yellow" if changed else "dim"
            t.add_row(f"  {name}", f"[{style}]{v:.5g}[/]  ← {init:.5g}")
        return t

    # ── Numba 워밍업 ──────────────────────────────────────────────────────
    if _NUMBA:
        console.print("[dim]Numba DTW JIT 컴파일 중...[/dim]", end="")
        _d = np.random.rand(60).astype(np.float64)
        _dtw_1d_nb(_d, _d, 6)
        console.print(" [green]완료[/green]")
    else:
        console.print("[yellow]Numba 미설치 — numpy fallback 사용 (느림). pip install numba 권장[/yellow]")

    console.print(
        f"[bold green]최적화 시작[/bold green]  "
        f"파라미터 {N_PARAMS}개 | popsize={popsize} | max_evals={max_evals}\n"
        f"비교 코너: {selected_corners} | w_disp={w_disp}, w_vel={w_vel}")

    # ── 최적화 루프 ───────────────────────────────────────────────────────
    # ProcessPoolExecutor를 루프 밖에서 한 번만 생성해 프로세스 기동 오버헤드 절감
    _pool = ProcessPoolExecutor(max_workers=n_workers) if n_workers > 1 else None

    with contextlib.ExitStack() as _stack:
        live = _stack.enter_context(
            Live(_status_table(), refresh_per_second=0.5, console=console))
        if _pool is not None:
            _stack.enter_context(_pool)

        gen_idx = 0

        while not es.stop():
            solutions = es.ask()

            # 이번 세대 eval_id 미리 일괄 할당
            base_eid = eval_no[0] + 1
            eval_no[0] += len(solutions)

            # 워커 인자 목록 — 각 평가마다 고유 디렉터리 사용 (병렬 충돌 방지)
            args_list = [
                (base_eid + k, solutions[k].tolist(), base_cfg,
                 str(Path(sim_temp) / f"e{base_eid + k:05d}"),
                 t_meas, meas_pos,
                 selected_corners, w_disp, w_vel, dtw_radius, suppress_output,
                 response_weights)
                for k in range(len(solutions))
            ]

            if _pool is not None:
                raw = list(_pool.map(_worker_eval, args_list))
            else:
                raw = [_worker_eval(a) for a in args_list]

            costs = []
            for r in raw:
                cost     = r['cost']
                f_d      = r['f_d']
                f_v      = r['f_v']
                theta    = np.asarray(r['theta'])
                dt       = r['elapsed']
                eid      = r['eid']
                diverged = r['diverged']

                costs.append(cost)

                # best 갱신
                if cost < best_cost[0]:
                    best_cost[0]  = cost
                    best_theta[0] = theta.copy()
                    if not diverged and r['sim_pos'] is not None:
                        best_sim[0] = (r['sim_pos'].copy(), r['sim_t'].copy())
                    with open(best_pkl, 'wb') as fp:
                        pickle.dump({'theta': theta, 'cost': cost,
                                     'param_names': PARAM_NAMES}, fp)

                # CSV 기록
                det = r.get('detail', {})
                corner_vals = []
                for _c in selected_corners:
                    _cd = det.get(_c, {})
                    corner_vals.append(f'{_cd["f_disp"]:.6f}' if _cd else '')
                    corner_vals.append(f'{_cd["f_vel"]:.6f}'  if _cd else '')
                with open(log_csv, 'a', newline='', encoding='utf-8') as fp:
                    csv.writer(fp).writerow(
                        [gen_idx, eid, f'{cost:.6f}', f'{f_d:.6f}',
                         f'{f_v:.6f}', f'{es.sigma:.5f}',
                         f'{dt:.1f}', diverged]
                        + corner_vals
                        + [f'{v:.6g}' for v in theta])

            es.tell(solutions, costs)
            cost_hist.append(min(costs))
            sigma_hist.append(es.sigma)
            gen_idx += 1

            # 10세대마다 오버레이 PNG 저장 (백그라운드 스레드)
            if gen_idx % 10 == 0 and best_sim[0] is not None:
                _sp, _st = best_sim[0]
                _path = out / f"overlay_gen{gen_idx:04d}.png"
                _cost = best_cost[0]

                def _bg_overlay(sp=_sp, st=_st, p=_path, co=_cost):
                    save_overlay_png(t_meas, meas_pos, sp, st,
                                     selected_corners, co, p)

                threading.Thread(target=_bg_overlay, daemon=True).start()

            live.update(_status_table())

    # ── 최종 리포트 ───────────────────────────────────────────────────────
    console.print(f"\n[bold green]최적화 완료[/bold green]")
    console.print(f"  Best cost : {best_cost[0]:.6f}")
    console.print(f"  총 평가  : {eval_no[0]} 회")
    console.print(f"  종료 이유: [yellow]{es.stop()}[/yellow]")
    console.print(f"  결과 폴더: {out}/")

    save_convergence_png(cost_hist, sigma_hist, out / "convergence.png")

    if best_sim[0] is not None:
        sp, st = best_sim[0]
        save_overlay_png(t_meas, meas_pos, sp, st,
                         selected_corners, best_cost[0],
                         out / "overlay_final.png")
        save_drop_dir_png(t_meas, meas_pos, sp, st,
                          selected_corners, best_cost[0],
                          out / "drop_dir_final.png")

    console.print("\n[bold]최적 파라미터 결과:[/bold]")
    for i, (name, init, lo, hi, _) in enumerate(PARAM_DEFS):
        v = best_theta[0][i]
        console.print(f"  {name:28s}: {v:>12.6g}   (초기값: {init:.6g})")

    return best_theta[0]


# ── 엔트리 포인트 ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── 사용자 설정 ──────────────────────────────────────────────────────
    TRAJ_FILE        = os.path.join(curr_dir, "resources", "profiles.txt")
    RESULT_DIR       = os.path.join(curr_dir, "opt_traject_results",
                                    datetime.now().strftime("%Y%m%d_%H%M%S"))
    SELECTED_CORNERS = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']  # 전체 코너 비교
    SIGMA0           = 0.25           # 초기 탐색 반경 (정규화 공간)
    POPSIZE          = 10             # 세대당 샘플 수
    MAX_EVALS        = 500            # 최대 시뮬레이션 실행 횟수
    W_DISP           = 0.50           # 변위 DTW 가중치
    W_VEL            = 0.50           # 속도 DTW 가중치
    N_WORKERS        = 4              # 병렬 프로세스 수 (1 = 순차, 권장: CPU코어수//sim_nthread)
    SUPPRESS_OUTPUT  = True           # False: 시뮬레이터 출력을 터미널에 표시

    # 코너별·응답별 가중치 — 0 으로 설정하면 해당 응답이 비용 함수에서 제외됩니다.
    RESPONSE_WEIGHTS = {c: {'disp': 1.0, 'vel': 1.0} for c in SELECTED_CORNERS}
    # 예시: C3 속도 응답 제외
    # RESPONSE_WEIGHTS['C3']['vel'] = 0.0

    LAUNCH_MONITOR   = True            # True: 최적화 시작 시 모니터 UI 자동 실행
    # ─────────────────────────────────────────────────────────────────────

    if LAUNCH_MONITOR:
        import subprocess
        monitor_script = os.path.join(curr_dir, "monitor_opt_traject.py")
        subprocess.Popen(
            [sys.executable, monitor_script, RESULT_DIR],
            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0,
        )

    best = run_optimization(
        traj_file        = TRAJ_FILE,
        result_dir       = RESULT_DIR,
        selected_corners = SELECTED_CORNERS,
        sigma0           = SIGMA0,
        popsize          = POPSIZE,
        max_evals        = MAX_EVALS,
        w_disp           = W_DISP,
        w_vel            = W_VEL,
        n_workers        = N_WORKERS,
        suppress_output  = SUPPRESS_OUTPUT,
        response_weights = RESPONSE_WEIGHTS,
    )
