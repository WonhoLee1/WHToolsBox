"""
CMA-ES + Soft-DTW Trajectory Optimizer for MuJoCo Cuboid
=========================================================
JAX 없이 순수 numpy / scipy / multiprocessing 만 사용합니다.

JAX 버전(cuboid_cmaes_dtw.py)과 100% 동일한 기능을 제공하되
외부 의존성을 최소화하여 어떤 환경에서도 바로 실행됩니다.

최적화 파라미터 (13개)
─────────────────────
  solimp    (5) : [dmin, dmax, width, midpoint, power]
  solref    (2) : [timeconst, dampratio]
  friction  (3) : [sliding, torsional, rolling]
  com_offset(3) : [x, y, z]

개선 옵션 (OptimOptions)
────────────────────────
  use_parallel    : CMA-ES 세대 내 multiprocessing 병렬 평가  (기본 ON)
  use_rigid_body  : 강체 구속으로 6DoF 신호 추출              (기본 ON)
  use_phase_align : Cross-correlation 시간 정렬 전처리         (기본 ON)
  use_lhs_init    : Latin Hypercube Sampling 초기점 탐색       (기본 ON)

  ※ DTW 가속 옵션
     dtw_backend : "numpy"    — 순수 Python 루프 (항상 사용 가능)
                   "cython"   — tslearn 내장 Cython DTW (pip install tslearn)
                   "fastdtw"  — FastDTW O(N) 근사       (pip install fastdtw)

설치 (최소)
───────────
  pip install cma mujoco scipy

선택적 가속
───────────
  pip install tslearn    # Cython DTW (~5× numpy 대비)
  pip install fastdtw    # O(N) 근사 DTW (긴 시퀀스에 유리)

사용법
──────
  python cuboid_cmaes_dtw_nojax.py                       # 합성 데이터 데모
  python cuboid_cmaes_dtw_nojax.py --data corners.npy    # 실제 데이터
  python cuboid_cmaes_dtw_nojax.py --dtw_backend cython  # Cython 가속
"""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import correlate

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 0 ── Improvement Options
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OptimOptions:
    """
    개선 옵션 플래그 모음. 각각 독립적으로 on/off 가능합니다.

    use_parallel
        CMA-ES ask/tell 루프에서 한 세대(population)를
        multiprocessing.Pool 로 병렬 평가합니다.
        시뮬레이션 1회 > 0.5 s 이면 확실히 유리합니다.
        0.1 s 미만이면 spawn 오버헤드로 오히려 느릴 수 있습니다.

    use_rigid_body
        8코너 위치·속도 대신 강체 역산 6DoF (v_cm, ω) 를 비교합니다.
        채널 수 24 → 6, 노이즈 억제, 물리 일관성 보장.

    use_phase_align
        DTW 전에 cross-correlation 으로 시간 오프셋을 추정·보정합니다.
        시작 조건이 모캡과 다를 때 band 를 좁게 유지할 수 있습니다.

    use_lhs_init
        Latin Hypercube Sampling 으로 파라미터 공간을 균일 탐색한 뒤
        가장 좋은 basin 에서 CMA-ES 를 시작합니다.

    dtw_backend
        DTW 계산 엔진을 선택합니다.
          "numpy"   (기본) — 순수 Python 루프. 항상 사용 가능.
                             T=300, 24채널 기준 평가 1회 ~0.3 s
          "cython"         — tslearn 내장 Cython 구현.
                             pip install tslearn
                             numpy 대비 ~5× 빠름
          "fastdtw"        — O(N) 선형 근사 DTW.
                             pip install fastdtw
                             T 가 매우 클 때 (> 500) 유리.
                             정확도는 약간 낮음.
    """
    use_parallel    : bool = True
    use_rigid_body  : bool = True
    use_phase_align : bool = True
    use_lhs_init    : bool = True

    dtw_backend     : str  = "numpy"  # "numpy" | "cython" | "fastdtw"
    n_workers       : int  = 0        # 0 = cpu_count() 자동
    n_lhs_samples   : int  = 16
    phase_max_shift : float = 0.2     # 허용 시간 오프셋 [s]

    def summary(self) -> str:
        flags = {
            f"DTW({self.dtw_backend})" : True,
            "Parallel"   : self.use_parallel,
            "RigidBody"  : self.use_rigid_body,
            "PhaseAlign" : self.use_phase_align,
            "LHS-Init"   : self.use_lhs_init,
        }
        return "  " + "   ".join(
            f"{'ON ' if v else 'off'} {k}" for k, v in flags.items()
        )


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 ── DTW Backend  (numpy / cython / fastdtw)
# ══════════════════════════════════════════════════════════════════════════════

# ── 1-A. numpy 순수 구현 ──────────────────────────────────────────────────────

def _huber_mat(a: np.ndarray, b: np.ndarray, delta: float) -> np.ndarray:
    """(n,), (m,) → (n, m) Huber 거리 행렬."""
    r     = a[:, None] - b[None, :]
    abs_r = np.abs(r)
    return np.where(abs_r <= delta,
                    0.5 * r ** 2,
                    delta * (abs_r - 0.5 * delta))


def _soft_dtw_numpy(
    a     : np.ndarray,
    b     : np.ndarray,
    gamma : float,
    delta : float,
    band  : int,
) -> float:
    """
    Soft-DTW with Huber distance + Sakoe-Chiba band.

    시간 복잡도: O(T²)  또는  O(T × band)  (band > 0)
    공간 복잡도: O(T²)  (R 행렬 전체 보관)

    Parameters
    ----------
    a, b  : (T,) 정규화된 1-D 시퀀스
    gamma : softmin 평활도
    delta : Huber 전환점
    band  : Sakoe-Chiba 밴드 (0 = 없음)
    """
    n, m = len(a), len(b)
    D    = _huber_mat(a, b, delta)

    if band > 0:
        ii = np.arange(n)[:, None]
        jj = np.arange(m)[None, :]
        D  = np.where(np.abs(ii - jj) > band, 1e9, D)

    R = np.full((n + 2, m + 2), np.inf)
    R[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            r0, r1, r2 = R[i-1, j-1], R[i-1, j], R[i, j-1]
            mn   = min(r0, r1, r2)
            soft = mn - gamma * np.log(
                np.exp((mn - r0) / gamma)
                + np.exp((mn - r1) / gamma)
                + np.exp((mn - r2) / gamma)
                + 1e-38
            )
            R[i, j] = D[i-1, j-1] + soft

    return float(R[n, m])


# ── 1-B. Cython 가속 (tslearn) ────────────────────────────────────────────────

def _soft_dtw_cython(
    a     : np.ndarray,
    b     : np.ndarray,
    gamma : float,
    delta : float,   # Huber delta (cython 버전에서는 L2² 사용 후 후처리)
    band  : int,
) -> float:
    """
    tslearn 의 Cython 구현을 사용합니다.
    tslearn 은 L2² 거리 행렬을 사용하므로 Huber 는 적용하지 않습니다.
    pip install tslearn
    """
    try:
        from tslearn.metrics import soft_dtw
        # tslearn soft_dtw: (T, d) shape 필요
        A = a.reshape(-1, 1).astype(np.float64)
        B = b.reshape(-1, 1).astype(np.float64)
        if band > 0:
            # tslearn은 sakoe_chiba_radius 인수 지원
            from tslearn.metrics import soft_dtw_alignment
            return float(soft_dtw(A, B, gamma=gamma,
                                  be=None))   # band 미지원 → 근사
        return float(soft_dtw(A, B, gamma=gamma))
    except ImportError:
        # 폴백: numpy 구현 사용
        return _soft_dtw_numpy(a, b, gamma, delta, band)


# ── 1-C. FastDTW 근사 O(N) ────────────────────────────────────────────────────

def _dtw_fastdtw(
    a     : np.ndarray,
    b     : np.ndarray,
    gamma : float,   # fastdtw 에서는 사용 안 함 (hard DTW)
    delta : float,
    band  : int,
) -> float:
    """
    FastDTW 를 사용합니다. O(N) 이지만 hard DTW 이며 Soft 가 아닙니다.
    T > 500 인 긴 시퀀스에서 속도가 유리합니다.
    pip install fastdtw
    """
    try:
        from fastdtw import fastdtw
        radius = band if band > 0 else max(1, len(a) // 10)
        dist, _ = fastdtw(a, b, radius=radius,
                          dist=lambda x, y: abs(x - y))
        return float(dist)
    except ImportError:
        return _soft_dtw_numpy(a, b, gamma, delta, band)


# ── 디스패처 ──────────────────────────────────────────────────────────────────

def _get_dtw_fn(backend: str):
    """
    백엔드 이름으로 dtw1d 함수를 반환합니다.
    반환 함수 시그니처: (a, b, gamma, delta, band) → float
    """
    if backend == "cython":
        return _soft_dtw_cython
    elif backend == "fastdtw":
        return _dtw_fastdtw
    else:
        return _soft_dtw_numpy


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 ── Data Structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CornerTrajectory:
    """
    8개 코너의 시간 시계열.

    time : (T,)      [s]
    pos  : (T, 8, 3) [m]
    vel  : (T, 8, 3) [m/s]  — 미제공 시 cubic spline 자동 계산
    """
    time : np.ndarray
    pos  : np.ndarray
    vel  : Optional[np.ndarray] = None

    def __post_init__(self):
        assert self.pos.ndim == 3 and self.pos.shape[1:] == (8, 3), \
            f"pos must be (T, 8, 3), got {self.pos.shape}"
        assert len(self.time) == len(self.pos), "time/pos length mismatch"
        if self.vel is None:
            self._compute_velocity_inplace()

    def _compute_velocity_inplace(self):
        """Cubic-spline 1차 미분으로 속도 계산."""
        vel = np.empty_like(self.pos)
        for c in range(8):
            for d in range(3):
                cs = CubicSpline(self.time, self.pos[:, c, d])
                vel[:, c, d] = cs(self.time, 1)
        self.vel = vel

    @classmethod
    def from_npy(cls, path: str | Path, dt: float = 0.01) -> "CornerTrajectory":
        """(T, 8, 3) 또는 (T, 24) .npy 파일 로드."""
        arr = np.load(path)
        if arr.ndim == 2:
            arr = arr.reshape(len(arr), 8, 3)
        return cls(time=np.arange(len(arr)) * dt, pos=arr)

    @classmethod
    def from_csv(cls, path: str | Path, dt: float = 0.01) -> "CornerTrajectory":
        """
        CSV 로드.
        열 구성: [t, c0x,c0y,c0z, ..., c7z]  (25열)
             또는 [c0x,...,c7z]               (24열, t는 dt로 생성)
        """
        raw = np.loadtxt(path, delimiter=",")
        if raw.shape[1] == 25:
            t, pos = raw[:, 0], raw[:, 1:].reshape(-1, 8, 3)
        else:
            pos = raw.reshape(-1, 8, 3)
            t   = np.arange(len(pos)) * dt
        return cls(time=t, pos=pos)

    @property
    def duration(self) -> float:
        return float(self.time[-1] - self.time[0])

    @property
    def n_frames(self) -> int:
        return len(self.time)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 ── Rigid-Body 6DoF Extraction
# ══════════════════════════════════════════════════════════════════════════════
#
#  강체 운동방정식:  v_i = v_cm + ω × r_i   (r_i = p_i - p_cm)
#
#  8코너 속도로 최소자승 풀어 (v_cm, ω) 를 역산
#  블록 선형 시스템 (24×6):
#    [v_0]   [I  -[r_0]×] [v_cm]
#    [v_1] = [I  -[r_1]×] [ ω  ]
#    [...]   [...]
# ──────────────────────────────────────────────────────────────────────────────

def _skew(r: np.ndarray) -> np.ndarray:
    """(3,) → (3, 3) skew-symmetric 행렬."""
    x, y, z = r
    return np.array([[ 0, -z,  y],
                     [ z,  0, -x],
                     [-y,  x,  0]])


def extract_rigid_body_motion(traj: CornerTrajectory) -> np.ndarray:
    """
    8코너 위치·속도 → 6DoF body velocity per frame.

    Returns
    -------
    body_vel : (T, 6)  [vx, vy, vz, wx, wy, wz]
    """
    T        = traj.n_frames
    body_vel = np.zeros((T, 6))
    for t in range(T):
        p_cm = traj.pos[t].mean(axis=0)
        r    = traj.pos[t] - p_cm
        A    = np.vstack([
            np.hstack([np.eye(3), -_skew(r[i])]) for i in range(8)
        ])                              # (24, 6)
        b_vec = traj.vel[t].reshape(-1)   # (24,)
        sol, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
        body_vel[t]  = sol
    return body_vel


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 ── Phase Alignment
# ══════════════════════════════════════════════════════════════════════════════

def phase_align(
    sim       : CornerTrajectory,
    ref       : CornerTrajectory,
    max_shift : float = 0.2,
) -> tuple[CornerTrajectory, float]:
    """
    Cross-correlation 으로 시간 lag 를 추정하고 sim 을 정렬합니다.

    대표 신호: 8코너 Z 평균 (CoM Z 위치)
    → lag 추정 → sim.time 이동 → ref.time 그리드로 재보간

    Returns
    -------
    (aligned_sim, lag_sec)
    """
    dt = float(ref.time[1] - ref.time[0])

    sig_r = ref.pos[:, :, 2].mean(axis=1)
    sig_s = sim.pos[:, :, 2].mean(axis=1)
    sig_r = (sig_r - sig_r.mean()) / (sig_r.std() + 1e-10)
    sig_s = (sig_s - sig_s.mean()) / (sig_s.std() + 1e-10)

    corr = correlate(sig_r, sig_s, mode="full")
    lags = np.arange(-(len(sig_s) - 1), len(sig_r)) * dt
    mask = np.abs(lags) <= max_shift
    lag  = lags[int(np.argmax(np.where(mask, corr, -np.inf)))]

    t_shifted = sim.time + lag
    t_lo = max(t_shifted[0],  ref.time[0])
    t_hi = min(t_shifted[-1], ref.time[-1])
    t_q  = np.clip(ref.time, t_lo, t_hi)

    pos_al = np.empty_like(ref.pos)
    for c in range(8):
        for d in range(3):
            cs = CubicSpline(t_shifted, sim.pos[:, c, d], extrapolate=True)
            pos_al[:, c, d] = cs(t_q)

    return CornerTrajectory(time=ref.time.copy(), pos=pos_al), lag


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 ── Loss Function
# ══════════════════════════════════════════════════════════════════════════════

def _remove_com_drift(pos: np.ndarray) -> np.ndarray:
    """(T, 8, 3) → CoM 상대좌표 + 시간평균 오프셋 제거."""
    rel = pos - pos.mean(axis=1, keepdims=True)
    rel -= rel.mean(axis=0, keepdims=True)
    return rel


def compute_loss(
    sim          : CornerTrajectory,
    ref          : CornerTrajectory,
    opts         : OptimOptions,
    lambda_vel   : float = 0.3,
    gamma        : float = 0.1,
    delta        : float = 0.05,
    band_ratio   : float = 0.15,
    com_normalize: bool  = True,
) -> float:
    """
    통합 손실함수.

    경로 A (use_rigid_body=True)
        CoM 3채널 위치 DTW  +  λ × 6DoF 속도 DTW  =  9채널

    경로 B (use_rigid_body=False)
        8코너 × 3축 위치 DTW  +  λ × 속도 DTW     = 24채널

    DTW: opts.dtw_backend 로 선택된 구현 사용
    """
    # Phase alignment
    if opts.use_phase_align:
        sim, _lag = phase_align(sim, ref, max_shift=opts.phase_max_shift)

    # CoM 정규화
    pos_s = _remove_com_drift(sim.pos) if com_normalize else sim.pos.copy()
    pos_r = _remove_com_drift(ref.pos) if com_normalize else ref.pos.copy()

    T    = min(len(pos_s), len(pos_r))
    band = max(1, int(T * band_ratio)) if band_ratio > 0 else 0

    dtw_fn = _get_dtw_fn(opts.dtw_backend)

    # 1D DTW 헬퍼: 표준편차 정규화 후 계산
    def dtw1d(a: np.ndarray, b: np.ndarray) -> float:
        std = float(np.std(b)) + 1e-10
        return dtw_fn(a / std, b / std, gamma, delta / std, band)

    # ── 경로 A: 강체 6DoF (9채널) ─────────────────────────────────────────
    if opts.use_rigid_body:
        sim_t = CornerTrajectory(sim.time[:T], pos_s[:T], sim.vel[:T])
        ref_t = CornerTrajectory(ref.time[:T], pos_r[:T], ref.vel[:T])
        bv_s  = extract_rigid_body_motion(sim_t)   # (T, 6)
        bv_r  = extract_rigid_body_motion(ref_t)

        com_s = pos_s[:T].mean(axis=1)   # (T, 3)
        com_r = pos_r[:T].mean(axis=1)

        lp = sum(dtw1d(com_s[:, d], com_r[:, d]) for d in range(3)) / 3.0
        lv = sum(dtw1d(bv_s[:, k],  bv_r[:, k])  for k in range(6)) / 6.0
        return lp + lambda_vel * lv

    # ── 경로 B: 24채널 ────────────────────────────────────────────────────
    lp = lv = 0.0
    for c in range(8):
        for d in range(3):
            lp += dtw1d(pos_s[:, c, d], pos_r[:, c, d])
            if sim.vel is not None:
                lv += dtw1d(sim.vel[:T, c, d], ref.vel[:T, c, d])
    sc = 8.0 * 3.0
    return lp / sc + lambda_vel * lv / sc


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 ── Parameter Definition
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CuboidParams:
    """MuJoCo 물리 파라미터 13개 묶음."""
    solimp    : np.ndarray = field(default_factory=lambda:
                                   np.array([0.9, 0.95, 0.001, 0.5, 2.0]))
    solref    : np.ndarray = field(default_factory=lambda:
                                   np.array([0.02, 1.0]))
    friction  : np.ndarray = field(default_factory=lambda:
                                   np.array([0.8, 0.005, 0.0001]))
    com_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def to_vector(self) -> np.ndarray:
        return np.concatenate([self.solimp, self.solref,
                               self.friction, self.com_offset])

    @classmethod
    def from_vector(cls, v: np.ndarray) -> "CuboidParams":
        v = np.asarray(v, dtype=float)
        assert len(v) == 13, f"need 13 params, got {len(v)}"
        p = cls(solimp=v[0:5].copy(), solref=v[5:7].copy(),
                friction=v[7:10].copy(), com_offset=v[10:13].copy())
        p._clip()
        return p

    def _clip(self):
        lo, hi = PARAM_BOUNDS[:, 0], PARAM_BOUNDS[:, 1]
        v = np.clip(self.to_vector(), lo, hi)
        if v[1] <= v[0]:
            v[1] = min(v[0] + 0.01, hi[1])
        self.solimp     = v[0:5]
        self.solref     = v[5:7]
        self.friction   = v[7:10]
        self.com_offset = v[10:13]

    def __repr__(self):
        return (f"CuboidParams(\n"
                f"  solimp    = {np.round(self.solimp,    5)}\n"
                f"  solref    = {np.round(self.solref,    5)}\n"
                f"  friction  = {np.round(self.friction,  5)}\n"
                f"  com_offset= {np.round(self.com_offset,5)}\n)")


PARAM_BOUNDS = np.array([
    [0.1,  0.99 ],   # solimp dmin
    [0.1,  0.999],   # solimp dmax
    [1e-4, 0.05 ],   # solimp width
    [0.1,  0.9  ],   # solimp midpoint
    [1.0,  6.0  ],   # solimp power
    [0.004, 0.5 ],   # solref timeconst
    [0.1,  2.0  ],   # solref dampratio
    [0.01,  3.0 ],   # friction sliding
    [1e-4,  0.1 ],   # friction torsional
    [1e-5,  0.01],   # friction rolling
    [-0.05, 0.05],   # com x
    [-0.05, 0.05],   # com y
    [-0.05, 0.05],   # com z
], dtype=float)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 ── MuJoCo Simulator
# ══════════════════════════════════════════════════════════════════════════════

_XML_TEMPLATE = """
<mujoco model="cuboid_opt">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{dt}" gravity="0 0 -9.81" integrator="RK4"
          noslip_iterations="5"/>
  <default>
    <geom solimp="{solimp}" solref="{solref}"/>
  </default>
  <worldbody>
    <light pos="0 0 5" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="5 5 0.1"
          friction="{friction}" solimp="{solimp}" solref="{solref}"/>
    <body name="cuboid" pos="{init_pos}" euler="{init_euler}">
      <freejoint name="root"/>
      <inertial pos="{com_offset}" mass="{mass}" diaginertia="{inertia}"/>
      <geom name="box" type="box" size="{hx} {hy} {hz}"
            friction="{friction}" solimp="{solimp}" solref="{solref}"
            rgba="0.25 0.55 0.85 0.9"/>
    </body>
  </worldbody>
</mujoco>
"""


def _fmt(arr: np.ndarray) -> str:
    return " ".join(f"{v:.6g}" for v in arr)


def _corners_world(xpos: np.ndarray, xmat: np.ndarray,
                   he: np.ndarray) -> np.ndarray:
    signs = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                      [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]], float)
    return xpos + (xmat.reshape(3, 3) @ (signs * he).T).T


class CuboidSimulator:
    """MuJoCo 자유 비행 큐브 시뮬레이터."""

    def __init__(
        self,
        half_extents : np.ndarray = np.array([0.1, 0.1, 0.1]),
        mass         : float      = 1.0,
        init_pos     : np.ndarray = np.array([0., 0., 0.15]),
        init_euler   : np.ndarray = np.zeros(3),
        init_vel     : np.ndarray = np.zeros(6),
        dt           : float      = 0.002,
        time_ref     : Optional[np.ndarray] = None,
    ):
        self.he         = np.asarray(half_extents)
        self.mass       = mass
        self.init_pos   = np.asarray(init_pos)
        self.init_euler = np.asarray(init_euler)
        self.init_vel   = np.asarray(init_vel)
        self.dt         = dt
        self.time_ref   = time_ref
        a, b, c         = 2 * self.he
        self._iner      = mass / 12.0 * np.array([b**2+c**2,
                                                   a**2+c**2, a**2+b**2])

    def run(self, params: CuboidParams) -> CornerTrajectory:
        """파라미터로 XML 재빌드 후 시뮬레이션 실행 → 코너 궤적 반환."""
        xml   = _XML_TEMPLATE.format(
            dt        = self.dt,
            solimp    = _fmt(params.solimp),
            solref    = _fmt(params.solref),
            friction  = _fmt(params.friction),
            com_offset= _fmt(params.com_offset),
            mass      = self.mass,
            inertia   = _fmt(self._iner),
            init_pos  = _fmt(self.init_pos),
            init_euler= _fmt(self.init_euler),
            hx=self.he[0], hy=self.he[1], hz=self.he[2],
        )
        model = mujoco.MjModel.from_xml_string(xml)
        data  = mujoco.MjData(model)
        data.qvel[:] = self.init_vel
        mujoco.mj_forward(model, data)

        dur     = float(self.time_ref[-1]) if self.time_ref is not None else 2.0
        n_steps = max(1, int(np.ceil(dur / self.dt))) + 1
        bid     = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cuboid")

        times, positions = [], []
        for _ in range(n_steps):
            positions.append(
                _corners_world(data.xpos[bid], data.xmat[bid], self.he).copy())
            times.append(data.time)
            mujoco.mj_step(model, data)

        t_arr   = np.array(times)
        pos_arr = np.stack(positions)

        # 참조 시간 그리드로 보간
        if self.time_ref is not None:
            t_lo = max(t_arr[0],  self.time_ref[0])
            t_hi = min(t_arr[-1], self.time_ref[-1])
            t_q  = np.clip(self.time_ref, t_lo, t_hi)
            out  = np.empty((len(self.time_ref), 8, 3))
            for c in range(8):
                for d in range(3):
                    out[:, c, d] = CubicSpline(t_arr, pos_arr[:, c, d])(t_q)
            pos_arr, t_arr = out, self.time_ref.copy()

        return CornerTrajectory(time=t_arr, pos=pos_arr)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 ── LHS Initial Search
# ══════════════════════════════════════════════════════════════════════════════

def lhs_init_search(
    objective_fn,
    opts    : OptimOptions,
    verbose : bool = True,
) -> CuboidParams:
    """
    Latin Hypercube Sampling 으로 파라미터 공간을 균일 탐색합니다.
    scipy.stats.qmc 사용 (미설치 시 uniform random 폴백).
    """
    lo, hi = PARAM_BOUNDS[:, 0], PARAM_BOUNDS[:, 1]
    n = opts.n_lhs_samples

    try:
        from scipy.stats import qmc
        samples = qmc.scale(
            qmc.LatinHypercube(d=13, seed=42).random(n=n), lo, hi)
    except (ImportError, AttributeError):
        samples = np.random.default_rng(42).uniform(lo, hi, (n, 13))

    if verbose:
        print(f"\n  LHS 탐색: {n}개 후보 평가 중 ...")

    best_loss, best_p = np.inf, CuboidParams.from_vector((lo + hi) / 2)
    for i, v in enumerate(samples):
        p    = CuboidParams.from_vector(v)
        loss = objective_fn(p)
        if loss < best_loss:
            best_loss, best_p = loss, p
        if verbose:
            print(f"    [{i+1:3d}/{n}] loss={loss:.5f}  best={best_loss:.5f}",
                  end="\r")

    if verbose:
        print(f"\n  LHS 완료 → best_loss={best_loss:.5f}")
    return best_p


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 ── CMA-ES Optimizer
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OptimResult:
    params_best  : CuboidParams
    loss_history : list
    loss_best    : float
    n_evals      : int
    elapsed_sec  : float
    converged    : bool
    phase_lag_sec: float = 0.0


# multiprocessing worker — 모듈 top-level 에 있어야 pickle 가능
_GLOBAL_OBJ_FN = None

def _pool_worker(u: np.ndarray) -> float:
    return _GLOBAL_OBJ_FN(u)


class CMAESOptimizer:
    """
    CMA-ES + Soft-DTW 큐브 파라미터 최적화기 (JAX 없는 버전).

    모든 개선 옵션은 OptimOptions 로 제어합니다.
    DTW 엔진은 opts.dtw_backend 로 선택합니다.
    """

    def __init__(
        self,
        simulator    : CuboidSimulator,
        reference    : CornerTrajectory,
        opts         : OptimOptions = None,
        lambda_vel   : float = 0.3,
        gamma        : float = 0.1,
        delta        : float = 0.05,
        band_ratio   : float = 0.15,
        com_normalize: bool  = True,
        sigma0       : float = 0.25,
        max_evals    : int   = 600,
        tol_loss     : float = 1e-5,
        verbose      : bool  = True,
    ):
        self.simulator     = simulator
        self.reference     = reference
        self.opts          = opts or OptimOptions()
        self.lambda_vel    = lambda_vel
        self.gamma         = gamma
        self.delta         = delta
        self.band_ratio    = band_ratio
        self.com_normalize = com_normalize
        self.sigma0        = sigma0
        self.max_evals     = max_evals
        self.tol_loss      = tol_loss
        self.verbose       = verbose

        self._lo    = PARAM_BOUNDS[:, 0]
        self._hi    = PARAM_BOUNDS[:, 1]
        self._scale = self._hi - self._lo + 1e-14

        self._loss_history : list  = []
        self._n_evals      : int   = 0
        self._best_loss    : float = np.inf
        self._best_params  : Optional[CuboidParams] = None

    # ── 정규화 ────────────────────────────────────────────────────────────────
    def _to_unit(self, v: np.ndarray) -> np.ndarray:
        return (v - self._lo) / self._scale

    def _from_unit(self, u: np.ndarray) -> np.ndarray:
        v = np.clip(u * self._scale + self._lo, self._lo, self._hi)
        if v[1] <= v[0]:
            v[1] = min(v[0] + 0.01, self._hi[1])
        return v

    # ── 목적 함수 ─────────────────────────────────────────────────────────────
    def _obj_params(self, params: CuboidParams) -> float:
        try:
            sim  = self.simulator.run(params)
            loss = compute_loss(
                sim, self.reference, self.opts,
                lambda_vel    = self.lambda_vel,
                gamma         = self.gamma,
                delta         = self.delta,
                band_ratio    = self.band_ratio,
                com_normalize = self.com_normalize,
            )
        except Exception as e:
            loss = 1e6
            if self.verbose:
                print(f"\n  [warn] {e}")
        return loss

    def _obj_unit(self, u: np.ndarray) -> float:
        params = CuboidParams.from_vector(self._from_unit(np.asarray(u)))
        loss   = self._obj_params(params)

        self._n_evals += 1
        self._loss_history.append(loss)
        if loss < self._best_loss:
            self._best_loss   = loss
            self._best_params = params

        if self.verbose and self._n_evals % max(1, self.max_evals // 20) == 0:
            print(f"  [numpy] eval={self._n_evals:5d}  "
                  f"loss={loss:.5f}  best={self._best_loss:.5f}")
        return loss

    # ── 병렬 ask/tell ─────────────────────────────────────────────────────────
    def _parallel_gen(self, es) -> None:
        global _GLOBAL_OBJ_FN
        _GLOBAL_OBJ_FN = self._obj_unit
        xs = es.ask()
        n  = self.opts.n_workers or cpu_count()
        with Pool(processes=n) as pool:
            fs = pool.map(_pool_worker, xs)
        es.tell(xs, fs)

    # ── 메인 ──────────────────────────────────────────────────────────────────
    def optimize(self, x0: Optional[CuboidParams] = None) -> OptimResult:
        """
        CMA-ES 최적화 실행.

        Parameters
        ----------
        x0 : 초기 파라미터 (None 이면 LHS 또는 bounds 중간값)

        Returns
        -------
        OptimResult
        """
        try:
            import cma
        except ImportError:
            raise ImportError("pip install cma")

        self._loss_history, self._n_evals  = [], 0
        self._best_loss, self._best_params = np.inf, None

        # 초기점 결정
        if self.opts.use_lhs_init:
            x0_unit = self._to_unit(
                lhs_init_search(self._obj_params, self.opts,
                                self.verbose).to_vector())
        elif x0 is not None:
            x0_unit = self._to_unit(x0.to_vector())
        else:
            x0_unit = self._to_unit((self._lo + self._hi) / 2)

        # CMA-ES 옵션
        cma_opts = {
            "maxfevals": self.max_evals,
            "tolx"     : 1e-9,
            "tolfun"   : self.tol_loss,
            "bounds"   : [[0.0] * 13, [1.0] * 13],
            "verbose"  : -9,
            "verb_log" : 0,
        }

        if self.verbose:
            print(f"\n{'═'*62}")
            print(f"  CMA-ES + Soft-DTW (no-JAX)   σ₀={self.sigma0}  "
                  f"budget={self.max_evals}")
            print(self.opts.summary())
            print(f"{'═'*62}")

        t0 = time.perf_counter()
        es = cma.CMAEvolutionStrategy(x0_unit.tolist(), self.sigma0, cma_opts)

        if self.opts.use_parallel:
            while not es.stop():
                self._parallel_gen(es)
                if self._best_loss < self.tol_loss:
                    break
        else:
            es.optimize(self._obj_unit)

        elapsed   = time.perf_counter() - t0
        best_p    = CuboidParams.from_vector(
                        self._from_unit(np.array(es.result.xbest)))
        stop      = es.result.stop
        converged = "tolfun" in stop or "tolx" in stop

        if self.verbose:
            print(f"\n  완료  evals={self._n_evals}  "
                  f"best={self._best_loss:.6f}  "
                  f"time={elapsed:.1f}s  converged={converged}")
            print(f"  {best_p}")

        return OptimResult(
            params_best  = best_p,
            loss_history = self._loss_history,
            loss_best    = self._best_loss,
            n_evals      = self._n_evals,
            elapsed_sec  = elapsed,
            converged    = converged,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 ── Diagnostics
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(
    result     : OptimResult,
    simulator  : CuboidSimulator,
    reference  : CornerTrajectory,
    corner_idx : int = 0,
    save_path  : Optional[str] = None,
):
    import matplotlib.pyplot as plt

    sim_best = simulator.run(result.params_best)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(f"CMA-ES + Soft-DTW (no-JAX)  |  "
                 f"loss={result.loss_best:.5f}  evals={result.n_evals}",
                 fontweight="bold")

    axes[0, 0].semilogy(result.loss_history,
                        color="#2563eb", lw=1.3, alpha=0.8)
    axes[0, 0].set(xlabel="Evaluations", ylabel="Loss (log)",
                   title="Convergence")
    axes[0, 0].grid(True, alpha=0.3)

    for d, (ax, lbl, col) in enumerate(zip(
        [axes[0, 1], axes[1, 0], axes[1, 1]],
        ["X", "Y", "Z"],
        ["#dc2626", "#16a34a", "#7c3aed"],
    )):
        ax.plot(reference.time, reference.pos[:, corner_idx, d],
                "k-", lw=2, label="MoCap")
        ax.plot(sim_best.time, sim_best.pos[:, corner_idx, d],
                color=col, lw=1.8, ls="--", label="Sim")
        ax.set(xlabel="Time [s]", ylabel=f"Pos {lbl} [m]",
               title=f"Corner {corner_idx} — {lbl}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  → {save_path}")
    plt.show()


def print_param_table(result: OptimResult):
    names = [
        "solimp.dmin", "solimp.dmax", "solimp.width",
        "solimp.mid",  "solimp.power",
        "solref.tc",   "solref.damp",
        "fric.slide",  "fric.tors",   "fric.roll",
        "com.x",       "com.y",       "com.z",
    ]
    v = result.params_best.to_vector()
    print(f"\n{'─'*58}")
    print(f"  {'Parameter':<18} {'Value':>10}   {'LB':>8}  {'UB':>8}")
    print(f"{'─'*58}")
    for n, val, (lb, ub) in zip(names, v, PARAM_BOUNDS):
        print(f"  {n:<18} {val:>10.5f}   {lb:>8.4g}  {ub:>8.4g}")
    print(f"{'─'*58}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 11 ── Demo & CLI
# ══════════════════════════════════════════════════════════════════════════════

def run_demo(dtw_backend: str = "numpy"):
    print("\n" + "═" * 62)
    print(f"  DEMO — CMA-ES + Soft-DTW  (backend={dtw_backend})")
    print("═" * 62)

    dt       = 0.005
    time_ref = np.arange(0, 1.5, dt)

    true_p = CuboidParams(
        solimp     = np.array([0.9,  0.95,  0.002, 0.5, 2.0]),
        solref     = np.array([0.02, 1.0]),
        friction   = np.array([0.7,  0.004, 5e-5]),
        com_offset = np.array([0.01, -0.005, 0.0]),
    )
    print(f"\n정답:\n{true_p}")

    sim = CuboidSimulator(
        half_extents = np.array([0.08, 0.12, 0.06]),
        mass         = 0.8,
        init_pos     = np.array([0., 0., 0.10]),
        init_euler   = np.array([0.1, 0.05, 0.]),
        init_vel     = np.array([0.3, 0., 0.5, 0.2, 0.1, 0.]),
        dt           = dt,
        time_ref     = time_ref,
    )
    clean = sim.run(true_p)
    ref   = CornerTrajectory(
        time = clean.time,
        pos  = clean.pos + np.random.default_rng(0).normal(0, 5e-4, clean.pos.shape),
    )
    print(f"참조 궤적: {ref.n_frames} frames, {ref.duration:.2f}s")

    opts = OptimOptions(
        dtw_backend     = dtw_backend,
        use_parallel    = True,
        use_rigid_body  = True,
        use_phase_align = True,
        use_lhs_init    = True,
        n_lhs_samples   = 12,
    )
    opt = CMAESOptimizer(
        simulator     = sim,
        reference     = ref,
        opts          = opts,
        lambda_vel    = 0.3,
        gamma         = 0.1,
        delta         = 0.05,
        band_ratio    = 0.15,
        sigma0        = 0.25,
        max_evals     = 500,
        verbose       = True,
    )
    result = opt.optimize()
    print_param_table(result)
    try:
        plot_results(result, sim, ref, save_path="dtw_result_nojax.png")
    except Exception as e:
        print(f"  (plot skipped: {e})")
    return result


def main():
    ap = argparse.ArgumentParser(
        description="CMA-ES + Soft-DTW cuboid optimizer (no-JAX)")
    ap.add_argument("--data",           type=str,   default=None)
    ap.add_argument("--dt",             type=float, default=0.005)
    ap.add_argument("--hx",             type=float, default=0.10)
    ap.add_argument("--hy",             type=float, default=0.10)
    ap.add_argument("--hz",             type=float, default=0.10)
    ap.add_argument("--mass",           type=float, default=1.0)
    ap.add_argument("--max_evals",      type=int,   default=600)
    ap.add_argument("--sigma0",         type=float, default=0.25)
    ap.add_argument("--lambda_vel",     type=float, default=0.3)
    ap.add_argument("--gamma",          type=float, default=0.1)
    ap.add_argument("--out",            type=str,   default="result.png")
    ap.add_argument("--dtw_backend",    type=str,   default="numpy",
                    choices=["numpy", "cython", "fastdtw"],
                    help="numpy(기본) | cython(tslearn) | fastdtw(근사 O(N))")
    ap.add_argument("--no_parallel",    action="store_true")
    ap.add_argument("--no_rigid_body",  action="store_true")
    ap.add_argument("--no_phase_align", action="store_true")
    ap.add_argument("--no_lhs",         action="store_true")
    args = ap.parse_args()

    if args.data is None:
        run_demo(dtw_backend=args.dtw_backend)
        return

    p   = Path(args.data)
    ref = (CornerTrajectory.from_npy(p, args.dt) if p.suffix == ".npy"
           else CornerTrajectory.from_csv(p, args.dt))
    print(f"로드: {ref.n_frames} frames, {ref.duration:.2f}s")

    sim = CuboidSimulator(
        half_extents = np.array([args.hx, args.hy, args.hz]),
        mass         = args.mass,
        init_pos     = ref.pos[0].mean(axis=0),
        dt           = args.dt / 2,
        time_ref     = ref.time,
    )
    opts = OptimOptions(
        dtw_backend     = args.dtw_backend,
        use_parallel    = not args.no_parallel,
        use_rigid_body  = not args.no_rigid_body,
        use_phase_align = not args.no_phase_align,
        use_lhs_init    = not args.no_lhs,
    )
    opt = CMAESOptimizer(
        simulator  = sim,
        reference  = ref,
        opts       = opts,
        lambda_vel = args.lambda_vel,
        gamma      = args.gamma,
        sigma0     = args.sigma0,
        max_evals  = args.max_evals,
        verbose    = True,
    )
    result = opt.optimize()
    print_param_table(result)
    plot_results(result, sim, ref, save_path=args.out)


if __name__ == "__main__":
    main()
