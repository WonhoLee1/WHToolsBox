import os
import re
import sys
import logging
import numpy as np
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

def compute_corner_kinematics(
    center_pos: np.ndarray, 
    center_mat: np.ndarray, 
    center_vel: np.ndarray, 
    center_acc: np.ndarray, 
    box_w: float, box_h: float, box_d: float
) -> List[Dict[str, np.ndarray]]:
    """
    조립체 중심의 위치, 회전, 속도, 가속도 데이터로부터 8개 모서리 꼭지점의 
    글로벌 위치/속도/가속도를 강체 운동학(Rigid Body Kinematics) 공식으로 역산합니다.
    
    Args:
        center_pos (np.ndarray): 중심점의 3차원 글로벌 위치 [x, y, z]
        center_mat (np.ndarray): 3x3 회전 행렬 (body xmat)
        center_vel (np.ndarray): 6자유도 속도 [wx, wy, wz, vx, vy, vz]
        center_acc (np.ndarray): 6자유도 가속도 [alpha_x, alpha_y, alpha_z, ax, ay, az]
        box_w (float): 박스의 가로 길이 (Width, m)
        box_h (float): 박스의 세로 길이 (Height, m)
        box_d (float): 박스의 깊이 (Depth, m)
    
    Returns:
        List[Dict]: 8개 모서리의 {'pos': ndarray, 'vel': ndarray, 'acc': ndarray} 리스트
    """
    w = center_vel[0:3]     # 각속도 (Angular velocity)
    v = center_vel[3:6]     # 선속도 (Linear velocity)
    alpha = center_acc[0:3] # 각가속도 (Angular acceleration)
    a = center_acc[3:6]     # 선가속도 (Linear acceleration)
    
    corners_local = []
    # 8개 꼭지점의 로컬 좌표 생성
    for x in [-box_w / 2, box_w / 2]:
        for y in [-box_h / 2, box_h / 2]:
            for z in [-box_d / 2, box_d / 2]:
                corners_local.append(np.array([x, y, z]))
    
    results = []
    for loc in corners_local:
        # 글로벌 오프셋 벡터 (r = R * r_local)
        r = center_mat @ loc
        
        # 선속도 공식: v_p = v_cg + w × r
        v_corner = v + np.cross(w, r)
        
        # 선가속도 공식: a_p = a_cg + α × r + w × (w × r)
        a_corner = a + np.cross(alpha, r) + np.cross(w, np.cross(w, r))
        
        results.append({
            'pos': center_pos + r,
            'vel': v_corner,
            'acc': a_corner
        })
    return results

def calculate_required_aux_masses(
    config: Dict[str, Any],
    target_mass: float, 
    target_cog: Optional[Union[List[float], np.ndarray]] = None, 
    target_moi: Optional[Union[List[float], np.ndarray]] = None, 
    num_masses: int = 8,
    base_mci: Optional[Tuple[float, np.ndarray, np.ndarray]] = None
) -> List[Dict[str, Any]]:
    """
    설계 목표치(Target Mass, CoG, MoI)를 달성하기 위해 필요한 추가 보정 질량(Aux Masses)의 
    최적 위치와 크기를 역산합니다.
    """
    # [WHTOOLS] 재귀 방지: 외부에서 base 정보를 주거나, 직접 계산(재귀 없는 버전) 수행
    if base_mci is not None:
        m_base, c_base, i_base = base_mci
    else:
        # [WHTOOLS] Circular Import 방지를 위해 로컬 임포트 사용
        from run_discrete_builder.whtb_physics import _get_assembly_inertia_base
        temp_cfg = config.copy()
        temp_cfg["chassis_aux_masses"] = []
        temp_cfg["component_aux"] = {}
        m_base, c_base, i_base, _ = _get_assembly_inertia_base(temp_cfg)
    
    m_base = float(m_base)
    c_base = np.array(c_base)
    i_base = np.array(i_base)
    
    t_mass = target_mass if target_mass is not None else m_base
    t_cog  = np.array(target_cog) if (target_cog is not None and len(target_cog) == 3) else c_base
    t_moi  = np.array(target_moi) if (target_moi is not None and len(target_moi) >= 3) else None
    
    # 추가 필요 질량 (Target - Current)
    m_aux = t_mass - m_base
    if m_aux < 0:
        # 목표 질량이 현재보다 작으면 최소한의 질량으로 CoG만 보정 시도
        m_aux = 1e-4 
        
    # 보정 질량계의 평균 중심 좌표 (M_total * C_total = M_base * C_base + M_aux * C_aux)
    pos_aux = (t_cog * t_mass - m_base * c_base) / (m_aux if m_aux > 0 else 1e-6)

    bw, bh, bd = config.get('box_w', 2.0), config.get('box_h', 1.4), config.get('box_d', 0.25)
    limit_x, limit_y, limit_z = bw/2.0 * 2.0, bh/2.0 * 2.0, bd/2.0 * 2.0

    def to_pos(p):
        return [
            float(np.clip(p[0], -limit_x, limit_x)),
            float(np.clip(p[1], -limit_y, limit_y)),
            float(np.clip(p[2], -limit_z, limit_z))
        ]

    aux_masses = []

    # 배치 로직: 보충 질량 개수에 따라 기하학적으로 배분
    if num_masses <= 1 or t_moi is None:
        aux_masses.append({
            "name" : "AutoBalance_Single",
            "pos"  : to_pos(pos_aux),
            "mass" : float(m_aux),
            "size" : [0.01, 0.01, 0.01]
        })
    elif num_masses == 2:
        m_each = m_aux / 2.0
        i_needed = t_moi[1] - i_base[1] if t_moi is not None else 0
        dx = math.sqrt(max(0.005, i_needed / (2.0 * m_each)))
        for sx in [-1, 1]:
            p = [pos_aux[0] + sx * dx, pos_aux[1], pos_aux[2]]
            aux_masses.append({"name": f"AutoBalance_{len(aux_masses)+1}", "pos": to_pos(p), "mass": m_each, "size": [0.01]*3})
            
    else: 
        m_each = m_aux / 8.0
        # [WHTOOLS] 최적화 기반 밸런싱 (Scipy 활용)
        # 목표: Target MoI와의 오차를 최소화하는 dx, dy, dz 및 개별 질량 배분(alpha, beta, gamma) 탐색
        from scipy.optimize import minimize

        # 1. 기저 모델을 타겟 CoG로 이동시켰을 때의 관성 (평행축 정리)
        d = t_cog - c_base
        i_at_t = np.zeros(6)
        i_at_t[:3] = i_base[:3] + m_base * np.array([d[1]**2 + d[2]**2, d[0]**2 + d[2]**2, d[0]**2 + d[1]**2])
        if len(i_base) >= 6:
            i_at_t[3:6] = i_base[3:6] - m_base * np.array([d[0]*d[1], d[0]*d[2], d[1]*d[2]])
        
        di_target = t_moi - i_at_t

        def get_actual_moi_contribution(p):
            dx, dy, dz, a, b, g = p
            # Diagonal
            ixx = m_aux * (dy**2 + dz**2)
            iyy = m_aux * (dx**2 + dz**2)
            izz = m_aux * (dx**2 + dy**2)
            # Products — MuJoCo tensor convention: Ixy = -Σm·xi·yi
            ixy = -m_aux * dx * dy * a
            ixz = -m_aux * dx * dz * b
            iyz = -m_aux * dy * dz * g
            return np.array([ixx, iyy, izz, ixy, ixz, iyz])

        def objective(p):
            actual = get_actual_moi_contribution(p)
            # 가중치 부여 (Diagonal에 더 높은 우선순위)
            weights = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
            err = (actual - di_target) * weights
            return np.sum(err**2)

        # 제약 조건: m_i >= 0 (8개 지점)
        # m_i = m_avg * (1 + a*sx*sy + b*sx*sz + g*sy*sz)
        def constraint_mass(p):
            _, _, _, a, b, g = p
            m_min = 1.0
            for sx in [-1, 1]:
                for sy in [-1, 1]:
                    for sz in [-1, 1]:
                        m_min = min(m_min, 1.0 + a*sx*sy + b*sx*sz + g*sy*sz)
            return m_min # >= 0

        # 초기값 설정 (Analytic solution 기반)
        dx_init = math.sqrt(max(0.001, (di_target[1] + di_target[2] - di_target[0]) / (2.0 * m_aux))) if m_aux > 0 else 0.1
        dy_init = math.sqrt(max(0.001, (di_target[0] + di_target[2] - di_target[1]) / (2.0 * m_aux))) if m_aux > 0 else 0.1
        dz_init = math.sqrt(max(0.001, (di_target[0] + di_target[1] - di_target[2]) / (2.0 * m_aux))) if m_aux > 0 else 0.1
        p0 = [dx_init, dy_init, dz_init, 0.0, 0.0, 0.0]
        
        bounds = [(0.001, limit_x), (0.001, limit_y), (0.001, limit_z), (-0.95, 0.95), (-0.95, 0.95), (-0.95, 0.95)]
        cons = {'type': 'ineq', 'fun': constraint_mass}

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Values in x were outside bounds.*")
            res = minimize(objective, p0, bounds=bounds, constraints=cons, method='SLSQP', options={'maxiter': 100})
        
        dx_f, dy_f, dz_f, a_f, b_f, g_f = res.x
        
        m_list = []
        for sx in [-1, 1]:
            for sy in [-1, 1]:
                for sz in [-1, 1]:
                    m_this = m_each * (1.0 + a_f*sx*sy + b_f*sx*sz + g_f*sy*sz)
                    m_list.append(max(m_this, 1e-6))
        
        # 질량 보존 강제
        m_sum = sum(m_list)
        scale = m_aux / m_sum if m_sum > 1e-9 else 1.0
        
        idx = 0
        for sx in [-1, 1]:
            for sy in [-1, 1]:
                for sz in [-1, 1]:
                    m_final = m_list[idx] * scale
                    p = [pos_aux[0] + sx * dx_f, pos_aux[1] + sy * dy_f, pos_aux[2] + sz * dz_f]
                    aux_masses.append({
                        "name": f"AutoBalance_{len(aux_masses)+1}",
                        "pos": to_pos(p),
                        "mass": float(m_final),
                        "size": [0.01, 0.01, 0.01]
                    })
                    idx += 1
                    
    return aux_masses


class TerminalTable:
    """터미널 출력용 텍스트 테이블.

    열 구분선 없이 헤더 상하에만 구분선을 표시합니다.

    Usage:
        t = TerminalTable(["Corner", "X (m)", "Y (m)", "Z (m)"],
                          align=["l", "r", "r", "r"])
        t.add_row(["C1", "-0.98253", "1.65455", "0.00364"])
        print(t.render())          # 문자열 반환
        t.print()                  # 바로 출력
    """

    ALIGN_LEFT  = "l"
    ALIGN_RIGHT = "r"
    ALIGN_CENTER = "c"

    def __init__(self, headers: List[str],
                 align: Optional[List[str]] = None,
                 padding: int = 1,
                 line_char: str = "-"):
        self.headers   = headers
        self.align     = align or ["l"] * len(headers)
        self.padding   = padding
        self.line_char = line_char
        self._rows: List[List[str]] = []

    def add_row(self, row: List) -> None:
        self._rows.append([str(v) for v in row])

    def _col_widths(self) -> List[int]:
        widths = [len(h) for h in self.headers]
        for row in self._rows:
            for i, cell in enumerate(row):
                if i < len(widths):
                    widths[i] = max(widths[i], len(cell))
        return widths

    def _fmt_cell(self, text: str, width: int, align: str) -> str:
        p = " " * self.padding
        if align == self.ALIGN_RIGHT:
            return p + text.rjust(width) + p
        elif align == self.ALIGN_CENTER:
            return p + text.center(width) + p
        else:
            return p + text.ljust(width) + p

    def render(self) -> str:
        widths = self._col_widths()
        total  = sum(w + self.padding * 2 for w in widths)
        sep    = self.line_char * total

        lines = [sep]
        lines.append("".join(
            self._fmt_cell(h, widths[i], self.align[i] if i < len(self.align) else "l")
            for i, h in enumerate(self.headers)
        ))
        lines.append(sep)
        for row in self._rows:
            lines.append("".join(
                self._fmt_cell(row[i] if i < len(row) else "", widths[i],
                               self.align[i] if i < len(self.align) else "l")
                for i in range(len(self.headers))
            ))
        lines.append(sep)
        return "\n".join(lines)

    def print(self) -> None:
        print(self.render())


# ─── Session Logger ──────────────────────────────────────────────────────────

_ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;]*[mABCDEFGHJKSTfhilmnprsu]|\x1b\][^\x07]*\x07|\x1b[@-_][0-?]*[ -/]*[@-~]')


class _TeeStream:
    """stdout을 터미널과 파일에 동시 출력하는 스트림 래퍼."""
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> None:
        for s in self._streams:
            try:
                is_file = not (hasattr(s, 'isatty') and s.isatty())
                s.write(_ANSI_ESCAPE.sub('', data) if is_file else data)
            except Exception:
                pass

    def flush(self) -> None:
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        return hasattr(self._streams[0], 'isatty') and self._streams[0].isatty()

    def fileno(self) -> int:
        return self._streams[0].fileno()


class _StripRichMarkup(logging.Filter):
    """Rich 마크업 태그([bold red] 등)를 파일 로그에서 제거하는 필터."""
    _MARKUP = re.compile(r'\[/?[a-zA-Z0-9 _#]*\]')

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self._MARKUP.sub('', record.msg)
        return True


class WHToolsSessionLogger:
    """
    DropSimulator 생성 시 자동으로 시작되는 세션 로그 매니저.

    - 모든 print() 출력과 WHTS_Engine 로거를 whtoolsbox.log 파일에 저장
    - 매 세션마다 파일을 새로 덮어씀
    - 다른 세션이 파일을 점유 중이면 whtoolsbox_1.log, whtoolsbox_2.log 순으로 증가
    - .lock 파일 + PID 확인으로 좀비 락 자동 해제
    """
    _log_path:     Optional[Path] = None
    _lock_path:    Optional[Path] = None
    _log_file                     = None
    _file_handler: Optional[logging.FileHandler] = None
    _orig_stdout                  = None
    _started: bool                = False

    @classmethod
    def start(cls, base_dir: Optional[Path] = None) -> None:
        if cls._started:
            return
        base_dir = Path(base_dir) if base_dir else Path.cwd()

        log_path, lock_path = cls._acquire(base_dir)
        if log_path is None:
            return  # 슬롯 확보 실패 → 조용히 패스

        cls._log_path  = log_path
        cls._lock_path = lock_path

        # 로그 파일 열기 (overwrite)
        cls._log_file = open(log_path, 'w', encoding='utf-8', buffering=1)

        # stdout tee
        cls._orig_stdout = sys.stdout
        sys.stdout = _TeeStream(sys.stdout, cls._log_file)

        # logging FileHandler — Rich 마크업 제거 후 파일에 기록
        cls._file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        cls._file_handler.setFormatter(
            logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
        )
        cls._file_handler.addFilter(_StripRichMarkup())
        logging.getLogger("WHTS_Engine").addHandler(cls._file_handler)

        cls._started = True
        print(f"[WHTools] Session log → {log_path}")

    @classmethod
    def get_log_path(cls) -> Optional[Path]:
        return cls._log_path

    @classmethod
    def release(cls) -> None:
        """세션 종료 시 호출 — lock 파일 제거, 핸들러/스트림 복원."""
        if cls._lock_path and cls._lock_path.exists():
            try:
                cls._lock_path.unlink()
            except Exception:
                pass
        if cls._file_handler:
            logging.getLogger("WHTS_Engine").removeHandler(cls._file_handler)
            cls._file_handler = None
        if cls._log_file:
            try:
                cls._log_file.close()
            except Exception:
                pass
            cls._log_file = None
        if cls._orig_stdout:
            sys.stdout = cls._orig_stdout
            cls._orig_stdout = None
        cls._started   = False
        cls._log_path  = None
        cls._lock_path = None

    # ── 내부 헬퍼 ──────────────────────────────────────────────────────────

    @classmethod
    def _acquire(cls, base_dir: Path):
        """사용 가능한 (log_path, lock_path) 쌍을 반환. 실패 시 (None, None)."""
        candidates = [base_dir / "whtoolsbox.log"] + \
                     [base_dir / f"whtoolsbox_{i}.log" for i in range(1, 20)]

        for log_path in candidates:
            lock_path = log_path.with_suffix('.lock')
            if not lock_path.exists():
                # lock 없음 → 독점 생성 시도
                try:
                    fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.write(fd, str(os.getpid()).encode())
                    os.close(fd)
                    return log_path, lock_path
                except FileExistsError:
                    continue  # 경합 발생 → 다음 슬롯
            else:
                # lock 존재 → PID 살아있는지 확인
                try:
                    pid = int(lock_path.read_text().strip())
                    alive = cls._pid_alive(pid)
                except Exception:
                    alive = False

                if not alive:
                    # 좀비 락 → 교체 후 획득
                    try:
                        lock_path.write_text(str(os.getpid()))
                        return log_path, lock_path
                    except Exception:
                        continue
                # 살아있는 세션 → 다음 슬롯
        return None, None

    @classmethod
    def _pid_alive(cls, pid: int) -> bool:
        try:
            import psutil
            return psutil.pid_exists(pid)
        except ImportError:
            try:
                os.kill(pid, 0)
                return True
            except (OSError, ProcessLookupError):
                return False

import configparser
def get_external_tool_path(tool_name):
    try:
        cfg_path = None
        
        # PyInstaller 빌드 환경(frozen)인 경우
        if getattr(sys, 'frozen', False):
            # 1순위: 실행 파일(.exe) 바로 옆 루트 폴더 조회
            exe_dir = os.path.dirname(sys.executable)
            path_exe = os.path.join(exe_dir, 'external_tools_config.ini')
            if os.path.exists(path_exe):
                cfg_path = path_exe
            else:
                # 2순위: 없을 경우 번들 내부(_internal/ 루트) 조회
                bundle_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                path_bundle = os.path.join(bundle_dir, 'external_tools_config.ini')
                if os.path.exists(path_bundle):
                    cfg_path = path_bundle
                    
        # 일반 개발 환경이거나, 빌드 환경에서 둘 다 못 찾은 경우
        if not cfg_path:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cfg_path = os.path.join(base_dir, 'external_tools_config.ini')
            
        if cfg_path and os.path.exists(cfg_path):
            parser = configparser.ConfigParser()
            parser.read(cfg_path, encoding='utf-8')
            if 'Executables' in parser and tool_name in parser['Executables']:
                return parser['Executables'][tool_name]
    except Exception:
        pass
    return None

