"""
[WHTOOLS] run_drop_simulator Package - v6.0
이 패키지는 MuJoCo 기반의 낙하 시뮬레이션 엔진과 PySide6 기반의 분석 툴을 제공합니다.

Key Components:
    - DropSimulator: 시뮬레이션 메인 엔진 (engine)
    - DropSimResult: 결과 데이터 컨테이너 (data)
    - ControlPanel: PySide6 기반 실시간 제어 패널
"""

from .whts_engine import DropSimulator
from .whts_data import DropSimResult
from .whts_control_panel import ControlPanel
from .whts_utils import compute_corner_kinematics, calculate_required_aux_masses

__version__ = "6.0.0"
__author__ = "WHTOOLS"

__all__ = [
    "DropSimulator",
    "DropSimResult",
    "ControlPanel",
    "compute_corner_kinematics",
    "calculate_required_aux_masses"
]
