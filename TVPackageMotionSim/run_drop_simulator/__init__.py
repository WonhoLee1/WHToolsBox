"""
[WHTOOLS] run_drop_simulator Package - v4.0
이 패키지는 MuJoCo 기반의 낙하 시뮬레이션 엔진과 분석 툴을 모듈화하여 제공합니다.

Key Components:
    - DropSimulator: 시뮬레이션 메인 엔진 (engine)
    - DropSimResult: 결과 데이터 컨테이너 (data)
    - ConfigEditor: 실시간 설정 UI (gui)
"""

from .whts_engine import DropSimulator
from .whts_data import DropSimResult
from .whts_gui import ConfigEditor
from .whts_utils import compute_corner_kinematics, calculate_required_aux_masses

__version__ = "4.0.0"
__author__ = "WHTOOLS"

__all__ = [
    "DropSimulator",
    "DropSimResult",
    "ConfigEditor",
    "compute_corner_kinematics",
    "calculate_required_aux_masses"
]
