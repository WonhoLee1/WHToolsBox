# -*- coding: utf-8 -*-
"""
[WHTOOLS] Drop Simulation v4.0 - Main Execution Script
이 스크립트는 'run_drop_simulator' 패키지를 사용하여 낙하 시뮬레이션을 수행합니다.

주요 변경점:
    1. 코드 모듈화: 엔진, GUI, 데이터 모델을 분산하여 유지보수성을 극대화함.
    2. 다중 부품 분석: RRG, PBA 지표를 통해 부품별 구조적 안정성을 상세히 평가함.
    3. 고해상도 리포팅: 시뮬레이션 종료 시 자동으로 postprocess_ui와 연동됨.
"""

import sys
import os
from typing import Dict, Any, Optional

# [WHTOOLS] 모듈 경로 추가 (필요 시)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run_drop_simulator import DropSimulator

def run_standard_simulation(custom_config: Optional[Dict[str, Any]] = None) -> None:
    """
    표준 낙하 시뮬레이션을 실행합니다.
    
    Args:
        custom_config (Dict, optional): 특정 시나리오를 위한 커스텀 설정.
    """
    # 1. 시뮬레이터 인스턴스 생성
    sim = DropSimulator(config=custom_config)
    
    # 2. 시뮬레이션 구동
    # setup() -> simulate() 순으로 호출됩니다.
    sim.log(">> [Main] V4 Simulation Engine 시작...")
    sim.simulate()
    
    sim.log(">> [Main] 모든 프로세스가 완료되었습니다.")

if __name__ == "__main__":
    # 사용자 정의 설정 (예: 낙하 모드 변경 등)
    # config = {"drop_height": 0.8, "drop_mode": "PARCEL"}
    run_standard_simulation()
