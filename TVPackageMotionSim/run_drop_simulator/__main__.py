# -*- coding: utf-8 -*-
"""
[WHTOOLS] Drop Simulator Module Runner
패키지 모드(python -m run_drop_simulator)로 시뮬레이션을 실행합니다.
"""

import sys
import os
from .whts_engine import DropSimulator

def main():
    """표준 낙하 시뮬레이션을 실행합니다."""
    # 시뮬레이터 인스턴스 생성 (디폴트 설정 사용)
    sim = DropSimulator()
    
    # 시뮬레이션 구동
    sim.log(">> [Module Runner] V4 Simulation Engine 시작...")
    sim.simulate()
    
    sim.log(">> [Module Runner] 모든 프로세스가 완료되었습니다.")

if __name__ == "__main__":
    main()
