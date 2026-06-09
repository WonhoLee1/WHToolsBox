# Implementation Plan - Air Drag & Squeeze Force Data Export & Threading Fix

## Goal Description
1. 시뮬레이션 완료 결과(data/engineering.csv 및 PNG 그래프)에 `air drag`와 `squeeze force` 데이터가 누락되는 문제를 해결합니다.
2. 다중 스레드 솔버 또는 백그라운드 QThread 실행 시 공기역학 제어 콜백이 누락되어 물리량이 `0.0`으로 출력되는 문제를 수정합니다.

## Background Context
* **문제 1 (스레드 충돌)**: 비 GUI 스레드(`SimThread`)에서 Matplotlib 플로팅을 처리하다가 스레드 세이프티 위반으로 예외가 발생하여 결과 익스포트(`data` 폴더 생성)가 통째로 취소되었습니다.
* **문제 2 (콜백 누락)**: MuJoCo 제어 콜백이 `_mujoco_thread_registry`를 통해 스레드 ID (`threading.get_ident()`) 매칭 방식으로 등록 및 실행되도록 구현되어 있었습니다. 그러나 다중 스레드 솔버(`sim_nthread > 1`) 및 QThread 컨텍스트 하에서는 MuJoCo의 콜백 실행 스레드 ID가 등록 시점과 달라져 매칭에 실패했고, 공기역학 연산 루프가 무시되어 결과 데이터가 항상 `0.0`으로 인출되었습니다.

## Proposed Changes
1. **Matplotlib 백엔드 고정 (`run_drop_simulator/wht_export_sim_result.py`)**:
   * pyplot 임포트 전 `matplotlib.use('Agg')`를 선언하여 비 GUI 스레드 환경에서도 GUI 충돌 없이 그래프 파일과 CSV를 저장하도록 변경합니다.
2. **글로벌 콜백 단일 인스턴스화 (`run_drop_simulator/whts_engine.py`)**:
   * `_mujoco_thread_registry`를 제거하고 글로벌 변수 `_global_mujoco_control_callback_instance`를 통해 단일 인스턴스로 콜백을 관리하도록 변경합니다.
   * `setup()` 함수 시작 시 글로벌 변수를 한 번만 선언하고 내부의 중복 `global` 선언을 통합하여 SyntaxError를 예방합니다.

## Verification Plan
* `run_drop_simulation_cases_v6.py` 실행을 통해 시뮬레이션 완료 후 결과 디렉토리(예: `results/rds-YYYYMMDD_HHMMSS/data/`) 내에 `engineering.csv`, `engineering-air_drag.png`, `engineering-air_squeeze.png`가 모두 생성되는지 검증합니다.
* `engineering.csv` 내에 `air_drag` 및 `air_squeeze` 데이터가 0이 아닌 물리량으로 완벽히 채워지는지 수치를 검사합니다.
