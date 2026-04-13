# Implementation Plan - [v6.2] Report Integrity & Metric Sync

시나리오는 완주하지만 리포트 수치가 0으로 나오거나 로그가 실종되는 현상을 해결합니다.

## User Review Required

> [!IMPORTANT]
> - **응력 데이터 정합성**: 현재 리포트의 `BS(MPa)`는 MuJoCo 블록 회전 기반의 추정치입니다. 이를 JAX 쉘 해석 기반의 **Von-Mises 응력**으로 교체하여 더 정밀한 결과를 제공할 예정입니다.
> - **로그 가시성**: 병렬 분석 중에도 `[PART-OK]` 로그가 즉시 출력되도록 `flush=True`를 강제 적용합니다.

## Proposed Changes

### [Analysis & Reporting Engine]

#### [MODIFY] [whts_multipostprocessor_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_multipostprocessor_engine.py)
- `ShellDeformationAnalyzer.analyze` 내의 `print`문에 `flush=True` 추가.
- `PlateAssemblyManager`에 `show_report()` 메서드 추가: 모든 파트의 JAX 분석 결과를 종합하여 `rich.table` 등으로 출력.
- 결과 딕셔너리에 `'Bending Stress [MPa]'` 키를 추가하여 `'Von-Mises [MPa]'` 데이터와 동기화 (레거시 리포터 호환성).

#### [MODIFY] [run_drop_simulation_cases_v6.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulation_cases_v6.py)
- `run_analysis_and_dashboard_minimal` 함수 마지막에 `manager.show_report()` 호출 추가.
- `QtVisualizerV2` 호출부의 `try-except` 구문을 강화하여 GUI 실패 시에도 프로세스가 `0` 코드로 종료되도록 보장.

## Verification Plan

### Automated Tests
- `python run_drop_simulation_cases_v6.py` 실행 후:
    1. 로그에서 `[PART-OK] ... (Markers: 16)` 메시지 확인.
    2. 최종 리포트 테이블의 `BS(MPa)` 컬럼이 `0.00`이 아닌 유효한 수치인지 확인.
    3. 대시보드 종료 후 터미널에 `Exit code: 0`이 뜨는지 확인.
