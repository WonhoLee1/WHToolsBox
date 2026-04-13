# Implementation Plan - [v6.7] Result Persistence & Standalone Viewer

시뮬레이션 종료 후에도 결과를 확인할 수 있도록 데이터를 영구 저장하고, ParaView 크래시 문제를 우회하는 전용 독립 뷰어를 구축합니다.

## User Review Required

> [!IMPORTANT]
> - **데이터 저장**: 시뮬레이션 결과가 `results/latest_results.pkl`에 저장됩니다. (용량 확보 필요)
> - **독립 실행**: 이제 `python view_results_v6.py` 명령으로 시뮬레이션 없이 대시보드만 열 수 있습니다.

## Proposed Changes

### [Persistence & Viewer]

#### [MODIFY] [run_drop_simulation_cases_v6.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulation_cases_v6.py)
- `manager`의 결과를 저장하는 `save_results()` 함수 호출 추가.

#### [NEW] [view_results_v6.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/view_results_v6.py)
- 저장된 데이터를 로드하여 `QtVisualizerV2`를 독립적으로 실행하는 스크립트.

#### [MODIFY] [whts_exporter.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_exporter.py)
- `VTKHDF` 데이터 압축(GZIP) 및 청크 최적화로 ParaView 로드 안정성 강화.

## Verification Plan

### Manual Verification
1. 시뮬레이션 실행 후 `latest_results.pkl` 생성 확인.
2. `python view_results_v6.py` 실행 시 대시보드 창이 뜨는지 확인.
3. ParaView에서 강화된 `vtkhdf` 로드 시 크래시 여부 재점검.
