# Implementation Plan - [v7.4] Robust Alignment & VTKHDF Completion

시뮬레이션 폭주 상황에서도 해석의 끈을 놓지 않는 '강건한 정합성' 로직을 도입하고, ParaView 6.0 호환성을 위한 VTKHDF 규격 및 API를 완결합니다.

## User Review Required

> [!IMPORTANT]
> - **강건한 정렬 (RANSAC/Optim)**: SVD가 실패하거나 오차가 큰 경우, 정상 마커들만 골라내어 기준 평면을 잡는 최적화 기법을 도입합니다. (폭주한 마커에 의한 평면 왜곡 방지)
> - **VTKHDF 오프셋 완결**: `ConnectivityIdOffsets`, `CellOffsets` 등 ParaView가 시계열 데이터에서 요구하는 모든 오프셋 어레이를 `whts_exporter.py`에 추가합니다.
> - **API 예외 처리**: ParaView 6.0의 `DescriptiveStatistics` 속성 누락 문제를 해결합니다.

## Proposed Changes

### [Robust Alignment Engine]

#### [MODIFY] [whts_multipostprocessor_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_multipostprocessor_engine.py)
- `remove_rigid_motion`: R-RMSE가 임계치를 넘을 경우, 가중치 기반 최소자승법(WLS) 또는 RANSAC을 통해 지배적인 평면 경향성을 추출하는 로직 추가.

### [VTKHDF & Dashboard Fix]

#### [MODIFY] [whts_exporter.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_exporter.py)
- `export_to_vtkhdf`: `Steps/ConnectivityIdOffsets`, `Steps/CellOffsets` 데이터셋 생성 및 데이터 주입.
- `launch_paraview`: `DescriptiveStatistics` 필터 속성 설정 시 버전별 호환성 코드 보강 (`hasattr` 점검 강화).

## Verification Plan

### Automated Tests
1. `python run_drop_simulation_cases_v6.py` 실행 후 ParaView에서 `Result.vtkhdf` 로드 시 오프셋 관련 에러가 사라졌는지 확인.
2. 붕괴된 파트에서도 기준 평면이 (튀지 않고) 최대한 합리적으로 설정되는지 시각적으로 확인.

### Manual Verification
1. ParaView 6.0 Dashboard가 크래시 없이 기동되는지 확인.
