# Implementation Plan - [v6.9c] Unified ParaView & Numerical Integrity Patch

ParaView 대시보드 실행 에러를 해결하고, 해석 결과의 물리적 상식성을 확보하기 위해 피팅 엔진을 대폭 강화합니다.

## User Review Required

> [!IMPORTANT]
> - **ParaView Reader 수정**: `HDFReader` 오류를 해결하기 위해 범용 로더인 **`OpenDataFile`**을 사용하여 호환성을 확보합니다.
> - **2차 피팅 강제**: 마커가 부족한 부품(16개 이하)은 수학적 변곡점 폭주를 막기 위해 **2차(Quadratic)**로 차수를 낮춥니다.
> - **단위계 확행**: 영률(E)이 MPa 단위로 정확히 교정되는지 로그를 실시간으로 출력합니다.

## Proposed Changes

### [ParaView Dashboard Fix]

#### [MODIFY] [whts_exporter.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_exporter.py)
- `launch_paraview` 및 `register_paraview_macro`: `HDFReader` 호출부를 **`VTKHDFReader`** 또는 **`OpenDataFile`**로 수정.

### [Numerical Deep Stabilization]

#### [MODIFY] [whts_multipostprocessor_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_multipostprocessor_engine.py)
- `PlateConfig`: `reg_lambda`를 **`0.1`**로 대폭 상향하여 곡률 폭주 방지.
- `analyze`: 마커 16개 이하일 때 `max_safe_deg`를 **2**로 강제 하향.
- `__init__`: 영률 보정 로직을 보강하고 `print(..., flush=True)`를 추가하여 가시성 확보.

## Verification Plan

### Automated Tests
1. `python run_drop_simulation_cases_v6.py` 실행 후 로그 상의 응력이 **500 MPa 미만**(정상 범위)인지 확인.
2. ParaView가 에러 없이 자동 실행되어 대시보드가 출력되는지 확인.

### Manual Verification
1. `Open Data` 창에서 `Result.vtkhdf`가 정상적으로 로드되는지 확인.
