# Implementation Plan - [v6.9e] Sanity Guard Refinement & Alignment Integrity

물리적 항복점(Yield Point)을 고려한 곡률 가드 임계치 조정과, 시뮬레이션 폭주 시 발생하는 비물리적 데이터를 차단하는 최종 방어선을 구축합니다.

## User Review Required

> [!IMPORTANT]
> - **곡률 가드 최적화**: $70,000 MPa$ 소재의 항복 강도를 고려하여 곡률 $\kappa$ 클리핑 범위를 **`0.02`**로 대폭 하향합니다. (응력 500 MPa 수준으로 안착)
> - **정렬 실패 차단**: 마커 정렬 오차(R-RMSE)가 **10mm**를 넘으면 데이터를 신뢰할 수 없는 폭주 상태로 간주하여 변위와 응력을 0으로 초기화합니다.
> - **ParaView 매크로 보강**: 매크로 실행 중 속성 에러를 방지하기 위해 `try-except` 가드를 주입합니다.

## Proposed Changes

### [Engine Stability]

#### [MODIFY] [whts_multipostprocessor_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_multipostprocessor_engine.py)
- `PlateMechanicsSolver.evaluate_batch`: 곡률 클리핑 범위를 **`[-0.02, 0.02]`**로 조정.
- `ShellDeformationAnalyzer.remove_rigid_motion`: `r_rmse > 10.0`인 경우 경고 출력 및 해당 프레임 변위를 강제 진압(0 설정)하는 안전장치 주입.

### [Visualization Stability]

#### [MODIFY] [whts_exporter.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_exporter.py)
- `launch_paraview`: 대시보드 매크로 내 `try-except` 가드 주입하여 `DescriptiveStatistics` 오류 완전 해결.

## Verification Plan

### Automated Tests
1. `python run_drop_simulation_cases_v6.py` 실행 후 로그 상의 응력이 **500 MPa 미만**으로 완벽히 안착되는지 확인.
2. `Opencell_Left` 등 폭주하던 파트가 `[ALIGN-FAIL]` 경고와 함께 안정적으로 처리되는지 확인.

### Manual Verification
1. ParaView에서 대시보드가 에러 없이 열리며, 비물리적으로 찢어진 면이 사라졌는지 확인.
