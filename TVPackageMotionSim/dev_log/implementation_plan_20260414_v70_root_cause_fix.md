# Implementation Plan - [RCA-Fix] Fundamental Unit Integrity & Guard Removal

단위계 자동 판정의 맹점을 해결하고, 모든 데이터를 mm 단위로 강제 동기화하여 수치 폭주의 근본 원인을 제거합니다. 이 과정에서 설치된 임시 가드(Clipping)들을 철거하여 물리적 신뢰도를 회복합니다.

## User Review Required

> [!IMPORTANT]
> - **단위계 강제 통일**: `scale_result_to_mm` 함수에 `marker_pos_history`를 추가하여, 미터-밀리미터 혼선 가능성을 0%로 만듭니다.
> - **임시 가드 철거**: 응력 정상화가 보장되므로, 비물리적이었던 곡률 클리핑과 초강력 규세화(1.0)를 제거하고 공학적 기본값(`1e-4`)으로 환원합니다.
> - **정렬 안정성**: `sigma`에 최소 임계값을 두어 측면 부품의 회전 폭주를 수학적으로 방지합니다.

## Proposed Changes

### [Fundamental Unit Fix]

#### [MODIFY] [whts_multipostprocessor_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_multipostprocessor_engine.py)
- `scale_result_to_mm`: `marker_pos_history`, `marker_vel_history`를 스케일링 목록에 추가.
- `ShellDeformationAnalyzer.analyze`: 불안정한 `if < 2.0` 판정 로직 제거.
- `ShellDeformationAnalyzer.fit_reference_plane`: `sigma` 계산 시 `max(sigma, 50.0)` 적용하여 가중치 쏠림 방지.

### [Sanity Restoration]

#### [MODIFY] [whts_multipostprocessor_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_multipostprocessor_engine.py)
- `PlateConfig`: `reg_lambda`를 **`1e-4`**로 원복 (물리적 정밀도 회복).
- `PlateMechanicsSolver.evaluate_batch`: 곡률 클리핑 제거 (데이터가 정상화되었으므로 불필요).

### [ParaView Consistency]

#### [MODIFY] [whts_exporter.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_exporter.py)
- 대시보드 매크로의 `ModelVariables` 호환성 구문 유지.

## Verification Plan

### Automated Tests
1. `python run_drop_simulation_cases_v6.py` 실행 후 로그 상의 응력이 가드 없이도 **100 MPa 내외**의 깨끗한 수치를 보이는지 확인.
2. `Opencell_Left/Right`의 R-RMSE가 **1.0mm 이하**로 극도로 안정화되는지 확인.

### Manual Verification
1. ParaView에서 부드럽고 상식적인 변형 형상(Camber 등)이 관찰되는지 확인.
