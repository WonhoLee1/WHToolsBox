# Implementation Plan - [v7.2] Plate Theory Audit & Numerical Smoothing

평판이론의 고전 수식을 전수 조사하여 계수의 무결성을 확보하고, 수치적 노이즈를 억제하여 최종 해석 결과의 신뢰도를 공학적 정점에 올려놓습니다.

## User Review Required

> [!IMPORTANT]
> - **평판이론 수식 검증**: Kirchhoff, Mindlin, Von Karman 각 이론의 계수($6/t^2, D, 1-\nu^2$ 등)가 mm 단위계에서 물리적으로 완벽한지 재검토하고 교정합니다.
> - **수치적 평활화**: 규제화 계수를 **`0.01`**로 설정하여 마커 노이즈에 의한 수치 폭주를 원천 차단합니다.
> - **폭주 감지**: 정렬 오차(RMSE)가 **10mm**를 초과하는 파트는 물리적 붕괴 상태로 간주하여 리포트 오염을 방지합니다.

## Proposed Changes

### [Theoretical Integrity]

#### [MODIFY] [whts_multipostprocessor_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_multipostprocessor_engine.py)
- `evaluate_batch`: 각 이론별 응력/전단력 계산 수식의 계수 및 부호 전수 점검 및 교정.
- `PlateConfig`: `reg_lambda`를 **`0.01`**로 고정하여 안정성 극대화.

### [Reliability Guard]

#### [MODIFY] [whts_multipostprocessor_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_multipostprocessor_engine.py)
- `ShellDeformationAnalyzer.analyze`: `Avg R-RMSE > 10.0`인 경우 해당 파트의 응력과 변위를 `0.0`으로 소거하고 `[PHYSICS-CRASH]` 경고 리포트 주입.

## Verification Plan

### Automated Tests
1. `python run_drop_simulation_cases_v6.py` 실행 후 `Opencell_Front` 응력이 공학계 상식(1~50 MPa)으로 안착했는지 확인.
2. 붕괴된 측면 파트들이 리포트 상에서 깨끗하게 정제(0 혹은 경고)되는지 확인.

### Manual Verification
1. ParaView에서 각 부품의 응력 분포가 이론적 기대치(Camber의 정점 등)와 일치하는지 시각적으로 확인.
