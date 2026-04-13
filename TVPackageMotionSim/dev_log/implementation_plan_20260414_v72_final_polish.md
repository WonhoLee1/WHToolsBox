# Implementation Plan - [v7.2] Numerical Smoothing & Explosion Guard

수치적 노이즈를 억제하여 응력을 현실화하고, 시뮬레이션 자체가 붕괴된 파트(Exploded)로부터 발생하는 비물리적 가짜 데이터를 원천 차단합니다.

## User Review Required

> [!IMPORTANT]
> - **수치적 평활화(Refined lambda)**: 마커 노이즈에 의한 응력 점프를 막기 위해 `reg_lambda`를 **`0.01`**로 조정합니다. (488 MPa -> ~10 MPa 기대)
> - **시뮬레이션 폭주 감지**: 강체 정렬 오차(RMSE)가 **10mm**를 넘거나 변위가 비정상적인 파트는 **`[EXPLODED]`**로 처리하여 리포트 오염을 방지합니다.
> - **전 차원 스케일링 완결**: $1/L, 1/L^2, 1/L^3$ 스케일링을 모든 해석 모델(Mindlin, Von Karman 포함)에 완벽 적용합니다.

## Proposed Changes

### [Engine Refinement]

#### [MODIFY] [whts_multipostprocessor_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_multipostprocessor_engine.py)
- `PlateConfig`: `reg_lambda`를 **`0.01`**로 상향.
- `evaluate_batch`: 전단력(Vx, Vy) 및 비선형항(VON_KARMAN)에 대한 스케일링 수식 최종 보정.
- `ShellDeformationAnalyzer.analyze`: 폭주 데이터 감지 로직 추가 (RMSE > 10mm인 경우 Stress=0 처리 및 경고 출력).

## Verification Plan

### Automated Tests
1. `python run_drop_simulation_cases_v6.py` 실행 후 `Opencell_Front` 응력이 **30 MPa 미만**인지 확인.
2. 폭주하는 측면 부품(`Cushion_Left` 등)이 리포트에서 **`[EXPLODED]`** 혹은 `0.00`으로 안정적으로 표현되는지 확인.

### Manual Verification
1. ParaView 시각화에서 폭주한 파트가 찢어지는 현상이 사라지거나(0 처리 시), 시각적 경고가 명확한지 확인.
