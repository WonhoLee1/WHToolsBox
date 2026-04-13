# Implementation Plan - [v6.9b] Incremental Alignment & Physics Integrity

사용자의 직관을 반영하여, 이전 프레임의 회전 상태를 참조하는 **증분 정렬(Incremental Alignment)** 방식을 도입합니다. 이를 통해 측면 부품의 회전 폭주를 막고 분석의 연속성을 확보합니다.

## User Review Required

> [!IMPORTANT]
> - **증분 정렬 (Incremental Rotation)**: SVD 계산 시 이전 프레임의 회전 정보(`prev_R`)를 참조값으로 사용하여, 마커가 일직선인 경우에도 회전 안정성을 100% 확보합니다. (사용자 제안 반영)
> - **영률(E) 강제 정규화**: 수만 배 부풀려진 응력을 정상화하기 위해 영률 단위를 MPa로 내부 강제 고정합니다.
> - **물리적 검류**: JAX 계산 전후에 물리적 한계치(Displacement < 50mm, Stress < 1000MPa)를 상식 수준에서 대폭 강화합니다.

## Proposed Changes

### [Alignment Guard Strategy]

#### [MODIFY] [whts_multipostprocessor_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_multipostprocessor_engine.py)
- `remove_rigid_motion`: 
    - 이전 프레임의 회전 행렬 `prev_R`을 정렬의 "Seed"로 활용.
    - SVD 결과가 불안정(Rank deficient)할 경우, `prev_R`을 기준으로 회전 행렬을 Orthogonalize 하여 연속성을 강제함.

### [Core Physics Fix]

#### [MODIFY] [whts_multipostprocessor_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_multipostprocessor_engine.py)
- `ShellDeformationAnalyzer.__init__`: 영률 $E$ 단위를 MPa로 일관되게 고정.
- `PlateConfig`: 기본 규제화 계수(`reg_lambda`)를 상향하여 데이터 부족 구간의 피팅 발산 방지.

## Verification Plan

### Automated Tests
1. `sim_v6_integrity_v69b_test.txt` 로그에서 `Opencell_Left/Right`의 변위가 **30mm 이하**로 매끄럽게 연속되는지 확인.
2. 응력 값이 클리핑 없이도 **200 MPa 미만**으로 정상 산출되는지 확인.

### Manual Verification
1. 독립 뷰어(`view_results_v6.py`)에서 측면 부품이 튀거나 회전하지 않고 제품 본체와 함께 자연스럽게 거동하는지 확인.
