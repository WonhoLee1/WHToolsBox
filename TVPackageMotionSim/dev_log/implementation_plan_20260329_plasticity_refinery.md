# Implementation Plan - Plasticity Algorithm and Visualization Refinement

소성 변형 알고리즘을 물리적으로 더 정확하게 개선하고, 시뮬레이션 과정에서 변형 정도에 따라 실시간으로 색상이 변하도록 시각화 기능을 강화합니다.

## User Review Required

> [!IMPORTANT]
> - **방향성 수축**: 접촉 법선(Normal) 벡터를 분석하여 3축 중 실제로 압축이 일어나는 특정 로컬 축의 크기만 감소시킵니다.
> - **실시간 색상 전이**: 시뮬레이션 루프 내에서 변형률(Strain)을 계산하고, 노란색(초기)에서 주황/빨간색(변형 심화)으로 색상을 실시간 업데이트합니다.

## Proposed Changes

### 1. [Simulator Engine] `whts_engine.py`

#### [MODIFY] [_apply_plasticity_v2](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_engine.py)
기존의 고정된 장축(Major Axis) 수축 방식에서 **동적 축 선택 방식**으로 전환합니다.

- **로컬 축 탐지**: `contact.frame`에서 얻은 접촉 법선을 해당 지오메트리의 `geom_xmat`(회전 행렬)을 이용해 로컬 좌표계로 변환합니다.
- **최적 수축 축 선택**: 변환된 로컬 법선 벡터 중 절대값이 가장 큰 성분을 가진 축(X, Y, Z 중 하나)을 수축 대상 축으로 정합니다.
- **실시간 색상 업데이트**:
    - `(현재 크기 / 초기 크기)` 비유를 통해 변형률을 산출합니다.
    - 변형이 깊어질수록 `[1, 1, 0]`(노란색)에서 `[1, 0, 0]`(빨간색)으로 서서히 변하도록 `geom_rgba`를 매 스텝 업데이트합니다.

### 2. [Case Script] `run_drop_simulation_cases_v4.py`

#### [VERIFY]
- 고해상도 격자(`chassis_div` 등 수정된 부분)에서 소성 변형이 의도한 방향으로 일어나는지 Viewer를 통해 확인합니다.

## Open Questions

- 소성 변형 시 부피 보존(Volume conservation)을 위해 다른 두 축을 약간 확장하는 로직도 추가할까요? (현재는 단순 수축만 고려)
- 색상 변화의 임계값(어느 정도 변형되었을 때 완전히 빨간색이 될지)을 별도의 설정값(`plastic_color_limit`)으로 분리할까요?

## Verification Plan

### Automated Tests
- `python run_drop_simulation_cases_v4.py` 실행.
- 로그를 통해 `NaN/Inf` 발생 여부 재확인.

### Manual Verification
- MuJoCo Viewer에서 충격 시 모서리 블록들이 압축 방향에 따라 얇아지는지 관찰.
- 블록 색상이 노란색 -> 주황색 -> 빨간색으로 실시간 전이되는지 확인.
