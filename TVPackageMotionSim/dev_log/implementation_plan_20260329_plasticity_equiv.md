# Implementation Plan - Plasticity Algorithm and Visualization Refinement (v3)

소성 변형 알고리즘을 물리적으로 더 정확하게 개선하고, **등가 변형률(Equivalent Strain)** 개념을 도입하여 시뮬레이션 과정에서 변형 정도에 따라 실시간으로 색상이 변하도록 시각화 기능을 강화합니다.

## User Review Required

> [!IMPORTANT]
> - **방향성 수축**: 접촉 법선(Normal) 벡터를 분석하여 3축 중 실제로 압축이 일어나는 특정 로컬 축의 크기만 감소시킵니다.
> - **등가 변형률 기반 색상 전이**: 한 개의 축만 고려하던 방식에서 벗어나, 3축의 모든 변류율을 종합한 **등가 변형률**을 기준으로 노란색에서 파란색으로 실시간 업데이트합니다.

## Proposed Changes

### 1. [Simulator Engine] `whts_engine.py`

#### [MODIFY] [_apply_plasticity_v2](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_engine.py)

기존의 단순 수축 방식에서 **등가 변형률 기반 시각화** 방식으로 고도화합니다.

- **로컬 축 탐지 및 수축**: 
  - `local_normal = model.geom_xmat[g_id].T @ world_normal`를 통해 압축 축을 찾아 `geom_size`를 줄입니다.
- **등가 변형률(Equivalent Strain) 수식**: 
  - **공식**: $\epsilon_{eq} = \sqrt{\epsilon_x^2 + \epsilon_y^2 + \epsilon_{z}^2}$ (SRSS 방식)
  - **의미**: 각 축의 누적된 영구 변형을 벡터 합으로 계산하여 전체적인 손상도를 산출합니다.
  - **장점**: 충돌 방향이 바뀌거나 접촉이 없어져도 이미 발생한 변형 상태가 색상(파란색)으로 유지됩니다.
- **실시간 색상 전이**:
  - `strain_norm = np.clip(equiv_strain / color_limit, 0.0, 1.0)`
  - 노란색($[1, 1, 0]$) $\rightarrow$ 파란색($[0, 0, 1]$) 보간.

### 2. [Case Script] `run_drop_simulation_cases_v4.py`

#### [VERIFY]
- `cush_yield_pressure`, `plastic_color_limit` 등의 파라미터를 통해 민감도를 실시간 조정하며 확인합니다.

## Verification Plan

### Automated Tests
- `python run_drop_simulation_cases_v4.py` 실행.

### Manual Verification
- 충격 발생 시 노란색 블록이 **실시간으로 파란색**으로 변하며, 접촉이 종료된 후에도 해당 색상이 유지되는지 확인.
- 블록이 3방향 중 실제 힘을 받는 방향으로만 정교하게 찌그러지는지 확인.
