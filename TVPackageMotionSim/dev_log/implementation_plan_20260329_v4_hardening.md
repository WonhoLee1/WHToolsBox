# Implementation Plan - Plasticity & Hardening Strategy (v4)

소성 변형 알고리즘에 **가공 경화(Isotropic Hardening)** 모델을 도입하여, 재료가 압축될수록 변형 저항력이 강해지는 실제 물리 현상을 정교하게 모사합니다.

## User Review Required

> [!IMPORTANT]
> - **가공 경화(Hardening)**: 등가 변형률($\epsilon_{eq}$)이 증가함에 따라 항복 강도(Yield Stress)를 동적으로 상향시킵니다.
> - **물리적 임계점**: 한번 변형된 블록은 다음 충격 시 더 큰 에너지가 가해져야만 추가 변형이 일어나며, 그렇지 않을 경우 순수 탄성 거동을 수행합니다.

## Proposed Changes

### 1. [Simulator Engine] `whts_engine.py`

#### [MODIFY] [_apply_plasticity_v2](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_engine.py)

항복 판정 로직에 하드닝 항을 추가합니다.

- **항복 강도 진화(Yield Evolution)**: 
  - **공식**: $\sigma_{yield, current} = \sigma_{yield, 0} + H \cdot \epsilon_{eq}$
  - $\sigma_{yield, 0}$: 초기 항복 압력 (`cush_yield_pressure`)
  - $H$: 하드닝 계수 (`plastic_hardening_modulus`)
  - $\epsilon_{eq}$: 현재 누적된 등가 변형률
- **거동 제어**: 
  - `Pressure > current_yield` 조건에서만 `geom_size` 감소가 발생합니다.
- **시각화 유지**: 
  - 등가 변형률 기반의 파란색 전이는 그대로 유지하여 누적된 손상도를 시각화합니다.

### 2. [Case Script] `run_drop_simulation_cases_v4.py`

#### [UPDATE]
- `cfg["plastic_hardening_modulus"] = 2000.0` 설정 추가.
- 초기 항복과 하드닝이 조화롭게 작동하도록 파라미터 밸런싱.

## Verification Plan

### Automated Tests
- `python run_drop_simulation_cases_v4.py` 실행.

### Manual Verification
- **1차 충격**: 블록이 찌그러지며 파란색으로 변하는지 확인.
- **2차 충격**: 동일 부위에 약한 충격이 가해졌을 때, 추가적인 크기 감소 없이 탄성 반발만 일어나는지 Viewer에서 확인.
- **로그 확인**: `current_yield`가 상승함에 따라 `reduction`이 발생하는 빈도가 줄어드는지 점검.
