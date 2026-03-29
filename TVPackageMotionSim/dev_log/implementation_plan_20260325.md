# Implementation Plan - Cushion Corner & Plasticity Algorithm Refinement

본 문서는 쿠션의 코너/엣지 부분에 특화된 물리 파라미터(solref, solimp) 할당과, 소성 변형(Plasticity) 알고리즘의 실시간성 강화를 위한 수정 계획을 담고 있습니다.

## 1. 개요 (Overview)
1. **쿠션 코너 식별 로직 변경**: 현재 12개의 모서리 전체를 대상으로 하는 로직을 사용자가 정의한 "4개의 수직 엣지(8개 코너점 및 그 사이의 Z방향 geom)"로 한정합니다.
2. **소성 변형 알고리즘 고도화**: 최대 침투 이후 회복될 때까지 기다리지 않고, 침투량이 감소하기 시작하는 즉시 영구 변형(색상, 크기, 위치)을 적용하도록 변경합니다.
3. **설정 연동**: `cush_yield_stress` 및 `enable_plasticity` 설정을 시뮬레이션 루틴에 정확히 반영합니다.

## 2. 제안된 변경 사항 (Proposed Changes)

---
### 2.1. [Component: run_discrete_builder]
쿠션 모델 생성 시 코너/엣지 블록을 식별하는 기준을 수정합니다.

#### [MODIFY] [run_discrete_builder/__init__.py](file:///c:/Users/GOODMAN/WHToolsBox/test_box_mujoco/run_discrete_builder/__init__.py)
- `BCushion.is_edge_block(i, j, k)` 메서드를 수정하여 `(bx and by)` 조건만 체크하도록 변경합니다. 이는 X-Y 평면의 모서리(Z축 방향 엣지)만을 선택하게 됩니다.

---
### 2.2. [Component: run_drop_simulation]
시뮬레이션 루프 내의 소성 변형 로직을 실시간 방식으로 변경합니다.

#### [MODIFY] [run_drop_simulation.py](file:///c:/Users/GOODMAN/WHToolsBox/test_box_mujoco/run_drop_simulation.py)
- `apply_plastic_deformation` 함수 내부의 변형 적용 타이밍을 수정합니다.
    - 현재: `recovery >= state['max_p'] * plasticity_ratio` 일 때 적용.
    - 변경: `curr_p < state['max_p']` (침투가 줄어들기 시작) 시점에 즉시 영구 변형을 가산 및 적용.
- 소성 변형 대상 geom 판별 시 `run_discrete_builder`의 `is_edge_block`과 동일한 "Corner" 기준을 적용합니다.
- `cush_yield_stress` 설정을 활용하여 항복 임계값을 관리합니다.

## 3. 검증 계획 (Verification Plan)

### 자동화 테스트 (Automated Tests)
- **코너 파라미터 확인**: `create_model`을 통해 생성된 XML을 검사하여, 수직 엣지에 해당하는 geom들의 클래스가 `contact_bcushion_edge`로 올바르게 지정되었는지 확인합니다.
- **소성 변형 동작 확인**: `run_drop_simulation.py`를 실행하여, 바닥 충돌 후 쿠션 코너부가 파란색(또는 어두운 색)으로 변하며 영구적인 크기 축소가 발생하는지 GUI(Viewer)를 통해 육안으로 확인합니다.

### 수동 검증 (Manual Verification)
- 사용자가 직접 시뮬레이션을 실행하여, 코너 낙하 시 해당 부위의 변형이 즉각적으로 시각화되는지 확인 부탁드립니다.

---
**작성일**: 2026-03-25
**작성자**: Antigravity (Assistant)
