# Walkthrough - Cushion Corner & Plasticity Refinement

쿠션의 코너 부위 물리 파라미터 할당 로직과 실시간 소성 변형 알고리즘 수정을 완료하였습니다.

## 1. 주요 변경 사항 및 구현 내용

### 1.1. 쿠션 코너 식별 로직 수정
- **변경 전**: 12개 모서리(Edges) 전체를 대상으로 `_edge` 클래스 할당.
- **변경 후**: 4개의 수직 엣지(Vertical Edges, 8개 코너점 포함)로 범위를 한정.
- **파일**: [run_discrete_builder/__init__.py](file:///c:/Users/GOODMAN/WHToolsBox/test_box_mujoco/run_discrete_builder/__init__.py)
- **효과**: 정면 또는 코너 낙하 시 실제 충격이 집중되는 수직 기둥 부위에만 `cush_corner_solref/imp`가 정확히 적용됩니다.

### 1.2. 실시간 소성 변형 알고리즘 고도화 및 버그 수정
- **버그 수정**: 
    - 대소문자 구분 문제(`g_bcushion_` vs `g_BCushion_`)로 인해 필터링이 누락되어 중앙 쿠션이 잘못 변형되던 현상을 해결하였습니다.
    - 항복 응력(`cush_yield_stress`) 기본값을 `0.1 -> 0.01 MPa`로 조정하여 테스트 시 변형이 더 잘 관찰되도록 하였습니다.
    - 변형 축을 **Z축(두께 방향)**으로 우선 고정하여 수직 낙하 시의 물리적 타당성을 높였습니다.
- **색상 변화 체계 (Color Mapping)**:
    - **노란색 (Yellow)**: 정상 상태 (Highlighted Corner)
    - **파란색 계열 (Light Blue -> Deep Blue)**: 소성 변형 진행 중 (**5mm** 변형 시 Deep Blue 도달)
    - **민감도 개선**: 기존 3cm에서 5mm로 임계값을 대폭 낮추어 미세한 변형도 즉시 파란색으로 보이도록 최적화하였습니다.
- **디버그 로깅 추가**:
    - 쿠션 코너가 항복 응력을 초과하여 활성화될 때 `[Plasticity] Corner Activated` 로그가 출력됩니다.
    - 실제 영구 변형이 일어날 때 `[Plasticity] ... Deforming` 로그와 변형량(mm)이 출력됩니다.

### 1.3. 설정 및 초기화 보강
- `run_test` 함수 내에서 `cush_yield_stress` 및 `enable_plasticity` 설정을 시뮬레이션 루틴에 동기화하였습니다.
- 시뮬레이션 리셋(`Backspace` 또는 `R`) 시 변형된 기하학 정보(Size, Pos, Color)가 원본으로 복구되도록 초기화 로직을 추가하였습니다.

## 2. 검증 방법 (How to Test)

1. `run_drop_simulation.py`를 실행합니다. (`enable_plasticity=True` 확인됨)
2. 시뮬레이션 뷰어에서 쿠션 코너가 바닥에 충돌하는 것을 관찰합니다.
3. 충돌 후 튕겨 나올 때, 해당 코너 블록이 즉시 **파란색**으로 변하며 물리적으로도 **압착**된 상태를 유지하는지 확인합니다.

---
**작성일**: 2026-03-25
**작성자**: Antigravity (Assistant)
