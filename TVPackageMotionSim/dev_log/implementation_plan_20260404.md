# [V5.2.8.2] 듀얼 모드 배치 구조 해석 엔진 및 벤치마킹 계획

사용자님의 제안에 따라, NumPy 기반 표준 연산과 JAX 기반 가속 연산을 선택할 수 있는 듀얼 엔진 구조를 도입합니다.

## User Review Required

> [!IMPORTANT]
> - **벤치마킹 기능**: 각 방식의 연산 소요 시간(Processing Time)을 리포트에 함께 출력하여 JAX의 도입 효과를 수치로 확인하실 수 있습니다.
> - **설정 옵션**: `use_jax_reporting` 설정을 통해 주 분석 엔진을 전환할 수 있습니다.

## Proposed Changes

### [Core Engine]

#### [MODIFY] [whts_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_engine.py)
- `_main_loop()`: `compute_structural_step_metrics(self)` 호출 제거 (성능 복구 핵심).
- `_wrap_up()`: 시뮬레이션 종료 후 `compute_batch_structural_metrics(self)`를 호출하여 누적된 `quat_hist`를 기반으로 결과 산출.

### [Reporting Engine]

#### [MODIFY] [whts_reporting.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_reporting.py)
- `compute_batch_structural_metrics(sim)` 함수 구현:
    - **Standard Mode (NumPy)**: 하위 호환성 및 검증을 위한 일반 루프 연산.
    - **Accelerated Mode (JAX)**: `vmap`, `jit`을 활용한 병렬 연산 (JAX 버전).
    - **Performance Log**: 분석 완료 후 각 엔진의 소요 시간(Analysis Time)을 비교 출력.

## Open Questions

> [!QUESTION]
> - 두 방식의 결과값이 부동 소수점 오차 범위 내에서 일치하는지 자동으로 검증하는 로직(Validation)을 추가할까요?

## Verification Plan

### Automated Tests
- 시뮬레이션 종료 시 터미널에 `[ ANALYSIS BENCHMARK ]` 섹션이 나타나고 각 방식의 시간이 출력되는지 확인.
- 최종 리포트 데이터가 0이 아닌 유효값인지 확인.
