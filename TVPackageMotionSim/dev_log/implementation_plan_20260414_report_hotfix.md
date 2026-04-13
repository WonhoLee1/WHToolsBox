# Implementation Plan - [v6.6] Report Output Refinement

피팅 발산 방어 로직은 성공했으나, 리포트 출력 시 배열 데이터를 스칼라로 변환하지 못한 버그를 수정합니다.

## User Review Required

> [!NOTE]
> - **데이터 축약**: 리포트 테이블에는 시뮬레이션 전체 시간/공간 중의 최댓값이 표시됩니다.
> - **안전성 확보**: `float()` 변환을 강제하여 포맷팅 에러를 방지합니다.

## Proposed Changes

### [UI & Reporting]

#### [MODIFY] [whts_multipostprocessor_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_multipostprocessor_engine.py)
- `show_report` 메서드 내에서 `m_stress`, `m_disp` 등을 추출할 때 `np.nanmax()`를 사용하여 단일 수치로 확정.

## Verification Plan

### Automated Tests
- `python run_drop_simulation_cases_v6.py` 재실행 후:
    1. **에러 유무**: `TypeError` 없이 리포트가 끝까지 출력되는지 확인.
    2. **수치 확인**: `Max Disp`가 클리핑되어 10mm 이내의 정상 수치로 나오는지 재검증.
