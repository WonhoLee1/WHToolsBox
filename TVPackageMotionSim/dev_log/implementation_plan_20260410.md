# [WHTOOLS] v6.py: 최소 정보 기반 구조 해석 검증 계획

본 계획은 `run_drop_simulation_cases_v5.py`를 개선하여, `ShellDeformationAnalyzer`에 치수(`W`, `H`)나 2D 오프셋(`o_data_hint`)과 같은 보조 정보 없이 **순수 마커 궤적 데이터**만으로도 정밀 구조 해석이 가능한지 검증하는 것을 목표로 합니다.

## User Review Required

> [!IMPORTANT]
> - `v6.py`에서는 `get_assembly_data_from_sim`에서 생성되는 `offsets` 데이터를 의도적으로 배제하고 테스트합니다.
> - `ShellDeformationAnalyzer`의 SVD 기반 자율 로컬 좌표계 생성 로직이 시뮬레이션의 물리적 방향과 일치하는지 대시보드에서 육안 확인이 필요합니다.
> - 기존 `v5.py` 대비 분석 속도나 정확도에 유의미한 차이가 있는지 비교합니다.

## Proposed Changes

### [Component Name] Simulation & Analysis Pipeline

---

#### [NEW] [run_drop_simulation_cases_v6.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulation_cases_v6.py)
- `run_analysis_and_dashboard_minimal` 함수 구현:
    - `get_assembly_data_from_sim` 호출 후 `assembly_markers`만 활용.
    - `ShellDeformationAnalyzer` 생성 시 `W=0, H=0` (기본값) 사용.
    - `o_data_hint`를 설정하지 않고 `m_data_hist`만 주입.
- `run_digital_twin_pipeline`이 위 최소화된 함수를 호출하도록 수정.
- 기존 시뮬레이션 설정(`test_case_1_setup`) 유지.

---

### [Backup & Logging]

#### [NEW] [implementation_plan_20260410.md](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/dev_log/implementation_plan_20260410.md) (Backup)
- 본 계획서의 사본 저장 및 버전 관리.

## Open Questions

- `o_data_hint` 없이 SVD만으로 좌표계를 잡을 경우, 판의 X-Y 축이 뒤바뀔 가능성이 있습니다. (이 경우 `Analyzer` 내부의 장단축비 기반 보정 로직이 잘 작동하는지 확인이 필요합니다.)
- `mode='kinematic'`으로 추출한 마커 데이터를 쓸 때, `offsets` 없이도 완벽한 정렬이 유지되는지 확인이 필요합니다.

## Verification Plan

### Automated Tests
- `python run_drop_simulation_cases_v6.py` 실행.
- 콘솔 로그에서 `[PART-OK]` 메시지가 정상 출력되는지 확인.
- `Avg F-RMSE` 및 `Avg R-RMSE` 값이 `v5.py` 결과와 유사한지 비교 (수치적 안정성 확인).

### Manual Verification
- 실행 후 나타나는 **Qt Dashboard**에서 각 파트의 변형 형상이 물리적으로 타당한지(예: 꺾임 방향 등) 확인.
- Perspective View에서 마커 정렬 상태 확인.
