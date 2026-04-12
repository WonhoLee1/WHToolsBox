# [Goal] ImportError Fix & Reporting Engine Restoration

`whts_reporting.py` 파일이 편집 과정에서 잘려나가(Truncated) `compute_ssr_shell_metrics` 등을 로드하지 못하는 문제를 해결하고, 시뮬레이션 파이프라인의 후처리 단계를 복구합니다.

## User Review Required

> [!NOTE]
> **파일 복원**: `whts_reporting.py`가 약 260라인 부근에서 잘려 있는 것을 확인했습니다. 백업 데이터를 바탕으로 누락된 SSR(Structural Surface Reconstruction) 엔진과 최종 리포트 출력 로직을 통합 복원합니다.

## Proposed Changes

### 1. Reporting Engine Restoration

#### [MODIFY] [whts_reporting.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_reporting.py)
- 누락된 함수들 복원:
  - `compute_critical_timestamps`: 임계 시점 검출
  - `finalize_simulation_results`: Rich 기반 터미널 리포트 출력
  - `apply_rank_heatmap`: MuJoCo 뷰어 내 변형 랭크 가시화
  - `compute_ssr_shell_metrics`: 레거시 UI용 SSR 연산 엔진

---

### 2. Simulation Engine Cleanup

#### [MODIFY] [whts_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_engine.py)
- `_wrap_up` 로직 보완:
  - 파일 저장 및 분석 단계가 안정적으로 종료되도록 보장합니다.

## Verification Plan

### Automated Tests
- `run_drop_simulation_cases_v6.py` 재실행:
  - 시뮬레이션 종료 후 "ImportError" 없이 터미널 리포트가 출력되는지 확인.
  - 레거시 Tkinter UI가 정상적으로 팝업되는지 확인.

### Manual Verification
- ParaView 대시보드와 레거시 UI가 충돌 없이 각각 독립적으로 작동하는지 확인.
