# WHTOOLS Issue Tracker

## [P1] Codebase Synchronization & Mapping Logic Refinement

- **Status**: 진행 중
- **Issue Description**:
  - 로컬 시뮬레이션 엔진(`whts_engine.py`)과 분석 툴(`plate_by_markers_v2.py`)이 최신 GitHub 상태와 동기화되어야 함.
  - 특히 `whts_mapping.py` 파일명이 날짜별로 파편화되어 있어(`whts_mapping_D260406.py`), 이를 표준화하여 엔진에서 정상 참조되도록 수정 필요.
- **Requester**: User
- **History**:
  - 2026-04-06: GitHub 동기화 요청 (`plate_by_markers_v2.py`, `whts_mapping.py`).
  - 확인 결과 GitHub의 마지막 업데이트는 4월 4일이며, 로컬의 4월 6일 버전이 더 최신 로직을 포함하고 있음.
- **Action Plan**:
  1. GitHub의 4월 4일 버전 백업 및 로컬 최신 로직과의 병합 검토.
  2. `whts_mapping.py` 파일명 표준화 및 `whts_postprocess_engine_v2.py` 호출 구조 안정화.
  3. GitHub 최신 파일을 `whts_mapping_aaa.py`로 다운로드하여 로컬 로직과의 차이점 정밀 분석.

## [P2] Headless Mode Process Hang Resolution

- **Status**: 완료 (지속 관찰 필요)
- **Issue Description**:
  - MuJoCo 시뮬레이션이 Headless 모드(UI 미사용)에서 `mainloop()`로 인해 프로세스가 종료되지 않는 이슈.
- **Resolution**: `if self.ctrl_open_ui:` 가드를 추가하여 명시적 요청 시에만 UI 루틴 진입하도록 수정 완료.

## [P3] Digital Twin Accuracy Metrics (RMSE)

- **Status**: 완료
- **Issue Description**:
  - JAX SSR 엔진 분석 시 프레임별 RMSE 지표를 터미널에 출력하여 분석 정밀도 피드백 제공 필요.
- **Resolution**: `AdvancedPlateOptimizer` 및 `run_analysis`에 RMSE 연산 및 출력 로직 추가 (GitHub 4/4 업데이트 반영됨).

## [P4] 90-Degree In-Plane Rotation of Side Faces

- **Status**: 완료
- **Issue Description**:
  - `ShellDeformationAnalyzer`에서 PCA 기반 기저 산출 시, 세로로 긴(tall-narrow) 측면 패널의 기저축이 90도 회전되어 렌더링되는 문제.
- **Resolution**:
  - `o_data_hint`를 이용한 Kabsch 캘리브레이션 로직을 복구하여 설계 좌표계와 3D 기저를 동기화함.
  - `fit_reference_plane` 메서드가 힌트 좌표를 참조하여 로컬 기저를 산출하도록 수정 완료.
