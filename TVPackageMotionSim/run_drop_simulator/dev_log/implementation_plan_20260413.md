# [FIX] Robust Structural Analysis, Friction Config & Transient Export

주요 에러(Friction TypeError, SVD non-convergence 및 KeyError)를 해결하고, 전체 시뮬레이션 및 분석 파이프라인의 견고함(Robustness)을 강화합니다.

## User Review Required

> [!IMPORTANT]
> - **Friction Config 수정**: `run_drop_simulation_cases_v5.py`에서와 같이 `friction`을 리스트(`[0.3, 0.3]`)로 설정할 때 빌더가 비정상 종료되는 현상을 수정합니다.
> - **SVD 수치 안정성**: 파트의 최소 치수가 0에 가까울 경우(`4x1` 마커 배치 등) `sigma`가 0이 되어 발생하던 SVD 수렴 오류를 수정합니다.
> - **이슈 트래커 도입**: 반복되는 문제점과 개선 사항을 체계적으로 관리하기 위해 `dev_log/issue_tracker.md`를 생성합니다.

## Proposed Changes

### 1. Configuration & Physics (Friction Fix)

#### [MODIFY] [whtb_config.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_config.py)

- `get_friction_standard`: `mu` 인자가 리스트로 들어올 경우(`[0.3, 0.3]` 등)를 대비하여 첫 번째 요소를 기준으로 5축 마찰 계수를 생성하도록 로직을 강화합니다. (TypeError 방지)

---

### 2. Post-Processor Engine (Numerical Stability)

#### [MODIFY] [whts_multipostprocessor_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_multipostprocessor_engine.py)

- `fit_reference_plane`: `sigma` 계산 시 최소값(Floor)을 적용하고, 가중치 합이 0일 경우의 예외 처리를 추가합니다.
- `remove_rigid_motion`: SVD 수행 전 매트릭스 유효성(NaN 체크)을 확인합니다.
- `analyze`: 해석 과정에서의 치명적 오류 발생 시 `self.results`를 비우고 명확하게 실패를 반환합니다.

---

### 3. Result Exporter (Safety Guard)

#### [MODIFY] [whts_exporter.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_exporter.py)

- `export_to_vtkhdf`: `analyzer.results`가 비어있거나 필수 키(`Displacement [mm]`)가 없는 파트는 로그를 출력하고 건너뜜으로써 전체 내보내기가 중단되는 것을 방지합니다.
- `export_to_glb`: 동일하게 결과 유효성 체크를 강화합니다.

---

### 4. Dashboard UI (Initialization Fix)

#### [MODIFY] [whts_multipostprocessor_ui.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_multipostprocessor_ui.py)

- `QtVisualizerV2`: 대시보드 초기화 시 첫 번째 파트가 아닌, **해석에 성공한 첫 번째 파트**를 기준으로 데이터 필드 목록을 구성합니다.

---

### 5. Issue Management

#### [NEW] [issue_tracker.md](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/dev_log/issue_tracker.md)

- 반복되는 설정 오류 및 수치 불안정 이슈를 체계적으로 관리하기 위한 이슈 트래커 파일을 생성합니다.

## Verification Plan

### Automated Tests
- `run_drop_simulation_cases_v6.py` 및 `v5.py`를 실행하여 비정상 종료 없이 분석과 내보내기가 완료되는지 확인합니다.
- 결과물인 `Result.vtkhdf`가 생성되는지 확인합니다.

### Manual Verification
- ParaView에서 VTKHDF 파일을 열어 시계열 데이터가 정상적으로 표시되는지 확인합니다.
- 대시보드 UI를 실행하여 데이터가 누락된 파트가 안전하게 처리되는지 확인합니다.
