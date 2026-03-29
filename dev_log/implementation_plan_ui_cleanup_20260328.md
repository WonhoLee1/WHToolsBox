# [Plan] UI 코드 중복 제거 및 런타임 에러 수정

`postprocess_ui.py` 파일 내에 동일한 기능을 수행하는 함수들이 중복 정의되어 있어, 실시간 업데이트 시 인자 개수 불일치(`TypeError`) 및 시뮬레이션 중단 현상이 발생하고 있습니다. 이를 정리하여 안정성을 확보합니다.

## User Review Required
- **중복 함수 제거**: 기존에 여러 번 정의된 `_on_show_contour_frame`, `_get_contour_grid_at` 등을 하나로 통합합니다.
- **인자 표준화**: 모든 컨투어 관련 함수가 `(step, comp, metric)` 순서의 인자를 일관되게 사용하도록 수정합니다.

## Proposed Changes

### [Component] Post-Processing UI (postprocess_ui.py)

#### [MODIFY] [postprocess_ui.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/postprocess_ui.py)
- **함수 통합 및 중복 제거**:
    - `_get_contour_grid_at`: 최신 버전(Line 1023)을 유지하고 하단의 `_get_contour_grid_for_comp` 제거.
    - `_on_show_contour_frame`: 가장 완성도가 높은 매트릭스 레이아웃 버전으로 단일화.
    - `_overlay_pba_vectors`: 부품별/시점별 벡터 오버레이 로직 통합.
- **버그 수정**:
    - `_on_save_contour_frames`에서 `_get_contour_grid_at` 호출 시 인자 누락 여부 확인 및 수정.
    - 실시간 연동 시 `self.sim.time_history`가 비어있을 경우의 예외 처리 강화.

## Verification Plan

### Automated Tests
- `run_drop_simulation_v3.py` 실행 후 `구조 해석 2D Field Contour` 탭에서 `실시간 연동` 체크.
- 시뮬레이션 진행 중 컨투어가 에러 없이 갱신되는지 확인.
- `매트릭스 컨투어 생성` 버튼 클릭 시 팝업 창 정상 출력 확인.

### Manual Verification
- `!! [PlotError]` 메시지가 더 이상 발생하지 않고 시뮬레이션이 끝까지 완주되는지 확인.
