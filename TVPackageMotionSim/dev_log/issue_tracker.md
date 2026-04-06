# [WHTOOLS] Issue Tracker

본 파일은 시뮬레이션 개선 및 수정에 대한 요구사항을 관리하고, 반복되는 이슈가 발생하지 않도록 추적하는 관리 문서입니다.

## 🟢 Open Issues

| ID | Issue Description | Status | Date | Note |
|:---|:---|:---|:---|:---|
| #001 | `get_default_config()`를 `test_run_case_1` 기반으로 최적화 | Pending | 2026-04-05 | 기본값 상향 및 내부 구조 가독성 강화 |
| #002 | 파라미터 네이밍 표준화 (`oc_` -> `opencell_`, `occ_` -> `opencellcoh_`) | Pending | 2026-04-05 | 프로젝트 전반의 변수명 일체감 확보 |
| #003 | 솔버 내부에 산재한 `.get()` 기본값을 `get_default_config`로 통합 | Pending | 2026-04-05 | 설정 관리의 'Single Source of Truth' 강화 |
| #004 | Headless 시뮬레이션 종료 시 `mainloop` 프리징 현상 해결 | Completed | 2026-04-06 | Lazy UI Init 및 Guard 로직 도입 (V5.4.2) |
| #005 | use_postprocess_v2의 PySide6 기반 고도화 요구사항 반영 | Completed | 2026-04-06 | 서브프로세스 독립 실행 및 V2 UI 연동 최적화 |
| #006 | `use_postprocess_ui` 레거시 기능을 V2 Dashboard로 완전 이식 | In Progress | 2026-04-06 | 기구학/구조해석 탭 및 데이터 연동 로직 추가 |

## 🟣 Completed Issues

*최근 해결된 이슈가 여기에 표시됩니다.*

## 🔴 Fixed Bugs & Gotchas

### 반복적인 실수를 방지하기 위한 기술적 메모

1. **Config Key 동기화**: `mat_*` 딕셔너리 내부의 `solref` 등은 외부 파라미터 수정 후 반드시 재조립되어야 함. (현재 `get_default_config` 끝단에서 처리 중)

2. **Path Encoding**: Windows 환경에서 한글 경로 포함 시 인코딩 문제 주의 (UTF-8 명시)

3. **UI Guarding**: Headless 모드 시뮬레이션 시 `tk.Tk()`를 명시적으로 `if enable_UI` 조건으로 감싸거나, `ctrl_open_ui`가 False일 때 `_wrap_up`에서 `return`하도록 하여 터미널 중단을 방지한다.

4. **V2 Dashboard**: V2 UI는 PySide6 기반이므로 Tkinter와 동일 프로세스에서 실행 시 충돌 가능성이 크다. 반드시 `subprocess`를 통해 별도 프로세스로 분리 실행한다.
