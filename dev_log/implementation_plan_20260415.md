# [WHTOOLS] UI 코드 구조 고도화 및 가독성 개선 계획

`whts_multipostprocessor_ui.py` 파일의 복잡한 로직을 체계적으로 구조화하여, 사람과 AI 에이전트 모두가 이해하기 쉬운 고품질 코드로 리팩토링합니다.

## User Review Required

> [!IMPORTANT]
> **원칙**: 기능의 변경이나 삭제는 절대 없으며, 오직 **가독성(Readability)**, **구조화(Structuring)**, **주석 최적화(Documentation)** 에만 집중합니다.
> 특히 최근에 추가된 `NoneType` 방어 로직과 `v7.5.5` 패치 사항들이 누락되지 않도록 엄격히 관리합니다.

## Proposed Changes

### [WHTOOLS UI]

#### [MODIFY] [whts_multipostprocessor_ui.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_multipostprocessor_ui.py)

1.  **클래스 및 메서드 그룹화**:
    *   `# === [Section Name] ===` 형태의 대형 구분자를 도입하여 시각적 인지 속도를 높임.
    *   그룹 구성:
        *   `Data Models & Configs`: `PlotSlotConfig`, `DashboardConfig`
        *   `Helper Windows`: `VisibilityToolWindow`, `AddPlotDialog`, `AboutDialog`
        *   `Main Application (QtVisualizerV2)`:
            *   `Core Lifecycle`: `__init__`, `update_frame`, `_ctrl_slot`
            *   `UI Components`: `_init_ui`, `_init_3d_controls`, `_init_2d_controls`, `_init_animation_dock`
            *   `Visualization Engines`: `_init_3d_view`, `_init_2d_plots`, `_update_2d_plots`
            *   `Event Handlers`: `keyPressEvent`, `_on_axis_clicked`, `_show_part_menu`
2.  **가독성 개선**:
    *   세미콜론(`;`)을 사용한 한 줄 다중 명령문을 지양하고 개별 라인으로 분리.
    *   변수명과 함수 인자에 대한 상세 설명(Docstrings) 표준화 (`Google Style` 또는 `reStructuredText` 지향).
3.  **WHTOOLS 페르소나 주석**:
    *   각 기능의 설계 의도와 수치 해석적 배경을 설명하는 고급 기술 주석 추가.

## Verification Plan

### Automated Tests
- 리팩토링 후 `run_drop_simulation_cases_v6.py` 및 `run_post_only_v5.py`를 실행하여 UI가 정상적으로 런칭되는지 확인.
- 애니메이션 제어, 필드 변경, 레이아웃 변경 등 핵심 기능의 정상 작동 여부 재검토.

### Manual Verification
- `Visibility Manager` 열기/닫기, 컬러바 정합성, 3D 뷰의 Perspective 모드 등 시각적 요소가 이전과 동일하게 유지되는지 확인.
