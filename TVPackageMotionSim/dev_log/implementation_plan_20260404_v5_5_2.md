# Dashboard 메뉴바 확장 및 동적 설정 기능 추가 계획 (V5.5.2)

이 계획은 대시보드 상단에 메뉴바를 추가하고, 사용자가 실시간으로 3D/2D 폰트 및 그래프 테마를 변경할 수 있는 기능을 구현하는 것을 목표로 합니다.

## User Review Required

> [!IMPORTANT]
> - **폰트 다이얼로그 연동:** 표준 `QFontDialog`를 호출하여 사용자가 선택한 폰트를 3D View(PyVista)와 2D Plot(Matplotlib)에 즉시 적용합니다.
> - **테마 변경의 영향:** Matplotlib 테마 변경 시 기존 그래프의 색상이나 스타일이 초기화될 수 있으므로, 변경 후 `draw()`를 명시적으로 호출하여 갱신합니다.
> - **About 다이얼로그:** `resources/logo.png` 파일을 배너로 사용하여 전문적인 정보를 표시합니다.

## Proposed Changes

### [Component] UI Framework (PySide6)

#### [MODIFY] [`plate_by_markers_v2.py`](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/plate_by_markers_v2.py)

- `QtVisualizerV2._init_ui()` 수정: `_init_menus()` 호출 추가.
- `_init_menus()` 신규 메서드 추가:
    - `Setting` 메뉴: `3D View Font`, `2D Plot Font`, `2D Plot Theme` (Submenu).
    - `Help` 메뉴: `About`.
- `_change_3d_font()` 슬롯 추가:
    - `QFontDialog` 호출.
    - PyVista의 Scalar Bar 및 Text Overlay 폰트 정보 업데이트.
- `_change_2d_font()` 슬롯 추가:
    - `QFontDialog` 호출.
    - `matplotlib.rcParams` 업데이트 및 Canvas Redraw.
- `_set_2d_theme(theme_name)` 슬롯 추가:
    - `plt.style.use()` 적용.
    - Canvas Redraw.
- `_show_about()` 메서드 추가:
    - 배너 이미지(logo.png)가 포함된 커스텀 다이얼로그 전시.

## Verification Plan

### Automated Tests
- 대시보드 실행 후 각 메뉴의 다이얼로그가 정상적으로 뜨는지 확인.
- 폰트 변경 시 3D View의 텍스트 크기와 2D Plot의 레이블이 실시간으로 바뀌는지 확인.
- Matplotlib 테마 변경 시 그래프 스타일이 즉각적으로 변하는지 확인.

### Manual Verification
- About 창의 로고 배너가 정상적인 비율로 출력되는지 확인.
- 메뉴 선택 후 프로그램의 안정성 체크.
