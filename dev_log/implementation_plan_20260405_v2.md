# [Goal] QtVisualizerV2 UI 최적화 및 2D Plot 엔진 고도화 (v1.2)

`plate_by_markers_v2.py`의 UI 성능을 개선하고, 사용자 경험(UX)을 강화하기 위해 2D 플롯 엔진과 레이아웃 관리 시스템을 리팩토링합니다.

## User Review Required

> [!IMPORTANT]
> **Pop-out 동기화**: 이제 별도의 창(`Pop-out`)으로 띄운 그래프들도 메인 윈도우의 애니메이션 재생 및 슬라이더 이동과 실시간으로 동기화됩니다.
> **성능 최적화**: 레이아웃 변경 시 `FigureCanvas`를 통째로 다시 만들던 방식을 개선하여, 기존 캔버스를 유지하고 서브플롯만 동적으로 재구성합니다.
> **활성 슬롯 강조**: 선택된 차트의 테두리를 `WHTOOLS Blue (#1A73E8)` 색상으로 강조하여 가독성을 높입니다.

## Proposed Changes

### 1. `_init_2d_plots` 및 레이아웃 관리 리팩토링

- **[Optimization]**: `FigureCanvas`와 `NavigationToolbar`를 `__init__`에서 한 번만 생성하도록 변경.
- **[Refactor]**: `_init_2d_plots`에서는 `self.canv.figure.clear()` 후 새로운 레이아웃에 맞춰 `add_subplot`만 수행.
- **[Safety]**: 레이아웃 변경 시 기존 `self.ims`, `self.vline` 캐시를 초기화하여 데이터 무결성 확보.

### 2. 애니메이션 및 Pop-out 동기화 로직 강화

- **[Feature]**: `update_frame` 함수 내에서 `self.pop_win`이 열려 있는 경우, `_update_pop_out_plots(f)`를 호출하여 실시간 동기화.
- **[Sync]**: 메인 윈도우의 `Sync` 체크박스 상태에 따라 2D/3D 연동 여부를 엄격히 제어.

### 3. 'Add Plot' 다이얼로그 및 세부 UI 개선

- **[UI]**: `AddPlotDialog`에서 현재 설정된 파트와 데이터 키가 기본값으로 선택되도록 개선.
- **[Visual]**: Matplotlib 차트의 모든 텍스트(Title, Label, Tick)에 `self.f_font_size` 전역 설정 반영.

### 4. 3D 뷰 및 지면 제어 보완

- **[Feature]**: 3D 뷰의 마우스 메뉴(Context Menu)에 `Reset Camera` 및 `View Isometric` 숏컷 추가 확인 및 보강.

## Verification Plan

### Automated Tests

- 레이아웃 전환(1x1 -> 3x2 -> 2x1) 10회 반복 시 메모리 누수 및 컴포넌트 유실 여부 확인.
- `Pop-out` 창 활성화 상태에서 애니메이션 재생 시 FPS(Frames Per Second) 유지율 확인.

### Manual Verification

- 차트 클릭 시 테두리 색상 변경 확인.
- `Add Plot` 다이얼로그를 통해 데이터 변경 시 즉각적으로 차트가 업데이트되는지 확인.
- Pop-out 창의 슬라이더 동기화 확인.

## dev_log 작업 관리

- `dev_log/task_20260405_v2.md`에 진행 상황 기록.
- 완료 후 `dev_log/walkthrough_20260405_v2.md` 작성.
