# [Goal] 설정 기반(Config-driven) 3D-2D 분리형 전문가 대시보드 개편 (v1.1)

유동적인 분석 환경을 위해 UI의 모든 초기 상태를 코드로 제어할 수 있는 `DashboardConfig` 시스템을 완성하고, 3D와 2D 영역을 스플리터로 분리하는 대규모 구조 개편을 마무리합니다.

## User Review Required

> [!IMPORTANT]
> **축 클릭 선택 시스템**: 차트 영역을 마우스로 클릭하면 해당 슬롯이 '활성' 상태가 되어 `+ Add Plot` 시 자동으로 대상 슬롯이 지정됩니다. 활성 슬롯은 시각적으로 강조(Highlight)됩니다.
> **팝업 레이아웃 미러링**: Pop-out 기능을 실행할 때, 현재 대시보드에 설정된 레이아웃(1x1 ~ 3x2)과 모든 슬롯의 데이터(Contour/Curve/파트 정보)를 그대로 유지하여 새 창을 띄웁니다.
> **배너 위치**: 로고 배너는 **3D 제어 패널(Group Box) 내의 좌측**에 배치됩니다.

## Proposed Changes

### 1. `QtVisualizerV2` 클래스 리팩토링 및 UI 구조화

#### [MODIFY] `plate_by_markers_v2.py:QtVisualizerV2` (file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/plate_by_markers_v2.py)
- `_init_ui`: `Main Splitter` 기반으로 3D(좌)와 2D(우) 영역을 확실히 분리.
- `_init_3d_panel`: `QGroupBox("3D View Control")` 내부에 로고와 컨트롤들을 `QHBoxLayout`으로 수평 배치.
- `_init_2d_panel`: 메인 차트 캔버스와 컨트롤 바(레이아웃 선택, 동기화, 보간, 추가 버튼)를 수직으로 배치.

---

### 2. 동적 2D 플롯 엔진 및 인터랙션 고도화

#### [MODIFY] `_init_2d_plots`
- `self.current_layout` (1x1, 1x2, 2x2, 3x2)에 맞춰 `Figure.add_subplot`을 유동적으로 수행.
- **축 클릭 이벤트**: `self.canv.mpl_connect('button_press_event', self._on_axis_clicked)` 등록.
- **활성 슬롯 강조**: 선택된 축의 spine 색상 및 두께를 변경하여 강조 시각화.

#### [NEW] `_show_add_plot_dialog()`
- `QDialog`를 통해 슬롯 번호(기본값은 활성 슬롯), 대상 파트, 플롯 타입(Contour/Curve), 데이터 키를 선택.
- 'Apply' 즉시 `self.plot_slots`를 갱신하고 `update_frame` 호출.

---

### 3. 애니메이션 및 데이터 연동 최적화

#### [MODIFY] `update_frame` & `_update_2d_plots`
- `self.cmb_step` (Step 1~10) 값을 애니메이션 프레임 증분값으로 사용.
- `self.plot_slots` 리스트를 순회하며 설정된 모든 슬롯의 데이터를 동적으로 업데이트.

---

### 4. Pop-out 기능 강화

#### [MODIFY] `_pop_out_2d`
- 메인 창의 `self.current_layout`과 `self.plot_slots` 정보를 복제하여 새 창의 `Figure` 구성을 그대로 재현.

## Open Questions

- 활성 슬롯 강조 시, Matplotlib 기본 스타일 외에 특별히 선호하시는 색상이 있으신가요? (기본값은 WHTOOLS Blue 계열 혹은 강조용 Red를 고려 중입니다.)

## Verification Plan

### Automated Tests
- `DashboardConfig` 초기화 시 복합 레이아웃 정상 로딩 확인.
- `create_cube_markers` 예제 데이터를 이용한 6개 면 동시 렌더링 부하 테스트.

### Manual Verification
- 차트 클릭 시 테두리 강조 및 `Add Plot` 시 슬롯 자동 선택 여부 확인.
- 레이아웃 변경 시 기존 차트 설정 데이터가 유효한 범위 내에서 유지되는지 확인.
- Pop-out 창이 메인 대시보드와 동일한 레이아웃으로 뜨는지 확인.
