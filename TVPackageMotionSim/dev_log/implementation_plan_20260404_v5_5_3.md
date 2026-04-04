# PyVista 폰트 오류 해결 및 D2Coding 폰트 적용 계획 (V5.5.3)

PyVista에서 `'cascadia code'`를 폰트 패밀리로 인식하지 못해 발생한 `KeyError`를 해결하고, 사용자가 지정한 `D2Coding` 폰트를 시스템 전반(PyVista, Matplotlib)에 안전하게 적용하는 계획입니다.

## User Review Required

> [!IMPORTANT]
> - **PyVista 폰트 적용 방식 변경:** PyVista의 기본 `font_family`는 시스템 폰트 이름을 직접 인식하는 데 제한이 있습니다. 따라서 사용자가 제공한 `D2Coding...ttf` 파일의 **절대 경로**를 `font_file` 매개변수로 직접 전달하는 방식을 사용합니다.
> - **폰트 파일 위치:** `run_drop_simulator/resources/D2Coding-Ver1.3.2-20180524-ligature.ttf` 경로의 존재를 확인했으며, 이를 기본 폰트로 사용합니다.
> - **Matplotlib 연동:** Matplotlib에서도 동일한 TTF 파일을 `font_manager`를 통해 등록하여 스타일 일관성을 유지합니다.

## Proposed Changes

### [Component] UI/Visualization Engine

#### [MODIFY] [`plate_by_markers_v2.py`](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/plate_by_markers_v2.py)

- **폰트 경로 상수화:** `QtVisualizerV2` 초기화 시 폰트 파일의 절대 경로를 `self.font_path`로 저장합니다.
- **PyVista 3D View (`_init_3d_view`):**
    - `add_point_labels`: `font_family` 대신 `text_property`를 생성하고 `set_font_file()`을 호출하여 설정합니다.
    - `add_scalar_bar`: 반환된 actor의 `label_text_property`와 `title_text_property`에 `SetFontFile()`을 호출합니다.
    - `add_text` (통계 오버레이): `font_file=self.font_path` 인자를 사용합니다.
- **Matplotlib 2D Plot (`_init_2d_plots`):**
    - `matplotlib.font_manager.fontManager.addfont(self.font_path)`를 호출하여 폰트를 등록합니다.
    - `rcParams['font.family']`를 등록된 폰트 이름으로 설정합니다.
- **동적 설정 슬롯 (`_change_3d_font`, `_change_2d_font`):**
    - 폰트 다이얼로그에서 선택된 폰트가 시스템 표준 폰트인 경우와 파일인 경우를 구분하여 처리하도록 로직을 보강합니다. (임시적으로 D2Coding을 기본값으로 강제 고정)

## Verification Plan

### Automated Tests
- 대시보드 실행 시 `KeyError` 없이 정상적으로 UI가 뜨는지 확인.
- 3D 뷰의 마커 라벨과 범례 폰트가 D2Coding(가독성 높은 코딩 폰트)으로 가독성 있게 표시되는지 확인.
- 2D 그래프의 축 레이블과 타이틀 폰트 확인.

### Manual Verification
- `Setting > About` 창을 열어 폰트가 깨지지 않고 잘 나오는지 확인.
- 폰트 파일 경로가 유효하지 않을 경우를 대비한 가벼운 예외 처리(Fallback to 'arial') 확인.
