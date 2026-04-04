# 대시보드 메뉴바 확장 및 동적 설정 기능 구현 완료 (v5.5.2)

안녕하세요, **WHTOOLS**입니다. 대시보드의 사용 편의성과 커스터마이징 기능을 대폭 강화한 **버전 5.5.2** 업데이트가 완료되었습니다. 이제 상단 메뉴바를 통해 분석 환경을 자유롭게 조정하실 수 있습니다.

## 1. 주요 업데이트 내용

### 1.1. 새로운 메뉴바 시스템
- **Setting 메뉴:**
    - **3D View Font:** PyVista 화면 내의 통계 정보 및 범례 폰트를 실시간으로 변경할 수 있습니다.
    - **2D Plot Font:** Matplotlib 그래프의 텍스트 크기와 폰트를 일괄 조정합니다.
    - **2D Plot Theme:** Matplotlib이 지원하는 수십 가지의 테마(Solarize, ggplot, dark_background 등)를 즉시 적용할 수 있습니다.
- **Help 메뉴:**
    - **About:** 제품 정보, 버전, 기술 스택 및 공식 **logo.png** 배너가 포함된 전문가 수준의 안내 창을 제공합니다.

### 1.2. 동수적 시각화 엔진 고도화
- 폰트 및 테마 변경 시 화면 전체를 다시 그리지 않고 필요한 컴포넌트만 정밀하게 `update` 및 `draw` 하도록 최적화하였습니다.
- 테마 변경 시 발생할 수 있는 레이아웃 틀어짐을 방지하기 위해 `tight_layout` 자동 보정 로직을 포함했습니다.

## 2. 변경된 코드 및 리소스
- [`plate_by_markers_v2.py`](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/plate_by_markers_v2.py): 메뉴바 초기화 및 폰트/테마 변경 슬롯 로직 추가.
- **리소스 경로:** `resources/logo.png` (About 창 배너용).

## 3. 사용 가이드
1. 대시보드 상단의 `Setting` 메뉴를 클릭합니다.
2. `3D View Font` 또는 `2D Plot Font`를 선택하여 원하는 글꼴과 크기를 적용해 보세요.
3. `2D Plot Theme` 하위 메뉴에서 다양한 시각적 스타일을 실험해 보실 수 있습니다.

> [!TIP]
> 발표용 자료를 만드실 때는 `Setting > 2D Plot Theme > bmh` 또는 `ggplot` 테마를 사용하시면 더욱 깔끔한 그래프를 얻으실 수 있습니다.

---
**WHTOOLS**는 엔지니어의 작업 효율을 위한 최적의 도구를 지향합니다. 추가적인 기능 요청이나 개선 사항이 있으시면 언제든 말씀해 주세요.
