# Implementation Plan - MuJoCo Window Alignment & Model Naming Setup

본 문서에는 시뮬레이션 리로드 시 MuJoCo 시뮬레이터 뷰어 창의 Cascading(점진적 우하단 이동) 현상을 해결하기 위한 창 위치 고정 아키텍처와 생성되는 XML 파일 내의 모델 대표 명칭을 프리미엄 사양으로 변경하는 구체적인 구현 계획을 수록합니다.

## User Review Required

> [!IMPORTANT]
> **Windows OS API 종속성 및 DPI 배율 동기화**
> - MuJoCo 뷰어 창의 위치를 감지 및 제어하기 위해 Windows OS API인 `ctypes.windll.user32.GetWindowRect` 및 `MoveWindow`를 사용합니다.
> - 사용자가 수동으로 드래그해 놓은 최종 위치를 픽셀 단위로 완벽하게 복원 및 유지하기 위해 가상 프레임과 가시 가림막(DWM 그림자 등)을 배제한 윈도우 본연의 좌표계(물리 좌표계)를 100ms 주기로 안전하게 추적 및 실시간 캐싱합니다.
> - 이로 인해 OS의 화면 배율(DPI Ratio) 설정에 영향받지 않는 절대적인 픽셀 매칭 및 고정이 가능해집니다.

## Proposed Changes

---

### [Component 1] TVPackageMotionSim Drop Simulator

#### [MODIFY] [whts_control_panel.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_control_panel.py)
- **생성자 `__init__` 수정**:
  - `self._last_mujoco_hwnd = None` 필드를 추가하여 이전 뷰어 창의 윈도우 핸들을 캐싱합니다.
  - `self._last_mujoco_rect = None` 필드를 추가하여 직전 뷰어 창의 실제 물리 좌표 `(left, top, width, height)` 정보를 기억합니다.
- **`_align_with_mujoco_window` 메소드 수정**:
  - `EnumWindows`로 MuJoCo 창을 찾았을 때, 이전에 저장된 `_last_mujoco_hwnd`와 핸들이 달라진 상태라면 "새 창이 떴음(리로드 격발)"으로 판정합니다.
  - 이 경우, 캐싱된 `_last_mujoco_rect` 정보가 존재한다면 `ctypes.windll.user32.MoveWindow` API를 호출하여 새 창을 이전 창의 위치와 크기로 단 한 픽셀의 오차도 없이 강제 고정합니다.
  - 이후 `_mujoco_aligned = False`로 리셋하여 컨트롤 센터 창도 새 뷰어 위치에 맞게 정상적으로 재정렬(DWM extended frame bounds 기준)될 수 있게 유도합니다.
  - 매 주기마다 `GetWindowRect`를 이용해 가시 윈도우 좌표 `(left, top, width, height)`를 실시간으로 캐시하여 사용자가 수동으로 수정한 최신 위치를 상시 보존합니다.

---

### [Component 2] TVPackageMotionSim Discrete Builder

#### [MODIFY] [whtb_builder.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_builder.py)
- **XML 스트림 생성 로직 수정 (라인 371)**:
  - `<mujoco model="discrete_custom_box">`로 하드코딩되어 있던 모델의 기본 명칭 부분을 프리미엄 명칭인 `<mujoco model="Samsung Electronics TV Package Drop Motion Simulation">`으로 완전히 변경합니다.

---

### [Component 3] Centralized UI Theme

#### [MODIFY] [whts_theme.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_theme.py)
- **`TREE_QSS` 스타일 시트 조밀화**:
  - 설정 트리 뷰 `self.config_tree`의 각 행의 상하좌우 간격이 더욱 조밀하고 완성도 높게 표현되도록 `QTreeWidget::item` 선택자의 패딩 값을 `padding: 3px;`에서 `padding: 2px;`로 정교하게 변경합니다.
  - 변경 전 안전하게 recovery 가능한 `whts_theme_backup_20260521.py` 백업본을 작성합니다.

---

## Verification Plan

### Automated Tests
- 구문 오류 무결성 검증을 수행합니다.
  ```powershell
  python -m py_compile c:\Users\GOODMAN\WHToolsBox\TVPackageMotionSim\run_drop_simulator\whts_control_panel.py
  python -m py_compile c:\Users\GOODMAN\WHToolsBox\TVPackageMotionSim\run_discrete_builder\whtb_builder.py
  python -m py_compile c:\Users\GOODMAN\WHToolsBox\TVPackageMotionSim\run_drop_simulator\whts_theme.py
  ```

### Manual Verification
- 컨트롤 패널에서 **"Create & Reload Model"** 버튼을 클릭하여 모델 리로드를 수행합니다.
- 새로 뜨는 MuJoCo 창이 이전 창의 위치에서 단 1픽셀도 어긋나지 않고 완벽하게 동일한 자리에 생성되는지 눈으로 확인합니다.
- 사용자가 MuJoCo 창을 수동으로 임의의 위치로 드래그하여 옮긴 뒤, 다시 모델 리로드를 수행했을 때 해당 최종 사용자 위치에 창이 정확하게 복원되어 뜨는지 검증합니다.
- MuJoCo 뷰어 상단 타이틀 바 및 라이브 XML 에디터 본문에서 모델명이 `Samsung Electronics TV Package Drop Motion Simulation`으로 성공적으로 로드되는지 확인합니다.
- 설정 창 중앙의 `self.config_tree`의 각 리스트 항목들이 전방향 `2px` 패딩 설정에 맞춰 조밀하고 수려하게 나열되는지 검증합니다.

