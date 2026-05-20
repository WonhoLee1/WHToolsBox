# [Goal Description]
1. Control Center UI에서 Back 및 Reset 버튼을 눌렀을 때 시뮬레이션의 물리적 시간(Time)이 정상적으로 초기화되거나 이전 프레임의 시간으로 되돌아가지 못하고, 계속 누적되어 증가하는 문제를 해결합니다. (구현 완료)
2. 'Create & Reload Model' 실행 시 새로 뜨는 MuJoCo 뷰어 창이 윈도우 OS의 캐스케이딩(Cascading) 규칙으로 인해 조금씩 우하단으로 밀려나는 현상을 방지하고, 이전 창 위치(또는 사용자가 수동으로 지정한 마지막 창 위치)에 고정되어 뜨도록 수정합니다.
3. 시뮬레이션 모델 XML의 기본 대표 이름인 `discrete_custom_box`를 `Samsung Electronics TV Package Drop Motion Simulation`으로 고도화하여 프리미엄 명칭의 전문성을 극대화합니다.

## User Review Required
- **MuJoCo 창 위치 고정 로직:** Windows 환경에서 Win32 API (`ctypes`)를 이용해 이전에 활성화되어 있던 "MuJoCo" 타이틀 창의 정확한 윈도우 좌표와 가시 크기를 Control Center 인스턴스에 캐싱(`self.last_mujoco_pos`)해둡니다. 리로드 등으로 새 창이 떴을 때, 해당 캐시가 존재하고 새 창 위치가 달라졌다면 `MoveWindow` API를 이용해 캐시된 위치로 강제 이동 및 크기 고정을 수행합니다.
- **모델 대표 명칭 변경:** `whtb_builder.py`에서 XML을 내보낼 때 생성되는 `<mujoco model="...">` 태그 내의 기본 문자열을 수정하므로, 하위 호환성 및 기존 시뮬레이션 엔진 로직에는 영향을 미치지 않습니다.

## Open Questions
- 없음.

## Proposed Changes

### Component: Drop Simulator Control Center (`run_drop_simulator`)
새로운 MuJoCo 창이 뜰 때 기존에 수동으로 정렬해 놓았거나 이전 세션에서 활성화되어 있던 위치를 완벽하게 유지하도록 제어 코드를 수정합니다.

---

#### [MODIFY] [whts_control_panel.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_control_panel.py)

- `ControlPanel.__init__(self, simulator)`
  - 생성자 하단에 이전 MuJoCo 창의 스크린 픽셀 위치 정보를 저장할 캐시 필드 `self.last_mujoco_pos = None`을 추가합니다.
- `ControlPanel._align_with_mujoco_window(self)`
  - MuJoCo 창의 핸들(`hwnd`)을 획득하고 크기 및 위치(`rect`)를 받아온 후, `self.last_mujoco_pos`가 존재하고 실제 위치가 오프셋된 경우 `ctypes.windll.user32.MoveWindow` API를 호출하여 이전 위치로 되돌려 고정시킵니다.
  - 동시에, 유효한 픽셀 위치일 때 해당 윈도우의 크기 및 위치를 `self.last_mujoco_pos` 필드에 상시 최신화합니다. (이를 통해 사용자가 임의로 옮겨놓은 위치도 다음 리로드 시 완벽히 보존됩니다.)

### Component: Discrete XML Builder (`run_discrete_builder`)
생성되는 XML 내의 모델 대표 이름을 삼성 기기 낙하 시뮬레이션 공식 타이틀로 업그레이드합니다.

---

#### [MODIFY] [whtb_builder.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_builder.py)

- `create_model(export_path, config, logger)`
  - XML 스트림 생성부(라인 371 부근)의 `<mujoco model="discrete_custom_box">`를 `<mujoco model="Samsung Electronics TV Package Drop Motion Simulation">`으로 수정합니다.

## Verification Plan

### Automated Tests
- `python -m py_compile` 명령을 활용하여 수정한 파일들의 구문 오류가 없음을 컴파일 검증합니다.

### Manual Verification
1. Control Center UI 실행 후 `New Model Setup` 또는 `Create & Reload Model`을 수행하여 MuJoCo 창을 생성합니다.
2. 생성된 MuJoCo 창을 임의의 위치(예: 모니터의 우측 끝)로 드래그하여 옮겨 놓습니다.
3. `Create & Reload Model` 버튼을 다시 클릭하여 리로드를 진행합니다.
4. 새로 열리는 MuJoCo 뷰어 창이 OS 기본값에 따라 우하단으로 어긋나지 않고, **직전에 옮겨 놓았던 위치와 정확히 일치하는 곳에 번쩍임 없이 부드럽게 고정되어 뜨는지** 검증합니다.
5. 리로드된 새로운 XML 소스코드 또는 Live Editor 창을 열어 `<mujoco model="Samsung Electronics TV Package Drop Motion Simulation">` 명칭이 정상적으로 로드되었는지 확인합니다.
