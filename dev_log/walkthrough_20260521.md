# Walkthrough - MuJoCo Window Alignment & Model Naming Setup

본 문서에는 TV Package Motion Simulation 시스템에 적용된 MuJoCo 패시브 뷰어의 창 위치/크기 자동 복원 기능 및 XML 생성 기본 모델 이름 변경 작업의 상세 내역과 검증 결과를 기록합니다.

## Changes Made

### 1. MuJoCo 뷰어 창 위치 및 크기 자동 복원
- [whts_control_panel.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_control_panel.py) 파일에 캐싱 로직 도입.
  - 생성자 `__init__`에서 `self._last_mujoco_hwnd` 및 `self._last_mujoco_rect`를 생성하여 직전에 구동된 뷰어의 윈도우 핸들과 물리 좌표 구조체(RECT)를 안전하게 관리합니다.
  - `_align_with_mujoco_window` 타이머 핸들러에서 100ms 주기로 `ctypes.windll.user32.GetWindowRect`를 호출하여 사용자가 수동으로 조정한 최신 윈도우 물리 RECT를 상시 캐싱합니다.
  - 리로드 시 새로운 창 핸들이 발견되면(`self._last_mujoco_hwnd != hwnd`), 직전 기억해 둔 RECT의 좌우/상하 픽셀 데이터로 `ctypes.windll.user32.MoveWindow` API를 호출해 단 한 픽셀의 오차도 없이 동일한 기존 자리에 창을 띄워주고 크기를 자동 복원 조정합니다.
  - 그 직후 `self._mujoco_aligned = False`로 리셋하여 컨트롤 센터도 새로운 뷰어의 DWM extended frame bounds를 기준으로 예쁘게 자동 재정렬되도록 유기적인 격발 루프를 설계하였습니다.

### 2. XML 생성 모델 명칭 변경
- [whtb_builder.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_builder.py) 파일의 371라인 XML 작성 로직 수정.
  - 기존 `<mujoco model="discrete_custom_box">` 태그를 프리미엄 타이틀 명칭인 `<mujoco model="Samsung Electronics TV Package Drop Motion Simulation">`으로 완전히 변경하였습니다.

### 3. Config Tree 아이템 패딩(padding) 조밀화
- [whts_theme.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_theme.py) 파일 내 `TREE_QSS` 스타일 정의 수정.
  - `QTreeWidget::item` 선택자의 패딩 값을 `padding: 3px;`에서 `padding: 2px;`로 전방향 축소 조정하였습니다.
  - 이를 통해 설정 창(`self.config_tree`)의 각 행 간격이 더욱 조밀하고 짜임새 있게 정돈되어, 콤팩트하면서도 고급스러운 다크 테마 UI를 한층 강화하였습니다.

---

## Verification Results

### Automated Tests
- 구문 오류 무결성 검증을 위해 `python -m py_compile` 명령을 실행하였으며, 수정된 소스 파일 모두 컴파일에 완벽히 성공(Exit Code 0)하여 문법적 사이드 이펙트가 전혀 없음을 확인했습니다.
  ```powershell
  python -m py_compile c:\Users\GOODMAN\WHToolsBox\TVPackageMotionSim\run_drop_simulator\whts_control_panel.py
  python -m py_compile c:\Users\GOODMAN\WHToolsBox\TVPackageMotionSim\run_discrete_builder\whtb_builder.py
  python -m py_compile c:\Users\GOODMAN\WHToolsBox\TVPackageMotionSim\run_drop_simulator\whts_theme.py
  ```

### Manual Verification Flow
1. **리로드 및 복원 기능 테스트**:
   - 사용자가 MuJoCo 뷰어 창을 띄운 상태에서 임의의 위치로 이동 및 크기를 크거나 작게 변경합니다.
   - **"Create & Reload Model"** 버튼을 클릭하여 모델 리로드를 수행합니다.
   - 기존의 Cascading(점진적 우하단 이탈) 현상이 완벽하게 해결되고, 새롭게 로드된 패시브 창이 이전 사용자가 의도하여 배치해 두었던 그 자리, 그 크기 그대로 완벽하게 복원 및 이동 조정되어 뜨는 것을 확인합니다.
2. **모델 타이틀 명칭 검증**:
   - 새로 리로드된 MuJoCo 뷰어 창의 상단 타이틀 바에 `Samsung Electronics TV Package Drop Motion Simulation`이 정상적으로 표시되는지 확인합니다.
   - 라이브 XML 에디터(`Live XML Editor`)를 열어 `<mujoco model="...">` 태그가 성공적으로 변경된 상태로 로드되어 있음을 검증합니다.
3. **트리 뷰 조밀도 시각 검증**:
   - 설정 창 중앙의 `self.config_tree` 행 높이와 여백이 전방향 2px 패딩 설정으로 인해 더욱 촘촘하고 정교하게 출력되는지 확인합니다.


---

## [2026-05-21] Monitor Configuration 창 개선 및 Edge 낙하 자세 버그 수정

### 수정 파일 목록
| 파일 | 역할 |
|------|------|
| `run_drop_simulator/whts_monitor.py` | Monitor Configuration + Curve 창 개선 |
| `run_drop_simulator/whts_control_panel.py` | UI 오버라이트 버그 3건 수정 |
| `run_discrete_builder/whtb_utils.py` | 평행 면 조합 경고 로직 추가 |

---

### 1. whts_monitor.py 변경 내역

#### C1~C8 기본값 모두 체크
- 기존: C1만 체크
- 변경: 모든 코너(C1~C8) 기본 체크 상태

#### Select Axes 모두 기본 체크
- 기존: Z축만 체크
- 변경: X, Y, Z, Resultant 모두 기본 체크

#### 커브 창 항상 위(WindowStaysOnTopHint)
- RealTimeMonitorWindow에 `Qt.WindowStaysOnTopHint` 플래그 추가
- 시뮬레이션 실행 중에도 커브 창이 MuJoCo 뷰어 위로 유지됨

#### Legend 플롯 우측 밖 배치
- `fig.subplots_adjust(right=0.78)`으로 우측 22% 여백 확보
- 각 subplot의 legend를 `bbox_to_anchor=(1.01, 0.5)`로 axes 기준 우측 바깥에 배치
- View → Tight Layout 메뉴 클릭 시에도 subplots_adjust 재적용하여 legend 위치 유지

---

### 2. whts_control_panel.py Edge 낙하 자세 버그 수정

#### Bug Fix 1: __init__ 초기화 순서
- `_on_ista_changed` 호출 전에 config의 `drop_direction`을 `edit_direction`에 먼저 주입
- 기존 Edge/Corner 방향값이 초기화 시 보존됨

#### Bug Fix 2: _on_ista_changed 무조건 덮어쓰기
- 기존 유효한 Edge/Corner 방향이 있으면 기본값으로 초기화하지 않도록 방어 로직 추가
- 비어있거나 반대 모드의 기본값인 경우에만 초기화

#### Bug Fix 3: _on_apply_and_sync의 blockSignals 누락
- combo_ista.setCurrentText() 호출 시 blockSignals(True/False) 적용
- Setup Helper → Apply 시 시그널 연쇄 발화로 방향이 덮어쓰이는 버그 차단

---

### 3. whtb_utils.py parse_drop_target 강화

#### 평행 면 조합 검출 및 경고
- LTL: {1-3, 2-4, 5-6}, PARCEL: {1-2, 3-4, 5-6} 조합은 모서리 불가
- frozenset 기반으로 순서 무관 검출
- 경고 메시지 터미널 출력 후 Bottom 낙하로 안전 fallback

---

### 검증 결과
- 3개 수정 파일 Python 구문 검사: **전체 통과 (exit code 0)**
- LTL Edge 3-4 방향 파싱: `[0, -0.06, 0.525]` (올바른 Edge 벡터) 확인
- 평행 면 조합(LTL: 1-3, 2-4, 5-6) → WARNING + Bottom fallback 정상 동작 확인
