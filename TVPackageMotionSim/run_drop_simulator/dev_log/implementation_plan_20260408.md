# Implementation Plan - WHTOOLS Dashboard UI Refinement

[WHTOOLS] 대시보드의 가독성과 사용성을 높이기 위해 요청하신 UI 개선 사항을 반영합니다. 기존 메뉴바 체제를 제거하고 탭 기반의 통합 제어 환경으로 전환합니다.

## User Review Required

> [!IMPORTANT]
> 1. **메뉴바 제거**: 상단 메뉴바(`mb = self.menuBar()`)를 완전히 제거하고 모든 기능을 탭으로 이관합니다.
> 2. **탭 높이 변경**: 탭 영역의 높이를 기존 120px에서 약 180px~ 200px로 확장하여 더 많은 제어 위젯을 수용합니다.
> 3. **Floor 설정**: Context Menu에서 간단한 입력을 받아 Floor의 위치와 크기를 실시간으로 변경하는 로직이 추가됩니다.

## Proposed Changes

### [UI] [whts_multipostprocessor_ui.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_multipostprocessor_ui.py)

#### 1. 레이아웃 및 탭 구조 재구성
- **로고 확대**: `scaledToHeight(60)` -> `scaledToHeight(90)`으로 변경하여 브랜드 시인성 강화.
- **메뉴바 제거**: `self.menuBar()` 관련 코드 삭제.
- **Settings 탭 추가**: `⚙️ Settings` 탭을 생성하고 다음 기능을 배치:
    - Visibility Manager 실행 버튼
    - Reset Camera 버튼
    - About WHTOOLS 버튼
    - Animation Step (Skip frames) 제어 (`QSpinBox`)
- **2D Plot Control 이관**: 기존 2D 패널 좌측 상단에 있던 `2D Plot Control` 그룹박스를 `📈 2D Field & Curves` 탭 내부로 이동.

#### 2. 3D 시각화 제어 개선
- **Field & Range 그룹 수정**: `Min Note`, `Max Note` 체크박스를 `Field & Range` 그룹박스 내부의 하단 레이아웃에 배치.
- **Perspective 설정**: `self.ch_per.setChecked(True)`를 기본값으로 적용.
- **Animation Step 복구**: `self.anim_step`을 사용자가 조정할 수 있는 UI 추가 (Settings 탭).

#### 3. Context Menu (우클릭 메뉴) 고도화
- **PyVista 단축키 메뉴 정제**: `_ PyVista Key:` 텍스트를 제거하고, 클릭 시 즉시 해당 기능(Wireframe, Reset, Point/Surface 등)이 실행되도록 핸들러 연결.
- **Floor 제어 기능 추가**:
    - `Floor Settings` 메뉴 항목 추가.
    - 실행 시 간단한 Input Dialog들을 통해 `Origin (예: 0,0,0)`, `Direction (Normal)`, `Size (W, H)`를 입력받아 `self.ground` 메쉬를 업데이트.

## Open Questions

- Floor의 `Direction` 입력 시 `(0,0,1)` 같은 벡터 형태 외에 `XY`, `YZ`, `ZX` 면 선택 방식을 병행할지 확인이 필요합니다. (현재는 벡터 입력을 기본으로 구현 예정)

## Verification Plan

### Automated Tests
- `python whts_multipostprocessor.py`를 실행하여 레이아웃이 깨지지 않고 탭 이동이 원활한지 확인.
- 각 버튼(About, Reset, Visibility)의 동작 여부 확인.

### Manual Verification
- **우클릭 메뉴**: Floor 위치(Origin)를 `0,0,100`으로 변경했을 때 바닥면이 상승하는지 확인.
- **애니메이션**: Step을 5로 설정하고 재생 시 프레임이 5씩 건너뛰는지 확인.
- **2D 탭**: 탭 내에서 Layout 변경 시 그래프 영역이 정상적으로 갱신되는지 확인.
