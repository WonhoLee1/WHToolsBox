# Professional Dashboard Enhancement Plan (V5.4.0)

본 계획은 Qt 기반 시각화 대시보드를 전문 FEM 소프트웨어 수준의 인터페이스와 기능을 갖추도록 고도화하는 것을 목표로 합니다.

## User Review Required

> [!IMPORTANT]
> **단축키 충돌 주의**: `Ctrl + Shift + 1~N` 단축키는 현재 Windows 또는 다른 상위 앱과 충돌할 가능성이 있습니다. 구현 후 테스트가 필요합니다.
> **초기 방향 보정**: `Top/Bottom`과 `Front/Rear` 배치가 반전된 문제를 `whts_mapping.py` 단계에서 재검토할 예정입니다.

## Proposed Changes

### 1. 3D Visualization & UX (`plate_by_markers_v2.py`)

#### Field Mode 확장
- [ ] `Field` 콤보박스에 `Body Color`, `Face Color` 옵션 추가.
- [ ] **Body Color**: 파트 이름의 키워드(`cushion`, `opencell`, `chassis` 등)별로 고정 컬러맵 할당.
- [ ] **Face Color**: 각 `analyzer` 인덱스 기반 고유 색상 할당.

#### Context Menu (Right Click) 강화
- [ ] **Views**: `XY`, `YZ`, `ZX` (Front/Back) 및 `Isometric` (4개 방향) 서브메뉴 추가.
- [ ] **Visibility**: `Mesh Edge (Line) Visibility`, `Floor Visibility` (원점 기준 평면 선택형) 추가.
- [ ] **Linkage**: `Part - Marker Visibility Link` 체크 박스 메뉴 추가.
- [ ] **Axes**: 좌측 하단 `add_axes()` 위젯 추가.

#### 단축키 시스템 (`QShortcut`)
- [ ] `f`, `F`: Fit View (Reset Camera).
- [ ] `Ctrl + Shift + 1~N`: 표준 뷰 전환.

#### 가시성 연동 로직
- [ ] `MarkerActor` 클래스 보강: 포인트(Sphere)와 라벨(Text) 공동 가시성 제어.
- [ ] `Visibility Link` 활성 시 파트 트리 조작과 마커 그룹 조작 동기화.

### 2. 2D Plotting & Styling (`plate_by_markers_v2.py`, `mpl_extension.py`)

#### Plot 제어 기능
- [ ] 2D Plot 하단에 `Interpolate` 체크박스 추가 (기본 On).
- [ ] `On` 시 `pcolormesh(shading='linear')` 또는 `contourf` 사용하여 부드러운 이미지 생성.

#### Matplotlib 스타일 최적화
- [ ] 전역 폰트 크기 `9pt`, 범례 `8pt` 강제 적용.
- [ ] 축 라벨 표준화: `Position X [mm]`, `Position Y [mm]`, `Time [s]`, `Magnitude [Unit]`.

#### 팝업 기능
- [ ] 각 차트 옆/위에 `Pop-out` 버튼 추가.
- [ ] 클릭 시 새로운 Matplotlib 창 창조 및 상단 툴바 활성화.

### 3. Data Mapping & Alignment (`whts_mapping.py`)

- [ ] `is_on_face` 로직 및 `face_offsets` 계산 시 `Top/Front/Rear/Bottom` 좌표축 매핑 재검토.
- [ ] 시뮬레이션 데이터의 Z-up 방향이 대시보드 3D 뷰의 정면과 일치하도록 초기 카메라 각도 설정 보정.

## Verification Plan

### Automated/Manual Tests
- [ ] **Visibility Sync**: 파트 리스트에서 하나를 끌 때 마커 점과 이름이 동시에 사라지는지 확인.
- [ ] **Shortcuts**: `f` 키 입력 시 화면 중앙 정렬 확인.
- [ ] **Color Modes**: `Body Color` 선택 시 모든 쿠션이 동일한 색상으로 변하는지 확인.
- [ ] **Pop-out**: 새 창이 뜨고 줌/팬 툴바가 정상 작동하는지 확인.

### User Feedback
- [ ] 보정된 초기 방향이 사용자의 의도(Top은 위, Front는 앞)와 일치하는지 확인 필요.
