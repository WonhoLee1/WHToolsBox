# [Goal] 자율 구조 해석 및 ParaView 분석 자동화 파이프라인 (Phase 2)

기존의 자율 구조 해석(v6.py) 및 데이터 내보내기 파이프라인을 고도화하여, 사용자가 **ParaView**를 실행하는 즉시 **3D 변형 형상**과 **2D 시계열 그래프**가 결합된 전용 대시보드가 자동으로 구성되도록 합니다. 또한 파이썬 코드에서 ParaView 매크로를 자동으로 등록하여 유지보수성을 극대화합니다.

## User Review Required

> [!IMPORTANT]
> **ParaView 버전 호환성**: 시스템에서 감지된 `ParaView 6.0.1`에 맞춰 `--script` 인자를 활용한 자동 로딩 기능을 구현합니다.
> 
> **매크로 자동 등록**: 사용자의 `AppData` 내에 `Macros` 폴더가 없는 경우 자동으로 생성하고 `WHTOOLS_Dashboard.py`를 영구 등록합니다.

## Proposed Changes

### [ParaView Automation Engine]

#### [NEW] [whts_paraview_setup.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_paraview_setup.py)
ParaView의 파이썬 엔진(PVPython)에서 실행될 초기 레이아웃 구성 스크립트입니다.
- **기능**:
  - `_Assembly_Full_Sequence.pvd` 로드 및 뷰 가시성 설정.
  - 레이아웃 분할 (좌측: 3D Render View, 우측: XY Chart View).
  - `Plot Data Over Time` 필터를 통한 실시간 응력/변위 그래프 생성.

#### [MODIFY] [whts_exporter.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_exporter.py)
- **추가**: `register_paraview_macro()` 메서드
  - `os.environ['APPDATA']`를 통해 ParaView 매크로 폴더 탐색.
  - `whts_paraview_setup.py`의 고정 버전(유지보수용)을 해당 폴더에 복사.
- **수정**: `launch_paraview()` 메서드
  - `paraview.exe --script=...` 인자를 사용하여 방금 생성된 데이터에 최적화된 초기 뷰를 띄움.

#### [MODIFY] [run_drop_simulation_cases_v6.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulation_cases_v6.py)
- 해석 및 저장 완료 후 `exporter.register_paraview_macro()` 및 `exporter.launch_paraview()` 호출.

## Verification Plan

### Automated Tests
- `v6.py` 실행 완료 후 ParaView 창이 떴을 때:
  1. 화면이 좌우로 자동 분할되는가?
  2. 우측 차트에 시간에 따른 최대 응력 곡선이 나타나는가?
  3. ParaView의 `Macros` 메뉴에 `WHTOOLS_Dashboard` 버튼이 존재하는가?

### Manual Verification
- ParaView를 수동으로 켰을 때도 `Macros` 버튼이 잘 보이는지 확인.
