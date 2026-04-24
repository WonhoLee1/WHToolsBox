# [Refactoring] plate_by_markers.py - Code Reorganization and Documentation

`plate_by_markers.py`의 코드를 보다 체계적이고 가독성 높은 구조로 개선합니다. 과도한 축약을 지양하고, 상세한 한글 주석을 추가하며, AI agent가 이해하고 확장하기 쉬운 객체지향 구조로 다듬습니다.

## User Review Required

> [!IMPORTANT]
> - **Variable Names**: `kin`, `sol`, `cfg` 등의 축약어를 `kinematics`, `solver`, `config` 등으로 대대적으로 변경합니다.
> - **Logic Preservation**: 핵심적인 JAX 기반 Kabsch 알고리즘 및 Kirchhoff 평판 역학 계산 로직은 유지하되, 인터페이스만 정리합니다.
> - **Dependency**: `koreanize-matplotlib`를 추가하여 시각화 시 한글 깨짐 문제를 해결합니다.

## Proposed Changes

### Core Logic Module

#### [MODIFY] [plate_by_markers.py](file:///c:/Users/GOODMAN/WHToolsBox/plate_by_markers.py) (Major Overhaul)
- **Class Refactoring**:
    - `PlateConfig`: 필드명을 명확히 하고 Docstring 추가.
    - `KinematicsManager`: `RigidBodyKinematicsManager`로 변경 고려 또는 내부 변수명 확장.
    - `KirchhoffPlateOptimizer`: 기저 함수 계산 및 최적화 로직의 변수명을 수학적 의미에 맞게 확장.
    - `PlateMechanicsSolver`: 물리량 계산 로직(Stress, Strain, Von-Mises)을 함수화하여 분리.
- **Orchestration**:
    - `ShellDeformationAnalyzer`: 데이터 로딩과 해석 실행의 책임을 명확히 분리.
- **Visualization**:
    - `QtVisualizer`: `QtDeformationVisualizer` 등으로 명확히 하고, UI 생성 로직을 `_setup_ui_layout`, `_setup_3d_view`, `_setup_plot_widgets` 등으로 세분화.
    - `koreanize_matplotlib.setup_export()` 적용 (설치 여부 확인 필요).

### Infrastructure

#### [NEW] [implementation_plan_20260414.md](file:///c:/Users/GOODMAN/WHToolsBox/dev_log/implementation_plan_20260414.md)
#### [NEW] [walkthrough_20260414.md](file:///c:/Users/GOODMAN/WHToolsBox/dev_log/walkthrough_20260414.md)
#### [NEW] [task_20260414.md](file:///c:/Users/GOODMAN/WHToolsBox/dev_log/task_20260414.md)

## Open Questions

> [!QUESTION]
> 1. `koreanize-matplotlib`가 현재 환경에 설치되어 있습니까? 설치가 필요하다면 `pip install` 명령을 실행해도 될까요?
> 2. `PlateMechanicsSolver`에서 계산하는 물리량 외에 추가로 분석이 필요한 항목이 있습니까? (예: 특정 방향의 변형 에너지 분포 등)

## Verification Plan

### Automated Tests
- `main` 블록의 `create_example_markers`를 통한 샘플 데이터 해석이 정상적으로 수행되는지 확인.
- JAX `jit` 컴파일 오류 여부 확인.

### Manual Verification
- 시각화 GUI가 정상적으로 로드되는지 확인.
- 한글 폰트가 차트에서 제대로 표시되는지 확인.
- 3D 뷰의 View 모드(Global/Local) 전환이 물리적으로 타당하게 작동하는지 확인.
