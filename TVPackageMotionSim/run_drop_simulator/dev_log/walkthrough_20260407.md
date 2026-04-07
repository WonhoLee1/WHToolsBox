# Walkthrough: WHTOOLS Post-Processor Architecture Refactoring

[WHTOOLS]의 포스트 프로세서 아키텍처를 고도화하여, 유지보수성과 확장성을 극대화한 모듈형 구조로 리팩토링을 완료했습니다. 기존의 거대했던 `plate_by_markers_v2.py`를 수치 해석 엔진과 시각화 UI로 물리적으로 분리했습니다.

## 1. 주요 변경 사항 (Core Refactoring)

### 1.1. 해석 엔진 분리 (`whts_multipostprocessor_engine.py`)
- **역할**: JAX 가속 수치 해석, 물리적 필드 생성, 강체 운동 제거(Alignment) 전담.
- **특징**: `PySide6`나 `PyVista` 같은 UI 라이브러리에 대한 의존성을 100% 제거하였습니다. 이로써 서버 환경이나 Headless 모드에서도 독립적으로 해석 로직을 호출할 수 있습니다.
- **주요 클래스**: `ShellDeformationAnalyzer`, `PlateAssemblyManager`, `PlateMechanicsSolver`.

### 1.2. 시각화 UI 분리 (`whts_multipostprocessor_ui.py`)
- **역할**: PyVista(3D) 및 Matplotlib(2D)를 활용한 인터랙티브 대시보드 인터페이스 관리.
- **특징**: 엔진 모듈을 임포트하여 데이터를 시각화하며, `Visibility Manager` 고도화 및 런타임 안정성(HasAttr 체크 등)을 강화했습니다.
- **주요 클래스**: `QtVisualizerV2`, `VisibilityToolWindow`, `AddPlotDialog`.

### 1.3. 통합 실행기 구축 및 레거시 대응
- **신규 실행기**: `whts_multipostprocessor.py`를 통해 전체 시스템을 한 번에 구동할 수 있습니다.
- **레거시 래퍼**: 기존 `plate_by_markers_v2.py`를 **Deprecated Wrapper**로 전환했습니다. 기존 코드를 임포트하는 외부 스크립트들이 수정 없이도 동작하도록 하위 호환성을 확보했습니다.
- **외부 의존성 업데이트**: `run_drop_simulation_cases_v5.py`의 임포트 경로를 신규 모듈 구조에 맞게 최적화했습니다.

## 2. 작업 결과물 (Deliverables)

- [x] `legacy/plate_by_markers_v2.py`: 원본 코드 보존 및 아카이빙
- [x] [engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_multipostprocessor_engine.py): 해석 엔진 모듈
- [x] [ui.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_multipostprocessor_ui.py): 시각화 UI 모듈
- [x] [launcher.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_multipostprocessor.py): 통합 실행기
- [x] [v5_cases.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulation_cases_v5.py): 의존성 업데이트 완료

## 3. 검증 결과 (Verification)

> [!IMPORTANT]
> **통합 테스트 통과**
> 1. `whts_multipostprocessor.py` 독립 실행 시 데모 데이터(큐브 6면) 해석 및 대시보드 로딩 확인.
> 2. `KeyError` 및 `AttributeError` (초기화 순서 관련) 전수 해결 및 런타임 안정성 확보.
> 3. `plate_by_markers_v2.py`를 통한 간접 호출 시 경고 메시지 출력 및 정상 동작 확인.

이제 공학적 해석 엔진과 사용자 인터페이스가 깔끔하게 분리되어, 향후 신규 해석 이론 추가나 웹 기반 대시보드 확장 시 더욱 유연하게 대응할 수 있는 기반이 마련되었습니다.
