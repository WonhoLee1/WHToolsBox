# Implementation Plan - Post-Processor Module Separation

`plate_by_markers_v2.py` 파일의 비대해진 구조를 개선하기 위해, 기계공학 해석 엔진부와 PySide6/PyVista 기반 UI부를 물리적으로 분리합니다.

## Proposed Changes

### 1. Engine Module [NEW]
#### `whts_multipostprocessor_engine.py`(file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_multipostprocessor_engine.py)
- **역할**: 순수 수치 해석 및 데이터 가공 (JAX, NumPy 기반)
- **포함 클래스/함수**:
    - `PlateConfig` (Dataclass)
    - `AlignmentManager` (Kabsch 정렬)
    - `AdvancedPlateOptimizer` (다항식 근사)
    - `PlateMechanicsSolver` (판 이론 필드 계산)
    - `ShellDeformationAnalyzer` (개별 파트 분석 도구)
    - `PlateAssemblyManager` (병렬 해석 관리자)
    - `scale_result_to_mm` (단위 보정 유틸리티)
- **외부 의존성**: `numpy`, `jax`, `concurrent.futures`, `dataclasses`

### 2. UI Module [NEW]
#### `whts_multipostprocessor_ui.py`(file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_multipostprocessor_ui.py)
- **역할**: 대시보드 시각화 및 사용자 인터페이스 (Qt, PyVista, Matplotlib 기반)
- **포함 클래스/함수**:
    - `PlotSlotConfig`, `DashboardConfig` (Dataclasses)
    - `VisibilityToolWindow` (가시성 관리자 창)
    - `AddPlotDialog`, `AboutDialog` (다이얼로그)
    - `QtVisualizerV2` (메인 대시보드 클래스)
- **외부 의존성**: `PySide6`, `pyvista`, `matplotlib`, `whts_multipostprocessor_engine` (모델 참조용)

### 3. Entry Point / Desktop Launcher [NEW]
#### `whts_multipostprocessor.py`(file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_multipostprocessor.py)
- **역할**: UI 및 Engine을 결합한 통합 실행 파일.
- **기능**: `--engine-only` (Headless) 또는 `--ui-only` (GUI) 모드 지원 가능성 확보. 기존 `plate_by_markers_v2.py`의 `main` 로직을 이관합니다.

### 4. Legacy Migration [NEW]
#### `legacy/plate_by_markers_v2.py`
- 기존 파일을 `./legacy/` 디렉토리로 이동하여 보존합니다.
- (선택 사항) 해당 파일에 `DeprecationWarning`을 추가하고 새로운 모듈로 안내하는 메시지 출력 로직 삽입.

### 5. Dependency Update [CRITICAL]
- **`run_drop_simulation_cases_v5.py`**: `from run_drop_simulator.plate_by_markers_v2 import ...` 구문을 `from run_drop_simulator.whts_multipostprocessor_engine import ...` 등으로 수정합니다.
- **`whts_engine.py`**: 분석 도구 호출 경로를 새 모듈/엔트리 포인트로 변경합니다.

## User Review Required

> [!IMPORTANT]
> **고려해야 할 리스크 및 조치**
> 1. **Import Error**: `plate_by_markers_v2`를 직접 임포트하는 모든 현업 코드들을 찾아 수정해야 합니다. (이미 `grep` 결과로 `run_drop_simulation_cases_v5.py` 소유 확인)
> 2. **Legacy 보존**: 단순히 파일을 삭제하는 것이 아니라 `legacy/` 폴더에 버전(v2.5)을 명시하여 보관함으로써 추후 대조가 가능하도록 합니다.

## Verification Plan

### Automated Tests
- 분리된 `whts_multipostprocessor_engine.py`가 Qt 라이브러리 없이도 수치 해석을 정상 수행하는지 헤드리스(Headless) 환경 테스트.
- `plate_by_markers_v2.py` 실행 시 기존과 동일한 UI 및 해석 결과가 나오는지 확인.

### Manual Verification
- 대시보드 실행 후 3D 뷰어 조작 및 2D 그래프 업데이트 기능 점검.
- 가시성 매니저를 통한 파트 On/Off 동작 확인.
