# Implementation Plan - Project Structure Refinement & Multi-case Update

`run_drop_simulator` 패키지를 독립 실행 가능한 모듈로 완성하고, 루트의 모든 버전 관리 스크립트(`v3`, `v4`) 및 기존 케이스 스크립트를 `legacy` 폴더로 이동하여 정리합니다.

## User Review Required

> [!IMPORTANT]
> - **구조 통합**: `run_drop_simulation_v4.py`의 로직은 패키지 내부(`__main__.py`)로 이동하여 `python -m run_drop_simulator`로 실행하도록 변경합니다.
> - **케이스 스크립트 업데이트**: 기존 `run_drop_simulation_cases.py`를 계승한 `run_drop_simulation_cases_v4.py`를 생성하고 신규 패키지를 사용하도록 수정합니다.
> - **레거시 아카이브**: `v3`, `v4` 파일 및 기존 케이스 파일을 모두 `./legacy_reference/` 폴더로 이동하여 루트 디렉토리를 최소화합니다.

## Proposed Changes

### 1. [Simulator Package] `run_drop_simulator` 고도화

#### [NEW] [__main__.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/__main__.py)
- `run_drop_simulation_v4.py`의 실행부 코드를 이식합니다.

### 2. [Integration] 케이스 실행 스크립트 업데이트

#### [NEW] [run_drop_simulation_cases_v4.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulation_cases_v4.py)
- 기존 `run_drop_simulation_cases.py`를 기능적으로 계승합니다.
- **Import 수정**: `from run_drop_simulator import DropSimulator`

### 3. [Cleanup] Legacy Archive (./legacy_reference/)

#### [MOVE] [v3, v4, cases legacy](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/legacy_reference/)
- 아래 파일들을 `./legacy_reference/` 폴더로 이동합니다:
    - `run_drop_simulation_v3.py`
    - `run_drop_simulation_v4.py`
    - `run_drop_simulation_cases.py`

## Open Questions

- 현재 백그라운드에서 실행 중인 `python run_drop_simulation_v4.py` 프로세스를 제가 중지(Terminate)해도 될까요? 파일 이동을 위해 프로세스 종료가 선행되어야 합니다.

## Verification Plan

### Automated Tests
- `python -m run_drop_simulator` 실행 확인.
- `python run_drop_simulation_cases_v4.py` 실행 확인.

---

### 작업 후 루트 디렉토리 예상 구조
```text
/TVPackageMotionSim/
  ├── run_drop_simulator/ (Package)
  ├── run_discrete_builder/ (Package)
  ├── run_drop_simulation_cases_v4.py (New Runner)
  ├── legacy_reference/ (Archived Scripts)
  └── dev_log/ (Documentation)
```
