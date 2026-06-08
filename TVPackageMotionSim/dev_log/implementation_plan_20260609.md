# Implementation Plan - Integrating OpenRadioss Execution Scripts into Project (2026-06-09)

이 계획서는 외부 `openradioss_gui` 디렉토리에 존재하는 `runopenradioss.py` 및 `inp2rad.py` 파일을 프로젝트 내부로 이관(migration)하여, PyInstaller로 빌드된 단일 실행 파일(`.exe`)에서도 무한 루프 없이 안정적으로 OpenRadioss 해석이 실행될 수 있도록 개선하는 작업을 정의합니다.

## User Review Required

> [!IMPORTANT]
>
> - `runopenradioss.py`가 프로젝트 내부로 들어오게 되면, 해당 스크립트는 더 이상 외부의 독립적인 OpenRadioss GUI 환경에 종속되지 않고 프로젝트 전용으로 동작하게 됩니다.
> - `sys.executable`을 활용한 별도 파이썬 프로세스 호출 방식이 제거되고, 프로젝트의 현재 프로세스(혹은 백그라운드 스레드) 상에서 `RunOpenRadioss` 클래스를 직접 로드하여 실행합니다.

## Proposed Changes

### [OpenRadioss Execution Scripts Integration]

이 단계에서는 외부의 OpenRadioss 실행 관련 소스 코드들을 프로젝트의 모듈 경로 안으로 복사하고, 의존성 관계 및 빌드 설정을 조정합니다.

#### [NEW] [runopenradioss.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/runopenradioss.py)

- 외부 `D:\OpenRadioss_win64\OpenRadioss\openradioss_gui\runopenradioss.py`를 프로젝트의 `run_drop_simulator` 디렉토리 하위로 복사합니다.
- 임포트 시 `tkinter`가 없거나 로드 오류가 발생하는 경우를 대비해 `tkinter` 로드 부분에 `try-except` 구문을 적용해 무두방(Headless) 환경 대응성을 강화합니다.
- `RunOpenRadioss`의 `d3plot_conversion` 이나 `convert_anim_to_vtkhdf` 시 발생하는 모듈 임포트 경로 문제를 로컬 디렉토리에 맞게 보완합니다.

#### [NEW] [inp2rad.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/inp2rad.py)

- 외부 `D:\OpenRadioss_win64\OpenRadioss\openradioss_gui\inp2rad.py`를 프로젝트의 `run_drop_simulator` 디렉토리 하위로 복사하여 `.inp` 형식의 메쉬 파일을 `.rad` 파일로 번역하는 변환 모듈을 로컬화합니다.

#### [MODIFY] [whts_radioss_builder.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_radioss_builder.py)

- `whts_radioss_builder.py`의 `run` 함수(line 117~187)를 전면 수정합니다.
- 기존의 임시 파이썬 래퍼 스크립트를 파일로 쓰거나 `-c` 파라미터와 `_sys.executable`로 서브프로세스를 생성하던 방식을 걷어냅니다.
- 내부적으로 로컬 모듈 `from .runopenradioss import RunOpenRadioss`를 직접 가져와서, `RunOpenRadioss` 인스턴스를 생성하고 즉시 `batch_run()` 메소드를 호출하도록 변경합니다.
- 메인 GUI 스레드가 차단되지 않도록 콜백 및 비동기 처리 흐름에 영향이 없는지 검토합니다.

#### [MODIFY] [drop_simulator_v6.spec](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/drop_simulator_v6.spec)

- `run_drop_simulator/runopenradioss.py`와 `run_drop_simulator/inp2rad.py` 파일이 PyInstaller 분석(Analysis) 과정에서 감지되도록 `hiddenimports`에 누락이 없는지 검토하고 필요한 경우 명시적으로 추가합니다.
- 필요시 `tkinter` 관련 DLL 또는 외부 툴 종속성을 추가로 수집할 수 있도록 설정을 보강합니다.

---

## Verification Plan

### Automated & Manual Verification

- **일반 실행 환경 검증**:
  - `python run_drop_simulation_cases_v6.py` 또는 `v7`을 구동하여 Radioss Run 시뮬레이션 케이스가 정상 구동하는지 확인합니다.
  - 로그 창에 `[Radioss] Starter ...` 및 `[Radioss] Engine ...` 등의 진행 상황 메시지가 실시간으로 출력되는지 검증합니다.
- **PyInstaller 빌드 파일 실행 검증**:
  - `build_drop_simulator_v6.ps1`을 실행하여 새롭게 `WHTools_DropSimulator_v6.exe`를 빌드합니다.
  - 생성된 `.exe`를 실행하고 **Radioss Run** 기능을 작동시켰을 때, `~v6.exe`가 다시 재귀 실행되는 버그가 완전히 해결되었고 백그라운드에서 `OpenRadioss` 엔진이 무사히 동작하여 결과 파일을 생성하는지 검증합니다.
