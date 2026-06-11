# Implementation Plan - Integrating OpenRadioss Execution Scripts into Project (2026-06-09)

이 계획서는 외부 `openradioss_gui` 디렉토리에 존재하는 `runopenradioss.py` 및 `inp2rad.py` 파일을 프로젝트 내부로 이관(migration)하고, PyInstaller로 빌드된 단일 실행 파일(`.exe`)에서도 무한 루프 없이 안정적으로 OpenRadioss 해석이 실행될 수 있도록 개선하는 작업 및 사용자의 추가 수정 요구사항을 정의합니다.

## User Review Required

> [!IMPORTANT]
> **Git Push 누락 상황**:
> - 로컬 커밋은 원격 브랜치 `origin/master`와 동기화되어 있으나, 로컬 전용 브랜치 `D0410`은 원격에 생성/푸시되지 않은 상태입니다.
> - 작업 트리(Working Tree)에 `dev_log/*_20260607.md`, `do_restore.py`, 각종 `.bak` 파일 등 최근 작성 파일들이 커밋 및 푸시되지 않은 **Untracked** 상태로 존재하고 있습니다.
> - 변경 및 작성된 파일들을 확인 후 Commit & Push 처리가 필요합니다.

## Proposed Changes

### [OpenRadioss Execution Scripts Integration]

외부의 OpenRadioss 실행 관련 소스 코드들을 프로젝트의 모듈 경로 안으로 복사하고, 의존성 관계 및 빌드 설정을 조정합니다.

#### [NEW] [runopenradioss.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/runopenradioss.py)

- 외부 `D:\OpenRadioss_win64\OpenRadioss\openradioss_gui\runopenradioss.py`를 프로젝트의 `run_drop_simulator` 디렉토리 하위로 복사합니다. (완료)
- `tkinter` 로드 부분에 `try-except` 구문을 적용해 무두방(Headless) 환경 대응성을 강화합니다. (완료)
- **[추가 수정]** `job_process` 메서드에서 stdout을 출력할 때 불필요한 앞 공백 `"  "`을 붙여서 가공 출력하는 대신, OpenRadioss가 제공하는 원래의 stdout 라인을 그대로 출력하도록 `print(line.strip())`으로 변경합니다.

#### [NEW] [inp2rad.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/inp2rad.py)

- 외부 `D:\OpenRadioss_win64\OpenRadioss\openradioss_gui\inp2rad.py`를 프로젝트의 `run_drop_simulator` 디렉토리 하위로 복사하여 `.inp` 형식의 메쉬 파일을 `.rad` 파일로 번역하는 변환 모듈을 로컬화합니다. (완료)

#### [MODIFY] [whts_radioss_builder.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_radioss_builder.py)

- `whts_radioss_builder.py`의 `run` 함수를 수정합니다.
- 기존의 임시 파이썬 래퍼 스크립트를 파일로 쓰거나 `-c` 파라미터와 `_sys.executable`로 서브프로세스를 생성하던 방식을 걷어냅니다. (완료)
- 내부적으로 로컬 모듈 `from .runopenradioss import RunOpenRadioss`를 직접 가져와서 실행합니다. (완료)
- **[추가 수정]** Radioss 실행 시 불필요한 디버그 안내 정보가 인쇄되지 않도록 `RunOpenRadioss(command, debug=0)`으로 `debug` 값을 `0`으로 수정합니다.
- **[추가 수정]** `CallbackStream`을 통해 가로챈 sys.stdout 데이터를 UI 콜백에 전달할 때, 접두어 `[Radioss] `를 붙이지 않고 원래의 `val` 데이터 그대로 전달하여 원본 stdout과 터미널 메시지가 일치하도록 개선합니다.

#### [MODIFY] [drop_simulator_v6.spec](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/drop_simulator_v6.spec)

- **[추가 수정]** PyInstaller 빌드 시 누락되었던 내부 프로젝트 모듈(`whts_` 모듈 18개, `whtb_` 모듈 6개) 및 사용 확률이 높은 공통 모듈류를 `hiddenimports`에 대폭 확충합니다.
- 누락되기 쉬운 외부 라이브러리(`numba`, `openpyxl`, `lxml` 등)도 포함하여 실행 시 `ModuleNotFoundError` 발생을 완전 차단합니다.

#### [MODIFY] [whts_utils.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_utils.py)

- **[추가 수정]** 프로그램 실행 시 `WHToolsSessionLogger._acquire` 함수에서 `candidates` 후보군에 국한하지 않고, 디렉토리 내에 존재하는 모든 `*.lock` 파일들을 스캔하도록 수정합니다.
- 각 `.lock` 파일 내부의 PID를 분석하여 현재 구동 중인 프로세스가 없을 경우(잡고 있는 경우가 없는 stale lock), 해당 락 파일 및 이에 대응하는 로그 파일(`.log`)을 안전하게 자동 일괄 삭제합니다.

#### [MODIFY] [whts_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_engine.py)

- **[추가 수정]** 시뮬레이션이 종료되어 `simulation_result.pkl` 파일을 저장하는 `_build_and_save_result` 시점에, 자동으로 `SimulationDataExporter`를 활용해 CSV 및 Matplotlib 기반의 PNG 결과 그래프를 생성하여 저장하도록 설정합니다. (기본 동작화)

#### [MODIFY] [whts_control_panel.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_control_panel.py)

- **[추가 수정]** 상단 `🔍 View` 메뉴 아래에 결과 폴더를 파일 탐색기(Explorer)로 열어주는 `📂 Open Result Folder in Explorer` 액션을 추가하고 핸들러 `_on_open_result_folder`를 연동합니다.
- **[추가 수정]** `Run Engine` 시 비동기 `RadiossEngineWorker`의 로그 출력(`sig_log`)이 `self.sim.log`를 타서 `WHTS_Engine` 로거에 의해 중복/장황하게 출력(RichHandler 양식)되지 않도록, `print(msg, flush=True)`로 직접 연결해 가공 없이 깨끗하게 출력합니다.

#### [MODIFY] [run_drop_simulation_cases_v6.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulation_cases_v6.py)

- **[추가 수정]** 콘솔 가로 폭을 120열로 강제 조정하는 윈도우 API 설정을 `configure_terminal_font` 내에 추가하고, 기동 시 터미널을 깨끗하게 비운 뒤 멋진 `WHTOOLS` ASCII Art를 대형 폰트로 터미널에 먼저 인쇄하도록 설정합니다.

---

## Verification Plan

### Automated & Manual Verification

- **일반 실행 환경 검증**:
  - `python run_drop_simulation_cases_v6.py`를 구동하여 Radioss Run 시뮬레이션 케이스가 정상 구동하는지 확인합니다.
  - 로그 창에 디버그성 출력 없이, OpenRadioss Starter 및 Engine의 원본 stdout만 깨짐 없이 그대로 출력되는지 검증합니다.
  - 프로그램 재실행 시, 기존에 생성되었던 프로세스 미점유 stale 락 파일 및 대응 로그 파일이 깔끔하게 자동 삭제되는지 검증합니다.
- **PyInstaller 빌드 파일 실행 검증**:
  - `build_drop_simulator_v6.ps1`을 실행하여 새롭게 `WHTools_DropSimulator_v6.exe`를 빌드합니다.
  - 생성된 `.exe`를 실행하고 **Radioss Run** 기능을 작동시켰을 때, 누락된 모듈 오류 없이 정상 실행되며, 백그라운드에서 OpenRadioss 엔진이 무사히 동작하여 결과 파일을 생성하는지 검증합니다.
