# Walkthrough - OpenRadioss Integration, PyInstaller Setup, & Log Improvements (2026-06-09)

이 가이드에서는 PyInstaller 빌드 파일(`.exe`) 환경에서 발생하던 모듈 누락 오류와 무한 루프 버그를 완전히 해결하고, 사용자 요청에 맞춰 Radioss 실행 로그 포맷 및 UI 리소스 오류를 개선하고, 프로젝트 락 파일들을 정리한 변경 요약을 기술합니다.

## 변경 사항 요약

### 1. 스크립트 로컬화 및 로깅 개선
*   **스크립트 로컬화 (완료)**: 외부 `openradioss_gui` 디렉토리에 있던 [runopenradioss.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/runopenradioss.py) 및 [inp2rad.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/inp2rad.py)를 `run_drop_simulator` 하위로 완전히 마이그레이션했습니다.
*   **디버그성 로그 축소 및 가공 제거**:
    - [whts_radioss_builder.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_radioss_builder.py)에서 `RunOpenRadioss` 생성 시 `debug=0`을 넘겨주어 클래스 초기화 시 인쇄되던 다량의 디버그 라인을 제거했습니다.
    - `CallbackStream`에서 `[Radioss]` 접두사를 제거하여 원래 stdout 텍스트 그대로 UI 로그로 내보내도록 수정했습니다.
    - `runopenradioss.py` 내의 `job_process`에서도 불필요하게 덧붙던 앞공백 `"  "`을 지우고 원래 로그 라인이 그대로 나오도록 `print(line.strip())` 형태로 변경했습니다.

### 2. PyInstaller 빌드 환경 모듈 누락 완치
*   **Hidden Imports 대폭 확장**: [drop_simulator_v6.spec](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/drop_simulator_v6.spec) 파일 내의 `hiddenimports` 배열에 18개의 프로젝트 `whts_` 모듈, 6개의 `whtb_` 빌더 모듈, 공통 유틸 모듈 및 누락되기 쉬운 외부 핵심 라이브러리(`numba`, `lxml`, `openpyxl`, `scipy`의 각종 서브패키지 등)를 명시적으로 선언하여 exe 실행 시 `ModuleNotFoundError`가 발생하던 현상을 완전히 해결했습니다.
*   **빌드 검증**: `build_drop_simulator_v6.ps1` 스크립트를 통해 PyInstaller 빌드를 정상적으로 수행하였으며, 오류 없이 `WHTools_DropSimulator_v6.exe` 단일 패키지가 성공적으로 생성되었습니다.

### 3. UI 로고 PNG 참조 오류 수정
*   **누락 이미지 검출**: 대시보드 UI 파일인 [whts_multipostprocessor_ui.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_multipostprocessor_ui.py)에서 `resources/logo.png` 파일을 로드하려고 했으나, 실제 폴더 내에는 `sidebar_logo.png`만 존재하여 로고가 보이지 않던 버그를 확인했습니다.
*   **수정 사항**: `self.logo_path` 경로 설정을 `resources/sidebar_logo.png`로 올바르게 수정하여 렌더링 누락 오류를 완치했습니다.

### 4. Lock 파일 클린업 및 Stale Lock 일괄 제거 개선 (Clean up)
*   **일회성 정리**: 프로젝트 루트 경로 및 하위 `TVPackageMotionSim` 내에 남아있던 여러 동기화 락 파일들(`*.lock`)을 깨끗하게 일괄 삭제하였습니다.
*   **자동화 로직 고도화**: [whts_utils.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_utils.py)의 `WHToolsSessionLogger._acquire`를 개선하여, 프로그램 기동 시 지정된 candidates 후보군 뿐만 아니라 **디렉토리 내의 모든 *.lock 파일**을 전수 조사합니다.
*   락 파일 내부에 적혀 있는 PID가 활성 프로세스가 아닐 경우(좀비 락), 해당 락 파일과 대응되는 로그 파일(`.log`)을 안전하게 자동 일괄 삭제하도록 구현하여, 비정상 종료 등으로 발생한 잔여 락 파일들의 누적을 완전히 차단합니다.

### 5. 시뮬레이션 종료 시 결과 CSV 및 PNG 그래프 기본 자동 출력 (Auto-Export)
*   **기본 동작화**: [whts_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_engine.py)에서 시뮬레이션 목표 타임 도달 등으로 `simulation_result.pkl` 결과 파일을 쓰는 시점에, 자동으로 `SimulationDataExporter`를 import하여 실행하도록 코드를 연동했습니다.
*   시뮬레이션이 종료되면 별도 유틸을 직접 돌리지 않아도 결과 디렉토리 하위에 자동으로 `data` 폴더가 생성되고 파트별 변위/속도/가속도의 CSV 및 PNG 그래프 30여 개가 자동으로 일괄 파일 익스포트됩니다.

### 6. UI 메뉴 연결 및 Radioss 로깅 양식 개선
*   **결과 폴더 탐색기 열기 추가**: [whts_control_panel.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_control_panel.py)의 상단 `View` 메뉴 하위에 `Open Result Folder in Explorer` 메뉴를 연동하여 결과 데이터를 쉽게 찾을 수 있도록 윈도우 탐색기 구동 로직을 추가했습니다.
*   **로깅 중복/가공 제거**: `Run Engine` 시 비동기 `RadiossEngineWorker`의 stdout 출력이 `self.sim.log`를 타서 발생하던 로깅 포맷(RichHandler 타임스탬프 및 파일 정보 추가)의 중복 및 장황한 출력을 제거하고, `print()`를 통해 윈도우 원래의 stdout 형태로 한 번만 깔끔하게 찍히도록 시그널 연결을 수정했습니다.

### 7. 콘솔 가로 폭(120열) 및 ASCII Art 타이틀 지원
*   **콘솔 120열 세팅**: `configure_terminal_font` 내에 윈도우 API(`SetConsoleScreenBufferSize`, `SetConsoleWindowInfo`)를 적용하여 콘솔창이 기동될 때 가로 폭이 120열로 강제 피팅되어 시뮬레이션 로그가 줄바꿈 없이 깔끔하게 보이도록 개선했습니다.
*   **ASCII Art**: 프로그램 기동 시 터미널 화면을 클리어한 뒤 대형 `WHTOOLS` ASCII Art 문자 타일을 인쇄하도록 변경하여 시각적 직관성을 향상했습니다.

---

## 검증 결과
*   PyInstaller 빌드 태스크가 정상 종료되고 exe 실행 파일 패키지가 온전하게 빌드 완료되었습니다.
*   디버그 로그 노이즈 제거 및 접두사 없는 순수 stdout 라인 출력이 잘 적용된 것을 확인했습니다.
*   프로그램 재실행 시, 프로세스가 살아있지 않은 stale 락 파일들 및 대응 로그 파일이 `_acquire` 단계에서 완벽하게 자동 정리되는 것을 검증하였습니다.
*   시뮬레이션 목표 시점 도달 완료 시, `.pkl` 결과 저장뿐만 아니라 `data` 폴더 하위에 CSV와 PNG 결과 차트가 자동으로 잘 뽑혀서 파일 시스템에 저장되는 것을 완벽하게 확인했습니다.
*   Run Engine 시, 타임스탬프와 파일 라인이 덕지덕지 붙어 중복 출력되던 현상이 완전히 사라지고 깨끗한 원본 stdout이 터미널에 정상 인쇄됨을 확인했습니다.
*   프로그램 기동 시, 콘솔창이 120열로 정확하게 피팅되며 대형 `WHTOOLS` 로고 아트 타일이 멋지게 인쇄되는 것을 검증했습니다.
