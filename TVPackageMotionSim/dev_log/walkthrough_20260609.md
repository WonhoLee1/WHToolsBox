# Walkthrough - OpenRadioss Integration & PyInstaller Loop Fix (2026-06-09)

이 가이드에서는 PyInstaller 빌드 파일(`.exe`) 환경에서 Radioss Run 수행 시 발생하던 무한 재귀 실행 오류를 해결하기 위해 진행한 코드 변경점과 빌드 설정 개선 사항을 요약합니다.

## 변경 사항 요약

### 1. 스크립트 로컬화 (Migration)
*   **복사 대상**: 외부의 `openradioss_gui` 디렉토리에 있었던 핵심 배치 구동 스크립트들을 프로젝트 내부로 이관하였습니다.
    *   [runopenradioss.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/runopenradioss.py) -> [NEW]
    *   [inp2rad.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/inp2rad.py) -> [NEW]

### 2. 코드 개선 (In-Process Call)
*   **오류 원인 해소**:
    *   기존에는 `whts_radioss_builder.py`에서 `sys.executable`을 사용하여 임시 래퍼 파이썬 프로세스를 호출했습니다. PyInstaller 빌드 환경에서는 `sys.executable`이 자기 자신(`.exe`)을 가리켜 무한 프로세스 생성 루프가 발생하는 버그가 있었습니다.
*   **수정 사항**:
    *   [whts_radioss_builder.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_radioss_builder.py)에서 외부 프로세스 호출 구조를 걷어내고, 복사해 온 로컬 `runopenradioss` 모듈의 `RunOpenRadioss` 클래스를 **인프로세스(현재 프로세스 내부)에서 직접 import하여 실행**하도록 개선하였습니다.
    *   출력 스트림(`sys.stdout`)의 리디렉션을 위한 버퍼형 `CallbackStream`을 정의하여, 직접 호출 환경에서도 UI에 실시간 빌드/해석 로그가 정상 반영되도록 인터페이스를 설계했습니다.
    *   `runopenradioss.py` 내부의 `tkinter` 및 `inp2rad` 임포트 오류를 방어하여 무두방(Headless) CLI 실행 환경 호환성을 높였습니다.

### 3. PyInstaller 빌드 설정 수정
*   [drop_simulator_v6.spec](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/drop_simulator_v6.spec) 파일의 `hidden_imports`에 마이그레이션된 `run_drop_simulator.runopenradioss`와 `run_drop_simulator.inp2rad` 모듈을 명시적으로 수집 선언하여 빌드 누락이 없도록 개선했습니다.
*   PowerShell 세션 전파 문제로 인한 빌드 에러를 방지하기 위해 [build_drop_simulator_v6.ps1](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/build_drop_simulator_v6.ps1) 스크립트에서 PyInstaller 실행 시 명시적으로 가상환경을 타겟팅(`conda run -n vdmc pyinstaller`)하도록 구동 방식을 변경했습니다.

---

## 검증 결과
*   `conda run -n vdmc python -c "import run_drop_simulator.runopenradioss; print('Import OK!')"` 테스트를 통해 모듈 임포트 결함이 없음을 확인했습니다.
*   에이전트 단의 백그라운드 빌드는 사용자의 중단 요청에 따라 모두 강제 종료되었으며, 사용자가 직접 수동 빌드 및 실행을 원활하게 진행할 수 있도록 준비가 완료되었습니다.
