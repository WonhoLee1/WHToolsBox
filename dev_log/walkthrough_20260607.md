# 작업 완료 보고서 (2026-06-07)

## 변경 및 해결 내역

### 1. LS-PrePost 실행 시 세션 파일 에러 수정
* **원인:** LS-PrePost가 실행될 때 작업 디렉터리(CWD)가 루트 폴더로 지정되어 있었고, 실행 파일 인자가 상대 경로(`.k` 파일)로 전달되면서 LS-PrePost 내부에서 임시 세션 파일을 이동(move)하려다가 경로 해석 실패 또는 쓰기 권한 충돌이 발생하였습니다.
* **조치:** 
  * [whts_control_panel.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_control_panel.py) 파일의 `_on_open_lsprepost` 내 `subprocess.Popen` 호출 부분을 수정했습니다.
  * 실행 대상 `.k` 파일의 경로를 절대 경로(`abs_k_file`)로 파싱하여 안전하게 전달합니다.
  * 자식 프로세스의 `cwd` 매개변수를 결과 폴더(`abs_k_file.parent`)로 지정하여, 세션 파일 생성 및 이동이 해당 결과 디렉터리 내에서 원활하게 처리되도록 보장했습니다.

### 2. Radioss Engine 파일에 H3D 출력 기능 포함
* **원인:** 해석 종료 후 ParaView나 외부 포스트프로세서 등에서 결과 분석을 용이하게 하도록, Altair 표준인 `.h3d` 포맷 결과를 함께 추출할 수 있는 수치 제어가 필요했습니다.
* **조치:**
  * [whts_radioss_builder.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_radioss_builder.py) 파일의 `_write_engine` 함수를 수정했습니다.
  * 엔진 덱(`_0001.rad`) 파일 출력 시 기존 `/ANIM/DT` 및 데이터 지정 카드들에 더하여, H3D 가시화 결과를 추출하는 설정 카드들을 주입하였습니다.

### 3. 해석 진행 중 실시간 ParaView 변환 가시화 기능 추가
* **원인:** 기존에는 해석이 완료되기 전이거나 이미 이전 시점의 `.vtkhdf`가 존재하면 실시간 결과 업데이트가 되지 않았으며, 해석기가 작성하고 있는 불완전한 애니메이션 파일을 무리하게 파싱하려다 시스템 충돌이 날 위험이 있었습니다.
* **조치:**
  * [whts_control_panel.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_control_panel.py)의 `_on_open_paraview` 함수를 보완했습니다.
  * 해석 백그라운드 스레드(`self._radioss_worker`)의 진행 유무를 파악하여 실행 중일 경우 기존 `.vtkhdf`를 무시하고 갱신하도록 구성했습니다.
  * Windows I/O 점유 특성에 따라 현재 OpenRadioss가 디스크에 작성 중인 임시/미완성 애니메이션 파일들은 변환 대상에서 안전하게 감지 및 스킵하고, 완료된 파일들만 엮어서 `.vtkhdf`를 생성하는 견고한 잠금(Lock) 확인 알고리즘을 도입했습니다.
  * 기존 `.vtkhdf`가 ParaView에서 가독/점유 중이어서 삭제가 되지 않는 경우, 경고 메시지를 노출하여 사용자 충돌을 방어했습니다.

### 4. Radioss H3D 엔진 출력 카드 문법 오류 수정
* **원인:** 기존에 `/ANIM/H3D/*` 형식의 비표준 카드가 주입되어 OpenRadioss 솔버 기동 시 문법 위반(Syntax Error)으로 해석이 즉각 종료되는 문제가 발생했습니다.
* **조치:**
  * [whts_radioss_builder.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_radioss_builder.py) 파일의 `_write_engine` 메소드를 수정했습니다.
  * 비표준인 `/ANIM/H3D/*` 카드를 OpenRadioss 표준 카드인 `/H3D/DT`, `/H3D/NODA/VEL`, `/H3D/NODA/DISP`, `/H3D/ELEM/EPSP`, `/H3D/ELEM/VONM`으로 정정하였습니다.

### 5. Radioss H3D 변위 카드 상세 규격 수정
* **원인:** 교정한 `/H3D/NODA/DISP` 카드 적용 시, 변위를 나타내는 절점 카드의 규격이 실제로는 `DISP`가 아닌 `DIS`였기 때문에 `NON-EXISTENT /H3D OPTION` 오류로 인해 해석이 비정상 종료되었습니다.
* **조치:**
  * [whts_radioss_builder.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_radioss_builder.py) 내 `_write_engine` 함수에 기입된 `/H3D/NODA/DISP`를 공식 명칭인 `/H3D/NODA/DIS`로 긴급 교체하여 해석 문법 정합성을 보장했습니다.

### 6. Radioss H3D 요소 출력 카드 명시적 세분화 및 호환성 고도화
* **원인:** 일반 엘리먼트 지시어(`/H3D/ELEM/...`) 대신 쉘 및 솔리드 요소의 공식 스펙을 참고하여, 컴파일러 및 엔진 버전별 파서 해석 오동작을 원천적으로 막고 완벽한 호환성을 제공해야 했습니다.
* **조치:**
  * [whts_radioss_builder.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_radioss_builder.py) 파일의 `_write_engine` 함수를 수정했습니다.
  * 기존 `/H3D/ELEM/EPSP` 및 `/H3D/ELEM/VONM` 카드를 제거하고, 명시적으로 쉘 결과인 `/H3D/SHELL/EPSP`, `/H3D/SHELL/VONM` 및 솔리드 결과인 `/H3D/SOLID/EPSP`, `/H3D/SOLID/VONM` 카드로 변경 주입했습니다.

### 7. 엔진 실행 시 터미널 중복 출력 로그 제거
* **원인:** 백그라운드 해석 시 프로세스 출력 수집기에서 `print` 함수를 통해 콘솔에 쏘는 것과, 수집기가 전달한 콜백이 `logger.info`를 실행하여 터미널 콘솔에 기록하는 것이 이중으로 동작하여 화면이 중복 프린트로 오염되었습니다.
* **조치:**
  * [whts_radioss_builder.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_radioss_builder.py) 내 `run` 함수를 수정하여, 메시지 수집 `callback`이 주입된 상황에는 `print` 출력을 우회하고 로거에만 일임하도록 분기를 구성했습니다.

## 테스트 및 검증 결과
* 코드 구조적 정합성을 확인했으며, 솔버 기동 시 문법 에러 없이 정상적으로 연산이 시작되고 H3D 출력이 올바르게 기입됨을 확인했습니다. 터미널에는 오직 한 줄씩만 깔끔한 로그 포맷팅으로 엔진 출력 메시지가 인출되어 중복 출력이 제거되었습니다.
