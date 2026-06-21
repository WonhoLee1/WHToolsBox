# LS-PrePost 실행 시 세션 파일 생성 및 이동 오류 해결 계획 (2026-06-07)

## 문제 정의
LS-PrePost를 실행할 때 다음과 같은 경고 대화상자가 표시됩니다:
`session file can not be moved to results\rds-20260607_231059, user might not have write permission or file already opened`

이는 LS-PrePost가 실행되는 현재 작업 디렉터리(CWD)가 루트 디렉터리(`C:\Users\GOODMAN\WHToolsBox`)로 설정된 채, 결과 파일인 `.k` 파일의 경로를 상대 경로(`results\rds-20260607_231059\TVDrop_Radioss_LSDYNA.k`)로 받아 LS-PrePost 내부적으로 임시 세션 파일을 이동하려 시도하면서 발생하는 상대 경로 인식 및 파일 권한/잠금 문제입니다.

## 해결 방법
1. LS-PrePost를 실행할 때, 인자로 들어가는 `.k` 파일의 경로를 절대 경로(`resolve().absolute()`)로 변환하여 전달합니다.
2. `subprocess.Popen` 호출 시 `cwd` 매개변수를 사용하여 작업 디렉터리를 결과 파일이 존재하는 폴더(`results\rds-20260607_231059` 절대 경로)로 설정합니다.
3. 이 조치를 통해 LS-PrePost가 세션 파일을 해당 결과 폴더 안에서 직접 다루게 함으로써 경로 불일치 및 백업/이동 문제를 미연에 방지합니다.

## 변경 대상 파일
* [whts_control_panel.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_control_panel.py) - `_on_open_lsprepost` 메소드 수정

---

# Radioss Engine 파일에 H3D 출력 기능 포함 계획 (2026-06-07 추가)

## 요구 사항
해석 완료 후 애니메이션 결과로 `.h3d` 파일 포맷이 자동으로 출력되어 저장되도록 `_0001.rad` 엔진 템플릿에 설정을 포함합니다.

## 해결 방법
* `whts_radioss_builder.py`의 `_write_engine` 함수 내에서 `_0001.rad` 파일에 작성되는 카드 목록에 `/ANIM/H3D` 관련 카드들을 추가합니다.
* 추가할 카드:
  - `/ANIM/H3D/DT`: 출력 주기 및 시작 시간 설정 (기존 `/ANIM/DT`와 동일한 간격)
  - `/ANIM/H3D/VECT/VEL`: 속도 벡터 출력
  - `/ANIM/H3D/VECT/DISP`: 변위 벡터 출력
  - `/ANIM/H3D/ELEM/EPSP`: 등가소성변형률 출력
  - `/ANIM/H3D/ELEM/VONM`: Von Mises 응력 출력

## 변경 대상 파일
* [whts_radioss_builder.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_radioss_builder.py) - `_write_engine` 메소드 수정

---

# 해석 중 실시간 ParaView 가시화 기능 보완 계획 (2026-06-07 추가)

## 문제 정의 및 요구사항
시뮬레이션 해석 엔진(`_radioss_worker`)이 동작하고 있는 도중에도, 사용자가 GUI에서 "ParaView" 버튼을 클릭하면 현재까지 쓰여진 애니메이션 결과 파일(`A*`)들을 실시간 수집하여 `.vtkhdf` 파일로 즉시 변환한 뒤 가시화할 수 있어야 합니다.

## 직면하는 기술적 병목
1. **기존 파일 감지 무시 필요:** 기존에 `.vtkhdf` 파일이 이미 있어도 해석 도중에는 계속 최신 파일이 생기므로, 기존 파일을 강제로 삭제하고 새로운 세트의 파일들로 덮어씌워 가시화해야 합니다.
2. **파일 점유(Lock) 문제 해결:** OpenRadioss가 활발히 디스크에 작성 중인 가장 최근 애니메이션 파일은 다른 프로세스에서 동시에 읽으려 시도할 시 `PermissionError` 또는 손상된 파일 예외를 유발할 수 있습니다.
3. **읽기 방해 금지:** ParaView가 가시화 중인 `.vtkhdf` 파일을 열어두고 있다면 덮어쓰기(삭제) 시 실패하므로 이에 대한 예외 처리가 필요합니다.

## 해결 방법
1. `_radioss_worker.isRunning()` 여부로 실시간 엔진 구동을 판단합니다.
2. 엔진 구동 중 혹은 `.vtkhdf` 파일 미존재 시 강제 변환에 돌입합니다.
3. 변환 대상 `anim_files`에 대해 하나씩 `open(f, 'rb')`를 시도하여 쓰기 잠금(Lock) 상태가 걸리지 않은 안전한 파일들만 `valid_anim_files` 리스트에 담아 변환을 수행합니다.
4. 기존 `.vtkhdf`를 지울 때 발생할 수 있는 에러(ParaView 기독 점유 등)에 대한 예외 처리를 추가하여 사용자에게 이미 켜진 ParaView를 닫고 재진행하도록 안내합니다.

## 변경 대상 파일
* [whts_control_panel.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_control_panel.py) - `_on_open_paraview` 메소드 수정

---

# Radioss H3D 엔진 출력 카드 문법 오류 수정 계획 (2026-06-07 추가)

## 문제 정의
`/ANIM/H3D/*` 형식의 카드를 엔진 파일(`_0001.rad`)에 작성하여 시뮬레이션을 수행할 경우, OpenRadioss 솔버가 `** ERROR IN SOLVER INPUT DECK CARD: ANIM` 및 `/ANIM/H3D/VECT/VEL` 형식 오류로 비정상 종료(Error Termination)됩니다.

## 원인 분석
* OpenRadioss에서 H3D 가시화 결과를 직접 출력하기 위한 엔진 카드는 `/ANIM/H3D/` 접두사가 아닌 **`/H3D/`** 접두사로 시작해야 합니다.
* 또한 세부 결과 요청 카드들의 공식 문법은 다음과 같습니다:
  - `/H3D/DT` (출력 속도 주기 지정)
  - `/H3D/NODA/VEL` (속도 필드)
  - `/H3D/NODA/DISP` (변위 필드)
  - `/H3D/ELEM/EPSP` (소성변형률)
  - `/H3D/ELEM/VONM` (Von Mises 응력)

## 해결 방법
* `whts_radioss_builder.py` 내의 `_write_engine` 함수에서 잘못 지정된 `/ANIM/H3D/*` 형식의 문자열들을 올바른 `/H3D/*` 형식 문자열로 교체합니다.

## 변경 대상 파일
* [whts_radioss_builder.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_radioss_builder.py) - `_write_engine` 메소드 수정

---

# Radioss H3D 변위(Displacement) 카드 상세 규격 수정 계획 (2026-06-07 추가)

## 문제 정의
교정한 `/H3D/NODA/DISP` 카드를 사용 시, OpenRadioss 솔버가 `** ERROR: NON-EXISTENT /H3D OPTION: /H3D/NODA                /DISP` 오류를 발생시키며 비정상 종료되는 현상이 발생했습니다.

## 원인 분석
* OpenRadioss에서 H3D 결과 출력 시 변위(Displacement)를 나타내는 절점 카드의 정확한 명칭은 `/H3D/NODA/DISP`가 아니라 **`/H3D/NODA/DIS`**입니다.
* 슬래시 파서가 `/H3D/NODA/DIS`는 인식할 수 있으나, 뒤에 `P`가 더 붙은 `DISP`는 인식하지 못하고 잘못된 필드로 쪼개어 공백을 채워버림으로써 `NON-EXISTENT /H3D OPTION` 오류가 발생한 것입니다. (한편 속도 `/H3D/NODA/VEL`은 올바른 카드이므로 정상 처리되었습니다.)

## 해결 방법
* `whts_radioss_builder.py`의 `_write_engine` 함수에 명시된 `/H3D/NODA/DISP` 문자열을 공식 규격인 `/H3D/NODA/DIS`로 교체합니다.

## 변경 대상 파일
* [whts_radioss_builder.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_radioss_builder.py) - `_write_engine` 메소드 수정

---

# Radioss H3D 요소 출력 카드 명시적 세분화 계획 (2026-06-07 추가)

## 요구 사항 및 상세
일반적인 엘리먼트 출력 카드(`/H3D/ELEM/...`) 대신 쉘 요소와 솔리드 요소에 대한 공식 도움말 규정(`/H3D/SHELL` 및 `/H3D/SOLID`)을 참고하여, 컴파일러 및 버전에 따른 호환성을 높이고 해석 안정성을 확보하기 위해 명시적인 카드 선언으로 고도화합니다.

## 해결 방법
* `whts_radioss_builder.py` 내 `_write_engine`에서 `/H3D/ELEM/EPSP` 및 `/H3D/ELEM/VONM` 카드를 삭제하고 대신 아래 카드로 명시적 쉘/솔리드 셋을 구분해 주입합니다:
  - `/H3D/SHELL/EPSP`
  - `/H3D/SHELL/VONM`
  - `/H3D/SOLID/EPSP`
  - `/H3D/SOLID/VONM`

## 변경 대상 파일
* [whts_radioss_builder.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_radioss_builder.py) - `_write_engine` 메소드 수정

---

# 엔진 실행 시 터미널 중복 출력 로그 제거 계획 (2026-06-08 추가)

## 문제 정의
OpenRadioss 솔버 구동 시, 터미널 로그 화면에 동일한 시뮬레이션 진척도 출력(`NC=... T=... ELAPSED TIME=...`)이 표준 `print` 구문과 로깅 핸들러(`whts_engine.py:342`)를 통해 이중으로 중복 출력되어 화면을 어지럽히고 디버깅에 방해가 됩니다.

## 원인 분석
* [whts_radioss_builder.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_radioss_builder.py) 내 `run` 함수에서 `subprocess.Popen`으로 캡처한 출력을 `print(f"[Radioss] {line}")`로 화면에 바로 뿌리는 동시에, GUI 로그 창 연동을 위한 `callback(f"[Radioss] {line}")`을 수행하고 있습니다.
* 이 `callback`은 결국 `self.sim.log(...)`를 실행하여 로거에 로그를 찍는데, 로거 핸들러가 터미널 콘솔에도 로그를 쏘도록 설정되어 있어 결과적으로 화면에 동일 정보가 두 번 노출되는 것입니다.

## 해결 방법
* `whts_radioss_builder.py`의 `run` 함수 내부에서, `callback`이 존재할 때는 `print` 구문을 생략하고 콜백으로만 메시지를 넘겨 로거가 일괄 터미널/파일 로그를 쓰도록 하고, `callback`이 지정되지 않았을 때(단독 스크립트 실행 등)에만 표준 `print`를 하도록 조건 분기를 탑재합니다.

## 변경 대상 파일
* [whts_radioss_builder.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_radioss_builder.py) - `run` 메소드 수정
