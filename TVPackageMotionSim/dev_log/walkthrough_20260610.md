# Walkthrough - Resolving Air Drag & Squeeze Force Data Export & Threading Issues

## Changes Made
1. **`run_drop_simulator/wht_export_sim_result.py`**:
   * Matplotlib 백엔드를 비 GUI 용인 `Agg`로 고정(`matplotlib.use('Agg')`)하여 백그라운드 스레드에서 차트 생성 시 스레드 충돌 및락 문제를 예방했습니다.
2. **`run_drop_simulator/whts_engine.py`**:
   * **글로벌 콜백 단일 인스턴스 패치**: 기존 스레드 ID 기반의 매칭 레지스트리 방식에서 벗어나, 프로세스 전역에서 단 하나의 콜백 인스턴스를 가지도록 `_global_mujoco_control_callback_instance` 변수를 도입하여 등록 및 처리하도록 수정했습니다. 이로 인해 다중 스레드 솔버나 QThread 환경에서도 스레드 ID 불일치 현상 없이 공기역학 콜백이 100% 호출됩니다.
   * **SyntaxError 방지**: `setup()` 함수의 시작 부분에 단 한 번만 `global _global_mujoco_control_callback_instance`를 선언하고, 함수 내부의 중복 `global` 선언들을 제거하여 파이썬 구문 오류를 수정했습니다.
   * **상세 디버그 로깅**: 결과 저장 및 익스포트 실패 시 터미널 로그에 `traceback.format_exc()`를 인쇄하여 구체적인 예외를 추적할 수 있도록 보완했습니다.

## Verification & Testing
* **시뮬레이션 가동 및 데이터 확인**: `run_drop_simulation_cases_v6.py`를 실행하여 2.0초의 물리 시뮬레이션을 완료한 결과, 스레드 크래시나 예외 메시지 없이 `data` 폴더가 정상 생성되고 `engineering.csv` 내에 `air_drag` 및 `air_squeeze`에 0이 아닌 유의미한 값들이 수집되었습니다:
* **air_drag**: `-1.778` ~ `40.578 N` 범위의 물리력 수집
  * **air_squeeze**: 지면 임계 근접 시점인 낙하 직전에 `1598.72 N` 피크 압착력 수집
* **차트 및 데이터 저장**: `results/rds-20260610_004556/data/` 경로 아래 `engineering-air_drag.png` 및 `engineering-air_squeeze.png` 차트 이미지 파일과 CSV가 완벽하게 저장되었습니다.

---

## 3. ASCII Art Spelling Correction & Console Output Optimization
### Changes Made
1. **`run_drop_simulation_cases_v6.py`**:
   * **아스키 아트 타이틀 수정**: 로고의 소문자 `s` (`WHTOOLs`)를 대문자 `S` (`WHTOOLS`)의 아스키 아트로 재생성하여 최종 타이틀을 **`WHTOOLS TV Drop Motion Simulator`**로 오타 없이 완전히 대문자화했습니다.
   * **콘솔 출력 동기화 (TeeLogger Lock 보강)**: `TeeLogger` 클래스의 `write` 함수 내에서 `self.stream.write(data)` 및 `self.stream.flush()`와 디스크 로그 파일 쓰기 영역 전체를 `self.lock` 스레드 락 블록 내부로 감쌌습니다.
     * 이 조치를 통해 멀티스레드나 비동기 스트림 환경에서 `rich.console.Console`이 표(Table)나 레포트 문자열을 조각내어 출력할 때 다른 스레드의 출력이 중간에 끼어들어 출력이 깨지거나 꼬이는 현상을 방지했습니다.

### Verification & Testing
* **아스키 아트 테스트**: 임시 검증 스크립트를 통해 `Consolas` 9pt 글꼴 및 120열 버퍼 크기 변경 하에 타이틀 로고가 깨짐 없이 깔끔하게 렌더링됨을 확인했습니다.
* **로그 격리 검증**: 단독 프로세스 실행 환경에서 `Stiffness Calculation Report` 및 `Inertia Correction` 물리 표 보고서가 비동기 로깅으로 인해 문자열이 조각나 섞이지 않고, 본래의 정렬된 직사각형 표 형태를 깨끗하게 유지하며 인쇄되는 것을 확인했습니다.

---

## 4. PyInstaller Executable Import Path Correction (`whts_ista_helper.py`)
### Changes Made
1. **`run_drop_simulator/whts_control_panel.py`**:
   * **상대 임포트 적용**: `_update_all()` 함수 내부에서 `ISTA6ASimulator`와 `IstaFaceMapper`를 가져올 때, 절대 경로 방식인 `from whts_ista_helper import ...` 대신 **패키지 상대 임포트 방식인 `from .whts_ista_helper import ...`**로 수정했습니다.
   * **원인 및 효과**: PyInstaller로 빌드된 단일 실행 파일(`.exe`) 환경에서는 `run_drop_simulator` 경로가 파이썬 기본 `sys.path` 에 자동으로 잡히지 않아 `ModuleNotFoundError`를 유발할 수 있습니다. 이번 상대 경로 수정을 통해 실행 파일 압축 해제 임시 디렉토리(`_MEIPASS`) 내에서 컨트롤 패널과 동일한 패키지 폴더 내의 `whts_ista_helper.py`를 오류 없이 즉각 로드할 수 있도록 패키지 호환성을 완성했습니다.

---

## 5. INI Configuration Editor Menu Integration (`external_tools_config.ini`)
### Changes Made
1. **`run_drop_simulator/whts_control_panel.py`**:
   * **View 메뉴 아이템 추가**: 상단 메뉴바의 `🔍 View` 메뉴 하단에 구분선(Separator)을 넣고 **`⚙️ Edit External Tools Config (INI)`** 액션을 추가했습니다.
   * **자율 경로 탐색 핸들러 연동**: 해당 메뉴 클릭 시 호출될 `_on_edit_external_tools_config` 함수를 작성했습니다. 이 함수는 실행 환경(CWD, 스크립트 실행 폴더, PyInstaller 배포 위치 등)에 맞추어 `external_tools_config.ini` 파일을 자율 탐색(Candiates search)한 뒤, OS 수준의 `startfile` API를 통해 **시스템 기본 텍스트 에디터(윈도 메모장 등)**로 열어 즉시 편집할 수 있게 연동합니다.

---

## 6. Batch RDS Multi-Threaded Parallel Execution Fix
### Changes Made
1. **`run_drop_simulator/whts_engine.py`**:
   * **모델 메모리 주소 기반 콜백 맵핑 (`_mujoco_model_registry`) 구현**:
     기존 `_mujoco_thread_registry` (스레드 ID 매핑)를 완전히 폐지하고, 각 시뮬레이션 인스턴스의 MjModel 파이썬 객체 고유 주소(`id(self.model)`)를 키값으로 삼아 개별 제어 콜백을 전역 딕셔너리(`_mujoco_model_registry`)에 관리하도록 리팩토링했습니다.
     * 이 방식을 통해 다중 스레드 솔버(`sim_nthread > 1`)를 켤 때 MuJoCo C++ 워커 스레드 ID가 파이썬 메인/QThread ID와 달라 매칭에 실패하는 구조적 한계를 완벽히 해결했습니다.
   * **전역 콜백 공유 안전화 및 클린업 변경**:
     개별 시뮬레이터 인스턴스의 `setup` 및 `_wrap_up` 과정에서 전역 제어 콜백 설정을 아예 해제해 버리던 무차별적인 `mujoco.set_mjcb_control(None)` 호출을 삭제했습니다. 대신, 전역 `mjcb_control`에 고정 바인딩된 상태를 유지하며 자신의 모델 객체 주소(`id(self.model)`)를 `_mujoco_model_registry`에서 추가 및 수거(pop)하도록 수정했습니다.
     * 이로 인해 다수의 Parallel workers가 동시에 실행되는 상황에서 한 스레드가 끝나서 `None`을 세팅할 때 다른 활성 스레드의 콜백이 돌연 초기화되어 **`py_mjcb_control is null`** 크래시를 발생시키는 동시성 버그를 완벽하게 예방했습니다.
2. **`run_drop_simulator/whts_control_panel.py`**:
   * **잔존 레거시 코드 제거**: `BatchRdsWorker`의 `run_one` 실행 블록이 끝날 때 구버전 `_mujoco_thread_registry`를 직접 임포트해 pop하려다 발생했던 `cannot import name '_mujoco_thread_registry'` 임포트 오류의 원인을 제거하고 정리했습니다.

---

## 7. UI Telemetry Callback & Usage Tracking Integration
### Changes Made
1. **`run_discrete_builder/whtb_config.py`**:
   * **기본 텔레메트리 덤프 함수 주입**: 기본 설정 사양(`_build_default_dict`)에 `telemetry_callback` 키를 신설하고 기본값으로 아무런 동작도 하지 않는 더미 람다 함수(`lambda event_name: None`)를 지정하여, 별도 등록이 없는 경우에도 시뮬레이션 및 UI 흐름에 오류 없이 무해하게 작동하도록 보장했습니다.
2. **`run_drop_simulator/whts_control_panel.py`**:
   * **자율 트래킹 시스템 구축 (`_setup_ui_telemetry`)**: 
     상위 윈도우 생성 완료 시점(`__init__`)에 `self.findChildren(QPushButton)` 및 `self.findChildren(QAction)`을 활용해 Control Center UI 내의 모든 버튼과 메뉴 아이템 객체들을 자율 탐색 및 수집하도록 구현했습니다.
   * **텔레메트리 연동 시그널 슬롯 연결**:
     탐색된 각 컴포넌트의 클릭/트리거 시그널에 `self._trigger_telemetry` 슬롯 함수를 다중 연결(Qt Multi-slot connectivity)하여, 기존 동작의 수정이나 왜곡 없이 버튼 클릭/메뉴 트리거 시 실시간으로 `telemetry_callback`을 호출하도록 연동을 마쳤습니다.
   * **순수 기능명 자동 정제 (`clean_name`)**:
     위젯 라벨에서 이모지(🆕, 💾, ▶ 등) 및 UI 데코레이션 문자를 정규식으로 자동 제거하는 헬퍼 함수를 구현했습니다. 이를 통해 호출 시 기존의 임의 명칭 대신 **정제된 순수 기능명(예: `New Model Setup`, `Save Config (JSON)`, `Play`, `Pause`, `Reset` 등)**이 콜백 함수의 인자값으로 완벽하게 넘어가도록 보정했습니다. (슬라이더 및 QSlider 컴포넌트는 요구사항에 맞게 연동 대상에서 배제)

---

## 8. Telemetry Callback Targets Filtering & Mapping
### Changes Made
1. **`run_drop_simulator/whts_control_panel.py`**:
   * **QAbstractButton 기반 확장 탐색**: 기존 `QPushButton`만 탐색하던 방식에서 `QAbstractButton`을 탐색하도록 변경하여, QToolButton 계열인 **`📋 Log Motion`** 등의 컴포넌트도 누락 없이 텔레메트리 추적 대상에 포함되도록 범위를 확장했습니다.
   * **특정 기능명 우선 필터링 (`ALLOWED_FEATURES`)**: 무차별적인 전체 로깅 대신 사용자가 요청한 **9가지 핵심 기능명**(`New Model Setup`, `Play`, `Monitor`, `Cam. Info.`, `Log Motion`, `Str. Analysis`, `Generate Model`, `Run Engine`, `ParaView`)만 우선하여 콜백을 호출하도록 필터를 적용했습니다.
   * **재생 상태 매핑 (`Pause -> Play`)**: `Play` 버튼을 누르는 도중 UI 텍스트가 토글 상태인 `Pause`로 변환된 경우에도, 텔레메트리 기록이 통계적으로 일관되게 남을 수 있도록 **`Pause` 이벤트를 `Play` 로 자동 변환하여 전달**하는 헬퍼 맵핑을 구현했습니다.

### Verification & Testing
* **테스트 검증 완료**: 업데이트된 테스트 스크립트([test_telemetry.py](file:///C:/Users/GOODMAN/.gemini/antigravity-ide/brain/fccb4c5e-dd8f-40ff-897a-90f8f92979e7/scratch/test_telemetry.py))를 통해 비대상 버튼 클릭 시 조용히 무시되는 것과, `Play`, `Pause`, `Monitor` 클릭 시 각각 `['Play', 'Play', 'Monitor']` 형태로 필터 및 매핑이 완벽하게 일치하여 텔레메트리에 남는 것을 성공적으로 확인 및 검증했습니다.
