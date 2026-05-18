# 목표 시간(sim_duration) 조절용 SpinBox UI 추가 구현 계획서 (2026-05-18)

## 1. 개요 및 목표
현재 시뮬레이션의 목표 시간(`sim_duration`)은 `lbl_time`에 단순 텍스트로만 표시되고 있으며, UI 상에서 실시간으로 수정할 수 있는 방법이 없습니다. 
이를 개선하여 사용자가 GUI 상에서 목표 시간을 직접 수정할 수 있도록 `QDoubleSpinBox`를 도입합니다.
- **수정 대상 파일:** `c:\Users\GOODMAN\WHToolsBox\TVPackageMotionSim\run_drop_simulator\whts_control_panel.py`
- **핵심 기능:**
  - `sim_duration` 조절용 `QDoubleSpinBox` 추가
  - 0.5초 단위로 값 조정 가능 (`setSingleStep(0.5)`)
  - 키보드를 통한 직접 실수 입력 허용
  - 다른 동작(설정 불러오기 등)에 의해 `sim_duration`이 바뀔 때 실시간 동기화 지원

## 2. 변경 계획 세부사항

### 2.1 [MODIFY] [whts_control_panel.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_control_panel.py)

#### 2.1.1 UI 초기화 (`_init_ui` 메서드)
- 기존의 `self.lbl_time = QLabel("Time: 0.000 / 0.000 s")` 텍스트 레이블을 `Time: 0.000 s` 형태로 분할 표시하도록 변경합니다.
- `QDoubleSpinBox` 인 `self.spin_duration`을 생성합니다.
  - 범위: `0.1` ~ `100.0`
  - 단일 스텝: `0.5`
  - 소수점 자리수: `3` (정밀한 직접 실수 입력 지원)
  - 폰트: `Consolas, 10`
  - 초기값: `self.sim.config.get("sim_duration", 1.0)`
  - 시그널 바인딩: `valueChanged` 시그널을 `self._on_duration_changed`와 연결
- `status_layout` 내부에서 가로 레이아웃 `time_layout = QHBoxLayout()`을 생성하여 시간 레이블과 목표 설정 스핀 박스를 깔끔하게 정렬합니다:
  `Time: 0.000 s / Target: [ 1.000 ] s` 형태의 세련된 가로 배치를 구현합니다.
- `lbl_time`을 단독 레이아웃으로 뺏기 때문에, 기존의 `for lbl in [...]` 루프에서 `lbl_time`을 제외하고 나머지 상태 레이블들만 일괄 폰트 및 레이아웃 추가를 하도록 조절합니다.

#### 2.1.2 상태 업데이트 루프 (`_update_status` 메서드)
- 매 주기마다 불리는 `_update_status` 내에서 시간 정보 텍스트를 업데이트하는 부분을 간소화합니다:
  `self.lbl_time.setText(f"Time: {curr_time:.3f} s")`
- 사용자가 스핀 박스에 포커스를 주어 수동 입력하고 있는 중이 아닐 때는, 외부(예: Config Load)에서 바뀐 최신 `sim_duration`을 스핀 박스에 실시간 동기화합니다:
  ```python
  if not self.spin_duration.hasFocus():
      self.spin_duration.blockSignals(True)
      self.spin_duration.setValue(target_time)
      self.spin_duration.blockSignals(False)
  ```

#### 2.1.3 목표 시간 변경 콜백 함수 추가 (`_on_duration_changed` 메서드)
- 사용자가 스핀 박스의 값을 조작하거나 직접 실수를 입력하여 변경했을 때, 시뮬레이터 객체의 `self.sim.config["sim_duration"]`을 실시간으로 갱신하는 메서드를 추가합니다.
- `RULE[user_global]`에 따라 인자와 복합 설명을 담은 완벽한 Docstring을 포함시킵니다.

## 3. 검증 계획
1. **정적 검증:** Python 코드 구문 오류(Syntax Error) 및 Lint 에러 검사.
2. **동적 검증:** 시뮬레이터 GUI 제어반(`whts_control_panel.py`)을 실행하여:
   - 스핀 박스가 `Simulation Status` 그룹 내부의 `Time` 영역 우측에 잘 위치하는지 확인.
   - 마우스 휠 또는 스핀 박스 버튼 조작 시 `0.5` 단위로 증감하는지 확인.
   - 키보드로 `0.75` 등 임의의 실수를 입력했을 때 원활히 반영되는지 확인.
   - 시뮬레이션 동작 시 타겟 지점(`sim_duration`)에 도달했을 때 `Collection Complete` 상태로 정확히 전이되는지 테스트.
