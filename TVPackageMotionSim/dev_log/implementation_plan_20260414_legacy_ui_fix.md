# Implementation Plan - Legacy UI (Tkinter) Ghost Window Fix

Legacy UI (`whts_postprocess_ui.py`) 사용 시 발생하는 빈 `tk` 창(Ghost Window) 문제를 해결하고 초기화 로직을 정비합니다.

## User Review Required

> [!IMPORTANT]
> - `PostProcessingUI` 클래스의 생성자(`__init__`) 시그니처가 변경됩니다. (`master` 인자 추가)
> - 엔진(`whts_engine.py`)에서 UI를 호출하는 방식이 명시적으로 `tk_root`를 넘겨주는 방식으로 변경됩니다.

## Proposed Changes

### [Component] Post-Processing UI (`whts_postprocess_ui.py`)

#### [MODIFY] [whts_postprocess_ui.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_postprocess_ui.py)
- `__init__(self, parent_sim, master=None)`으로 변경하고 `super().__init__(master)`를 호출합니다.
- `on_simulation_complete` 메서드 등에서 초기화 시점의 `withdraw` 상태를 명확히 관리합니다.

### [Component] Simulation Engine (`whts_engine.py`)

#### [MODIFY] [whts_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_engine.py)
- `_wrap_up` 메서드에서 `PostProcessingUI(self, master=self.tk_root)`와 같이 명시적으로 마스터 윈도우를 전달합니다.
- `tk_root` 프로퍼티에서 `withdraw()`가 확실히 실행되도록 보장합니다.

## Open Questions
- Legacy UI에서 `Matplotlib` 백엔드를 `TkAgg`로 고정하시겠습니까? (현재 `QtAgg` 시도 후 실패 시 `TkAgg` 폴백 로직이 들어가 있어 환경에 따라 비일관적일 수 있습니다.)

## Verification Plan

### Manual Verification
- `run_drop_simulation_cases_v6.py`에서 `use_postprocess_ui = True`로 설정 후 시뮬레이션 실행.
- 시뮬레이션 종료 시 빈 창 없이 `WHTOOLS Post-Processing Explorer v4` 창만 정상적으로 뜨는지 확인.
