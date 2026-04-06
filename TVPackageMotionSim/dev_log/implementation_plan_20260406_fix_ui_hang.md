# [Fix] Simulation UI Hang and KeyboardInterrupt (V5.4.2)

Headless 모드에서 원치 않는 Tkinter 메인 루프 진입으로 인한 세션 행(Hang) 현상을 해결하고, v2 Dashboard의 PySide6 기반 연동을 강화합니다.

## User Review Required

> [!IMPORTANT]
> 이번 수정은 `simulate(enable_UI=False)` 시 **Tkinter 리소스를 전혀 생성하지 않도록** 설계를 변경합니다. 따라서 SSH나 CLI 환경에서의 안정성이 극대화됩니다.

## Proposed Changes

### [Component] run_drop_simulator

#### [MODIFY] [whts_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_engine.py)
- **Lazy UI Initialization**: `tk.Tk()` 생성을 `simulate()` 초기 단계에서 제거하고, 실제 UI가 필요한 시점(`ctrl_open_ui`가 True이거나 `ConfigEditor` 호출 시)에만 생성하도록 변경합니다.
- **Robust Guard in `_wrap_up`**: `ctrl_open_ui` 플래그를 최우선으로 검사하여, 명시적인 UI 요청이 없는 경우 `mainloop()`에 절대 진입하지 않도록 강제합니다.
- **V2 UI Alignment**: `use_postprocess_v2`가 활성화된 경우, Tkinter 대신 PySide6 기반의 `whts_postprocess_ui_v2.py`가 우선 실행되도록 로직을 정돈합니다.
- **Resource Cleanup**: 시뮬레이션 종료 시 `tk_root`가 존재할 때만 `destroy()`를 호출하도록 안정화합니다.

### [Component] run_drop_simulation_cases

#### [MODIFY] [run_drop_simulation_cases_v4.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulation_cases_v4.py)
- **Uncommenting Main Execution**: 사용자의 요구사항에 맞춰 `test_run_case_1` 혹은 `test_run_case_2`를 정상 실행 가능하도록 주석을 해제합니다.
- **Defaulting to Headless**: 기본적으로 `enable_UI=False`를 유지하여 대량 케이스 실행 시 중단을 방지합니다.

## Verification Plan

### Automated Tests
1. `python run_drop_simulation_cases_v4.py` 실행:
   - UI 없이 시뮬레이션이 끝까지 실행되고 결과(`.pkl`)가 생성되는지 확인.
   - 프로세스가 중단(Hang)되지 않고 터미널로 복귀하는지 확인.
2. `enable_UI=True`로 변경 후 실행:
   - 시뮬레이션 종료 후 정상적으로 UI가 팝업되는지 확인.
