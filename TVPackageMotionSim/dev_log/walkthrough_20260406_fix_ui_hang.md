# [Walkthrough] Simulation Headless Fix & V2 UI Alignment

이 가이드는 시뮬레이션이 종료 단계에서 멈추지 않고(Headless 보호), 사용자가 요청한 PySide6 기반 V2 UI가 정상적으로 호출되도록 하는 과정을 설명합니다.

## 📋 핵심 수정 사항

1. **Lazy UI Initialization**: `whts_engine.py`에서 `tk.Tk()`를 필요한 시점에만 생성하도록 변경했습니다. 이제 `--enable_UI=False` 모드에서는 어떠한 UI 라이브러리도 메모리에 로드되지 않습니다.
2. **Robust Execution Guard**: `_wrap_up()` 단계에서 `ctrl_open_ui` 플래그를 검사하여, 명시적인 UI 호출 명령이 없는 경우 즉시 프로세스를 종료하도록 강제했습니다.
3. **PySide6 V2 Migration**: `use_postprocess_v2` 옵션 활성화 시 사용자가 요구한 PySide6 기반의 신형 대시보드가 서브프로세스로 실행되도록 로직을 일원화했습니다.

## 🛠️ 실행 방법

기존에 실행 중인 시뮬레이션 프로세스가 있다면 터미널에서 `Ctrl+C`로 종료한 뒤 아래 명령어로 재시작하세요.

### 1. Headless 일괄 실행 (Hang 방지)
```powershell
python run_drop_simulation_cases_v4.py
```
이 방식은 모든 시뮬레이션을 순차적으로 실행한 뒤, UI를 띄우지 않고 깔끔하게 종료됩니다.

### 2. 시뮬레이션 후 V2 UI(PySide6) 자동 호출
`run_drop_simulation_cases_v4.py` 내부의 `test_run_case_2`는 이미 `use_postprocess_v2: True`로 설정되어 있습니다.
시뮬레이션 도중 또는 종료 후 UI를 보고 싶다면 시뮬레이션 뷰어에서 `K` 키를 누르거나, 스크립트 호출 시 `enable_UI=True`를 전달하세요.

## 🏁 마무리
이제 대량의 시뮬레이션을 수행하더라도 UI 블로킹으로 인해 프로세스가 멈추는 현상 없이 안정적으로 데이터를 수집할 수 있습니다.
추가적인 UI 기능 개선이나 물리 엔진 튜닝이 필요하시면 말씀해 주세요.
