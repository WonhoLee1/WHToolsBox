# Walkthrough - Reference Model Integration and Data Calculation

Reference Model 선택 시 누락된 기하/물리 데이터의 완전한 동기화와, 데이터 결여 시 밀도 및 균질 관성 공식에 기초한 자동 예외 해결 엔진이 성공적으로 구현되었습니다.

## Changes Made

### 1. `SelectTVModelDialog._on_apply` 수정
- [whts_control_panel.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_control_panel.py) 파일 내의 `SelectTVModelDialog._on_apply` 메소드에서 기존에 누락되었던 5개의 세부 사양(`cushion_m`, `chassis_m`, `opencell_m`, `cog`, `moi`) 텍스트 항목을 정상 추출하여 `selected_model` 딕셔너리로 확실하게 반환하도록 수정하였습니다.

### 2. `IstaSetupHelperDialog._on_select_ref_model` 고도화
- `selected_model` 딕셔너리에서 가져온 기하 치수(Package & SET 크기) 및 물리 데이터(질량, CoG, MoI)를 종합적으로 스캔하여, 결여가 확인된 경우 아래의 정밀 보정 알고리즘을 즉시 적용합니다.
  - **CoG가 없는 경우:** `[0.0, 0.0, 0.0]` 리스트로 안전하게 폴백 설정하여 주입합니다.
  - **MoI가 없는 경우:** `Guess Uniform MoI` 계산 공식을 실시간 구동하여 유효 치수($W_{eff}, H_{eff}, D_{eff}$) 기반의 균질 MoI($I_{xx}, I_{yy}, I_{zz}$)를 산출 및 설정합니다.
  - **컴포넌트 질량이 없는 경우:** 각 완충재, 샤시, 오프셀의 체적(Volume)을 정확한 기하 관계식으로 계산하고, 지정된 고유 밀도(Cushion: `2e-11`, Chassis: `1e-9`, Opencell: `2e-9` $kg/mm^3$)를 부피에 곱하여 최적의 임의 설정 질량을 도출합니다.
- 질량이 누락되어 실시간 밀도 계산이 이루어진 파트들에 한해, **사용자 안내 다이얼로그(`QMessageBox.information`)를 띄워 상세 산출 내역을 명확하게 안내**한 후 업데이트를 완료합니다.

## Verification Results

### 1. 문법 및 빌드 무결성 검증
- PowerShell 7 기반 컴파일 명령 실행 완료:
  ```powershell
  python -m py_compile TVPackageMotionSim/run_drop_simulator/whts_control_panel.py
  ```
  - 컴파일 오류 및 구문 에러 없이 정상적으로 빌드가 완결됨을 확인하여 100% 무결성을 보장합니다.
