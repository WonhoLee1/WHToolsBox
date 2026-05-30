# [Goal Description]

Reference Model 선택 시 누락된 물리 데이터(CoG, MoI, Component Masses)들을 `config` 로 완벽하게 동기화 및 주입하고, 누락된 데이터에 대해 예외 처리 및 물리적인 규격 계산을 수행하는 개선 작업을 수행합니다.

## User Review Required

> [!NOTE]
> 본 개선 사항은 `SelectTVModelDialog`의 테이블 선택 모델에서 데이터를 온전히 추출하도록 수정하고, `IstaSetupHelperDialog`에서 모델 선택 시 데이터의 결여 여부를 판별하여 다음과 같은 물리 보정 로직을 자동 수행합니다:
> 1. **CoG 누락 시:** `[0.0, 0.0, 0.0]` 으로 기본 설정하여 전달.
> 2. **MoI 누락 시:** `Guess Uniform MoI` 균질 관성 모멘트 공식을 적용하여 실시간 자동 계산 후 주입.
> 3. **Component Masses(Chassis, Cushion, Opencell) 누락 시:** 각 파트의 형상 체적(Volume)을 계산하고, 지정된 밀도(Cushion: `2e-11 kg/mm³`, Chassis: `1e-9 kg/mm³`, Opencell: `2e-9 kg/mm³`)를 적용하여 현실적인 질량으로 변환한 뒤 사용자에게 알림 다이얼로그를 팝업하고 `config`를 업데이트합니다.

## Open Questions

> [!TIP]
> 질량 계산 시 밀도 단위가 `kg/mm³` 기준이므로 부피를 `mm³` 단위로 변환(`volume_m³ * 1e9`)하여 연산합니다. 이 경우 75인치급 모델 기준 Cushion 질량은 약 `3.0 kg` 등 매우 현실적인 값이 도출됩니다.

---

## Proposed Changes

### TVPackageMotionSim Control Panel

#### [MODIFY] [whts_control_panel.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_control_panel.py)

1. `SelectTVModelDialog._on_apply` 수정:
   - 선택된 모델의 `cushion_m`, `chassis_m`, `opencell_m`, `cog`, `moi` 데이터를 테이블 위젯 아이템에서 정상적으로 추출하여 `selected_model` 딕셔너리에 반환하도록 수정합니다.
2. `IstaSetupHelperDialog._on_select_ref_model` 수정:
   - `selected_model` 딕셔너리로부터 질량, CoG, MoI 데이터를 온전히 파싱합니다.
   - 질량이 누락된 경우, Package 및 SET 치수 데이터를 기반으로 Cushion, Chassis, Opencell 부피(Volume)를 산출하고, 각각 밀도(`2e-11`, `1e-9`, `2e-9`)와 곱하여(mm³ 기반) 질량을 임의 계산합니다.
   - 질량 계산이 적용된 경우 사용자에게 알림 정보 창(`QMessageBox.information`)을 띄워 알립니다.
   - CoG가 누락된 경우 `[0.0, 0.0, 0.0]`을 주입합니다.
   - MoI가 누락된 경우 `Guess Uniform` 균질 관성 계산 공식을 사용하여 MoI를 자동 계산하여 주입합니다.
   - 이 모든 데이터가 `config`의 `components`, `chassis_cog`, `chassis_moi`, `components_balance` 에 빈틈없이 적용되도록 동기화합니다.

---

## Verification Plan

### Automated Tests
- 없음 (UI 및 물리 파라미터 통합 동기화 테스트)

### Manual Verification
- `whts_control_panel.py`를 실행하여 [Setup] 버튼 클릭 후 Reference Model을 선택합니다.
- CoG, MoI 또는 질량 정보가 누락된 임의의 모델을 선택하거나, 데이터 누락 가상 시나리오를 시뮬레이션하여 경고 창 팝업 및 config 정상 주입을 검증합니다.
