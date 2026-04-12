# Implementation Plan - Fix Weld Constraint Error in whtb_builder.py

`chassis_use_weld=False` (또는 `use_internal_weld=False`) 설정 시, 모든 블록이 하나의 `<body>` 내에 담기게 되지만, 보조 질량을 용접하는 로직은 여전히 개별 블록 바디(`b_bchassis_i_j_k`)를 참조하고 있습니다. 이를 현재 설정에 맞게 동적으로 바디 이름을 결정하도록 수정합니다.

## User Review Required

> [!IMPORTANT]
> 이 변경은 `chassis_use_weld` 옵션이 `False`일 때 보조 질량(Auxiliary Mass)을 올바른 바디(부품 전체 바디)에 용접하도록 합니다. 
> 만약 보조 질량을 특정 블록의 로컬 좌표계에 더 정확히 구속하고 싶다면 `chassis_use_weld=True`를 사용하는 것이 권장되지만, `False` 모드에서도 시스템이 죽지 않도록 부품 전체 바디에 용접하는 방식을 적용합니다.

## Proposed Changes

### [run_discrete_builder]

#### [MODIFY] [whtb_builder.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_builder.py)

- 보조 질량 용접 로직(라인 241-242 근처)을 수정합니다.
- `b_chassis.use_internal_weld` 값에 따라 `body2` 이름을 다음과 같이 결정합니다:
    - `True`인 경우: `b_{b_chassis.name.lower()}_{ci}_{cj}_{ck}` (기세 방식)
    - `False`인 경우: `b_chassis.name` (부품 전체 바디)
- `b_aux_mass`에 대해서도 동일한 논리(혹시 모르니)를 적용하여 `body1` 이름을 결정합니다.

## Verification Plan

### Automated Tests
- `run_drop_simulation_cases_v5.py`의 `test_case_2_setup`을 다시 실행하여 `ValueError: unknown element` 에러가 사라지는지 확인합니다.
- 생성된 `simulation_model.xml` 파일을 열어 `<equality>` 섹션의 `<weld>` 태그들이 올바른 바디들을 참조하고 있는지 확인합니다.

### Manual Verification
- 시뮬레이션이 시작되고 보조 질량을 포함한 제품 어셈블리가 정상적으로 낙하하는지 MuJoCo Viewer 또는 로그를 통해 확인합니다.
