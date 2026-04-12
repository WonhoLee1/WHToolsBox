# [WHTOOLS] Multi-Part Cushion Splitting Implementation Plan

본 계획서는 기존의 단일 덩어리 거대 쿠션(`BCushion`)을 내부 제품(TV Assembly)의 구성 요소인 OpenCell, Cohesive Tape, Chassis 등의 Z-레이어에 맞춰 물리적으로 분할하는 기능을 구현하기 위한 설계도입니다.

## User Review Required

> [!IMPORTANT]
> **분할 기준 및 자동화**: 내부 부품의 두께(`opencell_d`, `chassis_d` 등) 변화에 따라 쿠션의 분할 위치가 자동으로 동기화되도록 구현할 예정입니다. 만약 특정 레이어에서 쿠션이 필요 없는 경우(예: Tape 레이어 옆면) 이를 제거할 수 있는 옵션도 포함할까요?

> [!NOTE]
> **강체 거동 vs 변형 거동**: 분할된 쿠션 파트들은 각각 독립된 `BaseDiscreteBody`로 생성되므로, 필요한 경우 개별적으로 `use_weld=False`를 설정하여 완전 강체(Rigid Body)로 취급함으로써 연산 속도를 확보할 수 있습니다.

## Proposed Changes

### [Discrete Builder] (run_discrete_builder/)

내부 부품의 Z-범위를 계산하고, 이를 기반으로 여러 개의 `BCushion` 인스턴스를 생성하도록 빌더 로직을 확장합니다.

#### [MODIFY] [whtb_builder.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_builder.py)
- `create_model` 함수 내에서 `include_cushion` 로직을 리팩토링합니다.
- `assy_group` 내부 부품들의 `local_offset`과 `depth` 정보를 취합하여 `split_planes_z` 배열을 생성합니다.
- 루프를 돌며 각 영역에 해당하는 `BCushion_Front`, `BCushion_Mid`, `BCushion_Rear` 등의 객체를 생성하고 `root_container`에 추가합니다.
- 분할된 쿠션들 사이에 `weld` 제약 조건을 자동으로 생성하는 로직을 추가합니다.

#### [MODIFY] [whtb_config.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_config.py)
- `get_default_config`에 `cushion_split_mode` (bool) 옵션을 추가합니다.
- 각 분할된 쿠션 파트별로 질량(`mass`)을 어떻게 분배할지(체적 비례 또는 명시적 할당)에 대한 설정을 추가합니다.

#### [MODIFY] [whtb_models.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_models.py)
- `BCushion` 클래스의 `is_cavity` 로직이 자신이 담당하는 Z-영역 외의 블록은 자동으로 제외하거나, 생성 시점에 `depth`와 오프셋을 조절하여 호환성을 유지하도록 수정합니다.

---

## Verification Plan

### Automated Tests
- `run_discrete_builder/whtb_builder.py` 단독 실행을 통해 분할된 쿠션이 포함된 XML이 정상 생성되는지 확인합니다.
- MuJoCo Viewer에서 각 쿠션 파트가 노란색/흰색 등으로 구분되어 가시화되는지 확인합니다.

### Manual Verification
- `test_case_1_setup` (Corner Drop) 시뮬레이션을 실행하여, 쿠션이 분할된 상태에서도 물리적 연속성이 유지(Weld 작동 여부)되는지 점검합니다.
- 분할된 면(Interface)에서 비정상적인 침투(Penetration)나 떨림 현상이 발생하는지 관찰합니다.

## Open Questions

1. **파트 네이밍 규칙**: `BCushion_Part1_OpenCell`, `BCushion_Part2_Chassis` 와 같이 내부 파트 이름을 추종하는 네이밍을 선호하시나요?
2. **접합부 물성**: 분할된 쿠션 사이의 용접(`weld`) 강도는 쿠션 내부의 `weld` 강도와 동일하게 설정할까요, 아니면 별도의 `inter_cushion_weld` 클래스를 정의할까요?
