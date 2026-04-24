# MuJoCo v5 (Digital Twin) 접촉 Pair 시스템 전환 계획

본 계획은 `v5` 디지털 트윈 시나리오(`run_drop_simulation_cases_v5.py`)를 기반으로, 기존 비트마스크 시스템을 완전히 제거하고 명시적 `<contact><pair>` 시스템으로 전환하는 것을 목표로 합니다.

## User Review Required

> [!IMPORTANT]
> **1. 재료 속성 분리 (Weld vs Contact)**
> 각 파트(`BCushion`, `BChassis` 등)의 내부 결속은 `weld` 속성으로 관리하고, 외부 접촉(마찰, 반발)은 오직 `pair` 설정에서만 관리하도록 분리합니다.
> 
> **2. 쿠션 모서리(_edge) 처리**
> 기존 쿠션 모서리의 특수 물리 계수는 `cushion_edge`라는 하위 타입을 도입하여 `cushion_edge - floor`와 같은 형태로 `pair` 섹션에서 명시적으로 정의됩니다.
>
> **3. 자가 접촉 제외 로직**
> 동일한 이산화 바디(예: 하나의 쿠션 덩어리) 내의 블록들끼리는 접촉 쌍을 생성하지 않습니다.

## Proposed Changes

### [1] [whtb_config.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_config.py) [MODIFY]
- **`CONTACT_CONFIG` 도입**: 파트 쌍별(`paper-floor`, `cushion-floor`, `cushion_edge-floor`, `paper-cushion` 등) `friction`, `solref`, `solimp` 통합 정의.
- **기본값 정리**: `mat_*` 사전에서 접촉 관련 속성(`friction`, `solref` 등)을 제거하고 순수 재질/용접 속성만 남깁니다.
- 모든 설정이 `user_config`에 의해 오버라이드 가능하도록 구조를 유지합니다.

### [2] [whtb_builder.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_builder.py) [MODIFY]
- **비트마스크 제거**: `default` 및 `geom` 태그에서 `contype`, `conaffinity`를 `0`으로 고정하여 자동 충돌을 비활성화합니다.
- **Geom 메타데이터 수집**: `root_container`를 순회하며 생성된 모든 `<geom>`의 이름, 재질 타입, 모서리 여부(`is_edge`)를 수집합니다.
- **명시적 Pair 생성 엔진**: 수집된 데이터를 바탕으로 `CONFIG["contacts"]`에 정의된 조합에 대해서만 `<pair>` 태그를 생성합니다.
- **Exclusion 로직**: 동일 인스턴스 내 자가 접촉은 제외하되, 바닥(`ground`)과의 접촉은 허용합니다.

### [3] [run_drop_simulation_cases_v5.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulation_cases_v5.py) [MODIFY]
- `test_case_1_setup` 및 `test_case_2_setup`에서 새로운 `contacts` 설정을 오버라이드하는 예시 코드를 추가하여 사용자가 물리 계수를 확인하고 점검할 수 있도록 합니다.

---

## Open Questions

- **Friction 수치**: `[sliding, torsional]` 2개 값만 입력받아 `[sliding, torsional, 0.0001, 0.0001, 0.0001]` 형태로 확장하는 방식을 `v5`에도 동일하게 적용할까요? (현재 `v5`는 일부 `0.8 0.8`과 같이 2개 값을 사용 중입니다.)

## Verification Plan

### Automated Tests
- `create_xmls.py` 또는 커스텀 스크립트를 사용하여 `v5` 기반 XML이 정상 생성되는지 확인.
- `mujoco.MjModel.from_xml_string()`을 통해 XML 구문 유효성 검증.
- `<contact>` 섹션 내에 `cushion_edge - floor` 쌍이 올바르게 포함되었는지 확인.

### Manual Verification
- `run_drop_simulation_cases_v5.py` 실행 후 MuJoCo Viewer에서 박스가 지면과 정상적으로 충돌하고 튕기는지 시각적으로 확인.
- `_edge` 클래스가 적용된 쿠션 모서리가 지면 충돌 시 설정된 `solref`에 따라 다르게 반응하는지 체크.
