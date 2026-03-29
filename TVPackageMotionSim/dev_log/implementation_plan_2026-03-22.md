# MuJoCo Weld 및 Contact 파라미터 최적화 계획 (2026-03-22)

MuJoCo 시뮬레이션의 물리적 정확도를 높이기 위해 `weld`와 `contact` 파라미터를 분리하고, 부품별 계층 관리 구조를 도입하며, 쿠션 모서리 접촉 특성을 차별화합니다.

## User Review Required

> [!IMPORTANT]
> - `weld` 파라미터가 MuJoCo의 `<default>` 클래스 기반으로 통합 관리됩니다. XML 내 수천 개의 태그가 간소화되어 가독성과 수정 편의성이 대폭 향상됩니다.
> - 쿠션 부품의 모서리(Edge/Corner) 블록은 지면과의 접촉 시 일반 블록보다 강화된(또는 부드러운) 별도의 `solref`, `solimp` 값을 적용받습니다.

## Proposed Changes

### [Discrete Builder] (run_discrete_builder.py)

#### [MODIFY] [run_discrete_builder.py](file:///c:/Users/GOODMAN/WHToolsBox/test_box_mujoco/run_discrete_builder.py)
- `get_default_config`: `weld`용 `solref`/`solimp`와 `contact`용 파라미터(특히 지면 접촉용)를 분리하여 정의.
- `BaseDiscreteBody.get_weld_xml_strings`: 내부 Weld 생성 시 `class="weld_부품명"`을 사용하도록 변경.
- `create_model`: 
    - `<default>` 섹션에 각 부품별 `weld`전용 클래스와 **타품종 간 연결용(`weld_bopencellcohesive`)**, **보조 질량용(`weld_aux`)** 클래스를 정의하여 파라미터 집중 관리.
    - 부품 간 및 보조 질량 Weld 생성 로직에서 하드코딩된 값을 제거하고 클래스 참조 방식으로 통일.
- `BCushion.is_edge_block`: 블록 인덱스를 분석하여 모서리/코너 여부를 판별하는 로직 추가.
- `BaseDiscreteBody.get_worldbody_xml_strings`: 모서리 블록인 경우 `cush_edge_solref` 등을 적용하도록 수정.

### [Simulation Runner] (run_drop_simulation.py)

#### [MODIFY] [run_drop_simulation.py](file:///c:/Users/GOODMAN/WHToolsBox/test_box_mujoco/run_drop_simulation.py)
- `test_run_case_1`: 새로운 파라미터(`cush_weld_solref`, `cush_contact_solref`, `cush_edge_solref` 등)를 설정값에 포함.

## Verification Plan

### Automated Tests
- `run_drop_simulation.py` 실행을 통해 XML 생성 성공 및 스키마 위반 여부 확인.
- 생성된 XML의 `weld` 태그들이 `class`를 사용하고 있는지 검수.
- 모서리 블록의 `geom` 태그에 차별화된 파라미터가 적용되었는지 확인.
