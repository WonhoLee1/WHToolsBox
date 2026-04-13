# [Goal] XML Weld Class Generation Correction

MuJoCo XML 생성 시 `<default>`에 정의된 `weld` 클래스를 실제 `<weld>` 요소가 사용하도록 로직을 수정합니다. 이를 통해 `-1000.0`과 같은 특수 강성 파라미터가 정확히 반영되도록 합니다.

## User Review Required

> [!IMPORTANT]
> - `BCushion`의 경우 `is_corner_block` 조건이 참인 블록이 포함된 용접 쌍에 대해 `weld_bcushion_corner` 클래스를 적용합니다.
> - 기존의 명시적인 `solref`, `solimp` 속성은 제거되어 클래스 상속을 따르게 됩니다.

## Proposed Changes

### [run_discrete_builder]

---

#### [MODIFY] [whtb_base.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_base.py)
- `BaseDiscreteBody.get_weld_xml_strings` 수정:
    - `class="weld_{self.name.lower()}"` 속성을 추가합니다.
    - `solref`, `solimp` 속성을 제거합니다.

#### [MODIFY] [whtb_models.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_models.py)
- `BCushion.get_weld_xml_strings` 수정:
    - 연결되는 두 블록 중 하나라도 `is_corner_block`인 경우 `weld_bcushion_corner` 클래스를 적용합니다.
    - 그 외에는 `weld_bcushion` 클래스를 적용합니다.
    - 하드코딩된 `solref`, `solimp` 문자열 생성을 제거합니다.

## Open Questions
- 없음 (사용자가 `is_corner_block` 사용을 확정함)

## Verification Plan

### Automated Tests
- `run_drop_simulation_cases_v6.py` 실행 후 생성된 `temp_drop_sim.xml` 파일 검독.
    - `<equality>` 섹션 내의 `<weld>` 요소에 `class` 속성이 있는지 확인.
    - `solref`, `solimp` 속성이 사라졌는지 확인.

### Manual Verification
- MuJoCo Viewer를 통해 완충재의 물리적 거동(stiffness)이 의도한 대로 나타나는지 확인.
