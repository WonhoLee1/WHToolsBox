# [WHTOOLS] Config Default 및 파라미터 네이밍 표준화 계획

`get_default_config()` 함수를 `test_run_case_1` 기반으로 최적화하고, 프로젝트 전반의 파라미터 네이밍을 표준화(Short-name 제거)하여 유지보수성을 향상시킵니다.

## User Review Required

> [!IMPORTANT]
> **파라미터 네이밍 변경 (Breaking Changes)**:
> - `oc_` -> `opencell_`
> - `_oc` -> `_opencell` (예: `mass_oc` -> `mass_opencell`)
> - `occ_` -> `opencellcoh_`
> - `_occ` -> `_opencellcoh` (예: `mass_occ` -> `mass_opencellcoh`)
> - `chas_d` -> `chassis_d`
>
> 위 변경 사항에 따라 `whtb_builder.py`, `whts_engine.py`, `whts_utils.py` 뿐만 아니라 **`run_drop_simulation_cases_v4.py`, `run_drop_simulation_cases_v5.py` 내부의 설정 키들도 일괄 수정**하여 최신 표준을 따르도록 합니다.

## Proposed Changes

### 1. Configuration Core (`run_discrete_builder/`)

#### [MODIFY] [whtb_config.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_config.py)
- `test_run_case_1`의 모든 물리/기하 파라미터를 디폴트 값으로 설정.
- 솔버 코드(`.get()`)에서 사용되던 기본값들(SSR 관련, 소성 변형 관련 등)을 모두 명시적으로 추가.
- 내부 구조를 **Geometry, Physics, Simulation, Component, Mass, Air, PostProcess** 카테고리로 분류하여 정리.
- `oc_`, `occ_` 관련 키를 `opencell_`, `opencellcoh_`로 변경.
- **하위 호환성 레이어**: 기존 `oc_`, `occ_` 등의 키로 입력이 들어와도 내부적으로 `opencell_`, `opencellcoh_`로 매핑되도록 처리하여 기존 테스트 코드가 수정 없이 동작하게 함.

#### [MODIFY] [whtb_builder.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_builder.py)
- 변경된 파라미터 네이밍(`opencell_div`, `opencellcoh_d` 등)에 맞춰 참조 코드 수정.

### 2. Simulation Engine (`run_drop_simulator/`)

#### [MODIFY] [whts_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_engine.py)
- `_aerodynamics_callback`, `_collect_history` 등에서 사용하는 `config.get()` 참조 키를 표준화된 이름으로 수정.

#### [MODIFY] [whts_reporting.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_reporting.py)
- SSR 관련 파라미터(`ssr_resolution`, `ssr_thickness` 등)를 `self.config`에서 직접 참조하도록 최적화.

#### [MODIFY] [whts_utils.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_utils.py)
- `compute_corner_kinematics` 호출 시 사용하는 키(`box_w` 등) 확인 및 동기화.

### 3. Scenario Cases (`/`)

#### [MODIFY] [run_drop_simulation_cases_v4.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulation_cases_v4.py)
#### [MODIFY] [run_drop_simulation_cases_v5.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulation_cases_v5.py)
- 각 Case 함수 내에서 정의된 `cfg["oc_..."]`, `cfg["mass_oc"]` 등을 새로운 표준 네이밍으로 일괄 교체.

## Open Questions

> [!QUESTION]
> - `test_run_case_1`에서 설정하는 `chassis_aux_masses`의 구체적인 리스트 데이터도 디폴트에 포함할까요? (현재는 빈 리스트가 기본값입니다.) -> Case 1의 `[{"name": "InertiaAux_Single", ...}]`을 기본으로 넣겠습니다.
> - `occ_`를 `opencellcoh_`로 변경할 때, `coh`는 `Cohesive`를 의미하는 것으로 이해했습니다. 맞을까요? (코드상 `BOpenCellCohesive`와 매칭됨)

## Verification Plan

### Automated Tests
- `python run_drop_simulation_cases_v4.py` 실행을 통해 `test_run_case_1`이 정상 작동하는지 확인.
- `get_default_config()`를 단독 호출하여 반환된 딕셔너리의 키 이름과 값이 의도한 대로(Case 1 기반) 설정되었는지 확인하는 스크립트 작성.

### Manual Verification
- `whtb_config.py`의 내부 구조가 가독성 있게 정리되었는지 코드 리뷰.
- `oc_div` 입력 시 `opencell_div`로 내부 반영되는지 확인.
