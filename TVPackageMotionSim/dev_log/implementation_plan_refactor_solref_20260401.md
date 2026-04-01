# [WHTOOLS] 시뮬레이션 설정 파라미터 변수명 전역 리팩토링 계획 (v2/2026-04-01)

시뮬레이션 설정(`cfg`)에서 사용되는 `solref` 관련 파라미터들이 MuJoCo의 물리적 의미(Time Constant, Damping Ratio)를 보다 정확히 반영하도록 변수명을 변경합니다.

- `~_solref_stiff` → `~_solref_timec` (Time Constant)
- `~_solref_damp` → `~_solref_dampr` (Damping Ratio)

## Proposed Changes

### [TVPackageMotionSim](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim)

#### [MODIFY] [run_discrete_builder/whtb_config.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_config.py)
- `get_default_config` 함수 내의 모든 관련 변수명 및 딕셔너리 키를 변경합니다.
- `cush`, `tape`, `cell`, `tv`, `ground` 등의 접두사가 붙은 모든 `solref_stiff/damp` 쌍을 수정합니다.
- `weld` 관련 파라미터(`cush_weld_solref_stiff` 등)도 동일하게 수정합니다.

#### [MODIFY] [run_drop_simulation_cases_v4.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulation_cases_v4.py)
- `test_run_case_1`, `test_run_case_2` 등에서 `cfg`를 설정하는 모든 코드를 새로운 변수명으로 업데이트합니다.
- 이전 작업에서 추가한 주석의 내용도 변수명 변경에 맞춰 미세 조정합니다.

#### [MODIFY] [run_cushion_optimization.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_cushion_optimization.py) 및 [run_stiffness_optimization.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_stiffness_optimization.py)
- 최적화 알고리즘에서 탐색하는 파라미터 키 값을 새로운 이름으로 변경합니다.
- 결과 출력 및 로그 기록 시의 변수명도 통일합니다.

#### [MODIFY] [run_discrete_builder/whtb_builder.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_builder.py)
- `get_single_body_instance` 및 `create_model` 함수에서 `config` 딕셔너리를 참조하는 부분을 수정합니다.

#### [MODIFY] [run_discrete_builder/whtb_models.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_models.py)
- `BCushion` 클래스 등에서 `corner_weld_solref` 관련 로직 확인 및 필요시 키 명칭 수정.

## Verification Plan

### Automated Tests
- `python -m py_compile`을 사용하여 모든 수정된 파일의 구문 오류 여부를 확인합니다.
- `test_run_case_1(enable_UI=False)`를 짧게 실행하여 설정값이 정상적으로 로드되고 MuJoCo XML이 생성되는지 확인합니다.

### Manual Verification
- 생성된 MuJoCo XML 내의 `solref` 값이 의도한 대로 (`timeconst dampratio`) 올바르게 들어갔는지 확인합니다.
- 전역 검색(`grep`)을 통해 누락된 변수명이 없는지 최종 확인합니다.
