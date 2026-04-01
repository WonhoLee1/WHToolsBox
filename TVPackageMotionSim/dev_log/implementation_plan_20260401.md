# [WHTOOLS] 시뮬레이션 설정 파라미터 주석 추가 계획 (2026-04-01)

`test_run_case_1()` 함수 내의 `cfg` 변수 세팅 과정에서 사용된 각 키(Key)들에 대해, 엔지니어링 관점에서의 상세 설명을 한글 주석으로 추가합니다. 

## Proposed Changes

### [TVPackageMotionSim](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim)

#### [MODIFY] [run_drop_simulation_cases_v4.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulation_cases_v4.py)
- `test_run_case_1` 함수 내의 `cfg` 딕셔너리 설정 부분에 한글 주석을 추가합니다. 
- 각 파라미터의 물리적 의미, 단위, 그리고 시눌레이션 영향도를 명시합니다.

## Verification Plan

### Automated Tests
- `python run_drop_simulation_cases_v4.py` 명령을 실행하여 문법 오류가 없는지 확인합니다.

### Manual Verification
- 주석의 내용이 WHTOOLS 엔지니어링 표준에 부합하는지 최종 검토합니다.
