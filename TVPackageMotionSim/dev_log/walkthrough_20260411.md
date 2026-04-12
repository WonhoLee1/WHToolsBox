# Walkthrough - Fix Weld Constraint Error in whtb_builder.py

안녕하세요, **WHTOOLS**입니다.

`run_drop_simulation_cases_v5.py` 실행 시 발생하던 `ValueError: unknown element 'b_bchassis_1_1_0'` 에러를 수정하였습니다. 이번 패치를 통해 어셈블리의 이산화(Discrete) 가동 여부와 관계없이 보조 질량(Auxiliary Mass)이 제품 본체에 안정적으로 결합됩니다.

## 1. 작업 내용

### 1.1. 오류 원인 파악
- **설정**: `chassis_use_weld=False` (단일 바디 모드)
- **문제**: 단일 바디 모드에서는 개별 격자 바디(`b_bchassis_...`)가 생성되지 않음에도 불구하고, 용접(Weld) 제약 조건이 해당 이름을 참조하여 MuJoCo 모델 로딩이 실패함.

### 1.2. 해결 방법
- `whtb_builder.py`의 보조 질량 용접 로직을 수정하였습니다.
- 부품의 `use_internal_weld` 설정 값을 실시간으로 체크하여, 용접 대상 바디의 명칭을 다음과 같이 동적으로 선택합니다:
    - `use_internal_weld=True`: `b_{name.lower()}_{i}_{j}_{k}` (개별 블록 바디)
    - `use_internal_weld=False`: `{name}` (부품 통합 바디)

## 2. 변경 파일

- [MODIFY] [whtb_builder.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_builder.py)

## 3. 검증 결과

- `run_drop_simulation_cases_v5.py`의 `test_case_2_setup` (섀시 용접 해제 모드) 실행 결과, 모델 생성 및 시뮬레이션 루프가 에러 없이 정상적으로 완료됨을 확인하였습니다.
- **로그 확인**:
  ```text
  [01:42:00] ℹ️ [Headless] UI 오픈 요청이 없으므로 시뮬레이션을 정상 종료합니다.
  Exit code: 0
  ```

## 4. 마치며

이번 수정을 통해 대규모 어셈블리의 성능 최적화를 위해 일부 부품의 이산화 용접을 비활성화(`use_weld=False`)하더라도, 시스템의 물리적 구속 조건이 깨지지 않고 안정적으로 작동하는 기반을 마련하였습니다.

> [!tip]
> 모든 작업 관련 문서(Plan, Task, Walkthrough)는 `./dev_log/` 폴더 내에 오늘 날짜(`20260411`)로 백업되었습니다.

추가적으로 개선이 필요한 사항이 있으면 언제든 말씀해 주세요!
