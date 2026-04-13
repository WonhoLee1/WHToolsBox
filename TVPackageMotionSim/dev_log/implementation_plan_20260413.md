# [WHTOOLS] `whtb_physics.py` 복구 및 시뮬레이션 파이프라인 안정화

시뮬레이션 실행 중 `run_discrete_builder.whtb_physics` 모듈을 찾을 수 없어 발생하는 `ModuleNotFoundError`를 해결하기 위해, 누락된 물리 해석 및 밸런싱 모듈을 복구하고 관련 경로와 설정을 최적화합니다.

## User Review Required

> [!IMPORTANT]
> - `whtb_physics.py` 파일이 현재 워크스페이스에서 누락되어 시뮬레이션이 중단되고 있습니다. 이 파일은 컴포넌트의 질량, 무게중심(CoG), 관성 모멘트(MoI)를 분석하고 목표치에 맞게 보정 질량(Aux Masses)을 자동 배치하는 핵심 로직을 포함합니다.
> - 기존 `whts_utils.py`의 `calculate_required_aux_masses`와 `whtb_builder.py`의 `create_model` 사이의 순환 참조 가능성을 차단하기 위해, 물리 분석 로직을 `whtb_physics.py`로 완전히 일원화합니다.

## Proposed Changes

### 1. `run_discrete_builder` 폴더 (빌더 엔진 하부)

---

#### [NEW] [whtb_physics.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_physics.py)
- `analyze_and_balance_components(config, verbose=True)` 함수 구현.
- `BaseDiscreteBody.calculate_inertia()`를 활용하여 어셈블리의 기초 관성을 측정.
- `config["components_balance"]` 또는 `target_mass` 설정을 기반으로 부족한 질량과 관성을 계산.
- 보정용 `BAuxBoxMass` 객체 데이터를 생성하여 `config["component_aux"]`에 등록.
- `rich` 라이브러리를 사용하여 시뮬레이션 시작 전 물리 분석 결과 테이블 출력.

#### [MODIFY] [whtb_builder.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_builder.py)
- `create_model` 함수 내부에서 `whtb_physics.py`를 임포트할 때 발생할 수 있는 경로 문제를 방지하기 위해 상대 임포트 또는 절대 경로 처리를 강화합니다.
- `config.get("component_aux", {})`를 통해 전달된 보정 질량을 모델에 추가하는 로직을 견고하게 유지합니다.

### 2. `run_drop_simulator` 폴더 (시뮬레이션 엔진 하부)

---

#### [MODIFY] [whts_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_engine.py)
- `apply_balancing` 메서드에서 중복된 `calculate_required_aux_masses` 호출을 제거하고, `create_model`이 내부적으로 수행하는 `analyze_and_balance_components`에 의존하도록 구조를 개선합니다.

#### [MODIFY] [whts_utils.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_utils.py)
- `calculate_required_aux_masses` 함수가 더 이상 `create_model`을 호출하지 않도록 하거나, `whtb_physics`의 기능을 활용하도록 리팩토링하여 순환 참조를 방지합니다.

## Open Questions

> [!QUESTION]
> - 현재 `v5`와 `v6` 스크립트 모두 동일한 물리 모듈을 참조하고 있습니다. `v6`에서는 "최소 정보(Minimalist)" 분석을 지향하고 있는데, 초기 밸런싱 단계에서도 동일한 엄격한 물리 검증을 적용할까요? 아니면 `v6` 전용의 경량화된 밸런싱 로직이 필요하신가요? (일단은 정밀도 유지를 위해 `v5`와 공유하는 정밀 로직을 적용할 예정입니다.)

## Verification Plan

### Automated Tests
1. **모듈 임포트 테스트**: `python -c "from TVPackageMotionSim.run_discrete_builder.whtb_physics import analyze_and_balance_components; print('Success')"` 명령으로 임포트 여부 확인.
2. **풀 파이프라인 실행**: `python TVPackageMotionSim/run_drop_simulation_cases_v6.py`를 실행하여 `Case 1`이 물리 분석 테이블을 출력하고 시뮬레이션 단계로 진입하는지 확인.

### Manual Verification
- 터미널에 출력되는 `Assembly Physics Analysis` 테이블의 `Final (Balanced)` 질량이 `Target (Req)` 값(예: 25.0kg)과 일치하는지 확인.
- 생성된 `simulation_model.xml` 파일 내에 `InertiaAux_` 명칭을 가진 보정 질량 블록들이 올바르게 포함되었는지 육안 점검.
