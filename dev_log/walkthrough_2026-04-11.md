# MuJoCo v5 접촉 시스템 리팩토링 결과 보고서

디지털 트윈 파이프라인(`v5`)의 접촉 제어 시스템을 기존 비트마스크(Bitmask) 방식에서 **명시적 Pair 방식**으로 완전히 전환하였습니다. 이를 통해 복잡한 파트 간 상호작용과 쿠션 모서리(`_edge`)의 물리 특성을 정밀하게 제어할 수 있게 되었습니다.

## 주요 변경 사항

### 1. 설정 구조의 단일화 (UCM)
- [whtb_config.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_config.py)
    - `CONTACT_PAIRS` 딕셔너리를 도입하여 파트 조합별(`paper-floor`, `cushion-floor` 등) 마찰력 및 물리 계수를 한 곳에서 관리합니다.
    - 기존 매터리얼 매핑(`mat_*`)에서 불필요하게 흩어져 있던 접촉 속성들을 제거하고 순수 재질/시각화 정보만 남겼습니다.

### 2. 동적 Pair 생성 엔진 구현
- [whtb_builder.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_builder.py)
    - 모든 Geom의 `contype/conaffinity`를 `0`으로 고정하여 기존의 자동 충돌 방식을 비활성화했습니다.
    - 바디 계층 구조를 순회하며 모든 블록의 메타데이터(타입, 모서리 여부)를 수집하여 필요한 조합에 대해서만 `<pair>` 태그를 생성합니다.
    - **동일 파트 내 자가 접촉 제외 로직**을 적용하여 계산 효율성을 높였습니다.

### 3. 쿠션 모서리(_edge) 특화 대응
- 쿠션 블록 중 모서리에 해당하는 블록은 자동으로 `cushion_edge` 타입으로 분류됩니다.
- 이를 통해 `cushion_edge - floor` 쌍에 대해 일반 쿠션 면과는 다른 물리 계수(`solref` 등)를 적용할 수 있도록 페어링 로직을 강화했습니다.

### 4. 시뮬레이션 케이스 적용
- [run_drop_simulation_cases_v5.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulation_cases_v5.py)
    - `test_case_1_setup` 및 `test_case_2_setup`에 통합된 `contacts` 설정을 적용하여 사용자가 시뮬레이션 직전에 물리 계수를 쉽게 파악하고 수정할 수 있도록 노출했습니다.

## 검증 결과

> [!CHECK]
> **XML 유효성 검사**
> `scratch/test_v5_contacts.py`를 통해 생성된 XML이 MuJoCo `MjModel`에 정상적으로 로드됨을 확인하였습니다.
> - 총 1,405개의 Geom을 포함한 복잡한 어셈블리에 대해 약 2,000개 이상의 명시적 Pair가 올바르게 생성되었습니다.
> - `contact_bauxboxmass` 등 누락되었던 클래스 정의 오류를 해결하였습니다.

> [!TIP]
> **추후 활용 방안**
> 이제 파트 간의 접촉이 불안정할 경우, 각 파트의 재질 속성을 건드릴 필요 없이 `run_drop_simulation_cases_v5.py`의 `contacts` 테이블 내 수치만 조정하여 즉시 해결할 수 있습니다.
