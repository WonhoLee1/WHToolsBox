# Implementation Plan: Mass Balancing 고도화 및 Config 통합

안녕하세요, **WHTOOLS**입니다.
오늘은 낙하 시뮬레이션의 물리적 정확성을 결정짓는 핵심 요소인 **질량 보정(Mass Balancing)** 기능을 고도화하고, 이를 시뮬레이션 설정(Config)의 기본 항목으로 통합하는 작업을 진행하겠습니다.

---

## 1. 목표 (Objectives)
- **Config 통합**: `enable_target_balancing` 옵션을 통해 시뮬레이터가 자동으로 질량 보정을 수행하도록 통합.
- **유연한 보정 개수**: 1, 2, 3, 4, 8개의 질량체를 선택적으로 사용하여 보정할 수 있도록 지원.
- **영역 제한 (Bounding Box)**: 보정용 질량체가 항상 패키징 박스(박스+쿠션) 내부에 위치하도록 좌표 제한 로직 구현.
- **심화 분석**: CoG만 보정할 경우 변경된 MoI를 비교하여 출력하는 분석 기능 추가.

---

## 2. 제안된 변경 사항 (Proposed Changes)

### 2.1. `run_drop_simulation_v3.py`
- **`DropSimulator.__init__`**: 기본 설정에 balancing 관련 파라미터 추가.
- **`DropSimulator.setup`**: `enable_target_balancing` 활성화 시 자동으로 `apply_balancing` 호출.
- **`calculate_required_aux_masses`**:
    - `num_masses` 인자 추가 및 케이스별 로직 (1: 단일 점, 2: X축 대칭, 4: XY 평면 대칭 등) 구현.
    - `box_w`, `box_h`, `box_d`를 기반으로 한 Clipping 추가.
- **`apply_balancing`**: Baseline, Target, Final 상태를 한눈에 볼 수 있는 요약 테이블 출력 로직 추가.

### 2.2. `run_drop_simulation_cases.py`
- 새로운 Config 옵션을 사용하여 케이스별로 상이한 질량 보정 시나리오를 테스트할 수 있게 업데이트.

---

## 3. 검증 계획 (Verification Plan)

### 자동 테스트 및 수치 검증
- `run_drop_simulation_v3.py` 단독 실행 시 balancing 로그 확인.
- 보정 후 MuJoCo 모델에서 직접 `total_mass`, `cog`, `moi`를 추출하여 목표치와의 오차(Error %) 계산 및 출력.

---

**작성자**: WHTOOLS (Antigravity)  
**날짜**: 2026-03-26
