# [Goal] 완충재(Cushion, Cushion_corner)의 torquescale 이론적 자동화 공식 도입

완충재의 3D 연속체 역학(Solid Continuum) 모델링에 맞춰, 전단/인장 탄성 강성비($G/E$) 및 이산 격자 해상도(div)에 따른 힌지 보정(Hinge scale)을 반영한 이론적 `torquescale` 자동 계산 공식을 도입합니다.

## User Review Required

> [!NOTE]
> * 기존 `cushion`과 `cushion_corner` 용접 강성의 하드코딩 값(`0.001`, `0.1`)을 지우고, 이산 격자 분할도(`div`)에 따라 굽힘 유연성을 최적화하는 수식 계산 결과로 대체합니다.
> * 격자 평균 분할 수($N_{avg}$)가 3일 때 기존 튜닝값인 `0.001`에 매칭되도록 수식 내 격자 보정 팩터($\alpha_{\text{grid}}$)의 파라미터를 보정하여 기존 해석과의 연속성을 최대한 보장합니다.

## Proposed Changes

### 1. `run_discrete_builder` 유틸리티 레이어

#### [MODIFY] [whtb_utils.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_utils.py)
* `calculate_cushion_torquescale` 함수 추가:
  - 등방성 3D 연속체 이론식 $G = \frac{E}{2(1+\nu)}$ 를 활용하여 기초 굽힘/인장비 $G/E$ 산출.
  - 평균 격자 수 $N_{avg} = \text{mean}(div)$ 에 비례해 증가하는 격자 보정 로그 스케일 함수 $\alpha_{\text{grid}} = 10^{-\frac{4.0}{N_{avg} - 1}}$ 적용.
  - 모서리부(`is_corner=True`)일 경우 기하학적 모서리 강성을 모사하기 위해 100배 증가(최대 1.0 제한) 캡 보정 적용.
* `__init__.py`에서 해당 함수를 임포트하여 노출.

---

### 2. 시뮬레이션 케이스 레이어

#### [MODIFY] [run_drop_simulation_cases_v6.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulation_cases_v6.py)
* `calculate_cushion_torquescale`를 임포트 목록에 추가.
* `cushion` 및 `cushion_corner` 용접 강성 정의 시 이론 공식을 호출하여 자동 생성된 값을 `cfg["welds"]`에 주입.

---

## Verification Plan

### Automated/Manual Verification
- 시뮬레이션 모델 생성(XML 빌드)을 로컬로 구동하여 `cushion` 및 `cushion_corner`에 이론적으로 계산된 `torquescale`이 정확히 XML에 기입되는지 확인합니다:
  - $3\times3\times3$ 격자 분할 기준:
    - Cushion `torquescale` $\approx 0.00476$
    - Cushion Corner `torquescale` $\approx 0.476$
- 시뮬레이션의 빌드 에러가 없는지(`TypeError` 등 없이 XML이 완벽히 파싱되는지) 검증합니다.
