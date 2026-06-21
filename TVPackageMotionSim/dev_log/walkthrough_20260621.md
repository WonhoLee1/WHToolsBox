# [Walkthrough] 완충재(Cushion)의 torquescale 이론적 자동화 공식 적용 완료

완충재(`cushion`)와 완충재 코너(`cushion_corner`)에 대해 기존 하드코딩된 값을 무시하고, 3D Solid 연속체 역학($G/E$ 비율)과 이산 격자 크기 보정(Hinge scale)을 반영한 **이론적 자동화 공식**을 적용하여 모델 빌드 및 기동 검증까지 완료하였습니다.

## 주요 변경 사항

### 1. 이론적 torquescale 산출 로직 구현
* **[whtb_utils.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_utils.py)**
  * `calculate_cushion_torquescale(div, nu, is_corner)` 함수 구현:
    * **연속체 이론비 ($G/E$)**: 포아송 비 $\nu=0.05$ 기준 $\frac{1}{2(1+\nu)} \approx 0.4762$ 계산.
    * **격자 크기 보정 ($\alpha_{\text{grid}}$)**: 평균 격자 수($N_{avg}$)에 비선형 스케일 팩터 $10^{-\frac{4.0}{N_{avg}-1}}$ 를 적용하여, 격자가 거칠어질 때 힌지처럼 거동하도록 보간.
    * **코너 캡 보정**: 코너 기하 강성을 고려해 평면부 대비 100배 스케일링(최대 1.0 제한).
  * `run_discrete_builder/__init__.py`에 해당 함수를 추가하여 노출 완료.

### 2. 완충재 일반/코너 용접부 분리 파싱 적용
* **[whtb_models.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_models.py)**
  * `BCushion.get_weld_xml_strings()` 함수 수정:
    * 기존에는 `welds.cushion` 키만 파싱하여 일반/코너 구분 없이 단일 `torquescale`로 적용되던 한계를 해결.
    * `cushion`과 `cushion_corner` 키를 각각 파싱하여, 코너 블록 검증 여부(`is_corner_block`)에 따라 `weld_bcushion`과 `weld_bcushion_corner` 클래스에 알맞은 `torquescale`이 동적으로 주입되도록 수정.

### 3. 시뮬레이션 케이스 적용
* **[run_drop_simulation_cases_v6.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulation_cases_v6.py)**, **[run_drop_simulation_cases_v7.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulation_cases_v7.py)**, **[run_drop_simulation_cases_doe.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulation_cases_doe.py)**, **[run_optimization_framework.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_parameter_study/run_optimization_framework.py)**, **[run_drop_simulation_case_opt_traject.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulation_case_opt_traject.py)**
  * 기존 하드코딩 `torquescale` 정의를 제거하고 신규 함수 `calculate_cushion_torquescale`를 임포트하여 welds 딕셔너리에 동적 할당 완료.

---

## 검증 결과

### 1. XML 생성 검증
`scratch/test_run.py`를 통해 $3 \times 3 \times 3$ 격자 분할 기준으로 모델 빌드를 검사한 결과, `simulation_model.xml`에 다음과 같이 성공적으로 구분 적용되었습니다:
* **일반 완충재 용접부 (`weld_bcushion`)**: `torquescale="0.004762"` (부드러운 압축 거동 유도)
* **모서리 완충재 용접부 (`weld_bcushion_corner`)**: `torquescale="0.476190"` (L자형 꺾임 구조 유지)

### 2. 시뮬레이션 동작 검증
임시 테스트 시뮬레이션(0.01초 단기 해석) 구동 결과, 수치적 폭발이나 빌드 오류 없이 JAX 배치 해석 및 결과 pickle 저장까지 에러 없이 완벽히 동작하였습니다.
