# WHTOOLS 최적화 및 DOE 프레임워크 구현 완료 보고서 (2026-06-03)

본 문서는 MuJoCo 낙하 시뮬레이션을 대상으로 설계 파라미터의 튜닝, 실험계획법(DOE) 기반 배치 실행, 결과 Overlay 비교 및 조건별 최적 설계안(Best Case)을 자동 추출하는 **최적화 & DOE 프레임워크**의 최종 구현 명세를 다룹니다.

## 주요 작업 완료 사항

### 1. 물리 파라미터 연동 안정성 개선
- **대상 파일**: [whtb_physics.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_physics.py)
- **내용**: `target_inertia`가 `[Ixx, Iyy, Izz]`의 3성분으로 전달될 때, `(3,)`과 `(6,)` 크기 불일치로 인해 브로드캐스팅 오류(`ValueError`)가 발생하던 문제를 해결하기 위해, 3성분 검출 시 자동으로 `[Ixx, Iyy, Izz, 0.0, 0.0, 0.0]`의 6성분으로 패딩 및 보정하는 예방 패치를 적용했습니다.

### 2. 최적화 및 DOE 핵심 엔진 구축
- **대상 파일**: [whts_optimization_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_optimization_engine.py)
- **기능**:
  - **DOE Generator**: Latin Hypercube Sampling(LHS, NumPy 자체 고신뢰성 구현), Random, Full Factorial 기법을 지원하여 이산(Discrete)/연속(Continuous) 튜닝 변수를 처리합니다.
  - **DOE Batch Runner**: Base JSON 설정을 파싱하여 각 케이스별 설정으로 시뮬레이터를 연속 기동하고, 결과 요약 및 이력 데이터를 `results/DOE/case_{id}/` 폴더에 격리 저장합니다.
  - **Optimization Evaluator**: 다중 제약 조건 필터링 및 목적 함수에 최적화된 설계 후보안을 검색합니다.

### 3. Gooey 탭형 입력기 및 PySide6 최적화 대시보드 구현
- **대상 파일**: [whts_optimization_ui.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_optimization_ui.py)
- **기능**:
  - **Gooey Parser**: `@Gooey` 데코레이터와 `navigation="Tabbed"` 속성을 적용해, Base Config / Geometry / Physics / Target CoG / Dashboard 탭 구조를 구축하고 그룹별 체크박스 기반의 튜닝 위젯 배치를 완료했습니다.
  - **Target CoG (X, Y, Z) 지원**: 타겟 무게중심의 세 축 성분에 대해 각각 `-0.020`에서 `0.020` [m]의 범위에서 discrete/continuous 튜닝을 지원합니다. 엔진 배치 러너는 해당 파라미터 유입 시 `components_balance`에 매핑 후 auto-balancing 물리 보정을 자동으로 재연산합니다.
  - **DOE Monitor UI (최적화 대시보드)**: 
    - **DOE Run Cases Selector**: 여러 케이스를 다중 선택하여 하나의 Matplotlib 차트에 Z-변위 및 지면 접촉력을 중첩(Overlay) 가시화합니다. (9pt 한글 폰트 및 다크 테마 적용)
    - **Optimal Target Selection**: 슬라이더 및 파이썬 식(Expression) 형태의 고급 룰 필터를 제공하여 이를 만족하는 최적 파라미터 세트를 한눈에 볼 수 있도록 도출합니다.

### 4. 프레임워크 통합 실행 런처
- **대상 파일**: [run_optimization_framework.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_optimization_framework.py)
- **내용**: `config.json` 설정 파일 부재 시 자동으로 default 설정을 자동 빌드하고 프레임워크를 원클릭으로 구동해 주는 런처입니다.

---

## 검증 (Verification) 결과

### 1. DOE 샘플링 엔진 테스트
- `scratch/test_doe.py` 스크립트를 기동하여 LHS, Random, Full Factorial 샘플링이 정의된 이산/연속 경계에 따라 정확하고 고르게 분포를 생성함을 수치적으로 확인 완료했습니다.

### 2. 대시보드 코어 로직 및 UI 검증
- `scratch/test_ui_dashboard.py` 테스트를 통해 요약 JSON 로딩, 테이블 연동 및 다중 제약 조건/고급 파이썬 필터식(`max_stress_mpa < 180.0` 등)에 기반하여 최적안(Best Case 2)을 한치의 오차 없이 추천함을 통과하였습니다.

---

## 파일 구조 및 링크

- **런처 진입점**: [run_optimization_framework.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_parameter_study/run_optimization_framework.py)
- **UI 및 다시보드**: [whts_optimization_ui.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_parameter_study/whts_optimization_ui.py)
- **배치 연산 엔진**: [whts_optimization_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_parameter_study/whts_optimization_engine.py)
- **물리 픽스 패치**: [whtb_physics.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_physics.py)
