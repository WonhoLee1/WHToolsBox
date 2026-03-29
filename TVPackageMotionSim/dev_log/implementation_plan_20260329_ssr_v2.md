# SSR 기반 고정밀 구조 해석 및 통합 UI 고도화 계획

기존의 2D 컨투어 시각화에만 사용되던 SSR(Structural Surface Reconstruction) 기법을 구조 지표 엔진과 통합하여, 모델화된 연속 곡면으로부터 정밀한 응력 및 변형 지표를 추출하는 기능을 구현합니다. 사용자는 포스트 프로세싱 단계에서 필요에 따라 특정 구간/시간 해상도에 대해 SSR 분석을 수행하고 그 결과를 분석에 활용할 수 있습니다.

---

## 1. 개요 및 설계 원칙

- **On-Demand Calculation**: 시나리오 중 매 스텝 계산하는 대신, 포스트 UI에서 사용자가 요청할 때(`Advanced SSR Analysis` 버튼) 지정된 프레임/시간 간격에 대해 계산합니다.
- **Shared Logic**: 컨투어 가시화 로직과 구조 지표 산출 로직이 동일한 SSR 엔진을 공유하도록 구현합니다.
- **Temporal Fallback**: 특정 시점에 SSR 데이터가 없는 경우, 가장 가까운 과거/미래의 계산된 데이터를 표시하여 부드러운 분석 환경을 제공합니다.
- **Unified Interface**: 구조 해석 탭과 2D 컨투어 탭에 동일한 분석 도구 진입점을 제공합니다.

---

## 2. 세부 구현 계획

### 2.1. 데이터 엔진 고도화 (`whts_engine.py`, `whts_data.py`)

#### [PATCH] `nominal_local_pos` 속성 추가 및 보존
- **Problem**: `postprocess_ui.py`에서 `self.sim.nominal_local_pos` 접근 시 `AttributeError` 발생.
- **Fix**:
    - `DropSimulator`의 `_discover_components` 수행 시 각 바디의 초기 로컬 좌표를 `self.nominal_local_pos` 딕셔너리에 저장합니다.
    - `DropSimResult` 클래스에 해당 필드를 추가하여 `.pkl` 저장 및 로드 시에도 데이터가 유지되도록 합니다.

### 2.2. SSR 핵심 연산 모듈 (`whts_reporting.py` 또는 `whts_utils.py`)

#### [NEW] `compute_ssr_shell_metrics(positions, values, thickness, E, nu=0.3)`
- **Input**: 블록 위치(X, Y), 각도/변위 값, 쉘 두께($t$), 영률($E$), 푸아송 비($\nu$).
- **Logic**:
    1. `scipy.interpolate.Rbf`를 이용한 연속 곡면 $w(x, y)$ 생성.
    2. 고해상도 그리드에서 2계 도함수(곡률, Curvature) $\kappa_{xx}, \kappa_{yy}, \kappa_{xy}$ 산출.
    3. Shell 이론 기반 모멘트 및 최대 등가 응력(Von-Mises) 계산.
- **Output**: 최대 응력값, 최대 응력 발생 위치, 고해상도 그리드 데이터.

### 2.3. 포스트 프로세싱 UI 고도화 (`postprocess_ui.py`)

#### [NEW] `Advanced SSR Analysis` 통합 툴 창
- **UI**: 두 탭에 버튼 추가 $\rightarrow$ 클릭 시 팝업 창 오픈.
- **입력 항목**:
    - 대상 컴포넌트 선택 (Checklist)
    - 분석 범위 및 간격 (Total Frames 또는 Delta Time)
    - 강성 및 두께 파라미터 확인/수정.
- **동작**:
    - "Run Analysis" 클릭 시 배경 스레드에서 지정된 스텝들에 대해 SSR 연산 수행.
    - 수행 결과를 `self.sim.metrics[comp]['ssr_results']`에 시계열로 저장.

#### [MODIFY] 시각화 및 리포팅 연동
- **2D 컨투어**: 데이터 부재 시 `min(calculated_steps, key=lambda s: abs(s - current_step))` 로직으로 가장 가까운 SSR 결과 표시.
- **구조 해석**: SSR 계산이 완료되면 요약 테이블에 "Peak SSR Stress" 항목을 동적으로 추가 표시.

---

## 3. 검증 계획

### 3.1. 기능 검증
- [ ] `nominal_local_pos` 에러 수정 확인 및 컨투어 정상 작동 여부.
- [ ] 다양한 프레임 간격 설정에 따른 SSR 지표 산출 정확도 및 소요 시간 확인.
- [ ] 애니메이션 재생 시 계산된 SSR 데이터의 정상적인 Fallback 표시 확인.

### 3.2. 성능 최적화
- [ ] RBF 보간 해상도를 동적으로 조절할 수 있도록 옵션화하여 분석 속도 밸런싱.

---

> [!IMPORTANT]
> **사용자 피드백 요청**
> - **버튼 이름 추천**: `Advanced Shell-Metric Analysis (SSR)` 외에 `High-Fidelity SSR Stress Analysis` 또는 `Precision Surface Metrics` 등 선호하시는 명칭이 있으신가요?
> - **데이터 보존 정책**: SSR 재계산 결과는 현재 세션 중에만 유지되며, `.pkl`에 다시 저장할지 여부를 선택하게 할까요? (기본은 휘발성 권장)
