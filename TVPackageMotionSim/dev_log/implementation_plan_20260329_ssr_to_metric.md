# SSR(Structural Surface Reconstruction) 기반 고정밀 구조 지표 산출 계획

대상 시뮬레이터의 기존 구조 해석 지표는 물리적으로 분할된 "이산화된(Discrete) 블록"의 각도/위치 변화 및 인접 요소 간 상대 구배(RRG)를 바탕으로 도출되었습니다. 
하지만 이전에 포스트 프로세싱 UI에 구현했던 **고정밀 모드(SSR, Structural Surface Reconstruction)**는 주어진 각 위치 상태를 2D 평면 보간(Radial Basis Function 등)을 통해 **연속된 곡면 형상 함수(Continuous Surface)**로 구성해내는 기능입니다.

이 기법을 단순히 '결과 시각화'에 그치지 않고 본 시뮬레이션 **해석 파이프라인(엔진 단위)**으로 이관하여, 매 타임스텝마다 Shell 근사 이론에 기반한 곡률(Curvature) 행렬과 등가 응력(Von-Mises Stress 등)을 도출하는 계획입니다. 이 접근은 국부적인 노이즈를 억제하면서 취약 위치와 실제 최대 변형/응력을 극도의 해상도로 파악할 수 있게 해줍니다.

---

> [!WARNING]
> **성능 영향 검토 필요**
> 매 스텝마다 모든 부품에 대한 연속 곡면 피팅(Fitting) 곡률 연산을 수행하면 시뮬레이션 속도(FPS)가 유의미하게 저하될 수 있습니다. 
> 따라서 기본적으로는 이 기능의 활성화 여부를 Option(`enable_ssr_metrics`)으로 관리하거나, 전체 스텝 계산이 아닌 Peak 시점 전후로만 집중 수행하는 방안도 검토가 필요합니다. 본 계획은 **"활성화 옵션(Option) 제어를 통한 매 스텝 동적 도출 추적 지원"**으로 방향을 잡습니다.

---

## 1. 개요 및 이론적 배경

- **기존 방식**: (Block A 각도 - Block B 각도) / 거리 $\rightarrow$ RRG(Relative Rotation Gradient), 이웃 블록 간 상대 단순 차분.
- **SSR 산출 방식**:
  1. 부품 내 각 블록의 국소 지오메트리 좌표 $(x, y)$와 변위/변형 벡터 $W, \theta$ 등을 추출.
  2. RBF(Radial Basis Function)나 Thin-Plate Spline 모델을 통해 전체 연속 변형 필드(곡면 함수 $w(x,y)$) 추정.
  3. 해석 모델을 미분하여 Shell 평면의 2차 구배(곡률, Curvature) $\kappa_{xx}, \kappa_{yy}, \kappa_{xy}$ 연산.
  4. 기 확보된 재질/형상 강성($E$, $I$, 두께 $t$, 푸아송 비 $\nu$)과 연계해 탄성 모멘트 $M$ 및 단위 응력(Stress) 연산.

## 2. 부적합(Bug) 패치

### [MODIFY] `whts_engine.py` / `whts_data.py` - `nominal_local_pos` 속성 보존
에러 로그에서 `DropSimulator` 객체 혹은 Loaded Result 인스턴스의 `nominal_local_pos` 속성이 소거되어 UI 컨투어 렌더링 중 크래시가 발생하는 것을 확인했습니다.
- `DropSimResult` 클래스(`whts_data.py`)에 `nominal_local_pos` 속성 추가 (Dict 형태).
- `whts_engine.py`의 `_wrap_up` 과정에서 `DropSimResult` 인스턴스화 시 데이터에 포함.

## 3. 구조 해석(Reporting) 모듈 고도화 (`whts_reporting.py`)

### 3.1. SSR 강성/응력 역산 래퍼 추가
블록 묶음(Component 단위)의 3D 공간 데이터와 각도 편차 데이터를 받아 Shell 이론을 통해 분석합니다.

#### [NEW] `_compute_ssr_shell_metrics(comp_name, positions, bend_angles, twist_angles, config)`
- **Input**: 부품 구성 블록의 중심 Base Local 좌표 배열(X,Y), 각 그리드 상의 현재 step 벤딩/비틀림 각도, 부품 강성 파라미터.
- **Logic**:
  1. `scipy.interpolate.Rbf` 등을 이용해 평면 스플라인 적합 곡선 생성 (혹은 다항 회귀 기반).
  2. $dx, dy$ 미소 간격의 분석용 고유 그리드(High-Resolution Mesh) 생성.
  3. 그리드에서의 2계 미분(2nd Derivatives)으로 곡률 $(\frac{\partial^2 w}{\partial x^2}, \frac{\partial^2 w}{\partial y^2})$ 확보.
  4. (제공된 강성치/두께를 적용한) Max Principal Stress $\sigma_1, \sigma_2$, 혹은 Maximum Von-Mises 응력 반환.
- **Output**: SSR Max Stress (`float`), Peak Location `(x, y)`.

### 3.2. 정위치 로직 연동 (`compute_structural_step_metrics` 갱신)
- 기존 각도 수집 로직 후단에 다음을 병합합니다.
```python
# 설정에서 켜져 있을 때만(ssr_enabled == True) 평가 
if sim.config.get("enable_ssr_metrics", False) and len(list_of_angles) >= 3:
    ssr_stress, ssr_loc = _compute_ssr_shell_metrics(...)
    # 시계열 dict에 로깅 (max_ssr_stress_hist)
```

## 4. UI 및 리포팅 연동 (`postprocess_ui.py`)

### [MODIFY] `postprocess_ui.py`
1. 컨투어(2D) 뷰어 엔진이 파일 로드 시 `nominal_local_pos`를 안전하게 조회하도록 예외처리 추가 (`getattr`).
2. Global Summary Table에 "Max SSR Stress" 열을 동적으로 추가 (데이터가 존재하는 경우).
3. Critical Timestamps 자동 탐지(`whts_reporting.py` 내) 조건에 SSR Peak(시뮬레이션 중 가장 높은 SSR 응력이 발생한 시점) 항목을 신규 추가.

---

> [!QUESTION] **사용자 확인 요청 사항**
> 1. 매 타임스텝마다 SSR을 위한 2D 곡률 연산(회귀 및 RBF 평가)을 수행할 경우, **초당 프레임율(FPS) 저하 및 생성되는 `.pkl` 결과 데이터의 용량 증가**가 예상됩니다. 괜찮으실까요?
> 2. 구성 설정 상 기본값(Option) 이름은 `enable_ssr_metrics`(옵션: False) 형태로 파라미터 제어를 넣는 것이 적합할지 의견 부탁드립니다.
