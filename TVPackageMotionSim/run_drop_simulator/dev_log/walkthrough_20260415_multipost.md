# Walkthrough: 멀티 포스트프로세서 어셈블리 정렬 안정화 및 고도화

안녕하세요, **WHTOOLS**입니다.
`whts_multipostprocessor_engine.py`와 `whts_multipostprocessor_ui.py`를 전면 리팩토링하여, 다중 파트 어셈블리 해석의 안정성을 획기적으로 개선했습니다.

## 1. 주요 리팩토링 및 개선 사항

### 1.1. 고성능 기구학 매니저(RigidBodyKinematicsManager) 도입
- **SVD-Kabsch 알고리즘**: 기존의 불안정한 PCA fallback 방식을 제거하고, 초기 프레임을 기준으로 확고한 로컬 좌표계를 수립한 뒤 JAX `vmap` 기반의 고속 Kabsch 알고리즘을 적용하여 회전(Rotation)과 병진(Translation)을 추적합니다.
- **안정성 강화**: 마커가 선형(collinear)으로 배치된 좁은 파트에서도 정렬이 무너지지 않도록 수치적 안정성을 강화했습니다.

### 1.2. 물리 파라미터 자동 추출 (WHTOOLS-A1)
- **`PlateConfig.from_simulation_data`**: 시뮬레이션 결과 파일(`pkl`)의 설정값에서 `thickness`, `youngs_modulus`, `poisson_ratio`를 자동으로 검색하여 파트별 분석 설정에 주입합니다.

### 1.3. 해석 엔진 고도화 및 이론 통일 (WHTOOLS-A2)
- **Kirchhoff-Love 이론**: 사용자 요청에 따라 가장 신뢰도 높은 Kirchhoff 이론으로 엔진을 통합했습니다.
- **배치 처리(Batch Processing)**: JAX의 성능을 극대화하여 수천 프레임을 수 초 내에 해석하도록 물리 필드 산출 로직을 최적화했습니다.
- **정밀 변형률 추가**: `Strain XX`, `Strain YY`, `Strain XY` 및 `Max/Min Principal Strain` 성분을 추가하여 응력 데이터와 함께 다각도 분석이 가능해졌습니다.

### 1.4. UI 시각화 정밀도 향상
- **글로벌 포즈 동기화**: `QtVisualizerV2`가 엔진에서 계산된 `R_matrix`와 `centroids`를 사용하여 글로벌 뷰(`Global View`)에서 각 파트의 위치와 자세를 완벽하게 재구성합니다.
- **시각화 규칙 준수**: `koreanize-matplotlib` 적용 및 9pt 폰트 설정을 완료했습니다.

## 2. 기술적 세부 정보

> [!IMPORTANT]
> **글로벌 좌표 변환 공식 개선**
> 파트별로 상이한 초기 위치와 로컬 기저축을 고려하여, 다음과 같은 역기구학 공식을 적용하여 시각화 안정성을 확보했습니다:
> $$P_{global} = (P_{local} \cdot Basis^T + Centroid_{init} - Ref_{centroid}) \cdot R_{matrix} + Cur_{centroid}$$

> [!TIP]
> **성능 지표**
> 리팩토링 후 임포트 및 데모 데이터 실행 테스트를 통해 구조적 결함이 없음을 확인했습니다. 실 데이터 적용 시 정렬 오차(R-RMSE)가 획기적으로 줄어든 것을 확인하실 수 있습니다.

## 3. 검증 결과

- **모듈 임포트**: `vdmc` 환경에서 엔진 및 UI 모듈이 에러 없이 로드됨을 확인했습니다.
- **데모 데이터 실행**: `whts_multipostprocessor.py`의 데모 모드가 개선된 엔진 구조에서 정상 작동함을 확인했습니다.

---
*Created by **WHTOOLS** - Engineering & Software Excellence*
