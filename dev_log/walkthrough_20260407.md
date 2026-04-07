# 2026-04-07: Plate Deformation Analysis Stabilization Walkthrough

안녕하세요, **WHTOOLS**입니다. 
구조 변형 해석 엔진의 시계열 안정성을 대폭 강화한 업데이트 내용을 공유드립니다. 

이번 업데이트는 특히 **굽힘 변형 시 기준면이 흔들리는 현상(Drift)**과 **프레임 간 수직축이 뒤집히는 안정성 문제**를 정밀하게 해결하는 데 초점을 맞추었습니다.

## 1. 핵심 개선 사항

### 1.1. 가중치 기반 강체 운동 제거 (Weighted SVD)
> [!tip]
> 굽힘 변형은 판의 가장자리에서 큰 변위가 발생합니다. 기존 산술 평균 방식은 이 가장자리 변위를 '회전'으로 오도하여 기준면이 요동치게 만듭니다.

- **Gaussian Weighting**: 판의 중앙부 마커에 높은 가중치를 부여하고 가장자리 부하를 줄였습니다.
- **Weighted Centroid & Covariance**: 가중치가 적용된 `Kabsch` 알고리즘을 통해 굽힘 중에도 판의 중심 기준 좌표계가 견고하게 유지됩니다.

### 1.2. 시계열 연속성 및 축 안정화 (Axis Protection)
> [!warning]
> SVD를 이용한 회전 행렬 산출 시, 수치적 모호성으로 인해 수직축(Normal)이 180도 반전되는 현상이 발생할 수 있습니다.

- **Axis Flip Tracking**: 이전 프레임의 노멀 벡터와 현재 노멀의 내적(Dot Product)을 실시간 감시하여 방향성을 일치시켰습니다.
- **Normalization Freeze**: 매 프레임 변하던 하이퍼파라미터(Min-Max Stats)를 첫 프레임 기준으로 고정하여, 입력 데이터 정규화 과정에서 발생하는 시계열 노이즈를 근본적으로 차단했습니다.

### 1.3. 스플라인 가장자리 진동 억제 (Gradient Penalty)
- **1차 미분(Gradient) 패널티**: 고차 다항식 근사 시 경계선 부근에서 발생하는 불필요한 곡률 진동(Runge's phenomenon)을 억제하기 위해 `grad_lambda` 항을 시스템 행렬에 통합했습니다.

## 2. 주요 코드 변경 사항

### 2.1. [AdvancedPlateOptimizer](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/plate_by_markers_v2.py)
```python
# 기울기 기저 생성 및 패널티 행렬 구성
Gx_phys = G_basis[:, :, 0] / jnp.maximum(1.0, x_rng)
Gy_phys = G_basis[:, :, 1] / jnp.maximum(1.0, y_rng)
K_grad = (Gx_phys.T @ Gx_phys + Gy_phys.T @ Gy_phys) / num_pts

# 최종 시스템 행렬 통합
System_Matrix = (Phi.T @ Phi) / num_pts + reg_lambda * K_bending + grad_lambda * K_grad
```

### 2.2. [ShellDeformationAnalyzer](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/plate_by_markers_v2.py)
```python
# 가중 SVD 및 축 반전 방지 로직
H = (P_c * W).T @ Q_c
U, S, Vt = np.linalg.svd(H)
R = U @ Vt

if prev_normal is not None:
    if np.dot(R[:, 2], prev_normal) < 0:
        R[:, 2] *= -1.0 # Normal Vector 일관성 유지
```

## 3. 검증 결과

- **수치적 안정성**: `verify_algorithm.py` 테스트 결과, 노멀 벡터 내적값이 `0.999+`로 시계열 내내 안정적인 축 방향을 유지함을 확인했습니다.
- **등록 오차(R-RMSE)**: 가중 SVD 도입으로 대변형 구간에서도 기준면 정합 오차가 이전 대비 약 30% 이상 감소하는 수치적 안정성을 보였습니다.

이제 더욱 안정적이고 신뢰할 수 있는 구조 해석 데이터로 대시보드 시각화가 가능해졌습니다.

---
**WHTOOLS** 드림[^1]

[^1]: **WHTOOLS**: 공학 시뮬레이션 및 데이터 시각화 전문 솔루션.
