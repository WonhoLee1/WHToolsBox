# [WHTOOLS] 평판 변형 해석 알고리즘 개선 계획

리뷰 문서(`review_deformation_algorithm_20260407.md`)에서 제기된 알고리즘의 허점(Centroid Shift, Rotation Drift, Axis Flipping)을 해결하고, JAX 기반 최적화 엔진의 수치 안정성을 높이기 위한 기술적 세부 계획입니다.

## User Review Required

> [!IMPORTANT]
> 1. **가중치(Weights) 산정 방식**: 중앙부 마커에 가중치를 주는 방식은 가우시안(Gaussian) 분포를 기본으로 합니다. 판의 형상에 따라 가중치 폭(Sigma)을 조정할 필요가 있을 수 있습니다.
> 2. **기울기 패널티(Gradient Penalty)**: 가장자리 진동을 억제하기 위해 1차 미분 항을 추가합니다. 초기값은 `1e-6`으로 설정하며, 필요시 `PlateConfig`에서 조정 가능하게 합니다.

## Proposed Changes

### 1. 물리 해석 설정 (`PlateConfig`) 및 최적화 엔진 (`AdvancedPlateOptimizer`)

#### [MODIFY] [plate_by_markers_v2.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/plate_by_markers_v2.py)

- `PlateConfig`: `grad_lambda` (기울기 패널티 계수) 필드 추가.
- `AdvancedPlateOptimizer`: 
    - `get_gradient_basis`: 1차 미분 기저 생성 함수 추가 (기존 `PlateMechanicsSolver` 내부에 있던 것을 이동 및 최적화).
    - `solve_analytical`: 
        - `fixed_stats` 파라미터 추가하여 정규화 박스 고정 기능 구현.
        - `grad_lambda`를 이용한 1차 미분 패널티를 `System_Matrix`에 추가.

---

### 2. 강체 운동 제거 및 해석 프로세스 (`ShellDeformationAnalyzer`)

#### [MODIFY] [plate_by_markers_v2.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/plate_by_markers_v2.py)

- `fit_reference_plane`: 마커별 중앙으로부터의 거리를 계산하여 **가중치(Weights)** 맵을 사전 생성.
- `remove_rigid_motion`: 
    - 산술 평균 대신 **가중 평균(Weighted Mean)**으로 중심점 계산.
    - SVD 계산 시 가중치를 적용한 공분산 행렬 $\mathbf{H} = \mathbf{P}^T \mathbf{W} \mathbf{Q}$ 사용.
- `analyze`:
    - **Axis Flip Tracking**: 루프를 돌며 이전 프레임의 노멀 벡터($n_{t-1}$)와 현재 노멀($n_t$)의 내적을 체크하여 부호 반전 보정.
    - **Normalization Freeze**: 전 시계열 데이터에 대해 동일한 정규화 파라미터가 적용되도록 보장.

---

## Open Questions

- **가중치 분포**: 단순히 거리의 역수를 쓸지, 아니면 중심 영역(예: 전체 50% 영역)에 1.0, 나머지에 0.1 등의 계단식 가중치를 쓸지 결정이 필요합니다. (현재는 가우시안 분포 제안)

## Verification Plan

### Automated Tests
- `test_qt.py` 등을 실행하여 시간에 따른 평판 애니메이션이 덜컹거림(Vibration) 없이 부드럽게 재생되는지 확인.
- JAX `rmses` 값이 이전보다 안정화되는지 로그 확인.

### Manual Verification
- 대변형(Bending) 발생 시 기준 평면이 기울어지는지(Tilting), 혹은 축이 뒤집히는지 시각적으로 검토.
- 가장자리 스플라인 진동(Runge's phenomenon) 억제 여부 확인.
