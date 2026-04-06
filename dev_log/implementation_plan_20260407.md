# [WHTOOLS] Integrated Shell Deformation Analysis Engine v2.5 안정화 계획

`plate_by_markers.py`의 성공적인 구현 사례를 바탕으로, 현재 `plate_by_markers_v2.py`에서 발생하는 좌표계 불일치 및 분석 루틴의 런타임 오류를 해결합니다.

## User Review Required

> [!IMPORTANT]
> **Kabsch 알고리즘의 회전 행렬 방향성**
> 현재 `v2`는 $Q \to P$ 회전을 산출하면서 $P \to Q$로 주석 처리가 되어 있습니다. 3D 뷰어의 `update_frame` 로직과 일치시키기 위해 이를 $P \to Q$ (Forward Mapping) 방향으로 통일합니다.

> [!WARNING]
> **analyze() 루틴의 대대적 정리**
> 현재 `v2`의 `analyze()` 메서드에는 정의되지 않은 변수(`p_ref_jax`, `all_rot_matrices` 등)와 중복된 차수 결정 로직이 포함되어 있습니다. 이를 `plate_by_markers.py` 수준의 정갈한 코드로 리팩토링합니다.

## Proposed Changes

### [Component] Shell Deformation Engine

#### [MODIFY] [plate_by_markers_v2.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/plate_by_markers_v2.py)

- **remove_rigid_motion()**: 
  - Kabsch 알고리즘 결과인 `Rotation_Matrix`를 $P \to Q$ (Reference to Current) 방향으로 수정.
  - SVD 결과 $M = P^T Q = U S V^T$ 일 때, $R = U V^T$ (또는 반사 보정 포함)를 적용하여 $Q = P R$ 관계를 성립시킴.
- **analyze()**:
  - `all_w`, `all_R`, `all_cq`, `all_rmses` 등 루프 내부 변수와 결과 패키징 변수명 일치화.
  - `p_ref_jax` 등 누락된 JAX 텐서 정의 추가.
  - 중복된 '차수 적응 장치' 섹션 통합 및 논리 간소화.
- **QtVisualizerV2.update_frame()**:
  - `Global` 모드에서의 좌표 복원 식을 `p_world = (p_local @ basis + c_ref - c_P_kabsch) @ R + c_Q_kabsch` 형태로 정밀 검정.
  - (`v2`에서는 `c_ref`가 곧 `c_P_kabsch`이므로 `p_local @ basis @ R + c_Q` 형태가 될 것으로 예상)

## Open Questions

- `plate_by_markers.py`의 `centroid_0` (바운딩 박스 중심) 방식을 `v2`에도 도입하시겠습니까? 아니면 현재의 `ref_center` (산술 평균 중심) 방식을 유지하시겠습니까? 
  - *제안*: 수치적 안정성을 위해 `plate_by_markers.py`의 `centroid_0` 방식을 따르는 것이 좋습니다.

## Verification Plan

### Automated Tests
- `run_drop_simulation_cases_v5.py` 실행을 통한 엔드-투-엔드 검증.
- 개별 파트 분석 결과의 F-RMSE 및 R-RMSE 값이 $10^{-6}$ 수준 이하인지 확인.

### Manual Verification
- 대시보드 재생 시 `Global` 뷰에서 파트 간의 정렬 상태가 시뮬레이션 원본과 일치하는지 육안 확인.
- `Local` 뷰에서 마커의 면내 유동(In-plane drift)이 완전히 제거되었는지 확인.
