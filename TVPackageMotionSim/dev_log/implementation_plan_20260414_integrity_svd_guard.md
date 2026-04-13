# Implementation Plan - [v6.9a] Structural Analysis Integrity & SVD Guard

수학적 발산으로 인한 비현실적인 수치를 차단하고, 특히 마커 배치가 불리한 측면 부품(Opencell Left/Right)의 정렬 안정성을 확보합니다.

## User Review Required

> [!IMPORTANT]
> - **SVD 보호 시스템**: 마커가 직선상에 배치되어 회전이 불안정한 경우(측면 부품), 회전 자유도를 제한하여 변위 폭주(393mm 등)를 원천 차단합니다.
> - **물리적 검증(Physical Check)**: 계산된 변위가 부품 크기의 50%를 초과할 경우 "수학적 오차"로 간주하여 Safe-Response(마커 기준값)로 전환합니다.
> - **JAX 의존도 조절**: JAX는 계산 가속을 위해 사용하되, 입력값과 결과값의 논리적 필터링을 파이썬 레벨에서 엄격히 수행합니다.

## Proposed Changes

### [Alignment & Kinematics]

#### [MODIFY] [whts_multipostprocessor_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_multipostprocessor_engine.py)
- `remove_rigid_motion`: SVD 수행 후 특이값의 비율(S1/S3)을 체크하여, 평면성이 부족한 경우(측면 부품 등) 억지 회전을 막는 **Orthogonal Constraint** 로직 추가.
- `analyze`: 응력 계산 전 `E` 단위를 MPa로 교정($10^{-6}$ 필터링).

### [Safety & Stabilization]

#### [MODIFY] [whts_multipostprocessor_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_multipostprocessor_engine.py)
- **Divergence Guard**: 피팅 결과가 물리적으로 불가능한 수준일 경우, `Max_Disp_Verified`뿐만 아니라 전체 필드를 마커 중심의 보수적 데이터로 대체하는 로직 강화.

## Verification Plan

### Automated Tests
1. `Opencell_Left/Right`의 최대 변위가 **50mm 미만**(현실적 범위)으로 잡히는지 로그 확인.
2. 응력 값이 **1,000 MPa 미만**인지 확인.

### Manual Verification
1. ParaView에서 측면 부품이 제멋대로 회전하거나 찢겨 보이지 않는지 시각적으로 확인.
