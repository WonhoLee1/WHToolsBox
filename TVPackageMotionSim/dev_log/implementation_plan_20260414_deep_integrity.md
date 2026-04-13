# Implementation Plan - [v6.9b] Deep Numerical Integrity & Scaling Fix

임시 방편(클리핑)을 넘어, 응력 산출의 근본 원인인 단위계 미스매치와 SVD 정렬 불안정성을 완전히 해결합니다.

## User Review Required

> [!IMPORTANT]
> - **영률(E) 단위 고정**: 외부 입력에 관계없이 모든 파트의 `E`를 **1,000 ~ 70,000 MPa** 범위 내로 강제 한정하여 수치 폭주를 원천 차단합니다.
> - **SVD 가드 강화**: 특이값 비율 임계치를 `0.05`에서 **`0.15`**로 대폭 상향하여, 조금이라도 선형에 가까운 마커 배치는 회전을 무시합니다.
> - **규제화(Regularization) 강화**: `reg_lambda`를 10배 높여 다항식이 데이터를 무리하게 쫓아가다 발산하는 것을 막습니다.

## Proposed Changes

### [Core Mechanics Logic]

#### [MODIFY] [whts_multipostprocessor_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_multipostprocessor_engine.py)
- `ShellDeformationAnalyzer.__init__`: 영률 보정 로직을 `E > 1e6` 뿐만 아니라 하한선까지 두어 MPa 단위로 강제 고정.
- `remove_rigid_motion`: `planar_ratio < 0.15` 적용 및 회전 필터링 강화.

### [Fitting Stability]

#### [MODIFY] [whts_multipostprocessor_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_multipostprocessor_engine.py)
- `PlateConfig`: 기본 `reg_lambda`를 상향하여 피팅 안정성 확보.

## Verification Plan

### Automated Tests
1. `latest_results.pkl`을 로드하여 `Opencell_Left`의 응력이 **200 MPa 미만**(정상 범위)인지 확인.
2. 모든 부품의 최대 변위가 **50mm 미만**으로 잡히는지 로그 확인.

### Manual Verification
1. ParaView에서 부품들이 떨리거나 폭발하는 형태가 아닌, 부드러운 굽힘 형상을 보이는지 확인.
