# Implementation Plan - [v7.4] Intelligent Hybrid Alignment & Visual Integrity

SVD 결과의 품질을 스스로 판단하여 최적화 기반 정합으로 자동 전환하는 지능형 엔진을 구축하고, ParaView 6.0 및 VTKHDF 완결성을 확보합니다.

## User Review Required

> [!IMPORTANT]
> - **하이브리드 정합 (SVD + Optimization)**: SVD 결과를 R-RMSE로 평가하여, 임계치 초과 시 최적화 엔진(Min-Dist)을 가동합니다. (폭주 상황 대응)
> - **차수 고정 무결성**: 마커 개수에 따른 차수 자동 후퇴를 방지하고, 규제화(Regularization)를 통해 설정 차수를 유지합니다.
> - **VTKHDF 시계열 오프셋**: `ConnectivityIdOffsets`, `CellOffsets` 등을 주입하여 시각화 에러를 해결합니다.

## Proposed Changes

### [Intelligent Robust Engine]

#### [MODIFY] [whts_multipostprocessor_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_multipostprocessor_engine.py)
- `remove_rigid_motion`: 
    - SVD 정렬 후 RMSE 평가 로직 추가.
    - `rmse > threshold` 일 경우 `scipy.optimize.minimize`를 이용한 강건 정합(Robust Alignment) 수행.
- `fit_reference_plane`: `poly_degree` 후퇴 로직 제거 및 Ridge 규제 강도 최적화.

#### [MODIFY] [whts_mapping.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_mapping.py)
- `extract_face_markers`: `use_virtual_markers=False` 기본값 설정 및 옵션화.

### [Visual Pipeline Fix]

#### [MODIFY] [whts_exporter.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_exporter.py)
- `export_to_vtkhdf`: `Steps/ConnectivityIdOffsets`, `Steps/CellOffsets` 데이터셋 생성 로직 추가.
- `launch_paraview`: ParaView 6.0 API 대응 (`Variables` -> `ModelVariables` 또는 `add_attribute`).

## Verification Plan

### Automated Tests
1. `python run_drop_simulation_cases_v6.py` 실행 후 로그에서 `[ROBUST-ALIGN]` 트리거 여부 확인.
2. ParaView에서 오프셋 에러 발생 여부 및 통계 필터 작동 확인.
