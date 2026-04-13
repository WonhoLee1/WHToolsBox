# Implementation Plan - [v7.4] Robust Alignment & Authentic Mapping

사용자님의 제안에 따라 가상 마커 의존도를 낮추고 '자동 차수 후퇴' 로직을 거두어내며, 최적화 기반 정함과 ParaView 6.0 완결성을 확보합니다.

## User Review Required

> [!IMPORTANT]
> - **가상 마커 선택권**: `use_virtual_markers=False`를 기본값으로 설정하여 실데이터 기반 해석을 지향합니다.
> - **차수 일관성 확보**: 마커가 부족할 때 자동으로 1차로 떨어뜨리던 로직을 제거하거나 사용자 설정 차수를 유지하도록 보강합니다. (응력 0 현상 방지)
> - **최적화 기반 정합 (L2-Minimization)**: SVD 대신 거리 최소화 최적화를 통해 기준 평면을 잡는 로직을 추가하여 강건성(Robustness)을 확보합니다.
> - **VTKHDF 시계열 메타데이터**: `ConnectivityIdOffsets`, `CellOffsets` 등을 주입하여 ParaView 로딩 에러를 해결합니다.

## Proposed Changes

### [Authentic Mapping & Robust Engine]

#### [MODIFY] [whts_mapping.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_mapping.py)
- `extract_face_markers`: `use_virtual_markers` 인자 추가 및 기본값 False 설정.
- 이름 매핑 시 통합 강체보다 이산화된 바디들을 먼저 찾도록 로직 보강. (마커 4개 문제의 근본 해결)

#### [MODIFY] [whts_multipostprocessor_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_multipostprocessor_engine.py)
- `fit_reference_plane`: `poly_degree=1` 자동 후퇴 로직 제거. 규제화 및 최적화(L2-Min)를 통해 설정 차수 유지.
- `remove_rigid_motion`: 사용자 제안 '거리 최소화 최적화' 적용. (Outlier 강건성 확보)

### [Visual Pipeline Fix]

#### [MODIFY] [whts_exporter.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_exporter.py)
- `export_to_vtkhdf`: `Steps/ConnectivityIdOffsets`, `Steps/CellOffsets` 데이터셋 생성 로직 추가.
- `launch_paraview`: ParaView 6.0 API 대응 (`Variables` -> `ModelVariables` 또는 `add_attribute`).

## Verification Plan

### Automated Tests
1. `python run_drop_simulation_cases_v6.py` 실행 후 ParaView에서 오프셋 관련 에러가 사라졌는지 확인.
2. `Chassis`, `Opencell` 리포트에서 마커 16개(실데이터)가 정확히 추출되는지 확인.
3. `Max Stress`가 자동 후퇴 없이 설정된 차수(4차)로 정밀하게 계산되는지 확인.
