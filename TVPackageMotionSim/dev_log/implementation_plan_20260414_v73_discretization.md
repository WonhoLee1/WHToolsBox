# Implementation Plan - [v7.3] Grid Discretization & High-Fidelity Stress Recovery

Opencell 및 Chassis 부품이 시뮬레이션에서 단일 강체로 통합되는 현상을 해결하여, 4차 다항식 해석에 필요한 충분한 마커(16개 이상)를 확보하고 응력 해석의 무결성을 복구합니다.

## User Review Required

> [!IMPORTANT]
> - **격자 분할 강제**: `opencell`과 `chassis`가 시뮬레이션 상에서 실제로 여러 개의 블록으로 물리적 분할이 이루어지도록 `v6.py` 설정을 수정합니다. (현재 (0,0,0) 단일 블록으로 생성되는 문제 해결)
> - **해석 임계값 복구**: 마커가 16개 이상 확보되면, 자동으로 `poly_degree=4`가 활성화되어 정밀한 응력 분포가 도출됩니다.

## Proposed Changes

### [Simulation Setup Fix]

#### [MODIFY] [run_drop_simulation_cases_v6.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulation_cases_v6.py)
- `test_case_1_setup`: `opencell` 및 `chassis`의 `div` 설정을 재점검하고, 빌더가 이를 통합하지 않도록 `use_weld` 옵션과 함께 분할 무결성을 보장하는 추가 플래그(있을 경우)를 적용합니다.

### [Mapping Engine Enhancement]

#### [MODIFY] [whts_mapping.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_mapping.py)
- `extract_face_markers`: 부품이 단일 블록이더라도 표면에서 더 많은 샘플링 포인트를 추출할 수 있도록 보간 노드(Interpolated Nodes) 생성 로직 검토.

## Verification Plan

### Automated Tests
1. `python run_drop_simulation_cases_v6.py` 실행 후 `Opencell_Front`의 마커 개수가 **16개**로 증가했는지 확인.
2. 리포트의 `Max Stress [MPa]` 컬럼에 **0.00이 아닌 유의미한 수치**가 리포트되는지 확인.

### Manual Verification
1. ParaView 시각화에서 `Opencell` 파트가 단일 면이 아닌 쪼개진 격자 구조로 표현되는지 확인.
