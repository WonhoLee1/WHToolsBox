# Implementation Plan - Marker Integrity & Density Reinforcement

모든 파트(Opencell 포함)의 표면 노드(코너점)들이 누락 없이 분석에 참여하도록 마커 추출 엔진을 강화하고, 추출 결과를 투명하게 보고합니다.

## User Review Required

> [!IMPORTANT]
> - **마커 추출 개수 고지**: 모든 분석 결과에 추출된 마커 수가 명시됩니다.
> - **격자 완결성 보장**: 3x3 배열 시 반드시 16개의 노드가 수집되도록 인덱싱 로직을 강화합니다.
> - **Thin Part Support**: 얇은 파트에서 마커 부족으로 인해 `nan`이 발생하는 현상을 방지하기 위한 수치 보정 로직이 적용됩니다.

## Proposed Changes

### [Component] Marker Mapping Engine (`whts_mapping.py`)

#### [MODIFY] [whts_mapping.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_mapping.py)
- **Grid Indexing Refinement**: `max_indices` 계산 시 0-based 인덱스와 실제 노드 개수(N+1) 간의 매핑을 더 정교하게 수정.
- **Node Accumulator Guard**: `node_idx` 생성 시 부동 소수점 오차로 인한 노드 누락 방지.
- **Surface Coverage Check**: 모든 어셈블리의 외곽면이 100% 커버되는지 내부 체크 루틴 추가.

### [Component] Analysis Engine (`whts_multipostprocessor_engine.py`)

#### [MODIFY] [whts_multipostprocessor_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_multipostprocessor_engine.py)
- **Detailed Component Log**: `[PART-OK] {Name} | Markers: {Count} | Grid: {NxM} | RMSE: ...` 형태로 로그 상세화.
- **Adaptive Degree Selection**: 수집된 마커의 기하학적 분포(Aspect Ratio)를 분석하여 최적의 분석 차수 자동 선택.

## Open Questions
- 분석 대상인 `Opencell` 파트의 각 블록 크기가 동일한가요? 혹은 가변적인가요? (균일 격자 가정이 무너질 경우 PSR 알고리즘의 가중치를 조정해야 합니다.)

## Verification Plan

### Automated Tests
- 3x3x3 그리드 더미 데이터를 주입하여 정확히 16개(면당)의 마커와 적절한 차수가 로그에 찍히는지 확인.

### Manual Verification
- 수정 후 시뮬레이션 재실행 시 `Opencell_Right`의 마커 수가 사용자 의도(풍부한 점)에 부합하는지 로그 확인.
