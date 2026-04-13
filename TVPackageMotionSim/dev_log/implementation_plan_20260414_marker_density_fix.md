# Implementation Plan - Marker Density Verification & NaN Stability Fix

Opencell(3x3x3) 등 복합 블록 구조에서 마커가 의도한 대로(16개 노드) 추출되는지 검증하고, 수치적 불안정성으로 인한 `nan` 발생 및 크래시를 방지합니다.

## User Review Required

> [!IMPORTANT]
> - 이제 분석 로그에 **추출된 마커의 총 개수**가 명시됩니다. (`[PART-OK] ... Markers: 16`)
> - `nan` 발생 시 `0.0`으로 자동 치환하여 ParaView 익스포트 및 대시보드 크래시를 방지합니다.

## Proposed Changes

### [Component] Post-Processor Engine (`whts_multipostprocessor_engine.py`)

#### [MODIFY] [whts_multipostprocessor_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_multipostprocessor_engine.py)
- **Log Refinement**: `PART-OK` 로그에 `Markers: {N}` 정보 추가.
- **Numerical Guard**: SVD 및 PCA 연산 전 데이터의 유효성(Variance check) 검증 강화.
- **NaN Handling**: 결과 데이터 생성 후 `np.nan_to_num`을 적용하여 오염된 데이터가 대시보드로 넘어가지 않도록 차단.

### [Component] Coordinate Mapping (`whts_mapping.py`)

#### [MODIFY] [whts_mapping.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_mapping.py)
- **Node Accumulator Logic Check**: 블록 간 공유되는 노드가 올바르게 병합(Accumulation)되어 3x3 배열 기준 16개가 나오는지 로직 검증.
- **Thin Part Support**: 얇은 파트(One-dimension thin)를 위한 PCA 축 결정 알고리즘 개선.

## Open Questions
- Opencell의 블록 간 간격이 혹시 떨어져 있나요? (Weld가 True이므로 붙어있을 것으로 가정하지만, 떨어져 있을 경우 마커 병합 로직을 수정해야 합니다.)

## Verification Plan

### Manual Verification
- `Opencell` 모델 재실행 후 로그에 마커 개수가 16개(혹은 의도한 수)로 찍히는지 확인.
- `nan` 발생 경고가 떠도 시뮬레이션 종료 후 대시보드가 정상적으로 뜨는지 확인.
