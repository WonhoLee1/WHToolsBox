# Implementation Plan - [RCA] Final Root Cause Analysis & Fix

임시방편(Clipping)을 배제하고, `Opencell_Left/Right` 부품의 수치가 폭주하는 근본적인 원인을 규명하고 해결합니다.

## User Review Required

> [!IMPORTANT]
> - **진단 우선**: 마커의 인덱스가 참조 데이터와 시뮬레이션 데이터 사이에서 꼬였는지 확인하기 위해 `t=0` 시점의 정밀 로그를 출력합니다.
> - **가설 검증**: 214mm RMSE는 보통 "인덱싱 오류" 또는 "좌표계 뒤집힘"에서 발생합니다. 이를 수정하면 클리핑 없이도 정상 수치가 나와야 합니다.

## Proposed Changes

### [Diagnostic Step]

#### [MODIFY] [whts_multipostprocessor_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_multipostprocessor_engine.py)
- `ShellDeformationAnalyzer.fit_reference_plane`: 참조 마커의 바운딩 박스와 무게중심을 상세히 출력하는 디버그 코드 주입.
- `ShellDeformationAnalyzer.analyze`: 첫 번째 프레임(`t=0`)의 정렬 RMSE를 로그의 최상단에 노출.

### [Fix Strategy (Hypothesis based)]

#### [MODIFY] [run_drop_simulation_cases_v6.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulation_cases_v6.py) (검토 후 수정)
- 만약 마커 추출 로직(`get_part_markers`)에서 좌/우 부품의 인덱스가 대칭적으로 뒤집혔다면 이를 바로잡음.

## Verification Plan

### Automated Tests
1. `t=0` 시점의 RMSE가 **0.1mm 이하**로 떨어지는지 확인 (현재 214mm 추정).
2. 인덱스 수정 후, 클리핑 가드 없이도 변위가 **10mm 내외**로 출력되는지 확인.

### Manual Verification
1. ParaView에서 `Opencell_Left` 파트의 마커들이 제 위치에 점으로 예쁘게 찍혀 있는지 확인.
