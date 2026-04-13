# Implementation Plan - [v6.5] Structural Analysis Numerical Stabilization

[v6.4]에서 확인된 비정상적인 변형량($283mm$) 폭주 문제를 해결하고 물리적 타당성을 확보합니다.

## User Review Required

> [!IMPORTANT]
> - **피팅 차수 제한**: 16개 이하의 마커 데이터셋에 대해서는 다항식 차수를 3차 이하로 제한하여 외곽 발산을 방지합니다.
> - **물리적 가이드**: `Max Disp`가 실제 마커 변위보다 지나치게 클 경우(예: 2배 이상), 리포트에서 이를 유효하지 않은 데이터로 간주하거나 마커 기반의 값을 우선 출력합니다.

## Proposed Changes

### [Structural Analysis Engine]

#### [MODIFY] [whts_multipostprocessor_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_multipostprocessor_engine.py)
- `analyze` 메서드 내 `deg_x`, `deg_y` 결정 로직 보강: 마커 밀도에 따라 더 보수적인 차수 선정.
- `AdvancedPlateOptimizer`의 정규화 강도 조절 및 발산 체크 로직 추가.
- 리포트 출력 시 `Max Disp`를 "마커 최대 변위" 기준으로 정제.

## Verification Plan

### Automated Tests
- `python run_drop_simulation_cases_v6.py` 재실행 후:
    1. **로그 확인**: `Fit (X mm) > Markers (Y mm)` 경고가 사라지거나 현저히 줄어드는지 확인.
    2. **리포트 확인**: `Max Disp` 수치가 10mm 내외의 상식적인 범위로 들어오는지 확인.
