# Implementation Plan - [v6.8] Data Diet & Persistence Optimization

저장되는 데이터의 정밀도를 조절하고 불필요한 필드를 제거하여 파일 용량을 획기적으로 줄입니다.

## User Review Required

> [!NOTE]
> - **용량 기대 효과**: 787MB → **약 100~150MB** (약 80% 절감 예상)
> - **정밀도**: 시뮬레이션 결과값의 정밀도를 `float64`에서 `float32`로 낮춥니다. (시각화 분석에는 지장 없음)

## Proposed Changes

### [Data Optimization]

#### [MODIFY] [run_drop_simulation_cases_v6.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulation_cases_v6.py)
- `dump_data` 생성 시 `float32` 변환 적용.
- `exclude_fields` 리스트를 통해 곡률(Curvature) 등 대용량 보조 필드를 필터링.

#### [MODIFY] [view_results_v6.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/view_results_v6.py)
- 최적화된 결과 구조(Missing fields)에 대한 방어 로직 추가.

## Verification Plan

### Manual Verification
1. 시뮬레이션 재실행 후 `latest_results.pkl` 용량 확인.
2. `view_results_v6.py`를 실행하여 대시보드에서 그래프 및 변형 형상이 정상적으로 표시되는지 확인.
