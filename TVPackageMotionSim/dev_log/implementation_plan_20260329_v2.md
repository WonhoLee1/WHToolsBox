# 구조 해석 지표 출력 누락 수정 계획

구조 해석 그래프 생성 시 Bending Z와 Twist Angle을 제외한 나머지 지표(RRG, Tilt-X/Y, GTI, GBI 등)가 출력되지 않는 문제를 해결합니다.

## User Review Required

> [!IMPORTANT]
> - 이제 **Bending X/Y (Tilt 분성 성분)**, **RRG (상대 회전 구배)**, **GTI (Global Tilt Index)**, **GBI (Global Bending Index)** 지표가 그래프로 정상 출력됩니다.
> - 기존에 연산 로직만 있고 데이터 저장 기능이 누락되었던 부분을 보완합니다.

## Proposed Changes

### [Reporting] `run_drop_simulator/whts_reporting.py`

#### [MODIFY] [whts_reporting.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_reporting.py)
`compute_structural_step_metrics` 함수를 다음과 같이 업그레이드합니다.
- **Tilt 분해**: 전체 Bending 외에 X축 및 Y축 방향의 기울기 성분(`bend_x_deg`, `bend_y_deg`)을 추가로 연산하여 `all_blocks_bend_x/y` 리스트에 저장합니다.
- **부엄별 지표 집계**: 각 부품별로 최대 RRG를 추적하여 `max_rrg_hist`에 저장합니다.
- **전역 지표 연산 (GTI/GBI)**: 부품별 전역 지표인 GTI와 GBI를 매 스텝 연산하여 `sim.structural_time_series['comp_global_metrics']`에 누적합니다.
- **PBA 데이터 강화**: 각 부품별 PBA 주축 강도를 추적할 수 있도록 데이터를 보강합니다.

## Open Questions
- **X/Y Bending 정의**: 사용자 환경에서 Bending X/Y가 각각 Y축 및 X축 회전에 의한 기울기를 의미하는지 확인이 필요하나, 일반적인 틸트 성분 분해 방식(atan2 기반)으로 우선 적용합니다.

## Verification Plan

### Automated Tests
- `python run_drop_simulation_cases_v4.py` 실행 후 Post-Processing UI에서:
    1. 모든 구조 해석 지표(RRG, PBA, Bend X/Y, GTI, GBI)를 선택하고 그래프 생성 버튼을 클릭합니다.
    2. 모든 그래프에 데이터 포인트가 정상적으로 출력되는지 확인합니다.
