# Implementation Plan - [v6.3] Final Unit Sync & Stability

마커 개수 무결성 증명 후, 비정상적으로 폭주하는 응력 수치를 물리적으로 타당한 범위(MPa)로 보충하고 파이프라인을 최종 안정화합니다.

## User Review Required

> [!IMPORTANT]
> - **단위계 동기화**: 마커 좌표계($mm$)와 재료 물성치($Pa$) 간의 불일치를 해결합니다. 해석 엔진 내부의 모든 연산을 $mm, N, MPa$ 단위계로 통일합니다.
> - **종료 코드 안정화**: GUI 미지원 환경에서 `SystemExit` 발생 시 발생하는 비정상 종료 코드를 방어합니다.

## Proposed Changes

### [Numerical Scaling Fix]

#### [MODIFY] [whts_multipostprocessor_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_multipostprocessor_engine.py)
- `PlateMechanicsSolver` 내의 강성 행렬($D$) 계산 시 $mm$ 단위계에 맞춰 $E$ 값을 $MPa$($1/10^6$)로 스케일링하여 적용.
- `ShellDeformationAnalyzer` 초기화 시 `W, H`가 0인 경우 `o_data`의 범위를 기반으로 자동 계산하도록 보완 (해석 정밀도 향상).

#### [MODIFY] [run_drop_simulation_cases_v6.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulation_cases_v6.py)
- `sys.exit(0)` 호출 방식을 개선하여 터미널 리다이렉션 환경에서도 `Exit code: 0`을 유지하도록 처리.

## Verification Plan

### Automated Tests
- `python run_drop_simulation_cases_v6.py` 실행:
    1. 리포트 테이블의 `Max Stress [MPa]`가 현실적인 수치(예: 0.1 ~ 500.0)로 출력되는지 확인.
    2. 로그에서 `Exit code: 0` 확인.
